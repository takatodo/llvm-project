//===- LoopCarriedComputationReuse.cpp -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements reuse of a pure loop-body computation whose affine
// accesses are translated by one iteration.
//
//===----------------------------------------------------------------------===//

#include "LoopCarriedReuseUtils.h"
#include "mlir/Dialect/Affine/Transforms/Passes.h"

#include "mlir/Analysis/AliasAnalysis.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"

#include <cstdint>
#include <optional>

namespace mlir {
namespace affine {
#define GEN_PASS_DEF_AFFINELOOPCARRIEDCOMPUTATIONREUSE
#include "mlir/Dialect/Affine/Transforms/Passes.h.inc"
} // namespace affine
} // namespace mlir

using namespace mlir;
using namespace mlir::affine;

namespace {

constexpr unsigned maxComputationDepth = 256;

struct ReuseCandidate {
  Value earlierRoot;
  Value laterRoot;
  SmallVector<Operation *> earlierOps;
  SmallVector<Value> sources;
};

/// Match two side-effect-free single-result DAGs. The only varying leaves are
/// affine loads separated by one loop iteration. Equal values must be defined
/// outside the loop, so an existing iter_arg cannot become a reusable context.
class ShiftedDAGMatcher {
public:
  explicit ShiftedDAGMatcher(AffineForOp loop) : loop(loop) {}

  LogicalResult match(Value earlier, Value later) {
    return matchImpl(earlier, later, /*depth=*/0);
  }

  SmallVector<Operation *> takeEarlierOps() { return std::move(earlierOps); }

  ArrayRef<Operation *> getLaterOps() const { return laterOps; }

  bool hasTranslation() const { return hasTranslatedLoad; }

  SmallVector<Value> takeSources() {
    return SmallVector<Value>(sources.begin(), sources.end());
  }

private:
  LogicalResult matchImpl(Value earlier, Value later, unsigned depth) {
    if (depth > maxComputationDepth)
      return failure();
    if (earlier == later)
      return success(loop.isDefinedOutsideOfLoop(earlier));
    if (earlier.getType() != later.getType())
      return failure();

    auto knownEarlier = earlierToLater.find(earlier);
    if (knownEarlier != earlierToLater.end())
      return success(knownEarlier->second == later);
    auto knownLater = laterToEarlier.find(later);
    if (knownLater != laterToEarlier.end())
      return success(knownLater->second == earlier);

    auto earlierLoad = earlier.getDefiningOp<AffineLoadOp>();
    auto laterLoad = later.getDefiningOp<AffineLoadOp>();
    if (earlierLoad || laterLoad) {
      if (!earlierLoad || !laterLoad)
        return failure();
      FailureOr<loop_carried_reuse::LoadAcrossIterationMatch> accessMatch =
          loop_carried_reuse::matchLoadsAcrossOneIteration(earlierLoad,
                                                           laterLoad, loop);
      if (failed(accessMatch))
        return failure();
      for (Operation *operation : accessMatch->earlierIndexOps)
        if (seenEarlier.insert(operation).second)
          earlierOps.push_back(operation);
      mapValues(earlier, later);
      record(earlierLoad, laterLoad);
      sources.insert(accessMatch->source);
      hasTranslatedLoad |= accessMatch->isTranslated;
      return success();
    }

    Operation *earlierOp = earlier.getDefiningOp();
    Operation *laterOp = later.getDefiningOp();
    if (!earlierOp || !laterOp || earlierOp == laterOp ||
        earlierOp->getParentOp() != loop || laterOp->getParentOp() != loop ||
        earlierOp->getNumResults() != 1 || laterOp->getNumResults() != 1 ||
        earlierOp->getNumRegions() != 0 || laterOp->getNumRegions() != 0 ||
        !isMemoryEffectFree(earlierOp) || !isMemoryEffectFree(laterOp) ||
        !isSpeculatable(earlierOp) || !isSpeculatable(laterOp))
      return failure();

    mapValues(earlier, later);
    auto flags = static_cast<OperationEquivalence::Flags>(
        OperationEquivalence::IgnoreLocations |
        OperationEquivalence::IgnoreDiscardableAttrs |
        OperationEquivalence::IgnoreCommutativity);
    if (!OperationEquivalence::isEquivalentTo(
            earlierOp, laterOp,
            [&](Value earlierOperand, Value laterOperand) {
              return matchImpl(earlierOperand, laterOperand, depth + 1);
            },
            /*markEquivalent=*/nullptr, flags))
      return failure();

    record(earlierOp, laterOp);
    return success();
  }

  void mapValues(Value earlier, Value later) {
    earlierToLater.try_emplace(earlier, later);
    laterToEarlier.try_emplace(later, earlier);
  }

  void record(Operation *earlier, Operation *later) {
    if (seenEarlier.insert(earlier).second)
      earlierOps.push_back(earlier);
    if (seenLater.insert(later).second)
      laterOps.push_back(later);
  }

  AffineForOp loop;
  DenseMap<Value, Value> earlierToLater;
  DenseMap<Value, Value> laterToEarlier;
  llvm::SmallPtrSet<Operation *, 16> seenEarlier;
  llvm::SmallPtrSet<Operation *, 16> seenLater;
  llvm::SmallSetVector<Value, 4> sources;
  SmallVector<Operation *> earlierOps;
  SmallVector<Operation *> laterOps;
  bool hasTranslatedLoad = false;
};

static std::optional<ReuseCandidate>
findReuseCandidate(AffineForOp loop, AliasAnalysis &aliasAnalysis) {
  SmallVector<Operation *> bodyOps;
  for (Operation &operation : loop.getBody()->without_terminator())
    bodyOps.push_back(&operation);

  // Search consumers backwards so that a whole translated computation is
  // chosen before a translated subexpression inside that computation.
  for (Operation *consumer : llvm::reverse(bodyOps)) {
    for (unsigned laterIndex = 0; laterIndex < consumer->getNumOperands();
         ++laterIndex) {
      for (unsigned earlierIndex = 0; earlierIndex < consumer->getNumOperands();
           ++earlierIndex) {
        if (earlierIndex == laterIndex)
          continue;
        Value earlierRoot = consumer->getOperand(earlierIndex);
        Value laterRoot = consumer->getOperand(laterIndex);
        if (!earlierRoot.getDefiningOp() || !laterRoot.getDefiningOp() ||
            isa<AffineLoadOp>(earlierRoot.getDefiningOp()))
          continue;

        ShiftedDAGMatcher matcher(loop);
        if (failed(matcher.match(earlierRoot, laterRoot)) ||
            !matcher.hasTranslation())
          continue;
        if (llvm::is_contained(matcher.getLaterOps(),
                               earlierRoot.getDefiningOp()))
          continue;

        SmallVector<Operation *> earlierOps = matcher.takeEarlierOps();
        SmallVector<Value> sources = matcher.takeSources();
        if (earlierOps.empty() || sources.empty() ||
            !loop_carried_reuse::isSafeToPreload(loop, sources, earlierOps,
                                                 aliasAnalysis))
          continue;

        llvm::SmallPtrSet<Operation *, 16> earlierSet(earlierOps.begin(),
                                                      earlierOps.end());
        if (llvm::any_of(earlierOps, [&](Operation *operation) {
              return llvm::any_of(operation->getOperands(), [&](Value operand) {
                if (operand == loop.getInductionVar() ||
                    loop.isDefinedOutsideOfLoop(operand))
                  return false;
                Operation *definingOp = operand.getDefiningOp();
                return !definingOp || !earlierSet.contains(definingOp);
              });
            }))
          continue;

        return ReuseCandidate{earlierRoot, laterRoot, std::move(earlierOps),
                              std::move(sources)};
      }
    }
  }
  return std::nullopt;
}

struct AffineLoopCarriedComputationReuse
    : public affine::impl::AffineLoopCarriedComputationReuseBase<
          AffineLoopCarriedComputationReuse> {
  void runOnOperation() override {
    AliasAnalysis &aliasAnalysis = getAnalysis<AliasAnalysis>();
    SmallVector<AffineForOp> loops;
    getOperation().walk<WalkOrder::PostOrder>(
        [&](AffineForOp loop) { loops.push_back(loop); });

    IRRewriter rewriter(&getContext());
    for (AffineForOp loop : loops) {
      std::optional<ReuseCandidate> candidate =
          findReuseCandidate(loop, aliasAnalysis);
      if (candidate && failed(loop_carried_reuse::materializeLoopCarriedValue(
                           rewriter, loop, candidate->earlierRoot,
                           candidate->laterRoot, candidate->earlierOps))) {
        signalPassFailure();
        return;
      }
    }
  }
};

} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::affine::createAffineLoopCarriedComputationReusePass() {
  return std::make_unique<AffineLoopCarriedComputationReuse>();
}

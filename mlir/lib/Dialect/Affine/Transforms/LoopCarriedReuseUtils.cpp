//===- LoopCarriedReuseUtils.cpp - Loop-carried reuse support -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "LoopCarriedReuseUtils.h"

#include "mlir/Dialect/Affine/IR/AffineValueMap.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"

#include <optional>

using namespace mlir;
using namespace mlir::affine;

namespace mlir::affine::loop_carried_reuse {
namespace {

constexpr unsigned maxIndexComputationDepth = 256;

static void canonicalizeAccess(AffineLoadOp load, AffineMap &map,
                               SmallVectorImpl<Value> &operands) {
  map = load.getAffineMap();
  llvm::append_range(operands, load.getMapOperands());
  fullyComposeAffineMapAndOperands(&map, &operands);
  map = simplifyAffineMap(map);
  canonicalizeMapAndOperands(&map, &operands);
}

static LogicalResult collectAffineApplyDependencies(
    Value value, AffineForOp loop, llvm::SmallPtrSetImpl<Operation *> &seen,
    SmallVectorImpl<Operation *> &operations, unsigned depth) {
  if (depth > maxIndexComputationDepth)
    return failure();
  if (value == loop.getInductionVar() || loop.isDefinedOutsideOfLoop(value))
    return success();

  auto apply = value.getDefiningOp<AffineApplyOp>();
  if (!apply || apply->getParentOp() != loop)
    return failure();
  if (seen.contains(apply))
    return success();
  for (Value operand : apply.getMapOperands())
    if (failed(collectAffineApplyDependencies(operand, loop, seen, operations,
                                              depth + 1)))
      return failure();
  if (seen.insert(apply).second)
    operations.push_back(apply);
  return success();
}

static bool isSourceStable(AffineForOp loop, Value source,
                           AliasAnalysis &aliasAnalysis) {
  bool stable = true;
  (void)loop.getBody()->walk<WalkOrder::PostOrder>([&](Operation *operation) {
    // Recursive-effect operations derive their effects from nested operations
    // unless they also expose direct effects. The post-order walk has already
    // checked those nested operations.
    if (operation->hasTrait<OpTrait::HasRecursiveMemoryEffects>() &&
        !isa<MemoryEffectOpInterface>(operation))
      return WalkResult::advance();
    if (aliasAnalysis.getModRef(operation, source).isMod()) {
      stable = false;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return stable;
}

static bool hasAtLeastTwoIterations(AffineForOp loop) {
  if (loop.hasConstantBounds()) {
    int64_t lowerBound = loop.getConstantLowerBound();
    int64_t upperBound = loop.getConstantUpperBound();
    if (upperBound <= lowerBound)
      return false;
    uint64_t span =
        static_cast<uint64_t>(upperBound) - static_cast<uint64_t>(lowerBound);
    return span > static_cast<uint64_t>(loop.getStepAsInt());
  }

  std::optional<APInt> tripCount = loop.getStaticTripCount();
  return tripCount && !tripCount->isNegative() && tripCount->ugt(1);
}

} // namespace

FailureOr<LoadAcrossIterationMatch>
matchLoadsAcrossOneIteration(AffineLoadOp earlier, AffineLoadOp later,
                             AffineForOp loop) {
  if (!earlier || !later || earlier == later ||
      earlier->getParentOp() != loop || later->getParentOp() != loop ||
      earlier.getMemRef() != later.getMemRef() ||
      earlier.getType() != later.getType() ||
      !loop.isDefinedOutsideOfLoop(earlier.getMemRef()))
    return failure();

  AffineMap earlierMap, laterMap;
  SmallVector<Value> earlierOperands, laterOperands;
  canonicalizeAccess(earlier, earlierMap, earlierOperands);
  canonicalizeAccess(later, laterMap, laterOperands);

  auto hasUnsupportedLoopLocalOperand = [&](ArrayRef<Value> operands) {
    return llvm::any_of(operands, [&](Value operand) {
      return operand != loop.getInductionVar() &&
             !loop.isDefinedOutsideOfLoop(operand);
    });
  };
  if (hasUnsupportedLoopLocalOperand(earlierOperands) ||
      hasUnsupportedLoopLocalOperand(laterOperands))
    return failure();

  MLIRContext *context = loop.getContext();
  SmallVector<AffineExpr> dimReplacements;
  SmallVector<AffineExpr> symbolReplacements;
  for (unsigned i = 0; i < earlierMap.getNumDims(); ++i)
    dimReplacements.push_back(getAffineDimExpr(i, context));
  for (unsigned i = 0; i < earlierMap.getNumSymbols(); ++i)
    symbolReplacements.push_back(getAffineSymbolExpr(i, context));

  bool isTranslated = false;
  for (auto [index, operand] : llvm::enumerate(earlierOperands)) {
    if (operand != loop.getInductionVar())
      continue;
    isTranslated = true;
    if (index < earlierMap.getNumDims())
      dimReplacements[index] = dimReplacements[index] + loop.getStepAsInt();
    else
      symbolReplacements[index - earlierMap.getNumDims()] =
          symbolReplacements[index - earlierMap.getNumDims()] +
          loop.getStepAsInt();
  }
  AffineMap shiftedEarlierMap = earlierMap.replaceDimsAndSymbols(
      dimReplacements, symbolReplacements, earlierMap.getNumDims(),
      earlierMap.getNumSymbols());
  if (AffineValueMap(shiftedEarlierMap, earlierOperands) !=
      AffineValueMap(laterMap, laterOperands))
    return failure();

  LoadAcrossIterationMatch match;
  match.source = earlier.getMemRef();
  match.isTranslated = isTranslated;
  llvm::SmallPtrSet<Operation *, 8> seen;
  for (Value operand : earlier.getMapOperands())
    if (failed(collectAffineApplyDependencies(
            operand, loop, seen, match.earlierIndexOps, /*depth=*/0)))
      return failure();
  return match;
}

bool isSafeToPreload(AffineForOp loop, ValueRange sources,
                     ArrayRef<Operation *> prologueOps,
                     AliasAnalysis &aliasAnalysis) {
  if (loop.getLowerBoundMap().getNumResults() != 1 ||
      !hasAtLeastTwoIterations(loop) || sources.empty() || prologueOps.empty())
    return false;
  if (llvm::any_of(sources, [&](Value source) {
        return !isSourceStable(loop, source, aliasAnalysis);
      }))
    return false;

  Operation *root = prologueOps.back();
  if (!root || root->getParentOp() != loop)
    return false;
  llvm::SmallPtrSet<Operation *, 16> prologueSet(prologueOps.begin(),
                                                 prologueOps.end());
  for (Operation &operation : loop.getBody()->without_terminator()) {
    bool reachedRoot = &operation == root;
    if (!prologueSet.contains(&operation) &&
        !isa<AffineReadOpInterface, AffineWriteOpInterface>(operation) &&
        !isPure(&operation))
      return false;
    if (reachedRoot)
      return true;
  }
  return false;
}

FailureOr<AffineForOp>
materializeLoopCarriedValue(IRRewriter &rewriter, AffineForOp loop,
                            Value earlierRoot, Value laterRoot,
                            ArrayRef<Operation *> prologueOps) {
  if (prologueOps.empty() ||
      prologueOps.back() != earlierRoot.getDefiningOp() ||
      earlierRoot.getType() != laterRoot.getType())
    return failure();

  llvm::SmallPtrSet<Operation *, 16> available;
  for (Operation *operation : prologueOps) {
    if (!operation || operation->getParentOp() != loop)
      return failure();
    for (Value operand : operation->getOperands()) {
      if (operand == loop.getInductionVar() ||
          loop.isDefinedOutsideOfLoop(operand))
        continue;
      Operation *definingOp = operand.getDefiningOp();
      if (!definingOp || !available.contains(definingOp))
        return failure();
    }
    available.insert(operation);
  }

  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(loop);

  Value lowerBound;
  if (loop.hasConstantLowerBound())
    lowerBound = arith::ConstantIndexOp::create(rewriter, loop.getLoc(),
                                                loop.getConstantLowerBound());
  else
    lowerBound =
        AffineApplyOp::create(rewriter, loop.getLoc(), loop.getLowerBoundMap(),
                              loop.getLowerBoundOperands());

  IRMapping mapping;
  mapping.map(loop.getInductionVar(), lowerBound);
  SmallVector<Operation *> clonedOps;
  clonedOps.reserve(prologueOps.size());
  for (Operation *operation : prologueOps)
    clonedOps.push_back(rewriter.clone(*operation, mapping));
  Value initial = mapping.lookup(earlierRoot);

  BlockArgument carried;
  FailureOr<LoopLikeOpInterface> replacement = loop.replaceWithAdditionalYields(
      rewriter, initial, /*replaceInitOperandUsesInLoop=*/false,
      [&](OpBuilder &, Location, ArrayRef<BlockArgument> newArguments) {
        carried = newArguments.front();
        return SmallVector<Value>{laterRoot};
      });
  if (failed(replacement)) {
    for (Operation *operation : llvm::reverse(clonedOps))
      rewriter.eraseOp(operation);
    if (lowerBound.use_empty())
      rewriter.eraseOp(lowerBound.getDefiningOp());
    return failure();
  }

  earlierRoot.replaceAllUsesWith(carried);
  for (Operation *operation : llvm::reverse(prologueOps))
    if (isOpTriviallyDead(operation))
      rewriter.eraseOp(operation);
  return cast<AffineForOp>(*replacement);
}

} // namespace mlir::affine::loop_carried_reuse

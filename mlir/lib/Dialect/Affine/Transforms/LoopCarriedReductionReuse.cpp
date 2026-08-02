//===- LoopCarriedReductionReuse.cpp - Carry partial reductions ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements reuse of partial reductions over affine loads whose
// accesses are translated by one loop iteration.
//
//===----------------------------------------------------------------------===//

#include "LoopCarriedReuseUtils.h"
#include "mlir/Dialect/Affine/Transforms/Passes.h"

#include "mlir/Analysis/AliasAnalysis.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"

#include <optional>

namespace mlir {
namespace affine {
#define GEN_PASS_DEF_AFFINELOOPCARRIEDREDUCTIONREUSE
#include "mlir/Dialect/Affine/Transforms/Passes.h.inc"
} // namespace affine
} // namespace mlir

using namespace mlir;
using namespace mlir::affine;

namespace {

/// Bound the quadratic translated-load correspondence search independently of
/// target profitability policy.
constexpr unsigned maxReductionLeaves = 256;

/// Match modular integer addition. Overflow promises are rejected because
/// regrouping may change poison.
static bool isReductionCombiner(Operation *operation) {
  auto add = dyn_cast<arith::AddIOp>(operation);
  return add && add.getOverflowFlags() == arith::IntegerOverflowFlags::none;
}

struct ReductionTree {
  Operation *root;
  SmallVector<Operation *> combiners;
  SmallVector<AffineLoadOp> leaves;
};

/// Flatten a homogeneous, single-use reduction tree into affine-load leaves.
static std::optional<ReductionTree> matchReductionTree(Operation *root,
                                                       AffineForOp loop) {
  ReductionTree result{root};
  SmallVector<Operation *> worklist{root};
  llvm::SmallPtrSet<Operation *, 16> seenCombiners;
  llvm::SmallPtrSet<Operation *, 16> seenLoads;
  while (!worklist.empty()) {
    Operation *node = worklist.pop_back_val();
    if (!seenCombiners.insert(node).second || !isReductionCombiner(node))
      return std::nullopt;
    result.combiners.push_back(node);

    for (Value operand : node->getOperands()) {
      Operation *child = operand.getDefiningOp();
      if (child && child->getBlock() == loop.getBody() &&
          isReductionCombiner(child)) {
        if (!operand.hasOneUse())
          return std::nullopt;
        worklist.push_back(child);
        continue;
      }
      auto load = operand.getDefiningOp<AffineLoadOp>();
      if (!load || load->getBlock() != loop.getBody() ||
          !seenLoads.insert(load).second)
        return std::nullopt;
      result.leaves.push_back(load);
    }
  }
  if (result.leaves.size() < 4 || result.leaves.size() > maxReductionLeaves)
    return std::nullopt;
  llvm::stable_sort(result.leaves, [](AffineLoadOp lhs, AffineLoadOp rhs) {
    return lhs->isBeforeInBlock(rhs);
  });
  return result;
}

struct ProvenReductionReuse {
  Operation *root;
  SmallVector<Operation *> combiners;
  // groups[column][member], ordered from the oldest to the newest column.
  SmallVector<SmallVector<AffineLoadOp>> groups;
  SmallVector<Operation *> prologueOps;
};

/// Partition the reduction leaves into equal-length translated chains, then
/// transpose the chains into the partial reductions carried by the rewrite.
static std::optional<ProvenReductionReuse>
matchAndProveReductionReuse(ReductionTree tree, AffineForOp loop,
                            AliasAnalysis &aliasAnalysis) {
  unsigned numLeaves = tree.leaves.size();
  SmallVector<int> successor(numLeaves, -1);
  SmallVector<int> predecessor(numLeaves, -1);
  SmallVector<SmallVector<Operation *>> indexOps(numLeaves);
  llvm::SmallSetVector<Value, 4> sources;

  for (unsigned earlierIndex = 0; earlierIndex < numLeaves; ++earlierIndex) {
    for (unsigned laterIndex = 0; laterIndex < numLeaves; ++laterIndex) {
      FailureOr<loop_carried_reuse::LoadAcrossIterationMatch> match =
          loop_carried_reuse::matchLoadsAcrossOneIteration(
              tree.leaves[earlierIndex], tree.leaves[laterIndex], loop);
      if (failed(match) || !match->isTranslated)
        continue;
      // Duplicate accesses make the correspondence ambiguous. Do not choose
      // an arbitrary pairing because that would make deletion unsound.
      if (successor[earlierIndex] != -1 || predecessor[laterIndex] != -1)
        return std::nullopt;
      successor[earlierIndex] = laterIndex;
      predecessor[laterIndex] = earlierIndex;
      indexOps[earlierIndex] = std::move(match->earlierIndexOps);
      sources.insert(match->source);
    }
  }

  SmallVector<unsigned> heads;
  for (unsigned index = 0; index < numLeaves; ++index)
    if (predecessor[index] == -1)
      heads.push_back(index);
  if (heads.size() < 2)
    return std::nullopt;

  llvm::BitVector visited(numLeaves);
  SmallVector<SmallVector<unsigned>> chains;
  std::optional<unsigned> chainLength;
  for (unsigned head : heads) {
    SmallVector<unsigned> chain;
    for (int current = head; current != -1; current = successor[current]) {
      if (visited.test(current))
        return std::nullopt;
      visited.set(current);
      chain.push_back(current);
    }
    if (chain.size() < 2)
      return std::nullopt;
    if (!chainLength)
      chainLength = chain.size();
    else if (*chainLength != chain.size())
      return std::nullopt;
    chains.push_back(std::move(chain));
  }
  if (visited.count() != numLeaves)
    return std::nullopt;

  ProvenReductionReuse plan;
  plan.root = tree.root;
  plan.combiners = std::move(tree.combiners);
  plan.groups.resize(*chainLength);
  DenseMap<Operation *, unsigned> leafIndices;
  for (auto [index, load] : llvm::enumerate(tree.leaves))
    leafIndices[load] = index;
  for (ArrayRef<unsigned> chain : chains)
    for (auto [column, leafIndex] : llvm::enumerate(chain))
      plan.groups[column].push_back(tree.leaves[leafIndex]);

  llvm::SmallPtrSet<Operation *, 16> treeOperations(plan.combiners.begin(),
                                                    plan.combiners.end());
  llvm::SmallPtrSet<Operation *, 16> prologueSet;
  for (ArrayRef<AffineLoadOp> column : ArrayRef(plan.groups).drop_back()) {
    for (AffineLoadOp load : column) {
      if (llvm::any_of(load->getUsers(), [&](Operation *user) {
            return !treeOperations.contains(user);
          }))
        return std::nullopt;
      unsigned leafIndex = leafIndices.lookup(load);
      prologueSet.insert(indexOps[leafIndex].begin(),
                         indexOps[leafIndex].end());
      prologueSet.insert(load);
    }
  }
  for (Operation &operation : loop.getBody()->without_terminator())
    if (prologueSet.contains(&operation))
      plan.prologueOps.push_back(&operation);

  if (!loop_carried_reuse::isSafeToPreload(loop, sources.getArrayRef(),
                                           plan.prologueOps, aliasAnalysis))
    return std::nullopt;
  return plan;
}

static std::optional<ProvenReductionReuse>
findReductionReuse(AffineForOp loop, AliasAnalysis &aliasAnalysis) {
  for (Operation &operation : loop.getBody()->without_terminator()) {
    if (!isReductionCombiner(&operation) || operation.getResult(0).use_empty())
      continue;
    if (operation.getResult(0).hasOneUse()) {
      Operation *user = *operation.getResult(0).getUsers().begin();
      if (user->getBlock() == operation.getBlock() && isReductionCombiner(user))
        continue;
    }
    std::optional<ReductionTree> tree = matchReductionTree(&operation, loop);
    if (!tree)
      continue;
    if (std::optional<ProvenReductionReuse> plan =
            matchAndProveReductionReuse(std::move(*tree), loop, aliasAnalysis))
      return plan;
  }
  return std::nullopt;
}

static Value
buildReduction(OpBuilder &builder, Location location, ValueRange values,
               SmallVectorImpl<Operation *> *createdOps = nullptr) {
  Value result = values.front();
  for (Value value : values.drop_front()) {
    result = arith::AddIOp::create(builder, location, result, value);
    if (createdOps)
      createdOps->push_back(result.getDefiningOp());
  }
  return result;
}

static LogicalResult rewriteReductionReuse(IRRewriter &rewriter,
                                           AffineForOp loop,
                                           ProvenReductionReuse plan) {
  OpBuilder::InsertionGuard guard(rewriter);
  Location location = plan.root->getLoc();

  llvm::SmallPtrSet<Operation *, 16> available;
  for (Operation *operation : plan.prologueOps) {
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

  rewriter.setInsertionPoint(loop);
  Value lowerBound;
  if (loop.hasConstantLowerBound())
    lowerBound = arith::ConstantIndexOp::create(rewriter, location,
                                                loop.getConstantLowerBound());
  else
    lowerBound =
        AffineApplyOp::create(rewriter, location, loop.getLowerBoundMap(),
                              loop.getLowerBoundOperands());

  IRMapping mapping;
  mapping.map(loop.getInductionVar(), lowerBound);
  SmallVector<Operation *> prologue;
  for (Operation *operation : plan.prologueOps)
    prologue.push_back(rewriter.clone(*operation, mapping));

  SmallVector<Value> initialPartials;
  for (ArrayRef<AffineLoadOp> column : ArrayRef(plan.groups).drop_back()) {
    SmallVector<Value> values;
    for (AffineLoadOp load : column)
      values.push_back(mapping.lookup(load.getResult()));
    initialPartials.push_back(
        buildReduction(rewriter, location, values, &prologue));
  }

  rewriter.setInsertionPoint(plan.root);
  SmallVector<Value> newestValues(llvm::map_range(
      plan.groups.back(), [](AffineLoadOp load) { return load.getResult(); }));
  SmallVector<Operation *> newestOperations;
  Value newestPartial =
      buildReduction(rewriter, location, newestValues, &newestOperations);

  FailureOr<LoopLikeOpInterface> replacement = loop.replaceWithAdditionalYields(
      rewriter, initialPartials, /*replaceInitOperandUsesInLoop=*/false,
      [&](OpBuilder &, Location, ArrayRef<BlockArgument> newArguments) {
        SmallVector<Value> nextState;
        llvm::append_range(nextState, newArguments.drop_front());
        nextState.push_back(newestPartial);
        return nextState;
      });
  if (failed(replacement)) {
    for (Operation *operation : llvm::reverse(newestOperations))
      rewriter.eraseOp(operation);
    for (Operation *operation : llvm::reverse(prologue))
      rewriter.eraseOp(operation);
    if (lowerBound.use_empty())
      rewriter.eraseOp(lowerBound.getDefiningOp());
    return failure();
  }

  auto newLoop = cast<AffineForOp>(*replacement);
  ValueRange carried =
      newLoop.getRegionIterArgs().take_back(initialPartials.size());
  rewriter.setInsertionPoint(plan.root);
  SmallVector<Value> partials(carried.begin(), carried.end());
  partials.push_back(newestPartial);
  Value combined = buildReduction(rewriter, location, partials);
  rewriter.replaceAllUsesWith(plan.root->getResult(0), combined);

  llvm::SmallPtrSet<Operation *, 32> deadSet(plan.combiners.begin(),
                                             plan.combiners.end());
  deadSet.insert(plan.prologueOps.begin(), plan.prologueOps.end());
  SmallVector<Operation *> deadCandidates(deadSet.begin(), deadSet.end());
  bool erased = true;
  while (erased) {
    erased = false;
    for (Operation *&candidate : deadCandidates) {
      if (candidate && candidate->use_empty() &&
          (isa<AffineLoadOp>(candidate) || isOpTriviallyDead(candidate))) {
        rewriter.eraseOp(candidate);
        candidate = nullptr;
        erased = true;
      }
    }
  }
  return success();
}

struct AffineLoopCarriedReductionReuse
    : public affine::impl::AffineLoopCarriedReductionReuseBase<
          AffineLoopCarriedReductionReuse> {
  void runOnOperation() override {
    AliasAnalysis &aliasAnalysis = getAnalysis<AliasAnalysis>();
    SmallVector<AffineForOp> loops;
    getOperation().walk<WalkOrder::PostOrder>(
        [&](AffineForOp loop) { loops.push_back(loop); });

    IRRewriter rewriter(&getContext());
    for (AffineForOp loop : loops) {
      std::optional<ProvenReductionReuse> plan =
          findReductionReuse(loop, aliasAnalysis);
      if (!plan)
        continue;
      if (failed(rewriteReductionReuse(rewriter, loop, std::move(*plan)))) {
        signalPassFailure();
        return;
      }
    }
  }
};

} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::affine::createAffineLoopCarriedReductionReusePass() {
  return std::make_unique<AffineLoopCarriedReductionReuse>();
}

//===- LoopCarriedReuseUtils.h - Loop-carried reuse support ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Private analysis and materialization shared by affine loop-carried reuse
// transformations.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_LIB_DIALECT_AFFINE_TRANSFORMS_LOOPCARRIEDREUSEUTILS_H
#define MLIR_LIB_DIALECT_AFFINE_TRANSFORMS_LOOPCARRIEDREUSEUTILS_H

#include "mlir/Analysis/AliasAnalysis.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LLVM.h"

namespace mlir::affine::loop_carried_reuse {

/// A proof that evaluating `earlier` one loop iteration later accesses the
/// same element as `later` in the current iteration. `isTranslated` records
/// whether the access actually depends on the loop induction variable;
/// invariant accesses are useful as leaves of a larger translated producer.
/// `earlierIndexOps` is a topologically ordered, closed slice of loop-local
/// affine.apply operations needed to materialize `earlier` at the first
/// iteration.
struct LoadAcrossIterationMatch {
  SmallVector<Operation *> earlierIndexOps;
  Value source;
  bool isTranslated;
};

/// Match affine loads whose accesses agree after advancing `earlier` by one
/// iteration of `loop`. The match may be invariant; clients that require a
/// translated edge must check `isTranslated`. Unsupported loop-local operands
/// and uncomposable access expressions fail closed.
FailureOr<LoadAcrossIterationMatch>
matchLoadsAcrossOneIteration(AffineLoadOp earlier, AffineLoadOp later,
                             AffineForOp loop);

/// Return true only when the loop executes at least twice, every source is
/// stable, and moving `prologueOps` before the loop does not cross a blocking
/// operation in the first iteration.
bool isSafeToPreload(AffineForOp loop, ValueRange sources,
                     ArrayRef<Operation *> prologueOps,
                     AliasAnalysis &aliasAnalysis);

/// Compute `earlierRoot` at the first iteration, append its value to the loop
/// state, replace its in-loop uses with that state, and yield `laterRoot` for
/// the next iteration. `prologueOps` must be a topologically ordered, closed
/// slice ending in the operation that defines `earlierRoot`.
FailureOr<AffineForOp>
materializeLoopCarriedValue(IRRewriter &rewriter, AffineForOp loop,
                            Value earlierRoot, Value laterRoot,
                            ArrayRef<Operation *> prologueOps);

} // namespace mlir::affine::loop_carried_reuse

#endif // MLIR_LIB_DIALECT_AFFINE_TRANSFORMS_LOOPCARRIEDREUSEUTILS_H

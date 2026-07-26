//===- OnlineAttention.h - Online Attention transformation ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_LINALG_TRANSFORMS_ONLINEATTENTION_H
#define MLIR_DIALECT_LINALG_TRANSFORMS_ONLINEATTENTION_H

#include <cstdint>

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir::linalg {

/// Numerical permissions for the online Attention rewrite. These describe
/// observable floating-point behavior and must be authorized by the caller;
/// they do not replace structural matching of the source IR.
struct OnlineAttentionNumericalContract {
  Type probabilityStorageType;
  Type accumulatorType;
  bool scoresAreFinite = false;
  bool valuesAreFinite = false;
  bool allMaskedRowsAreImpossible = false;
  bool allowKeyTileReassociation = false;
  bool allowSignedZeroChange = false;
};

struct OnlineAttentionOptions {
  int64_t keyTileSize = 0;
  OnlineAttentionNumericalContract numerical;
};

struct OnlineAttentionRewriteResult {
  scf::ForOp keyLoop;
  Value result;
};

/// Rewrite the canonical tensor-semantics chain
///
///   linalg.matmul(Q, K^T) -> scale -> [additive mask] -> linalg.softmax
///       -> linalg.matmul(P, V)
///
/// into a key-tiled online-softmax loop. The QK and PV computations remain
/// structured `linalg.matmul` operations for a subsequent target-specific
/// scheduling adapter. If present, the mask must be a rank-two tensor with the
/// score shape. The numerical contract applies after scaling and masking and
/// explicitly authorizes all observable changes. The function fails without
/// modifying the IR when the chain or numerical contract is unsupported.
FailureOr<OnlineAttentionRewriteResult>
rewriteAttentionToOnlineTiles(RewriterBase &rewriter, SoftmaxOp softmax,
                              const OnlineAttentionOptions &options);

} // namespace mlir::linalg

#endif // MLIR_DIALECT_LINALG_TRANSFORMS_ONLINEATTENTION_H

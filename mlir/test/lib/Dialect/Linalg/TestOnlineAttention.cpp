//===- TestOnlineAttention.cpp - Test online Attention rewrite -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/Transforms/OnlineAttention.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/STLExtras.h"

using namespace mlir;

namespace {

constexpr StringLiteral kHintAttr = "online_attention.hint";
constexpr StringLiteral kTransformedAttr = "online_attention.transformed";
constexpr StringLiteral kAlgorithm = "stable-online-softmax";

struct ParsedHint {
  linalg::OnlineAttentionOptions options;
  DictionaryAttr numericalContract;
};

static bool hasExactKeys(DictionaryAttr dictionary,
                         ArrayRef<StringRef> expectedKeys) {
  if (!dictionary || dictionary.size() != expectedKeys.size())
    return false;
  return llvm::all_of(expectedKeys,
                      [&](StringRef key) { return dictionary.get(key); });
}

static FailureOr<ParsedHint> parseHint(func::FuncOp function) {
  auto hint = function->getAttrOfType<DictionaryAttr>(kHintAttr);
  if (!hasExactKeys(hint,
                    {"algorithm", "key_tile", "numerical_contract", "version"}))
    return failure();

  auto algorithm = hint.getAs<StringAttr>("algorithm");
  auto keyTile = hint.getAs<IntegerAttr>("key_tile");
  auto contract = hint.getAs<DictionaryAttr>("numerical_contract");
  auto version = hint.getAs<IntegerAttr>("version");
  if (!algorithm || algorithm.getValue() != kAlgorithm || !keyTile ||
      keyTile.getInt() <= 0 || !version || version.getInt() != 1)
    return failure();

  if (!hasExactKeys(contract, {"accumulation", "all_masked",
                               "probability_storage", "reassociation",
                               "score_domain", "signed_zero", "value_domain"}))
    return failure();
  auto accumulation = contract.getAs<StringAttr>("accumulation");
  auto allMasked = contract.getAs<StringAttr>("all_masked");
  auto probabilityStorage = contract.getAs<StringAttr>("probability_storage");
  auto reassociation = contract.getAs<StringAttr>("reassociation");
  auto scoreDomain = contract.getAs<StringAttr>("score_domain");
  auto signedZero = contract.getAs<StringAttr>("signed_zero");
  auto valueDomain = contract.getAs<StringAttr>("value_domain");
  if (!accumulation || accumulation.getValue() != "f32" || !allMasked ||
      allMasked.getValue() != "impossible" || !probabilityStorage ||
      probabilityStorage.getValue() != "f16" || !reassociation ||
      reassociation.getValue() != "key-tile" || !scoreDomain ||
      scoreDomain.getValue() != "finite" || !signedZero ||
      signedZero.getValue() != "ignore" || !valueDomain ||
      valueDomain.getValue() != "finite")
    return failure();

  return ParsedHint{linalg::OnlineAttentionOptions{
                        keyTile.getInt(),
                        linalg::OnlineAttentionNumericalContract{
                            Float16Type::get(function.getContext()),
                            Float32Type::get(function.getContext()),
                            /*scoresAreFinite=*/true,
                            /*valuesAreFinite=*/true,
                            /*allMaskedRowsAreImpossible=*/true,
                            /*allowKeyTileReassociation=*/true,
                            /*allowSignedZeroChange=*/true}},
                    contract};
}

struct TestLinalgAttentionToOnlineTiles
    : public PassWrapper<TestLinalgAttentionToOnlineTiles,
                         OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestLinalgAttentionToOnlineTiles)

  StringRef getArgument() const final {
    return "test-linalg-attention-to-online-tiles";
  }

  StringRef getDescription() const final {
    return "rewrite hinted canonical Linalg Attention to tiled online softmax";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<arith::ArithDialect, linalg::LinalgDialect, math::MathDialect,
                scf::SCFDialect, tensor::TensorDialect>();
  }

  void runOnOperation() override {
    func::FuncOp function = getOperation();
    FailureOr<ParsedHint> hint = parseHint(function);
    if (failed(hint) || function->hasAttr(kTransformedAttr))
      return;

    SmallVector<linalg::SoftmaxOp> softmaxOps;
    function.walk(
        [&](linalg::SoftmaxOp softmax) { softmaxOps.push_back(softmax); });
    if (softmaxOps.size() != 1)
      return;

    IRRewriter rewriter(&getContext());
    FailureOr<linalg::OnlineAttentionRewriteResult> result =
        linalg::rewriteAttentionToOnlineTiles(rewriter, softmaxOps.front(),
                                              hint->options);
    if (failed(result))
      return;

    function->setAttr(
        kTransformedAttr,
        rewriter.getDictionaryAttr(
            {rewriter.getNamedAttr("algorithm",
                                   rewriter.getStringAttr(kAlgorithm)),
             rewriter.getNamedAttr("key_tile", rewriter.getI64IntegerAttr(
                                                   hint->options.keyTileSize)),
             rewriter.getNamedAttr("numerical_contract",
                                   hint->numericalContract)}));
  }
};

} // namespace

namespace mlir::test {

void registerTestLinalgAttentionToOnlineTiles() {
  PassRegistration<TestLinalgAttentionToOnlineTiles>();
}

} // namespace mlir::test

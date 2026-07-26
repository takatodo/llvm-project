//===- OnlineAttention.cpp - Online Attention transformation -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Linalg/Transforms/OnlineAttention.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Matchers.h"

#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/STLExtras.h"

using namespace mlir;

namespace mlir::linalg {
namespace {

constexpr StringLiteral kAlgorithm = "stable-online-softmax";
constexpr StringLiteral kTileRoleAttr = "online_attention.tile_role";

struct ZeroInit {
  linalg::FillOp fill;
  tensor::EmptyOp empty;
};

struct AttentionMatch {
  linalg::MatmulOp qk;
  linalg::GenericOp scaleOp;
  linalg::GenericOp maskOp;
  linalg::SoftmaxOp softmax;
  linalg::MatmulOp pv;
  ZeroInit qkInit;
  ZeroInit pvInit;
  tensor::EmptyOp scaleEmpty;
  tensor::EmptyOp maskEmpty;
  tensor::EmptyOp probabilityEmpty;
  Value q;
  Value k;
  Value v;
  Value scale;
  Value mask;
  Value zero;
  int64_t queryCount;
  int64_t keyTile;
  OnlineAttentionNumericalContract numerical;
};

struct AdditiveMaskMatch {
  linalg::GenericOp op;
  tensor::EmptyOp empty;
  Value lhs;
  Value rhs;
};

static FailureOr<ZeroInit> matchZeroInit(Value value) {
  auto fill = value.getDefiningOp<linalg::FillOp>();
  if (!fill || fill->getNumResults() != 1 || !fill->getResult(0).hasOneUse() ||
      fill.getInputs().size() != 1 || fill.getOutputs().size() != 1 ||
      !matchPattern(fill.getInputs().front(), m_PosZeroFloat()))
    return failure();

  auto empty = fill.getOutputs().front().getDefiningOp<tensor::EmptyOp>();
  if (!empty || !empty->getResult(0).hasOneUse())
    return failure();
  return ZeroInit{fill, empty};
}

static bool hasShapeAndElementType(Value value, ArrayRef<int64_t> shape,
                                   Type elementType) {
  auto type = dyn_cast<RankedTensorType>(value.getType());
  return type && type.getShape() == shape &&
         type.getElementType() == elementType && !type.getEncoding();
}

static FailureOr<std::pair<linalg::GenericOp, Value>>
matchScale(Value value, MLIRContext *context) {
  auto generic = value.getDefiningOp<linalg::GenericOp>();
  if (!generic || generic->getNumResults() != 1 ||
      !generic->getResult(0).hasOneUse() || generic.getInputs().size() != 1 ||
      generic.getOutputs().size() != 1)
    return failure();

  AffineMap identity = AffineMap::getMultiDimIdentityMap(2, context);
  if (generic.getIndexingMapsArray() !=
          SmallVector<AffineMap>{identity, identity} ||
      generic.getIteratorTypesArray() !=
          SmallVector<utils::IteratorType>{utils::IteratorType::parallel,
                                           utils::IteratorType::parallel})
    return failure();

  Block &body = generic.getRegion().front();
  if (body.getNumArguments() != 2 ||
      std::distance(body.begin(), body.end()) != 2)
    return failure();
  auto multiply = dyn_cast<arith::MulFOp>(body.front());
  auto yield = dyn_cast<linalg::YieldOp>(body.back());
  if (!multiply || !yield || yield.getValues().size() != 1 ||
      yield.getValues().front() != multiply.getResult())
    return failure();

  Value scale;
  if (multiply.getLhs() == body.getArgument(0))
    scale = multiply.getRhs();
  else if (multiply.getRhs() == body.getArgument(0))
    scale = multiply.getLhs();
  else
    return failure();
  if (!scale.getType().isF32() ||
      scale.getParentRegion() == &generic.getRegion())
    return failure();

  return std::make_pair(generic, scale);
}

static FailureOr<AdditiveMaskMatch> matchAdditiveMask(Value value,
                                                      MLIRContext *context) {
  auto generic = value.getDefiningOp<linalg::GenericOp>();
  if (!generic || generic->getNumResults() != 1 ||
      !generic->getResult(0).hasOneUse() || generic.getInputs().size() != 2 ||
      generic.getOutputs().size() != 1)
    return failure();

  AffineMap identity = AffineMap::getMultiDimIdentityMap(2, context);
  if (generic.getIndexingMapsArray() !=
          SmallVector<AffineMap>{identity, identity, identity} ||
      generic.getIteratorTypesArray() !=
          SmallVector<utils::IteratorType>{utils::IteratorType::parallel,
                                           utils::IteratorType::parallel})
    return failure();

  Block &body = generic.getRegion().front();
  if (body.getNumArguments() != 3 ||
      std::distance(body.begin(), body.end()) != 2)
    return failure();
  auto add = dyn_cast<arith::AddFOp>(body.front());
  auto yield = dyn_cast<linalg::YieldOp>(body.back());
  if (!add || !yield || yield.getValues().size() != 1 ||
      yield.getValues().front() != add.getResult())
    return failure();

  Value lhs = body.getArgument(0);
  Value rhs = body.getArgument(1);
  if (!((add.getLhs() == lhs && add.getRhs() == rhs) ||
        (add.getLhs() == rhs && add.getRhs() == lhs)))
    return failure();

  auto empty = generic.getOutputs().front().getDefiningOp<tensor::EmptyOp>();
  if (!empty || !empty->getResult(0).hasOneUse())
    return failure();
  return AdditiveMaskMatch{generic, empty, generic.getInputs()[0],
                           generic.getInputs()[1]};
}

static FailureOr<AttentionMatch>
matchAttention(SoftmaxOp softmax, const OnlineAttentionOptions &options) {
  const OnlineAttentionNumericalContract &numerical = options.numerical;
  if (options.keyTileSize <= 0 || !numerical.probabilityStorageType ||
      !numerical.probabilityStorageType.isF16() || !numerical.accumulatorType ||
      !numerical.accumulatorType.isF32() || !numerical.scoresAreFinite ||
      !numerical.valuesAreFinite || !numerical.allMaskedRowsAreImpossible ||
      !numerical.allowKeyTileReassociation || !numerical.allowSignedZeroChange)
    return failure();

  if (!softmax.hasPureTensorSemantics() || softmax.getDimension() != 1 ||
      softmax->getNumResults() != 1 || !softmax->getResult(0).hasOneUse())
    return failure();

  auto pv = dyn_cast<linalg::MatmulOp>(*softmax->getResult(0).user_begin());
  if (!pv || !pv.hasPureTensorSemantics() || pv.getInputs().size() != 2 ||
      pv.getOutputs().size() != 1 ||
      pv.getInputs().front() != softmax->getResult(0) ||
      pv.getIndexingMapsArray() !=
          linalg::MatmulOp::getDefaultIndexingMaps(softmax.getContext()))
    return failure();

  linalg::GenericOp scaleOp;
  linalg::GenericOp maskOp;
  tensor::EmptyOp maskEmpty;
  Value scaleValue;
  Value mask;
  if (FailureOr<std::pair<linalg::GenericOp, Value>> scale =
          matchScale(softmax.getInput(), softmax.getContext());
      succeeded(scale)) {
    scaleOp = scale->first;
    scaleValue = scale->second;
  } else {
    FailureOr<AdditiveMaskMatch> additiveMask =
        matchAdditiveMask(softmax.getInput(), softmax.getContext());
    if (failed(additiveMask))
      return failure();

    FailureOr<std::pair<linalg::GenericOp, Value>> lhsScale =
        matchScale(additiveMask->lhs, softmax.getContext());
    FailureOr<std::pair<linalg::GenericOp, Value>> rhsScale =
        matchScale(additiveMask->rhs, softmax.getContext());
    if (succeeded(lhsScale) == succeeded(rhsScale))
      return failure();

    auto &matchedScale = succeeded(lhsScale) ? *lhsScale : *rhsScale;
    scaleOp = matchedScale.first;
    scaleValue = matchedScale.second;
    mask = succeeded(lhsScale) ? additiveMask->rhs : additiveMask->lhs;
    maskOp = additiveMask->op;
    maskEmpty = additiveMask->empty;
  }

  auto qk = scaleOp.getInputs().front().getDefiningOp<linalg::MatmulOp>();
  if (!qk || !qk.hasPureTensorSemantics() || qk.getInputs().size() != 2 ||
      qk.getOutputs().size() != 1 || !qk->getResult(0).hasOneUse())
    return failure();

  MLIRContext *context = softmax.getContext();
  AffineExpr qDim = getAffineDimExpr(0, context);
  AffineExpr kDim = getAffineDimExpr(1, context);
  AffineExpr reductionDim = getAffineDimExpr(2, context);
  SmallVector<AffineMap> expectedQkMaps{
      AffineMap::get(3, 0, {qDim, reductionDim}, context),
      AffineMap::get(3, 0, {kDim, reductionDim}, context),
      AffineMap::get(3, 0, {qDim, kDim}, context)};
  if (qk.getIndexingMapsArray() != expectedQkMaps)
    return failure();

  FailureOr<ZeroInit> qkInit = matchZeroInit(qk.getOutputs().front());
  FailureOr<ZeroInit> pvInit = matchZeroInit(pv.getOutputs().front());
  if (failed(qkInit) || failed(pvInit))
    return failure();

  auto scaleEmpty =
      scaleOp.getOutputs().front().getDefiningOp<tensor::EmptyOp>();
  auto probabilityEmpty = softmax.getOutput().getDefiningOp<tensor::EmptyOp>();
  if (!scaleEmpty || !probabilityEmpty ||
      !scaleEmpty->getResult(0).hasOneUse() ||
      !probabilityEmpty->getResult(0).hasOneUse())
    return failure();

  Value q = qk.getInputs()[0];
  Value k = qk.getInputs()[1];
  Value v = pv.getInputs()[1];
  auto qType = dyn_cast<RankedTensorType>(q.getType());
  auto kType = dyn_cast<RankedTensorType>(k.getType());
  auto vType = dyn_cast<RankedTensorType>(v.getType());
  if (!qType || !kType || !vType || qType.getRank() != 2 ||
      kType.getRank() != 2 || vType.getRank() != 2 ||
      !qType.getElementType().isF16() || !kType.getElementType().isF16() ||
      !vType.getElementType().isF16() || qType.getEncoding() ||
      kType.getEncoding() || vType.getEncoding() || qType.getDimSize(0) <= 0 ||
      qType.getDimSize(1) != kType.getDimSize(1) ||
      kType.getDimSize(0) != vType.getDimSize(0) || qType.getDimSize(1) <= 0 ||
      vType.getDimSize(1) <= 0)
    return failure();

  int64_t queryCount = qType.getDimSize(0);
  int64_t keyCount = kType.getDimSize(0);
  int64_t keyTile = options.keyTileSize;
  if (keyCount <= 0 || keyCount % keyTile != 0)
    return failure();

  Type f32 = Float32Type::get(context);
  if (!hasShapeAndElementType(qk->getResult(0), {queryCount, keyCount}, f32) ||
      !hasShapeAndElementType(scaleOp->getResult(0), {queryCount, keyCount},
                              f32) ||
      (mask && (!hasShapeAndElementType(mask, {queryCount, keyCount}, f32) ||
                !hasShapeAndElementType(maskOp->getResult(0),
                                        {queryCount, keyCount}, f32))) ||
      !hasShapeAndElementType(softmax->getResult(0), {queryCount, keyCount},
                              f32) ||
      !hasShapeAndElementType(pv->getResult(0),
                              {queryCount, vType.getDimSize(1)}, f32))
    return failure();

  return AttentionMatch{
      qk,         scaleOp,          maskOp,   softmax,
      pv,         *qkInit,          *pvInit,  scaleEmpty,
      maskEmpty,  probabilityEmpty, q,        k,
      v,          scaleValue,       mask,     qkInit->fill.getInputs().front(),
      queryCount, keyTile,          numerical};
}

static Value createEmpty(OpBuilder &builder, Location location,
                         ArrayRef<int64_t> shape, Type elementType) {
  return tensor::EmptyOp::create(builder, location, shape, elementType);
}

static Value createFilledTensor(OpBuilder &builder, Location location,
                                ArrayRef<int64_t> shape, Type elementType,
                                Value fillValue) {
  Value empty = createEmpty(builder, location, shape, elementType);
  return linalg::FillOp::create(builder, location, ValueRange{fillValue},
                                ValueRange{empty})
      .getResult(0);
}

static Value createScale(OpBuilder &builder, Location location, Value input,
                         Value scale) {
  auto inputType = cast<RankedTensorType>(input.getType());
  Value output = createEmpty(builder, location, inputType.getShape(),
                             inputType.getElementType());
  AffineMap identity = AffineMap::getMultiDimIdentityMap(inputType.getRank(),
                                                         builder.getContext());
  SmallVector<AffineMap> maps{identity, identity};
  SmallVector<utils::IteratorType> iterators(inputType.getRank(),
                                             utils::IteratorType::parallel);
  return linalg::GenericOp::create(
             builder, location, inputType, ValueRange{input},
             ValueRange{output}, maps, iterators,
             [&](OpBuilder &nestedBuilder, Location nestedLocation,
                 ValueRange arguments) {
               Value scaled = arith::MulFOp::create(
                   nestedBuilder, nestedLocation, arguments[0], scale);
               linalg::YieldOp::create(nestedBuilder, nestedLocation, scaled);
             })
      .getResult(0);
}

static Value createAdditiveMask(OpBuilder &builder, Location location,
                                Value input, Value mask) {
  auto inputType = cast<RankedTensorType>(input.getType());
  Value output = createEmpty(builder, location, inputType.getShape(),
                             inputType.getElementType());
  AffineMap identity = AffineMap::getMultiDimIdentityMap(inputType.getRank(),
                                                         builder.getContext());
  SmallVector<AffineMap> maps{identity, identity, identity};
  SmallVector<utils::IteratorType> iterators(inputType.getRank(),
                                             utils::IteratorType::parallel);
  return linalg::GenericOp::create(
             builder, location, inputType, ValueRange{input, mask},
             ValueRange{output}, maps, iterators,
             [&](OpBuilder &nestedBuilder, Location nestedLocation,
                 ValueRange arguments) {
               Value masked = arith::AddFOp::create(
                   nestedBuilder, nestedLocation, arguments[0], arguments[1]);
               linalg::YieldOp::create(nestedBuilder, nestedLocation, masked);
             })
      .getResult(0);
}

static Value createRowReduction(OpBuilder &builder, Location location,
                                Value input, Value initial, bool maximum) {
  MLIRContext *context = builder.getContext();
  AffineExpr row = getAffineDimExpr(0, context);
  AffineExpr column = getAffineDimExpr(1, context);
  SmallVector<AffineMap> maps{AffineMap::get(2, 0, {row, column}, context),
                              AffineMap::get(2, 0, {row}, context)};
  SmallVector<utils::IteratorType> iterators{utils::IteratorType::parallel,
                                             utils::IteratorType::reduction};
  auto resultType = cast<RankedTensorType>(initial.getType());
  return linalg::GenericOp::create(
             builder, location, resultType, ValueRange{input},
             ValueRange{initial}, maps, iterators,
             [&](OpBuilder &nestedBuilder, Location nestedLocation,
                 ValueRange arguments) {
               Value result = maximum ? Value(arith::MaxNumFOp::create(
                                            nestedBuilder, nestedLocation,
                                            arguments[0], arguments[1]))
                                      : Value(arith::AddFOp::create(
                                            nestedBuilder, nestedLocation,
                                            arguments[0], arguments[1]));
               linalg::YieldOp::create(nestedBuilder, nestedLocation, result);
             })
      .getResult(0);
}

static Value createTileWeights(OpBuilder &builder, Location location,
                               Value scores, Value rowMaximum) {
  auto scoreType = cast<RankedTensorType>(scores.getType());
  Value output = createEmpty(builder, location, scoreType.getShape(),
                             scoreType.getElementType());
  MLIRContext *context = builder.getContext();
  AffineExpr row = getAffineDimExpr(0, context);
  AffineExpr column = getAffineDimExpr(1, context);
  SmallVector<AffineMap> maps{AffineMap::get(2, 0, {row, column}, context),
                              AffineMap::get(2, 0, {row}, context),
                              AffineMap::get(2, 0, {row, column}, context)};
  SmallVector<utils::IteratorType> iterators(2, utils::IteratorType::parallel);
  return linalg::GenericOp::create(
             builder, location, scoreType, ValueRange{scores, rowMaximum},
             ValueRange{output}, maps, iterators,
             [&](OpBuilder &nestedBuilder, Location nestedLocation,
                 ValueRange arguments) {
               Value centered = arith::SubFOp::create(
                   nestedBuilder, nestedLocation, arguments[0], arguments[1]);
               Value weight =
                   math::ExpOp::create(nestedBuilder, nestedLocation, centered);
               linalg::YieldOp::create(nestedBuilder, nestedLocation, weight);
             })
      .getResult(0);
}

static Value createTruncatedWeights(OpBuilder &builder, Location location,
                                    Value weights) {
  auto sourceType = cast<RankedTensorType>(weights.getType());
  Type f16 = builder.getF16Type();
  auto resultType = RankedTensorType::get(sourceType.getShape(), f16);
  Value output = createEmpty(builder, location, resultType.getShape(), f16);
  AffineMap identity =
      AffineMap::getMultiDimIdentityMap(2, builder.getContext());
  SmallVector<AffineMap> maps{identity, identity};
  SmallVector<utils::IteratorType> iterators(2, utils::IteratorType::parallel);
  return linalg::GenericOp::create(
             builder, location, resultType, ValueRange{weights},
             ValueRange{output}, maps, iterators,
             [&](OpBuilder &nestedBuilder, Location nestedLocation,
                 ValueRange arguments) {
               Value narrowed = arith::TruncFOp::create(
                   nestedBuilder, nestedLocation, f16, arguments[0]);
               linalg::YieldOp::create(nestedBuilder, nestedLocation, narrowed);
             })
      .getResult(0);
}

static SmallVector<Value, 2> createMaximumAndScale(OpBuilder &builder,
                                                   Location location,
                                                   Value oldMaximum,
                                                   Value tileMaximum) {
  auto rowType = cast<RankedTensorType>(oldMaximum.getType());
  SmallVector<Value> outputs{oldMaximum,
                             createEmpty(builder, location, rowType.getShape(),
                                         rowType.getElementType())};
  AffineMap identity =
      AffineMap::getMultiDimIdentityMap(1, builder.getContext());
  SmallVector<AffineMap> maps(4, identity);
  SmallVector<utils::IteratorType> iterators{utils::IteratorType::parallel};
  SmallVector<Type> resultTypes(2, rowType);
  auto maximumAndScale = linalg::GenericOp::create(
      builder, location, resultTypes, ValueRange{oldMaximum, tileMaximum},
      outputs, maps, iterators,
      [&](OpBuilder &nestedBuilder, Location nestedLocation,
          ValueRange arguments) {
        Value nextMaximum = arith::MaxNumFOp::create(
            nestedBuilder, nestedLocation, arguments[0], arguments[1]);
        Value delta = arith::SubFOp::create(nestedBuilder, nestedLocation,
                                            arguments[0], nextMaximum);
        Value scale = math::ExpOp::create(nestedBuilder, nestedLocation, delta);
        linalg::YieldOp::create(nestedBuilder, nestedLocation,
                                ValueRange{nextMaximum, scale});
      });
  SmallVector<Value, 2> results;
  for (Value result : maximumAndScale.getResults())
    results.push_back(result);
  return results;
}

static Value createRescaledSum(OpBuilder &builder, Location location,
                               Value oldSum, Value tileSum, Value scale) {
  auto rowType = cast<RankedTensorType>(oldSum.getType());
  AffineMap identity =
      AffineMap::getMultiDimIdentityMap(1, builder.getContext());
  SmallVector<AffineMap> maps(4, identity);
  SmallVector<utils::IteratorType> iterators{utils::IteratorType::parallel};
  return linalg::GenericOp::create(
             builder, location, rowType, ValueRange{oldSum, tileSum, scale},
             ValueRange{oldSum}, maps, iterators,
             [&](OpBuilder &nestedBuilder, Location nestedLocation,
                 ValueRange arguments) {
               Value scaledOldSum = arith::MulFOp::create(
                   nestedBuilder, nestedLocation, arguments[0], arguments[2]);
               Value nextSum = arith::AddFOp::create(
                   nestedBuilder, nestedLocation, scaledOldSum, arguments[1]);
               linalg::YieldOp::create(nestedBuilder, nestedLocation, nextSum);
             })
      .getResult(0);
}

static Value createOutputMerge(OpBuilder &builder, Location location,
                               Value oldOutput, Value tileOutput, Value scale) {
  auto outputType = cast<RankedTensorType>(oldOutput.getType());
  MLIRContext *context = builder.getContext();
  AffineExpr row = getAffineDimExpr(0, context);
  AffineExpr column = getAffineDimExpr(1, context);
  AffineMap matrixMap = AffineMap::get(2, 0, {row, column}, context);
  AffineMap rowMap = AffineMap::get(2, 0, {row}, context);
  SmallVector<AffineMap> maps{matrixMap, matrixMap, rowMap, matrixMap};
  SmallVector<utils::IteratorType> iterators(2, utils::IteratorType::parallel);
  return linalg::GenericOp::create(
             builder, location, outputType,
             ValueRange{oldOutput, tileOutput, scale}, ValueRange{oldOutput},
             maps, iterators,
             [&](OpBuilder &nestedBuilder, Location nestedLocation,
                 ValueRange arguments) {
               Value oldScaled = arith::MulFOp::create(
                   nestedBuilder, nestedLocation, arguments[0], arguments[2]);
               Value merged = arith::AddFOp::create(
                   nestedBuilder, nestedLocation, oldScaled, arguments[1]);
               linalg::YieldOp::create(nestedBuilder, nestedLocation, merged);
             })
      .getResult(0);
}

static Value createNormalization(OpBuilder &builder, Location location,
                                 Value outputAccumulator, Value rowSum) {
  auto outputType = cast<RankedTensorType>(outputAccumulator.getType());
  Value output = createEmpty(builder, location, outputType.getShape(),
                             outputType.getElementType());
  MLIRContext *context = builder.getContext();
  AffineExpr row = getAffineDimExpr(0, context);
  AffineExpr column = getAffineDimExpr(1, context);
  AffineMap matrixMap = AffineMap::get(2, 0, {row, column}, context);
  AffineMap rowMap = AffineMap::get(2, 0, {row}, context);
  SmallVector<AffineMap> maps{matrixMap, rowMap, matrixMap};
  SmallVector<utils::IteratorType> iterators(2, utils::IteratorType::parallel);
  return linalg::GenericOp::create(
             builder, location, outputType,
             ValueRange{outputAccumulator, rowSum}, ValueRange{output}, maps,
             iterators,
             [&](OpBuilder &nestedBuilder, Location nestedLocation,
                 ValueRange arguments) {
               Value normalized = arith::DivFOp::create(
                   nestedBuilder, nestedLocation, arguments[0], arguments[1]);
               linalg::YieldOp::create(nestedBuilder, nestedLocation,
                                       normalized);
             })
      .getResult(0);
}

static OnlineAttentionRewriteResult rewriteAttention(RewriterBase &rewriter,
                                                     AttentionMatch &match) {
  Location location = match.qk.getLoc();
  // All matched operands necessarily dominate the consuming PV matmul. Mask
  // and V producers need not dominate the earlier scale operation.
  rewriter.setInsertionPoint(match.pv);
  Type f32 = rewriter.getF32Type();
  auto kType = cast<RankedTensorType>(match.k.getType());
  auto vType = cast<RankedTensorType>(match.v.getType());
  int64_t keyCount = kType.getDimSize(0);
  int64_t reductionSize = kType.getDimSize(1);
  int64_t valueSize = vType.getDimSize(1);

  llvm::APFloat negativeInfinity =
      llvm::APFloat::getInf(cast<FloatType>(f32).getFloatSemantics(),
                            /*negative=*/true);
  Value negativeInfinityValue = arith::ConstantOp::create(
      rewriter, location, FloatAttr::get(f32, negativeInfinity));
  Value zero = match.zero;
  Value maximumInit = createFilledTensor(rewriter, location, {match.queryCount},
                                         f32, negativeInfinityValue);
  Value sumInit =
      createFilledTensor(rewriter, location, {match.queryCount}, f32, zero);
  Value outputInit = createFilledTensor(
      rewriter, location, {match.queryCount, valueSize}, f32, zero);

  Value lowerBound = arith::ConstantIndexOp::create(rewriter, location, 0);
  Value upperBound =
      arith::ConstantIndexOp::create(rewriter, location, keyCount);
  Value step =
      arith::ConstantIndexOp::create(rewriter, location, match.keyTile);

  auto loop = scf::ForOp::create(
      rewriter, location, lowerBound, upperBound, step,
      ValueRange{maximumInit, sumInit, outputInit},
      [&](OpBuilder &builder, Location bodyLocation, Value keyBase,
          ValueRange state) {
        auto f16 = builder.getF16Type();
        auto keyTileType =
            RankedTensorType::get({match.keyTile, reductionSize}, f16);
        auto valueTileType =
            RankedTensorType::get({match.keyTile, valueSize}, f16);
        SmallVector<OpFoldResult> offsets{keyBase, builder.getIndexAttr(0)};
        SmallVector<OpFoldResult> keySizes{builder.getIndexAttr(match.keyTile),
                                           builder.getIndexAttr(reductionSize)};
        SmallVector<OpFoldResult> valueSizes{
            builder.getIndexAttr(match.keyTile),
            builder.getIndexAttr(valueSize)};
        SmallVector<OpFoldResult> strides{builder.getIndexAttr(1),
                                          builder.getIndexAttr(1)};
        Value keyTile =
            tensor::ExtractSliceOp::create(builder, bodyLocation, keyTileType,
                                           match.k, offsets, keySizes, strides);
        Value valueTile = tensor::ExtractSliceOp::create(
            builder, bodyLocation, valueTileType, match.v, offsets, valueSizes,
            strides);

        auto scoreType =
            RankedTensorType::get({match.queryCount, match.keyTile}, f32);
        Value scoreInit = createFilledTensor(builder, bodyLocation,
                                             scoreType.getShape(), f32, zero);
        MLIRContext *context = builder.getContext();
        AffineExpr qDim = getAffineDimExpr(0, context);
        AffineExpr kDim = getAffineDimExpr(1, context);
        AffineExpr reductionDim = getAffineDimExpr(2, context);
        SmallVector<AffineMap> qkMaps{
            AffineMap::get(3, 0, {qDim, reductionDim}, context),
            AffineMap::get(3, 0, {kDim, reductionDim}, context),
            AffineMap::get(3, 0, {qDim, kDim}, context)};
        SmallVector<NamedAttribute> qkAttributes{
            builder.getNamedAttr("indexing_maps",
                                 builder.getAffineMapArrayAttr(qkMaps)),
            builder.getNamedAttr(kTileRoleAttr, builder.getStringAttr("qk"))};
        Value scores =
            linalg::MatmulOp::create(builder, bodyLocation, scoreType,
                                     ValueRange{match.q, keyTile},
                                     ValueRange{scoreInit}, qkAttributes)
                .getResult(0);
        Value scaledScores =
            createScale(builder, bodyLocation, scores, match.scale);
        if (match.mask) {
          auto maskTileType =
              RankedTensorType::get({match.queryCount, match.keyTile}, f32);
          SmallVector<OpFoldResult> maskOffsets{builder.getIndexAttr(0),
                                                keyBase};
          SmallVector<OpFoldResult> maskSizes{
              builder.getIndexAttr(match.queryCount),
              builder.getIndexAttr(match.keyTile)};
          Value maskTile = tensor::ExtractSliceOp::create(
              builder, bodyLocation, maskTileType, match.mask, maskOffsets,
              maskSizes, strides);
          scaledScores =
              createAdditiveMask(builder, bodyLocation, scaledScores, maskTile);
        }

        Value tileMaximumInit =
            createFilledTensor(builder, bodyLocation, {match.queryCount}, f32,
                               negativeInfinityValue);
        Value tileMaximum = createRowReduction(builder, bodyLocation,
                                               scaledScores, tileMaximumInit,
                                               /*maximum=*/true);
        SmallVector<Value, 2> maximumAndScale =
            createMaximumAndScale(builder, bodyLocation, state[0], tileMaximum);
        Value nextMaximum = maximumAndScale[0];
        Value rescale = maximumAndScale[1];
        Value weights =
            createTileWeights(builder, bodyLocation, scaledScores, nextMaximum);
        Value tileSumInit = createFilledTensor(builder, bodyLocation,
                                               {match.queryCount}, f32, zero);
        Value tileSum =
            createRowReduction(builder, bodyLocation, weights, tileSumInit,
                               /*maximum=*/false);
        Value narrowedWeights =
            createTruncatedWeights(builder, bodyLocation, weights);

        auto tileOutputType =
            RankedTensorType::get({match.queryCount, valueSize}, f32);
        Value tileOutputInit = createFilledTensor(
            builder, bodyLocation, tileOutputType.getShape(), f32, zero);
        auto pv = linalg::MatmulOp::create(
            builder, bodyLocation, tileOutputType,
            ValueRange{narrowedWeights, valueTile}, ValueRange{tileOutputInit});
        pv->setAttr(kTileRoleAttr, builder.getStringAttr("pv"));

        Value nextSum = createRescaledSum(builder, bodyLocation, state[1],
                                          tileSum, rescale);
        Value nextOutput = createOutputMerge(builder, bodyLocation, state[2],
                                             pv.getResult(0), rescale);
        scf::YieldOp::create(builder, bodyLocation,
                             ValueRange{nextMaximum, nextSum, nextOutput});
      });
  loop->setAttr("online_attention.algorithm",
                rewriter.getStringAttr(kAlgorithm));
  loop->setAttr("online_attention.key_tile",
                rewriter.getI64IntegerAttr(match.keyTile));
  loop->setAttr(
      "online_attention.mask",
      rewriter.getStringAttr(match.mask ? "finite-additive" : "none"));
  DictionaryAttr numericalContract = rewriter.getDictionaryAttr(
      {rewriter.getNamedAttr("accumulation", rewriter.getStringAttr("f32")),
       rewriter.getNamedAttr("all_masked",
                             rewriter.getStringAttr("impossible")),
       rewriter.getNamedAttr("probability_storage",
                             rewriter.getStringAttr("f16")),
       rewriter.getNamedAttr("reassociation",
                             rewriter.getStringAttr("key-tile")),
       rewriter.getNamedAttr("score_domain", rewriter.getStringAttr("finite")),
       rewriter.getNamedAttr("signed_zero", rewriter.getStringAttr("ignore")),
       rewriter.getNamedAttr("value_domain",
                             rewriter.getStringAttr("finite"))});
  loop->setAttr("online_attention.numerical_contract", numericalContract);
  DictionaryAttr statefulReductionContract = rewriter.getDictionaryAttr(
      {rewriter.getNamedAttr("algorithm", rewriter.getStringAttr(kAlgorithm)),
       rewriter.getNamedAttr("finalize", rewriter.getStringAttr(
                                             "weighted-output-div-normalizer")),
       rewriter.getNamedAttr(
           "init",
           rewriter.getArrayAttr({rewriter.getStringAttr("negative-infinity"),
                                  rewriter.getStringAttr("positive-zero"),
                                  rewriter.getStringAttr("positive-zero")})),
       rewriter.getNamedAttr(
           "merge", rewriter.getStringAttr("rescaled-max-sum-weighted-output")),
       rewriter.getNamedAttr(
           "state_roles",
           rewriter.getArrayAttr({rewriter.getStringAttr("maximum"),
                                  rewriter.getStringAttr("normalizer"),
                                  rewriter.getStringAttr("weighted-output")})),
       rewriter.getNamedAttr("tail", rewriter.getStringAttr("none")),
       rewriter.getNamedAttr(
           "update", rewriter.getStringAttr("tile-local-online-softmax")),
       rewriter.getNamedAttr("update_count", rewriter.getI64IntegerAttr(
                                                 keyCount / match.keyTile)),
       rewriter.getNamedAttr("update_granularity",
                             rewriter.getI64IntegerAttr(match.keyTile)),
       rewriter.getNamedAttr("version", rewriter.getI64IntegerAttr(1))});
  loop->setAttr("online_attention.stateful_reduction",
                statefulReductionContract);

  rewriter.setInsertionPointAfter(loop);
  Value normalized = createNormalization(rewriter, location, loop.getResult(2),
                                         loop.getResult(1));
  rewriter.replaceOp(match.pv, normalized);

  rewriter.eraseOp(match.softmax);
  if (match.maskOp) {
    rewriter.eraseOp(match.maskOp);
    rewriter.eraseOp(match.maskEmpty);
  }
  rewriter.eraseOp(match.scaleOp);
  rewriter.eraseOp(match.qk);
  rewriter.eraseOp(match.probabilityEmpty);
  rewriter.eraseOp(match.scaleEmpty);
  rewriter.eraseOp(match.qkInit.fill);
  rewriter.eraseOp(match.qkInit.empty);
  rewriter.eraseOp(match.pvInit.fill);
  rewriter.eraseOp(match.pvInit.empty);

  return OnlineAttentionRewriteResult{loop, normalized};
}

} // namespace

FailureOr<OnlineAttentionRewriteResult>
rewriteAttentionToOnlineTiles(RewriterBase &rewriter, SoftmaxOp softmax,
                              const OnlineAttentionOptions &options) {
  FailureOr<AttentionMatch> match = matchAttention(softmax, options);
  if (failed(match))
    return failure();
  return rewriteAttention(rewriter, *match);
}

} // namespace mlir::linalg

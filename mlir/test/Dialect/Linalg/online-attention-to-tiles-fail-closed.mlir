// RUN: mlir-opt -test-linalg-attention-to-online-tiles %s | FileCheck %s

#qkA = affine_map<(q, k, d) -> (q, d)>
#qkB = affine_map<(q, k, d) -> (k, d)>
#qkOut = affine_map<(q, k, d) -> (q, k)>
#identity2 = affine_map<(q, k) -> (q, k)>

// A dynamic contraction dimension does not provide a static slice size. The
// transform must leave the complete canonical chain unchanged.
func.func @dynamic_reduction_dimension(
    %q: tensor<16x?xf16>,
    %k: tensor<32x?xf16>,
    %v: tensor<32x64xf16>) -> tensor<16x64xf32>
    attributes {
      online_attention.hint = {
        algorithm = "stable-online-softmax",
        key_tile = 16 : i64,
        numerical_contract = {
          accumulation = "f32",
          all_masked = "impossible",
          probability_storage = "f16",
          reassociation = "key-tile",
          score_domain = "finite",
          signed_zero = "ignore",
          value_domain = "finite"
        },
        version = 1 : i64
      }
    } {
  %zero = arith.constant 0.0 : f32
  %scale = arith.constant 1.250000e-01 : f32

  %scoreEmpty = tensor.empty() : tensor<16x32xf32>
  %scoreInit = linalg.fill ins(%zero : f32)
      outs(%scoreEmpty : tensor<16x32xf32>) -> tensor<16x32xf32>
  %scores = linalg.matmul
      indexing_maps = [#qkA, #qkB, #qkOut]
      ins(%q, %k : tensor<16x?xf16>, tensor<32x?xf16>)
      outs(%scoreInit : tensor<16x32xf32>) -> tensor<16x32xf32>

  %scaledEmpty = tensor.empty() : tensor<16x32xf32>
  %scaled = linalg.generic {
      indexing_maps = [#identity2, #identity2],
      iterator_types = ["parallel", "parallel"]}
      ins(%scores : tensor<16x32xf32>)
      outs(%scaledEmpty : tensor<16x32xf32>) {
    ^bb0(%score: f32, %unused: f32):
      %value = arith.mulf %score, %scale : f32
      linalg.yield %value : f32
  } -> tensor<16x32xf32>

  %probabilityEmpty = tensor.empty() : tensor<16x32xf32>
  %probabilities = linalg.softmax dimension(1)
      ins(%scaled : tensor<16x32xf32>)
      outs(%probabilityEmpty : tensor<16x32xf32>) -> tensor<16x32xf32>

  %outputEmpty = tensor.empty() : tensor<16x64xf32>
  %outputInit = linalg.fill ins(%zero : f32)
      outs(%outputEmpty : tensor<16x64xf32>) -> tensor<16x64xf32>
  %output = linalg.matmul
      ins(%probabilities, %v : tensor<16x32xf32>, tensor<32x64xf16>)
      outs(%outputInit : tensor<16x64xf32>) -> tensor<16x64xf32>
  return %output : tensor<16x64xf32>
}

// CHECK-LABEL: func.func @dynamic_reduction_dimension(
// CHECK-NOT: online_attention.transformed
// CHECK: linalg.matmul
// CHECK-SAME: tensor<16x?xf16>, tensor<32x?xf16>
// CHECK: linalg.softmax
// CHECK: linalg.matmul
// CHECK: return

// A negative-zero accumulator is observable for signed-zero-sensitive
// floating-point code. The rewrite synthesizes positive-zero accumulators, so
// it must not accept this chain without a contract that permits the change.
func.func @negative_zero_accumulator(
    %q: tensor<16x64xf16>,
    %k: tensor<32x64xf16>,
    %v: tensor<32x64xf16>) -> tensor<16x64xf32>
    attributes {
      online_attention.hint = {
        algorithm = "stable-online-softmax",
        key_tile = 16 : i64,
        numerical_contract = {
          accumulation = "f32",
          all_masked = "impossible",
          probability_storage = "f16",
          reassociation = "key-tile",
          score_domain = "finite",
          signed_zero = "ignore",
          value_domain = "finite"
        },
        version = 1 : i64
      }
    } {
  %negativeZero = arith.constant -0.0 : f32
  %scale = arith.constant 1.250000e-01 : f32

  %scoreEmpty = tensor.empty() : tensor<16x32xf32>
  %scoreInit = linalg.fill ins(%negativeZero : f32)
      outs(%scoreEmpty : tensor<16x32xf32>) -> tensor<16x32xf32>
  %scores = linalg.matmul
      indexing_maps = [#qkA, #qkB, #qkOut]
      ins(%q, %k : tensor<16x64xf16>, tensor<32x64xf16>)
      outs(%scoreInit : tensor<16x32xf32>) -> tensor<16x32xf32>

  %scaledEmpty = tensor.empty() : tensor<16x32xf32>
  %scaled = linalg.generic {
      indexing_maps = [#identity2, #identity2],
      iterator_types = ["parallel", "parallel"]}
      ins(%scores : tensor<16x32xf32>)
      outs(%scaledEmpty : tensor<16x32xf32>) {
    ^bb0(%score: f32, %unused: f32):
      %value = arith.mulf %score, %scale : f32
      linalg.yield %value : f32
  } -> tensor<16x32xf32>

  %probabilityEmpty = tensor.empty() : tensor<16x32xf32>
  %probabilities = linalg.softmax dimension(1)
      ins(%scaled : tensor<16x32xf32>)
      outs(%probabilityEmpty : tensor<16x32xf32>) -> tensor<16x32xf32>

  %outputEmpty = tensor.empty() : tensor<16x64xf32>
  %outputInit = linalg.fill ins(%negativeZero : f32)
      outs(%outputEmpty : tensor<16x64xf32>) -> tensor<16x64xf32>
  %output = linalg.matmul
      ins(%probabilities, %v : tensor<16x32xf32>, tensor<32x64xf16>)
      outs(%outputInit : tensor<16x64xf32>) -> tensor<16x64xf32>
  return %output : tensor<16x64xf32>
}

// CHECK-LABEL: func.func @negative_zero_accumulator(
// CHECK-NOT: online_attention.transformed
// CHECK: arith.constant -0.000000e+00
// CHECK: linalg.matmul
// CHECK: linalg.softmax
// CHECK: linalg.matmul
// CHECK: return

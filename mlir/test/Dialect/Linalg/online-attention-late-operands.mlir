// RUN: mlir-opt -test-linalg-attention-to-online-tiles %s | FileCheck %s

#qkA = affine_map<(q, k, d) -> (q, d)>
#qkB = affine_map<(q, k, d) -> (k, d)>
#qkOut = affine_map<(q, k, d) -> (q, k)>
#identity2 = affine_map<(q, k) -> (q, k)>

// Mask and V only have to dominate their original consumers. They may be
// produced after the scale and softmax respectively, so the replacement loop
// must be inserted at the original PV matmul rather than at an earlier op.
func.func @late_mask_and_value_producers(
    %q: tensor<16x64xf16>,
    %k: tensor<32x64xf16>,
    %v: tensor<32x64xf16>,
    %mask: tensor<16x32xf32>) -> tensor<16x64xf32>
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

  %lateMask = tensor.extract_slice %mask[0, 0] [16, 32] [1, 1]
      : tensor<16x32xf32> to tensor<16x32xf32>
  %maskedEmpty = tensor.empty() : tensor<16x32xf32>
  %masked = linalg.generic {
      indexing_maps = [#identity2, #identity2, #identity2],
      iterator_types = ["parallel", "parallel"]}
      ins(%scaled, %lateMask : tensor<16x32xf32>, tensor<16x32xf32>)
      outs(%maskedEmpty : tensor<16x32xf32>) {
    ^bb0(%score: f32, %bias: f32, %unused: f32):
      %value = arith.addf %score, %bias : f32
      linalg.yield %value : f32
  } -> tensor<16x32xf32>

  %probabilityEmpty = tensor.empty() : tensor<16x32xf32>
  %probabilities = linalg.softmax dimension(1)
      ins(%masked : tensor<16x32xf32>)
      outs(%probabilityEmpty : tensor<16x32xf32>) -> tensor<16x32xf32>

  %lateV = tensor.extract_slice %v[0, 0] [32, 64] [1, 1]
      : tensor<32x64xf16> to tensor<32x64xf16>
  %outputEmpty = tensor.empty() : tensor<16x64xf32>
  %outputInit = linalg.fill ins(%zero : f32)
      outs(%outputEmpty : tensor<16x64xf32>) -> tensor<16x64xf32>
  %output = linalg.matmul
      ins(%probabilities, %lateV
          : tensor<16x32xf32>, tensor<32x64xf16>)
      outs(%outputInit : tensor<16x64xf32>) -> tensor<16x64xf32>
  return %output : tensor<16x64xf32>
}

// CHECK-LABEL: func.func @late_mask_and_value_producers(
// CHECK: %[[LATE_MASK:.+]] = tensor.extract_slice
// CHECK: %[[LATE_V:.+]] = tensor.extract_slice
// CHECK: scf.for
// CHECK: tensor.extract_slice %[[LATE_V]]
// CHECK: tensor.extract_slice %[[LATE_MASK]]
// CHECK: online_attention.mask = "finite-additive"
// CHECK: return
// CHECK-NOT: linalg.softmax

// RUN: mlir-opt -test-linalg-attention-to-online-tiles %s | FileCheck %s

#qkA = affine_map<(q, k, d) -> (q, d)>
#qkB = affine_map<(q, k, d) -> (k, d)>
#qkOut = affine_map<(q, k, d) -> (q, k)>
#identity2 = affine_map<(q, k) -> (q, k)>

func.func @query_tile_32(
    %q: tensor<32x64xf16>,
    %k: tensor<32x64xf16>,
    %v: tensor<32x64xf16>) -> tensor<32x64xf32>
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

  %scoreEmpty = tensor.empty() : tensor<32x32xf32>
  %scoreInit = linalg.fill ins(%zero : f32)
      outs(%scoreEmpty : tensor<32x32xf32>) -> tensor<32x32xf32>
  %scores = linalg.matmul
      indexing_maps = [#qkA, #qkB, #qkOut]
      ins(%q, %k : tensor<32x64xf16>, tensor<32x64xf16>)
      outs(%scoreInit : tensor<32x32xf32>) -> tensor<32x32xf32>

  %scaledEmpty = tensor.empty() : tensor<32x32xf32>
  %scaled = linalg.generic {
      indexing_maps = [#identity2, #identity2],
      iterator_types = ["parallel", "parallel"]}
      ins(%scores : tensor<32x32xf32>)
      outs(%scaledEmpty : tensor<32x32xf32>) {
    ^bb0(%score: f32, %unused: f32):
      %value = arith.mulf %score, %scale : f32
      linalg.yield %value : f32
  } -> tensor<32x32xf32>

  %probabilityEmpty = tensor.empty() : tensor<32x32xf32>
  %probabilities = linalg.softmax dimension(1)
      ins(%scaled : tensor<32x32xf32>)
      outs(%probabilityEmpty : tensor<32x32xf32>) -> tensor<32x32xf32>

  %outputEmpty = tensor.empty() : tensor<32x64xf32>
  %outputInit = linalg.fill ins(%zero : f32)
      outs(%outputEmpty : tensor<32x64xf32>) -> tensor<32x64xf32>
  %output = linalg.matmul
      ins(%probabilities, %v : tensor<32x32xf32>, tensor<32x64xf16>)
      outs(%outputInit : tensor<32x64xf32>) -> tensor<32x64xf32>
  return %output : tensor<32x64xf32>
}

// CHECK-LABEL: func.func @query_tile_32(
// CHECK-SAME: online_attention.transformed
// CHECK-NOT: tensor<32x32xf32>
// CHECK: %[[LOOP:.+]]:3 = scf.for
// CHECK: linalg.matmul
// CHECK-SAME: tensor<32x64xf16>, tensor<16x64xf16>
// CHECK-SAME: tensor<32x16xf32>
// CHECK: linalg.matmul
// CHECK-SAME: tensor<32x16xf16>, tensor<16x64xf16>
// CHECK-SAME: tensor<32x64xf32>
// CHECK: return
// CHECK-NOT: linalg.softmax

// RUN: mlir-opt -test-linalg-attention-to-online-tiles %s | FileCheck %s
// RUN: mlir-opt -test-linalg-attention-to-online-tiles \
// RUN:   -test-linalg-attention-to-online-tiles %s | \
// RUN:   FileCheck %s --check-prefix=IDEMPOTENT

#qkA = affine_map<(q, k, d) -> (q, d)>
#qkB = affine_map<(q, k, d) -> (k, d)>
#qkOut = affine_map<(q, k, d) -> (q, k)>
#identity2 = affine_map<(q, k) -> (q, k)>

// The numerical contract deliberately exposes the two observable differences
// from the canonical expression: regrouping by 16-key tiles and narrowing
// tile-local probabilities to f16 for the PV contraction.
func.func @canonical_hinted(
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

// CHECK-LABEL: func.func @canonical_hinted(
// CHECK-SAME: online_attention.hint
// CHECK-SAME: online_attention.transformed
// CHECK-NOT: tensor<16x32xf32>
// CHECK: %[[LOOP:.+]]:3 = scf.for
// CHECK-SAME: step
// CHECK-SAME: iter_args
// CHECK: tensor.extract_slice
// CHECK-SAME: tensor<32x64xf16> to tensor<16x64xf16>
// CHECK: tensor.extract_slice
// CHECK-SAME: tensor<32x64xf16> to tensor<16x64xf16>
// CHECK: linalg.matmul
// CHECK-SAME: tensor<16x64xf16>, tensor<16x64xf16>
// CHECK-SAME: tensor<16x16xf32>
// CHECK: arith.truncf
// CHECK: linalg.matmul
// CHECK-SAME: tensor<16x16xf16>, tensor<16x64xf16>
// CHECK-SAME: tensor<16x64xf32>
// CHECK: scf.yield
// CHECK: } {online_attention.algorithm = "stable-online-softmax"
// CHECK-SAME: online_attention.mask = "none"
// CHECK-SAME: online_attention.numerical_contract = {
// CHECK-SAME: accumulation = "f32"
// CHECK-SAME: all_masked = "impossible"
// CHECK-SAME: score_domain = "finite"
// CHECK-SAME: signed_zero = "ignore"
// CHECK-SAME: value_domain = "finite"
// CHECK-SAME: online_attention.stateful_reduction = {
// CHECK-SAME: finalize = "weighted-output-div-normalizer"
// CHECK-SAME: merge = "rescaled-max-sum-weighted-output"
// CHECK-SAME: state_roles = ["maximum", "normalizer", "weighted-output"]
// CHECK-SAME: tail = "none"
// CHECK-SAME: update = "tile-local-online-softmax"
// CHECK-SAME: update_count = 2
// CHECK-SAME: update_granularity = 16
// CHECK-SAME: version = 1
// CHECK: return
// CHECK-NOT: linalg.softmax

// IDEMPOTENT-LABEL: func.func @canonical_hinted(
// IDEMPOTENT-COUNT-1: online_attention.transformed
// IDEMPOTENT-COUNT-1: scf.for
// IDEMPOTENT-NOT: linalg.softmax

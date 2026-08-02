// RUN: mlir-opt %s --transform-interpreter --split-input-file \
// RUN:   --verify-diagnostics --canonicalize --cse | FileCheck %s

func.func @coupled_logsumexp(
    %maximums: tensor<16xf32>, %sums: tensor<16xf32>,
    %maximumInit: tensor<f32>, %sumInit: tensor<f32>)
    -> (tensor<f32>, tensor<f32>) {
  %state:2 = linalg.reduce
      ins(%maximums, %sums : tensor<16xf32>, tensor<16xf32>)
      outs(%maximumInit, %sumInit : tensor<f32>, tensor<f32>)
      dimensions = [0]
      {partial_reduction_contract = "associative_commutative_identity"}
      (%leftMaximum: f32, %leftSum: f32,
       %rightMaximum: f32, %rightSum: f32) {
        %maximum = arith.maximumf %leftMaximum, %rightMaximum : f32
        %leftDelta = arith.subf %leftMaximum, %maximum : f32
        %leftScale = math.exp %leftDelta : f32
        %rightDelta = arith.subf %rightMaximum, %maximum : f32
        %rightScale = math.exp %rightDelta : f32
        %scaledLeft = arith.mulf %leftScale, %leftSum : f32
        %scaledRight = arith.mulf %rightScale, %rightSum : f32
        %sum = arith.addf %scaledLeft, %scaledRight : f32
        linalg.yield %maximum, %sum : f32, f32
      }
  return %state#0, %state#1 : tensor<f32>, tensor<f32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(
      %root: !transform.any_op {transform.readonly}) {
    %target = transform.structured.match ops{["linalg.reduce"]} in %root
      : (!transform.any_op) -> !transform.any_op
    %maximumIdentity, %sumIdentity, %partial, %merge, %loop =
      transform.structured.tile_reduction_using_for %target
        by tile_sizes = [4]
        : (!transform.any_op) -> (!transform.any_op, !transform.any_op,
            !transform.any_op, !transform.any_op, !transform.any_op)
    transform.yield
  }
}

// CHECK-LABEL: func.func @coupled_logsumexp
// CHECK:       scf.for
// CHECK:         linalg.generic
// CHECK:           math.exp
// CHECK:       linalg.reduce
// CHECK-SAME:    partial_reduction_contract
// CHECK:         math.exp

// -----

func.func @coupled_complex_product_using_forall(
    %realParts: tensor<16xi64>, %imaginaryParts: tensor<16xi64>,
    %realIdentity: tensor<i64>, %imaginaryIdentity: tensor<i64>)
    -> (tensor<i64>, tensor<i64>) {
  %product:2 = linalg.reduce
      ins(%realParts, %imaginaryParts : tensor<16xi64>, tensor<16xi64>)
      outs(%realIdentity, %imaginaryIdentity : tensor<i64>, tensor<i64>)
      dimensions = [0]
      {partial_reduction_contract = "associative_commutative_identity"}
      (%leftReal: i64, %leftImaginary: i64,
       %rightReal: i64, %rightImaginary: i64) {
        %realProduct = arith.muli %leftReal, %rightReal : i64
        %imaginaryProduct =
          arith.muli %leftImaginary, %rightImaginary : i64
        %real = arith.subi %realProduct, %imaginaryProduct : i64
        %crossLeft = arith.muli %leftReal, %rightImaginary : i64
        %crossRight = arith.muli %leftImaginary, %rightReal : i64
        %imaginary = arith.addi %crossLeft, %crossRight : i64
        linalg.yield %real, %imaginary : i64, i64
      }
  return %product#0, %product#1 : tensor<i64>, tensor<i64>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(
      %root: !transform.any_op {transform.readonly}) {
    %target = transform.structured.match ops{["linalg.reduce"]} in %root
      : (!transform.any_op) -> !transform.any_op
    %realIdentity, %imaginaryIdentity, %partial, %merge, %forall =
      transform.structured.tile_reduction_using_forall %target
        by num_threads = [4] tile_sizes = []
        : (!transform.any_op) -> (!transform.any_op, !transform.any_op,
            !transform.any_op, !transform.any_op, !transform.any_op)
    transform.yield
  }
}

// CHECK-LABEL: func.func @coupled_complex_product_using_forall
// CHECK:       scf.forall
// CHECK:         linalg.reduce
// CHECK:           arith.muli
// CHECK:       linalg.reduce
// CHECK-SAME:    partial_reduction_contract
// CHECK:         arith.muli

// -----

func.func @dynamic_multidimensional_tail(
    %realParts: tensor<?x10xi64>, %imaginaryParts: tensor<?x10xi64>,
    %realIdentity: tensor<i64>, %imaginaryIdentity: tensor<i64>)
    -> (tensor<i64>, tensor<i64>) {
  %product:2 = linalg.reduce
      ins(%realParts, %imaginaryParts : tensor<?x10xi64>, tensor<?x10xi64>)
      outs(%realIdentity, %imaginaryIdentity : tensor<i64>, tensor<i64>)
      dimensions = [0, 1]
      {partial_reduction_contract = "associative_commutative_identity"}
      (%leftReal: i64, %leftImaginary: i64,
       %rightReal: i64, %rightImaginary: i64) {
        %realProduct = arith.muli %leftReal, %rightReal : i64
        %imaginaryProduct =
          arith.muli %leftImaginary, %rightImaginary : i64
        %real = arith.subi %realProduct, %imaginaryProduct : i64
        %crossLeft = arith.muli %leftReal, %rightImaginary : i64
        %crossRight = arith.muli %leftImaginary, %rightReal : i64
        %imaginary = arith.addi %crossLeft, %crossRight : i64
        linalg.yield %real, %imaginary : i64, i64
      }
  return %product#0, %product#1 : tensor<i64>, tensor<i64>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(
      %root: !transform.any_op {transform.readonly}) {
    %target = transform.structured.match ops{["linalg.reduce"]} in %root
      : (!transform.any_op) -> !transform.any_op
    %realIdentity, %imaginaryIdentity, %partial, %merge, %loops =
      transform.structured.tile_reduction_using_for %target
        by tile_sizes = [3, 4]
        : (!transform.any_op) -> (!transform.any_op, !transform.any_op,
            !transform.any_op, !transform.any_op, !transform.any_op)
    transform.yield
  }
}

// The dynamic first dimension may be zero at runtime.  The static second
// dimension deliberately has a tail because 10 is not divisible by 4.
// CHECK-LABEL: func.func @dynamic_multidimensional_tail
// CHECK:       tensor.dim
// CHECK:       scf.for
// CHECK:         scf.for
// CHECK:       linalg.reduce
// CHECK-SAME:    partial_reduction_contract

// -----

func.func @zero_extent(
    %left: tensor<0x10xi32>, %right: tensor<0x10xi32>,
    %leftIdentity: tensor<i32>, %rightIdentity: tensor<i32>)
    -> (tensor<i32>, tensor<i32>) {
  %state:2 = linalg.reduce
      ins(%left, %right : tensor<0x10xi32>, tensor<0x10xi32>)
      outs(%leftIdentity, %rightIdentity : tensor<i32>, tensor<i32>)
      dimensions = [0, 1]
      {partial_reduction_contract = "associative_commutative_identity"}
      (%leftValue: i32, %rightValue: i32,
       %leftState: i32, %rightState: i32) {
        %nextLeft = arith.addi %leftValue, %leftState : i32
        %nextRight = arith.addi %rightValue, %rightState : i32
        linalg.yield %nextLeft, %nextRight : i32, i32
      }
  return %state#0, %state#1 : tensor<i32>, tensor<i32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(
      %root: !transform.any_op {transform.readonly}) {
    %target = transform.structured.match ops{["linalg.reduce"]} in %root
      : (!transform.any_op) -> !transform.any_op
    %leftIdentity, %rightIdentity, %partial, %merge, %loops =
      transform.structured.tile_reduction_using_for %target
        by tile_sizes = [3, 4]
        : (!transform.any_op) -> (!transform.any_op, !transform.any_op,
            !transform.any_op, !transform.any_op, !transform.any_op)
    transform.yield
  }
}

// CHECK-LABEL: func.func @zero_extent
// CHECK:       linalg.broadcast
// CHECK:       linalg.broadcast
// CHECK:       linalg.reduce
// CHECK-SAME:    partial_reduction_contract

// -----

func.func @buffer_semantics_fails_closed(
    %values: memref<16xi32>, %identity: memref<i32>) {
  // expected-error @below {{'linalg.reduce' op expected operation to have tensor semantics}}
  linalg.reduce
      ins(%values : memref<16xi32>)
      outs(%identity : memref<i32>)
      dimensions = [0]
      {partial_reduction_contract = "associative_commutative_identity"}
      (%value: i32, %state: i32) {
        %next = arith.addi %value, %state : i32
        linalg.yield %next : i32
      }
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(
      %root: !transform.any_op {transform.readonly}) {
    %target = transform.structured.match ops{["linalg.reduce"]} in %root
      : (!transform.any_op) -> !transform.any_op
    // expected-error@+2 {{failed to tile using partial reduction}}
    %identity, %partial, %merge, %loop =
      transform.structured.tile_reduction_using_for %target
        by tile_sizes = [4]
        : (!transform.any_op) -> (!transform.any_op, !transform.any_op,
            !transform.any_op, !transform.any_op)
    transform.yield
  }
}

// -----

func.func private @unknown_effect(%value: i32) -> i32

func.func @nested_unknown_effect_fails_closed(
    %values: tensor<16xi32>, %identity: tensor<i32>) -> tensor<i32> {
  // expected-error @below {{'linalg.reduce' op coupled partial-reduction combiner must be memory-effect-free}}
  %sum = linalg.reduce
      ins(%values : tensor<16xi32>)
      outs(%identity : tensor<i32>)
      dimensions = [0]
      {partial_reduction_contract = "associative_commutative_identity"}
      (%value: i32, %state: i32) {
        %next = scf.execute_region -> i32 {
          %called = func.call @unknown_effect(%value) : (i32) -> i32
          scf.yield %called : i32
        }
        %merged = arith.addi %state, %next : i32
        linalg.yield %merged : i32
      }
  return %sum : tensor<i32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(
      %root: !transform.any_op {transform.readonly}) {
    %target = transform.structured.match ops{["linalg.reduce"]} in %root
      : (!transform.any_op) -> !transform.any_op
    // expected-error@+2 {{failed to tile using partial reduction}}
    %identity, %partial, %merge, %loop =
      transform.structured.tile_reduction_using_for %target
        by tile_sizes = [4]
        : (!transform.any_op) -> (!transform.any_op, !transform.any_op,
            !transform.any_op, !transform.any_op)
    transform.yield
  }
}

// -----

func.func @missing_contract_fails_closed(
    %left: tensor<16xf32>, %right: tensor<16xf32>,
    %leftInit: tensor<f32>, %rightInit: tensor<f32>)
    -> (tensor<f32>, tensor<f32>) {
  // expected-error @below {{'linalg.reduce' op Failed to anaysis the reduction operation.}}
  %state:2 = linalg.reduce
      ins(%left, %right : tensor<16xf32>, tensor<16xf32>)
      outs(%leftInit, %rightInit : tensor<f32>, tensor<f32>)
      dimensions = [0]
      (%leftValue: f32, %rightValue: f32,
       %leftState: f32, %rightState: f32) {
        %both = arith.addf %leftValue, %rightValue : f32
        %nextLeft = arith.addf %leftState, %both : f32
        %nextRight = arith.addf %rightState, %nextLeft : f32
        linalg.yield %nextLeft, %nextRight : f32, f32
      }
  return %state#0, %state#1 : tensor<f32>, tensor<f32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(
      %root: !transform.any_op {transform.readonly}) {
    %target = transform.structured.match ops{["linalg.reduce"]} in %root
      : (!transform.any_op) -> !transform.any_op
    // expected-error@+2 {{failed to tile using partial reduction}}
    %leftIdentity, %rightIdentity, %partial, %leftMerge, %rightMerge, %loop =
      transform.structured.tile_reduction_using_for %target
        by tile_sizes = [4]
        : (!transform.any_op) -> (!transform.any_op, !transform.any_op,
            !transform.any_op, !transform.any_op, !transform.any_op,
            !transform.any_op)
    transform.yield
  }
}

// -----

func.func @effectful_combiner_fails_closed(
    %values: tensor<16xi32>, %init: tensor<i32>, %sink: memref<1xi32>)
    -> tensor<i32> {
  // expected-error @below {{'linalg.reduce' op coupled partial-reduction combiner must be memory-effect-free}}
  %sum = linalg.reduce
      ins(%values : tensor<16xi32>)
      outs(%init : tensor<i32>)
      dimensions = [0]
      {partial_reduction_contract = "associative_commutative_identity"}
      (%value: i32, %state: i32) {
        %zero = arith.constant 0 : index
        memref.store %value, %sink[%zero] : memref<1xi32>
        %next = arith.addi %value, %state : i32
        linalg.yield %next : i32
      }
  return %sum : tensor<i32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(
      %root: !transform.any_op {transform.readonly}) {
    %target = transform.structured.match ops{["linalg.reduce"]} in %root
      : (!transform.any_op) -> !transform.any_op
    // expected-error@+2 {{failed to tile using partial reduction}}
    %identity, %partial, %merge, %loop =
      transform.structured.tile_reduction_using_for %target
        by tile_sizes = [4]
        : (!transform.any_op) -> (!transform.any_op, !transform.any_op,
            !transform.any_op, !transform.any_op)
    transform.yield
  }
}

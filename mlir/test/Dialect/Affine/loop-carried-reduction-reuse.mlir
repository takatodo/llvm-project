// RUN: mlir-opt %s --affine-loop-carried-reduction-reuse > %t.once
// RUN: mlir-opt %t.once --affine-loop-carried-reduction-reuse > %t.twice
// RUN: diff %t.once %t.twice
// RUN: mlir-opt %t.once --canonicalize --cse | FileCheck %s

// Two translated member chains become two carried column reductions. The
// steady-state loop retains only the newest column's two loads.

// CHECK-LABEL: func.func @box2x3
// CHECK: %[[A0:.*]] = affine.load %[[IN:.*]][0, 0]
// CHECK: %[[A1:.*]] = affine.load %[[IN]][1, 0]
// CHECK: %[[B0:.*]] = affine.load %[[IN]][0, 1]
// CHECK: %[[B1:.*]] = affine.load %[[IN]][1, 1]
// CHECK: %[[PA:.*]] = arith.addi %[[A0]], %[[A1]]
// CHECK: %[[PB:.*]] = arith.addi %[[B0]], %[[B1]]
// CHECK: affine.for %[[I:.*]] = 0 to 4
// CHECK-SAME: iter_args(%[[CA:.*]] = %[[PA]], %[[CB:.*]] = %[[PB]])
// CHECK-NEXT: %[[C0:.*]] = affine.load %[[IN]][0, %[[I]] + 2]
// CHECK-NEXT: %[[C1:.*]] = affine.load %[[IN]][1, %[[I]] + 2]
// CHECK-NEXT: %[[PC:.*]] = arith.addi %[[C0]], %[[C1]]
// CHECK-NEXT: %[[AB:.*]] = arith.addi %[[CA]], %[[CB]]
// CHECK-NEXT: %[[SUM:.*]] = arith.addi %[[AB]], %[[PC]]
// CHECK-NEXT: affine.store %[[SUM]]
// CHECK-NEXT: affine.yield %[[CB]], %[[PC]]
func.func @box2x3(%in0: memref<2x6xi32>, %out0: memref<4xi32>) {
  %in, %out = memref.distinct_objects %in0, %out0
      : memref<2x6xi32>, memref<4xi32>
  affine.for %i = 0 to 4 {
    %a0 = affine.load %in[0, %i] : memref<2x6xi32>
    %a1 = affine.load %in[1, %i] : memref<2x6xi32>
    %b0 = affine.load %in[0, %i + 1] : memref<2x6xi32>
    %b1 = affine.load %in[1, %i + 1] : memref<2x6xi32>
    %c0 = affine.load %in[0, %i + 2] : memref<2x6xi32>
    %c1 = affine.load %in[1, %i + 2] : memref<2x6xi32>
    %s0 = arith.addi %a0, %a1 : i32
    %s1 = arith.addi %s0, %b0 : i32
    %s2 = arith.addi %s1, %b1 : i32
    %s3 = arith.addi %s2, %c0 : i32
    %sum = arith.addi %s3, %c1 : i32
    affine.store %sum, %out[%i] : memref<4xi32>
  }
  return
}

// Access comparison composes affine.apply and uses the loop step rather than
// assuming a unit index translation.

// CHECK-LABEL: func.func @step_and_affine_apply
// CHECK: affine.for %[[I:.*]] = 0 to 8 step 2 iter_args
// CHECK-COUNT-2: affine.load
// CHECK: affine.yield
func.func @step_and_affine_apply(%in0: memref<2x10xi32>,
                                 %out0: memref<4xi32>) {
  %in, %out = memref.distinct_objects %in0, %out0
      : memref<2x10xi32>, memref<4xi32>
  affine.for %i = 0 to 8 step 2 {
    %j0 = affine.apply affine_map<(d0) -> (d0)>(%i)
    %j1 = affine.apply affine_map<(d0) -> (d0 + 2)>(%i)
    %a0 = affine.load %in[0, %j0] : memref<2x10xi32>
    %a1 = affine.load %in[1, %j0] : memref<2x10xi32>
    %b0 = affine.load %in[0, %j1] : memref<2x10xi32>
    %b1 = affine.load %in[1, %j1] : memref<2x10xi32>
    %s0 = arith.addi %a0, %a1 : i32
    %s1 = arith.addi %s0, %b0 : i32
    %sum = arith.addi %s1, %b1 : i32
    %oi = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%i)
    affine.store %sum, %out[%oi] : memref<4xi32>
  }
  return
}

// Overflow promises make integer regrouping invalid.

// CHECK-LABEL: func.func @integer_overflow_contract
// CHECK: affine.for
// CHECK-NOT: iter_args
func.func @integer_overflow_contract(%in0: memref<2x5xi32>,
                                     %out0: memref<4xi32>) {
  %in, %out = memref.distinct_objects %in0, %out0
      : memref<2x5xi32>, memref<4xi32>
  affine.for %i = 0 to 4 {
    %a0 = affine.load %in[0, %i] : memref<2x5xi32>
    %a1 = affine.load %in[1, %i] : memref<2x5xi32>
    %b0 = affine.load %in[0, %i + 1] : memref<2x5xi32>
    %b1 = affine.load %in[1, %i + 1] : memref<2x5xi32>
    %p0 = arith.addi %a0, %a1 overflow<nsw> : i32
    %p1 = arith.addi %p0, %b0 overflow<nsw> : i32
    %sum = arith.addi %p1, %b1 overflow<nsw> : i32
    affine.store %sum, %out[%i] : memref<4xi32>
  }
  return
}

// A one-member chain belongs to raw load carry, not partial-reduction reuse.

// CHECK-LABEL: func.func @one_member
// CHECK: affine.for
// CHECK-NOT: iter_args
func.func @one_member(%in0: memref<5xi32>, %out0: memref<4xi32>) {
  %in, %out = memref.distinct_objects %in0, %out0
      : memref<5xi32>, memref<4xi32>
  affine.for %i = 0 to 4 {
    %a = affine.load %in[%i] : memref<5xi32>
    %b = affine.load %in[%i + 1] : memref<5xi32>
    %sum = arith.addi %a, %b : i32
    affine.store %sum, %out[%i] : memref<4xi32>
  }
  return
}

// Keeping a leading load alive outside the tree prevents a guaranteed net
// load reduction, so the candidate is rejected.

// CHECK-LABEL: func.func @leading_external_use
// CHECK: affine.for
// CHECK-NOT: iter_args
func.func @leading_external_use(%in0: memref<2x5xi32>,
                                %out0: memref<4xi32>,
                                %side0: memref<4xi32>) {
  %in, %out, %side = memref.distinct_objects %in0, %out0, %side0
      : memref<2x5xi32>, memref<4xi32>, memref<4xi32>
  affine.for %i = 0 to 4 {
    %a0 = affine.load %in[0, %i] : memref<2x5xi32>
    %a1 = affine.load %in[1, %i] : memref<2x5xi32>
    %b0 = affine.load %in[0, %i + 1] : memref<2x5xi32>
    %b1 = affine.load %in[1, %i + 1] : memref<2x5xi32>
    %p0 = arith.addi %a0, %a1 : i32
    %p1 = arith.addi %p0, %b0 : i32
    %sum = arith.addi %p1, %b1 : i32
    affine.store %sum, %out[%i] : memref<4xi32>
    affine.store %a0, %side[%i] : memref<4xi32>
  }
  return
}

// A preload must not cross a potentially non-terminating prefix operation.

// CHECK-LABEL: func.func @non_speculatable_prefix
// CHECK: affine.for
// CHECK-NOT: iter_args
func.func @non_speculatable_prefix(%in0: memref<?x?xi32>,
                                   %out0: memref<2xi32>, %continue: i1) {
  %in, %out = memref.distinct_objects %in0, %out0
      : memref<?x?xi32>, memref<2xi32>
  affine.for %i = 0 to 2 {
    scf.while : () -> () {
      scf.condition(%continue)
    } do {
      scf.yield
    }
    %a0 = affine.load %in[0, %i] : memref<?x?xi32>
    %a1 = affine.load %in[1, %i] : memref<?x?xi32>
    %b0 = affine.load %in[0, %i + 1] : memref<?x?xi32>
    %b1 = affine.load %in[1, %i + 1] : memref<?x?xi32>
    %p0 = arith.addi %a0, %a1 : i32
    %p1 = arith.addi %p0, %b0 : i32
    %sum = arith.addi %p1, %b1 : i32
    affine.store %sum, %out[%i] : memref<2xi32>
  }
  return
}

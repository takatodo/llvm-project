// RUN: mlir-opt %s --pass-pipeline='builtin.module(func.func(affine-loop-carried-reduction-reuse,affine-loop-carried-computation-reuse),canonicalize,cse)' > %t.partial-first
// RUN: mlir-opt %s --pass-pipeline='builtin.module(func.func(affine-loop-carried-computation-reuse,affine-loop-carried-reduction-reuse),canonicalize,cse)' > %t.computation-first
// RUN: diff %t.partial-first %t.computation-first
// RUN: FileCheck %s < %t.partial-first

// The reduction adapter owns the homogeneous tree, while generic computation
// reuse remains a no-op before and after it. The two orders must therefore
// reach the same canonical form rather than introduce nested carried states.

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

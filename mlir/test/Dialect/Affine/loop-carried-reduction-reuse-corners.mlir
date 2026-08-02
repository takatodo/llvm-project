// RUN: mlir-opt %s --affine-loop-carried-reduction-reuse | FileCheck %s

#lb = affine_map<(d0) -> (d0)>
#ub_minus1 = affine_map<(d0) -> (d0 - 1)>

// Existing loop-carried state remains first and keeps its result semantics.

// CHECK-LABEL: func.func @existing_iter_arg
// CHECK: %[[RESULT:.*]]:2 = affine.for
// CHECK-SAME: iter_args(%[[OLD:.*]] = %{{.*}}, %[[PARTIAL:.*]] = %{{.*}})
// CHECK: affine.yield %{{.*}}, %{{.*}}
// CHECK: return %[[RESULT]]#0
func.func @existing_iter_arg(%in0: memref<2x5xi32>,
                             %out0: memref<4xi32>) -> i32 {
  %in, %out = memref.distinct_objects %in0, %out0
      : memref<2x5xi32>, memref<4xi32>
  %zero = arith.constant 0 : i32
  %result = affine.for %i = 0 to 4 iter_args(%acc = %zero) -> i32 {
    %a0 = affine.load %in[0, %i] : memref<2x5xi32>
    %a1 = affine.load %in[1, %i] : memref<2x5xi32>
    %b0 = affine.load %in[0, %i + 1] : memref<2x5xi32>
    %b1 = affine.load %in[1, %i + 1] : memref<2x5xi32>
    %p0 = arith.addi %a0, %a1 : i32
    %p1 = arith.addi %p0, %b0 : i32
    %sum = arith.addi %p1, %b1 : i32
    affine.store %sum, %out[%i] : memref<4xi32>
    %next = arith.xori %acc, %sum : i32
    affine.yield %next : i32
  }
  return %result : i32
}

// Member chains may come from different proven-distinct stable sources.

// CHECK-LABEL: func.func @different_sources
// CHECK: affine.for {{.*}} iter_args
func.func @different_sources(%a0: memref<5xi32>, %b0: memref<5xi32>,
                             %out0: memref<4xi32>) {
  %a, %b, %out = memref.distinct_objects %a0, %b0, %out0
      : memref<5xi32>, memref<5xi32>, memref<4xi32>
  affine.for %i = 0 to 4 {
    %aLeft = affine.load %a[%i] : memref<5xi32>
    %bLeft = affine.load %b[%i] : memref<5xi32>
    %aRight = affine.load %a[%i + 1] : memref<5xi32>
    %bRight = affine.load %b[%i + 1] : memref<5xi32>
    %p0 = arith.addi %aLeft, %bLeft : i32
    %p1 = arith.addi %p0, %aRight : i32
    %sum = arith.addi %p1, %bRight : i32
    affine.store %sum, %out[%i] : memref<4xi32>
  }
  return
}

// Every member chain must cover the same number of translated positions.

// CHECK-LABEL: func.func @unequal_chain_lengths
// CHECK: affine.for
// CHECK-NOT: iter_args
func.func @unequal_chain_lengths(%in0: memref<2x6xi32>,
                                 %out0: memref<4xi32>) {
  %in, %out = memref.distinct_objects %in0, %out0
      : memref<2x6xi32>, memref<4xi32>
  affine.for %i = 0 to 4 {
    %a0 = affine.load %in[0, %i] : memref<2x6xi32>
    %a1 = affine.load %in[0, %i + 1] : memref<2x6xi32>
    %a2 = affine.load %in[0, %i + 2] : memref<2x6xi32>
    %b0 = affine.load %in[1, %i] : memref<2x6xi32>
    %b1 = affine.load %in[1, %i + 1] : memref<2x6xi32>
    %p0 = arith.addi %a0, %a1 : i32
    %p1 = arith.addi %p0, %a2 : i32
    %p2 = arith.addi %p1, %b0 : i32
    %sum = arith.addi %p2, %b1 : i32
    affine.store %sum, %out[%i] : memref<4xi32>
  }
  return
}

// Duplicate accesses have more than one possible translated pairing.

// CHECK-LABEL: func.func @ambiguous_pairing
// CHECK: affine.for
// CHECK-NOT: iter_args
func.func @ambiguous_pairing(%in0: memref<5xi32>, %out0: memref<4xi32>) {
  %in, %out = memref.distinct_objects %in0, %out0
      : memref<5xi32>, memref<4xi32>
  affine.for %i = 0 to 4 {
    %a0 = affine.load %in[%i] : memref<5xi32>
    %a1 = affine.load %in[%i] : memref<5xi32>
    %b0 = affine.load %in[%i + 1] : memref<5xi32>
    %b1 = affine.load %in[%i + 1] : memref<5xi32>
    %p0 = arith.addi %a0, %a1 : i32
    %p1 = arith.addi %p0, %b0 : i32
    %sum = arith.addi %p1, %b1 : i32
    affine.store %sum, %out[%i] : memref<4xi32>
  }
  return
}

// A possible modification of any carried source rejects the whole plan.

// CHECK-LABEL: func.func @source_modified
// CHECK: affine.for
// CHECK-NOT: iter_args
func.func @source_modified(%in0: memref<2x5xi32>,
                           %out0: memref<4xi32>) {
  %in, %out = memref.distinct_objects %in0, %out0
      : memref<2x5xi32>, memref<4xi32>
  affine.for %i = 0 to 4 {
    %a0 = affine.load %in[0, %i] : memref<2x5xi32>
    %a1 = affine.load %in[1, %i] : memref<2x5xi32>
    %b0 = affine.load %in[0, %i + 1] : memref<2x5xi32>
    %b1 = affine.load %in[1, %i + 1] : memref<2x5xi32>
    %p0 = arith.addi %a0, %a1 : i32
    %p1 = arith.addi %p0, %b0 : i32
    %sum = arith.addi %p1, %b1 : i32
    affine.store %sum, %out[%i] : memref<4xi32>
    affine.store %sum, %in[0, %i] : memref<2x5xi32>
  }
  return
}

// Without a no-alias proof, the output write may modify the input source.

// CHECK-LABEL: func.func @unknown_alias
// CHECK: affine.for
// CHECK-NOT: iter_args
func.func @unknown_alias(%in: memref<2x5xi32>, %out: memref<4xi32>) {
  affine.for %i = 0 to 4 {
    %a0 = affine.load %in[0, %i] : memref<2x5xi32>
    %a1 = affine.load %in[1, %i] : memref<2x5xi32>
    %b0 = affine.load %in[0, %i + 1] : memref<2x5xi32>
    %b1 = affine.load %in[1, %i + 1] : memref<2x5xi32>
    %p0 = arith.addi %a0, %a1 : i32
    %p1 = arith.addi %p0, %b0 : i32
    %sum = arith.addi %p1, %b1 : i32
    affine.store %sum, %out[%i] : memref<4xi32>
  }
  return
}

// Preloading is not profitable or safe when the loop may execute once.

// CHECK-LABEL: func.func @one_trip
// CHECK: affine.for
// CHECK-NOT: iter_args
func.func @one_trip(%in0: memref<2x2xi32>, %out0: memref<1xi32>) {
  %in, %out = memref.distinct_objects %in0, %out0
      : memref<2x2xi32>, memref<1xi32>
  affine.for %i = 0 to 1 {
    %a0 = affine.load %in[0, %i] : memref<2x2xi32>
    %a1 = affine.load %in[1, %i] : memref<2x2xi32>
    %b0 = affine.load %in[0, %i + 1] : memref<2x2xi32>
    %b1 = affine.load %in[1, %i + 1] : memref<2x2xi32>
    %p0 = arith.addi %a0, %a1 : i32
    %p1 = arith.addi %p0, %b0 : i32
    %sum = arith.addi %p1, %b1 : i32
    affine.store %sum, %out[%i] : memref<1xi32>
  }
  return
}

// A zero-trip reduction must not create prologue loads outside the loop.

// CHECK-LABEL: func.func @zero_trip
// CHECK-NOT: affine.load
// CHECK: affine.for
// CHECK-NOT: iter_args
func.func @zero_trip(%in0: memref<2x2xi32>, %out0: memref<1xi32>) {
  %in, %out = memref.distinct_objects %in0, %out0
      : memref<2x2xi32>, memref<1xi32>
  affine.for %i = 0 to 0 {
    %a0 = affine.load %in[0, %i] : memref<2x2xi32>
    %a1 = affine.load %in[1, %i] : memref<2x2xi32>
    %b0 = affine.load %in[0, %i + 1] : memref<2x2xi32>
    %b1 = affine.load %in[1, %i + 1] : memref<2x2xi32>
    %p0 = arith.addi %a0, %a1 : i32
    %p1 = arith.addi %p0, %b0 : i32
    %sum = arith.addi %p1, %b1 : i32
    affine.store %sum, %out[%i] : memref<1xi32>
  }
  return
}

// A statically empty symbolic interval must also remain unchanged.

// CHECK-LABEL: func.func @symbolic_negative_trip
// CHECK-NOT: affine.load
// CHECK: affine.for
// CHECK-NOT: iter_args
func.func @symbolic_negative_trip(%in0: memref<2x?xi32>,
                                  %out0: memref<?xi32>, %start: index) {
  %in, %out = memref.distinct_objects %in0, %out0
      : memref<2x?xi32>, memref<?xi32>
  affine.for %i = #lb(%start) to #ub_minus1(%start) {
    %a0 = affine.load %in[0, %i] : memref<2x?xi32>
    %a1 = affine.load %in[1, %i] : memref<2x?xi32>
    %b0 = affine.load %in[0, %i + 1] : memref<2x?xi32>
    %b1 = affine.load %in[1, %i + 1] : memref<2x?xi32>
    %p0 = arith.addi %a0, %a1 : i32
    %p1 = arith.addi %p0, %b0 : i32
    %sum = arith.addi %p1, %b1 : i32
    affine.store %sum, %out[%i] : memref<?xi32>
  }
  return
}

// A non-speculatable operation after the producer loads is not crossed by the
// prologue and therefore does not block the rewrite.

// CHECK-LABEL: func.func @blocker_after_producers
// CHECK: affine.for {{.*}} iter_args
func.func @blocker_after_producers(%in0: memref<2x3xi32>,
                                   %out0: memref<2xi32>, %continue: i1) {
  %in, %out = memref.distinct_objects %in0, %out0
      : memref<2x3xi32>, memref<2xi32>
  affine.for %i = 0 to 2 {
    %a0 = affine.load %in[0, %i] : memref<2x3xi32>
    %a1 = affine.load %in[1, %i] : memref<2x3xi32>
    %b0 = affine.load %in[0, %i + 1] : memref<2x3xi32>
    %b1 = affine.load %in[1, %i + 1] : memref<2x3xi32>
    %p0 = arith.addi %a0, %a1 : i32
    %p1 = arith.addi %p0, %b0 : i32
    %sum = arith.addi %p1, %b1 : i32
    scf.while : () -> () {
      scf.condition(%continue)
    } do {
      scf.yield
    }
    affine.store %sum, %out[%i] : memref<2xi32>
  }
  return
}

// A shared internal combiner cannot be deleted or regrouped as a strict tree.

// CHECK-LABEL: func.func @shared_combiner_subtree
// CHECK: affine.for
// CHECK-NOT: iter_args
func.func @shared_combiner_subtree(%in0: memref<2x3xi32>,
                                   %out0: memref<2x2xi32>) {
  %in, %out = memref.distinct_objects %in0, %out0
      : memref<2x3xi32>, memref<2x2xi32>
  affine.for %i = 0 to 2 {
    %a0 = affine.load %in[0, %i] : memref<2x3xi32>
    %a1 = affine.load %in[1, %i] : memref<2x3xi32>
    %b0 = affine.load %in[0, %i + 1] : memref<2x3xi32>
    %b1 = affine.load %in[1, %i + 1] : memref<2x3xi32>
    %p0 = arith.addi %a0, %a1 : i32
    affine.store %p0, %out[0, %i] : memref<2x2xi32>
    %p1 = arith.addi %p0, %b0 : i32
    %sum = arith.addi %p1, %b1 : i32
    affine.store %sum, %out[1, %i] : memref<2x2xi32>
  }
  return
}

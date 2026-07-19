// RUN: mlir-opt %s -split-input-file -test-affine-access-analysis -verify-diagnostics | FileCheck %s

// CHECK-LABEL: func @loop_simple
func.func @loop_simple(%A : memref<?x?xf32>, %B : memref<?x?x?xf32>) {
   %c0 = arith.constant 0 : index
   %M = memref.dim %A, %c0 : memref<?x?xf32>
   affine.for %i = 0 to %M {
     affine.for %j = 0 to %M {
       affine.load %A[%c0, %i] : memref<?x?xf32>
       // expected-remark@above {{contiguous along loop 0}}
       // expected-remark@above {{invariant along loop 1}}
       affine.load %A[%c0, 8 * %i + %j] : memref<?x?xf32>
       // expected-remark@above {{contiguous along loop 1}}
       // Note/FIXME: access stride isn't being checked.
       // expected-remark@-3 {{contiguous along loop 0}}

       // These are all non-contiguous along both loops. Nothing is emitted.
       affine.load %A[%i, %c0] : memref<?x?xf32>
       // expected-remark@above {{invariant along loop 1}}
       // Note/FIXME: access stride isn't being checked.
       affine.load %A[%i, 8 * %j] : memref<?x?xf32>
       // expected-remark@above {{contiguous along loop 1}}
       affine.load %A[%j, 4 * %i] : memref<?x?xf32>
       // expected-remark@above {{contiguous along loop 0}}
     }
   }
   return
}

// -----

// CHECK-LABEL: func @loop_unsimplified
func.func @loop_unsimplified(%A : memref<100xf32>) {
   affine.for %i = 0 to 100 {
     affine.load %A[2 * %i - %i - %i] : memref<100xf32>
     // expected-remark@above {{invariant along loop 0}}

     %m = affine.apply affine_map<(d0) -> (-2 * d0)>(%i)
     %n = affine.apply affine_map<(d0) -> (2 * d0)>(%i)
     affine.load %A[(%m + %n) floordiv 2] : memref<100xf32>
     // expected-remark@above {{invariant along loop 0}}
   }
   return
}

// -----

#map = affine_map<(d0) -> (d0 * 16)>
#map1 = affine_map<(d0) -> (d0 * 16 + 16)>
#map2 = affine_map<(d0) -> (d0)>
#map3 = affine_map<(d0) -> (d0 + 1)>

func.func @tiled(%arg0: memref<*xf32>) {
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<1x224x224x64xf32>
  %cast = memref.cast %arg0 : memref<*xf32> to memref<64xf32>
  affine.for %arg1 = 0 to 4 {
    affine.for %arg2 = 0 to 224 {
      affine.for %arg3 = 0 to 14 {
        %alloc_0 = memref.alloc() : memref<1x16x1x16xf32>
        affine.for %arg4 = #map(%arg1) to #map1(%arg1) {
          affine.for %arg5 = #map(%arg3) to #map1(%arg3) {
            // TODO: here and below, the access isn't really invariant
            // along tile-space IVs where the intra-tile IVs' bounds
            // depend on them.
            %0 = affine.load %cast[%arg4] : memref<64xf32>
            // expected-remark@above {{contiguous along loop 3}}
            // expected-remark@above {{invariant along loop 0}}
            // expected-remark@above {{invariant along loop 1}}
            // expected-remark@above {{invariant along loop 2}}
            // expected-remark@above {{invariant along loop 4}}
            affine.store %0, %alloc_0[0, %arg1 * -16 + %arg4, 0, %arg3 * -16 + %arg5] : memref<1x16x1x16xf32>
            // expected-remark@above {{contiguous along loop 4}}
            // expected-remark@above {{contiguous along loop 2}}
            // expected-remark@above {{invariant along loop 1}}
          }
        }
        affine.for %arg4 = #map(%arg1) to #map1(%arg1) {
          affine.for %arg5 = #map2(%arg2) to #map3(%arg2) {
            affine.for %arg6 = #map(%arg3) to #map1(%arg3) {
              %0 = affine.load %alloc_0[0, %arg1 * -16 + %arg4, -%arg2 + %arg5, %arg3 * -16 + %arg6] : memref<1x16x1x16xf32>
              // expected-remark@above {{contiguous along loop 5}}
              // expected-remark@above {{contiguous along loop 2}}
              affine.store %0, %alloc[0, %arg5, %arg6, %arg4] : memref<1x224x224x64xf32>
              // expected-remark@above {{contiguous along loop 3}}
              // expected-remark@above {{invariant along loop 0}}
              // expected-remark@above {{invariant along loop 1}}
              // expected-remark@above {{invariant along loop 2}}
            }
          }
        }
        memref.dealloc %alloc_0 : memref<1x16x1x16xf32>
      }
    }
  }
  return
}

// -----

func.func @opwise_shift_raw_same_iteration(%A: memref<16xf32>, %v: f32) {
  // Moving the store later than the load reverses their same-iteration RAW
  // dependence.
  // expected-remark@+1 {{invalid operation shifts}}
  affine.for %i = 0 to 16 {
    affine.store %v, %A[%i] : memref<16xf32>
    %0 = affine.load %A[%i] : memref<16xf32>
  } {test.shifts = array<i64: 1, 0, 0>}
  return
}

// -----

func.func @opwise_shift_delayed_consumer(%A: memref<16xf32>, %v: f32) {
  // Delaying the load preserves the RAW dependence.
  // expected-remark@+1 {{valid operation shifts}}
  affine.for %i = 0 to 16 {
    affine.store %v, %A[%i] : memref<16xf32>
    %0 = affine.load %A[%i] : memref<16xf32>
  } {test.shifts = array<i64: 0, 1, 0>}
  return
}

// -----

func.func @opwise_shift_distance_one(%A: memref<16xf32>, %v: f32) {
  // The store and the next iteration's load would be scheduled at the same
  // time, with the load first.
  // expected-remark@+1 {{invalid operation shifts}}
  affine.for %i = 1 to 16 {
    affine.store %v, %A[%i] : memref<16xf32>
    %0 = affine.load %A[%i - 1] : memref<16xf32>
  } {test.shifts = array<i64: 1, 0, 0>}

  // The same dependence must be found when its source occurs after its
  // destination in the loop body: store(i) -> load(i + 1).
  // expected-remark@+1 {{invalid operation shifts}}
  affine.for %i = 1 to 16 {
    %0 = affine.load %A[%i - 1] : memref<16xf32>
    affine.store %v, %A[%i] : memref<16xf32>
  } {test.shifts = array<i64: 0, 1, 0>}
  return
}

// -----

func.func @opwise_shift_distance_two(%A: memref<16xf32>, %v: f32) {
  // One iteration of distance remains after applying the shifts.
  // expected-remark@+1 {{valid operation shifts}}
  affine.for %i = 2 to 16 {
    affine.store %v, %A[%i] : memref<16xf32>
    %0 = affine.load %A[%i - 2] : memref<16xf32>
  } {test.shifts = array<i64: 1, 0, 0>}

  // Reversing the body order does not change the distance threshold.
  // expected-remark@+1 {{valid operation shifts}}
  affine.for %i = 2 to 16 {
    %0 = affine.load %A[%i - 2] : memref<16xf32>
    affine.store %v, %A[%i] : memref<16xf32>
  } {test.shifts = array<i64: 0, 1, 0>}
  return
}

// -----

func.func @opwise_shift_non_unit_step(%A: memref<16xf32>, %v: f32) {
  // A distance of two IV units is only one iteration for a step-two loop.
  // expected-remark@+1 {{invalid operation shifts}}
  affine.for %i = 2 to 16 step 2 {
    affine.store %v, %A[%i] : memref<16xf32>
    %0 = affine.load %A[%i - 2] : memref<16xf32>
  } {test.shifts = array<i64: 1, 0, 0>}

  // Two step-two iterations of distance remain ordered.
  // expected-remark@+1 {{valid operation shifts}}
  affine.for %i = 4 to 16 step 2 {
    affine.store %v, %A[%i] : memref<16xf32>
    %0 = affine.load %A[%i - 4] : memref<16xf32>
  } {test.shifts = array<i64: 1, 0, 0>}
  return
}

// -----

#semi_affine_div = affine_map<(d0)[s0] -> (d0 floordiv s0)>

func.func @opwise_shift_semi_affine_divisor(
    %A: memref<?xf32>, %divisor: index, %v: f32) {
  // Unsupported access relations must not make an unsafe schedule appear
  // legal.
  // expected-remark@+1 {{invalid operation shifts}}
  affine.for %i = 0 to 16 {
    %storeIndex = affine.apply #semi_affine_div(%i)[%divisor]
    affine.store %v, %A[%storeIndex] : memref<?xf32>
    %loadIndex = affine.apply #semi_affine_div(%i)[%divisor]
    %0 = affine.load %A[%loadIndex] : memref<?xf32>
  } {test.shifts = array<i64: 1, 1, 0, 0, 0>}
  return
}

// -----

func.func @opwise_shift_memory_dependence_kinds(
    %A: memref<16xf32>, %v0: f32, %v1: f32) {
  // Moving a read after a store reverses a WAR dependence.
  // expected-remark@+1 {{invalid operation shifts}}
  affine.for %i = 0 to 16 {
    %0 = affine.load %A[%i] : memref<16xf32>
    affine.store %v0, %A[%i] : memref<16xf32>
  } {test.shifts = array<i64: 1, 0, 0>}

  // Moving the first store after the second reverses a WAW dependence.
  // expected-remark@+1 {{invalid operation shifts}}
  affine.for %i = 0 to 16 {
    affine.store %v0, %A[%i] : memref<16xf32>
    affine.store %v1, %A[%i] : memref<16xf32>
  } {test.shifts = array<i64: 1, 0, 0>}

  // Read-after-read ordering is not a dependence.
  // expected-remark@+1 {{valid operation shifts}}
  affine.for %i = 0 to 16 {
    %0 = affine.load %A[%i] : memref<16xf32>
    %1 = affine.load %A[%i] : memref<16xf32>
  } {test.shifts = array<i64: 1, 0, 0>}
  return
}

// -----

func.func @opwise_shift_disjoint_elements(%A: memref<2xf32>, %v: f32) {
  // Affine dependence analysis proves that these accesses to the same memref
  // never touch the same element.
  // expected-remark@+1 {{valid operation shifts}}
  affine.for %i = 0 to 4 {
    affine.store %v, %A[0] : memref<2xf32>
    %0 = affine.load %A[1] : memref<2xf32>
  } {test.shifts = array<i64: 1, 0, 0>}
  return
}

// -----

func.func @opwise_shift_larger_gap(%A: memref<16xf32>, %v: f32) {
  // A dependence distance equal to the shift gap is invalid.
  // expected-remark@+1 {{invalid operation shifts}}
  affine.for %i = 2 to 16 {
    affine.store %v, %A[%i] : memref<16xf32>
    %0 = affine.load %A[%i - 2] : memref<16xf32>
  } {test.shifts = array<i64: 2, 0, 0>}

  // A larger dependence distance remains ordered.
  // expected-remark@+1 {{valid operation shifts}}
  affine.for %i = 3 to 16 {
    affine.store %v, %A[%i] : memref<16xf32>
    %0 = affine.load %A[%i - 3] : memref<16xf32>
  } {test.shifts = array<i64: 2, 0, 0>}
  return
}

// -----

#nonnegative = affine_set<(d0) : (d0 >= 0)>

func.func @opwise_shift_nested_accesses(%A: memref<16xf32>, %v: f32) {
  // Nested accesses inherit the shift of their top-level ancestor operation.
  // expected-remark@+1 {{invalid operation shifts}}
  affine.for %i = 0 to 16 {
    affine.if #nonnegative(%i) {
      affine.store %v, %A[%i] : memref<16xf32>
    }
    affine.if #nonnegative(%i) {
      %0 = affine.load %A[%i] : memref<16xf32>
    }
  } {test.shifts = array<i64: 1, 0, 0>}
  return
}

// -----

func.func @opwise_shift_iter_args(
    %A: memref<4xi32>, %init: i32) -> i32 {
  %c1 = arith.constant 1 : i32
  // Delaying the yielded value makes it meet the next iteration's unshifted
  // use in the same time slice, where the use would be emitted first.
  // expected-remark@+1 {{invalid operation shifts}}
  %result = affine.for %i = 0 to 4
      iter_args(%acc = %init) -> (i32) {
    affine.store %acc, %A[%i] : memref<4xi32>
    %next = arith.addi %acc, %c1 : i32
    affine.yield %next : i32
  } {test.shifts = array<i64: 0, 1, 1>}
  return %result : i32
}

// -----

func.func @opwise_shift_may_alias_arguments(
    %A: memref<16xf32>, %B: memref<16xf32>, %v: f32) {
  // The two function arguments may name the same object, in which case moving
  // the store after the load would reverse a same-iteration RAW dependence.
  // expected-remark@+1 {{invalid operation shifts}}
  affine.for %i = 0 to 16 {
    affine.store %v, %A[%i] : memref<16xf32>
    %0 = affine.load %B[%i] : memref<16xf32>
  } {test.shifts = array<i64: 1, 0, 0>}
  return
}

// -----

func.func @opwise_shift_may_alias_reads(
    %A: memref<16xf32>, %B: memref<16xf32>) {
  // Aliasing does not constrain the relative order of two reads.
  // expected-remark@+1 {{valid operation shifts}}
  affine.for %i = 0 to 16 {
    %0 = affine.load %A[%i] : memref<16xf32>
    %1 = affine.load %B[%i] : memref<16xf32>
  } {test.shifts = array<i64: 1, 0, 0>}
  return
}

// -----

func.func @opwise_shift_cast_alias(%A: memref<16xf32>, %v: f32) {
  %alias = memref.cast %A : memref<16xf32> to memref<?xf32>
  // Distinct SSA values that are known to reference the same object must still
  // preserve their write/read ordering.
  // expected-remark@+1 {{invalid operation shifts}}
  affine.for %i = 0 to 16 {
    affine.store %v, %A[%i] : memref<16xf32>
    %0 = affine.load %alias[%i] : memref<?xf32>
  } {test.shifts = array<i64: 1, 0, 0>}
  return
}

// -----

func.func @opwise_shift_subview_alias(%A: memref<?xf32>, %v: f32) {
  %alias = memref.subview %A[0] [16] [1]
      : memref<?xf32> to memref<16xf32, strided<[1]>>
  // A view and its source are distinct SSA values but may name the same
  // elements.
  // expected-remark@+1 {{invalid operation shifts}}
  affine.for %i = 0 to 16 {
    affine.store %v, %A[%i] : memref<?xf32>
    %0 = affine.load %alias[%i] : memref<16xf32, strided<[1]>>
  } {test.shifts = array<i64: 1, 0, 0>}
  return
}

// -----

func.func @opwise_shift_distinct_arguments(
    %A0: memref<16xf32>, %B0: memref<16xf32>, %v: f32) {
  %A, %B = memref.distinct_objects %A0, %B0
      : memref<16xf32>, memref<16xf32>
  // Proven-distinct objects have no cross-memref dependence.
  // expected-remark@+1 {{valid operation shifts}}
  affine.for %i = 0 to 16 {
    affine.store %v, %A[%i] : memref<16xf32>
    %0 = affine.load %B[%i] : memref<16xf32>
  } {test.shifts = array<i64: 1, 0, 0>}
  return
}

// -----

func.func @opwise_shift_non_affine_memory_effect(%A: memref<16xf32>, %v: f32) {
  // The strict entry point rejects reordering involving a memory operation
  // whose indexing relation cannot be represented by Affine analysis.
  // expected-remark@+1 {{invalid operation shifts}}
  affine.for %i = 0 to 16 {
    memref.store %v, %A[%i] : memref<16xf32>
    %0 = affine.load %A[%i] : memref<16xf32>
  } {test.shifts = array<i64: 1, 0, 0>}
  return
}

// -----

func.func @opwise_shift_dealloc(%A: memref<1xf32>) {
  // Moving the load after the deallocation would introduce a use-after-free.
  // expected-remark@+1 {{invalid operation shifts}}
  affine.for %i = 0 to 1 {
    %0 = affine.load %A[0] : memref<1xf32>
    memref.dealloc %A : memref<1xf32>
  } {test.shifts = array<i64: 1, 0, 0>}
  return
}

// -----

func.func private @unknown_effect(memref<16xf32>)

func.func @opwise_shift_unknown_memory_effect(%A: memref<16xf32>) {
  // An operation without recursively modeled memory effects must not be
  // reordered relative to a memory access.
  // expected-remark@+1 {{invalid operation shifts}}
  affine.for %i = 0 to 16 {
    func.call @unknown_effect(%A) : (memref<16xf32>) -> ()
    %0 = affine.load %A[%i] : memref<16xf32>
  } {test.shifts = array<i64: 1, 0, 0>}
  return
}

// -----

func.func @opwise_shift_preserves_non_affine_effect(
    %A: memref<16xf32>, %v: f32) {
  // An unsupported memory effect is harmless when its relative schedule is
  // unchanged.
  // expected-remark@+1 {{valid operation shifts}}
  affine.for %i = 0 to 16 {
    memref.store %v, %A[%i] : memref<16xf32>
    %0 = affine.load %A[%i] : memref<16xf32>
  } {test.shifts = array<i64: 0, 0, 0>}
  return
}

// -----

// CHECK-LABEL: func.func @opwise_shift_producer_consumer
// CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
// CHECK: affine.load {{.*}}[%[[C0]]] : memref<4xi32>
// CHECK: affine.for %[[I:.*]] = 1 to 4 {
// CHECK: affine.store {{.*}}[%[[I]]] : memref<4xi32>
// CHECK: %[[PREV:.*]] = affine.apply {{.*}}(%[[I]])
// CHECK: affine.load {{.*}}[%[[PREV]]] : memref<4xi32>
// CHECK: affine.store {{.*}}[%[[PREV]]] : memref<4xi32>
// CHECK: %[[LAST:.*]] = affine.apply
// CHECK: affine.load {{.*}}[%[[LAST]]] : memref<4xi32>
func.func @opwise_shift_producer_consumer(
    %A0: memref<4xi32>, %T0: memref<4xi32>, %B0: memref<4xi32>) {
  %A, %T, %B = memref.distinct_objects %A0, %T0, %B0
      : memref<4xi32>, memref<4xi32>, memref<4xi32>
  // Delaying the consumer pipelines it behind the producer.
  // expected-remark@+1 {{valid operation shifts}}
  affine.for %i = 0 to 4 {
    %a = affine.load %A[%i] : memref<4xi32>
    %produced = arith.muli %a, %a : i32
    affine.store %produced, %T[%i] : memref<4xi32>
    %t = affine.load %T[%i] : memref<4xi32>
    %consumed = arith.addi %t, %t : i32
    affine.store %consumed, %B[%i] : memref<4xi32>
  } {test.apply_shifts, test.shifts = array<i64: 0, 0, 0, 1, 1, 1, 0>}
  return
}

// -----

func.func @opwise_shift_reversed_producer_consumer(
    %A0: memref<4xi32>, %T0: memref<4xi32>, %B0: memref<4xi32>) {
  %A, %T, %B = memref.distinct_objects %A0, %T0, %B0
      : memref<4xi32>, memref<4xi32>, memref<4xi32>
  // Delaying the producer behind the consumer reverses their RAW dependence.
  // expected-remark@+1 {{invalid operation shifts}}
  affine.for %i = 0 to 4 {
    %a = affine.load %A[%i] : memref<4xi32>
    %produced = arith.muli %a, %a : i32
    affine.store %produced, %T[%i] : memref<4xi32>
    %t = affine.load %T[%i] : memref<4xi32>
    %consumed = arith.addi %t, %t : i32
    affine.store %consumed, %B[%i] : memref<4xi32>
  } {test.shifts = array<i64: 1, 1, 1, 0, 0, 0, 0>}
  return
}

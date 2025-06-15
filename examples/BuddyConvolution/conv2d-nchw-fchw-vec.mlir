// RUN: buddy-opt %s \
// RUN: 	-conv2d-nhwc-fhwc-vectorization \
// RUN: 	-convert-linalg-to-loops \
// RUN: 	-lower-affine \
// RUN: 	-arith-bufferize \
// RUN: 	-convert-scf-to-cf \
// RUN: 	-convert-vector-to-llvm \
// RUN: 	-convert-arith-to-llvm \
// RUN: 	-finalize-memref-to-llvm \
// RUN: 	-convert-func-to-llvm \
// RUN: 	-reconcile-unrealized-casts \
// RUN: | mlir-cpu-runner -e main -entry-point-result=void \
// RUN:     -shared-libs=%mlir_runner_utils_dir/libmlir_runner_utils%shlibext \
// RUN:     -shared-libs=%mlir_runner_utils_dir/libmlir_c_runner_utils%shlibext \
// RUN: | FileCheck %s
#map = affine_map<(d0, d1) -> (d0 + d1)>
module {
  func.func private @printMemrefF32(memref<*xf32>)
  func.func private @rtclock() -> f64
  func.func @conv_2d_nchw_fchw(%arg0: memref<?x?x?x?xf32>, %arg1: memref<?x?x?x?xf32>, %arg2: memref<?x?x?x?xf32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %c32 = arith.constant 5 : index
    %dim = memref.dim %arg0, %c0 : memref<?x?x?x?xf32>
    %dim_0 = memref.dim %arg0, %c1 : memref<?x?x?x?xf32>
    %dim_1 = memref.dim %arg1, %c0 : memref<?x?x?x?xf32>
    %dim_2 = memref.dim %arg1, %c2 : memref<?x?x?x?xf32>
    %dim_3 = memref.dim %arg1, %c3 : memref<?x?x?x?xf32>
    %dim_4 = memref.dim %arg2, %c2 : memref<?x?x?x?xf32>
    %dim_5 = memref.dim %arg2, %c3 : memref<?x?x?x?xf32> 

    // %upbound_tmp = arith.subi %dim_3, %c32 : index
    // %upbound = arith.addi %upbound_tmp, %c1 : index

    scf.for %arg3 = %c0 to %dim step %c1 {
      scf.for %arg4 = %c0 to %dim_1 step %c1 {
        scf.for %arg5 = %c0 to %dim_4 step %c1 {
          scf.for %arg6 = %c0 to %dim_5 step %c1 {
            scf.for %arg7 = %c0 to %dim_0 step %c1 {
              scf.for %arg8 = %c0 to %dim_2 step %c1 {
                scf.for %arg9 = %c0 to %dim_3 step %c32 iter_args (%iter_idx = %c0) -> (index) {
                  %0 = affine.apply #map(%arg5, %arg8)
                  %1 = affine.apply #map(%arg6, %arg9)
                  %2 = vector.load %arg0[%arg3, %arg7, %0, %1] : memref<?x?x?x?xf32>, vector<5xf32>
                  %3 = vector.load %arg1[%arg4, %arg7, %arg8, %arg9] : memref<?x?x?x?xf32>, vector<5xf32>
                  %4 = vector.load %arg2[%arg3, %arg4, %arg5, %arg6] : memref<?x?x?x?xf32>, vector<5xf32>
                  %5 = vector.fma %2, %3, %4 : vector<5xf32>
                  vector.store %5, %arg2[%arg3, %arg4, %arg5, %arg6] : memref<?x?x?x?xf32>, vector<5xf32>
                  %tmp4 = arith.addi %arg9, %c32  : index
                  scf.yield %tmp4 : index
                }
              }
            }
          }
        }
      }
    }
    return
  }

  func.func @alloc_f32(%arg0: index, %arg1: index, %arg2: index, %arg3: index, %arg4: f32) -> memref<?x?x?x?xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %0 = memref.alloc(%arg0, %arg1, %arg2, %arg3) : memref<?x?x?x?xf32>
    scf.for %idx0 = %c0 to %arg0 step %c1 {
      scf.for %idx1 = %c0 to %arg1 step %c1 {
        scf.for %idx2 = %c0 to %arg2 step %c1 {
          scf.for %idx3 = %c0 to %arg3 step %c1 {
            memref.store %arg4, %0[%idx0, %idx1, %idx2, %idx3] : memref<?x?x?x?xf32>
          }
        }
      }
    }
    return %0 : memref<?x?x?x?xf32>
  }

  func.func @main() {
    %f0 = arith.constant 0.000000e+00 : f32
    %f2 = arith.constant 2.000000e+00 : f32
    %f3 = arith.constant 3.000000e+00 : f32

    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %c5 = arith.constant 5 : index
    %c6 = arith.constant 6 : index
    %c8 = arith.constant 8 : index
    %c12 = arith.constant 12 : index
    %c16 = arith.constant 16 : index
    %c24 = arith.constant 24 : index
    %c28 = arith.constant 28 : index

    // %v0 = call @alloc_f32(%c1, %c12, %c12, %c6, %f2) : (index, index, index, index, f32) -> memref<?x?x?x?xf32>
    // %v1 = call @alloc_f32(%c16, %c5, %c5, %c6, %f3) : (index, index, index, index, f32) -> memref<?x?x?x?xf32>
    // %v2 = call @alloc_f32(%c1, %c8, %c8, %c16, %f0) : (index, index, index, index, f32) -> memref<?x?x?x?xf32>
    
    %v0 = call @alloc_f32(%c1, %c8, %c12, %c12, %f2) : (index, index, index, index, f32) -> memref<?x?x?x?xf32>
    %v1 = call @alloc_f32(%c16, %c8, %c5, %c5, %f3) : (index, index, index, index, f32) -> memref<?x?x?x?xf32>
    %v2 = call @alloc_f32(%c1, %c16, %c8, %c8, %f0) : (index, index, index, index, f32) -> memref<?x?x?x?xf32>

    %t_start = call @rtclock() : () -> f64
    call @conv_2d_nchw_fchw(%v0, %v1, %v2) : (memref<?x?x?x?xf32>, memref<?x?x?x?xf32>, memref<?x?x?x?xf32>) -> ()
    %t_end = call @rtclock() : () -> f64

    // All the elements of the MemRef are the same,
    // only check the first line to verify the correctness.
    // CHECK: Unranked Memref
    // CHECK: [
    // CHECK: [
    // CHECK: [
    // CHECK: [900{{(, 900)*}}],
    %print_v2 = memref.cast %v2 : memref<?x?x?x?xf32> to memref<*xf32>
    call @printMemrefF32(%print_v2) : (memref<*xf32>) -> ()

    %time = arith.subf %t_end, %t_start : f64
    vector.print %time : f64

    memref.dealloc %v0 : memref<?x?x?x?xf32>
    memref.dealloc %v1 : memref<?x?x?x?xf32>
    memref.dealloc %v2 : memref<?x?x?x?xf32>
    return
  }
}

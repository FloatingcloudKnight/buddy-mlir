// RUN: buddy-opt %s \
// RUN:     -convert-linalg-to-affine-loops \
// RUN:     -lower-affine \
// RUN:     -convert-vector-to-scf \
// RUN:     -convert-vector-to-llvm \
// RUN:     -finalize-memref-to-llvm  \
// RUN:     -convert-scf-to-cf \
// RUN:     -convert-openmp-to-llvm \
// RUN:     -convert-math-to-llvm \
// RUN:     -convert-math-to-libm \
// RUN:     -convert-arith-to-llvm \
// RUN:     -convert-func-to-llvm \
// RUN:     -expand-strided-metadata \
// RUN:     -finalize-memref-to-llvm \
// RUN:     -reconcile-unrealized-casts \
// RUN: | mlir-cpu-runner -e main -entry-point-result=void \
// RUN:     -shared-libs=%mlir_runner_utils_dir/libmlir_runner_utils%shlibext \
// RUN:     -shared-libs=%mlir_runner_utils_dir/libmlir_c_runner_utils%shlibext \
// RUN: | FileCheck %s
#map = affine_map<(d0) -> (d0)>

func.func private @rtclock() -> f64
func.func private @printMemrefF32(memref<*xf32>)

func.func @test(%arg1 : memref<?x?x?xf32>, %arg0 : memref<?x?x?x?xf32>, %c :  memref<?x?x?x?xf32>) {
  %t_start = call @rtclock() : () -> f64
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %c32 = arith.constant 32 : index
  %cst = arith.constant 0.000000e+00 : f32
  %2 = vector.splat %cst : vector<32xf32>
  %dim = memref.dim %arg1, %c0 : memref<?x?x?xf32>
  %dim_0 = memref.dim %arg1, %c1 : memref<?x?x?xf32>
  %dim_1 = memref.dim %arg1, %c2 : memref<?x?x?xf32>
  %dim_2 = memref.dim %arg0, %c3 : memref<?x?x?x?xf32>
  %5 = arith.subi %dim_2, %c32 : index
  %6 = arith.addi %5, %c1 : index
  affine.for %arg2 = #map(%c0) to #map(%dim) {
    affine.for %arg3 = #map(%c0) to #map(%dim_0) {
      %11 = scf.for %arg4 = %c0 to %6 step %c32 iter_args(%arg5 = %c0) -> (index) {
        %17 = scf.for %arg6 = %c0 to %dim_1 step %c1 iter_args(%arg7 = %2) -> (vector<32xf32>) {
          %19 = memref.load %arg1[%arg2, %arg3, %arg6] : memref<?x?x?xf32>
          %20 = vector.splat %19 : vector<32xf32>
          %21 = vector.load %arg0[%c0, %arg6, %arg2, %arg4] : memref<?x?x?x?xf32>, vector<32xf32>
          %22 = vector.fma %20, %21, %arg7 : vector<32xf32>
          scf.yield %22 : vector<32xf32>
        }
        vector.store %17, %c[%c0, %arg3, %arg2, %arg4] : memref<?x?x?x?xf32>, vector<32xf32>
        %18 = arith.addi %arg4, %c32 : index
        scf.yield %18 : index
      }
      %12 = arith.subi %dim_2, %11 : index
      %13 = vector.create_mask %12 : vector<32xi1>
      %14 = vector.maskedload %c[%c0, %arg3, %arg2, %11], %13, %2 : memref<?x?x?x?xf32>, vector<32xi1>, vector<32xf32> into vector<32xf32>
      %15 = scf.for %arg4 = %c0 to %dim_1 step %c1 iter_args(%arg5 = %14) -> (vector<32xf32>) {
        %16 = memref.load %arg1[%arg2, %arg3, %arg4] : memref<?x?x?xf32>
        %17 = vector.splat %16 : vector<32xf32>
        %18 = vector.maskedload %arg0[%c0, %arg4, %arg2, %11], %13, %2 : memref<?x?x?x?xf32>, vector<32xi1>, vector<32xf32> into vector<32xf32>
        %19 = vector.fma %17, %18, %arg5 : vector<32xf32>
        scf.yield %19 : vector<32xf32>
      }
      vector.maskedstore %c[%c0, %arg3, %arg2, %11], %13, %15 : memref<?x?x?x?xf32>, vector<32xi1>, vector<32xf32>
    }
  }
  
  %t_end = call @rtclock() : () -> f64
  %time = arith.subf %t_end, %t_start : f64
  // Print timings.
  vector.print %time : f64
  return
}

func.func @alloc_f32(%dim0: index, %dim1: index, %dim2: index, %arg4: f32) -> memref<?x?x?xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %0 = memref.alloc(%dim0, %dim1, %dim2) : memref<?x?x?xf32>
  scf.for %idx0 = %c0 to %dim0 step %c1 {
    scf.for %idx1 = %c0 to %dim1 step %c1 {
      scf.for %idx2 = %c0 to %dim2 step %c1 {
        memref.store %arg4, %0[%idx0, %idx1, %idx2] : memref<?x?x?xf32>
      }
    }
  }
  return %0 : memref<?x?x?xf32>
}

func.func @alloc_f32_4(%dim0: index, %dim1: index, %dim2: index, %dim3: index, %arg4: f32) -> memref<?x?x?x?xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %0 = memref.alloc(%dim0, %dim1, %dim2, %dim3) : memref<?x?x?x?xf32>
  scf.for %idx0 = %c0 to %dim0 step %c1 {
    scf.for %idx1 = %c0 to %dim1 step %c1 {
      scf.for %idx2 = %c0 to %dim2 step %c1 {
        scf.for %idx3 = %c0 to %dim3 step %c1 {
        memref.store %arg4, %0[%idx0, %idx1, %idx2, %idx3] : memref<?x?x?x?xf32>
        }
      }
    }
  }
  return %0 : memref<?x?x?x?xf32>
}

func.func @main(){
  %c1 = arith.constant 1 : index
  %c32 = arith.constant 32 : index
  %c40 = arith.constant 40 : index
  %c128 = arith.constant 128 : index
  %f0 = arith.constant 0.0 : f32
  %f2 = arith.constant 2.0 : f32
  %f3 = arith.constant 3.0 : f32

  %m0 = call @alloc_f32(%c32, %c40, %c40, %f2) : (index, index, index, f32) -> memref<?x?x?xf32>
  %m1 = call @alloc_f32_4(%c1, %c40, %c32, %c128, %f3) : (index, index, index, index, f32) -> memref<?x?x?x?xf32>
  %m2 = call @alloc_f32_4(%c1, %c40, %c32, %c128, %f0) : (index, index, index, index, f32) -> memref<?x?x?x?xf32>

  call @test(%m0, %m1, %m2) : (memref<?x?x?xf32>, memref<?x?x?x?xf32>, memref<?x?x?x?xf32>) -> ()

  %printed_m2 = memref.cast %m2 : memref<?x?x?x?xf32> to memref<*xf32>

  // CHECK: Unranked Memref base@ = {{.*}} rank = 4 offset = 0 sizes = [1, 40, 32, 128] strides = [163840, 4096, 128, 1] data = 
  // CHECK-NEXT: [
  // CHECK: [240{{(, 240)*}}]
  call @printMemrefF32(%printed_m2) : (memref<*xf32>) -> ()

  return
}

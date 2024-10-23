#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0) -> (d0 ceildiv 32)>
#map2 = affine_map<(d0, d1) -> (d0 + d1 - 1)>
module {
  func.func private @printMemrefF32(memref<*xf32>)
  func.func @conv_2d(%arg0: memref<?x?xf32>, %arg1: memref<?x?xf32>, %arg2: memref<?x?xf32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c32 = arith.constant 32 : index
    %cst = arith.constant 0.000000e+00 : f32
    %0 = vector.splat %cst : vector<32xf32>
    %dim = memref.dim %arg1, %c0 : memref<?x?xf32>
    %dim_0 = memref.dim %arg1, %c1 : memref<?x?xf32>
    %dim_1 = memref.dim %arg2, %c0 : memref<?x?xf32>
    %dim_2 = memref.dim %arg2, %c1 : memref<?x?xf32>
    affine.for %arg3 = #map(%c0) to #map(%dim_1) {
      affine.for %arg4 = #map(%c0) to #map(%dim) {
        affine.for %arg5 = #map(%c0) to #map(%dim_0) {
          affine.for %arg6 = #map(%c0) to #map1(%dim_2) {
            %1 = memref.load %arg1[%arg4, %arg5] : memref<?x?xf32>
            %2 = arith.index_cast %c0 : index to i32
            %3 = arith.sitofp %2 : i32 to f32
            %4 = arith.cmpf one, %1, %3 : f32
            scf.if %4 {
              %5 = vector.broadcast %1 : f32 to vector<32xf32>
              %6 = arith.muli %arg6, %c32 : index
              %7 = arith.subi %dim_2, %6 : index
              %8 = arith.cmpi sge, %7, %c32 : index
              scf.if %8 {
                %9 = affine.vector_load %arg0[%arg3 + %arg4, %arg5 + %arg6 * 32] : memref<?x?xf32>, vector<32xf32>
                %10 = affine.vector_load %arg2[%arg3, %arg6 * 32] : memref<?x?xf32>, vector<32xf32>
                %11 = vector.fma %9, %5, %10 : vector<32xf32>
                affine.vector_store %11, %arg2[%arg3, %arg6 * 32] : memref<?x?xf32>, vector<32xf32>
              } else {
                %9 = vector.create_mask %7 : vector<32xi1>
                %10 = arith.addi %arg3, %arg4 : index
                %11 = arith.muli %arg6, %c32 : index
                %12 = arith.addi %arg5, %11 : index
                %13 = vector.maskedload %arg0[%10, %12], %9, %0 : memref<?x?xf32>, vector<32xi1>, vector<32xf32> into vector<32xf32>
                %14 = vector.maskedload %arg2[%arg3, %11], %9, %0 : memref<?x?xf32>, vector<32xi1>, vector<32xf32> into vector<32xf32>
                %15 = vector.fma %13, %5, %14 : vector<32xf32>
                vector.maskedstore %arg2[%arg3, %11], %9, %15 : memref<?x?xf32>, vector<32xi1>, vector<32xf32>
              }
            }
          }
        }
      }
    }
    return
  }
  func.func @alloc_f32(%arg0: index, %arg1: index, %arg2: f32) -> memref<?x?xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %alloc = memref.alloc(%arg0, %arg1) : memref<?x?xf32>
    scf.for %arg3 = %c0 to %arg0 step %c1 {
      scf.for %arg4 = %c0 to %arg1 step %c1 {
        memref.store %arg2, %alloc[%arg3, %arg4] : memref<?x?xf32>
      }
    }
    return %alloc : memref<?x?xf32>
  }
  func.func @main() {
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant 1.000000e+00 : f32
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %c3_1 = arith.constant 3 : index
    %c8 = arith.constant 8 : index
    %0 = affine.apply #map2(%c8, %c3_1)
    %1 = call @alloc_f32(%0, %0, %cst_0) : (index, index, f32) -> memref<?x?xf32>
    %2 = call @alloc_f32(%c3_1, %c3_1, %cst_0) : (index, index, f32) -> memref<?x?xf32>
    %3 = call @alloc_f32(%c8, %c8, %cst) : (index, index, f32) -> memref<?x?xf32>
    call @conv_2d(%1, %2, %3) : (memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>) -> ()
    %cast = memref.cast %3 : memref<?x?xf32> to memref<*xf32>
    call @printMemrefF32(%cast) : (memref<*xf32>) -> ()
    memref.dealloc %1 : memref<?x?xf32>
    memref.dealloc %2 : memref<?x?xf32>
    memref.dealloc %3 : memref<?x?xf32>
    return
  }
}


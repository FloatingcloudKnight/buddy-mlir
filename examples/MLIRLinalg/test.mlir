module{
    func.func private @printMemrefF32(memref<*xf32>)

    func.func @pooling_nhwc_max(%a : memref<?x?x?x?xf32>, %b : memref<?x?xf32>, %c : memref<?x?x?x?xf32>) {
      linalg.pooling_nhwc_max  
        ins(%a, %b : memref<?x?x?x?xf32>, memref<?x?xf32>) 
        outs(%c : memref<?x?x?x?xf32>)
      return
    }

    func.func @main(){
      // Set up dims.
      %N = arith.constant 1 : index
      %iH = arith.constant 4 : index
      %iW = arith.constant 4 : index
      %kH = arith.constant 2 : index
      %kW = arith.constant 2 : index
      %oH = arith.constant 3 : index
      %oW = arith.constant 3 : index
      %iC = arith.constant 1 : index

      // Set Init Value.
      %cf1 = arith.constant 1.0 : f32

      %A = memref.alloc(%N, %iH, %iW, %iC) : memref<?x?x?x?xf32>
      %B = memref.alloc(%kH, %kW) : memref<?x?xf32>
      %C = memref.alloc(%N, %oH, %oW, %iC) : memref<?x?x?x?xf32>

      linalg.fill
      ins(%cf1 : f32)
      outs(%A:memref<?x?x?x?xf32>)

      linalg.fill
      ins(%cf1 : f32)
      outs(%B:memref<?x?xf32>)

      linalg.fill
      ins(%cf1 : f32)
      outs(%C:memref<?x?x?x?xf32>)

      call @pooling_nhwc_max(%A, %B, %C) : (memref<?x?x?x?xf32>, memref<?x?xf32>, memref<?x?x?x?xf32>) -> ()

      %print_C = memref.cast %C : memref<?x?x?x?xf32> to memref<*xf32>
      call @printMemrefF32(%print_C) : (memref<*xf32>) -> ()

      memref.dealloc %C : memref<?x?x?x?xf32>
      memref.dealloc %B : memref<?x?xf32>
      memref.dealloc %A : memref<?x?x?x?xf32>
      return 
    }
}

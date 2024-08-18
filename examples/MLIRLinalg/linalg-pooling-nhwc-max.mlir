module{
    memref.global "private" @input : memref<1x4x4x1xf32> = 
        dense<[[[[1.], [2.], [3.], [4.]], 
                [[4.], [3.], [5.], [8.]],
                [[4.], [5.], [3.], [8.]], 
                [[9.], [8.], [5.], [1.]]]]>
    memref.global "private" @kernel : memref<2x2xf32> = dense<0.0>
    memref.global "private" @output : memref<1x3x3x1xf32> = dense<0.0>

    func.func private @printMemrefF32(memref<*xf32>)

    func.func @pooling_nhwc_max(%a : memref<?x?x?x?xf32>, %b : memref<?x?xf32>, %c : memref<?x?x?x?xf32>) {
      linalg.pooling_nhwc_max  
        ins(%a, %b : memref<?x?x?x?xf32>, memref<?x?xf32>) 
        outs(%c : memref<?x?x?x?xf32>)
      return
    }

    func.func @main(){
      %input = memref.get_global @input : memref<1x4x4x1xf32>
      %kernel = memref.get_global @kernel : memref<2x2xf32>
      %output = memref.get_global @output : memref<1x3x3x1xf32>

      %a = memref.cast %input : memref<1x4x4x1xf32> to memref<?x?x?x?xf32>
      %b = memref.cast %kernel : memref<2x2xf32> to memref<?x?xf32>
      %c = memref.cast %output : memref<1x3x3x1xf32> to memref<?x?x?x?xf32>

      call @pooling_nhwc_max(%a, %b, %c) : (memref<?x?x?x?xf32>, memref<?x?xf32>, memref<?x?x?x?xf32>) -> ()
      
      %print_c = memref.cast %c : memref<?x?x?x?xf32> to memref<*xf32>
      call @printMemrefF32(%print_c) : (memref<*xf32>) -> ()

      return 
    }
}
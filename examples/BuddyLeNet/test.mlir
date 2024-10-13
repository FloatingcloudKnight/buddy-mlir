module {
  func.func @subgraph0(%arg0: tensor<1x1x28x28xf32>, %arg1: tensor<6x1x5x5xf32>, %arg2: tensor<6xf32>, %arg3: tensor<16x6x5x5xf32>, %arg4: tensor<16xf32>, %arg5: tensor<120x256xf32>, %arg6: tensor<120xf32>, %arg7: tensor<84x120xf32>, %arg8: tensor<84xf32>, %arg9: tensor<10x84xf32>, %arg10: tensor<10xf32>) -> tensor<1x10xf32> {
    %4 = tosa.conv2d %arg0, %arg1, %arg2 {dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>} : (tensor<1x1x28x28xf32>, tensor<6x1x5x5xf32>, tensor<6xf32>) -> tensor<1x6x24x24xf32>
    %7 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x6x24x24xf32>}> : () -> tensor<1x6x24x24xf32>
    %8 = tosa.maximum %4, %7 : (tensor<1x6x24x24xf32>, tensor<1x6x24x24xf32>) -> tensor<1x6x24x24xf32>
    %11 = tosa.max_pool2d %8 {kernel = array<i64: 2, 2>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 2, 2>} : (tensor<1x6x24x24xf32>) -> tensor<1x6x12x12xf32>
    %18 = tosa.conv2d %11, %arg3, %arg4 {dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>} : (tensor<1x6x12x12xf32>, tensor<16x6x5x5xf32>, tensor<16xf32>) -> tensor<1x16x8x8xf32>
    %21 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x16x8x8xf32>}> : () -> tensor<1x16x8x8xf32>
    %22 = tosa.maximum %18, %21 : (tensor<1x16x8x8xf32>, tensor<1x16x8x8xf32>) -> tensor<1x16x8x8xf32>
    %25 = tosa.max_pool2d %22 {kernel = array<i64: 2, 2>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 2, 2>} : (tensor<1x16x8x8xf32>) -> tensor<1x16x4x4xf32>
    %28 = tosa.reshape %25 {new_shape = array<i64: 1, 256>} : (tensor<1x16x4x4xf32>) -> tensor<1x256xf32>
    %29 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %30 = tosa.transpose %arg5, %29 : (tensor<120x256xf32>, tensor<2xi32>) -> tensor<256x120xf32>
    %31 = tosa.reshape %28 {new_shape = array<i64: 1, 1, 256>} : (tensor<1x256xf32>) -> tensor<1x1x256xf32>
    %32 = tosa.reshape %30 {new_shape = array<i64: 1, 256, 120>} : (tensor<256x120xf32>) -> tensor<1x256x120xf32>
    %33 = tosa.matmul %31, %32 : (tensor<1x1x256xf32>, tensor<1x256x120xf32>) -> tensor<1x1x120xf32>              // 操作数必须是3D Tensor
    %34 = tosa.reshape %33 {new_shape = array<i64: 1, 120>} : (tensor<1x1x120xf32>) -> tensor<1x120xf32>
    %35 = tosa.reshape %arg6 {new_shape = array<i64: 1, 120>} : (tensor<120xf32>) -> tensor<1x120xf32>
    %36 = tosa.add %35, %34 : (tensor<1x120xf32>, tensor<1x120xf32>) -> tensor<1x120xf32>
    %37 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x120xf32>}> : () -> tensor<1x120xf32>
    %38 = tosa.maximum %36, %37 : (tensor<1x120xf32>, tensor<1x120xf32>) -> tensor<1x120xf32>
    %39 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %40 = tosa.transpose %arg7, %39 : (tensor<84x120xf32>, tensor<2xi32>) -> tensor<120x84xf32>
    %41 = tosa.reshape %38 {new_shape = array<i64: 1, 1, 120>} : (tensor<1x120xf32>) -> tensor<1x1x120xf32>
    %42 = tosa.reshape %40 {new_shape = array<i64: 1, 120, 84>} : (tensor<120x84xf32>) -> tensor<1x120x84xf32>
    %43 = tosa.matmul %41, %42 : (tensor<1x1x120xf32>, tensor<1x120x84xf32>) -> tensor<1x1x84xf32>
    %44 = tosa.reshape %43 {new_shape = array<i64: 1, 84>} : (tensor<1x1x84xf32>) -> tensor<1x84xf32>
    %45 = tosa.reshape %arg8 {new_shape = array<i64: 1, 84>} : (tensor<84xf32>) -> tensor<1x84xf32>
    %46 = tosa.add %45, %44 : (tensor<1x84xf32>, tensor<1x84xf32>) -> tensor<1x84xf32>
    %47 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x84xf32>}> : () -> tensor<1x84xf32>
    %48 = tosa.maximum %46, %47 : (tensor<1x84xf32>, tensor<1x84xf32>) -> tensor<1x84xf32>
    %49 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %50 = tosa.transpose %arg9, %49 : (tensor<10x84xf32>, tensor<2xi32>) -> tensor<84x10xf32>
    %51 = tosa.reshape %48 {new_shape = array<i64: 1, 1, 84>} : (tensor<1x84xf32>) -> tensor<1x1x84xf32>
    %52 = tosa.reshape %50 {new_shape = array<i64: 1, 84, 10>} : (tensor<84x10xf32>) -> tensor<1x84x10xf32>
    %53 = tosa.matmul %51, %52 : (tensor<1x1x84xf32>, tensor<1x84x10xf32>) -> tensor<1x1x10xf32>
    %54 = tosa.reshape %53 {new_shape = array<i64: 1, 10>} : (tensor<1x1x10xf32>) -> tensor<1x10xf32>
    %55 = tosa.reshape %arg10 {new_shape = array<i64: 1, 10>} : (tensor<10xf32>) -> tensor<1x10xf32>
    %56 = tosa.add %55, %54 : (tensor<1x10xf32>, tensor<1x10xf32>) -> tensor<1x10xf32>
    return %56 : tensor<1x10xf32>
  }
}


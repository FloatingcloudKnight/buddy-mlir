add_mlir_library(MatMulOptimization
	MatMulOptimize.cpp
  MatMulVectorization.cpp
  MatMulParallelVectorization.cpp
  BatchMatMulOptimize.cpp
  BatchMatMulOptimize8.cpp
  BatchMatMulOptimize16.cpp
  BatchMatMulOptimize32.cpp
  BatchMatMulOptimize128.cpp
  BatchMatMulOptimize256.cpp
  BatchMatMulTileOptimize.cpp
  BatchMatMulSCFOptimize.cpp
  MatMulTransposeBVec.cpp
  BatchMatMulOptimize.cpp
  BatchMatMulTileOptimize.cpp
  BatchMatMulSCFOptimize.cpp
  LINK_LIBS PUBLIC
  BuddyUtils
)

add_mlir_library(BatchMatMulOptimization
  BatchMatMulOptimize.cpp
  BatchMatMulOptimize8.cpp
  BatchMatMulOptimize16.cpp
  BatchMatMulOptimize32.cpp
  BatchMatMulOptimize128.cpp
  BatchMatMulOptimize256.cpp
)

add_mlir_library(MatMulParallelVectorization
  MatMulParallelVectorization.cpp
)

add_mlir_library(MatMulTransposeBVec
  MatMulTransposeBVec.cpp 
)

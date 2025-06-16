add_mlir_library(TransposeOptimization
  BuiltinTransposeVectorization.cpp
  TransposeFusionVectorization.cpp
  TransposeTest.cpp
  LINK_LIBS PUBLIC
  BuddyUtils
)

add_mlir_library(CBConvVectorization
  CBConvVectorization.cpp
  GEMMPointwiseConv2DNhwcHwcf.cpp
  PoolingVectorization.cpp
  PoolingNhwcMaxAdaptVectorization.cpp
  PoolingNhwcMaxVectorization.cpp
  ConvNhwcFhwcAdaptVectorization.cpp
  ConvNhwcFhwcVectorization.cpp
  
  LINK_LIBS PUBLIC
  BuddyUtils
  )

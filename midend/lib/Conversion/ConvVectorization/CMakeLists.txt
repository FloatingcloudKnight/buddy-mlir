add_mlir_library(CBConvVectorization
  CBConvVectorization.cpp
  GEMMPointwiseConv2DNhwcHwcf.cpp
  PoolingVectorization.cpp
  PoolingNhwcMaxAdaptVectorization.cpp
  PoolingNhwcMaxVectorization.cpp
  PoolingNhwcMaxVectorization4.cpp
  PoolingNhwcMaxVectorization8.cpp
  PoolingNhwcMaxVectorization32.cpp
  PoolingNhwcMaxVectorization64.cpp
  PoolingNhwcMaxVectorization128.cpp
  Conv2dNhwcFhwcAdaptVectorization.cpp
  Conv2dNhwcFhwcVectorization2.cpp
  Conv2dNhwcFhwcVectorization4.cpp
  Conv2dNhwcFhwcVectorization16.cpp
  Conv2dNhwcFhwcVectorization32.cpp
  Conv2dNhwcFhwcVectorization64.cpp
  Conv2dNhwcFhwcVectorization128.cpp
  Conv2dNhwcFhwcVectorization.cpp
  
  LINK_LIBS PUBLIC
  BuddyUtils
  )

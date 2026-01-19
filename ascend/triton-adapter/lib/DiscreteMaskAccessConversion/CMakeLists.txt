add_triton_library(DiscreteMaskAccessConversion
  DiscreteMaskAccessConversionPass.cpp

  DEPENDS
  DiscreteMaskAccessConversionPassIncGen

  LINK_LIBS
  MLIRIR
  MLIRPass
  MLIRTransforms
  MLIRSupport
  TritonIR
)

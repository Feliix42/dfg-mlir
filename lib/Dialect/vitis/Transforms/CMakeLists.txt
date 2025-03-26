add_mlir_conversion_library(MergeCastChain
    MergeCastChain.cpp

    DEPENDS
    VitisTransformsIncGen

    LINK_COMPONENTS
    Core

    LINK_LIBS PUBLIC
    MLIRDialectUtils
    MLIRTransformUtils
)

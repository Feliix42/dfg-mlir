add_mlir_conversion_library(MergeCastChain
    MergeCastChain.cpp

    DEPENDS
        DFGMLIRVitisTransformsIncGen

    LINK_COMPONENTS
    Core

    LINK_LIBS PUBLIC
    MLIRDialectUtils
    MLIRTransformUtils
)

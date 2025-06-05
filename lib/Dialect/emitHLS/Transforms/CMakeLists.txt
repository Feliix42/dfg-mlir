add_mlir_conversion_library(DFGMLIREmitHLSTransforms
    InsertIncludes.cpp
    MergeCastChain.cpp

    DEPENDS
        DFGMLIREmitHLSTransformsIncGen

    LINK_COMPONENTS
    Core

    LINK_LIBS PUBLIC
    MLIRDialectUtils
    MLIRTransformUtils
)

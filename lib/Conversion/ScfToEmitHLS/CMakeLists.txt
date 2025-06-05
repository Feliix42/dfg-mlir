add_mlir_conversion_library(DFGMLIRScfToEmitHLS
    ScfToEmitHLS.cpp

    DEPENDS
        DFGMLIRConversionIncGen

    LINK_COMPONENTS
        Core

    LINK_LIBS PUBLIC
        MLIRDialectUtils
        MLIRTransformUtils
        MLIRControlFlowDialect
        MLIRSCFDialect
        DFGMLIRDfgDialect
        DFGMLIREmitHLSDialect
)
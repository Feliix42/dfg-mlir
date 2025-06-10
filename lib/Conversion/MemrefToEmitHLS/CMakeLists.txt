add_mlir_conversion_library(DFGMLIRMemrefToEmitHLS
MemrefToEmitHLS.cpp

    DEPENDS
        DFGMLIRConversionIncGen

    LINK_COMPONENTS
        Core

    LINK_LIBS PUBLIC
        MLIRDialectUtils
        MLIRTransformUtils
        MLIRControlFlowDialect
        MLIRMemRefDialect
        DFGMLIRDfgDialect
        DFGMLIREmitHLSDialect
)
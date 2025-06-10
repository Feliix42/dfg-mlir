add_mlir_conversion_library(DFGMLIRMathToEmitHLS
MathToEmitHLS.cpp

    DEPENDS
        DFGMLIRConversionIncGen

    LINK_COMPONENTS
        Core

    LINK_LIBS PUBLIC
        MLIRDialectUtils
        MLIRTransformUtils
        MLIRControlFlowDialect
        MLIRMathDialect
        DFGMLIRDfgDialect
        DFGMLIREmitHLSDialect
)
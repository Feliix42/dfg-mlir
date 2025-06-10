add_mlir_conversion_library(DFGMLIRArithIndexToEmitHLS
ArithIndexToEmitHLS.cpp

    DEPENDS
        DFGMLIRConversionIncGen

    LINK_COMPONENTS
        Core

    LINK_LIBS PUBLIC
        MLIRDialectUtils
        MLIRTransformUtils
        MLIRControlFlowDialect
        MLIRArithDialect
        MLIRMathDialect
        DFGMLIRDfgDialect
        DFGMLIREmitHLSDialect
)
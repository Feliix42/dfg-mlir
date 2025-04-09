add_mlir_conversion_library(DFGMLIRArithIndexToVitis
ArithIndexToVitis.cpp

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
        DFGMLIRVitisDialect
)
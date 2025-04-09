add_mlir_conversion_library(DFGMLIRMathToVitis
MathToVitis.cpp

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
        DFGMLIRVitisDialect
)
add_mlir_conversion_library(DFGMLIRDfgToFunc
        DfgToFunc.cpp

    DEPENDS
        DFGMLIRConversionIncGen

    LINK_COMPONENTS
        Core

    LINK_LIBS PUBLIC
        MLIRDialectUtils
        MLIRTransformUtils
        MLIRFuncDialect
        MLIRControlFlowDialect
        DFGMLIRDfgDialect
)
add_mlir_conversion_library(DFGMLIRLinalgToDfg
LinalgToDfg.cpp

    DEPENDS
        DFGMLIRConversionIncGen

    LINK_COMPONENTS
        Core

    LINK_LIBS PUBLIC
        MLIRDialectUtils
        MLIRTransformUtils
        MLIRControlFlowDialect
        DFGMLIRDfgDialect
)
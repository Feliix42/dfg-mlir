add_mlir_conversion_library(DFGMLIRDfgToVitis
    DfgToVitis.cpp

    DEPENDS
        DFGMLIRConversionIncGen

    LINK_COMPONENTS
        Core

    LINK_LIBS PUBLIC
        MLIRDialectUtils
        MLIRTransformUtils
        MLIRControlFlowDialect
        DFGMLIRDfgDialect
        DFGMLIRVitisDialect
)
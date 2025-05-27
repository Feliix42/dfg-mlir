add_mlir_conversion_library(DFGMLIRScfToVitis
    ScfToVitis.cpp

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
        DFGMLIRVitisDialect
)
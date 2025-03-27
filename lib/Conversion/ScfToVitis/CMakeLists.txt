add_mlir_conversion_library(ScfToVitis
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
        DfgDialect
        VitisDialect
)
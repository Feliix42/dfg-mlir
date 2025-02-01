add_mlir_conversion_library(ScfToVitis
    ScfToVitis.cpp

    DEPENDS
        ConversionIncGen

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
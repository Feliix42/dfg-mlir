add_mlir_conversion_library(MemrefToVitis
MemrefToVitis.cpp

    DEPENDS
        ConversionIncGen

    LINK_COMPONENTS
        Core

    LINK_LIBS PUBLIC
        MLIRDialectUtils
        MLIRTransformUtils
        MLIRControlFlowDialect
        MLIRMemRefDialect
        DfgDialect
        VitisDialect
)
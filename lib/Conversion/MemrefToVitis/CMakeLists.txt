add_mlir_conversion_library(DFGMLIRMemrefToVitis
MemrefToVitis.cpp

    DEPENDS
        DFGMLIRConversionIncGen

    LINK_COMPONENTS
        Core

    LINK_LIBS PUBLIC
        MLIRDialectUtils
        MLIRTransformUtils
        MLIRControlFlowDialect
        MLIRMemRefDialect
        DFGMLIRDfgDialect
        DFGMLIRVitisDialect
)
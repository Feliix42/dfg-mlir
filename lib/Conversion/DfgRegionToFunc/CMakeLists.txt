add_mlir_conversion_library(DFGMLIRDfgRegionToFunc
        DfgRegionToFunc.cpp

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

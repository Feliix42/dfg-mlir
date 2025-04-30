add_mlir_conversion_library(DFGMLIRFuncToDfg
    FuncToDfg.cpp

    DEPENDS
        DFGMLIRConversionIncGen

    LINK_COMPONENTS
        Core

    LINK_LIBS PUBLIC
        MLIRDialectUtils
        MLIRTransformUtils
        MLIRFuncDialect
        DFGMLIRDfgDialect
)
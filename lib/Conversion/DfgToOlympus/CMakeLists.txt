add_mlir_conversion_library(DFGMLIRDfgToOlympus
        DfgToOlympus.cpp

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


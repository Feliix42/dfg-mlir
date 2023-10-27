add_mlir_conversion_library(DfgToOlympus
        DfgToOlympus.cpp

    DEPENDS
        ConversionIncGen

    LINK_COMPONENTS
        Core

    LINK_LIBS PUBLIC
        MLIRDialectUtils
        MLIRTransformUtils
        MLIRFuncDialect
        DfgDialect
)


add_mlir_conversion_library(DfgInsertOlympusWrapper
        DfgInsertOlympusWrapper.cpp

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


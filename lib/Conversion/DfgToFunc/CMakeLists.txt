add_mlir_conversion_library(DfgToFunc
        DfgToFunc.cpp

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
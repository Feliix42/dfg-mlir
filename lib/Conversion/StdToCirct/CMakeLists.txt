add_mlir_conversion_library(StdToCirct
    StdToCirct.cpp

    DEPENDS
    ConversionIncGen

    LINK_COMPONENTS
    Core

    LINK_LIBS PUBLIC
    MLIRDialectUtils
    MLIRTransformUtils
    MLIRArithDialect
    CIRCTFIRRTL
)
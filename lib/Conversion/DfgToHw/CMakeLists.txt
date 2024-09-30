add_mlir_conversion_library(DfgToHw
    DfgToHw.cpp

    DEPENDS
    ConversionIncGen

    LINK_COMPONENTS
    Core

    LINK_LIBS PUBLIC
    MLIRDialectUtils
    MLIRTransformUtils
    DfgDialect
    CIRCTHandshake
    CIRCTComb
    CIRCTHW
    CIRCTSV
)
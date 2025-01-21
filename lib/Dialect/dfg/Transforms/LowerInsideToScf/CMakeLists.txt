add_mlir_conversion_library(DfgLowerInsideToScf
    LowerInsideToScf.cpp

    DEPENDS
    DfgPassesIncGen

    LINK_COMPONENTS
    Core

    LINK_LIBS PUBLIC
    MLIRDialectUtils
    MLIRTransformUtils
    MLIRTosaToLinalg
)
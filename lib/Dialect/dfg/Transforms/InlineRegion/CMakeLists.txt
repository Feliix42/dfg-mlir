add_mlir_conversion_library(InlineRegion
    InlineRegion.cpp

    DEPENDS
    DfgPassesIncGen

    LINK_COMPONENTS
    Core

    LINK_LIBS PUBLIC
    MLIRDialectUtils
    MLIRTransformUtils
)
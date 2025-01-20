add_mlir_conversion_library(Bufferize
    Bufferize.cpp

    DEPENDS
    DfgPassesIncGen

    LINK_COMPONENTS
    Core

    LINK_LIBS PUBLIC
    MLIRDialectUtils
    MLIRTransformUtils
    MLIRBufferizationDialect
)
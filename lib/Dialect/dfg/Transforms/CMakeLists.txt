add_mlir_conversion_library(MLIRDfgTransforms
    BufferizableOpInterfaceImpl.cpp
    FlattenMemref.cpp
    InlineRegion.cpp
    LowerInsideToScf.cpp
    OperatorToProcess.cpp
    PrintOperatorToYaml.cpp

    DEPENDS
    DfgTransformsIncGen

    LINK_COMPONENTS
    Core

    LINK_LIBS PUBLIC
    MLIRDialectUtils
    MLIRTransformUtils
    MLIRBufferizationDialect
    MLIRIR
    MLIRPass
    MLIRTransforms
    MLIRSupport
)

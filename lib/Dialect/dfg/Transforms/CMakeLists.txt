add_mlir_conversion_library(DFGMLIRDfgTransforms
    BufferizableOpInterfaceImpl.cpp
    InlineRegion.cpp
    LowerInsideToScf.cpp
    OperatorToProcess.cpp
    PrintOperatorToYaml.cpp

    DEPENDS
        DFGMLIRDfgTransformsIncGen

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

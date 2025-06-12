add_mlir_conversion_library(DFGMLIRDfgTransforms
    BufferizableOpInterfaceImpl.cpp
    InlineRegion.cpp
    InlineScalarArgument.cpp
    LowerInsideToScf.cpp
    OperatorToProcess.cpp
    PrintGraph.cpp
    PrintOperatorToYaml.cpp
    RemoveScalarGlobals.cpp
    RetrieveGraphLayers.cpp
    SoftTranspose.cpp
    mergingregions.cpp

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

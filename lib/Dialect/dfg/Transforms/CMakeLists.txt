add_mlir_conversion_library(OperatorToProcess
    OperatorToProcess.cpp

    DEPENDS
    DfgPassesIncGen

    LINK_COMPONENTS
    Core

    LINK_LIBS PUBLIC
    MLIRDialectUtils
    MLIRTransformUtils
)
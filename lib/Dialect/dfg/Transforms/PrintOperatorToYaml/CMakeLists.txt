add_mlir_conversion_library(PrintOperatorToYaml
    PrintOperatorToYaml.cpp

    DEPENDS
    DfgPassesIncGen

    LINK_COMPONENTS
    Core

    LINK_LIBS PUBLIC
    MLIRDialectUtils
    MLIRTransformUtils

    LLVMObjectYAML
)
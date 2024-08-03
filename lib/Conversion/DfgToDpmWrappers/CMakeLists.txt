add_mlir_conversion_library(DfgToDpmWrappers
        DfgToDpmWrappers.cpp

    DEPENDS
        ConversionIncGen

    LINK_COMPONENTS
        Core

    LINK_LIBS PUBLIC
        MLIRDialectUtils
        MLIRTransformUtils
        MLIRFuncDialect
        MLIRControlFlowDialect
        MLIRLLVMCommonConversion
        MLIRLLVMDialect
        DfgDialect
)

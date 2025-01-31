add_mlir_conversion_library(ArithIndexToVitis
ArithIndexToVitis.cpp

    DEPENDS
        ConversionIncGen

    LINK_COMPONENTS
        Core

    LINK_LIBS PUBLIC
        MLIRDialectUtils
        MLIRTransformUtils
        MLIRControlFlowDialect
        MLIRArithDialect
        MLIRMathDialect
        DfgDialect
        VitisDialect
)
add_mlir_conversion_library(MathToVitis
MathToVitis.cpp

    DEPENDS
        ConversionIncGen

    LINK_COMPONENTS
        Core

    LINK_LIBS PUBLIC
        MLIRDialectUtils
        MLIRTransformUtils
        MLIRControlFlowDialect
        MLIRMathDialect
        DfgDialect
        VitisDialect
)
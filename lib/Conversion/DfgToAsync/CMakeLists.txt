add_mlir_conversion_library(DfgToAsync
        DfgToAsync.cpp

    DEPENDS
        ConversionIncGen
    
    LINK_COMPONENTS
        Core
    
    LINK_LIBS PUBLIC
        MLIRDialectUtils
        MLIRTransformUtils
        MLIRAsyncDialect
        DfgDialect
)
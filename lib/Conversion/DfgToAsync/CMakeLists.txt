add_mlir_conversion_library(DfgToAsync
        DfgToAsync.cpp

    DEPENDS
        DFGMLIRConversionIncGen
    
    LINK_COMPONENTS
        Core
    
    LINK_LIBS PUBLIC
        MLIRDialectUtils
        MLIRTransformUtils
        MLIRAsyncDialect
        DFGMLIRDfgDialect
)
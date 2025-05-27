add_mlir_conversion_library(DFGMLIRDfgToAsync
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
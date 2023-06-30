add_mlir_conversion_library(DfgToCirct
        DfgToCirct.cpp

    DEPENDS
        ConversionIncGen
    
    LINK_COMPONENTS
        Core
    
    LINK_LIBS PUBLIC
        MLIRDialectUtils
        MLIRTransformUtils
        DfgDialect
        CIRCTHandshake
)
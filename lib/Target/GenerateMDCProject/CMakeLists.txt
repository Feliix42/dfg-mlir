add_mlir_translation_library(DFGMLIRMDCTargetProject
    TranslateRegistration.cpp
    GenerateMDCProject.cpp
    
    LINK_LIBS PUBLIC
    DFGMLIRDfgDialect
    MLIRIR
    MLIRSupport
    MLIRTranslateLib
    MLIRParser
)

add_mlir_translation_library(DFGMLIREmitHLSTargetProject
    TranslateRegistration.cpp
    GenerateVitisProject.cpp

    LINK_LIBS PUBLIC
    DFGMLIREmitHLSDialect
    MLIRIR
    MLIRSupport
    MLIRTranslateLib
)

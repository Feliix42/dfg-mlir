add_mlir_translation_library(DFGMLIRVitisTargetProject
    TranslateRegistration.cpp
    GenerateVitisProject.cpp

    LINK_LIBS PUBLIC
    DFGMLIRVitisDialect
    MLIRIR
    MLIRSupport
    MLIRTranslateLib
)

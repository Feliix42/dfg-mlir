add_mlir_translation_library(VitisTargetProject
    TranslateRegistration.cpp
    GenerateVitisProject.cpp

    LINK_LIBS PUBLIC
    DFGMLIRVitisDialect
    MLIRIR
    MLIRSupport
    MLIRTranslateLib
)

add_mlir_translation_library(VitisTargetTcl
    TranslateRegistration.cpp
    TranslateToVitisTcl.cpp

    LINK_LIBS PUBLIC
    DFGMLIRVitisDialect
    MLIRIR
    MLIRSupport
    MLIRTranslateLib
)

add_mlir_translation_library(VitisTargetCpp
    TranslateRegistration.cpp
    TranslateToVitisCpp.cpp

    LINK_LIBS PUBLIC
    VitisDialect
    MLIRIR
    MLIRSupport
    MLIRTranslateLib
)

add_mlir_translation_library(VitisTargetTcl
    TranslateRegistration.cpp
    TranslateToVitisTcl.cpp

    LINK_LIBS PUBLIC
    VitisDialect
    MLIRIR
    MLIRSupport
    MLIRTranslateLib
)

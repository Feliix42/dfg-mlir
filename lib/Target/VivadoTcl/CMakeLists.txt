add_mlir_translation_library(VivadoTargetTcl
    TranslateRegistration.cpp
    TranslateToVivadoTcl.cpp

    LINK_LIBS PUBLIC
    DfgDialect
    MLIRIR
    MLIRSupport
    MLIRTranslateLib
)

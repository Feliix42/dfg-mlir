add_mlir_translation_library(VivadoTargetTcl
    TranslateRegistration.cpp
    TranslateToVivadoTcl.cpp

    LINK_LIBS PUBLIC
    DFGMLIRDfgDialect
    MLIRIR
    MLIRSupport
    MLIRTranslateLib
)

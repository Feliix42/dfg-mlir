# ###############################################################################
# dfg-translate
#
# The dfg-translate tool.
# ###############################################################################

project(dfg-translate)

add_executable(${PROJECT_NAME}
    dfg-translate.cpp
)

# Link all standard MLIR dialect and conversion libs.
get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(translation_libs GLOBAL PROPERTY MLIR_TRANSLATION_LIBS)
get_property(circt_dialect_libs GLOBAL PROPERTY CIRCT_DIALECT_LIBS)
get_property(circt_translation_libs GLOBAL PROPERTY CIRCT_TRANSLATION_LIBS)
target_link_libraries(${PROJECT_NAME}
    PRIVATE
    MLIRLspServerLib

    VitisTargetCpp

    ${circt_dialect_libs}
    ${circt_translation_libs}

    ${dialect_libs}
    ${translation_libs}
)

# ###############################################################################
# dfg-opt
#
# The dfg-mlir optimizer driver.
# ###############################################################################

project(dfg-opt)

add_executable(${PROJECT_NAME}
    dfg-opt.cpp
)

# Link all standard MLIR dialect and conversion libs.
get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
target_link_libraries(${PROJECT_NAME}
    PRIVATE
    ${dialect_libs}
    ${conversion_libs}

    MLIROptLib
    MLIRTransforms
)

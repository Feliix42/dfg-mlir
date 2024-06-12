################################################################################
# DfgIncGen
#
# MLIR DFG dialect generated includes.
################################################################################

add_custom_target(DfgIncGen)

mlir_gen_enums(Dfg)

# Interfaces.
add_subdirectory(Interfaces)
# Attributes, Dialect, Operations and Types.
add_subdirectory(IR)
# Passes
add_subdirectory(Transforms)

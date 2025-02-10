################################################################################
# VitisIncGen
#
# MLIR Vitis dialect generated includes.
################################################################################

add_custom_target(VitisIncGen)

mlir_gen_enums(Vitis)

# Interfaces.
# add_subdirectory(Interfaces)
# Attributes, Dialect, Operations and Types.
add_subdirectory(IR)
# Passes
add_subdirectory(Transforms)

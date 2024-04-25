################################################################################
# DfgIncGen
#
# MLIR DFG dialect generated includes.
################################################################################

add_custom_target(DfgIncGen)

# Interfaces.
# add_subdirectory(Interfaces)
# Attributes, Dialect, Operations and Types.
add_subdirectory(IR)
add_subdirectory(Transforms)

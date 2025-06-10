# add_subdirectory(Interfaces)
add_subdirectory(IR)
add_subdirectory(Transforms)

add_library(DFGMLIREmitHLSEnums OBJECT
Enums.cpp
)
add_dependencies(DFGMLIREmitHLSEnums DFGMLIREmitHLSEnumsIncGen)

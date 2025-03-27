add_subdirectory(Interfaces)
add_subdirectory(IR)
add_subdirectory(Transforms)

add_library(DFGMLIRDfgEnums OBJECT
    Enums.cpp
)
add_dependencies(DFGMLIRDfgEnums DFGMLIRDfgEnumsIncGen)

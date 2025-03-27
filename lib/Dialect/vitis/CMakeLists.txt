# add_subdirectory(Interfaces)
add_subdirectory(IR)
add_subdirectory(Transforms)

add_library(DFGMLIRVitisEnums OBJECT
Enums.cpp
)
add_dependencies(DFGMLIRVitisEnums DFGMLIRVitisEnumsIncGen)

add_subdirectory(Interfaces)
add_subdirectory(IR)
add_subdirectory(Transforms)

add_library(DfgEnums OBJECT
    Enums.cpp
)
add_dependencies(DfgEnums DfgEnumsIncGen)

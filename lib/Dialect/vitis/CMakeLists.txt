# add_subdirectory(Interfaces)
add_subdirectory(IR)
# add_subdirectory(Transforms)

add_library(VitisEnums OBJECT
Enums.cpp
)
add_dependencies(VitisEnums VitisEnumsIncGen)

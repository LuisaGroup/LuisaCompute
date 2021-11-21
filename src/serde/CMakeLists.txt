set(LUISA_COMPUTE_SERDE
NoSqlPython.cpp
PythonLib.cpp
SerdeExtern.cpp
SimpleBinaryJson.cpp
SimpleJsonLoader.cpp
SimpleJsonValue.cpp

SimpleParser.cpp
YamlParser.cpp
)

add_library(luisa-compute-serde SHARED ${LUISA_COMPUTE_SERDE})
target_link_libraries(luisa-compute-serde PUBLIC luisa-compute-core luisa-compute-vstl)
set_target_properties(luisa-compute-serde PROPERTIES
        UNITY_BUILD ON)
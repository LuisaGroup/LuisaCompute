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

add_library(luisa-compute-serde MODULE ${LUISA_COMPUTE_SERDE})
target_link_libraries(luisa-compute-serde PRIVATE luisa-compute-core luisa-compute-vstl)
set_target_properties(luisa-compute-serde PROPERTIES
        MSVC_RUNTIME_LIBRARY MultiThreadedDLL
        UNITY_BUILD ON)
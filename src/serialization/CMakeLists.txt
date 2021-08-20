set(LUISA_COMPUTE_SERIALIZATION_SOURCES
    dummy.cpp)

add_library(luisa-compute-serialization SHARED ${LUISA_COMPUTE_SERIALIZATION_SOURCES})
target_link_libraries(luisa-compute-serialization PUBLIC luisa-compute-ast)
set_target_properties(luisa-compute-serialization PROPERTIES
                      WINDOWS_EXPORT_ALL_SYMBOLS ON
                      UNITY_BUILD ON)

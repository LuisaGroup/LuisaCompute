##[[[]]]
    set(LUISA_COMPUTE_SERIALIZE_SOURCES
        astserializer.cpp
        funcserializer.cpp
    )
##[[[]]]
add_library(luisa-compute-serialize SHARED ${LUISA_COMPUTE_SERIALIZE_SOURCES})
target_link_libraries(luisa-compute-serialize PUBLIC luisa-compute-vstl luisa-compute-ast)
add_dependencies(luisa-compute-serialize luisa-compute-serde)
set_target_properties(luisa-compute-serialize PROPERTIES
        UNITY_BUILD ON
        WINDOWS_EXPORT_ALL_SYMBOLS ON)

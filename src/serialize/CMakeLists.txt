set(LUISA_COMPUTE_SERIALIZE_SOURCES
        serialize.h
        traits.h
        serialize.cpp
        key_value_pair.h)
add_library(luisa-compute-serialize INTERFACE ${LUISA_COMPUTE_SERIALIZE_SOURCES})
target_link_libraries(luisa-compute-serialize INTERFACE luisa-compute-vstl luisa-compute-ast)
set_target_properties(luisa-compute-serialize PROPERTIES
        UNITY_BUILD ${LUISA_COMPUTE_ENABLE_UNITY_BUILD})

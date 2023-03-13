set(LUISA_BACKEND_COMMON_SOURCES
        default_binary_io.cpp
        default_binary_io.h
        native_resource_ext.h
        resource_generator.h
        string_scratch.cpp
        string_scratch.h
        tex_compress_ext.h)

add_library(luisa-compute-backend-common STATIC ${LUISA_BACKEND_COMMON_SOURCES})
target_link_libraries(luisa-compute-backend-common PUBLIC
        luisa-compute-runtime
        luisa-compute-dsl)

set_target_properties(luisa-compute-backend-common PROPERTIES
        UNITY_BUILD ${LUISA_COMPUTE_ENABLE_UNITY_BUILD})

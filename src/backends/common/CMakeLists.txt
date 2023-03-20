set(LUISA_BACKEND_COMMON_SOURCES
        command_reorder_visitor.h
        default_binary_io.cpp default_binary_io.h
        resource_generator.h
        resource_tracker.h
        string_scratch.cpp string_scratch.h)

if (LUISA_COMPUTE_ENABLE_RUST)
    list(APPEND LUISA_BACKEND_COMMON_SOURCES
            rust_device_common.cpp rust_device_common.h)
endif ()

add_library(luisa-compute-backend-common STATIC ${LUISA_BACKEND_COMMON_SOURCES})
target_link_libraries(luisa-compute-backend-common PUBLIC
        luisa-compute-runtime
        luisa-compute-dsl
        luisa-compute-ir)

if (UNIX AND NOT APPLE)
    target_compile_options(luisa-compute-backend-common PRIVATE -fPIC)
endif ()

set_target_properties(luisa-compute-backend-common PROPERTIES
        UNITY_BUILD ${LUISA_COMPUTE_ENABLE_UNITY_BUILD})

if (LUISA_COMPUTE_ENABLE_TENSOR)
    set(LUISA_COMPUTE_TENSOR_SOURCES
            tensor.cpp)

    add_library(luisa-compute-tensor SHARED ${LUISA_COMPUTE_TENSOR_SOURCES})
    target_link_libraries(luisa-compute-tensor PUBLIC luisa-compute-core)
    target_compile_definitions(luisa-compute-tensor
            PRIVATE LC_TENSOR_EXPORT_DLL=1
            PUBLIC LUISA_ENABLE_TENSOR=1)
    set_target_properties(luisa-compute-tensor PROPERTIES
            UNITY_BUILD ${LUISA_COMPUTE_ENABLE_UNITY_BUILD}
            OUTPUT_NAME lc-tensor)
    luisa_compute_install(tensor SOURCES ${LUISA_COMPUTE_TENSOR_SOURCES})
else ()
    add_library(luisa-compute-tensor INTERFACE)
    luisa_compute_install(tensor)
endif ()

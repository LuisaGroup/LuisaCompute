find_package(LLVM CONFIG)
if (LLVM_FOUND)
    message(STATUS "Build with ISPC")

    ##[[[]]]
    set(ISPC_BACKEND_SOURCES
            runtime/ispc_codegen.cpp
            runtime/ispc_codegenutil.cpp
            runtime/ispc_compiler.cpp
            runtime/ispc_device.cpp
            runtime/ispc_runtime.cpp
            runtime/ispc_shader.cpp
            runtime/ispc_jit_module.cpp
            )
    ##[[[]]]

    luisa_compute_add_backend(ispc SOURCES ${ISPC_BACKEND_SOURCES})
    set_target_properties(luisa-compute-backend-ispc PROPERTIES
            WINDOWS_EXPORT_ALL_SYMBOLS OFF
            UNITY_BUILD ON)

    # copy device libs
    set(ISPC_DEVICE_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/ISPCTest")
    add_custom_target(luisa-compute-backend-ispc-copy-include ALL
            COMMAND ${CMAKE_COMMAND} -E copy_directory
            "${ISPC_DEVICE_INCLUDE_DIR}"
            "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/backends/ispc_device_include")
    add_dependencies(luisa-compute-backend-ispc luisa-compute-backend-ispc-copy-include)
    foreach (CONFIG ${CMAKE_CONFIGURATION_TYPES})
        string(TOLOWER ${CONFIG} config_lower)
        string(TOUPPER ${CONFIG} CONFIG_UPPER)
        add_custom_target(luisa-compute-backend-ispc-copy-include-${config_lower} ALL
                COMMAND ${CMAKE_COMMAND} -E copy_directory
                "${ISPC_DEVICE_INCLUDE_DIR}"
                "${CMAKE_RUNTIME_OUTPUT_DIRECTORY_${CONFIG_UPPER}}/backends/ispc_device_include")
        add_dependencies(luisa-compute-backend-ispc luisa-compute-backend-ispc-copy-include-${config_lower})
    endforeach ()

    # link with llvm for jit
    llvm_map_components_to_libnames(
            LLVM_LIBS core executionengine support mcjit nativecodegen irreader passes analysis)
    target_include_directories(luisa-compute-backend-ispc PRIVATE ${LLVM_INCLUDE_DIRS})
    target_link_libraries(luisa-compute-backend-ispc PRIVATE ${LLVM_LIBS})
endif ()
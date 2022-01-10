message(STATUS "Build with ISPC")

##[[[]]]
set(ISPC_BACKEND_SOURCES
        runtime/ispc_codegen.cpp
        runtime/ispc_codegenutil.cpp
        runtime/ispc_device.cpp
        runtime/ispc_runtime.cpp
        runtime/ispc_shader.cpp
        runtime/ispc_mesh.cpp
        runtime/ispc_accel.cpp
        )
##[[[]]]

luisa_compute_add_backend(ispc SOURCES ${ISPC_BACKEND_SOURCES})
set_target_properties(luisa-compute-backend-ispc PROPERTIES
        WINDOWS_EXPORT_ALL_SYMBOLS OFF
        UNITY_BUILD ON)

# copy device libs
set(ISPC_DEVICE_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/ISPCTest")
if (CMAKE_BUILD_TYPE)
    add_custom_target(lc-ispc-copy-support ALL
            COMMAND ${CMAKE_COMMAND} -E copy_directory
            "${ISPC_DEVICE_INCLUDE_DIR}"
            "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/backends/ispc_support")
    add_dependencies(luisa-compute-backend-ispc lc-ispc-copy-support)
endif ()
foreach (CONFIG ${CMAKE_CONFIGURATION_TYPES})
    string(TOLOWER ${CONFIG} config_lower)
    string(TOUPPER ${CONFIG} CONFIG_UPPER)
    add_custom_target(lc-ispc-copy-support-${config_lower} ALL
            COMMAND ${CMAKE_COMMAND} -E copy_directory
            "${ISPC_DEVICE_INCLUDE_DIR}"
            "${CMAKE_RUNTIME_OUTPUT_DIRECTORY_${CONFIG_UPPER}}/backends/ispc_support")
    add_dependencies(luisa-compute-backend-ispc lc-ispc-copy-support-${config_lower})
endforeach ()

if (WIN32)
    option(LUISA_COMPUTE_ENABLE_ISPC_LLVM_JIT "Enable LLVM JIT for ISPC kernels" ON)
    if (LUISA_COMPUTE_ENABLE_ISPC_LLVM_JIT)
        find_package(LLVM)
    endif ()
else ()
    find_package(LLVM REQUIRED)
endif ()

if (LLVM_FOUND) # link with llvm for jit
    message(STATUS "Use LLVM JIT for ISPC kernels")
    llvm_map_components_to_libnames(
            LLVM_LIBS core executionengine support mcjit nativecodegen irreader passes analysis)
    target_sources(luisa-compute-backend-ispc PRIVATE runtime/ispc_jit_module.cpp)
    message(STATUS "LLVM: ${LLVM_INCLUDE_DIRS}")
    target_include_directories(luisa-compute-backend-ispc PRIVATE ${LLVM_INCLUDE_DIRS})
    target_link_libraries(luisa-compute-backend-ispc PRIVATE ${LLVM_LIBS})
    target_compile_definitions(luisa-compute-backend-ispc PRIVATE LUISA_COMPUTE_ISPC_LLVM_JIT)
    if (CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
        target_compile_options(luisa-compute-backend-ispc PRIVATE /wd4624 /wd4996) # do not complain about LLVM
    endif ()
else () # fallback to DLL
    message(STATUS "Use runtime DLL loading for ISPC kernels")
    target_sources(luisa-compute-backend-ispc PRIVATE runtime/ispc_dll_module.cpp)
endif ()

# LLVMConfig.cmake may export the CRT mode used to build LLVM. Preserve the
# project's value so including LLVM never makes a later ABI check tautological.
set(_luisa_simd_had_msvc_runtime_library FALSE)
if (DEFINED CMAKE_MSVC_RUNTIME_LIBRARY)
    set(_luisa_simd_had_msvc_runtime_library TRUE)
    set(_luisa_simd_project_msvc_runtime_library
            "${CMAKE_MSVC_RUNTIME_LIBRARY}")
endif ()
find_package(LLVM CONFIG REQUIRED)
if (_luisa_simd_had_msvc_runtime_library)
    set(CMAKE_MSVC_RUNTIME_LIBRARY
            "${_luisa_simd_project_msvc_runtime_library}")
else ()
    unset(CMAKE_MSVC_RUNTIME_LIBRARY)
endif ()

if (LLVM_LINK_LLVM_DYLIB AND TARGET LLVM)
    set(_luisa_simd_llvm_libraries LLVM)
else ()
    llvm_map_components_to_libnames(_luisa_simd_llvm_libraries
            core
            support
            analysis
            passes
            executionengine
            orcjit
            nativecodegen)
endif ()

message(STATUS "Build SIMD CPU codegen core with LLVM ${LLVM_PACKAGE_VERSION}")

add_library(luisa-compute-simd-schedule STATIC
        schedule/schedule_ir.cpp
        schedule/warp_uniformity.cpp
        schedule/xir_to_schedule.cpp)
target_compile_features(luisa-compute-simd-schedule PUBLIC cxx_std_20)
target_include_directories(luisa-compute-simd-schedule PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/schedule)
target_link_libraries(luisa-compute-simd-schedule PUBLIC
        luisa-compute-xir)
set_target_properties(luisa-compute-simd-schedule PROPERTIES
        POSITION_INDEPENDENT_CODE ON
        UNITY_BUILD OFF)

add_library(luisa-compute-simd-llvm STATIC
        llvm/llvm_jit.cpp
        llvm/llvm_schedule_codegen.cpp
        llvm/llvm_value_layout.cpp
        llvm/llvm_warp_collectives.cpp)
target_compile_features(luisa-compute-simd-llvm PUBLIC cxx_std_20)
target_include_directories(luisa-compute-simd-llvm
        PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/llvm
        ${CMAKE_CURRENT_SOURCE_DIR}/schedule)
target_include_directories(luisa-compute-simd-llvm SYSTEM PUBLIC
        ${LLVM_INCLUDE_DIRS})
target_link_libraries(luisa-compute-simd-llvm
        PUBLIC luisa-compute-simd-schedule
        PRIVATE ${_luisa_simd_llvm_libraries})
target_compile_definitions(luisa-compute-simd-llvm
        PRIVATE ${LLVM_DEFINITIONS})
set_target_properties(luisa-compute-simd-llvm PROPERTIES
        POSITION_INDEPENDENT_CODE ON
        UNITY_BUILD OFF)

add_library(luisa-compute-simd-compiler STATIC
        simd_compiler.cpp)
target_compile_features(luisa-compute-simd-compiler PUBLIC cxx_std_20)
target_include_directories(luisa-compute-simd-compiler PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(luisa-compute-simd-compiler PUBLIC
        luisa-compute-simd-llvm)
set_target_properties(luisa-compute-simd-compiler PROPERTIES
        POSITION_INDEPENDENT_CODE ON
        UNITY_BUILD OFF)

luisa_compute_add_backend(simd SOURCES
        ../fallback/fallback_texture.cpp
        runtime/simd_buffer.cpp
        runtime/simd_device.cpp
        runtime/simd_event.cpp
        runtime/simd_shader.cpp
        runtime/simd_stream.cpp
        runtime/simd_texture.cpp)
target_link_libraries(luisa-compute-backend-simd PRIVATE
        luisa-compute-simd-compiler)
set_target_properties(luisa-compute-backend-simd PROPERTIES
        UNITY_BUILD OFF)

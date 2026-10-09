file(GLOB LUISA_COMPUTE_SPIRV_SOURCES CONFIGURE_DEPENDS spirv_codegen/*.cpp)
# The bundled static glslang target provides its SPIR-V builder and frontend
# internals together, including the private APIs used by this code generator.
set(_LUISA_GLSLANG_DIR
        "${CMAKE_CURRENT_LIST_DIR}/../../../ext/glslang")
add_library(luisa-compute-spirv STATIC ${LUISA_COMPUTE_SPIRV_SOURCES})
set_target_properties(luisa-compute-spirv PROPERTIES
        POSITION_INDEPENDENT_CODE ON
        OUTPUT_NAME luisa-spirv)
target_include_directories(luisa-compute-spirv PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<BUILD_INTERFACE:${_LUISA_GLSLANG_DIR}>)
target_link_libraries(luisa-compute-spirv PUBLIC
        luisa-compute-runtime
        luisa-compute-xir
        glslang
        SPIRV-Tools-opt)
if (NOT WIN32)
    target_compile_options(luisa-compute-spirv PUBLIC
            $<$<CXX_COMPILER_ID:Clang,AppleClang>:-fms-extensions>)
endif ()
target_precompile_headers(luisa-compute-spirv PRIVATE spirv_codegen/lc_spirv_pch.h)

# Keep build-time SPIR-V validation available even though the upstream
# SPIRV-Tools command-line executables are disabled in the bundled build.
add_executable(luisa_validate_spirv validate_spirv.cpp)
set_target_properties(luisa_validate_spirv PROPERTIES
        EXCLUDE_FROM_ALL TRUE
        EXCLUDE_FROM_DEFAULT_BUILD TRUE)
target_link_libraries(luisa_validate_spirv PRIVATE SPIRV-Tools-static)

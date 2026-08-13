file(GLOB LUISA_COMPUTE_SPIRV_SOURCES CONFIGURE_DEPENDS spirv_codegen/*.cpp)
# Build glslang SPIR-V builder sources directly into this lib (mirrors xmake lc-glslang).
# The upstream `SPIRV` shared target hides spv:: symbols (visibility=hidden), which makes
# linking against spv::Builder etc. fail at runtime. Compiling the sources here gives us
# full access to spv::Builder, GlslangToSpv, etc. without depending on shared lib exports.
set(_LUISA_GLSLANG_DIR
        "${CMAKE_CURRENT_LIST_DIR}/../../../ext/glslang")
set(_LUISA_GLSLANG_SPIRV_DIR "${_LUISA_GLSLANG_DIR}/SPIRV")
set(LUISA_COMPUTE_SPIRV_GLSLANG_SOURCES
        ${_LUISA_GLSLANG_SPIRV_DIR}/GlslangToSpv.cpp
        ${_LUISA_GLSLANG_SPIRV_DIR}/InReadableOrder.cpp
        ${_LUISA_GLSLANG_SPIRV_DIR}/Logger.cpp
        ${_LUISA_GLSLANG_SPIRV_DIR}/SpvBuilder.cpp
        ${_LUISA_GLSLANG_SPIRV_DIR}/SpvPostProcess.cpp
        ${_LUISA_GLSLANG_SPIRV_DIR}/doc.cpp
        ${_LUISA_GLSLANG_SPIRV_DIR}/SpvTools.cpp
        ${_LUISA_GLSLANG_SPIRV_DIR}/disassemble.cpp)
add_library(luisa-compute-spirv STATIC
        ${LUISA_COMPUTE_SPIRV_SOURCES}
        ${LUISA_COMPUTE_SPIRV_GLSLANG_SOURCES})
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

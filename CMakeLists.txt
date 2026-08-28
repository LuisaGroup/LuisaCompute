cmake_minimum_required(VERSION 3.26...3.29)
cmake_policy(VERSION 3.26)

set(CMAKE_POLICY_DEFAULT_CMP0042 NEW)
set(CMAKE_POLICY_DEFAULT_CMP0048 NEW)
set(CMAKE_POLICY_DEFAULT_CMP0063 NEW)
set(CMAKE_POLICY_DEFAULT_CMP0068 NEW)
set(CMAKE_POLICY_DEFAULT_CMP0069 NEW)
set(CMAKE_POLICY_DEFAULT_CMP0072 NEW)
set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)
set(CMAKE_POLICY_DEFAULT_CMP0091 NEW)
set(CMAKE_POLICY_DEFAULT_CMP0135 NEW)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# set basic compilation options
include(scripts/setup_compilation.cmake)

# parse version from include/luisa/version.h
file(READ "${CMAKE_CURRENT_SOURCE_DIR}/include/luisa/version.h" LUISA_VERSION_H_CONTENTS)
string(REGEX MATCH "#define LUISA_COMPUTE_VERSION_MAJOR ([0-9]+)" _ ${LUISA_VERSION_H_CONTENTS})
set(LUISA_COMPUTE_VERSION_MAJOR "${CMAKE_MATCH_1}")
string(REGEX MATCH "#define LUISA_COMPUTE_VERSION_MINOR ([0-9]+)" _ ${LUISA_VERSION_H_CONTENTS})
set(LUISA_COMPUTE_VERSION_MINOR "${CMAKE_MATCH_1}")
string(REGEX MATCH "#define LUISA_COMPUTE_VERSION_PATCH ([0-9]+)" _ ${LUISA_VERSION_H_CONTENTS})
set(LUISA_COMPUTE_VERSION_PATCH "${CMAKE_MATCH_1}")

set(LUISA_COMPUTE_VERSION
        "${LUISA_COMPUTE_VERSION_MAJOR}.${LUISA_COMPUTE_VERSION_MINOR}.${LUISA_COMPUTE_VERSION_PATCH}"
        CACHE STRING "Version of LuisaCompute"
        FORCE)
message(STATUS "Configuring LuisaCompute ${LUISA_COMPUTE_VERSION}...")

# define the project
project(LuisaCompute LANGUAGES C CXX VERSION ${LUISA_COMPUTE_VERSION})

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

if (${CMAKE_CURRENT_SOURCE_DIR} STREQUAL ${CMAKE_SOURCE_DIR})
    set(LUISA_COMPUTE_MASTER_PROJECT ON)
else ()
    set(LUISA_COMPUTE_MASTER_PROJECT OFF)
endif ()

option(LUISA_COMPUTE_ENABLE_PACKAGE_DISTRIBUTION
        "Configure a relocatable CPack distribution build" OFF)

set(_LUISA_COMPUTE_DEFAULT_DEVELOPER_FEATURES ON)
set(_LUISA_COMPUTE_DEFAULT_BUILD_TESTS ${LUISA_COMPUTE_MASTER_PROJECT})
if (LUISA_COMPUTE_ENABLE_PACKAGE_DISTRIBUTION)
    # The fallback backend has additional toolchain/runtime distribution
    # requirements. It remains opt-in for a binary SDK while the native GPU
    # backend for the host platform is packaged and tested end to end.
    set(_LUISA_COMPUTE_DEFAULT_DEVELOPER_FEATURES OFF)
    set(_LUISA_COMPUTE_DEFAULT_BUILD_TESTS OFF)
endif ()

if (NOT SKBUILD AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/scripts/options.cmake")
    message(STATUS "Loading bootstrap options from \"scripts/options.cmake\"")
    include(scripts/options.cmake)
else ()
    option(LUISA_COMPUTE_ENABLE_DSL "Enable C++ DSL" ON)
    option(LUISA_COMPUTE_ENABLE_TENSOR "Enable C++ DSL tensor extension" OFF)
    option(LUISA_COMPUTE_ENABLE_DX "Enable DirectX backend" ON)
    option(LUISA_COMPUTE_ENABLE_METAL "Enable Metal backend" ON)
    option(LUISA_COMPUTE_ENABLE_METAL4 "Enable experimental Metal 4 AIR backend" OFF)
    option(LUISA_COMPUTE_ENABLE_HIP "Enable HIP backend" OFF) # still WIP
    option(LUISA_COMPUTE_ENABLE_CUDA "Enable CUDA backend" ON)
    option(LUISA_COMPUTE_ENABLE_CUDA_EXT_LCUB "Enable CUDA extension: LCUB" OFF)
    option(LUISA_COMPUTE_ENABLE_VULKAN "Enable Vulkan backend" ON)
    option(LUISA_COMPUTE_ENABLE_FALLBACK "Enable Fallback backend" ${_LUISA_COMPUTE_DEFAULT_DEVELOPER_FEATURES})
    option(LUISA_COMPUTE_ENABLE_GUI "Enable GUI support" ON)
    option(LUISA_COMPUTE_ENABLE_CLANG_CXX "Enable ClangTooling-based C++ shading language" OFF)
    option(LUISA_COMPUTE_BUILD_TESTS "Build tests for LuisaCompute" ${_LUISA_COMPUTE_DEFAULT_BUILD_TESTS})
    option(LUISA_COMPUTE_ENABLE_SAFE_MODE "Enable safe mode for runtime" OFF)
endif ()

# The SIMD CPU backend is staged behind an explicit option until the Phase 2
# runtime shell reaches parity with the selected fallback core suite.
option(LUISA_COMPUTE_ENABLE_SIMD "Enable experimental SIMD CPU backend" OFF)

# iOS device tests are signed application bundles rather than host executables.
# Keep them separate from the ordinary test suite so cross-compiling the runtime
# does not implicitly pull test applications, provisioning, or UIKit into a
# production build.
option(LUISA_COMPUTE_BUILD_IOS_TESTS
        "Build the opt-in Metal4 iOS device conformance applications" OFF)
option(LUISA_COMPUTE_BUILD_IOS_EXAMPLES
        "Build the opt-in interactive Metal4 iOS example applications" OFF)
option(LUISA_COMPUTE_BUILD_IOS_BENCHMARKS
        "Build the opt-in iOS Metal backend comparison application" OFF)
set(LUISA_COMPUTE_IOS_BENCHMARK_BACKEND "metal4" CACHE STRING
        "Backend linked into the iOS comparison application (metal or metal4)")
set_property(CACHE LUISA_COMPUTE_IOS_BENCHMARK_BACKEND PROPERTY STRINGS
        metal metal4)
set(LUISA_COMPUTE_IOS_BENCHMARK_BUNDLE_ID "" CACHE STRING
        "Optional bundle identifier for signing the iOS comparison application")
if ((LUISA_COMPUTE_BUILD_IOS_TESTS OR
        LUISA_COMPUTE_BUILD_IOS_EXAMPLES OR
        LUISA_COMPUTE_BUILD_IOS_BENCHMARKS) AND NOT APPLE)
    message(FATAL_ERROR
            "The iOS applications require an Apple host/toolchain.")
endif ()
if ((LUISA_COMPUTE_BUILD_IOS_TESTS OR
        LUISA_COMPUTE_BUILD_IOS_EXAMPLES) AND
        NOT LUISA_COMPUTE_ENABLE_METAL4)
    message(FATAL_ERROR
            "The iOS applications require LUISA_COMPUTE_ENABLE_METAL4=ON.")
endif ()
if (LUISA_COMPUTE_BUILD_IOS_EXAMPLES AND
        NOT CMAKE_SYSTEM_NAME STREQUAL "iOS")
    message(FATAL_ERROR
            "LUISA_COMPUTE_BUILD_IOS_EXAMPLES requires an iOS toolchain.")
endif ()
if (LUISA_COMPUTE_BUILD_IOS_BENCHMARKS)
    if (NOT CMAKE_SYSTEM_NAME STREQUAL "iOS")
        message(FATAL_ERROR
                "LUISA_COMPUTE_BUILD_IOS_BENCHMARKS requires an iOS toolchain.")
    endif ()
    string(TOLOWER "${LUISA_COMPUTE_IOS_BENCHMARK_BACKEND}"
            _LUISA_COMPUTE_IOS_BENCHMARK_BACKEND)
    if (_LUISA_COMPUTE_IOS_BENCHMARK_BACKEND STREQUAL "metal4")
        if (NOT LUISA_COMPUTE_ENABLE_METAL4)
            message(FATAL_ERROR
                    "The Metal4 iOS benchmark requires LUISA_COMPUTE_ENABLE_METAL4=ON.")
        endif ()
    elseif (_LUISA_COMPUTE_IOS_BENCHMARK_BACKEND STREQUAL "metal")
        if (NOT LUISA_COMPUTE_ENABLE_METAL)
            message(FATAL_ERROR
                    "The old Metal iOS benchmark requires LUISA_COMPUTE_ENABLE_METAL=ON.")
        endif ()
    else ()
        message(FATAL_ERROR
                "LUISA_COMPUTE_IOS_BENCHMARK_BACKEND must be 'metal' or 'metal4'.")
    endif ()
endif ()

option(LUISA_COMPUTE_ENABLE_VK_XIR_SPIRV "Enable XIR to SPIR-V code generation for Vulkan backend" ON)
option(LUISA_COMPUTE_ENABLE_VK_AST_LLVM_SPIRV "Use AST LLVM codegen for Vulkan backend SPIR-V generation" OFF)
option(LUISA_COMPUTE_VULKAN_ENABLE_DXC_COMPATIBILITY
        "Enable legacy Vulkan HLSL-to-SPIR-V compatibility through DXC" ON)
if (LUISA_COMPUTE_ENABLE_VK_XIR_SPIRV AND LUISA_COMPUTE_ENABLE_VK_AST_LLVM_SPIRV)
    message(FATAL_ERROR "Vulkan compute shader codegen options are mutually exclusive. "
            "Use LUISA_COMPUTE_ENABLE_VK_XIR_SPIRV for the default native SPIR-V path, "
            "or LUISA_COMPUTE_ENABLE_VK_AST_LLVM_SPIRV for the experimental LLVM SPIR-V path.")
endif ()

option(LUISA_COMPUTE_USE_SYSTEM_LIBS "Prefer system libraries over bundled ones"
        ${LUISA_COMPUTE_ENABLE_PACKAGE_DISTRIBUTION})

option(LUISA_COMPUTE_PACKAGE_REQUIRE_SYSTEM_LIBS
        "Fail a distribution build when a preferred system library is unavailable" OFF)

option(LUISA_COMPUTE_ENABLE_DX_CUDA_INTEROP
        "Enable CUDA interop in the DirectX backend when CUDAToolkit is available"
        ${_LUISA_COMPUTE_DEFAULT_DEVELOPER_FEATURES})

option(LUISA_COMPUTE_USE_SYSTEM_STL "Use system STL instead of the bundled EASTL" ON)
option(LUISA_COMPUTE_USE_SYSTEM_GLFW "Use system GLFW instead of the bundled one" ${LUISA_COMPUTE_USE_SYSTEM_LIBS})
option(LUISA_COMPUTE_USE_SYSTEM_LMDB "Use system LMDB instead of the bundled one" ${LUISA_COMPUTE_USE_SYSTEM_LIBS})
option(LUISA_COMPUTE_USE_SYSTEM_REPROC "Use system reproc instead of the bundled one" ${LUISA_COMPUTE_USE_SYSTEM_LIBS})
option(LUISA_COMPUTE_USE_SYSTEM_SPDLOG "Use system spdlog instead of the bundled one" ${LUISA_COMPUTE_USE_SYSTEM_LIBS})
option(LUISA_COMPUTE_USE_SYSTEM_XXHASH "Use system xxHash instead of the bundled one" ${LUISA_COMPUTE_USE_SYSTEM_LIBS})
option(LUISA_COMPUTE_USE_SYSTEM_YYJSON "Use system yyjson instead of the bundled one" ${LUISA_COMPUTE_USE_SYSTEM_LIBS})
option(LUISA_COMPUTE_USE_SYSTEM_MAGIC_ENUM "Use system magic enum instead of the bundled one" ${LUISA_COMPUTE_USE_SYSTEM_LIBS})
option(LUISA_COMPUTE_USE_SYSTEM_MARL "Use system marl instead of the bundled one" ${LUISA_COMPUTE_USE_SYSTEM_LIBS})
option(LUISA_COMPUTE_DISABLE_WIN_MESSAGE_BOX "Disable windows message box" ${LUISA_COMPUTE_DISABLE_WIN_MESSAGE_BOX})

if (LUISA_COMPUTE_ENABLE_PACKAGE_DISTRIBUTION AND
        LUISA_COMPUTE_USE_SYSTEM_LIBS)
    include(cmake/LuisaComputeSystemDependencies.cmake)
endif ()

option(LUISA_COMPUTE_ENABLE_SANITIZERS "Enable sanitizers" OFF)
option(LUISA_COMPUTE_ENABLE_UNITY_BUILD "Enable unity build (which might accelerate the compilation but could break VS Code's clangd indexing)" OFF)
option(LUISA_COMPUTE_CHECK_BACKEND_DEPENDENCIES "Check dependencies of backends (and disable the unsatisfied ones instead of reporting a configuration error)" ON)

if (UNIX AND NOT APPLE)
    option(LUISA_COMPUTE_ENABLE_WAYLAND "Enable Wayland support in GUI and Vulkan swapchains" OFF)
else ()
    set(LUISA_COMPUTE_ENABLE_WAYLAND OFF)
endif ()

if (LUISA_COMPUTE_ENABLE_HIP)
    option(LUISA_COMPUTE_HIP_PLATFORM_NVIDIA "Use HIP with NVIDIA backend (requires HIP SDK with CUDA support)" OFF)
    if (LUISA_COMPUTE_HIP_PLATFORM_NVIDIA)
        set(CMAKE_HIP_PLATFORM "nvidia")
        set(HIP_PLATFORM "nvidia")
    endif ()
endif ()

# validate the options
include(scripts/validate_options.cmake)

# enable objc and objcxx for Apple platforms
if (APPLE)
    if (LUISA_COMPUTE_ENABLE_METAL OR LUISA_COMPUTE_ENABLE_METAL4 OR LUISA_COMPUTE_ENABLE_GUI)
        enable_language(OBJC)
        enable_language(OBJCXX)
    endif ()
endif ()

# Open Image Denoise
option(LUISA_COMPUTE_DOWNLOAD_OIDN "Download OpenImageDenoise for denoiser extension" OFF)

# nvCOMP
if (LUISA_COMPUTE_ENABLE_CUDA)
    option(LUISA_COMPUTE_DOWNLOAD_NVCOMP "Download the nvCOMP library for CUDA GPU decompression" OFF)
endif ()

# setup the build output directories
include(scripts/setup_output_dirs.cmake)

# rpath
set(CMAKE_MACOSX_RPATH ON)
set(CMAKE_SKIP_BUILD_RPATH OFF)
set(CMAKE_BUILD_RPATH_USE_ORIGIN ON)
set(CMAKE_BUILD_WITH_INSTALL_RPATH ON)
# Keep installed binaries relocatable. Appending linker search directories here
# leaks dependency build paths into RUNPATH (for example, a locally built
# libuuid directory) even when the exported CMake targets themselves are clean.
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH OFF)

if (APPLE)
    set(CMAKE_INSTALL_RPATH "@loader_path;@loader_path/../bin;@loader_path/../lib")
elseif (UNIX)
    set(CMAKE_INSTALL_RPATH "$ORIGIN;$ORIGIN/../bin;$ORIGIN/../lib")
endif ()

if (NOT CMAKE_INSTALL_PREFIX)
    set(CMAKE_INSTALL_PREFIX "dist")
endif ()

if (SKBUILD)
    set(CMAKE_INSTALL_INCLUDEDIR ${SKBUILD_NULL_DIR})
    set(CMAKE_INSTALL_DOCDIR ${SKBUILD_NULL_DIR})
    if (WIN32)
        set(CMAKE_INSTALL_LIBDIR ${SKBUILD_NULL_DIR})
    else ()
        set(CMAKE_INSTALL_LIBDIR ${SKBUILD_PLATLIB_DIR}/luisa/dylibs)
    endif ()
    set(CMAKE_INSTALL_BINDIR ${SKBUILD_PLATLIB_DIR}/luisa/dylibs)
else ()
    include(GNUInstallDirs)
    set(LUISA_COMPUTE_INSTALL_CMAKEDIR
            "${CMAKE_INSTALL_LIBDIR}/cmake/LuisaCompute"
            CACHE STRING "Installation directory for LuisaCompute CMake package files")
    mark_as_advanced(LUISA_COMPUTE_INSTALL_CMAKEDIR)
endif ()

install(FILES README.md LICENSE
        DESTINATION ${CMAKE_INSTALL_DOCDIR}/)

# include
file(GLOB_RECURSE LUISA_COMPUTE_HEADERS CONFIGURE_DEPENDS
        "include/luisa/*.h"
        "include/luisa/*.hpp")
add_library(luisa-compute-include INTERFACE ${LUISA_COMPUTE_HEADERS})
target_include_directories(luisa-compute-include INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_compile_features(luisa-compute-include INTERFACE c_std_11 cxx_std_20)

if (UNIX AND NOT APPLE AND LUISA_COMPUTE_ENABLE_WAYLAND)
    target_compile_definitions(luisa-compute-include INTERFACE LUISA_ENABLE_WAYLAND=1)
endif ()

# add platform macros
if (WIN32)
    target_compile_definitions(luisa-compute-include INTERFACE LUISA_PLATFORM_WINDOWS=1)
elseif (UNIX)
    target_compile_definitions(luisa-compute-include INTERFACE LUISA_PLATFORM_UNIX=1)
    if (APPLE)
        target_compile_definitions(luisa-compute-include INTERFACE LUISA_PLATFORM_APPLE=1)
        if (CMAKE_SYSTEM_NAME STREQUAL "iOS")
            target_compile_definitions(luisa-compute-include INTERFACE LUISA_PLATFORM_IOS=1)
        endif ()
    endif ()
endif ()

# sanitizers
if (LUISA_COMPUTE_ENABLE_SANITIZERS)
    if (CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
        set(LUISA_COMPUTE_SANITIZER_FLAGS "/fsanitize=address")
        target_compile_options(luisa-compute-include INTERFACE ${LUISA_COMPUTE_SANITIZER_FLAGS})
    else ()
        set(LUISA_COMPUTE_SANITIZER_FLAGS "-fsanitize=address" "-fsanitize=leak" "-fsanitize=undefined")
        target_compile_options(luisa-compute-include INTERFACE ${LUISA_COMPUTE_SANITIZER_FLAGS})
        target_link_options(luisa-compute-include INTERFACE ${LUISA_COMPUTE_SANITIZER_FLAGS})
    endif ()
    mark_as_advanced(LUISA_COMPUTE_SANITIZER_FLAGS)
endif ()

if (LUISA_COMPUTE_ENABLE_DSL)
    target_compile_definitions(luisa-compute-include INTERFACE LUISA_ENABLE_DSL=1)
endif ()

install(TARGETS luisa-compute-include
        EXPORT LuisaComputeTargets)

install(DIRECTORY include/luisa
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
        FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp")

# Global definition to avoid MSVC ABI break.
# See https://github.com/microsoft/STL/releases/tag/vs-2022-17.10 for details.
if (WIN32)
    add_compile_definitions(_DISABLE_CONSTEXPR_MUTEX_CONSTRUCTOR)
    target_compile_definitions(luisa-compute-include INTERFACE _DISABLE_CONSTEXPR_MUTEX_CONSTRUCTOR)
endif ()

add_subdirectory(utils)
add_subdirectory(src)

if (LUISA_COMPUTE_BUILD_IOS_EXAMPLES)
    add_subdirectory(examples/ios)
endif ()
if (LUISA_COMPUTE_BUILD_IOS_BENCHMARKS)
    add_subdirectory(examples/ios/benchmark)
endif ()

if (NOT SKBUILD)
    include(CMakePackageConfigHelpers)
    get_property(LUISA_COMPUTE_BUILT_BACKENDS
            GLOBAL PROPERTY LUISA_COMPUTE_BUILT_BACKENDS)
    if (NOT LUISA_COMPUTE_BUILT_BACKENDS)
        set(LUISA_COMPUTE_BUILT_BACKENDS "")
    endif ()
    get_property(LUISA_COMPUTE_DX_CUDA_INTEROP_ENABLED
            GLOBAL PROPERTY LUISA_COMPUTE_DX_CUDA_INTEROP_ENABLED)
    if (NOT LUISA_COMPUTE_DX_CUDA_INTEROP_ENABLED)
        set(LUISA_COMPUTE_DX_CUDA_INTEROP_ENABLED OFF)
    endif ()
    if (PROJECT_VERSION_MAJOR EQUAL 0)
        set(LUISA_COMPUTE_VERSION_COMPATIBILITY SameMinorVersion)
    else ()
        set(LUISA_COMPUTE_VERSION_COMPATIBILITY SameMajorVersion)
    endif ()
    configure_package_config_file(
            cmake/LuisaComputeConfig.cmake.in
            "${CMAKE_CURRENT_BINARY_DIR}/LuisaComputeConfig.cmake"
            INSTALL_DESTINATION "${LUISA_COMPUTE_INSTALL_CMAKEDIR}"
            PATH_VARS
            CMAKE_INSTALL_BINDIR
            CMAKE_INSTALL_LIBDIR
            CMAKE_INSTALL_INCLUDEDIR)
    write_basic_package_version_file(
            "${CMAKE_CURRENT_BINARY_DIR}/LuisaComputeConfigVersion.cmake"
            VERSION "${PROJECT_VERSION}"
            COMPATIBILITY ${LUISA_COMPUTE_VERSION_COMPATIBILITY})
    install(FILES
            "${CMAKE_CURRENT_BINARY_DIR}/LuisaComputeConfig.cmake"
            "${CMAKE_CURRENT_BINARY_DIR}/LuisaComputeConfigVersion.cmake"
            DESTINATION "${LUISA_COMPUTE_INSTALL_CMAKEDIR}")
endif ()

if (LUISA_COMPUTE_BUILD_TESTS)
    enable_testing()
    add_subdirectory(examples)
    add_subdirectory(tutorials)
endif()

if (NOT SKBUILD AND LUISA_COMPUTE_ENABLE_PACKAGE_DISTRIBUTION)
    include(cmake/LuisaComputeCPack.cmake)
endif ()

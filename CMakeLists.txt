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

if (CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
    set(LUISA_COMPUTE_MASTER_PROJECT ON)
else ()
    set(LUISA_COMPUTE_MASTER_PROJECT OFF)
endif ()

if (NOT SKBUILD AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/scripts/options.cmake")
    message(STATUS "Loading bootstrap options from \"scripts/options.cmake\"")
    include(scripts/options.cmake)
else ()
    option(LUISA_COMPUTE_ENABLE_DSL "Enable C++ DSL" ON)
    option(LUISA_COMPUTE_ENABLE_TENSOR "Enable C++ DSL tensor extension" OFF)
    option(LUISA_COMPUTE_ENABLE_DX "Enable DirectX backend" ON)
    option(LUISA_COMPUTE_ENABLE_METAL "Enable Metal backend" ON)
    option(LUISA_COMPUTE_ENABLE_CUDA "Enable CUDA backend" ON)
    option(LUISA_COMPUTE_ENABLE_CUDA_EXT_LCUB "Enable CUDA extension: LCUB" OFF)
    option(LUISA_COMPUTE_ENABLE_VULKAN "Enable Vulkan backend" ON)
    option(LUISA_COMPUTE_ENABLE_CPU "Enable CPU backend" ON)
    option(LUISA_COMPUTE_ENABLE_FALLBACK "Enable Fallback backend" ON)
    option(LUISA_COMPUTE_ENABLE_REMOTE "Enable Remote backend" ON)
    option(LUISA_COMPUTE_ENABLE_GUI "Enable GUI support" ON)
    option(LUISA_COMPUTE_ENABLE_CLANG_CXX "Enable ClangTooling-based C++ shading language" OFF)
    option(LUISA_COMPUTE_BUILD_TESTS "Build tests for LuisaCompute" ${LUISA_COMPUTE_MASTER_PROJECT})
    option(LUISA_COMPUTE_EMBREE_ZIP_PATH "Path to embree zip file. Use this if embree fails to download" "")
endif ()

option(LUISA_COMPUTE_USE_SYSTEM_LIBS "Prefer system libraries over bundled ones" OFF)

option(LUISA_COMPUTE_USE_SYSTEM_STL "Use system STL instead of the bundled EASTL" ${LUISA_COMPUTE_USE_SYSTEM_LIBS})
option(LUISA_COMPUTE_USE_SYSTEM_GLFW "Use system GLFW instead of the bundled one" ${LUISA_COMPUTE_USE_SYSTEM_LIBS})
option(LUISA_COMPUTE_USE_SYSTEM_LMDB "Use system LMDB instead of the bundled one" ${LUISA_COMPUTE_USE_SYSTEM_LIBS})
option(LUISA_COMPUTE_USE_SYSTEM_REPROC "Use system reproc instead of the bundled one" ${LUISA_COMPUTE_USE_SYSTEM_LIBS})
option(LUISA_COMPUTE_USE_SYSTEM_SPDLOG "Use system spdlog instead of the bundled one" ${LUISA_COMPUTE_USE_SYSTEM_LIBS})
option(LUISA_COMPUTE_USE_SYSTEM_XXHASH "Use system xxHash instead of the bundled one" ${LUISA_COMPUTE_USE_SYSTEM_LIBS})
option(LUISA_COMPUTE_USE_SYSTEM_YYJSON "Use system yyjson instead of the bundled one" ${LUISA_COMPUTE_USE_SYSTEM_LIBS})
option(LUISA_COMPUTE_USE_SYSTEM_VOLK "Use system volk instead of the bundled one" ${LUISA_COMPUTE_USE_SYSTEM_LIBS})
option(LUISA_COMPUTE_USE_SYSTEM_MAGIC_ENUM "Use system magic enum instead of the bundled one" ${LUISA_COMPUTE_USE_SYSTEM_LIBS})
option(LUISA_COMPUTE_USE_SYSTEM_MARL "Use system marl instead of the bundled one" ${LUISA_COMPUTE_USE_SYSTEM_LIBS})

option(LUISA_COMPUTE_ENABLE_SANITIZERS "Enable sanitizers" OFF)
option(LUISA_COMPUTE_ENABLE_UNITY_BUILD "Enable unity build (which might accelerate the compilation but could break VS Code's clangd indexing)" OFF)
option(LUISA_COMPUTE_CHECK_BACKEND_DEPENDENCIES "Check dependencies of backends (and disable the unsatisfied ones instead of reporting a configuration error)" ON)

if (UNIX AND NOT APPLE)
    option(LUISA_COMPUTE_ENABLE_WAYLAND "Enable Wayland support in GUI and Vulkan swapchains" OFF)
endif ()

# validate the options
include(scripts/validate_options.cmake)

# enable objc and objcxx for Apple platforms
if (APPLE)
    if (LUISA_COMPUTE_ENABLE_METAL OR LUISA_COMPUTE_ENABLE_GUI)
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
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH ON)

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
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_compile_features(luisa-compute-include INTERFACE c_std_11 cxx_std_20)

if (LUISA_COMPUTE_ENABLE_RUST)
    target_compile_definitions(luisa-compute-include INTERFACE LUISA_ENABLE_IR=1)
endif ()

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

# Generate luisa-config.h
# Collect all configuration flags for export
set(LUISA_ENABLE_IR 0)
set(LUISA_ENABLE_DSL 0)
set(LUISA_ENABLE_GUI 0)
set(LUISA_ENABLE_API 0)
set(LUISA_ENABLE_XIR 0)
set(LUISA_ENABLE_CLANGCXX 0)
set(LUISA_ENABLE_TENSOR 0)
set(LUISA_ENABLE_WAYLAND 0)
set(LUISA_PLATFORM_WINDOWS 0)
set(LUISA_PLATFORM_UNIX 0)
set(LUISA_PLATFORM_APPLE 0)
set(LUISA_USE_SYSTEM_STL 0)
set(LUISA_USE_SYSTEM_SPDLOG 0)
set(LUISA_USE_SYSTEM_XXHASH 0)
set(LUISA_USE_SYSTEM_MAGIC_ENUM 0)
set(LUISA_USE_SYSTEM_GLFW 0)
set(LUISA_USE_SYSTEM_REPROC 0)
set(LUISA_USE_SYSTEM_YYJSON 0)
set(LUISA_USE_SYSTEM_MARL 0)
set(LUISA_USE_SYSTEM_LMDB 0)
set(LUISA_USE_SYSTEM_VULKAN 0)

# Core feature flags
if (LUISA_COMPUTE_ENABLE_RUST)
    set(LUISA_ENABLE_IR 1)
endif ()

if (LUISA_COMPUTE_ENABLE_DSL)
    set(LUISA_ENABLE_DSL 1)
endif ()

if (LUISA_COMPUTE_ENABLE_GUI)
    set(LUISA_ENABLE_GUI 1)
endif ()

# Note: LUISA_ENABLE_API, LUISA_ENABLE_XIR, LUISA_ENABLE_CLANGCXX, LUISA_ENABLE_TENSOR
# are set as PUBLIC definitions on their respective component targets, not on luisa-compute-include.
# They are always available when those components are built, so we set them based on whether
# the components are enabled.
set(LUISA_ENABLE_API 1)  # Always built
set(LUISA_ENABLE_XIR 1)  # Always built

if (LUISA_COMPUTE_ENABLE_CLANG_CXX)
    set(LUISA_ENABLE_CLANGCXX 1)
endif ()

if (LUISA_COMPUTE_ENABLE_TENSOR)
    set(LUISA_ENABLE_TENSOR 1)
endif ()

# Platform-specific features
if (UNIX AND NOT APPLE AND LUISA_COMPUTE_ENABLE_WAYLAND)
    set(LUISA_ENABLE_WAYLAND 1)
endif ()

# Platform detection
if (WIN32)
    set(LUISA_PLATFORM_WINDOWS 1)
elseif (UNIX)
    set(LUISA_PLATFORM_UNIX 1)
    if (APPLE)
        set(LUISA_PLATFORM_APPLE 1)
    endif ()
endif ()

# System library usage flags
if (LUISA_COMPUTE_USE_SYSTEM_STL)
    set(LUISA_USE_SYSTEM_STL 1)
endif ()

if (LUISA_COMPUTE_USE_SYSTEM_SPDLOG)
    set(LUISA_USE_SYSTEM_SPDLOG 1)
endif ()

if (LUISA_COMPUTE_USE_SYSTEM_XXHASH)
    set(LUISA_USE_SYSTEM_XXHASH 1)
endif ()

if (LUISA_COMPUTE_USE_SYSTEM_MAGIC_ENUM)
    set(LUISA_USE_SYSTEM_MAGIC_ENUM 1)
endif ()

if (LUISA_COMPUTE_USE_SYSTEM_GLFW)
    set(LUISA_USE_SYSTEM_GLFW 1)
endif ()

if (LUISA_COMPUTE_USE_SYSTEM_REPROC)
    set(LUISA_USE_SYSTEM_REPROC 1)
endif ()

if (LUISA_COMPUTE_USE_SYSTEM_YYJSON)
    set(LUISA_USE_SYSTEM_YYJSON 1)
endif ()

if (LUISA_COMPUTE_USE_SYSTEM_MARL)
    set(LUISA_USE_SYSTEM_MARL 1)
endif ()

if (LUISA_COMPUTE_USE_SYSTEM_LMDB)
    set(LUISA_USE_SYSTEM_LMDB 1)
endif ()

if (LUISA_COMPUTE_USE_SYSTEM_VOLK)
    set(LUISA_USE_SYSTEM_VULKAN 1)
endif ()

configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/include/luisa/luisa-config.h.in
    ${CMAKE_CURRENT_BINARY_DIR}/include/luisa/luisa-config.h
    @ONLY)

# Generate pkg-config file
set(LUISA_COMPUTE_PKGCONFIG_CFLAGS "")
set(LUISA_COMPUTE_PKGCONFIG_LIBS "")

# Helper function to add compile flag to pkg-config
function(add_pkgconfig_cflag flag_name flag_value)
    if (${flag_value})
        list(APPEND LUISA_COMPUTE_PKGCONFIG_CFLAGS "-D${flag_name}=${flag_value}")
        set(LUISA_COMPUTE_PKGCONFIG_CFLAGS "${LUISA_COMPUTE_PKGCONFIG_CFLAGS}" PARENT_SCOPE)
    endif ()
endfunction()

# Core feature flags
add_pkgconfig_cflag(LUISA_ENABLE_IR ${LUISA_ENABLE_IR})
add_pkgconfig_cflag(LUISA_ENABLE_DSL ${LUISA_ENABLE_DSL})
add_pkgconfig_cflag(LUISA_ENABLE_GUI ${LUISA_ENABLE_GUI})
add_pkgconfig_cflag(LUISA_ENABLE_API ${LUISA_ENABLE_API})
add_pkgconfig_cflag(LUISA_ENABLE_XIR ${LUISA_ENABLE_XIR})
add_pkgconfig_cflag(LUISA_ENABLE_CLANGCXX ${LUISA_ENABLE_CLANGCXX})
add_pkgconfig_cflag(LUISA_ENABLE_TENSOR ${LUISA_ENABLE_TENSOR})
add_pkgconfig_cflag(LUISA_ENABLE_WAYLAND ${LUISA_ENABLE_WAYLAND})

# Platform flags
add_pkgconfig_cflag(LUISA_PLATFORM_WINDOWS ${LUISA_PLATFORM_WINDOWS})
add_pkgconfig_cflag(LUISA_PLATFORM_UNIX ${LUISA_PLATFORM_UNIX})
add_pkgconfig_cflag(LUISA_PLATFORM_APPLE ${LUISA_PLATFORM_APPLE})

# System library flags
add_pkgconfig_cflag(LUISA_USE_SYSTEM_STL ${LUISA_USE_SYSTEM_STL})
add_pkgconfig_cflag(LUISA_USE_SYSTEM_SPDLOG ${LUISA_USE_SYSTEM_SPDLOG})
add_pkgconfig_cflag(LUISA_USE_SYSTEM_XXHASH ${LUISA_USE_SYSTEM_XXHASH})
add_pkgconfig_cflag(LUISA_USE_SYSTEM_MAGIC_ENUM ${LUISA_USE_SYSTEM_MAGIC_ENUM})
add_pkgconfig_cflag(LUISA_USE_SYSTEM_GLFW ${LUISA_USE_SYSTEM_GLFW})
add_pkgconfig_cflag(LUISA_USE_SYSTEM_REPROC ${LUISA_USE_SYSTEM_REPROC})
add_pkgconfig_cflag(LUISA_USE_SYSTEM_YYJSON ${LUISA_USE_SYSTEM_YYJSON})
add_pkgconfig_cflag(LUISA_USE_SYSTEM_MARL ${LUISA_USE_SYSTEM_MARL})
add_pkgconfig_cflag(LUISA_USE_SYSTEM_LMDB ${LUISA_USE_SYSTEM_LMDB})
add_pkgconfig_cflag(LUISA_USE_SYSTEM_VULKAN ${LUISA_USE_SYSTEM_VULKAN})

# Add Windows-specific flag
if (LUISA_PLATFORM_WINDOWS)
    list(APPEND LUISA_COMPUTE_PKGCONFIG_CFLAGS "-D_DISABLE_CONSTEXPR_MUTEX_CONSTRUCTOR")
endif ()

# Convert lists to space-separated strings
string(REPLACE ";" " " LUISA_COMPUTE_PKGCONFIG_CFLAGS "${LUISA_COMPUTE_PKGCONFIG_CFLAGS}")
string(REPLACE ";" " " LUISA_COMPUTE_PKGCONFIG_LIBS "${LUISA_COMPUTE_PKGCONFIG_LIBS}")

configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/luisacompute.pc.in
    ${CMAKE_CURRENT_BINARY_DIR}/luisacompute.pc
    @ONLY)

install(TARGETS luisa-compute-include
        EXPORT LuisaComputeTargets)

install(DIRECTORY include/luisa
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
        FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp")

# Install generated luisa-config.h
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/include/luisa/luisa-config.h
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/luisa)

# Install pkg-config file
if (NOT SKBUILD)
    install(FILES ${CMAKE_CURRENT_BINARY_DIR}/luisacompute.pc
            DESTINATION ${CMAKE_INSTALL_LIBDIR}/pkgconfig)
endif ()

# Global definition to avoid MSVC ABI break.
# See https://github.com/microsoft/STL/releases/tag/vs-2022-17.10 for details.
if (WIN32)
    add_compile_definitions(_DISABLE_CONSTEXPR_MUTEX_CONSTRUCTOR)
    target_compile_definitions(luisa-compute-include INTERFACE _DISABLE_CONSTEXPR_MUTEX_CONSTRUCTOR)
endif ()

add_subdirectory(utils)
add_subdirectory(src)

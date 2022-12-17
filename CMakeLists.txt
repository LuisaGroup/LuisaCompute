cmake_minimum_required(VERSION 3.18...3.23)
cmake_policy(VERSION 3.18)

set(CMAKE_POLICY_DEFAULT_CMP0042 NEW)
set(CMAKE_POLICY_DEFAULT_CMP0048 NEW)
set(CMAKE_POLICY_DEFAULT_CMP0069 NEW)
set(CMAKE_POLICY_DEFAULT_CMP0072 NEW)
set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)
set(CMAKE_POLICY_DEFAULT_CMP0091 NEW)
set(CMAKE_POLICY_DEFAULT_CMP0135 NEW)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

project(LuisaCompute LANGUAGES C CXX VERSION 0.3.1)

option(LUISA_COMPUTE_ENABLE_DX "Enable DirectX backend" ON)
option(LUISA_COMPUTE_ENABLE_METAL "Enable Metal backend" ON)
option(LUISA_COMPUTE_ENABLE_CUDA "Enable CUDA backend" ON)
option(LUISA_COMPUTE_ENABLE_ISPC "Enable ISPC backend" OFF)
option(LUISA_COMPUTE_ENABLE_VULKAN "Enable Vulkan backend" OFF)
option(LUISA_COMPUTE_ENABLE_LLVM "Enable LLVM backend" ON)

option(LUISA_COMPUTE_ENABLE_PYTHON "Enable Python binding" ON)
option(LUISA_COMPUTE_ENABLE_GUI "Enable GUI support" ON)
option(LUISA_COMPUTE_DOWNLOAD_PREBUILT_EMBREE "Download pre-built Embree if available" ON)

if (APPLE AND LUISA_COMPUTE_ENABLE_METAL)
    enable_language(OBJC)
    enable_language(OBJCXX)
    set(CMAKE_OBJC_STANDARD 11)
    set(CMAKE_OBJCXX_STANDARD 20)
    set(CMAKE_OBJC_STANDARD_REQUIRED ON)
    set(CMAKE_OBJCXX_STANDARD_REQUIRED ON)
endif ()

if (APPLE)
    set(CMAKE_INSTALL_RPATH "@loader_path;@loader_path/../bin;@loader_path/../lib")
elseif (UNIX)
    set(CMAKE_INSTALL_RPATH "$ORIGIN;$ORIGIN/../bin;$ORIGIN/../lib")
endif ()

if (NOT CMAKE_INSTALL_PREFIX)
    set(CMAKE_INSTALL_PREFIX "dist")
endif ()

if (NOT DEFINED LUISA_COMPUTE_MASTER_PROJECT)
    if (CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
        set(LUISA_COMPUTE_MASTER_PROJECT ON)
    else ()
        set(LUISA_COMPUTE_MASTER_PROJECT OFF)
    endif ()
endif ()

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)
set(CMAKE_CXX_EXTENSIONS OFF)
set(BUILD_SHARED_LIBS ON)

if (CMAKE_SYSTEM_PROCESSOR MATCHES "AMD64" OR
        CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64")
    # enable AVX2 for embree on x64
    if (CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /arch:AVX2")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /arch:AVX2")
    else ()
        option(LUISA_COMPUTE_MARCH "-march option for GCC-like compilers" "native")
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -march=${LUISA_COMPUTE_MARCH} -mf16c")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=${LUISA_COMPUTE_MARCH} -mf16c")
    endif ()
endif ()

if (NOT CMAKE_SIZEOF_VOID_P EQUAL 8)
    message(FATAL_ERROR "LuisaCompute only supports 64-bit platforms")
endif ()

if (NOT CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
    option(LUISA_COMPUTE_ENABLE_SCCACHE "Enable sccache to speed up compilation" ON)
    if (LUISA_COMPUTE_ENABLE_SCCACHE)
        find_program(SCCACHE_EXE sccache)
        if (SCCACHE_EXE)
            message(STATUS "Build with sccache: ${SCCACHE_EXE}")
            set(CMAKE_C_COMPILER_LAUNCHER ${SCCACHE_EXE})
            set(CMAKE_CXX_COMPILER_LAUNCHER ${SCCACHE_EXE})
            set(CMAKE_OBJC_COMPILER_LAUNCHER ${SCCACHE_EXE})
            set(CMAKE_OBJCXX_COMPILER_LAUNCHER ${SCCACHE_EXE})
        else ()
            message(STATUS "Could not find sccache")
        endif ()
    endif ()
endif ()

if (CMAKE_CXX_COMPILER_ID MATCHES "Clang" AND
        CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 15.0)
    add_compile_options(-Wno-deprecated-builtins)
endif ()

option(LUISA_COMPUTE_BUILD_TESTS "Build tests for LuisaCompute" ${LUISA_COMPUTE_MASTER_PROJECT})
option(LUISA_COMPUTE_ENABLE_UNITY_BUILD "Enable unity build" OFF)
option(LUISA_PROJECT_DYLIB_PATH "dylib path" "luisa/dylibs")

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")

set(CMAKE_FIND_PACKAGE_SORT_ORDER NATURAL)
set(CMAKE_FIND_PACKAGE_SORT_DIRECTION DEC)

include(GNUInstallDirs)

install(FILES README.md README_Python_en.md README_Python_zh.md LICENSE
        TYPE DATA)

add_subdirectory(src)

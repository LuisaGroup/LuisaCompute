cmake_minimum_required(VERSION 3.26)
cmake_policy(SET CMP0091 NEW)
set(CMAKE_MSVC_RUNTIME_LIBRARY MultiThreaded)
project(benchmark_cuda_tirx_subgroup LANGUAGES CXX)

if (NOT WIN32 OR NOT CMAKE_CXX_COMPILER_ID STREQUAL "MSVC" OR NOT CMAKE_SIZEOF_VOID_P EQUAL 8)
    message(FATAL_ERROR "This standalone diagnostic matches the existing Windows x64 MSVC Release build only.")
endif ()
if (CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_CONFIGURATION_TYPES Release CACHE STRING "Production-compatible configuration" FORCE)
elseif (NOT CMAKE_BUILD_TYPE STREQUAL "Release")
    message(FATAL_ERROR "Configure with -DCMAKE_BUILD_TYPE=Release.")
endif ()

get_filename_component(_repo_default "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)
set(LUISA_REPO "${_repo_default}" CACHE PATH "Existing repository, read only")
set(LUISA_EXISTING_BUILD "${LUISA_REPO}/build-msvc-llvm" CACHE PATH "Existing Release build, read only")
set(CUDAToolkit_ROOT "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.4" CACHE PATH "System CUDA toolkit")
find_package(CUDAToolkit 13.4 REQUIRED)
if (NOT CUDAToolkit_VERSION MATCHES "^13\\.4\\.")
    message(FATAL_ERROR "This frozen diagnostic expects system CUDA 13.4.")
endif ()

foreach (_name core ast runtime tile)
    set(_dll "${LUISA_EXISTING_BUILD}/bin/luisa-${_name}.dll")
    set(_lib "${LUISA_EXISTING_BUILD}/lib/luisa-${_name}.lib")
    if (NOT EXISTS "${_dll}" OR NOT EXISTS "${_lib}")
        message(FATAL_ERROR "Missing existing ${_name} import library/DLL; build Luisa separately first.")
    endif ()
    add_library(luisa-existing-${_name} SHARED IMPORTED)
    set_target_properties(luisa-existing-${_name} PROPERTIES
        IMPORTED_LOCATION "${_dll}" IMPORTED_IMPLIB "${_lib}")
endforeach ()
set(_anchor "${LUISA_EXISTING_BUILD}/bin/benchmark_tile_workloads.exe")
if (NOT EXISTS "${_anchor}")
    message(FATAL_ERROR "Missing production Context anchor: ${_anchor}")
endif ()

add_executable(benchmark_cuda_tirx_subgroup benchmark_cuda_tirx_subgroup.cpp)
set_target_properties(benchmark_cuda_tirx_subgroup PROPERTIES
    CXX_STANDARD 20 CXX_STANDARD_REQUIRED YES CXX_EXTENSIONS NO
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_BINARY_DIR}/bin")
target_compile_options(benchmark_cuda_tirx_subgroup PRIVATE
    /utf-8 /O2 /Ob2 /GR /EHsc /Zc:preprocessor /Zc:__cplusplus /wd4005 /wd4068 /wd5105)
target_compile_definitions(benchmark_cuda_tirx_subgroup PRIVATE
    LUISA_TILE_BENCH_CUDA_EVENTS=1
    NDEBUG LUISA_PLATFORM_WINDOWS=1 LUISA_USE_SYSTEM_STL=1
    FMT_EXCEPTIONS=0 FMT_HEADER_ONLY=1 FMT_USE_CONSTEVAL=0 FMT_USE_NOEXCEPT=1
    MARL_DLL=1 MARL_USE_SYSTEM_STL=1 REPROCXX_SHARED REPROC_SHARED
    SPDLOG_DISABLE_DEFAULT_LOGGER SPDLOG_FWRITE_UNLOCKED SPDLOG_NO_EXCEPTIONS SPDLOG_NO_THREAD_ID
    XXH_INLINE_ALL YYJSON_IMPORTS=1 NOMINMAX=1 WIN32_LEAN_AND_MEAN=1
    _CRT_SECURE_NO_WARNINGS=1 _DISABLE_CONSTEXPR_MUTEX_CONSTRUCTOR
    _ENABLE_EXTENDED_ALIGNED_STORAGE=1 _UNICODE=1
    "LUISA_DIAGNOSTIC_RUNTIME_ANCHOR=\"${_anchor}\"")
target_include_directories(benchmark_cuda_tirx_subgroup PRIVATE
    "${LUISA_REPO}/include" "${LUISA_REPO}/src/tests" "${LUISA_REPO}/src/tests/common"
    "${LUISA_REPO}/src/ext/BTree/include" "${LUISA_REPO}/src/ext/spdlog/include"
    "${LUISA_REPO}/src/ext/xxHash" "${LUISA_REPO}/src/ext/magic_enum/include"
    "${LUISA_REPO}/src/ext/reproc/reproc/include" "${LUISA_REPO}/src/ext/reproc/reproc++/include"
    "${LUISA_REPO}/src/ext/marl/include" "${LUISA_REPO}/src/ext/half/include"
    "${LUISA_REPO}/src/ext/stb" "${LUISA_REPO}/src/ext/yyjson/src")
target_link_libraries(benchmark_cuda_tirx_subgroup PRIVATE
    luisa-existing-tile luisa-existing-runtime luisa-existing-ast luisa-existing-core CUDA::cuda_driver)

# No add_subdirectory, generated production file, post-build copy or test launch.

# The borrowed TIRx options use the same pinned headers as the main build.
target_include_directories(benchmark_cuda_tirx_subgroup PRIVATE
    "${LUISA_REPO}/src/ext/tvm/include"
    "${LUISA_REPO}/src/ext/tvm/3rdparty/tvm-ffi/include"
    "${LUISA_REPO}/src/ext/tvm/3rdparty/tvm-ffi/3rdparty/dlpack/include")
target_link_libraries(benchmark_cuda_tirx_subgroup PRIVATE
    "${LUISA_EXISTING_BUILD}/lib/luisa-tile-bridge-tirx.lib"
    "${LUISA_REPO}/.deps/tvm-msvc-cuda/lib/tvm_ffi.lib")

cmake_minimum_required(VERSION 3.26)

project(LuisaComputePackageConsumer LANGUAGES CXX)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/stage")
foreach (_config IN ITEMS Debug Release RelWithDebInfo MinSizeRel)
    string(TOUPPER "${_config}" _config_upper)
    set("CMAKE_RUNTIME_OUTPUT_DIRECTORY_${_config_upper}"
            "${CMAKE_BINARY_DIR}/stage")
endforeach ()
unset(_config)
unset(_config_upper)

find_package(LuisaCompute CONFIG REQUIRED)
set(_package_version "${LuisaCompute_VERSION}")
set(_package_components)
if (LuisaCompute_WITH_DSL)
    list(APPEND _package_components dsl)
endif ()
if (LuisaCompute_WITH_GUI)
    list(APPEND _package_components gui)
endif ()
if (LuisaCompute_WITH_RUST OR rust IN_LIST LuisaCompute_COMPONENTS)
    message(FATAL_ERROR
            "CPack distributions must not contain Rust integration")
endif ()
if (LUISA_COMPUTE_PACKAGE_EXPECT_BACKEND)
    list(APPEND _package_components "${LUISA_COMPUTE_PACKAGE_EXPECT_BACKEND}")
endif ()
find_package(LuisaCompute "${_package_version}" EXACT CONFIG REQUIRED
        COMPONENTS ${_package_components})

if (NOT TARGET luisa::compute)
    message(FATAL_ERROR "LuisaCompute package did not define luisa::compute")
endif ()
if (NOT TARGET luisa::luisa-compute-include)
    message(FATAL_ERROR
            "LuisaCompute package did not export its public header target")
endif ()
get_target_property(_compute_features luisa::compute INTERFACE_COMPILE_FEATURES)
if (NOT cxx_std_20 IN_LIST _compute_features)
    message(FATAL_ERROR
            "luisa::compute does not export its C++20 requirement: "
            "${_compute_features}")
endif ()
if (NOT LuisaCompute_CXX_STANDARD EQUAL 20)
    message(FATAL_ERROR
            "LuisaCompute package reported an unexpected C++ standard: "
            "${LuisaCompute_CXX_STANDARD}")
endif ()
foreach (_public_option IN ITEMS
        LUISA_COMPUTE_ENABLE_DSL
        LUISA_COMPUTE_ENABLE_GUI
        LUISA_COMPUTE_ENABLE_RUST
        LUISA_COMPUTE_ENABLE_WAYLAND
        LUISA_COMPUTE_ENABLE_SAFE_MODE
        LUISA_COMPUTE_ENABLE_CLANG_CXX
        LUISA_COMPUTE_ENABLE_DX_CUDA_INTEROP)
    if (NOT "${LuisaCompute_BUILD_OPTIONS}" MATCHES
            "(^|;)${_public_option}=(ON|OFF)($|;)")
        message(FATAL_ERROR
                "LuisaCompute_BUILD_OPTIONS does not report ${_public_option}: "
                "${LuisaCompute_BUILD_OPTIONS}")
    endif ()
endforeach ()
unset(_public_option)
foreach (_dependency IN ITEMS
        STL GLFW LMDB REPROC SPDLOG XXHASH YYJSON MAGIC_ENUM MARL)
    if (_dependency STREQUAL "GLFW" AND NOT LuisaCompute_WITH_GUI)
        if (_dependency IN_LIST LuisaCompute_SYSTEM_DEPENDENCIES OR
                _dependency IN_LIST LuisaCompute_BUNDLED_DEPENDENCIES)
            message(FATAL_ERROR
                    "Disabled GUI dependency GLFW was reported as packaged")
        endif ()
        continue()
    endif ()
    if (LuisaCompute_USE_SYSTEM_${_dependency})
        if (NOT _dependency IN_LIST LuisaCompute_SYSTEM_DEPENDENCIES OR
                _dependency IN_LIST LuisaCompute_BUNDLED_DEPENDENCIES)
            message(FATAL_ERROR
                    "System dependency metadata is inconsistent for ${_dependency}")
        endif ()
    elseif (NOT _dependency IN_LIST LuisaCompute_BUNDLED_DEPENDENCIES OR
            _dependency IN_LIST LuisaCompute_SYSTEM_DEPENDENCIES)
        message(FATAL_ERROR
                "Bundled dependency metadata is inconsistent for ${_dependency}")
    endif ()
endforeach ()
unset(_dependency)
get_target_property(_include_features
        luisa::luisa-compute-include INTERFACE_COMPILE_FEATURES)
if (NOT cxx_std_20 IN_LIST _include_features)
    message(FATAL_ERROR
            "The exported public header target lost cxx_std_20: "
            "${_include_features}")
endif ()
get_target_property(_include_directories
        luisa::luisa-compute-include INTERFACE_INCLUDE_DIRECTORIES)
if (NOT LuisaCompute_INCLUDE_DIR IN_LIST _include_directories)
    message(FATAL_ERROR
            "The exported public header target does not contain the installed "
            "include directory: ${_include_directories}")
endif ()
get_target_property(_include_definitions
        luisa::luisa-compute-include INTERFACE_COMPILE_DEFINITIONS)
if (WIN32)
    if (NOT "LUISA_PLATFORM_WINDOWS=1" IN_LIST _include_definitions OR
            NOT "_DISABLE_CONSTEXPR_MUTEX_CONSTRUCTOR" IN_LIST
            _include_definitions)
        message(FATAL_ERROR
                "The Windows ABI definitions were not exported: "
                "${_include_definitions}")
    endif ()
elseif (UNIX)
    if (NOT "LUISA_PLATFORM_UNIX=1" IN_LIST _include_definitions)
        message(FATAL_ERROR
                "The Unix platform definition was not exported: "
                "${_include_definitions}")
    endif ()
endif ()
get_target_property(_include_options
        luisa::luisa-compute-include INTERFACE_COMPILE_OPTIONS)
if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND
        NOT "-Wno-pragmas" IN_LIST _include_options)
    message(FATAL_ERROR
            "The required GNU public compile option was not exported: "
            "${_include_options}")
endif ()
if (NOT TARGET luisa::validation_layer)
    message(FATAL_ERROR "LuisaCompute package did not define luisa::validation_layer")
endif ()
if (NOT LuisaCompute_VERSION)
    message(FATAL_ERROR "LuisaCompute package did not report its version")
endif ()

foreach (_backend IN LISTS LuisaCompute_BACKENDS)
    if (NOT TARGET "luisa::backend_${_backend}")
        message(FATAL_ERROR
                "LuisaCompute package reports backend '${_backend}' but did not "
                "define target luisa::backend_${_backend}")
    endif ()
    get_target_property(_backend_type "luisa::backend_${_backend}" TYPE)
    if (NOT _backend_type STREQUAL "MODULE_LIBRARY")
        message(FATAL_ERROR
                "Exported backend target luisa::backend_${_backend} has "
                "unexpected type ${_backend_type}")
    endif ()
endforeach ()

list(LENGTH LuisaCompute_BACKENDS _backend_count)
set(_expected_backend "${LUISA_COMPUTE_PACKAGE_EXPECT_BACKEND}")
if (_expected_backend AND NOT _expected_backend IN_LIST LuisaCompute_BACKENDS)
    message(FATAL_ERROR
            "Expected backend '${_expected_backend}' is not in the installed "
            "backend list: ${LuisaCompute_BACKENDS}")
endif ()

set(_expect_dxc 0)
if (WIN32 AND EXISTS "${LuisaCompute_RUNTIME_DIR}/dxcompiler.dll")
    set(_expect_dxc 1)
elseif (UNIX AND NOT APPLE AND
        EXISTS "${LuisaCompute_RUNTIME_DIR}/libdxcompiler.so")
    set(_expect_dxc 1)
endif ()

add_executable(luisa_compute_package_consumer main.cpp)
target_link_libraries(luisa_compute_package_consumer PRIVATE luisa::compute)
target_compile_definitions(luisa_compute_package_consumer PRIVATE
        LUISA_PACKAGE_VERSION_MAJOR=${LuisaCompute_VERSION_MAJOR}
        LUISA_PACKAGE_VERSION_MINOR=${LuisaCompute_VERSION_MINOR}
        LUISA_PACKAGE_VERSION_PATCH=${LuisaCompute_VERSION_PATCH}
        LUISA_PACKAGE_EXPECTED_BACKEND_COUNT=${_backend_count}
        LUISA_PACKAGE_EXPECT_DXC=${_expect_dxc}
        "LUISA_PACKAGE_WITH_DSL=$<BOOL:${LuisaCompute_WITH_DSL}>"
        "LUISA_PACKAGE_WITH_GUI=$<BOOL:${LuisaCompute_WITH_GUI}>"
        "LUISA_PACKAGE_WITH_WAYLAND=$<BOOL:${LuisaCompute_WITH_WAYLAND}>"
        "LUISA_PACKAGE_WITH_SAFE_MODE=$<BOOL:${LuisaCompute_WITH_SAFE_MODE}>"
        "LUISA_PACKAGE_WITH_CLANG_CXX=$<BOOL:${LuisaCompute_WITH_CLANG_CXX}>"
        "LUISA_PACKAGE_USE_SYSTEM_STL=$<BOOL:${LuisaCompute_USE_SYSTEM_STL}>"
        "LUISA_PACKAGE_USE_SYSTEM_GLFW=$<BOOL:${LuisaCompute_USE_SYSTEM_GLFW}>"
        "LUISA_PACKAGE_USE_SYSTEM_LMDB=$<BOOL:${LuisaCompute_USE_SYSTEM_LMDB}>"
        "LUISA_PACKAGE_USE_SYSTEM_REPROC=$<BOOL:${LuisaCompute_USE_SYSTEM_REPROC}>"
        "LUISA_PACKAGE_USE_SYSTEM_SPDLOG=$<BOOL:${LuisaCompute_USE_SYSTEM_SPDLOG}>"
        "LUISA_PACKAGE_USE_SYSTEM_XXHASH=$<BOOL:${LuisaCompute_USE_SYSTEM_XXHASH}>"
        "LUISA_PACKAGE_USE_SYSTEM_YYJSON=$<BOOL:${LuisaCompute_USE_SYSTEM_YYJSON}>"
        "LUISA_PACKAGE_USE_SYSTEM_MAGIC_ENUM=$<BOOL:${LuisaCompute_USE_SYSTEM_MAGIC_ENUM}>"
        "LUISA_PACKAGE_USE_SYSTEM_MARL=$<BOOL:${LuisaCompute_USE_SYSTEM_MARL}>"
        "LUISA_PACKAGE_EXPECTED_BACKEND=\"${_expected_backend}\"")
luisa_compute_deploy_runtime(TARGET luisa_compute_package_consumer)

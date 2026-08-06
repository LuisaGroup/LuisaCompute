cmake_minimum_required(VERSION 3.26)

project(LuisaComputePackageConsumer LANGUAGES CXX)

find_package(LuisaCompute CONFIG REQUIRED)
set(_package_version "${LuisaCompute_VERSION}")
set(_package_components dsl)
if (LuisaCompute_WITH_RUST)
    list(APPEND _package_components rust)
    set(_package_with_rust 1)
else ()
    set(_package_with_rust 0)
endif ()
if (LUISA_COMPUTE_PACKAGE_EXPECT_BACKEND)
    list(APPEND _package_components "${LUISA_COMPUTE_PACKAGE_EXPECT_BACKEND}")
endif ()
find_package(LuisaCompute "${_package_version}" EXACT CONFIG REQUIRED
        COMPONENTS ${_package_components})

if (NOT TARGET luisa::compute)
    message(FATAL_ERROR "LuisaCompute package did not define luisa::compute")
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
        LUISA_PACKAGE_WITH_RUST=${_package_with_rust}
        "LUISA_PACKAGE_EXPECTED_BACKEND=\"${_expected_backend}\"")
luisa_compute_deploy_runtime(TARGET luisa_compute_package_consumer)

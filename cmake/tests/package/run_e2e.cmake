cmake_minimum_required(VERSION 3.26)

foreach (_required IN ITEMS
        LUISA_COMPUTE_E2E_PACKAGE_FILE
        LUISA_COMPUTE_E2E_PACKAGE_ROOT_NAME
        LUISA_COMPUTE_E2E_SOURCE_DIR
        LUISA_COMPUTE_E2E_BINARY_DIR
        LUISA_COMPUTE_E2E_WORK_DIR
        LUISA_COMPUTE_E2E_GENERATOR)
    if (NOT DEFINED ${_required} OR "${${_required}}" STREQUAL "")
        message(FATAL_ERROR "${_required} is required")
    endif ()
endforeach ()

if (NOT EXISTS "${LUISA_COMPUTE_E2E_PACKAGE_FILE}")
    message(FATAL_ERROR
            "CPack artifact does not exist: ${LUISA_COMPUTE_E2E_PACKAGE_FILE}")
endif ()

set(_checksum_file "${LUISA_COMPUTE_E2E_PACKAGE_FILE}.sha256")
if (NOT EXISTS "${_checksum_file}")
    message(FATAL_ERROR
            "CPack checksum does not exist: ${_checksum_file}")
endif ()
file(READ "${_checksum_file}" _checksum_contents)
string(REGEX MATCH "^[0-9A-Fa-f]+" _expected_checksum
        "${_checksum_contents}")
file(SHA256 "${LUISA_COMPUTE_E2E_PACKAGE_FILE}" _actual_checksum)
string(TOLOWER "${_expected_checksum}" _expected_checksum)
string(TOLOWER "${_actual_checksum}" _actual_checksum)
if (NOT _expected_checksum STREQUAL _actual_checksum)
    message(FATAL_ERROR
            "CPack SHA-256 mismatch for ${LUISA_COMPUTE_E2E_PACKAGE_FILE}: "
            "expected ${_expected_checksum}, got ${_actual_checksum}")
endif ()

set(_extract_dir "${LUISA_COMPUTE_E2E_WORK_DIR}/extract")
set(_consumer_source "${LUISA_COMPUTE_E2E_WORK_DIR}/consumer-source")
set(_consumer_build "${LUISA_COMPUTE_E2E_WORK_DIR}/consumer-build")
file(REMOVE_RECURSE "${LUISA_COMPUTE_E2E_WORK_DIR}")
file(MAKE_DIRECTORY "${_extract_dir}" "${_consumer_source}")

execute_process(
        COMMAND "${CMAKE_COMMAND}" -E tar xf
        "${LUISA_COMPUTE_E2E_PACKAGE_FILE}"
        WORKING_DIRECTORY "${_extract_dir}"
        RESULT_VARIABLE _extract_result)
if (NOT _extract_result EQUAL 0)
    message(FATAL_ERROR
            "Failed to extract ${LUISA_COMPUTE_E2E_PACKAGE_FILE}")
endif ()

set(_package_root
        "${_extract_dir}/${LUISA_COMPUTE_E2E_PACKAGE_ROOT_NAME}")
if (NOT IS_DIRECTORY "${_package_root}")
    message(FATAL_ERROR
            "CPack archive did not contain the expected top-level directory: "
            "${LUISA_COMPUTE_E2E_PACKAGE_ROOT_NAME}")
endif ()

set(LUISA_COMPUTE_PACKAGE_ROOT "${_package_root}")
set(LUISA_COMPUTE_FORBIDDEN_PATHS
        "${LUISA_COMPUTE_E2E_SOURCE_DIR}"
        "${LUISA_COMPUTE_E2E_BINARY_DIR}")
include("${LUISA_COMPUTE_E2E_SOURCE_DIR}/cmake/tests/package/audit_install.cmake")

file(COPY
        "${LUISA_COMPUTE_E2E_SOURCE_DIR}/cmake/tests/package/CMakeLists.txt"
        "${LUISA_COMPUTE_E2E_SOURCE_DIR}/cmake/tests/package/main.cpp"
        DESTINATION "${_consumer_source}")

file(GLOB_RECURSE _package_configs LIST_DIRECTORIES FALSE
        "${_package_root}/*/LuisaComputeConfig.cmake")
list(LENGTH _package_configs _package_config_count)
if (NOT _package_config_count EQUAL 1)
    message(FATAL_ERROR
            "Expected exactly one LuisaComputeConfig.cmake in the package; "
            "found ${_package_config_count}: ${_package_configs}")
endif ()
list(GET _package_configs 0 _package_config)
get_filename_component(_package_config_dir "${_package_config}" DIRECTORY)

set(_configure_command
        "${CMAKE_COMMAND}"
        -S "${_consumer_source}"
        -B "${_consumer_build}"
        -G "${LUISA_COMPUTE_E2E_GENERATOR}"
        "-DCMAKE_FIND_USE_PACKAGE_REGISTRY=OFF"
        "-DCMAKE_FIND_USE_SYSTEM_PACKAGE_REGISTRY=OFF"
        "-DLuisaCompute_DIR=${_package_config_dir}"
        "-DLUISA_COMPUTE_PACKAGE_EXPECT_BACKEND=${LUISA_COMPUTE_E2E_EXPECT_BACKEND}")
if (LUISA_COMPUTE_E2E_CONFIG)
    list(APPEND _configure_command
            "-DCMAKE_BUILD_TYPE=${LUISA_COMPUTE_E2E_CONFIG}")
endif ()
if (LUISA_COMPUTE_E2E_CXX_COMPILER)
    list(APPEND _configure_command
            "-DCMAKE_CXX_COMPILER=${LUISA_COMPUTE_E2E_CXX_COMPILER}")
endif ()
if (LUISA_COMPUTE_E2E_PREFIX_PATH)
    list(APPEND _configure_command
            "-DCMAKE_PREFIX_PATH=${LUISA_COMPUTE_E2E_PREFIX_PATH}")
endif ()
if (LUISA_COMPUTE_E2E_OSX_SYSROOT)
    list(APPEND _configure_command
            "-DCMAKE_OSX_SYSROOT=${LUISA_COMPUTE_E2E_OSX_SYSROOT}")
endif ()
if (LUISA_COMPUTE_E2E_OSX_ARCHITECTURES)
    list(APPEND _configure_command
            "-DCMAKE_OSX_ARCHITECTURES=${LUISA_COMPUTE_E2E_OSX_ARCHITECTURES}")
endif ()

execute_process(
        COMMAND ${_configure_command}
        RESULT_VARIABLE _configure_result)
if (NOT _configure_result EQUAL 0)
    message(FATAL_ERROR "The isolated package consumer failed to configure")
endif ()

set(_build_command "${CMAKE_COMMAND}" --build "${_consumer_build}")
if (LUISA_COMPUTE_E2E_CONFIG)
    list(APPEND _build_command --config "${LUISA_COMPUTE_E2E_CONFIG}")
endif ()
execute_process(
        COMMAND ${_build_command}
        RESULT_VARIABLE _build_result)
if (NOT _build_result EQUAL 0)
    message(FATAL_ERROR "The isolated package consumer failed to build")
endif ()

# The deploy helper must have copied every LuisaCompute runtime file needed by
# the consumer. Remove the extracted SDK before running to prove that the
# executable does not accidentally resolve anything from the package prefix.
file(REMOVE_RECURSE "${_extract_dir}")
if (WIN32)
    set(_consumer_executable
            "${_consumer_build}/stage/luisa_compute_package_consumer.exe")
else ()
    set(_consumer_executable
            "${_consumer_build}/stage/luisa_compute_package_consumer")
endif ()
if (NOT EXISTS "${_consumer_executable}")
    message(FATAL_ERROR
            "The package consumer executable was not staged: "
            "${_consumer_executable}")
endif ()
execute_process(
        COMMAND "${_consumer_executable}"
        RESULT_VARIABLE _run_result)
if (NOT _run_result EQUAL 0)
    message(FATAL_ERROR
            "The staged package consumer failed after the SDK was removed")
endif ()

message(STATUS
        "LuisaCompute CPack E2E test passed: ${LUISA_COMPUTE_E2E_PACKAGE_FILE}")

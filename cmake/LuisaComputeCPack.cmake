set(CPACK_PACKAGE_NAME "LuisaCompute")
set(CPACK_PACKAGE_VENDOR "LuisaGroup")
set(CPACK_PACKAGE_CONTACT "LuisaGroup")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY
        "Cross-platform high-performance rendering compute framework")
set(CPACK_PACKAGE_HOMEPAGE_URL
        "https://github.com/LuisaGroup/LuisaCompute")
set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE")
set(CPACK_RESOURCE_FILE_README "${CMAKE_CURRENT_SOURCE_DIR}/README.md")
set(CPACK_PACKAGE_VERSION "${PROJECT_VERSION}")
set(CPACK_PACKAGE_VERSION_MAJOR "${PROJECT_VERSION_MAJOR}")
set(CPACK_PACKAGE_VERSION_MINOR "${PROJECT_VERSION_MINOR}")
set(CPACK_PACKAGE_VERSION_PATCH "${PROJECT_VERSION_PATCH}")
set(CPACK_PACKAGE_DIRECTORY "${CMAKE_BINARY_DIR}/package")
set(CPACK_PACKAGING_INSTALL_PREFIX "/")
set(CPACK_INCLUDE_TOPLEVEL_DIRECTORY ON)
set(CPACK_MONOLITHIC_INSTALL ON)
set(CPACK_PACKAGE_CHECKSUM SHA256)
set(CPACK_THREADS 0)
set(CPACK_VERBATIM_VARIABLES ON)

if (WIN32)
    set(_luisa_compute_package_platform windows)
    set(_luisa_compute_default_cpack_generator ZIP)
    set(_luisa_compute_package_extension .zip)
elseif (APPLE)
    set(_luisa_compute_package_platform macos)
    set(_luisa_compute_default_cpack_generator TGZ)
    set(_luisa_compute_package_extension .tar.gz)
else ()
    set(_luisa_compute_package_platform linux)
    set(_luisa_compute_default_cpack_generator TGZ)
    set(_luisa_compute_package_extension .tar.gz)
endif ()

string(TOLOWER "${CMAKE_SYSTEM_PROCESSOR}" _luisa_compute_package_arch)
if (NOT _luisa_compute_package_arch)
    set(_luisa_compute_package_arch unknown)
endif ()
set(CPACK_PACKAGE_FILE_NAME
        "LuisaCompute-${PROJECT_VERSION}-${_luisa_compute_package_platform}-${_luisa_compute_package_arch}")
if (NOT CPACK_GENERATOR)
    set(CPACK_GENERATOR "${_luisa_compute_default_cpack_generator}")
endif ()
set(_luisa_compute_selected_cpack_generator "${CPACK_GENERATOR}")
set(_luisa_compute_binary_package_file_name "${CPACK_PACKAGE_FILE_NAME}")

include(CPack)

# The built-in package target is always available. The E2E target invokes CPack
# directly because CMake's special global `package` target cannot portably be
# used with add_dependencies (notably with Unix Makefiles). Native package
# generators can still be selected through CPACK_GENERATOR and tested by
# downstream packaging.
if ("${_luisa_compute_selected_cpack_generator}" STREQUAL
        "${_luisa_compute_default_cpack_generator}")
    if (WIN32)
        set(_luisa_compute_native_backend dx)
    elseif (APPLE)
        set(_luisa_compute_native_backend metal)
    else ()
        set(_luisa_compute_native_backend vk)
    endif ()
    get_property(_luisa_compute_built_backends
            GLOBAL PROPERTY LUISA_COMPUTE_BUILT_BACKENDS)
    if (_luisa_compute_native_backend IN_LIST _luisa_compute_built_backends)
        set(_luisa_compute_expected_backend
                "${_luisa_compute_native_backend}")
    elseif (_luisa_compute_built_backends)
        list(GET _luisa_compute_built_backends 0
                _luisa_compute_expected_backend)
    else ()
        set(_luisa_compute_expected_backend "")
    endif ()

    set(_luisa_compute_package_artifact
            "${CPACK_PACKAGE_DIRECTORY}/${_luisa_compute_binary_package_file_name}${_luisa_compute_package_extension}")
    add_custom_target(luisa-compute-package-e2e
            COMMAND "${CMAKE_CPACK_COMMAND}"
            --config "${CMAKE_BINARY_DIR}/CPackConfig.cmake"
            -C "$<CONFIG>"
            COMMAND "${CMAKE_COMMAND}"
            "-DLUISA_COMPUTE_E2E_PACKAGE_FILE=${_luisa_compute_package_artifact}"
            "-DLUISA_COMPUTE_E2E_PACKAGE_ROOT_NAME=${_luisa_compute_binary_package_file_name}"
            "-DLUISA_COMPUTE_E2E_SOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}"
            "-DLUISA_COMPUTE_E2E_BINARY_DIR=${CMAKE_BINARY_DIR}"
            "-DLUISA_COMPUTE_E2E_WORK_DIR=${CMAKE_BINARY_DIR}/package-e2e"
            "-DLUISA_COMPUTE_E2E_GENERATOR=${CMAKE_GENERATOR}"
            "-DLUISA_COMPUTE_E2E_CONFIG=$<CONFIG>"
            "-DLUISA_COMPUTE_E2E_CXX_COMPILER=${CMAKE_CXX_COMPILER}"
            "-DLUISA_COMPUTE_E2E_PREFIX_PATH=${CMAKE_PREFIX_PATH}"
            "-DLUISA_COMPUTE_E2E_OSX_SYSROOT=${CMAKE_OSX_SYSROOT}"
            "-DLUISA_COMPUTE_E2E_OSX_ARCHITECTURES=${CMAKE_OSX_ARCHITECTURES}"
            "-DLUISA_COMPUTE_E2E_EXPECT_BACKEND=${_luisa_compute_expected_backend}"
            -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/tests/package/run_e2e.cmake"
            COMMENT "Testing the extracted LuisaCompute distribution package"
            VERBATIM)
endif ()

unset(_luisa_compute_default_cpack_generator)
unset(_luisa_compute_binary_package_file_name)
unset(_luisa_compute_built_backends)
unset(_luisa_compute_expected_backend)
unset(_luisa_compute_native_backend)
unset(_luisa_compute_package_arch)
unset(_luisa_compute_package_artifact)
unset(_luisa_compute_package_extension)
unset(_luisa_compute_package_platform)
unset(_luisa_compute_selected_cpack_generator)

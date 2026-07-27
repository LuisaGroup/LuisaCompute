# LLVMConfig.cmake exports its own CMAKE_MSVC_RUNTIME_LIBRARY value into the
# including directory. Preserve the project's setting so the package cannot
# silently change the ABI of targets created after find_package().
unset(_luisa_compute_project_msvc_runtime_library)
if (DEFINED LUISA_COMPUTE_PROJECT_MSVC_RUNTIME_LIBRARY_WAS_DEFINED)
    set(_luisa_compute_had_msvc_runtime_library
            ${LUISA_COMPUTE_PROJECT_MSVC_RUNTIME_LIBRARY_WAS_DEFINED})
    if (_luisa_compute_had_msvc_runtime_library)
        set(_luisa_compute_project_msvc_runtime_library
                "${LUISA_COMPUTE_PROJECT_MSVC_RUNTIME_LIBRARY}")
    endif ()
else ()
    # Defensive fallback for embedding this directory outside the normal
    # top-level configure path.
    set(_luisa_compute_had_msvc_runtime_library FALSE)
    if (DEFINED CMAKE_MSVC_RUNTIME_LIBRARY)
        set(_luisa_compute_had_msvc_runtime_library TRUE)
        set(_luisa_compute_project_msvc_runtime_library
                "${CMAKE_MSVC_RUNTIME_LIBRARY}")
    endif ()
endif ()
find_package(LLVM CONFIG REQUIRED)
# Capture LLVMConfig's actual value before restoring the project snapshot.
set(_luisa_compute_llvm_msvc_runtime_library
        "${CMAKE_MSVC_RUNTIME_LIBRARY}")
if (_luisa_compute_had_msvc_runtime_library)
    set(CMAKE_MSVC_RUNTIME_LIBRARY
            "${_luisa_compute_project_msvc_runtime_library}")
else ()
    unset(CMAKE_MSVC_RUNTIME_LIBRARY)
endif ()

if (MSVC)
    # CMake's default under CMP0091 is the DLL CRT with a Debug suffix selected
    # per configuration. Compare the CRT family separately from build mode so
    # the standard default and LLVM's single-config spelling compare honestly.
    set(_luisa_compute_effective_project_msvc_runtime_library
            "${_luisa_compute_project_msvc_runtime_library}")
    if (NOT _luisa_compute_effective_project_msvc_runtime_library)
        set(_luisa_compute_effective_project_msvc_runtime_library
                "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")
    endif ()
    if (NOT _luisa_compute_llvm_msvc_runtime_library)
        message(FATAL_ERROR
                "LLVM ${LLVM_PACKAGE_VERSION} does not report its MSVC runtime "
                "library. The AST-to-LLVM SPIR-V facade exchanges C++ objects, "
                "so the ABI cannot be verified safely.")
    endif ()
    string(FIND "${_luisa_compute_effective_project_msvc_runtime_library}"
            "DLL" _luisa_compute_project_uses_dll_crt)
    string(FIND "${_luisa_compute_llvm_msvc_runtime_library}"
            "DLL" _luisa_compute_llvm_uses_dll_crt)
    set(_luisa_compute_project_dll_crt FALSE)
    set(_luisa_compute_llvm_dll_crt FALSE)
    if (NOT _luisa_compute_project_uses_dll_crt EQUAL -1)
        set(_luisa_compute_project_dll_crt TRUE)
    endif ()
    if (NOT _luisa_compute_llvm_uses_dll_crt EQUAL -1)
        set(_luisa_compute_llvm_dll_crt TRUE)
    endif ()
    if ((_luisa_compute_project_dll_crt AND
         NOT _luisa_compute_llvm_dll_crt) OR
        (_luisa_compute_llvm_dll_crt AND
         NOT _luisa_compute_project_dll_crt))
        message(FATAL_ERROR
                "LLVM ${LLVM_PACKAGE_VERSION} and LuisaCompute use different "
                "MSVC runtime-library families: LLVM='"
                "${_luisa_compute_llvm_msvc_runtime_library}', project='"
                "${_luisa_compute_effective_project_msvc_runtime_library}'.")
    endif ()

    set(_luisa_compute_llvm_configurations)
    if (TARGET LLVM)
        get_target_property(_luisa_compute_llvm_configurations LLVM
                IMPORTED_CONFIGURATIONS)
    elseif (TARGET LLVMSupport)
        get_target_property(_luisa_compute_llvm_configurations LLVMSupport
                IMPORTED_CONFIGURATIONS)
    endif ()
    if (NOT _luisa_compute_llvm_configurations OR
        _luisa_compute_llvm_configurations MATCHES "-NOTFOUND$")
        set(_luisa_compute_llvm_configurations ${LLVM_CONFIGURATION_TYPES})
    endif ()
    if (NOT _luisa_compute_llvm_configurations AND LLVM_BUILD_TYPE)
        set(_luisa_compute_llvm_configurations ${LLVM_BUILD_TYPE})
    endif ()
    if (NOT _luisa_compute_llvm_configurations)
        message(FATAL_ERROR
                "LLVM ${LLVM_PACKAGE_VERSION} does not report usable imported "
                "configurations; the MSVC iterator-debug ABI cannot be checked.")
    endif ()

    set(_luisa_compute_llvm_has_debug FALSE)
    set(_luisa_compute_llvm_has_nondebug FALSE)
    foreach (_configuration IN LISTS _luisa_compute_llvm_configurations)
        string(TOUPPER "${_configuration}" _configuration_upper)
        if (_configuration_upper STREQUAL "DEBUG")
            set(_luisa_compute_llvm_has_debug TRUE)
        else ()
            set(_luisa_compute_llvm_has_nondebug TRUE)
        endif ()
    endforeach ()
    if (CMAKE_CONFIGURATION_TYPES)
        set(_luisa_compute_project_configurations
                ${CMAKE_CONFIGURATION_TYPES})
    elseif (CMAKE_BUILD_TYPE)
        set(_luisa_compute_project_configurations ${CMAKE_BUILD_TYPE})
    else ()
        set(_luisa_compute_project_configurations Release)
    endif ()
    foreach (_configuration IN LISTS _luisa_compute_project_configurations)
        string(TOUPPER "${_configuration}" _configuration_upper)
        if (_configuration_upper STREQUAL "DEBUG" AND
            NOT _luisa_compute_llvm_has_debug)
            message(FATAL_ERROR
                    "LuisaCompute enables the Debug configuration, but LLVM "
                    "${LLVM_PACKAGE_VERSION} has no Debug import. Restrict "
                    "CMAKE_CONFIGURATION_TYPES to non-Debug configurations or "
                    "provide a matching Debug LLVM build.")
        elseif (NOT _configuration_upper STREQUAL "DEBUG" AND
                NOT _luisa_compute_llvm_has_nondebug)
            message(FATAL_ERROR
                    "LuisaCompute enables the ${_configuration} configuration, "
                    "but LLVM ${LLVM_PACKAGE_VERSION} only provides Debug. "
                    "Provide a matching non-Debug LLVM build.")
        endif ()
    endforeach ()
endif ()

find_path(LUISA_COMPUTE_LLVM_SPIRV_INTRINSICS_INCLUDE_DIR
        NAMES llvm/IR/IntrinsicsSPIRV.h
        PATHS ${LLVM_INCLUDE_DIRS}
        NO_DEFAULT_PATH
        NO_CACHE)
if (NOT LUISA_COMPUTE_LLVM_SPIRV_INTRINSICS_INCLUDE_DIR)
    message(FATAL_ERROR
            "The Vulkan AST-to-LLVM SPIR-V path requires an LLVM installation "
            "with the native SPIR-V target headers (missing "
            "llvm/IR/IntrinsicsSPIRV.h in ${LLVM_INCLUDE_DIRS}).")
endif ()

list(FIND LLVM_TARGETS_TO_BUILD "SPIRV" _luisa_llvm_spirv_target_index)
if (_luisa_llvm_spirv_target_index EQUAL -1)
    message(FATAL_ERROR
            "The selected LLVM ${LLVM_PACKAGE_VERSION} installation was not "
            "built with the native SPIR-V target. Reconfigure LLVM with "
            "LLVM_TARGETS_TO_BUILD including SPIRV, or disable "
            "LUISA_COMPUTE_ENABLE_VK_AST_LLVM_SPIRV.")
endif ()

file(GLOB LUISA_COMPUTE_SPIRV_LLVM_SOURCES CONFIGURE_DEPENDS *.cpp *.h)
add_library(luisa-compute-spirv-llvm STATIC
        ${LUISA_COMPUTE_SPIRV_LLVM_SOURCES})
set_target_properties(luisa-compute-spirv-llvm PROPERTIES
        POSITION_INDEPENDENT_CODE ON
        OUTPUT_NAME luisa-spirv-llvm)
target_include_directories(luisa-compute-spirv-llvm
        PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
        PRIVATE
        ${LLVM_INCLUDE_DIRS})
separate_arguments(LUISA_COMPUTE_SPIRV_LLVM_DEFINITIONS
        NATIVE_COMMAND "${LLVM_DEFINITIONS}")
set(LUISA_COMPUTE_SPIRV_LLVM_PRIVATE_DEFINITIONS)
foreach (_definition IN LISTS LUISA_COMPUTE_SPIRV_LLVM_DEFINITIONS)
    string(REGEX REPLACE "^-D" "" _definition "${_definition}")
    if (_definition MATCHES "^_GLIBCXX_USE_CXX11_ABI=[01]$")
        string(REGEX REPLACE ".*=" "" _llvm_glibcxx_abi
                "${_definition}")
        include(CheckCXXSourceCompiles)
        set(_abi_check
                "LUISA_COMPUTE_PROJECT_GLIBCXX_ABI_${_llvm_glibcxx_abi}")
        # Do not retain a stale success when the compiler or global ABI flags
        # are changed in an existing build directory.
        unset(${_abi_check} CACHE)
        unset(${_abi_check})
        check_cxx_source_compiles(
                "#include <string>\n#if !defined(_GLIBCXX_USE_CXX11_ABI) || _GLIBCXX_USE_CXX11_ABI != ${_llvm_glibcxx_abi}\n#error incompatible libstdc++ ABI\n#endif\nint main() { return 0; }"
                ${_abi_check})
        if (NOT ${_abi_check})
            message(FATAL_ERROR
                    "LLVM ${LLVM_PACKAGE_VERSION} uses "
                    "_GLIBCXX_USE_CXX11_ABI=${_llvm_glibcxx_abi}, but the "
                    "LuisaCompute toolchain uses a different libstdc++ ABI. "
                    "The LLVM SPIR-V facade exchanges C++ containers, so "
                    "mixing the ABIs is unsafe.")
        endif ()
        # The facade returns C++ containers across the static-library boundary,
        # so its consumer must use the same libstdc++ ABI as LLVM. Do not leak
        # LLVM's unrelated feature-test macros into the Vulkan backend.
        target_compile_definitions(luisa-compute-spirv-llvm PUBLIC
                "${_definition}")
    else ()
        list(APPEND LUISA_COMPUTE_SPIRV_LLVM_PRIVATE_DEFINITIONS
                "${_definition}")
    endif ()
endforeach ()
target_compile_definitions(luisa-compute-spirv-llvm PRIVATE
        ${LUISA_COMPUTE_SPIRV_LLVM_PRIVATE_DEFINITIONS})
target_link_libraries(luisa-compute-spirv-llvm
        PUBLIC
        luisa-compute-runtime
        PRIVATE
        SPIRV-Tools-opt)

set(LUISA_COMPUTE_SPIRV_LLVM_COMPONENTS
        core
        support
        bitwriter
        transformutils
        analysis
        codegen
        target
        mc
        spirvcodegen
        spirvdesc
        spirvinfo
        spirvanalysis)

# llvm_config knows which components are already present in the monolithic
# LLVM dylib and retains any target-specific archives that are not. Linking the
# imported LLVM target alone is insufficient for packages whose dylib omits the
# experimental SPIR-V target.
if (TARGET LLVM)
    llvm_config(luisa-compute-spirv-llvm USE_SHARED
            ${LUISA_COMPUTE_SPIRV_LLVM_COMPONENTS})
else ()
    if (WIN32)
        llvm_map_components_to_libnames(
                _luisa_compute_spirv_llvm_component_targets
                ${LUISA_COMPUTE_SPIRV_LLVM_COMPONENTS})
        foreach (_llvm_component_target IN LISTS
                _luisa_compute_spirv_llvm_component_targets)
            if (TARGET ${_llvm_component_target})
                get_target_property(_llvm_component_target_type
                        ${_llvm_component_target} TYPE)
                if (_llvm_component_target_type STREQUAL "SHARED_LIBRARY")
                    message(FATAL_ERROR
                            "The Windows LLVM package provides shared component "
                            "library ${_llvm_component_target} without the "
                            "monolithic LLVM target. Use an LLVM package with "
                            "static component libraries so runtime DLL staging "
                            "is explicit and complete.")
                endif ()
            endif ()
        endforeach ()
    endif ()
    target_link_directories(luisa-compute-spirv-llvm PRIVATE
            ${LLVM_LIBRARY_DIRS})
    llvm_config(luisa-compute-spirv-llvm
            ${LUISA_COMPUTE_SPIRV_LLVM_COMPONENTS})
endif ()

# A monolithic LLVM DLL is an imported runtime dependency, not an artifact of
# the static facade. Stage and install it explicitly so a Windows Vulkan build
# does not depend on the user's PATH after linking successfully.
if (WIN32 AND TARGET LLVM)
    get_target_property(_luisa_compute_llvm_target_type LLVM TYPE)
    if (_luisa_compute_llvm_target_type STREQUAL "SHARED_LIBRARY")
        add_custom_target(luisa-compute-spirv-llvm-copy-runtime ALL
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "$<TARGET_FILE:LLVM>"
                "$<TARGET_FILE_DIR:luisa-compute-core>"
                COMMENT "Staging the LLVM runtime for the Vulkan backend"
                VERBATIM)
        add_dependencies(luisa-compute-spirv-llvm-copy-runtime
                luisa-compute-core)
        add_dependencies(luisa-compute-spirv-llvm
                luisa-compute-spirv-llvm-copy-runtime)
        install(IMPORTED_RUNTIME_ARTIFACTS LLVM
                RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
    endif ()
endif ()

# Distribution builds prefer dependencies supplied by the host package manager,
# but remain buildable on platforms where a dependency has no suitable system
# package. The ordinary LUISA_COMPUTE_USE_SYSTEM_* semantics are unchanged
# outside LUISA_COMPUTE_ENABLE_PACKAGE_DISTRIBUTION: explicitly requesting a
# system dependency there remains strict.

function(_luisa_compute_resolve_system_dependency option_name display_name found)
    if (NOT ${option_name})
        return()
    endif ()
    if (found)
        message(STATUS "Package distribution: using system ${display_name}")
        return()
    endif ()
    if (LUISA_COMPUTE_PACKAGE_REQUIRE_SYSTEM_LIBS)
        message(FATAL_ERROR
                "Package distribution requires system ${display_name}, but it "
                "was not found or failed its compatibility probe "
                "(${option_name}=ON).")
    endif ()
    # Keep the user's cache preference intact so installing a system package
    # and reconfiguring can select it without clearing the build tree. The
    # normal directory-scope value controls this configure pass only.
    set("${option_name}" OFF PARENT_SCOPE)
    message(STATUS
            "Package distribution: system ${display_name} is unavailable or "
            "incompatible; using bundled copy")
endfunction()

# Probe template-heavy or API-version-sensitive packages in a separate CMake
# project. Imported targets are directory-scoped, so a rejected system package
# never collides with the bundled target of the same name in LuisaCompute.
function(_luisa_compute_try_system_dependency result display_name project_name
        source_dir target_name)
    set(_luisa_compute_probe_flags)
    foreach (_luisa_compute_probe_variable IN ITEMS
            CMAKE_BUILD_TYPE
            CMAKE_PREFIX_PATH
            CMAKE_FIND_ROOT_PATH
            CMAKE_SYSROOT
            CMAKE_OSX_SYSROOT
            CMAKE_OSX_ARCHITECTURES
            ${ARGN})
        if (DEFINED ${_luisa_compute_probe_variable} AND
                NOT "${${_luisa_compute_probe_variable}}" STREQUAL "")
            string(REPLACE ";" "\\;" _luisa_compute_probe_value
                    "${${_luisa_compute_probe_variable}}")
            list(APPEND _luisa_compute_probe_flags
                    "-D${_luisa_compute_probe_variable}:STRING=${_luisa_compute_probe_value}")
        endif ()
    endforeach ()
    try_compile(_luisa_compute_probe_result
            PROJECT "${project_name}"
            SOURCE_DIR "${source_dir}"
            TARGET "${target_name}"
            NO_CACHE
            CMAKE_FLAGS ${_luisa_compute_probe_flags}
            OUTPUT_VARIABLE _luisa_compute_probe_output)
    if (NOT _luisa_compute_probe_result)
        message(VERBOSE
                "System ${display_name} compatibility probe failed:\n"
                "${_luisa_compute_probe_output}")
    endif ()
    set("${result}" "${_luisa_compute_probe_result}" PARENT_SCOPE)
endfunction()

if (LUISA_COMPUTE_USE_SYSTEM_STL)
    message(STATUS "Package distribution: using the system C++ standard library")
endif ()

if (LUISA_COMPUTE_USE_SYSTEM_SPDLOG)
    # A package being discoverable does not make it usable. Ubuntu 24.04's
    # spdlog 1.12/external fmt 9 pair, for example, fails in FMT_STRING's
    # consteval parser with Clang 20. Probe in an isolated CMake directory so a
    # rejected imported target cannot collide with the bundled spdlog aliases.
    _luisa_compute_try_system_dependency(
            _luisa_compute_dependency_found
            spdlog
            LuisaComputeSystemSpdlogProbe
            "${CMAKE_CURRENT_LIST_DIR}/probes/spdlog"
            luisa-compute-system-spdlog-probe
            spdlog_DIR fmt_DIR)
    _luisa_compute_resolve_system_dependency(
            LUISA_COMPUTE_USE_SYSTEM_SPDLOG spdlog
            ${_luisa_compute_dependency_found})
endif ()

if (LUISA_COMPUTE_USE_SYSTEM_XXHASH)
    find_package(xxHash CONFIG QUIET)
    set(_luisa_compute_dependency_found FALSE)
    if (TARGET xxHash::xxhash)
        set(_luisa_compute_dependency_found TRUE)
    else ()
        find_package(PkgConfig QUIET)
        if (PkgConfig_FOUND)
            pkg_check_modules(LUISA_COMPUTE_PACKAGE_XXHASH QUIET libxxhash)
            if (LUISA_COMPUTE_PACKAGE_XXHASH_FOUND)
                set(_luisa_compute_dependency_found TRUE)
            endif ()
        endif ()
    endif ()
    _luisa_compute_resolve_system_dependency(
            LUISA_COMPUTE_USE_SYSTEM_XXHASH xxHash
            ${_luisa_compute_dependency_found})
endif ()

if (LUISA_COMPUTE_USE_SYSTEM_MAGIC_ENUM)
    find_package(magic_enum CONFIG QUIET)
    if (TARGET magic_enum::magic_enum)
        set(_luisa_compute_dependency_found TRUE)
    else ()
        set(_luisa_compute_dependency_found FALSE)
    endif ()
    _luisa_compute_resolve_system_dependency(
            LUISA_COMPUTE_USE_SYSTEM_MAGIC_ENUM magic_enum
            ${_luisa_compute_dependency_found})
endif ()

if (LUISA_COMPUTE_ENABLE_GUI AND LUISA_COMPUTE_USE_SYSTEM_GLFW)
    # Luisa's GUI needs GLFW 3.4's runtime-platform API. In particular,
    # Ubuntu 24.04 exposes a valid glfw target from 3.3.10 that is too old to
    # compile glfwGetPlatform()/GLFW_PLATFORM_WAYLAND. Test the actual API, not
    # only package discovery or an unreliable downstream version variable.
    _luisa_compute_try_system_dependency(
            _luisa_compute_dependency_found
            GLFW
            LuisaComputeSystemGLFWProbe
            "${CMAKE_CURRENT_LIST_DIR}/probes/glfw"
            luisa-compute-system-glfw-probe
            glfw3_DIR)
    _luisa_compute_resolve_system_dependency(
            LUISA_COMPUTE_USE_SYSTEM_GLFW GLFW
            ${_luisa_compute_dependency_found})
elseif (NOT LUISA_COMPUTE_ENABLE_GUI)
    set(LUISA_COMPUTE_USE_SYSTEM_GLFW OFF)
endif ()

if (LUISA_COMPUTE_USE_SYSTEM_REPROC)
    find_package(reproc CONFIG QUIET)
    find_package(reproc++ CONFIG QUIET)
    if ((TARGET reproc AND TARGET reproc++) OR
            (TARGET reproc::reproc AND TARGET reproc::reproc++))
        set(_luisa_compute_dependency_found TRUE)
    else ()
        set(_luisa_compute_dependency_found FALSE)
    endif ()
    if ((TARGET reproc OR TARGET reproc++ OR
            TARGET reproc::reproc OR TARGET reproc::reproc++) AND
            NOT _luisa_compute_dependency_found)
        message(FATAL_ERROR
                "Package distribution found an incomplete system reproc "
                "installation. Both reproc and reproc++ are required; "
                "install the missing package or explicitly set "
                "LUISA_COMPUTE_USE_SYSTEM_REPROC=OFF.")
    endif ()
    _luisa_compute_resolve_system_dependency(
            LUISA_COMPUTE_USE_SYSTEM_REPROC reproc
            ${_luisa_compute_dependency_found})
endif ()

if (LUISA_COMPUTE_USE_SYSTEM_YYJSON)
    find_package(yyjson CONFIG QUIET)
    if (TARGET yyjson::yyjson)
        set(_luisa_compute_dependency_found TRUE)
    else ()
        set(_luisa_compute_dependency_found FALSE)
    endif ()
    _luisa_compute_resolve_system_dependency(
            LUISA_COMPUTE_USE_SYSTEM_YYJSON yyjson
            ${_luisa_compute_dependency_found})
endif ()

if (LUISA_COMPUTE_USE_SYSTEM_MARL)
    find_package(marl CONFIG QUIET)
    if (TARGET marl::marl)
        set(_luisa_compute_dependency_found TRUE)
    else ()
        set(_luisa_compute_dependency_found FALSE)
    endif ()
    _luisa_compute_resolve_system_dependency(
            LUISA_COMPUTE_USE_SYSTEM_MARL marl
            ${_luisa_compute_dependency_found})
endif ()

if (LUISA_COMPUTE_USE_SYSTEM_LMDB)
    find_package(unofficial-lmdb CONFIG QUIET)
    set(_luisa_compute_dependency_found FALSE)
    if (TARGET unofficial::lmdb::lmdb)
        set(_luisa_compute_dependency_found TRUE)
    else ()
        find_package(PkgConfig QUIET)
        if (PkgConfig_FOUND)
            pkg_check_modules(LUISA_COMPUTE_PACKAGE_LMDB QUIET lmdb)
            if (LUISA_COMPUTE_PACKAGE_LMDB_FOUND)
                set(_luisa_compute_dependency_found TRUE)
            endif ()
        endif ()
    endif ()
    _luisa_compute_resolve_system_dependency(
            LUISA_COMPUTE_USE_SYSTEM_LMDB LMDB
            ${_luisa_compute_dependency_found})
endif ()

unset(_luisa_compute_dependency_found)

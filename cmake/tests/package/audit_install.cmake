cmake_minimum_required(VERSION 3.26)

if (NOT LUISA_COMPUTE_PACKAGE_ROOT)
    message(FATAL_ERROR "LUISA_COMPUTE_PACKAGE_ROOT is required")
endif ()
cmake_path(ABSOLUTE_PATH LUISA_COMPUTE_PACKAGE_ROOT NORMALIZE)
if (NOT IS_DIRECTORY "${LUISA_COMPUTE_PACKAGE_ROOT}")
    message(FATAL_ERROR
            "LuisaCompute package root does not exist: "
            "${LUISA_COMPUTE_PACKAGE_ROOT}")
endif ()

function(_luisa_compute_reject_forbidden_runtime_path binary kind value)
    if (NOT IS_ABSOLUTE "${value}")
        return()
    endif ()
    cmake_path(CONVERT "${value}" TO_CMAKE_PATH_LIST
            _luisa_compute_value NORMALIZE)
    if (WIN32)
        string(TOLOWER "${_luisa_compute_value}" _luisa_compute_value)
    endif ()
    foreach (_luisa_compute_forbidden IN LISTS LUISA_COMPUTE_FORBIDDEN_PATHS)
        if (NOT _luisa_compute_forbidden)
            continue()
        endif ()
        cmake_path(CONVERT "${_luisa_compute_forbidden}" TO_CMAKE_PATH_LIST
                _luisa_compute_forbidden NORMALIZE)
        if (WIN32)
            string(TOLOWER "${_luisa_compute_forbidden}"
                    _luisa_compute_forbidden)
        endif ()
        string(FIND "${_luisa_compute_value}"
                "${_luisa_compute_forbidden}" _luisa_compute_match)
        if (_luisa_compute_match EQUAL 0)
            message(FATAL_ERROR
                    "Installed binary ${binary} contains ${kind} path ${value}, "
                    "which refers to producer path ${_luisa_compute_forbidden}")
        endif ()
    endforeach ()
endfunction()

# Exported package files must not refer back to the producer checkout, build
# tree, or the pre-relocation install prefix. Scan both slash conventions so
# the same check works for Windows-generated files.
file(GLOB_RECURSE _LuisaCompute_CMAKE_FILES LIST_DIRECTORIES FALSE
        "${LUISA_COMPUTE_PACKAGE_ROOT}/*.cmake")
foreach (_LuisaCompute_CMAKE_FILE IN LISTS _LuisaCompute_CMAKE_FILES)
    file(READ "${_LuisaCompute_CMAKE_FILE}" _LuisaCompute_CMAKE_CONTENT)
    if (WIN32)
        string(TOLOWER "${_LuisaCompute_CMAKE_CONTENT}"
                _LuisaCompute_CMAKE_CONTENT)
    endif ()
    foreach (_LuisaCompute_FORBIDDEN IN LISTS LUISA_COMPUTE_FORBIDDEN_PATHS)
        if (NOT _LuisaCompute_FORBIDDEN)
            continue()
        endif ()
        cmake_path(CONVERT "${_LuisaCompute_FORBIDDEN}" TO_CMAKE_PATH_LIST
                _LuisaCompute_FORBIDDEN_CMAKE NORMALIZE)
        string(REPLACE "/" "\\" _LuisaCompute_FORBIDDEN_NATIVE
                "${_LuisaCompute_FORBIDDEN_CMAKE}")
        if (WIN32)
            string(TOLOWER "${_LuisaCompute_FORBIDDEN_CMAKE}"
                    _LuisaCompute_FORBIDDEN_CMAKE)
            string(TOLOWER "${_LuisaCompute_FORBIDDEN_NATIVE}"
                    _LuisaCompute_FORBIDDEN_NATIVE)
        endif ()
        string(FIND "${_LuisaCompute_CMAKE_CONTENT}"
                "${_LuisaCompute_FORBIDDEN_CMAKE}" _LuisaCompute_FOUND_CMAKE)
        string(FIND "${_LuisaCompute_CMAKE_CONTENT}"
                "${_LuisaCompute_FORBIDDEN_NATIVE}" _LuisaCompute_FOUND_NATIVE)
        if (NOT _LuisaCompute_FOUND_CMAKE EQUAL -1 OR
            NOT _LuisaCompute_FOUND_NATIVE EQUAL -1)
            message(FATAL_ERROR
                    "Installed package file ${_LuisaCompute_CMAKE_FILE} "
                    "contains forbidden producer path "
                    "${_LuisaCompute_FORBIDDEN}")
        endif ()
    endforeach ()
endforeach ()

file(GLOB_RECURSE _LuisaCompute_RUNTIME_FILES LIST_DIRECTORIES FALSE
        "${LUISA_COMPUTE_PACKAGE_ROOT}/bin/*"
        "${LUISA_COMPUTE_PACKAGE_ROOT}/lib/*")

# ---------------------------------------------------------------------------
# Runtime closure.
# A shared library that an installed binary links against but that never made it
# into the package only fails at load time on the customer's machine
# ("libglslang-default-resource-limits.so.16: cannot open shared object file"),
# which is far away from the missing install() rule that caused it. Every
# library produced by this build must therefore appear in the package too.
# Libraries that come from outside the build (libc, libvulkan, the CUDA driver,
# system Qt/GLFW, ...) are not produced by it and stay exempt.
# ---------------------------------------------------------------------------
set(_LuisaCompute_PACKAGED_FILE_NAMES "")
foreach (_LuisaCompute_RUNTIME_FILE IN LISTS _LuisaCompute_RUNTIME_FILES)
    get_filename_component(_LuisaCompute_PACKAGED_FILE_NAME
            "${_LuisaCompute_RUNTIME_FILE}" NAME)
    list(APPEND _LuisaCompute_PACKAGED_FILE_NAMES
            "${_LuisaCompute_PACKAGED_FILE_NAME}")
endforeach ()
list(REMOVE_DUPLICATES _LuisaCompute_PACKAGED_FILE_NAMES)

set(_LuisaCompute_BUILT_SHARED_FILE_NAMES "")
if (DEFINED LUISA_COMPUTE_E2E_BINARY_DIR AND
        IS_DIRECTORY "${LUISA_COMPUTE_E2E_BINARY_DIR}")
    foreach (_LuisaCompute_OUTPUT_DIR IN ITEMS bin lib)
        set(_LuisaCompute_OUTPUT_DIR
                "${LUISA_COMPUTE_E2E_BINARY_DIR}/${_LuisaCompute_OUTPUT_DIR}")
        if (NOT IS_DIRECTORY "${_LuisaCompute_OUTPUT_DIR}")
            continue ()
        endif ()
        file(GLOB_RECURSE _LuisaCompute_BUILT_FILES
                LIST_DIRECTORIES FALSE
                "${_LuisaCompute_OUTPUT_DIR}/*")
        foreach (_LuisaCompute_BUILT_FILE IN LISTS _LuisaCompute_BUILT_FILES)
            get_filename_component(_LuisaCompute_BUILT_FILE_NAME
                    "${_LuisaCompute_BUILT_FILE}" NAME)
            list(APPEND _LuisaCompute_BUILT_SHARED_FILE_NAMES
                    "${_LuisaCompute_BUILT_FILE_NAME}")
        endforeach ()
    endforeach ()
    list(REMOVE_DUPLICATES _LuisaCompute_BUILT_SHARED_FILE_NAMES)
endif ()

# Reject a dependency that this build produced but the package does not ship.
function(_luisa_compute_require_packaged_dependency binary kind load_name)
    if (_LuisaCompute_BUILT_SHARED_FILE_NAMES STREQUAL "")
        return() # no build tree to compare against; nothing is known
    endif ()
    if (IS_ABSOLUTE "${load_name}")
        # Already rejected for producer paths; nothing else to say about it.
        return()
    endif ()
    get_filename_component(_load_name "${load_name}" NAME)
    if (NOT _load_name IN_LIST _LuisaCompute_BUILT_SHARED_FILE_NAMES)
        return() # external (system or vendor) library
    endif ()
    if (_load_name IN_LIST _LuisaCompute_PACKAGED_FILE_NAMES)
        return()
    endif ()
    message(FATAL_ERROR
            "Installed binary ${binary} requires ${kind} ${_load_name}, which "
            "this build produces but the package does not install. Add the "
            "missing install() rule (see cmake/tests/package/audit_install.cmake).")
endfunction()

if (CMAKE_HOST_SYSTEM_NAME STREQUAL "Linux")
    find_program(_LuisaCompute_READELF NAMES readelf llvm-readelf REQUIRED)
    foreach (_LuisaCompute_RUNTIME_FILE IN LISTS _LuisaCompute_RUNTIME_FILES)
        execute_process(
                COMMAND "${_LuisaCompute_READELF}" -d
                "${_LuisaCompute_RUNTIME_FILE}"
                RESULT_VARIABLE _LuisaCompute_READELF_RESULT
                OUTPUT_VARIABLE _LuisaCompute_DYNAMIC_SECTION
                ERROR_QUIET)
        if (NOT _LuisaCompute_READELF_RESULT EQUAL 0)
            continue()
        endif ()
        string(REGEX MATCHALL
                "\\(NEEDED\\)[^\n]*"
                _LuisaCompute_NEEDED_LINES
                "${_LuisaCompute_DYNAMIC_SECTION}")
        foreach (_LuisaCompute_NEEDED_LINE IN LISTS _LuisaCompute_NEEDED_LINES)
            if (NOT _LuisaCompute_NEEDED_LINE MATCHES "\\[([^]]*)\\]")
                continue ()
            endif ()
            if (IS_ABSOLUTE "${CMAKE_MATCH_1}")
                _luisa_compute_reject_forbidden_runtime_path(
                        "${_LuisaCompute_RUNTIME_FILE}" "DT_NEEDED"
                        "${CMAKE_MATCH_1}")
            endif ()
            _luisa_compute_require_packaged_dependency(
                    "${_LuisaCompute_RUNTIME_FILE}" "DT_NEEDED"
                    "${CMAKE_MATCH_1}")
        endforeach ()
        string(REGEX MATCHALL
                "\\((RPATH|RUNPATH)\\)[^\n]*"
                _LuisaCompute_RPATH_LINES
                "${_LuisaCompute_DYNAMIC_SECTION}")
        foreach (_LuisaCompute_RPATH_LINE IN LISTS _LuisaCompute_RPATH_LINES)
            if (NOT _LuisaCompute_RPATH_LINE MATCHES "\\[([^]]*)\\]")
                continue()
            endif ()
            string(REPLACE ":" ";" _LuisaCompute_RPATH_ENTRIES
                    "${CMAKE_MATCH_1}")
            foreach (_LuisaCompute_RPATH IN LISTS _LuisaCompute_RPATH_ENTRIES)
                _luisa_compute_reject_forbidden_runtime_path(
                        "${_LuisaCompute_RUNTIME_FILE}" "RPATH"
                        "${_LuisaCompute_RPATH}")
            endforeach ()
        endforeach ()
    endforeach ()
elseif (APPLE)
    find_program(_LuisaCompute_OTOOL NAMES otool REQUIRED)
    foreach (_LuisaCompute_RUNTIME_FILE IN LISTS _LuisaCompute_RUNTIME_FILES)
        execute_process(
                COMMAND "${_LuisaCompute_OTOOL}" -l
                "${_LuisaCompute_RUNTIME_FILE}"
                RESULT_VARIABLE _LuisaCompute_OTOOL_RESULT
                OUTPUT_VARIABLE _LuisaCompute_LOAD_COMMANDS
                ERROR_QUIET)
        if (NOT _LuisaCompute_OTOOL_RESULT EQUAL 0)
            continue()
        endif ()
        string(REPLACE "\n" ";" _LuisaCompute_LOAD_COMMANDS
                "${_LuisaCompute_LOAD_COMMANDS}")
        set(_LuisaCompute_EXPECT_RPATH FALSE)
        foreach (_LuisaCompute_LOAD_LINE IN LISTS _LuisaCompute_LOAD_COMMANDS)
            string(STRIP "${_LuisaCompute_LOAD_LINE}" _LuisaCompute_LOAD_LINE)
            if (_LuisaCompute_LOAD_LINE STREQUAL "cmd LC_RPATH")
                set(_LuisaCompute_EXPECT_RPATH TRUE)
            elseif (_LuisaCompute_EXPECT_RPATH AND
                    _LuisaCompute_LOAD_LINE MATCHES "^path ([^ ]+)")
                set(_LuisaCompute_EXPECT_RPATH FALSE)
                _luisa_compute_reject_forbidden_runtime_path(
                        "${_LuisaCompute_RUNTIME_FILE}" "LC_RPATH"
                        "${CMAKE_MATCH_1}")
            endif ()
        endforeach ()

        execute_process(
                COMMAND "${_LuisaCompute_OTOOL}" -L
                "${_LuisaCompute_RUNTIME_FILE}"
                RESULT_VARIABLE _LuisaCompute_OTOOL_LINK_RESULT
                OUTPUT_VARIABLE _LuisaCompute_LINKED_LIBRARIES
                ERROR_QUIET)
        if (_LuisaCompute_OTOOL_LINK_RESULT EQUAL 0)
            string(REPLACE "\n" ";" _LuisaCompute_LINKED_LIBRARIES
                    "${_LuisaCompute_LINKED_LIBRARIES}")
            foreach (_LuisaCompute_LINK_LINE IN LISTS _LuisaCompute_LINKED_LIBRARIES)
                string(STRIP "${_LuisaCompute_LINK_LINE}"
                        _LuisaCompute_LINK_LINE)
                # otool may exit successfully for a non-Mach-O input while
                # printing a diagnostic that begins with the input filename.
                # Accept only the actual load-name grammar emitted by `-L`;
                # this also supports valid dependency paths containing spaces.
                if (NOT _LuisaCompute_LINK_LINE MATCHES
                        "^(/.+) \\(compatibility version .+\\)$")
                    continue()
                endif ()
                set(_LuisaCompute_LOAD_NAME "${CMAKE_MATCH_1}")
                _luisa_compute_reject_forbidden_runtime_path(
                        "${_LuisaCompute_RUNTIME_FILE}" "Mach-O load-name"
                        "${_LuisaCompute_LOAD_NAME}")
                if (_LuisaCompute_LOAD_NAME MATCHES "^@rpath/")
                    _luisa_compute_require_packaged_dependency(
                            "${_LuisaCompute_RUNTIME_FILE}" "Mach-O load-name"
                            "${_LuisaCompute_LOAD_NAME}")
                endif ()
            endforeach ()
        endif ()
    endforeach ()
endif ()

message(STATUS
        "LuisaCompute install audit passed: ${LUISA_COMPUTE_PACKAGE_ROOT}")

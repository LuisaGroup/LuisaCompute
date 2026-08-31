add_library(luisa-compute-ext INTERFACE)

if (APPLE AND LUISA_COMPUTE_ENABLE_METAL4)
    find_package(LLVM CONFIG REQUIRED)
    if (NOT LLVM_VERSION_MAJOR EQUAL 21)
        message(FATAL_ERROR
                "The in-tree llvm-downgrade sources currently require LLVM 21, "
                "but LLVM ${LLVM_PACKAGE_VERSION} was found. Point LLVM_DIR at LLVM 21.")
    endif ()
    # LLVM may have been configured on a generic Unix host while targeting
    # iOS, in which case its exported LLVMSupport target carries a standalone
    # librt dependency. iOS provides those APIs from libSystem and has no
    # librt archive, so remove the invalid dependency before this static
    # target captures LLVM's transitive link interface.
    if (CMAKE_SYSTEM_NAME STREQUAL "iOS" AND TARGET LLVMSupport)
        get_target_property(_luisa_llvm_support_link_libraries
                LLVMSupport INTERFACE_LINK_LIBRARIES)
        if (_luisa_llvm_support_link_libraries)
            list(FILTER _luisa_llvm_support_link_libraries
                    EXCLUDE REGEX "^rt$")
            set_property(TARGET LLVMSupport PROPERTY
                    INTERFACE_LINK_LIBRARIES
                    "${_luisa_llvm_support_link_libraries}")
        endif ()
    endif ()

    set(LUISA_LLVM_DOWNGRADE_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/llvm-downgrade")
    set(LUISA_LLVM_DOWNGRADE_PATCH
            "${CMAKE_CURRENT_SOURCE_DIR}/llvm-downgrade-llvm21-air.patch")
    set(LUISA_LLVM_DOWNGRADE_EXPECTED_COMMIT
            "1e04ee99aff7c059606502ee86d03eb7d1c5d781")
    set(LUISA_LLVM_DOWNGRADE_EXPECTED_SOURCE_FINGERPRINT
            "7679006d0b2b529998de888e03a812ff311632f58f5d8e4a3c07fca3519c02b6")
    if (NOT EXISTS "${LUISA_LLVM_DOWNGRADE_SOURCE_DIR}/src/BitcodeWriter140.cpp")
        message(FATAL_ERROR
                "The llvm-downgrade submodule is missing. Run "
                "'git submodule update --init src/ext/llvm-downgrade'.")
    endif ()

    file(GLOB_RECURSE LUISA_LLVM_DOWNGRADE_MIRROR_INPUTS CONFIGURE_DEPENDS
            LIST_DIRECTORIES false
            "${LUISA_LLVM_DOWNGRADE_SOURCE_DIR}/CMakeLists.txt"
            "${LUISA_LLVM_DOWNGRADE_SOURCE_DIR}/LICENSE.TXT"
            "${LUISA_LLVM_DOWNGRADE_SOURCE_DIR}/README.md"
            "${LUISA_LLVM_DOWNGRADE_SOURCE_DIR}/common/*"
            "${LUISA_LLVM_DOWNGRADE_SOURCE_DIR}/include/*"
            "${LUISA_LLVM_DOWNGRADE_SOURCE_DIR}/src/*"
            "${LUISA_LLVM_DOWNGRADE_SOURCE_DIR}/test/*"
            "${LUISA_LLVM_DOWNGRADE_SOURCE_DIR}/tools/*")
    list(SORT LUISA_LLVM_DOWNGRADE_MIRROR_INPUTS)
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
            "${LUISA_LLVM_DOWNGRADE_PATCH}"
            ${LUISA_LLVM_DOWNGRADE_MIRROR_INPUTS})

    find_package(Git REQUIRED)
    if (EXISTS "${LUISA_LLVM_DOWNGRADE_SOURCE_DIR}/.git")
        execute_process(
                COMMAND "${CMAKE_COMMAND}" -E env
                --unset=GIT_DIR --unset=GIT_WORK_TREE
                "${GIT_EXECUTABLE}" -C "${LUISA_LLVM_DOWNGRADE_SOURCE_DIR}"
                rev-parse --verify HEAD
                RESULT_VARIABLE LUISA_LLVM_DOWNGRADE_HEAD_RESULT
                OUTPUT_VARIABLE LUISA_LLVM_DOWNGRADE_HEAD
                ERROR_VARIABLE LUISA_LLVM_DOWNGRADE_HEAD_ERROR
                OUTPUT_STRIP_TRAILING_WHITESPACE)
        if (NOT LUISA_LLVM_DOWNGRADE_HEAD_RESULT EQUAL 0)
            message(FATAL_ERROR
                    "Failed to inspect the llvm-downgrade submodule revision.\n"
                    "${LUISA_LLVM_DOWNGRADE_HEAD_ERROR}")
        endif ()
        if (NOT LUISA_LLVM_DOWNGRADE_HEAD STREQUAL
                LUISA_LLVM_DOWNGRADE_EXPECTED_COMMIT)
            message(FATAL_ERROR
                    "The llvm-downgrade submodule is at "
                    "${LUISA_LLVM_DOWNGRADE_HEAD}, but LuisaCompute requires "
                    "${LUISA_LLVM_DOWNGRADE_EXPECTED_COMMIT}.")
        endif ()
        execute_process(
                COMMAND "${CMAKE_COMMAND}" -E env
                --unset=GIT_DIR --unset=GIT_WORK_TREE
                "${GIT_EXECUTABLE}" -C "${LUISA_LLVM_DOWNGRADE_SOURCE_DIR}"
                status --porcelain=v1 --untracked-files=all
                RESULT_VARIABLE LUISA_LLVM_DOWNGRADE_STATUS_RESULT
                OUTPUT_VARIABLE LUISA_LLVM_DOWNGRADE_STATUS
                ERROR_VARIABLE LUISA_LLVM_DOWNGRADE_STATUS_ERROR
                OUTPUT_STRIP_TRAILING_WHITESPACE)
        if (NOT LUISA_LLVM_DOWNGRADE_STATUS_RESULT EQUAL 0)
            message(FATAL_ERROR
                    "Failed to inspect the llvm-downgrade submodule status.\n"
                    "${LUISA_LLVM_DOWNGRADE_STATUS_ERROR}")
        endif ()
        if (NOT LUISA_LLVM_DOWNGRADE_STATUS STREQUAL "")
            message(FATAL_ERROR
                    "The llvm-downgrade submodule must be pristine. Found:\n"
                    "${LUISA_LLVM_DOWNGRADE_STATUS}")
        endif ()
    endif ()

    set(LUISA_LLVM_DOWNGRADE_FINGERPRINT_INPUT
            "base:${LUISA_LLVM_DOWNGRADE_EXPECTED_COMMIT}\n")
    foreach (LUISA_LLVM_DOWNGRADE_INPUT IN LISTS
            LUISA_LLVM_DOWNGRADE_MIRROR_INPUTS)
        file(RELATIVE_PATH LUISA_LLVM_DOWNGRADE_INPUT_RELATIVE
                "${LUISA_LLVM_DOWNGRADE_SOURCE_DIR}"
                "${LUISA_LLVM_DOWNGRADE_INPUT}")
        file(SHA256 "${LUISA_LLVM_DOWNGRADE_INPUT}"
                LUISA_LLVM_DOWNGRADE_INPUT_SHA256)
        string(APPEND LUISA_LLVM_DOWNGRADE_FINGERPRINT_INPUT
                "${LUISA_LLVM_DOWNGRADE_INPUT_RELATIVE}:"
                "${LUISA_LLVM_DOWNGRADE_INPUT_SHA256}\n")
    endforeach ()
    string(SHA256 LUISA_LLVM_DOWNGRADE_SOURCE_FINGERPRINT
            "${LUISA_LLVM_DOWNGRADE_FINGERPRINT_INPUT}")
    if (NOT LUISA_LLVM_DOWNGRADE_SOURCE_FINGERPRINT STREQUAL
            LUISA_LLVM_DOWNGRADE_EXPECTED_SOURCE_FINGERPRINT)
        message(FATAL_ERROR
                "The llvm-downgrade source contents do not match pinned commit "
                "${LUISA_LLVM_DOWNGRADE_EXPECTED_COMMIT}. Expected source "
                "fingerprint ${LUISA_LLVM_DOWNGRADE_EXPECTED_SOURCE_FINGERPRINT}, "
                "found ${LUISA_LLVM_DOWNGRADE_SOURCE_FINGERPRINT}.")
    endif ()
    file(SHA256 "${LUISA_LLVM_DOWNGRADE_PATCH}"
            LUISA_LLVM_DOWNGRADE_PATCH_SHA256)
    string(APPEND LUISA_LLVM_DOWNGRADE_FINGERPRINT_INPUT
            "overlay:${LUISA_LLVM_DOWNGRADE_PATCH_SHA256}\n")
    string(SHA256 LUISA_LLVM_DOWNGRADE_FINGERPRINT
            "${LUISA_LLVM_DOWNGRADE_FINGERPRINT_INPUT}")

    # Keep the pinned third-party checkout pristine. The LLVM 21 and AIR typed-
    # pointer extensions live in the parent repository as a reviewable patch,
    # which is applied to a disposable copy in the build tree. Reuse a matching
    # mirror so ordinary reconfiguration does not invalidate its object files.
    set(LUISA_LLVM_DOWNGRADE_DIR
            "${CMAKE_CURRENT_BINARY_DIR}/llvm-downgrade-llvm21-air")
    set(LUISA_LLVM_DOWNGRADE_FINGERPRINT_FILE
            "${LUISA_LLVM_DOWNGRADE_DIR}/.luisa-source-fingerprint")
    set(LUISA_LLVM_DOWNGRADE_REFRESH_MIRROR ON)
    if (EXISTS "${LUISA_LLVM_DOWNGRADE_FINGERPRINT_FILE}")
        file(READ "${LUISA_LLVM_DOWNGRADE_FINGERPRINT_FILE}"
                LUISA_LLVM_DOWNGRADE_CACHED_FINGERPRINT)
        string(STRIP "${LUISA_LLVM_DOWNGRADE_CACHED_FINGERPRINT}"
                LUISA_LLVM_DOWNGRADE_CACHED_FINGERPRINT)
        if (LUISA_LLVM_DOWNGRADE_CACHED_FINGERPRINT STREQUAL
                LUISA_LLVM_DOWNGRADE_FINGERPRINT)
            set(LUISA_LLVM_DOWNGRADE_REFRESH_MIRROR OFF)
        endif ()
    endif ()
    if (LUISA_LLVM_DOWNGRADE_REFRESH_MIRROR)
        file(REMOVE_RECURSE "${LUISA_LLVM_DOWNGRADE_DIR}")
        file(MAKE_DIRECTORY "${LUISA_LLVM_DOWNGRADE_DIR}")
        file(COPY
                "${LUISA_LLVM_DOWNGRADE_SOURCE_DIR}/CMakeLists.txt"
                "${LUISA_LLVM_DOWNGRADE_SOURCE_DIR}/LICENSE.TXT"
                "${LUISA_LLVM_DOWNGRADE_SOURCE_DIR}/README.md"
                "${LUISA_LLVM_DOWNGRADE_SOURCE_DIR}/common"
                "${LUISA_LLVM_DOWNGRADE_SOURCE_DIR}/include"
                "${LUISA_LLVM_DOWNGRADE_SOURCE_DIR}/src"
                "${LUISA_LLVM_DOWNGRADE_SOURCE_DIR}/test"
                "${LUISA_LLVM_DOWNGRADE_SOURCE_DIR}/tools"
                DESTINATION "${LUISA_LLVM_DOWNGRADE_DIR}")
        set(LUISA_LLVM_DOWNGRADE_GIT_ENV
                "GIT_CEILING_DIRECTORIES=${CMAKE_CURRENT_BINARY_DIR}")
        execute_process(
                COMMAND "${CMAKE_COMMAND}" -E env
                --unset=GIT_DIR --unset=GIT_WORK_TREE
                "${LUISA_LLVM_DOWNGRADE_GIT_ENV}"
                "${GIT_EXECUTABLE}" apply --check --whitespace=nowarn
                "${LUISA_LLVM_DOWNGRADE_PATCH}"
                WORKING_DIRECTORY "${LUISA_LLVM_DOWNGRADE_DIR}"
                RESULT_VARIABLE LUISA_LLVM_DOWNGRADE_PATCH_CHECK_RESULT
                OUTPUT_VARIABLE LUISA_LLVM_DOWNGRADE_PATCH_CHECK_OUTPUT
                ERROR_VARIABLE LUISA_LLVM_DOWNGRADE_PATCH_CHECK_ERROR)
        if (NOT LUISA_LLVM_DOWNGRADE_PATCH_CHECK_RESULT EQUAL 0)
            message(FATAL_ERROR
                    "The LuisaCompute llvm-downgrade patch no longer applies to "
                    "the pinned submodule.\n"
                    "${LUISA_LLVM_DOWNGRADE_PATCH_CHECK_OUTPUT}"
                    "${LUISA_LLVM_DOWNGRADE_PATCH_CHECK_ERROR}")
        endif ()
        execute_process(
                COMMAND "${CMAKE_COMMAND}" -E env
                --unset=GIT_DIR --unset=GIT_WORK_TREE
                "${LUISA_LLVM_DOWNGRADE_GIT_ENV}"
                "${GIT_EXECUTABLE}" apply --whitespace=nowarn
                "${LUISA_LLVM_DOWNGRADE_PATCH}"
                WORKING_DIRECTORY "${LUISA_LLVM_DOWNGRADE_DIR}"
                RESULT_VARIABLE LUISA_LLVM_DOWNGRADE_PATCH_RESULT
                OUTPUT_VARIABLE LUISA_LLVM_DOWNGRADE_PATCH_OUTPUT
                ERROR_VARIABLE LUISA_LLVM_DOWNGRADE_PATCH_ERROR)
        if (NOT LUISA_LLVM_DOWNGRADE_PATCH_RESULT EQUAL 0)
            message(FATAL_ERROR
                    "Failed to prepare the patched llvm-downgrade sources.\n"
                    "${LUISA_LLVM_DOWNGRADE_PATCH_OUTPUT}"
                    "${LUISA_LLVM_DOWNGRADE_PATCH_ERROR}")
        endif ()
        file(WRITE "${LUISA_LLVM_DOWNGRADE_FINGERPRINT_FILE}"
                "${LUISA_LLVM_DOWNGRADE_FINGERPRINT}\n")
    endif ()

    add_library(luisa-compute-ext-llvm-downgrade STATIC
            llvm_downgrade.cpp llvm_downgrade.h
            ${LUISA_LLVM_DOWNGRADE_DIR}/common/legacy_opts.cpp
            ${LUISA_LLVM_DOWNGRADE_DIR}/src/BitcodeWriter140.cpp
            ${LUISA_LLVM_DOWNGRADE_DIR}/src/ModuleRewriter140.cpp
            ${LUISA_LLVM_DOWNGRADE_DIR}/src/PointerRewriter.cpp
            ${LUISA_LLVM_DOWNGRADE_DIR}/src/PointerRewriter.h
            ${LUISA_LLVM_DOWNGRADE_DIR}/src/ValueEnumerator140.cpp
            ${LUISA_LLVM_DOWNGRADE_DIR}/src/ValueEnumerator140.h)
    target_include_directories(luisa-compute-ext-llvm-downgrade
            BEFORE PRIVATE ${LUISA_LLVM_DOWNGRADE_DIR}/include)
    target_include_directories(luisa-compute-ext-llvm-downgrade
            PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}
            ${LUISA_LLVM_DOWNGRADE_DIR}/src
            ${LUISA_LLVM_DOWNGRADE_DIR}/common
            ${LLVM_INCLUDE_DIRS})
    separate_arguments(LUISA_LLVM_DOWNGRADE_DEFINITIONS NATIVE_COMMAND "${LLVM_DEFINITIONS}")
    target_compile_options(luisa-compute-ext-llvm-downgrade PRIVATE
            ${LUISA_LLVM_DOWNGRADE_DEFINITIONS}
            -include ${LUISA_LLVM_DOWNGRADE_DIR}/common/LegacyOpts.h)
    if (NOT LLVM_ENABLE_RTTI)
        target_compile_options(luisa-compute-ext-llvm-downgrade PRIVATE -fno-rtti)
    endif ()
    target_compile_definitions(luisa-compute-ext-llvm-downgrade PRIVATE LLVMDG_HAS_140=1)
    target_compile_features(luisa-compute-ext-llvm-downgrade PUBLIC cxx_std_20)
    set_target_properties(luisa-compute-ext-llvm-downgrade PROPERTIES
            POSITION_INDEPENDENT_CODE ON
            FOLDER ext
            OUTPUT_NAME luisa-ext-llvm-downgrade)
    llvm_map_components_to_libnames(LUISA_LLVM_DOWNGRADE_LIBS
            bitwriter core support transformutils)
    target_link_libraries(luisa-compute-ext-llvm-downgrade
            PRIVATE ${LUISA_LLVM_DOWNGRADE_LIBS})
endif ()

find_package(Threads REQUIRED)
target_link_libraries(luisa-compute-ext INTERFACE Threads::Threads)

add_custom_target(luisa-compute-ext-copy ALL
        COMMENT "Copy necessary files for third-party libraries")
add_dependencies(luisa-compute-ext-copy luisa-compute-ext)

function(luisa_compute_install_extension target)
    cmake_parse_arguments("EXT" "" "INCLUDE;HEADER_DESTINATION" "INCLUDES;HEADER_FILES" ${ARGN})
    install(TARGETS ${target}
            EXPORT LuisaComputeTargets
            LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
            ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
            RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
            PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/luisa/ext)
    if (EXT_INCLUDE)
        install(DIRECTORY ${EXT_INCLUDE}
                DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/luisa/ext
                FILES_MATCHING REGEX ".*\\.(h|hpp|hxx|hxx|inl)$")
    endif ()
    if (EXT_INCLUDES)
        foreach (include ${EXT_INCLUDES})
            install(DIRECTORY ${include}
                    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/luisa/ext
                    FILES_MATCHING REGEX ".*\\.(h|hpp|hxx|hxx|inl)$")
        endforeach ()
    endif ()
    if (EXT_HEADER_FILES)
        install(FILES ${EXT_HEADER_FILES}
                DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/luisa/ext/${EXT_HEADER_DESTINATION})
    endif ()
endfunction()

# include/luisa/core/stl/map.h publicly includes <fc/btree.h>. Keep the bundled
# header-only dependency on the same transitive interface as the other core
# STL dependencies so every CMake consumer sees it in both build and install
# trees. The upstream BTree project is intentionally not added as a CMake
# subdirectory: its standalone build config enables tests and mutates the
# enclosing project, while Luisa only consumes its public headers.
add_library(luisa-compute-ext-btree INTERFACE)
target_include_directories(luisa-compute-ext-btree INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/BTree/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/luisa/ext>)
target_link_libraries(luisa-compute-ext INTERFACE luisa-compute-ext-btree)
luisa_compute_install_extension(luisa-compute-ext-btree
        INCLUDE BTree/include/fc)

if (LUISA_COMPUTE_USE_SYSTEM_SPDLOG)
    find_package(spdlog REQUIRED)
    target_link_libraries(luisa-compute-ext INTERFACE spdlog::spdlog_header_only)
    target_compile_definitions(luisa-compute-ext INTERFACE LUISA_USE_SYSTEM_SPDLOG=1)
else ()
    set(SPDLOG_ENABLE_PCH OFF CACHE BOOL "" FORCE)
    set(SPDLOG_BUILD_SHARED ON CACHE BOOL "" FORCE)
    set(SPDLOG_NO_EXCEPTIONS ON CACHE BOOL "" FORCE)
    set(SPDLOG_NO_THREAD_ID ON CACHE BOOL "" FORCE)
    set(SPDLOG_FMT_EXTERNAL OFF CACHE BOOL "" FORCE)
    set(SPDLOG_FMT_EXTERNAL_HO OFF CACHE BOOL "" FORCE)
    set(SPDLOG_DISABLE_DEFAULT_LOGGER ON CACHE BOOL "" FORCE)
    add_subdirectory(spdlog)
    target_compile_definitions(spdlog_header_only INTERFACE
            FMT_USE_CONSTEVAL=0
            FMT_EXCEPTIONS=0
            FMT_HEADER_ONLY=1
            FMT_USE_NOEXCEPT=1)
    target_compile_definitions(spdlog PUBLIC
            FMT_USE_CONSTEVAL=0
            FMT_EXCEPTIONS=0
            FMT_HEADER_ONLY=1
            FMT_USE_NOEXCEPT=1)
    target_include_directories(spdlog_header_only INTERFACE
            $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/luisa/ext>)
    set_target_properties(spdlog PROPERTIES EXCLUDE_FROM_ALL ON)
    target_link_libraries(luisa-compute-ext INTERFACE spdlog::spdlog_header_only)
    luisa_compute_install_extension(spdlog_header_only INCLUDE spdlog/include/spdlog)
endif ()

if (LUISA_COMPUTE_USE_SYSTEM_XXHASH)
    find_package(xxHash CONFIG)
    if (TARGET xxHash::xxhash)
        target_link_libraries(luisa-compute-ext INTERFACE xxHash::xxhash)
    else () # via pkg-config
        find_package(PkgConfig REQUIRED)
        pkg_check_modules(XXHASH REQUIRED IMPORTED_TARGET libxxhash)
        target_link_libraries(luisa-compute-ext INTERFACE PkgConfig::XXHASH)
    endif ()
    target_compile_definitions(luisa-compute-ext INTERFACE LUISA_USE_SYSTEM_XXHASH=1)
else ()
    add_library(xxhash INTERFACE)
    target_include_directories(xxhash INTERFACE
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/xxHash>
            $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/luisa/ext/xxHash>)
    target_compile_definitions(xxhash INTERFACE XXH_INLINE_ALL)
    set_target_properties(xxhash PROPERTIES
            UNITY_BUILD ${LUISA_COMPUTE_ENABLE_UNITY_BUILD}
            OUTPUT_NAME luisa-ext-xxhash)
    target_link_libraries(luisa-compute-ext INTERFACE xxhash)
    luisa_compute_install_extension(xxhash HEADER_FILES
            xxHash/xxh3.h
            xxHash/xxhash.h
            HEADER_DESTINATION xxHash)
endif ()

# magic enum
if (LUISA_COMPUTE_USE_SYSTEM_MAGIC_ENUM)
    find_package(magic_enum REQUIRED)
    target_link_libraries(luisa-compute-ext INTERFACE magic_enum::magic_enum)
    target_compile_definitions(luisa-compute-ext INTERFACE LUISA_USE_SYSTEM_MAGIC_ENUM=1)
else ()
    add_library(magic_enum INTERFACE)
    target_include_directories(magic_enum INTERFACE
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/magic_enum/include>
            $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/luisa/ext>)
    target_link_libraries(luisa-compute-ext INTERFACE magic_enum)
    luisa_compute_install_extension(magic_enum INCLUDE magic_enum/include/magic_enum)
endif ()

# glfw
if (LUISA_COMPUTE_ENABLE_GUI AND NOT CMAKE_SYSTEM_NAME STREQUAL "iOS")
    if (LUISA_COMPUTE_USE_SYSTEM_GLFW)
        find_package(glfw3 REQUIRED)
        if (TARGET glfw)
            target_link_libraries(luisa-compute-ext INTERFACE glfw)
        else ()
            target_link_libraries(luisa-compute-ext INTERFACE glfw3::glfw)
        endif ()
        target_compile_definitions(luisa-compute-ext INTERFACE LUISA_USE_SYSTEM_GLFW=1)
    else ()
        if (UNIX AND NOT APPLE)
            set(GLFW_BUILD_WAYLAND ${LUISA_COMPUTE_ENABLE_WAYLAND} CACHE BOOL "" FORCE)
        endif ()
        set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
        set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
        set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
        set(GLFW_INSTALL OFF CACHE BOOL "" FORCE)
        set(GLFW_LIBRARY_TYPE SHARED CACHE STRING "" FORCE)
        add_subdirectory(glfw)
        set_target_properties(glfw PROPERTIES OUTPUT_NAME luisa-ext-glfw)
        target_include_directories(glfw INTERFACE
                $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/luisa/ext>)
        target_link_libraries(luisa-compute-ext INTERFACE glfw)
        luisa_compute_install_extension(glfw INCLUDE glfw/include/GLFW)
    endif ()
endif ()

if (LUISA_COMPUTE_USE_SYSTEM_STL)
    target_compile_definitions(luisa-compute-ext INTERFACE LUISA_USE_SYSTEM_STL=1)
else ()
    add_subdirectory(EASTL)
    target_include_directories(EASTL-interface INTERFACE
            $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/luisa/ext>)
    target_include_directories(EABase INTERFACE
            $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/luisa/ext>)
    set_target_properties(EASTL-object PROPERTIES UNITY_BUILD ${LUISA_COMPUTE_ENABLE_UNITY_BUILD})
    target_link_libraries(luisa-compute-ext INTERFACE EASTL-interface)
    luisa_compute_install_extension(EASTL-interface INCLUDE EASTL/include/EASTL)
    luisa_compute_install_extension(EABase INCLUDE EASTL/packages/EABase/include/Common/EABase)
    target_compile_features(EASTL-interface INTERFACE cxx_std_20)
    # silence deprecation warnings from EASTL for now (we will fix them when the upstream does)
    target_compile_definitions(EASTL-interface INTERFACE EASTL_DEPRECATIONS_FOR_2024_APRIL=EA_DISABLED)
    if (CMAKE_CXX_COMPILER_ID MATCHES "Clang" AND
            CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 15.0)
        target_compile_options(EASTL-interface INTERFACE -Wno-deprecated-builtins)
    endif ()
endif ()

# reproc
if (LUISA_COMPUTE_USE_SYSTEM_REPROC)
    find_package(reproc REQUIRED)
    find_package(reproc++ REQUIRED)
    if (TARGET reproc::reproc AND TARGET reproc::reproc++)
        target_link_libraries(luisa-compute-ext INTERFACE
                reproc::reproc reproc::reproc++)
    else ()
        target_link_libraries(luisa-compute-ext INTERFACE reproc reproc++)
    endif ()
    target_compile_definitions(luisa-compute-ext INTERFACE LUISA_USE_SYSTEM_REPROC=1)
else ()
    set(REPROC_DEVELOP OFF CACHE BOOL "" FORCE)
    set(REPROC_TEST OFF CACHE BOOL "" FORCE)
    set(REPROC_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(REPROC_WARNINGS ON CACHE BOOL "" FORCE)
    set(REPROC_TIDY OFF CACHE BOOL "" FORCE)
    set(REPROC_SANITIZERS OFF CACHE BOOL "" FORCE)
    set(REPROC_WARNINGS_AS_ERRORS OFF CACHE BOOL "" FORCE)
    set(REPROC_OBJECT_LIBRARIES ON CACHE BOOL "" FORCE)
    set(REPROC_INSTALL OFF CACHE BOOL "" FORCE)
    set(REPROC_INSTALL_PKGCONFIG OFF CACHE BOOL "" FORCE)
    set(REPROC++ ON CACHE BOOL "" FORCE)
    set(REPROC_MULTITHREADED ON CACHE BOOL "" FORCE)
    add_subdirectory(reproc)
    if (UNIX AND NOT APPLE)
        set_target_properties(reproc PROPERTIES POSITION_INDEPENDENT_CODE ON)
        set_target_properties(reproc++ PROPERTIES POSITION_INDEPENDENT_CODE ON)
    endif ()
    add_library(reproc-interface INTERFACE)
    target_compile_definitions(reproc-interface INTERFACE REPROC_SHARED REPROCXX_SHARED)
    target_include_directories(reproc-interface INTERFACE
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/reproc/reproc/include>
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/reproc/reproc++/include>
            $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/luisa/ext>)
    target_link_libraries(luisa-compute-ext INTERFACE reproc-interface)
    install(TARGETS reproc-interface EXPORT LuisaComputeTargets)

    target_compile_definitions(reproc PRIVATE REPROC_BUILDING PUBLIC REPROC_SHARED)
    target_compile_definitions(reproc++ PRIVATE REPROCXX_BUILDING PUBLIC REPROCXX_SHARED)
    target_link_libraries(reproc PUBLIC reproc-interface)
    target_link_libraries(reproc++ PUBLIC reproc-interface)

    install(DIRECTORY reproc/reproc/include/reproc
            DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/luisa/ext
            FILES_MATCHING REGEX ".*\\.h$")
    install(DIRECTORY reproc/reproc++/include/reproc++
            DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/luisa/ext
            FILES_MATCHING REGEX ".*\\.hpp$")
    # Note: we do not link `reproc` and `reproc++` to `luisa-compute-ext` but
    #       instead to `luisa-compute-core` since this is an OBJECT library.
endif ()

# yyjson
if (LUISA_COMPUTE_USE_SYSTEM_YYJSON)
    find_package(yyjson REQUIRED)
    target_link_libraries(luisa-compute-ext INTERFACE yyjson::yyjson)
    target_compile_definitions(luisa-compute-ext INTERFACE LUISA_USE_SYSTEM_YYJSON=1)
else ()
    add_library(luisa-compute-ext-yyjson-interface INTERFACE)
    target_include_directories(luisa-compute-ext-yyjson-interface INTERFACE
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/yyjson/src>
            $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/luisa/ext>)
    target_compile_definitions(luisa-compute-ext-yyjson-interface INTERFACE YYJSON_IMPORTS=1)
    luisa_compute_install_extension(luisa-compute-ext-yyjson-interface
            HEADER_FILES yyjson/src/yyjson.h)
    add_library(luisa-compute-ext-yyjson OBJECT yyjson/src/yyjson.c yyjson/src/yyjson.h)
    target_link_libraries(luisa-compute-ext-yyjson PUBLIC luisa-compute-ext-yyjson-interface)
    target_compile_definitions(luisa-compute-ext-yyjson PRIVATE YYJSON_EXPORTS=1)
    if (UNIX AND NOT APPLE)
        set_target_properties(luisa-compute-ext-yyjson PROPERTIES POSITION_INDEPENDENT_CODE ON)
    endif ()
endif ()

# marl
if (LUISA_COMPUTE_USE_SYSTEM_MARL)
    find_package(marl CONFIG REQUIRED)
    target_link_libraries(luisa-compute-ext INTERFACE marl::marl)
    target_compile_definitions(luisa-compute-ext INTERFACE LUISA_USE_SYSTEM_MARL=1)
else ()
    add_library(luisa-compute-ext-marl-interface INTERFACE)
    target_include_directories(luisa-compute-ext-marl-interface INTERFACE
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/marl/include>
            $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/luisa/ext>)
    target_compile_definitions(luisa-compute-ext-marl-interface INTERFACE MARL_DLL=1)
    target_link_libraries(luisa-compute-ext INTERFACE luisa-compute-ext-marl-interface)
    luisa_compute_install_extension(luisa-compute-ext-marl-interface INCLUDE marl/include/marl)

    file(GLOB LUISA_COMPUTE_MARL_HEADERS CONFIGURE_DEPENDS marl/include/marl/*.h)
    file(GLOB LUISA_COMPUTE_MARL_C_SOURCES CONFIGURE_DEPENDS marl/src/*.c)
    file(GLOB LUISA_COMPUTE_MARL_ASM_SOURCES CONFIGURE_DEPENDS marl/src/*.S)
    file(GLOB LUISA_COMPUTE_MARL_CXX_SOURCES CONFIGURE_DEPENDS marl/src/*.cpp)
    # remove the manual unity build file "build.marl.cpp" and let CMake handle it
    list(FILTER LUISA_COMPUTE_MARL_CXX_SOURCES EXCLUDE REGEX ".*build.marl.cpp$")
    add_library(luisa-compute-ext-marl OBJECT ${LUISA_COMPUTE_MARL_HEADERS} ${LUISA_COMPUTE_MARL_CXX_SOURCES})
    if (LUISA_COMPUTE_USE_SYSTEM_STL)
        target_compile_definitions(luisa-compute-ext-marl-interface INTERFACE MARL_USE_SYSTEM_STL=1)
    else ()
        target_link_libraries(luisa-compute-ext-marl-interface INTERFACE EASTL-interface)
    endif ()
    target_compile_definitions(luisa-compute-ext-marl PRIVATE MARL_BUILDING_DLL=1)
    target_link_libraries(luisa-compute-ext-marl PRIVATE luisa-compute-ext-marl-interface)

    set_target_properties(luisa-compute-ext-marl PROPERTIES UNITY_BUILD ${LUISA_COMPUTE_ENABLE_UNITY_BUILD})
    if (NOT WIN32)
        enable_language(ASM)
        target_sources(luisa-compute-ext-marl PRIVATE ${LUISA_COMPUTE_MARL_C_SOURCES} ${LUISA_COMPUTE_MARL_ASM_SOURCES})
    endif ()
    if (UNIX AND NOT APPLE)
        set_target_properties(luisa-compute-ext-marl PROPERTIES POSITION_INDEPENDENT_CODE ON)
    endif ()
endif ()

# lmdb
if (LUISA_COMPUTE_USE_SYSTEM_LMDB)
    find_package(unofficial-lmdb CONFIG)
    if (TARGET unofficial::lmdb::lmdb)
        target_link_libraries(luisa-compute-ext INTERFACE unofficial::lmdb::lmdb)
    else () # via pkg-config
        find_package(PkgConfig REQUIRED)
        pkg_check_modules(LMDB REQUIRED IMPORTED_TARGET lmdb)
        target_link_libraries(luisa-compute-ext INTERFACE PkgConfig::LMDB)
    endif ()
    target_compile_definitions(luisa-compute-ext INTERFACE LUISA_USE_SYSTEM_LMDB=1)
else ()
    add_library(luisa-compute-ext-lmdb OBJECT
            liblmdb/mdb.c liblmdb/lmdb.h
            liblmdb/midl.c liblmdb/midl.h)
    target_include_directories(luisa-compute-ext-lmdb PUBLIC
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/liblmdb>)
    if (UNIX AND NOT APPLE)
        set_target_properties(luisa-compute-ext-lmdb PROPERTIES POSITION_INDEPENDENT_CODE ON)
    endif ()
    if (WIN32)
        target_compile_options(luisa-compute-ext-lmdb PRIVATE "-UWIN32_LEAN_AND_MEAN")
    endif ()
    set_target_properties(luisa-compute-ext-lmdb PROPERTIES
            UNITY_BUILD ${LUISA_COMPUTE_ENABLE_UNITY_BUILD}
            OUTPUT_NAME luisa-ext-lmdb)
endif ()

# TODO: support system half and stb

# half
add_library(half INTERFACE)
target_include_directories(half INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/half/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/luisa/ext>)
target_link_libraries(luisa-compute-ext INTERFACE half)
luisa_compute_install_extension(half HEADER_FILES half/include/half.hpp)
if (CMAKE_CXX_COMPILER_ID MATCHES "Clang" AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 20.0)
    target_compile_options(half INTERFACE
            -Wno-invalid-specialization
            -Wno-deprecated-literal-operator
            -Wno-elaborated-enum-base)
endif ()

# stb
add_subdirectory(stb)
target_include_directories(luisa-compute-ext-stb-interface INTERFACE
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/luisa/ext>)
target_link_libraries(luisa-compute-ext INTERFACE luisa-compute-ext-stb-interface)

luisa_compute_install_extension(luisa-compute-ext-stb-interface HEADER_FILES
        stb/stb/stb_image.h
        stb/stb/stb_image_resize2.h
        stb/stb/stb_image_write.h
        HEADER_DESTINATION stb)

# glslang
# Vulkan's backend-private compute kernels are compiled to SPIR-V during the
# build. Use the bundled, version-pinned glslang executable so those artifacts
# are reproducible and never need to be checked into the source tree.
set(ENABLE_GLSLANG_BINARIES ${LUISA_COMPUTE_ENABLE_VULKAN} CACHE BOOL "" FORCE)
set(GLSLANG_TESTS OFF CACHE BOOL "" FORCE)
set(ENABLE_HLSL ${LUISA_COMPUTE_ENABLE_VULKAN} CACHE BOOL "" FORCE)
set(BUILD_EXTERNAL OFF CACHE BOOL "" FORCE)
set(ENABLE_OPT OFF CACHE BOOL "" FORCE)
set(ALLOW_EXTERNAL_SPIRV_TOOLS OFF CACHE BOOL "" FORCE)
add_subdirectory(glslang)
target_link_libraries(luisa-compute-ext INTERFACE glslang)
install(TARGETS glslang EXPORT LuisaComputeTargets)

if (SKBUILD OR LUISA_COMPUTE_FORCE_PYTHON_BINDINGS)
    add_subdirectory(pybind11)
endif ()

if (LUISA_COMPUTE_ENABLE_VULKAN)
# SPIRV-Tools
set(SPIRV-Headers_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/spirv-headers CACHE PATH "" FORCE)
set(SPIRV_HEADER_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/spirv-headers/include CACHE PATH "" FORCE)
set(SPIRV_SKIP_EXECUTABLES ON CACHE BOOL "" FORCE)
set(SPIRV_SKIP_TESTS ON CACHE BOOL "" FORCE)
set(SKIP_SPIRV_TOOLS_INSTALL ON CACHE BOOL "" FORCE)
set(SPIRV_WERROR OFF CACHE BOOL "" FORCE)
add_subdirectory(SPIRV-Tools)
# SPIRV-Tools adds /WX to its base MSVC warning list even when SPIRV_WERROR is
# disabled. Append /WX- so warnings in the bundled dependency stay non-fatal.
if (MSVC)
    foreach(_spirv_target IN ITEMS
            SPIRV-Tools SPIRV-Tools-static SPIRV-Tools-shared
            SPIRV-Tools-opt SPIRV-Tools-link SPIRV-Tools-reduce
            SPIRV-Tools-lint SPIRV-Tools-diff)
        if (TARGET ${_spirv_target})
            # target_compile_options cannot be used on ALIAS targets; resolve first
            get_target_property(_spirv_aliased ${_spirv_target} ALIASED_TARGET)
            if (_spirv_aliased)
                target_compile_options(${_spirv_aliased} PRIVATE /WX-)
            else()
                target_compile_options(${_spirv_target} PRIVATE /WX-)
            endif()
            unset(_spirv_aliased)
        endif()
    endforeach()
    unset(_spirv_target)
endif()
endif ()


install(TARGETS luisa-compute-ext EXPORT LuisaComputeTargets)

if (LUISA_COMPUTE_ENABLE_VULKAN OR
    LUISA_COMPUTE_ENABLE_CUDA OR
    LUISA_COMPUTE_ENABLE_FALLBACK OR
    LUISA_COMPUTE_ENABLE_HIP)
    add_subdirectory(volk)
endif ()

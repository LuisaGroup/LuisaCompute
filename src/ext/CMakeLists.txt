add_library(luisa-compute-ext INTERFACE)

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
    target_include_directories(xxhash INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/xxHash>)
    target_compile_definitions(xxhash INTERFACE XXH_INLINE_ALL)
    set_target_properties(xxhash PROPERTIES
            UNITY_BUILD ${LUISA_COMPUTE_ENABLE_UNITY_BUILD}
            OUTPUT_NAME lc-ext-xxhash)
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
    target_include_directories(magic_enum INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/magic_enum/include>)
    target_link_libraries(luisa-compute-ext INTERFACE magic_enum)
    luisa_compute_install_extension(magic_enum INCLUDE magic_enum/include/magic_enum)
endif ()

# glfw
if (LUISA_COMPUTE_ENABLE_GUI)
    if (LUISA_COMPUTE_USE_SYSTEM_GLFW)
        find_package(glfw3 REQUIRED)
        target_link_libraries(luisa-compute-ext INTERFACE glfw)
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
        set_target_properties(glfw PROPERTIES OUTPUT_NAME lc-ext-glfw)
        target_link_libraries(luisa-compute-ext INTERFACE glfw)
        luisa_compute_install_extension(glfw)
    endif ()
endif ()

if (LUISA_COMPUTE_USE_SYSTEM_STL)
    target_compile_definitions(luisa-compute-ext INTERFACE LUISA_USE_SYSTEM_STL=1)
else ()
    add_subdirectory(EASTL)
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
    target_link_libraries(luisa-compute-ext INTERFACE reproc reproc++)
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
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/reproc/reproc++/include>)
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
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/yyjson/src>)
    target_compile_definitions(luisa-compute-ext-yyjson-interface INTERFACE YYJSON_IMPORTS=1)
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
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/marl/include>)
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
    install(TARGETS luisa-compute-ext-marl EXPORT LuisaComputeTargets)
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
            OUTPUT_NAME lc-ext-lmdb)
    install(TARGETS luisa-compute-ext-lmdb EXPORT LuisaComputeTargets)
endif ()

# TODO: support system half and stb

# half
add_library(half INTERFACE)
target_include_directories(half INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/half/include>)
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
target_link_libraries(luisa-compute-ext INTERFACE luisa-compute-ext-stb-interface)

luisa_compute_install_extension(luisa-compute-ext-stb-interface HEADER_FILES
        stb/stb/stb_image.h
        stb/stb/stb_image_resize2.h
        stb/stb/stb_image_write.h
        HEADER_DESTINATION stb)

if (SKBUILD OR LUISA_COMPUTE_FORCE_PYTHON_BINDINGS)
    add_subdirectory(pybind11)
endif ()

install(TARGETS luisa-compute-ext EXPORT LuisaComputeTargets)

# volk
if (LUISA_COMPUTE_ENABLE_VULKAN)
    if (LUISA_COMPUTE_USE_SYSTEM_VOLK)
        find_package(volk REQUIRED)
        target_link_libraries(luisa-compute-ext INTERFACE volk::volk)
        target_compile_definitions(luisa-compute-ext INTERFACE LUISA_USE_SYSTEM_VULKAN=1)
    else ()
        add_subdirectory(volk)
    endif ()
endif ()

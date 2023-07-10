add_library(luisa-compute-ext INTERFACE)

add_custom_target(luisa-compute-ext-copy ALL
        COMMENT "Copy necessary files for third-party libraries")
add_dependencies(luisa-compute-ext-copy luisa-compute-ext)

function(luisa_compute_install_extension target)
    cmake_parse_arguments("EXT" "" "INCLUDE;HEADER_DESTINATION" "HEADER_FILES" ${ARGN})
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
    if (EXT_HEADER_FILES)
        install(FILES ${EXT_HEADER_FILES}
                DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/luisa/ext/${EXT_HEADER_DESTINATION})
    endif ()
endfunction()

set(SPDLOG_ENABLE_PCH OFF CACHE BOOL "" FORCE)
set(SPDLOG_BUILD_SHARED ON CACHE BOOL "" FORCE)
set(SPDLOG_NO_EXCEPTIONS ON CACHE BOOL "" FORCE)
set(SPDLOG_NO_THREAD_ID ON CACHE BOOL "" FORCE)
set(SPDLOG_FMT_EXTERNAL OFF CACHE BOOL "" FORCE)
set(SPDLOG_FMT_EXTERNAL_HO OFF CACHE BOOL "" FORCE)
set(SPDLOG_DISABLE_DEFAULT_LOGGER ON CACHE BOOL "" FORCE)
add_subdirectory(spdlog)
target_compile_definitions(spdlog_header_only INTERFACE
        FMT_CONSTEVAL=constexpr
        FMT_EXCEPTIONS=0
        FMT_HEADER_ONLY=1
        FMT_USE_NOEXCEPT=1)
target_compile_definitions(spdlog PUBLIC
        FMT_CONSTEVAL=constexpr
        FMT_EXCEPTIONS=0
        FMT_HEADER_ONLY=1
        FMT_USE_NOEXCEPT=1)
set_target_properties(spdlog PROPERTIES EXCLUDE_FROM_ALL ON)
target_link_libraries(luisa-compute-ext INTERFACE spdlog::spdlog_header_only)
luisa_compute_install_extension(spdlog_header_only INCLUDE spdlog/include/spdlog)

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

add_subdirectory(stb)
target_link_libraries(luisa-compute-ext INTERFACE stb)
set_target_properties(stb PROPERTIES OUTPUT_NAME lc-ext-stb)
luisa_compute_install_extension(stb HEADER_FILES
        stb/stb/stb_image.h
        stb/stb/stb_image_resize.h
        stb/stb/stb_image_write.h
        HEADER_DESTINATION stb)

add_library(magic_enum INTERFACE)
target_include_directories(magic_enum INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/magic_enum/include>)
target_link_libraries(luisa-compute-ext INTERFACE magic_enum)
luisa_compute_install_extension(magic_enum HEADER_FILES
        magic_enum/include/magic_enum.hpp
        magic_enum/include/magic_enum_containers.hpp
        magic_enum/include/magic_enum_flags.hpp
        magic_enum/include/magic_enum_format.hpp
        magic_enum/include/magic_enum_fuse.hpp
        magic_enum/include/magic_enum_iostream.hpp
        magic_enum/include/magic_enum_switch.hpp)

if (LUISA_COMPUTE_ENABLE_GUI)
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

add_subdirectory(EASTL)
set_target_properties(EASTL PROPERTIES
        UNITY_BUILD ${LUISA_COMPUTE_ENABLE_UNITY_BUILD}
        OUTPUT_NAME lc-ext-eastl)
target_link_libraries(luisa-compute-ext INTERFACE EASTL)
luisa_compute_install_extension(EASTL INCLUDE EASTL/include/EASTL)
luisa_compute_install_extension(EABase INCLUDE EASTL/packages/EABase/include/Common/EABase)
if (CMAKE_CXX_COMPILER_ID MATCHES "Clang" AND
        CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 15.0)
    target_compile_options(EASTL PUBLIC -Wno-deprecated-builtins)
endif ()

add_library(unordered_dense INTERFACE)
target_include_directories(unordered_dense INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/unordered_dense/include>)
target_link_libraries(luisa-compute-ext INTERFACE unordered_dense)
luisa_compute_install_extension(unordered_dense INCLUDE unordered_dense/include)

add_library(half INTERFACE)
target_include_directories(half INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/half/include>)
target_link_libraries(luisa-compute-ext INTERFACE half)
luisa_compute_install_extension(half INCLUDE half/include)

if (SKBUILD)
    add_subdirectory(pybind11)
endif ()

install(TARGETS luisa-compute-ext EXPORT LuisaComputeTargets)

# for rust bindings
if (LUISA_COMPUTE_ENABLE_RUST)
    add_subdirectory(corrosion)
endif ()

add_library(luisa-compute-ext INTERFACE)

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
        stb/stb/stb_image_resize2.h
        stb/stb/stb_image_write.h
        HEADER_DESTINATION stb)

add_library(magic_enum INTERFACE)
target_include_directories(magic_enum INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/magic_enum/include>)
target_link_libraries(luisa-compute-ext INTERFACE magic_enum)
luisa_compute_install_extension(magic_enum INCLUDE magic_enum/include/magic_enum)

if (LUISA_COMPUTE_ENABLE_GUI)

    # glfw
    set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
    set(GLFW_INSTALL OFF CACHE BOOL "" FORCE)
    set(GLFW_LIBRARY_TYPE SHARED CACHE STRING "" FORCE)
    add_subdirectory(glfw)
    set_target_properties(glfw PROPERTIES OUTPUT_NAME lc-ext-glfw)
    target_link_libraries(luisa-compute-ext INTERFACE glfw)
    luisa_compute_install_extension(glfw)

    # imgui
    set(LUISA_IMGUI_SOURCES
            imgui/imgui.cpp
            imgui/imgui_demo.cpp
            imgui/imgui_draw.cpp
            imgui/imgui_tables.cpp
            imgui/imgui_widgets.cpp
            imgui/backends/imgui_impl_glfw.cpp)
    set(LUISA_IMGUI_HEADERS
            imgui/imconfig.h
            imgui/imgui.h
            imgui/imgui_internal.h
            imgui/imstb_rectpack.h
            imgui/imstb_textedit.h
            imgui/imstb_truetype.h
            imgui/backends/imgui_impl_glfw.h)
    add_library(luisa-compute-ext-imgui SHARED
            ${LUISA_IMGUI_SOURCES}
            ${LUISA_IMGUI_HEADERS})
    set_target_properties(luisa-compute-ext-imgui PROPERTIES
            UNITY_BUILD ${LUISA_COMPUTE_ENABLE_UNITY_BUILD}
            OUTPUT_NAME lc-ext-imgui
            PUBLIC_HEADER "${LUISA_IMGUI_HEADERS}")
    target_compile_definitions(luisa-compute-ext-imgui PUBLIC GLFW_INCLUDE_NONE)
    if (WIN32)
        target_compile_definitions(luisa-compute-ext-imgui PRIVATE
                "IMGUI_API=__declspec(dllexport)"
                "IMGUI_IMPL_API=__declspec(dllexport)")
    endif ()
    target_link_libraries(luisa-compute-ext-imgui PUBLIC glfw)
    target_include_directories(luisa-compute-ext-imgui PUBLIC
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/imgui>
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/imgui/backends>)
    luisa_compute_install_extension(luisa-compute-ext-imgui)
    target_link_libraries(luisa-compute-ext INTERFACE luisa-compute-ext-imgui)

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

add_library(half INTERFACE)
target_include_directories(half INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/half/include>)
target_link_libraries(luisa-compute-ext INTERFACE half)
luisa_compute_install_extension(half HEADER_FILES half/include/half.hpp)

# reproc
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
    target_compile_options(reproc PUBLIC "-fPIC")
    target_compile_options(reproc++ PUBLIC "-fPIC")
endif ()
target_include_directories(luisa-compute-ext INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/reproc/reproc/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/reproc/reproc++/include>)
install(DIRECTORY reproc/reproc/include/reproc
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/luisa/ext
        FILES_MATCHING REGEX ".*\\.h$")
install(DIRECTORY reproc/reproc++/include/reproc++
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/luisa/ext
        FILES_MATCHING REGEX ".*\\.hpp$")
# Note: we do not link `reproc` and `reproc++` to `luisa-compute-ext` but
#       instead to `luisa-compute-core` since this is an OBJECT library.

if (SKBUILD OR LUISA_COMPUTE_FORCE_PYTHON_BINDINGS)
    add_subdirectory(pybind11)
endif ()

install(TARGETS luisa-compute-ext EXPORT LuisaComputeTargets)

# for rust bindings
if (LUISA_COMPUTE_ENABLE_RUST)
    add_subdirectory(corrosion)
endif ()

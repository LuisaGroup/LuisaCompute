add_library(luisa-compute-ext INTERFACE)

if (LUISA_COMPUTE_ENABLE_MIMALLOC)
    set(MI_OVERRIDE OFF CACHE BOOL "" FORCE)
    set(MI_XMALLOC ON CACHE BOOL "" FORCE)
    set(MI_USE_CXX OFF CACHE BOOL "" FORCE)
    set(MI_OSX_ZONE ON CACHE BOOL "" FORCE)
    set(MI_BUILD_STATIC ON CACHE BOOL "" FORCE)
    set(MI_BUILD_SHARED OFF CACHE BOOL "" FORCE)
    set(MI_BUILD_OBJECT OFF CACHE BOOL "" FORCE)
    set(MI_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    add_subdirectory(mimalloc)
    if (MSVC)
        target_compile_options(mimalloc-static PRIVATE /W0)
        target_compile_definitions(mimalloc-static PUBLIC _CRT_SECURE_NO_WARNINGS=1)
    endif ()
    set_target_properties(mimalloc-static PROPERTIES UNITY_BUILD ON)
endif ()

set(SPDLOG_ENABLE_PCH ON CACHE BOOL "" FORCE)
set(SPDLOG_BUILD_SHARED ON CACHE BOOL "" FORCE)
set(SPDLOG_NO_EXCEPTIONS ON CACHE BOOL "" FORCE)
add_subdirectory(spdlog)
target_compile_definitions(spdlog INTERFACE FMT_CONSTEVAL=inline)
set_target_properties(spdlog PROPERTIES UNITY_BUILD ON)
target_compile_definitions(spdlog PUBLIC FMT_EXCEPTIONS=0)
target_link_libraries(luisa-compute-ext INTERFACE spdlog)

add_library(xxhash INTERFACE)
target_include_directories(xxhash INTERFACE xxHash)
target_compile_definitions(xxhash INTERFACE XXH_INLINE_ALL)
set_target_properties(xxhash PROPERTIES UNITY_BUILD ON)
target_link_libraries(luisa-compute-ext INTERFACE xxhash)

add_library(asio SHARED asio/asio/src/asio.cpp)
target_include_directories(asio PUBLIC asio/asio/include)
target_compile_definitions(asio PUBLIC
        ASIO_STANDALONE=1
        ASIO_DYN_LINK=1
        ASIO_HAS_STD_INVOKE_RESULT=1
        $<$<PLATFORM_ID:Windows>:_WIN32_WINNT=0x0A00>)
target_link_libraries(luisa-compute-ext INTERFACE asio)

add_subdirectory(stb)
target_link_libraries(luisa-compute-ext INTERFACE stb)

add_subdirectory(json)
target_link_libraries(luisa-compute-ext INTERFACE nlohmann_json::nlohmann_json)

add_subdirectory(glad)
target_link_libraries(luisa-compute-ext INTERFACE glad)

find_package(GLFW3 CONFIG)
if (NOT GLFW3_FOUND)
    set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
    set(GLFW_INSTALL OFF CACHE BOOL "" FORCE)
    set(GLFW_LIBRARY_TYPE SHARED CACHE STRING "" FORCE)
    add_subdirectory(glfw)
    set_target_properties(glfw PROPERTIES UNITY_BUILD ON)
endif ()
target_link_libraries(luisa-compute-ext INTERFACE glfw)

add_subdirectory(imgui)
target_link_libraries(luisa-compute-ext INTERFACE imgui)

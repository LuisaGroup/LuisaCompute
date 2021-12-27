add_library(luisa-compute-ext INTERFACE)

set(FMT_OS OFF CACHE BOOL "" FORCE)
add_subdirectory(fmt)
target_compile_definitions(fmt-header-only INTERFACE
        FMT_CONSTEVAL=constexpr
        FMT_EXCEPTIONS=0
        FMT_HEADER_ONLY=1
        FMT_USE_NOEXCEPT=1)
target_link_libraries(luisa-compute-ext INTERFACE fmt::fmt-header-only)

set(SPDLOG_ENABLE_PCH OFF CACHE BOOL "" FORCE)
set(SPDLOG_BUILD_SHARED ON CACHE BOOL "" FORCE)
set(SPDLOG_NO_EXCEPTIONS OFF CACHE BOOL "" FORCE)
set(SPDLOG_NO_THREAD_ID ON CACHE BOOL "" FORCE)
set(SPDLOG_FMT_EXTERNAL_HO ON CACHE BOOL "" FORCE)
set(SPDLOG_DISABLE_DEFAULT_LOGGER ON CACHE BOOL "" FORCE)
add_subdirectory(spdlog)
set_target_properties(spdlog PROPERTIES UNITY_BUILD ON)
target_link_libraries(spdlog PUBLIC fmt::fmt-header-only)
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
        ASIO_HAS_STD_INVOKE_RESULT=1)
if (WIN32)
    target_compile_definitions(asio PUBLIC _WIN32_WINNT=0x0A00)
    target_link_libraries(asio PUBLIC ws2_32 wsock32 bcrypt)
endif ()
target_link_libraries(luisa-compute-ext INTERFACE asio)

add_subdirectory(stb)
target_link_libraries(luisa-compute-ext INTERFACE stb)

set(JSON_MultipleHeaders ON CACHE BOOL "" FORCE)
set(JSON_SystemInclude ON CACHE BOOL "" FORCE)
add_subdirectory(json)
target_link_libraries(luisa-compute-ext INTERFACE nlohmann_json::nlohmann_json)

if (LUISA_COMPUTE_ENABLE_GUI)
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
endif ()

set(EMBREE_TUTORIALS OFF CACHE BOOL "" FORCE)
set(EMBREE_TASKING_SYSTEM INTERNAL CACHE STRING "" FORCE)
set(CMAKE_CXX_STANDARD 11)
add_subdirectory(embree)
target_link_libraries(luisa-compute-ext INTERFACE embree)

add_subdirectory(EASTL)
target_link_libraries(luisa-compute-ext INTERFACE EASTL)

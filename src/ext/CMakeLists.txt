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


#add_subdirectory(stb)
#target_link_libraries(luisa-compute-ext INTERFACE stb)

set(JSON_MultipleHeaders ON CACHE BOOL "" FORCE)
set(JSON_SystemInclude ON CACHE BOOL "" FORCE)
add_subdirectory(json)
target_link_libraries(luisa-compute-ext INTERFACE nlohmann_json::nlohmann_json)

add_subdirectory(EASTL)
target_link_libraries(luisa-compute-ext INTERFACE EASTL)

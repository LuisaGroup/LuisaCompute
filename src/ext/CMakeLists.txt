add_library(luisa-compute-ext INTERFACE)

#set(WITH_WERROR OFF CACHE BOOL "" FORCE)
#set(SKIP_PERFORMANCE_COMPARISON ON CACHE BOOL "" FORCE)
#add_subdirectory(cereal)
#target_link_libraries(luisa-compute-ext INTERFACE cereal::cereal)
target_include_directories(luisa-compute-ext INTERFACE cereal/include)

add_subdirectory(fmt)
set_target_properties(fmt PROPERTIES UNITY_BUILD ON)
target_compile_definitions(fmt PUBLIC FMT_EXCEPTIONS=0)

set(SPDLOG_BUILD_SHARED ON CACHE BOOL "" FORCE)
set(SPDLOG_ENABLE_PCH ON CACHE BOOL "" FORCE)
set(SPDLOG_FMT_EXTERNAL ON CACHE BOOL "" FORCE)
set(SPDLOG_FMT_EXTERNAL_HO OFF CACHE BOOL "" FORCE)
add_subdirectory(spdlog)
set_target_properties(spdlog PROPERTIES UNITY_BUILD ON)
target_compile_definitions(spdlog PUBLIC SPDLOG_NO_EXCEPTIONS=1)
target_link_libraries(luisa-compute-ext INTERFACE spdlog::spdlog)

find_package(Threads REQUIRED)
target_link_libraries(luisa-compute-ext INTERFACE Threads::Threads)

set(CMAKE_C_STANDARD 99)
add_subdirectory(xxHash/cmake_unofficial)
set_target_properties(xxhash PROPERTIES UNITY_BUILD ON)
target_link_libraries(luisa-compute-ext INTERFACE xxHash::xxhash)

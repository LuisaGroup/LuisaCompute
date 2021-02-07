add_library(luisa-compute-ext INTERFACE)

find_package(OpenImageIO)  # TODO
target_link_libraries(luisa-compute-ext INTERFACE OpenImageIO::OpenImageIO)

find_package(OpenCV REQUIRED)
target_link_libraries(luisa-compute-ext INTERFACE ${OpenCV_LIBS})

set(WITH_WERROR OFF CACHE BOOL "" FORCE)
set(SKIP_PERFORMANCE_COMPARISON ON CACHE BOOL "" FORCE)
add_subdirectory(cereal)
target_link_libraries(luisa-compute-ext INTERFACE cereal::cereal)

find_package(spdlog REQUIRED)
target_link_libraries(luisa-compute-ext INTERFACE spdlog::spdlog)

find_package(Threads REQUIRED)
target_link_libraries(luisa-compute-ext INTERFACE Threads::Threads)

add_subdirectory(xxHash/cmake_unofficial)
target_link_libraries(luisa-compute-ext INTERFACE xxHash::xxhash)

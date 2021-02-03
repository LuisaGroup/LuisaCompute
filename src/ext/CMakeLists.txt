add_library(luisa-compute-ext INTERFACE)

find_package(OpenImageIO REQUIRED)
target_link_libraries(luisa-compute-ext INTERFACE OpenImageIO::OpenImageIO)

find_package(OpenCV REQUIRED)
target_link_libraries(luisa-compute-ext INTERFACE ${OpenCV_LIBS})

add_subdirectory(cereal)
target_link_libraries(luisa-compute-ext INTERFACE cereal::cereal)

find_package(spdlog REQUIRED)
target_link_libraries(luisa-compute-ext INTERFACE spdlog::spdlog)



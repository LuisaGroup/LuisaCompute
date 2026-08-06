cmake_minimum_required(VERSION 3.26)

project(LuisaComputePackageConsumer LANGUAGES CXX)

find_package(LuisaCompute 0 CONFIG REQUIRED)

if (NOT TARGET luisa::compute)
    message(FATAL_ERROR "LuisaCompute package did not define luisa::compute")
endif ()

add_executable(luisa_compute_package_consumer main.cpp)
target_link_libraries(luisa_compute_package_consumer PRIVATE luisa::compute)

cmake_minimum_required(VERSION 3.26)

project(LuisaComputeSystemSpdlogProbe LANGUAGES CXX)

find_package(spdlog CONFIG REQUIRED)

add_executable(luisa-compute-system-spdlog-probe main.cpp)
target_compile_features(luisa-compute-system-spdlog-probe PRIVATE cxx_std_20)
target_link_libraries(luisa-compute-system-spdlog-probe PRIVATE
        spdlog::spdlog_header_only)

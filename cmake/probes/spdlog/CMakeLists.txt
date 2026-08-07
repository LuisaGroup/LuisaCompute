cmake_minimum_required(VERSION 3.26)

project(LuisaComputeSystemSpdlogProbe LANGUAGES CXX)

find_package(spdlog CONFIG QUIET)

get_filename_component(_luisa_compute_source_dir
        "${CMAKE_CURRENT_SOURCE_DIR}/../../.." ABSOLUTE)

add_executable(luisa-compute-system-spdlog-probe main.cpp)
target_compile_features(luisa-compute-system-spdlog-probe PRIVATE cxx_std_20)
if (TARGET spdlog::spdlog_header_only)
    target_compile_definitions(luisa-compute-system-spdlog-probe PRIVATE
            LUISA_USE_SYSTEM_SPDLOG=1
            LUISA_USE_SYSTEM_STL=1)
    target_include_directories(luisa-compute-system-spdlog-probe PRIVATE
            "${_luisa_compute_source_dir}/include"
            "${_luisa_compute_source_dir}/src/ext/half/include")
    target_link_libraries(luisa-compute-system-spdlog-probe PRIVATE
            spdlog::spdlog_header_only)
else ()
    # A missing package is a normal negative probe result. Keep the nested
    # project configurable so try_compile can return FALSE to the resolver
    # instead of aborting the parent configure.
    target_compile_definitions(luisa-compute-system-spdlog-probe PRIVATE
            LUISA_COMPUTE_SYSTEM_DEPENDENCY_MISSING=1)
endif ()

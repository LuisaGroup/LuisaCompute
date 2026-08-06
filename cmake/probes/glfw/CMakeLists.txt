cmake_minimum_required(VERSION 3.26)

project(LuisaComputeSystemGLFWProbe LANGUAGES CXX)

find_package(glfw3 CONFIG REQUIRED)

if (TARGET glfw)
    set(_luisa_compute_glfw_target glfw)
elseif (TARGET glfw3::glfw)
    set(_luisa_compute_glfw_target glfw3::glfw)
else ()
    message(FATAL_ERROR "glfw3 did not define a supported CMake target")
endif ()

add_executable(luisa-compute-system-glfw-probe main.cpp)
target_compile_features(luisa-compute-system-glfw-probe PRIVATE cxx_std_20)
target_link_libraries(luisa-compute-system-glfw-probe PRIVATE
        "${_luisa_compute_glfw_target}")

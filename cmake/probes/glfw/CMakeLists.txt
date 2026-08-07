cmake_minimum_required(VERSION 3.26)

project(LuisaComputeSystemGLFWProbe LANGUAGES CXX)

find_package(glfw3 CONFIG QUIET)

if (TARGET glfw)
    set(_luisa_compute_glfw_target glfw)
elseif (TARGET glfw3::glfw)
    set(_luisa_compute_glfw_target glfw3::glfw)
endif ()

add_executable(luisa-compute-system-glfw-probe main.cpp)
target_compile_features(luisa-compute-system-glfw-probe PRIVATE cxx_std_20)
if (_luisa_compute_glfw_target)
    target_link_libraries(luisa-compute-system-glfw-probe PRIVATE
            "${_luisa_compute_glfw_target}")
else ()
    # See the spdlog probe: absence should make try_compile return FALSE, not
    # make its nested configure fatal to the distribution build.
    target_compile_definitions(luisa-compute-system-glfw-probe PRIVATE
            LUISA_COMPUTE_SYSTEM_DEPENDENCY_MISSING=1)
endif ()

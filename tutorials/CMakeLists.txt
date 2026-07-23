# Tutorials for LuisaCompute
# 11 step-by-step tutorials aligned with the documentation structure.
# Each tutorial is in its own subdirectory with a main.cpp.

function(luisa_compute_add_tutorial name)
    luisa_compute_add_executable(${name} ${ARGN})
    target_include_directories(${name} PRIVATE ${CMAKE_SOURCE_DIR}/src/tests/ ${CMAKE_SOURCE_DIR}/src/tests/common/ ${CMAKE_SOURCE_DIR}/examples/)
endfunction()

if (NOT DEFINED LUISA_COMPUTE_TUTORIAL_TEST_BACKEND)
    if (LUISA_COMPUTE_ENABLE_FALLBACK)
        set(_luisa_compute_tutorial_test_backend fallback)
    elseif (LUISA_COMPUTE_ENABLE_VULKAN)
        set(_luisa_compute_tutorial_test_backend vk)
    elseif (LUISA_COMPUTE_ENABLE_DX)
        set(_luisa_compute_tutorial_test_backend dx)
    elseif (LUISA_COMPUTE_ENABLE_METAL)
        set(_luisa_compute_tutorial_test_backend metal)
    elseif (LUISA_COMPUTE_ENABLE_HIP)
        set(_luisa_compute_tutorial_test_backend hip)
    elseif (LUISA_COMPUTE_ENABLE_CUDA)
        set(_luisa_compute_tutorial_test_backend cuda)
    else ()
        set(_luisa_compute_tutorial_test_backend "")
    endif ()
    set(LUISA_COMPUTE_TUTORIAL_TEST_BACKEND "${_luisa_compute_tutorial_test_backend}"
            CACHE STRING "Backend used by offline tutorial CTests")
endif ()

# Helper: register a CTest target for a tutorial executable (offline mode).
function(luisa_compute_register_tutorial_ctest name)
    if (LUISA_COMPUTE_TUTORIAL_TEST_BACKEND STREQUAL "")
        return()
    endif ()
    add_test(NAME ${name}
             COMMAND ${name} ${LUISA_COMPUTE_TUTORIAL_TEST_BACKEND} --offline)
    set_tests_properties(${name} PROPERTIES LABELS "tutorial")
endfunction()

# --- Tutorials ---
luisa_compute_add_tutorial(tutorial_01_mandelbrot         01_mandelbrot/main.cpp)
luisa_compute_add_tutorial(tutorial_02_path_tracing       02_path_tracing/main.cpp)
if (LUISA_COMPUTE_ENABLE_GUI)
    luisa_compute_add_tutorial(tutorial_03_mpm_fluid      03_mpm_fluid/main.cpp)
    luisa_compute_add_tutorial(tutorial_06_image_processing   06_image_processing/main.cpp)
    luisa_compute_add_tutorial(tutorial_04_game_of_life    04_game_of_life/main.cpp)
    luisa_compute_add_tutorial(tutorial_05_wave_equation   05_wave_equation/main.cpp)
    luisa_compute_add_tutorial(tutorial_07_nbody           07_nbody/main.cpp)
    luisa_compute_add_tutorial(tutorial_08_fire_particles  08_fire_particles/main.cpp)
    luisa_compute_add_tutorial(tutorial_09_reaction_diffusion 09_reaction_diffusion/main.cpp)
    luisa_compute_add_tutorial(tutorial_10_voxel_raytracer 10_voxel_raytracer/main.cpp)
    luisa_compute_add_tutorial(tutorial_11_black_hole      11_black_hole/main.cpp)
endif ()

# --- CTest registrations (all run in offline mode) ---
luisa_compute_register_tutorial_ctest(tutorial_01_mandelbrot)
luisa_compute_register_tutorial_ctest(tutorial_02_path_tracing)
if (LUISA_COMPUTE_ENABLE_GUI)
    luisa_compute_register_tutorial_ctest(tutorial_03_mpm_fluid)
    luisa_compute_register_tutorial_ctest(tutorial_06_image_processing)
    luisa_compute_register_tutorial_ctest(tutorial_04_game_of_life)
    luisa_compute_register_tutorial_ctest(tutorial_05_wave_equation)
    luisa_compute_register_tutorial_ctest(tutorial_07_nbody)
    luisa_compute_register_tutorial_ctest(tutorial_08_fire_particles)
    luisa_compute_register_tutorial_ctest(tutorial_09_reaction_diffusion)
    luisa_compute_register_tutorial_ctest(tutorial_10_voxel_raytracer)
    luisa_compute_register_tutorial_ctest(tutorial_11_black_hole)
endif ()

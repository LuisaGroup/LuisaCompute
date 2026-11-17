# Regression fixtures intentionally live in this repository, not LuisaComputeGallery.
# Keep the historical test_* targets available when examples are moved or repackaged.
function(luisa_add_regression_fixture name)
    luisa_compute_add_executable(${name} ${ARGN})
    target_include_directories(${name} PRIVATE
        ${PROJECT_SOURCE_DIR}/src/tests ${PROJECT_SOURCE_DIR}/src/tests/common
        ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/common)
endfunction()

if (LUISA_COMPUTE_ENABLE_GUI)
    luisa_add_regression_fixture(test_path_tracing
        rendering/path_tracing.cpp
        rendering/path_tracing_test.h)
    luisa_add_regression_fixture(test_path_tracing_camera
        rendering/path_tracing_camera.cpp)
    luisa_add_regression_fixture(test_path_tracing_cutout
        rendering/path_tracing_cutout.cpp)
    luisa_add_regression_fixture(test_path_tracing_hdr
        rendering/path_tracing_hdr.cpp)
    luisa_add_regression_fixture(test_path_tracing_nested_callable
        rendering/path_tracing_nested_callable.cpp)
    luisa_add_regression_fixture(test_path_tracing_ray_masks
        rendering/path_tracing_ray_masks.cpp)
    luisa_add_regression_fixture(test_path_tracing_spectrum
        rendering/path_tracing_spectrum.cpp)
    luisa_add_regression_fixture(test_photon_mapping
        rendering/photon_mapping.cpp)
    luisa_add_regression_fixture(test_sdf_renderer
        rendering/sdf_renderer.cpp)
    luisa_add_regression_fixture(test_blackhole
        rendering/blackhole.cpp)
    luisa_add_regression_fixture(test_voxel_raytracer
        rendering/voxel_raytracer.cpp)
    luisa_add_regression_fixture(test_procedural
        rendering/procedural.cpp)
    luisa_add_regression_fixture(test_shader_toy_spacex
        rendering/shader_toy_spacex.cpp)
    luisa_add_regression_fixture(test_shader_visuals_present
        rendering/shader_visuals_present.cpp)
    luisa_add_regression_fixture(test_fire_simulation
        simulation/fire_simulation.cpp)
    luisa_add_regression_fixture(test_mpm3d
        simulation/mpm3d.cpp)
    luisa_add_regression_fixture(test_mpm88
        simulation/mpm88.cpp)
    luisa_add_regression_fixture(test_nbody_simulation
        simulation/nbody_simulation.cpp)
    luisa_add_regression_fixture(test_wave_equation
        simulation/wave_equation.cpp)
    if (TARGET luisa-compute-xir)
        luisa_add_regression_fixture(test_path_tracing_xir2ast
            rendering/path_tracing_xir2ast.cpp)
        luisa_add_regression_fixture(test_sdf_renderer_xir2ast
            rendering/sdf_renderer_xir2ast.cpp)
    endif ()
endif ()

luisa_add_regression_fixture(test_helloworld
    compute/helloworld.cpp)
luisa_add_regression_fixture(test_gdeflate
    compute/compact/main.cpp
    compute/compact/gdeflate.cpp)
luisa_add_regression_fixture(test_shader_toy
    rendering/shader_toy.cpp)
luisa_add_regression_fixture(test_image_processing
    compute/image_processing.cpp)
luisa_add_regression_fixture(test_game_of_life
    simulation/game_of_life.cpp)
# test_cluster_launch_control already has dedicated unit/runtime coverage.
luisa_add_regression_fixture(test_async_copy_prefetch
    compute/async_copy_prefetch.cpp)
luisa_add_regression_fixture(test_software_lbvh
    compute/software_lbvh.cpp
    compute/lbvh/lbvh_storage.cpp
    compute/lbvh/lbvh_sort.cpp
    compute/lbvh/blas.cpp
    compute/lbvh/tlas.cpp
    compute/lbvh/software_lbvh.cpp)
luisa_add_regression_fixture(test_software_lbvh_test
    compute/lbvh/test/lbvh_test.cpp
    compute/lbvh/lbvh_storage.cpp
    compute/lbvh/lbvh_sort.cpp
    compute/lbvh/blas.cpp
    compute/lbvh/tlas.cpp
    compute/lbvh/software_lbvh.cpp)
luisa_add_regression_fixture(test_multi_head_attention
    ml/multi_head_attention.cpp
    ml/attention_kernels.cpp
    ml/attention_host_data.cpp
    ml/attention_cpu_reference.cpp
    ml/attention_runner.cpp
    ml/paged_attention.cpp)

if (TARGET test_path_tracing_spectrum)
    set(_spectrum_cpp "${CMAKE_CURRENT_BINARY_DIR}/srgb_to_fourier_embedded.cpp")
    set(_spectrum_h "${CMAKE_CURRENT_BINARY_DIR}/srgb_to_fourier_embedded.h")
    add_custom_command(OUTPUT ${_spectrum_cpp} ${_spectrum_h}
        COMMAND $<TARGET_FILE:luisa_embed_device_lib>
            ${PROJECT_SOURCE_DIR}/src/tests/SRGBToFourierEvenPacked.dat
            --unsigned --prefix luisa_compute_ -o ${_spectrum_cpp} -h ${_spectrum_h}
        DEPENDS luisa_embed_device_lib ${PROJECT_SOURCE_DIR}/src/tests/SRGBToFourierEvenPacked.dat
        VERBATIM)
    target_sources(test_path_tracing_spectrum PRIVATE ${_spectrum_cpp} ${_spectrum_h})
    target_include_directories(test_path_tracing_spectrum PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
endif ()

if (LUISA_COMPUTE_ENABLE_GUI)
    # Finite, headless rendering coverage for the AIR-only Metal4 backend.
    if (TARGET luisa-compute-backend-metal4)
        function(luisa_metal4_add_render_test target)
            add_test(NAME metal4_air_render_${target}
                    COMMAND $<TARGET_FILE:${target}> metal4 --offline ${ARGN})
            set_tests_properties(metal4_air_render_${target} PROPERTIES
                    LABELS "integration;integration_metal4;rendering"
                    TIMEOUT 300)
        endfunction()

        luisa_metal4_add_render_test(test_path_tracing --spp 1)
        luisa_metal4_add_render_test(test_path_tracing_camera --spp 1)
        luisa_metal4_add_render_test(test_path_tracing_cutout --spp 1)
        luisa_metal4_add_render_test(test_path_tracing_hdr --spp 1)
        luisa_metal4_add_render_test(test_path_tracing_nested_callable --spp 1)
        luisa_metal4_add_render_test(test_path_tracing_ray_masks --spp 1)
        luisa_metal4_add_render_test(test_path_tracing_spectrum --spp 1)
        luisa_metal4_add_render_test(test_photon_mapping --spp 1)
        luisa_metal4_add_render_test(test_sdf_renderer --spp 1)
        luisa_metal4_add_render_test(test_blackhole)
        luisa_metal4_add_render_test(test_voxel_raytracer)
        luisa_metal4_add_render_test(test_procedural --spp 1)
        luisa_metal4_add_render_test(test_shader_toy)
        luisa_metal4_add_render_test(test_shader_toy_spacex)
        luisa_metal4_add_render_test(test_shader_visuals_present)
    endif ()

endif ()

if (TARGET luisa-compute-backend-simd)
    add_dependencies(test_helloworld luisa-compute-backend-simd)
    add_test(NAME simd_test_helloworld
            COMMAND $<TARGET_FILE:test_helloworld> simd)
    set_tests_properties(simd_test_helloworld PROPERTIES
            LABELS "integration_simd"
            TIMEOUT 30
            WORKING_DIRECTORY $<TARGET_FILE_DIR:test_helloworld>)

    foreach (_simd_graphics_example IN ITEMS
            image_processing
            shader_toy
            game_of_life
            fire_simulation)
        set(_simd_graphics_target
                test_${_simd_graphics_example})
        if (TARGET ${_simd_graphics_target})
            add_dependencies(${_simd_graphics_target}
                    luisa-compute-backend-simd)
            add_test(NAME simd_graphics_${_simd_graphics_example}
                    COMMAND $<TARGET_FILE:${_simd_graphics_target}>
                    simd --offline -c
                    ${CMAKE_SOURCE_DIR}/docs/gallery/test_${_simd_graphics_example}.png)
            set_tests_properties(
                    simd_graphics_${_simd_graphics_example} PROPERTIES
                    LABELS "integration_simd;graphics_simd"
                    TIMEOUT 120
                    WORKING_DIRECTORY
                    $<TARGET_FILE_DIR:${_simd_graphics_target}>)
        endif ()
    endforeach ()
endif ()


# coroutine (headless)
luisa_add_regression_fixture(test_coro_external_stage_debug
        coro/external_stage_debug.cpp)
luisa_add_regression_fixture(test_coro_external_stage_neural_sdf
        coro/external_stage_neural_sdf.cpp)
luisa_add_regression_fixture(test_coro_on_demand_texture
        coro/on_demand_texture.cpp)
if (TARGET luisa-compute-backend-metal)
    add_test(NAME metal_test_coro_external_stage_debug
            COMMAND $<TARGET_FILE:test_coro_external_stage_debug> metal)
    set_tests_properties(metal_test_coro_external_stage_debug PROPERTIES
            LABELS "integration_metal;coro_external_stage"
            TIMEOUT 120
            WORKING_DIRECTORY
            $<TARGET_FILE_DIR:test_coro_external_stage_debug>)
    add_test(NAME metal_test_coro_external_stage_neural_sdf
            COMMAND $<TARGET_FILE:test_coro_external_stage_neural_sdf>
            metal --test)
    set_tests_properties(
            metal_test_coro_external_stage_neural_sdf PROPERTIES
            LABELS "integration_metal;coro_external_stage;graphics_metal"
            TIMEOUT 180
            WORKING_DIRECTORY
            $<TARGET_FILE_DIR:test_coro_external_stage_neural_sdf>)
    add_test(NAME metal_test_coro_on_demand_texture
            COMMAND $<TARGET_FILE:test_coro_on_demand_texture>
            metal --test)
    set_tests_properties(
            metal_test_coro_on_demand_texture PROPERTIES
            LABELS "integration_metal;coro_external_stage;graphics_metal"
            TIMEOUT 120
            WORKING_DIRECTORY
            $<TARGET_FILE_DIR:test_coro_on_demand_texture>)
endif ()

if (TARGET luisa-compute-backend-metal4)
    add_test(NAME metal4_test_coro_on_demand_texture
            COMMAND $<TARGET_FILE:test_coro_on_demand_texture> metal4 --test)
    set_tests_properties(metal4_test_coro_on_demand_texture PROPERTIES
            LABELS "integration;integration_metal4;integration_coro"
            TIMEOUT 120
            WORKING_DIRECTORY $<TARGET_FILE_DIR:test_coro_on_demand_texture>)
endif ()

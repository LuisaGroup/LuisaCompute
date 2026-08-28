function(luisa_compute_add_example name)
    cmake_parse_arguments(LCE "MIRROR_AS_TEST" "" "" ${ARGN})
    set(srcs ${LCE_UNPARSED_ARGUMENTS})
    luisa_compute_add_executable(${name} ${srcs})
    target_include_directories(${name} PRIVATE ${CMAKE_SOURCE_DIR}/src/tests/ ${CMAKE_SOURCE_DIR}/src/tests/common/ ${CMAKE_SOURCE_DIR}/examples/)
    if (LCE_MIRROR_AS_TEST)
        string(REGEX REPLACE "^example_" "test_" mirror "${name}")
        if (NOT mirror STREQUAL name AND NOT TARGET ${mirror})
            luisa_compute_add_executable(${mirror} ${srcs})
            target_include_directories(${mirror} PRIVATE ${CMAKE_SOURCE_DIR}/src/tests/ ${CMAKE_SOURCE_DIR}/src/tests/common/ ${CMAKE_SOURCE_DIR}/examples/)
            set_target_properties(${name} PROPERTIES LUISA_TEST_MIRROR ${mirror})
        endif ()
    endif ()
endfunction()

function(luisa_example_pair_link name)
    target_link_libraries(${name} ${ARGN})
    get_target_property(mirror ${name} LUISA_TEST_MIRROR)
    if (mirror)
        target_link_libraries(${mirror} ${ARGN})
    endif ()
endfunction()

if (LUISA_COMPUTE_ENABLE_GUI)
    # rendering
    luisa_compute_add_example(example_path_tracing rendering/path_tracing.cpp MIRROR_AS_TEST)
    luisa_compute_add_example(example_path_tracing_camera rendering/path_tracing_camera.cpp MIRROR_AS_TEST)
    luisa_compute_add_example(example_path_tracing_cutout rendering/path_tracing_cutout.cpp MIRROR_AS_TEST)
    luisa_compute_add_example(example_path_tracing_hdr rendering/path_tracing_hdr.cpp MIRROR_AS_TEST)
    luisa_compute_add_example(example_path_tracing_nested_callable rendering/path_tracing_nested_callable.cpp MIRROR_AS_TEST)
    luisa_compute_add_example(example_path_tracing_ray_masks rendering/path_tracing_ray_masks.cpp MIRROR_AS_TEST)
    luisa_compute_add_example(example_path_tracing_spectrum rendering/path_tracing_spectrum.cpp MIRROR_AS_TEST)
    # embed SRGBToFourierEvenPacked.dat into the spectrum example/test
    set(_spectrum_dat "${CMAKE_SOURCE_DIR}/src/tests/SRGBToFourierEvenPacked.dat")
    set(_spectrum_embed_cpp "${CMAKE_CURRENT_LIST_DIR}/rendering/srgb_to_fourier_embedded.cpp")
    set(_spectrum_embed_h "${CMAKE_CURRENT_LIST_DIR}/rendering/srgb_to_fourier_embedded.h")
    add_custom_target(luisa-compute-example-spectrum-embed
            COMMAND $<TARGET_FILE:luisa_embed_device_lib> ${_spectrum_dat} --unsigned --prefix luisa_compute_ -o ${_spectrum_embed_cpp} -h ${_spectrum_embed_h}
            BYPRODUCTS ${_spectrum_embed_cpp} ${_spectrum_embed_h}
            DEPENDS luisa_embed_device_lib ${_spectrum_dat})
    add_dependencies(example_path_tracing_spectrum luisa-compute-example-spectrum-embed)
    target_sources(example_path_tracing_spectrum PRIVATE ${_spectrum_embed_cpp} ${_spectrum_embed_h})
    get_target_property(_spectrum_mirror example_path_tracing_spectrum LUISA_TEST_MIRROR)
    if (_spectrum_mirror)
        add_dependencies(${_spectrum_mirror} luisa-compute-example-spectrum-embed)
        target_sources(${_spectrum_mirror} PRIVATE ${_spectrum_embed_cpp} ${_spectrum_embed_h})
    endif ()
    luisa_compute_add_example(example_photon_mapping rendering/photon_mapping.cpp MIRROR_AS_TEST)
    luisa_compute_add_example(example_sdf_renderer rendering/sdf_renderer.cpp MIRROR_AS_TEST)
    luisa_compute_add_example(example_blackhole rendering/blackhole.cpp MIRROR_AS_TEST)
    luisa_compute_add_example(example_voxel_raytracer rendering/voxel_raytracer.cpp MIRROR_AS_TEST)
    luisa_compute_add_example(example_procedural rendering/procedural.cpp MIRROR_AS_TEST)
    luisa_compute_add_example(example_shader_toy_spacex rendering/shader_toy_spacex.cpp MIRROR_AS_TEST)
    luisa_compute_add_example(example_shader_visuals_present rendering/shader_visuals_present.cpp MIRROR_AS_TEST)

    if (TARGET luisa-compute-xir)
        luisa_compute_add_example(example_path_tracing_xir2ast rendering/path_tracing_xir2ast.cpp MIRROR_AS_TEST)
        luisa_compute_add_example(example_sdf_renderer_xir2ast rendering/sdf_renderer_xir2ast.cpp MIRROR_AS_TEST)
    endif ()

    # simulation
    luisa_compute_add_example(example_fire_simulation simulation/fire_simulation.cpp MIRROR_AS_TEST)
    luisa_compute_add_example(example_mpm3d simulation/mpm3d.cpp MIRROR_AS_TEST)
    luisa_compute_add_example(example_mpm88 simulation/mpm88.cpp MIRROR_AS_TEST)
    luisa_compute_add_example(example_nbody_simulation simulation/nbody_simulation.cpp MIRROR_AS_TEST)
    luisa_compute_add_example(example_wave_equation simulation/wave_equation.cpp MIRROR_AS_TEST)

    # gui
    luisa_compute_add_example(example_imgui gui/imgui.cpp)
    # mnist requires CUDA + DX/VK interop and is built as a shared library, not a standalone executable
    if (LUISA_COMPUTE_ENABLE_CUDA)
        add_library(example_mnist SHARED gui/mnist.cpp)
        target_link_libraries(example_mnist PRIVATE luisa-compute)
        target_include_directories(example_mnist PRIVATE ${CMAKE_SOURCE_DIR}/src/tests/ ${CMAKE_SOURCE_DIR}/src/tests/common/ ${CMAKE_SOURCE_DIR}/examples/)
        set_target_properties(example_mnist PROPERTIES EXCLUDE_FROM_ALL TRUE)
    endif ()
    luisa_compute_add_example(example_swapchain gui/swapchain.cpp)
    luisa_compute_add_example(example_swapchain_static gui/swapchain_static.cpp)
    if (WIN32 AND LUISA_COMPUTE_ENABLE_DX)
        luisa_compute_add_example(example_win_hdr gui/win_hdr.cpp)
    endif ()

endif ()

# compute (headless)
luisa_compute_add_example(example_bindless_mip compute/bindless_mip.cpp)
luisa_compute_add_example(example_coro_sdf_renderer rendering/coro_sdf_renderer.cpp)
luisa_compute_add_example(example_helloworld compute/helloworld.cpp MIRROR_AS_TEST)
luisa_compute_add_example(example_gdeflate compute/compact/main.cpp compute/compact/gdeflate.cpp MIRROR_AS_TEST)
luisa_compute_add_example(example_tensor_stub tensor/main.cpp)
# collect all tensor kernel source files
file(GLOB _tensor_kernels CONFIGURE_DEPENDS tensor/kernel_*.cpp)
# tile-language split runners: TinyCNN inference (--cnn) and polynomial-fit
# training (--poly-fit); keep in sync with examples/xmake.lua.
target_sources(example_tensor_stub PRIVATE
 ${_tensor_kernels}
 tensor/cnn_kernels.cpp
 tensor/cnn_inference.cpp
 tensor/poly_fit_kernels.cpp
 tensor/poly_fit.cpp
 tensor/linear_regression_kernels.cpp
 tensor/linear_regression.cpp
 tensor/mlp.cpp
 tensor/mnist.cpp
 tensor/rnn.cpp
 tensor/tensor_basics.cpp)
luisa_compute_add_example(example_shader_toy rendering/shader_toy.cpp MIRROR_AS_TEST)
luisa_compute_add_example(example_image_processing compute/image_processing.cpp MIRROR_AS_TEST)
luisa_compute_add_example(example_game_of_life simulation/game_of_life.cpp MIRROR_AS_TEST)
luisa_compute_add_example(example_cluster_launch_control compute/cluster_launch_control.cpp MIRROR_AS_TEST)
luisa_compute_add_example(example_async_copy_prefetch compute/async_copy_prefetch.cpp MIRROR_AS_TEST)
luisa_compute_add_example(example_multi_head_attention
        ml/multi_head_attention.cpp
        ml/attention_kernels.cpp
        ml/attention_host_data.cpp
        ml/attention_cpu_reference.cpp
        ml/attention_runner.cpp
        MIRROR_AS_TEST)

# coroutine (headless)
luisa_compute_add_example(example_coro_external_stage_debug
        coro/external_stage_debug.cpp)
luisa_compute_add_example(example_coro_external_stage_neural_sdf
        coro/external_stage_neural_sdf.cpp)
if (LUISA_COMPUTE_BUILD_TESTS AND TARGET luisa-compute-backend-metal)
    add_test(NAME metal_example_coro_external_stage_debug
            COMMAND $<TARGET_FILE:example_coro_external_stage_debug> metal)
    set_tests_properties(metal_example_coro_external_stage_debug PROPERTIES
            LABELS "integration_metal;coro_external_stage"
            TIMEOUT 120
            WORKING_DIRECTORY
            $<TARGET_FILE_DIR:example_coro_external_stage_debug>)
    add_test(NAME metal_example_coro_external_stage_neural_sdf
            COMMAND $<TARGET_FILE:example_coro_external_stage_neural_sdf>
            metal --test)
    set_tests_properties(
            metal_example_coro_external_stage_neural_sdf PROPERTIES
            LABELS "integration_metal;coro_external_stage;graphics_metal"
            TIMEOUT 180
            WORKING_DIRECTORY
            $<TARGET_FILE_DIR:example_coro_external_stage_neural_sdf>)
endif ()
luisa_compute_add_example(example_coro_path_tracing rendering/coro_path_tracing.cpp)
luisa_compute_add_example(example_coro_path_tracing_wavefront rendering/coro_path_tracing.cpp)
target_compile_definitions(example_coro_path_tracing_wavefront
        PRIVATE LUISA_CORO_PATH_TRACING_SAMPLE_DISPATCH_DEFAULT=1)

# --- GUI toolkit integration examples (wxWidgets, Qt) ---

# =============================================================================
# Extension examples (backend-specific device extensions)
# =============================================================================

if (LUISA_COMPUTE_ENABLE_CUDA AND LUISA_COMPUTE_ENABLE_CUDA_EXT_LCUB)
    luisa_compute_add_example(example_cuda_lcub extension/cuda_lcub.cpp)
    luisa_example_pair_link(example_cuda_lcub PRIVATE CUDA::cudart CUDA::cuda_driver)
endif ()

if (LUISA_COMPUTE_ENABLE_DX)
    luisa_compute_add_example(example_dml extension/dml.cpp)
    luisa_compute_add_example(example_dstorage extension/dstorage.cpp)
    luisa_compute_add_example(example_dx_supersampling extension/dx_supersampling.cpp)
    luisa_compute_add_example(example_fsr3 extension/fsr3.cpp)
    luisa_compute_add_example(example_supersampling extension/supersampling.cpp)
    luisa_compute_add_example(example_dx_custom_sdk dx/custom_sdk/custom_sdk.cpp)
endif ()

if (LUISA_COMPUTE_ENABLE_CLANG_CXX)
    luisa_compute_add_example(example_path_tracing_clangcxx extension/path_tracing_clangcxx.cpp)
    # clangcxx_compiler is xmake-only; no CMake target available
endif ()

# =============================================================================
# Interop examples (cross-backend interoperability)
# =============================================================================

if (LUISA_COMPUTE_ENABLE_CUDA AND LUISA_COMPUTE_ENABLE_DX)
    luisa_compute_add_example(example_cuda_dx_interop interop/cuda_dx_interop.cpp)
endif ()

if (LUISA_COMPUTE_ENABLE_CUDA AND LUISA_COMPUTE_ENABLE_VULKAN)
    luisa_compute_add_example(example_cuda_vk_interop interop/cuda_vk_interop.cpp)
endif ()

# =============================================================================
# GUI toolkit integration examples (wxWidgets, Qt)
# =============================================================================

option(LUISA_COMPUTE_EXAMPLE_WX_INTEROP "Build wxWidgets-interoperability examples." ON)
if (LUISA_COMPUTE_ENABLE_GUI AND LUISA_COMPUTE_EXAMPLE_WX_INTEROP)
    find_package(wxWidgets COMPONENTS core base)
    if (wxWidgets_FOUND)
        include(${wxWidgets_USE_FILE})
        message(STATUS "Found wxWidgets: ${wxWidgets_INCLUDE_DIRS}")

        if (UNIX AND NOT APPLE)
            foreach (lib ${wxWidgets_LIBRARIES})
                if (lib MATCHES ".*wx_gtk4.*")
                    set(LUISA_REQUIRED_GTK_LIB gtk4)
                elseif (lib MATCHES ".*wx_gtk3.*")
                    set(LUISA_REQUIRED_GTK_LIB gtk+-3.0)
                elseif (lib MATCHES ".*wx_gtk2.*")
                    set(LUISA_REQUIRED_GTK_LIB gtk+-2.0)
                endif ()
            endforeach ()
        endif ()

        if (LUISA_REQUIRED_GTK_LIB)
            find_package(PkgConfig)
            if (PkgConfig_FOUND)
                pkg_check_modules(GTK ${LUISA_REQUIRED_GTK_LIB})
            endif ()
            if (GTK_FOUND)
                message(STATUS "Found GTK: ${GTK_INCLUDE_DIRS}")
            endif ()
        endif ()

        if (NOT LUISA_REQUIRED_GTK_LIB OR GTK_FOUND)
            if (NOT TARGET luisa-compute-gui-wx)
                add_library(luisa-compute-gui-wx INTERFACE)
                target_include_directories(luisa-compute-gui-wx INTERFACE
                        ${GTK_INCLUDE_DIRS}
                        ${wxWidgets_INCLUDE_DIRS})
                target_link_libraries(luisa-compute-gui-wx INTERFACE
                        ${GTK_LIBRARIES}
                        ${wxWidgets_LIBRARIES})
            endif ()
            luisa_compute_add_example(example_swapchain_wx gui/swapchain_wx.cpp)
            luisa_example_pair_link(example_swapchain_wx PRIVATE luisa-compute-gui-wx)
        endif ()
    endif ()
endif ()

option(LUISA_COMPUTE_EXAMPLE_QT_INTEROP "Build Qt-interoperability examples." ON)
if (LUISA_COMPUTE_ENABLE_GUI AND LUISA_COMPUTE_EXAMPLE_QT_INTEROP)
    if (WIN32 AND NOT MSVC AND CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        message(WARNING "Clang has trouble with Qt on Windows. The Qt example will not be built.")
    else ()
        find_package(Qt6 COMPONENTS Widgets Core Gui QUIET CONFIG)
        if (Qt6_FOUND AND APPLE)
            try_compile(LUISA_COMPUTE_HAS_AGL
                    SOURCE_FROM_CONTENT "try_agl.cpp" "int main() {}"
                    LINK_LIBRARIES "-framework AGL")
            if (NOT LUISA_COMPUTE_HAS_AGL)
                message(WARNING "Found Qt6 but could not find AGL framework. Qt GUI examples will not be built.")
                set(Qt6_FOUND OFF)
            endif ()
        endif ()
        if (Qt6_FOUND AND TARGET Qt::Widgets AND TARGET Qt::Core AND TARGET Qt::Gui AND TARGET Qt::GuiPrivate)
            luisa_compute_add_example(example_swapchain_qt gui/swapchain_qt.cpp)
            luisa_example_pair_link(example_swapchain_qt PRIVATE Qt::Widgets Qt::Core Qt::Gui Qt::GuiPrivate)
        endif ()
    endif ()
endif ()

if (TARGET luisa-compute-backend-simd)
    add_dependencies(example_helloworld luisa-compute-backend-simd)
    add_test(NAME simd_example_helloworld
            COMMAND $<TARGET_FILE:example_helloworld> simd)
    set_tests_properties(simd_example_helloworld PROPERTIES
            LABELS "integration_simd"
            TIMEOUT 30
            WORKING_DIRECTORY $<TARGET_FILE_DIR:example_helloworld>)

    foreach (_simd_graphics_example IN ITEMS
            image_processing
            shader_toy
            game_of_life
            fire_simulation)
        set(_simd_graphics_target
                example_${_simd_graphics_example})
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

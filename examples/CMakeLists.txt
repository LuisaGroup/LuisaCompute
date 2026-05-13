function(luisa_compute_add_example name)
    luisa_compute_add_executable(${name} ${ARGN})
    target_include_directories(${name} PRIVATE ${CMAKE_SOURCE_DIR}/src/tests/ ${CMAKE_SOURCE_DIR}/examples/)
endfunction()

if (LUISA_COMPUTE_ENABLE_GUI)
    # rendering
    luisa_compute_add_example(example_path_tracing rendering/path_tracing.cpp)
    luisa_compute_add_example(example_path_tracing_camera rendering/path_tracing_camera.cpp)
    luisa_compute_add_example(example_path_tracing_cutout rendering/path_tracing_cutout.cpp)
    luisa_compute_add_example(example_path_tracing_hdr rendering/path_tracing_hdr.cpp)
    luisa_compute_add_example(example_path_tracing_nested_callable rendering/path_tracing_nested_callable.cpp)
    luisa_compute_add_example(example_path_tracing_ray_masks rendering/path_tracing_ray_masks.cpp)
    luisa_compute_add_example(example_path_tracing_spectrum rendering/path_tracing_spectrum.cpp)
    luisa_compute_add_example(example_photon_mapping rendering/photon_mapping.cpp)
    luisa_compute_add_example(example_sdf_renderer rendering/sdf_renderer.cpp)
    luisa_compute_add_example(example_blackhole rendering/blackhole.cpp)
    luisa_compute_add_example(example_voxel_raytracer rendering/voxel_raytracer.cpp)
    luisa_compute_add_example(example_procedural rendering/procedural.cpp)
    luisa_compute_add_example(example_shader_toy rendering/shader_toy.cpp)
    luisa_compute_add_example(example_shader_toy_spacex rendering/shader_toy_spacex.cpp)
    luisa_compute_add_example(example_shader_visuals_present rendering/shader_visuals_present.cpp)

    if (LUISA_COMPUTE_ENABLE_RUST)
        luisa_compute_add_example(example_path_tracing_ir rendering/path_tracing_ir.cpp)
        luisa_compute_add_example(example_sdf_renderer_ir rendering/sdf_renderer_ir.cpp)
    endif ()

    # simulation
    luisa_compute_add_example(example_fire_simulation simulation/fire_simulation.cpp)
    luisa_compute_add_example(example_game_of_life simulation/game_of_life.cpp)
    luisa_compute_add_example(example_mpm3d simulation/mpm3d.cpp)
    luisa_compute_add_example(example_mpm88 simulation/mpm88.cpp)
    luisa_compute_add_example(example_nbody_simulation simulation/nbody_simulation.cpp)
    luisa_compute_add_example(example_wave_equation simulation/wave_equation.cpp)

    # gui
    luisa_compute_add_example(example_imgui gui/imgui.cpp)
    # mnist requires CUDA + DX/VK interop and is built as a shared library, not a standalone executable
    if (LUISA_COMPUTE_ENABLE_CUDA)
        add_library(example_mnist SHARED gui/mnist.cpp)
        target_link_libraries(example_mnist PRIVATE luisa-compute)
        target_include_directories(example_mnist PRIVATE ${CMAKE_SOURCE_DIR}/src/tests/ ${CMAKE_SOURCE_DIR}/examples/)
        set_target_properties(example_mnist PROPERTIES EXCLUDE_FROM_ALL TRUE)
    endif ()
    luisa_compute_add_example(example_swapchain gui/swapchain.cpp)
    luisa_compute_add_example(example_swapchain_static gui/swapchain_static.cpp)
    luisa_compute_add_example(example_win_hdr gui/win_hdr.cpp)

    # compute (GUI-dependent)
    luisa_compute_add_example(example_image_processing compute/image_processing.cpp)
endif ()

# compute (headless)
luisa_compute_add_example(example_helloworld compute/helloworld.cpp)

# --- GUI toolkit integration examples (wxWidgets, Qt) ---

# =============================================================================
# Extension examples (backend-specific device extensions)
# =============================================================================

if (LUISA_COMPUTE_ENABLE_CUDA AND LUISA_COMPUTE_ENABLE_CUDA_EXT_LCUB)
    luisa_compute_add_example(example_cuda_lcub extension/cuda_lcub.cpp)
    target_link_libraries(example_cuda_lcub PRIVATE CUDA::cudart CUDA::cuda_driver)
endif ()

if (LUISA_COMPUTE_ENABLE_DX)
    luisa_compute_add_example(example_dml extension/dml.cpp)
    luisa_compute_add_example(example_dstorage extension/dstorage.cpp)
    luisa_compute_add_example(example_dx_supersampling extension/dx_supersampling.cpp)
    luisa_compute_add_example(example_fsr3 extension/fsr3.cpp)
    luisa_compute_add_example(example_supersampling extension/supersampling.cpp)
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
            add_library(luisa-compute-gui-wx INTERFACE)
            target_include_directories(luisa-compute-gui-wx INTERFACE
                    ${GTK_INCLUDE_DIRS}
                    ${wxWidgets_INCLUDE_DIRS})
            target_link_libraries(luisa-compute-gui-wx INTERFACE
                    ${GTK_LIBRARIES}
                    ${wxWidgets_LIBRARIES})
            luisa_compute_add_example(example_swapchain_wx gui/swapchain_wx.cpp)
            target_link_libraries(example_swapchain_wx PRIVATE luisa-compute-gui-wx)
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
            target_link_libraries(example_swapchain_qt PRIVATE Qt::Widgets Qt::Core Qt::Gui Qt::GuiPrivate)
        endif ()
    endif ()
endif ()

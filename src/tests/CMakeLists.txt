set(test_common_source
        common/test_main.cpp
        common/config.h
        common/doctest.h
        common/cornell_box.h
        common/tiny_obj_loader.h
        common/tinyexr.h)

function(luisa_compute_test_suite name)
    file(GLOB_RECURSE test_${name}_source next/test/${name}/**.cpp)
    luisa_compute_add_executable(test_${name} ${test_common_source} ${test_${name}_source})
    target_include_directories(test_${name} PRIVATE ./)
    target_precompile_headers(test_${name} PRIVATE pch.h)
endfunction()

luisa_compute_test_suite(feat)

#file(GLOB_RECURSE test_all_source next/test/**.cpp)
#luisa_compute_add_executable(test_all ${test_common_source} ${test_all_source})
#target_include_directories(test_all PRIVATE ./)

luisa_compute_add_executable(test_main
        common/test_main.cpp
        common/config.h
        common/doctest.h
        common/cornell_box.h
        common/tiny_obj_loader.h
        common/tinyexr.h)

function(luisa_compute_add_test name)
    target_sources(test_main PRIVATE test_${name}.cpp)
endfunction()

luisa_compute_add_test(context)
luisa_compute_add_test(dsl)
luisa_compute_add_test(buffer_io)

if (LUISA_COMPUTE_ENABLE_RUST)
    luisa_compute_add_executable(test_autodiff test_autodiff.cpp)
    luisa_compute_add_executable(test_autodiff_full test_autodiff_full.cpp)
    luisa_compute_add_executable(test_ast2ir test_ast2ir.cpp)
    luisa_compute_add_executable(test_ast2ir_headless test_ast2ir_headless.cpp)
    luisa_compute_add_executable(test_ast2ir_ir2ast test_ast2ir_ir2ast.cpp)

    if (LUISA_COMPUTE_ENABLE_GUI)
        luisa_compute_add_executable(test_kernel_ir test_kernel_ir.cpp)
        luisa_compute_add_executable(test_sdf_renderer_ir test_sdf_renderer_ir.cpp)
        luisa_compute_add_executable(test_path_tracing_ir test_path_tracing_ir.cpp)
    endif ()
endif ()
luisa_compute_add_executable(test_ir_v2 test_ir_v2.cpp)
luisa_compute_add_executable(test_helloworld test_helloworld.cpp)
luisa_compute_add_executable(test_type test_type.cpp)
luisa_compute_add_executable(test_ast test_ast.cpp)
luisa_compute_add_executable(test_binding_group test_binding_group.cpp)
luisa_compute_add_executable(test_binding_group_template test_binding_group_template.cpp)
luisa_compute_add_executable(test_copy test_copy.cpp)
luisa_compute_add_executable(test_constant test_constant.cpp)
luisa_compute_add_executable(test_cpu_callable test_cpu_callable.cpp)
luisa_compute_add_executable(test_dsl_multithread test_dsl_multithread.cpp)
luisa_compute_add_executable(test_dsl_sugar test_dsl_sugar.cpp)
luisa_compute_add_executable(test_dstorage test_dstorage.cpp)
luisa_compute_add_executable(test_dstorage_decompression test_dstorage_decompression.cpp)
luisa_compute_add_executable(test_indirect test_indirect.cpp)
luisa_compute_add_executable(test_present test_present.cpp)
luisa_compute_add_executable(test_indirect_rtx test_indirect_rtx.cpp)
luisa_compute_add_executable(test_runtime test_runtime.cpp)
luisa_compute_add_executable(test_printer test_printer.cpp)
luisa_compute_add_executable(test_printer_custom_callback test_printer_custom_callback.cpp)
luisa_compute_add_executable(test_callable test_callable.cpp)
luisa_compute_add_executable(test_texture_io test_texture_io.cpp)
luisa_compute_add_executable(test_texture_compress test_texture_compress.cpp)
luisa_compute_add_executable(test_atomic test_atomic.cpp)
luisa_compute_add_executable(test_atomic_queue test_atomic_queue.cpp)
luisa_compute_add_executable(test_shared_memory test_shared_memory.cpp)
luisa_compute_add_executable(test_bindless test_bindless.cpp)
luisa_compute_add_executable(test_sampler test_sampler.cpp)
luisa_compute_add_executable(test_bindless_buffer test_bindless_buffer.cpp)
luisa_compute_add_executable(test_rtx test_rtx.cpp)
luisa_compute_add_executable(test_sdf_renderer test_sdf_renderer.cpp)
luisa_compute_add_executable(test_procedural test_procedural.cpp)
luisa_compute_add_executable(test_procedural_callable test_procedural_callable.cpp)
luisa_compute_add_executable(test_mipmap test_mipmap.cpp)
luisa_compute_add_executable(test_native_include test_native_include.cpp)
luisa_compute_add_executable(test_select_device test_select_device.cpp)
luisa_compute_add_executable(test_sparse_texture test_sparse_texture.cpp)
luisa_compute_add_executable(test_soa test_soa.cpp)
luisa_compute_add_executable(test_soa_subview test_soa_subview.cpp)
luisa_compute_add_executable(test_soa_simple test_soa_simple.cpp)
luisa_compute_add_executable(test_raytracing_weekend test_raytracing_weekend/main.cpp)
luisa_compute_add_executable(test_dml test_dml.cpp)
luisa_compute_add_executable(test_oso_parser test_oso_parser.cpp)
luisa_compute_add_executable(test_cuda_dx_interop test_cuda_dx_interop.cpp)
luisa_compute_add_executable(test_pinned_mem test_pinned_mem.cpp)

if (LUISA_COMPUTE_ENABLE_CLANG_CXX)
    luisa_compute_add_executable(clangcxx_compiler clangcxx_compiler.cpp)
    luisa_compute_add_executable(test_clang_cxx test_clang_cxx.cpp)
    luisa_compute_add_executable(test_path_tracing_clangcxx test_path_tracing_clangcxx.cpp)
endif ()

if (LUISA_COMPUTE_ENABLE_GUI)
    luisa_compute_add_executable(test_swapchain test_swapchain.cpp)
    luisa_compute_add_executable(test_swapchain_static test_swapchain_static.cpp)
    luisa_compute_add_executable(test_game_of_life test_game_of_life.cpp)
    luisa_compute_add_executable(test_photon_mapping test_photon_mapping.cpp)
    luisa_compute_add_executable(test_raster test_raster.cpp)
    luisa_compute_add_executable(test_shader_toy test_shader_toy.cpp)
    luisa_compute_add_executable(test_path_tracing test_path_tracing.cpp)
    luisa_compute_add_executable(test_path_tracing_camera test_path_tracing_camera.cpp)
    luisa_compute_add_executable(test_path_tracing_nested_callable test_path_tracing_nested_callable.cpp)
    luisa_compute_add_executable(test_denoiser test_denoiser.cpp)
    luisa_compute_add_executable(test_path_tracing_cutout test_path_tracing_cutout.cpp)
    luisa_compute_add_executable(test_curve test_curve.cpp)
    luisa_compute_add_executable(test_curve_pbrt test_curve_pbrt.cpp)
    luisa_compute_add_executable(test_curve_pbrt_diffuse test_curve_pbrt_diffuse.cpp)
    luisa_compute_add_executable(test_normal_encoding test_normal_encoding.cpp)
    luisa_compute_add_executable(test_mpm88 test_mpm88.cpp)
    luisa_compute_add_executable(test_mpm3d test_mpm3d.cpp)
    luisa_compute_add_executable(test_shader_visuals_present test_shader_visuals_present.cpp)
    luisa_compute_add_executable(test_texture3d test_texture3d.cpp)
    luisa_compute_add_executable(test_imgui test_imgui.cpp)
    luisa_compute_add_executable(test_motion_blur test_motion_blur.cpp)
endif ()

if (LUISA_COMPUTE_ENABLE_CUDA AND LUISA_COMPUTE_ENABLE_CUDA_EXT_LCUB)
    find_package(CUDAToolkit QUIET)
    if (CUDAToolkit_FOUND)
        luisa_compute_add_executable(test_cuda_lcub test_cuda_lcub.cpp)
        target_link_libraries(test_cuda_lcub PUBLIC CUDA::cudart CUDA::cuda_driver)
    endif ()
endif ()

# GUI interaction with wxWidgets
option(LUISA_COMPUTE_TEST_WX_INTEROP "Build wxWidgets-interoperability tests." ON)
if (LUISA_COMPUTE_TEST_WX_INTEROP)
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

            # demos using wxWidgets
            luisa_compute_add_executable(test_swapchain_wx test_swapchain_wx.cpp)
            target_link_libraries(test_swapchain_wx PRIVATE luisa-compute-gui-wx)
        endif ()
    endif ()
endif ()

# GUI interaction with Qt
option(LUISA_COMPUTE_TEST_QT_INTEROP "Build Qt-interoperability tests." ON)
if (LUISA_COMPUTE_TEST_QT_INTEROP)
    if (WIN32 AND NOT MSVC AND CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        # FIXME: Qt adds the invalid flags -Zc:__cplusplus and -permissive- to clang
        message(WARNING "Clang has trouble with Qt on Windows. The Qt example will not be built.")
    else ()
        find_package(Qt6 COMPONENTS Widgets Core Gui QUIET CONFIG)
        if (Qt6_FOUND AND TARGET Qt::Widgets AND TARGET Qt::Core AND TARGET Qt::Gui)
            message(STATUS "Found Qt")
            luisa_compute_add_executable(test_swapchain_qt test_swapchain_qt.cpp)
            target_link_libraries(test_swapchain_qt PRIVATE Qt::Widgets Qt::Core Qt::Gui)
        endif ()
    endif ()
endif ()

local lc_enable_gui = has_config("lc_enable_gui")

local function example_proj(name, source, gui_dep, callable, project_kind)
    if gui_dep and not lc_enable_gui then return end
    target(name)
    add_deps("lc-backends-dummy", {inherit = false, links = false})
    _config_project({project_kind = project_kind or "binary"})
    add_files(source)
    add_includedirs("$(projectdir)/src/tests/", "$(projectdir)/src/tests/common/", "$(projectdir)/examples/")
    add_deps("lc-runtime", "lc-dsl", "lc-vstl", "stb-image")
    if lc_enable_gui then add_deps("lc-gui") end
    if gui_dep then add_defines("LUISA_ENABLE_GUI") end
    if callable then callable() end
    target_end()
end

-- rendering
example_proj("example_path_tracing", "rendering/path_tracing.cpp", true)
example_proj("example_path_tracing_camera", "rendering/path_tracing_camera.cpp", true)
example_proj("example_path_tracing_cutout", "rendering/path_tracing_cutout.cpp", true)
example_proj("example_path_tracing_hdr", "rendering/path_tracing_hdr.cpp", true)
example_proj("example_path_tracing_nested_callable", "rendering/path_tracing_nested_callable.cpp", true)
example_proj("example_path_tracing_ray_masks", "rendering/path_tracing_ray_masks.cpp", true)
example_proj("example_path_tracing_spectrum", "rendering/path_tracing_spectrum.cpp", true, function()
    add_rules("utils.bin2obj", {extensions = {".dat"}})
    add_files("$(projectdir)/src/tests/SRGBToFourierEvenPacked.dat")
    add_defines("LUISA_BIN_2_OBJ")
end)
example_proj("example_photon_mapping", "rendering/photon_mapping.cpp", true)
example_proj("example_sdf_renderer", "rendering/sdf_renderer.cpp", true)
example_proj("example_blackhole", "rendering/blackhole.cpp", true)
example_proj("example_voxel_raytracer", "rendering/voxel_raytracer.cpp", true)
example_proj("example_procedural", "rendering/procedural.cpp", true)
example_proj("example_shader_toy", "rendering/shader_toy.cpp", true)
example_proj("example_shader_toy_spacex", "rendering/shader_toy_spacex.cpp", true)
example_proj("example_shader_visuals_present", "rendering/shader_visuals_present.cpp", true)
if has_config("lc_enable_xir") then
    example_proj("example_path_tracing_xir2ast", "rendering/path_tracing_xir2ast.cpp", true)
    example_proj("example_sdf_renderer_xir2ast", "rendering/sdf_renderer_xir2ast.cpp", true)
end

-- simulation
example_proj("example_fire_simulation", "simulation/fire_simulation.cpp", true)
example_proj("example_game_of_life", "simulation/game_of_life.cpp", true)
example_proj("example_mpm3d", "simulation/mpm3d.cpp", true)
example_proj("example_mpm88", "simulation/mpm88.cpp", true)
example_proj("example_nbody_simulation", "simulation/nbody_simulation.cpp", true)
example_proj("example_wave_equation", "simulation/wave_equation.cpp", true)

-- gui
example_proj("example_imgui", "gui/imgui.cpp", true)
if has_config("lc_cuda_backend") then
    example_proj("example_mnist", "gui/mnist.cpp", true, nil, "shared")
end
example_proj("example_swapchain", "gui/swapchain.cpp", true)
example_proj("example_swapchain_static", "gui/swapchain_static.cpp", true)
example_proj("example_win_hdr", "gui/win_hdr.cpp", true)

    -- compute
    example_proj("example_helloworld", "compute/helloworld.cpp", false)
    example_proj("example_tensor_stub", "tensor/main.cpp", false, function()
        add_files("tensor/kernel_*.cpp")
        add_files("tensor/cnn_kernels.cpp", "tensor/cnn_inference.cpp")
    end)
example_proj("example_tile_bench", "compute/tile_bench.cpp", false)
    example_proj("example_cluster_launch_control", "compute/cluster_launch_control.cpp", false)
    example_proj("example_async_copy_prefetch", "compute/async_copy_prefetch.cpp", false)
    example_proj("example_image_processing", "compute/image_processing.cpp", true)
    if has_config("lc_enable_xir") then
        local function coro_example_proj(name, source, gui_dep, callable)
            example_proj(name, source, gui_dep, function()
                add_deps("lc-coro")
                if callable then callable() end
            end)
        end
        coro_example_proj("example_coro_sdf_renderer", "rendering/coro_sdf_renderer.cpp", false)
        coro_example_proj("example_coro_path_tracing", "rendering/coro_path_tracing.cpp", false)
        coro_example_proj("example_coro_path_tracing_wavefront", "rendering/coro_path_tracing.cpp", false, function()
            add_defines("LUISA_CORO_PATH_TRACING_SAMPLE_DISPATCH_DEFAULT=1")
        end)
    end
    example_proj("example_multi_head_attention", "ml/multi_head_attention.cpp", false, function()
        add_files("ml/attention_kernels.cpp", "ml/attention_host_data.cpp", "ml/attention_cpu_reference.cpp", "ml/attention_runner.cpp")
    end)
    includes("compute/tokenize")
    includes("compute/compact")

-- extension
if has_config("lc_dx_backend") then
    example_proj("example_dml", "extension/dml.cpp", false)
    example_proj("example_dstorage", "extension/dstorage.cpp", false)
    -- example_proj("example_dx_supersampling", "extension/dx_supersampling.cpp", true)
    example_proj("example_supersampling", "extension/supersampling.cpp", true)
end

if has_config("lc_cuda_ext_lcub") then
    example_proj("example_cuda_lcub", "extension/cuda_lcub.cpp", false, function()
        add_deps("lc-compute-cuda-ext-lcub")
    end)
end

if not is_mode("debug") and has_config("lc_enable_clangcxx") then
    example_proj("example_path_tracing_clangcxx", "extension/path_tracing_clangcxx.cpp", true, function()
        add_deps("lc-clangcxx")
    end)
    includes("extension/clangcxx_compiler")
end

-- interop
if has_config("lc_dx_cuda_interop") then
    example_proj("example_cuda_dx_interop", "interop/cuda_dx_interop.cpp", false)
end
if has_config("lc_vk_cuda_interop") then
    example_proj("example_cuda_vk_interop", "interop/cuda_vk_interop.cpp", false)
end

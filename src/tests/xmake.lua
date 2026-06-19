local lc_enable_gui = has_config("lc_enable_gui")

local function test_proj(name, source, gui_dep, callable, kind)
    if gui_dep and not lc_enable_gui then
        return
    end
    target(name)
    add_deps("lc-backends-dummy", {inherit = false, links = false})
    _config_project({project_kind = kind or "binary"})
    add_files(source)
    add_includedirs("./", "./common")
    add_deps("lc-runtime", "lc-dsl", "lc-vstl", "stb-image")
    if lc_enable_gui then
        add_deps("lc-gui")
    end
    if gui_dep then
        add_defines("LUISA_ENABLE_GUI")
    end
    if callable then
        callable()
    end
    target_end()
end

-- unit/core
test_proj("test_basic_traits", "unit/core/test_basic_traits.cpp")
test_proj("test_basic_types", "unit/core/test_basic_types.cpp")
test_proj("test_binary_file_stream", "unit/core/test_binary_file_stream.cpp")
test_proj("test_binary_io", "unit/core/test_binary_io.cpp")
test_proj("test_clock", "unit/core/test_clock.cpp")
test_proj("test_dynamic_module", "unit/core/test_dynamic_module.cpp")
test_proj("test_first_fit", "unit/core/test_first_fit.cpp")
test_proj("test_logging", "unit/core/test_logging.cpp")
test_proj("test_mathematics", "unit/core/test_mathematics.cpp")
test_proj("test_matrix", "unit/core/test_matrix.cpp")
test_proj("test_pool", "unit/core/test_pool.cpp")
test_proj("test_type", "unit/core/test_type.cpp")
if has_config('lc_vk_backend_use_xir_spirv') then
    test_proj("test_glslang_spirv", "unit/ext/test_glslang_spirv.cpp", false, function()
        add_deps("lc-glslang")
    end)
end
if has_config("lc_vk_backend") then
    test_proj("test_spirv_motion_patch", "unit/ext/test_spirv_motion_patch.cpp", false, function()
        add_files("../backends/vk/spirv_motion_patch.cpp")
        add_includedirs("../backends/vk")
    end)
end
test_proj("test_normal_encoding", "unit/dsl/test_normal_encoding.cpp", true)

-- unit/ast
test_proj("test_ast", "unit/ast/test_ast.cpp")
test_proj("test_ast_basic", "unit/ast/test_ast_basic.cpp")
test_proj("test_builtin_kernel", "unit/ast/test_builtin_kernel.cpp", false, function()
    add_includedirs("$(projectdir)/src/runtime")
end)
test_proj("test_manual_ast", "unit/ast/test_manual_ast.cpp")

-- unit/dsl
test_proj("test_binding_group", "unit/dsl/test_binding_group.cpp")
test_proj("test_binding_group_template", "unit/dsl/test_binding_group_template.cpp")
test_proj("test_callable", "unit/dsl/test_callable.cpp")
test_proj("test_constant", "unit/dsl/test_constant.cpp")
test_proj("test_dsl", "unit/dsl/test_dsl.cpp")
test_proj("test_dsl_sugar", "unit/dsl/test_dsl_sugar.cpp")
test_proj("test_dsl_multithread", "unit/dsl/test_dsl_multithread.cpp")
test_proj("test_soa", "unit/dsl/test_soa.cpp")
test_proj("test_soa_subview", "unit/dsl/test_soa_subview.cpp")
test_proj("test_soa_simple", "unit/dsl/test_soa_simple.cpp")
test_proj("test_device_math", "unit/dsl/test_device_math.cpp")
test_proj("test_nested_callable", "unit/dsl/test_nested_callable.cpp")
test_proj("test_calc", "unit/dsl/test_calc.cpp")
test_proj("test_dsl_matrix", "unit/dsl/test_matrix.cpp")
test_proj("test_var", "unit/dsl/test_var.cpp")
test_proj("test_8bit", "unit/dsl/test_8bit.cpp")

-- unit/runtime
test_proj("test_atomic", "unit/runtime/test_atomic.cpp")
test_proj("test_atomic_queue", "unit/runtime/test_atomic_queue.cpp")
test_proj("test_byte_buffer", "unit/runtime/test_byte_buffer.cpp")
test_proj("test_context", "unit/runtime/test_context.cpp")
test_proj("test_copy", "unit/runtime/test_copy.cpp")
test_proj("test_cpu_callable", "unit/runtime/test_cpu_callable.cpp")
test_proj("test_decoupled_look_back", "unit/runtime/test_decoupled_look_back.cpp")
test_proj("test_complex_kernel", "unit/runtime/test_complex_kernel.cpp")
test_proj("test_mipmap", "unit/runtime/test_mipmap.cpp")
test_proj("test_pinned_mem", "unit/runtime/test_pinned_mem.cpp")
test_proj("test_printer", "unit/runtime/test_printer.cpp")
test_proj("test_printer_custom_callback", "unit/runtime/test_printer_custom_callback.cpp")
test_proj("test_sampler", "unit/runtime/test_sampler.cpp")
test_proj("test_shared_memory", "unit/runtime/test_shared_memory.cpp")
test_proj("test_softmax", "unit/runtime/test_softmax.cpp")
-- test_tensor requires lc-tensor which is currently disabled
-- test_proj("test_tensor", "unit/runtime/test_tensor.cpp")
test_proj("test_texture_compress", "unit/runtime/test_texture_compress.cpp")
test_proj("test_texture_io", "unit/runtime/test_texture_io.cpp")
test_proj("test_warp", "unit/runtime/test_warp.cpp")
test_proj("test_warp_prefix_scan", "unit/runtime/test_warp_prefix_scan.cpp")
test_proj("test_mha_warp_reduction", "unit/runtime/test_mha_warp_reduction.cpp")
test_proj("test_buffer_io", "unit/runtime/test_buffer_io.cpp")
test_proj("test_buffer", "unit/runtime/test_buffer.cpp")
test_proj("test_out_of_range", "unit/runtime/test_out_of_range.cpp")
test_proj("test_buffer_view", "unit/runtime/test_buffer_view.cpp")
test_proj("test_device_test", "unit/runtime/test_device.cpp")
test_proj("test_external_buffer", "unit/runtime/test_external_buffer.cpp")
test_proj("test_gemm", "unit/runtime/test_gemm.cpp")
test_proj("test_shared_mem", "unit/runtime/test_shared_mem.cpp")

-- unit/coro
test_proj("test_coro_scheduler_base", "unit/coro/test_coro_scheduler_base.cpp")
test_proj("test_coro_multisplit", "unit/coro/test_coro_multisplit.cpp")
test_proj("test_coro_compaction", "unit/coro/test_coro_compaction.cpp")
test_proj("test_coro_radix_sort", "unit/coro/test_coro_radix_sort.cpp")

-- unit/xir
if has_config("lc_enable_xir") then
    local function coro_xir_test_proj(name, source)
        test_proj(name, source, false, function()
            add_defines("LUISA_ENABLE_XIR")
            add_deps("lc-coro")
        end)
    end
    test_proj("test_ast_to_xir", "unit/xir/test_ast_to_xir.cpp", false, function()
        add_defines("LUISA_ENABLE_XIR")
    end)
    test_proj("test_xir_passes", "unit/xir/test_xir_passes.cpp", false, function()
        add_defines("LUISA_ENABLE_XIR")
    end)
    test_proj("test_xir_pass_licm", "unit/xir/test_xir_pass_licm.cpp", false, function()
        add_defines("LUISA_ENABLE_XIR")
    end)
    test_proj("test_xir_pass_indvar_simplify", "unit/xir/test_xir_pass_indvar_simplify.cpp", false, function()
        add_defines("LUISA_ENABLE_XIR")
    end)
    test_proj("test_xir2ast_translators", "unit/xir/test_xir2ast_translators.cpp", false, function()
        add_defines("LUISA_ENABLE_XIR")
    end)
    test_proj("test_xir_pass_restructure_cfg", "unit/xir/test_xir_pass_restructure_cfg.cpp", false, function()
        add_defines("LUISA_ENABLE_XIR")
    end)
    coro_xir_test_proj("test_coro_graph", "unit/coro/test_coro_graph.cpp")
    coro_xir_test_proj("test_coro_state_machine", "unit/coro/test_coro_state_machine.cpp")
    coro_xir_test_proj("test_coro_persistent", "unit/coro/test_coro_persistent.cpp")
    coro_xir_test_proj("test_coro_persistent_opt", "unit/coro/test_coro_persistent_opt.cpp")
    coro_xir_test_proj("test_coro_persistent_integration", "unit/coro/test_coro_persistent_integration.cpp")
    coro_xir_test_proj("test_coro_soa_layout", "unit/coro/test_coro_soa_layout.cpp")
    coro_xir_test_proj("test_coro_wavefront", "unit/coro/test_coro_wavefront.cpp")
    coro_xir_test_proj("test_coro_all_schedulers", "unit/coro/test_coro_all_schedulers.cpp")
    coro_xir_test_proj("test_coro_wavefront_integration", "unit/coro/test_coro_wavefront_integration.cpp")
    coro_xir_test_proj("test_coro_pipeline_1suspend", "unit/coro/test_coro_pipeline_1suspend.cpp")
    coro_xir_test_proj("test_coro_pipeline_3suspend", "unit/coro/test_coro_pipeline_3suspend.cpp")
end

-- integration/runtime
test_proj("test_aot", "integration/runtime/test_aot.cpp", true)
test_proj("test_device_debugger", "integration/runtime/test_device_debugger.cpp")
test_proj("test_dstorage_decompression", "integration/runtime/test_dstorage_decompression.cpp", true)
test_proj("test_procedural_callable", "integration/runtime/test_procedural_callable.cpp")
test_proj("test_rtx", "integration/runtime/test_rtx.cpp")
if has_config("lc_enable_osl") then
    test_proj("test_oso_parser", "integration/runtime/test_oso_parser.cpp", false, function()
        add_deps("lc-osl")
    end)
end

test_proj("test_bindless", "integration/runtime/test_bindless.cpp", true)
test_proj("test_bindless_buffer", "integration/runtime/test_bindless_buffer.cpp", true)
test_proj("test_curve", "integration/runtime/test_curve.cpp", true)
test_proj("test_curve_pbrt", "integration/runtime/test_curve_pbrt.cpp", true)
test_proj("test_curve_pbrt_diffuse", "integration/runtime/test_curve_pbrt_diffuse.cpp", true)
test_proj("test_denoiser", "integration/runtime/test_denoiser.cpp", true)
test_proj("test_indirect", "integration/runtime/test_indirect.cpp", true)
test_proj("test_indirect_rtx", "integration/runtime/test_indirect_rtx.cpp", true)
test_proj("test_motion_blur", "integration/runtime/test_motion_blur.cpp", true)
test_proj("test_motion_blur_vk", "integration/runtime/test_motion_blur_vk.cpp", true)
test_proj("test_vk_spirv_codegen_path", "unit/runtime/test_vk_spirv_codegen_path.cpp")
test_proj("test_mesh", "integration/runtime/motion_blur/test_mesh.cpp", true)
test_proj("test_mesh_2", "integration/runtime/motion_blur/test_mesh_2.cpp", true)
test_proj("test_mesh_3", "integration/runtime/motion_blur/test_mesh_3.cpp", true)
test_proj("test_mesh_4", "integration/runtime/motion_blur/test_mesh_4.cpp", true)
test_proj("test_native_include", "integration/runtime/test_native_include.cpp", true)
test_proj("test_present", "integration/runtime/test_present.cpp", true)
test_proj("test_runtime", "integration/runtime/test_runtime.cpp", true)
test_proj("test_select_device", "integration/runtime/test_select_device.cpp", true)
test_proj("test_texture3d", "integration/runtime/test_texture3d.cpp", true)
test_proj("test_transient_resource", "integration/runtime/test_transient_resource.cpp", true, function()
    add_files("integration/runtime/transient_resource_device/*.cpp")
end)

-- integration/runtime: CUDA-only tests
if has_config("lc_cuda_backend") then
    test_proj("test_cuda_graph", "integration/runtime/test_cuda_graph.cpp")
end

-- integration/runtime: DX-only tests
if has_config("lc_dx_backend") then
    test_proj("test_raster", "integration/runtime/test_raster.cpp", true)
    test_proj("test_memory_compact", "integration/runtime/test_memory_compact.cpp", false, function()
        if has_config("lc_vk_backend") then
            add_deps("lc-volk")
        end
    end)
    -- test_work_graph disabled: work_graph headers not found
    -- test_proj("test_work_graph", "integration/runtime/test_work_graph.cpp")
end

-- integration/ir (depends on lc-ir which requires Rust/lc-rust)
if has_config("lc_enable_ir") then
    test_proj("test_autodiff", "integration/ir/test_autodiff.cpp")
    test_proj("test_autodiff_full", "integration/ir/test_autodiff_full.cpp")
    test_proj("test_ast2ir", "integration/ir/test_ast2ir.cpp")
    test_proj("test_ast2ir_headless", "integration/ir/test_ast2ir_headless.cpp")
    test_proj("test_ast2ir_ir2ast", "integration/ir/test_ast2ir_ir2ast.cpp")
    test_proj("test_kernel_ir", "integration/ir/test_kernel_ir.cpp", true)
end

if has_config("lc_enable_xir") then
    test_proj("test_xir2ast_roundtrip", "integration/ir/test_xir2ast_roundtrip.cpp", false, function()
        add_defines("LUISA_ENABLE_XIR")
    end)
end

-- unit/dsl: standalone GPU tests
test_proj("test_dsl_mathematic", "unit/dsl/test_dsl_mathematic.cpp")

-- unit/runtime: FP4/FP8 quantization tests
test_proj("test_fp4", "unit/runtime/test_fp4.cpp")
test_proj("test_fp4_quantization", "unit/runtime/test_fp4_quantization.cpp")
test_proj("test_fp8", "unit/runtime/test_fp8.cpp")
test_proj("test_fp8_quantization", "unit/runtime/test_fp8_quantization.cpp")

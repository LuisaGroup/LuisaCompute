# =============================================================================
# LuisaCompute Tests
# Organized into: unit tests, integration tests
# Uses Boost.UT (single-header) for test framework
# =============================================================================

# --- CTest integration ---
enable_testing()

# Helper: add a standalone test executable, include path, AND register with CTest.
# Usage: luisa_compute_add_test(name source LABELS label1;label2 [ARGS arg1 ...])
# Tests without LABELS compile but are NOT registered with CTest.
function(luisa_compute_add_test name source)
    cmake_parse_arguments(PARSE_ARGV 2 CT "" "" "LABELS;ARGS")
    luisa_compute_add_executable(${name} ${source})
    # Expose `./` so includes like "test_device.h" (now in common/) resolve via the search path
    # below, and "common/foo.h" continues to work. Add common/ explicitly so plain
    # #include "test_device.h" / "reference_image.h" / "cornell_box.h" all resolve.
    target_include_directories(${name} PRIVATE ./ ./common)
    if (CT_LABELS)
        add_test(NAME ${name} COMMAND ${name} ${CT_ARGS})
        set_tests_properties(${name} PROPERTIES LABELS "${CT_LABELS}")
    endif ()
endfunction()

# =============================================================================
# Unit Tests
# =============================================================================

# --- unit/core: standalone tests (no backend needed, auto-run via CTest) ---
luisa_compute_add_test(test_basic_traits       unit/core/test_basic_traits.cpp       LABELS "unit;unit_core")
luisa_compute_add_test(test_basic_types        unit/core/test_basic_types.cpp        LABELS "unit;unit_core")
luisa_compute_add_test(test_binary_file_stream unit/core/test_binary_file_stream.cpp LABELS "unit;unit_core")
luisa_compute_add_test(test_binary_io          unit/core/test_binary_io.cpp          LABELS "unit;unit_core")
luisa_compute_add_test(test_clock              unit/core/test_clock.cpp              LABELS "unit;unit_core")
luisa_compute_add_test(test_dynamic_module     unit/core/test_dynamic_module.cpp     LABELS "unit;unit_core")
luisa_compute_add_test(test_first_fit          unit/core/test_first_fit.cpp          LABELS "unit;unit_core")
luisa_compute_add_test(test_logging            unit/core/test_logging.cpp            LABELS "unit;unit_core")
luisa_compute_add_test(test_mathematics        unit/core/test_mathematics.cpp        LABELS "unit;unit_core")
luisa_compute_add_test(test_matrix             unit/core/test_matrix.cpp             LABELS "unit;unit_core")
luisa_compute_add_test(test_pool               unit/core/test_pool.cpp               LABELS "unit;unit_core")
luisa_compute_add_test(test_type               unit/core/test_type.cpp               LABELS "unit;unit_core")
luisa_compute_add_test(test_stl_containers     unit/core/test_stl_containers.cpp     LABELS "unit;unit_core")
luisa_compute_add_test(test_hash               unit/core/test_hash.cpp               LABELS "unit;unit_core")
luisa_compute_add_test(test_spin_mutex         unit/core/test_spin_mutex.cpp         LABELS "unit;unit_core")
luisa_compute_add_test(test_fiber              unit/core/test_fiber.cpp              LABELS "unit;unit_core")
luisa_compute_add_test(test_platform_utils     unit/core/test_platform.cpp           LABELS "unit;unit_core")
luisa_compute_add_test(test_glslang_spirv      unit/ext/test_glslang_spirv.cpp       LABELS "unit;unit_ext")
target_link_libraries(test_glslang_spirv PRIVATE glslang-default-resource-limits)

if (LUISA_COMPUTE_ENABLE_GUI)
endif ()

# --- unit/ast: standalone tests (need backend, NOT auto-run via CTest) ---
luisa_compute_add_test(test_ast             unit/ast/test_ast.cpp)
luisa_compute_add_test(test_ast_basic       unit/ast/test_ast_basic.cpp)
luisa_compute_add_test(test_builtin_kernel  unit/ast/test_builtin_kernel.cpp)
target_include_directories(test_builtin_kernel PRIVATE ${CMAKE_SOURCE_DIR}/src/runtime)
luisa_compute_add_test(test_manual_ast      unit/ast/test_manual_ast.cpp)

# --- unit/dsl: standalone tests (need backend, NOT auto-run via CTest) ---
luisa_compute_add_test(test_binding_group          unit/dsl/test_binding_group.cpp)
luisa_compute_add_test(test_binding_group_template unit/dsl/test_binding_group_template.cpp)
luisa_compute_add_test(test_callable               unit/dsl/test_callable.cpp)
luisa_compute_add_test(test_constant               unit/dsl/test_constant.cpp)
luisa_compute_add_test(test_dsl                    unit/dsl/test_dsl.cpp)
luisa_compute_add_test(test_dsl_sugar              unit/dsl/test_dsl_sugar.cpp)
luisa_compute_add_test(test_dsl_multithread        unit/dsl/test_dsl_multithread.cpp)
luisa_compute_add_test(test_soa                    unit/dsl/test_soa.cpp)
luisa_compute_add_test(test_soa_subview            unit/dsl/test_soa_subview.cpp)
luisa_compute_add_test(test_soa_simple             unit/dsl/test_soa_simple.cpp)
luisa_compute_add_test(test_device_math            unit/dsl/test_device_math.cpp)
luisa_compute_add_test(test_nested_callable        unit/dsl/test_nested_callable.cpp)
luisa_compute_add_test(test_normal_encoding        unit/dsl/test_normal_encoding.cpp)
luisa_compute_add_test(test_calc                   unit/dsl/test_calc.cpp)
luisa_compute_add_test(test_dsl_matrix             unit/dsl/test_matrix.cpp)
luisa_compute_add_test(test_var                    unit/dsl/test_var.cpp)
luisa_compute_add_test(test_8bit                   unit/dsl/test_8bit.cpp)
luisa_compute_add_test(test_polymorphic            unit/dsl/test_polymorphic.cpp)
luisa_compute_add_test(test_dsl_autodiff           unit/dsl/test_autodiff.cpp)

# --- unit/runtime: standalone tests (need backend, NOT auto-run via CTest) ---
luisa_compute_add_test(test_atomic              unit/runtime/test_atomic.cpp)
luisa_compute_add_test(test_atomic_queue        unit/runtime/test_atomic_queue.cpp)
luisa_compute_add_test(test_byte_buffer         unit/runtime/test_byte_buffer.cpp)
luisa_compute_add_test(test_context             unit/runtime/test_context.cpp)
luisa_compute_add_test(test_copy                unit/runtime/test_copy.cpp)
luisa_compute_add_test(test_cpu_callable        unit/runtime/test_cpu_callable.cpp)
luisa_compute_add_test(test_decoupled_look_back unit/runtime/test_decoupled_look_back.cpp)
luisa_compute_add_test(test_matrix_multiply     unit/runtime/test_matrix_multiply.cpp)
luisa_compute_add_test(test_mipmap              unit/runtime/test_mipmap.cpp)
luisa_compute_add_test(test_pinned_mem          unit/runtime/test_pinned_mem.cpp)
luisa_compute_add_test(test_printer             unit/runtime/test_printer.cpp)
luisa_compute_add_test(test_printer_custom_callback unit/runtime/test_printer_custom_callback.cpp)
luisa_compute_add_test(test_sampler             unit/runtime/test_sampler.cpp)
luisa_compute_add_test(test_shared_memory       unit/runtime/test_shared_memory.cpp)
luisa_compute_add_test(test_softmax             unit/runtime/test_softmax.cpp)
if (LUISA_COMPUTE_ENABLE_TENSOR AND FALSE) # disabled: link errors against missing Tensor symbols
    luisa_compute_add_test(test_tensor              unit/runtime/test_tensor.cpp)
endif ()
luisa_compute_add_test(test_texture_compress    unit/runtime/test_texture_compress.cpp)
luisa_compute_add_test(test_texture_io          unit/runtime/test_texture_io.cpp)
luisa_compute_add_test(test_warp                unit/runtime/test_warp.cpp)
luisa_compute_add_test(test_warp_prefix_scan    unit/runtime/test_warp_prefix_scan.cpp)
luisa_compute_add_test(test_buffer_io           unit/runtime/test_buffer_io.cpp)
luisa_compute_add_test(test_buffer              unit/runtime/test_buffer.cpp)
luisa_compute_add_test(test_buffer_view         unit/runtime/test_buffer_view.cpp)
luisa_compute_add_test(test_device_test         unit/runtime/test_device.cpp)
luisa_compute_add_test(test_external_buffer     unit/runtime/test_external_buffer.cpp)
luisa_compute_add_test(test_gemm                unit/runtime/test_gemm.cpp)
luisa_compute_add_test(test_shared_mem          unit/runtime/test_shared_mem.cpp)

# --- unit/xir: standalone tests (no backend needed, auto-run via CTest) ---
luisa_compute_add_test(test_ast_to_xir unit/xir/test_ast_to_xir.cpp LABELS "unit;unit_xir")
luisa_compute_add_test(test_xir_module      unit/xir/test_xir_module.cpp      LABELS "unit;unit_xir")
luisa_compute_add_test(test_xir_builder     unit/xir/test_xir_builder.cpp     LABELS "unit;unit_xir")
luisa_compute_add_test(test_xir_translators unit/xir/test_xir_translators.cpp LABELS "unit;unit_xir")
luisa_compute_add_test(test_xir_passes      unit/xir/test_xir_passes.cpp      LABELS "unit;unit_xir")
luisa_compute_add_test(test_xir_pass_lower_break_continue unit/xir/test_xir_pass_lower_break_continue.cpp LABELS "unit;unit_xir")
luisa_compute_add_test(test_xir_pass_destructure_cfg unit/xir/test_xir_pass_destructure_cfg.cpp LABELS "unit;unit_xir")
luisa_compute_add_test(test_xir_pass_simplify_cfg unit/xir/test_xir_pass_simplify_cfg.cpp LABELS "unit;unit_xir")
luisa_compute_add_test(test_xir_pass_restructure_cfg unit/xir/test_xir_pass_restructure_cfg.cpp LABELS "unit;unit_xir")
luisa_compute_add_test(test_xir_pass_early_return_elimination unit/xir/test_xir_pass_early_return_elimination.cpp LABELS "unit;unit_xir")

# =============================================================================
# Integration Tests (require GPU backends, NOT auto-run via CTest)
# Run manually: ./<test_name> <backend>
# =============================================================================

# --- integration/runtime: standalone tests ---
luisa_compute_add_test(test_device_debugger         integration/runtime/test_device_debugger.cpp)
luisa_compute_add_test(test_dstorage_decompression  integration/runtime/test_dstorage_decompression.cpp)
luisa_compute_add_test(test_procedural_callable      integration/runtime/test_procedural_callable.cpp)
luisa_compute_add_test(test_rtx                      integration/runtime/test_rtx.cpp)
luisa_compute_add_test(test_oso_parser               integration/runtime/test_oso_parser.cpp)

if (LUISA_COMPUTE_ENABLE_GUI)
    luisa_compute_add_test(test_bindless               integration/runtime/test_bindless.cpp)
    luisa_compute_add_test(test_bindless_buffer         integration/runtime/test_bindless_buffer.cpp)
    luisa_compute_add_test(test_curve                   integration/runtime/test_curve.cpp)
    luisa_compute_add_test(test_curve_pbrt              integration/runtime/test_curve_pbrt.cpp)
    luisa_compute_add_test(test_curve_pbrt_diffuse      integration/runtime/test_curve_pbrt_diffuse.cpp)
    luisa_compute_add_test(test_denoiser                integration/runtime/test_denoiser.cpp)
    luisa_compute_add_test(test_indirect                integration/runtime/test_indirect.cpp)
    luisa_compute_add_test(test_indirect_rtx            integration/runtime/test_indirect_rtx.cpp)
    luisa_compute_add_test(test_motion_blur             integration/runtime/test_motion_blur.cpp)
    luisa_compute_add_test(test_motion_blur_vk          integration/runtime/test_motion_blur_vk.cpp)
    luisa_compute_add_test(test_mesh                    integration/runtime/motion_blur/test_mesh.cpp)
    luisa_compute_add_test(test_mesh_2                  integration/runtime/motion_blur/test_mesh_2.cpp)
    luisa_compute_add_test(test_mesh_3                  integration/runtime/motion_blur/test_mesh_3.cpp)
    luisa_compute_add_test(test_mesh_4                  integration/runtime/motion_blur/test_mesh_4.cpp)
    luisa_compute_add_test(test_native_include          integration/runtime/test_native_include.cpp)
    luisa_compute_add_test(test_present                 integration/runtime/test_present.cpp)
    luisa_compute_add_test(test_runtime                 integration/runtime/test_runtime.cpp)
    luisa_compute_add_test(test_select_device           integration/runtime/test_select_device.cpp)
    luisa_compute_add_test(test_texture3d               integration/runtime/test_texture3d.cpp)

    luisa_compute_add_executable(test_transient_resource
            integration/runtime/test_transient_resource.cpp
            integration/runtime/transient_resource_device/managed_first_fit.cpp
            integration/runtime/transient_resource_device/managed_first_fit.h
            integration/runtime/transient_resource_device/transient_resource_device.cpp
            integration/runtime/transient_resource_device/transient_resource_device.h)
    target_include_directories(test_transient_resource PRIVATE ./ ./common)
endif ()

# --- integration/runtime: DX-only tests ---
if (LUISA_COMPUTE_ENABLE_DX)
    luisa_compute_add_test(test_raster         integration/runtime/test_raster.cpp         ARGS dx)
    luisa_compute_add_test(test_memory_compact  integration/runtime/test_memory_compact.cpp ARGS dx)
    # test_work_graph disabled: work_graph headers not yet available
    # luisa_compute_add_test(test_work_graph      integration/runtime/test_work_graph.cpp     ARGS dx)
endif ()

# --- integration/runtime: HIP/HIPRT tests (AMD GPU only) ---
if (LUISA_COMPUTE_ENABLE_HIP)
    luisa_compute_add_test(test_hiprt_standalone integration/runtime/test_hiprt_standalone.cpp)
endif ()

# --- integration/ir: standalone tests ---
if (LUISA_COMPUTE_ENABLE_RUST)
    luisa_compute_add_test(test_autodiff        integration/ir/test_autodiff.cpp)
    luisa_compute_add_test(test_autodiff_full    integration/ir/test_autodiff_full.cpp)
    luisa_compute_add_test(test_ast2ir           integration/ir/test_ast2ir.cpp)
    luisa_compute_add_test(test_ast2ir_headless  integration/ir/test_ast2ir_headless.cpp)
    luisa_compute_add_test(test_ast2ir_ir2ast    integration/ir/test_ast2ir_ir2ast.cpp)

    if (LUISA_COMPUTE_ENABLE_GUI)
        luisa_compute_add_test(test_kernel_ir integration/ir/test_kernel_ir.cpp)
    endif ()
endif ()

# --- unit/dsl: standalone GPU tests ---
luisa_compute_add_test(test_dsl_mathematic       unit/dsl/test_dsl_mathematic.cpp)

# --- unit/runtime: FP4/FP8 quantization tests ---
luisa_compute_add_test(test_fp4                  unit/runtime/test_fp4.cpp)
luisa_compute_add_test(test_fp4_quantization     unit/runtime/test_fp4_quantization.cpp)
luisa_compute_add_test(test_fp8                  unit/runtime/test_fp8.cpp)
luisa_compute_add_test(test_fp8_quantization     unit/runtime/test_fp8_quantization.cpp)


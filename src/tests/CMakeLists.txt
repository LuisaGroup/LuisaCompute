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

# --- unit/simd: dependency-light Schedule IR semantic model ---
# Keep this target independent from LLVM, Embree, and the runtime backends so
# divergent-control semantics can be tested in the smallest host build.
add_executable(test_simd_cohort_scheduler
        unit/simd/test_cohort_scheduler.cpp)
target_compile_features(test_simd_cohort_scheduler PRIVATE cxx_std_20)
target_include_directories(test_simd_cohort_scheduler PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../backends/simd/schedule)
add_test(NAME test_simd_cohort_scheduler
        COMMAND test_simd_cohort_scheduler)
set_tests_properties(test_simd_cohort_scheduler PROPERTIES
        LABELS "unit;unit_simd")

add_executable(test_simd_scheduler_model
        unit/simd/test_scheduler_model.cpp)
target_compile_features(test_simd_scheduler_model PRIVATE cxx_std_20)
target_include_directories(test_simd_scheduler_model PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../backends/simd/schedule)
add_test(NAME test_simd_scheduler_model
        COMMAND test_simd_scheduler_model)
set_tests_properties(test_simd_scheduler_model PROPERTIES
        LABELS "unit;unit_simd")

add_executable(test_simd_reference_warp_collectives
        unit/simd/test_reference_warp_collectives.cpp)
target_compile_features(test_simd_reference_warp_collectives PRIVATE cxx_std_20)
target_include_directories(test_simd_reference_warp_collectives PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/unit/simd
        ${CMAKE_CURRENT_SOURCE_DIR}/../backends/simd/schedule)
add_test(NAME test_simd_reference_warp_collectives
        COMMAND test_simd_reference_warp_collectives)
set_tests_properties(test_simd_reference_warp_collectives PROPERTIES
        LABELS "unit;unit_simd")

add_executable(test_simd_schedule_ir
        unit/simd/test_schedule_ir.cpp
        ../backends/simd/schedule/schedule_ir.cpp)
target_compile_features(test_simd_schedule_ir PRIVATE cxx_std_20)
target_include_directories(test_simd_schedule_ir PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../backends/simd/schedule)
add_test(NAME test_simd_schedule_ir
        COMMAND test_simd_schedule_ir)
set_tests_properties(test_simd_schedule_ir PROPERTIES
        LABELS "unit;unit_simd")

add_executable(test_simd_thread_pool
        unit/simd/test_simd_thread_pool.cpp
        ../backends/simd/runtime/simd_thread_pool.cpp)
target_compile_features(test_simd_thread_pool PRIVATE cxx_std_20)
target_include_directories(test_simd_thread_pool PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/../backends/simd/runtime)
target_link_libraries(test_simd_thread_pool PRIVATE
        luisa-compute-core)
add_test(NAME test_simd_thread_pool
        COMMAND test_simd_thread_pool)
set_tests_properties(test_simd_thread_pool PROPERTIES
        LABELS "unit;unit_simd")

luisa_compute_add_test(test_simd_warp_uniformity
        unit/simd/test_warp_uniformity.cpp
        LABELS "unit;unit_simd;unit_xir")
target_sources(test_simd_warp_uniformity PRIVATE
        ../backends/simd/schedule/warp_uniformity.cpp)
target_include_directories(test_simd_warp_uniformity PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../backends/simd/schedule)

luisa_compute_add_test(test_simd_xir_to_schedule
        unit/simd/test_xir_to_schedule.cpp
        LABELS "unit;unit_simd;unit_xir")
target_sources(test_simd_xir_to_schedule PRIVATE
        ../backends/simd/schedule/block_barrier.cpp
        ../backends/simd/schedule/schedule_ir.cpp
        ../backends/simd/schedule/warp_uniformity.cpp
        ../backends/simd/schedule/xir_to_schedule.cpp)
target_include_directories(test_simd_xir_to_schedule PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../backends/simd/schedule)

if (TARGET luisa-compute-simd-llvm)
    add_executable(test_fallback_llvm_native_math
            unit/fallback/test_fallback_llvm_native_math.cpp
            ../backends/fallback/fallback_codegen.cpp)
    target_compile_features(test_fallback_llvm_native_math PRIVATE cxx_std_20)
    target_link_libraries(test_fallback_llvm_native_math PRIVATE
            luisa-compute-simd-llvm)
    target_include_directories(test_fallback_llvm_native_math PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/../backends/fallback
            ${LLVM_INCLUDE_DIRS})
    add_test(NAME test_fallback_llvm_native_math
            COMMAND test_fallback_llvm_native_math)
    set_tests_properties(test_fallback_llvm_native_math PROPERTIES
            LABELS "unit;unit_fallback;unit_llvm;integration_fallback")

    add_executable(test_simd_llvm_native_math
            unit/simd/test_llvm_native_math.cpp
            unit/simd/test_llvm_native_math_fast.cpp)
    target_compile_features(test_simd_llvm_native_math PRIVATE cxx_std_20)
    target_link_libraries(test_simd_llvm_native_math PRIVATE
            luisa-compute-simd-llvm)
    target_include_directories(test_simd_llvm_native_math PRIVATE
            ${LLVM_INCLUDE_DIRS})
    add_test(NAME test_simd_llvm_native_math
            COMMAND test_simd_llvm_native_math)
    set_tests_properties(test_simd_llvm_native_math PROPERTIES
            LABELS "unit;unit_simd;unit_llvm;integration_simd")

    add_executable(benchmark_llvm_native_math
            unit/simd/benchmark_llvm_native_math.cpp)
    target_compile_features(benchmark_llvm_native_math PRIVATE cxx_std_20)
    target_link_libraries(benchmark_llvm_native_math PRIVATE
            luisa-compute-simd-llvm)
    target_include_directories(benchmark_llvm_native_math PRIVATE
            ${LLVM_INCLUDE_DIRS})

    add_executable(benchmark_simd_predicated_if
            unit/simd/benchmark_predicated_if_conversion.cpp)
    target_compile_features(benchmark_simd_predicated_if PRIVATE cxx_std_20)
    target_link_libraries(benchmark_simd_predicated_if PRIVATE
            luisa-compute-simd-compiler)
    target_include_directories(benchmark_simd_predicated_if PRIVATE
            ${LLVM_INCLUDE_DIRS})

    add_executable(benchmark_simd_loop_unswitch
            unit/simd/benchmark_loop_unswitch.cpp)
    target_compile_features(benchmark_simd_loop_unswitch PRIVATE cxx_std_20)
    target_link_libraries(benchmark_simd_loop_unswitch PRIVATE
            luisa-compute-simd-compiler)
    target_include_directories(benchmark_simd_loop_unswitch PRIVATE
            ${LLVM_INCLUDE_DIRS})

    add_executable(benchmark_simd_gemm_assembly
            unit/simd/benchmark_simd_gemm_assembly.cpp)
    target_compile_features(benchmark_simd_gemm_assembly PRIVATE cxx_std_20)
    target_link_libraries(benchmark_simd_gemm_assembly PRIVATE
            luisa-compute-simd-compiler
            luisa-compute-dsl)
    target_include_directories(benchmark_simd_gemm_assembly PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/../backends/simd
            ${LLVM_INCLUDE_DIRS})

    luisa_compute_add_test(benchmark_simd_ray_query
            unit/simd/benchmark_simd_ray_query.cpp)
    luisa_compute_add_test(benchmark_simd_procedural_ray_query
            unit/simd/benchmark_simd_procedural_ray_query.cpp)

    add_executable(test_simd_llvm_value_layout
            unit/simd/test_llvm_value_layout.cpp)
    target_compile_features(test_simd_llvm_value_layout PRIVATE cxx_std_20)
    target_link_libraries(test_simd_llvm_value_layout PRIVATE
            luisa-compute-simd-llvm)
    target_include_directories(test_simd_llvm_value_layout PRIVATE
            ${LLVM_INCLUDE_DIRS})
    add_test(NAME test_simd_llvm_value_layout
            COMMAND test_simd_llvm_value_layout)
    set_tests_properties(test_simd_llvm_value_layout PROPERTIES
            LABELS "unit;unit_simd;unit_llvm")

    add_executable(test_simd_llvm_warp_collectives
            unit/simd/test_llvm_warp_collectives.cpp)
    target_compile_features(test_simd_llvm_warp_collectives PRIVATE cxx_std_20)
    target_link_libraries(test_simd_llvm_warp_collectives PRIVATE
            luisa-compute-simd-llvm)
    target_include_directories(test_simd_llvm_warp_collectives PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/unit/simd
            ${LLVM_INCLUDE_DIRS})
    add_test(NAME test_simd_llvm_warp_collectives
            COMMAND test_simd_llvm_warp_collectives)
    set_tests_properties(test_simd_llvm_warp_collectives PROPERTIES
            LABELS "unit;unit_simd;unit_llvm")

    add_executable(test_simd_llvm_schedule_codegen
            unit/simd/test_llvm_schedule_codegen.cpp)
    target_compile_features(test_simd_llvm_schedule_codegen PRIVATE cxx_std_20)
    target_link_libraries(test_simd_llvm_schedule_codegen PRIVATE
            luisa-compute-simd-compiler
            luisa-compute-dsl)
    target_include_directories(test_simd_llvm_schedule_codegen PRIVATE
            ${LLVM_INCLUDE_DIRS})
    add_test(NAME test_simd_llvm_schedule_codegen
            COMMAND test_simd_llvm_schedule_codegen)
    set_tests_properties(test_simd_llvm_schedule_codegen PROPERTIES
            LABELS "unit;unit_simd;unit_llvm;integration_simd")
endif ()

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
luisa_compute_add_test(test_foreground_moments unit/core/test_foreground_moments.cpp LABELS "unit;unit_core")
target_include_directories(test_foreground_moments PRIVATE ${CMAKE_SOURCE_DIR}/examples/common)
luisa_compute_add_test(test_example_options unit/core/test_example_options.cpp LABELS "unit;unit_core")
target_include_directories(test_example_options PRIVATE ${CMAKE_SOURCE_DIR}/examples/common)
luisa_compute_add_test(test_path_tracing_sample_plan unit/core/test_path_tracing_sample_plan.cpp LABELS "unit;unit_core")
target_include_directories(test_path_tracing_sample_plan PRIVATE ${CMAKE_SOURCE_DIR}/examples/common)
luisa_compute_add_test(test_nbody_render_plan unit/core/test_nbody_render_plan.cpp LABELS "unit;unit_core")
target_include_directories(test_nbody_render_plan PRIVATE ${CMAKE_SOURCE_DIR}/examples/common)
luisa_compute_add_test(test_photon_mapping_plan unit/core/test_photon_mapping_plan.cpp LABELS "unit;unit_core")
target_include_directories(test_photon_mapping_plan PRIVATE ${CMAKE_SOURCE_DIR}/examples/common)
luisa_compute_add_test(test_sparse_buffer_tile_plan unit/core/test_sparse_buffer_tile_plan.cpp LABELS "unit;unit_core")
target_include_directories(test_sparse_buffer_tile_plan PRIVATE ../runtime)
luisa_compute_add_test(test_sparse_texture_tile_plan unit/core/test_sparse_texture_tile_plan.cpp LABELS "unit;unit_core")
target_include_directories(test_sparse_texture_tile_plan PRIVATE ../runtime)
luisa_compute_add_test(test_hip_curve_bounds   unit/core/test_hip_curve_bounds.cpp   LABELS "unit;unit_core")
target_sources(test_hip_curve_bounds PRIVATE ../backends/hip/hip_curve_bounds.cpp)
target_include_directories(test_hip_curve_bounds PRIVATE ../backends/hip)
luisa_compute_add_test(test_hip_llvm_pipeline unit/core/test_hip_llvm_pipeline.cpp LABELS "unit;unit_core")
target_sources(test_hip_llvm_pipeline PRIVATE ../backends/hip/llvm_codegen/hip_llvm_pipeline.cpp)
target_include_directories(test_hip_llvm_pipeline PRIVATE ../backends/hip/llvm_codegen)
luisa_compute_add_test(test_glslang_spirv      unit/ext/test_glslang_spirv.cpp       LABELS "unit;unit_ext")
target_link_libraries(test_glslang_spirv PRIVATE glslang-default-resource-limits)
luisa_compute_add_test(test_command_reorder_bindless unit/ext/test_command_reorder_bindless.cpp LABELS "unit;unit_ext")
target_include_directories(test_command_reorder_bindless PRIVATE ../backends/common)
luisa_compute_add_test(test_spirv_motion_patch unit/ext/test_spirv_motion_patch.cpp  LABELS "unit;unit_ext")
target_sources(test_spirv_motion_patch PRIVATE ../backends/vk/spirv_motion_patch.cpp)
target_include_directories(test_spirv_motion_patch PRIVATE ../backends/vk)
if (TARGET luisa-compute-backend-vk)
    luisa_compute_add_test(test_vk_user_compute_codegen_route unit/ext/test_vk_user_compute_codegen_route.cpp LABELS "unit;unit_ext;vk")
    target_include_directories(test_vk_user_compute_codegen_route PRIVATE ../backends/vk)
    luisa_compute_add_test(test_vk_sparse_residency_registry unit/ext/test_vk_sparse_residency_registry.cpp LABELS "unit;unit_ext;vk")
    target_include_directories(test_vk_sparse_residency_registry PRIVATE ../backends/vk)
    target_sources(test_vk_sparse_residency_registry PRIVATE ../backends/vk/sparse_residency_registry.cpp)
    luisa_compute_add_test(test_vk_shader_binary_contract unit/ext/test_vk_shader_binary_contract.cpp LABELS "unit;unit_ext;vk")
    target_include_directories(test_vk_shader_binary_contract PRIVATE ../backends/vk)
    target_sources(test_vk_shader_binary_contract PRIVATE ../backends/vk/shader_artifact_codec.cpp)
    if (TARGET luisa-compute-spirv)
        target_link_libraries(test_vk_shader_binary_contract PRIVATE luisa-compute-spirv)
        if (LUISA_COMPUTE_ENABLE_VK_XIR_SPIRV)
            target_compile_definitions(test_vk_shader_binary_contract PRIVATE LUISA_XIR_TO_SPIRV)
        elseif (LUISA_COMPUTE_ENABLE_VK_AST_LLVM_SPIRV)
            target_compile_definitions(test_vk_shader_binary_contract PRIVATE LUISA_AST_LLVM_TO_SPIRV)
        endif ()
    endif ()
    luisa_compute_add_test(test_vk_saved_argument_contract unit/ext/test_vk_saved_argument_contract.cpp LABELS "unit;unit_ext;vk")
    target_include_directories(test_vk_saved_argument_contract PRIVATE ../backends/vk)
    luisa_compute_add_test(test_vk_indirect_dispatch unit/ext/test_vk_indirect_dispatch.cpp LABELS "unit;unit_ext;vk")
    target_include_directories(test_vk_indirect_dispatch PRIVATE ../backends/common ../backends/vk)
    luisa_compute_add_test(test_vk_resource_barrier_contract unit/ext/test_vk_resource_barrier_contract.cpp LABELS "unit;unit_ext;vk")
    target_include_directories(test_vk_resource_barrier_contract PRIVATE ../backends/vk)
    luisa_compute_add_test(test_vk_result_contract unit/ext/test_vk_result_contract.cpp LABELS "unit;unit_ext;vk")
    target_include_directories(test_vk_result_contract PRIVATE ../backends/vk)
    target_sources(test_vk_result_contract PRIVATE ../backends/vk/VulkanTools.cpp)
    luisa_compute_add_test(test_vk_cuda_interop_texture_plan unit/ext/test_vk_cuda_interop_texture_plan.cpp LABELS "unit;unit_ext;vk")
    target_include_directories(test_vk_cuda_interop_texture_plan PRIVATE ../backends/vk)
    foreach (_vulkan_header_test IN ITEMS
            test_vk_resource_barrier_contract
            test_vk_result_contract
            test_vk_cuda_interop_texture_plan)
        # These tests consume backend planning headers that expose Vulkan
        # types. Inherit the vendored Vulkan headers from Volk instead of
        # relying on a system Vulkan SDK being installed on the CI host.
        target_link_libraries(${_vulkan_header_test} PRIVATE luisa-compute-ext-volk)
    endforeach ()
    # Requires a Vulkan physical device. Keep it out of unattended CTest and
    # run it explicitly; the executable capability-skips unsupported sparse
    # residency while forcing Vulkan validation when the layer is installed.
    luisa_compute_add_test(test_vk_sparse_runtime unit/runtime/test_vk_sparse_runtime.cpp)
    target_link_libraries(test_vk_sparse_runtime PRIVATE luisa-compute-ext-volk)
    # Exercises a fatal runtime contract through a child process and requires
    # a physical Vulkan device, so keep it out of unattended CTest runs.
    luisa_compute_add_test(test_vk_native_route_guard unit/runtime/test_vk_native_route_guard.cpp)
    if (LUISA_COMPUTE_ENABLE_VK_XIR_SPIRV)
        target_compile_definitions(test_vk_native_route_guard PRIVATE
                LUISA_TEST_VK_HAS_NATIVE_XIR_SPIRV=1)
    else ()
        target_compile_definitions(test_vk_native_route_guard PRIVATE
                LUISA_TEST_VK_HAS_NATIVE_XIR_SPIRV=0)
    endif ()
endif ()
luisa_compute_add_test(test_argument_block_layout unit/ext/test_argument_block_layout.cpp LABELS "unit;unit_ext")
target_include_directories(test_argument_block_layout PRIVATE ../backends/common)
luisa_compute_add_test(test_pixel_storage unit/core/test_pixel_storage.cpp LABELS "unit;unit_core")
if (NOT LUISA_COMPUTE_USE_SYSTEM_STL)
    luisa_compute_add_test(test_eastl_allocation unit/core/test_eastl_allocation.cpp LABELS "unit;unit_core")
endif ()
if (TARGET luisa-compute-hlsl-codegen)
    luisa_compute_add_test(test_hlsl_validation_codegen unit/ext/test_hlsl_validation_codegen.cpp LABELS "unit;unit_ext;hlsl")
    target_include_directories(test_hlsl_validation_codegen PRIVATE ../backends/common/hlsl)
    target_link_libraries(test_hlsl_validation_codegen PRIVATE
            luisa-compute-hlsl-codegen
            SPIRV-Tools-opt)
endif ()
if (TARGET luisa-compute-spirv)
    luisa_compute_add_test(test_spirv_buffer_layout unit/ext/test_spirv_buffer_layout.cpp LABELS "unit;unit_ext;spirv")
    target_link_libraries(test_spirv_buffer_layout PRIVATE luisa-compute-spirv)
    luisa_compute_add_test(test_spirv_constant_ubo_layout unit/ext/test_spirv_constant_ubo_layout.cpp LABELS "unit;unit_ext;spirv")
    target_link_libraries(test_spirv_constant_ubo_layout PRIVATE luisa-compute-spirv)
    luisa_compute_add_test(test_spirv_optimizer unit/ext/test_spirv_optimizer.cpp LABELS "unit;unit_ext;spirv")
    target_link_libraries(test_spirv_optimizer PRIVATE luisa-compute-spirv)
    luisa_compute_add_test(test_spirv_float_atomics unit/ext/test_spirv_float_atomics.cpp LABELS "unit;unit_ext;spirv")
    target_link_libraries(test_spirv_float_atomics PRIVATE luisa-compute-spirv)
    luisa_compute_add_test(test_spirv_ray_culling unit/ext/test_spirv_ray_culling.cpp LABELS "unit;unit_ext;spirv")
    target_link_libraries(test_spirv_ray_culling PRIVATE luisa-compute-spirv)
    luisa_compute_add_test(test_spirv_xir_dialect unit/ext/test_spirv_xir_dialect.cpp LABELS "unit;unit_ext;spirv")
    target_link_libraries(test_spirv_xir_dialect PRIVATE luisa-compute-spirv)
    luisa_compute_add_test(test_spirv_handoff_boundaries unit/ext/test_spirv_handoff_boundaries.cpp LABELS "unit;unit_ext;spirv")
    target_link_libraries(test_spirv_handoff_boundaries PRIVATE luisa-compute-spirv)
    luisa_compute_add_test(test_spirv_volatile_buffers unit/ext/test_spirv_volatile_buffers.cpp LABELS "unit;unit_ext;spirv")
    target_link_libraries(test_spirv_volatile_buffers PRIVATE luisa-compute-spirv)
    luisa_compute_add_test(test_spirv_aggregate_indices unit/ext/test_spirv_aggregate_indices.cpp LABELS "unit;unit_ext;spirv")
    target_link_libraries(test_spirv_aggregate_indices PRIVATE luisa-compute-spirv)
    luisa_compute_add_test(test_spirv_pointer_legalization unit/ext/test_spirv_pointer_legalization.cpp LABELS "unit;unit_ext;spirv")
    target_link_libraries(test_spirv_pointer_legalization PRIVATE luisa-compute-spirv)
    luisa_compute_add_test(test_spirv_callable_reference_handoff unit/ext/test_spirv_callable_reference_handoff.cpp LABELS "unit;unit_ext;spirv")
    target_link_libraries(test_spirv_callable_reference_handoff PRIVATE luisa-compute-spirv)
    luisa_compute_add_test(test_spirv_instruction_layout unit/ext/test_spirv_instruction_layout.cpp LABELS "unit;unit_ext;spirv")
    target_link_libraries(test_spirv_instruction_layout PRIVATE luisa-compute-spirv)
    luisa_compute_add_test(test_spirv_control_flow_plan unit/ext/test_spirv_control_flow_plan.cpp LABELS "unit;unit_ext;spirv")
    target_link_libraries(test_spirv_control_flow_plan PRIVATE luisa-compute-spirv)
    luisa_compute_add_test(test_spirv_target_feature_mask unit/ext/test_spirv_target_feature_mask.cpp LABELS "unit;unit_ext;spirv")
    target_link_libraries(test_spirv_target_feature_mask PRIVATE luisa-compute-spirv)
    luisa_compute_add_test(test_spirv_target_feature_codegen unit/ext/test_spirv_target_feature_codegen.cpp LABELS "unit;unit_ext;spirv")
    target_link_libraries(test_spirv_target_feature_codegen PRIVATE luisa-compute-spirv)
    luisa_compute_add_test(test_spirv_runtime_target_plan unit/ext/test_spirv_runtime_target_plan.cpp LABELS "unit;unit_ext;spirv")
    target_link_libraries(test_spirv_runtime_target_plan PRIVATE luisa-compute-spirv)
    luisa_compute_add_test(test_spirv_raw_float_constants unit/ext/test_spirv_raw_float_constants.cpp LABELS "unit;unit_ext;spirv")
    target_link_libraries(test_spirv_raw_float_constants PRIVATE luisa-compute-spirv)
    luisa_compute_add_test(test_spirv_integer_numeric_boundaries unit/ext/test_spirv_integer_numeric_boundaries.cpp LABELS "unit;unit_ext;spirv")
    target_link_libraries(test_spirv_integer_numeric_boundaries PRIVATE luisa-compute-spirv)
    if (LUISA_COMPUTE_ENABLE_VK_XIR_SPIRV)
        luisa_compute_add_test(test_spirv_bindless_xir_usage unit/ext/test_spirv_bindless_xir_usage.cpp LABELS "unit;unit_ext;spirv")
        target_link_libraries(test_spirv_bindless_xir_usage PRIVATE luisa-compute-spirv)
    endif ()
endif ()
if (TARGET luisa-compute-spirv-llvm)
    luisa_compute_add_test(test_spirv_llvm_facade unit/ext/test_spirv_llvm_facade.cpp LABELS "unit;unit_ext;spirv;spirv_llvm")
    target_link_libraries(test_spirv_llvm_facade PRIVATE
            luisa-compute-spirv-llvm
            SPIRV-Tools-opt)
endif ()

if (LUISA_COMPUTE_ENABLE_GUI)
endif ()

# --- unit/ast: standalone tests (backend-free tests are labeled for CTest) ---
luisa_compute_add_test(test_ast             unit/ast/test_ast.cpp)
luisa_compute_add_test(test_ast_basic       unit/ast/test_ast_basic.cpp)
luisa_compute_add_test(test_function_builder_dag unit/ast/test_function_builder_dag.cpp LABELS "unit;unit_ast")
luisa_compute_add_test(test_async_copy_ast  unit/ast/test_async_copy_ast.cpp LABELS "unit;unit_ast")
luisa_compute_add_test(test_bindless_write_usage unit/ast/test_bindless_write_usage.cpp LABELS "unit;unit_ast")
luisa_compute_add_test(test_builtin_kernel  unit/ast/test_builtin_kernel.cpp)
target_include_directories(test_builtin_kernel PRIVATE ${CMAKE_SOURCE_DIR}/src/runtime)
luisa_compute_add_test(test_manual_ast      unit/ast/test_manual_ast.cpp)
luisa_compute_add_test(test_cooperative_vector unit/ast/test_cooperative_vector.cpp)

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
luisa_compute_add_test(test_async_copy              unit/dsl/test_async_copy.cpp)
luisa_compute_add_test(test_async_copy_cuda        unit/runtime/test_async_copy_cuda.cpp)
luisa_compute_add_test(test_cluster_launch_control  unit/runtime/test_cluster_launch_control.cpp)

# --- unit/runtime: standalone tests (need backend, NOT auto-run via CTest) ---
luisa_compute_add_test(test_accel_build_modes   unit/runtime/test_accel_build_modes.cpp)
luisa_compute_add_test(test_accel_visibility    unit/runtime/test_accel_visibility.cpp)
luisa_compute_add_test(test_hip_rt_scratch_reuse unit/runtime/test_hip_rt_scratch_reuse.cpp)
luisa_compute_add_test(test_hip_shader_cache    unit/runtime/test_hip_shader_cache.cpp)
luisa_compute_add_test(test_hip_callable_boundary unit/runtime/test_hip_callable_boundary.cpp)
luisa_compute_add_test(test_atomic              unit/runtime/test_atomic.cpp)
luisa_compute_add_test(test_atomic_queue        unit/runtime/test_atomic_queue.cpp)
luisa_compute_add_test(test_byte_buffer         unit/runtime/test_byte_buffer.cpp)
luisa_compute_add_test(test_context             unit/runtime/test_context.cpp)
luisa_compute_add_test(test_copy                unit/runtime/test_copy.cpp)
luisa_compute_add_test(test_cpu_callable        unit/runtime/test_cpu_callable.cpp)
luisa_compute_add_test(test_decoupled_look_back unit/runtime/test_decoupled_look_back.cpp)
luisa_compute_add_test(test_complex_kernel      unit/runtime/test_complex_kernel.cpp)
if (TARGET luisa-compute-backend-fallback)
    # Uses two independent fallback Device instances and a shared in-memory
    # BinaryIO to verify cold compilation, object-cache reuse, and dynamic
    # uniform arguments. Kept out of unattended CTest because it requires the
    # optional LLVM/Embree fallback backend at runtime.
    luisa_compute_add_test(test_fallback_shader_cache unit/runtime/test_fallback_shader_cache.cpp)
endif ()
luisa_compute_add_test(test_bindless_mip        unit/runtime/test_bindless_mip.cpp)
luisa_compute_add_test(test_direct_texture_sampling unit/runtime/test_direct_texture_sampling.cpp)
luisa_compute_add_test(test_metal_codegen_regressions unit/runtime/test_metal_codegen_regressions.cpp)
luisa_compute_add_test(test_mipmap              unit/runtime/test_mipmap.cpp)
luisa_compute_add_test(test_motion_instance_keyframes unit/runtime/test_motion_instance_keyframes.cpp LABELS "unit;unit_runtime")
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
luisa_compute_add_test(test_pbrt_curve_parser  unit/runtime/test_pbrt_curve_parser.cpp LABELS "unit;unit_runtime")
luisa_compute_add_test(test_texture_io          unit/runtime/test_texture_io.cpp)
luisa_compute_add_test(test_timeline_event      unit/runtime/test_timeline_event.cpp)
luisa_compute_add_test(test_warp                unit/runtime/test_warp.cpp)
luisa_compute_add_test(test_warp_prefix_scan    unit/runtime/test_warp_prefix_scan.cpp)
luisa_compute_add_test(test_warp_sparse_collectives unit/runtime/test_warp_sparse_collectives.cpp)
luisa_compute_add_test(test_hip_wave_size       unit/runtime/test_hip_wave_size.cpp)
luisa_compute_add_test(test_mha_warp_reduction  unit/runtime/test_mha_warp_reduction.cpp)
if (TARGET luisa-compute-backend-vk AND LUISA_COMPUTE_ENABLE_VK_XIR_SPIRV)
    # Requires an explicit backend argument and a physical Vulkan device, so
    # build it for the native path but keep it out of unattended CTest runs.
    luisa_compute_add_test(test_vk_shader_cache unit/runtime/test_vk_shader_cache.cpp)
    luisa_compute_add_test(test_vk_spirv_codegen_path unit/runtime/test_vk_spirv_codegen_path.cpp)
    target_link_libraries(test_vk_spirv_codegen_path PRIVATE
            luisa-compute-spirv
            luisa-compute-ext-volk)
    target_include_directories(test_vk_spirv_codegen_path PRIVATE ../backends/common)
    target_compile_options(test_vk_spirv_codegen_path PRIVATE
            $<$<CXX_COMPILER_ID:MSVC>:/bigobj>)
    if (LUISA_COMPUTE_VULKAN_ENABLE_DXC_COMPATIBILITY)
        target_compile_definitions(test_vk_spirv_codegen_path PRIVATE
                LUISA_TEST_VK_HAS_DXC_COMPATIBILITY=1)
    else ()
        target_compile_definitions(test_vk_spirv_codegen_path PRIVATE
                LUISA_TEST_VK_HAS_DXC_COMPATIBILITY=0)
    endif ()
endif ()
luisa_compute_add_test(test_ray_query_world_ray unit/runtime/test_ray_query_world_ray.cpp)
luisa_compute_add_test(test_buffer_io           unit/runtime/test_buffer_io.cpp)
luisa_compute_add_test(test_buffer              unit/runtime/test_buffer.cpp)
luisa_compute_add_test(test_out_of_range        unit/runtime/test_out_of_range.cpp)
luisa_compute_add_test(test_buffer_view         unit/runtime/test_buffer_view.cpp)
luisa_compute_add_test(test_device_test         unit/runtime/test_device.cpp)
luisa_compute_add_test(test_external_buffer     unit/runtime/test_external_buffer.cpp)
luisa_compute_add_test(test_gemm                unit/runtime/test_gemm.cpp)
luisa_compute_add_test(benchmark_simd_gemm      unit/simd/benchmark_simd_gemm.cpp)
luisa_compute_add_test(test_hip_codegen_arithmetic unit/runtime/test_hip_codegen_arithmetic.cpp)
luisa_compute_add_test(test_hip_curve_ray_query unit/runtime/test_hip_curve_ray_query.cpp)
luisa_compute_add_test(test_hip_motion_instance_matrix unit/runtime/test_hip_motion_instance_matrix.cpp)
luisa_compute_add_test(test_hip_motion_instance_device_ops unit/runtime/test_hip_motion_instance_device_ops.cpp)
luisa_compute_add_test(test_hip_motion_instance_srt unit/runtime/test_hip_motion_instance_srt.cpp)
luisa_compute_add_test(test_hip_motion_mesh unit/runtime/test_hip_motion_mesh.cpp)
luisa_compute_add_test(test_hip_motion_ray_query unit/runtime/test_hip_motion_ray_query.cpp)
luisa_compute_add_test(test_hip_static_motion_instance unit/runtime/test_hip_static_motion_instance.cpp)
luisa_compute_add_test(test_hip_motion_workgroup_rows unit/runtime/test_hip_motion_workgroup_rows.cpp)
luisa_compute_add_test(test_hip_ray_query_pipeline unit/runtime/test_hip_ray_query_pipeline.cpp)
luisa_compute_add_test(test_hip_signed_texture_io unit/runtime/test_hip_signed_texture_io.cpp)
luisa_compute_add_test(test_shared_mem          unit/runtime/test_shared_mem.cpp)

# XIR optimization integration executables. These require an explicit backend
# argument (and the native XIR -> SPIR-V path for the Vulkan-specific coverage),
# so build them in every configured test build but keep them out of unattended
# CTest runs.
luisa_compute_add_test(test_dsl_fuse_buffer_reads unit/runtime/test_dsl_fuse_buffer_reads.cpp)
luisa_compute_add_test(test_dsl_indvar_strength_reduce unit/runtime/test_dsl_indvar_strength_reduce.cpp)
luisa_compute_add_test(test_dsl_slp_vectorization unit/runtime/test_dsl_slp_vectorization.cpp)
luisa_compute_add_test(test_dsl_xir_optimize_vk unit/runtime/test_dsl_xir_optimize_vk.cpp)

# --- unit/xir: standalone tests (no backend needed, auto-run via CTest) ---
luisa_compute_add_test(test_ast_to_xir unit/xir/test_ast_to_xir.cpp LABELS "unit;unit_xir")
luisa_compute_add_test(test_xir_module      unit/xir/test_xir_module.cpp      LABELS "unit;unit_xir")
luisa_compute_add_test(test_xir_builder     unit/xir/test_xir_builder.cpp     LABELS "unit;unit_xir")
luisa_compute_add_test(test_xir_verifier    unit/xir/test_xir_verifier.cpp    LABELS "unit;unit_xir")
luisa_compute_add_test(test_xir_translators unit/xir/test_xir_translators.cpp LABELS "unit;unit_xir")
luisa_compute_add_test(test_xir_interchange unit/xir/test_xir_interchange.cpp LABELS "unit;unit_xir")
luisa_compute_add_test(test_xir2ast_translators unit/xir/test_xir2ast_translators.cpp LABELS "unit;unit_xir")
luisa_compute_add_test(test_xir_passes      unit/xir/test_xir_passes.cpp      LABELS "unit;unit_xir")
luisa_compute_add_test(test_xir_aggregate_field_bitmask unit/xir/test_xir_aggregate_field_bitmask.cpp LABELS "unit;unit_xir")
luisa_compute_add_test(test_xir_pass_alias_analysis unit/xir/test_xir_pass_alias_analysis.cpp LABELS "unit;unit_xir")
luisa_compute_add_test(test_xir_pass_core_analyses unit/xir/test_xir_pass_core_analyses.cpp LABELS "unit;unit_xir")
luisa_compute_add_test(test_xir_pass_defer_local_aggregate_load unit/xir/test_xir_pass_defer_local_aggregate_load.cpp LABELS "unit;unit_xir;unit_coro")
luisa_compute_add_test(test_xir_pass_coro_rematerialize unit/xir/test_xir_pass_coro_rematerialize.cpp LABELS "unit;unit_xir;unit_coro")
luisa_compute_add_test(test_xir_pass_coro_alloca_scope unit/xir/test_xir_pass_coro_alloca_scope.cpp LABELS "unit;unit_xir;unit_coro")
luisa_compute_add_test(test_xir_pass_pointer_usage unit/xir/test_xir_pass_pointer_usage.cpp LABELS "unit;unit_xir")
luisa_compute_add_test(test_xir_pass_scalar_evolution unit/xir/test_xir_pass_scalar_evolution.cpp LABELS "unit;unit_xir")
luisa_compute_add_test(test_xir_integer_alignment unit/xir/test_xir_integer_alignment.cpp LABELS "unit;unit_xir")
luisa_compute_add_test(test_xir_pass_unused_callable_removal unit/xir/test_xir_pass_unused_callable_removal.cpp LABELS "unit;unit_xir")
luisa_compute_add_test(test_xir_pass_licm unit/xir/test_xir_pass_licm.cpp LABELS "unit;unit_xir")
luisa_compute_add_test(test_xir_pass_lower_break_continue unit/xir/test_xir_pass_lower_break_continue.cpp LABELS "unit;unit_xir")
luisa_compute_add_test(test_xir_coro_cfg_distill unit/xir/test_coro_cfg_distill.cpp LABELS "unit;unit_xir;unit_coro")
luisa_compute_add_test(test_xir_coro_materialize unit/xir/test_coro_materialize.cpp LABELS "unit;unit_xir;unit_coro")
luisa_compute_add_test(test_coro_dead_field unit/xir/test_coro_dead_field.cpp LABELS "unit;unit_xir;unit_coro")
luisa_compute_add_test(test_coro_frame_size unit/xir/test_coro_frame_size.cpp LABELS "unit;unit_xir;unit_coro")
luisa_compute_add_test(test_xir_coro_reg2mem unit/xir/test_coro_reg2mem.cpp LABELS "unit;unit_xir;unit_coro")
luisa_compute_add_test(test_xir_coro_split unit/xir/test_coro_split.cpp LABELS "unit;unit_xir;unit_coro")
luisa_compute_add_test(test_coro_restructure_validation unit/xir/test_coro_restructure_validation.cpp LABELS "unit;unit_xir;unit_coro")
luisa_compute_add_test(test_xir_coro_xir2ast unit/xir/test_coro_xir2ast.cpp LABELS "unit;unit_xir;unit_coro")
luisa_compute_add_test(test_xir_coro_pipeline unit/xir/test_coro_pipeline.cpp LABELS "unit;unit_xir;unit_coro")
luisa_compute_add_test(test_coro_frame_desc unit/dsl/test_coro_frame_desc.cpp LABELS "unit;unit_xir;unit_coro")
luisa_compute_add_test(test_coro_frame_runtime unit/dsl/test_coro_frame_runtime.cpp)
luisa_compute_add_test(test_coro_compile_trigger unit/dsl/test_coro_compile_trigger.cpp LABELS "unit;unit_xir;unit_coro")
luisa_compute_add_test(test_coro_func            unit/dsl/test_coro_func.cpp            LABELS "unit;unit_xir;unit_coro")
luisa_compute_add_test(test_coro_suspend         unit/dsl/test_coro_suspend.cpp         LABELS "unit;unit_xir;unit_coro")
luisa_compute_add_test(test_coro_graph unit/coro/test_coro_graph.cpp LABELS "unit;unit_coro")
luisa_compute_add_test(test_coro_scheduler_base unit/coro/test_coro_scheduler_base.cpp LABELS "unit;unit_coro")
luisa_compute_add_test(test_coro_state_machine unit/coro/test_coro_state_machine.cpp)
luisa_compute_add_test(test_coro_persistent    unit/coro/test_coro_persistent.cpp)
luisa_compute_add_test(test_coro_persistent_opt unit/coro/test_coro_persistent_opt.cpp)
luisa_compute_add_test(test_coro_persistent_integration unit/coro/test_coro_persistent_integration.cpp)
luisa_compute_add_test(test_coro_soa_layout     unit/coro/test_coro_soa_layout.cpp)
luisa_compute_add_test(test_coro_wavefront      unit/coro/test_coro_wavefront.cpp)
luisa_compute_add_test(test_coro_radix_sort     unit/coro/test_coro_radix_sort.cpp)
luisa_compute_add_test(test_coro_all_schedulers unit/coro/test_coro_all_schedulers.cpp)
luisa_compute_add_test(test_coro_multisplit     unit/coro/test_coro_multisplit.cpp     LABELS "unit;unit_coro")
luisa_compute_add_test(test_coro_compaction    unit/coro/test_coro_compaction.cpp    LABELS "unit;unit_coro")
luisa_compute_add_test(test_coro_wavefront_integration unit/coro/test_coro_wavefront_integration.cpp)
luisa_compute_add_test(test_coro_pipeline_1suspend unit/coro/test_coro_pipeline_1suspend.cpp)
luisa_compute_add_test(test_coro_pipeline_3suspend unit/coro/test_coro_pipeline_3suspend.cpp)
luisa_compute_add_test(test_xir_pass_destructure_cfg unit/xir/test_xir_pass_destructure_cfg.cpp LABELS "unit;unit_xir")
luisa_compute_add_test(test_xir_pass_simplify_cfg unit/xir/test_xir_pass_simplify_cfg.cpp LABELS "unit;unit_xir")
luisa_compute_add_test(test_xir_pass_early_cse unit/xir/test_xir_pass_early_cse.cpp LABELS "unit;unit_xir")
luisa_compute_add_test(test_xir_pass_lower_ray_query_to_pipeline unit/xir/test_xir_pass_lower_ray_query_to_pipeline.cpp LABELS "unit;unit_xir")
luisa_compute_add_test(test_xir_pass_reconstruct_ray_query_loop unit/xir/test_xir_pass_reconstruct_ray_query_loop.cpp LABELS "unit;unit_xir")
luisa_compute_add_test(test_xir_pass_lower_irreducible_cfg unit/xir/test_xir_pass_lower_irreducible_cfg.cpp LABELS "unit;unit_xir;unit_coro")
luisa_compute_add_test(test_xir_pass_fuse_consecutive_buffer_reads unit/xir/test_xir_pass_fuse_consecutive_buffer_reads.cpp LABELS "unit;unit_xir")
luisa_compute_add_test(test_xir_pass_restructure_cfg unit/xir/test_xir_pass_restructure_cfg.cpp LABELS "unit;unit_xir")
luisa_compute_add_test(test_xir_pass_convergence_region unit/xir/test_xir_pass_convergence_region.cpp LABELS "unit;unit_xir")
luisa_compute_add_test(test_xir_pass_post_dom_tree unit/xir/test_xir_pass_post_dom_tree.cpp LABELS "unit;unit_xir")
luisa_compute_add_test(test_xir_pass_autodiff_cfg unit/xir/test_xir_pass_autodiff_cfg.cpp LABELS "unit;unit_xir")
luisa_compute_add_test(test_xir_pass_early_return_elimination unit/xir/test_xir_pass_early_return_elimination.cpp LABELS "unit;unit_xir")
luisa_compute_add_test(test_xir_pass_indvar_simplify unit/xir/test_xir_pass_indvar_simplify.cpp LABELS "unit;unit_xir")
luisa_compute_add_test(test_xir_pass_loop_rotation unit/xir/test_xir_pass_loop_rotation.cpp LABELS "unit;unit_xir")
luisa_compute_add_test(test_xir_pass_loop_fusion unit/xir/test_xir_pass_loop_fusion.cpp LABELS "unit;unit_xir")
luisa_compute_add_test(test_xir_pass_loop_vectorization unit/xir/test_xir_pass_loop_vectorization.cpp LABELS "unit;unit_xir")
luisa_compute_add_test(test_xir_pass_sccp unit/xir/test_xir_pass_sccp.cpp LABELS "unit;unit_xir")
luisa_compute_add_test(test_xir_pass_slp_vectorization unit/xir/test_xir_pass_slp_vectorization.cpp LABELS "unit;unit_xir")
luisa_compute_add_test(test_xir_pass_pipeline unit/xir/test_xir_pass_pipeline.cpp LABELS "unit;unit_xir")
luisa_compute_add_test(test_xir_pass_mutation_safety unit/xir/test_xir_pass_mutation_safety.cpp LABELS "unit;unit_xir")

# =============================================================================
# Integration Tests (require GPU backends, NOT auto-run via CTest)
# Run manually: ./<test_name> <backend>
# =============================================================================

# --- integration/runtime: standalone tests ---
luisa_compute_add_test(test_device_debugger         integration/runtime/test_device_debugger.cpp)
luisa_compute_add_test(test_dstorage_decompression  integration/runtime/test_dstorage_decompression.cpp)
luisa_compute_add_test(test_procedural_callable      integration/runtime/test_procedural_callable.cpp)
luisa_compute_add_test(test_rtx                      integration/runtime/test_rtx.cpp)
luisa_compute_add_test(test_aot                      integration/runtime/test_aot.cpp)
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
    luisa_compute_add_test(test_raster         integration/runtime/test_raster.cpp)
    luisa_compute_add_test(test_memory_compact integration/runtime/test_memory_compact.cpp)
    # test_work_graph disabled: work_graph headers not yet available
    # luisa_compute_add_test(test_work_graph      integration/runtime/test_work_graph.cpp     ARGS dx)
endif ()

# --- integration/runtime: CUDA-only tests ---
if (LUISA_COMPUTE_ENABLE_CUDA)
    luisa_compute_add_test(test_cuda_graph integration/runtime/test_cuda_graph.cpp)
endif ()

# --- integration/xir: standalone tests ---
luisa_compute_add_test(test_xir2ast_roundtrip integration/xir/test_xir2ast_roundtrip.cpp LABELS "unit;unit_xir")

# --- unit/dsl: standalone GPU tests ---
luisa_compute_add_test(test_dsl_mathematic       unit/dsl/test_dsl_mathematic.cpp)

# --- unit/runtime: FP4/FP8 quantization tests ---
luisa_compute_add_test(test_fp4                  unit/runtime/test_fp4.cpp)
luisa_compute_add_test(test_fp4_quantization     unit/runtime/test_fp4_quantization.cpp)
luisa_compute_add_test(test_fp8                  unit/runtime/test_fp8.cpp)
luisa_compute_add_test(test_fp8_quantization     unit/runtime/test_fp8_quantization.cpp)

# The SIMD backend is a host-only module, so its supported runtime conformance
# subset is safe to run unattended without a GPU or driver installation.
if (TARGET luisa-compute-backend-simd)
    luisa_compute_add_test(test_simd_runtime_widths
            unit/simd/test_simd_runtime_widths.cpp
            LABELS integration_simd runtime_simd)
    luisa_compute_add_test(test_simd_local_memory
            unit/simd/test_simd_local_memory.cpp
            LABELS integration_simd runtime_simd)
    luisa_compute_add_test(test_simd_atomics
            unit/simd/test_simd_atomics.cpp
            LABELS integration_simd runtime_simd)
    luisa_compute_add_test(test_simd_bindless_buffers
            unit/simd/test_simd_bindless_buffers.cpp
            LABELS integration_simd runtime_simd)
    luisa_compute_add_test(test_simd_bindless_textures
            unit/simd/test_simd_bindless_textures.cpp
            LABELS integration_simd runtime_simd)
    luisa_compute_add_test(test_simd_arithmetic
            unit/simd/test_simd_arithmetic.cpp
            LABELS integration_simd runtime_simd)
    luisa_compute_add_test(test_simd_accel
            unit/simd/test_simd_accel.cpp
            LABELS integration_simd runtime_simd)
    luisa_compute_add_test(test_simd_curves
            unit/simd/test_simd_curves.cpp
            LABELS integration_simd runtime_simd)
    luisa_compute_add_test(test_simd_procedural
            unit/simd/test_simd_procedural.cpp
            LABELS integration_simd runtime_simd)
    foreach (_simd_backend_test IN ITEMS
            test_simd_runtime_widths
            test_simd_local_memory
            test_simd_atomics
            test_simd_bindless_buffers
            test_simd_bindless_textures
            test_simd_arithmetic
            test_simd_accel
            test_simd_curves
            test_simd_procedural)
        add_dependencies(${_simd_backend_test}
                luisa-compute-backend-simd)
    endforeach ()
    add_test(NAME test_simd_accel_incoherent_oracle
            COMMAND $<TARGET_FILE:test_simd_accel>)
    set_tests_properties(test_simd_accel_incoherent_oracle PROPERTIES
            LABELS "integration_simd;runtime_simd"
            ENVIRONMENT "LUISA_SIMD_DISABLE_COHERENT_W16_DIRECT_TRACE=1")

    foreach (_simd_runtime_test IN ITEMS
            test_buffer
            test_buffer_view
            test_buffer_io
            test_texture_io
            test_direct_texture_sampling
            test_accel_build_modes
            test_accel_visibility
            test_ray_query_world_ray
            test_dsl_slp_vectorization
            test_printer
            test_printer_custom_callback
            test_shared_mem
            test_shared_memory
            test_softmax
            test_warp
            test_warp_prefix_scan
            test_warp_sparse_collectives)
        add_dependencies(${_simd_runtime_test}
                luisa-compute-backend-simd)
        add_test(NAME simd_${_simd_runtime_test}
                COMMAND $<TARGET_FILE:${_simd_runtime_test}> simd)
        set_tests_properties(simd_${_simd_runtime_test} PROPERTIES
                LABELS "integration_simd;runtime_simd")
    endforeach ()
    foreach (_simd_shared_width IN ITEMS 1 2 4 16)
        foreach (_simd_shared_test IN ITEMS
                test_shared_mem
                test_shared_memory
                test_softmax)
            add_test(
                    NAME simd_${_simd_shared_test}_w${_simd_shared_width}
                    COMMAND $<TARGET_FILE:${_simd_shared_test}> simd)
            set_tests_properties(
                    simd_${_simd_shared_test}_w${_simd_shared_width}
                    PROPERTIES
                    LABELS "integration_simd;runtime_simd"
                    ENVIRONMENT
                    "LUISA_SIMD_WARP_WIDTH=${_simd_shared_width}")
        endforeach ()
    endforeach ()
    add_dependencies(test_bindless_mip
            luisa-compute-backend-simd)
    add_test(NAME simd_test_bindless_mip
            COMMAND $<TARGET_FILE:test_bindless_mip> simd)
    set_tests_properties(simd_test_bindless_mip PROPERTIES
            LABELS "integration_simd;runtime_simd")
endif ()

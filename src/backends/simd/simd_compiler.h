#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "llvm/llvm_jit.h"
#include "llvm/llvm_schedule_codegen.h"

namespace luisa::compute::xir {
class Function;
}// namespace luisa::compute::xir

namespace luisa::compute {
class Function;
}// namespace luisa::compute

namespace luisa::compute::simd {

struct SIMDCompiledKernel {
    std::unique_ptr<LLVMJIT> jit{};
    void *entry{nullptr};
    // Optional runtime-only block packet wrapper; see
    // LLVMScheduleCodegenResult::packet_batch_entry.
    void *packet_batch_entry{nullptr};
    // Optional runtime-only consecutive-block wrapper; see
    // LLVMScheduleCodegenResult::block_batch_entry.
    void *block_batch_entry{nullptr};
    size_t argument_buffer_size{0u};
    std::vector<SIMDLLVMPrintFormat> print_formats{};
    // Pre-schedule rewrite feedback for diagnostics/tests.
    size_t fast_math_identity_count{0u};
    size_t fast_math_radix_pow_count{0u};
    size_t decomposed_aggregate_alloca_count{0u};
    size_t inserted_aggregate_leaf_alloca_count{0u};
    size_t predicated_diamond_count{0u};
    size_t predicated_instruction_count{0u};
    size_t predicated_phi_count{0u};
    size_t predicated_refinement_round_count{0u};
    size_t predicated_forwarded_phi_count{0u};
    size_t predicated_forwarding_block_count{0u};
    size_t predicated_widened_update_diamond_count{0u};
    size_t predicated_wide_select_ladder_diamond_count{0u};
    size_t predicated_ray_query_filter_diamond_count{0u};
    size_t factored_select_count{0u};
    size_t unswitched_loop_count{0u};
    size_t guarded_unswitched_loop_count{0u};
    size_t unswitched_cloned_block_count{0u};
    size_t unswitched_cloned_instruction_count{0u};
    size_t unswitched_live_out_count{0u};
    size_t schedule_block_count{0u};
    size_t convergence_point_count{0u};
    bool scalar_frame_metadata{false};
    size_t state_slot_count{0u};
    size_t coalesced_state_slot_count{0u};
    size_t general_colored_state_slot_count{0u};
    size_t spilled_instruction_count{0u};
    size_t cold_state_slot_count{0u};
    size_t stack_pinned_state_slot_count{0u};
    size_t ray_query_count{0u};
    size_t ray_query_scratch_slot_count{0u};
    size_t ray_query_scratch_bytes{0u};
    size_t ray_query_status_slot_count{0u};
    size_t ray_query_state_handle_slot_count{0u};
    size_t compact_surface_filter_state_count{0u};
    size_t output_only_empty_surface_filter_state_count{0u};
    size_t direct_output_surface_filter_state_count{0u};
    size_t direct_ray_query_pipeline_count{0u};
    size_t post_reconstruction_ray_query_pipeline_count{0u};
    size_t resident_ray_query_pipeline_count{0u};
    size_t surface_filter_ray_query_pipeline_count{0u};
    size_t uniform_buffer_broadcast_count{0u};
    size_t contiguous_buffer_read_count{0u};
    size_t contiguous_buffer_write_count{0u};
    size_t transposed_buffer_read_count{0u};
    size_t transposed_buffer_write_count{0u};
    size_t paired_leaf_gather_count{0u};
    size_t biased_narrow_buffer_gather_count{0u};
    size_t interleaved_scalar_buffer_read_group_count{0u};
    size_t interleaved_scalar_buffer_read_count{0u};
    size_t interleaved_scalar_buffer_read_alias_guard_count{0u};
    size_t guarded_native_texture_read_count{0u};
    size_t guarded_native_texture_write_count{0u};
    size_t guarded_byte4_texture_write_count{0u};
    size_t predicated_memory_diamond_count{0u};
    size_t predicated_memory_instruction_count{0u};
    size_t local_predicated_diamond_count{0u};
    size_t local_predicated_two_sided_diamond_count{0u};
    size_t local_predicated_assignment_diamond_count{0u};
    size_t local_predicated_block_count{0u};
    size_t local_predicated_instruction_count{0u};
    size_t nested_predicated_region_count{0u};
    size_t nested_predicated_block_count{0u};
    size_t nested_predicated_instruction_count{0u};
    size_t chained_predicated_region_count{0u};
    size_t chained_predicated_transition_count{0u};
    size_t chained_predicated_block_count{0u};
    size_t chained_predicated_nested_tail_count{0u};
    size_t chained_predicated_terminal_block_count{0u};
    size_t chained_predicated_terminal_instruction_count{0u};
    size_t predicated_loop_count{0u};
    size_t predicated_loop_block_count{0u};
    size_t predicated_loop_instruction_count{0u};
    size_t predicated_loop_batch_iteration_count{0u};
    size_t structured_early_exit_loop_count{0u};
    size_t structured_early_exit_loop_block_count{0u};
    size_t structured_early_exit_loop_instruction_count{0u};
    size_t structured_early_exit_loop_absorbed_block_count{0u};
    bool native_predicated_loop{false};
    size_t cohort_uniform_loop_branch_count{0u};
    size_t coherent_mask_reuse_count{0u};
    size_t all_on_region_version_count{0u};
    size_t all_on_region_block_count{0u};
    size_t all_on_region_instruction_count{0u};
    size_t convergence_token_guard_count{0u};
    size_t return_frame_guard_count{0u};
    size_t direct_divergent_child_count{0u};
    size_t unit_dimension_mask_elision_count{0u};
    size_t linear_1d_thread_id_count{0u};
    size_t linear_1d_packet_tail_narrowing_count{0u};
    size_t linear_1d_block_coalescing_count{0u};
    size_t shared_memory_size{0u};
    size_t block_barrier_count{0u};
    size_t block_barrier_loop_epoch_count{0u};
    bool cooperative_block{false};
    bool direct_control_flow{false};
    size_t predicated_acyclic_surface_filter_handler_count{0u};
    uint32_t warp_width{0u};
    std::string target_triple{};
    // Populated only when explicitly requested for diagnostics.
    std::string llvm_ir{};
    std::string assembly{};
    std::vector<std::string> diagnostics{};

    [[nodiscard]] bool succeeded() const noexcept {
        return jit != nullptr &&
               (static_cast<uint32_t>(entry != nullptr) +
                    static_cast<uint32_t>(packet_batch_entry != nullptr) +
                    static_cast<uint32_t>(block_batch_entry != nullptr) ==
                1u) &&
               diagnostics.empty();
    }
};

// Compiles already-canonicalized XIR through Schedule IR to a host ORC entry.
// The returned function uses the packet ABI documented in
// llvm_schedule_codegen.h. Unsupported Phase-2 features are returned as
// diagnostics instead of being silently scalarized.
[[nodiscard]] SIMDCompiledKernel compile_simd_kernel(
    const xir::Function *function, uint32_t warp_width,
    std::string_view entry_name = {}, bool enable_fast_math = false,
    bool enable_uniform_buffer_broadcast = true,
    bool enable_lane_affine_buffer = true,
    bool capture_assembly = false,
    uint32_t dispatch_worker_count = 1u,
    bool enable_packet_batch_entry = false,
    bool enable_block_batch_entry = false);

// Translates a DSL/AST kernel to XIR, legalizes its structured control flow,
// inlines callables, promotes local SSA storage, and then invokes the packet
// compiler above. This is the front door used by the runtime backend.
[[nodiscard]] SIMDCompiledKernel compile_simd_kernel(
    const compute::Function &kernel, uint32_t warp_width,
    std::string_view entry_name = {}, bool enable_fast_math = false,
    bool capture_assembly = false,
    uint32_t dispatch_worker_count = 1u,
    bool enable_packet_batch_entry = false,
    bool enable_block_batch_entry = false);

}// namespace luisa::compute::simd

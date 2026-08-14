#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "llvm/llvm_jit.h"

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
    size_t argument_buffer_size{0u};
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
    size_t factored_select_count{0u};
    size_t unswitched_loop_count{0u};
    size_t unswitched_cloned_block_count{0u};
    size_t unswitched_cloned_instruction_count{0u};
    size_t unswitched_live_out_count{0u};
    size_t schedule_block_count{0u};
    size_t convergence_point_count{0u};
    size_t state_slot_count{0u};
    size_t spilled_instruction_count{0u};
    size_t cold_state_slot_count{0u};
    size_t stack_pinned_state_slot_count{0u};
    size_t ray_query_count{0u};
    size_t ray_query_scratch_slot_count{0u};
    size_t ray_query_scratch_bytes{0u};
    size_t ray_query_status_slot_count{0u};
    size_t ray_query_state_handle_slot_count{0u};
    size_t uniform_buffer_broadcast_count{0u};
    size_t contiguous_buffer_read_count{0u};
    size_t contiguous_buffer_write_count{0u};
    size_t paired_leaf_gather_count{0u};
    size_t predicated_memory_diamond_count{0u};
    size_t predicated_memory_instruction_count{0u};
    size_t coherent_mask_reuse_count{0u};
    size_t convergence_token_guard_count{0u};
    size_t direct_divergent_child_count{0u};
    bool direct_control_flow{false};
    uint32_t warp_width{0u};
    std::string target_triple{};
    // Populated only when explicitly requested by a diagnostic benchmark.
    std::string assembly{};
    std::vector<std::string> diagnostics{};

    [[nodiscard]] bool succeeded() const noexcept {
        return jit != nullptr && entry != nullptr && diagnostics.empty();
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
    bool capture_assembly = false);

// Translates a DSL/AST kernel to XIR, legalizes its structured control flow,
// inlines callables, promotes local SSA storage, and then invokes the packet
// compiler above. This is the front door used by the runtime backend.
[[nodiscard]] SIMDCompiledKernel compile_simd_kernel(
    const compute::Function &kernel, uint32_t warp_width,
    std::string_view entry_name = {}, bool enable_fast_math = false,
    bool capture_assembly = false);

}// namespace luisa::compute::simd

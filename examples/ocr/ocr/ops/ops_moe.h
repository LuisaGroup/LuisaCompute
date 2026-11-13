#pragma once
// OCR engine -- Mixture-of-Experts operators (Phase 10). Device-free DSL
// factories + independent double-accumulation host oracles.
//
// Correctness-first MoE building blocks for the DeepSeek-OCR-2 vertical slice:
//   * FP32 router projection (ops_gemm) + FP32 max-subtracted softmax.
//   * Deterministic top-k selection with LOWEST-INDEX tie-break and normalized
//     expert weights (the selected softmax probabilities are renormalized to
//     sum to 1). No approximate router shortcut.
//   * Histogram/counts, exclusive-prefix offsets, and a token-to-expert
//     permutation (expert-sorted slot order, stable within an expert by slot
//     index). Gather/scatter move rows to/from expert-sorted order.
//   * Reference expert math is a loop over ACTIVE experts (host drives the
//     per-expert GEMMs with the exact active row count); the gathered rows are
//     already expert-sorted, so each active expert's gate/up/SwiGLU/down is a
//     plain per-expert GEMM. NO grouped persistent GEMM framework (Phase 11).
//   * Weighted restore to token order PLUS two shared dense experts.
//
// All activations/accumulators are FP32. The router weight is dense FP32
// [hidden, num_experts] (layout kn); quantized router/expert storage is NOT
// part of Phase 10 (an explicit later arm). Slot = token*top_k + j.
#include <cstdint>

#include <luisa/core/stl/vector.h>
#include <luisa/dsl/syntax.h>

#include "ocr/runtime/dispatch.h"
#include "ocr/status.h"

namespace luisa::ocr {

// Router geometry contract. num_experts/top_k are EXACT (the DeepSeek contract
// is 64 routed experts / top-6 / two shared); hidden dims may scale, the expert
// count and top-k may NOT.
struct MoeRouterConfig {
    uint32_t num_experts{0};
    uint32_t top_k{0};
    [[nodiscard]] Status validate() const;
};

[[nodiscard]] luisa::string moe_variant_key(luisa::string_view op);

// ---------------------------------------------------------------------------
// Device kernels
// ---------------------------------------------------------------------------

// FP32 softmax over expert logits: probs[t,e] = softmax(logits[t,:]).
// One thread per token; max-subtracted, FP32 accumulation. A non-finite row or
// zero sum writes zeros defensively (host validation rejects non-finite input
// BEFORE launch).
using MoeRouterSoftmaxKernel = compute::Kernel1D<void(
    compute::Buffer<float> /*logits [tokens, experts]*/,
    compute::Buffer<float> /*probs [tokens, experts]*/,
    uint32_t /*tokens*/, uint32_t /*experts*/)>;
[[nodiscard]] MoeRouterSoftmaxKernel make_moe_router_softmax_kernel() noexcept;

// Deterministic top-k over softmax probs. ids[t*top_k+j] (uint32 expert index),
// weights[t*top_k+j] (float, renormalized to sum 1 over the selected experts).
// Tie-break: equal probability selects the LOWEST expert index. One thread per
// token; k selection passes, each excluding already-chosen experts.
using MoeTopKKernel = compute::Kernel1D<void(
    compute::Buffer<float> /*probs [tokens, experts]*/,
    compute::Buffer<uint32_t> /*ids [tokens*top_k]*/,
    compute::Buffer<float> /*weights [tokens*top_k]*/,
    uint32_t /*tokens*/, uint32_t /*experts*/, uint32_t /*top_k*/)>;
[[nodiscard]] MoeTopKKernel make_moe_topk_kernel() noexcept;

// Histogram: counts[e] = number of slots whose selected expert is e.
// One thread per expert, loops all slots (deterministic, no atomics).
using MoeHistogramKernel = compute::Kernel1D<void(
    compute::Buffer<uint32_t> /*ids [slots]*/,
    compute::Buffer<uint32_t> /*counts [experts]*/,
    uint32_t /*slots*/, uint32_t /*experts*/)>;
[[nodiscard]] MoeHistogramKernel make_moe_histogram_kernel() noexcept;

// Exclusive prefix sum of counts -> offsets[e] (start row of expert e in the
// expert-sorted order). Single thread; experts is small (64).
using MoeOffsetsKernel = compute::Kernel1D<void(
    compute::Buffer<uint32_t> /*counts [experts]*/,
    compute::Buffer<uint32_t> /*offsets [experts]*/,
    uint32_t /*experts*/)>;
[[nodiscard]] MoeOffsetsKernel make_moe_offsets_kernel() noexcept;

// Token-to-expert permutation. One thread per expert walks slots in ascending
// order and appends matching slots at offsets[e]+run. Outputs:
//   perm_slot[dst]  = original slot index (token*top_k + j)
//   perm_token[dst] = original token index (slot / top_k)
//   slot_pos[slot]  = dst (inverse map, for weighted scatter restore)
using MoePermutationKernel = compute::Kernel1D<void(
    compute::Buffer<uint32_t> /*ids [slots]*/,
    compute::Buffer<uint32_t> /*offsets [experts]*/,
    compute::Buffer<uint32_t> /*perm_slot [slots]*/,
    compute::Buffer<uint32_t> /*slot_pos [slots]*/,
    compute::Buffer<uint32_t> /*perm_token [slots]*/,
    uint32_t /*slots*/, uint32_t /*experts*/, uint32_t /*top_k*/)>;
[[nodiscard]] MoePermutationKernel make_moe_permutation_kernel() noexcept;

// Gather rows into expert-sorted order: gathered[s,:] = x[perm_token[s],:].
// One thread per (slot, hidden element).
using MoeGatherKernel = compute::Kernel1D<void(
    compute::Buffer<float> /*x [tokens, hidden]*/,
    compute::Buffer<uint32_t> /*perm_token [slots]*/,
    compute::Buffer<float> /*gathered [slots, hidden]*/,
    uint32_t /*slots*/, uint32_t /*hidden*/)>;
[[nodiscard]] MoeGatherKernel make_moe_gather_kernel() noexcept;

// Weighted restore to token order + two shared dense experts:
//   out[t,d] = shared0[t,d] + shared1[t,d]
//            + sum_j weights[t*top_k+j] * expert_out[slot_pos[t*top_k+j], d]
// expert_out is the UNWEIGHTED per-slot expert output (expert-sorted order).
// One thread per (token, hidden element).
using MoeScatterKernel = compute::Kernel1D<void(
    compute::Buffer<float> /*expert_out [slots, hidden]*/,
    compute::Buffer<uint32_t> /*slot_pos [slots]*/,
    compute::Buffer<float> /*weights [slots]*/,
    compute::Buffer<float> /*shared0 [tokens, hidden]*/,
    compute::Buffer<float> /*shared1 [tokens, hidden]*/,
    compute::Buffer<float> /*out [tokens, hidden]*/,
    uint32_t /*tokens*/, uint32_t /*hidden*/, uint32_t /*top_k*/)>;
[[nodiscard]] MoeScatterKernel make_moe_scatter_kernel() noexcept;

// ---------------------------------------------------------------------------
// Dispatch helpers (TOTAL threads; every factory has one).
// ---------------------------------------------------------------------------
[[nodiscard]] Result<LaunchPlan1D> plan_moe_router_dispatch(uint64_t tokens);
[[nodiscard]] Result<LaunchPlan1D> plan_moe_topk_dispatch(uint64_t tokens);
[[nodiscard]] Result<LaunchPlan1D> plan_moe_histogram_dispatch(uint64_t experts);
[[nodiscard]] Result<LaunchPlan1D> plan_moe_offsets_dispatch(uint64_t experts);
[[nodiscard]] Result<LaunchPlan1D> plan_moe_permutation_dispatch(uint64_t experts);
[[nodiscard]] Result<LaunchPlan1D> plan_moe_gather_dispatch(uint64_t slots, uint64_t hidden);
[[nodiscard]] Result<LaunchPlan1D> plan_moe_scatter_dispatch(uint64_t tokens, uint64_t hidden);

// Host validators: reject malformed routing inputs BEFORE launch.
[[nodiscard]] Status validate_moe_router_inputs(uint64_t tokens, uint64_t hidden,
                                                const MoeRouterConfig &router);
[[nodiscard]] Status validate_moe_topk_inputs(uint64_t tokens, const MoeRouterConfig &router);

// Validate a routing TABLE (device readback or oracle output) before it drives
// any gather/scatter subview: counts must sum to `slots`, offsets must be their
// exclusive prefix, and offset+count must fit inside `slots`. Returns a Status
// (never aborts), so a corrupt readback can never slice out of bounds.
[[nodiscard]] Status validate_moe_routing_tables(luisa::span<const uint32_t> counts,
                                                 luisa::span<const uint32_t> offsets,
                                                 uint64_t slots);

// ---------------------------------------------------------------------------
// Independent host oracles (double accumulation; NO device/DSL calls).
// ---------------------------------------------------------------------------

// Full routing result for one token batch (projection + softmax + top-k +
// histogram + offsets + permutation). All arrays sized for `tokens`/`slots`.
struct MoeRouting {
    luisa::vector<float> logits;     // [tokens, experts]
    luisa::vector<float> probs;      // [tokens, experts]
    luisa::vector<uint32_t> ids;     // [tokens*top_k]
    luisa::vector<float> weights;    // [tokens*top_k] normalized
    luisa::vector<uint32_t> counts;  // [experts]
    luisa::vector<uint32_t> offsets; // [experts]
    luisa::vector<uint32_t> perm_slot;  // [slots]
    luisa::vector<uint32_t> slot_pos;   // [slots]
    luisa::vector<uint32_t> perm_token; // [slots]
};

// Router projection (double-accum GEMM) + softmax + deterministic top-k +
// histogram/offsets/permutation. `router_w` is [hidden, experts] row-major.
[[nodiscard]] Status ref_moe_route(luisa::span<const float> x, luisa::span<const float> router_w,
                                   uint32_t tokens, uint32_t hidden, const MoeRouterConfig &router,
                                   MoeRouting &out);
// Softmax only (for isolated router-softmax parity).
[[nodiscard]] Status ref_moe_softmax(luisa::span<const float> logits, uint32_t tokens, uint32_t experts,
                                     luisa::vector<float> &probs);
// Deterministic top-k only (ids + normalized weights) from probs.
[[nodiscard]] Status ref_moe_topk(luisa::span<const float> probs, uint32_t tokens, uint32_t experts,
                                  uint32_t top_k, luisa::vector<uint32_t> &ids, luisa::vector<float> &weights);
// Gather rows: gathered[s,:] = x[perm_token[s],:].
[[nodiscard]] Status ref_moe_gather(luisa::span<const float> x, luisa::span<const uint32_t> perm_token,
                                    uint32_t slots, uint32_t hidden, luisa::vector<float> &out);
// One expert's SwiGLU FFN over `rows` gathered rows: down(silu(gate(x))*up(x)).
// gate/up [hidden, inter], down [inter, hidden]. Double accumulation.
[[nodiscard]] Status ref_moe_expert_ffn(luisa::span<const float> x, luisa::span<const float> gate_w,
                                        luisa::span<const float> up_w, luisa::span<const float> down_w,
                                        uint32_t rows, uint32_t hidden, uint32_t inter,
                                        luisa::vector<float> &out);
// Weighted restore + two shared experts (mirrors MoeScatterKernel).
[[nodiscard]] Status ref_moe_scatter(luisa::span<const float> expert_out, luisa::span<const uint32_t> slot_pos,
                                     luisa::span<const float> weights, luisa::span<const float> shared0,
                                     luisa::span<const float> shared1, uint32_t tokens, uint32_t hidden,
                                     uint32_t top_k, luisa::vector<float> &out);

// One routed expert's weight triple (gate/up/down), expert-sorted layout.
struct MoeExpertWeights {
    luisa::vector<float> gate_w, up_w, down_w;
};

// Reference MoE layer: loop over ACTIVE experts (counts[e] > 0), run each
// expert's FFN on its gathered rows, then weighted restore + two shared experts.
// `routed` has num_experts entries; `shared0/shared1` are dense expert weights.
// Outputs let tests compare EVERY contribution:
//   expert_out  [slots, hidden]   unweighted per-slot routed output (sorted)
//   weighted    [tokens*top_k, hidden] per-slot WEIGHTED contribution, token order
//   shared0_out / shared1_out [tokens, hidden] the two shared dense experts
//   out         [tokens, hidden]  final token-order MoE output
[[nodiscard]] Status ref_moe_layer(luisa::span<const float> x, const MoeRouting &routing,
                                   luisa::span<const MoeExpertWeights> routed,
                                   const MoeExpertWeights &shared0, const MoeExpertWeights &shared1,
                                   uint32_t tokens, uint32_t hidden, uint32_t inter, uint32_t shared_inter,
                                   luisa::vector<float> &expert_out, luisa::vector<float> &weighted,
                                   luisa::vector<float> &shared0_out, luisa::vector<float> &shared1_out,
                                   luisa::vector<float> &out);

}// namespace luisa::ocr

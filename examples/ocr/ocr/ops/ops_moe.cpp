// OCR engine -- Mixture-of-Experts operator implementations (Phase 10).
// Device-free DSL factories + independent double-accumulation host oracles.
// See ops_moe.h for the exact routing/permutation/expert contract.
#include "ocr/ops/ops_moe.h"

#include <algorithm>
#include <cmath>
#include <limits>

#include <luisa/core/stl/format.h>
#include <luisa/dsl/sugar.h>

#include "ocr/ops/cpu_reference.h"
#include "ocr/ops/ops_common.h"
#include "ocr/runtime/shader_cache.h"

namespace luisa::ocr {

using namespace luisa::compute;

// ---------------------------------------------------------------------------
// Config / keys / validators
// ---------------------------------------------------------------------------

[[nodiscard]] Status MoeRouterConfig::validate() const {
    if (num_experts == 0u) {
        return Status::error(StatusCode::invalid_argument, "MoE num_experts must be nonzero", "num_experts");
    }
    if (top_k == 0u) {
        return Status::error(StatusCode::invalid_argument, "MoE top_k must be nonzero", "top_k");
    }
    if (top_k > num_experts) {
        return Status::error(StatusCode::invalid_argument, "MoE top_k must be <= num_experts", "top_k");
    }
    return Status::success();
}

[[nodiscard]] luisa::string moe_variant_key(luisa::string_view op) {
    return make_shader_key("ocr_moe", op);
}

[[nodiscard]] Status validate_moe_router_inputs(uint64_t tokens, uint64_t hidden,
                                                const MoeRouterConfig &router) {
    auto s = router.validate();
    if (!s.ok()) { return s; }
    if (hidden == 0u) {
        return Status::error(StatusCode::invalid_argument, "MoE hidden must be nonzero", "hidden");
    }
    uint64_t logits = 0;
    if (!checked_mul(tokens, router.num_experts, logits)) {
        return Status::error(StatusCode::out_of_memory, "MoE router logit count overflows", "tokens*experts");
    }
    if (logits > std::numeric_limits<uint32_t>::max()) {
        return Status::error(StatusCode::invalid_argument,
                             "MoE router logits exceed the uint32 kernel index envelope", "tokens*experts");
    }
    return Status::success();
}

[[nodiscard]] Status validate_moe_topk_inputs(uint64_t tokens, const MoeRouterConfig &router) {
    auto s = router.validate();
    if (!s.ok()) { return s; }
    uint64_t slots = 0;
    if (!checked_mul(tokens, router.top_k, slots)) {
        return Status::error(StatusCode::out_of_memory, "MoE slot count overflows", "tokens*top_k");
    }
    if (slots > std::numeric_limits<uint32_t>::max()) {
        return Status::error(StatusCode::invalid_argument,
                             "MoE slots exceed the uint32 kernel index envelope", "tokens*top_k");
    }
    return Status::success();
}

[[nodiscard]] Status validate_moe_routing_tables(luisa::span<const uint32_t> counts,
                                                 luisa::span<const uint32_t> offsets,
                                                 uint64_t slots) {
    if (counts.size() != offsets.size()) {
        return Status::error(StatusCode::invalid_argument, "routing counts/offsets size mismatch", "routing");
    }
    uint64_t sum = 0;
    uint64_t running = 0;
    for (size_t e = 0; e < counts.size(); e++) {
        if (static_cast<uint64_t>(offsets[e]) != running) {
            return Status::error(StatusCode::invalid_argument,
                                 luisa::format("routing offsets[{}] = {} is not the exclusive prefix {}",
                                               e, offsets[e], running),
                                 "offsets");
        }
        sum += counts[e];
        running += counts[e];
        if (running > slots) {
            return Status::error(StatusCode::invalid_argument,
                                 "routing offset+count exceeds the slot count", "counts");
        }
    }
    if (sum != slots) {
        return Status::error(StatusCode::invalid_argument,
                             luisa::format("routing counts sum {} does not equal the slot count {}", sum, slots),
                             "counts");
    }
    return Status::success();
}

// ---------------------------------------------------------------------------
// Device kernels
// ---------------------------------------------------------------------------

[[nodiscard]] MoeRouterSoftmaxKernel make_moe_router_softmax_kernel() noexcept {
    MoeRouterSoftmaxKernel kernel = [](BufferFloat logits, BufferFloat probs,
                                       Var<uint> tokens, Var<uint> experts) noexcept {
        set_block_size(256u, 1u, 1u);
        set_name("ocr_moe_router_softmax");
        const auto t = ocr_linear_id();
        $if (t < tokens & experts > 0u) {
            const auto base = t * experts;
            // Defensive: a non-finite row writes zeros (host validation rejects
            // non-finite input before launch; this keeps a skipped validation
            // from producing NaN-driven garbage).
            Var<uint> finite = def(1u);
            // TRUE -inf initial: seed the running max with element 0 instead of a
            // finite sentinel. A sentinel above the real maximum (e.g. -3.0e38
            // for a row of -3.4e38) makes every exp() underflow to 0, so the sum
            // is 0 and the division yields NaN. Seeding from a real element
            // guarantees exp(max-max)==1, hence sum >= 1 for any finite row.
            Var<float> max_v = def(logits.read(base));
            $for (e, experts) {
                const auto v = logits.read(base + e);
                $if (isnan(v) | isinf(v)) { finite = 0u; };
                max_v = max(max_v, v);
            };
            $if (finite != 0u) {
                Var<float> sum = def(0.0f);
                $for (e, experts) { sum += exp(logits.read(base + e) - max_v); };
                // Explicit zero-sum guard (defensive; matches the host oracle).
                $if (sum > 0.0f) {
                    $for (e, experts) {
                        probs.write(base + e, exp(logits.read(base + e) - max_v) / sum);
                    };
                } $else {
                    $for (e, experts) { probs.write(base + e, 0.0f); };
                };
            } $else {
                $for (e, experts) { probs.write(base + e, 0.0f); };
            };
        };
    };
    return kernel;
}

[[nodiscard]] MoeTopKKernel make_moe_topk_kernel() noexcept {
    MoeTopKKernel kernel = [](BufferFloat probs, BufferUInt ids, BufferFloat weights,
                              Var<uint> tokens, Var<uint> experts, Var<uint> top_k) noexcept {
        set_block_size(256u, 1u, 1u);
        set_name("ocr_moe_topk");
        const auto t = ocr_linear_id();
        $if (t < tokens & experts > 0u & top_k > 0u & top_k <= experts) {
            const auto base = t * experts;
            const auto wbase = t * top_k;
            Var<float> wsum = def(0.0f);
            $for (j, top_k) {
                Var<uint> best = def(0u);
                Var<float> best_v = def(-1.0f);
                $for (e, experts) {
                    // Exclude experts already chosen for this token.
                    Var<uint> taken = def(0u);
                    $for (prev, j) {
                        $if (ids.read(wbase + prev) == e) { taken = 1u; };
                    };
                    $if (taken == 0u) {
                        const auto v = probs.read(base + e);
                        // Strict >: the FIRST (lowest-index) expert wins ties.
                        $if (v > best_v) {
                            best_v = v;
                            best = e;
                        };
                    };
                };
                ids.write(wbase + j, best);
                weights.write(wbase + j, best_v);
                wsum += best_v;
            };
            // Normalize the selected weights (defensive zero guard).
            $if (wsum > 0.0f) {
                $for (j, top_k) { weights.write(wbase + j, weights.read(wbase + j) / wsum); };
            } $else {
                $for (j, top_k) { weights.write(wbase + j, 0.0f); };
            };
        };
    };
    return kernel;
}

[[nodiscard]] MoeHistogramKernel make_moe_histogram_kernel() noexcept {
    MoeHistogramKernel kernel = [](BufferUInt ids, BufferUInt counts,
                                   Var<uint> slots, Var<uint> experts) noexcept {
        set_block_size(256u, 1u, 1u);
        set_name("ocr_moe_histogram");
        const auto e = ocr_linear_id();
        $if (e < experts) {
            Var<uint> c = def(0u);
            $for (s, slots) {
                $if (ids.read(s) == e) { c += 1u; };
            };
            counts.write(e, c);
        };
    };
    return kernel;
}

[[nodiscard]] MoeOffsetsKernel make_moe_offsets_kernel() noexcept {
    MoeOffsetsKernel kernel = [](BufferUInt counts, BufferUInt offsets, Var<uint> experts) noexcept {
        // Valid CUDA block size (multiple of 32); only thread 0 does the serial
        // exclusive-prefix scan (experts is small, e.g. 64).
        set_block_size(32u, 1u, 1u);
        set_name("ocr_moe_offsets");
        $if (ocr_linear_id() == 0u) {
            Var<uint> acc = def(0u);
            $for (e, experts) {
                offsets.write(e, acc);
                acc += counts.read(e);
            };
        };
    };
    return kernel;
}

[[nodiscard]] MoePermutationKernel make_moe_permutation_kernel() noexcept {
    MoePermutationKernel kernel = [](BufferUInt ids, BufferUInt offsets, BufferUInt perm_slot,
                                     BufferUInt slot_pos, BufferUInt perm_token,
                                     Var<uint> slots, Var<uint> experts, Var<uint> top_k) noexcept {
        set_block_size(256u, 1u, 1u);
        set_name("ocr_moe_permutation");
        const auto e = ocr_linear_id();
        $if (e < experts & top_k > 0u) {
            Var<uint> run = def(offsets.read(e));
            // Ascending slot order => stable within an expert.
            $for (s, slots) {
                $if (ids.read(s) == e) {
                    perm_slot.write(run, s);
                    slot_pos.write(s, run);
                    perm_token.write(run, s / top_k);
                    run += 1u;
                };
            };
        };
    };
    return kernel;
}

[[nodiscard]] MoeGatherKernel make_moe_gather_kernel() noexcept {
    MoeGatherKernel kernel = [](BufferFloat x, BufferUInt perm_token, BufferFloat gathered,
                                Var<uint> slots, Var<uint> hidden) noexcept {
        set_block_size(256u, 1u, 1u);
        set_name("ocr_moe_gather");
        const auto id = ocr_linear_id();
        $if (id < slots * hidden) {
            const auto s = id / hidden;
            const auto d = id % hidden;
            const auto token = perm_token.read(s);
            gathered.write(id, x.read(token * hidden + d));
        };
    };
    return kernel;
}

[[nodiscard]] MoeScatterKernel make_moe_scatter_kernel() noexcept {
    MoeScatterKernel kernel = [](BufferFloat expert_out, BufferUInt slot_pos, BufferFloat weights,
                                 BufferFloat shared0, BufferFloat shared1, BufferFloat out,
                                 Var<uint> tokens, Var<uint> hidden, Var<uint> top_k) noexcept {
        set_block_size(256u, 1u, 1u);
        set_name("ocr_moe_scatter");
        const auto id = ocr_linear_id();
        $if (id < tokens * hidden) {
            const auto t = id / hidden;
            const auto d = id % hidden;
            Var<float> acc = shared0.read(id) + shared1.read(id);
            $for (j, top_k) {
                const auto slot = t * top_k + j;
                const auto dst = slot_pos.read(slot);
                acc += weights.read(slot) * expert_out.read(dst * hidden + d);
            };
            out.write(id, acc);
        };
    };
    return kernel;
}

// ---------------------------------------------------------------------------
// Dispatch helpers
// ---------------------------------------------------------------------------

[[nodiscard]] Result<LaunchPlan1D> plan_moe_router_dispatch(uint64_t tokens) { return plan_threads(tokens); }
[[nodiscard]] Result<LaunchPlan1D> plan_moe_topk_dispatch(uint64_t tokens) { return plan_threads(tokens); }
[[nodiscard]] Result<LaunchPlan1D> plan_moe_histogram_dispatch(uint64_t experts) { return plan_threads(experts); }
[[nodiscard]] Result<LaunchPlan1D> plan_moe_offsets_dispatch(uint64_t experts) {
    // Single-thread prefix; one work item regardless of expert count.
    return plan_threads(experts == 0u ? 0u : 1u);
}
[[nodiscard]] Result<LaunchPlan1D> plan_moe_permutation_dispatch(uint64_t experts) { return plan_threads(experts); }
[[nodiscard]] Result<LaunchPlan1D> plan_moe_gather_dispatch(uint64_t slots, uint64_t hidden) {
    uint64_t count = 0;
    if (!checked_mul(slots, hidden, count)) {
        return Result<LaunchPlan1D>::failure(StatusCode::out_of_memory, "gather element count overflows", "slots*hidden");
    }
    return plan_threads(count);
}
[[nodiscard]] Result<LaunchPlan1D> plan_moe_scatter_dispatch(uint64_t tokens, uint64_t hidden) {
    uint64_t count = 0;
    if (!checked_mul(tokens, hidden, count)) {
        return Result<LaunchPlan1D>::failure(StatusCode::out_of_memory, "scatter element count overflows", "tokens*hidden");
    }
    return plan_threads(count);
}

// ---------------------------------------------------------------------------
// Independent host oracles (double accumulation)
// ---------------------------------------------------------------------------

[[nodiscard]] Status ref_moe_softmax(luisa::span<const float> logits, uint32_t tokens, uint32_t experts,
                                     luisa::vector<float> &probs) {
    if (experts == 0u) {
        return Status::error(StatusCode::invalid_argument, "softmax experts must be nonzero", "experts");
    }
    const uint64_t need = static_cast<uint64_t>(tokens) * experts;
    if (logits.size() != need) {
        return Status::error(StatusCode::invalid_argument, "softmax logits size mismatch", "logits");
    }
    probs.resize(static_cast<size_t>(need));
    for (uint32_t t = 0; t < tokens; t++) {
        const size_t base = static_cast<size_t>(t) * experts;
        double max_v = -std::numeric_limits<double>::infinity();
        bool finite = true;
        for (uint32_t e = 0; e < experts; e++) {
            const double v = logits[base + e];
            if (!std::isfinite(v)) { finite = false; }
            max_v = std::max(max_v, v);
        }
        if (!finite) {
            for (uint32_t e = 0; e < experts; e++) { probs[base + e] = 0.0f; }
            continue;
        }
        double sum = 0.0;
        for (uint32_t e = 0; e < experts; e++) { sum += std::exp(static_cast<double>(logits[base + e]) - max_v); }
        // Explicit zero-sum guard, identical to the device kernel's defensive
        // branch (a finite row always has sum >= 1 because max_v is a real
        // element, so this only fires for degenerate/underflowing input).
        if (!(sum > 0.0)) {
            for (uint32_t e = 0; e < experts; e++) { probs[base + e] = 0.0f; }
            continue;
        }
        for (uint32_t e = 0; e < experts; e++) {
            probs[base + e] = static_cast<float>(std::exp(static_cast<double>(logits[base + e]) - max_v) / sum);
        }
    }
    return Status::success();
}

[[nodiscard]] Status ref_moe_topk(luisa::span<const float> probs, uint32_t tokens, uint32_t experts,
                                  uint32_t top_k, luisa::vector<uint32_t> &ids, luisa::vector<float> &weights) {
    if (experts == 0u || top_k == 0u || top_k > experts) {
        return Status::error(StatusCode::invalid_argument, "invalid top-k geometry", "top_k");
    }
    const uint64_t need = static_cast<uint64_t>(tokens) * experts;
    if (probs.size() != need) {
        return Status::error(StatusCode::invalid_argument, "topk probs size mismatch", "probs");
    }
    const size_t slots = static_cast<size_t>(tokens) * top_k;
    ids.resize(slots);
    weights.resize(slots);
    for (uint32_t t = 0; t < tokens; t++) {
        const size_t base = static_cast<size_t>(t) * experts;
        const size_t wbase = static_cast<size_t>(t) * top_k;
        luisa::vector<bool> taken(experts, false);
        double wsum = 0.0;
        for (uint32_t j = 0; j < top_k; j++) {
            uint32_t best = 0;
            double best_v = -1.0;
            for (uint32_t e = 0; e < experts; e++) {
                if (taken[e]) { continue; }
                const double v = probs[base + e];
                if (v > best_v) {
                    best_v = v;
                    best = e;
                }
            }
            taken[best] = true;
            ids[wbase + j] = best;
            weights[wbase + j] = static_cast<float>(best_v);
            wsum += best_v;
        }
        if (wsum > 0.0) {
            for (uint32_t j = 0; j < top_k; j++) {
                weights[wbase + j] = static_cast<float>(static_cast<double>(weights[wbase + j]) / wsum);
            }
        } else {
            for (uint32_t j = 0; j < top_k; j++) { weights[wbase + j] = 0.0f; }
        }
    }
    return Status::success();
}

[[nodiscard]] Status ref_moe_route(luisa::span<const float> x, luisa::span<const float> router_w,
                                   uint32_t tokens, uint32_t hidden, const MoeRouterConfig &router,
                                   MoeRouting &out) {
    auto s = validate_moe_router_inputs(tokens, hidden, router);
    if (!s.ok()) { return s; }
    const uint32_t experts = router.num_experts, top_k = router.top_k;
    const uint64_t slots64 = static_cast<uint64_t>(tokens) * top_k;
    // Router projection: logits[t,e] = sum_h x[t,h] * router_w[h,e] (double acc).
    luisa::vector<float> logits;
    s = ref_gemm(x, router_w, {}, tokens, experts, hidden, RefGemmEpilogue::none, logits);
    if (!s.ok()) { return s; }
    out.logits = logits;
    s = ref_moe_softmax(logits, tokens, experts, out.probs);
    if (!s.ok()) { return s; }
    s = ref_moe_topk(out.probs, tokens, experts, top_k, out.ids, out.weights);
    if (!s.ok()) { return s; }
    // Histogram.
    out.counts.assign(experts, 0u);
    for (uint64_t slot = 0; slot < slots64; slot++) { out.counts[out.ids[static_cast<size_t>(slot)]]++; }
    // Exclusive-prefix offsets.
    out.offsets.resize(experts);
    uint32_t acc = 0;
    for (uint32_t e = 0; e < experts; e++) {
        out.offsets[e] = acc;
        acc += out.counts[e];
    }
    // Permutation (stable within expert by ascending slot).
    const size_t slots = static_cast<size_t>(slots64);
    out.perm_slot.resize(slots);
    out.slot_pos.resize(slots);
    out.perm_token.resize(slots);
    luisa::vector<uint32_t> run(experts);
    for (uint32_t e = 0; e < experts; e++) { run[e] = out.offsets[e]; }
    for (size_t slot = 0; slot < slots; slot++) {
        const uint32_t e = out.ids[slot];
        const uint32_t dst = run[e]++;
        out.perm_slot[dst] = static_cast<uint32_t>(slot);
        out.slot_pos[slot] = dst;
        out.perm_token[dst] = static_cast<uint32_t>(slot / top_k);
    }
    return Status::success();
}

[[nodiscard]] Status ref_moe_gather(luisa::span<const float> x, luisa::span<const uint32_t> perm_token,
                                    uint32_t slots, uint32_t hidden, luisa::vector<float> &out) {
    if (hidden == 0u) {
        return Status::error(StatusCode::invalid_argument, "gather hidden must be nonzero", "hidden");
    }
    if (perm_token.size() != slots) {
        return Status::error(StatusCode::invalid_argument, "gather perm_token size mismatch", "perm_token");
    }
    // M5: validate the source extent AND every index BEFORE reading, so a
    // malformed permutation can never read past x.
    if (x.size() % hidden != 0u) {
        return Status::error(StatusCode::invalid_argument, "gather x size is not a whole number of rows", "x");
    }
    const uint64_t tokens = x.size() / hidden;
    for (size_t s = 0; s < perm_token.size(); s++) {
        if (static_cast<uint64_t>(perm_token[s]) >= tokens) {
            return Status::error(StatusCode::invalid_argument,
                                 luisa::format("gather perm_token[{}] = {} is outside the {} source rows",
                                               s, perm_token[s], tokens),
                                 "perm_token");
        }
    }
    out.resize(static_cast<size_t>(slots) * hidden);
    for (uint32_t s = 0; s < slots; s++) {
        const uint32_t token = perm_token[s];
        for (uint32_t d = 0; d < hidden; d++) {
            out[static_cast<size_t>(s) * hidden + d] = x[static_cast<size_t>(token) * hidden + d];
        }
    }
    return Status::success();
}

[[nodiscard]] Status ref_moe_expert_ffn(luisa::span<const float> x, luisa::span<const float> gate_w,
                                        luisa::span<const float> up_w, luisa::span<const float> down_w,
                                        uint32_t rows, uint32_t hidden, uint32_t inter,
                                        luisa::vector<float> &out) {
    luisa::vector<float> gate, up, act;
    auto s = ref_gemm(x, gate_w, {}, rows, inter, hidden, RefGemmEpilogue::none, gate);
    if (!s.ok()) { return s; }
    s = ref_gemm(x, up_w, {}, rows, inter, hidden, RefGemmEpilogue::none, up);
    if (!s.ok()) { return s; }
    s = ref_gated_act(gate, up, 0u, act);// 0 = silu (SwiGLU)
    if (!s.ok()) { return s; }
    return ref_gemm(act, down_w, {}, rows, hidden, inter, RefGemmEpilogue::none, out);
}

[[nodiscard]] Status ref_moe_scatter(luisa::span<const float> expert_out, luisa::span<const uint32_t> slot_pos,
                                     luisa::span<const float> weights, luisa::span<const float> shared0,
                                     luisa::span<const float> shared1, uint32_t tokens, uint32_t hidden,
                                     uint32_t top_k, luisa::vector<float> &out) {
    if (hidden == 0u || top_k == 0u) {
        return Status::error(StatusCode::invalid_argument, "scatter hidden/top_k must be nonzero", "hidden/top_k");
    }
    const size_t th = static_cast<size_t>(tokens) * hidden;
    const size_t slots = static_cast<size_t>(tokens) * top_k;
    if (slot_pos.size() != slots || weights.size() != slots || shared0.size() != th || shared1.size() != th) {
        return Status::error(StatusCode::invalid_argument, "scatter input size mismatch", "scatter");
    }
    // M6: validate the expert_out extent AND every destination slot BEFORE
    // reading, so a malformed inverse map can never read past expert_out.
    if (expert_out.size() != slots * hidden) {
        return Status::error(StatusCode::invalid_argument,
                             "scatter expert_out size does not match slots*hidden", "expert_out");
    }
    for (size_t slot = 0; slot < slots; slot++) {
        if (static_cast<size_t>(slot_pos[slot]) >= slots) {
            return Status::error(StatusCode::invalid_argument,
                                 luisa::format("scatter slot_pos[{}] = {} is outside the {} slots",
                                               slot, slot_pos[slot], slots),
                                 "slot_pos");
        }
    }
    out.resize(th);
    for (uint32_t t = 0; t < tokens; t++) {
        for (uint32_t d = 0; d < hidden; d++) {
            const size_t idx = static_cast<size_t>(t) * hidden + d;
            double acc = static_cast<double>(shared0[idx]) + static_cast<double>(shared1[idx]);
            for (uint32_t j = 0; j < top_k; j++) {
                const size_t slot = static_cast<size_t>(t) * top_k + j;
                const uint32_t dst = slot_pos[slot];
                acc += static_cast<double>(weights[slot]) * expert_out[static_cast<size_t>(dst) * hidden + d];
            }
            out[idx] = static_cast<float>(acc);
        }
    }
    return Status::success();
}

[[nodiscard]] Status ref_moe_layer(luisa::span<const float> x, const MoeRouting &routing,
                                   luisa::span<const MoeExpertWeights> routed,
                                   const MoeExpertWeights &shared0, const MoeExpertWeights &shared1,
                                   uint32_t tokens, uint32_t hidden, uint32_t inter, uint32_t shared_inter,
                                   luisa::vector<float> &expert_out, luisa::vector<float> &weighted,
                                   luisa::vector<float> &shared0_out, luisa::vector<float> &shared1_out,
                                   luisa::vector<float> &out) {
    // M7: reject an empty token batch BEFORE dividing ids.size() by tokens.
    if (tokens == 0u) {
        return Status::error(StatusCode::invalid_argument, "MoE layer tokens must be nonzero", "tokens");
    }
    if (hidden == 0u || inter == 0u || shared_inter == 0u) {
        return Status::error(StatusCode::invalid_argument, "MoE layer dims must be nonzero", "hidden/inter");
    }
    const uint32_t experts = static_cast<uint32_t>(routing.counts.size());
    if (routing.ids.size() % tokens != 0u) {
        return Status::error(StatusCode::invalid_argument,
                             "routing ids size is not a whole number of token rows", "ids");
    }
    const uint32_t top_k = static_cast<uint32_t>(routing.ids.size() / tokens);
    if (top_k == 0u || experts == 0u || top_k > experts) {
        return Status::error(StatusCode::invalid_argument, "invalid routing geometry", "top_k");
    }
    if (routing.weights.size() != routing.ids.size() || routing.offsets.size() != experts ||
        routing.slot_pos.size() != routing.ids.size() || routing.perm_token.size() != routing.ids.size()) {
        return Status::error(StatusCode::invalid_argument, "routing array sizes are inconsistent", "routing");
    }
    if (routed.size() != experts) {
        return Status::error(StatusCode::invalid_argument, "routed expert count mismatch", "routed");
    }
    const size_t slots = routing.ids.size();
    // Validate the histogram/offsets with the SAME shared validator the device
    // execution path uses, before slicing gathered rows.
    if (auto vs = validate_moe_routing_tables(routing.counts, routing.offsets, slots); !vs.ok()) { return vs; }
    // Gather token rows into expert-sorted order.
    luisa::vector<float> gathered;
    auto s = ref_moe_gather(x, routing.perm_token, static_cast<uint32_t>(slots), hidden, gathered);
    if (!s.ok()) { return s; }
    // Loop over ACTIVE experts only.
    expert_out.assign(slots * hidden, 0.0f);
    for (uint32_t e = 0; e < experts; e++) {
        const uint32_t count = routing.counts[e];
        if (count == 0u) { continue; }
        const uint32_t off = routing.offsets[e];
        const size_t gbase = static_cast<size_t>(off) * hidden;
        luisa::vector<float> ffn_out;
        s = ref_moe_expert_ffn(luisa::span<const float>{gathered.data() + gbase, static_cast<size_t>(count) * hidden},
                               routed[e].gate_w, routed[e].up_w, routed[e].down_w,
                               count, hidden, inter, ffn_out);
        if (!s.ok()) { return s; }
        for (size_t i = 0; i < ffn_out.size(); i++) { expert_out[gbase + i] = ffn_out[i]; }
    }
    // Per-slot WEIGHTED contribution in TOKEN order (for contribution parity).
    weighted.assign(slots * hidden, 0.0f);
    for (size_t slot = 0; slot < slots; slot++) {
        const uint32_t dst = routing.slot_pos[slot];
        const float w = routing.weights[slot];
        for (uint32_t d = 0; d < hidden; d++) {
            weighted[slot * hidden + d] = w * expert_out[static_cast<size_t>(dst) * hidden + d];
        }
    }
    // Two shared dense experts over ALL token rows.
    s = ref_moe_expert_ffn(x, shared0.gate_w, shared0.up_w, shared0.down_w, tokens, hidden, shared_inter, shared0_out);
    if (!s.ok()) { return s; }
    s = ref_moe_expert_ffn(x, shared1.gate_w, shared1.up_w, shared1.down_w, tokens, hidden, shared_inter, shared1_out);
    if (!s.ok()) { return s; }
    // Weighted restore + shared.
    return ref_moe_scatter(expert_out, routing.slot_pos, routing.weights, shared0_out, shared1_out,
                           tokens, hidden, top_k, out);
}

}// namespace luisa::ocr

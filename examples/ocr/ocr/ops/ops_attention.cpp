// OCR engine -- attention operator implementations (Phase 2).
#include "ocr/ops/ops_attention.h"

#include <algorithm>

#include <luisa/dsl/sugar.h>

#include "ocr/ops/ops_common.h"
#include "ocr/quantization.h"
#include "ocr/runtime/shader_cache.h"

namespace luisa::ocr {

using namespace luisa::compute;

[[nodiscard]] luisa::string_view ops_attention_mask_kind_name(AttentionMaskKind kind) noexcept {
    switch (kind) {
        case AttentionMaskKind::bidirectional: return "bidirectional";
        case AttentionMaskKind::causal: return "causal";
        case AttentionMaskKind::windowed: return "windowed";
        case AttentionMaskKind::prefix_causal: return "prefix_causal";
        case AttentionMaskKind::custom_block_mask: return "custom_block_mask";
        default: return "unknown";
    }
}

[[nodiscard]] Status AttentionMaskDescriptor::validate(uint32_t q_len, uint32_t k_len, uint32_t q_heads) const {
    switch (kind) {
        case AttentionMaskKind::bidirectional:
        case AttentionMaskKind::causal: break;
        case AttentionMaskKind::windowed:
            if (window == 0u) {
                return Status::error(StatusCode::invalid_argument, "windowed mask requires window > 0", "window");
            }
            break;
        case AttentionMaskKind::prefix_causal: break;
        case AttentionMaskKind::custom_block_mask: {
            if (block_size == 0u) {
                return Status::error(StatusCode::invalid_argument, "custom block mask requires block_size > 0", "block_size");
            }
            const uint64_t qb = (static_cast<uint64_t>(q_len) + block_size - 1u) / block_size;
            const uint64_t kb = (static_cast<uint64_t>(k_len) + block_size - 1u) / block_size;
            uint64_t need = 0;
            if (!checked_mul(qb, kb, need) || block_mask.size() != need) {
                return Status::error(StatusCode::invalid_argument, "block_mask size does not match the block grid", "block_mask");
            }
            break;
        }
        default:
            return Status::error(StatusCode::invalid_argument, "unknown attention mask kind", "kind");
    }
    if (!rel_bias.empty()) {
        uint64_t need = 0;
        if (!checked_mul(static_cast<uint64_t>(q_heads) * q_len, k_len, need) || rel_bias.size() != need) {
            return Status::error(StatusCode::invalid_argument, "rel_bias size does not match [q_heads,q_len,k_len]", "rel_bias");
        }
    }
    return Status::success();
}

[[nodiscard]] Status validate_attention_decode_inputs(luisa::span<const uint32_t> kv_len,
                                                      uint32_t kv_capacity,
                                                      const AttentionMaskDescriptor &mask) {
    for (auto len : kv_len) {
        if (len > kv_capacity) {
            return Status::error(StatusCode::invalid_argument,
                                 "kv_len exceeds kv_capacity", "kv_len");
        }
    }
    if (mask.kind == AttentionMaskKind::custom_block_mask) {
        if (mask.block_size == 0u) {
            return Status::error(StatusCode::invalid_argument,
                                 "custom block mask requires block_size > 0", "block_size");
        }
        // The query block index is position/block_size and is unbounded at
        // decode time; the mask must cover the caller's maximum position.
        // Minimum: one query block row over the full key-block grid.
        const uint64_t k_blocks = (static_cast<uint64_t>(kv_capacity) + mask.block_size - 1u) / mask.block_size;
        if (mask.block_mask.size() < k_blocks || mask.block_mask.size() % k_blocks != 0u) {
            return Status::error(StatusCode::invalid_argument,
                                 "decode block_mask must be a whole number of query-block rows over the key-block grid",
                                 "block_mask");
        }
    }
    if (mask.kind == AttentionMaskKind::windowed && mask.window == 0u) {
        return Status::error(StatusCode::invalid_argument, "windowed mask requires window > 0", "window");
    }
    return Status::success();
}

[[nodiscard]] Status validate_attention_decode_inputs(luisa::span<const uint32_t> kv_len,
                                                      luisa::span<const int32_t> positions,
                                                      uint32_t kv_capacity,
                                                      const AttentionMaskDescriptor &mask) {
    auto base = validate_attention_decode_inputs(kv_len, kv_capacity, mask);
    if (!base.ok()) { return base; }
    if (positions.size() != kv_len.size()) {
        return Status::error(StatusCode::invalid_argument,
                             "positions and kv_len must have the same batch size", "positions");
    }
    if (mask.kind == AttentionMaskKind::custom_block_mask && mask.block_size != 0u) {
        // The mask must cover the query block of the maximum NONNEGATIVE
        // position; negative positions are fail-safe masked in-kernel.
        int32_t max_pos = -1;
        for (auto p : positions) { max_pos = std::max(max_pos, p); }
        if (max_pos >= 0) {
            const uint64_t k_blocks = (static_cast<uint64_t>(kv_capacity) + mask.block_size - 1u) / mask.block_size;
            const uint64_t qb = static_cast<uint64_t>(max_pos) / mask.block_size;
            uint64_t need = 0;
            if (!checked_mul(qb + 1u, k_blocks, need) || need > mask.block_mask.size()) {
                return Status::error(StatusCode::invalid_argument,
                                     "decode block_mask does not cover the maximum position's query block",
                                     "block_mask");
            }
        }
    }
    return Status::success();
}

[[nodiscard]] luisa::vector<uint32_t> pack_block_mask_words(luisa::span<const uint8_t> mask) {
    luisa::vector<uint32_t> words((mask.size() + 3u) / 4u, 0u);
    for (size_t i = 0; i < mask.size(); i++) {
        words[i / 4u] |= static_cast<uint32_t>(mask[i]) << (8u * (i % 4u));
    }
    return words;
}

namespace {

// Mask predicate for prefill: may query qi attend key ki? Mirrors the host
// oracle (cpu_reference.cpp mask_allows) but as device control flow.
[[nodiscard]] Var<bool> prefill_mask_allows(Var<uint> qi, Var<uint> ki, Var<uint> q_len, Var<uint> k_len,
                                            Var<uint> kind, Var<uint> window, Var<uint> prefix_len,
                                            Var<uint> block_size, const BufferUInt &block_mask) noexcept {
    // causal diagonal aligned to the end of the key sequence.
    const auto offset = cast<int>(k_len) - cast<int>(q_len);
    const auto q_abs = cast<int>(qi) + offset;
    const auto kk = cast<int>(ki);
    Var<bool> allowed = def(true);
    $if (kind == static_cast<uint>(AttentionMaskKind::causal)) {
        allowed = kk <= q_abs;
    } $elif (kind == static_cast<uint>(AttentionMaskKind::windowed)) {
        allowed = (kk <= q_abs) & (kk > q_abs - cast<int>(window));
    } $elif (kind == static_cast<uint>(AttentionMaskKind::prefix_causal)) {
        allowed = (ki < prefix_len) | (kk <= q_abs);
    } $elif (kind == static_cast<uint>(AttentionMaskKind::custom_block_mask)) {
        $if (block_size == 0u) {
            allowed = false;
        } $else {
            const auto qb = qi / block_size;
            const auto kb = ki / block_size;
            const auto k_blocks = (k_len + block_size - 1u) / block_size;
            const auto idx = qb * k_blocks + kb;
            const auto word = block_mask.read(idx >> 2u);
            const auto byte = (word >> ((idx & 3u) * 8u)) & 0xFFu;
            allowed = byte != 0u;
        };
    };
    return allowed;
}

// Decode mask predicate: query at absolute position `pos`, key index ki.
// Fail-safe: only the four known non-custom kinds can allow by default;
// CUSTOM_BLOCK_MASK reads the caller mask (block_size==0 -> nothing allowed);
// any unknown kind allows nothing (never silently bidirectional).
[[nodiscard]] Var<bool> decode_mask_allows(Var<uint> ki, Var<int> pos, Var<uint> kind, Var<uint> window,
                                           Var<uint> prefix_len, Var<uint> block_size,
                                           Var<uint> kv_capacity, Var<uint> mask_bytes,
                                           const BufferUInt &block_mask) noexcept {
    const auto kk = cast<int>(ki);
    Var<bool> allowed = def(false);
    $if (kind == static_cast<uint>(AttentionMaskKind::bidirectional)) {
        allowed = true;
    } $elif (kind == static_cast<uint>(AttentionMaskKind::causal)) {
        allowed = kk <= pos;
    } $elif (kind == static_cast<uint>(AttentionMaskKind::windowed)) {
        allowed = (kk <= pos) & (kk > pos - cast<int>(window));
    } $elif (kind == static_cast<uint>(AttentionMaskKind::prefix_causal)) {
        allowed = (ki < prefix_len) | (kk <= pos);
    } $elif (kind == static_cast<uint>(AttentionMaskKind::custom_block_mask)) {
        $if (block_size != 0u & pos >= 0) {
            const auto qb = cast<uint>(pos) / block_size;
            const auto kb = ki / block_size;
            const auto k_blocks = (kv_capacity + block_size - 1u) / block_size;
            const auto idx = qb * k_blocks + kb;
            // OOB guard: a mask index beyond the bound byte count is
            // disallowed WITHOUT reading (large positions stay fail-safe).
            $if (idx < mask_bytes) {
                const auto word = block_mask.read(idx >> 2u);
                const auto byte = (word >> ((idx & 3u) * 8u)) & 0xFFu;
                allowed = byte != 0u;
            };
        };
    };
    return allowed;
}

// One QK dot score with optional softcap + relative bias.
[[nodiscard]] Var<float> qk_score(const BufferFloat &q, const BufferFloat &k, Var<uint> q_base, Var<uint> k_base,
                                  Var<uint> head_dim, Var<float> scale, Var<float> softcap) noexcept {
    Var<float> dot = def(0.0f);
    $for (d, head_dim) { dot += q.read(q_base + d) * k.read(k_base + d); };
    Var<float> score = dot * scale;
    $if (softcap > 0.0f) { score = softcap * tanh(score / softcap); };
    return score;
}

// Reference prefill: one thread per (query, head). Two passes over keys.
void attention_prefill_reference(const BufferFloat &q, const BufferFloat &k, const BufferFloat &v, const BufferFloat &out,
                                 const BufferUInt &block_mask, const BufferFloat &rel_bias,
                                 Var<uint> q_len, Var<uint> k_len, Var<uint> q_heads, Var<uint> kv_heads,
                                 Var<uint> head_dim, Var<float> scale, Var<float> softcap,
                                 Var<uint> kind, Var<uint> window, Var<uint> prefix_len,
                                 Var<uint> block_size, Var<uint> has_rel_bias) noexcept {
    const auto row = ocr_linear_id();
    $if (row < q_len * q_heads) {
        const auto qi = row / q_heads;
        const auto h = row % q_heads;
        const auto group = q_heads / kv_heads;
        const auto kvh = h / group;
        const auto q_base = (qi * q_heads + h) * head_dim;
        const auto o_base = (qi * q_heads + h) * head_dim;
        // Deterministic zero for a fully-masked row.
        $for (d, head_dim) { out.write(o_base + d, 0.0f); };
        // Pass 1: max score over allowed keys.
        Var<float> max_score = def(-3.0e38f);
        Var<uint> any = def(0u);
        $for (ki, k_len) {
            $if (prefill_mask_allows(qi, ki, q_len, k_len, kind, window, prefix_len, block_size, block_mask)) {
                const auto k_base = (ki * kv_heads + kvh) * head_dim;
                auto score = qk_score(q, k, q_base, k_base, head_dim, scale, softcap);
                $if (has_rel_bias != 0u) { score += rel_bias.read((h * q_len + qi) * k_len + ki); };
                max_score = max(max_score, score);
                any = 1u;
            };
        };
            $if (any != 0u) {
            // Pass 2: sum of exp and weighted V accumulation.
            Var<float> sum = def(0.0f);
            $for (ki, k_len) {
                $if (prefill_mask_allows(qi, ki, q_len, k_len, kind, window, prefix_len, block_size, block_mask)) {
                    const auto k_base = (ki * kv_heads + kvh) * head_dim;
                    auto score = qk_score(q, k, q_base, k_base, head_dim, scale, softcap);
                    $if (has_rel_bias != 0u) { score += rel_bias.read((h * q_len + qi) * k_len + ki); };
                    sum += exp(score - max_score);
                };
            };
            $for (d, head_dim) {
                Var<float> acc = def(0.0f);
                $for (ki, k_len) {
                    $if (prefill_mask_allows(qi, ki, q_len, k_len, kind, window, prefix_len, block_size, block_mask)) {
                        const auto k_base = (ki * kv_heads + kvh) * head_dim;
                        auto score = qk_score(q, k, q_base, k_base, head_dim, scale, softcap);
                        $if (has_rel_bias != 0u) { score += rel_bias.read((h * q_len + qi) * k_len + ki); };
                        const auto w = exp(score - max_score) / sum;
                        acc += w * v.read(k_base + d);
                    };
                };
                out.write(o_base + d, acc);
            };
        };
    };
}

// Online prefill: one thread per (query, head). Single pass, running FP32
// max/sum and a running output accumulator rescaled on each new max.
void attention_prefill_online(const BufferFloat &q, const BufferFloat &k, const BufferFloat &v, const BufferFloat &out,
                              const BufferUInt &block_mask, const BufferFloat &rel_bias,
                              Var<uint> q_len, Var<uint> k_len, Var<uint> q_heads, Var<uint> kv_heads,
                              Var<uint> head_dim, Var<float> scale, Var<float> softcap,
                              Var<uint> kind, Var<uint> window, Var<uint> prefix_len,
                              Var<uint> block_size, Var<uint> has_rel_bias) noexcept {
    const auto row = ocr_linear_id();
    $if (row < q_len * q_heads) {
        const auto qi = row / q_heads;
        const auto h = row % q_heads;
        const auto group = q_heads / kv_heads;
        const auto kvh = h / group;
        const auto q_base = (qi * q_heads + h) * head_dim;
        const auto o_base = (qi * q_heads + h) * head_dim;
        Var<float> run_max = def(-3.0e38f);
        Var<float> run_sum = def(0.0f);
        // Running output accumulator (one per head_dim element).
        $for (d, head_dim) { out.write(o_base + d, 0.0f); };
        $for (ki, k_len) {
            $if (prefill_mask_allows(qi, ki, q_len, k_len, kind, window, prefix_len, block_size, block_mask)) {
                const auto k_base = (ki * kv_heads + kvh) * head_dim;
                auto score = qk_score(q, k, q_base, k_base, head_dim, scale, softcap);
                $if (has_rel_bias != 0u) { score += rel_bias.read((h * q_len + qi) * k_len + ki); };
                const auto new_max = max(run_max, score);
                const auto rescale = exp(run_max - new_max);
                const auto p = exp(score - new_max);
                run_sum = run_sum * rescale + p;
                $for (d, head_dim) {
                    const auto prev = out.read(o_base + d);
                    out.write(o_base + d, prev * rescale + p * v.read(k_base + d));
                };
                run_max = new_max;
            };
        };
        $if (run_sum > 0.0f) {
            $for (d, head_dim) { out.write(o_base + d, out.read(o_base + d) / run_sum); };
        };
    };
}

}// namespace

[[nodiscard]] AttentionPrefillKernel make_attention_prefill_reference_kernel() noexcept {
    AttentionPrefillKernel kernel = [](BufferFloat q, BufferFloat k, BufferFloat v, BufferFloat out,
                                       BufferUInt block_mask, BufferFloat rel_bias,
                                       Var<uint> q_len, Var<uint> k_len, Var<uint> q_heads, Var<uint> kv_heads,
                                       Var<uint> head_dim, Var<float> scale, Var<float> softcap,
                                       Var<uint> kind, Var<uint> window, Var<uint> prefix_len,
                                       Var<uint> block_size, Var<uint> has_rel_bias) noexcept {
        set_block_size(256u, 1u, 1u);
        set_name("ocr_attention_prefill_reference");
        attention_prefill_reference(q, k, v, out, block_mask, rel_bias, q_len, k_len, q_heads, kv_heads,
                                    head_dim, scale, softcap, kind, window, prefix_len, block_size, has_rel_bias);
    };
    return kernel;
}

[[nodiscard]] AttentionPrefillKernel make_attention_prefill_online_kernel() noexcept {
    AttentionPrefillKernel kernel = [](BufferFloat q, BufferFloat k, BufferFloat v, BufferFloat out,
                                       BufferUInt block_mask, BufferFloat rel_bias,
                                       Var<uint> q_len, Var<uint> k_len, Var<uint> q_heads, Var<uint> kv_heads,
                                       Var<uint> head_dim, Var<float> scale, Var<float> softcap,
                                       Var<uint> kind, Var<uint> window, Var<uint> prefix_len,
                                       Var<uint> block_size, Var<uint> has_rel_bias) noexcept {
        set_block_size(256u, 1u, 1u);
        set_name("ocr_attention_prefill_online");
        attention_prefill_online(q, k, v, out, block_mask, rel_bias, q_len, k_len, q_heads, kv_heads,
                                 head_dim, scale, softcap, kind, window, prefix_len, block_size, has_rel_bias);
    };
    return kernel;
}

[[nodiscard]] AttentionDecodeKernel make_attention_decode_kernel() noexcept {
    AttentionDecodeKernel kernel = [](BufferFloat q, BufferFloat k, BufferFloat v, BufferFloat out,
                                      BufferUInt kv_len, BufferInt positions, BufferUInt block_mask,
                                      Var<uint> batch, Var<uint> kv_capacity, Var<uint> q_heads, Var<uint> kv_heads,
                                      Var<uint> head_dim, Var<float> scale, Var<float> softcap,
                                      Var<uint> kind, Var<uint> window, Var<uint> prefix_len,
                                      Var<uint> block_size, Var<uint> mask_bytes) noexcept {
        set_block_size(256u, 1u, 1u);
        set_name("ocr_attention_decode");
        const auto row = ocr_linear_id();
        $if (row < batch * q_heads) {
            const auto b = row / q_heads;
            const auto h = row % q_heads;
            const auto group = q_heads / kv_heads;
            const auto kvh = h / group;
            // S2 defensive clamp: an invalid kv_len can never read OOB.
            const auto len = min(kv_len.read(b), kv_capacity);
            const auto pos = positions.read(b);
            const auto q_base = (b * q_heads + h) * head_dim;
            const auto o_base = (b * q_heads + h) * head_dim;
            // Deterministic zero for a fully-masked / empty-KV row.
            $for (d, head_dim) { out.write(o_base + d, 0.0f); };
            Var<float> max_score = def(-3.0e38f);
            Var<uint> any = def(0u);
            $for (ki, len) {
                $if (decode_mask_allows(ki, pos, kind, window, prefix_len, block_size, kv_capacity, mask_bytes, block_mask)) {
                    const auto k_base = ((b * kv_capacity + ki) * kv_heads + kvh) * head_dim;
                    const auto score = qk_score(q, k, q_base, k_base, head_dim, scale, softcap);
                    max_score = max(max_score, score);
                    any = 1u;
                };
            };
            $if (any != 0u) {
                Var<float> sum = def(0.0f);
                $for (ki, len) {
                    $if (decode_mask_allows(ki, pos, kind, window, prefix_len, block_size, kv_capacity, mask_bytes, block_mask)) {
                        const auto k_base = ((b * kv_capacity + ki) * kv_heads + kvh) * head_dim;
                        const auto score = qk_score(q, k, q_base, k_base, head_dim, scale, softcap);
                        sum += exp(score - max_score);
                    };
                };
                $for (d, head_dim) {
                    Var<float> acc = def(0.0f);
                    $for (ki, len) {
                        $if (decode_mask_allows(ki, pos, kind, window, prefix_len, block_size, kv_capacity, mask_bytes, block_mask)) {
                            const auto k_base = ((b * kv_capacity + ki) * kv_heads + kvh) * head_dim;
                            const auto score = qk_score(q, k, q_base, k_base, head_dim, scale, softcap);
                            const auto w = exp(score - max_score) / sum;
                            acc += w * v.read(k_base + d);
                        };
                    };
                    out.write(o_base + d, acc);
                };
            };
        };
    };
    return kernel;
}

[[nodiscard]] Result<LaunchPlan1D> plan_attention_prefill_dispatch(uint64_t q_len, uint64_t q_heads) {
    uint64_t rows = 0;
    if (!checked_mul(q_len, q_heads, rows)) {
        return Result<LaunchPlan1D>::failure(StatusCode::invalid_argument, "attention row count overflow", "q_len*q_heads");
    }
    return plan_threads(rows, 256u);
}

[[nodiscard]] Result<LaunchPlan1D> plan_attention_decode_dispatch(uint64_t batch, uint64_t q_heads) {
    uint64_t rows = 0;
    if (!checked_mul(batch, q_heads, rows)) {
        return Result<LaunchPlan1D>::failure(StatusCode::invalid_argument, "decode row count overflow", "batch*q_heads");
    }
    return plan_threads(rows, 256u);
}

[[nodiscard]] luisa::string attention_variant_key(luisa::string_view variant) {
    return make_shader_key("attention", variant);
}

}// namespace luisa::ocr
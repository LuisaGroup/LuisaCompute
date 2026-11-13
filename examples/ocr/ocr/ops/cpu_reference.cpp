// OCR engine -- independent naive host oracles (Phase 2). Host-only.
#include "ocr/ops/cpu_reference.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace luisa::ocr {

namespace {

[[nodiscard]] double gelu_erf_d(double x) noexcept {
    return 0.5 * x * (1.0 + std::erf(x * 0.70710678118654752440));
}
[[nodiscard]] double gelu_tanh_d(double x) noexcept {
    const double inner = 0.79788456080286535588 * (x + 0.044715 * x * x * x);
    return 0.5 * x * (1.0 + std::tanh(inner));
}
[[nodiscard]] double silu_d(double x) noexcept { return x / (1.0 + std::exp(-x)); }
[[nodiscard]] double hard_swish_d(double x) noexcept {
    return x * std::clamp(x + 3.0, 0.0, 6.0) / 6.0;
}

[[nodiscard]] Status check_size(size_t got, size_t want, const char *what) {
    if (got != want) {
        return Status::error(StatusCode::invalid_argument,
                             luisa::string{what} + " span size mismatch", what);
    }
    return Status::success();
}

}// namespace

// ---------------------------------------------------------------------------
// GEMM
// ---------------------------------------------------------------------------

[[nodiscard]] Status ref_gemm(luisa::span<const float> a, luisa::span<const float> b,
                              luisa::span<const float> aux, uint32_t m, uint32_t n, uint32_t k,
                              RefGemmEpilogue epilogue, luisa::vector<float> &out) {
    auto s = check_size(a.size(), static_cast<size_t>(m) * k, "a");
    if (!s.ok()) { return s; }
    s = check_size(b.size(), static_cast<size_t>(k) * n, "b");
    if (!s.ok()) { return s; }
    const bool needs_bias = epilogue == RefGemmEpilogue::bias ||
                            epilogue == RefGemmEpilogue::bias_gelu_erf ||
                            epilogue == RefGemmEpilogue::bias_gelu_tanh ||
                            epilogue == RefGemmEpilogue::bias_relu ||
                            epilogue == RefGemmEpilogue::bias_relu6 ||
                            epilogue == RefGemmEpilogue::bias_silu ||
                            epilogue == RefGemmEpilogue::bias_hard_swish ||
                            epilogue == RefGemmEpilogue::bias_residual;
    const bool needs_residual = epilogue == RefGemmEpilogue::residual ||
                                epilogue == RefGemmEpilogue::bias_residual;
    if (needs_bias || needs_residual) {
        // bias_residual aux = [bias N | residual M*N]; residual = M*N; bias = N.
        const size_t want = epilogue == RefGemmEpilogue::bias_residual
                                ? static_cast<size_t>(n) + static_cast<size_t>(m) * n
                                : (needs_residual ? static_cast<size_t>(m) * n : n);
        s = check_size(aux.size(), want, "aux");
        if (!s.ok()) { return s; }
    }
    out.assign(static_cast<size_t>(m) * n, 0.0f);
    for (uint32_t i = 0; i < m; i++) {
        for (uint32_t j = 0; j < n; j++) {
            double acc = 0.0;
            for (uint32_t t = 0; t < k; t++) {
                acc += static_cast<double>(a[static_cast<size_t>(i) * k + t]) *
                       static_cast<double>(b[static_cast<size_t>(t) * n + j]);
            }
            double v = acc;
            const size_t idx = static_cast<size_t>(i) * n + j;
            if (needs_bias) { v += aux[j]; }
            switch (epilogue) {
                case RefGemmEpilogue::bias_gelu_erf: v = gelu_erf_d(v); break;
                case RefGemmEpilogue::bias_gelu_tanh: v = gelu_tanh_d(v); break;
                case RefGemmEpilogue::bias_relu: v = std::max(v, 0.0); break;
                case RefGemmEpilogue::bias_relu6: v = std::clamp(v, 0.0, 6.0); break;
                case RefGemmEpilogue::bias_silu: v = silu_d(v); break;
                case RefGemmEpilogue::bias_hard_swish: v = hard_swish_d(v); break;
                default: break;
            }
            if (needs_residual) {
                // bias_residual: residual lives after the N bias entries.
                v += aux[epilogue == RefGemmEpilogue::bias_residual ? n + idx : idx];
            }
            out[idx] = static_cast<float>(v);
        }
    }
    return Status::success();
}

[[nodiscard]] Status ref_gated_act(luisa::span<const float> gate, luisa::span<const float> up,
                                   uint32_t act, luisa::vector<float> &out) {
    auto s = check_size(gate.size(), up.size(), "gate/up");
    if (!s.ok()) { return s; }
    out.resize(gate.size());
    for (size_t i = 0; i < gate.size(); i++) {
        double g = gate[i];
        switch (act) {
            case 0: g = silu_d(g); break;
            case 1: g = gelu_erf_d(g); break;
            case 2: g = gelu_tanh_d(g); break;
            case 3: g = std::max(g, 0.0); break;
            case 4: break;// identity
            default:
                return Status::error(StatusCode::invalid_argument, "unknown gated activation", "act");
        }
        out[i] = static_cast<float>(g * static_cast<double>(up[i]));
    }
    return Status::success();
}

[[nodiscard]] Status ref_dequant_int4(luisa::span<const uint8_t> packed, uint64_t count,
                                      luisa::span<const float> scales, luisa::span<const uint8_t> zero_points,
                                      uint32_t group_size, bool symmetric, luisa::vector<float> &out) {
    // Reuse the Phase 1 codec: it IS the independent host semantics definition.
    out.resize(static_cast<size_t>(count));
    return dequantize_int4(packed, count, scales, zero_points, group_size, symmetric, out);
}

[[nodiscard]] Status ref_dequant_int8(luisa::span<const int8_t> codes, uint64_t count,
                                      luisa::span<const float> scales, uint32_t group_size,
                                      luisa::vector<float> &out) {
    out.resize(static_cast<size_t>(count));
    return dequantize_int8(codes, count, scales, group_size, out);
}

[[nodiscard]] Status ref_round_half(luisa::span<const float> values, bool bf16, luisa::vector<float> &out) {
    out.resize(values.size());
    for (size_t i = 0; i < values.size(); i++) {
        out[i] = bf16 ? bf16_to_fp32(fp32_to_bf16(values[i])) : fp16_to_fp32(fp32_to_fp16(values[i]));
    }
    return Status::success();
}

// ---------------------------------------------------------------------------
// Normalization
// ---------------------------------------------------------------------------

[[nodiscard]] Status ref_layer_norm(luisa::span<const float> x, luisa::span<const float> gamma,
                                    luisa::span<const float> beta, uint32_t rows, uint32_t h,
                                    float eps, luisa::vector<float> &out) {
    auto s = check_size(x.size(), static_cast<size_t>(rows) * h, "x");
    if (!s.ok()) { return s; }
    if (!gamma.empty()) {
        s = check_size(gamma.size(), h, "gamma");
        if (!s.ok()) { return s; }
    }
    if (!beta.empty()) {
        s = check_size(beta.size(), h, "beta");
        if (!s.ok()) { return s; }
    }
    out.resize(x.size());
    for (uint32_t r = 0; r < rows; r++) {
        const size_t base = static_cast<size_t>(r) * h;
        double mean = 0.0;
        for (uint32_t i = 0; i < h; i++) { mean += x[base + i]; }
        mean /= h;
        double var = 0.0;
        for (uint32_t i = 0; i < h; i++) {
            const double d = x[base + i] - mean;
            var += d * d;
        }
        var /= h;
        const double inv = 1.0 / std::sqrt(var + eps);
        for (uint32_t i = 0; i < h; i++) {
            double v = (x[base + i] - mean) * inv;
            if (!gamma.empty()) { v *= gamma[i]; }
            if (!beta.empty()) { v += beta[i]; }
            out[base + i] = static_cast<float>(v);
        }
    }
    return Status::success();
}

[[nodiscard]] Status ref_rms_norm(luisa::span<const float> x, luisa::span<const float> gamma,
                                  uint32_t rows, uint32_t h, float eps, luisa::vector<float> &out) {
    auto s = check_size(x.size(), static_cast<size_t>(rows) * h, "x");
    if (!s.ok()) { return s; }
    if (!gamma.empty()) {
        s = check_size(gamma.size(), h, "gamma");
        if (!s.ok()) { return s; }
    }
    out.resize(x.size());
    for (uint32_t r = 0; r < rows; r++) {
        const size_t base = static_cast<size_t>(r) * h;
        double sumsq = 0.0;
        for (uint32_t i = 0; i < h; i++) { sumsq += static_cast<double>(x[base + i]) * x[base + i]; }
        const double inv = 1.0 / std::sqrt(sumsq / h + eps);
        for (uint32_t i = 0; i < h; i++) {
            double v = x[base + i] * inv;
            if (!gamma.empty()) { v *= gamma[i]; }
            out[base + i] = static_cast<float>(v);
        }
    }
    return Status::success();
}

[[nodiscard]] Status ref_l2_norm(luisa::span<const float> x, uint32_t rows, uint32_t h,
                                 float floor, luisa::vector<float> &out) {
    auto s = check_size(x.size(), static_cast<size_t>(rows) * h, "x");
    if (!s.ok()) { return s; }
    out.resize(x.size());
    for (uint32_t r = 0; r < rows; r++) {
        const size_t base = static_cast<size_t>(r) * h;
        double sumsq = 0.0;
        for (uint32_t i = 0; i < h; i++) { sumsq += static_cast<double>(x[base + i]) * x[base + i]; }
        const double norm = std::max(std::sqrt(sumsq), static_cast<double>(floor));
        for (uint32_t i = 0; i < h; i++) { out[base + i] = static_cast<float>(x[base + i] / norm); }
    }
    return Status::success();
}

[[nodiscard]] Status ref_row_sum(luisa::span<const float> x, uint32_t rows, uint32_t h,
                                 luisa::vector<float> &out) {
    auto s = check_size(x.size(), static_cast<size_t>(rows) * h, "x");
    if (!s.ok()) { return s; }
    out.resize(rows);
    for (uint32_t r = 0; r < rows; r++) {
        double acc = 0.0;
        for (uint32_t i = 0; i < h; i++) { acc += x[static_cast<size_t>(r) * h + i]; }
        out[r] = static_cast<float>(acc);
    }
    return Status::success();
}

[[nodiscard]] Status ref_residual_add(luisa::span<const float> a, luisa::span<const float> b,
                                      luisa::vector<float> &out) {
    auto s = check_size(a.size(), b.size(), "a/b");
    if (!s.ok()) { return s; }
    out.resize(a.size());
    for (size_t i = 0; i < a.size(); i++) { out[i] = a[i] + b[i]; }
    return Status::success();
}

// ---------------------------------------------------------------------------
// RoPE
// ---------------------------------------------------------------------------

namespace {

// Rotate pair (a,b) by angle: (a cos - b sin, a sin + b cos).
inline void rotate_pair(double &a, double &b, double angle) noexcept {
    const double c = std::cos(angle);
    const double s = std::sin(angle);
    const double na = a * c - b * s;
    const double nb = a * s + b * c;
    a = na;
    b = nb;
}

[[nodiscard]] Status rope_impl(luisa::span<const float> x, luisa::span<const int32_t> positions,
                               luisa::span<const uint32_t> axis_of_pair, uint32_t tokens,
                               uint32_t heads, uint32_t head_dim, uint32_t axes, float theta,
                               bool interleaved, luisa::vector<float> &out) {
    if (head_dim % 2u != 0u) {
        return Status::error(StatusCode::invalid_argument, "head_dim must be even for RoPE", "head_dim");
    }
    const uint32_t pairs = head_dim / 2u;
    auto s = check_size(x.size(), static_cast<size_t>(tokens) * heads * head_dim, "x");
    if (!s.ok()) { return s; }
    s = check_size(positions.size(), static_cast<size_t>(tokens) * axes, "positions");
    if (!s.ok()) { return s; }
    if (axis_of_pair.size() != pairs) {
        return Status::error(StatusCode::invalid_argument, "axis_of_pair size must equal head_dim/2", "axis_of_pair");
    }
    out.resize(x.size());
    for (uint32_t t = 0; t < tokens; t++) {
        for (uint32_t h = 0; h < heads; h++) {
            const size_t base = (static_cast<size_t>(t) * heads + h) * head_dim;
            for (uint32_t d = 0; d < pairs; d++) {
                const uint32_t axis = axis_of_pair[d];
                if (axis >= axes) {
                    return Status::error(StatusCode::invalid_argument, "RoPE axis out of range", "axis_of_pair");
                }
                const double pos = positions[static_cast<size_t>(t) * axes + axis];
                const double inv_freq = std::pow(static_cast<double>(theta),
                                                 -2.0 * static_cast<double>(d) / static_cast<double>(head_dim));
                const double angle = pos * inv_freq;
                const uint32_t i0 = interleaved ? d : 2u * d;
                const uint32_t i1 = interleaved ? d + pairs : 2u * d + 1u;
                double a = x[base + i0];
                double b = x[base + i1];
                rotate_pair(a, b, angle);
                out[base + i0] = static_cast<float>(a);
                out[base + i1] = static_cast<float>(b);
            }
        }
    }
    return Status::success();
}

}// namespace

[[nodiscard]] Status ref_rope_1d(luisa::span<const float> x, luisa::span<const int32_t> positions,
                                 uint32_t tokens, uint32_t heads, uint32_t head_dim, float theta,
                                 luisa::vector<float> &out) {
    luisa::vector<uint32_t> axis_of_pair(head_dim / 2u, 0u);
    return rope_impl(x, positions, axis_of_pair, tokens, heads, head_dim, 1u, theta, false, out);
}

[[nodiscard]] Status ref_rope_rotate_half(luisa::span<const float> x, luisa::span<const int32_t> positions,
                                          uint32_t tokens, uint32_t heads, uint32_t head_dim, float theta,
                                          luisa::vector<float> &out) {
    luisa::vector<uint32_t> axis_of_pair(head_dim / 2u, 0u);
    return rope_impl(x, positions, axis_of_pair, tokens, heads, head_dim, 1u, theta, true, out);
}

[[nodiscard]] Status ref_rope_axial(luisa::span<const float> x, luisa::span<const int32_t> positions,
                                    luisa::span<const uint32_t> axis_of_pair, uint32_t tokens,
                                    uint32_t heads, uint32_t head_dim, uint32_t axes, float theta,
                                    luisa::vector<float> &out) {
    return rope_impl(x, positions, axis_of_pair, tokens, heads, head_dim, axes, theta, false, out);
}

[[nodiscard]] Status ref_rope_interleaved(luisa::span<const float> x, luisa::span<const int32_t> positions,
                                          luisa::span<const uint32_t> axis_of_pair, uint32_t tokens,
                                          uint32_t heads, uint32_t head_dim, uint32_t axes, float theta,
                                          luisa::vector<float> &out) {
      return rope_impl(x, positions, axis_of_pair, tokens, heads, head_dim, axes, theta, true, out);
}

namespace {
[[nodiscard]] Status rope_partial_impl(luisa::span<const float> x, luisa::span<const int32_t> positions,
                                       luisa::span<const uint32_t> axis_of_pair, uint32_t tokens,
                                       uint32_t heads, uint32_t head_dim, uint32_t rotary_dim,
                                       uint32_t axes, float theta, bool interleaved, luisa::vector<float> &out) {
    if (head_dim % 2u != 0u || rotary_dim % 2u != 0u || rotary_dim == 0u || rotary_dim > head_dim) {
        return Status::error(StatusCode::invalid_argument,
                             "partial rotary requires 0 < rotary_dim <= head_dim, both even", "rotary_dim");
    }
    auto s = check_size(x.size(), static_cast<size_t>(tokens) * heads * head_dim, "x");
    if (!s.ok()) { return s; }
    s = check_size(positions.size(), static_cast<size_t>(tokens) * axes, "positions");
    if (!s.ok()) { return s; }
    if (axis_of_pair.size() != rotary_dim / 2u) {
        return Status::error(StatusCode::invalid_argument, "axis_of_pair size must equal rotary_dim/2", "axis_of_pair");
    }
    out.resize(x.size());
    const uint32_t pairs = rotary_dim / 2u;
    for (uint32_t t = 0; t < tokens; t++) {
        for (uint32_t h = 0; h < heads; h++) {
            const size_t base = (static_cast<size_t>(t) * heads + h) * head_dim;
            for (uint32_t d = 0; d < pairs; d++) {
                const uint32_t axis = axis_of_pair[d];
                if (axis >= axes) {
                    return Status::error(StatusCode::invalid_argument, "RoPE axis out of range", "axis_of_pair");
                }
                const double pos = positions[static_cast<size_t>(t) * axes + axis];
                const double inv_freq = std::pow(static_cast<double>(theta),
                                                 -2.0 * static_cast<double>(d) / static_cast<double>(rotary_dim));
                const uint32_t i0 = interleaved ? d : 2u * d;
                const uint32_t i1 = interleaved ? d + pairs : 2u * d + 1u;
                double a = x[base + i0];
                double b = x[base + i1];
                rotate_pair(a, b, pos * inv_freq);
                out[base + i0] = static_cast<float>(a);
                out[base + i1] = static_cast<float>(b);
            }
            // Unrotated tail: copied unchanged.
            for (uint32_t i = rotary_dim; i < head_dim; i++) {
                out[base + i] = x[base + i];
            }
        }
    }
    return Status::success();
}
}// namespace

[[nodiscard]] Status ref_rope_partial_adjacent(luisa::span<const float> x, luisa::span<const int32_t> positions,
                                               luisa::span<const uint32_t> axis_of_pair, uint32_t tokens,
                                               uint32_t heads, uint32_t head_dim, uint32_t rotary_dim,
                                               uint32_t axes, float theta, luisa::vector<float> &out) {
    return rope_partial_impl(x, positions, axis_of_pair, tokens, heads, head_dim, rotary_dim, axes, theta, false, out);
}

[[nodiscard]] Status ref_rope_partial_rotate_half(luisa::span<const float> x, luisa::span<const int32_t> positions,
                                                  luisa::span<const uint32_t> axis_of_pair, uint32_t tokens,
                                                  uint32_t heads, uint32_t head_dim, uint32_t rotary_dim,
                                                  uint32_t axes, float theta, luisa::vector<float> &out) {
    return rope_partial_impl(x, positions, axis_of_pair, tokens, heads, head_dim, rotary_dim, axes, theta, true, out);
}

[[nodiscard]] Status ref_rope_1d_fp32_angle(luisa::span<const float> x, luisa::span<const int32_t> positions,
                                            uint32_t tokens, uint32_t heads, uint32_t head_dim, float theta,
                                            luisa::vector<float> &out) {
    if (head_dim % 2u != 0u) {
        return Status::error(StatusCode::invalid_argument, "head_dim must be even for RoPE", "head_dim");
    }
    auto s = check_size(x.size(), static_cast<size_t>(tokens) * heads * head_dim, "x");
    if (!s.ok()) { return s; }
    s = check_size(positions.size(), tokens, "positions");
    if (!s.ok()) { return s; }
    out.resize(x.size());
    const float log2_theta = std::log2(theta);
    for (uint32_t t = 0; t < tokens; t++) {
        for (uint32_t h = 0; h < heads; h++) {
            const size_t base = (static_cast<size_t>(t) * heads + h) * head_dim;
            for (uint32_t d = 0; d < head_dim / 2u; d++) {
                // Exact device evaluation order, all in float.
                const float inv_freq = std::exp2(log2_theta * (-2.0f * static_cast<float>(d) / static_cast<float>(head_dim)));
                const float angle = static_cast<float>(positions[t]) * inv_freq;
                const float c = std::cos(angle);
                const float sn = std::sin(angle);
                const float a = x[base + 2u * d];
                const float b = x[base + 2u * d + 1u];
                out[base + 2u * d] = a * c - b * sn;
                out[base + 2u * d + 1u] = a * sn + b * c;
            }
        }
    }
    return Status::success();
}

[[nodiscard]] Status ref_sinusoidal_table(uint32_t len, uint32_t dim, float theta,
                                          luisa::vector<float> &out) {
    if (dim % 2u != 0u) {
        return Status::error(StatusCode::invalid_argument, "sinusoidal dim must be even", "dim");
    }
    out.resize(static_cast<size_t>(len) * dim);
    for (uint32_t p = 0; p < len; p++) {
        for (uint32_t d = 0; d < dim / 2u; d++) {
            const double inv_freq = std::pow(static_cast<double>(theta),
                                             -2.0 * static_cast<double>(d) / static_cast<double>(dim));
            const double angle = static_cast<double>(p) * inv_freq;
            out[static_cast<size_t>(p) * dim + 2u * d] = static_cast<float>(std::sin(angle));
            out[static_cast<size_t>(p) * dim + 2u * d + 1u] = static_cast<float>(std::cos(angle));
        }
    }
    return Status::success();
}

// ---------------------------------------------------------------------------
// Attention
// ---------------------------------------------------------------------------

namespace {

// Whether query q may attend key k under the mask (absolute positions).
[[nodiscard]] bool mask_allows(const RefAttentionMask &mask, uint32_t q, uint32_t k,
                               uint32_t q_len, uint32_t k_len) noexcept {
    switch (mask.kind) {
        case RefMaskKind::bidirectional: return true;
        case RefMaskKind::causal: {
            // Align the causal diagonal to the END of the key sequence so a
            // short query block over a long KV (decode/prefill tail) works.
            const int64_t offset = static_cast<int64_t>(k_len) - static_cast<int64_t>(q_len);
            return static_cast<int64_t>(k) <= static_cast<int64_t>(q) + offset;
        }
        case RefMaskKind::windowed: {
            const int64_t offset = static_cast<int64_t>(k_len) - static_cast<int64_t>(q_len);
            const int64_t q_abs = static_cast<int64_t>(q) + offset;
            const int64_t kk = k;
            return kk <= q_abs && kk > q_abs - static_cast<int64_t>(mask.window);
        }
        case RefMaskKind::prefix_causal: {
            if (k < mask.prefix_len) { return true; }
            const int64_t offset = static_cast<int64_t>(k_len) - static_cast<int64_t>(q_len);
            return static_cast<int64_t>(k) <= static_cast<int64_t>(q) + offset;
        }
        case RefMaskKind::custom_block: {
            if (mask.block_size == 0u) { return false; }
            const uint32_t qb = q / mask.block_size;
            const uint32_t kb = k / mask.block_size;
            const uint32_t k_blocks = (k_len + mask.block_size - 1u) / mask.block_size;
            const size_t idx = static_cast<size_t>(qb) * k_blocks + kb;
            return idx < mask.block_mask.size() && mask.block_mask[idx] != 0u;
        }
        default: return false;
    }
}

}// namespace

[[nodiscard]] Status ref_attention(luisa::span<const float> q, luisa::span<const float> k,
                                   luisa::span<const float> v, uint32_t q_len, uint32_t k_len,
                                   uint32_t q_heads, uint32_t kv_heads, uint32_t head_dim,
                                   float scale, float softcap, const RefAttentionMask &mask,
                                   luisa::vector<float> &out) {
    if (kv_heads == 0u || q_heads % kv_heads != 0u) {
        return Status::error(StatusCode::invalid_argument, "q_heads must be a multiple of kv_heads", "heads");
    }
    auto s = check_size(q.size(), static_cast<size_t>(q_len) * q_heads * head_dim, "q");
    if (!s.ok()) { return s; }
    s = check_size(k.size(), static_cast<size_t>(k_len) * kv_heads * head_dim, "k");
    if (!s.ok()) { return s; }
    s = check_size(v.size(), k.size(), "v");
    if (!s.ok()) { return s; }
    if (!mask.rel_bias.empty()) {
        s = check_size(mask.rel_bias.size(), static_cast<size_t>(q_heads) * q_len * k_len, "rel_bias");
        if (!s.ok()) { return s; }
    }
    const uint32_t group = q_heads / kv_heads;
    out.assign(static_cast<size_t>(q_len) * q_heads * head_dim, 0.0f);
    luisa::vector<double> scores(k_len);
    for (uint32_t qi = 0; qi < q_len; qi++) {
        for (uint32_t h = 0; h < q_heads; h++) {
            const uint32_t kvh = h / group;
            const size_t q_base = (static_cast<size_t>(qi) * q_heads + h) * head_dim;
            double max_score = -std::numeric_limits<double>::infinity();
            bool any = false;
            for (uint32_t ki = 0; ki < k_len; ki++) {
                if (!mask_allows(mask, qi, ki, q_len, k_len)) { scores[ki] = -std::numeric_limits<double>::infinity(); continue; }
                const size_t k_base = (static_cast<size_t>(ki) * kv_heads + kvh) * head_dim;
                double dot = 0.0;
                for (uint32_t d = 0; d < head_dim; d++) {
                    dot += static_cast<double>(q[q_base + d]) * k[k_base + d];
                }
                double score = dot * scale;
                if (softcap > 0.0f) { score = softcap * std::tanh(score / softcap); }
                if (!mask.rel_bias.empty()) {
                    score += mask.rel_bias[(static_cast<size_t>(h) * q_len + qi) * k_len + ki];
                }
                scores[ki] = score;
                max_score = std::max(max_score, score);
                any = true;
            }
            if (!any) { continue; }// fully masked row -> zero output
            double sum = 0.0;
            for (uint32_t ki = 0; ki < k_len; ki++) {
                scores[ki] = std::exp(scores[ki] - max_score);
                sum += scores[ki];
            }
            const size_t o_base = (static_cast<size_t>(qi) * q_heads + h) * head_dim;
            for (uint32_t d = 0; d < head_dim; d++) {
                double acc = 0.0;
                for (uint32_t ki = 0; ki < k_len; ki++) {
                    const size_t v_base = (static_cast<size_t>(ki) * kv_heads + kvh) * head_dim;
                    acc += (scores[ki] / sum) * v[v_base + d];
                }
                out[o_base + d] = static_cast<float>(acc);
            }
        }
    }
    return Status::success();
}

[[nodiscard]] Status ref_attention_decode(luisa::span<const float> q, luisa::span<const float> k,
                                          luisa::span<const float> v, luisa::span<const uint32_t> kv_len,
                                          luisa::span<const int32_t> positions, uint32_t batch,
                                          uint32_t kv_capacity, uint32_t q_heads, uint32_t kv_heads,
                                          uint32_t head_dim, float scale, float softcap,
                                          const RefAttentionMask &mask, luisa::vector<float> &out) {
    if (kv_heads == 0u || q_heads % kv_heads != 0u) {
        return Status::error(StatusCode::invalid_argument, "q_heads must be a multiple of kv_heads", "heads");
    }
    auto s = check_size(q.size(), static_cast<size_t>(batch) * q_heads * head_dim, "q");
    if (!s.ok()) { return s; }
    s = check_size(k.size(), static_cast<size_t>(batch) * kv_capacity * kv_heads * head_dim, "k");
    if (!s.ok()) { return s; }
    s = check_size(v.size(), k.size(), "v");
    if (!s.ok()) { return s; }
    s = check_size(kv_len.size(), batch, "kv_len");
    if (!s.ok()) { return s; }
    s = check_size(positions.size(), batch, "positions");
    if (!s.ok()) { return s; }
    const uint32_t group = q_heads / kv_heads;
    out.assign(static_cast<size_t>(batch) * q_heads * head_dim, 0.0f);
    luisa::vector<double> scores(kv_capacity);
    for (uint32_t b = 0; b < batch; b++) {
        const uint32_t len = kv_len[b];
        if (len > kv_capacity) {
            return Status::error(StatusCode::invalid_argument, "kv_len exceeds capacity", "kv_len");
        }
        for (uint32_t h = 0; h < q_heads; h++) {
            const uint32_t kvh = h / group;
            const size_t q_base = (static_cast<size_t>(b) * q_heads + h) * head_dim;
            double max_score = -std::numeric_limits<double>::infinity();
            bool any = false;
            for (uint32_t ki = 0; ki < len; ki++) {
                // Decode: the query is at absolute position positions[b]; a
                // causal mask allows keys up to that position. Windowed uses
                // the same absolute position. Prefix/custom apply directly.
                bool allowed = true;
                switch (mask.kind) {
                    case RefMaskKind::causal: allowed = static_cast<int32_t>(ki) <= positions[b]; break;
                    case RefMaskKind::windowed:
                        allowed = static_cast<int32_t>(ki) <= positions[b] &&
                                  static_cast<int32_t>(ki) > positions[b] - static_cast<int32_t>(mask.window);
                        break;
                    case RefMaskKind::prefix_causal:
                        allowed = ki < mask.prefix_len || static_cast<int32_t>(ki) <= positions[b];
                        break;
                    case RefMaskKind::custom_block: {
                        if (mask.block_size == 0u) { allowed = false; break; }
                        const uint32_t qb = static_cast<uint32_t>(positions[b]) / mask.block_size;
                        const uint32_t kb = ki / mask.block_size;
                        const uint32_t k_blocks = (kv_capacity + mask.block_size - 1u) / mask.block_size;
                        const size_t idx = static_cast<size_t>(qb) * k_blocks + kb;
                        allowed = idx < mask.block_mask.size() && mask.block_mask[idx] != 0u;
                        break;
                    }
                    default: break;
                }
                if (!allowed) { scores[ki] = -std::numeric_limits<double>::infinity(); continue; }
                const size_t k_base = ((static_cast<size_t>(b) * kv_capacity + ki) * kv_heads + kvh) * head_dim;
                double dot = 0.0;
                for (uint32_t d = 0; d < head_dim; d++) { dot += static_cast<double>(q[q_base + d]) * k[k_base + d]; }
                double score = dot * scale;
                if (softcap > 0.0f) { score = softcap * std::tanh(score / softcap); }
                scores[ki] = score;
                max_score = std::max(max_score, score);
                any = true;
            }
            if (!any) { continue; }
            double sum = 0.0;
            for (uint32_t ki = 0; ki < len; ki++) {
                scores[ki] = std::exp(scores[ki] - max_score);
                sum += scores[ki];
            }
            const size_t o_base = (static_cast<size_t>(b) * q_heads + h) * head_dim;
            for (uint32_t d = 0; d < head_dim; d++) {
                double acc = 0.0;
                for (uint32_t ki = 0; ki < len; ki++) {
                    const size_t v_base = ((static_cast<size_t>(b) * kv_capacity + ki) * kv_heads + kvh) * head_dim;
                    acc += (scores[ki] / sum) * v[v_base + d];
                }
                out[o_base + d] = static_cast<float>(acc);
            }
        }
    }
    return Status::success();
}

// ---------------------------------------------------------------------------
// Image
// ---------------------------------------------------------------------------

[[nodiscard]] Status ref_image_resize_normalize(luisa::span<const uint8_t> src, uint32_t sw, uint32_t sh,
                                                uint32_t dw, uint32_t dh, bool bgr_to_rgb, bool bilinear,
                                                luisa::span<const float> mean, luisa::span<const float> std_dev,
                                                luisa::vector<float> &out) {
    auto s = check_size(src.size(), static_cast<size_t>(sw) * sh * 3u, "src");
    if (!s.ok()) { return s; }
    if (!mean.empty()) {
        s = check_size(mean.size(), 3u, "mean");
        if (!s.ok()) { return s; }
        s = check_size(std_dev.size(), 3u, "std_dev");
        if (!s.ok()) { return s; }
    }
    const float scale_x = static_cast<float>(sw) / static_cast<float>(dw);
    const float scale_y = static_cast<float>(sh) / static_cast<float>(dh);
    const size_t plane = static_cast<size_t>(dw) * dh;
    out.resize(plane * 3u);
    auto sample = [&](uint32_t c, float x, float y) -> double {
        // Channel swap on read when converting BGR->RGB.
        const uint32_t sc = bgr_to_rgb ? (2u - c) : c;
        if (bilinear) {
            const float x0f = std::floor(x);
            const float y0f = std::floor(y);
            const int32_t x0 = static_cast<int32_t>(x0f);
            const int32_t y0 = static_cast<int32_t>(y0f);
            const float fx = x - x0f;
            const float fy = y - y0f;
            auto at = [&](int32_t px, int32_t py) -> double {
                const int32_t cx = std::clamp(px, 0, static_cast<int32_t>(sw) - 1);
                const int32_t cy = std::clamp(py, 0, static_cast<int32_t>(sh) - 1);
                return src[(static_cast<size_t>(cy) * sw + cx) * 3u + sc];
            };
            const double v00 = at(x0, y0), v10 = at(x0 + 1, y0), v01 = at(x0, y0 + 1), v11 = at(x0 + 1, y0 + 1);
            const double top = v00 + (v10 - v00) * fx;
            const double bottom = v01 + (v11 - v01) * fx;
            return top + (bottom - top) * fy;
        }
        // Area: box average over the destination footprint.
        const uint32_t dx = static_cast<uint32_t>(x);
        const uint32_t dy = static_cast<uint32_t>(y);
        const float xb = dx * scale_x, xe = (dx + 1) * scale_x;
        const float yb = dy * scale_y, ye = (dy + 1) * scale_y;
        const uint32_t x0 = static_cast<uint32_t>(std::floor(xb));
        const uint32_t x1 = std::min(sw, static_cast<uint32_t>(std::ceil(xe)));
        const uint32_t y0 = static_cast<uint32_t>(std::floor(yb));
        const uint32_t y1 = std::min(sh, static_cast<uint32_t>(std::ceil(ye)));
        double sum = 0.0, weight = 0.0;
        for (uint32_t yy = y0; yy < y1; yy++) {
            const double wy = std::min(ye, static_cast<float>(yy + 1)) - std::max(yb, static_cast<float>(yy));
            for (uint32_t xx = x0; xx < x1; xx++) {
                const double wx = std::min(xe, static_cast<float>(xx + 1)) - std::max(xb, static_cast<float>(xx));
                const double w = wx * wy;
                sum += src[(static_cast<size_t>(yy) * sw + xx) * 3u + sc] * w;
                weight += w;
            }
        }
        return weight > 0.0 ? sum / weight : 0.0;
    };
    for (uint32_t y = 0; y < dh; y++) {
        for (uint32_t x = 0; x < dw; x++) {
            for (uint32_t c = 0; c < 3u; c++) {
                double value;
                if (sw == dw && sh == dh) {
                    const uint32_t sc = bgr_to_rgb ? (2u - c) : c;
                    value = src[(static_cast<size_t>(y) * sw + x) * 3u + sc];
                } else if (bilinear) {
                    value = sample(c, (x + 0.5f) * scale_x - 0.5f, (y + 0.5f) * scale_y - 0.5f);
                } else {
                    value = sample(c, static_cast<float>(x), static_cast<float>(y));
                }
                double normalized = value / 255.0;
                if (!mean.empty()) { normalized = (normalized - mean[c]) / std_dev[c]; }
                out[static_cast<size_t>(c) * plane + static_cast<size_t>(y) * dw + x] = static_cast<float>(normalized);
            }
        }
    }
    return Status::success();
}

// ---------------------------------------------------------------------------
// Conv / pool
// ---------------------------------------------------------------------------

[[nodiscard]] Status ref_conv2d(luisa::span<const float> x, luisa::span<const float> w,
                                luisa::span<const float> bias, uint32_t n, uint32_t in_c, uint32_t h,
                                uint32_t w_in, uint32_t out_c, uint32_t kh, uint32_t kw,
                                uint32_t stride_h, uint32_t stride_w, uint32_t pad_h, uint32_t pad_w,
                                uint32_t dil_h, uint32_t dil_w, uint32_t groups,
                                luisa::vector<float> &out, uint32_t &out_h, uint32_t &out_w) {
    if (groups == 0u || in_c % groups != 0u || out_c % groups != 0u ||
        in_c == 0u || out_c == 0u || n == 0u || h == 0u || w_in == 0u) {
        return Status::error(StatusCode::invalid_argument, "invalid conv groups/channels/dims", "groups");
    }
    if (stride_h == 0u || stride_w == 0u || dil_h == 0u || dil_w == 0u || kh == 0u || kw == 0u) {
        return Status::error(StatusCode::invalid_argument, "invalid conv geometry", "geometry");
    }
    const uint32_t eff_kh = (kh - 1u) * dil_h + 1u;
    const uint32_t eff_kw = (kw - 1u) * dil_w + 1u;
    if (h + 2u * pad_h < eff_kh || w_in + 2u * pad_w < eff_kw) {
        return Status::error(StatusCode::invalid_argument, "conv input smaller than kernel", "geometry");
    }
    out_h = (h + 2u * pad_h - eff_kh) / stride_h + 1u;
    out_w = (w_in + 2u * pad_w - eff_kw) / stride_w + 1u;
    auto s = check_size(x.size(), static_cast<size_t>(n) * in_c * h * w_in, "x");
    if (!s.ok()) { return s; }
    s = check_size(w.size(), static_cast<size_t>(out_c) * (in_c / groups) * kh * kw, "w");
    if (!s.ok()) { return s; }
    if (!bias.empty()) {
        s = check_size(bias.size(), out_c, "bias");
        if (!s.ok()) { return s; }
    }
    out.assign(static_cast<size_t>(n) * out_c * out_h * out_w, 0.0f);
    const uint32_t icg = in_c / groups;
    const uint32_t ocg = out_c / groups;
    for (uint32_t ni = 0; ni < n; ni++) {
        for (uint32_t g = 0; g < groups; g++) {
            for (uint32_t oc = 0; oc < ocg; oc++) {
                const uint32_t oc_abs = g * ocg + oc;
                for (uint32_t oy = 0; oy < out_h; oy++) {
                    for (uint32_t ox = 0; ox < out_w; ox++) {
                        double acc = bias.empty() ? 0.0 : static_cast<double>(bias[oc_abs]);
                        for (uint32_t ic = 0; ic < icg; ic++) {
                            const uint32_t ic_abs = g * icg + ic;
                            for (uint32_t ky = 0; ky < kh; ky++) {
                                const int64_t iy = static_cast<int64_t>(oy * stride_h) + ky * dil_h - pad_h;
                                if (iy < 0 || iy >= h) { continue; }
                                for (uint32_t kx = 0; kx < kw; kx++) {
                                    const int64_t ix = static_cast<int64_t>(ox * stride_w) + kx * dil_w - pad_w;
                                    if (ix < 0 || ix >= w_in) { continue; }
                                    const size_t xi = ((static_cast<size_t>(ni) * in_c + ic_abs) * h + iy) * w_in + ix;
                                    const size_t wi = ((static_cast<size_t>(oc_abs) * icg + ic) * kh + ky) * kw + kx;
                                    acc += static_cast<double>(x[xi]) * w[wi];
                                }
                            }
                        }
                        const size_t oi = ((static_cast<size_t>(ni) * out_c + oc_abs) * out_h + oy) * out_w + ox;
                        out[oi] = static_cast<float>(acc);
                    }
                }
            }
        }
    }
    return Status::success();
}

[[nodiscard]] Status ref_conv1d(luisa::span<const float> x, luisa::span<const float> w,
                                luisa::span<const float> bias, uint32_t n, uint32_t c, uint32_t len,
                                uint32_t out_c, uint32_t k, uint32_t stride, uint32_t pad,
                                uint32_t dilation, uint32_t groups, bool causal,
                                luisa::vector<float> &out, uint32_t &out_len) {
    if (groups == 0u || c % groups != 0u || out_c % groups != 0u || stride == 0u || dilation == 0u || k == 0u) {
        return Status::error(StatusCode::invalid_argument, "invalid conv1d geometry", "geometry");
    }
    const uint32_t eff_k = (k - 1u) * dilation + 1u;
    const uint32_t pad_left = causal ? eff_k - 1u : pad;
    const uint32_t pad_right = causal ? 0u : pad;
    if (len + pad_left + pad_right < eff_k) {
        return Status::error(StatusCode::invalid_argument, "conv1d input smaller than kernel", "geometry");
    }
    out_len = (len + pad_left + pad_right - eff_k) / stride + 1u;
    auto s = check_size(x.size(), static_cast<size_t>(n) * c * len, "x");
    if (!s.ok()) { return s; }
    s = check_size(w.size(), static_cast<size_t>(out_c) * (c / groups) * k, "w");
    if (!s.ok()) { return s; }
    if (!bias.empty()) {
        s = check_size(bias.size(), out_c, "bias");
        if (!s.ok()) { return s; }
    }
    out.assign(static_cast<size_t>(n) * out_c * out_len, 0.0f);
    const uint32_t icg = c / groups;
    const uint32_t ocg = out_c / groups;
    for (uint32_t ni = 0; ni < n; ni++) {
        for (uint32_t g = 0; g < groups; g++) {
            for (uint32_t oc = 0; oc < ocg; oc++) {
                const uint32_t oc_abs = g * ocg + oc;
                for (uint32_t ot = 0; ot < out_len; ot++) {
                    double acc = bias.empty() ? 0.0 : static_cast<double>(bias[oc_abs]);
                    for (uint32_t ic = 0; ic < icg; ic++) {
                        const uint32_t ic_abs = g * icg + ic;
                        for (uint32_t kt = 0; kt < k; kt++) {
                            const int64_t it = static_cast<int64_t>(ot * stride) + kt * dilation - pad_left;
                            if (it < 0 || it >= len) { continue; }
                            const size_t xi = (static_cast<size_t>(ni) * c + ic_abs) * len + it;
                            const size_t wi = (static_cast<size_t>(oc_abs) * icg + ic) * k + kt;
                            acc += static_cast<double>(x[xi]) * w[wi];
                        }
                    }
                    out[(static_cast<size_t>(ni) * out_c + oc_abs) * out_len + ot] = static_cast<float>(acc);
                }
            }
        }
    }
    return Status::success();
}

[[nodiscard]] Status ref_batch_norm(luisa::span<const float> x, luisa::span<const float> mean,
                                    luisa::span<const float> var, luisa::span<const float> gamma,
                                    luisa::span<const float> beta, uint32_t n, uint32_t c,
                                    uint32_t spatial, float eps, luisa::vector<float> &out) {
    auto s = check_size(x.size(), static_cast<size_t>(n) * c * spatial, "x");
    if (!s.ok()) { return s; }
    for (auto span : {mean, var, gamma, beta}) {
        s = check_size(span.size(), c, "bn param");
        if (!s.ok()) { return s; }
    }
    out.resize(x.size());
    for (uint32_t ni = 0; ni < n; ni++) {
        for (uint32_t ci = 0; ci < c; ci++) {
            const double inv = 1.0 / std::sqrt(static_cast<double>(var[ci]) + eps);
            for (uint32_t p = 0; p < spatial; p++) {
                const size_t idx = (static_cast<size_t>(ni) * c + ci) * spatial + p;
                out[idx] = static_cast<float>((x[idx] - mean[ci]) * inv * gamma[ci] + beta[ci]);
            }
        }
    }
    return Status::success();
}

[[nodiscard]] Status ref_avg_pool2d(luisa::span<const float> x, uint32_t n, uint32_t c, uint32_t h,
                                    uint32_t w, uint32_t kh, uint32_t kw, uint32_t sh, uint32_t sw,
                                    uint32_t ph, uint32_t pw, luisa::vector<float> &out,
                                    uint32_t &out_h, uint32_t &out_w) {
    if (kh == 0u || kw == 0u || sh == 0u || sw == 0u) {
        return Status::error(StatusCode::invalid_argument, "invalid pool geometry", "geometry");
    }
    if (h + 2u * ph < kh || w + 2u * pw < kw) {
        return Status::error(StatusCode::invalid_argument, "pool input smaller than kernel", "geometry");
    }
    out_h = (h + 2u * ph - kh) / sh + 1u;
    out_w = (w + 2u * pw - kw) / sw + 1u;
    auto s = check_size(x.size(), static_cast<size_t>(n) * c * h * w, "x");
    if (!s.ok()) { return s; }
    out.assign(static_cast<size_t>(n) * c * out_h * out_w, 0.0f);
    for (uint32_t ni = 0; ni < n; ni++) {
        for (uint32_t ci = 0; ci < c; ci++) {
            for (uint32_t oy = 0; oy < out_h; oy++) {
                for (uint32_t ox = 0; ox < out_w; ox++) {
                    double sum = 0.0;
                    uint32_t count = 0;
                    for (uint32_t ky = 0; ky < kh; ky++) {
                        const int64_t iy = static_cast<int64_t>(oy * sh) + ky - ph;
                        if (iy < 0 || iy >= h) { continue; }
                        for (uint32_t kx = 0; kx < kw; kx++) {
                            const int64_t ix = static_cast<int64_t>(ox * sw) + kx - pw;
                            if (ix < 0 || ix >= w) { continue; }
                            sum += x[((static_cast<size_t>(ni) * c + ci) * h + iy) * w + ix];
                            count++;
                        }
                    }
                    out[((static_cast<size_t>(ni) * c + ci) * out_h + oy) * out_w + ox] =
                        count > 0u ? static_cast<float>(sum / count) : 0.0f;
                }
            }
        }
    }
    return Status::success();
}

[[nodiscard]] Status ref_adaptive_avg_pool2d(luisa::span<const float> x, uint32_t n, uint32_t c,
                                             uint32_t h, uint32_t w, uint32_t out_h, uint32_t out_w,
                                             luisa::vector<float> &out) {
    if (out_h == 0u || out_w == 0u || h == 0u || w == 0u) {
        return Status::error(StatusCode::invalid_argument, "invalid adaptive pool geometry", "geometry");
    }
    auto s = check_size(x.size(), static_cast<size_t>(n) * c * h * w, "x");
    if (!s.ok()) { return s; }
    out.assign(static_cast<size_t>(n) * c * out_h * out_w, 0.0f);
    for (uint32_t ni = 0; ni < n; ni++) {
        for (uint32_t ci = 0; ci < c; ci++) {
            for (uint32_t oy = 0; oy < out_h; oy++) {
                // PyTorch adaptive_avg_pool2d bin edges: start = floor(i*in/out),
                // end = ceil((i+1)*in/out).
                const uint32_t y0 = static_cast<uint32_t>(static_cast<uint64_t>(oy) * h / out_h);
                const uint32_t y1 = static_cast<uint32_t>((static_cast<uint64_t>(oy + 1) * h + out_h - 1u) / out_h);
                for (uint32_t ox = 0; ox < out_w; ox++) {
                    const uint32_t x0 = static_cast<uint32_t>(static_cast<uint64_t>(ox) * w / out_w);
                    const uint32_t x1 = static_cast<uint32_t>((static_cast<uint64_t>(ox + 1) * w + out_w - 1u) / out_w);
                    double sum = 0.0;
                    uint32_t count = 0;
                    for (uint32_t yy = y0; yy < y1; yy++) {
                        for (uint32_t xx = x0; xx < x1; xx++) {
                            sum += x[((static_cast<size_t>(ni) * c + ci) * h + yy) * w + xx];
                            count++;
                        }
                    }
                    out[((static_cast<size_t>(ni) * c + ci) * out_h + oy) * out_w + ox] =
                        count > 0u ? static_cast<float>(sum / count) : 0.0f;
                }
            }
        }
    }
    return Status::success();
}

[[nodiscard]] Status ref_pixel_unshuffle(luisa::span<const float> x, uint32_t n, uint32_t c,
                                         uint32_t h, uint32_t w, uint32_t r, luisa::vector<float> &out) {
    if (r == 0u || h % r != 0u || w % r != 0u) {
        return Status::error(StatusCode::invalid_argument, "pixel unshuffle requires divisible dims", "r");
    }
    auto s = check_size(x.size(), static_cast<size_t>(n) * c * h * w, "x");
    if (!s.ok()) { return s; }
    const uint32_t oh = h / r, ow = w / r, oc = c * r * r;
    out.resize(static_cast<size_t>(n) * oc * oh * ow);
    for (uint32_t ni = 0; ni < n; ni++) {
        for (uint32_t ci = 0; ci < c; ci++) {
            for (uint32_t y = 0; y < h; y++) {
                for (uint32_t xx = 0; xx < w; xx++) {
                    const uint32_t oy = y / r, ox = xx / r;
                    const uint32_t ry = y % r, rx = xx % r;
                    const uint32_t oc_idx = ci * r * r + ry * r + rx;
                    out[((static_cast<size_t>(ni) * oc + oc_idx) * oh + oy) * ow + ox] =
                        x[((static_cast<size_t>(ni) * c + ci) * h + y) * w + xx];
                }
            }
        }
    }
    return Status::success();
}

[[nodiscard]] Status ref_concat_channels(luisa::span<const float> a, luisa::span<const float> b,
                                         uint32_t n, uint32_t ca, uint32_t cb, uint32_t spatial,
                                         luisa::vector<float> &out) {
    auto s = check_size(a.size(), static_cast<size_t>(n) * ca * spatial, "a");
    if (!s.ok()) { return s; }
    s = check_size(b.size(), static_cast<size_t>(n) * cb * spatial, "b");
    if (!s.ok()) { return s; }
    const uint32_t oc = ca + cb;
    out.resize(static_cast<size_t>(n) * oc * spatial);
    for (uint32_t ni = 0; ni < n; ni++) {
        for (uint32_t ci = 0; ci < ca; ci++) {
            for (uint32_t p = 0; p < spatial; p++) {
                out[(static_cast<size_t>(ni) * oc + ci) * spatial + p] =
                    a[(static_cast<size_t>(ni) * ca + ci) * spatial + p];
            }
        }
        for (uint32_t ci = 0; ci < cb; ci++) {
            for (uint32_t p = 0; p < spatial; p++) {
                out[(static_cast<size_t>(ni) * oc + ca + ci) * spatial + p] =
                    b[(static_cast<size_t>(ni) * cb + ci) * spatial + p];
            }
        }
    }
    return Status::success();
}

[[nodiscard]] Status ref_channel_shuffle(luisa::span<const float> x, uint32_t n, uint32_t c,
                                         uint32_t spatial, uint32_t groups, luisa::vector<float> &out) {
    if (groups == 0u || c % groups != 0u) {
        return Status::error(StatusCode::invalid_argument, "channel shuffle requires divisible channels", "groups");
    }
    auto s = check_size(x.size(), static_cast<size_t>(n) * c * spatial, "x");
    if (!s.ok()) { return s; }
    out.resize(x.size());
    const uint32_t cpg = c / groups;
    for (uint32_t ni = 0; ni < n; ni++) {
        for (uint32_t ci = 0; ci < c; ci++) {
            const uint32_t g = ci / cpg;
            const uint32_t j = ci % cpg;
            const uint32_t dst = j * groups + g;
            for (uint32_t p = 0; p < spatial; p++) {
                out[(static_cast<size_t>(ni) * c + dst) * spatial + p] =
                    x[(static_cast<size_t>(ni) * c + ci) * spatial + p];
            }
        }
    }
    return Status::success();
}

// ---------------------------------------------------------------------------
// Decode / vision
// ---------------------------------------------------------------------------

[[nodiscard]] Status ref_embedding_lookup(luisa::span<const float> table, luisa::span<const int32_t> ids,
                                          uint32_t vocab, uint32_t hidden, luisa::vector<float> &out) {
    auto s = check_size(table.size(), static_cast<size_t>(vocab) * hidden, "table");
    if (!s.ok()) { return s; }
    out.resize(ids.size() * hidden);
    for (size_t t = 0; t < ids.size(); t++) {
        const int32_t id = ids[t];
        if (id < 0 || static_cast<uint32_t>(id) >= vocab) {
            return Status::error(StatusCode::invalid_argument, "embedding id out of range", "ids");
        }
        for (uint32_t d = 0; d < hidden; d++) {
            out[t * hidden + d] = table[static_cast<size_t>(id) * hidden + d];
        }
    }
    return Status::success();
}

[[nodiscard]] Status ref_mm_scatter(luisa::span<const float> embeddings, luisa::span<const float> features,
                                    uint32_t seq, uint32_t hidden, uint32_t start,
                                    luisa::vector<float> &out) {
    auto s = check_size(embeddings.size(), static_cast<size_t>(seq) * hidden, "embeddings");
    if (!s.ok()) { return s; }
    if (features.size() % hidden != 0u) {
        return Status::error(StatusCode::invalid_argument, "features size not a multiple of hidden", "features");
    }
    const uint32_t count = static_cast<uint32_t>(features.size() / hidden);
    if (start + count > seq) {
        return Status::error(StatusCode::invalid_argument, "scatter range exceeds sequence", "start");
    }
    out.assign(embeddings.begin(), embeddings.end());
    for (uint32_t t = 0; t < count; t++) {
        for (uint32_t d = 0; d < hidden; d++) {
            out[(static_cast<size_t>(start + t)) * hidden + d] = features[static_cast<size_t>(t) * hidden + d];
        }
    }
    return Status::success();
}

[[nodiscard]] Status ref_argmax(luisa::span<const float> logits, uint32_t rows, uint32_t cols,
                                luisa::vector<int32_t> &out) {
    auto s = check_size(logits.size(), static_cast<size_t>(rows) * cols, "logits");
    if (!s.ok()) { return s; }
    out.resize(rows);
    for (uint32_t r = 0; r < rows; r++) {
        uint32_t best = 0u;
        float best_v = logits[static_cast<size_t>(r) * cols];
        for (uint32_t c = 1u; c < cols; c++) {
            const float v = logits[static_cast<size_t>(r) * cols + c];
            if (v > best_v) { best_v = v; best = c; }
        }
        out[r] = static_cast<int32_t>(best);
    }
    return Status::success();
}

[[nodiscard]] Status ref_kv_append(luisa::span<const float> src_k, luisa::span<const float> src_v,
                                   luisa::span<float> dst_k, luisa::span<float> dst_v,
                                   uint32_t count, uint32_t kv_dim, uint32_t start, uint32_t capacity) {
    const size_t src_count = static_cast<size_t>(count) * kv_dim;
    const size_t dst_count = static_cast<size_t>(capacity) * kv_dim;
    auto s = check_size(src_k.size(), src_count, "src_k");
    if (!s.ok()) { return s; }
    s = check_size(src_v.size(), src_count, "src_v");
    if (!s.ok()) { return s; }
    s = check_size(dst_k.size(), dst_count, "dst_k");
    if (!s.ok()) { return s; }
    s = check_size(dst_v.size(), dst_count, "dst_v");
    if (!s.ok()) { return s; }
    if (static_cast<uint64_t>(start) + count > capacity) {
        return Status::error(StatusCode::out_of_memory, "KV append overruns token capacity", "capacity");
    }
    for (uint32_t t = 0; t < count; t++) {
        for (uint32_t d = 0; d < kv_dim; d++) {
            const size_t src = static_cast<size_t>(t) * kv_dim + d;
            const size_t dst = static_cast<size_t>(start + t) * kv_dim + d;
            dst_k[dst] = src_k[src];
            dst_v[dst] = src_v[src];
        }
    }
    return Status::success();
}

[[nodiscard]] Status ref_softmax(luisa::span<const float> logits, uint32_t rows, uint32_t cols,
                                 float temperature, luisa::vector<float> &out) {
    auto s = check_size(logits.size(), static_cast<size_t>(rows) * cols, "logits");
    if (!s.ok()) { return s; }
    if (!(temperature > 0.0f) || !std::isfinite(temperature)) {
        return Status::error(StatusCode::invalid_argument, "temperature must be finite and positive", "temperature");
    }
    out.resize(logits.size());
    for (uint32_t r = 0; r < rows; r++) {
        const size_t base = static_cast<size_t>(r) * cols;
        double max_v = -std::numeric_limits<double>::infinity();
        for (uint32_t c = 0; c < cols; c++) { max_v = std::max(max_v, static_cast<double>(logits[base + c]) / temperature); }
        double sum = 0.0;
        for (uint32_t c = 0; c < cols; c++) {
            const double e = std::exp(static_cast<double>(logits[base + c]) / temperature - max_v);
            out[base + c] = static_cast<float>(e);
            sum += e;
        }
        for (uint32_t c = 0; c < cols; c++) { out[base + c] = static_cast<float>(out[base + c] / sum); }
    }
    return Status::success();
}

[[nodiscard]] Status ref_add_position(luisa::span<const float> x, luisa::span<const float> pos,
                                      luisa::span<const int32_t> pos_ids, uint32_t tokens,
                                      uint32_t hidden, luisa::vector<float> &out) {
    auto s = check_size(x.size(), static_cast<size_t>(tokens) * hidden, "x");
    if (!s.ok()) { return s; }
    s = check_size(pos_ids.size(), tokens, "pos_ids");
    if (!s.ok()) { return s; }
    if (pos.size() % hidden != 0u) {
        return Status::error(StatusCode::invalid_argument, "pos table size not a multiple of hidden", "pos");
    }
    const uint32_t pos_rows = static_cast<uint32_t>(pos.size() / hidden);
    out.resize(x.size());
    for (uint32_t t = 0; t < tokens; t++) {
        const int32_t p = pos_ids[t];
        if (p < 0 || static_cast<uint32_t>(p) >= pos_rows) {
            return Status::error(StatusCode::invalid_argument, "position id out of range", "pos_ids");
        }
        for (uint32_t d = 0; d < hidden; d++) {
            out[static_cast<size_t>(t) * hidden + d] =
                x[static_cast<size_t>(t) * hidden + d] + pos[static_cast<size_t>(p) * hidden + d];
        }
    }
    return Status::success();
}

[[nodiscard]] Status ref_deepstack_inject(luisa::span<const float> x, luisa::span<const float> proj,
                                          uint32_t tokens, uint32_t hidden, uint32_t start,
                                          luisa::vector<float> &out) {
    auto s = check_size(x.size(), static_cast<size_t>(tokens) * hidden, "x");
    if (!s.ok()) { return s; }
    if (proj.size() % hidden != 0u) {
        return Status::error(StatusCode::invalid_argument, "proj size not a multiple of hidden", "proj");
    }
    const uint32_t count = static_cast<uint32_t>(proj.size() / hidden);
    if (start + count > tokens) {
        return Status::error(StatusCode::invalid_argument, "injection range exceeds tokens", "start");
    }
    out.assign(x.begin(), x.end());
    for (uint32_t t = 0; t < count; t++) {
        for (uint32_t d = 0; d < hidden; d++) {
            out[(static_cast<size_t>(start + t)) * hidden + d] += proj[static_cast<size_t>(t) * hidden + d];
        }
    }
    return Status::success();
}

}// namespace luisa::ocr
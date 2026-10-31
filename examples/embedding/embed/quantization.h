// Per-tensor symmetric INT8 quantization helpers for BGM3 v3 packages
// (INT8 quantization, phase 1: packer + loader). Plan reference:
// examples/embedding/embedding_model_plan.md (INT8 quantization).
//
// Purpose
//   ONE header-only, host-only transcription of the v3 quantization contract, so
//   the package loader (embed/package.cpp), the executor upload path and the
//   host tests all use identical math:
//     absmax = max |x| over the tensor (a nonfinite element is fail-closed)
//     scale  = absmax / 127                (exactly 1.0 for an all-zero tensor)
//     code   = clamp(RNE(x / scale), -127, +127)   (RNE = round half to even)
//     value  = code * scale                (the only dequantization)
//   A v3 blob is exactly element_count signed int8 bytes (no per-blob header);
//   the positive fp32 scale is stored in the reserved trailing 4 bytes
//   (offset 44) of that tensor's 48-byte table entry. v1/v2 packages keep those
//   bytes zero and are untouched by this file.
//
// Plan of use (phase 1, no GPU code here)
//   1. pack: tools/pack_model.cpp --precision int8 quantizes every tensor, writes
//      the per-tensor scale into the table entry, then its internal --check path
//      dequantizes on read-back (so the parity gate measures the true effect).
//   2. load: embed::Package::load exposes Tensor::int8_data / Tensor::int8_scale
//      (validated finite and > 0) and never routes a v3 blob through
//      data/half_data.
//   3. upload (phase 2, parent): either keep the codes and dequantize in the
//      kernel (q * scale) or call int8_dequantize_tensor() once host-side.
//
// Host-only: no lc-dsl / device headers, no kernels, no exceptions, no RTTI, no
// STL containers (luisa::span only). The packer tool links no lc libraries and
// therefore carries a mirrored copy of this math in pack_model.cpp; the two MUST
// agree bit for bit (the same arrangement as the bf16/fp16 codecs there).
#pragma once

#include <cmath>
#include <cstdint>
#include <cstddef>

#include <luisa/core/logging.h>
#include <luisa/core/stl/memory.h>// luisa::span

namespace luisa::embed {

// Symmetric int8 code range: [-127, +127]. -128 is never produced, so that
// q * scale stays exactly symmetric around zero.
constexpr int32_t kInt8QuantMax = 127;

// Scale used for a tensor whose absmax is zero. Any positive scale maps an
// all-zero tensor to all-zero codes; 1.0 keeps the stored scale valid.
constexpr float kInt8UnitScale = 1.0f;

// A stored/loaded v3 scale must be finite and strictly positive (loader gate:
// NaN, +-inf, 0 and negative scales are all rejected).
[[nodiscard]] inline bool int8_scale_is_valid(float scale) noexcept {
    return std::isfinite(scale) && scale > 0.0f;
}

// Round to nearest integer, ties to even. Implemented explicitly (instead of
// std::nearbyint) so the result never depends on the ambient FP rounding mode;
// double arithmetic keeps the tie test exact over the whole quantized range.
[[nodiscard]] inline double int8_round_nearest_even(double x) noexcept {
    const double base = std::floor(x);
    const double frac = x - base;// in [0, 1)
    if (frac > 0.5) { return base + 1.0; }
    if (frac < 0.5) { return base; }
    return std::fmod(base, 2.0) == 0.0 ? base : base + 1.0;
}

// One fp32 value -> one int8 code, clamped into [-127, +127]. `scale` must have
// passed int8_scale_is_valid().
[[nodiscard]] inline int8_t int8_quantize(float v, float scale) noexcept {
    const double q = int8_round_nearest_even(static_cast<double>(v) / static_cast<double>(scale));
    if (q > static_cast<double>(kInt8QuantMax)) { return static_cast<int8_t>(kInt8QuantMax); }
    if (q < -static_cast<double>(kInt8QuantMax)) { return static_cast<int8_t>(-kInt8QuantMax); }
    return static_cast<int8_t>(q); // exact: |q| <= 127
}

// One stored code -> the dequantized fp32 value (int8 -> float is exact, so this
// is a single multiply).
[[nodiscard]] inline float int8_dequantize(int8_t q, float scale) noexcept {
    return static_cast<float>(q) * scale;
}

// max |x| over the whole tensor. A NaN / +-inf element is fatal: a nonfinite
// weight would silently poison every downstream GEMM, so quantization is
// fail-closed.
[[nodiscard]] inline float int8_tensor_absmax(luisa::span<const float> values) noexcept {
    float absmax = 0.0f;
    for (size_t i = 0u; i < values.size(); i++) {
        const float v = values[i];
        if (!std::isfinite(v)) [[unlikely]] {
            LUISA_ERROR("INT8 quantization: nonfinite value {} at element {} of {} (fail-closed).",
                        v, i, values.size());
        }
        const float a = std::fabs(v);
        absmax = a > absmax ? a : absmax;
    }
    return absmax;
}

// absmax -> per-tensor scale. An all-zero tensor (absmax == 0) gets 1.0; a
// negative/nonfinite absmax, or one whose scale would underflow/overflow, is
// fatal.
[[nodiscard]] inline float int8_scale_from_absmax(float absmax) noexcept {
    if (!std::isfinite(absmax) || absmax < 0.0f) [[unlikely]] {
        LUISA_ERROR("INT8 quantization: invalid absmax {} (must be finite and >= 0).", absmax);
    }
    if (absmax == 0.0f) { return kInt8UnitScale; }
    const float scale = absmax / static_cast<float>(kInt8QuantMax);
    if (!int8_scale_is_valid(scale)) [[unlikely]] {
        LUISA_ERROR("INT8 quantization: absmax {} yields the unrepresentable scale {} (fail-closed).",
                    absmax, scale);
    }
    return scale;
}

// Quantize a whole tensor in place of `dst` (same element count as `src`) and
// return the per-tensor scale the caller must store in the table entry.
[[nodiscard]] inline float int8_quantize_tensor(luisa::span<const float> src,
                                               luisa::span<int8_t> dst) noexcept {
    if (src.size() != dst.size()) [[unlikely]] {
        LUISA_ERROR("INT8 quantization: source has {} elements but the destination blob has {}.",
                    src.size(), dst.size());
    }
    const float scale = int8_scale_from_absmax(int8_tensor_absmax(src));
    for (size_t i = 0u; i < src.size(); i++) {
        dst[i] = int8_quantize(src[i], scale);
    }
    return scale;
}

// Dequantize a stored v3 blob into fp32 (the loader / host-upload path). The
// scale comes from the tensor's table entry and is re-validated here.
inline void int8_dequantize_tensor(luisa::span<const int8_t> src, luisa::span<float> dst,
                                   float scale) noexcept {
    if (!int8_scale_is_valid(scale)) [[unlikely]] {
        LUISA_ERROR("INT8 dequantization: the stored scale {} is not finite and positive (fail-closed).",
                    scale);
    }
    if (src.size() != dst.size()) [[unlikely]] {
        LUISA_ERROR("INT8 dequantization: the blob has {} elements but the destination has {}.",
                    src.size(), dst.size());
    }
    for (size_t i = 0u; i < src.size(); i++) {
        dst[i] = int8_dequantize(src[i], scale);
    }
}

}// namespace luisa::embed
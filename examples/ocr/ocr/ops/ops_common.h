#pragma once
// OCR engine -- shared device-side DSL helpers (Phase 2).
//
// Header-only device functions used by the ops kernels. These are recorded
// into the AST when a factory lambda runs; they contain NO host state and NO
// device object. All loops over runtime bounds use DSL control flow.
//
// Conventions:
// - Activations/accumulators are FP32. Storage-only packed reads (fp16/bf16/
//   int4/int8) decode to FP32 before any arithmetic.
// - Flattened 2D launches recover the linear id with ocr_linear_id() and the
//   (row, col) of a padded 2D grid with ocr_row_col().
#include <luisa/dsl/syntax.h>

namespace luisa::ocr {

// Linear thread id for a (possibly flattened) 1D launch. dispatch_size() is
// the TOTAL thread count per dimension, so this is exact on every backend and
// needs no extra kernel argument. When the host plan folded block rows into
// dispatch.y, dispatch_size().x is the padded x thread count.
[[nodiscard]] inline auto ocr_linear_id() noexcept {
    return compute::dispatch_y() * compute::dispatch_size().x + compute::dispatch_x();
}

// Recover (row, col) from a flattened 2D launch: the linear id spans
// padded_cols columns per row. Caller guards row < rows && col < cols.
inline void ocr_flattened_row_col(compute::Var<uint32_t> padded_cols,
                                  compute::Var<uint32_t> &row, compute::Var<uint32_t> &col) noexcept {
    const auto linear = ocr_linear_id();
    row = linear / padded_cols;
    col = linear % padded_cols;
}

// Exact bf16 -> fp32: the low 16 bits are the bf16 payload, shifted to the top.
[[nodiscard]] inline auto ocr_bf16_to_f32(compute::Var<uint32_t> bits) noexcept {
    return compute::as<float>(bits << 16u);
}

// Exact fp16 -> fp32 (all halves incl. subnormals are representable in fp32).
// Portable bit decomposition, no hardware half arithmetic.
[[nodiscard]] inline auto ocr_fp16_to_f32(compute::Var<uint32_t> h) noexcept {
    using namespace compute;
    Var<uint32_t> sign = (h >> 15u) & 1u;
    Var<uint32_t> exp = (h >> 10u) & 0x1Fu;
    Var<uint32_t> mant = h & 0x3FFu;
    Var<uint32_t> out_bits = def(0u);
    $if (exp == 0u) {
        $if (mant == 0u) {
            out_bits = sign << 31u;// +/- zero
        } $else {
            // Subnormal: normalize. value = mant * 2^-24; find leading bit.
            Var<uint32_t> e = def(0u);
            Var<uint32_t> m = def(mant);
            $while ((m & 0x400u) == 0u) {
                m = m << 1u;
                e += 1u;
            };
            m = m & 0x3FFu;
            out_bits = (sign << 31u) | ((127u - 15u - e + 1u) << 23u) | (m << 13u);
        };
    } $elif (exp == 0x1Fu) {
        out_bits = (sign << 31u) | (0xFFu << 23u) | (mant << 13u);// inf / NaN
    } $else {
        out_bits = (sign << 31u) | ((exp + 127u - 15u) << 23u) | (mant << 13u);
    };
    return as<float>(out_bits);
}

// Decode one packed fp16/bf16 element from a word array. `index` is the logical
// element index; two elements per word, low half first. `is_bf16` is a
// compile-time host bool captured by the factory.
[[nodiscard]] inline auto ocr_read_half(const compute::BufferVar<uint32_t> &packed,
                                        compute::Var<uint32_t> index, bool is_bf16) noexcept {
    using namespace compute;
    const auto word = packed.read(index >> 1u);
    const auto shift = (index & 1u) * 16u;
    const auto bits = (word >> shift) & 0xFFFFu;
    if (is_bf16) { return ocr_bf16_to_f32(bits); }
    return ocr_fp16_to_f32(bits);
}

// Decode one grouped INT4 element (Phase 1 semantics): unsigned nibble 0..15,
// LOW nibble first within each byte; eight nibbles per uint32 word, element
// 8j+c in bits[4c+3:4c]. Symmetric zero point is implicitly 8. Groups are
// contiguous along the flattened row-major order; the last group may be
// partial. dequant = (code - zp[g]) * scale[g].
[[nodiscard]] inline auto ocr_read_int4(const compute::BufferVar<uint32_t> &packed,
                                        const compute::BufferVar<float> &scales,
                                        const compute::BufferVar<uint32_t> &zero_points,
                                        compute::Var<uint32_t> index,
                                        compute::Var<uint32_t> group_size,
                                        bool symmetric) noexcept {
    using namespace compute;
    const auto word = packed.read(index >> 3u);
    const auto shift = (index & 7u) * 4u;
    const auto code = (word >> shift) & 0xFu;
    const auto group = index / group_size;
    const auto scale = scales.read(group);
    Var<uint32_t> zp = def(8u);
    if (!symmetric) { zp = zero_points.read(group); }
    const auto signed_code = as<int32_t>(code) - as<int32_t>(zp);
    return cast<float>(signed_code) * scale;
}

// Decode one grouped symmetric INT8 element: signed byte -127..127, four per
// word, element 4j+c in bits[8c+7:8c], sign-extended. dequant = code*scale[g].
[[nodiscard]] inline auto ocr_read_int8(const compute::BufferVar<uint32_t> &packed,
                                        const compute::BufferVar<float> &scales,
                                        compute::Var<uint32_t> index,
                                        compute::Var<uint32_t> group_size) noexcept {
    using namespace compute;
    const auto word = packed.read(index >> 2u);
    const auto shift = (index & 3u) * 8u;
    const auto byte = (word >> shift) & 0xFFu;
    // Sign-extend the low 8 bits to int32.
    const auto signed_byte = as<int32_t>(byte << 24u) >> 24u;
    const auto group = index / group_size;
    return cast<float>(signed_byte) * scales.read(group);
}

// FP32 activation epilogues (compile-time selected by the factory).
// Portable erf: the HLSL target (dx / vk-HLSL route) has no erf intrinsic, so
// erf is evaluated with the Abramowitz & Stegun 7.1.26 rational approximation
// (max abs error 1.5e-7), composed only of abs/exp/select so every backend
// computes the same value. Evaluated on |x/sqrt(2)| with the odd sign restored.
[[nodiscard]] inline auto ocr_erf(compute::Var<float> x) noexcept {
    using namespace compute;
    const auto z = abs(x);
    const auto t = 1.0f / (1.0f + 0.3275911f * z);
    const auto poly = t * (0.254829592f +
                           t * (-0.284496736f +
                                t * (1.421413741f +
                                     t * (-1.453152027f + t * 1.061405429f))));
    const auto erf_z = 1.0f - poly * exp(-(z * z));
    return select(-erf_z, erf_z, x >= 0.0f);
}
[[nodiscard]] inline auto ocr_gelu_erf(compute::Var<float> x) noexcept {
    return 0.5f * x * (1.0f + ocr_erf(x * 0.70710678118654752440f));
}
[[nodiscard]] inline auto ocr_gelu_tanh(compute::Var<float> x) noexcept {
    const auto inner = 0.79788456080286535588f * (x + 0.044715f * x * x * x);
    return 0.5f * x * (1.0f + compute::tanh(inner));
}
[[nodiscard]] inline auto ocr_relu(compute::Var<float> x) noexcept {
    return compute::max(x, 0.0f);
}
[[nodiscard]] inline auto ocr_relu6(compute::Var<float> x) noexcept {
    return compute::clamp(x, 0.0f, 6.0f);
}
[[nodiscard]] inline auto ocr_silu(compute::Var<float> x) noexcept {
    return x / (1.0f + compute::exp(-x));
}
[[nodiscard]] inline auto ocr_hard_swish(compute::Var<float> x) noexcept {
    return x * compute::clamp(x + 3.0f, 0.0f, 6.0f) / 6.0f;
}
[[nodiscard]] inline auto ocr_sigmoid(compute::Var<float> x) noexcept {
    return 1.0f / (1.0f + compute::exp(-x));
}

}// namespace luisa::ocr
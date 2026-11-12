// OCR engine -- quantization codecs + checked host arithmetic (Phase 1).
// See quantization.h for the layout contracts. No exceptions, no RTTI.
#include "ocr/quantization.h"

#include <cmath>
#include <cstring>

namespace luisa::ocr {

[[nodiscard]] bool checked_element_count(luisa::span<const uint64_t> dims, uint64_t &out) noexcept {
    uint64_t count = 1;
    for (auto d : dims) {
        if (!checked_mul(count, d, count)) { return false; }
    }
    out = count;
    return true;
}

// ---------------------------------------------------------------------------
// FP16 / BF16 (explicit round-to-nearest-even)
// ---------------------------------------------------------------------------

[[nodiscard]] float fp16_to_fp32(uint16_t h) noexcept {
    const uint32_t sign = static_cast<uint32_t>(h & 0x8000u) << 16u;
    const uint32_t exponent = (h >> 10u) & 0x1fu;
    const uint32_t mantissa = h & 0x3ffu;
    uint32_t bits = 0;
    if (exponent == 0) {
        if (mantissa == 0) {
            bits = sign;// +/-0
        } else {
            // Subnormal half: normalize into a float normal.
            uint32_t e = 127u - 15u + 1u;
            uint32_t m = mantissa;
            while ((m & 0x400u) == 0) {
                m <<= 1u;
                e -= 1u;
            }
            m &= 0x3ffu;
            bits = sign | (e << 23u) | (m << 13u);
        }
    } else if (exponent == 0x1fu) {
        // Inf / NaN (keep the payload; quiet the NaN).
        bits = sign | 0x7f800000u | (mantissa << 13u);
        if (mantissa != 0) { bits |= 0x00400000u; }
    } else {
        bits = sign | ((exponent + 127u - 15u) << 23u) | (mantissa << 13u);
    }
    float f = 0.0f;
    std::memcpy(&f, &bits, sizeof(f));
    return f;
}

[[nodiscard]] uint16_t fp32_to_fp16(float f) noexcept {
    uint32_t bits = 0;
    std::memcpy(&bits, &f, sizeof(bits));
    const uint32_t sign = (bits >> 16u) & 0x8000u;
    const int32_t exponent = static_cast<int32_t>((bits >> 23u) & 0xffu) - 127;
    const uint32_t mantissa = bits & 0x7fffffu;
    if (exponent == 128) {
        // Inf / NaN.
        if (mantissa == 0) { return static_cast<uint16_t>(sign | 0x7c00u); }
        uint16_t payload = static_cast<uint16_t>(mantissa >> 13u);
        if (payload == 0) { payload = 1; }// never turn NaN into inf
        return static_cast<uint16_t>(sign | 0x7c00u | payload | 0x0200u);
    }
    if (exponent > 15) {
        // Overflow -> +/-inf (IEEE RNE for round-to-nearest).
        return static_cast<uint16_t>(sign | 0x7c00u);
    }
    if (exponent >= -14) {
        // Normal half: RNE on the 13 dropped bits.
        const uint32_t half_m = mantissa >> 13u;
        const uint32_t rem = mantissa & 0x1fffu;
        uint32_t m = half_m;
        if (rem > 0x1000u || (rem == 0x1000u && (half_m & 1u) != 0)) { m += 1; }
        uint32_t e = static_cast<uint32_t>(exponent + 15);
        if (m == 0x400u) {// mantissa carried into the exponent
            m = 0;
            e += 1;
        }
        if (e >= 0x1fu) { return static_cast<uint16_t>(sign | 0x7c00u); }
        return static_cast<uint16_t>(sign | (e << 10u) | m);
    }
    if (exponent >= -24) {
        // Subnormal half (includes rounding up to the smallest normal).
        const uint32_t full = mantissa | 0x800000u;// 1.m, 24 bits
        const uint32_t shift = static_cast<uint32_t>(-exponent - 14 + 13);// 14..24
        const uint32_t m = full >> shift;
        const uint32_t rem = full & ((1u << shift) - 1u);
        const uint32_t halfway = 1u << (shift - 1u);
        uint32_t r = m;
        if (rem > halfway || (rem == halfway && (m & 1u) != 0)) { r += 1; }
        return static_cast<uint16_t>(sign | r);// r==0x400 becomes the smallest normal
    }
    if (exponent == -25) {
        // Value in [2^-25, 2^-24): compare against half of the smallest
        // subnormal (2^-25). Exactly 2^-25 (mantissa==0) is a tie -> round to
        // even (0). Anything strictly above rounds up to the smallest
        // subnormal 0x0001. (The previous code flushed these to zero.)
        if (mantissa != 0) { return static_cast<uint16_t>(sign | 0x0001u); }
        return static_cast<uint16_t>(sign);
    }
    // exponent < -25: strictly below half of the smallest subnormal -> +/-0.
    return static_cast<uint16_t>(sign);
}

[[nodiscard]] float bf16_to_fp32(uint16_t b) noexcept {
    const uint32_t bits = static_cast<uint32_t>(b) << 16u;
    float f = 0.0f;
    std::memcpy(&f, &bits, sizeof(f));
    return f;
}

[[nodiscard]] uint16_t fp32_to_bf16(float f) noexcept {
    uint32_t bits = 0;
    std::memcpy(&bits, &f, sizeof(bits));
    const uint32_t exponent = (bits >> 23u) & 0xffu;
    const uint32_t mantissa = bits & 0x7fffffu;
    if (exponent == 0xffu && mantissa != 0) {
        // NaN: keep it a quiet NaN with a nonzero payload.
        uint16_t upper = static_cast<uint16_t>(bits >> 16u);
        upper |= 0x0040u;
        if ((upper & 0x007fu) == 0) { upper |= 0x0001u; }
        return upper;
    }
    // RNE on the low 16 bits (inf/zero pass through unchanged).
    const uint32_t lsb = (bits >> 16u) & 1u;
    bits += 0x7fffu + lsb;
    return static_cast<uint16_t>(bits >> 16u);
}

// ---------------------------------------------------------------------------
// Grouped quantization
// ---------------------------------------------------------------------------

[[nodiscard]] luisa::string_view quant_kind_name(QuantKind kind) noexcept {
    switch (kind) {
        case QuantKind::none: return "none";
        case QuantKind::int4_symmetric: return "int4_symmetric";
        case QuantKind::int4_asymmetric: return "int4_asymmetric";
        case QuantKind::int8_symmetric: return "int8_symmetric";
        default: return "unknown";
    }
}

[[nodiscard]] bool group_count_for(uint64_t count, uint32_t group_size, uint64_t &out) noexcept {
    if (!is_supported_group_size(group_size)) { return false; }
    uint64_t padded = 0;
    if (!checked_add(count, group_size - 1u, padded)) { return false; }
    out = padded / group_size;
    return true;
}

[[nodiscard]] bool int4_packed_size(uint64_t count, uint64_t &out) noexcept {
    uint64_t padded = 0;
    if (!checked_add(count, 1u, padded)) { return false; }
    out = padded / 2u;
    return true;
}

[[nodiscard]] Status validate_quant_metadata(
    QuantKind kind, uint32_t group_size, uint64_t element_count,
    luisa::span<const float> scales, luisa::span<const uint8_t> zero_points) {
    if (kind == QuantKind::none) {
        return Status::error(StatusCode::invalid_argument, "quantization metadata requires a quantized kind", "quant.kind");
    }
    if (kind != QuantKind::int4_symmetric && kind != QuantKind::int4_asymmetric && kind != QuantKind::int8_symmetric) {
        return Status::error(StatusCode::unsupported, "unknown quantization kind", "quant.kind");
    }
    if (!is_supported_group_size(group_size)) {
        return Status::error(StatusCode::unsupported, "group size must be 64 or 128", "quant.group_size");
    }
    uint64_t groups = 0;
    if (!group_count_for(element_count, group_size, groups)) {
        return Status::error(StatusCode::invalid_argument, "element count overflows group computation", "quant.element_count");
    }
    if (static_cast<uint64_t>(scales.size()) != groups) {
        return Status::error(StatusCode::invalid_argument, "scale count does not match the group count", "quant.scales");
    }
    const bool asymmetric = kind == QuantKind::int4_asymmetric;
    if (asymmetric) {
        if (static_cast<uint64_t>(zero_points.size()) != groups) {
            return Status::error(StatusCode::invalid_argument, "zero-point count does not match the group count", "quant.zero_points");
        }
        for (auto zp : zero_points) {
            if (zp > 15u) {
                return Status::error(StatusCode::invalid_argument, "INT4 zero point outside 0..15", "quant.zero_points");
            }
        }
    } else if (!zero_points.empty()) {
        return Status::error(StatusCode::invalid_argument, "symmetric quantization must not store zero points (implicit)", "quant.zero_points");
    }
    for (auto s : scales) {
        if (!std::isfinite(s) || s < 0.0f) {
            return Status::error(StatusCode::invalid_argument, "quantization scale must be finite and non-negative", "quant.scales");
        }
    }
    return Status::success();
}

[[nodiscard]] Status pack_int4_codes(luisa::span<const uint8_t> codes,
                                     luisa::vector<uint8_t> &packed_out) {
    for (auto c : codes) {
        if (c > 15u) {
            return Status::error(StatusCode::invalid_argument, "INT4 code outside 0..15", "codes");
        }
    }
    uint64_t packed_size = 0;
    if (!int4_packed_size(codes.size(), packed_size)) {
        return Status::error(StatusCode::invalid_argument, "code count overflows packed size", "codes");
    }
    packed_out.clear();
    packed_out.resize(static_cast<size_t>(packed_size), 0u);
    for (size_t i = 0; i < codes.size(); i++) {
        packed_out[i / 2u] |= static_cast<uint8_t>(codes[i] << ((i % 2u) * 4u));
    }
    return Status::success();
}

[[nodiscard]] Status unpack_int4_codes(luisa::span<const uint8_t> packed,
                                       uint64_t count, luisa::span<uint8_t> codes_out) {
    uint64_t packed_size = 0;
    if (!int4_packed_size(count, packed_size)) {
        return Status::error(StatusCode::invalid_argument, "element count overflows packed size", "count");
    }
    if (static_cast<uint64_t>(packed.size()) != packed_size) {
        return Status::error(StatusCode::invalid_argument, "packed byte size does not match the element count", "packed");
    }
    if (static_cast<uint64_t>(codes_out.size()) != count) {
        return Status::error(StatusCode::invalid_argument, "output span does not match the element count", "codes_out");
    }
    for (uint64_t i = 0; i < count; i++) {
        const auto byte = packed[static_cast<size_t>(i / 2u)];
        codes_out[static_cast<size_t>(i)] = static_cast<uint8_t>((byte >> ((i % 2u) * 4u)) & 0x0fu);
    }
    if ((count % 2u) != 0 && (packed.back() & 0xf0u) != 0) {
        return Status::error(StatusCode::invalid_argument, "nonzero padding nibble in odd-length INT4 payload", "packed");
    }
    return Status::success();
}

[[nodiscard]] Status dequantize_int4(
    luisa::span<const uint8_t> packed, uint64_t element_count,
    luisa::span<const float> scales, luisa::span<const uint8_t> zero_points,
    uint32_t group_size, bool symmetric, luisa::span<float> out) {
    const auto kind = symmetric ? QuantKind::int4_symmetric : QuantKind::int4_asymmetric;
    auto meta = validate_quant_metadata(kind, group_size, element_count, scales, zero_points);
    if (!meta.ok()) { return meta; }
    uint64_t packed_size = 0;
    if (!int4_packed_size(element_count, packed_size) ||
        static_cast<uint64_t>(packed.size()) != packed_size) {
        return Status::error(StatusCode::invalid_argument, "packed byte size does not match the element count", "packed");
    }
    if (static_cast<uint64_t>(out.size()) != element_count) {
        return Status::error(StatusCode::invalid_argument, "output span does not match the element count", "out");
    }
    if ((element_count % 2u) != 0 && !packed.empty() && (packed.back() & 0xf0u) != 0) {
        return Status::error(StatusCode::invalid_argument, "nonzero padding nibble in odd-length INT4 payload", "packed");
    }
    for (uint64_t i = 0; i < element_count; i++) {
        const auto byte = packed[static_cast<size_t>(i / 2u)];
        const auto code = static_cast<int32_t>((byte >> ((i % 2u) * 4u)) & 0x0fu);
        const auto g = static_cast<size_t>(i / group_size);
        const int32_t zp = symmetric ? 8 : static_cast<int32_t>(zero_points[g]);
        out[static_cast<size_t>(i)] = static_cast<float>(code - zp) * scales[g];
    }
    return Status::success();
}

[[nodiscard]] Status dequantize_int8(
    luisa::span<const int8_t> codes, uint64_t element_count,
    luisa::span<const float> scales, uint32_t group_size, luisa::span<float> out) {
    auto meta = validate_quant_metadata(QuantKind::int8_symmetric, group_size, element_count, scales, {});
    if (!meta.ok()) { return meta; }
    if (static_cast<uint64_t>(codes.size()) != element_count) {
        return Status::error(StatusCode::invalid_argument, "code count does not match the element count", "codes");
    }
    if (static_cast<uint64_t>(out.size()) != element_count) {
        return Status::error(StatusCode::invalid_argument, "output span does not match the element count", "out");
    }
    for (uint64_t i = 0; i < element_count; i++) {
        const auto code = codes[static_cast<size_t>(i)];
        if (code == -128) {
            return Status::error(StatusCode::invalid_argument, "INT8 code -128 is reserved and must not appear", "codes");
        }
        out[static_cast<size_t>(i)] = static_cast<float>(code) * scales[static_cast<size_t>(i / group_size)];
    }
    return Status::success();
}

namespace {

// Round-to-nearest-even of a finite value into [lo, hi].
[[nodiscard]] int32_t round_clamped(float v, int32_t lo, int32_t hi) noexcept {
    const float r = std::nearbyintf(v);// host default rounding mode is RNE
    float clamped = r;
    if (clamped < static_cast<float>(lo)) { clamped = static_cast<float>(lo); }
    if (clamped > static_cast<float>(hi)) { clamped = static_cast<float>(hi); }
    return static_cast<int32_t>(clamped);
}

}// namespace

[[nodiscard]] Status quantize_int4(
    luisa::span<const float> values, uint32_t group_size, bool symmetric,
    luisa::vector<uint8_t> &packed_out, luisa::vector<float> &scales_out,
    luisa::vector<uint8_t> &zero_points_out) {
    if (!is_supported_group_size(group_size)) {
        return Status::error(StatusCode::unsupported, "group size must be 64 or 128", "group_size");
    }
    for (auto v : values) {
        if (!std::isfinite(v)) {
            return Status::error(StatusCode::invalid_argument, "cannot quantize a non-finite value", "values");
        }
    }
    uint64_t groups = 0;
    if (!group_count_for(values.size(), group_size, groups)) {
        return Status::error(StatusCode::invalid_argument, "element count overflows group computation", "values");
    }
    luisa::vector<uint8_t> codes;
    codes.resize(values.size(), 0u);
    scales_out.clear();
    scales_out.resize(static_cast<size_t>(groups), 1.0f);
    zero_points_out.clear();
    if (!symmetric) { zero_points_out.resize(static_cast<size_t>(groups), 8u); }
    for (uint64_t g = 0; g < groups; g++) {
        const auto begin = static_cast<size_t>(g * group_size);
        const auto end = static_cast<size_t>(std::min<uint64_t>(values.size(), (g + 1) * group_size));
        float lo = 0.0f, hi = 0.0f;
        for (size_t i = begin; i < end; i++) {
            lo = std::min(lo, values[i]);
            hi = std::max(hi, values[i]);
        }
        float scale = 1.0f;
        int32_t zp = 8;
        if (symmetric) {
            const float max_abs = std::max(std::fabs(lo), std::fabs(hi));
            if (max_abs > 0.0f) { scale = max_abs / 7.0f; }
            // Clamp the scale so the extreme code never rounds past +/-7.
            scale = std::max(scale, max_abs / 7.5f);
        } else {
            const float range = hi - lo;
            if (range > 0.0f) {
                scale = range / 15.0f;
                scale = std::max(scale, range / 15.5f);
                zp = round_clamped(-lo / scale, 0, 15);
            }
        }
        // Guard against a denormal/zero scale produced by extreme inputs.
        if (!(scale > 0.0f) || !std::isfinite(scale)) { scale = 1.0f; zp = 8; }
        scales_out[static_cast<size_t>(g)] = scale;
        if (!symmetric) { zero_points_out[static_cast<size_t>(g)] = static_cast<uint8_t>(zp); }
        for (size_t i = begin; i < end; i++) {
            const int32_t code = round_clamped(values[i] / scale + static_cast<float>(zp), 0, 15);
            codes[i] = static_cast<uint8_t>(code);
        }
    }
    return pack_int4_codes(codes, packed_out);
}

[[nodiscard]] Status quantize_int8(
    luisa::span<const float> values, uint32_t group_size,
    luisa::vector<int8_t> &codes_out, luisa::vector<float> &scales_out) {
    if (!is_supported_group_size(group_size)) {
        return Status::error(StatusCode::unsupported, "group size must be 64 or 128", "group_size");
    }
    for (auto v : values) {
        if (!std::isfinite(v)) {
            return Status::error(StatusCode::invalid_argument, "cannot quantize a non-finite value", "values");
        }
    }
    uint64_t groups = 0;
    if (!group_count_for(values.size(), group_size, groups)) {
        return Status::error(StatusCode::invalid_argument, "element count overflows group computation", "values");
    }
    codes_out.clear();
    codes_out.resize(values.size(), 0);
    scales_out.clear();
    scales_out.resize(static_cast<size_t>(groups), 1.0f);
    for (uint64_t g = 0; g < groups; g++) {
        const auto begin = static_cast<size_t>(g * group_size);
        const auto end = static_cast<size_t>(std::min<uint64_t>(values.size(), (g + 1) * group_size));
        float max_abs = 0.0f;
        for (size_t i = begin; i < end; i++) {
            max_abs = std::max(max_abs, std::fabs(values[i]));
        }
        float scale = 1.0f;
        if (max_abs > 0.0f) {
            scale = std::max(max_abs / 127.0f, max_abs / 127.5f);
        }
        if (!(scale > 0.0f) || !std::isfinite(scale)) { scale = 1.0f; }
        scales_out[static_cast<size_t>(g)] = scale;
        for (size_t i = begin; i < end; i++) {
            codes_out[i] = static_cast<int8_t>(round_clamped(values[i] / scale, -127, 127));
        }
    }
    return Status::success();
}

}// namespace luisa::ocr
#pragma once
// OCR engine -- host-side quantization codecs and checked host arithmetic
// (Phase 1). Host-only: no DSL/runtime device includes.
//
// Contents:
// - Checked 64-bit add/mul/align helpers shared by every host artifact parser
//   (package, safetensors, image, document, preprocess, feature bundle).
// - Explicit round-to-nearest-even FP32<->FP16/BF16 bit conversions. There is
//   NO silent reinterpretation between FP16 and BF16 anywhere in the engine.
// - Grouped symmetric/asymmetric INT4 (packed codes + per-group FP32 scales
//   + per-group uint8 zero points, group size 64 or 128) and optional grouped
//   symmetric INT8 codecs, with fail-closed validation of every range.
//
// INT4 layout contract (v1, shared with tools/pack_model.py):
// - Codes are unsigned nibbles 0..15, two per byte, LOW nibble first (element
//   2*i in bits 0..3, element 2*i+1 in bits 4..7). Odd element counts pad the
//   final high nibble with 0.
// - Groups are contiguous along the flattened row-major element order; the
//   final group may be partial. group_count = ceil(N / group_size).
// - Asymmetric: dequant = (code - zero_point[g]) * scale[g], zero_point in
//   0..15 stored as one byte per group.
// - Symmetric: zero point is IMPLICITLY 8 (code 8 == 0.0); no zero-point
//   bytes are stored. dequant = (code - 8) * scale[g].
// - Scales are FP32, finite and non-negative. A zero scale decodes the whole
//   group to zero; the host quantizer emits scale 1 for all-zero groups.
//
// INT8 layout contract (v1): signed codes -127..127 (never -128), grouped
// symmetric, dequant = code * scale[g], same group rules as INT4.
#include <cstdint>

#include <luisa/core/stl/memory.h>
#include <luisa/core/stl/vector.h>

#include "ocr/status.h"

namespace luisa::ocr {

// ---------------------------------------------------------------------------
// Checked 64-bit host arithmetic. Every size/offset computation in the host
// artifact parsers goes through these; on overflow they return false and
// leave `out` untouched.
// ---------------------------------------------------------------------------

[[nodiscard]] inline bool checked_add(uint64_t a, uint64_t b, uint64_t &out) noexcept {
    if (b > UINT64_MAX - a) { return false; }
    out = a + b;
    return true;
}

[[nodiscard]] inline bool checked_mul(uint64_t a, uint64_t b, uint64_t &out) noexcept {
    if (a != 0 && b > UINT64_MAX / a) { return false; }
    out = a * b;
    return true;
}

// Align `value` up to `alignment` (a power of two). Fails on overflow or a
// non-power-of-two alignment.
[[nodiscard]] inline bool checked_align_up(uint64_t value, uint64_t alignment, uint64_t &out) noexcept {
    if (alignment == 0 || (alignment & (alignment - 1)) != 0) { return false; }
    const auto mask = alignment - 1;
    if (value > UINT64_MAX - mask) { return false; }
    out = (value + mask) & ~mask;
    return true;
}

// Product of all dimensions (checked). An empty shape yields 1 (scalar).
[[nodiscard]] bool checked_element_count(luisa::span<const uint64_t> dims, uint64_t &out) noexcept;

// ---------------------------------------------------------------------------
// Explicit RNE floating-point conversions (bit-exact, no hardware dependent
// rounding mode). NaN inputs produce a quiet NaN of the target format;
// infinities are preserved; FP32 values that overflow FP16 become +/-inf
// (IEEE RNE), which artifact validators then reject where finiteness is
// required (e.g. quantization scales).
// ---------------------------------------------------------------------------

[[nodiscard]] float fp16_to_fp32(uint16_t h) noexcept;
[[nodiscard]] uint16_t fp32_to_fp16(float f) noexcept;
[[nodiscard]] float bf16_to_fp32(uint16_t b) noexcept;
[[nodiscard]] uint16_t fp32_to_bf16(float f) noexcept;

// ---------------------------------------------------------------------------
// Grouped quantization
// ---------------------------------------------------------------------------

enum class QuantKind : uint32_t {
    none = 0,
    int4_symmetric = 1,
    int4_asymmetric = 2,
    int8_symmetric = 3,
};

[[nodiscard]] luisa::string_view quant_kind_name(QuantKind kind) noexcept;

// Only 64 and 128 are supported group sizes in format v1 (plan §6.2).
[[nodiscard]] constexpr bool is_supported_group_size(uint32_t group_size) noexcept {
    return group_size == 64u || group_size == 128u;
}

// Number of groups covering `count` elements (checked; count may be 0).
[[nodiscard]] bool group_count_for(uint64_t count, uint32_t group_size, uint64_t &out) noexcept;
// Packed INT4 byte size for `count` codes (checked).
[[nodiscard]] bool int4_packed_size(uint64_t count, uint64_t &out) noexcept;

// Validate the metadata of one quantized tensor before any decode. Checks
// kind/group-size consistency, scale/zero-point counts, scale finiteness and
// sign, and zero-point ranges. `scales`/`zero_points` are the actual stored
// arrays (zero_points empty for symmetric kinds).
[[nodiscard]] Status validate_quant_metadata(
    QuantKind kind, uint32_t group_size, uint64_t element_count,
    luisa::span<const float> scales, luisa::span<const uint8_t> zero_points);

// Pack/unpack raw unsigned nibble codes (0..15). unpack validates that the
// padding nibble of an odd count is zero and that packed.size() is exactly
// ceil(count/2).
[[nodiscard]] Status pack_int4_codes(luisa::span<const uint8_t> codes,
                                     luisa::vector<uint8_t> &packed_out);
[[nodiscard]] Status unpack_int4_codes(luisa::span<const uint8_t> packed,
                                       uint64_t count, luisa::span<uint8_t> codes_out);

// Dequantize INT4/INT8 to FP32. All sizes are validated against `element_count`
// and the metadata; malformed input returns a Status, never reads OOB.
[[nodiscard]] Status dequantize_int4(
    luisa::span<const uint8_t> packed, uint64_t element_count,
    luisa::span<const float> scales, luisa::span<const uint8_t> zero_points,
    uint32_t group_size, bool symmetric, luisa::span<float> out);
[[nodiscard]] Status dequantize_int8(
    luisa::span<const int8_t> codes, uint64_t element_count,
    luisa::span<const float> scales, uint32_t group_size, luisa::span<float> out);

// Host quantizers (used by the packer, fixtures and round-trip tests).
// Deterministic: per-group min/max (asymmetric) or max-abs (symmetric) with
// RNE code rounding; all-zero groups get scale 1 and code == zero point.
[[nodiscard]] Status quantize_int4(
    luisa::span<const float> values, uint32_t group_size, bool symmetric,
    luisa::vector<uint8_t> &packed_out, luisa::vector<float> &scales_out,
    luisa::vector<uint8_t> &zero_points_out);
[[nodiscard]] Status quantize_int8(
    luisa::span<const float> values, uint32_t group_size,
    luisa::vector<int8_t> &codes_out, luisa::vector<float> &scales_out);

}// namespace luisa::ocr
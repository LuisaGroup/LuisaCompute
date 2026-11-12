#pragma once
// Shared helpers for OCR Phase 2 device tests (synthetic data only).
// Deterministic LCG (no <random>), luisa::span transfers, tolerance compare.
#include <cmath>
#include <cstdint>

#include <luisa/core/logging.h>
#include <luisa/core/stl/vector.h>
#include <luisa/runtime/buffer.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/stream.h>

#include "ut/ut.hpp"

#include "ocr/quantization.h"

namespace luisa::ocr::test {

struct Lcg {
    uint32_t state = 0x12345678u;
    [[nodiscard]] uint32_t next() noexcept {
        state = state * 1664525u + 1013904223u;
        return state >> 8;
    }
    [[nodiscard]] float uniform(float lo, float hi) noexcept {
        return lo + (hi - lo) * (static_cast<float>(next() % 8388608u) / 8388608.f);
    }
    void fill(luisa::vector<float> &v, float lo, float hi) noexcept {
        for (auto &x : v) { x = uniform(lo, hi); }
    }
};

template<typename T>
[[nodiscard]] compute::Buffer<T> upload(compute::Device &device, compute::Stream &stream,
                                        luisa::span<const T> data) {
    auto buf = device.create_buffer<T>(data.size() == 0 ? 1 : data.size());
    if (!data.empty()) { stream << buf.copy_from(data); }
    return buf;
}

template<typename T>
void download(compute::Stream &stream, const compute::Buffer<T> &buf, luisa::vector<T> &out, size_t count) {
    out.resize(count);
    if (count != 0) { stream << buf.view(0, count).copy_to(luisa::span<T>{out}); }
}

// Per-element gate |got-ref| <= atol + rtol*|ref|; NaN must match NaN.
// Accumulates failures, one expect at the end (test skill convention).
inline void expect_close(luisa::span<const float> got, luisa::span<const float> ref,
                         float atol, float rtol, const char *what) {
    bool ok = got.size() == ref.size();
    size_t first_bad = 0;
    float worst = 0.f;
    for (size_t i = 0; ok && i < got.size(); i++) {
        const float g = got[i], r = ref[i];
        if (std::isnan(r)) {
            if (!std::isnan(g)) { ok = false; first_bad = i; }
            continue;
        }
        const float diff = std::abs(g - r);
        worst = std::max(worst, diff);
        if (!(diff <= atol + rtol * std::abs(r))) { ok = false; first_bad = i; }
    }
    if (!ok && first_bad < got.size() && first_bad < ref.size()) {
        LUISA_WARNING("{}: mismatch at [{}]: got {} expected {} (worst abs diff {})",
                      what, first_bad, got[first_bad], ref[first_bad], worst);
    }
    boost::ut::expect(ok) << what;
}

// Pack fp16/bf16 values (already host-rounded) into uint32 words, low half first.
[[nodiscard]] inline luisa::vector<uint32_t> pack_half_words(luisa::span<const float> values, bool bf16,
                                                             luisa::vector<float> &rounded) {
    rounded.resize(values.size());
    luisa::vector<uint32_t> words((values.size() + 1u) / 2u, 0u);
    for (size_t i = 0; i < values.size(); i++) {
        const uint16_t h = bf16 ? fp32_to_bf16(values[i]) : fp32_to_fp16(values[i]);
        rounded[i] = bf16 ? bf16_to_fp32(h) : fp16_to_fp32(h);
        words[i / 2u] |= static_cast<uint32_t>(h) << (16u * (i % 2u));
    }
    return words;
}

// Pack INT4 codes (0..15) into uint32 words: 8 nibbles/word, low nibble first.
[[nodiscard]] inline luisa::vector<uint32_t> pack_int4_words(luisa::span<const uint8_t> codes) {
    luisa::vector<uint32_t> words((codes.size() + 7u) / 8u, 0u);
    for (size_t i = 0; i < codes.size(); i++) {
        words[i / 8u] |= static_cast<uint32_t>(codes[i] & 0xFu) << (4u * (i % 8u));
    }
    return words;
}

// Pack signed INT8 codes into uint32 words: 4 bytes/word, low byte first.
[[nodiscard]] inline luisa::vector<uint32_t> pack_int8_words(luisa::span<const int8_t> codes) {
    luisa::vector<uint32_t> words((codes.size() + 3u) / 4u, 0u);
    for (size_t i = 0; i < codes.size(); i++) {
        words[i / 4u] |= static_cast<uint32_t>(static_cast<uint8_t>(codes[i])) << (8u * (i % 4u));
    }
    return words;
}

[[nodiscard]] inline luisa::vector<uint32_t> widen_bytes(luisa::span<const uint8_t> bytes) {
    luisa::vector<uint32_t> out(bytes.size());
    for (size_t i = 0; i < bytes.size(); i++) { out[i] = bytes[i]; }
    return out;
}

}// namespace luisa::ocr::test
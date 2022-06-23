//
// Created by mike on 22-6-9.
//

#pragma once

#include <core/stl.h>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
using float16_t = int16_t;
using float32x4_t = __m128;
using int64x2_t = __m128i;
#elif defined(__aarch64__)
#include <arm_neon.h>
#else
#error Unsupported platform for the LLVM backend to correctly detect the ABI
#endif

namespace luisa::compute::llvm::detail {

[[nodiscard]] inline auto decode_float4(int64_t v0, int64_t v1) noexcept {
    return luisa::bit_cast<float4>(int64x2_t{v0, v1});
}

[[nodiscard]] inline auto decode_int4(int64_t v0, int64_t v1) noexcept {
    return luisa::bit_cast<int4>(int64x2_t{v0, v1});
}

[[nodiscard]] inline auto decode_uint4(int64_t v0, int64_t v1) noexcept {
    return luisa::bit_cast<uint4>(int64x2_t{v0, v1});
}

[[nodiscard]] inline auto decode_uint3(int64_t v0, int64_t v1) noexcept {
    return luisa::bit_cast<uint3>(int64x2_t{v0, v1});
}

[[nodiscard]] inline auto decode_uint2(int64_t x) noexcept {
    return luisa::bit_cast<uint2>(x);
}

[[nodiscard]] inline auto decode_float2(int64_t x) noexcept {
    return luisa::bit_cast<float2>(x);
}

[[nodiscard]] inline auto encode_int4(int4 x) noexcept {
    return luisa::bit_cast<float32x4_t>(x);
}

[[nodiscard]] inline auto encode_uint4(uint4 x) noexcept {
    return luisa::bit_cast<float32x4_t>(x);
}

[[nodiscard]] inline auto encode_float4(float4 x) noexcept {
    return luisa::bit_cast<float32x4_t>(x);
}

}// namespace luisa::compute::llvm::detail

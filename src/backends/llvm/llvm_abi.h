//
// Created by mike on 22-6-9.
//

#pragma once

#include <core/intrin.h>
#include <core/stl.h>

namespace luisa::compute::llvm::detail {

[[nodiscard]] inline auto decode_float4(int64_t v0, int64_t v1) noexcept {
    return luisa::bit_cast<float4>(std::array{v0, v1});
}

[[nodiscard]] inline auto decode_int4(int64_t v0, int64_t v1) noexcept {
    return luisa::bit_cast<int4>(std::array{v0, v1});
}

[[nodiscard]] inline auto decode_uint4(int64_t v0, int64_t v1) noexcept {
    return luisa::bit_cast<uint4>(std::array{v0, v1});
}

[[nodiscard]] inline auto decode_uint3(int64_t v0, int64_t v1) noexcept {
    return luisa::bit_cast<uint3>(std::array{v0, v1});
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

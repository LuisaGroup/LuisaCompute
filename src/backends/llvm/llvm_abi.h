//
// Created by mike on 22-6-9.
//

#pragma once

#include <core/stl.h>

namespace luisa::compute::llvm::detail {

struct alignas(16) ulong2 {
    uint64_t x;
    uint64_t y;
};

struct alignas(16) ulong4 {
    uint64_t x;
    uint64_t y;
    uint64_t z;
    uint64_t w;
};

[[nodiscard]] inline auto decode_float4(uint64_t v0, uint64_t v1) noexcept {
    return luisa::bit_cast<float4>(ulong2{v0, v1});
}

[[nodiscard]] inline auto decode_int4(uint64_t v0, uint64_t v1) noexcept {
    return luisa::bit_cast<int4>(ulong2{v0, v1});
}

[[nodiscard]] inline auto decode_uint4(uint64_t v0, uint64_t v1) noexcept {
    return luisa::bit_cast<uint4>(ulong2{v0, v1});
}

[[nodiscard]] inline auto decode_uint3(uint64_t v0, uint64_t v1) noexcept {
    return luisa::bit_cast<uint3>(ulong2{v0, v1});
}

[[nodiscard]] inline auto decode_uint2(uint64_t x) noexcept {
    return luisa::bit_cast<uint2>(x);
}

[[nodiscard]] inline auto decode_float2(uint64_t x) noexcept {
    return luisa::bit_cast<float2>(x);
}

[[nodiscard]] inline auto encode_int4(int4 x) noexcept {
    return luisa::bit_cast<ulong2>(x);
}

[[nodiscard]] inline auto encode_uint4(uint4 x) noexcept {
    return luisa::bit_cast<ulong2>(x);
}

[[nodiscard]] inline auto encode_float4(float4 x) noexcept {
    return luisa::bit_cast<ulong2>(x);
}

}// namespace luisa::compute::llvm::detail

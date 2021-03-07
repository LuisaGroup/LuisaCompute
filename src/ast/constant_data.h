//
// Created by Mike Smith on 2021/3/6.
//

#pragma once

#include <variant>
#include <memory>
#include <span>
#include <mutex>
#include <vector>
#include <array>

#include <core/data_types.h>
#include <core/concepts.h>

namespace luisa::compute {

class ConstantData {

public:
    using View = std::variant<
        std::span<const bool>, std::span<const float>, std::span<const char>, std::span<const uchar>, std::span<const short>, std::span<const ushort>, std::span<const int>, std::span<const uint>,
        std::span<const bool2>, std::span<const float2>, std::span<const char2>, std::span<const uchar2>, std::span<const short2>, std::span<const ushort2>, std::span<const int2>, std::span<const uint2>,
        std::span<const bool3>, std::span<const float3>, std::span<const char3>, std::span<const uchar3>, std::span<const short3>, std::span<const ushort3>, std::span<const int3>, std::span<const uint3>,
        std::span<const bool4>, std::span<const float4>, std::span<const char4>, std::span<const uchar4>, std::span<const short4>, std::span<const ushort4>, std::span<const int4>, std::span<const uint4>,
        std::span<const float3x3>, std::span<const float4x4>>;

private:
    std::unique_ptr<std::byte[]> _storage;
    View _view;
    uint64_t _hash;
    
    ConstantData(std::unique_ptr<std::byte[]> s, View v, uint64_t hash) noexcept
        : _storage{std::move(s)}, _view{v}, _hash{hash}{}

public:
    [[nodiscard]] static uint64_t create(View data) noexcept;
    [[nodiscard]] static View view(uint64_t hash) noexcept;
};

}// namespace luisa::compute

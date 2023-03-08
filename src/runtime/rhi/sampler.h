//
// Created by Mike Smith on 2021/7/2.
//

#pragma once

#include <cstdint>
#include <core/basic_types.h>

namespace luisa::compute {

class Sampler {

public:
    enum struct Filter : uint8_t {
        POINT,
        LINEAR_POINT,
        LINEAR_LINEAR,
        ANISOTROPIC
    };

    enum struct Address : uint8_t {
        EDGE,
        REPEAT,
        MIRROR,
        ZERO
    };

private:
    Filter _filter{Filter::POINT};
    Address _address{Address::EDGE};

public:
    constexpr Sampler() noexcept = default;
    constexpr Sampler(Filter filter, Address address) noexcept
        : _filter{filter}, _address{address} {}

    [[nodiscard]] uint code() const noexcept { return (to_underlying(_filter) << 2u) | (to_underlying(_address)); }
    [[nodiscard]] static constexpr auto decode(uint code) noexcept {
        return Sampler{static_cast<Filter>(code >> 2u),
                       static_cast<Address>(code & 0x03u)};
    }

    [[nodiscard]] static constexpr auto point_edge() noexcept { return Sampler{Filter::POINT, Address::EDGE}; }
    [[nodiscard]] static constexpr auto point_repeat() noexcept { return Sampler{Filter::POINT, Address::REPEAT}; }
    [[nodiscard]] static constexpr auto point_mirror() noexcept { return Sampler{Filter::POINT, Address::MIRROR}; }
    [[nodiscard]] static constexpr auto point_zero() noexcept { return Sampler{Filter::POINT, Address::ZERO}; }
    [[nodiscard]] static constexpr auto linear_point_edge() noexcept { return Sampler{Filter::LINEAR_POINT, Address::EDGE}; }
    [[nodiscard]] static constexpr auto linear_point_repeat() noexcept { return Sampler{Filter::LINEAR_POINT, Address::REPEAT}; }
    [[nodiscard]] static constexpr auto linear_point_mirror() noexcept { return Sampler{Filter::LINEAR_POINT, Address::MIRROR}; }
    [[nodiscard]] static constexpr auto linear_point_zero() noexcept { return Sampler{Filter::LINEAR_POINT, Address::ZERO}; }
    [[nodiscard]] static constexpr auto linear_linear_edge() noexcept { return Sampler{Filter::LINEAR_LINEAR, Address::EDGE}; }
    [[nodiscard]] static constexpr auto linear_linear_repeat() noexcept { return Sampler{Filter::LINEAR_LINEAR, Address::REPEAT}; }
    [[nodiscard]] static constexpr auto linear_linear_mirror() noexcept { return Sampler{Filter::LINEAR_LINEAR, Address::MIRROR}; }
    [[nodiscard]] static constexpr auto linear_linear_zero() noexcept { return Sampler{Filter::LINEAR_LINEAR, Address::ZERO}; }
    [[nodiscard]] static constexpr auto anisotropic_edge() noexcept { return Sampler{Filter::ANISOTROPIC, Address::EDGE}; }
    [[nodiscard]] static constexpr auto anisotropic_repeat() noexcept { return Sampler{Filter::ANISOTROPIC, Address::REPEAT}; }
    [[nodiscard]] static constexpr auto anisotropic_mirror() noexcept { return Sampler{Filter::ANISOTROPIC, Address::MIRROR}; }
    [[nodiscard]] static constexpr auto anisotropic_zero() noexcept { return Sampler{Filter::ANISOTROPIC, Address::ZERO}; }

    [[nodiscard]] auto address() const noexcept { return _address; }
    [[nodiscard]] auto filter() const noexcept { return _filter; }
    [[nodiscard]] auto set_address(Address a) noexcept { _address = a; }
    [[nodiscard]] auto set_filter(Filter f) noexcept { _filter = f; }
    [[nodiscard]] auto operator==(Sampler rhs) const noexcept { return code() == rhs.code(); }
};

}// namespace luisa::compute

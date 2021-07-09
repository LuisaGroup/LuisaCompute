//
// Created by Mike Smith on 2021/7/2.
//

#pragma once

#include <cstdint>
#include <core/hash.h>
#include <core/basic_types.h>

namespace luisa::compute {

class TextureSampler {

public:
    struct Hash {
        [[nodiscard]] auto operator()(TextureSampler s) const noexcept {
            return s.hash();
        }
    };

    enum struct AddressMode : uint8_t {
        EDGE,
        REPEAT,
        MIRROR,
        ZERO
    };

    enum struct FilterMode : uint8_t {
        NEAREST,
        LINEAR
    };

    enum struct MipFilterMode : uint8_t {
        NONE,
        NEAREST,
        LINEAR
    };

    [[nodiscard]] static auto compute_mip_levels(uint3 size, uint requested_levels) noexcept {
        auto max_size = std::max({size.x, size.y, size.z});
        auto max_levels = 0u;
        while (max_size != 0u) {
            max_size >>= 1u;
            max_levels++;
        }
        return requested_levels == 0u
                   ? max_levels
                   : std::min(requested_levels, max_levels);
    }

    [[nodiscard]] static auto compute_mip_levels(uint2 size, uint requested_levels) noexcept {
        return compute_mip_levels(make_uint3(size, 1u), requested_levels);
    }

private:
    uint _max_anisotropy{1u};
    std::array<AddressMode, 3u> _address{AddressMode::EDGE, AddressMode::EDGE, AddressMode::EDGE};
    FilterMode _filter{FilterMode::LINEAR};
    MipFilterMode _mip_filter{MipFilterMode::NEAREST};
    bool _normalized_coords{true};

public:
    auto &set_max_anisotropy(uint a) noexcept {
        _max_anisotropy = a;
        return *this;
    }

    auto &set_address_mode(AddressMode mode) noexcept {
        _address[0] = _address[1] = _address[2] = mode;
        return *this;
    }

    auto &set_address_mode(AddressMode u, AddressMode v, AddressMode w = AddressMode::EDGE) noexcept {
        _address = {u, v, w};
        return *this;
    }

    auto &set_filter_mode(FilterMode mode) noexcept {
        _filter = mode;
        return *this;
    }

    auto &set_mip_filter_mode(MipFilterMode mode) noexcept {
        _mip_filter = mode;
        return *this;
    }

    auto &set_normalized(bool b) noexcept {
        _normalized_coords = b;
        return *this;
    }

    [[nodiscard]] auto max_anisotropy() const noexcept { return _max_anisotropy; }
    [[nodiscard]] auto address_mode() const noexcept { return _address; }
    [[nodiscard]] auto filter_mode() const noexcept { return _filter; }
    [[nodiscard]] auto mip_filter_mode() const noexcept { return _mip_filter; }
    [[nodiscard]] auto normalized() const noexcept { return _normalized_coords; }
    [[nodiscard]] uint64_t hash() const noexcept { return xxh3_hash64(this, sizeof(TextureSampler)); }
    [[nodiscard]] auto operator==(TextureSampler rhs) const noexcept { return memcmp(this, &rhs, sizeof(TextureSampler)) == 0; }
};

}// namespace luisa::compute

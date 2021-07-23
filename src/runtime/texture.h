//
// Created by Mike Smith on 2021/7/2.
//

#pragma once

#include <cstdint>

#include <core/hash.h>
#include <core/basic_types.h>
#include <core/mathematics.h>
#include <runtime/pixel.h>
#include <runtime/buffer.h>
#include <runtime/image.h>
#include <runtime/volume.h>

namespace luisa::compute {

class TextureSampler {

public:
    struct Hash {
        [[nodiscard]] auto operator()(TextureSampler s) const noexcept { return s.code(); }
    };

    enum struct Filter : uint32_t {
        POINT,
        BILINEAR,
        TRILINEAR,
        ANISOTROPIC
    };

    enum struct Address : uint32_t {
        EDGE,
        REPEAT,
        MIRROR,
        ZERO
    };

private:
    Filter _filter{Filter::POINT};
    Address _address{Address::EDGE};

public:
    constexpr TextureSampler() noexcept = default;
    constexpr TextureSampler(Filter filter, Address address) noexcept
        : _filter{filter}, _address{address} {}

    [[nodiscard]] uint code() const noexcept { return (to_underlying(_filter) << 2u) | (to_underlying(_address)); }
    [[nodiscard]] static constexpr auto decode(uint code) noexcept {
        return TextureSampler{static_cast<Filter>(code >> 2u),
                              static_cast<Address>(code & 0x03u)};
    }

    [[nodiscard]] static constexpr auto point_edge() noexcept { return TextureSampler{Filter::POINT, Address::EDGE}; }
    [[nodiscard]] static constexpr auto point_repeat() noexcept { return TextureSampler{Filter::POINT, Address::REPEAT}; }
    [[nodiscard]] static constexpr auto point_mirror() noexcept { return TextureSampler{Filter::POINT, Address::MIRROR}; }
    [[nodiscard]] static constexpr auto point_zero() noexcept { return TextureSampler{Filter::POINT, Address::ZERO}; }
    [[nodiscard]] static constexpr auto bilinear_edge() noexcept { return TextureSampler{Filter::BILINEAR, Address::EDGE}; }
    [[nodiscard]] static constexpr auto bilinear_repeat() noexcept { return TextureSampler{Filter::BILINEAR, Address::REPEAT}; }
    [[nodiscard]] static constexpr auto bilinear_mirror() noexcept { return TextureSampler{Filter::BILINEAR, Address::MIRROR}; }
    [[nodiscard]] static constexpr auto bilinear_zero() noexcept { return TextureSampler{Filter::BILINEAR, Address::ZERO}; }
    [[nodiscard]] static constexpr auto trilinear_edge() noexcept { return TextureSampler{Filter::TRILINEAR, Address::EDGE}; }
    [[nodiscard]] static constexpr auto trilinear_repeat() noexcept { return TextureSampler{Filter::TRILINEAR, Address::REPEAT}; }
    [[nodiscard]] static constexpr auto trilinear_mirror() noexcept { return TextureSampler{Filter::TRILINEAR, Address::MIRROR}; }
    [[nodiscard]] static constexpr auto trilinear_zero() noexcept { return TextureSampler{Filter::TRILINEAR, Address::ZERO}; }
    [[nodiscard]] static constexpr auto anisotropic_edge() noexcept { return TextureSampler{Filter::ANISOTROPIC, Address::EDGE}; }
    [[nodiscard]] static constexpr auto anisotropic_repeat() noexcept { return TextureSampler{Filter::ANISOTROPIC, Address::REPEAT}; }
    [[nodiscard]] static constexpr auto anisotropic_mirror() noexcept { return TextureSampler{Filter::ANISOTROPIC, Address::MIRROR}; }
    [[nodiscard]] static constexpr auto anisotropic_zero() noexcept { return TextureSampler{Filter::ANISOTROPIC, Address::ZERO}; }

    [[nodiscard]] auto address() const noexcept { return _address; }
    [[nodiscard]] auto filter() const noexcept { return _filter; }
    [[nodiscard]] auto set_address(Address a) noexcept { _address = a; }
    [[nodiscard]] auto set_filter(Filter f) noexcept { _filter = f; }
    [[nodiscard]] auto operator==(TextureSampler rhs) const noexcept { return code() == rhs.code(); }
};

namespace detail {

template<typename Texture>
[[nodiscard]] inline auto validate_mip_level(Texture t, uint level) noexcept {
    auto valid = level < t.mip_levels();
    if (!valid) {
        LUISA_WARNING_WITH_LOCATION(
            "Invalid mipmap level {} (max = {}) for heap texture #{}.",
            level, t.mip_levels() - 1u, t.handle());
    }
    return valid;
}

}// namespace detail

class Heap;

class Texture2D {

private:
    uint64_t _handle;
    PixelStorage _storage;
    uint _mip_levels;
    uint2 _size;

private:
    friend class Heap;
    Texture2D(uint64_t handle, PixelStorage storage, uint mip_levels, uint2 size) noexcept
        : _handle{handle}, _storage{storage}, _mip_levels{mip_levels}, _size{size} {}

public:
    [[nodiscard]] uint64_t handle() const noexcept { return _handle; }
    [[nodiscard]] PixelStorage storage() const noexcept { return _storage; }
    [[nodiscard]] uint mip_levels() const noexcept { return _mip_levels; }
    [[nodiscard]] uint2 size() const noexcept { return _size; }

    [[nodiscard]] Command *load(const void *pixels, uint mip_level = 0u) noexcept;
    [[nodiscard]] Command *load(ImageView<float> image, uint mip_level = 0u) noexcept;

    template<typename T>
    [[nodiscard]] Command *load(BufferView<T> buffer, uint mip_level = 0u) noexcept {
        if (!detail::validate_mip_level(*this, mip_level)) { return nullptr; }
        auto mipmap_size = max(_size >> mip_level, 1u);
        return BufferToTextureCopyCommand::create(
            buffer.handle(), buffer.offset_bytes(),
            _handle, _storage,
            mip_level, make_uint3(0u), make_uint3(mipmap_size, 1u));
    }
};

class Texture3D {

private:
    uint64_t _handle;
    PixelStorage _storage;
    uint _mip_levels;
    uint3 _size;

private:
    friend class Heap;
    Texture3D(uint64_t handle, PixelStorage storage, uint mip_levels, uint3 size) noexcept
        : _handle{handle},
          _storage{storage},
          _mip_levels{mip_levels},
          _size{size} {}

public:
    [[nodiscard]] uint64_t handle() const noexcept { return _handle; }
    [[nodiscard]] PixelStorage storage() const noexcept { return _storage; }
    [[nodiscard]] uint mip_levels() const noexcept { return _mip_levels; }
    [[nodiscard]] uint3 size() const noexcept { return _size; }

    [[nodiscard]] Command *load(const void *pixels, uint mip_level = 0u) noexcept;
    [[nodiscard]] Command *load(VolumeView<float> image, uint mip_level = 0u) noexcept;

    template<typename T>
    [[nodiscard]] Command *load(BufferView<T> buffer, uint mip_level = 0u) noexcept {
        if (!detail::validate_mip_level(*this, mip_level)) { return nullptr; }
        auto mipmap_size = max(_size >> mip_level, 1u);
        return BufferToTextureCopyCommand::create(
            buffer.handle(), buffer.offset_bytes(),
            _handle, _storage,
            mip_level, make_uint3(0u), mipmap_size);
    }
};

}// namespace luisa::compute

//
// Created by Mike Smith on 2021/4/18.
//

#pragma once

#include <runtime/pixel.h>
#include <runtime/resource.h>
#include <runtime/mipmap.h>

namespace luisa::compute {

template<typename T>
class VolumeView;

template<typename T>
struct Expr;

// Volumes are 3D textures without sampling, i.e., 3D surfaces.
template<typename T>
class Volume : public Resource {

    static_assert(std::disjunction_v<
                  std::is_same<T, int>,
                  std::is_same<T, uint>,
                  std::is_same<T, float>>);

private:
    PixelStorage _storage{};
    uint _mip_levels{};
    uint3 _size{};

private:
    friend class Device;
    Volume(Device::Interface *device, PixelStorage storage, uint3 size, uint mip_levels = 1u) noexcept
        : Resource{
              device, Tag::TEXTURE,
              device->create_texture(
                  pixel_storage_to_format<T>(storage), 3u,
                  size.x, size.y, size.z,
                  detail::max_mip_levels(size, mip_levels),
                  {}, std::numeric_limits<uint64_t>::max(), 0u)},
          _storage{storage}, _mip_levels{detail::max_mip_levels(size, mip_levels)}, _size{size} {}

    Volume(Device::Interface *device, PixelStorage storage, uint width, uint height, uint depth, uint mip_levels = 1u) noexcept
        : Volume{device, storage, make_uint3(width, height, depth), mip_levels} {}

public:
    Volume() noexcept = default;
    using Resource::operator bool;

    [[nodiscard]] auto native_handle() const noexcept { return device()->texture_native_handle(handle()); }
    [[nodiscard]] auto mip_levels() const noexcept { return _mip_levels; }
    [[nodiscard]] auto size() const noexcept { return _size; }
    [[nodiscard]] auto storage() const noexcept { return _storage; }

    [[nodiscard]] auto view() const noexcept { return VolumeView<T>{handle(), _storage, _mip_levels, {}, _size}; }
    [[nodiscard]] auto view(uint3 offset, uint3 size) const noexcept {
        if (any(offset + size >= _size)) [[unlikely]] {
            LUISA_ERROR_WITH_LOCATION(
                "Invalid offset[{}, {}, {}] and size[{}, {}, {}] of view "
                "for volume #{} with size[{}, {}, {}].",
                offset.x, offset.y, offset.z, size.x, size.y, size.z,
                handle(), _size.x, _size.y, _size.z);
        }
        return VolumeView<T>{handle(), _storage, _mip_levels, offset, size};
    }
    [[nodiscard]] auto level(uint l) const noexcept { return view().level(l); }

    template<typename UVW>
    [[nodiscard]] decltype(auto) read(UVW &&uvw) const noexcept {
        return this->view().read(std::forward<UVW>(uvw));
    }

    template<typename UVW, typename Value>
    [[nodiscard]] decltype(auto) write(UVW &&uvw, Value &&value) const noexcept {
        return this->view().write(
            std::forward<UVW>(uvw),
            std::forward<Value>(value));
    }

    template<typename UVW, typename I>
    [[nodiscard]] decltype(auto) read(UVW &&uvw, I &&level) const noexcept {
        return this->view().read(std::forward<UVW>(uvw), std::forward<I>(level));
    }

    template<typename UVW, typename Value, typename I>
    [[nodiscard]] decltype(auto) write(UVW &&uvw, Value &&value, I &&level) const noexcept {
        return this->view().write(
            std::forward<UVW>(uvw),
            std::forward<Value>(value),
            std::forward<I>(level));
    }

    template<typename U>
    [[nodiscard]] auto copy_to(U &&dst) const noexcept { return view().copy_to(std::forward<U>(dst)); }
    template<typename U>
    [[nodiscard]] auto copy_from(U &&dst) const noexcept { return view().copy_from(std::forward<U>(dst)); }
};

template<typename T>
class VolumeView {

private:
    uint64_t _handle;
    PixelStorage _storage;
    uint _mip_levels;
    uint3 _offset;
    uint3 _size;

private:
    friend class Volume<T>;

    constexpr explicit VolumeView(
        uint64_t handle,
        PixelStorage storage,
        uint mip_levels,
        uint3 offset,
        uint3 size) noexcept
        : _handle{handle},
          _storage{storage},
          _mip_levels{mip_levels},
          _offset{offset},
          _size{size} {

        if (any(_offset >= _size)) [[unlikely]] {
            LUISA_ERROR_WITH_LOCATION(
                "Invalid offset[{}, {}, {}] and size[{}, {}, {}] for volume #{}.",
                _offset.x, _offset.y, _offset.z, _size.x, _size.y, _size.z, _handle);
        }
    }

public:
    VolumeView(const Volume<T> &volume) noexcept : VolumeView{volume.view()} {}

    [[nodiscard]] auto handle() const noexcept { return _handle; }
    [[nodiscard]] auto size() const noexcept { return _size; }
    [[nodiscard]] auto storage() const noexcept { return _storage; }
    [[nodiscard]] auto offset() const noexcept { return _offset; }
    [[nodiscard]] auto mip_levels() const noexcept { return _mip_levels; }

    [[nodiscard]] auto level(uint l) const noexcept {
        if (l >= _mip_levels) [[unlikely]] {
            LUISA_ERROR_WITH_LOCATION(
                "Invalid mipmap level {} (max = {}).",
                l, _mip_levels - 1u);
        }
        return detail::MipmapView{
            _handle, max(_size >> l, 1u),
            _offset >> l, l, _storage};
    }

    [[nodiscard]] auto subview(uint3 offset, uint3 size) const noexcept {
        return VolumeView{_handle, _storage, _offset + offset, size};
    }

    template<typename U>
    [[nodiscard]] auto copy_to(U &&dst) const noexcept { return level(0u).copy_to(std::forward<U>(dst)); }
    template<typename U>
    [[nodiscard]] auto copy_from(U &&src) const noexcept { return level(0u).copy_from(std::forward<U>(src)); }

    template<typename UVW>
    [[nodiscard]] decltype(auto) read(UVW &&uvw) const noexcept {
        return Expr<Volume<T>>{*this}.read(std::forward<UVW>(uvw));
    }

    template<typename UVW, typename Value>
    [[nodiscard]] decltype(auto) write(UVW &&uvw, Value &&value) const noexcept {
        return Expr<Volume<T>>{*this}.write(
            std::forward<UVW>(uvw),
            std::forward<Value>(value));
    }

    template<typename UVW, typename I>
    [[nodiscard]] decltype(auto) read(UVW &&uvw, I &&level) const noexcept {
        return Expr<Volume<T>>{*this}.read(std::forward<UVW>(uvw), std::forward<I>(level));
    }

    template<typename UVW, typename Value, typename I>
    [[nodiscard]] decltype(auto) write(UVW &&uvw, Value &&value, I &&level) const noexcept {
        return Expr<Volume<T>>{*this}.write(
            std::forward<UVW>(uvw),
            std::forward<Value>(value),
            std::forward<I>(level));
    }
};

template<typename T>
VolumeView(const Volume<T> &) -> VolumeView<T>;

template<typename T>
VolumeView(VolumeView<T>) -> VolumeView<T>;

template<typename T>
struct is_volume : std::false_type {};

template<typename T>
struct is_volume<Volume<T>> : std::true_type {};

template<typename T>
struct is_volume_view : std::false_type {};

template<typename T>
struct is_volume_view<VolumeView<T>> : std::true_type {};

template<typename T>
using is_volume_or_view = std::disjunction<is_volume<T>, is_volume_view<T>>;

template<typename T>
constexpr auto is_volume_v = is_volume<T>::value;

template<typename T>
constexpr auto is_volume_view_v = is_volume_view<T>::value;

template<typename T>
constexpr auto is_volume_or_view_v = is_volume_or_view<T>::value;

}// namespace luisa::compute

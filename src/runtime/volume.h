//
// Created by Mike Smith on 2021/4/18.
//

#pragma once

#include <runtime/pixel.h>
#include <runtime/resource.h>
#include <runtime/mipmap.h>
#include <runtime/sampler.h>
#include <runtime/device.h>

namespace luisa::compute {

namespace detail {
LC_RUNTIME_API void error_volume_invalid_mip_levels(size_t level, size_t mip) noexcept;
}

template<typename T>
class VolumeView;

template<typename T>
struct Expr;

// Volumes are 3D textures without sampling, i.e., 3D surfaces.
template<typename T>
class Volume final : public Resource {

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
    Volume(DeviceInterface *device, PixelStorage storage, uint3 size, uint mip_levels = 1u, Sampler sampler = {}) noexcept
        : Resource{
              device, Tag::TEXTURE,
              device->create_texture(
                  pixel_storage_to_format<T>(storage), 3u,
                  size.x, size.y, size.z,
                  detail::max_mip_levels(size, mip_levels))},
          _storage{storage}, _mip_levels{detail::max_mip_levels(size, mip_levels)}, _size{size} {}

public:
    Volume() noexcept = default;
    using Resource::operator bool;
    [[nodiscard]] auto mip_levels() const noexcept { return _mip_levels; }
    [[nodiscard]] auto size() const noexcept { return _size; }
    [[nodiscard]] auto byte_size() const noexcept {
        size_t byte_size = 0;
        auto size = _size;
        for (size_t i = 0; i < _mip_levels; ++i) {
            byte_size += pixel_storage_size(_storage, size.x, size.y, size.z);
            size >>= uint3(1);
        }
        return byte_size;
    }
    [[nodiscard]] auto storage() const noexcept { return _storage; }
    [[nodiscard]] auto format() const noexcept { return pixel_storage_to_format<T>(_storage); }

    [[nodiscard]] auto view(uint32_t level) const noexcept {
        if (level >= _mip_levels) [[unlikely]] {
            detail::error_volume_invalid_mip_levels(level, _mip_levels);
        }
        auto mip_size = luisa::max(_size >> level, 1u);
        return VolumeView<T>{handle(), _storage, level, mip_size};
    }

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

    template<typename U>
    [[nodiscard]] auto copy_to(U &&dst) const noexcept {
        return this->view(0).copy_to(std::forward<U>(dst));
    }
    template<typename U>
    [[nodiscard]] auto copy_from(U &&dst) const noexcept {
        return this->view(0).copy_from(std::forward<U>(dst));
    }
};

template<typename T>
class VolumeView {

private:
    uint64_t _handle;
    PixelStorage _storage;
    uint _level;
    uint3 _size;

private:
    friend class Volume<T>;
    friend class detail::MipmapView;

    constexpr explicit VolumeView(
        uint64_t handle, PixelStorage storage, uint level, uint3 size) noexcept
        : _handle{handle}, _storage{storage}, _level{level}, _size{size} {}

    [[nodiscard]] auto _as_mipmap() const noexcept {
        return detail::MipmapView{
            _handle, _size, _level, _storage};
    }

public:
    VolumeView(const Volume<T> &volume) noexcept : VolumeView{volume.view(0u)} {}

    [[nodiscard]] auto handle() const noexcept { return _handle; }
    [[nodiscard]] auto size() const noexcept { return _size; }
    [[nodiscard]] auto byte_size() const noexcept {
        return pixel_storage_size(_storage, _size.x, _size.y, _size.z);
    }
    [[nodiscard]] auto storage() const noexcept { return _storage; }
    [[nodiscard]] auto level() const noexcept { return _level; }
    [[nodiscard]] auto format() const noexcept { return pixel_storage_to_format<T>(_storage); }

    template<typename U>
    [[nodiscard]] auto copy_to(U &&dst) const noexcept { return _as_mipmap().copy_to(std::forward<U>(dst)); }
    template<typename U>
    [[nodiscard]] auto copy_from(U &&src) const noexcept { return _as_mipmap().copy_from(std::forward<U>(src)); }

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

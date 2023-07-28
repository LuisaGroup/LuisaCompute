#pragma once

#include <luisa/core/mathematics.h>
#include <luisa/runtime/rhi/pixel.h>
#include <luisa/runtime/rhi/resource.h>
#include <luisa/runtime/mipmap.h>
#include <luisa/runtime/rhi/sampler.h>
#include <luisa/runtime/rhi/device_interface.h>

namespace luisa::compute {

namespace detail {

template<typename VolumeOrView>
class VolumeExprProxy;

LC_RUNTIME_API void error_volume_invalid_mip_levels(size_t level, size_t mip) noexcept;
LC_RUNTIME_API void volume_size_zero_error() noexcept;
}// namespace detail

template<typename T>
class VolumeView;

template<typename T>
class SparseVolume;

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
    friend class ResourceGenerator;
    Volume(DeviceInterface *device,
           const ResourceCreationInfo &create_info,
           PixelStorage storage,
           uint3 size, uint mip_levels) noexcept
        : Resource{device, Tag::TEXTURE, create_info},
          _storage{storage},
          _mip_levels{detail::max_mip_levels(size, mip_levels)},
          _size{size} {}

    Volume(DeviceInterface *device, PixelStorage storage, uint3 size,
           uint mip_levels = 1u, bool simultaneous_access = false) noexcept
        : Volume{device,
                 [&] {
                     if (size.x == 0 || size.y == 0 || size.z == 0) [[unlikely]] {
                         detail::volume_size_zero_error();
                     }
                     return device->create_texture(
                         pixel_storage_to_format<T>(storage), 3u,
                         size.x, size.y, size.z,
                         detail::max_mip_levels(size, mip_levels),
                         simultaneous_access);
                 }(),
                 storage, size, mip_levels} {}

public:
    Volume() noexcept = default;
    ~Volume() noexcept override {
        if (*this) { device()->destroy_texture(handle()); }
    }
    using Resource::operator bool;
    Volume(Volume &&) noexcept = default;
    Volume(Volume const &) noexcept = delete;
    Volume &operator=(Volume &&rhs) noexcept {
        _move_from(std::move(rhs));
        return *this;
    }
    Volume &operator=(Volume const &) noexcept = delete;
    // properties
    [[nodiscard]] auto mip_levels() const noexcept {
        _check_is_valid();
        return _mip_levels;
    }
    [[nodiscard]] auto size() const noexcept {
        _check_is_valid();
        return _size;
    }
    [[nodiscard]] auto storage() const noexcept {
        _check_is_valid();
        return _storage;
    }
    [[nodiscard]] auto format() const noexcept {
        _check_is_valid();
        return pixel_storage_to_format<T>(_storage);
    }
    [[nodiscard]] auto view(uint32_t level) const noexcept {
        _check_is_valid();
        if (level >= _mip_levels) [[unlikely]] {
            detail::error_volume_invalid_mip_levels(level, _mip_levels);
        }
        auto mip_size = luisa::max(_size >> level, 1u);
        return VolumeView<T>{handle(), _storage, level, mip_size};
    }
    [[nodiscard]] auto view() const noexcept { return view(0u); }

    // commands
    // copy image's data to pointer or another image
    template<typename U>
    [[nodiscard]] auto copy_to(U &&dst) const noexcept {
        _check_is_valid();
        return this->view(0).copy_to(std::forward<U>(dst));
    }
    // copy pointer or another image's data to image
    template<typename U>
    [[nodiscard]] auto copy_from(U &&dst) const noexcept {
        _check_is_valid();
        return this->view(0).copy_from(std::forward<U>(dst));
    }
    // DSL interface
    [[nodiscard]] auto operator->() const noexcept {
        _check_is_valid();
        return reinterpret_cast<const detail::VolumeExprProxy<Volume<T>> *>(this);
    }
};

// A VolumeView is a reference to a Volume without ownership
// (i.e., it will not destroy the underlying Volume on destruction).
template<typename T>
class VolumeView {

private:
    uint64_t _handle;
    PixelStorage _storage;
    uint _level;
    uint3 _size;

private:
    friend class Volume<T>;
    friend class SparseVolume<T>;
    friend class detail::MipmapView;

    [[nodiscard]] auto _as_mipmap() const noexcept {
        return detail::MipmapView{
            _handle, _size, _level, _storage};
    }

public:
    VolumeView(uint64_t handle, PixelStorage storage,
               uint level, uint3 size) noexcept
        : _handle{handle}, _storage{storage},
          _level{level}, _size{size} {}

    VolumeView(const Volume<T> &volume) noexcept : VolumeView{volume.view(0u)} {}

    // properties
    [[nodiscard]] auto handle() const noexcept { return _handle; }
    [[nodiscard]] auto size() const noexcept { return _size; }
    [[nodiscard]] auto size_bytes() const noexcept {
        return pixel_storage_size(_storage, _size);
    }
    [[nodiscard]] auto storage() const noexcept { return _storage; }
    [[nodiscard]] auto level() const noexcept { return _level; }
    [[nodiscard]] auto format() const noexcept { return pixel_storage_to_format<T>(_storage); }
    // commands
    // copy image's data to pointer or another image
    template<typename U>
    [[nodiscard]] auto copy_to(U &&dst) const noexcept { return _as_mipmap().copy_to(std::forward<U>(dst)); }
    // copy pointer or another image's data to image
    template<typename U>
    [[nodiscard]] auto copy_from(U &&src) const noexcept { return _as_mipmap().copy_from(std::forward<U>(src)); }
    // DSL interface
    [[nodiscard]] auto operator->() const noexcept {
        return reinterpret_cast<const detail::VolumeExprProxy<VolumeView<T>> *>(this);
    }
};

template<typename T>
VolumeView(const Volume<T> &) -> VolumeView<T>;

template<typename T>
VolumeView(VolumeView<T>) -> VolumeView<T>;

namespace detail {

template<typename T>
struct is_volume_impl : std::false_type {};

template<typename T>
struct is_volume_impl<Volume<T>> : std::true_type {};

template<typename T>
struct is_volume_view_impl : std::false_type {};

template<typename T>
struct is_volume_view_impl<VolumeView<T>> : std::true_type {};

}// namespace detail

template<typename T>
using is_volume = detail::is_volume_impl<std::remove_cvref_t<T>>;

template<typename T>
using is_volume_view = detail::is_volume_view_impl<std::remove_cvref_t<T>>;

template<typename T>
using is_volume_or_view = std::disjunction<is_volume<T>, is_volume_view<T>>;

template<typename T>
constexpr auto is_volume_v = is_volume<T>::value;

template<typename T>
constexpr auto is_volume_view_v = is_volume_view<T>::value;

template<typename T>
constexpr auto is_volume_or_view_v = is_volume_or_view<T>::value;

namespace detail {

template<typename VolumeOrView>
struct volume_element_impl {
    static_assert(always_false_v<VolumeOrView>);
};

template<typename T>
struct volume_element_impl<Volume<T>> {
    using type = T;
};

template<typename T>
struct volume_element_impl<VolumeView<T>> {
    using type = T;
};

}// namespace detail

template<typename VolumeOrView>
using volume_element = detail::volume_element_impl<std::remove_cvref_t<VolumeOrView>>;

template<typename VolumeOrView>
using volume_element_t = typename volume_element<VolumeOrView>::type;

}// namespace luisa::compute

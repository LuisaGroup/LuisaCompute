//
// Created by Mike Smith on 2021/4/18.
//

#pragma once

#include <runtime/pixel.h>
#include <runtime/device.h>

namespace luisa::compute {

template<typename T>
class VolumeView;

namespace detail {

template<typename T>
struct Expr;

}// namespace detail

// Volumes are 3D textures without sampling.
template<typename T>
class Volume : concepts::Noncopyable {

    static_assert(std::disjunction_v<
                  std::is_same<T, int>,
                  std::is_same<T, uint>,
                  std::is_same<T, float>>);

private:
    Device::Interface *_device;
    uint64_t _handle;
    PixelStorage _storage;
    uint3 _size;

private:
    friend class Device;
    Volume(Device &device, PixelStorage storage, uint width, uint height, uint depth) noexcept
        : _device{device.impl()},
          _handle{device.impl()->create_texture(
              pixel_storage_to_format<T>(storage), 3u,
              width, height, depth,
              1u, false)},
          _storage{storage},
          _size{width, height, depth} {}
    Volume(Device &device, PixelStorage storage, uint3 size) noexcept
        : Volume{device, storage, size.x, size.y, size.z} {}

public:
    Volume(Volume &&another) noexcept
        : _device{another._device},
          _handle{another._handle},
          _size{another._size},
          _storage{another._storage} { another._device = nullptr; }

    ~Volume() noexcept {
        if (_device != nullptr) {
            _device->dispose_texture(_handle);
        }
    }

    Volume &operator=(Volume &&rhs) noexcept {
        if (&rhs != this) {
            _device->dispose_texture(_handle);
            _device = rhs._device;
            _handle = rhs._handle;
            _size = rhs._size;
            _storage = rhs._storage;
            rhs._device = nullptr;
        }
        return *this;
    }

    [[nodiscard]] auto size() const noexcept { return _size; }
    [[nodiscard]] auto storage() const noexcept { return _storage; }

    [[nodiscard]] auto view() const noexcept { return VolumeView<T>{_handle, _storage, {}, _size}; }
    [[nodiscard]] auto view(uint3 offset, uint3 size) const noexcept {
        if (any(offset + size >= _size)) {
            LUISA_ERROR_WITH_LOCATION(
                "Invalid offset[{}, {}, {}] and size[{}, {}, {}] of view "
                "for volume #{} with size[{}, {}, {}].",
                offset.x, offset.y, offset.z, size.x, size.y, size.z,
                _handle, _size.x, _size.y, _size.z);
        }
        return VolumeView<T>{_handle, _storage, offset, size};
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

    [[nodiscard]] CommandHandle copy_to(void *data) const noexcept { return view().copy_to(data); }
    [[nodiscard]] CommandHandle copy_from(const void *data) const noexcept { return view().copy_from(data); }
};

template<typename T>
class VolumeView {

private:
    uint64_t _handle;
    PixelStorage _storage;
    uint3 _offset;
    uint3 _size;

private:
    friend class Volume<T>;

    constexpr explicit VolumeView(
        uint64_t handle,
        PixelStorage storage,
        uint3 offset,
        uint3 size) noexcept
        : _handle{handle},
          _storage{storage},
          _offset{offset},
          _size{size} {
        
        if (any(_offset >= _size)) {
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
    
    [[nodiscard]] auto subview(uint3 offset, uint3 size) const noexcept {
        return VolumeView{_handle, _storage, _offset + offset, size};
    }

    [[nodiscard]] auto copy_from(const void *data) const noexcept {
        return TextureUploadCommand::create(
            _handle, _storage,
            0u, _offset,
            _size, data);
    }

    [[nodiscard]] auto copy_to(void *data) const noexcept {
        return TextureDownloadCommand::create(
            _handle, _storage,
            0u, _offset,
            _size, data);
    }

    template<typename UVW>
    [[nodiscard]] decltype(auto) read(UVW &&uvw) const noexcept {
        return detail::Expr<Volume<T>>{*this}.read(std::forward<UVW>(uvw));
    }

    template<typename UVW, typename Value>
    [[nodiscard]] decltype(auto) write(UVW &&uvw, Value &&value) const noexcept {
        return detail::Expr<Volume<T>>{*this}.write(
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
constexpr auto is_volume_or_view_v = is_volume_or_view<T>::view;

}// namespace luisa::compute

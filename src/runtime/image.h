//
// Created by Mike Smith on 2021/3/29.
//

#pragma once

#include <runtime/command.h>
#include <runtime/device.h>

namespace luisa::compute {

template<typename T>
class ImageView;

namespace detail {

template<typename T>
struct Expr;

[[nodiscard]] auto valid_mipmap_levels(uint width, uint height, uint requested_levels) noexcept {
    auto rounded_size = next_pow2(std::min(width, height));
    auto max_levels = static_cast<uint>(std::log2(rounded_size));
    return requested_levels == 0u
               ? max_levels
               : std::min(requested_levels, max_levels);
}

}// namespace detail

// Images are textures without sampling.
template<typename T>
class Image : concepts::Noncopyable {

    static_assert(std::disjunction_v<
                  std::is_same<T, int>,
                  std::is_same<T, uint>,
                  std::is_same<T, float>>);

private:
    Device::Interface *_device;
    uint64_t _handle;
    uint2 _size;
    PixelStorage _storage;

private:
    friend class Device;
    Image(Device &device, PixelStorage storage, uint2 size) noexcept
        : _device{device.interface()},
          _handle{device.interface()->create_texture(
              pixel_storage_to_format<T>(storage), 2u,
              size.x, size.y, 1u,
              1u, false)},
          _size{size},
          _storage{storage} {}

    Image(Device &device, PixelStorage storage, uint width, uint height) noexcept
        : Image{device, storage, uint2{width, height}} {}

public:
    Image(Image &&another) noexcept
        : _device{another._device},
          _handle{another._handle},
          _size{another._size},
          _storage{another._storage} { another._device = nullptr; }

    ~Image() noexcept {
        if (_device != nullptr) {
            _device->dispose_texture(_handle);
        }
    }

    Image &operator=(Image &&rhs) noexcept {
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

    [[nodiscard]] auto view() const noexcept { return ImageView<T>{_handle, _storage, _size}; }

    template<typename UV>
    [[nodiscard]] decltype(auto) read(UV &&uv) const noexcept {
        return this->view().read(std::forward<UV>(uv));
    }

    template<typename UV, typename Value>
    [[nodiscard]] decltype(auto) write(UV &&uv, Value &&value) const noexcept {
        return this->view().write(
            std::forward<UV>(uv),
            std::forward<Value>(value));
    }

    [[nodiscard]] CommandHandle copy_to(void *data) const noexcept { return view().copy_to(data); }
    [[nodiscard]] CommandHandle copy_from(const void *data) const noexcept { return view().copy_from(data); }
};

template<typename T>
class ImageView {

private:
    uint64_t _handle;
    uint2 _size;
    PixelStorage _storage;

private:
    friend class Image<T>;

    constexpr explicit ImageView(
        uint64_t handle,
        PixelStorage storage,
        uint2 size) noexcept
        : _handle{handle},
          _size{size},
          _storage{storage} {}

public:
    ImageView(const Image<T> &image) noexcept : ImageView{image.view()} {}

    [[nodiscard]] auto handle() const noexcept { return _handle; }
    [[nodiscard]] auto size() const noexcept { return _size; }
    [[nodiscard]] auto storage() const noexcept { return _storage; }

    [[nodiscard]] auto copy_from(const void *data) const noexcept {
        return TextureUploadCommand::create(
            _handle, _storage,
            0u, uint3{},
            uint3{_size, 1u}, data);
    }

    [[nodiscard]] auto copy_to(void *data) const noexcept {
        return TextureDownloadCommand::create(
            _handle, _storage,
            0u, uint3{},
            uint3{_size, 1u}, data);
    }

    template<typename UV>
    [[nodiscard]] decltype(auto) read(UV &&uv) const noexcept {
        return detail::Expr<Image<T>>{*this}.read(std::forward<UV>(uv));
    }

    template<typename UV, typename Value>
    [[nodiscard]] decltype(auto) write(UV &&uv, Value &&value) const noexcept {
        return detail::Expr<Image<T>>{*this}.write(
            std::forward<UV>(uv),
            std::forward<Value>(value));
    }
};

template<typename T>
ImageView(const Image<T> &) -> ImageView<T>;

template<typename T>
ImageView(ImageView<T>) -> ImageView<T>;

}// namespace luisa::compute

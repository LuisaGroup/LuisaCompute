//
// Created by Mike Smith on 2021/3/29.
//

#pragma once

#include <runtime/command.h>
#include <runtime/device.h>

namespace luisa::compute {

template<typename T>
class ImageView;

template<typename T>
class BufferView;

namespace detail {

template<typename T>
struct Expr;

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
        : _device{device.impl()},
          _handle{device.impl()->create_texture(
              pixel_storage_to_format<T>(storage), 2u,
              size.x, size.y, 1u,
              1u, std::numeric_limits<uint64_t>::max())},
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

    [[nodiscard]] auto view() const noexcept { return ImageView<T>{_handle, _storage, {}, _size}; }
    [[nodiscard]] auto view(uint2 offset, uint2 size) const noexcept {
        if (any(offset + size >= _size)) [[unlikely]] {
            LUISA_ERROR_WITH_LOCATION(
                "Invalid offset[{}, {}] and size[{}, {}] of view "
                "for image #{} with size[{}, {}].",
                offset.x, offset.y, size.x, size.y,
                _handle, _size.x, _size.y);
        }
        return ImageView<T>{_handle, _storage, offset, size};
    }

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
    [[nodiscard]] CommandHandle copy_from(ImageView<T> src) const noexcept { return view().copy_from(src); }

    template<typename U>
    [[nodiscard]] CommandHandle copy_from(BufferView<U> src) const noexcept { return view().copy_from(src); }

    template<typename U>
    [[nodiscard]] CommandHandle copy_to(BufferView<U> src) const noexcept { return view().copy_to(src); }
};

template<typename T>
class ImageView {

private:
    uint64_t _handle;
    uint2 _size;
    uint2 _offset;
    PixelStorage _storage;

private:
    friend class Image<T>;

    constexpr explicit ImageView(
        uint64_t handle,
        PixelStorage storage,
        uint2 offset,
        uint2 size) noexcept
        : _handle{handle},
          _size{size},
          _offset{offset},
          _storage{storage} {

        if (any(_offset >= _size)) [[unlikely]] {
            LUISA_ERROR_WITH_LOCATION(
                "Invalid offset[{}, {}] and size[{}, {}] for image #{}.",
                _offset.x, _offset.y, _size.x, _size.y, _handle);
        }
    }

public:
    ImageView(const Image<T> &image) noexcept : ImageView{image.view()} {}

    [[nodiscard]] auto handle() const noexcept { return _handle; }
    [[nodiscard]] auto size() const noexcept { return _size; }
    [[nodiscard]] auto offset() const noexcept { return _offset; }
    [[nodiscard]] auto storage() const noexcept { return _storage; }

    [[nodiscard]] auto subview(uint2 offset, uint2 size) const noexcept {
        return ImageView{_handle, _storage, _offset + offset, size};
    }

    [[nodiscard]] auto copy_from(const void *data) const noexcept {
        return TextureUploadCommand::create(
            _handle, _storage,
            0u, uint3{_offset, 0u},
            uint3{_size, 1u}, data);
    }

    [[nodiscard]] auto copy_from(ImageView src) const noexcept {
        auto size = _size;
        if (!all(size == src._size)) {
            LUISA_WARNING_WITH_LOCATION(
                "ImageView sizes mismatch in copy command (src: [{}, {}], dest: [{}, {}]).",
                src._size.x, src._size.y, size.x, size.y);
            size = min(size, src._size);
        }
        return TextureCopyCommand::create(
            src._handle, _handle, 0u, 0u,
            uint3(src._offset, 0u), uint3(_offset, 0u),
            uint3(size, 1u));
    }

    template<typename U>
    [[nodiscard]] auto copy_from(BufferView<U> buffer) const noexcept {
        return BufferToTextureCopyCommand::create(
            buffer.handle(), buffer.offset_bytes(),
            _handle, _storage, 0u, uint3(_offset, 0u), uint3(_size, 1u));
    }

    template<typename U>
    [[nodiscard]] auto copy_to(BufferView<U> buffer) const noexcept {
        return TextureToBufferCopyCommand::create(
            buffer.handle(), buffer.offset_bytes(),
            _handle, _storage, 0u, uint3(_offset, 0u), uint3(_size, 1u));
    }

    [[nodiscard]] auto copy_to(void *data) const noexcept {
        return TextureDownloadCommand::create(
            _handle, _storage,
            0u, uint3{_offset, 0u},
            uint3{_size, 1u}, data);
    }

    template<typename UV>
    [[nodiscard]] decltype(auto) read(UV &&uv) const noexcept {
        return detail::Expr<Image<T>>{*this}.read(std::forward<UV>(uv));
    }

    template<typename UV, typename Value>
    void write(UV uv, Value &&value) const noexcept {
        detail::Expr<Image<T>>{*this}.write(
            std::forward<UV>(uv),
            std::forward<Value>(value));
    }
};

template<typename T>
ImageView(const Image<T> &) -> ImageView<T>;

template<typename T>
ImageView(ImageView<T>) -> ImageView<T>;

template<typename T>
struct is_image : std::false_type {};

template<typename T>
struct is_image<Image<T>> : std::true_type {};

template<typename T>
struct is_image_view : std::false_type {};

template<typename T>
struct is_image_view<ImageView<T>> : std::true_type {};

template<typename T>
using is_image_or_view = std::disjunction<is_image<T>, is_image_view<T>>;

template<typename T>
constexpr auto is_image_v = is_image<T>::value;

template<typename T>
constexpr auto is_image_view_v = is_image_view<T>::value;

template<typename T>
constexpr auto is_image_or_view_v = is_image_or_view<T>::view;

}// namespace luisa::compute

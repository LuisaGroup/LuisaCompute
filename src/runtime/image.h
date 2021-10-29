//
// Created by Mike Smith on 2021/3/29.
//

#pragma once

#include <core/mathematics.h>
#include <runtime/resource.h>
#include <runtime/mipmap.h>

namespace luisa::compute {

template<typename T>
class ImageView;

template<typename T>
class BufferView;

template<typename T>
struct Expr;

class Heap;

// Images are textures without sampling, i.e., surfaces.
template<typename T>
class Image : public Resource {

    static_assert(std::disjunction_v<
                  std::is_same<T, int>,
                  std::is_same<T, uint>,
                  std::is_same<T, float>>);

private:
    uint2 _size{};
    uint32_t _mip_levels{};
    PixelStorage _storage{};

private:
    friend class Device;
    Image(Device::Interface *device, PixelStorage storage, uint2 size, uint mip_levels = 1u) noexcept
        : Resource{
              device,
              Tag::TEXTURE,
              device->create_texture(
                  pixel_storage_to_format<T>(storage), 2u,
                  size.x, size.y, 1u,
                  detail::max_mip_levels(make_uint3(size, 1u), mip_levels), {},
                  std::numeric_limits<uint64_t>::max(), 0u)},
          _size{size}, _mip_levels{detail::max_mip_levels(make_uint3(size, 1u), mip_levels)}, _storage{storage} {}

public:
    Image() noexcept = default;
    using Resource::operator bool;

    [[nodiscard]] auto native_handle() const noexcept { return device()->texture_native_handle(handle()); }

    [[nodiscard]] auto size() const noexcept { return _size; }
    [[nodiscard]] auto mip_levels() const noexcept { return _mip_levels; }
    [[nodiscard]] auto storage() const noexcept { return _storage; }

    [[nodiscard]] auto view() const noexcept { return ImageView<T>{handle(), _storage, _mip_levels, {}, _size}; }
    [[nodiscard]] auto view(uint2 offset, uint2 size) const noexcept {
        if (any(offset + size >= _size)) [[unlikely]] {
            LUISA_ERROR_WITH_LOCATION(
                "Invalid offset[{}, {}] and size[{}, {}] of view "
                "for image #{} with size[{}, {}].",
                offset.x, offset.y, size.x, size.y,
                handle(), _size.x, _size.y);
        }
        return ImageView<T>{handle(), _storage, _mip_levels, offset, size};
    }
    [[nodiscard]] auto level(uint l) const noexcept { return view().level(l); }

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

    template<typename UV, typename I>
    [[nodiscard]] decltype(auto) read(UV &&uv, I &&level) const noexcept {
        return this->view().read(std::forward<UV>(uv), std::forward<I>(level));
    }

    template<typename UV, typename Value, typename I>
    [[nodiscard]] decltype(auto) write(UV &&uv, Value &&value, I &&level) const noexcept {
        return this->view().write(
            std::forward<UV>(uv),
            std::forward<Value>(value),
            std::forward<I>(level));
    }

    template<typename U>
    [[nodiscard]] auto copy_to(U &&dst) const noexcept { return view().copy_to(std::forward<U>(dst)); }
    template<typename U>
    [[nodiscard]] auto copy_from(U &&dst) const noexcept { return view().copy_from(std::forward<U>(dst)); }
};

template<typename T>
class ImageView {

private:
    uint64_t _handle;
    uint2 _size;
    uint2 _offset;
    uint _mip_levels;
    PixelStorage _storage;

private:
    friend class Image<T>;
    friend class Heap;

    constexpr ImageView(
        uint64_t handle,
        PixelStorage storage,
        uint mip_levels,
        uint2 offset,
        uint2 size) noexcept
        : _handle{handle},
          _size{size},
          _offset{offset},
          _mip_levels{mip_levels},
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
    [[nodiscard]] auto mip_levels() const noexcept { return _mip_levels; }
    [[nodiscard]] auto level(uint32_t l) const noexcept {
        if (l >= _mip_levels) [[unlikely]] {
            LUISA_ERROR_WITH_LOCATION(
                "Invalid mipmap level {} (max = {}).",
                l, _mip_levels - 1u);
        }
        return detail::MipmapView{
            _handle,
            max(make_uint3(_size, 1u) >> l, 1u),
            make_uint3(_offset, 0u) >> l,
            l, _storage};
    }

    [[nodiscard]] auto subview(uint2 offset, uint2 size) const noexcept {
        return ImageView{_handle, _storage, _offset + offset, size};
    }

    template<typename U>
    [[nodiscard]] auto copy_to(U &&dst) const noexcept { return level(0u).copy_to(std::forward<U>(dst)); }
    template<typename U>
    [[nodiscard]] auto copy_from(U &&src) const noexcept { return level(0u).copy_from(std::forward<U>(src)); }

    template<typename UV>
    [[nodiscard]] decltype(auto) read(UV &&uv) const noexcept {
        return Expr<Image<T>>{*this}.read(std::forward<UV>(uv));
    }

    template<typename UV, typename Value>
    void write(UV uv, Value &&value) const noexcept {
        Expr<Image<T>>{*this}.write(
            std::forward<UV>(uv),
            std::forward<Value>(value));
    }
    template<typename UV, typename I>
    [[nodiscard]] decltype(auto) read(UV &&uv, I &&l) const noexcept {
        return Expr<Image<T>>{*this}.read(std::forward<UV>(uv), std::forward<I>(l));
    }

    template<typename UV, typename Value, typename I>
    void write(UV uv, Value &&value, I &&l) const noexcept {
        Expr<Image<T>>{*this}.write(
            std::forward<UV>(uv),
            std::forward<Value>(value),
            std::forward<I>(l));
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
constexpr auto is_image_or_view_v = is_image_or_view<T>::value;

}// namespace luisa::compute

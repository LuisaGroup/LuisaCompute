//
// Created by Mike Smith on 2021/3/29.
//

#pragma once

#include <core/mathematics.h>
#include <runtime/resource.h>
#include <runtime/mipmap.h>
#include <runtime/sampler.h>
#include <runtime/device.h>

namespace luisa::compute {

namespace detail {
LC_RUNTIME_API void error_image_invalid_mip_levels(size_t level, size_t mip) noexcept;
}

template<typename T>
class ImageView;

template<typename T>
class BufferView;

template<typename T>
struct ImageExprProxy;

class BindlessArray;

// Images are textures without sampling, i.e., surfaces.

template<typename T>
constexpr bool is_legal_image_element = std::disjunction_v<
    std::is_same<T, int32_t>,
    std::is_same<T, uint>,
    std::is_same<T, float>>;
template<typename T>
class Image final : public Resource {
    static_assert(is_legal_image_element<T>);

private:
    uint2 _size{};
    uint32_t _mip_levels{};
    PixelStorage _storage{};

private:
    friend class Device;
    Image(DeviceInterface *device, PixelStorage storage, uint2 size, uint mip_levels = 1u) noexcept
        : Resource{
              device,
              Tag::TEXTURE,
              device->create_texture(
                  pixel_storage_to_format<T>(storage), 2u,
                  size.x, size.y, 1u,
                  detail::max_mip_levels(make_uint3(size, 1u), mip_levels))},
          _size{size}, _mip_levels{detail::max_mip_levels(make_uint3(size, 1u), mip_levels)}, _storage{storage} {}

public:
    Image() noexcept = default;
    using Resource::operator bool;
    [[nodiscard]] auto size() const noexcept { return _size; }
    [[nodiscard]] auto size_bytes() const noexcept {
        size_t byte_size = 0;
        auto size = _size;
        for (size_t i = 0; i < _mip_levels; ++i) {
            byte_size += pixel_storage_size(_storage, size.x, size.y, 1);
            size >>= uint2(1);
        }
        return byte_size;
    }
    [[nodiscard]] auto mip_levels() const noexcept { return _mip_levels; }
    [[nodiscard]] auto storage() const noexcept { return _storage; }
    [[nodiscard]] auto format() const noexcept { return pixel_storage_to_format<T>(_storage); }

    [[nodiscard]] auto view(uint32_t level) const noexcept {
        if (level >= _mip_levels) [[unlikely]] {
            detail::error_image_invalid_mip_levels(level, _mip_levels);
        }
        auto mip_size = luisa::max(_size >> level, 1u);
        return ImageView<T>{handle(), _storage, level, mip_size};
    }

    [[nodiscard]] auto view() const noexcept { return view(0u); }

    template<typename U>
    [[nodiscard]] auto copy_to(U &&dst) const noexcept {
        return this->view(0).copy_to(std::forward<U>(dst));
    }
    template<typename U>
    [[nodiscard]] auto copy_from(U &&dst) const noexcept {
        return this->view(0).copy_from(std::forward<U>(dst));
    }
    [[nodiscard]] auto operator->() const noexcept{
        return reinterpret_cast<ImageExprProxy<Image<T>> const*>(*this);
    }
};
class ViewExporter;
template<typename T>
class ImageView {

private:
    uint64_t _handle;
    uint2 _size;
    uint _level;
    PixelStorage _storage;

private:
    friend class Image<T>;
    friend class detail::MipmapView;
    friend class DepthBuffer;
    friend class ViewExporter;
    constexpr ImageView(
        uint64_t handle,
        PixelStorage storage,
        uint level,
        uint2 size) noexcept
        : _handle{handle},
          _size{size},
          _level{level},
          _storage{storage} {}

    [[nodiscard]] auto _as_mipmap() const noexcept {
        return detail::MipmapView{_handle, make_uint3(_size, 1u), _level, _storage};
    }

public:
    ImageView(const Image<T> &image) noexcept : ImageView{image.view(0u)} {}
    [[nodiscard]] auto handle() const noexcept { return _handle; }
    [[nodiscard]] auto size() const noexcept { return _size; }
    [[nodiscard]] auto size_bytes() const noexcept {
        return pixel_storage_size(_storage, _size.x, _size.y, 1);
    }
    [[nodiscard]] auto storage() const noexcept { return _storage; }
    [[nodiscard]] auto format() const noexcept { return pixel_storage_to_format<T>(_storage); }
    [[nodiscard]] auto level() const noexcept { return _level; }
    template<typename U>
    [[nodiscard]] auto copy_to(U &&dst) const noexcept { return _as_mipmap().copy_to(std::forward<U>(dst)); }
    template<typename U>
    [[nodiscard]] auto copy_from(U &&src) const noexcept { return _as_mipmap().copy_from(std::forward<U>(src)); }

    [[nodiscard]] auto operator->() const noexcept{
        return reinterpret_cast<ImageExprProxy<ImageView<T>> const*>(*this);
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

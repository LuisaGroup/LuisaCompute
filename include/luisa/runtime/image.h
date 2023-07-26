#pragma once

#include <luisa/core/mathematics.h>
#include <luisa/runtime/rhi/resource.h>
#include <luisa/runtime/mipmap.h>
#include <luisa/runtime/rhi/sampler.h>
#include <luisa/runtime/rhi/device_interface.h>

namespace luisa::compute {

namespace detail {

template<typename ImageOrView>
class ImageExprProxy;

LC_RUNTIME_API void error_image_invalid_mip_levels(size_t level, size_t mip) noexcept;
LC_RUNTIME_API void image_size_zero_error() noexcept;

}// namespace detail

template<typename T>
class ImageView;

template<typename T>
class SparseImage;

template<typename T>
class BufferView;

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
    friend class ResourceGenerator;
    Image(DeviceInterface *device,
          const ResourceCreationInfo &create_info,
          PixelStorage storage,
          uint2 size, uint mip_levels) noexcept
        : Resource{device, Tag::TEXTURE, create_info},
          _size{size},
          _mip_levels{detail::max_mip_levels(make_uint3(size, 1u), mip_levels)},
          _storage{storage} {}

    Image(DeviceInterface *device, PixelStorage storage, uint2 size,
          uint mip_levels = 1u, bool simultaneous_access = false) noexcept
        : Image{device,
                [&] {
                    if (size.x == 0 || size.y == 0) [[unlikely]] {
                        detail::image_size_zero_error();
                    }
                    return device->create_texture(
                        pixel_storage_to_format<T>(storage), 2u,
                        size.x, size.y, 1u,
                        detail::max_mip_levels(make_uint3(size, 1u), mip_levels),
                        simultaneous_access);
                }(),
                storage, size, mip_levels} {}

public:
    Image() noexcept = default;
    ~Image() noexcept override {
        if (*this) { device()->destroy_texture(handle()); }
    }
    using Resource::operator bool;
    Image(Image &&) noexcept = default;
    Image(Image const &) noexcept = delete;
    Image &operator=(Image &&rhs) noexcept {
        _move_from(std::move(rhs));
        return *this;
    }
    Image &operator=(Image const &) noexcept = delete;
    // properties
    [[nodiscard]] auto size() const noexcept {
        _check_is_valid();
        return _size;
    }
    [[nodiscard]] auto mip_levels() const noexcept {
        _check_is_valid();
        return _mip_levels;
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
            detail::error_image_invalid_mip_levels(level, _mip_levels);
        }
        auto mip_size = luisa::max(_size >> level, 1u);
        return ImageView<T>{handle(), _storage, level, mip_size};
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
        return reinterpret_cast<const detail::ImageExprProxy<Image<T>> *>(this);
    }
};

// An ImageView is a reference to an Image without ownership
// (i.e., it will not destroy the Image on destruction).
template<typename T>
class ImageView {

private:
    uint64_t _handle;
    uint2 _size;
    uint _level;
    PixelStorage _storage;

private:
    friend class Image<T>;
    friend class SparseImage<T>;
    friend class detail::MipmapView;
    friend class DepthBuffer;

    [[nodiscard]] auto _as_mipmap() const noexcept {
        return detail::MipmapView{_handle, make_uint3(_size, 1u), _level, _storage};
    }

public:
    ImageView(uint64_t handle, PixelStorage storage,
              uint level, uint2 size) noexcept
        : _handle{handle}, _size{size},
          _level{level}, _storage{storage} {}

    ImageView(const Image<T> &image) noexcept : ImageView{image.view(0u)} {}
    // properties
    [[nodiscard]] auto handle() const noexcept { return _handle; }
    [[nodiscard]] auto size() const noexcept { return _size; }
    [[nodiscard]] auto size_bytes() const noexcept {
        return pixel_storage_size(_storage, make_uint3(_size, 1u));
    }
    [[nodiscard]] auto storage() const noexcept { return _storage; }
    [[nodiscard]] auto format() const noexcept { return pixel_storage_to_format<T>(_storage); }
    [[nodiscard]] auto level() const noexcept { return _level; }
    // commands
    // copy image's data to pointer or another image
    template<typename U>
    [[nodiscard]] auto copy_to(U &&dst) const noexcept { return _as_mipmap().copy_to(std::forward<U>(dst)); }
    // copy pointer or another image's data to image
    template<typename U>
    [[nodiscard]] auto copy_from(U &&src) const noexcept { return _as_mipmap().copy_from(std::forward<U>(src)); }
    // DSL interface
    [[nodiscard]] auto operator->() const noexcept {
        return reinterpret_cast<const detail::ImageExprProxy<ImageView<T>> *>(this);
    }
};

template<typename T>
ImageView(const Image<T> &) -> ImageView<T>;

template<typename T>
ImageView(ImageView<T>) -> ImageView<T>;

namespace detail {

template<typename T>
struct is_image_impl : std::false_type {};

template<typename T>
struct is_image_impl<Image<T>> : std::true_type {};

template<typename T>
struct is_image_view_impl : std::false_type {};

template<typename T>
struct is_image_view_impl<ImageView<T>> : std::true_type {};

}// namespace detail

template<typename T>
using is_image = detail::is_image_impl<std::remove_cvref_t<T>>;

template<typename T>
using is_image_view = detail::is_image_view_impl<std::remove_cvref_t<T>>;

template<typename T>
using is_image_or_view = std::disjunction<is_image<T>, is_image_view<T>>;

template<typename T>
constexpr auto is_image_v = is_image<T>::value;

template<typename T>
constexpr auto is_image_view_v = is_image_view<T>::value;

template<typename T>
constexpr auto is_image_or_view_v = is_image_or_view<T>::value;

namespace detail {

template<typename ImageOrView>
struct image_element_impl {
    static_assert(always_false_v<ImageOrView>);
};

template<typename T>
struct image_element_impl<Image<T>> {
    using type = T;
};

template<typename T>
struct image_element_impl<ImageView<T>> {
    using type = T;
};

}// namespace detail

template<typename ImageOrView>
using image_element = detail::image_element_impl<std::remove_cvref_t<ImageOrView>>;

template<typename ImageOrView>
using image_element_t = typename image_element<ImageOrView>::type;

}// namespace luisa::compute

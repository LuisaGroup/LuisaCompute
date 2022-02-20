//
// Created by Mike Smith on 2021/10/28.
//

#pragma once

#include <runtime/command.h>

#ifdef min
#undef min
#endif

#ifdef max
#undef max
#endif

namespace luisa::compute {

template<typename T>
class Image;

template<typename T>
class Volume;

template<typename T>
class ImageView;

template<typename T>
class VolumeView;

template<typename T>
class BufferView;

namespace detail {

class MipmapView {

private:
    uint64_t _handle;
    uint3 _size;
    uint3 _offset;
    uint32_t _level;
    PixelStorage _storage;

public:
    MipmapView(uint64_t handle, uint3 size, uint3 offset, uint32_t level, PixelStorage storage) noexcept
        : _handle{handle},
          _size{size},
          _offset{offset},
          _level{level},
          _storage{storage} {
        LUISA_VERBOSE_WITH_LOCATION(
            "Mipmap: offset = [{}, {}, {}], size = [{}, {}, {}], level = {}.",
            offset.x, offset.y, offset.z, size.x, size.y, size.z, level);
    }

    [[nodiscard]] constexpr auto size_bytes() const noexcept {
        return _size.x * _size.y * _size.z * pixel_storage_size(_storage);
    }

    [[nodiscard]] auto copy_from(const void *data) const noexcept {
        return TextureUploadCommand::create(
            _handle, _storage, _level, _offset, _size, data);
    }

    template<typename T>
    [[nodiscard]] auto copy_from(const Image<T> &src) const noexcept {
        return copy_from(src.view());
    }

    template<typename T>
    [[nodiscard]] auto copy_from(const Volume<T> &src) const noexcept {
        return copy_from(src.view());
    }

    template<typename T>
    [[nodiscard]] auto copy_from(ImageView<T> src) const noexcept {
        return copy_from(src._as_mipmap());
    }

    template<typename T>
    [[nodiscard]] auto copy_from(VolumeView<T> src) const noexcept {
        return copy_from(src._as_mipmap());
    }

    [[nodiscard]] auto copy_from(MipmapView src) const noexcept {
        if (!all(_size == src._size)) [[unlikely]] {
            LUISA_ERROR_WITH_LOCATION(
                "MipmapView sizes mismatch in copy command "
                "(src: [{}, {}], dest: [{}, {}]).",
                src._size.x, src._size.y, _size.x, _size.y);
        }
        if (src._storage != _storage) [[unlikely]] {
            LUISA_ERROR_WITH_LOCATION(
                "MipmapView storages mismatch "
                "(src = {}, dst = {})",
                to_underlying(src._storage),
                to_underlying(_storage));
        }
        return TextureCopyCommand::create(
            _storage,
            src._handle, _handle,
            src._level, _level,
            src._offset, _offset, _size);
    }

    template<typename U>
    [[nodiscard]] auto copy_from(BufferView<U> buffer) const noexcept {
        if (auto size = size_bytes(); buffer.size_bytes() < size) {
            LUISA_ERROR_WITH_LOCATION(
                "No enough data (required = {} bytes) in buffer (size = {} bytes).",
                size, buffer.size_bytes());
        }
        return BufferToTextureCopyCommand::create(
            buffer.handle(), buffer.offset_bytes(),
            _handle, _storage, _level, _offset, _size);
    }

    template<typename U>
    [[nodiscard]] auto copy_to(BufferView<U> buffer) const noexcept {
        if (auto size = size_bytes(); buffer.size_bytes() < size) {
            LUISA_ERROR_WITH_LOCATION(
                "No enough data (required = {} bytes) in buffer (size = {} bytes).",
                size, buffer.size_bytes());
        }
        return TextureToBufferCopyCommand::create(
            buffer.handle(), buffer.offset_bytes(),
            _handle, _storage, _level, _offset, _size);
    }

    [[nodiscard]] auto copy_to(void *data) const noexcept {
        return TextureDownloadCommand::create(
            _handle, _storage, _level, _offset, _size, data);
    }

    template<typename T>
        requires requires(T t) { t.copy_from(std::declval<MipmapView>()); }
    [[nodiscard]] auto copy_to(T &&dst) const noexcept {
        return std::forward<T>(dst).copy_from(*this);
    }
};

[[nodiscard]] constexpr auto max_mip_levels(uint3 size, uint requested_levels) noexcept {
    auto max_size = std::max({size.x, size.y, size.z});
    auto max_levels = 0u;
    while (max_size != 0u) {
        max_size >>= 1u;
        max_levels++;
    }
    return requested_levels == 0u ? max_levels : std::min(requested_levels, max_levels);
}

}// namespace detail
}// namespace luisa::compute

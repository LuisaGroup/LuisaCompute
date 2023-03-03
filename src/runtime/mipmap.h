//
// Created by Mike Smith on 2021/10/28.
//

#pragma once

#include <runtime/command.h>

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

class LC_RUNTIME_API MipmapView {

private:
    uint64_t _handle;
    uint3 _size;
    uint32_t _level;
    PixelStorage _storage;

private:
    [[noreturn]] static void _error_mipmap_copy_buffer_size_mismatch(size_t mip_size, size_t buffer_size) noexcept;

public:
    MipmapView(uint64_t handle, uint3 size, uint32_t level, PixelStorage storage) noexcept;
    [[nodiscard]] constexpr auto size_bytes() const noexcept {
        return pixel_storage_size(_storage, _size);
    }

    [[nodiscard]] auto copy_from(const void *data) const noexcept {
        return TextureUploadCommand::create(
            _handle, _storage, _level, _size, data);
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

    [[nodiscard]] luisa::unique_ptr<TextureCopyCommand> copy_from(MipmapView src) const noexcept;
    template<typename U>
    [[nodiscard]] auto copy_from(BufferView<U> buffer) const noexcept {
        if (auto size = size_bytes(); buffer.size_bytes() < size) {
            _error_mipmap_copy_buffer_size_mismatch(size, buffer.size_bytes());
        }
        return BufferToTextureCopyCommand::create(
            buffer.handle(), buffer.offset_bytes(),
            _handle, _storage, _level, _size);
    }

    template<typename U>
    [[nodiscard]] auto copy_to(BufferView<U> buffer) const noexcept {
        if (auto size = size_bytes(); buffer.size_bytes() < size) {
            _error_mipmap_copy_buffer_size_mismatch(size, buffer.size_bytes());
        }
        return TextureToBufferCopyCommand::create(
            buffer.handle(), buffer.offset_bytes(),
            _handle, _storage, _level, _size);
    }

    [[nodiscard]] auto copy_to(void *data) const noexcept {
        return TextureDownloadCommand::create(
            _handle, _storage, _level, _size, data);
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

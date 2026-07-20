#pragma once

#include <luisa/core/logging.h>
#include <luisa/runtime/volume.h>
#include <luisa/runtime/sparse_texture.h>
#include <luisa/runtime/sparse_heap.h>

namespace luisa::compute {

template<typename T>
class SparseVolume final : public SparseTexture {

private:
    uint3 _size{};
    uint32_t _mip_levels{};
    PixelStorage _storage{};

private:
    friend class Device;
    friend class ResourceGenerator;
    SparseVolume(DeviceInterface *device, const SparseTextureCreationInfo &create_info, PixelStorage storage, uint3 size, uint mip_levels) noexcept
        : SparseTexture{
              device,
              create_info},
          _size{size}, _mip_levels{detail::max_mip_levels(size, mip_levels)}, _storage{storage} {
    }
    SparseVolume(DeviceInterface *device, PixelStorage storage, uint3 size, uint mip_levels, bool simultaneous_access) noexcept
        : SparseVolume{
              device,
              [&] {
                  if (size.x == 0 || size.y == 0 || size.z == 0) [[unlikely]] {
                      detail::volume_size_zero_error();
                  }
                  auto info = device->create_sparse_texture(
                      pixel_storage_to_format<T>(storage), 3u,
                      size.x, size.y, size.z,
                      detail::max_mip_levels(size, mip_levels), simultaneous_access);
#ifdef LUISA_ENABLE_SAFE_MODE
                  if (!info.valid()) {
                      LUISA_ERROR("Failed to create sparse volume.");
                  }
#endif
                  return info;
              }(),
              storage, size, mip_levels} {
    }
    [[nodiscard]] auto _check_tile_region(
        uint3 start_tile,
        uint3 tile_count,
        uint mip_level) const noexcept {
        return detail::check_sparse_texture_tile_region(
            3u, _size, _tile_size, _mip_levels, mip_level,
            start_tile, tile_count);
    }

public:
    SparseVolume() noexcept = default;
    using Resource::operator bool;
    using Resource::release;
    SparseVolume(SparseVolume &&) noexcept = default;
    SparseVolume(const SparseVolume &) noexcept = delete;
    SparseVolume &operator=(SparseVolume &&rhs) noexcept {
        _move_from(std::move(rhs));
        return *this;
    }
    SparseVolume &operator=(const SparseVolume &) noexcept = delete;

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
        return VolumeView<T>{native_handle(), handle(), _storage, level, mip_size};
    }
    [[nodiscard]] auto view() const noexcept {
        _check_is_valid();
        return view(0u);
    }
    [[nodiscard]] auto tile_size() const noexcept {
        _check_is_valid();
        return _tile_size;
    }

    [[nodiscard]] auto map_tile(uint3 start_tile, uint3 tile_count, uint mip_level, const SparseTextureHeap &heap) noexcept {
        _check_is_valid();
        static_cast<void>(_check_tile_region(
            start_tile, tile_count, mip_level));
        detail::check_sparse_heap_provenance(*this, heap);
        return SparseUpdateTile{
            .handle = handle(),
            .operations =
                SparseTextureMapOperation{
                    .start_tile = start_tile,
                    .tile_count = tile_count,
                    .allocated_heap = heap.handle(),
                    .mip_level = mip_level}};
    }
    [[nodiscard]] auto unmap_tile(uint3 start_tile, uint3 tile_count, uint mip_level) noexcept {
        _check_is_valid();
        static_cast<void>(_check_tile_region(
            start_tile, tile_count, mip_level));
        return SparseUpdateTile{
            .handle = handle(),
            .operations = SparseTextureUnMapOperation{
                .start_tile = start_tile,
                .tile_count = tile_count,
                .mip_level = mip_level}};
    }

    // command
    [[nodiscard]] auto copy_from(uint3 start_tile, uint3 tile_count, uint mip_level, const void *data) const noexcept {
        _check_is_valid();
        auto region = _check_tile_region(
            start_tile, tile_count, mip_level);
        return luisa::make_unique<TextureUploadCommand>(
            handle(), _storage, mip_level,
            region.extent, data, region.offset);
    }
    [[nodiscard]] auto copy_to(uint3 start_tile, uint3 tile_count, uint mip_level, void *data) const noexcept {
        _check_is_valid();
        auto region = _check_tile_region(
            start_tile, tile_count, mip_level);
        return luisa::make_unique<TextureDownloadCommand>(
            handle(), _storage, mip_level,
            region.extent, data, region.offset);
    }
    template<typename U>
    [[nodiscard]] auto copy_from(uint3 start_tile, uint3 tile_count, uint mip_level, BufferView<U> buffer_view) const noexcept {
        _check_is_valid();
        auto region = _check_tile_region(
            start_tile, tile_count, mip_level);
        detail::check_sparse_texture_copy_buffer_size(
            _storage, region.extent, buffer_view.size_bytes());
        return luisa::make_unique<BufferToTextureCopyCommand>(
            buffer_view.handle(), buffer_view.offset_bytes(),
            handle(), _storage, mip_level,
            region.extent, region.offset);
    }
    template<typename U>
    [[nodiscard]] auto copy_from(uint3 start_tile, uint3 tile_count, uint mip_level, const Buffer<U> &buffer) const noexcept {
        return copy_from(
            start_tile, tile_count, mip_level, buffer.view());
    }
    template<typename U>
    [[nodiscard]] auto copy_to(uint3 start_tile, uint3 tile_count, uint mip_level, BufferView<U> buffer_view) const noexcept {
        _check_is_valid();
        auto region = _check_tile_region(
            start_tile, tile_count, mip_level);
        detail::check_sparse_texture_copy_buffer_size(
            _storage, region.extent, buffer_view.size_bytes());
        return luisa::make_unique<TextureToBufferCopyCommand>(
            buffer_view.handle(), buffer_view.offset_bytes(),
            handle(), _storage, mip_level,
            region.extent, region.offset);
    }
    template<typename U>
    [[nodiscard]] auto copy_to(uint3 start_tile, uint3 tile_count, uint mip_level, const Buffer<U> &buffer) const noexcept {
        return copy_to(
            start_tile, tile_count, mip_level, buffer.view());
    }
};
namespace detail {
template<typename T>
struct is_sparse_volume_impl : std::false_type {};

template<typename T>
struct is_sparse_volume_impl<SparseVolume<T>> : std::true_type {};

template<typename T>
struct is_volume_impl<SparseVolume<T>> : std::true_type {};
}// namespace detail
template<typename T>
using is_sparse_volume = detail::is_sparse_volume_impl<std::remove_cvref_t<T>>;

template<typename T>
constexpr auto is_sparse_volume_v = is_sparse_volume<T>::value;
}// namespace luisa::compute

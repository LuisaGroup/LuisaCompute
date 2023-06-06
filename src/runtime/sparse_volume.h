#pragma once

#include <runtime/volume.h>
#include <runtime/sparse_texture.h>

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
    SparseVolume(DeviceInterface *device, const ResourceCreationInfo &create_info, PixelStorage storage, uint3 size, uint mip_levels) noexcept
        : SparseTexture{
              device,
              create_info},
          _size{size}, _mip_levels{mip_levels}, _storage{storage} {
    }
    SparseVolume(DeviceInterface *device, PixelStorage storage, uint3 size, uint mip_levels = 1u) noexcept
        : SparseVolume{
              device,
              [&] {
                  if (size.x == 0 || size.y == 0 || size.z == 0) [[unlikely]] {
                      detail::volume_size_zero_error();
                  }
                  return device->create_sparse_texture(
                      pixel_storage_to_format<T>(storage), 3u,
                      size.x, size.y, size.x,
                      detail::max_mip_levels(size, mip_levels));
              }(),
              storage, size, mip_levels} {
    }

public:
    using Resource::operator bool;
    SparseVolume(SparseVolume &&) noexcept = default;
    SparseVolume(const SparseVolume &) noexcept = delete;
    SparseVolume &operator=(SparseVolume &&rhs) noexcept {
        _move_from(std::move(rhs));
        return *this;
    }
    SparseVolume &operator=(const SparseVolume &) noexcept = delete;

    // properties
    [[nodiscard]] auto mip_levels() const noexcept { return _mip_levels; }
    [[nodiscard]] auto size() const noexcept { return _size; }
    [[nodiscard]] auto storage() const noexcept { return _storage; }
    [[nodiscard]] auto format() const noexcept { return pixel_storage_to_format<T>(_storage); }
    [[nodiscard]] auto view(uint32_t level) const noexcept {
        if (level >= _mip_levels) [[unlikely]] {
            detail::error_volume_invalid_mip_levels(level, _mip_levels);
        }
        auto mip_size = luisa::max(_size >> level, 1u);
        return VolumeView<T>{handle(), _storage, level, mip_size};
    }
    [[nodiscard]] auto view() const noexcept { return view(0u); }
    void map_tile(uint3 start_coord, uint3 size, uint mip_level) noexcept {
        _tiles.emplace_back(TileModification{
            .start_coord = start_coord,
            .size = size,
            .mip_level = mip_level,
            .operation = TileModification::Operation::Map});
    }
    void unmap_tile(uint3 start_coord, uint mip_level) noexcept {
        _tiles.emplace_back(TileModification{
            .start_coord = start_coord,
            .mip_level = mip_level,
            .operation = TileModification::Operation::UnMap});
    }

    // command
    [[nodiscard]] auto copy_from(uint3 start_coord, uint3 size, uint mip_level, const void *data) const noexcept {
        return luisa::make_unique<SparseTextureUploadCommand>(
            data, handle(), start_coord, size, _storage, mip_level);
    }
    template<typename U>
    [[nodiscard]] auto copy_from(uint3 start_coord, uint3 size, uint mip_level, BufferView<U> buffer_view) const noexcept {
        return luisa::make_unique<BufferToSparseTextureCopyCommand>(
            buffer_view.handle(), buffer_view.offset_bytes(), handle(), start_coord, size, _storage, mip_level);
    }
    template<typename U>
    [[nodiscard]] auto copy_from(uint3 start_coord, uint3 size, uint mip_level, const Buffer<U> &buffer) const noexcept {
        return luisa::make_unique<BufferToSparseTextureCopyCommand>(
            buffer.handle(), 0ull, handle(), start_coord, size, _storage, mip_level);
    }
};

}// namespace luisa::compute

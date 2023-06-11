#pragma once

#include <luisa/runtime/image.h>
#include <luisa/runtime/sparse_texture.h>

namespace luisa::compute {

template<typename T>
class SparseImage final : public SparseTexture {

private:
    uint2 _size{};
    uint32_t _mip_levels{};
    PixelStorage _storage{};

private:
    friend class Device;
    friend class ResourceGenerator;
    SparseImage(DeviceInterface *device, const SparseTextureCreationInfo &create_info, PixelStorage storage, uint2 size, uint mip_levels) noexcept
        : SparseTexture{
              device,
              create_info},
          _size{size}, _mip_levels{mip_levels}, _storage{storage} {
    }
    SparseImage(DeviceInterface *device, PixelStorage storage, uint2 size, uint mip_levels = 1u) noexcept
        : SparseImage{
              device,
              [&] {
                  if (size.x == 0 || size.y == 0) [[unlikely]] {
                      detail::image_size_zero_error();
                  }
                  return device->create_sparse_texture(
                      pixel_storage_to_format<T>(storage), 2u,
                      size.x, size.y, 1u,
                      detail::max_mip_levels(make_uint3(size, 1u), mip_levels));
              }(),
              storage, size, mip_levels} {
    }

public:
    using Resource::operator bool;
    SparseImage(SparseImage &&) noexcept = default;
    SparseImage(const SparseImage &) noexcept = delete;
    SparseImage &operator=(SparseImage &&rhs) noexcept {
        _move_from(std::move(rhs));
        return *this;
    }
    SparseImage &operator=(const SparseImage &) noexcept = delete;
    [[nodiscard]] auto view(uint32_t level) const noexcept {
        if (level >= _mip_levels) [[unlikely]] {
            detail::error_image_invalid_mip_levels(level, _mip_levels);
        }
        auto mip_size = luisa::max(_size >> level, 1u);
        return ImageView<T>{handle(), _storage, level, mip_size};
    }
    // properties
    [[nodiscard]] auto size() const noexcept { return _size; }
    [[nodiscard]] auto mip_levels() const noexcept { return _mip_levels; }
    [[nodiscard]] auto storage() const noexcept { return _storage; }
    [[nodiscard]] auto format() const noexcept { return pixel_storage_to_format<T>(_storage); }
    [[nodiscard]] auto view() const noexcept { return view(0u); }
    [[nodiscard]] auto tile_size() const noexcept { return _tile_size.xy(); }

    void map_tile(uint2 start_tile, uint2 tile_count, uint mip_level) noexcept {
        detail::check_sparse_tex2d_map(_size, _tile_size.xy(), start_tile, tile_count);
        _operations.emplace_back(SparseTextureMapOperation{
            .start_tile = make_uint3(start_tile, 0u),
            .tile_count = make_uint3(tile_count, 1u),
            .mip_level = mip_level});
    }
    void unmap_tile(uint2 start_tile, uint mip_level) noexcept {
        detail::check_sparse_tex2d_unmap(_size, _tile_size.xy(), start_tile);
        _operations.emplace_back(SparseTextureUnMapOperation{
            .start_tile = make_uint3(start_tile, 0u),
            .mip_level = mip_level});
    }

    // command
    [[nodiscard]] auto copy_from(uint2 start_tile, uint2 tile_count, uint mip_level, const void *data) const noexcept {
        return luisa::make_unique<TextureUploadCommand>(
            handle(), _storage, mip_level, make_uint3(tile_count * tile_size(), 1u), data, make_uint3(start_tile * tile_size(), 0u));
    }
    [[nodiscard]] auto copy_to(uint2 start_tile, uint2 tile_count, uint mip_level, void *data) const noexcept {
        return luisa::make_unique<TextureDownloadCommand>(
            handle(), _storage, mip_level, make_uint3(tile_count * tile_size(), 1u), data, make_uint3(start_tile * tile_size(), 0u));
    }
    template<typename U>
    [[nodiscard]] auto copy_from(uint2 start_tile, uint2 tile_count, uint mip_level, BufferView<U> buffer_view) const noexcept {
        return luisa::make_unique<BufferToTextureCopyCommand>(
            buffer_view.handle(), buffer_view.offset_bytes(), handle(), _storage, mip_level, make_uint3(tile_count * tile_size(), 1u), make_uint3(start_tile * tile_size(), 0u), _storage, mip_level);
    }
    template<typename U>
    [[nodiscard]] auto copy_from(uint2 start_tile, uint2 tile_count, uint mip_level, const Buffer<U> &buffer) const noexcept {
        return luisa::make_unique<BufferToTextureCopyCommand>(
            buffer.handle(), 0u, handle(), _storage, mip_level, make_uint3(tile_count * tile_size(), 1u), make_uint3(start_tile * tile_size(), 0u), _storage, mip_level);
    }
    template<typename U>
    [[nodiscard]] auto copy_to(uint2 start_tile, uint2 tile_count, uint mip_level, BufferView<U> buffer_view) const noexcept {
        return luisa::make_unique<TextureToBufferCopyCommand>(
            buffer_view.handle(), buffer_view.offset_bytes(), handle(), _storage, mip_level, make_uint3(tile_count * tile_size(), 1u), make_uint3(start_tile * tile_size(), 0u), _storage, mip_level);
    }
    template<typename U>
    [[nodiscard]] auto copy_to(uint2 start_tile, uint2 tile_count, uint mip_level, const Buffer<U> &buffer) const noexcept {
        return luisa::make_unique<TextureToBufferCopyCommand>(
            buffer.handle(), 0u, handle(), _storage, mip_level, make_uint3(tile_count * tile_size(), 1u), make_uint3(start_tile * tile_size(), 0u), _storage, mip_level);
    }
};
namespace detail {

template<typename T>
struct is_sparse_image_impl : std::false_type {};

template<typename T>
struct is_sparse_image_impl<SparseImage<T>> : std::true_type {};
}// namespace detail
template<typename T>
using is_sparse_image = detail::is_sparse_image_impl<std::remove_cvref_t<T>>;

template<typename T>
constexpr auto is_sparse_image_v = is_sparse_image<T>::value;
}// namespace luisa::compute


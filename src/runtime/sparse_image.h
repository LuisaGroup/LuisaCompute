#pragma once

#include <runtime/image.h>
#include <runtime/sparse_texture.h>

namespace luisa::compute {

template<typename T>
class SparseImage final : public SparseTexture {

private:
    uint2 _size{};
    uint32_t _mip_levels{};
    PixelStorage _storage{};

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

    void map_tile(uint2 start_coord, uint2 size, uint mip_level) noexcept {
        _tiles.emplace_back(TileModification{
            .start_coord = make_uint3(start_coord, 0u),
            .size = make_uint3(size, 1u),
            .mip_level = mip_level,
            .operation = TileModification::Operation::Map});
    }
    void unmap_tile(uint2 start_coord, uint mip_level) noexcept {
        _tiles.emplace_back(TileModification{
            .start_coord = make_uint3(start_coord, 0u),
            .mip_level = mip_level,
            .operation = TileModification::Operation::UnMap});
    }

    // command
    [[nodiscard]] auto copy_from(uint2 start_coord, uint2 size, uint mip_level, const void *data) const noexcept {
        return luisa::make_unique<SparseTextureUploadCommand>(
            data, handle(), make_uint3(start_coord, 0u), make_uint3(size, 1u), _storage, mip_level);
    }
    template <typename U>
    [[nodiscard]] auto copy_from(uint2 start_coord, uint2 size, uint mip_level, BufferView<U> buffer_view) const noexcept {
        return luisa::make_unique<BufferToSparseTextureCopyCommand>(
            buffer_view.handle(), buffer_view.offset_bytes(), handle(), make_uint3(start_coord, 0u), make_uint3(size, 1u), _storage, mip_level);
    }
    template <typename U>
    [[nodiscard]] auto copy_from(uint2 start_coord, uint2 size, uint mip_level, Buffer<U> buffer_view) const noexcept {
        return luisa::make_unique<BufferToSparseTextureCopyCommand>(
            buffer_view.handle(), 0ull, handle(), make_uint3(start_coord, 0u), make_uint3(size, 1u), _storage, mip_level);
    }
};

}// namespace luisa::compute

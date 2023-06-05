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
};
}// namespace luisa::compute
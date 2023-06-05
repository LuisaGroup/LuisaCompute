#pragma once

#include <runtime/image.h>
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
    SparseVolume &operator=(SparseVolume &&) noexcept = delete;// use _move_from in derived classes
    SparseVolume &operator=(const SparseVolume &) noexcept = delete;
};
}// namespace luisa::compute
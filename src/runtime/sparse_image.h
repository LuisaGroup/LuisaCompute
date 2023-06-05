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
    SparseImage &operator=(SparseImage &&) noexcept = delete;// use _move_from in derived classes
    SparseImage &operator=(const SparseImage &) noexcept = delete;
};
}// namespace luisa::compute
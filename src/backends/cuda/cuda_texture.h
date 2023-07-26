#pragma once

#include <array>
#include <cuda.h>

#include <luisa/core/spin_mutex.h>
#include <luisa/runtime/rhi/pixel.h>

namespace luisa::compute::cuda {

/**
 * @brief Struct of surface on CUDA
 * 
 */
struct alignas(16) CUDASurface {
    CUsurfObject handle;
    uint64_t storage;
};

static_assert(sizeof(CUDASurface) == 16u);

/**
 * @brief Mipmap array on CUDA
 * 
 */
class CUDATexture {

public:
    static constexpr auto max_level_count = 15u;
    using Binding = CUDASurface;

private:
    uint64_t _base_array;
    std::array<CUarray, max_level_count> _mip_arrays{};
    std::array<CUsurfObject, max_level_count> _mip_surfaces{};
    uint16_t _size[3];
    uint8_t _format;
    uint8_t _levels;

public:
    CUDATexture(uint64_t array, uint3 size, PixelFormat format, uint32_t levels) noexcept;
    ~CUDATexture() noexcept;
    [[nodiscard]] auto handle() const noexcept { return _base_array; }
    [[nodiscard]] auto format() const noexcept { return static_cast<PixelFormat>(_format); }
    [[nodiscard]] auto storage() const noexcept { return pixel_format_to_storage(format()); }
    [[nodiscard]] auto levels() const noexcept { return static_cast<size_t>(_levels); }
    [[nodiscard]] CUarray level(uint32_t i) const noexcept;
    [[nodiscard]] CUDASurface surface(uint32_t level) const noexcept;
    [[nodiscard]] uint3 size() const noexcept { return make_uint3(_size[0], _size[1], _size[2]); }
    [[nodiscard]] auto binding(uint32_t level) const noexcept { return surface(level); }
    void set_name(luisa::string &&name) noexcept;
};

static_assert(sizeof(CUDATexture) == 256u);

}// namespace luisa::compute::cuda


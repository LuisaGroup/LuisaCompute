#pragma once

#include <array>
#include <cuda.h>

#include <luisa/core/spin_mutex.h>
#include <luisa/runtime/rhi/pixel.h>

namespace luisa::compute::cuda {

/**
 * @brief Struct of surface on CUDA
 *
 * Layout of the `storage` field (packed):
 *   bits [0:15]   = width  (uint16)
 *   bits [16:31]  = height (uint16)
 *   bits [32:47]  = depth  (uint16)
 *   bits [48:55]  = PixelStorage enum (uint8)
 *   bits [56:63]  = reserved (0)
 */
struct alignas(16) CUDASurface {
    CUsurfObject handle;
    uint64_t storage;

    [[nodiscard]] static uint64_t encode_storage(uint16_t w, uint16_t h, uint16_t d, uint8_t pixel_storage) noexcept {
        return static_cast<uint64_t>(w) |
               (static_cast<uint64_t>(h) << 16u) |
               (static_cast<uint64_t>(d) << 32u) |
               (static_cast<uint64_t>(pixel_storage) << 48u);
    }
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

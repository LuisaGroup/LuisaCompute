#pragma once

#include <array>
#include <atomic>
#include <mutex>
#include <cuda.h>

#include <luisa/core/spin_mutex.h>
#include <luisa/runtime/rhi/pixel.h>
#include <luisa/runtime/rhi/sampler.h>

namespace luisa::compute::cuda {

class CUDADevice;
class CUDATexture;

/**
 * @brief Struct of surface on CUDA
 *
 * Layout of the `storage` field (packed):
 *   bits [0:15]   = width  (uint16)
 *   bits [16:31]  = height (uint16)
 *   bits [32:47]  = depth  (uint16)
 *   bits [48:55]  = PixelStorage enum (uint8)
 *   bits [56:63]  = base_level (uint8)
 */
struct alignas(16) CUDASurface {
    CUsurfObject handle;       // surface handle (0 when !needs_surface or BC texture)
    uint64_t storage;          // [0:15]=w, [16:31]=h, [32:47]=d, [48:55]=PixelStorage, [56:63]=base_level
    uint64_t sampler_table;    // 0 when unsampled or single static sampler; device ptr to 16x CUtexObject otherwise
    CUtexObject single_tex;    // non-zero for single static sampler; 0 otherwise

    [[nodiscard]] static uint64_t encode_storage(
        uint16_t w, uint16_t h, uint16_t d,
        uint8_t pixel_storage, uint8_t base_level = 0u) noexcept {
        return static_cast<uint64_t>(w) |
               (static_cast<uint64_t>(h) << 16u) |
               (static_cast<uint64_t>(d) << 32u) |
               (static_cast<uint64_t>(pixel_storage) << 48u) |
               (static_cast<uint64_t>(base_level) << 56u);
    }
};

static_assert(sizeof(CUDASurface) == 32u);

[[nodiscard]] CUDA_TEXTURE_DESC cuda_texture_descriptor(Sampler sampler) noexcept;
[[nodiscard]] CUtexObject create_cuda_texture_object(const CUDATexture *texture, Sampler sampler) noexcept;

/**
 * @brief Mipmap array on CUDA
 * 
 */
class CUDATexture {

public:
    static constexpr auto max_level_count = 15u;
    using Binding = CUDASurface;

private:
    struct SamplerCache {
        spin_mutex mutex{};
        std::array<CUtexObject, 16u> objects{};
        uint16_t uploaded_mask{0u};
        CUdeviceptr device_table{0ull};
    };

    uint64_t _base_array;
    std::array<CUarray, max_level_count> _mip_arrays{};
    std::array<CUsurfObject, max_level_count> _mip_surfaces{};
    uint16_t _size[3];
    uint8_t _format;
    uint8_t _levels;
    CUDADevice *_device{nullptr};
    mutable std::atomic<SamplerCache *> _sampler_cache{nullptr};

private:
    [[nodiscard]] SamplerCache *_get_or_create_sampler_cache() const noexcept;

public:
    CUDATexture(CUDADevice *device, uint64_t array, uint3 size, PixelFormat format, uint32_t levels) noexcept;
    ~CUDATexture() noexcept;
    [[nodiscard]] auto handle() const noexcept { return _base_array; }
    [[nodiscard]] auto format() const noexcept { return static_cast<PixelFormat>(_format); }
    [[nodiscard]] auto storage() const noexcept { return pixel_format_to_storage(format()); }
    [[nodiscard]] auto levels() const noexcept { return static_cast<size_t>(_levels); }
    [[nodiscard]] CUarray level(uint32_t i) const noexcept;
    [[nodiscard]] uint64_t encoded_storage(uint32_t level) const noexcept;
    [[nodiscard]] CUtexObject tex_object(Sampler sampler) const noexcept;
    [[nodiscard]] CUdeviceptr sampler_table(uint16_t sampler_mask, CUstream stream = nullptr) const noexcept;
    [[nodiscard]] CUDASurface binding(uint32_t level, uint16_t sampler_mask = 0u, bool needs_surface = true, CUstream stream = nullptr) const noexcept;
    [[nodiscard]] CUDASurface surface(uint32_t level) const noexcept { return binding(level, 0u, true, nullptr); }
    [[nodiscard]] uint3 size() const noexcept { return make_uint3(_size[0], _size[1], _size[2]); }
    [[nodiscard]] bool has_sampler_table() const noexcept {
        auto *cache = _sampler_cache.load(std::memory_order_acquire);
        if (cache == nullptr) { return false; }
        std::scoped_lock lock{cache->mutex};
        return cache->device_table != 0ull;
    }
    [[nodiscard]] uint32_t created_tex_object_count() const noexcept {
        auto *cache = _sampler_cache.load(std::memory_order_acquire);
        if (cache == nullptr) { return 0u; }
        std::scoped_lock lock{cache->mutex};
        uint32_t count = 0u;
        for (auto t : cache->objects) {
            if (t != 0ull) { ++count; }
        }
        return count;
    }
    void set_name(luisa::string &&name) noexcept;
};

static_assert(sizeof(CUDATexture) == 272u);

}// namespace luisa::compute::cuda

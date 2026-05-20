//
// Created by mike on 1/10/26.
//

#pragma once

#include <span>

#include <hip/hip_runtime.h>
#include <luisa/runtime/rhi/pixel.h>
#include <luisa/runtime/rhi/sampler.h>

namespace luisa::compute::hip {

struct alignas(16) HIPSurface {
    hipSurfaceObject_t handle;
    uint64_t storage;
};

struct alignas(16) HIPTextureObject {
    hipDeviceptr_t handles;
    uint64_t level_count;
};

class HIPTexture {

public:
    static constexpr auto max_level_count = 15u;
    using Binding = HIPSurface;

private:
    void *_base_array{};
    hipArray_t _mip_arrays[max_level_count]{};
    hipSurfaceObject_t _mip_surfaces[max_level_count]{};
    uint16_t _size[3] = {};
    uint8_t _format = {};
    uint8_t _levels : 4 = {};
    uint8_t _dimension : 4 = {};

public:
    HIPTexture() noexcept;
    ~HIPTexture() noexcept;
    [[nodiscard]] auto handle() const noexcept { return _base_array; }
    [[nodiscard]] auto format() const noexcept { return static_cast<PixelFormat>(_format); }
    [[nodiscard]] auto storage() const noexcept { return pixel_format_to_storage(format()); }
    [[nodiscard]] auto levels() const noexcept { return static_cast<size_t>(_levels); }
    [[nodiscard]] hipArray_t level(uint32_t i) const noexcept;
    [[nodiscard]] HIPSurface surface(uint32_t level) const noexcept;
    [[nodiscard]] auto size() const noexcept { return make_uint3(_size[0], _size[1], _size[2]); }
    [[nodiscard]] auto dimension() const noexcept { return static_cast<uint>(_dimension); }
    [[nodiscard]] auto is_mipmapped() const noexcept { return _levels > 1u; }
    [[nodiscard]] auto binding(uint32_t level) const noexcept { return surface(level); }
    void create_texture_objects(std::span<hipTextureObject_t> objects, Sampler s) const noexcept;

public:
    [[nodiscard]] static HIPTexture *create_device_texture(PixelFormat format, uint dim, uint3 size, uint32_t mip_levels) noexcept;
    [[nodiscard]] static HIPTexture *import_external_texture(uint64_t external_array, PixelFormat format, uint dim, uint3 size, uint32_t mip_levels) noexcept;
};

}// namespace luisa::compute::hip

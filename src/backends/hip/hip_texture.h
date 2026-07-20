//
// Created by mike on 1/10/26.
//

#pragma once

#include <cstddef>
#include <span>

#include <hip/hip_runtime.h>
#include <luisa/runtime/rhi/pixel.h>
#include <luisa/runtime/rhi/sampler.h>

namespace luisa::compute::hip {

struct alignas(16) HIPSurface {
    hipSurfaceObject_t handle;
    // Pointer to HIPDirectTextureDescriptor with the bound base mip encoded
    // in the low four bits. hipMalloc allocations are sufficiently aligned
    // for this tag and HIPTexture::binding validates the assumption.
    uint64_t descriptor;
};

struct alignas(16) HIPTextureObject {
    hipDeviceptr_t handles;
    uint64_t level_count;
};

// ROCm represents a texture object as 12 image descriptor dwords followed by
// 8 sampler descriptor dwords. The AMDGPU image intrinsics used by our LLVM
// backend consume the first 8 and 4 dwords respectively. Keep these compact
// descriptors separate: mip levels only need distinct image descriptors,
// while sampler descriptors can be shared by every level and texture slot.
static_assert(HIP_IMAGE_OBJECT_SIZE_DWORD == 12u);
static_assert(HIP_SAMPLER_OBJECT_OFFSET_DWORD == 12u);
static_assert(HIP_SAMPLER_OBJECT_SIZE_DWORD == 8u);

struct alignas(16) HIPImageDescriptor {
    uint32_t words[8];
};

struct alignas(16) HIPSamplerDescriptor {
    uint32_t words[4];
};

static_assert(sizeof(HIPImageDescriptor) == 32u && alignof(HIPImageDescriptor) == 16u);
static_assert(sizeof(HIPSamplerDescriptor) == 16u && alignof(HIPSamplerDescriptor) == 16u);

// Device-resident metadata for directly bound textures. Direct texture
// arguments remain 16 bytes (HIPSurface), while this table makes all mip
// levels and every runtime-selected sampler available to generated code.
struct alignas(16) HIPDirectTextureDescriptor {
    uint64_t level_count;
    uint64_t storage;
    uint64_t size_xy;
    uint64_t size_z;
    HIPImageDescriptor images[15];
    HIPSamplerDescriptor samplers[16];
};

static_assert(offsetof(HIPDirectTextureDescriptor, images) == 32u);
static_assert(offsetof(HIPDirectTextureDescriptor, samplers) == 512u);
static_assert(sizeof(HIPDirectTextureDescriptor) == 768u);
static_assert(alignof(HIPDirectTextureDescriptor) == 16u);

class HIPTexture {

public:
    static constexpr auto max_level_count = 15u;
    static constexpr auto direct_descriptor_mip_tag_bits = 4u;
    static constexpr auto direct_descriptor_mip_tag_mask =
        (1ull << direct_descriptor_mip_tag_bits) - 1ull;
    static_assert(max_level_count <= (1u << direct_descriptor_mip_tag_bits));
    static_assert(alignof(HIPDirectTextureDescriptor) >=
                  (1u << direct_descriptor_mip_tag_bits));
    using Binding = HIPSurface;

private:
    void *_base_array{};
    hipArray_t _mip_arrays[max_level_count]{};
    hipSurfaceObject_t _mip_surfaces[max_level_count]{};
    hipDeviceptr_t _direct_descriptor{};
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
    [[nodiscard]] HIPSurface binding(uint32_t level) const noexcept;
    void create_texture_objects(std::span<hipTextureObject_t> objects, Sampler s) const noexcept;
    void copy_image_descriptors(std::span<HIPImageDescriptor> descriptors) const noexcept;
    void copy_sampler_descriptors(std::span<HIPSamplerDescriptor> descriptors) const noexcept;

private:
    void _initialize_direct_descriptor() noexcept;

public:
    [[nodiscard]] static HIPTexture *create_device_texture(PixelFormat format, uint dim, uint3 size, uint32_t mip_levels) noexcept;
    [[nodiscard]] static HIPTexture *import_external_texture(uint64_t external_array, PixelFormat format, uint dim, uint3 size, uint32_t mip_levels) noexcept;
};

}// namespace luisa::compute::hip

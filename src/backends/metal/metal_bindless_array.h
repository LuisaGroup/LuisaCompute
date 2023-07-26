#pragma once

#include <luisa/runtime/rhi/command.h>
#include "../common/resource_tracker.h"
#include "metal_api.h"

namespace luisa::compute::metal {

class MetalDevice;
class MetalCommandEncoder;

class MetalBindlessArray {

public:
    struct alignas(16) Slot {
        std::byte *buffer;
        size_t buffer_size : 48;
        uint sampler2d : 8;
        uint sampler3d : 8;
        MTL::ResourceID texture2d;
        MTL::ResourceID texture3d;
    };
    static_assert(sizeof(Slot) == 32);

    struct Binding {
        uint64_t array;
    };

private:
    std::mutex _mutex;
    MTL::Buffer *_array;
    MTL::ComputePipelineState *_update;
    luisa::vector<MTL::Buffer *> _buffer_slots;
    luisa::vector<MTL::Texture *> _tex2d_slots;
    luisa::vector<MTL::Texture *> _tex3d_slots;
    ResourceTracker _buffer_tracker;
    ResourceTracker _texture_tracker;

public:
    MetalBindlessArray(MetalDevice *device, size_t size) noexcept;
    ~MetalBindlessArray() noexcept;
    void set_name(luisa::string_view name) noexcept;
    void update(MetalCommandEncoder &encoder, BindlessArrayUpdateCommand *cmd) noexcept;
    void mark_resource_usages(MTL::ComputeCommandEncoder *encoder) noexcept;
    [[nodiscard]] auto handle() const noexcept { return _array; }
    [[nodiscard]] auto binding() const noexcept { return Binding{_array->gpuAddress()}; }
};

}// namespace luisa::compute::metal


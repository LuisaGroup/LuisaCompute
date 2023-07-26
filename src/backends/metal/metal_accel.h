#pragma once

#include <luisa/core/spin_mutex.h>
#include <luisa/runtime/rhi/resource.h>
#include <luisa/runtime/rhi/command.h>
#include "../common/resource_tracker.h"
#include "metal_api.h"

namespace luisa::compute::metal {

class MetalDevice;
class MetalPrimitive;
class MetalCommandEncoder;

class MetalAccel {

public:
    static constexpr auto reserved_primitive_count = 1024u;

private:
    MTL::AccelerationStructure *_handle{nullptr};
    MTL::Buffer *_instance_buffer{nullptr};
    MTL::Buffer *_update_buffer{nullptr};
    MTL::InstanceAccelerationStructureDescriptor *_descriptor{nullptr};
    MTL::ComputePipelineState *_update;
    luisa::vector<MetalPrimitive *> _primitives;
    luisa::vector<MTL::Resource *> _resources;
    NS::String *_name{nullptr};
    AccelOption _option;
    bool _requires_rebuild{true};
    spin_mutex _mutex;

public:
    struct Binding {
        MTL::ResourceID handle;
        uint64_t instance_buffer;
    };

private:
    void _do_build(MetalCommandEncoder &encoder) noexcept;
    void _do_update(MetalCommandEncoder &encoder) noexcept;

public:
    MetalAccel(MetalDevice *device, const AccelOption &option) noexcept;
    ~MetalAccel() noexcept;
    void build(MetalCommandEncoder &encoder, AccelBuildCommand *command) noexcept;
    [[nodiscard]] auto handle() const noexcept { return _handle; }
    [[nodiscard]] auto instance_buffer() const noexcept { return _instance_buffer; }
    [[nodiscard]] auto binding() const noexcept { return Binding{_handle->gpuResourceID(), _instance_buffer->gpuAddress()}; }
    [[nodiscard]] auto pointer_to_handle() const noexcept { return const_cast<void *>(static_cast<const void *>(&_handle)); }
    void set_name(luisa::string_view name) noexcept;
    void mark_resource_usages(MetalCommandEncoder &encoder,
                              MTL::ComputeCommandEncoder *command_encoder,
                              MTL::ResourceUsage usage) noexcept;
};

}// namespace luisa::compute::metal


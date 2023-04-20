//
// Created by Mike Smith on 2023/4/20.
//

#pragma once

#include <runtime/rhi/resource.h>
#include <runtime/rhi/command.h>
#include <backends/common/resource_tracker.h>
#include <backends/metal/metal_api.h>

namespace luisa::compute::metal {

class MetalDevice;
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
    ResourceTracker _tracker;
    luisa::vector<MTL::AccelerationStructure *> _primitives;
    AccelOption _option;
    bool _requires_rebuild{true};

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
};

}// namespace luisa::compute::metal

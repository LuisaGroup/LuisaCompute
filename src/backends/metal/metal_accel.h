//
// Created by Mike Smith on 2023/4/20.
//

#pragma once

#include <runtime/rhi/resource.h>
#include <backends/metal/metal_api.h>

namespace luisa::compute::metal {

class MetalDevice;

class MetalAccel {

private:
    MTL::AccelerationStructure *_handle{};
    MTL::Buffer *_instance_buffer{};
    MTL::Buffer *_update_buffer{};
    MTL::ComputePipelineState *_update;
    AccelOption _option;

public:
    struct Binding {
        MTL::ResourceID handle;
        uint64_t instance_buffer;
    };

public:
    MetalAccel(MetalDevice *device, const AccelOption &option) noexcept;
    ~MetalAccel() noexcept;
    [[nodiscard]] auto handle() const noexcept { return _handle; }
    [[nodiscard]] auto instance_buffer() const noexcept { return _instance_buffer; }
    [[nodiscard]] auto binding() const noexcept { return Binding{_handle->gpuResourceID(), _instance_buffer->gpuAddress()}; }
};

}// namespace luisa::compute::metal

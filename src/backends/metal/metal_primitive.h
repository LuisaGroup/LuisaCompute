//
// Created by Mike Smith on 2023/4/20.
//

#pragma once

#include <core/stl/vector.h>
#include <runtime/rhi/resource.h>
#include <backends/metal/metal_api.h>

namespace luisa::compute::metal {

class MetalCommandEncoder;

class MetalPrimitive {

private:
    MTL::AccelerationStructure *_handle{nullptr};
    luisa::string _name;
    AccelOption _option;

private:
    virtual void _do_add_resources(luisa::vector<MTL::Resource *> &resources) const noexcept = 0;

protected:
    void _do_build(MetalCommandEncoder &encoder, MTL::PrimitiveAccelerationStructureDescriptor *descriptor) noexcept;
    void _do_update(MetalCommandEncoder &encoder, MTL::PrimitiveAccelerationStructureDescriptor *descriptor) noexcept;

public:
    MetalPrimitive(MTL::Device *device, const AccelOption &option) noexcept;
    virtual ~MetalPrimitive() noexcept;
    [[nodiscard]] auto handle() const noexcept { return _handle; }
    [[nodiscard]] auto option() const noexcept { return _option; }
    [[nodiscard]] auto name() const noexcept { return luisa::string_view{_name}; }
    void set_name(luisa::string_view name) noexcept { _name = name; }
    void add_resources(luisa::vector<MTL::Resource *> &resources) const noexcept;
};

}// namespace luisa::compute::metal

#pragma once

#include <luisa/runtime/rhi/command.h>
#include "metal_primitive.h"

namespace luisa::compute::metal {

class MetalCommandEncoder;

class MetalProceduralPrimitive : public MetalPrimitive {

private:
    MTL4::PrimitiveAccelerationStructureDescriptor *_descriptor{nullptr};
    MTL::PrimitiveAccelerationStructureDescriptor *_compatibility_descriptor{nullptr};
    MTL::Buffer *_aabb_buffer{nullptr};
    size_t _aabb_buffer_offset{0u};
    size_t _aabb_count{0u};

private:
    void _do_add_resources(luisa::vector<MTL::Resource *> &resources) const noexcept override;

public:
    MetalProceduralPrimitive(MTL::Device *device, const AccelOption &option) noexcept;
    ~MetalProceduralPrimitive() noexcept override;
    void build(MetalCommandEncoder &encoder, ProceduralPrimitiveBuildCommand *command) noexcept;
};

}// namespace luisa::compute::metal

//
// Created by Mike Smith on 2023/4/20.
//

#pragma once

#include <runtime/rhi/command.h>
#include <backends/metal/metal_primitive.h>

namespace luisa::compute::metal {

class MetalCommandEncoder;

class MetalMesh : public MetalPrimitive {

private:
    MTL::PrimitiveAccelerationStructureDescriptor *_descriptor{nullptr};

private:
    void _do_add_resources(luisa::vector<MTL::Resource *> &resources) const noexcept override;

public:
    MetalMesh(MTL::Device *device, const AccelOption &option) noexcept;
    ~MetalMesh() noexcept override;
    void build(MetalCommandEncoder &encoder, MeshBuildCommand *command) noexcept;
};

}// namespace luisa::compute::metal

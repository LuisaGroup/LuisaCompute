#pragma once

#include <luisa/runtime/rhi/command.h>
#include "metal_primitive.h"

namespace luisa::compute::metal {

class MetalMesh : public MetalPrimitive {

private:
    MTL4::PrimitiveAccelerationStructureDescriptor *_descriptor{nullptr};
    MTL::PrimitiveAccelerationStructureDescriptor *_compatibility_descriptor{nullptr};
    MTL::Buffer *_vertex_buffer{nullptr};
    MTL::Buffer *_triangle_buffer{nullptr};
    size_t _vertex_buffer_offset{0u};
    size_t _vertex_buffer_size{0u};
    size_t _vertex_stride{0u};
    size_t _triangle_buffer_offset{0u};
    size_t _triangle_buffer_size{0u};

private:
    void _do_add_resources(luisa::vector<MTL::Resource *> &resources) const noexcept override;

public:
    MetalMesh(MTL::Device *device, const AccelOption &option) noexcept;
    ~MetalMesh() noexcept override;
    void build(MetalCommandEncoder &encoder, MeshBuildCommand *command) noexcept;
};

}// namespace luisa::compute::metal

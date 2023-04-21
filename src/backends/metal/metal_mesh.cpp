//
// Created by Mike Smith on 2023/4/20.
//

#include <backends/metal/metal_mesh.h>

namespace luisa::compute::metal {

MetalMesh::MetalMesh(MTL::Device *device, const AccelOption &option) noexcept
    : MetalPrimitive{device, option} {

}

MetalMesh::~MetalMesh() noexcept {

}

void MetalMesh::_do_add_resources(luisa::vector<MTL::Resource *> &resources) const noexcept {

}

void MetalMesh::build(MetalCommandEncoder &encoder, MeshBuildCommand *command) noexcept {
}

}// namespace luisa::compute::metal

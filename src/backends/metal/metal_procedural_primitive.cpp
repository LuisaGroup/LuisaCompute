//
// Created by Mike Smith on 2023/4/20.
//

#include <backends/metal/metal_procedural_primitive.h>

namespace luisa::compute::metal {

MetalProceduralPrimitive::MetalProceduralPrimitive(MTL::Device *device,
                                                   const AccelOption &option) noexcept
    : MetalPrimitive{device, option} {}

MetalProceduralPrimitive::~MetalProceduralPrimitive() noexcept {
}

void MetalProceduralPrimitive::_do_add_resources(luisa::vector<MTL::Resource *> &resources) const noexcept {
}

void MetalProceduralPrimitive::build(MetalCommandEncoder &encoder,
                                     ProceduralPrimitiveBuildCommand *command) noexcept {
}

}// namespace luisa::compute::metal

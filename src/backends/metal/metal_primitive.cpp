//
// Created by Mike Smith on 2023/4/20.
//

#include <backends/metal/metal_primitive.h>

namespace luisa::compute::metal {

MetalPrimitive::MetalPrimitive(MTL::Device *device, const AccelOption &option) noexcept {
}

MetalPrimitive::~MetalPrimitive() noexcept {
}

void MetalPrimitive::add_resources(luisa::vector<MTL::Resource *> &resources) const noexcept {
    resources.emplace_back(_handle);
    _do_add_resources(resources);
}

}// namespace luisa::compute::metal

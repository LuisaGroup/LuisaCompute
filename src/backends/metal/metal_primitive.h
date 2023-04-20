//
// Created by Mike Smith on 2023/4/20.
//

#pragma once

#include <core/stl/vector.h>
#include <backends/metal/metal_api.h>

namespace luisa::compute::metal {

class MetalPrimitive {

private:
    MTL::AccelerationStructure *_handle{nullptr};

public:
    virtual ~MetalPrimitive() noexcept {}
    [[nodiscard]] auto handle() const noexcept { return _handle; }
    void add_resources(luisa::vector<MTL::Resource *> &resources) const noexcept {

    }
};

}// namespace luisa::compute::metal

//
// Created by Mike Smith on 2023/4/15.
//

#pragma once

#include <backends/metal/metal_api.h>

namespace luisa::compute::metal {

class MetalDevice;

class MetalBindlessArray {

public:
    MetalBindlessArray(MetalDevice *device, size_t size) noexcept;
    ~MetalBindlessArray() noexcept;
};

}// namespace luisa::compute::metal

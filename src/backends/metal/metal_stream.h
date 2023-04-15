//
// Created by Mike Smith on 2023/4/15.
//

#pragma once

#include <backends/metal/metal_api.h>

namespace luisa::compute::metal {

class MetalEvent;

class MetalStream {

private:
    MTL::CommandQueue *_queue;

public:
    void signal(MetalEvent *event) noexcept;
    void wait(MetalEvent *event) noexcept;
};

}// namespace luisa::compute::metal
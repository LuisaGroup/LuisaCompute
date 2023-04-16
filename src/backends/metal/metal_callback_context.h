//
// Created by Mike Smith on 2023/4/16.
//

#pragma once

namespace luisa::compute::metal {

struct MetalCallbackContext {
    virtual void recycle() noexcept = 0;
    virtual ~MetalCallbackContext() noexcept = default;
};

}// namespace luisa::compute::metal

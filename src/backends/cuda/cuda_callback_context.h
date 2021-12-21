//
// Created by Mike on 2021/12/11.
//

#pragma once

namespace luisa::compute::cuda {

struct CUDACallbackContext {
    virtual void recycle() noexcept = 0;
    virtual ~CUDACallbackContext() noexcept = default;
};

}

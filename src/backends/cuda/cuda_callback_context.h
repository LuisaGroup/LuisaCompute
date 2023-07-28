#pragma once

namespace luisa::compute::cuda {

struct CUDACallbackContext {
    virtual void recycle() noexcept = 0;
    virtual ~CUDACallbackContext() noexcept = default;
};

}// namespace luisa::compute::cuda


//
// Created by Mike on 8/1/2021.
//

#pragma once

#include <mutex>

#include <cuda.h>

#include <core/stl.h>
#include <core/spin_mutex.h>
#include <backends/cuda/cuda_callback_context.h>
#include <backends/cuda/cuda_ring_buffer.h>

namespace luisa::compute::cuda {

class CUDACallbackContext;

class CUDAStream {

private:
    CUstream _handle;
    CUDARingBuffer _upload_pool;
    luisa::queue<CUDACallbackContext *> _callbacks;
    spin_mutex _mutex;

public:
    CUDAStream() noexcept;
    ~CUDAStream() noexcept;
    [[nodiscard]] auto handle() const noexcept { return _handle; }
    [[nodiscard]] auto upload_pool() noexcept { return &_upload_pool; }
    void emplace_callback(CUDACallbackContext *cb) noexcept;
    void dispatch_callbacks() noexcept;
};

}

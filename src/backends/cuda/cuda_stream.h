//
// Created by Mike on 8/1/2021.
//

#pragma once

#include <mutex>

#include <cuda.h>

#include <core/stl.h>
#include <core/spin_mutex.h>
#include <backends/cuda/cuda_callback_context.h>
#include <backends/cuda/cuda_host_buffer_pool.h>

namespace luisa::compute::cuda {

class CUDACallbackContext;

/**
 * @brief Stream on CUDA
 * 
 */
class CUDAStream {

    // TODO: parallel compute/copy using multiple streams

private:
    CUstream _handle;
    CUDAHostBufferPool _upload_pool;
    luisa::queue<luisa::vector<CUDACallbackContext *>> _callback_lists;
    luisa::vector<CUDACallbackContext *> _current_callbacks;
    std::mutex _mutex;

public:
    CUDAStream() noexcept;
    ~CUDAStream() noexcept;
    /**
     * @brief Return handle of stream on CUDA
     * 
     * @return CUstream
     */
    [[nodiscard]] auto handle() const noexcept { return _handle; }
    /**
     * @brief Return CUDAHostBufferPool
     * 
     * @return address of CUDAHostBufferPool
     */
    [[nodiscard]] auto upload_pool() noexcept { return &_upload_pool; }
    /**
     * @brief Emplace callback context
     * 
     * @param cb callback context
     */
    void emplace_callback(CUDACallbackContext *cb) noexcept;
    /**
     * @brief Dispatch callbacks
     * 
     */
    void dispatch_callbacks() noexcept;
};

}

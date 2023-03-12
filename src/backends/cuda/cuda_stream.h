//
// Created by Mike on 8/1/2021.
//

#pragma once

#include <mutex>

#include <cuda.h>

#include <core/stl.h>
#include <core/spin_mutex.h>
#include <runtime/command_list.h>
#include <backends/cuda/cuda_callback_context.h>
#include <backends/cuda/cuda_host_buffer_pool.h>

namespace luisa::compute::cuda {

class CUDADevice;
class CUDACallbackContext;

/**
 * @brief Stream on CUDA
 * 
 */
class CUDAStream {

public:
    using CallbackContainer = luisa::vector<CUDACallbackContext *>;

private:
    CUDADevice *_device;
    CUDAHostBufferPool _upload_pool;
    std::mutex _mutex;
    luisa::queue<CallbackContainer> _callback_lists;
    CUstream _stream{};

public:
    explicit CUDAStream(CUDADevice *device) noexcept;
    ~CUDAStream() noexcept;

    [[nodiscard]] auto device() const noexcept { return _device; }
    [[nodiscard]] auto handle() const noexcept { return _stream; }
    [[nodiscard]] auto upload_pool() noexcept { return &_upload_pool; }
    void synchronize() noexcept;
    void signal(CUevent event) noexcept;
    void wait(CUevent event) noexcept;
    void callback(CallbackContainer &&callbacks) noexcept;
};

}// namespace luisa::compute::cuda

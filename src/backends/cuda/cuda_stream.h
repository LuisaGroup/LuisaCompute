//
// Created by Mike on 8/1/2021.
//

#pragma once

#include <mutex>
#include <condition_variable>

#include <cuda.h>

#include <luisa/core/stl.h>
#include <luisa/core/spin_mutex.h>
#include <luisa/runtime/command_list.h>
#include "cuda_callback_context.h"
#include "cuda_host_buffer_pool.h"

namespace luisa::compute::cuda {

class CUDADevice;
class CUDACallbackContext;

class CUDAStream;

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
    CUDAHostBufferPool _download_pool;
    luisa::queue<CallbackContainer> _callback_lists;
    CUstream _stream{};
    spin_mutex _dispatch_mutex;
    spin_mutex _callback_mutex;

public:
    explicit CUDAStream(CUDADevice *device) noexcept;
    virtual ~CUDAStream() noexcept;
    [[nodiscard]] auto device() const noexcept { return _device; }
    [[nodiscard]] auto handle() const noexcept { return _stream; }
    [[nodiscard]] auto upload_pool() noexcept { return &_upload_pool; }
    [[nodiscard]] auto download_pool() noexcept { return &_download_pool; }
    void dispatch(CommandList &&command_list) noexcept;
    void synchronize() noexcept;
    void signal(CUevent event) noexcept;
    void wait(CUevent event) noexcept;
    void callback(CallbackContainer &&callbacks) noexcept;
    void set_name(luisa::string &&name) noexcept;
};

}// namespace luisa::compute::cuda


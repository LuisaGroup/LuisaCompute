//
// Created by Mike on 8/1/2021.
//

#pragma once

#include <mutex>
#include <condition_variable>

#include <cuda.h>

#include <core/stl.h>
#include <core/spin_mutex.h>
#include <runtime/command_list.h>
#include <backends/cuda/cuda_callback_context.h>
#include <backends/cuda/cuda_host_buffer_pool.h>

namespace luisa::compute::cuda {

class CUDADevice;
class CUDACallbackContext;

class CUDAStream;
class CUDAIndirectDispatch;

class CUDAIndirectDispatchStream {

public:
    // use the nullptr as a stop token
    using Task = CUDAIndirectDispatch;
    static constexpr auto stop_token = static_cast<Task *>(nullptr);

private:
    CUDAStream *_parent;
    CUstream _stream;
    CUevent _event_to_wait;
    CUevent _event_to_signal;
    std::mutex _mutex;
    std::thread _thread;
    std::condition_variable _cv;
    luisa::queue<Task *> _tasks;

public:
    explicit CUDAIndirectDispatchStream(CUDAStream *parent) noexcept;
    ~CUDAIndirectDispatchStream() noexcept;
    void enqueue(ShaderDispatchCommand *command) noexcept;
    void stop() noexcept;
};

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
    luisa::queue<CallbackContainer> _callback_lists;
    luisa::unique_ptr<CUDAIndirectDispatchStream> _indirect_dispatch_thread;
    CUstream _stream{};
    uint _uid;
    spin_mutex _dispatch_mutex;
    spin_mutex _callback_mutex;
    spin_mutex _indirect_thread_creation_mutex;

public:
    explicit CUDAStream(CUDADevice *device) noexcept;
    virtual ~CUDAStream() noexcept;
    [[nodiscard]] auto device() const noexcept { return _device; }
    [[nodiscard]] auto handle() const noexcept { return _stream; }
    [[nodiscard]] auto upload_pool() noexcept { return &_upload_pool; }
    void dispatch(CommandList &&command_list) noexcept;
    void synchronize() noexcept;
    void signal(CUevent event) noexcept;
    void wait(CUevent event) noexcept;
    void callback(CallbackContainer &&callbacks) noexcept;
    [[nodiscard]] auto uid() const noexcept { return _uid; }
    void set_name(luisa::string &&name) noexcept;
};

}// namespace luisa::compute::cuda

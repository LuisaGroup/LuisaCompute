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

class CUDAIndirectDispatchStream {

public:
    struct Task {
        virtual void execute(CUstream stream) noexcept = 0;
        virtual ~Task() noexcept = default;
    };

    class TaskContext;

private:
    CUDAStream *_parent;
    CUstream _stream{nullptr};
    std::mutex _queue_mutex;
    std::thread _thread;
    std::condition_variable _cv;
    luisa::queue<TaskContext *> _task_contexts;
    CUdeviceptr _event{};
    uint64_t _event_value : 63;
    uint64_t _stop_requested : 1;

public:
    explicit CUDAIndirectDispatchStream(CUDAStream *parent) noexcept;
    ~CUDAIndirectDispatchStream() noexcept;
    void enqueue(Task *task) noexcept;
    void set_name(luisa::string_view name) noexcept;
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
    luisa::unique_ptr<CUDAIndirectDispatchStream> _indirect;
    CUstream _stream{};
    luisa::string _name;
    spin_mutex _name_mutex;
    spin_mutex _dispatch_mutex;
    spin_mutex _callback_mutex;
    spin_mutex _indirect_creation_mutex;

public:
    explicit CUDAStream(CUDADevice *device) noexcept;
    virtual ~CUDAStream() noexcept;
    [[nodiscard]] auto device() const noexcept { return _device; }
    [[nodiscard]] auto handle() const noexcept { return _stream; }
    [[nodiscard]] auto upload_pool() noexcept { return &_upload_pool; }
    [[nodiscard]] CUDAIndirectDispatchStream &indirect() noexcept;
    void dispatch(CommandList &&command_list) noexcept;
    void synchronize() noexcept;
    void signal(CUevent event) noexcept;
    void wait(CUevent event) noexcept;
    void callback(CallbackContainer &&callbacks) noexcept;
    void set_name(luisa::string &&name) noexcept;
};

}// namespace luisa::compute::cuda

#pragma once

#include <mutex>
#include <thread>
#include <condition_variable>

#include <cuda.h>

#include <luisa/core/stl.h>
#include <luisa/core/spin_mutex.h>
#include <luisa/runtime/command_list.h>
#include "cuda_callback_context.h"
#include "cuda_host_buffer_pool.h"

namespace luisa::compute::cuda {

class CUDADevice;
struct CUDACallbackContext;

class CUDAStream;
class CUDAEvent;

/**
 * @brief Stream on CUDA
 * 
 */
class CUDAStream {

public:
    using CallbackContainer = luisa::vector<CUDACallbackContext *>;

    static constexpr auto stop_ticket = std::numeric_limits<uint64_t>::max();

    struct CallbackPackage {
        uint64_t ticket;
        CallbackContainer callbacks;
    };

private:
    CUDADevice *_device;
    CUDAHostBufferPool _upload_pool;
    CUDAHostBufferPool _download_pool;
    std::thread _callback_thread;
    std::mutex _callback_mutex;
    std::condition_variable _callback_cv;
    CUDAEvent *_callback_event;
    std::atomic_uint64_t _current_ticket{0u};
    std::atomic_uint64_t _finished_ticket{0u};
    luisa::queue<CallbackPackage> _callback_lists;
    CUstream _stream{};
    spin_mutex _dispatch_mutex;

public:
    explicit CUDAStream(CUDADevice *device) noexcept;
    virtual ~CUDAStream() noexcept;
    [[nodiscard]] auto device() const noexcept { return _device; }
    [[nodiscard]] auto handle() const noexcept { return _stream; }
    [[nodiscard]] auto upload_pool() noexcept { return &_upload_pool; }
    [[nodiscard]] auto download_pool() noexcept { return &_download_pool; }
    void dispatch(CommandList &&command_list) noexcept;
    void synchronize() noexcept;
    void signal(CUDAEvent *event, uint64_t value) noexcept;
    void wait(CUDAEvent *event, uint64_t value) noexcept;
    void callback(CallbackContainer &&callbacks) noexcept;
    void set_name(luisa::string &&name) noexcept;
};

}// namespace luisa::compute::cuda


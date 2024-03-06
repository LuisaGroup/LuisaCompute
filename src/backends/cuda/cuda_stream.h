#pragma once

#include <mutex>
#include <thread>
#include <condition_variable>

#include <cuda.h>

#include <luisa/core/stl.h>
#include <luisa/core/spin_mutex.h>
#include <luisa/runtime/rhi/device_interface.h>
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
    volatile uint64_t *_callback_semaphore{nullptr};
    CUdeviceptr _callback_semaphore_device{0u};
    std::atomic_uint64_t _current_ticket{0u};
    std::atomic_uint64_t _finished_ticket{0u};
    luisa::queue<CallbackPackage> _callback_lists;
    CUstream _stream{};
    spin_mutex _dispatch_mutex;

private:
    using LogCallback = DeviceInterface::StreamLogCallback;
    LogCallback _log_callback;

public:
    explicit CUDAStream(CUDADevice *device) noexcept;
    virtual ~CUDAStream() noexcept;
    [[nodiscard]] auto device() const noexcept { return _device; }
    [[nodiscard]] auto handle() const noexcept { return _stream; }
    [[nodiscard]] auto upload_pool() noexcept { return &_upload_pool; }
    [[nodiscard]] auto download_pool() noexcept { return &_download_pool; }
    [[nodiscard]] auto &log_callback() const noexcept { return _log_callback; }
    void dispatch(CommandList &&command_list) noexcept;
    void synchronize() noexcept;
    void signal(CUDAEvent *event, uint64_t value) noexcept;
    void wait(CUDAEvent *event, uint64_t value) noexcept;
    void callback(CallbackContainer &&callbacks) noexcept;
    void set_name(luisa::string &&name) noexcept;
    void set_log_callback(LogCallback callback) noexcept;
};

}// namespace luisa::compute::cuda

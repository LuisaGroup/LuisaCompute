//
// Created by mike on 1/9/26.
//

#pragma once

#include <thread>
#include <atomic>
#include <mutex>
#include <condition_variable>

#include <hip/hip_runtime.h>

#include <luisa/core/stl/vector.h>
#include <luisa/core/stl/queue.h>
#include <luisa/runtime/rhi/device_interface.h>

#include "hip_stage_buffer_pool.h"

namespace luisa::compute::hip {

class HIPDevice;

class HIPStream {

public:
    using CallbackContainer = luisa::vector<HIPCallbackContext *>;
    static constexpr auto stop_ticket = std::numeric_limits<uint64_t>::max();
    struct CallbackPackage {
        uint64_t ticket;
        CallbackContainer callbacks;
    };

private:
    HIPDevice *_device;
    hipStream_t _stream{};
    HIPStageBufferPool _upload_pool;
    HIPStageBufferPool _download_pool;
    std::thread _callback_thread;
    std::mutex _callback_mutex;
    std::condition_variable _callback_cv;
    volatile uint64_t *_callback_semaphore{nullptr};
    hipDeviceptr_t _callback_semaphore_device{};
    std::atomic_uint64_t _current_ticket{0u};
    std::atomic_uint64_t _finished_ticket{0u};
    luisa::queue<CallbackPackage> _callback_lists{};
    spin_mutex _dispatch_mutex;
    bool _profiling_enabled{false};
    double _total_gpu_time_ms{0.0};
    uint64_t _dispatch_count{0u};

private:
    void _create_callback_semaphore() noexcept;
    void _destroy_callback_semaphore() noexcept;
    void _spawn_callback_thread() noexcept;
    void _shutdown_callback_thread() noexcept;

public:
    explicit HIPStream(HIPDevice *device) noexcept;
    ~HIPStream() noexcept;
    [[nodiscard]] auto device() const noexcept { return _device; }
    [[nodiscard]] auto handle() const noexcept { return _stream; }
    [[nodiscard]] auto upload_pool() noexcept { return &_upload_pool; }
    [[nodiscard]] auto download_pool() noexcept { return &_download_pool; }
    void dispatch(CommandList &&command_list) noexcept;
    void synchronize() noexcept;
    void callback(CallbackContainer &&callbacks) noexcept;
};

}// namespace luisa::compute::hip

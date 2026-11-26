//
// Created by mike on 1/9/26.
//

#pragma once

#include <thread>
#include <atomic>
#include <mutex>
#include <condition_variable>

#include <hip/hip_runtime.h>
#include <hiprt/hiprt.h>

#include <luisa/core/stl/vector.h>
#include <luisa/core/stl/queue.h>
#include <luisa/core/stl/string.h>
#include <luisa/core/stl/unordered_map.h>
#include <luisa/runtime/rhi/device_interface.h>
#include <luisa/runtime/rhi/command.h>

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

    struct PendingProfileEvent {
        hipEvent_t start{};
        hipEvent_t stop{};
        Command::Tag tag{};
        uint64_t shader_handle{0u};
        luisa::string shader_name;
    };

    struct ProfileAggregate {
        uint64_t dispatches{0u};
        double gpu_ms{0.0};
    };

    struct DeferredDownloadCopy {
        void *dst{};
        HIPStageBufferPool::View *staging{};
        size_t size{0u};
    };

private:
    HIPDevice *_device;
    hipStream_t _stream{};
    HIPStageBufferPool _upload_pool;
    HIPStageBufferPool _download_pool;
    hipDeviceptr_t _rt_scratch_buffer{};
    size_t _rt_scratch_capacity{};
    hipDeviceptr_t _rt_global_stack_buffer{};
    uint32_t _rt_global_stack_thread_capacity{};
    std::thread _callback_thread;
    std::mutex _callback_mutex;
    std::condition_variable _callback_cv;
    volatile uint64_t *_callback_semaphore{nullptr};
    hipDeviceptr_t _callback_semaphore_device{};
    std::atomic_uint64_t _current_ticket{0u};
    std::atomic_uint64_t _finished_ticket{0u};
    std::atomic_bool _hiprt_build_pending{false};
    luisa::queue<CallbackPackage> _callback_lists{};
    spin_mutex _dispatch_mutex;
    using LogCallback = DeviceInterface::StreamLogCallback;
    mutable std::mutex _log_callback_mutex;
    LogCallback _log_callback;
    bool _profiling_enabled{false};
    bool _summary_profiling_enabled{false};
    double _total_gpu_time_ms{0.0};
    double _other_gpu_time_ms{0.0};
    double _host_encode_ms{0.0};
    double _sync_wait_ms{0.0};
    uint64_t _dispatch_count{0u};
    uint64_t _commit_count{0u};
    uint64_t _sync_count{0u};
    luisa::vector<PendingProfileEvent> _pending_profile_events;
    luisa::unordered_map<luisa::string, ProfileAggregate> _shader_profile_totals;
    luisa::vector<DeferredDownloadCopy> _deferred_downloads;

private:
    void _create_callback_semaphore() noexcept;
    void _destroy_callback_semaphore() noexcept;
    void _spawn_callback_thread() noexcept;
    void _shutdown_callback_thread() noexcept;
    void _notify_hiprt_build_completed_after_synchronize() noexcept;
    void _drain_profile_events_after_sync() noexcept;
    void _drain_deferred_downloads_after_sync() noexcept;

public:
    explicit HIPStream(HIPDevice *device) noexcept;
    ~HIPStream() noexcept;
    [[nodiscard]] auto device() const noexcept { return _device; }
    [[nodiscard]] auto handle() const noexcept { return _stream; }
    [[nodiscard]] auto upload_pool() noexcept { return &_upload_pool; }
    [[nodiscard]] auto download_pool() noexcept { return &_download_pool; }
    [[nodiscard]] size_t deferred_download_count_locked() const noexcept {
        return _deferred_downloads.size();
    }
    void append_deferred_downloads_locked(
        luisa::vector<DeferredDownloadCopy> &&copies) noexcept;
    [[nodiscard]] luisa::vector<DeferredDownloadCopy>
    take_deferred_downloads_locked() noexcept;
    [[nodiscard]] static HIPCallbackContext *create_download_copy_context(
        luisa::vector<DeferredDownloadCopy> &&copies) noexcept;
    void flush_deferred_downloads() noexcept;
    // Temporary acceleration-structure builders on one stream are totally
    // ordered, so they share one allocation whose lifetime is the stream.
    [[nodiscard]] hipDeviceptr_t rt_scratch_buffer(size_t required_size) noexcept;
    // Static HIPRT stacks are indexed by the physical launch thread. Keeping
    // one grow-only allocation per stream makes reuse stream-ordered while
    // preventing cross-stream aliasing.
    [[nodiscard]] hiprtGlobalStackBuffer
    rt_global_stack_buffer(size_t required_thread_count) noexcept;
    void dispatch(CommandList &&command_list) noexcept;
    void synchronize() noexcept;
    void mark_hiprt_build_submitted() noexcept {
        // Completion is published only after this exact stream is drained.
        _hiprt_build_pending.store(true, std::memory_order_release);
    }
    void callback(CallbackContainer &&callbacks) noexcept;
    [[nodiscard]] LogCallback log_callback() const noexcept;
    void set_log_callback(LogCallback callback) noexcept;
};

}// namespace luisa::compute::hip

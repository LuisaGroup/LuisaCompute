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
#include "cuda_pdl_launch.h"

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
    std::atomic_uint64_t _callback_semaphore{0u};
    std::atomic_uint64_t _current_ticket{0u};
    std::atomic_uint64_t _finished_ticket{0u};
    luisa::queue<CallbackPackage> _callback_lists;
    CUstream _stream{};
    spin_mutex _dispatch_mutex;

    // Programmatic-dependent-launch state of this stream (see
    // cuda_pdl_launch.h). Read when a submission starts and written after every
    // command of a submission; also written by the host-side stream operations
    // below, which other threads may call, hence the atomic.
    std::atomic<cuda_pdl_detail::PdlPredecessor> _pdl_predecessor{
        cuda_pdl_detail::PdlPredecessor::unknown};

    // Staging blocks (upload/download pool views) go back to their pool through a
    // CUevent recorded at the end of the submission that used them, not through a
    // cuLaunchHostFunc + callback-thread round trip: the fence is queried by the
    // next dispatch()/synchronize().
    struct StagingBatch {
        CUevent event{};
        luisa::vector<CUDACallbackContext *> contexts;
    };
    spin_mutex _staging_mutex;
    luisa::vector<StagingBatch> _pending_staging;
    luisa::vector<CUevent> _free_events;
    void _reclaim_staging(bool all) noexcept;

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
    // Whether the stream slot the next launch enters is a kernel launch.
    [[nodiscard]] bool pdl_predecessor_is_kernel() const noexcept {
        return _pdl_predecessor.load(std::memory_order_acquire) ==
               cuda_pdl_detail::PdlPredecessor::kernel_launch;
    }
    void note_pdl_predecessor(bool is_kernel) noexcept {
        _pdl_predecessor.store(
            is_kernel ? cuda_pdl_detail::PdlPredecessor::kernel_launch
                      : cuda_pdl_detail::PdlPredecessor::non_kernel,
            std::memory_order_release);
    }
    // The stream no longer holds a known predecessor (a drain happened).
    void clear_pdl_predecessor() noexcept {
        _pdl_predecessor.store(cuda_pdl_detail::PdlPredecessor::unknown,
                               std::memory_order_release);
    }
    void dispatch(CommandList &&command_list) noexcept;
    void synchronize() noexcept;
    void signal(CUDAEvent *event, uint64_t value) noexcept;
    void wait(CUDAEvent *event, uint64_t value) noexcept;
    void callback(CallbackContainer &&callbacks) noexcept;
    /// Return staging pool views to the pool once the stream has passed this
    /// submission, without a host callback (records one CUevent).
    void defer_recycle(luisa::vector<CUDACallbackContext *> &&contexts) noexcept;
    void set_name(luisa::string &&name) noexcept;
    void set_log_callback(LogCallback callback) noexcept;
};

}// namespace luisa::compute::cuda

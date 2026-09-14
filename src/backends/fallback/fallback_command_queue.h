#pragma once

#include <atomic>
#include <mutex>
#include <thread>
#include <condition_variable>

#include <luisa/core/stl/queue.h>
#include <luisa/core/stl/functional.h>
#include <luisa/runtime/rhi/device_interface.h>

#ifdef LUISA_COMPUTE_USE_SYSTEM_PARALLEL_FOR

#if defined(LUISA_PLATFORM_APPLE)
#define LUISA_FALLBACK_USE_DISPATCH_QUEUE
#include <dispatch/dispatch.h>
#elif defined(LUISA_PLATFORM_WINDOWS)
#define LUISA_FALLBACK_USE_PPL
#include <ppl.h>
#elif defined(LUISA_COMPUTE_ENABLE_LIBDISPATCH)
#define LUISA_FALLBACK_USE_DISPATCH_QUEUE
#include <dispatch/dispatch.h>
#elif defined(LUISA_COMPUTE_ENABLE_TBB)
#define LUISA_FALLBACK_USE_TBB
#include <tbb/parallel_for.h>
#else
#define LUISA_FALLBACK_USE_AKARI_THREAD_POOL
#endif

#else
#define LUISA_FALLBACK_USE_AKARI_THREAD_POOL
#endif

namespace luisa::compute::fallback {

class AkrThreadPool;

class FallbackCommandQueue {

private:
    std::mutex _mutex;
    std::condition_variable _cv;
    std::mutex _synchronize_mutex;
    std::condition_variable _synchronize_cv;
    std::thread _dispatcher;
    luisa::queue<luisa::move_only_function<void()>> _tasks;
    size_t _in_flight_limit{0u};
    std::atomic_size_t _total_enqueue_count{0u};
    std::atomic_size_t _total_finish_count{0u};
    size_t _worker_count{0u};
    DeviceInterface::StreamLogCallback _log_callback;

#if defined(LUISA_FALLBACK_USE_DISPATCH_QUEUE)
    dispatch_queue_t _dispatch_queue{nullptr};
#elif defined(LUISA_FALLBACK_USE_AKARI_THREAD_POOL)
    luisa::unique_ptr<AkrThreadPool> _worker_pool;
#endif

private:
    void _run_dispatch_loop() noexcept;
    void _wait_for_task_queue_available() const noexcept;
    void _enqueue_task_no_wait(luisa::move_only_function<void()> &&task) noexcept;

public:
    explicit FallbackCommandQueue(size_t in_flight_limit, size_t num_threads) noexcept;
    ~FallbackCommandQueue() noexcept;
    void enqueue(luisa::move_only_function<void()> &&task) noexcept;
    void enqueue_parallel(uint n, luisa::move_only_function<void(uint)> &&task) noexcept;
    void synchronize() noexcept;

    void set_log_callback(DeviceInterface::StreamLogCallback callback) noexcept { _log_callback = std::move(callback); }
    [[nodiscard]] auto &log_callback() const noexcept { return _log_callback; }
};

}// namespace luisa::compute::fallback

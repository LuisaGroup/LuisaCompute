// Device-owned, lazily created bounded worker pool for ordered batch shader
// compilation (plan P5.1).
//
// The pool owns threads and per-worker compiler state; it owns no Vulkan object
// and creates none. A batch runs in two phases (see shader_prepare.h):
//
//   1. worker threads call `Device::_prepare_compute_shader()` for the job indices
//      they claim (dynamic scheduling, so a slow kernel cannot stall the queue)
//      and mark that job's ticket;
//   2. the calling thread walks the batch in index order, waits for ticket `i`,
//      calls `Device::_publish_compute_shader()` for it, and only then moves to
//      `i + 1`.
//
// Consequences that the tests rely on:
//  * publication order is the request order regardless of completion order, so a
//    batch creates the same shaders in the same order as a serial loop;
//  * the publishing thread never blocks a worker: workers only ever prepare, so a
//    head-of-line kernel cannot starve the rest of the batch - it only delays the
//    first publication, which is what an ordered batch means;
//  * every prepare path (including "this request is a serial fallback") marks its
//    ticket through an RAII guard, so no early return can hang the publisher;
//  * `stop()` drains the active batch before the workers exit, so a shutdown that
//    races a batch cannot leave a ticket unmarked;
//  * a batch submitted from a worker thread (nested compilation) is compiled
//    serially on that thread instead of waiting for a worker - the pool never
//    waits for itself.
#pragma once
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <thread>
#include <luisa/core/stl/memory.h>
#include <luisa/core/stl/vector.h>
#include <luisa/runtime/rhi/resource.h>
#include <luisa/backends/ext/vk_shader_compile_ext.h>
#include "shader_prepare.h"
namespace lc::vk {
class Device;
namespace detail {
class ShaderCompilePool {
public:
    // Counters are plain atomics: they never feed a decision.
    struct Counters {
        std::atomic<uint64_t> batches{0u};
        std::atomic<uint64_t> requests{0u};
        std::atomic<uint64_t> parallel_prepares{0u};
        std::atomic<uint64_t> serial_prepares{0u};
        std::atomic<uint64_t> cache_hits{0u};
        std::atomic<uint64_t> serial_fallbacks{0u};
        std::atomic<uint64_t> publish_order_waits{0u};
        std::atomic<uint64_t> publish_order_wait_ns{0u};
    };
private:
    struct Job {
        luisa::compute::ShaderOption option;
        luisa::compute::Function kernel;
        PreparedComputeShader prepared;
        // Guarded by the pool mutex; the publisher waits on it through `_cv`.
        bool prepared_done{false};
    };
    Device *_device;
    // Serializes whole batches (a batch is a bulk operation). `create_shader`
    // never takes this lock, so ordinary compilation is not serialized by it.
    std::mutex _batch_mtx;
    std::mutex _mtx;
    std::condition_variable _cv;
    // --- worker side state, guarded by `_mtx` ---
    luisa::vector<Job> *_jobs{nullptr};
    size_t _job_count{0u};
    std::atomic<size_t> _next_job{0u};
    bool _shutdown{false};
    std::atomic<bool> _stopped{false};
    // One context per worker thread, created on the thread that owns it.
    luisa::vector<luisa::unique_ptr<WorkerContext>> _contexts;
    luisa::vector<std::thread> _threads;
    std::atomic<uint32_t> _worker_count{0u};
    Counters _counters;
    // Lazily create (or resize) the worker set. Returns false when the pool has
    // been stopped for good. Called with `_batch_mtx` held.
    [[nodiscard]] bool _ensure_workers(uint32_t count) noexcept;
    void _stop_workers_locked() noexcept;
    // The worker main loop; `context` lives in `_contexts` and outlives the thread.
    void _worker_main(WorkerContext *context) noexcept;
    void _run_batch(luisa::span<const luisa::compute::VulkanShaderCompileRequest> requests,
                    luisa::vector<luisa::compute::VulkanShaderCompileResult> &results) noexcept;
    void _run_batch_serial(luisa::span<const luisa::compute::VulkanShaderCompileRequest> requests,
                           luisa::vector<luisa::compute::VulkanShaderCompileResult> &results) noexcept;
public:
    static constexpr uint32_t max_workers = 32u;
    // Marks one job's ticket. Public because the RAII guard in the worker loop
    // calls it; it is not part of the caller-facing contract.
    void complete_ticket(size_t index) noexcept;
    explicit ShaderCompilePool(Device *device) noexcept;
    ~ShaderCompilePool() noexcept;
    ShaderCompilePool(const ShaderCompilePool &) = delete;
    ShaderCompilePool(ShaderCompilePool &&) = delete;
    ShaderCompilePool &operator=(const ShaderCompilePool &) = delete;
    ShaderCompilePool &operator=(ShaderCompilePool &&) = delete;
    // Default worker count for a device: one fewer than the hardware concurrency
    // (the calling thread publishes), at least 1, at most `max_workers`.
    [[nodiscard]] static uint32_t default_worker_count() noexcept;
    // Prepare + publish one batch in request order. `worker_count == 1` uses the
    // same machinery with a single worker; `0` means `default_worker_count()`.
    // Never throws; returns the batch-level status, always with `results` sized to
    // `requests.size()`.
    [[nodiscard]] luisa::compute::VulkanShaderCompileStatus compile(
        luisa::span<const luisa::compute::VulkanShaderCompileRequest> requests,
        luisa::vector<luisa::compute::VulkanShaderCompileResult> &results,
        uint32_t worker_count) noexcept;
    // Stop and join every worker. Idempotent, and final: later batches report
    // `POOL_SHUTDOWN`. An active batch is drained first (see the header comment).
    void stop() noexcept;
    // True when the calling thread is one of this pool's workers.
    [[nodiscard]] static bool on_worker_thread() noexcept;
    [[nodiscard]] luisa::compute::VulkanShaderCompileStats stats() const noexcept;
};
}// namespace detail
}// namespace lc::vk
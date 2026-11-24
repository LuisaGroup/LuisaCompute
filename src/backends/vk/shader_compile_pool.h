// Device-owned, lazily started bounded worker pool for ordered batch shader
// compilation (plan P5.1).
//
// The pool owns the parallel preparation capacity and per-worker compiler state; it
// owns no Vulkan object and creates none. A batch runs in two phases (see
// shader_prepare.h):
//
//   1. preparation lanes call `Device::_prepare_compute_shader()` for the job indices
//      they claim (dynamic scheduling, so a slow kernel cannot stall the queue) and
//      then signal that job's ticket;
//   2. the calling thread walks the batch in index order, waits for ticket `i`, calls
//      `Device::_publish_compute_shader()` for it, and only then moves to `i + 1`.
//
// Consequences that the tests rely on:
//  * publication order is the request order regardless of completion order, so a
//    batch creates the same shaders in the same order as a serial loop;
//  * the publishing thread never blocks a lane: lanes only ever prepare, so a
//    head-of-line kernel cannot starve the rest of the batch - it only delays the
//    first publication, which is what an ordered batch means;
//  * every prepare path (including "this request is a serial fallback") signals its
//    ticket through the last statement of the lane, so no early return can hang the
//    publisher;
//  * `stop()` drains the active batch before the workers exit, so a shutdown that
//    races a batch cannot leave a ticket unmarked;
//  * a batch submitted from a preparation lane (nested compilation) is compiled
//    serially on that lane instead of waiting for a lane - the pool never waits for
//    itself. The nested test runs *before* the batch gate, because a nested call is
//    made while the batch that owns the gate is running.
//
// Executor selection
// ------------------
// Which of the two executors a pool drives is decided **once**, in the constructor,
// from the `marl::Scheduler` the *Device constructor* captured on its own thread
// (`marl::Scheduler::get()` there):
//
//  * `Executor::fiber` - a scheduler was bound on the device constructor thread.
//    With worker threads, preparation runs as marl tasks (fibers) on that existing
//    scheduler: the pool creates **no** thread, and a publisher that itself runs on a
//    scheduler fiber waits through marl events/counters, so it yields its fiber
//    instead of blocking an OS worker with a `std::condition_variable`. A scheduler
//    configured with zero workers has no parallel capacity, so batches use the
//    unchanged serial path on the caller rather than creating fallback threads.
//  * `Executor::std_thread` - no such scheduler was captured. The pool owns
//    `std::thread` workers, exactly as before.
//
// Detecting the executor at first compile, or at destruction, would be wrong: both
// may run on a different thread than the one that constructed the device, and a
// scheduler is bound per thread.
//
// **Lifetime of a captured scheduler**: it is *external*. It must outlive the device
// and every batch (i.e. be destroyed only after the device is gone), because the pool
// enqueues the next batch's lanes on it directly. The pool does not and cannot detect
// its destruction.
//
// Lane identity
// -------------
// "Is the calling flow one of this pool's lanes?" is answered with
// `luisa::fiber::get_id()` - the fiber identity on a marl worker, which stays stable
// when a fiber migrates between worker threads, and the thread identity for a plain
// thread. `thread_local` is deliberately not used: a fiber may migrate, so a
// thread-local marker would either miss or falsely report a lane.
#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <thread>

// Vulkan can include Xlib first, whose None macro collides with marl::Task::Flags::None.
// Hide it for both the direct and transitive marl includes, then restore it.
#pragma push_macro("None")
#undef None
#include <marl/scheduler.h>
#include <luisa/core/fiber.h>
#pragma pop_macro("None")

#include <luisa/backends/ext/vk_shader_compile_ext.h>
#include <luisa/core/stl/memory.h>
#include <luisa/core/stl/vector.h>
#include <luisa/runtime/rhi/resource.h>

#include "shader_prepare.h"

namespace lc::vk {
class Device;
namespace detail {
class ShaderCompilePool {
public:
    // Upper bound on parallel preparation lanes, whatever the executor. Also the
    // size of the lane table below, which is why it is declared first.
    static constexpr uint32_t max_workers = 32u;

    // Which executor this pool drives. See the file comment: it is selected in the
    // constructor from the scheduler the device captured on its constructing thread,
    // and never re-selected.
    enum class Executor : uint8_t {
        // No marl scheduler was captured: the pool owns `std::thread` workers.
        std_thread,
        // A marl scheduler was captured: preparation runs as marl tasks on it when
        // workers exist, and the pool never creates a thread of its own.
        fiber,
    };

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
        // Signalled by the lane that finished `prepared` and awaited by the publisher
        // for exactly this index. A marl event, not a `std::condition_variable`: a
        // publisher running on a scheduler fiber then suspends that fiber instead of
        // blocking the OS worker thread it happens to be on.
        luisa::fiber::event prepared_done;
    };

    Device *_device;
    // Decided in the constructor; immutable afterwards.
    Executor _executor{Executor::std_thread};
    // The scheduler captured on the device constructor thread (fiber executor only,
    // null otherwise). It is external and must outlive the device and its batches.
    marl::Scheduler *_scheduler{nullptr};

    // Serializes whole batches (a batch is a bulk operation). A fiber-aware gate, not
    // a `std::mutex`: a batch may be submitted from a scheduler fiber, and parking
    // that fiber's OS worker here would stall a batch whose lanes are queued on that
    // very worker. `create_shader` never takes this gate, so ordinary compilation is
    // not serialized by it.
    //
    // (`marl::mutex` alone is only a `std::mutex` wrapper and would park the worker
    // thread; `marl::ConditionVariable` is the fiber-aware primitive: it suspends the
    // calling fiber and hands its worker back to the scheduler.)
    class BatchGate {
    private:
        luisa::fiber::mutex _mtx;
        luisa::fiber::condition_variable _cv;
        bool _busy{false};

    public:
        BatchGate() noexcept = default;
        BatchGate(const BatchGate &) = delete;
        BatchGate &operator=(const BatchGate &) = delete;
        void acquire() noexcept {
            luisa::fiber::lock lock{_mtx};
            _cv.wait(lock, [this] { return !_busy; });
            _busy = true;
        }
        void release() noexcept {
            luisa::fiber::lock lock{_mtx};
            _busy = false;
            _cv.notify_one();
        }
    };
    // RAII hold of `BatchGate`, so no return path can leave a batch unfinished.
    class BatchHold {
    private:
        BatchGate &_gate;

    public:
        explicit BatchHold(BatchGate &gate) noexcept : _gate{gate} { _gate.acquire(); }
        BatchHold(const BatchHold &) = delete;
        BatchHold &operator=(const BatchHold &) = delete;
        ~BatchHold() noexcept { _gate.release(); }
    };

    // The one-and-only batch gate of this pool.
    BatchGate _batch;

    // --- std::thread executor: worker side state, guarded by `_mtx` ---
    // Only a sleep/wake channel between the publisher and the workers: the
    // per-job tickets are marl events, and a worker never publishes.
    std::mutex _mtx;
    std::condition_variable _cv;
    luisa::vector<Job> *_jobs{nullptr};
    size_t _job_count{0u};
    std::atomic<size_t> _next_job{0u};
    bool _shutdown{false};
    // One context per worker thread, created on the thread that owns it.
    luisa::vector<luisa::unique_ptr<WorkerContext>> _contexts;
    luisa::vector<std::thread> _threads;

    // --- shared ---
    // Identity of the flow that currently runs each preparation lane, or `kFreeLane`
    // when the lane is idle. A `std::thread` worker holds its lane for its whole
    // life; a fiber lane holds one for the duration of one marl task. This is what
    // makes nested compilation answerable without `thread_local`.
    std::atomic<uint64_t> _lane_flows[max_workers];
    std::atomic<uint32_t> _worker_count{0u};
    std::atomic<bool> _stopped{false};
    Counters _counters;

    // True when the calling flow is one of this pool's preparation lanes.
    [[nodiscard]] bool _on_lane() const noexcept;
    void _lane_enter(uint32_t lane) noexcept;
    void _lane_leave(uint32_t lane) noexcept;

    // `worker_count` to use when the caller passes 0.
    [[nodiscard]] uint32_t _default_worker_count() const noexcept;
    // Prepare the pool's executor for a batch of at most `count` lanes. Returns false
    // when the pool has been stopped for good. Called with the batch gate held.
    [[nodiscard]] bool _ensure_executor(uint32_t count) noexcept;
    // `std::thread` executor only: create (or resize) the worker set.
    [[nodiscard]] bool _ensure_thread_workers(uint32_t count) noexcept;
    void _stop_workers_locked() noexcept;

    // The worker main loop of the `std::thread` executor; `context` lives in
    // `_contexts` and outlives the thread.
    void _worker_main(WorkerContext *context) noexcept;
    // The body of one job, shared by both executors. Signals `job.prepared_done` as
    // its last touch of `job`.
    void _prepare_job(Job &job, WorkerContext *context) noexcept;
    // Copies `option`/`kernel` out of the caller's requests: they are not consumed and
    // publication happens later.
    [[nodiscard]] luisa::vector<Job> _build_jobs(
        luisa::span<const luisa::compute::VulkanShaderCompileRequest> requests) noexcept;

    // Both executors: publish every job in request order on the calling thread.
    void _publish_in_order(luisa::vector<Job> &jobs,
                           luisa::vector<luisa::compute::VulkanShaderCompileResult> &results) noexcept;
    void _run_batch_threaded(luisa::span<const luisa::compute::VulkanShaderCompileRequest> requests,
                             luisa::vector<luisa::compute::VulkanShaderCompileResult> &results) noexcept;
    void _run_batch_fiber(luisa::span<const luisa::compute::VulkanShaderCompileRequest> requests,
                          luisa::vector<luisa::compute::VulkanShaderCompileResult> &results,
                          uint32_t width) noexcept;
    void _run_batch_serial(luisa::span<const luisa::compute::VulkanShaderCompileRequest> requests,
                           luisa::vector<luisa::compute::VulkanShaderCompileResult> &results) noexcept;

public:
    // `scheduler` is the executor captured by the device constructor on its own
    // thread (`marl::Scheduler::get()` there), or null when nothing was bound. The
    // constructor decides the executor and creates nothing: workers and lanes are
    // created by the first batch.
    explicit ShaderCompilePool(Device *device, marl::Scheduler *scheduler) noexcept;
    ~ShaderCompilePool() noexcept;
    ShaderCompilePool(const ShaderCompilePool &) = delete;
    ShaderCompilePool(ShaderCompilePool &&) = delete;
    ShaderCompilePool &operator=(const ShaderCompilePool &) = delete;
    ShaderCompilePool &operator=(ShaderCompilePool &&) = delete;

    // Default worker count for a device that owns its own threads: one fewer than the
    // hardware concurrency (the calling thread publishes), at least 1, at most
    // `max_workers`.
    [[nodiscard]] static uint32_t default_worker_count() noexcept;

    // The executor this pool drives, and its name (`"fiber"` / `"std_thread"`).
    // Reported, never read back to change a decision.
    [[nodiscard]] Executor executor() const noexcept { return _executor; }
    [[nodiscard]] const char *executor_name() const noexcept;
    // Worker threads of the captured scheduler, 0 in the `std::thread` executor.
    [[nodiscard]] uint32_t captured_scheduler_worker_count() const noexcept;
    // True when the calling flow is one of this pool's preparation lanes. Answers the
    // same question the nested-compilation fallback asks, for diagnostics and tests.
    [[nodiscard]] bool in_preparation_flow() const noexcept { return _on_lane(); }

    // Prepare + publish one batch in request order. `worker_count == 1` uses the same
    // machinery with a single lane; `0` means `_default_worker_count()`.
    // Never throws; returns the batch-level status, always with `results` sized to
    // `requests.size()`.
    [[nodiscard]] luisa::compute::VulkanShaderCompileStatus compile(
        luisa::span<const luisa::compute::VulkanShaderCompileRequest> requests,
        luisa::vector<luisa::compute::VulkanShaderCompileResult> &results,
        uint32_t worker_count) noexcept;
    // Stop and join every worker. Idempotent, and final: later batches report
    // `POOL_SHUTDOWN`. An active batch is drained first (see the header comment).
    void stop() noexcept;
    [[nodiscard]] luisa::compute::VulkanShaderCompileStats stats() const noexcept;
};
}// namespace detail
}// namespace lc::vk
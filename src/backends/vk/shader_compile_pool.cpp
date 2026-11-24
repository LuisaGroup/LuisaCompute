#include "shader_compile_pool.h"

#include <algorithm>
#include <limits>

#include <marl/scheduler.h>

#include <luisa/core/clock.h>
#include <luisa/core/logging.h>

#include "device.h"

namespace lc::vk::detail {

namespace {
// A lane slot holding this value is free.
constexpr auto kFreeLane = uint64_t{0u};
// Stand-in identity for a flow whose `luisa::fiber::get_id()` is 0 (a plain thread
// whose id hash collapses to zero). It is never a fiber id - those carry the high bit
// - so it can never be confused with one, and a free slot (0) never matches it.
constexpr auto kZeroFlowId = std::numeric_limits<uint64_t>::max();

// Identity of the calling execution flow: the fiber identity on a marl worker (stable
// across a migration between worker threads) and the thread identity elsewhere. Never
// `thread_local`, and never the thread id alone: a lane is a fiber.
[[nodiscard]] uint64_t flow_id() noexcept {
    auto id = luisa::fiber::get_id();
    return id == kFreeLane ? kZeroFlowId : id;
}

// Answer a batch that cannot run at all: one result per request, no shader created.
void answer_shutdown(
    luisa::vector<luisa::compute::VulkanShaderCompileResult> &results) noexcept {
    for (auto &result : results) {
        result.status = luisa::compute::VulkanShaderCompileStatus::POOL_SHUTDOWN;
        result.message = "the shader compile pool was shut down";
        // No shader was created for this request, so the answer says so explicitly
        // instead of leaving a default-constructed handle behind.
        result.shader.invalidate();
    }
}

// Resolve one published result from the handle the backend returned.
void fill_result(luisa::compute::VulkanShaderCompileResult &result,
                 const ShaderCreationInfo &info) noexcept {
    result.shader = info;
    if (info.handle == 0u && !info.compile_ok) {
        result.status = luisa::compute::VulkanShaderCompileStatus::COMPILATION_FAILED;
        result.message = "the shader failed to compile";
    } else {
        result.status = luisa::compute::VulkanShaderCompileStatus::SUCCESS;
    }
}
}// namespace

uint32_t ShaderCompilePool::default_worker_count() noexcept {
    auto hardware = std::thread::hardware_concurrency();
    // One worker fewer than the hardware threads: the publishing thread is busy
    // with driver calls for most of the batch.
    auto count = hardware > 1u ? hardware - 1u : 1u;
    return std::min<uint32_t>(std::max<uint32_t>(count, 1u), max_workers);
}

ShaderCompilePool::ShaderCompilePool(Device *device, marl::Scheduler *scheduler) noexcept
    : _device{device} {
    for (auto &lane : _lane_flows) {
        lane.store(kFreeLane, std::memory_order_relaxed);
    }
    // Executor selection, exactly once, on the *device constructor's* thread: the
    // scheduler pointer was read there (`marl::Scheduler::get()`), so it is a fact
    // about that thread and cannot be recovered later from a batch's thread.
    if (scheduler != nullptr) {
        _executor = Executor::fiber;
        _scheduler = scheduler;
        LUISA_VERBOSE(
            "Vulkan shader compile pool: preparation uses the scheduler captured "
            "by the device constructor ({} worker thread(s)); the pool creates no "
            "thread of its own.",
            scheduler->config().workerThread.count);
    }
}

ShaderCompilePool::~ShaderCompilePool() noexcept {
    stop();
}

uint32_t ShaderCompilePool::_default_worker_count() const noexcept {
    if (_executor == Executor::fiber) {
        // The captured scheduler already owns the workers: use all of them, bounded,
        // never fewer than one.
        return std::min<uint32_t>(
            std::max<uint32_t>(captured_scheduler_worker_count(), 1u), max_workers);
    }
    return default_worker_count();
}

uint32_t ShaderCompilePool::captured_scheduler_worker_count() const noexcept {
    if (_scheduler == nullptr) { return 0u; }
    auto count = _scheduler->config().workerThread.count;
    return count <= 0 ? 0u : static_cast<uint32_t>(count);
}

const char *ShaderCompilePool::executor_name() const noexcept {
    return _executor == Executor::fiber ? "fiber" : "std_thread";
}

bool ShaderCompilePool::_on_lane() const noexcept {
    auto mine = flow_id();
    for (auto i = 0u; i < max_workers; i++) {
        if (_lane_flows[i].load(std::memory_order_acquire) == mine) { return true; }
    }
    return false;
}

void ShaderCompilePool::_lane_enter(uint32_t lane) noexcept {
    LUISA_ASSERT(lane < max_workers,
                 "Vulkan shader compile pool lane index out of range.");
    _lane_flows[lane].store(flow_id(), std::memory_order_release);
}

void ShaderCompilePool::_lane_leave(uint32_t lane) noexcept {
    LUISA_ASSERT(lane < max_workers,
                 "Vulkan shader compile pool lane index out of range.");
    _lane_flows[lane].store(kFreeLane, std::memory_order_release);
}

void ShaderCompilePool::stop() noexcept {
    if (_on_lane()) {
        // Called from inside this pool's own preparation lane (a `std::thread`
        // worker or a fiber lane). The batch gate below is held by the batch that is
        // running us, so taking it here would wait for ourselves - the same defect
        // class as the nested-compilation fallback in `compile()`. Latch the final
        // state instead: no new batch starts, and the drain/join happens in the
        // `stop()`/destructor call that runs outside the pool's flows.
        _stopped.store(true, std::memory_order_relaxed);
        return;
    }
    // The batch gate makes "stop during a batch" a wait, not a race: whichever flow
    // owns the batch finishes it (its lanes stay usable until it does), then this call
    // sets the final flag and joins.
    BatchHold batch_hold{_batch};
    _stopped.store(true, std::memory_order_relaxed);
    _stop_workers_locked();
}

void ShaderCompilePool::_stop_workers_locked() noexcept {
    if (_executor == Executor::fiber) {
        // Nothing to join: the lanes live on the external scheduler, and the batch
        // gate proves that the batch which owned them has returned, so no lane task is
        // queued or running any more.
        _worker_count.store(0u, std::memory_order_relaxed);
        return;
    }
    if (_threads.empty()) {
        // Nothing to join; make sure a later `_ensure_thread_workers` starts clean.
        std::lock_guard lock{_mtx};
        _shutdown = false;
        _worker_count.store(0u, std::memory_order_relaxed);
        return;
    }
    {
        std::lock_guard lock{_mtx};
        _shutdown = true;
    }
    _cv.notify_all();
    if (_on_lane()) {
        // Never join ourselves. The workers observe `_shutdown`, drain the active
        // batch (if any) and exit; a later `stop()`/`~ShaderCompilePool` from outside
        // the pool joins them.
        return;
    }
    for (auto &thread : _threads) {
        if (thread.joinable()) { thread.join(); }
    }
    _threads.clear();
    _contexts.clear();
    {
        std::lock_guard lock{_mtx};
        _shutdown = false;
    }
    _worker_count.store(0u, std::memory_order_relaxed);
}

bool ShaderCompilePool::_ensure_executor(uint32_t count) noexcept {
    if (_stopped.load(std::memory_order_relaxed)) { return false; }
    count = std::min<uint32_t>(std::max<uint32_t>(count, 1u), max_workers);
    if (_executor == Executor::fiber) {
        // Nothing to create: the captured scheduler owns the threads. The width is
        // only how many lane tasks one batch may run at once, and each batch enqueues
        // exactly `min(width, requests)` of them.
        _worker_count.store(count, std::memory_order_relaxed);
        return true;
    }
    return _ensure_thread_workers(count);
}

bool ShaderCompilePool::_ensure_thread_workers(uint32_t count) noexcept {
    if (_stopped.load(std::memory_order_relaxed)) { return false; }
    count = std::min<uint32_t>(std::max<uint32_t>(count, 1u), max_workers);
    if (_worker_count.load(std::memory_order_relaxed) == count &&
        _threads.size() == count) {
        return true;
    }
    // Resizing only ever happens between batches (the batch gate is held), so no
    // ticket can be outstanding here.
    _stop_workers_locked();
    if (_stopped.load(std::memory_order_relaxed)) { return false; }
    _contexts.reserve(count);
    for (auto i = 0u; i < count; i++) {
        auto context = luisa::make_unique<WorkerContext>();
        context->index = i;
        auto *raw = context.get();
        _contexts.emplace_back(std::move(context));
        _threads.emplace_back([this, raw] { _worker_main(raw); });
    }
    _worker_count.store(count, std::memory_order_relaxed);
    LUISA_VERBOSE("Vulkan shader compile pool: {} worker thread(s).", count);
    return true;
}

void ShaderCompilePool::_worker_main(WorkerContext *context) noexcept {
    // A `std::thread` worker never migrates, so it holds its lane for its whole life:
    // every flow on this thread is "inside the pool" until the thread exits.
    _lane_enter(context->index);
    auto has_unclaimed_job = [this] {
        return _jobs != nullptr &&
               _next_job.load(std::memory_order_relaxed) < _job_count;
    };
    for (;;) {
        Job *job = nullptr;
        {
            std::unique_lock lock{_mtx};
            _cv.wait(lock, [&] { return _shutdown || has_unclaimed_job(); });
            if (has_unclaimed_job()) {
                // Claiming under the lock keeps the claim and the "is there work"
                // predicate consistent, so a worker never claims index >= count.
                auto index = _next_job.fetch_add(1u, std::memory_order_relaxed);
                if (index < _job_count) { job = &(*_jobs)[index]; }
            } else if (_shutdown) {
                // Either no batch is active, or every job of the active batch has
                // been claimed (and its owner will signal its ticket). Both make it
                // safe to leave: the publisher never needs a thread that has not
                // claimed its ticket yet.
                break;
            }
        }
        if (job == nullptr) { continue; }
        _prepare_job(*job, context);
    }
    _lane_leave(context->index);
}

void ShaderCompilePool::_prepare_job(Job &job, WorkerContext *context) noexcept {
    job.prepared = _device->_prepare_compute_shader(
        job.option, job.kernel, context);
    _counters.parallel_prepares.fetch_add(1u, std::memory_order_relaxed);
    ++context->prepared;
    if (job.prepared.kind == PreparedComputeShader::Kind::SERIAL_FALLBACK) {
        _counters.serial_fallbacks.fetch_add(1u, std::memory_order_relaxed);
    } else if (job.prepared.kind == PreparedComputeShader::Kind::READY &&
               job.prepared.from_cache) {
        _counters.cache_hits.fetch_add(1u, std::memory_order_relaxed);
    }
    // Signalling the ticket is this flow's *last* touch of the job: after it the
    // publisher may consume the prepared record, and the batch frame may clear the
    // job array (see `_run_batch_threaded`), so nothing below may read `job`.
    job.prepared_done.signal();
}

luisa::compute::VulkanShaderCompileStatus ShaderCompilePool::compile(
    luisa::span<const luisa::compute::VulkanShaderCompileRequest> requests,
    luisa::vector<luisa::compute::VulkanShaderCompileResult> &results,
    uint32_t worker_count) noexcept {
    results.clear();
    results.resize(requests.size());
    if (requests.empty()) { return luisa::compute::VulkanShaderCompileStatus::INVALID_ARGUMENT; }
    // The nested test comes FIRST, before the batch gate. A nested call is made by a
    // lane of an *active* batch, i.e. while this object's batch gate is already held
    // by our own caller, so acquiring it here would deadlock before the serial
    // fallback below could run.
    if (_on_lane()) {
        if (_stopped.load(std::memory_order_relaxed)) {
            answer_shutdown(results);
            return luisa::compute::VulkanShaderCompileStatus::POOL_SHUTDOWN;
        }
        _counters.batches.fetch_add(1u, std::memory_order_relaxed);
        _counters.requests.fetch_add(requests.size(), std::memory_order_relaxed);
        // Nested compilation: waiting for a lane from a lane is a deadlock, so the
        // batch is compiled on this flow with the unchanged serial path.
        _counters.serial_prepares.fetch_add(requests.size(), std::memory_order_relaxed);
        _run_batch_serial(requests, results);
        return luisa::compute::VulkanShaderCompileStatus::FELL_BACK_TO_SERIAL;
    }
    BatchHold batch_hold{_batch};
    if (_stopped.load(std::memory_order_relaxed)) {
        answer_shutdown(results);
        return luisa::compute::VulkanShaderCompileStatus::POOL_SHUTDOWN;
    }
    _counters.batches.fetch_add(1u, std::memory_order_relaxed);
    _counters.requests.fetch_add(requests.size(), std::memory_order_relaxed);
    // A single-threaded marl scheduler has no worker on which an arbitrary caller
    // can enqueue preparation lanes. It still selects the scheduler executor (and
    // therefore never creates fallback std::threads); with no parallel capacity,
    // the honest execution mode is the unchanged serial path on the caller.
    if (_executor == Executor::fiber &&
        captured_scheduler_worker_count() == 0u) {
        _counters.serial_prepares.fetch_add(requests.size(), std::memory_order_relaxed);
        _run_batch_serial(requests, results);
        return luisa::compute::VulkanShaderCompileStatus::FELL_BACK_TO_SERIAL;
    }
    auto count = worker_count == 0u ? _default_worker_count() : worker_count;
    if (!_ensure_executor(count)) {
        _counters.serial_prepares.fetch_add(requests.size(), std::memory_order_relaxed);
        _run_batch_serial(requests, results);
        return luisa::compute::VulkanShaderCompileStatus::FELL_BACK_TO_SERIAL;
    }
    if (_executor == Executor::fiber) {
        _run_batch_fiber(requests, results,
                         _worker_count.load(std::memory_order_relaxed));
    } else {
        _run_batch_threaded(requests, results);
    }
    return luisa::compute::VulkanShaderCompileStatus::SUCCESS;
}

// Build the per-batch job list. `option` and `kernel` are copies: the caller's
// requests are not consumed, and publication happens later, possibly on a flow that
// is not the one that ran this function.
auto ShaderCompilePool::_build_jobs(
    luisa::span<const luisa::compute::VulkanShaderCompileRequest> requests) noexcept
    -> luisa::vector<Job> {
    luisa::vector<Job> jobs;
    jobs.reserve(requests.size());
    for (auto &request : requests) {
        Job job;
        job.option = request.option;
        job.kernel = request.kernel;
        jobs.emplace_back(std::move(job));
    }
    return jobs;
}

void ShaderCompilePool::_run_batch_serial(
    luisa::span<const luisa::compute::VulkanShaderCompileRequest> requests,
    luisa::vector<luisa::compute::VulkanShaderCompileResult> &results) noexcept {
    for (auto i = 0u; i < requests.size(); i++) {
        auto &request = requests[i];
        fill_result(results[i], _device->create_shader(request.option, request.kernel));
        results[i].serial_fallback = true;
    }
}

void ShaderCompilePool::_publish_in_order(
    luisa::vector<Job> &jobs,
    luisa::vector<luisa::compute::VulkanShaderCompileResult> &results) noexcept {
    // Publish in request order. The publisher is the only flow that creates Vulkan
    // objects, and it creates them for sequence number 0, 1, 2, ... - which is what
    // makes the batch order-deterministic.
    for (auto i = 0u; i < jobs.size(); i++) {
        auto &job = jobs[i];
        // A marl event, so a publisher that is itself a scheduler fiber suspends that
        // fiber here instead of blocking the OS worker thread under it. `is_signalled`
        // also keeps the diagnostic counters meaning "this index had to wait".
        if (!job.prepared_done.is_signalled()) {
            Clock clock;
            job.prepared_done.wait();
            auto ns = static_cast<uint64_t>(clock.toc() * 1.0e6);
            _counters.publish_order_waits.fetch_add(1u, std::memory_order_relaxed);
            _counters.publish_order_wait_ns.fetch_add(ns, std::memory_order_relaxed);
        }
        auto &result = results[i];
        if (job.prepared.kind == PreparedComputeShader::Kind::SERIAL_FALLBACK) {
            // The unchanged serial path, on the publishing flow: today's exact
            // behaviour for this request, including the process-global DXC use.
            fill_result(result, _device->create_shader(job.option, job.kernel));
            result.serial_fallback = true;
        } else {
            fill_result(result, _device->_publish_compute_shader(
                                    std::move(job.prepared), job.option, job.kernel));
        }
    }
}

void ShaderCompilePool::_run_batch_threaded(
    luisa::span<const luisa::compute::VulkanShaderCompileRequest> requests,
    luisa::vector<luisa::compute::VulkanShaderCompileResult> &results) noexcept {
    auto jobs = _build_jobs(requests);
    {
        std::lock_guard lock{_mtx};
        _jobs = &jobs;
        _job_count = jobs.size();
        _next_job.store(0u, std::memory_order_relaxed);
    }
    _cv.notify_all();
    _publish_in_order(jobs, results);
    // Every ticket is signalled by now, and the ticket is a worker's last touch of its
    // job, so clearing the batch here cannot pull the array out from under a worker:
    // a worker that has not claimed its index yet can no longer see it after this
    // store, and one that has claimed it already published its result.
    {
        std::lock_guard lock{_mtx};
        _jobs = nullptr;
        _job_count = 0u;
        _next_job.store(0u, std::memory_order_relaxed);
    }
}

void ShaderCompilePool::_run_batch_fiber(
    luisa::span<const luisa::compute::VulkanShaderCompileRequest> requests,
    luisa::vector<luisa::compute::VulkanShaderCompileResult> &results,
    uint32_t width) noexcept {
    LUISA_ASSERT(_scheduler != nullptr,
                 "Vulkan shader compile pool has no captured scheduler for the "
                 "fiber executor.");
    auto jobs = _build_jobs(requests);
    // One marl task per lane, never more tasks than there is work for (a one-request
    // batch must not occupy 32 fibers), at most `max_workers` because the lane table
    // is what identifies a lane.
    auto lanes = static_cast<uint32_t>(std::min<size_t>(
        std::max<uint32_t>(width, 1u), std::min<size_t>(jobs.size(), max_workers)));
    LUISA_ASSERT(lanes >= 1u && lanes <= max_workers,
                 "Vulkan shader compile pool lane count out of range.");
    // One `WorkerContext` per lane: it is the per-worker compiler state of
    // `shader_prepare.h`, and a lane is exactly one worker in this executor. The
    // contexts live in this frame, which outlives every lane task (see the counter
    // wait below).
    luisa::vector<WorkerContext> contexts(lanes);
    for (auto i = 0u; i < lanes; i++) { contexts[i].index = i; }
    // Dynamic claiming, exactly like the thread executor: a lane that finishes early
    // takes the next free index, so one slow kernel cannot stall the queue.
    std::atomic<size_t> next{0u};
    luisa::fiber::counter lanes_done{lanes};
    for (auto lane = 0u; lane < lanes; lane++) {
        auto *context = &contexts[lane];
        // Enqueued on the *captured* scheduler, not on whatever scheduler happens to
        // be bound to the batch's thread: `compile_shaders` may well be called from a
        // plain, unbound thread, and `Scheduler::enqueue` is the entry point that
        // needs no binding of its own thread (the constructor excluded the only case
        // where it would: a scheduler without worker threads).
        _scheduler->enqueue(marl::Task{[this, lane, context, &jobs, &next, lanes_done]() mutable noexcept {
            _lane_enter(lane);
            for (;;) {
                auto index = next.fetch_add(1u, std::memory_order_relaxed);
                if (index >= jobs.size()) { break; }
                _prepare_job(jobs[index], context);
            }
            _lane_leave(lane);
            lanes_done.done();
        }});
    }
    _publish_in_order(jobs, results);
    // Ordered publication is complete, so every ticket is signalled; the counter
    // additionally proves that no lane task is still running, which is what allows
    // `jobs`, `contexts` and `next` to die with this frame. A lane touches its job only
    // before signalling it, so publication never races the storage.
    lanes_done.wait();
}

luisa::compute::VulkanShaderCompileStats ShaderCompilePool::stats() const noexcept {
    luisa::compute::VulkanShaderCompileStats stats;
    stats.worker_count = _worker_count.load(std::memory_order_relaxed);
    stats.pool_created = stats.worker_count != 0u;
    stats.pool_shutdown = _stopped.load(std::memory_order_relaxed);
    stats.batches = _counters.batches.load(std::memory_order_relaxed);
    stats.requests = _counters.requests.load(std::memory_order_relaxed);
    stats.parallel_prepares = _counters.parallel_prepares.load(std::memory_order_relaxed);
    stats.serial_prepares = _counters.serial_prepares.load(std::memory_order_relaxed);
    stats.cache_hits = _counters.cache_hits.load(std::memory_order_relaxed);
    stats.serial_fallbacks = _counters.serial_fallbacks.load(std::memory_order_relaxed);
    stats.publish_order_waits = _counters.publish_order_waits.load(std::memory_order_relaxed);
    stats.publish_order_wait_ns = _counters.publish_order_wait_ns.load(std::memory_order_relaxed);
    return stats;
}
}// namespace lc::vk::detail
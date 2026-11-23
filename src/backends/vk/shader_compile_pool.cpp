#include "shader_compile_pool.h"
#include "device.h"
#include <algorithm>
#include <luisa/core/clock.h>
#include <luisa/core/logging.h>
namespace lc::vk::detail {

namespace {
// The worker that owns the calling thread, or null. Set for the whole life of a
// worker thread, cleared before it exits, never touched by the publishing thread -
// which is what makes "am I nested inside a pool?" answerable without a lock.
thread_local WorkerContext *g_current_worker = nullptr;

// Marks one job's ticket no matter how the prepare phase leaves (value, early
// return, or an exception-free error path that reports through the result). The
// publisher waits on exactly this flag, so a missing mark would be a head-of-line
// hang; the guard makes that impossible by construction.
struct TicketGuard {
    ShaderCompilePool *pool;
    size_t index;
    bool armed{true};
    TicketGuard(ShaderCompilePool *pool, size_t index) noexcept
        : pool{pool}, index{index} {}
    TicketGuard(const TicketGuard &) = delete;
    TicketGuard &operator=(const TicketGuard &) = delete;
    void dismiss() noexcept { armed = false; }
    ~TicketGuard() noexcept {
        if (armed) { pool->complete_ticket(index); }
    }
};
}// namespace

bool ShaderCompilePool::on_worker_thread() noexcept {
    return g_current_worker != nullptr;
}

uint32_t ShaderCompilePool::default_worker_count() noexcept {
    auto hardware = std::thread::hardware_concurrency();
    // One worker fewer than the hardware threads: the publishing thread is busy
    // with driver calls for most of the batch.
    auto count = hardware > 1u ? hardware - 1u : 1u;
    return std::min<uint32_t>(std::max<uint32_t>(count, 1u), max_workers);
}

ShaderCompilePool::ShaderCompilePool(Device *device) noexcept
    : _device{device} {}

ShaderCompilePool::~ShaderCompilePool() noexcept {
    stop();
}

void ShaderCompilePool::stop() noexcept {
    // The batch mutex makes "stop during a batch" a wait, not a race: whichever
    // thread owns the batch finishes it (workers stay alive until it does), then
    // this call sets the final flag and joins.
    std::lock_guard batch_lock{_batch_mtx};
    _stopped = true;
    _stop_workers_locked();
}

void ShaderCompilePool::_stop_workers_locked() noexcept {
    if (_threads.empty()) {
        // Nothing to join; make sure a later `_ensure_workers` starts clean.
        std::lock_guard lock{_mtx};
        _shutdown = false;
        _worker_count = 0u;
        return;
    }
    {
        std::lock_guard lock{_mtx};
        _shutdown = true;
    }
    _cv.notify_all();
    if (on_worker_thread()) {
        // Never join ourselves. The workers observe `_shutdown`, drain the active
        // batch (if any) and exit; a later `stop()`/`~ShaderCompilePool` from a
        // non-worker thread joins them.
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
    _worker_count = 0u;
}

bool ShaderCompilePool::_ensure_workers(uint32_t count) noexcept {
    if (_stopped) { return false; }
    count = std::min<uint32_t>(std::max<uint32_t>(count, 1u), max_workers);
    if (_worker_count == count && _threads.size() == count) { return true; }
    // Resizing only ever happens between batches (the batch mutex is held), so no
    // ticket can be outstanding here.
    _stop_workers_locked();
    if (_stopped) { return false; }
    _contexts.reserve(count);
    for (auto i = 0u; i < count; i++) {
        auto context = luisa::make_unique<WorkerContext>();
        context->index = i;
        auto *raw = context.get();
        _contexts.emplace_back(std::move(context));
        _threads.emplace_back([this, raw] { _worker_main(raw); });
    }
    _worker_count = count;
    LUISA_VERBOSE("Vulkan shader compile pool: {} worker thread(s).", count);
    return true;
}

void ShaderCompilePool::complete_ticket(size_t index) noexcept {
    {
        std::lock_guard lock{_mtx};
        if (_jobs != nullptr && index < _job_count) {
            (*_jobs)[index].prepared_done = true;
        }
    }
    _cv.notify_all();
}

void ShaderCompilePool::_worker_main(WorkerContext *context) noexcept {
    g_current_worker = context;
    for (;;) {
        Job *job = nullptr;
        size_t index = 0u;
        bool stop_now = false;
        {
            std::unique_lock lock{_mtx};
            _cv.wait(lock, [this] {
                return _shutdown ||
                       (_jobs != nullptr &&
                        _next_job.load(std::memory_order_relaxed) < _job_count);
            });
            if (_jobs != nullptr &&
                _next_job.load(std::memory_order_relaxed) < _job_count) {
                // Claiming under the lock keeps the claim and the "is there work"
                // predicate consistent, so a worker never claims index >= count.
                index = _next_job.fetch_add(1, std::memory_order_relaxed);
                if (index < _job_count) { job = &(*_jobs)[index]; }
            } else if (_shutdown) {
                // Either no batch is active, or every job of the active batch has
                // been claimed (and its owner will mark its ticket). Both make it
                // safe to leave: the publisher never needs a thread that has not
                // claimed its ticket yet.
                stop_now = true;
            }
        }
        if (stop_now) { break; }
        if (job == nullptr) { continue; }
        TicketGuard ticket{this, index};
        auto &prepared = job->prepared;
        prepared = _device->_prepare_compute_shader(
            job->option, job->kernel, context);
        _counters.parallel_prepares.fetch_add(1u, std::memory_order_relaxed);
        ++context->prepared;
        if (prepared.kind == PreparedComputeShader::Kind::SERIAL_FALLBACK) {
            _counters.serial_fallbacks.fetch_add(1u, std::memory_order_relaxed);
        } else if (prepared.kind == PreparedComputeShader::Kind::READY &&
                   prepared.from_cache) {
            _counters.cache_hits.fetch_add(1u, std::memory_order_relaxed);
        }
        ticket.dismiss();
        complete_ticket(index);
    }
    g_current_worker = nullptr;
}

luisa::compute::VulkanShaderCompileStatus ShaderCompilePool::compile(
    luisa::span<const luisa::compute::VulkanShaderCompileRequest> requests,
    luisa::vector<luisa::compute::VulkanShaderCompileResult> &results,
    uint32_t worker_count) noexcept {
    results.clear();
    results.resize(requests.size());
    if (requests.empty()) { return luisa::compute::VulkanShaderCompileStatus::INVALID_ARGUMENT; }
    std::lock_guard batch_lock{_batch_mtx};
    if (_stopped) {
        for (auto &result : results) {
            result.status = luisa::compute::VulkanShaderCompileStatus::POOL_SHUTDOWN;
            result.message = "the shader compile pool was shut down";
            // No shader was created for this request, so the answer says so
            // explicitly instead of leaving a default-constructed handle behind.
            result.shader.invalidate();
        }
        return luisa::compute::VulkanShaderCompileStatus::POOL_SHUTDOWN;
    }
    _counters.batches.fetch_add(1u, std::memory_order_relaxed);
    _counters.requests.fetch_add(requests.size(), std::memory_order_relaxed);
    if (on_worker_thread()) {
        // Nested compilation: waiting for a worker from a worker is a deadlock,
        // so the batch is compiled on this thread with the unchanged serial path.
        _counters.serial_prepares.fetch_add(requests.size(), std::memory_order_relaxed);
        _run_batch_serial(requests, results);
        return luisa::compute::VulkanShaderCompileStatus::FELL_BACK_TO_SERIAL;
    }
    auto count = worker_count == 0u ? default_worker_count() : worker_count;
    if (!_ensure_workers(count)) {
        _counters.serial_prepares.fetch_add(requests.size(), std::memory_order_relaxed);
        _run_batch_serial(requests, results);
        return luisa::compute::VulkanShaderCompileStatus::FELL_BACK_TO_SERIAL;
    }
    _run_batch(requests, results);
    return luisa::compute::VulkanShaderCompileStatus::SUCCESS;
}

namespace {
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

void ShaderCompilePool::_run_batch_serial(
    luisa::span<const luisa::compute::VulkanShaderCompileRequest> requests,
    luisa::vector<luisa::compute::VulkanShaderCompileResult> &results) noexcept {
    for (auto i = 0u; i < requests.size(); i++) {
        auto &request = requests[i];
        fill_result(results[i], _device->create_shader(request.option, request.kernel));
        results[i].serial_fallback = true;
    }
}

void ShaderCompilePool::_run_batch(
    luisa::span<const luisa::compute::VulkanShaderCompileRequest> requests,
    luisa::vector<luisa::compute::VulkanShaderCompileResult> &results) noexcept {
    luisa::vector<Job> jobs;
    jobs.reserve(requests.size());
    for (auto &request : requests) {
        Job job;
        job.option = request.option;
        job.kernel = request.kernel;
        jobs.emplace_back(std::move(job));
    }
    {
        std::lock_guard lock{_mtx};
        _jobs = &jobs;
        _job_count = jobs.size();
        _next_job.store(0u, std::memory_order_relaxed);
    }
    _cv.notify_all();
    // Publish in request order. The publisher is the only thread that creates
    // Vulkan objects, and it creates them for sequence number 0, 1, 2, ... - which
    // is what makes the batch order-deterministic.
    for (auto i = 0u; i < jobs.size(); i++) {
        auto &job = jobs[i];
        bool waited = false;
        {
            Clock clock;
            std::unique_lock lock{_mtx};
            if (!job.prepared_done) {
                waited = true;
                _cv.wait(lock, [&job] { return job.prepared_done; });
            }
            if (waited) {
                auto ns = static_cast<uint64_t>(clock.toc() * 1.0e6);
                _counters.publish_order_waits.fetch_add(1u, std::memory_order_relaxed);
                _counters.publish_order_wait_ns.fetch_add(ns, std::memory_order_relaxed);
            }
        }
        auto &result = results[i];
        if (job.prepared.kind == PreparedComputeShader::Kind::SERIAL_FALLBACK) {
            // The unchanged serial path, on the publishing thread: today's exact
            // behaviour for this request, including the process-global DXC use.
            fill_result(result, _device->create_shader(job.option, job.kernel));
            result.serial_fallback = true;
        } else {
            fill_result(result, _device->_publish_compute_shader(
                                    std::move(job.prepared), job.option, job.kernel));
        }
    }
    {
        std::lock_guard lock{_mtx};
        _jobs = nullptr;
        _job_count = 0u;
        _next_job.store(0u, std::memory_order_relaxed);
    }
}

luisa::compute::VulkanShaderCompileStats ShaderCompilePool::stats() const noexcept {
    luisa::compute::VulkanShaderCompileStats stats;
    stats.worker_count = _worker_count;
    stats.pool_created = _worker_count != 0u;
    stats.pool_shutdown = _stopped;
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
#include "simd_thread_pool.h"

#include <algorithm>

namespace luisa::compute::simd {

SIMDThreadPool::SIMDThreadPool(uint32_t worker_count) noexcept
    : _worker_cursors{std::make_unique<WorkerCursor[]>(
          std::max(worker_count, 1u))},
      _worker_count{std::max(worker_count, 1u)} {
    if (_worker_count <= 1u) { return; }
    _workers.reserve(_worker_count);
    for (auto worker = uint32_t{0u}; worker < _worker_count; worker++) {
        _workers.emplace_back([this, worker]() noexcept {
            _worker_loop(worker);
        });
    }
}

SIMDThreadPool::~SIMDThreadPool() noexcept {
    std::scoped_lock submit_lock{_submit_mutex};
    {
        std::scoped_lock work_lock{_work_mutex};
        _stopping = true;
        _generation++;
    }
    _work_available.notify_all();
    for (auto &worker : _workers) {
        if (worker.joinable()) { worker.join(); }
    }
}

void SIMDThreadPool::_worker_loop(uint32_t worker_index) noexcept {
    auto observed_generation = uint64_t{0u};
    std::unique_lock lock{_work_mutex};
    for (;;) {
        _work_available.wait(lock, [&]() noexcept {
            return _stopping || _generation != observed_generation;
        });
        if (_stopping) { return; }
        observed_generation = _generation;
        auto work = _work;
        if (worker_index >= work.active_workers) { continue; }
        lock.unlock();
        auto invoke_chunk = [&](uint64_t chunk) noexcept {
            auto begin = chunk * work.grain_size;
            auto end = begin +
                       std::min(work.grain_size, work.count - begin);
            work.invoke(work.context, begin, end);
        };
        auto try_claim = [&](uint32_t owner) noexcept {
            auto owned_chunk_count =
                (work.chunk_count - 1u - owner) /
                    work.active_workers +
                1u;
            auto local_chunk = _worker_cursors[owner].next.fetch_add(
                1u, std::memory_order_relaxed);
            if (local_chunk >= owned_chunk_count) { return false; }
            auto chunk = owner +
                         local_chunk * work.active_workers;
            invoke_chunk(chunk);
            return true;
        };

        // Drain the stable home sequence first. Once a worker becomes idle it
        // steals one chunk at a time in round-robin victim order, retaining
        // load balance without discarding cross-dispatch ownership locality.
        while (try_claim(worker_index)) {}
        for (;;) {
            auto found_work = false;
            for (auto offset = uint32_t{1u};
                 offset < work.active_workers; offset++) {
                auto victim = worker_index + offset;
                if (victim >= work.active_workers) {
                    victim -= work.active_workers;
                }
                found_work |= try_claim(victim);
            }
            if (!found_work) { break; }
        }
        lock.lock();
        if (--_working_workers == 0u) {
            _work_done.notify_one();
        }
    }
}

void SIMDThreadPool::_parallel_for(
    uint64_t count, uint64_t grain_size,
    void *context, Invoke invoke) noexcept {
    if (count == 0u) { return; }
    grain_size = std::max(grain_size, uint64_t{1u});
    std::scoped_lock submit_lock{_submit_mutex};
    auto chunk_count = (count - 1u) / grain_size + 1u;
    auto active_workers = static_cast<uint32_t>(std::min<uint64_t>(
        chunk_count, _workers.size()));
    if (active_workers <= 1u) {
        invoke(context, 0u, count);
        return;
    }
    std::unique_lock lock{_work_mutex};
    for (auto worker = uint32_t{0u}; worker < active_workers; worker++) {
        _worker_cursors[worker].next.store(
            0u, std::memory_order_relaxed);
    }
    _work = {
        .count = count,
        .grain_size = grain_size,
        .chunk_count = chunk_count,
        .active_workers = active_workers,
        .context = context,
        .invoke = invoke,
    };
    _working_workers = active_workers;
    _generation++;
    _work_available.notify_all();
    _work_done.wait(lock, [this]() noexcept {
        return _working_workers == 0u;
    });
    _work = {};
}

}// namespace luisa::compute::simd

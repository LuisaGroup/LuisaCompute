#include "simd_thread_pool.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string_view>
#include <luisa/core/stl/memory.h>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#elif defined(__linux__)
#include <pthread.h>
#include <sched.h>
#endif

namespace luisa::compute::simd {

namespace {

void pin_worker_thread(uint32_t worker_index) noexcept {
    if (const auto *raw = std::getenv("LUISA_SIMD_PINNED");
        raw == nullptr || std::string_view{raw} != "1") {
        return;
    }
#ifdef _WIN32
    const auto group_count = GetActiveProcessorGroupCount();
    uint32_t total_lps = 0u;
    for (WORD g = 0u; g < group_count; ++g) {
        total_lps += GetActiveProcessorCount(g);
    }
    if (total_lps == 0u) { return; }
    uint32_t target = worker_index % total_lps;
    for (WORD g = 0u; g < group_count; ++g) {
        const uint32_t count = GetActiveProcessorCount(g);
        if (target < count) {
            GROUP_AFFINITY aff{};
            aff.Group = g;
            aff.Mask = KAFFINITY{1u} << target;
            SetThreadGroupAffinity(GetCurrentThread(), &aff, nullptr);
            return;
        }
        target -= count;
    }
#elif defined(__linux__)
    cpu_set_t allowed;
    CPU_ZERO(&allowed);
    if (pthread_getaffinity_np(pthread_self(), sizeof(cpu_set_t), &allowed) == 0) {
        luisa::vector<uint32_t> cpus;
        for (uint32_t c = 0u; c < CPU_SETSIZE; ++c) {
            if (CPU_ISSET(c, &allowed)) { cpus.push_back(c); }
        }
        if (!cpus.empty()) {
            const auto chosen = cpus[worker_index % cpus.size()];
            cpu_set_t cpuset;
            CPU_ZERO(&cpuset);
            CPU_SET(chosen, &cpuset);
            pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
        }
    }
#else
    static_cast<void>(worker_index);
#endif
}

}// namespace

SIMDThreadPool::SIMDThreadPool(uint32_t worker_count) noexcept
    : _worker_cursors{std::make_unique<WorkerCursor[]>(
          std::max(worker_count, 1u))},
      _worker_count{std::max(worker_count, 1u)} {
    if (const auto *mode = std::getenv("LUISA_SIMD_THREAD_POOL_MODE")) {
        const auto sv = std::string_view{mode};
        if (sv == "contiguous") {
            _schedule_mode = ScheduleMode::contiguous;
        } else if (sv == "strided") {
            _schedule_mode = ScheduleMode::strided;
        } else if (sv == "dynamic") {
            _schedule_mode = ScheduleMode::dynamic;
        }
    }
    if (_worker_count <= 1u) { return; }
    _workers.reserve(_worker_count);
    for (auto worker = uint32_t{0u}; worker < _worker_count; worker++) {
        _workers.emplace_back([this, worker]() noexcept {
            pin_worker_thread(worker);
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
        const auto t0 = std::chrono::steady_clock::now();
        if (_schedule_mode == ScheduleMode::dynamic) {
            for (;;) {
                if (_dynamic_cursor.load(std::memory_order_relaxed) >=
                    work.chunk_count) {
                    break;
                }
                auto chunk = _dynamic_cursor.fetch_add(
                    1u, std::memory_order_relaxed);
                if (chunk >= work.chunk_count) { break; }
                invoke_chunk(chunk);
                _worker_cursors[worker_index].home_chunks++;
            }
        } else if (_schedule_mode == ScheduleMode::contiguous) {
            auto try_claim_contiguous = [&](uint32_t owner, bool is_home) noexcept {
                const auto end = _worker_cursors[owner].home_end;
                if (_worker_cursors[owner].next.load(std::memory_order_relaxed) >=
                    end) {
                    return false;
                }
                const auto chunk = _worker_cursors[owner].next.fetch_add(
                    1u, std::memory_order_relaxed);
                if (chunk >= end) { return false; }
                invoke_chunk(chunk);
                if (is_home) {
                    _worker_cursors[worker_index].home_chunks++;
                } else {
                    _worker_cursors[worker_index].stolen_chunks++;
                }
                return true;
            };
            while (try_claim_contiguous(worker_index, true)) {}
            for (;;) {
                auto found_work = false;
                for (auto offset = uint32_t{1u};
                     offset < work.active_workers; offset++) {
                    auto victim = worker_index + offset;
                    if (victim >= work.active_workers) {
                        victim -= work.active_workers;
                    }
                    found_work |= try_claim_contiguous(victim, false);
                }
                if (!found_work) { break; }
            }
        } else {
            auto try_claim_strided = [&](uint32_t owner, bool is_home) noexcept {
                auto owned_chunk_count =
                    (work.chunk_count - 1u - owner) /
                        work.active_workers +
                    1u;
                if (_worker_cursors[owner].next.load(std::memory_order_relaxed) >=
                    owned_chunk_count) {
                    return false;
                }
                auto local_chunk = _worker_cursors[owner].next.fetch_add(
                    1u, std::memory_order_relaxed);
                if (local_chunk >= owned_chunk_count) { return false; }
                auto chunk = owner +
                             local_chunk * work.active_workers;
                invoke_chunk(chunk);
                if (is_home) {
                    _worker_cursors[worker_index].home_chunks++;
                } else {
                    _worker_cursors[worker_index].stolen_chunks++;
                }
                return true;
            };
            while (try_claim_strided(worker_index, true)) {}
            for (;;) {
                auto found_work = false;
                for (auto offset = uint32_t{1u};
                     offset < work.active_workers; offset++) {
                    auto victim = worker_index + offset;
                    if (victim >= work.active_workers) {
                        victim -= work.active_workers;
                    }
                    found_work |= try_claim_strided(victim, false);
                }
                if (!found_work) { break; }
            }
        }
        const auto t1 = std::chrono::steady_clock::now();
        _worker_cursors[worker_index].busy_ns =
            static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0)
                    .count());
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
    const auto wall_t0 = std::chrono::steady_clock::now();
    std::unique_lock lock{_work_mutex};
    _dynamic_cursor.store(0u, std::memory_order_relaxed);
    const auto base_chunks = chunk_count / active_workers;
    const auto rem_chunks = chunk_count % active_workers;
    uint64_t slice_pos = 0u;
    for (auto worker = uint32_t{0u}; worker < active_workers; worker++) {
        const auto slice_len = base_chunks + (worker < rem_chunks ? 1u : 0u);
        _worker_cursors[worker].home_begin = slice_pos;
        _worker_cursors[worker].home_end = slice_pos + slice_len;
        _worker_cursors[worker].next.store(
            _schedule_mode == ScheduleMode::contiguous ? slice_pos : 0u,
            std::memory_order_relaxed);
        slice_pos += slice_len;
        _worker_cursors[worker].busy_ns = 0u;
        _worker_cursors[worker].home_chunks = 0u;
        _worker_cursors[worker].stolen_chunks = 0u;
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
    lock.unlock();
    if (const auto *prof = std::getenv("LUISA_SIMD_PROFILE_DISPATCH");
        prof != nullptr && std::string_view{prof} == "1" && count >= 64u) {
        const auto wall_ns = static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - wall_t0)
                .count());
        uint64_t min_ns = ~uint64_t{0u};
        uint64_t max_ns = 0u;
        uint64_t sum_ns = 0u;
        uint64_t total_stolen = 0u;
        for (auto w = uint32_t{0u}; w < active_workers; w++) {
            const auto ns = _worker_cursors[w].busy_ns;
            min_ns = std::min(min_ns, ns);
            max_ns = std::max(max_ns, ns);
            sum_ns += ns;
            total_stolen += _worker_cursors[w].stolen_chunks;
        }
        const double util =
            wall_ns > 0u
                ? static_cast<double>(sum_ns) /
                      (static_cast<double>(wall_ns) * active_workers)
                : 0.0;
        std::printf(
            "LUISA_SIMD_DISPATCH_PROFILE_JSON: "
            "{\"count\":%llu,\"grain\":%llu,\"chunks\":%llu,\"workers\":%u,"
            "\"wall_ms\":%.3f,\"min_worker_ms\":%.3f,\"mean_worker_ms\":%.3f,"
            "\"max_worker_ms\":%.3f,\"active_threads_equiv\":%.2f,"
            "\"utilization\":%.4f,\"stolen_chunks\":%llu}\n",
            static_cast<unsigned long long>(count),
            static_cast<unsigned long long>(grain_size),
            static_cast<unsigned long long>(chunk_count), active_workers,
            static_cast<double>(wall_ns) * 1e-6,
            static_cast<double>(min_ns) * 1e-6,
            static_cast<double>(sum_ns) * 1e-6 / active_workers,
            static_cast<double>(max_ns) * 1e-6,
            wall_ns > 0u ? static_cast<double>(sum_ns) / static_cast<double>(wall_ns) : 0.0,
            util, static_cast<unsigned long long>(total_stolen));
        std::fflush(stdout);
    }
}

}// namespace luisa::compute::simd

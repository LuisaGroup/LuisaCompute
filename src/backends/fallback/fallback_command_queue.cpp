#include "fallback_command_queue.h"

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string_view>

#if defined(_M_X64) || defined(__x86_64__)
#include <immintrin.h>
#elif defined(_M_ARM64) && defined(_MSC_VER) && !defined(__clang__)
#include <intrin.h>
#endif

#include <luisa/core/stl/vector.h>
#include <luisa/vstl/unique_ptr.h>
#include <luisa/core/logging.h>

#ifdef LUISA_FALLBACK_HAS_TBB
#include <oneapi/tbb/blocked_range.h>
#include <oneapi/tbb/parallel_for.h>
#include <oneapi/tbb/partitioner.h>
#include <oneapi/tbb/task_arena.h>
#include <oneapi/tbb/task_scheduler_observer.h>
#endif

#ifdef LUISA_FALLBACK_USE_AKARI_THREAD_POOL

#include <optional>
#ifdef LUISA_PLATFORM_WINDOWS
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <ppl.h>
#include <concrt.h>
#else
#include <pthread.h>
#endif

namespace luisa::compute::fallback {

namespace {

enum class BackendSchedulerKind : uint8_t {
    AkariOrig = 0,
    AkariOpt = 1,
    AkariWorkSteal = 2,
    Tbb = 3,
    Ppl = 4,
};

[[nodiscard]] BackendSchedulerKind query_backend_scheduler_kind() noexcept {
    if (const auto *raw = std::getenv("LUISA_FALLBACK_BACKEND_SCHEDULER")) {
        const std::string_view sv{raw};
        if (sv == "akari_orig" || sv == "akari") { return BackendSchedulerKind::AkariOrig; }
        if (sv == "akari_opt" || sv == "opt" || sv == "akari_spin") { return BackendSchedulerKind::AkariOpt; }
        if (sv == "akari_ws") { return BackendSchedulerKind::AkariWorkSteal; }
        if (sv == "tbb") {
#ifdef LUISA_FALLBACK_HAS_TBB
            return BackendSchedulerKind::Tbb;
#else
            return BackendSchedulerKind::AkariOpt;
#endif
        }
        if (sv == "ppl") {
#ifdef LUISA_PLATFORM_WINDOWS
            return BackendSchedulerKind::Ppl;
#else
            return BackendSchedulerKind::AkariOpt;
#endif
        }
    }
    return BackendSchedulerKind::AkariOpt;
}

[[nodiscard]] bool query_use_orig_queue_sync() noexcept {
    if (const auto *s = std::getenv("LUISA_FALLBACK_BACKEND_SCHEDULER")) {
        if (std::string_view{s} == "akari_orig") { return true; }
    }
    if (const auto *q = std::getenv("LUISA_FALLBACK_QUEUE_MODE")) {
        if (std::string_view{q} == "orig") { return true; }
    }
    return false;
}

[[nodiscard]] uint query_parallel_chunk() noexcept {
    if (const auto *raw = std::getenv("LUISA_FALLBACK_PARALLEL_CHUNK")) {
        std::string_view sv{raw};
        uint v = 0u;
        const auto [ptr, ec] = std::from_chars(sv.data(), sv.data() + sv.size(), v);
        if (ec == std::errc{} && ptr == sv.data() + sv.size() && v > 0u) {
            return v;
        }
    }
    return 1u;
}

[[nodiscard]] uint query_spin_iters(const char *env_name, uint default_val) noexcept {
    if (const auto *raw = std::getenv(env_name)) {
        std::string_view sv{raw};
        uint v = 0u;
        const auto [ptr, ec] = std::from_chars(sv.data(), sv.data() + sv.size(), v);
        if (ec == std::errc{} && ptr == sv.data() + sv.size()) {
            return v;
        }
    }
    return default_val;
}

inline void cpu_relax_pause() noexcept {
#if defined(_M_X64) || defined(__x86_64__)
    _mm_pause();
#elif defined(_M_ARM64) || defined(__aarch64__)
#if defined(_MSC_VER) && !defined(__clang__)
    __yield();
#else
    asm volatile("yield" ::: "memory");
#endif
#endif
}

[[nodiscard]] uint map_slot_to_logical_cpu(uint tid, std::string_view topo) noexcept {
    if (topo == "physical_first") {
        if (tid < 16u) { return tid * 2u; }
        if (tid < 32u) { return (tid - 16u) * 2u + 1u; }
    } else if (topo == "ccd_interleaved") {
        if (tid < 16u) {
            const uint core_in_ccd = tid / 2u;
            const uint ccd = tid % 2u;
            return ccd * 16u + core_in_ccd * 2u;
        } else if (tid < 32u) {
            const uint rem = tid - 16u;
            const uint core_in_ccd = rem / 2u;
            const uint ccd = rem % 2u;
            return ccd * 16u + core_in_ccd * 2u + 1u;
        }
    } else if (topo == "ccd0_pinned") {
        return tid % 16u;
    } else if (topo == "ccd1_pinned") {
        return 16u + (tid % 16u);
    } else if (topo == "ccd1_first") {
        return (tid + 16u) % 32u;
    }
    return tid;
}

#ifdef LUISA_PLATFORM_WINDOWS
void apply_current_or_target_thread_affinity(HANDLE handle, uint slot, uint &out_lp) noexcept {
    const std::string_view topo = []() noexcept -> std::string_view {
        if (const auto *raw = std::getenv("LUISA_FALLBACK_TOPOLOGY")) {
            return std::string_view{raw};
        }
        return "linear_pinned";
    }();
    if (topo == "unpinned" || topo == "os_floating") {
        out_lp = slot;
        return;
    }
    if (topo == "ccd0_pool") {
        GROUP_AFFINITY affinity{};
        affinity.Group = 0;
        affinity.Mask = 0x0000ffffull;
        out_lp = slot % 16u;
        SetThreadGroupAffinity(handle, &affinity, nullptr);
        return;
    }
    if (topo == "ccd1_pool") {
        GROUP_AFFINITY affinity{};
        affinity.Group = 0;
        affinity.Mask = 0xffff0000ull;
        out_lp = 16u + (slot % 16u);
        SetThreadGroupAffinity(handle, &affinity, nullptr);
        return;
    }
    const uint target_lp = map_slot_to_logical_cpu(slot, topo);
    out_lp = target_lp;
    GROUP_AFFINITY affinity{};
    affinity.Group = static_cast<WORD>(target_lp / 64u);
    affinity.Mask = 1ull << (target_lp % 64u);
    if (SetThreadGroupAffinity(handle, &affinity, nullptr) == 0) {
        LUISA_WARNING_WITH_LOCATION("Failed to pin thread to group");
    }
}
#endif

struct SingleRangeClaimState {
    uint begin{};
    uint end{};
    bool consumed{false};
    static bool claim(void *p, uint &out_b, uint &out_e) noexcept {
        auto *s = static_cast<SingleRangeClaimState *>(p);
        if (s->consumed || s->begin >= s->end) { return false; }
        out_b = s->begin;
        out_e = s->end;
        s->consumed = true;
        return true;
    }
};

}// namespace

struct Thread {

#ifdef LUISA_PLATFORM_WINDOWS
    HANDLE handle{};
#else
    pthread_t handle{};
#endif
    luisa::move_only_function<void()> f;
    static constexpr auto STACK_SIZE = 16_M;
    uint assigned_lp = 0u;

    template<class F>
    explicit Thread(uint tid [[maybe_unused]], F &&f) noexcept : f{std::forward<F>(f)}, assigned_lp{tid} {

#ifdef LUISA_PLATFORM_WINDOWS
        handle = CreateThread(nullptr, STACK_SIZE, [](void *arg) -> DWORD {
            auto *f = static_cast<luisa::move_only_function<void()> *>(arg);
            (*f)();
            return 0; }, &this->f, 0, nullptr);
        if (handle == nullptr) {
            LUISA_ERROR_WITH_LOCATION("Failed to create thread");
        }
        apply_current_or_target_thread_affinity(handle, tid, assigned_lp);
#else
        pthread_attr_t attr;
        pthread_attr_init(&attr);
        pthread_attr_setstacksize(&attr, STACK_SIZE);
#ifdef LUISA_PLATFORM_APPLE
        pthread_attr_set_qos_class_np(&attr, QOS_CLASS_USER_INTERACTIVE, 0);
#endif
        if (pthread_create(&handle, &attr, [](void *arg) noexcept -> void * {
            auto *f = static_cast<luisa::move_only_function<void()> *>(arg);
            (*f)();
            return nullptr; }, &this->f) != 0) {
            LUISA_ERROR_WITH_LOCATION("Failed to create thread");
        }
        pthread_attr_destroy(&attr);
#if !defined(LUISA_PLATFORM_APPLE)
        const std::string_view topo = []() noexcept -> std::string_view {
            if (const auto *raw = std::getenv("LUISA_FALLBACK_TOPOLOGY")) {
                return std::string_view{raw};
            }
            return "linear_pinned";
        }();
        if (topo != "unpinned" && topo != "os_floating") {
            cpu_set_t allowed;
            CPU_ZERO(&allowed);
            uint chosen_cpu = tid;
            if (pthread_getaffinity_np(pthread_self(), sizeof(cpu_set_t), &allowed) == 0) {
                luisa::vector<uint> allowed_cpus;
                for (uint c = 0u; c < CPU_SETSIZE; ++c) {
                    if (CPU_ISSET(c, &allowed)) { allowed_cpus.push_back(c); }
                }
                if (!allowed_cpus.empty()) {
                    const uint mapped = map_slot_to_logical_cpu(tid, topo);
                    chosen_cpu = allowed_cpus[mapped % allowed_cpus.size()];
                }
            }
            if (chosen_cpu < CPU_SETSIZE) {
                assigned_lp = chosen_cpu;
                cpu_set_t cpuset;
                CPU_ZERO(&cpuset);
                CPU_SET(chosen_cpu, &cpuset);
                if (pthread_setaffinity_np(handle, sizeof(cpu_set_t), &cpuset) != 0) {
                    LUISA_WARNING_WITH_LOCATION("Failed to pin thread affinity");
                }
            }
        }
#endif
#endif
    }
    Thread(const Thread &) = delete;
    Thread(Thread &&) = delete;
    Thread &operator=(const Thread &) = delete;
    Thread &operator=(Thread &&) = delete;

    ~Thread() noexcept {
#ifdef LUISA_PLATFORM_WINDOWS
        WaitForSingleObject(handle, INFINITE);
        CloseHandle(handle);
#else
        pthread_join(handle, nullptr);
#endif
    }
};

// ============================================================================
// 1. Original AkrThreadPool (Preserved verbatim + opt-in telemetry for A/B)
// ============================================================================
class AkrThreadPool {

public:
    struct alignas(64) ParallelFor {
        uint count = 0u;
        uint block_size = 0u;
        luisa::move_only_function<void(uint)> *task = nullptr;
    };

    struct alignas(64) WorkerDispatchStats {
        uint64_t wakeup_ns = 0ull;
        uint64_t finish_ns = 0ull;
        uint64_t exec_ns = 0ull;
        uint64_t max_task_ns = 0ull;
        uint64_t last_task_ns = 0ull;
        uint32_t tasks_done = 0u;
        uint32_t max_task_id = 0u;
        uint32_t last_task_id = 0u;
    };

private:
    luisa::vector<luisa::unique_ptr<Thread>> _threads;
    luisa::vector<WorkerDispatchStats> _stats;
    std::mutex _task_mutex, _submit_mutex;
    std::condition_variable _has_work, _work_done;
    ParallelFor _parallel_for = {};
    std::chrono::steady_clock::time_point _dispatch_start_tp{};
    uint64_t _dispatch_index = 0ull;
    uint _configured_chunk_size = 1u;
    bool _guided_mode = false;
    bool _sched_profile = false;

#ifdef __cpp_lib_hardware_interference_size
    static constexpr auto cache_line_size = std::hardware_destructive_interference_size;
#else
    static constexpr auto cache_line_size = 64u;
#endif
    alignas(cache_line_size) std::atomic_uint32_t _item_count = 0u;
    alignas(cache_line_size) std::atomic_uint64_t _task_generation = 0u;

    std::atomic_uint32_t _thread_working;
    std::atomic_bool _stopped = false;

    [[nodiscard]] bool _claim_chunk(uint count, uint min_chunk, size_t n_workers, uint &out_begin, uint &out_end) noexcept {
        if (!_guided_mode) {
            const uint cur = _item_count.load(std::memory_order_relaxed);
            if (cur >= count) { return false; }
            const uint i = _item_count.fetch_add(min_chunk, std::memory_order_relaxed);
            if (i >= count) { return false; }
            out_begin = i;
            out_end = std::min<uint>(i + min_chunk, count);
            return true;
        }
        uint cur = _item_count.load(std::memory_order_relaxed);
        while (cur < count) {
            const uint rem = count - cur;
            const uint denom = static_cast<uint>(std::max<size_t>(1u, n_workers * 4u));
            const uint step = std::min<uint>(rem, std::max<uint>(min_chunk, (rem + denom - 1u) / denom));
            if (_item_count.compare_exchange_weak(cur, cur + step, std::memory_order_relaxed, std::memory_order_relaxed)) {
                out_begin = cur;
                out_end = cur + step;
                return true;
            }
        }
        return false;
    }

public:
    explicit AkrThreadPool(size_t n_threads) noexcept : _stats(n_threads) {
        if (const auto *raw = std::getenv("LUISA_FALLBACK_SCHED_PROFILE")) {
            _sched_profile = std::string_view{raw} == "1";
        }
        if (const auto *raw = std::getenv("LUISA_FALLBACK_SCHED_MODE")) {
            _guided_mode = std::string_view{raw} == "guided";
        }
        _configured_chunk_size = query_parallel_chunk();
        for (size_t tid = 0; tid < n_threads; tid++) {
            _threads.emplace_back(std::move(luisa::make_unique<Thread>(static_cast<uint>(tid), [this, tid, n_threads] {
                auto last_task_generation = uint64_t{0};
                while (!_stopped.load(std::memory_order_seq_cst)) {
                    std::unique_lock lock{_task_mutex};
                    _has_work.wait(lock, [this, &last_task_generation] { return (last_task_generation != _task_generation.load() && _parallel_for.task != nullptr) || _stopped.load(std::memory_order_seq_cst); });
                    if (_stopped.load(std::memory_order_seq_cst)) { return; }
                    last_task_generation = _task_generation.load(std::memory_order_seq_cst);
                    auto [count, block_size, task] = _parallel_for;
                    const auto start_tp = _dispatch_start_tp;
                    lock.unlock();
                    if (!_sched_profile) {
                        if (!_guided_mode) {
                            while (_item_count.load(std::memory_order_relaxed) < count) {
                                auto i = _item_count.fetch_add(block_size, std::memory_order_relaxed);
                                for (uint j = i; j < std::min<uint>(i + block_size, count); j++) {
                                    (*task)(j);
                                }
                            }
                        } else {
                            uint begin_j = 0u, end_j = 0u;
                            while (_claim_chunk(count, block_size, n_threads, begin_j, end_j)) {
                                for (uint j = begin_j; j < end_j; j++) {
                                    (*task)(j);
                                }
                            }
                        }
                    } else {
                        const auto woke_tp = std::chrono::steady_clock::now();
                        uint64_t exec_ns = 0ull, max_t_ns = 0ull, last_t_ns = 0ull;
                        uint32_t done = 0u, max_id = 0u, last_id = 0u;
                        uint begin_j = 0u, end_j = 0u;
                        while (_claim_chunk(count, block_size, n_threads, begin_j, end_j)) {
                            for (uint j = begin_j; j < end_j; j++) {
                                const auto t0 = std::chrono::steady_clock::now();
                                (*task)(j);
                                const auto t1 = std::chrono::steady_clock::now();
                                const auto dt = static_cast<uint64_t>(
                                    std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count());
                                exec_ns += dt;
                                if (dt >= max_t_ns) { max_t_ns = dt; max_id = j; }
                                last_t_ns = dt;
                                last_id = j;
                                ++done;
                            }
                        }
                        const auto fin_tp = std::chrono::steady_clock::now();
                        auto &st = _stats[tid];
                        st.wakeup_ns = static_cast<uint64_t>(
                            std::chrono::duration_cast<std::chrono::nanoseconds>(woke_tp - start_tp).count());
                        st.finish_ns = static_cast<uint64_t>(
                            std::chrono::duration_cast<std::chrono::nanoseconds>(fin_tp - start_tp).count());
                        st.exec_ns = exec_ns;
                        st.max_task_ns = max_t_ns;
                        st.last_task_ns = last_t_ns;
                        st.tasks_done = done;
                        st.max_task_id = max_id;
                        st.last_task_id = last_id;
                    }
                    lock.lock();
                    if (_thread_working.fetch_sub(1, std::memory_order_seq_cst) == 1) {
                        _work_done.notify_all();
                    }
                }
            })));
        }
    }
    ~AkrThreadPool() noexcept {
        {
            std::scoped_lock lock{_task_mutex};
            _stopped.store(true, std::memory_order_seq_cst);
        }
        _has_work.notify_all();
        _threads.clear();
    }
    void parallel_for(uint count, luisa::move_only_function<void(uint)> &&task) noexcept {
        std::scoped_lock _lk{_submit_mutex};
        std::unique_lock lock{_task_mutex};
        _thread_working.store(static_cast<uint32_t>(_threads.size()), std::memory_order_seq_cst);
        _parallel_for = {count, _configured_chunk_size, &task};
        _item_count.store(0, std::memory_order_seq_cst);
        if (_sched_profile) {
            _dispatch_start_tp = std::chrono::steady_clock::now();
        }
        _task_generation.fetch_add(1, std::memory_order_seq_cst);
        _has_work.notify_all();
        _work_done.wait(lock, [this] { return _thread_working.load(std::memory_order_seq_cst) == 0; });
        const auto end_tp = _sched_profile ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
        _parallel_for = {};
        if (_sched_profile) {
            const uint64_t wall_ns = static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(end_tp - _dispatch_start_tp).count());
            uint64_t sum_exec = 0ull, sum_wakeup = 0ull, sum_fetch_ovh = 0ull, sum_tail_idle = 0ull;
            uint64_t max_wakeup = 0ull, min_fin = UINT64_MAX, max_fin = 0ull, sum_fin = 0ull;
            uint64_t max_task = 0ull, straggler_last_ns = 0ull;
            uint32_t max_task_id = 0u, straggler_tid = 0u, straggler_last_id = 0u;
            uint64_t ccd0_tasks = 0ull, ccd1_tasks = 0ull, ccd0_exec = 0ull, ccd1_exec = 0ull;
            const size_t W = _stats.size();
            for (size_t t = 0; t < W; ++t) {
                const auto &s = _stats[t];
                sum_exec += s.exec_ns;
                sum_wakeup += s.wakeup_ns;
                const uint64_t active_span = s.finish_ns >= s.wakeup_ns ? (s.finish_ns - s.wakeup_ns) : 0ull;
                sum_fetch_ovh += active_span >= s.exec_ns ? (active_span - s.exec_ns) : 0ull;
                sum_tail_idle += wall_ns >= s.finish_ns ? (wall_ns - s.finish_ns) : 0ull;
                if (s.wakeup_ns > max_wakeup) { max_wakeup = s.wakeup_ns; }
                if (s.finish_ns < min_fin) { min_fin = s.finish_ns; }
                if (s.finish_ns >= max_fin) {
                    max_fin = s.finish_ns;
                    straggler_tid = static_cast<uint32_t>(t);
                    straggler_last_id = s.last_task_id;
                    straggler_last_ns = s.last_task_ns;
                }
                sum_fin += s.finish_ns;
                if (s.max_task_ns >= max_task) {
                    max_task = s.max_task_ns;
                    max_task_id = s.max_task_id;
                }
                const uint lp = _threads[t]->assigned_lp;
                if (lp < 16u) {
                    ccd0_tasks += s.tasks_done;
                    ccd0_exec += s.exec_ns;
                } else {
                    ccd1_tasks += s.tasks_done;
                    ccd1_exec += s.exec_ns;
                }
            }
            const uint64_t mean_fin = W > 0 ? sum_fin / W : 0ull;
            std::printf(
                "LUISA_FALLBACK_SCHED_DISPATCH_JSON: "
                "{\"dispatch_idx\":%llu,\"workers\":%zu,\"tasks\":%u,\"chunk\":%u,"
                "\"wall_ns\":%llu,\"sum_exec_ns\":%llu,\"sum_wakeup_ns\":%llu,"
                "\"sum_fetch_overhead_ns\":%llu,\"sum_tail_idle_ns\":%llu,"
                "\"max_wakeup_ns\":%llu,\"min_finish_ns\":%llu,\"mean_finish_ns\":%llu,\"max_finish_ns\":%llu,"
                "\"max_task_ns\":%llu,\"max_task_id\":%u,"
                "\"straggler_tid\":%u,\"straggler_last_task_id\":%u,\"straggler_last_task_ns\":%llu,"
                "\"ccd0_tasks\":%llu,\"ccd1_tasks\":%llu,\"ccd0_exec_ns\":%llu,\"ccd1_exec_ns\":%llu}\n",
                static_cast<unsigned long long>(_dispatch_index++), W, count, _configured_chunk_size,
                static_cast<unsigned long long>(wall_ns),
                static_cast<unsigned long long>(sum_exec),
                static_cast<unsigned long long>(sum_wakeup),
                static_cast<unsigned long long>(sum_fetch_ovh),
                static_cast<unsigned long long>(sum_tail_idle),
                static_cast<unsigned long long>(max_wakeup),
                static_cast<unsigned long long>(min_fin == UINT64_MAX ? 0ull : min_fin),
                static_cast<unsigned long long>(mean_fin),
                static_cast<unsigned long long>(max_fin),
                static_cast<unsigned long long>(max_task), max_task_id,
                straggler_tid, straggler_last_id, static_cast<unsigned long long>(straggler_last_ns),
                static_cast<unsigned long long>(ccd0_tasks),
                static_cast<unsigned long long>(ccd1_tasks),
                static_cast<unsigned long long>(ccd0_exec),
                static_cast<unsigned long long>(ccd1_exec));
            std::fflush(stdout);
        }
    }
};

// ============================================================================
// 2. Optimized AkrFastThreadPool (Lock-free epoch + bounded spin-then-park +
//    single-last-worker completion + short-dispatch fast paths)
// ============================================================================
class AkrFastThreadPool {

private:
#ifdef __cpp_lib_hardware_interference_size
    static constexpr auto cache_line_size = std::hardware_destructive_interference_size;
#else
    static constexpr auto cache_line_size = 64u;
#endif

    struct alignas(64) WorkDesc {
        uint count = 0u;
        uint chunk_size = 1u;
        uint active_workers = 0u;
        bool is_worker_mode = false;
        bool use_work_stealing = true;
        bool spin_after_work = false;
        luisa::move_only_function<void(uint)> *item_task = nullptr;
        luisa::move_only_function<void(FallbackCommandQueue::ParallelClaimFn)> *worker_task = nullptr;
    };

    struct alignas(64) WorkerStealSlot {
        std::atomic_uint32_t next_chunk{0u};
        uint32_t end_chunk{0u};
    };

    luisa::vector<luisa::unique_ptr<Thread>> _threads;
    luisa::vector<WorkerStealSlot> _slots;
    std::mutex _submit_mutex;
    std::mutex _wake_mutex;
    std::condition_variable _has_work;
    std::mutex _done_mutex;
    std::condition_variable _work_done;

    WorkDesc _work = {};
    uint _configured_chunk = 1u;
    uint _worker_spin_iters = 4096u;
    uint _submit_spin_iters = 2048u;
    bool _guided_mode = false;
    bool _steal_enabled = true;

    alignas(cache_line_size) std::atomic_uint32_t _item_count{0u};
    alignas(cache_line_size) std::atomic_uint64_t _task_generation{0u};
    alignas(cache_line_size) std::atomic_uint32_t _remaining_workers{0u};
    alignas(cache_line_size) std::atomic_uint32_t _sleeping_workers{0u};
    alignas(cache_line_size) std::atomic_uint32_t _submitter_sleeping{0u};
    std::atomic_bool _stopped{false};

    [[nodiscard]] bool _claim_next_global(uint count, uint min_chunk, uint &out_begin, uint &out_end) noexcept {
        if (!_guided_mode) {
            const uint cur = _item_count.load(std::memory_order_relaxed);
            if (cur >= count) { return false; }
            const uint i = _item_count.fetch_add(min_chunk, std::memory_order_relaxed);
            if (i >= count) { return false; }
            out_begin = i;
            out_end = std::min<uint>(i + min_chunk, count);
            return true;
        }
        const size_t nw = _threads.size();
        uint cur = _item_count.load(std::memory_order_relaxed);
        while (cur < count) {
            const uint rem = count - cur;
            const uint denom = static_cast<uint>(std::max<size_t>(1u, nw * 4u));
            const uint step = std::min<uint>(rem, std::max<uint>(min_chunk, (rem + denom - 1u) / denom));
            if (_item_count.compare_exchange_weak(cur, cur + step, std::memory_order_relaxed, std::memory_order_relaxed)) {
                out_begin = cur;
                out_end = cur + step;
                return true;
            }
        }
        return false;
    }

    struct WorkerClaimCtx {
        AkrFastThreadPool *pool;
        uint tid;
        uint victim_offset;
        uint count;
        uint chunk;
        uint active_workers;
        bool use_steal;

        [[nodiscard]] bool next(uint &out_b, uint &out_e) noexcept {
            if (!use_steal) {
                return pool->_claim_next_global(count, chunk, out_b, out_e);
            }
            // Fast path: drain own L1-local contiguous range (1 atomic increment per local batch, or per chunk)
            auto &own = pool->_slots[tid];
            if (own.next_chunk.load(std::memory_order_relaxed) < own.end_chunk) {
                const uint c = own.next_chunk.fetch_add(1u, std::memory_order_relaxed);
                if (c < own.end_chunk) {
                    out_b = c * chunk;
                    out_e = std::min<uint>(out_b + chunk, count);
                    return true;
                }
            }
            // Steal path: dynamically steal 1 chunk at a time from remaining active workers
            while (victim_offset < active_workers) {
                const uint v = (tid + victim_offset) % active_workers;
                auto &vslot = pool->_slots[v];
                if (vslot.next_chunk.load(std::memory_order_relaxed) < vslot.end_chunk) {
                    const uint c = vslot.next_chunk.fetch_add(1u, std::memory_order_relaxed);
                    if (c < vslot.end_chunk) {
                        out_b = c * chunk;
                        out_e = std::min<uint>(out_b + chunk, count);
                        return true;
                    }
                }
                ++victim_offset;
            }
            return false;
        }

        static bool claim(void *p, uint &b, uint &e) noexcept {
            return static_cast<WorkerClaimCtx *>(p)->next(b, e);
        }
    };

    void _worker_loop(uint tid) noexcept {
        uint64_t last_gen = 0ull;
        bool allow_spin = false;
        for (;;) {
            // Phase 1: bounded user-space spin only after short micro-dispatches;
            // for full-grid renders, park immediately so finishing SMT siblings do
            // not execute PAUSE loops on cores still running straggler tiles.
            uint64_t cur_gen = _task_generation.load(std::memory_order_acquire);
            if (allow_spin && cur_gen == last_gen && !_stopped.load(std::memory_order_acquire)) {
                for (uint s = 0u; s < _worker_spin_iters; ++s) {
                    cpu_relax_pause();
                    cur_gen = _task_generation.load(std::memory_order_acquire);
                    if (cur_gen != last_gen || _stopped.load(std::memory_order_acquire)) {
                        break;
                    }
                }
            }
            // Phase 2: race-free sleep under _wake_mutex if still idle
            if (cur_gen == last_gen && !_stopped.load(std::memory_order_acquire)) {
                std::unique_lock lk{_wake_mutex};
                _sleeping_workers.fetch_add(1u, std::memory_order_seq_cst);
                _has_work.wait(lk, [this, last_gen] {
                    return _task_generation.load(std::memory_order_seq_cst) != last_gen ||
                           _stopped.load(std::memory_order_seq_cst);
                });
                _sleeping_workers.fetch_sub(1u, std::memory_order_relaxed);
                cur_gen = _task_generation.load(std::memory_order_acquire);
            }
            if (_stopped.load(std::memory_order_acquire)) {
                return;
            }
            last_gen = cur_gen;

            const WorkDesc w = _work;
            allow_spin = w.spin_after_work;
            if (tid < w.active_workers) {
                WorkerClaimCtx ctx{this, tid, 1u, w.count, w.chunk_size, w.active_workers, w.use_work_stealing};
                if (w.is_worker_mode) {
                    (*w.worker_task)(FallbackCommandQueue::ParallelClaimFn{&ctx, &WorkerClaimCtx::claim});
                } else {
                    uint b = 0u, e = 0u;
                    while (ctx.next(b, e)) {
                        for (uint j = b; j < e; ++j) {
                            (*w.item_task)(j);
                        }
                    }
                }
            }

            // Single-last-worker completion signal with seq_cst Dekker synchronization
            if (_remaining_workers.fetch_sub(1u, std::memory_order_seq_cst) == 1u) {
                if (_submitter_sleeping.load(std::memory_order_seq_cst) != 0u) {
                    { std::scoped_lock lk{_done_mutex}; }
                    _work_done.notify_one();
                }
            }
        }
    }

    void _dispatch_impl(WorkDesc &desc) noexcept {
        std::scoped_lock submit_lk{_submit_mutex};
        desc.use_work_stealing = _steal_enabled && !_guided_mode;
        desc.spin_after_work = desc.count < static_cast<uint>(_threads.size() * 4u);
        if (desc.use_work_stealing) {
            const uint n_chunks = (desc.count + desc.chunk_size - 1u) / desc.chunk_size;
            const uint aw = desc.active_workers;
            for (uint s = 0u; s < aw; ++s) {
                const uint cb = static_cast<uint>((static_cast<uint64_t>(s) * n_chunks) / aw);
                const uint ce = static_cast<uint>((static_cast<uint64_t>(s + 1u) * n_chunks) / aw);
                _slots[s].next_chunk.store(cb, std::memory_order_relaxed);
                _slots[s].end_chunk = ce;
            }
        } else {
            _item_count.store(0u, std::memory_order_relaxed);
        }
        _work = desc;
        _remaining_workers.store(static_cast<uint32_t>(_threads.size()), std::memory_order_relaxed);
        const uint64_t next_gen = _task_generation.load(std::memory_order_relaxed) + 1ull;
        _task_generation.store(next_gen, std::memory_order_seq_cst);
        if (_sleeping_workers.load(std::memory_order_seq_cst) != 0u) {
            { std::scoped_lock lk{_wake_mutex}; }
            _has_work.notify_all();
        }

        const uint sub_spins = desc.spin_after_work ? _submit_spin_iters : 0u;
        for (uint s = 0u; s < sub_spins; ++s) {
            if (_remaining_workers.load(std::memory_order_acquire) == 0u) {
                _work = {};
                return;
            }
            cpu_relax_pause();
        }
        if (_remaining_workers.load(std::memory_order_acquire) != 0u) {
            std::unique_lock done_lk{_done_mutex};
            _submitter_sleeping.store(1u, std::memory_order_seq_cst);
            _work_done.wait(done_lk, [this] {
                return _remaining_workers.load(std::memory_order_seq_cst) == 0u;
            });
            _submitter_sleeping.store(0u, std::memory_order_relaxed);
        }
        _work = {};
    }

public:
    explicit AkrFastThreadPool(size_t n_threads, bool enable_work_stealing = true) noexcept
        : _slots(std::max<size_t>(1u, n_threads)), _steal_enabled{enable_work_stealing} {
        _configured_chunk = query_parallel_chunk();
        _worker_spin_iters = query_spin_iters("LUISA_FALLBACK_SPIN_ITERS", 4096u);
        _submit_spin_iters = query_spin_iters("LUISA_FALLBACK_SUBMIT_SPIN_ITERS", 2048u);
        if (const auto *raw = std::getenv("LUISA_FALLBACK_SCHED_MODE")) {
            _guided_mode = std::string_view{raw} == "guided";
        }
        const size_t nw = std::max<size_t>(1u, n_threads);
        _threads.reserve(nw);
        for (size_t tid = 0; tid < nw; ++tid) {
            _threads.emplace_back(luisa::make_unique<Thread>(
                static_cast<uint>(tid), [this, tid] { _worker_loop(static_cast<uint>(tid)); }));
        }
    }

    ~AkrFastThreadPool() noexcept {
        {
            std::scoped_lock lk{_wake_mutex};
            _stopped.store(true, std::memory_order_seq_cst);
        }
        _has_work.notify_all();
        _threads.clear();
    }

    void parallel_for(uint count, luisa::move_only_function<void(uint)> &&task) noexcept {
        if (count == 0u) { return; }
        if (count == 1u || _threads.empty()) {
            for (uint i = 0u; i < count; ++i) { task(i); }
            return;
        }
        const uint chunk = _configured_chunk;
        const uint n_chunks = (count + chunk - 1u) / chunk;
        const uint active = std::min<uint>(static_cast<uint>(_threads.size()), n_chunks);
        WorkDesc desc{
            .count = count,
            .chunk_size = chunk,
            .active_workers = active,
            .is_worker_mode = false,
            .item_task = &task,
            .worker_task = nullptr,
        };
        _dispatch_impl(desc);
    }

    void parallel_workers(uint count, luisa::move_only_function<void(FallbackCommandQueue::ParallelClaimFn)> &&worker_fn) noexcept {
        if (count == 0u) { return; }
        if (count == 1u || _threads.empty()) {
            SingleRangeClaimState single{0u, count, false};
            worker_fn(FallbackCommandQueue::ParallelClaimFn{&single, &SingleRangeClaimState::claim});
            return;
        }
        const uint chunk = _configured_chunk;
        const uint n_chunks = (count + chunk - 1u) / chunk;
        const uint active = std::min<uint>(static_cast<uint>(_threads.size()), n_chunks);
        WorkDesc desc{
            .count = count,
            .chunk_size = chunk,
            .active_workers = active,
            .is_worker_mode = true,
            .item_task = nullptr,
            .worker_task = &worker_fn,
        };
        _dispatch_impl(desc);
    }
};

// ============================================================================
// 3. OneTBB Scheduler State (task_arena + partitioner + optional slot pinning)
// ============================================================================
class TbbSchedulerState {
#ifdef LUISA_FALLBACK_HAS_TBB
private:
    class PinningObserver final : public oneapi::tbb::task_scheduler_observer {
    public:
        explicit PinningObserver(oneapi::tbb::task_arena &arena) noexcept
            : oneapi::tbb::task_scheduler_observer(arena) {
            observe(true);
        }
        ~PinningObserver() override { observe(false); }
        void on_scheduler_entry(bool) override {
            const int slot = oneapi::tbb::this_task_arena::current_thread_index();
            if (slot < 0) { return; }
#ifdef LUISA_PLATFORM_WINDOWS
            uint dummy_lp = 0u;
            apply_current_or_target_thread_affinity(GetCurrentThread(), static_cast<uint>(slot), dummy_lp);
#elif !defined(LUISA_PLATFORM_APPLE)
            const std::string_view topo = []() noexcept -> std::string_view {
                if (const auto *raw = std::getenv("LUISA_FALLBACK_TOPOLOGY")) {
                    return std::string_view{raw};
                }
                return "unpinned";
            }();
            if (topo != "unpinned" && topo != "os_floating") {
                cpu_set_t allowed;
                CPU_ZERO(&allowed);
                uint chosen_cpu = static_cast<uint>(slot);
                if (pthread_getaffinity_np(pthread_self(), sizeof(cpu_set_t), &allowed) == 0) {
                    luisa::vector<uint> allowed_cpus;
                    for (uint c = 0u; c < CPU_SETSIZE; ++c) {
                        if (CPU_ISSET(c, &allowed)) { allowed_cpus.push_back(c); }
                    }
                    if (!allowed_cpus.empty()) {
                        const uint mapped = map_slot_to_logical_cpu(static_cast<uint>(slot), topo);
                        chosen_cpu = allowed_cpus[mapped % allowed_cpus.size()];
                    }
                }
                if (chosen_cpu < CPU_SETSIZE) {
                    cpu_set_t cpuset;
                    CPU_ZERO(&cpuset);
                    CPU_SET(chosen_cpu, &cpuset);
                    pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
                }
            }
#endif
        }
    };

    enum class PartitionerKind : uint8_t { Auto = 0, Simple = 1, Affinity = 2, Static = 3 };

    oneapi::tbb::task_arena _arena;
    luisa::unique_ptr<PinningObserver> _observer;
    oneapi::tbb::affinity_partitioner _affinity_part;
    uint _grain = 1u;
    PartitionerKind _part_kind = PartitionerKind::Auto;

public:
    explicit TbbSchedulerState(size_t num_threads) noexcept
        : _arena{static_cast<int>(std::max<size_t>(1u, num_threads))} {
        _grain = query_parallel_chunk();
        if (const auto *raw = std::getenv("LUISA_FALLBACK_TBB_PARTITIONER")) {
            const std::string_view sv{raw};
            if (sv == "simple") { _part_kind = PartitionerKind::Simple; }
            else if (sv == "affinity") { _part_kind = PartitionerKind::Affinity; }
            else if (sv == "static") { _part_kind = PartitionerKind::Static; }
        }
        _arena.initialize();
        const std::string_view topo = []() noexcept -> std::string_view {
            if (const auto *raw = std::getenv("LUISA_FALLBACK_TOPOLOGY")) {
                return std::string_view{raw};
            }
            return "unpinned";
        }();
        if (topo != "unpinned" && topo != "os_floating") {
            _observer = luisa::make_unique<PinningObserver>(_arena);
        }
    }

    template<typename RangeFn>
    void execute_range(uint n, RangeFn &&fn) noexcept {
        if (n == 0u) { return; }
        if (n == 1u) {
            fn(0u, 1u);
            return;
        }
        const oneapi::tbb::blocked_range<uint> range{0u, n, _grain};
        _arena.execute([&] {
            auto body = [&](const oneapi::tbb::blocked_range<uint> &r) noexcept {
                fn(r.begin(), r.end());
            };
            switch (_part_kind) {
                case PartitionerKind::Auto:
                    oneapi::tbb::parallel_for(range, body, oneapi::tbb::auto_partitioner{});
                    break;
                case PartitionerKind::Simple:
                    oneapi::tbb::parallel_for(range, body, oneapi::tbb::simple_partitioner{});
                    break;
                case PartitionerKind::Affinity:
                    oneapi::tbb::parallel_for(range, body, _affinity_part);
                    break;
                case PartitionerKind::Static:
                    oneapi::tbb::parallel_for(range, body, oneapi::tbb::static_partitioner{});
                    break;
            }
        });
    }
#else
public:
    explicit TbbSchedulerState(size_t) noexcept {}
    template<typename RangeFn>
    void execute_range(uint, RangeFn &&) noexcept {
        LUISA_ERROR_WITH_LOCATION("TBB support was not compiled into luisa-compute-backend-fallback.");
    }
#endif
};

}// namespace luisa::compute::fallback

#endif

namespace luisa::compute::fallback {

inline void FallbackCommandQueue::_run_dispatch_loop() noexcept {
#ifdef LUISA_PLATFORM_APPLE
    pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
#endif
#if defined(LUISA_FALLBACK_USE_AKARI_THREAD_POOL) && defined(LUISA_PLATFORM_WINDOWS)
    if (query_backend_scheduler_kind() == BackendSchedulerKind::Ppl) {
        const size_t hw = std::thread::hardware_concurrency();
        if (_worker_count > 0u && _worker_count != hw) {
            concurrency::SchedulerPolicy policy(
                2,
                concurrency::MinConcurrency, static_cast<unsigned int>(_worker_count),
                concurrency::MaxConcurrency, static_cast<unsigned int>(_worker_count));
            concurrency::CurrentScheduler::Create(policy);
        }
    }
#endif
    for (;;) {
        auto task = [this] {
            std::unique_lock lock{_mutex};
            _cv.wait(lock, [this] { return !_tasks.empty(); });
            auto task = std::move(_tasks.front());
            _tasks.pop();
            return task;
        }();
        if (!task) { break; }
        task();
        _total_finish_count.fetch_add(1u, std::memory_order_release);
        _total_finish_count.notify_all();
    }
#if defined(LUISA_FALLBACK_USE_DISPATCH_QUEUE)
    if (_dispatch_queue != nullptr) {
        dispatch_release(_dispatch_queue);
    }
#elif defined(LUISA_FALLBACK_USE_AKARI_THREAD_POOL)
    _worker_pool.reset();
    _fast_worker_pool.reset();
    _tbb_state.reset();
#endif
}

inline void FallbackCommandQueue::_wait_for_task_queue_available() const noexcept {
    if (_in_flight_limit == 0u) { return; }
    auto last_enqueue_count = _total_enqueue_count.load(std::memory_order_acquire);
    auto finish_count = _total_finish_count.load(std::memory_order_acquire);
    while (finish_count + _in_flight_limit <= last_enqueue_count) {
        _total_finish_count.wait(finish_count, std::memory_order_acquire);
        finish_count = _total_finish_count.load(std::memory_order_acquire);
    }
}

inline void FallbackCommandQueue::_enqueue_task_no_wait(luisa::move_only_function<void()> &&task) noexcept {
    {
        std::scoped_lock lock{_mutex};
        _total_enqueue_count.fetch_add(1u, std::memory_order_release);
        _tasks.push(std::move(task));
    }
    _cv.notify_one();
}

FallbackCommandQueue::FallbackCommandQueue(size_t in_flight_limit, size_t num_threads [[maybe_unused]]) noexcept
    : _in_flight_limit{in_flight_limit}, _worker_count{num_threads} {
    if (_worker_count == 0u) {
        if (const auto *raw = std::getenv("LUISA_FALLBACK_THREADS")) {
            std::string_view sv{raw};
            size_t parsed = 0u;
            const auto [ptr, ec] = std::from_chars(sv.data(), sv.data() + sv.size(), parsed);
            if (ec == std::errc{} && ptr == sv.data() + sv.size() && parsed > 0u) {
                _worker_count = parsed;
            }
        }
        if (_worker_count == 0u) {
            _worker_count = std::thread::hardware_concurrency();
        }
    }
    _dispatcher = std::thread{[this] { _run_dispatch_loop(); }};
}

FallbackCommandQueue::~FallbackCommandQueue() noexcept {
    _enqueue_task_no_wait({});
    _dispatcher.join();
}

void FallbackCommandQueue::synchronize() noexcept {
    if (query_use_orig_queue_sync()) {
        auto finished = false;
        _enqueue_task_no_wait([this, &finished] {
            {
                std::scoped_lock lock{_synchronize_mutex};
                finished = true;
            }
            _synchronize_cv.notify_one();
        });
        std::unique_lock lock{_synchronize_mutex};
        _synchronize_cv.wait(lock, [&finished] { return finished; });
        return;
    }
    size_t target = 0u;
    {
        std::scoped_lock lock{_mutex};
        target = _total_enqueue_count.load(std::memory_order_acquire);
    }
    for (uint32_t spin = 0u; spin < 256u; ++spin) {
        if (_total_finish_count.load(std::memory_order_acquire) >= target) {
            return;
        }
        cpu_relax_pause();
    }
    auto finish = _total_finish_count.load(std::memory_order_acquire);
    while (finish < target) {
        _total_finish_count.wait(finish, std::memory_order_acquire);
        finish = _total_finish_count.load(std::memory_order_acquire);
    }
}

void FallbackCommandQueue::enqueue(luisa::move_only_function<void()> &&task) noexcept {
    _wait_for_task_queue_available();
    _enqueue_task_no_wait(std::move(task));
}

void FallbackCommandQueue::enqueue_parallel(uint n, luisa::move_only_function<void(uint)> &&task) noexcept {
    static auto LUISA_SINGLE_THREADING = [] {
        using namespace std::string_view_literals;
        auto env = getenv("LUISA_SINGLE_THREADING");
        return env != nullptr && env == "1"sv;
    }();
    enqueue([this, n, task = std::move(task)]() mutable noexcept {
        if (LUISA_SINGLE_THREADING) {
            for (auto i = 0u; i < n; ++i) { task(i); }
            return;
        }
#if defined(LUISA_FALLBACK_USE_DISPATCH_QUEUE)
        if (_dispatch_queue == nullptr) {
#ifdef LUISA_PLATFORM_APPLE
            _dispatch_queue = dispatch_get_global_queue(QOS_CLASS_USER_INTERACTIVE, 0);
#else
            _dispatch_queue = dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_HIGH, 0);
#endif
            dispatch_retain(_dispatch_queue);
        }
        dispatch_apply_f(n, _dispatch_queue, &task, [](void *context, size_t idx) noexcept {
            auto task = static_cast<luisa::move_only_function<void(uint)> *>(context);
            (*task)(static_cast<uint>(idx));
        });
#elif defined(LUISA_FALLBACK_USE_PPL)
        concurrency::parallel_for(0u, n, task);
#elif defined(LUISA_FALLBACK_USE_TBB)
        tbb::parallel_for(0u, n, task);
#elif defined(LUISA_FALLBACK_USE_AKARI_THREAD_POOL)
        const auto sched_kind = query_backend_scheduler_kind();
        if (sched_kind == BackendSchedulerKind::AkariOrig) {
            if (_worker_pool == nullptr) {
                _worker_pool = luisa::make_unique<AkrThreadPool>(_worker_count);
            }
            _worker_pool->parallel_for(n, std::move(task));
        } else if (sched_kind == BackendSchedulerKind::AkariOpt || sched_kind == BackendSchedulerKind::AkariWorkSteal) {
            if (_fast_worker_pool == nullptr) {
                _fast_worker_pool = luisa::make_unique<AkrFastThreadPool>(
                    _worker_count, sched_kind == BackendSchedulerKind::AkariWorkSteal);
            }
            _fast_worker_pool->parallel_for(n, std::move(task));
        } else if (sched_kind == BackendSchedulerKind::Tbb) {
            if (_tbb_state == nullptr) {
                _tbb_state = luisa::make_unique<TbbSchedulerState>(_worker_count);
            }
            _tbb_state->execute_range(n, [&](uint b, uint e) noexcept {
                for (uint i = b; i < e; ++i) { task(i); }
            });
        } else {
#ifdef LUISA_PLATFORM_WINDOWS
            if (n == 0u) { return; }
            if (n == 1u) { task(0u); return; }
            const uint chunk = query_parallel_chunk();
            static const std::string_view ppl_part = []() noexcept -> std::string_view {
                if (const auto *raw = std::getenv("LUISA_FALLBACK_PPL_PARTITIONER")) {
                    return std::string_view{raw};
                }
                return "auto";
            }();
            if (chunk > 1u || ppl_part == "chunked") {
                const uint c = std::max<uint>(1u, chunk);
                const uint n_chunks = (n + c - 1u) / c;
                if (ppl_part == "static") {
                    concurrency::parallel_for(0u, n_chunks, [&](uint cid) noexcept {
                        const uint b = cid * c;
                        const uint e = std::min<uint>(b + c, n);
                        for (uint i = b; i < e; ++i) { task(i); }
                    }, concurrency::static_partitioner{});
                } else {
                    concurrency::parallel_for(0u, n_chunks, [&](uint cid) noexcept {
                        const uint b = cid * c;
                        const uint e = std::min<uint>(b + c, n);
                        for (uint i = b; i < e; ++i) { task(i); }
                    });
                }
            } else if (ppl_part == "static") {
                concurrency::parallel_for(0u, n, task, concurrency::static_partitioner{});
            } else {
                concurrency::parallel_for(0u, n, task);
            }
#endif
        }
#endif
    });
}

void FallbackCommandQueue::enqueue_parallel_workers(
    uint n, luisa::move_only_function<void(ParallelClaimFn)> &&worker_fn) noexcept {
    static auto LUISA_SINGLE_THREADING = [] {
        using namespace std::string_view_literals;
        auto env = getenv("LUISA_SINGLE_THREADING");
        return env != nullptr && env == "1"sv;
    }();
    enqueue([this, n, worker_fn = std::move(worker_fn)]() mutable noexcept {
        if (LUISA_SINGLE_THREADING) {
            SingleRangeClaimState single{0u, n, false};
            worker_fn(ParallelClaimFn{&single, &SingleRangeClaimState::claim});
            return;
        }
#if defined(LUISA_FALLBACK_USE_DISPATCH_QUEUE)
        if (n == 0u) { return; }
        if (_dispatch_queue == nullptr) {
#ifdef LUISA_PLATFORM_APPLE
            _dispatch_queue = dispatch_get_global_queue(QOS_CLASS_USER_INTERACTIVE, 0);
#else
            _dispatch_queue = dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_HIGH, 0);
#endif
            dispatch_retain(_dispatch_queue);
        }
        const uint chunk = std::max<uint>(1u, n / std::max<uint>(1u, static_cast<uint>(_worker_count * 8u)));
        const uint n_chunks = (n + chunk - 1u) / chunk;
        struct GcdCtx {
            luisa::move_only_function<void(ParallelClaimFn)> *fn;
            uint n;
            uint chunk;
        } ctx{&worker_fn, n, chunk};
        dispatch_apply_f(n_chunks, _dispatch_queue, &ctx, [](void *p, size_t cid) noexcept {
            auto *c = static_cast<GcdCtx *>(p);
            const uint b = static_cast<uint>(cid) * c->chunk;
            const uint e = std::min<uint>(b + c->chunk, c->n);
            SingleRangeClaimState single{b, e, false};
            (*c->fn)(ParallelClaimFn{&single, &SingleRangeClaimState::claim});
        });
#elif defined(LUISA_FALLBACK_USE_PPL)
        if (n == 0u) { return; }
        concurrency::parallel_for(0u, n, [&](uint idx) noexcept {
            SingleRangeClaimState single{idx, idx + 1u, false};
            worker_fn(ParallelClaimFn{&single, &SingleRangeClaimState::claim});
        });
#elif defined(LUISA_FALLBACK_USE_TBB)
        if (n == 0u) { return; }
        tbb::parallel_for(tbb::blocked_range<uint>(0u, n), [&](const tbb::blocked_range<uint> &r) noexcept {
            SingleRangeClaimState single{r.begin(), r.end(), false};
            worker_fn(ParallelClaimFn{&single, &SingleRangeClaimState::claim});
        });
#elif defined(LUISA_FALLBACK_USE_AKARI_THREAD_POOL)
        const auto sched_kind = query_backend_scheduler_kind();
        if (sched_kind == BackendSchedulerKind::AkariOpt || sched_kind == BackendSchedulerKind::AkariWorkSteal) {
            if (_fast_worker_pool == nullptr) {
                _fast_worker_pool = luisa::make_unique<AkrFastThreadPool>(
                    _worker_count, sched_kind == BackendSchedulerKind::AkariWorkSteal);
            }
            _fast_worker_pool->parallel_workers(n, std::move(worker_fn));
        } else if (sched_kind == BackendSchedulerKind::Tbb) {
            if (_tbb_state == nullptr) {
                _tbb_state = luisa::make_unique<TbbSchedulerState>(_worker_count);
            }
            _tbb_state->execute_range(n, [&](uint b, uint e) noexcept {
                SingleRangeClaimState single{b, e, false};
                worker_fn(ParallelClaimFn{&single, &SingleRangeClaimState::claim});
            });
        } else if (sched_kind == BackendSchedulerKind::Ppl) {
#ifdef LUISA_PLATFORM_WINDOWS
            if (n == 0u) { return; }
            if (n == 1u) {
                SingleRangeClaimState single{0u, 1u, false};
                worker_fn(ParallelClaimFn{&single, &SingleRangeClaimState::claim});
                return;
            }
            const uint chunk = query_parallel_chunk();
            static const std::string_view ppl_part = []() noexcept -> std::string_view {
                if (const auto *raw = std::getenv("LUISA_FALLBACK_PPL_PARTITIONER")) {
                    return std::string_view{raw};
                }
                return "auto";
            }();
            const uint c = std::max<uint>(1u, chunk);
            const uint n_chunks = (n + c - 1u) / c;
            if (ppl_part == "static") {
                concurrency::parallel_for(0u, n_chunks, [&](uint cid) noexcept {
                    SingleRangeClaimState single{cid * c, std::min<uint>((cid + 1u) * c, n), false};
                    worker_fn(ParallelClaimFn{&single, &SingleRangeClaimState::claim});
                }, concurrency::static_partitioner{});
            } else {
                concurrency::parallel_for(0u, n_chunks, [&](uint cid) noexcept {
                    SingleRangeClaimState single{cid * c, std::min<uint>((cid + 1u) * c, n), false};
                    worker_fn(ParallelClaimFn{&single, &SingleRangeClaimState::claim});
                });
            }
#endif
        } else {
            if (_worker_pool == nullptr) {
                _worker_pool = luisa::make_unique<AkrThreadPool>(_worker_count);
            }
            _worker_pool->parallel_for(n, [&](uint idx) noexcept {
                SingleRangeClaimState single{idx, idx + 1u, false};
                worker_fn(ParallelClaimFn{&single, &SingleRangeClaimState::claim});
            });
        }
#else
        SingleRangeClaimState single{0u, n, false};
        worker_fn(ParallelClaimFn{&single, &SingleRangeClaimState::claim});
#endif
    });
}

}// namespace luisa::compute::fallback

// Benchmark: reference-count-block pools.
//
// The runtime's shared-resource path allocates one 8-byte reference-count block per
// resource that gets copied and recycles it when the last handle dies. This benchmark
// measures the two candidate recycle bins on exactly that workload:
//
//   * `luisa::Pool<T, true>`      -- LIFO free list guarded by `luisa::spin_mutex`
//   * `luisa::ConcurrentPool<T>`  -- lock-free, one compare-and-swap per operation
//
// plus the general-purpose allocator as a baseline, so the win over
// `new_with_allocator`/`delete_with_allocator` is visible too.
//
// Each "operation" counted below is one allocate plus one deallocate, which is one
// resource promotion followed by one block release. `live` is the number of blocks a
// thread keeps checked out, i.e. how much of the pool it works inside:
//
//   * live = 0 -> allocate immediately followed by deallocate: all threads hammer the
//                 same slot, the worst case for both implementations;
//   * live = 8 -> each thread owns a small working set, the realistic case.
//
// Run: `./bin/debug/benchmark_refcount_pool.exe [threads...]`

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <thread>
#include <utility>
#include <vector>

#include <luisa/core/pool.h>
#include <luisa/core/stl/memory.h>

namespace {

/// Same shape as `luisa::compute::ResourceRefCount`.
struct RefCount {
    std::atomic<int64_t> count{1};
    std::atomic<uint64_t> state{0u};
};
static_assert(sizeof(RefCount) == 16u, "the benchmark must model the 16-byte block");

using Clock = std::chrono::steady_clock;

constexpr auto live_set_sizes = std::array<uint32_t, 2u>{0u, 8u};

template<typename Pool>
[[nodiscard]] auto pool_ops(Pool &pool) noexcept {
    return std::pair{
        [&pool] { return pool.create(); },
        [&pool](RefCount *p) { pool.destroy(p); }};
}

[[nodiscard]] auto allocator_ops() noexcept {
    return std::pair{
        [] { return luisa::new_with_allocator<RefCount>(); },
        [](RefCount *p) { luisa::delete_with_allocator(p); }};
}

/// Returns millions of allocate+deallocate pairs per second.
template<uint32_t live, typename Alloc, typename Free>
auto run(uint32_t thread_count, uint64_t iterations,
                       Alloc &&alloc, Free &&free_fn) noexcept {
    std::atomic<bool> start{false};
    std::vector<std::thread> threads;
    std::vector<uint64_t> sinks(thread_count, 0u);
    threads.reserve(thread_count);
    for (auto t = 0u; t < thread_count; t++) {
        threads.emplace_back([&, t] {
            std::array<RefCount *, live == 0u ? 1u : live> held{};
            if constexpr (live > 0u) {
                for (auto &p : held) { p = alloc(); }
            }
            while (!start.load(std::memory_order_acquire)) { std::this_thread::yield(); }
            uint64_t sink = 0u;
            if constexpr (live == 0u) {
                for (uint64_t i = 0u; i < iterations; i++) {
                    auto p = alloc();
                    sink += reinterpret_cast<uintptr_t>(p);
                    free_fn(p);
                }
            } else {
                for (uint64_t i = 0u; i < iterations; i++) {
                    auto p = alloc();
                    sink += reinterpret_cast<uintptr_t>(p);
                    auto slot = static_cast<size_t>(i % live);
                    free_fn(held[slot]);
                    held[slot] = p;
                }
                for (auto p : held) { free_fn(p); }
            }
            sinks[t] = sink;// keeps the compiler from eliding the whole loop
        });
    }
    auto begin = Clock::now();
    start.store(true, std::memory_order_release);
    for (auto &thread : threads) { thread.join(); }
    auto end = Clock::now();
    double seconds = std::chrono::duration<double>(end - begin).count();
    uint64_t sink = 0u;
    for (auto s : sinks) { sink += s; }
    if (sink == 0u) { std::fprintf(stderr, "unreachable\n"); }
    auto pairs = static_cast<double>(thread_count) * static_cast<double>(iterations);
    return pairs / seconds / 1e6;
}

void report(const char *name, uint32_t live, uint32_t threads, double mops) {
    std::printf("%-22s live=%-2u threads=%-3u %8.2f Mops/s\n", name, live, threads, mops);
}

}// namespace

int main(int argc, char *argv[]) {
    std::vector<uint32_t> thread_counts{1u, 2u, 4u, 8u, 16u};
    if (argc > 1) {
        thread_counts.clear();
        for (auto i = 1; i < argc; i++) {
            thread_counts.emplace_back(static_cast<uint32_t>(std::atoi(argv[i])));
        }
    }

    luisa::Pool<RefCount, true> mutex_pool;
    luisa::ConcurrentPool<RefCount> cas_pool;

    // Keep the total amount of work roughly constant so the runs stay comparable.
    constexpr auto total_work = uint64_t{8u} << 20u;

    std::printf("\n=== reference-count block pool benchmark (one op = alloc + free) ===\n");
    for (auto live : live_set_sizes) {
        std::printf("\n--- live set = %u ---\n", live);
        for (auto threads : thread_counts) {
            auto iterations = std::max<uint64_t>(total_work / threads, 1u << 16u);
            // warm up both pools so the first timed run does not pay for block growth
            if (live == 0u) {
                auto [a, f] = pool_ops(cas_pool);
                run<0u>(threads, iterations / 8u, a, f);
                auto [a2, f2] = pool_ops(mutex_pool);
                run<0u>(threads, iterations / 8u, a2, f2);
                auto [a3, f3] = allocator_ops();
                run<0u>(threads, iterations / 8u, a3, f3);
            } else {
                auto [a, f] = pool_ops(cas_pool);
                run<8u>(threads, iterations / 8u, a, f);
                auto [a2, f2] = pool_ops(mutex_pool);
                run<8u>(threads, iterations / 8u, a2, f2);
                auto [a3, f3] = allocator_ops();
                run<8u>(threads, iterations / 8u, a3, f3);
            }
            // measured runs
            {
                auto [a, f] = pool_ops(cas_pool);
                report("ConcurrentPool (CAS)", live, threads,
                       live == 0u ? run<0u>(threads, iterations, a, f)
                                  : run<8u>(threads, iterations, a, f));
            }
            {
                auto [a, f] = pool_ops(mutex_pool);
                report("Pool (spin_mutex)", live, threads,
                       live == 0u ? run<0u>(threads, iterations, a, f)
                                  : run<8u>(threads, iterations, a, f));
            }
            {
                auto [a, f] = allocator_ops();
                report("new_with_allocator", live, threads,
                       live == 0u ? run<0u>(threads, iterations, a, f)
                                  : run<8u>(threads, iterations, a, f));
            }
        }
    }
    std::printf("\n");
    return 0;
}

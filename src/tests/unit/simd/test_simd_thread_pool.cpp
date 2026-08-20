#include "ut/ut.hpp"

#include <array>
#include <atomic>
#include <chrono>
#include <mutex>
#include <thread>

#include "simd_thread_pool.h"

using namespace luisa::compute::simd;
using namespace boost::ut;
using namespace boost::ut::literals;

int main(int argc, char *argv[]) {
    boost::ut::detail::cfg::parse_arg_with_fallback(
        argc, const_cast<const char **>(argv));

    "parallel ranges execute exactly once on multiple persistent workers"_test = [] {
        constexpr auto item_count = 257u;
        std::array<std::atomic_uint32_t, item_count> visits{};
        std::mutex worker_ids_mutex;
        std::array<std::thread::id, 4u> worker_ids{};
        auto worker_id_count = uint32_t{0u};
        SIMDThreadPool pool{4u};
        pool.parallel_for(
            item_count, 1u,
            [&](uint64_t begin, uint64_t end) noexcept {
                {
                    std::scoped_lock lock{worker_ids_mutex};
                    auto id = std::this_thread::get_id();
                    auto found = false;
                    for (auto i = 0u; i < worker_id_count; i++) {
                        found |= worker_ids[i] == id;
                    }
                    if (!found && worker_id_count < worker_ids.size()) {
                        worker_ids[worker_id_count++] = id;
                    }
                }
                for (auto i = begin; i < end; i++) {
                    visits[i].fetch_add(1u, std::memory_order_relaxed);
                }
                std::this_thread::sleep_for(
                    std::chrono::microseconds{50});
            });

        expect(pool.worker_count() == 4u);
        expect(worker_id_count >= 2u)
            << "parallel work should engage multiple persistent workers";
        for (auto &visit : visits) {
            expect(visit.load(std::memory_order_relaxed) == 1u)
                << "every range element must execute exactly once";
        }
    };

    "serial and empty ranges preserve synchronous semantics"_test = [] {
        SIMDThreadPool pool{1u};
        auto caller = std::this_thread::get_id();
        auto executed = uint32_t{0u};
        auto executor = std::thread::id{};
        pool.parallel_for(
            19u, 4u,
            [&](uint64_t begin, uint64_t end) noexcept {
                expect(begin == 0u);
                expect(end == 19u);
                executed += static_cast<uint32_t>(end - begin);
                executor = std::this_thread::get_id();
            });
        expect(executed == 19u);
        expect(executor == caller)
            << "a one-worker pool should execute inline";

        auto empty_invocations = uint32_t{0u};
        pool.parallel_for(
            0u, 1u,
            [&](uint64_t, uint64_t) noexcept {
                empty_invocations++;
            });
        expect(empty_invocations == 0u);
    };

    "concurrent submissions are serialized without losing work"_test = [] {
        constexpr auto item_count = 193u;
        SIMDThreadPool pool{4u};
        std::array<std::atomic_uint32_t, item_count> first{};
        std::array<std::atomic_uint32_t, item_count> second{};
        auto submit = [&](auto &visits) noexcept {
            pool.parallel_for(
                item_count, 3u,
                [&](uint64_t begin, uint64_t end) noexcept {
                    for (auto i = begin; i < end; i++) {
                        visits[i].fetch_add(
                            1u, std::memory_order_relaxed);
                    }
                });
        };
        std::thread a{[&] { submit(first); }};
        std::thread b{[&] { submit(second); }};
        a.join();
        b.join();
        for (auto i = 0u; i < item_count; i++) {
            expect(first[i].load(std::memory_order_relaxed) == 1u);
            expect(second[i].load(std::memory_order_relaxed) == 1u);
        }
    };
}

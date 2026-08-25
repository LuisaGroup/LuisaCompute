#include "cohort_scheduler.h"

#include <array>
#include <cstddef>
#include <iostream>
#include <string_view>

using namespace luisa::compute::simd::schedule;

namespace {

[[nodiscard]] bool check(bool condition, const char *expression,
                         const char *file, int line) noexcept {
    if (!condition) {
        std::cerr << file << ':' << line << ": check failed: "
                  << expression << '\n';
    }
    return condition;
}

#define CHECK(EXPR)                                                           \
    do {                                                                      \
        if (!check(static_cast<bool>(EXPR), #EXPR, __FILE__, __LINE__)) {     \
            return false;                                                     \
        }                                                                     \
    } while (false)

template<size_t Width>
[[nodiscard]] LaneMask<Width> alternating_mask(bool even) noexcept {
    LaneMask<Width> result;
    for (auto lane = size_t{0u}; lane < Width; lane++) {
        if ((lane % 2u == 0u) == even) { result.set(lane); }
    }
    return result;
}

[[nodiscard]] bool test_lane_mask() noexcept {
    using Mask = LaneMask<128u>;
    constexpr auto full = Mask::full();
    constexpr auto first_65 = Mask::first_n(65u);
    static_assert(full.count() == 128u);
    static_assert(first_65.count() == 65u);
    static_assert(first_65.word(0u) == ~uint64_t{0u});
    static_assert(first_65.word(1u) == 1u);

    auto sparse = Mask::from_indices({0u, 1u, 6u, 64u, 127u, 200u});
    CHECK(sparse.count() == 5u);
    CHECK(sparse.first() == 0u);
    CHECK(sparse.test(127u));
    CHECK(!sparse.test(126u));
    CHECK((sparse & Mask::first_n(64u)).count() == 3u);
    CHECK((~Mask{}).count() == 128u);
    CHECK((Mask::full() - sparse).count() == 123u);
    return true;
}

template<size_t Width>
[[nodiscard]] bool run_diamond(SchedulingPolicy policy) noexcept {
    using Scheduler = CohortScheduler<Width>;
    using Mask = typename Scheduler::Mask;

    constexpr Continuation entry{0u, 0u, 0u};
    constexpr Continuation even_arm{1u, 1u, 0u};
    constexpr Continuation odd_arm{2u, 1u, 0u};
    constexpr Continuation join{3u, 7u, 0u};

    Scheduler scheduler{Mask::full(), policy};
    std::array<uint32_t, Width> values{};
    CHECK(scheduler.enqueue(entry, Mask::full()) == EnqueueResult::inserted);

    auto join_seen = false;
    auto steps = size_t{0u};
    while (auto cohort = scheduler.take()) {
        CHECK(++steps < 32u);
        switch (cohort->continuation.pc) {
            case 0u: {
                CHECK(scheduler.declare_convergence(join, cohort->mask));
                auto even = cohort->mask & alternating_mask<Width>(true);
                auto odd = cohort->mask & alternating_mask<Width>(false);
                auto even_result = scheduler.enqueue(even_arm, even);
                auto odd_result = scheduler.enqueue(odd_arm, odd);
                CHECK(even_result == EnqueueResult::inserted ||
                      even_result == EnqueueResult::empty);
                CHECK(odd_result == EnqueueResult::inserted ||
                      odd_result == EnqueueResult::empty);
                break;
            }
            case 1u:
            case 2u: {
                auto value = cohort->continuation.pc == 1u ? 10u : 20u;
                cohort->mask.for_each(
                    [&](auto lane) noexcept { values[lane] = value + lane; });
                auto result = scheduler.arrive(join, cohort->mask);
                CHECK(result == ConvergenceResult::waiting ||
                      result == ConvergenceResult::released);
                break;
            }
            case 3u: {
                CHECK(cohort->continuation == join);
                CHECK(cohort->mask == Mask::full());
                join_seen = true;
                CHECK(scheduler.terminate(cohort->mask));
                break;
            }
            default: CHECK(false);
        }
    }
    CHECK(join_seen);
    CHECK(scheduler.complete());
    for (auto lane = size_t{0u}; lane < Width; lane++) {
        auto expected = (lane % 2u == 0u ? 10u : 20u) + lane;
        CHECK(values[lane] == expected);
    }
    return true;
}

template<size_t Width>
[[nodiscard]] bool run_early_return(SchedulingPolicy policy) noexcept {
    using Scheduler = CohortScheduler<Width>;
    using Mask = typename Scheduler::Mask;

    constexpr Continuation entry{0u, 0u, 0u};
    constexpr Continuation returning{1u, 1u, 0u};
    constexpr Continuation continuing{2u, 1u, 0u};
    constexpr Continuation join{3u, 11u, 0u};

    Scheduler scheduler{Mask::full(), policy};
    CHECK(scheduler.enqueue(entry, Mask::full()) == EnqueueResult::inserted);
    auto join_mask = Mask{};
    while (auto cohort = scheduler.take()) {
        switch (cohort->continuation.pc) {
            case 0u: {
                CHECK(scheduler.declare_convergence(join, cohort->mask));
                auto return_mask = cohort->mask & Mask::single(0u);
                auto continue_mask = cohort->mask - return_mask;
                auto return_result = scheduler.enqueue(returning, return_mask);
                auto continue_result = scheduler.enqueue(continuing, continue_mask);
                CHECK(return_result == EnqueueResult::inserted);
                CHECK(continue_result == EnqueueResult::inserted ||
                      continue_result == EnqueueResult::empty);
                break;
            }
            case 1u:
                CHECK(scheduler.terminate(cohort->mask));
                break;
            case 2u: {
                auto result = scheduler.arrive(join, cohort->mask);
                CHECK(result == ConvergenceResult::waiting ||
                      result == ConvergenceResult::released);
                break;
            }
            case 3u:
                join_mask = cohort->mask;
                CHECK(scheduler.terminate(cohort->mask));
                break;
            default: CHECK(false);
        }
    }
    CHECK(scheduler.complete());
    CHECK(join_mask == (Mask::full() - Mask::single(0u)));
    return true;
}

template<size_t Width>
struct LoopResult {
    bool ok{false};
    std::array<uint32_t, Width> iterations{};
};

template<size_t Width>
[[nodiscard]] LoopResult<Width> run_loop(
    SchedulingPolicy policy) noexcept {
    using Scheduler = CohortScheduler<Width>;
    using Mask = typename Scheduler::Mask;

    constexpr Continuation entry{0u, 0u, 0u};
    constexpr Continuation loop_exit{3u, 23u, 0u};
    Scheduler scheduler{Mask::full(), policy};
    std::array<uint32_t, Width> remaining{};
    std::array<uint32_t, Width> initial{};
    std::array<uint32_t, Width> iterations{};
    for (auto lane = size_t{0u}; lane < Width; lane++) {
        initial[lane] = static_cast<uint32_t>((lane + 1u) % 4u);
        remaining[lane] = initial[lane];
    }
    if (scheduler.enqueue(entry, Mask::full()) != EnqueueResult::inserted) {
        return {};
    }

    auto exit_seen = false;
    auto steps = size_t{0u};
    while (auto cohort = scheduler.take()) {
        if (++steps >= 256u) { return {}; }
        switch (cohort->continuation.pc) {
            case 0u: {
                if (!scheduler.declare_convergence(loop_exit, cohort->mask)) {
                    return {};
                }
                if (scheduler.enqueue(
                        Continuation{1u, 17u, 0u},
                        cohort->mask) != EnqueueResult::inserted) {
                    return {};
                }
                break;
            }
            case 1u: {
                Mask done;
                Mask running;
                cohort->mask.for_each([&](auto lane) noexcept {
                    (remaining[lane] == 0u ? done : running).set(lane);
                });
                if (done.any()) {
                    auto result = scheduler.arrive(loop_exit, done);
                    if (result != ConvergenceResult::waiting &&
                        result != ConvergenceResult::released) {
                        return {};
                    }
                }
                if (running.any()) {
                    auto result = scheduler.enqueue(
                        Continuation{2u, 17u,
                                     cohort->continuation.loop_epoch},
                        running);
                    if (result != EnqueueResult::inserted &&
                        result != EnqueueResult::merged) {
                        return {};
                    }
                }
                break;
            }
            case 2u: {
                cohort->mask.for_each([&](auto lane) noexcept {
                    --remaining[lane];
                    ++iterations[lane];
                });
                auto result = scheduler.enqueue(
                    Continuation{1u, 17u,
                                 cohort->continuation.loop_epoch + 1u},
                    cohort->mask);
                if (result != EnqueueResult::inserted &&
                    result != EnqueueResult::merged) {
                    return {};
                }
                break;
            }
            case 3u:
                if (cohort->continuation != loop_exit ||
                    cohort->mask != Mask::full() ||
                    !scheduler.terminate(cohort->mask)) {
                    return {};
                }
                exit_seen = true;
                break;
            default: return {};
        }
    }
    if (!exit_seen || !scheduler.complete() || iterations != initial) {
        return {};
    }
    return {.ok = true, .iterations = iterations};
}

template<size_t Width>
[[nodiscard]] bool test_loop_policy_independence() noexcept {
    auto depth_first = run_loop<Width>(SchedulingPolicy::depth_first);
    auto largest = run_loop<Width>(SchedulingPolicy::largest_cohort);
    CHECK(depth_first.ok);
    CHECK(largest.ok);
    CHECK(depth_first.iterations == largest.iterations);
    return true;
}

template<size_t Width>
[[nodiscard]] bool test_collective_epoch_separation() noexcept {
    static_assert(Width >= 4u);
    using Scheduler = CohortScheduler<Width>;
    using Mask = typename Scheduler::Mask;
    auto epoch_3 = alternating_mask<Width>(true);
    auto epoch_4 = alternating_mask<Width>(false);
    Scheduler scheduler{Mask::full(), SchedulingPolicy::largest_cohort};
    CHECK(scheduler.enqueue(
              Continuation{42u, 9u, 3u}, epoch_3) ==
          EnqueueResult::inserted);
    CHECK(scheduler.enqueue(
              Continuation{42u, 9u, 4u}, epoch_4) ==
          EnqueueResult::inserted);
    CHECK(scheduler.ready_count() == 2u);

    auto first = scheduler.take();
    auto second = scheduler.take();
    CHECK(first.has_value());
    CHECK(second.has_value());
    CHECK(first->continuation.pc == second->continuation.pc);
    CHECK(first->continuation.convergence_token ==
          second->continuation.convergence_token);
    CHECK(first->continuation.loop_epoch !=
          second->continuation.loop_epoch);
    CHECK(!first->mask.intersects(second->mask));
    CHECK((first->mask | second->mask) == Mask::full());
    CHECK(scheduler.terminate(first->mask));
    CHECK(scheduler.terminate(second->mask));
    CHECK(scheduler.complete());
    return true;
}

template<size_t Width>
[[nodiscard]] bool test_partial_warp() noexcept {
    using Scheduler = CohortScheduler<Width>;
    using Mask = typename Scheduler::Mask;
    constexpr auto active_count = Width > 2u ? Width - 2u : Width;
    auto initial = Mask::first_n(active_count);
    Scheduler scheduler{initial};
    CHECK(scheduler.enqueue(Continuation{0u, 0u, 0u}, Mask::full()) ==
          EnqueueResult::inserted);
    auto cohort = scheduler.take();
    CHECK(cohort.has_value());
    CHECK(cohort->mask == initial);
    CHECK(cohort->mask.count() == active_count);
    CHECK(scheduler.terminate(cohort->mask));
    CHECK(scheduler.complete());
    return true;
}

[[nodiscard]] bool test_scheduler_invariants() noexcept {
    using Scheduler = CohortScheduler<4u>;
    using Mask = Scheduler::Mask;
    constexpr Continuation first{1u, 1u, 0u};
    constexpr Continuation second{2u, 1u, 0u};
    Scheduler scheduler;
    CHECK(scheduler.enqueue(first, Mask::single(0u)) ==
          EnqueueResult::inserted);
    CHECK(scheduler.enqueue(first, Mask::single(1u)) ==
          EnqueueResult::merged);
    CHECK(scheduler.enqueue(first, Mask::single(1u)) ==
          EnqueueResult::conflict);
    CHECK(scheduler.enqueue(second, Mask::single(0u)) ==
          EnqueueResult::conflict);
    auto cohort = scheduler.take();
    CHECK(cohort.has_value());
    CHECK(cohort->mask == Mask::from_indices({0u, 1u}));
    CHECK(scheduler.terminate(cohort->mask));
    CHECK(scheduler.live_mask() == Mask::from_indices({2u, 3u}));
    return true;
}

template<size_t Width>
[[nodiscard]] bool run_width_suite() noexcept {
    CHECK(run_diamond<Width>(SchedulingPolicy::depth_first));
    CHECK(run_diamond<Width>(SchedulingPolicy::largest_cohort));
    CHECK(run_early_return<Width>(SchedulingPolicy::depth_first));
    CHECK(run_early_return<Width>(SchedulingPolicy::largest_cohort));
    CHECK(test_loop_policy_independence<Width>());
    CHECK(test_partial_warp<Width>());
    return true;
}

}// namespace

int main() {
    struct Test {
        std::string_view name;
        bool (*run)() noexcept;
    };
    constexpr std::array tests{
        Test{"lane mask", &test_lane_mask},
        Test{"width 1", &run_width_suite<1u>},
        Test{"width 4", &run_width_suite<4u>},
        Test{"width 8", &run_width_suite<8u>},
        Test{"collective epochs 4", &test_collective_epoch_separation<4u>},
        Test{"collective epochs 8", &test_collective_epoch_separation<8u>},
        Test{"scheduler invariants", &test_scheduler_invariants},
    };

    auto failures = 0u;
    for (auto test : tests) {
        if (test.run()) {
            std::cout << "[pass] " << test.name << '\n';
        } else {
            std::cerr << "[fail] " << test.name << '\n';
            ++failures;
        }
    }
    return failures == 0u ? 0 : 1;
}

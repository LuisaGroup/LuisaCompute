#include "cohort_scheduler.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <optional>

using namespace luisa::compute::simd::schedule;

namespace {

constexpr auto width = 3u;
constexpr auto result_sentinel = uint32_t{0xdeadbeefu};
using Scheduler = CohortScheduler<width>;
using Mask = Scheduler::Mask;
using Cohort = Scheduler::CohortType;

enum : uint32_t {
    pc_entry,
    pc_inner_dispatch,
    pc_inner_add_ten,
    pc_early_return,
    pc_inner_add_thirty,
    pc_inner_join,
    pc_outer_add_forty,
    pc_shared,
    pc_outer_join,
    pc_loop_header,
    pc_loop_body,
    pc_loop_exit,
};

constexpr auto outer_token = uint32_t{101u};
constexpr auto inner_root_token = uint32_t{102u};
constexpr auto inner_outer_token = uint32_t{202u};
constexpr auto loop_token = uint32_t{103u};
constexpr Continuation outer_join{
    pc_outer_join, 0u, 0u};
constexpr Continuation loop_exit{
    pc_loop_exit, 0u, 0u};

[[nodiscard]] bool check(bool condition, const char *expression,
                         const char *file, int line) noexcept {
    if (!condition) {
        std::cerr << file << ':' << line << ": model check failed: "
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

struct ModelState {
    Scheduler scheduler{};
    std::array<uint8_t, width> input{};
    std::array<uint8_t, width> remaining{};
    std::array<uint32_t, width> values{};
    std::array<uint32_t, width> results{};
    size_t depth{0u};
};

struct ExplorationStats {
    size_t initial_states{0u};
    size_t terminal_schedules{0u};
    size_t transitions{0u};
    size_t max_depth{0u};
};

[[nodiscard]] constexpr bool enqueue_ok(EnqueueResult result) noexcept {
    return result == EnqueueResult::inserted ||
           result == EnqueueResult::merged ||
           result == EnqueueResult::empty;
}

[[nodiscard]] constexpr bool convergence_ok(
    ConvergenceResult result) noexcept {
    return result == ConvergenceResult::waiting ||
           result == ConvergenceResult::released ||
           result == ConvergenceResult::empty;
}

[[nodiscard]] size_t nonempty_count(
    std::initializer_list<Mask> masks) noexcept {
    return static_cast<size_t>(std::count_if(
        masks.begin(), masks.end(),
        [](Mask mask) noexcept { return mask.any(); }));
}

[[nodiscard]] bool route_inner_join(
    ModelState &state, const Cohort &cohort) noexcept {
    auto token = cohort.continuation.convergence_token;
    if (token == inner_root_token || token == inner_outer_token) {
        auto parent = token == inner_outer_token ? outer_token : 0u;
        return convergence_ok(state.scheduler.arrive(
            Continuation{pc_inner_join, parent, 0u}, cohort.mask));
    }
    return enqueue_ok(state.scheduler.enqueue(
        Continuation{pc_inner_join, token, 0u}, cohort.mask));
}

[[nodiscard]] bool route_outer_join(
    ModelState &state, const Cohort &cohort) noexcept {
    if (cohort.continuation.convergence_token == outer_token) {
        return convergence_ok(
            state.scheduler.arrive(outer_join, cohort.mask));
    }
    return enqueue_ok(
        state.scheduler.enqueue(outer_join, cohort.mask));
}

[[nodiscard]] bool execute_transition(
    ModelState &state, const Cohort &cohort) noexcept {
    auto token = cohort.continuation.convergence_token;
    auto epoch = cohort.continuation.loop_epoch;
    switch (cohort.continuation.pc) {
        case pc_entry: {
            Mask inner;
            Mask outer;
            cohort.mask.for_each([&](size_t lane) noexcept {
                (state.input[lane] < 4u ? inner : outer).set(lane);
            });
            auto divergent = inner.any() && outer.any();
            if (divergent &&
                !state.scheduler.declare_convergence(
                    outer_join, cohort.mask)) {
                return false;
            }
            auto child_token = divergent ? outer_token : token;
            return enqueue_ok(state.scheduler.enqueue(
                       Continuation{
                           pc_inner_dispatch, child_token, 0u},
                       inner)) &&
                   enqueue_ok(state.scheduler.enqueue(
                       Continuation{
                           pc_outer_add_forty, child_token, 0u},
                       outer));
        }
        case pc_inner_dispatch: {
            Mask add_ten;
            Mask early_return;
            Mask add_thirty;
            cohort.mask.for_each([&](size_t lane) noexcept {
                switch (state.input[lane] % 3u) {
                    case 0u: add_ten.set(lane); break;
                    case 1u: early_return.set(lane); break;
                    default: add_thirty.set(lane); break;
                }
            });
            auto divergent =
                nonempty_count({add_ten, early_return, add_thirty}) > 1u;
            auto inner_join = Continuation{
                pc_inner_join, token, 0u};
            if (divergent &&
                !state.scheduler.declare_convergence(
                    inner_join, cohort.mask)) {
                return false;
            }
            auto child_token = divergent ?
                token == outer_token ?
                    inner_outer_token :
                    inner_root_token :
                token;
            return enqueue_ok(state.scheduler.enqueue(
                       Continuation{
                           pc_inner_add_ten, child_token, 0u},
                       add_ten)) &&
                   enqueue_ok(state.scheduler.enqueue(
                       Continuation{
                           pc_early_return, child_token, 0u},
                       early_return)) &&
                   enqueue_ok(state.scheduler.enqueue(
                       Continuation{
                           pc_inner_add_thirty, child_token, 0u},
                       add_thirty));
        }
        case pc_inner_add_ten:
            cohort.mask.for_each([&](size_t lane) noexcept {
                state.values[lane] = 10u + static_cast<uint32_t>(lane);
            });
            return enqueue_ok(state.scheduler.enqueue(
                Continuation{pc_shared, token, 0u}, cohort.mask));
        case pc_early_return:
            cohort.mask.for_each([&](size_t lane) noexcept {
                state.results[lane] =
                    100u + static_cast<uint32_t>(lane);
            });
            return state.scheduler.terminate(cohort.mask);
        case pc_inner_add_thirty:
            cohort.mask.for_each([&](size_t lane) noexcept {
                state.values[lane] = 30u + static_cast<uint32_t>(lane);
            });
            return route_inner_join(state, cohort);
        case pc_inner_join:
            return route_outer_join(state, cohort);
        case pc_outer_add_forty:
            cohort.mask.for_each([&](size_t lane) noexcept {
                state.values[lane] = 40u + static_cast<uint32_t>(lane);
            });
            return enqueue_ok(state.scheduler.enqueue(
                Continuation{pc_shared, token, 0u}, cohort.mask));
        case pc_shared:
            if (token == inner_root_token ||
                token == inner_outer_token) {
                return route_inner_join(state, cohort);
            }
            return route_outer_join(state, cohort);
        case pc_outer_join:
            if (!state.scheduler.declare_convergence(
                    loop_exit, cohort.mask)) {
                return false;
            }
            return enqueue_ok(state.scheduler.enqueue(
                Continuation{pc_loop_header, loop_token, 0u},
                cohort.mask));
        case pc_loop_header: {
            Mask done;
            Mask running;
            cohort.mask.for_each([&](size_t lane) noexcept {
                (state.remaining[lane] == 0u ? done : running)
                    .set(lane);
            });
            auto arrived = done.none() ?
                ConvergenceResult::empty :
                state.scheduler.arrive(loop_exit, done);
            return convergence_ok(arrived) &&
                   enqueue_ok(state.scheduler.enqueue(
                       Continuation{pc_loop_body, loop_token, epoch},
                       running));
        }
        case pc_loop_body:
            cohort.mask.for_each([&](size_t lane) noexcept {
                --state.remaining[lane];
                ++state.values[lane];
            });
            return enqueue_ok(state.scheduler.enqueue(
                Continuation{
                    pc_loop_header, loop_token, epoch + 1u},
                cohort.mask));
        case pc_loop_exit:
            cohort.mask.for_each([&](size_t lane) noexcept {
                state.results[lane] = state.values[lane];
            });
            return state.scheduler.terminate(cohort.mask);
        default: return false;
    }
}

[[nodiscard]] std::array<uint32_t, width> scalar_reference(
    Mask active, const std::array<uint8_t, width> &input) noexcept {
    std::array<uint32_t, width> results;
    results.fill(result_sentinel);
    active.for_each([&](size_t lane) noexcept {
        auto value = input[lane];
        if (value < 4u && value % 3u == 1u) {
            results[lane] = 100u + static_cast<uint32_t>(lane);
            return;
        }
        auto base = value >= 4u ? 40u :
                    value % 3u == 0u ? 10u :
                                        30u;
        results[lane] = base + static_cast<uint32_t>(lane) +
                        static_cast<uint32_t>(value % 3u);
    });
    return results;
}

[[nodiscard]] bool explore_all_schedules(
    const ModelState &state,
    const std::array<uint32_t, width> &expected,
    ExplorationStats &stats) noexcept {
    if (state.depth > 64u ||
        !state.scheduler.quiescent_invariants_hold()) {
        return false;
    }
    stats.max_depth = std::max(stats.max_depth, state.depth);
    if (state.scheduler.complete()) {
        ++stats.terminal_schedules;
        return state.results == expected;
    }
    if (state.scheduler.stalled() ||
        state.scheduler.ready_count() == 0u) {
        return false;
    }
    auto choice_count = state.scheduler.ready_count();
    for (auto choice = size_t{0u}; choice < choice_count; choice++) {
        auto next = state;
        auto cohort = next.scheduler.take_at(choice);
        if (!cohort || !next.scheduler.invariants_hold() ||
            !execute_transition(next, *cohort)) {
            return false;
        }
        ++stats.transitions;
        ++next.depth;
        if (!explore_all_schedules(next, expected, stats)) {
            return false;
        }
    }
    return true;
}

[[nodiscard]] bool run_bounded_model_check() noexcept {
    ExplorationStats stats;
    for (auto active_bits = uint32_t{1u};
         active_bits < (1u << width); active_bits++) {
        Mask active;
        for (auto lane = size_t{0u}; lane < width; lane++) {
            active.set(lane, (active_bits & (1u << lane)) != 0u);
        }
        auto input_count = size_t{1u};
        for (auto i = size_t{0u}; i < active.count(); i++) {
            input_count *= 6u;
        }
        for (auto encoded = size_t{0u};
             encoded < input_count; encoded++) {
            std::array<uint8_t, width> input{};
            auto remaining_encoding = encoded;
            active.for_each([&](size_t lane) noexcept {
                input[lane] = static_cast<uint8_t>(
                    remaining_encoding % 6u);
                remaining_encoding /= 6u;
            });
            ModelState initial;
            initial.scheduler = Scheduler{active};
            initial.input = input;
            initial.results.fill(result_sentinel);
            for (auto lane = size_t{0u}; lane < width; lane++) {
                initial.remaining[lane] = input[lane] % 3u;
            }
            CHECK(initial.scheduler.enqueue(
                      Continuation{pc_entry, 0u, 0u}, active) ==
                  EnqueueResult::inserted);
            auto expected = scalar_reference(active, input);
            ++stats.initial_states;
            CHECK(explore_all_schedules(initial, expected, stats));
        }
    }
    CHECK(stats.initial_states == 342u);
    CHECK(stats.terminal_schedules >= stats.initial_states);
    CHECK(stats.transitions > stats.terminal_schedules);
    CHECK(stats.max_depth > 0u && stats.max_depth <= 64u);
    std::cout << "checked " << stats.initial_states
              << " initial states, " << stats.terminal_schedules
              << " complete scheduler interleavings, and "
              << stats.transitions << " transitions\n";
    return true;
}

}// namespace

int main() {
    if (!run_bounded_model_check()) {
        std::cerr << "[fail] bounded scheduler model\n";
        return 1;
    }
    std::cout << "[pass] bounded scheduler model\n";
    return 0;
}

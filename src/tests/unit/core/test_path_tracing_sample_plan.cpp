// Test for path-tracing sample-pass planning.
// This test covers exact dispatch counts, short tails, and infinite rendering.

#include "ut/ut.hpp"

#include "path_tracing_sample_plan.h"

#include <cstdint>
#include <initializer_list>

using namespace boost::ut;
using namespace boost::ut::literals;

namespace {

void expect_batches(
    uint64_t total_spp,
    uint32_t max_spp_per_dispatch,
    std::initializer_list<uint32_t> expected) {
    auto plan = luisa::ref::PathTracingSamplePassPlan{
        .total_spp = total_spp,
        .max_spp_per_dispatch = max_spp_per_dispatch,
    };
    uint64_t completed_spp = 0u;
    auto expected_batch = expected.begin();
    while (plan.has_next(completed_spp)) {
        expect(expected_batch != expected.end())
            << "the plan emitted more dispatches than expected";
        if (expected_batch == expected.end()) { return; }
        auto dispatch_spp = plan.next_dispatch_spp(completed_spp);
        expect(eq(dispatch_spp, *expected_batch));
        expect(dispatch_spp > 0u);
        expect(dispatch_spp <= max_spp_per_dispatch);
        completed_spp += dispatch_spp;
        ++expected_batch;
    }
    expect(expected_batch == expected.end())
        << "the plan emitted fewer dispatches than expected";
    expect(eq(completed_spp, total_spp))
        << "the plan must neither drop nor oversample requested paths";
    expect(eq(plan.next_dispatch_spp(completed_spp), 0u));
}

}// namespace

int main(int argc, char *argv[]) {
    boost::ut::detail::cfg::parse_arg_with_fallback(
        argc, const_cast<const char **>(argv));

    "path_tracing_sample_plan_handles_boundary_spp"_test = [] {
        expect_batches(1u, 64u, {1u});
        expect_batches(63u, 64u, {63u});
        expect_batches(64u, 64u, {64u});
        expect_batches(65u, 64u, {64u, 1u});
    };

    "path_tracing_sample_plan_preserves_default_1024_layout"_test = [] {
        static_assert(luisa::ref::DEFAULT_PATH_TRACING_SPP == 1024u);
        auto plan = luisa::ref::PathTracingSamplePassPlan{
            .total_spp = luisa::ref::DEFAULT_PATH_TRACING_SPP,
            .max_spp_per_dispatch = 64u,
        };
        uint64_t completed_spp = 0u;
        uint32_t dispatch_count = 0u;
        while (plan.has_next(completed_spp)) {
            auto dispatch_spp = plan.next_dispatch_spp(completed_spp);
            expect(eq(dispatch_spp, 64u));
            completed_spp += dispatch_spp;
            ++dispatch_count;
        }
        expect(eq(completed_spp, uint64_t{1024u}));
        expect(eq(dispatch_count, 16u));
    };

    "path_tracing_sample_plan_supports_single_sample_backends"_test = [] {
        auto plan = luisa::ref::PathTracingSamplePassPlan{
            .total_spp = 65u,
            .max_spp_per_dispatch = 1u,
        };
        uint64_t completed_spp = 0u;
        uint32_t dispatch_count = 0u;
        while (plan.has_next(completed_spp)) {
            auto dispatch_spp = plan.next_dispatch_spp(completed_spp);
            expect(eq(dispatch_spp, 1u));
            completed_spp += dispatch_spp;
            ++dispatch_count;
        }
        expect(eq(completed_spp, uint64_t{65u}));
        expect(eq(dispatch_count, 65u));
    };

    "path_tracing_sample_plan_handles_terminal_states"_test = [] {
        constexpr auto empty = luisa::ref::PathTracingSamplePassPlan{
            .total_spp = 0u,
            .max_spp_per_dispatch = 64u,
        };
        static_assert(!empty.has_next(0u));
        static_assert(empty.next_dispatch_spp(0u) == 0u);

        constexpr auto disabled = luisa::ref::PathTracingSamplePassPlan{
            .total_spp = 65u,
            .max_spp_per_dispatch = 0u,
        };
        static_assert(!disabled.has_next(0u));
        static_assert(disabled.next_dispatch_spp(0u) == 0u);

        constexpr auto infinite = luisa::ref::PathTracingSamplePassPlan{
            .max_spp_per_dispatch = 64u,
            .infinite = true,
        };
        expect(!empty.has_next(0u));
        expect(eq(empty.next_dispatch_spp(0u), 0u));
        expect(!disabled.has_next(0u));
        expect(eq(disabled.next_dispatch_spp(0u), 0u));
        expect(infinite.has_next(0u));
        expect(infinite.has_next(1'000'000u));
        expect(eq(infinite.next_dispatch_spp(1'000'000u), 64u));
    };
}

#include <limits>

#include "hip_geometry_memory_policy.h"
#include "ut/ut.hpp"

using namespace luisa::compute::hip;
using namespace boost::ut;
using namespace boost::ut::literals;

static auto test_hip_geometry_memory_policy = [] {
    "HIPRT high-quality scratch budget is an exact overflow-free ratio"_test = [] {
        expect(!hiprt_high_quality_scratch_exceeds_budget(0u, 0u));
        expect(hiprt_high_quality_scratch_exceeds_budget(1u, 0u));

        // The exact one-quarter boundary remains eligible for the
        // high-quality builder; one byte above it is memory constrained.
        expect(!hiprt_high_quality_scratch_exceeds_budget(1u, 4u));
        expect(hiprt_high_quality_scratch_exceeds_budget(2u, 4u));

        // Integer division must implement the rational strict inequality for
        // free-memory sizes that are not divisible by four.
        expect(!hiprt_high_quality_scratch_exceeds_budget(1u, 5u));
        expect(hiprt_high_quality_scratch_exceeds_budget(2u, 5u));
        expect(!hiprt_high_quality_scratch_exceeds_budget(1u, 7u));
        expect(hiprt_high_quality_scratch_exceeds_budget(2u, 7u));

        // These boundary cases would overflow a cross-multiplication by four.
        constexpr auto max_size = std::numeric_limits<size_t>::max();
        expect(!hiprt_high_quality_scratch_exceeds_budget(
            max_size / 4u, max_size));
        expect(hiprt_high_quality_scratch_exceeds_budget(
            max_size / 4u + 1u, max_size));
    };
    return 0;
}();

int main(int argc, char *argv[]) {
    boost::ut::detail::cfg::parse_arg_with_fallback(
        argc, const_cast<const char **>(argv));
}

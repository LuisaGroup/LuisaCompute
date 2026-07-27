// Test for deterministic N-body particle raster planning.
// This test covers packed depth/index ordering, stable ties, invalid inputs,
// and clipped 5x5 particle footprints.

#include "ut/ut.hpp"

#include "nbody_render_plan.h"

#include <array>
#include <limits>

using namespace luisa;
using namespace boost::ut;
using namespace boost::ut::literals;

int main(int argc, char *argv[]) {
    boost::ut::detail::cfg::parse_arg_with_fallback(
        argc, const_cast<const char **>(argv));

    "nbody_winner_encoding_orders_depth_then_index"_test = [] {
        using Encoding = ref::NBodyWinnerEncoding;
        auto near_high_index = Encoding::pack(3.0f, 1700u);
        auto far_low_index = Encoding::pack(4.0f, 1u);
        expect(Encoding::valid(near_high_index));
        expect(Encoding::valid(far_low_index));
        expect(near_high_index < far_low_index)
            << "nearer depth must win regardless of particle index";
        expect(eq(Encoding::select(near_high_index, far_low_index),
                  near_high_index));

        auto tied_high_index = Encoding::pack(3.0f, 1700u);
        auto tied_low_index = Encoding::pack(3.0f, 7u);
        expect(tied_low_index < tied_high_index)
            << "lower particle index must break equal-depth ties";
        expect(eq(Encoding::particle_index(tied_low_index), 7u));
        expect(eq(Encoding::depth_bucket(tied_low_index),
                  Encoding::depth_bucket(tied_high_index)));
    };

    "nbody_winner_encoding_is_schedule_independent"_test = [] {
        using Encoding = ref::NBodyWinnerEncoding;
        std::array candidates{
            Encoding::pack(4.5f, 2u),
            Encoding::pack(3.0f, 1400u),
            Encoding::pack(3.0f, 9u),
            Encoding::pack(8.0f, 0u),
        };
        auto fold = [](auto begin, auto end) noexcept {
            auto winner = Encoding::kInvalid;
            for (; begin != end; ++begin) {
                winner = Encoding::select(winner, *begin);
            }
            return winner;
        };
        auto forward = fold(candidates.cbegin(), candidates.cend());
        auto reverse = fold(candidates.crbegin(), candidates.crend());
        expect(eq(forward, reverse));
        expect(eq(forward, Encoding::pack(3.0f, 9u)));
    };

    "nbody_winner_encoding_rejects_unrepresentable_candidates"_test = [] {
        using Encoding = ref::NBodyWinnerEncoding;
        expect(eq(Encoding::pack(Encoding::kMinimumVisibleDistance, 0u),
                  Encoding::kInvalid));
        expect(eq(Encoding::pack(-1.0f, 0u), Encoding::kInvalid));
        expect(eq(Encoding::pack(std::numeric_limits<float>::infinity(), 0u),
                  Encoding::kInvalid));
        expect(eq(Encoding::pack(std::numeric_limits<float>::quiet_NaN(), 0u),
                  Encoding::kInvalid));
        expect(eq(Encoding::pack(1.0f, Encoding::kMaxParticleCount),
                  Encoding::kInvalid));

        auto last = Encoding::pack(
            std::numeric_limits<float>::max(),
            Encoding::kMaxParticleCount - 1u);
        expect(Encoding::valid(last));
        expect(last < Encoding::kInvalid);
        expect(eq(Encoding::particle_index(last),
                  Encoding::kMaxParticleCount - 1u));
    };

    "nbody_footprint_planner_clips_all_boundaries"_test = [] {
        auto interior = ref::plan_nbody_footprint(8, 6, 16, 12);
        expect(interior.valid());
        expect(eq(interior.min_x, 6));
        expect(eq(interior.min_y, 4));
        expect(eq(interior.max_x, 10));
        expect(eq(interior.max_y, 8));
        expect(eq(interior.pixel_count(), 25u));

        auto top_left = ref::plan_nbody_footprint(0, 0, 16, 12);
        expect(top_left.valid());
        expect(eq(top_left.min_x, 0));
        expect(eq(top_left.min_y, 0));
        expect(eq(top_left.max_x, 2));
        expect(eq(top_left.max_y, 2));
        expect(eq(top_left.pixel_count(), 9u));
        expect(top_left.contains(2, 2));
        expect(!top_left.contains(3, 2));

        auto bottom_right = ref::plan_nbody_footprint(15, 11, 16, 12);
        expect(bottom_right.valid());
        expect(eq(bottom_right.min_x, 13));
        expect(eq(bottom_right.min_y, 9));
        expect(eq(bottom_right.max_x, 15));
        expect(eq(bottom_right.max_y, 11));
        expect(eq(bottom_right.pixel_count(), 9u));

        expect(!ref::plan_nbody_footprint(-1, 0, 16, 12).valid());
        expect(!ref::plan_nbody_footprint(16, 0, 16, 12).valid());
        expect(!ref::plan_nbody_footprint(0, 12, 16, 12).valid());
        expect(!ref::plan_nbody_footprint(0, 0, 0, 12).valid());
    };
}

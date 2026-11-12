// Host-only test for the frontend half of opacity micromap support
// (VK_KHR_opacity_micromap / VK_EXT_opacity_micromap).
//
// It covers everything that can be proved without a device:
//
// - the Vulkan appendix's space-filling-curve micro-triangle order, in both
//   directions (`micro_triangle` and `micro_triangle_index`), including a full
//   round-trip over every micro triangle of levels 0..4, and the bit order of the
//   packed payload (LSB-first, per-triangle blocks);
// - the byte/bit arithmetic the backend sizes its allocations from
//   (`micro_triangle_count`, `triangle_data_byte_size`, `data_byte_size`,
//   `word_count`), including the four-state doubling and the 32-bit word rounding
//   that makes the baker's one-word-per-thread packing race-free;
// - the two-state / four-state repacking helpers of
//   `src/runtime/rtx/opacity_micromap_detail.h`, which is what an `upload()` of
//   host bits and a `readback()` of baked bits have to agree on.
//
// The device-side half (create/bake/readback/attach and the miss-through-a-hole
// assertion) lives in src/tests/unit/runtime/test_opacity_micromap.cpp.
#include "ut/ut.hpp"

#include <luisa/runtime/rtx/opacity_micromap.h>

#include "../../../src/runtime/rtx/opacity_micromap_detail.h"

#include <cmath>
#include <cstdint>
#include <vector>

using namespace boost::ut;
using namespace boost::ut::literals;
using namespace luisa;
using namespace luisa::compute;

namespace test_opacity_micromap_plan_detail {

namespace omd = opacity_micromap_detail;

[[nodiscard]] bool close_to(float lhs, float rhs) noexcept {
    return std::abs(lhs - rhs) < 1e-5f;
}

[[nodiscard]] bool close_to(float2 lhs, float2 rhs) noexcept {
    return close_to(lhs.x, rhs.x) && close_to(lhs.y, rhs.y);
}

// The corner *sets* of the four sub-triangles of a level-1 micromap, hand-derived
// from the specification text ("the sub-triangle nearest vertex 0, then the middle
// triangle with ordering flipped, then nearest vertex 1, then nearest vertex 2").
// Only the set matters for baking: every cut-out predicate is order independent.
struct ExpectedCorners {
    float2 c0;
    float2 c1;
    float2 c2;
};

[[nodiscard]] bool same_corners(const OpacityMicromap::MicroTriangle &t,
                               const ExpectedCorners &e) noexcept {
    auto all = [&](float2 x, float2 y, float2 z) noexcept {
        return close_to(t.c0, x) && close_to(t.c1, y) && close_to(t.c2, z);
    };
    // Any permutation of the three corners is the same micro triangle.
    return all(e.c0, e.c1, e.c2) || all(e.c0, e.c2, e.c1) ||
           all(e.c1, e.c0, e.c2) || all(e.c1, e.c2, e.c0) ||
           all(e.c2, e.c0, e.c1) || all(e.c2, e.c1, e.c0);
}

}// namespace test_opacity_micromap_plan_detail

int main() {

    using namespace test_opacity_micromap_plan_detail;
    using MicroTriangle = OpacityMicromap::MicroTriangle;

    "opacity_micromap_micro_triangle_count"_test = [] {
        expect(eq(OpacityMicromap::micro_triangle_count(0u), 1ull));
        expect(eq(OpacityMicromap::micro_triangle_count(1u), 4ull));
        expect(eq(OpacityMicromap::micro_triangle_count(2u), 16ull));
        expect(eq(OpacityMicromap::micro_triangle_count(5u), 1024ull));
        expect(eq(OpacityMicromap::micro_triangle_count(10u), 1ull << 20));
        // Clamped to the largest representable level instead of overflowing.
        expect(eq(OpacityMicromap::micro_triangle_count(15u), 1ull << 30));
        expect(eq(OpacityMicromap::micro_triangle_count(200u),
                  OpacityMicromap::micro_triangle_count(
                      OpacityMicromap::max_representable_subdivision_level)));
    };

    "opacity_micromap_data_byte_size"_test = [] {
        using Format = OpacityMicromap::Format;
        // Level 0: one micro triangle -> one bit -> one byte.
        expect(eq(OpacityMicromap::triangle_data_byte_size(Format::TwoState, 0u), 1ull));
        // Level 1: 4 bits -> one byte; four-state doubles it.
        expect(eq(OpacityMicromap::triangle_data_byte_size(Format::TwoState, 1u), 1ull));
        expect(eq(OpacityMicromap::triangle_data_byte_size(Format::FourState, 1u), 1ull));
        // Level 2: 16 bits -> two bytes; four-state 32 bits -> four bytes.
        expect(eq(OpacityMicromap::triangle_data_byte_size(Format::TwoState, 2u), 2ull));
        expect(eq(OpacityMicromap::triangle_data_byte_size(Format::FourState, 2u), 4ull));
        // Level 5: 1024 bits = 128 bytes per triangle, 1024 four-state bytes.
        expect(eq(OpacityMicromap::triangle_data_byte_size(Format::TwoState, 5u), 128ull));
        expect(eq(OpacityMicromap::triangle_data_byte_size(Format::FourState, 5u), 256ull));
        // Per-triangle blocks are only byte aligned, exactly like dataOffset.
        expect(eq(OpacityMicromap::data_byte_size(Format::TwoState, 1u, 3u), 3ull));
        expect(eq(OpacityMicromap::data_byte_size(Format::TwoState, 5u, 2u), 256ull));
        expect(eq(OpacityMicromap::data_byte_size(Format::FourState, 5u, 2u), 512ull));
        // The baker packs whole 32-bit words, so its dispatch size rounds up.
        expect(eq(omd::word_count(Format::TwoState, 1u, 3u), 1ull));
        expect(eq(omd::word_count(Format::TwoState, 5u, 1u), 32ull));
        expect(eq(omd::word_count(Format::FourState, 5u, 1u), 64ull));
        expect(eq(omd::micro_triangles_per_word(Format::TwoState), 32u));
        expect(eq(omd::micro_triangles_per_word(Format::FourState), 16u));
    };

    "opacity_micromap_level_one_corner_sets"_test = [] {
        // Level 0 is the base triangle: (0,0) is vertex 0, (1,0) vertex 1 and
        // (0,1) vertex 2 in (u, v) = (weight of vertex 1, weight of vertex 2).
        auto base = OpacityMicromap::micro_triangle(0u, 0u);
        expect(close_to(base.c0, make_float2(0.0f, 0.0f)));
        expect(close_to(base.c1, make_float2(1.0f, 0.0f)));
        expect(close_to(base.c2, make_float2(0.0f, 1.0f)));
        // Level 1, index 0 = nearest vertex 0: (v0, m01, m20).
        expect(same_corners(OpacityMicromap::micro_triangle(1u, 0u),
                            ExpectedCorners{make_float2(0.0f, 0.0f),
                                            make_float2(0.5f, 0.0f),
                                            make_float2(0.0f, 0.5f)}));
        // Level 1, index 1 = the middle triangle: the three edge midpoints.
        expect(same_corners(OpacityMicromap::micro_triangle(1u, 1u),
                            ExpectedCorners{make_float2(0.5f, 0.0f),
                                            make_float2(0.5f, 0.5f),
                                            make_float2(0.0f, 0.5f)}));
        // Level 1, index 2 = nearest vertex 1: (m01, v1, m12).
        expect(same_corners(OpacityMicromap::micro_triangle(1u, 2u),
                            ExpectedCorners{make_float2(0.5f, 0.0f),
                                            make_float2(1.0f, 0.0f),
                                            make_float2(0.5f, 0.5f)}));
        // Level 1, index 3 = nearest vertex 2: (m20, m12, v2).
        expect(same_corners(OpacityMicromap::micro_triangle(1u, 3u),
                            ExpectedCorners{make_float2(0.0f, 0.5f),
                                            make_float2(0.5f, 0.5f),
                                            make_float2(0.0f, 1.0f)}));
    };

    "opacity_micromap_index_round_trip"_test = [] {
        // The centroid of a micro triangle must map back to its own index
        // ("tie-breaks round into the middle triangle", so the centroids of the
        // four level-1 children are never ambiguous).
        for (auto level : {0u, 1u, 2u, 3u, 4u}) {
            auto count = OpacityMicromap::micro_triangle_count(level);
            for (auto index = 0ull; index < count; index++) {
                auto corners = OpacityMicromap::micro_triangle(level, index);
                auto centroid = make_float2(
                    (corners.c0.x + corners.c1.x + corners.c2.x) / 3.0f,
                    (corners.c0.y + corners.c1.y + corners.c2.y) / 3.0f);
                expect(eq(OpacityMicromap::micro_triangle_index(level, centroid), index))
                    << "level " << level << ", micro triangle " << index;
            }
        }
        // Vertex 0 is the fixed point of the curve (digit 0 all the way down). Vertex 2
        // is not: its child is *reversed*, so after the leading digit 3 the corner
        // becomes the child's own vertex 0 and the remaining digits are all 0.
        expect(eq(OpacityMicromap::micro_triangle_index(4u, make_float2(0.0f, 0.0f)), 0ull));
        expect(eq(OpacityMicromap::micro_triangle_index(4u, make_float2(0.0f, 1.0f)),
                  3ull * OpacityMicromap::micro_triangle_count(3u)));
        auto middle = OpacityMicromap::micro_triangle_index(3u, make_float2(1.0f / 3.0f, 1.0f / 3.0f));
        // All digits are 1 (the middle child) at every level of the curve.
        expect(eq(middle, 0b010101ull));
    };

    "opacity_micromap_two_state_packing"_test = [] {
        auto words = std::vector<uint>{0u, 0u};
        auto span = luisa::span<uint>{words};
        // Bit 0 of the first word is micro triangle 0, packed LSB first.
        omd::set_two_state(span, 0u, true);
        expect(eq(words[0], 1u));
        omd::set_two_state(span, 31u, true);
        expect(eq(words[0], 0x80000001u));
        // The next bit lands in the second word, not in the first.
        omd::set_two_state(span, 32u, true);
        expect(eq(words[1], 1u));
        expect(eq(words[0], 0x80000001u));
        // Clearing is exact.
        omd::set_two_state(span, 0u, false);
        expect(eq(words[0], 0x80000000u));
        expect(omd::get_two_state(span, 0u) == false);
        expect(omd::get_two_state(span, 31u) == true);
        expect(omd::get_two_state(span, 32u) == true);
        expect(omd::get_two_state(span, 33u) == false);
        // A triangle at level 5 owns 1024 bits = 32 whole words, so triangle 1
        // starts exactly at word 32.
        expect(eq(omd::word_count(OpacityMicromap::Format::TwoState, 5u, 1u), 32ull));
    };

    "opacity_micromap_four_state_packing"_test = [] {
        auto words = std::vector<uint>{0u};
        auto span = luisa::span<uint>{words};
        omd::set_four_state(span, 0u, omd::four_state_unknown_transparent);
        expect(eq(words[0], 2u));
        omd::set_four_state(span, 1u, omd::four_state_opaque);
        expect(eq(words[0], 2u | (1u << 2u)));
        omd::set_four_state(span, 15u, omd::four_state_unknown_opaque);
        expect(eq(words[0], 2u | (1u << 2u) | (3u << 30u)));
        expect(eq(omd::get_four_state(span, 0u), omd::four_state_unknown_transparent));
        expect(eq(omd::get_four_state(span, 1u), omd::four_state_opaque));
        expect(eq(omd::get_four_state(span, 15u), omd::four_state_unknown_opaque));
        expect(eq(omd::get_four_state(span, 2u), omd::four_state_transparent));
        // Overwriting a value must not disturb its neighbours.
        omd::set_four_state(span, 1u, omd::four_state_transparent);
        expect(eq(words[0], 2u | (3u << 30u)));
    };

    "opacity_micromap_corner_weights_sum_to_one"_test = [] {
        // Every micro triangle corner is a barycentric coordinate of the base
        // triangle, so `1 - u - v` (the weight of vertex 0) must stay in [0, 1].
        for (auto level : {1u, 2u, 5u}) {
            auto count = OpacityMicromap::micro_triangle_count(level);
            for (auto index = 0ull; index < count; index++) {
                auto corners = OpacityMicromap::micro_triangle(level, index);
                for (auto corner : {corners.c0, corners.c1, corners.c2}) {
                    auto w0 = 1.0f - corner.x - corner.y;
                    expect(w0 >= -1e-5f && w0 <= 1.0f + 1e-5f);
                    expect(corner.x >= -1e-5f && corner.x <= 1.0f + 1e-5f);
                    expect(corner.y >= -1e-5f && corner.y <= 1.0f + 1e-5f);
                }
            }
        }
    };
}

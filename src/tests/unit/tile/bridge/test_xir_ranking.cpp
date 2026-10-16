// Runtime tests for the public finite-value topk/sort library composition.
// Covers stable ties, signed zero, K boundaries, non-power-of-two dimensions,
// full values and int64 indices, input immutability and all buffer guards.
#include "ut/ut.hpp"
#include "test_device.h"
#include "tile_rank_test_utils.h"
#include <luisa/core/logging.h>
#include <luisa/runtime/stream.h>
#include <luisa/tile/runtime.h>
#include <array>

using namespace luisa;
using namespace luisa::compute;
using namespace boost::ut;
using namespace boost::ut::literals;

namespace {

void run(Device &device, test::tile_rank::Case fixture, bool tirx) {
    using namespace test::tile_rank;
    expect(fixture.kernel.valid());
    if (!fixture.kernel.valid()) { return; }
    LUISA_INFO("Ranking compile: R/N/K={}/{}/{}, descending={}, tirx={}",
               fixture.rows, fixture.columns, fixture.count, fixture.descending, tirx);
    auto shader = tile::compile(device, fixture.kernel, {.lowering = tirx ? tile::Lowering::TIRX : tile::Lowering::NATIVE});
    expect(static_cast<bool>(shader)) << shader.metadata().error;
    if (!shader) { return; }
    auto stream = device.create_stream(StreamTag::COMPUTE);
    auto input = device.create_buffer<float>(fixture.input.size() + 2u * GuardedData::pad);
    auto values = device.create_buffer<float>(fixture.expected_values.size() + 2u * GuardedData::pad);
    auto indices = device.create_buffer<int64_t>(fixture.expected_indices.size() + 2u * GuardedData::pad);
    for (auto pattern : {InputPattern::DUPLICATES, InputPattern::UNIQUE, InputPattern::ALL_EQUAL, InputPattern::SIGNED_ZERO}) {
        populate(fixture, pattern);
        GuardedData data{fixture};
        LUISA_INFO("Ranking dispatch: R/N/K={}/{}/{}, descending={}, pattern={}",
                   fixture.rows, fixture.columns, fixture.count, fixture.descending, static_cast<int32_t>(pattern));
        stream << input.copy_from(span{data.input}) << values.copy_from(span{data.values}) << indices.copy_from(span{data.indices})
               << shader(input.view(GuardedData::pad, fixture.input.size()),
                         values.view(GuardedData::pad, fixture.expected_values.size()),
                         indices.view(GuardedData::pad, fixture.expected_indices.size()))
                      .dispatch()
               << input.copy_to(span{data.input}) << values.copy_to(span{data.values}) << indices.copy_to(span{data.indices}) << synchronize();
        auto checked = validate(fixture, data);
        expect(eq(checked.value_mismatches, 0u)) << "R/N/K=" << fixture.rows << '/' << fixture.columns << '/' << fixture.count << " pattern=" << static_cast<int32_t>(pattern);
        expect(eq(checked.index_mismatches, 0u)) << "complete stable int64 indices";
        expect(eq(checked.input_mismatches, 0u)) << "input bitwise immutable";
        expect(eq(checked.guard_mismatches, 0u)) << "all three allocations retain prefix/suffix guards";
    }
    LUISA_INFO("Ranking complete: R/N/K={}/{}/{}, descending={}",
               fixture.rows, fixture.columns, fixture.count, fixture.descending);
}

}// namespace

int main(int argc, char *argv[]) {
    boost::ut::detail::cfg::parse_arg_with_fallback(argc, const_cast<const char **>(argv));
    LUISA_ASSERT(argc >= 2 && (string_view{argv[1]} == "simd" || string_view{argv[1]} == "metal4" || string_view{argv[1]} == "metal"),
                 "Usage: test_tile_xir_ranking <simd|metal4|metal> (metal explicitly uses TIRx)");
    auto [context, device] = test::create_device(argc, argv);
    auto tirx = string_view{argv[1]} == "metal";
    using test::tile_rank::Operation;
    "tile_ranking_topk_stable_values_indices"_test = [&] {
        for (auto dimensions : {std::array<int64_t, 3u>{1, 1, 1}, {3, 7, 1}, {3, 7, 3}, {3, 7, 7}, {3, 16, 8}, {17, 33, 5}, {3, 65, 64}, {3, 65, 65}}) {
            for (auto descending : {false, true}) {
                run(device, test::tile_rank::rows(Operation::TOPK, dimensions[0], dimensions[1], dimensions[2], descending), tirx);
            }
        }
    };
    "tile_ranking_sort_stable_values_indices"_test = [&] {
        for (auto dimensions : {std::array<int64_t, 2u>{1, 1}, {3, 7}, {3, 16}, {17, 33}, {3, 65}}) {
            for (auto descending : {false, true}) {
                run(device, test::tile_rank::rows(Operation::SORT, dimensions[0], dimensions[1], dimensions[1], descending), tirx);
            }
        }
    };
    "tile_ranking_oracle_rejects_corrupted_payloads"_test = [&] {
        // Mutation checks prove the shared checker observes each independent
        // obligation; they do not replace any device execution above.
        auto fixture = test::tile_rank::rows(Operation::TOPK, 3, 7, 3);
        test::tile_rank::GuardedData data{fixture};
        constexpr auto pad = test::tile_rank::GuardedData::pad;
        std::copy(fixture.expected_values.begin(), fixture.expected_values.end(), data.values.begin() + pad);
        std::copy(fixture.expected_indices.begin(), fixture.expected_indices.end(), data.indices.begin() + pad);
        expect(test::tile_rank::validate(fixture, data).passed());
        data.values[pad] += 1.0f;
        expect(eq(test::tile_rank::validate(fixture, data).value_mismatches, 1u));
        data.values[pad] = fixture.expected_values[0u];
        data.indices[pad] = -1;
        expect(eq(test::tile_rank::validate(fixture, data).index_mismatches, 1u));
        data.indices[pad] = fixture.expected_indices[0u];
        data.input[pad] += 1.0f;
        expect(eq(test::tile_rank::validate(fixture, data).input_mismatches, 1u));
        data.input[pad] = fixture.input[0u];
        data.input.front() = 0.0f;
        data.input.back() = 0.0f;
        data.values.front() = 0.0f;
        data.values.back() = 0.0f;
        data.indices.front() = 0;
        data.indices.back() = 0;
        expect(eq(test::tile_rank::validate(fixture, data).guard_mismatches, 6u));
    };
}

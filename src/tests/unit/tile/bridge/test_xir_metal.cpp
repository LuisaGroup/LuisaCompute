// Test the explicit TileIR -> XIR -> LLVM AIR -> Metal4 Runtime route.
// Full FP64 LLM oracles and offset guards cover whole-program/packet-local maps.

#include "ut/ut.hpp"
#include "test_device.h"
#include "tile_llm_test_utils.h"
#include <luisa/runtime/stream.h>
#include <luisa/tile/bridge/xir/planner.h>
#include <luisa/tile/runtime.h>
#include <limits>

using namespace luisa;
using namespace luisa::compute;
using namespace boost::ut;
using namespace boost::ut::literals;

namespace {

constexpr size_t padding = 17u;
constexpr float guard = -719.5f;

void check_guards(span<const float> values) {
    expect(std::all_of(values.begin(), values.begin() + padding, [](float x) { return x == guard; }));
    expect(std::all_of(values.end() - padding, values.end(), [](float x) { return x == guard; }));
}

void run(Device &device, const test::tile_llm::Case &fixture, uint32_t lanes) {
    LUISA_INFO("Metal4 Tile-XIR: {} rows={} width={} local_lanes={}",
               fixture.kernel.function().name(), fixture.shapes[3][0], fixture.shapes[3][1], lanes);
    expect(fixture.kernel.valid());
    if (!fixture.kernel.valid()) { return; }
    tile::bridge::xir::PlannerOptions planner{.block_size = 64u, .local_lanes = lanes};
    auto shader = tile::compile(device, fixture.kernel, {.threads_per_group = 64u, .xir = &planner},
                                {.enable_fast_math = false});
    expect(static_cast<bool>(shader)) << shader.metadata().error;
    if (!shader) { return; }
    const auto &metadata = shader.metadata();
    // Automatic mode is used only by the width-seven fallback fixture below;
    // its local domain cannot span the physical 32-lane packet.
    auto expected_lanes = lanes == 0u ? 1u : lanes;
    expect(metadata.realization.find("TileIR -> XIR SSA -> LLVM AIR -> Metal4 Runtime") != string::npos) << metadata.realization;
    expect(metadata.realization.find("source_format=XIR") != string::npos) << metadata.realization;
    expect(metadata.realization.find(format("local_lanes={};", expected_lanes)) != string::npos) << metadata.realization;
    expect(eq(metadata.dispatch_size.x, static_cast<uint32_t>(fixture.shapes[3][0]) * expected_lanes));
    expect(eq(metadata.dispatch_size.y, 1u));
    expect(eq(metadata.dispatch_size.z, 1u));
    expect(eq(shader.block_size().x, 64u));

    auto stream = device.create_stream(StreamTag::COMPUTE);
    std::array<vector<float>, 3u> inputs;
    vector<Buffer<float>> buffers;
    for (size_t i = 0u; i < inputs.size(); i++) {
        inputs[i].resize(fixture.inputs[i].size() + 2u * padding, guard);
        std::copy(fixture.inputs[i].begin(), fixture.inputs[i].end(), inputs[i].begin() + padding);
        buffers.emplace_back(device.create_buffer<float>(inputs[i].size()));
        stream << buffers.back().copy_from(span{inputs[i]});
    }
    vector<float> output(fixture.expected.size() + 2u * padding, guard);
    std::fill(output.begin() + padding, output.end() - padding, std::numeric_limits<float>::quiet_NaN());
    auto result = device.create_buffer<float>(output.size());
    stream << result.copy_from(span{output})
           << shader(buffers[0].view(padding, fixture.inputs[0].size()),
                     buffers[1].view(padding, fixture.inputs[1].size()),
                     buffers[2].view(padding, fixture.inputs[2].size()),
                     result.view(padding, fixture.expected.size()))
                  .dispatch()
           << result.copy_to(span{output});
    for (size_t i = 0u; i < inputs.size(); i++) { stream << buffers[i].copy_to(span{inputs[i]}); }
    stream << synchronize();
    check_guards(output);
    for (size_t i = 0u; i < fixture.expected.size(); i++) {
        auto actual = output[padding + i];
        auto expected = fixture.expected[i];
        expect(std::isfinite(actual) && std::abs(actual - expected) <= 5e-5 + 5e-5 * std::abs(expected))
            << "element " << i << " actual " << actual << " expected " << expected;
    }
    for (size_t i = 0u; i < inputs.size(); i++) {
        check_guards(inputs[i]);
        expect(std::equal(fixture.inputs[i].begin(), fixture.inputs[i].end(), inputs[i].begin() + padding))
            << "read-only input " << i << " changed";
    }
}
}// namespace

int main(int argc, char *argv[]) {
    if (argc < 2 || string_view{argv[1]} != "metal4") {
        LUISA_INFO("Usage: {} metal4 [test-filter] (no legacy Metal/TIRx fallback)", argv[0]);
        return 2;
    }
    boost::ut::detail::cfg::parse_arg_with_fallback(argc, const_cast<const char **>(argv));
    auto [context, device] = test::create_device(argc, argv);
    using test::tile_llm::RowOp;
    "tile_xir_metal4_llm_ragged_rows_and_packet_mapping"_test = [&] {
        for (auto op : {RowOp::RMS_NORM, RowOp::LAYER_NORM, RowOp::SWIGLU, RowOp::GELU_RESIDUAL, RowOp::MASKED_SOFTMAX}) {
            for (auto width : {7, 32, 65}) {
                auto fixture = test::tile_llm::rows(op, 17, width);
                run(device, fixture, 1u);
                run(device, fixture, width < 32 ? 0u : 32u);
                if (width < 32) {
                    tile::bridge::xir::PlannerOptions planner{.block_size = 64u, .local_lanes = 32u};
                    auto rejected = tile::compile(device, fixture.kernel, {.threads_per_group = 64u, .xir = &planner});
                    expect(!static_cast<bool>(rejected));
                    expect(rejected.metadata().error.find("local-axis distribution") != string::npos) << rejected.metadata().error;
                }
            }
        }
    };
    "tile_xir_metal4_bounded_wide_reduction_tails"_test = [&] {
        for (auto width : {128, 129}) {
            auto fixture = test::tile_llm::rows(RowOp::RMS_NORM, 17, width);
            for (auto lanes : {1u, 32u}) { run(device, fixture, lanes); }
        }
        for (auto width : {1024, 1025}) {
            auto fixture = test::tile_llm::rows(RowOp::MASKED_SOFTMAX, 17, width);
            for (auto lanes : {1u, 32u}) { run(device, fixture, lanes); }
        }
    };
    "tile_xir_metal4_rope_complete_program_mapping"_test = [&] {
        for (auto width : {6, 32, 66}) { run(device, test::tile_llm::rows(RowOp::ROPE, 17, width), 1u); }
    };
    return 0;
}

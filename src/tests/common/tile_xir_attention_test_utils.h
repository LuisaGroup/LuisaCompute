#pragma once

// Program-team projection/reduction and full online attention regressions
// shared by SIMD and Metal4. The key and value phases use different axes;
// ragged outputs must not predicate away lanes needed by a collective read.
#include "ut/ut.hpp"
#include "tile_llm_test_utils.h"
#include <luisa/core/logging.h>
#include <luisa/runtime/stream.h>
#include <luisa/tile/runtime.h>
#include <luisa/tile/bridge/xir/planner.h>
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <limits>

namespace luisa::test::tile_xir {

// A logical program owns several rows, but this reduction intentionally reads
// only its first row. The same scalar is then added to every retained row.
// This is the positive multi-axis admission case in test_xir.cpp, not a
// flattened reduction or one independent row sum per output row.
inline void program_team_first_row_reductions(compute::Device &device) {
    using namespace compute;
    using namespace compute::tile;
    using namespace boost::ut;
    constexpr auto programs = uint32_t{17u};
    constexpr auto pad = size_t{17u};
    constexpr auto guard = -719.5f;
    auto packet_width = device.compute_warp_size();
    auto block = std::max(32u, 2u * packet_width);
    for (auto height : {2u, 3u}) {
        for (auto width : {7u, 65u}) {
            auto kernel = tile_kernel("program_team_first_row_reduction", [=](TensorView<const float, 2> A, TensorView<float, 2> O) {
                              auto m = axis("m", height), n = axis("n", width);
                              for (auto &program : parallel(shape(programs))) {
                                  auto input = A[coord(program.index() * height, 0), shape(m, n)];
                                  auto sum = Scalar<float>{0.0f};
                                  for (auto &step : program.reduce(shape(n), reduction::unordered_tree)) {
                                      sum += input.at(coord(0, step.index()));
                                  }
                                  O(coord(program.index() * height, 0), shape(m, n)).store(input + sum);
                              }
                          }).capture(tensor_shape(programs * height, width), tensor_shape(programs * height, width));
            expect(kernel.valid()) << "height=" << height << " width=" << width;
            if (!kernel.valid()) { continue; }
            auto count = static_cast<size_t>(programs) * height * width;
            vector<float> original(count + 2u * pad, guard);
            for (size_t i = 0u; i < count; i++) {
                original[pad + i] = static_cast<float>(static_cast<int32_t>(i % 29u) - 14) * 0.25f + static_cast<float>(i / width) * 0.5f;
            }
            auto a = device.create_buffer<float>(original.size()), o = device.create_buffer<float>(original.size());
            auto stream = device.create_stream(StreamTag::COMPUTE);
            for (auto lanes : {1u, packet_width}) {
                auto options = bridge::xir::PlannerOptions{.block_size = block, .local_lanes = lanes};
                auto shader = compile(device, kernel, {.threads_per_group = block, .xir = &options}, {.enable_fast_math = false});
                expect(static_cast<bool>(shader)) << "height=" << height << " width=" << width << " lanes=" << lanes << ": " << shader.metadata().error;
                if (!shader) { continue; }
                expect(eq(shader.metadata().dispatch_size.x, programs * lanes));
                expect(shader.metadata().realization.find(format("local_lanes={};", lanes)) != string::npos);
                auto input = original;
                vector<float> output(original.size(), guard);
                std::fill(output.begin() + pad, output.end() - pad, std::numeric_limits<float>::quiet_NaN());
                stream << a.copy_from(span{input}) << o.copy_from(span{output})
                       << shader(a.view(pad, count), o.view(pad, count)).dispatch()
                       << a.copy_to(span{input}) << o.copy_to(span{output}) << synchronize();
                expect(input == original) << "first-row reduction must preserve the entire input and guards";
                expect(std::all_of(output.begin(), output.begin() + pad, [](float value) { return value == guard; }));
                expect(std::all_of(output.end() - pad, output.end(), [](float value) { return value == guard; }));
                for (size_t program = 0u; program < programs; program++) {
                    auto base = pad + program * height * width;
                    auto sum = 0.0;
                    for (size_t column = 0u; column < width; column++) { sum += static_cast<double>(original[base + column]); }
                    for (size_t row = 0u; row < height; row++) {
                        for (size_t column = 0u; column < width; column++) {
                            auto index = base + row * width + column;
                            auto expected = static_cast<double>(original[index]) + sum;
                            auto actual = output[index];
                            expect(std::isfinite(actual) && std::abs(static_cast<double>(actual) - expected) <= 1e-5)
                                << "height=" << height << " width=" << width << " lanes=" << lanes
                                << " program=" << program << " row=" << row << " column=" << column;
                        }
                    }
                }
            }
        }
    }
}

// A uniform extract and an owner-local extract share the very same source
// layout. Keeping the phase's local coordinate must not turn literal zero
// into this lane's column, including the inactive output lanes of a tail and
// multiple local slots. The shortened physical input also tests a nonzero
// fallback independently of local-slot padding. A nonzero literal projection
// remains unsupported.
inline void program_team_projection_reads(compute::Device &device) {
    using namespace compute;
    using namespace compute::tile;
    using namespace boost::ut;
    auto lanes = device.compute_warp_size();
    auto block = std::max(32u, 2u * lanes);
    constexpr auto pad = size_t{17u};
    constexpr auto guard = -719.5f;
    constexpr auto fallback = 7.25f;
    for (auto width : {7u, lanes + 3u, 2u * lanes + 1u}) {
        for (auto padded_input : {false, true}) {
            auto physical_width = padded_input ? width - 2u : width;
            auto kernel = tile_kernel("program_team_projection_reads", [=](TensorView<const float, 2> A, TensorView<float, 1> O) {
                              auto m = axis("m", 4), n = axis("n", width);
                              for (auto &program : parallel(shape(2))) {
                                  auto input = A.tile(coord(program.index() * 4, 0), shape(m, n), bounds::zero).load(fallback);
                                  auto output = map<float>(shape(n), [&](const Nest &element) {
                                      return input.at(coord(0, 0)) + 2.0f * input.at(coord(0, element.index())) + cast<float>(element.index());
                                  });
                                  O(coord(program.index() * width), shape(n)).store(output);
                              }
                          }).capture(tensor_shape(8, physical_width), tensor_shape(2u * width));
            expect(kernel.valid()) << "width=" << width << " padded_input=" << padded_input;
            if (!kernel.valid()) { continue; }
            auto options = bridge::xir::PlannerOptions{.block_size = block, .local_lanes = lanes};
            auto shader = compile(device, kernel, {.threads_per_group = block, .xir = &options}, {.enable_fast_math = false});
            expect(static_cast<bool>(shader)) << "width=" << width << " padded_input=" << padded_input << ": " << shader.metadata().error;
            if (!shader) { continue; }
            expect(eq(shader.metadata().dispatch_size.x, 2u * lanes));
            expect(shader.metadata().realization.find(format("local_lanes={};", lanes)) != string::npos);
            auto input_count = size_t{8u} * physical_width, output_count = size_t{2u} * width;
            vector<float> input(input_count + 2u * pad, guard), output(output_count + 2u * pad, guard);
            for (size_t i = 0u; i < input_count; i++) { input[pad + i] = static_cast<float>(static_cast<int32_t>(i) - 23) * 0.25f; }
            auto original = input;
            std::fill(output.begin() + pad, output.end() - pad, std::numeric_limits<float>::quiet_NaN());
            auto a = device.create_buffer<float>(input.size()), o = device.create_buffer<float>(output.size());
            auto stream = device.create_stream(StreamTag::COMPUTE);
            stream << a.copy_from(span{input}) << o.copy_from(span{output})
                   << shader(a.view(pad, input_count), o.view(pad, output_count)).dispatch()
                   << a.copy_to(span{input}) << o.copy_to(span{output}) << synchronize();
            expect(input == original) << "projection reads must preserve the entire input and guards";
            expect(std::all_of(output.begin(), output.begin() + pad, [](float value) { return value == guard; }));
            expect(std::all_of(output.end() - pad, output.end(), [](float value) { return value == guard; }));
            for (size_t program = 0u; program < 2u; program++) {
                for (size_t column = 0u; column < width; column++) {
                    auto base = pad + program * 4u * physical_width;
                    auto value = column < physical_width ? original[base + column] : fallback;
                    auto expected = original[base] + 2.0f * value + static_cast<float>(column);
                    auto actual = output[pad + program * width + column];
                    expect(std::isfinite(actual) && std::abs(actual - expected) <= 1e-6f) << "width=" << width << " padded_input=" << padded_input << " program=" << program << " column=" << column;
                }
            }
        }
    }
    program_team_first_row_reductions(device);
}

inline void sibling_parallel_outer_views_capture() {
    using namespace compute::tile;
    using namespace boost::ut;
    auto definition = tile_kernel("sibling_parallel_outer_views", [](TensorView<const float, 2> input,
                                                                     TensorView<float, 2> temporary,
                                                                     TensorView<float, 2> output) {
        auto rows = axis("rows", 3), columns = axis("columns", 7), one_row = axis("one_row", 1);
        for (auto &program : parallel(shape(1))) {
            auto bias = full<float>(shape(one_row, columns), 2.0f);
            for (auto &row : program.parallel(shape(rows))) {
                auto origin = coord(row.index(rows), 0);
                temporary(origin, shape(one_row, columns)).store(input[origin, shape(one_row, columns)] + bias);
            }
            for (auto &column : program.parallel(shape(columns))) {
                for (auto &row : column.serial(shape(rows))) {
                    auto origin = coord(row.index(rows), column.index(columns));
                    output(coord(column.index(columns), row.index(rows)), shape(1, 1))
                        .store(temporary[origin, shape(1, 1)] * 3.0f);
                }
            }
        }
    });
    auto kernel = definition.capture(tensor_shape(3, 7), tensor_shape(3, 7), tensor_shape(7, 3));
    expect(kernel.valid());
    if (!kernel.valid()) { return; }
    uint32_t parallels = 0u, serials = 0u, loads = 0u, stores = 0u;
    auto collect = [&](auto &&self, const Region &region) -> void {
        for (auto block : region.blocks()) {
            for (auto operation : block->operations()) {
                parallels += operation->kind() == OperationKind::PARALLEL ? 1u : 0u;
                serials += operation->kind() == OperationKind::SERIAL ? 1u : 0u;
                loads += operation->kind() == OperationKind::VIEW_LOAD ? 1u : 0u;
                stores += operation->kind() == OperationKind::VIEW_STORE ? 1u : 0u;
                for (const auto &child : operation->regions()) { self(self, *child); }
            }
        }
    };
    collect(collect, kernel.function().body());
    expect(eq(parallels, 3u));
    expect(eq(serials, 1u));
    expect(eq(loads, 2u));
    expect(eq(stores, 2u));
    // Capture/verification only: sibling execution-to-hardware mapping is
    // separate ongoing work. Do not force this fixture through an unsupported
    // backend map or confuse ancestor resource access with outer Tile mutation.
}

[[nodiscard]] inline tile_llm::Case attention_snapshot_fixture() {
    using namespace compute::tile;
    constexpr int64_t keys = 35, channels = 7, value_channels = 3, block_keys = 33;
    auto fixture = tile_llm::attention(1, 1, 1, 1, keys, channels, value_channels, 1, block_keys);
    auto scale = 1.0f / std::sqrt(static_cast<float>(channels));
    auto definition = tile_kernel("attention_definition_time_snapshots", [=](TensorView<float, 4> Q,
                                                                             TensorView<float, 4> K,
                                                                             TensorView<float, 4> V,
                                                                             TensorView<float, 4> O) {
        auto b = axis("b", 1), h = axis("h", 1), m = axis("m", 1), n = axis("n", block_keys);
        auto d = axis("d", channels), dv = axis("dv", value_channels);
        // Exactly one logical program: these deliberate writes cannot race
        // another query block or GQA head sharing the same key/value buffers.
        for (auto &nest : parallel(shape(1))) {
            auto query = Q.tile(coord(0, 0, 0, 0), shape(b, h, m, d), bounds::zero).load();
            Q(coord(0, 0, 0, 0), shape(b, h, m, d)).store(full<float>(shape(b, h, m, d), 13.0f));
            auto row_max = full<float>(shape(b, h, m), -1e30f);
            auto row_sum = zeros<float>(shape(b, h, m));
            auto acc = zeros<float>(shape(b, h, m, dv));
            for (auto &step : nest.pipeline(shape(ceil_div(keys, block_keys)), {.stages = 2u, .initiation_interval = 1u})) {
                auto k0 = step.index() * block_keys;
                step.stage("load");
                auto key = K.tile(coord(0, 0, k0, 0), shape(b, h, n, d), bounds::zero).load();
                auto value = V.tile(coord(0, 0, k0, 0), shape(b, h, n, dv), bounds::zero).load();
                K(coord(0, 0, k0, 0), shape(b, h, n, d), bounds::zero).store(full<float>(shape(b, h, n, d), 17.0f));
                V(coord(0, 0, k0, 0), shape(b, h, n, dv), bounds::zero).store(full<float>(shape(b, h, n, dv), 19.0f));
                step.stage("score");
                auto score = mma(query, key, zeros<float>(shape(b, h, m, n))) * scale;
                auto valid = iota(n) + k0 < keys;
                auto masked = ite(valid, score, -1e30f);
                auto next_max = max(row_max, reduce(masked, n, maximum));
                auto alpha = exp(row_max - next_max);
                auto probability = ite(valid, exp(masked - next_max), 0.0f);
                step.stage("update");
                row_sum = row_sum * alpha + reduce(probability, n, add);
                acc = mma(probability, value, acc * alpha);
                row_max = next_max;
            }
            O(coord(0, 0, 0, 0), shape(b, h, m, dv)).store(acc / row_sum);
        }
    });
    fixture.kernel = definition.capture(tensor_shape(1, 1, 1, channels), tensor_shape(1, 1, keys, channels),
                                        tensor_shape(1, 1, keys, value_channels), tensor_shape(1, 1, 1, value_channels));
    return fixture;
}

inline void program_team_attention(compute::Device &device, const tile_llm::Case &fixture,
                                   int64_t query_block, bool overwritten_inputs = false) {
    using namespace compute;
    using namespace compute::tile;
    using namespace boost::ut;
    auto lanes = device.compute_warp_size();
    auto block_size = std::max(32u, 2u * lanes);
    LUISA_ASSERT(query_block > 0 && lanes > 0u && fixture.shapes[3].size() == 4u, "Invalid program-team attention fixture");
    expect(fixture.kernel.valid());
    if (!fixture.kernel.valid()) { return; }
    auto options = bridge::xir::PlannerOptions{.block_size = block_size, .local_lanes = lanes};
    auto shader = compile(device, fixture.kernel, {.threads_per_group = block_size, .xir = &options},
                          {.enable_fast_math = false});
    expect(static_cast<bool>(shader)) << shader.metadata().error;
    if (!shader) { return; }
    const auto &metadata = shader.metadata();
    auto batches = fixture.shapes[3][0], heads = fixture.shapes[3][1], queries = fixture.shapes[3][2];
    auto value_channels = fixture.shapes[3][3], kv_heads = fixture.shapes[2][1], keys = fixture.shapes[2][2];
    auto programs = batches * heads * ceil_div(queries, query_block);
    expect(eq(metadata.dispatch_size.x, static_cast<uint32_t>(programs) * lanes));
    expect(eq(metadata.dispatch_size.y, 1u));
    expect(eq(metadata.dispatch_size.z, 1u));
    expect(eq(shader.block_size().x, block_size));
    expect(metadata.realization.find(format("local_lanes={};", lanes)) != string::npos) << metadata.realization;
    expect(metadata.realization.find("fast_math=false;") != string::npos) << metadata.realization;

    constexpr auto pad = size_t{17};
    constexpr auto guard = -719.5f;
    constexpr auto overwritten = std::array{13.0f, 17.0f, 19.0f};
    auto stream = device.create_stream(StreamTag::COMPUTE);
    vector<Buffer<float>> buffers;
    for (const auto &input : fixture.inputs) { buffers.emplace_back(device.create_buffer<float>(input.size() + 2u * pad)); }
    auto output = device.create_buffer<float>(fixture.expected.size() + 2u * pad);
    auto check_guards = [&](span<const float> values) {
        expect(std::all_of(values.begin(), values.begin() + pad, [](float value) { return value == guard; }));
        expect(std::all_of(values.end() - pad, values.end(), [](float value) { return value == guard; }));
    };
    for (auto uniform : {false, true}) {
        LUISA_INFO("Program-team attention: B={} Hq={} Hkv={} Q={} K={} D={} Dv={} BQ={} W={} uniform={} overwritten={}",
                   batches, heads, kv_heads, queries, keys, fixture.shapes[0][3], value_channels, query_block, lanes, uniform, overwritten_inputs);
        auto inputs = fixture.inputs;
        auto expected = fixture.expected;
        if (uniform) {
            // The same shader also has a closed-form oracle: zero Q/K makes
            // every visible key equiprobable. Nonconstant signed V includes
            // contributions owned by lanes with no valid output column.
            std::fill(inputs[0].begin(), inputs[0].end(), 0.0f);
            std::fill(inputs[1].begin(), inputs[1].end(), 0.0f);
            for (size_t i = 0u; i < inputs[2].size(); i++) {
                inputs[2][i] = static_cast<float>(static_cast<int64_t>((i * 17u + 3u) % 29u) - 14) * 0.125f;
            }
            for (int64_t batch = 0; batch < batches; batch++) {
                for (int64_t head = 0; head < heads; head++) {
                    auto kv_head = head / (heads / kv_heads);
                    for (int64_t q = 0; q < queries; q++) {
                        auto visible = keys - queries + q + 1;
                        for (int64_t channel = 0; channel < value_channels; channel++) {
                            auto sum = 0.0;
                            for (int64_t k = 0; k < visible; k++) {
                                sum += inputs[2][((batch * kv_heads + kv_head) * keys + k) * value_channels + channel];
                            }
                            expected[((batch * heads + head) * queries + q) * value_channels + channel] = sum / static_cast<double>(visible);
                        }
                    }
                }
            }
        }
        std::array<vector<float>, 3u> actual_inputs;
        for (size_t i = 0u; i < inputs.size(); i++) {
            actual_inputs[i].resize(inputs[i].size() + 2u * pad, guard);
            std::copy(inputs[i].begin(), inputs[i].end(), actual_inputs[i].begin() + pad);
            stream << buffers[i].copy_from(span{actual_inputs[i]});
        }
        vector<float> actual(fixture.expected.size() + 2u * pad, guard);
        std::fill(actual.begin() + pad, actual.end() - pad, std::numeric_limits<float>::quiet_NaN());
        stream << output.copy_from(span{actual})
               << shader(buffers[0].view(pad, inputs[0].size()), buffers[1].view(pad, inputs[1].size()),
                         buffers[2].view(pad, inputs[2].size()), output.view(pad, expected.size()))
                      .dispatch()
               << output.copy_to(span{actual});
        for (size_t i = 0u; i < inputs.size(); i++) { stream << buffers[i].copy_to(span{actual_inputs[i]}); }
        stream << synchronize();
        check_guards(actual);
        for (size_t i = 0u; i < expected.size(); i++) {
            expect(std::isfinite(actual[pad + i]) && std::abs(actual[pad + i] - expected[i]) <= 5e-5 + 5e-5 * std::abs(expected[i]))
                << "attention element=" << i << " uniform=" << uniform << " overwritten=" << overwritten_inputs
                << " actual=" << actual[pad + i] << " expected=" << expected[i];
        }
        for (size_t input = 0u; input < inputs.size(); input++) {
            check_guards(actual_inputs[input]);
            auto correct = true;
            for (size_t i = 0u; correct && i < inputs[input].size(); i++) {
                auto expected_value = overwritten_inputs ? overwritten[input] : inputs[input][i];
                correct = std::bit_cast<uint32_t>(actual_inputs[input][pad + i]) == std::bit_cast<uint32_t>(expected_value);
            }
            expect(correct) << "attention input=" << input << " uniform=" << uniform << " overwritten=" << overwritten_inputs;
        }
    }
}

}// namespace luisa::test::tile_xir

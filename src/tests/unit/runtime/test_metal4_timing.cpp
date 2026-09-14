// Test opt-in Metal4 Runtime feedback and precise dispatch timestamps.
// Checks real Tile output, sample boundaries, stream isolation and failures,
// including a standalone SIMT indirect range as the first sampled command;
// elapsed values are evidence, not performance thresholds.

#include "ut/ut.hpp"
#include "test_device.h"
#include "tile_llm_test_utils.h"
#include <luisa/core/logging.h>
#include <luisa/backends/ext/metal4_timing_ext.h>
#include <luisa/runtime/stream.h>
#include <luisa/runtime/dispatch_buffer.h>
#include <luisa/dsl/syntax.h>
#include <luisa/dsl/dispatch_indirect.h>
#include <luisa/tile/bridge/xir/planner.h>
#include <luisa/tile/runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <numeric>

using namespace luisa;
using namespace luisa::compute;
using namespace boost::ut;
using namespace boost::ut::literals;

namespace {

constexpr size_t padding = 17u;
constexpr float guard = -719.5f;

void check_sample(const Metal4TimingSample &sample, uint64_t sample_id,
                  uint32_t expected_dispatches, bool precise, uint64_t frequency,
                  uint3 dispatch_size, uint3 block_size, bool expect_empty_submission) {
    expect(sample.error.empty()) << sample.error;
    expect(!sample.overflow);
    expect(eq(sample.sample_id, sample_id));
    expect(eq(sample.dispatch_timestamps_enabled, precise));
    expect(eq(sample.dispatches.size(), static_cast<size_t>(expected_dispatches)));
    expect(eq(sample.timestamp_frequency_hz, precise ? frequency : uint64_t{0u}));
    auto dispatched = uint64_t{0u};
    auto empty_submissions = size_t{0u};
    vector<uint64_t> command_ordinals;
    for (const auto &command : sample.command_buffers) {
        expect(std::find(command_ordinals.begin(), command_ordinals.end(), command.ordinal) == command_ordinals.end());
        command_ordinals.emplace_back(command.ordinal);
        dispatched += command.dispatch_count;
        expect(!command.contains_non_dispatch_work);
        expect(gt(command.host_commit_begin_nanoseconds, uint64_t{0u}));
        expect(ge(command.host_commit_return_nanoseconds, command.host_commit_begin_nanoseconds));
        expect(ge(command.host_feedback_begin_nanoseconds, command.host_commit_begin_nanoseconds));
        expect(ge(command.host_callbacks_end_nanoseconds, command.host_feedback_begin_nanoseconds));
        expect(ge(command.host_completion_publish_nanoseconds, command.host_callbacks_end_nanoseconds));
        // Feedback may run before commit() returns. Its clock is not the GPU
        // clock, so neither that ordering nor a cross-clock difference is used.
        if (command.dispatch_count != 0u) { expect(command.valid); }
        if (command.valid) {
            expect(std::isfinite(command.gpu_begin_seconds) && std::isfinite(command.gpu_end_seconds));
            expect(command.gpu_begin_seconds > 0.0 && command.gpu_end_seconds > command.gpu_begin_seconds);
        }
        if (command.dispatch_count == 0u) { empty_submissions++; }
        auto associated = std::count_if(sample.dispatches.begin(), sample.dispatches.end(), [&](const auto &dispatch) {
            return dispatch.command_buffer_ordinal == command.ordinal;
        });
        expect(eq(static_cast<uint64_t>(associated), command.dispatch_count));
    }
    expect(eq(dispatched, static_cast<uint64_t>(expected_dispatches)));
    if (expect_empty_submission) { expect(gt(empty_submissions, size_t{0u})); }
    vector<uint64_t> dispatch_ordinals;
    for (const auto &dispatch : sample.dispatches) {
        expect(std::find(dispatch_ordinals.begin(), dispatch_ordinals.end(), dispatch.ordinal) == dispatch_ordinals.end());
        dispatch_ordinals.emplace_back(dispatch.ordinal);
        expect(std::find(command_ordinals.begin(), command_ordinals.end(), dispatch.command_buffer_ordinal) != command_ordinals.end());
        expect(all(dispatch.dispatch_size == dispatch_size));
        expect(all(dispatch.block_size == block_size));
        if (!sample.dispatches.empty()) { expect(eq(dispatch.shader_checksum, sample.dispatches.front().shader_checksum)); }
        if (precise) {
            expect(dispatch.valid);
            expect(gt(dispatch.begin_ticks, uint64_t{0u}));
            expect(gt(dispatch.end_ticks, dispatch.begin_ticks));
            expect(std::isfinite(dispatch.elapsed_nanoseconds) && dispatch.elapsed_nanoseconds > 0.0);
            if (frequency != 0u && dispatch.end_ticks > dispatch.begin_ticks) {
                auto expected = static_cast<long double>(dispatch.end_ticks - dispatch.begin_ticks) * 1.0e9L /
                                static_cast<long double>(frequency);
                expect(std::abs(static_cast<long double>(dispatch.elapsed_nanoseconds) - expected) <=
                       std::max(1.0e-6L, std::abs(expected) * 1.0e-12L));
            }
        } else {
            expect(!dispatch.valid);
            expect(eq(dispatch.begin_ticks, uint64_t{0u}));
            expect(eq(dispatch.end_ticks, uint64_t{0u}));
            expect(std::abs(dispatch.elapsed_nanoseconds) < 1e-12);
        }
    }
}

}// namespace

int main(int argc, char *argv[]) {
    // Member compare keeps both logical operands builtin bool, avoiding
    // Boost.UT's eager logical overloads when an argument or env var is absent.
    if (argc < 2 || string_view{argv[1]}.compare("metal4") != 0) {
        LUISA_INFO("Usage: {} metal4 [test-filter]; set LUISA_TEST_REQUIRE_METAL4_TIMESTAMPS=1 to require precise counters", argv[0]);
        return 2;
    }
    boost::ut::detail::cfg::parse_arg_with_fallback(argc, const_cast<const char **>(argv));
    auto [context, device] = test::create_device(argc, argv);
    auto timing = device.extension<Metal4TimingExt>();
    expect(timing != nullptr);
    if (timing == nullptr) { return 1; }
    auto capabilities = timing->capabilities();
    LUISA_INFO("Metal4 timing capabilities: feedback={}, timestamp_heap={}, frequency_hz={}, error='{}'",
               capabilities.command_buffer_feedback, capabilities.timestamp_heap,
               capabilities.timestamp_frequency_hz, capabilities.error);
    expect(capabilities.command_buffer_feedback);
    auto require_counters = std::getenv("LUISA_TEST_REQUIRE_METAL4_TIMESTAMPS");
    if (require_counters != nullptr && string_view{require_counters}.compare("1") == 0) {
        expect(capabilities.timestamp_heap) << capabilities.error;
        if (!capabilities.timestamp_heap) { return 1; }
    }
    if (capabilities.timestamp_heap) {
        expect(capabilities.error.empty()) << capabilities.error;
        expect(gt(capabilities.timestamp_frequency_hz, uint64_t{0u}));
    } else {
        LUISA_WARNING("Metal4 precise-counter correctness NOT VALIDATED: {}. Feedback-only checks still run.", capabilities.error);
    }

    auto fixture = test::tile_llm::rows(test::tile_llm::RowOp::SWIGLU, 17, 65);
    tile::bridge::xir::PlannerOptions planner{.block_size = 64u, .local_lanes = 32u};
    auto shader = tile::compile(device, fixture.kernel, {.threads_per_group = 64u, .xir = &planner});
    expect(static_cast<bool>(shader)) << shader.metadata().error;
    if (!shader) { return 1; }
    expect(shader.metadata().realization.find("TileIR -> XIR SSA -> LLVM AIR -> Metal4 Runtime") != string::npos);
    LUISA_INFO("Metal4 timing setup: creating primary stream");
    auto stream = device.create_stream(StreamTag::COMPUTE);
    LUISA_INFO("Metal4 timing setup: uploading shared inputs");
    vector<Buffer<float>> inputs;
    for (const auto &data : fixture.inputs) {
        inputs.emplace_back(device.create_buffer<float>(data.size()));
        stream << inputs.back().copy_from(span{data});
    }
    auto output = device.create_buffer<float>(fixture.expected.size() + 2u * padding);
    auto reset = [&](Stream &target, Buffer<float> &buffer) {
        vector<float> values(fixture.expected.size() + 2u * padding, guard);
        std::fill(values.begin() + padding, values.end() - padding, std::numeric_limits<float>::quiet_NaN());
        target << buffer.copy_from(span{values}) << synchronize();
    };
    auto dispatch = [&](Stream &target, Buffer<float> &buffer, uint32_t count) {
        CommandList commands;
        for (auto i = 0u; i < count; i++) {
            commands << shader(inputs[0], inputs[1], inputs[2], buffer.view(padding, fixture.expected.size())).dispatch();
        }
        target << commands.commit();
    };
    auto check_output = [&](Stream &target, Buffer<float> &buffer) {
        vector<float> values(buffer.size());
        target << buffer.copy_to(span{values}) << synchronize();
        for (size_t i = 0u; i < values.size(); i++) {
            if (i < padding || i >= values.size() - padding) {
                expect(eq(values[i], guard)) << "guard " << i;
            } else {
                auto expected = fixture.expected[i - padding];
                expect(std::isfinite(values[i]) && std::abs(values[i] - expected) <= 5e-5 + 5e-5 * std::abs(expected))
                    << "output " << i - padding;
            }
        }
    };
    LUISA_INFO("Metal4 timing setup: initial primary reset begin");
    reset(stream, output);// Also completes the shared read-only input uploads.
    LUISA_INFO("Metal4 timing setup: initial primary reset complete");

    "metal4_timing_feedback_counts_boundaries_and_empty_submissions"_test = [&] {
        LUISA_INFO("Metal4 timing case: feedback counts, boundaries and empty submissions");
        for (auto count : {1u, 3u}) {
            reset(stream, output);
            dispatch(stream, output, 2u);// Pending before begin: deliberately excluded.
            auto id = uint64_t{100u} + count;
            LUISA_INFO("Metal4 timing feedback: begin sample {}, expected_dispatches={}", id, count);
            auto started = timing->begin_sample(stream.handle(), id, 8u, false);
            expect(started);
            if (!started) { continue; }
            dispatch(stream, output, count);
            stream.synchronize();// Its empty CB belongs to the active sample.
            auto sample = timing->end_sample(stream.handle());
            check_sample(sample, id, count, false, 0u, shader.metadata().dispatch_size, shader.block_size(), true);
            dispatch(stream, output, 2u);// Outside the returned sample.
            check_output(stream, output);
            expect(!timing->end_sample(stream.handle()).error.empty());
            started = timing->begin_sample(stream.handle(), id + 10u, 8u, false);
            expect(started);
            if (!started) { continue; }
            auto empty = timing->end_sample(stream.handle());
            expect(empty.error.empty()) << empty.error;
            expect(eq(empty.sample_id, id + 10u));
            expect(empty.dispatches.empty());
            expect(empty.command_buffers.empty());// Neither boundary's own drain is recorded.
        }
    };

    if (capabilities.timestamp_heap) {
        "metal4_timing_precise_ticks_conversion_and_command_buffer_ownership"_test = [&] {
            LUISA_INFO("Metal4 timing case: precise ticks and command-buffer ownership");
            reset(stream, output);
            auto started = timing->begin_sample(stream.handle(), 200u, 8u, true);
            expect(started);
            if (!started) { return; }
            dispatch(stream, output, 1u);
            stream.synchronize();
            dispatch(stream, output, 2u);
            auto sample = timing->end_sample(stream.handle());
            check_sample(sample, 200u, 3u, true, capabilities.timestamp_frequency_hz,
                         shader.metadata().dispatch_size, shader.block_size(), true);
            check_output(stream, output);
        };
    }

    "metal4_timing_different_streams_have_independent_samples"_test = [&] {
        LUISA_INFO("Metal4 timing case: stream isolation; creating secondary stream");
        auto other_stream = device.create_stream(StreamTag::COMPUTE);
        auto other_output = device.create_buffer<float>(fixture.expected.size() + 2u * padding);
        LUISA_INFO("Metal4 timing isolation: primary reset begin");
        reset(stream, output);
        LUISA_INFO("Metal4 timing isolation: secondary reset begin");
        reset(other_stream, other_output);
        LUISA_INFO("Metal4 timing isolation: primary sample begin");
        auto first = timing->begin_sample(stream.handle(), 301u, 8u, false);
        LUISA_INFO("Metal4 timing isolation: secondary sample begin");
        auto second = timing->begin_sample(other_stream.handle(), 302u, 8u, false);
        expect(first && second);
        if (!first || !second) {
            if (first) { static_cast<void>(timing->end_sample(stream.handle())); }
            if (second) { static_cast<void>(timing->end_sample(other_stream.handle())); }
            return;
        }
        LUISA_INFO("Metal4 timing isolation: dispatching independent outputs");
        dispatch(stream, output, 2u);
        dispatch(other_stream, other_output, 3u);
        LUISA_INFO("Metal4 timing isolation: primary sample end");
        auto a = timing->end_sample(stream.handle());
        LUISA_INFO("Metal4 timing isolation: secondary sample end");
        auto b = timing->end_sample(other_stream.handle());
        check_sample(a, 301u, 2u, false, 0u, shader.metadata().dispatch_size, shader.block_size(), false);
        check_sample(b, 302u, 3u, false, 0u, shader.metadata().dispatch_size, shader.block_size(), false);
        check_output(stream, output);
        check_output(other_stream, other_output);
    };

    "metal4_timing_overflow_is_an_error_without_losing_dispatch_counts"_test = [&] {
        LUISA_INFO("Metal4 timing case: overflow preserves actual dispatches");
        reset(stream, output);
        auto started = timing->begin_sample(stream.handle(), 400u, 1u, false);
        expect(started);
        if (!started) { return; }
        dispatch(stream, output, 3u);
        auto sample = timing->end_sample(stream.handle());
        expect(eq(sample.sample_id, uint64_t{400u}));
        expect(sample.overflow);
        expect(!sample.error.empty());
        expect(eq(sample.dispatches.size(), size_t{1u}));// Retain the valid prefix as raw evidence only.
        auto total = std::accumulate(sample.command_buffers.begin(), sample.command_buffers.end(), uint64_t{0u},
                                     [](uint64_t count, const auto &command) { return count + command.dispatch_count; });
        expect(eq(total, uint64_t{3u}));
        check_output(stream, output);// Overflow must not truncate actual execution.
    };

    "metal4_timing_lifecycle_errors_and_normal_dispatch_after_disarm"_test = [&] {
        LUISA_INFO("Metal4 timing case: lifecycle errors and dispatch after disarm");
        expect(!timing->end_sample(stream.handle()).error.empty());
        expect(!timing->begin_sample(stream.handle(), 500u, 0u, false));
        auto started = timing->begin_sample(stream.handle(), 501u, 8u, false);
        expect(started);
        if (!started) { return; }
        expect(!timing->begin_sample(stream.handle(), 502u, 8u, false));
        dispatch(stream, output, 1u);
        auto sample = timing->end_sample(stream.handle());
        check_sample(sample, 501u, 1u, false, 0u, shader.metadata().dispatch_size, shader.block_size(), false);
        reset(stream, output);
        dispatch(stream, output, 2u);
        check_output(stream, output);
        expect(!timing->end_sample(stream.handle()).error.empty());
    };

    "metal4_timing_first_command_indirect_range_reports_unsupported_without_losing_work"_test = [&] {
        LUISA_INFO("Metal4 timing case: indirect range as the first sampled command");
        constexpr auto element_count = 65u;
        constexpr auto indirect_kernel_id = 7u;
        constexpr auto indirect_guard = uint32_t{0xd3adb33fu};
        auto indirect = device.create_indirect_dispatch_buffer(1u);
        auto indirect_output = device.create_buffer<uint32_t>(element_count + 2u * padding);
        // These are independent SIMT kernels, not Tile/SIMT DSL mixing. Their
        // setup dispatch and buffer initialization must finish before sampling.
        Kernel1D prepare_indirect = [](Var<IndirectDispatchBuffer> commands) noexcept {
            commands.set_dispatch_count(1u);
            commands.set_kernel(0u, make_uint3(32u, 1u, 1u),
                                make_uint3(element_count, 1u, 1u), indirect_kernel_id);
        };
        Kernel1D run_indirect = [](BufferVar<uint32_t> values) noexcept {
            set_block_size(32u, 1u, 1u);
            auto i = dispatch_x();
            values.write(i, i * 3u + kernel_id());
        };
        auto prepare_shader = device.compile(prepare_indirect);
        auto indirect_shader = device.compile(run_indirect);
        stream << prepare_shader(indirect).dispatch(1u) << synchronize();
        for (auto precise : {false, true}) {
            if (precise && !capabilities.timestamp_heap) { continue; }
            vector<uint32_t> values(indirect_output.size(), indirect_guard);
            stream << indirect_output.copy_from(span{values}) << synchronize();
            auto id = uint64_t{600u} + static_cast<uint64_t>(precise);
            auto started = timing->begin_sample(stream.handle(), id, 8u, precise);
            expect(started);
            if (!started) { continue; }
            // No upload or direct dispatch precedes this range in the sample.
            stream << indirect_shader(indirect_output.view(padding, element_count)).dispatch(indirect, 0u, 1u);
            auto sample = timing->end_sample(stream.handle());
            expect(eq(sample.sample_id, id));
            expect(eq(sample.dispatch_timestamps_enabled, precise));
            expect(eq(sample.timestamp_frequency_hz, precise ? capabilities.timestamp_frequency_hz : uint64_t{0u}));
            expect(sample.error == "Metal4 per-dispatch timing does not support indirect command ranges") << sample.error;
            expect(!sample.overflow);
            expect(sample.dispatches.empty());// An opaque range is not one measured direct dispatch.
            expect(eq(sample.command_buffers.size(), size_t{1u}));
            for (const auto &command : sample.command_buffers) {
                expect(command.contains_non_dispatch_work);
                expect(eq(command.dispatch_count, uint64_t{0u}));
                expect(command.valid);
                expect(std::isfinite(command.gpu_begin_seconds) && std::isfinite(command.gpu_end_seconds));
                expect(command.gpu_begin_seconds > 0.0 && command.gpu_end_seconds > command.gpu_begin_seconds);
                expect(gt(command.host_commit_begin_nanoseconds, uint64_t{0u}));
                expect(ge(command.host_commit_return_nanoseconds, command.host_commit_begin_nanoseconds));
                expect(ge(command.host_feedback_begin_nanoseconds, command.host_commit_begin_nanoseconds));
                expect(ge(command.host_callbacks_end_nanoseconds, command.host_feedback_begin_nanoseconds));
                expect(ge(command.host_completion_publish_nanoseconds, command.host_callbacks_end_nanoseconds));
            }
            // Unsupported timing must not suppress the actual range execution.
            stream << indirect_output.copy_to(span{values}) << synchronize();
            for (size_t i = 0u; i < values.size(); i++) {
                auto expected = i < padding || i >= values.size() - padding ?
                                    indirect_guard :
                                    static_cast<uint32_t>(i - padding) * 3u + indirect_kernel_id;
                expect(eq(values[i], expected)) << "indirect output or guard " << i;
            }
            expect(!timing->end_sample(stream.handle()).error.empty());
        }
    };
    return 0;
}

#pragma once

#include <luisa/backends/ext/metal4_timing_ext.h>
#include <luisa/core/logging.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/stream.h>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>

namespace luisa::test {

// Opt-in, separate from the uninstrumented host-wall benchmark. Retain raw
// records and paired feedback-only controls: precise counters can perturb GPU
// scheduling, and command-buffer time is not a pure kernel measurement.
class Metal4BenchmarkTiming final {
private:
    compute::Metal4TimingExt *_ext{nullptr};
    uint64_t _stream{0u};
    uint64_t _next_sample{0u};
    uint64_t _repetitions{0u};
    compute::Metal4TimingCapabilities _capabilities;
    vector<compute::Metal4TimingSample> _throughput;
    vector<compute::Metal4TimingSample> _latency;
    vector<compute::Metal4TimingSample> _control_throughput;
    vector<compute::Metal4TimingSample> _control_latency;

public:
    Metal4BenchmarkTiming(const compute::Device &device, const compute::Stream &stream, bool metal4) {
        auto setting = std::getenv("LUISA_TILE_BENCH_METAL4_TIMING");
        if (setting == nullptr || string_view{setting} == "0") { return; }
        LUISA_ASSERT(metal4 && string_view{setting} == "1", "Metal4 timing requires a Metal4 benchmark and a 0/1 setting");
        _ext = device.extension<compute::Metal4TimingExt>();
        LUISA_ASSERT(_ext != nullptr, "Metal4 timing extension unavailable");
        _stream = stream.handle();
        _capabilities = _ext->capabilities();
        LUISA_ASSERT(_capabilities.command_buffer_feedback && _capabilities.timestamp_heap &&
                         _capabilities.timestamp_frequency_hz != 0u,
                     "Metal4 dispatch timestamps unavailable: {}", _capabilities.error);
    }

    template<typename Submit>
    void measure(Submit &&submit, uint64_t repetitions, uint32_t samples) {
        if (_ext == nullptr) { return; }
        _repetitions = std::min<uint64_t>(repetitions, 64u);
        auto sample = [&](uint64_t count, bool counters) {
            LUISA_ASSERT(_ext->begin_sample(_stream, ++_next_sample, static_cast<uint32_t>(count), counters),
                         "Failed to begin Metal4 timing sample");
            compute::Metal4TimingSample result;
            {
                struct Scope final {
                    compute::Metal4TimingExt *ext;
                    uint64_t stream;
                    compute::Metal4TimingSample &result;
                    ~Scope() noexcept { result = ext->end_sample(stream); }
                } scope{_ext, _stream, result};
                submit(count);
            }
            LUISA_ASSERT(result.error.empty() && !result.overflow, "Metal4 timing failed: {}", result.error);
            LUISA_ASSERT(result.dispatches.size() == count && result.dispatch_timestamps_enabled == counters,
                         "Metal4 timing did not observe the exact requested dispatches");
            auto observed = uint64_t{0u};
            for (const auto &buffer : result.command_buffers) {
                LUISA_ASSERT((buffer.valid || (buffer.dispatch_count == 0u && !buffer.contains_non_dispatch_work)) &&
                                 std::isfinite(buffer.gpu_begin_seconds) && std::isfinite(buffer.gpu_end_seconds),
                             "Invalid Metal4 command-buffer timestamps");
                observed += buffer.dispatch_count;
            }
            LUISA_ASSERT(observed == count, "Metal4 command-buffer dispatch accounting differs from sample");
            for (const auto &dispatch : result.dispatches) {
                LUISA_ASSERT(!counters || (dispatch.valid && dispatch.end_ticks > dispatch.begin_ticks &&
                                           std::isfinite(dispatch.elapsed_nanoseconds) && dispatch.elapsed_nanoseconds > 0.0),
                             "Invalid Metal4 dispatch timestamps");
            }
            return result;
        };
        auto phase = [&](uint64_t count, auto &control, auto &instrumented) {
            for (auto i = 0u; i < samples; i++) {
                if (i % 2u == 0u) { control.emplace_back(sample(count, false)); }
                instrumented.emplace_back(sample(count, true));
                if (i % 2u != 0u) { control.emplace_back(sample(count, false)); }
            }
        };
        phase(_repetitions, _control_throughput, _throughput);
        phase(1u, _control_latency, _latency);
    }

    void print() const {
        if (_ext == nullptr) { return; }
        // Absolute GPU seconds lose small intervals at the surrounding
        // benchmark's shorter precision. Preserve the complete double value.
        auto previous_precision = std::cout.precision(17);
        auto boolean = [](bool value) { return value ? "true" : "false"; };
        std::cout << ",\"device_timing\":{\"method\":\"metal4_precise_dispatch_timestamps_v1\","
                     "\"scope\":\"instrumented_dispatch_intervals\",\"host_samples_instrumented\":false,"
                     "\"zero_overhead_kernel_time\":false,\"repetitions\":"
                  << _repetitions << ",\"capabilities\":{\"timestamp_heap\":" << boolean(_capabilities.timestamp_heap)
                  << ",\"legacy_dispatch_sampling\":" << boolean(_capabilities.legacy_dispatch_sampling)
                  << ",\"legacy_stage_sampling\":" << boolean(_capabilities.legacy_stage_sampling)
                  << ",\"timestamp_frequency_hz\":" << _capabilities.timestamp_frequency_hz << '}';
        auto records = [&](const char *name, const auto &samples) {
            std::cout << ",\"" << name << "\":[";
            auto separator = "";
            for (const auto &sample : samples) {
                std::cout << separator << "{\"sample_id\":" << sample.sample_id
                          << ",\"timestamp_frequency_hz\":" << sample.timestamp_frequency_hz
                          << ",\"dispatch_timestamps_enabled\":" << boolean(sample.dispatch_timestamps_enabled)
                          << ",\"overflow\":" << boolean(sample.overflow)
                          << ",\"error\":" << std::quoted(sample.error.c_str()) << ",\"command_buffers\":[";
                auto buffer_separator = "";
                for (const auto &buffer : sample.command_buffers) {
                    std::cout << buffer_separator << "{\"ordinal\":" << buffer.ordinal
                              << ",\"dispatch_count\":" << buffer.dispatch_count
                              << ",\"contains_non_dispatch_work\":" << boolean(buffer.contains_non_dispatch_work)
                              << ",\"gpu_begin_seconds\":" << buffer.gpu_begin_seconds
                              << ",\"gpu_end_seconds\":" << buffer.gpu_end_seconds
                              << ",\"host_commit_begin_ns\":" << buffer.host_commit_begin_nanoseconds
                              << ",\"host_commit_return_ns\":" << buffer.host_commit_return_nanoseconds
                              << ",\"host_feedback_begin_ns\":" << buffer.host_feedback_begin_nanoseconds
                              << ",\"host_callbacks_end_ns\":" << buffer.host_callbacks_end_nanoseconds
                              << ",\"host_completion_publish_ns\":" << buffer.host_completion_publish_nanoseconds
                              << ",\"valid\":" << boolean(buffer.valid) << '}';
                    buffer_separator = ",";
                }
                std::cout << "],\"dispatches\":[";
                auto dispatch_separator = "";
                for (const auto &dispatch : sample.dispatches) {
                    std::cout << dispatch_separator << "{\"ordinal\":" << dispatch.ordinal
                              << ",\"command_buffer_ordinal\":" << dispatch.command_buffer_ordinal
                              << ",\"shader_checksum\":" << std::quoted(std::to_string(dispatch.shader_checksum))
                              << ",\"dispatch_size\":[" << dispatch.dispatch_size.x << ',' << dispatch.dispatch_size.y << ',' << dispatch.dispatch_size.z
                              << "],\"block_size\":[" << dispatch.block_size.x << ',' << dispatch.block_size.y << ',' << dispatch.block_size.z
                              << "],\"begin_ticks\":" << dispatch.begin_ticks << ",\"end_ticks\":" << dispatch.end_ticks
                              << ",\"elapsed_ns\":" << dispatch.elapsed_nanoseconds << ",\"valid\":" << boolean(dispatch.valid) << '}';
                    dispatch_separator = ",";
                }
                std::cout << "]}";
                separator = ",";
            }
            std::cout << ']';
        };
        records("throughput", _throughput);
        records("latency", _latency);
        std::cout << ",\"control\":{\"method\":\"metal4_commit_feedback_v1\","
                     "\"scope\":\"command_buffer_gpu_intervals\",\"encoder_instrumentation\":false,\"repetitions\":"
                  << _repetitions;
        records("throughput", _control_throughput);
        records("latency", _control_latency);
        std::cout << "}}";
        std::cout.precision(previous_precision);
    }
};

}// namespace luisa::test

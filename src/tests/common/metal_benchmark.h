#pragma once

#include "metal_benchmark_timing.h"

#include <algorithm>
#include <cstdlib>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>

#ifdef __APPLE__
#include <dlfcn.h>
#endif

namespace luisa::test {

class MetalBenchmarkTiming final {
private:
    void *_library{nullptr};
    decltype(&luisa_metal_timing_begin) _begin{nullptr};
    decltype(&luisa_metal_timing_begin_control) _begin_control{nullptr};
    decltype(&luisa_metal_timing_end) _end{nullptr};
    decltype(&luisa_metal_timing_error) _error{nullptr};
    uint64_t _repetitions{0u};
    std::vector<LuisaMetalTimingResult> _throughput;
    std::vector<LuisaMetalTimingResult> _latency;
    std::vector<LuisaMetalTimingResult> _control_throughput;
    std::vector<LuisaMetalTimingResult> _control_latency;

public:
    explicit MetalBenchmarkTiming(bool metal) {
        if (auto path = std::getenv("LUISA_TILE_BENCH_METAL_TIMING")) {
            if (!metal) { throw std::runtime_error{"Metal device timing requested for a non-Metal benchmark"}; }
#ifdef __APPLE__
            _library = dlopen(path, RTLD_NOW | RTLD_LOCAL);
            if (_library == nullptr) { throw std::runtime_error{dlerror()}; }
            auto version = reinterpret_cast<decltype(&luisa_metal_timing_version)>(dlsym(_library, "luisa_metal_timing_version"));
            _begin = reinterpret_cast<decltype(_begin)>(dlsym(_library, "luisa_metal_timing_begin"));
            _begin_control = reinterpret_cast<decltype(_begin_control)>(dlsym(_library, "luisa_metal_timing_begin_control"));
            _end = reinterpret_cast<decltype(_end)>(dlsym(_library, "luisa_metal_timing_end"));
            _error = reinterpret_cast<decltype(_error)>(dlsym(_library, "luisa_metal_timing_error"));
            if (version == nullptr || version() != 2 || _begin == nullptr || _begin_control == nullptr || _end == nullptr || _error == nullptr) {
                dlclose(_library);
                _library = nullptr;
                throw std::runtime_error{"incompatible Metal benchmark timing library"};
            }
#else
            throw std::runtime_error{"Metal GPU counters require macOS"};
#endif
        }
    }
    ~MetalBenchmarkTiming() {
#ifdef __APPLE__
        if (_library != nullptr) { dlclose(_library); }
#endif
    }
    MetalBenchmarkTiming(const MetalBenchmarkTiming &) = delete;
    MetalBenchmarkTiming &operator=(const MetalBenchmarkTiming &) = delete;

    // submit must encode repetitions and wait for completion. Device sampling
    // is a SEPARATE phase after uninstrumented host-wall throughput/latency.
    template<typename Sync, typename Submit>
    void measure(Sync &&synchronize, Submit &&submit, uint64_t repetitions, uint32_t samples) {
        if (_library == nullptr) { return; }
        _repetitions = std::min<uint64_t>(repetitions, 64u);
        auto sample = [&](uint64_t count, bool counters) {
            synchronize();
            if (!(counters ? _begin(1024u) : _begin_control())) { throw std::runtime_error{_error()}; }
            LuisaMetalTimingResult result{};
            try {
                submit(count);
            } catch (...) {
                // Always remove process-local instrumentation on unwinding.
                static_cast<void>(_end(&result));
                throw;
            }
            if (!_end(&result)) { throw std::runtime_error{_error()}; }
            return result;
        };
        auto phase = [&](uint64_t count, auto &control, auto &instrumented) {
            for (auto i = 0u; i < samples; i++) {
                // Pair identical batches and alternate the probe/control order.
                if (i % 2u == 0u) { control.emplace_back(sample(count, false)); }
                instrumented.emplace_back(sample(count, true));
                if (i % 2u != 0u) { control.emplace_back(sample(count, false)); }
            }
        };
        phase(_repetitions, _control_throughput, _throughput);
        phase(1u, _control_latency, _latency);
    }

    // Emit a leading comma so ordinary host-only JSON remains byte-compatible.
    void print() const {
        if (_library == nullptr) { return; }
        std::cout << ",\"device_timing\":{\"method\":\"metal_compute_pass_timestamps_v1\","
                     "\"scope\":\"sum_of_compute_encoder_gpu_intervals\","
                     "\"host_samples_instrumented\":false,\"repetitions\":"
                  << _repetitions;
        auto print = [](const char *name, const auto &samples, bool counters) {
            std::cout << ",\"" << name << "\":[";
            auto separator = "";
            for (auto sample : samples) {
                std::cout << separator << '{';
                if (counters) {
                    std::cout << "\"compute_ns\":" << sample.compute_ns
                              << ",\"compute_span_ns\":" << sample.compute_span_ns
                              << ",\"calibration_cpu_ns\":" << sample.calibration_cpu_ns
                              << ",\"calibration_gpu_ticks\":" << sample.calibration_gpu_ticks
                              << ",\"compute_passes\":" << sample.compute_passes << ',';
                }
                std::cout << "\"command_buffer_ns\":" << sample.command_buffer_ns
                          << ",\"command_buffers\":" << sample.command_buffers << '}';
                separator = ",";
            }
            std::cout << ']';
        };
        print("throughput", _throughput, true);
        print("latency", _latency, true);
        std::cout << ",\"control\":{\"method\":\"metal_command_buffer_timestamps_v1\","
                     "\"scope\":\"sum_of_command_buffer_gpu_intervals\",\"encoder_instrumentation\":false,\"repetitions\":"
                  << _repetitions;
        print("throughput", _control_throughput, false);
        print("latency", _control_latency, false);
        std::cout << '}';
        std::cout << '}';
    }
};

}// namespace luisa::test

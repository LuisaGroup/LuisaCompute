// Runtime benchmark for lane-affine buffer lowering in a portable GEMM.
//
// The benchmark deliberately uses the ordinary DSL kernel rather than a
// backend-private entry point. It validates one result against an independent
// CPU implementation, warms the compiled shader, and then reports steady-
// state throughput for repeated dispatches. Set
// LUISA_SIMD_DISABLE_LANE_AFFINE_BUFFER=1 for a same-binary SIMD A/B.

#include "benchmark_simd_gemm_kernel.h"

#include <luisa/backends/ext/simd_config_ext.h>
#include <luisa/runtime/buffer.h>
#include <luisa/runtime/context.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/stream.h>

#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <optional>
#include <string_view>

using namespace luisa;
using namespace luisa::compute;

namespace {

constexpr auto matrix_size = simd::benchmark::gemm_matrix_size;
constexpr auto warmup_dispatch_count = 4u;
constexpr auto timed_dispatch_count = 128u;
constexpr auto sample_count = size_t{7u};

struct Options {
    std::string_view backend;
    uint32_t width{0u};
    uint32_t worker_count{0u};
};

[[nodiscard]] bool parse_uint32(
    std::string_view text, uint32_t &value) noexcept {
    auto result = std::from_chars(
        text.data(), text.data() + text.size(), value);
    return result.ec == std::errc{} &&
           result.ptr == text.data() + text.size();
}

[[nodiscard]] std::optional<Options> parse_options(
    int argc, char *argv[]) noexcept {
    if (argc < 2 || argv[1] == nullptr) {
        std::cerr << "Usage: " << (argc > 0 ? argv[0] : "benchmark")
                  << " <fallback|simd> [simd-width] [simd-worker-count]\n";
        return std::nullopt;
    }
    Options options{.backend = argv[1]};
    if (options.backend == "fallback") {
        if (argc != 2) {
            std::cerr << "Fallback benchmark takes no width or worker count\n";
            return std::nullopt;
        }
        return options;
    }
    if (options.backend != "simd" || argc < 3 || argv[2] == nullptr ||
        !parse_uint32(argv[2], options.width) ||
        (options.width != 1u && options.width != 2u &&
         options.width != 4u && options.width != 8u &&
         options.width != 16u)) {
        std::cerr << "SIMD benchmark requires width 1, 2, 4, 8, or 16\n";
        return std::nullopt;
    }
    if (argc >= 4 &&
        (argv[3] == nullptr ||
         !parse_uint32(argv[3], options.worker_count) ||
         options.worker_count == 0u)) {
        std::cerr << "Invalid SIMD worker count\n";
        return std::nullopt;
    }
    if (argc > 4) {
        std::cerr << "Too many benchmark arguments\n";
        return std::nullopt;
    }
    return options;
}

[[nodiscard]] bool validate(
    luisa::span<const float> lhs,
    luisa::span<const float> rhs,
    luisa::span<const float> actual) noexcept {
    for (auto row = uint32_t{0u}; row < matrix_size; row++) {
        for (auto column = uint32_t{0u}; column < matrix_size; column++) {
            auto expected = 0.0;
            for (auto inner = uint32_t{0u}; inner < matrix_size; inner++) {
                expected += static_cast<double>(
                                lhs[row * matrix_size + inner]) *
                            static_cast<double>(
                                rhs[inner * matrix_size + column]);
            }
            auto observed = actual[row * matrix_size + column];
            auto tolerance = 2.0e-4 +
                             2.0e-4 * std::abs(expected);
            if (!std::isfinite(observed) ||
                std::abs(static_cast<double>(observed) - expected) >
                    tolerance) {
                LUISA_WARNING(
                    "GEMM benchmark mismatch at ({}, {}): expected {}, got {} "
                    "(tolerance {}).",
                    row, column, expected, observed, tolerance);
                return false;
            }
        }
    }
    return true;
}

[[nodiscard]] double median(
    std::array<double, sample_count> values) noexcept {
    std::ranges::sort(values);
    return values[sample_count / 2u];
}

}// namespace

int main(int argc, char *argv[]) {
    auto options = parse_options(argc, argv);
    if (!options) { return 1; }
    Context context{argc > 0 ? argv[0] : ""};
    DeviceConfig config{};
    if (options->backend == "simd") {
        config.extension = luisa::make_unique<SIMDDeviceConfigExt>(
            options->width, options->worker_count);
    }
    auto device = context.create_device(
        options->backend,
        options->backend == "simd" ? &config : nullptr);
    auto element_count = static_cast<size_t>(matrix_size) * matrix_size;
    luisa::vector<float> lhs(element_count);
    luisa::vector<float> rhs(element_count);
    luisa::vector<float> output(
        element_count, std::numeric_limits<float>::quiet_NaN());
    for (auto i = size_t{0u}; i < element_count; i++) {
        auto lhs_value = static_cast<int32_t>((i * 17u + 3u) % 29u) - 14;
        auto rhs_value = static_cast<int32_t>((i * 11u + 5u) % 31u) - 15;
        lhs[i] = static_cast<float>(lhs_value) * 0.03125f;
        rhs[i] = static_cast<float>(rhs_value) * 0.025f;
    }

    auto lhs_buffer = device.create_buffer<float>(element_count);
    auto rhs_buffer = device.create_buffer<float>(element_count);
    auto output_buffer = device.create_buffer<float>(element_count);
    auto gemm = simd::benchmark::make_gemm_kernel();
    auto shader = device.compile(gemm);
    auto stream = device.create_stream();
    stream << lhs_buffer.copy_from(luisa::span{lhs})
           << rhs_buffer.copy_from(luisa::span{rhs})
           << shader(lhs_buffer, rhs_buffer, output_buffer)
                  .dispatch(matrix_size, matrix_size)
           << output_buffer.copy_to(luisa::span{output})
           << synchronize();
    if (!validate(lhs, rhs, output)) { return 2; }

    for (auto i = uint32_t{0u}; i < warmup_dispatch_count; i++) {
        stream << shader(lhs_buffer, rhs_buffer, output_buffer)
                      .dispatch(matrix_size, matrix_size);
    }
    stream << synchronize();
    std::array<double, sample_count> samples{};
    for (auto sample = size_t{0u}; sample < sample_count; sample++) {
        auto start = std::chrono::steady_clock::now();
        for (auto i = uint32_t{0u}; i < timed_dispatch_count; i++) {
            stream << shader(lhs_buffer, rhs_buffer, output_buffer)
                          .dispatch(matrix_size, matrix_size);
        }
        stream << synchronize();
        auto end = std::chrono::steady_clock::now();
        samples[sample] =
            std::chrono::duration<double>(end - start).count();
    }
    auto seconds = median(samples);
    auto operations = 2.0 * matrix_size * matrix_size * matrix_size *
                      timed_dispatch_count;
    std::cout << "luisa_gemm,backend=" << options->backend
              << ",width=" << options->width
              << ",workers=" << options->worker_count
              << ",size=" << matrix_size
              << ",dispatches=" << timed_dispatch_count
              << ",median_seconds=" << seconds
              << ",median_gflops=" << operations / seconds * 1.0e-9
              << ",samples_seconds=";
    for (auto sample = size_t{0u}; sample < sample_count; sample++) {
        if (sample != 0u) { std::cout << ';'; }
        std::cout << samples[sample];
    }
    std::cout << '\n';
    return 0;
}

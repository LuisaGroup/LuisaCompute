// Standalone ISPC side of the SIMD comparison suite.
// This source is compiled only by run.py and is not part of project CMake.

#include "simd_thread_pool.h"

#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <optional>
#include <string_view>
#include <thread>
#include <vector>

namespace {

constexpr auto sample_count = size_t{7u};
constexpr auto warmup_dispatch_count = uint32_t{2u};
constexpr auto mandelbrot_width = uint32_t{1536u};
constexpr auto mandelbrot_height = uint32_t{1024u};
constexpr auto mandelbrot_dispatch_count = uint32_t{16u};
constexpr auto stream_element_count = uint32_t{8u * 1024u * 1024u};
constexpr auto stream_dispatch_count = uint32_t{32u};
constexpr auto aos_element_count = uint32_t{4u * 1024u * 1024u};
constexpr auto aos_dispatch_count = uint32_t{16u};
constexpr auto gemm_matrix_size = uint32_t{256u};
constexpr auto gemm_dispatch_count = uint32_t{128u};
constexpr auto path_width = uint32_t{640u};
constexpr auto path_height = uint32_t{360u};
constexpr auto path_frame = uint32_t{17u};
constexpr auto path_dispatch_count = uint32_t{8u};

using Mandelbrot = void(
    uint32_t, uint32_t, uint32_t, uint32_t, uint32_t *);
using MaskedStream = void(
    const float *, const float *, const uint32_t *, float *,
    uint32_t, uint32_t);
using AosToSoa = void(
    const float *, float *, float *, float *, float *,
    uint32_t, uint32_t);
using Gemm = void(
    const float *, const float *, float *, uint32_t, uint32_t);
using PathTrace = void(
    uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, float *);

#define LUISA_DECLARE_VARIANT(suffix)               \
    extern "C" {                                    \
    Mandelbrot luisa_ispc_mandelbrot_##suffix;      \
    MaskedStream luisa_ispc_masked_stream_##suffix; \
    AosToSoa luisa_ispc_aos_to_soa_##suffix;        \
    Gemm luisa_ispc_gemm_##suffix;                  \
    PathTrace luisa_ispc_path_trace_##suffix;       \
    }

LUISA_DECLARE_VARIANT(avx2_w4)
LUISA_DECLARE_VARIANT(avx2_w8)
LUISA_DECLARE_VARIANT(avx512_w4)
LUISA_DECLARE_VARIANT(avx512_w8)
LUISA_DECLARE_VARIANT(avx512_w16)

#undef LUISA_DECLARE_VARIANT

struct Variant {
    std::string_view name;
    uint32_t width;
    Mandelbrot *mandelbrot;
    MaskedStream *masked_stream;
    AosToSoa *aos_to_soa;
    Gemm *gemm;
    PathTrace *path_trace;
};

#define LUISA_VARIANT(name, suffix, lanes)                  \
    Variant { name, lanes, &luisa_ispc_mandelbrot_##suffix, \
              &luisa_ispc_masked_stream_##suffix,           \
              &luisa_ispc_aos_to_soa_##suffix,              \
              &luisa_ispc_gemm_##suffix,                    \
              &luisa_ispc_path_trace_##suffix }

constexpr std::array variants{
    LUISA_VARIANT("avx2-i32x4", avx2_w4, 4u),
    LUISA_VARIANT("avx2-i32x8", avx2_w8, 8u),
    LUISA_VARIANT("avx512skx-x4", avx512_w4, 4u),
    LUISA_VARIANT("avx512skx-x8", avx512_w8, 8u),
    LUISA_VARIANT("avx512skx-x16", avx512_w16, 16u),
};

#undef LUISA_VARIANT

struct Options {
    uint32_t workers{0u};
    std::string_view target;
    std::string_view workload;
    std::string_view dump_path;
};

[[nodiscard]] std::optional<Options> parse_options(
    int argc, char *argv[]) noexcept {
    if (argc < 4 || argc > 5) {
        std::cerr << "Usage: " << (argc > 0 ? argv[0] : "ispc_runner")
                  << " <workers> <target> <workload> [dump-path]\n";
        return std::nullopt;
    }
    Options options{
        .target = argv[2],
        .workload = argv[3],
        .dump_path = argc == 5 ? std::string_view{argv[4]} :
                                 std::string_view{},
    };
    auto text = std::string_view{argv[1]};
    auto parsed = std::from_chars(
        text.data(), text.data() + text.size(), options.workers);
    if (parsed.ec != std::errc{} ||
        parsed.ptr != text.data() + text.size() ||
        options.workers == 0u) {
        std::cerr << "Invalid worker count.\n";
        return std::nullopt;
    }
    if (std::ranges::find_if(
            variants, [&](auto &&variant) noexcept {
                return variant.name == options.target;
            }) == variants.end()) {
        std::cerr << "Unknown target '" << options.target << "'.\n";
        return std::nullopt;
    }
    if (options.workload != "mandelbrot" &&
        options.workload != "masked_stream" &&
        options.workload != "aos_to_soa" &&
        options.workload != "gemm" &&
        options.workload != "path_trace") {
        std::cerr << "Unknown workload '" << options.workload << "'.\n";
        return std::nullopt;
    }
    return options;
}

[[nodiscard]] double median(
    std::array<double, sample_count> values) noexcept {
    std::ranges::sort(values);
    return values[sample_count / 2u];
}

template<typename T>
[[nodiscard]] uint64_t hash_values(const std::vector<T> &values) noexcept {
    constexpr auto offset = uint64_t{1469598103934665603ull};
    constexpr auto prime = uint64_t{1099511628211ull};
    auto hash = offset;
    auto *bytes = reinterpret_cast<const uint8_t *>(values.data());
    for (auto i = size_t{0u}; i < values.size() * sizeof(T); i++) {
        hash ^= bytes[i];
        hash *= prime;
    }
    return hash;
}

template<typename T>
[[nodiscard]] bool dump_values(
    std::string_view path, const std::vector<T> &values) noexcept {
    if (path.empty()) { return true; }
    std::ofstream stream{std::string{path}, std::ios::binary};
    stream.write(
        reinterpret_cast<const char *>(values.data()),
        static_cast<std::streamsize>(values.size() * sizeof(T)));
    return stream.good();
}

struct Measurement {
    std::array<double, sample_count> samples{};
    uint64_t item_count{0u};
    uint32_t dispatch_count{0u};
    uint64_t checksum{0u};
    double rate_numerator{0.0};
    double rate_scale{1.0e-6};
    std::string_view rate_unit{"mitems_per_second"};
};

template<typename Dispatch>
[[nodiscard]] Measurement measure(
    uint32_t dispatch_count, uint64_t item_count,
    Dispatch &&dispatch) noexcept {
    for (auto i = 0u; i < warmup_dispatch_count; i++) { dispatch(); }
    Measurement measurement{
        .item_count = item_count,
        .dispatch_count = dispatch_count,
    };
    for (auto sample = size_t{0u}; sample < sample_count; sample++) {
        auto begin = std::chrono::steady_clock::now();
        for (auto i = 0u; i < dispatch_count; i++) { dispatch(); }
        auto end = std::chrono::steady_clock::now();
        measurement.samples[sample] =
            std::chrono::duration<double>(end - begin).count();
    }
    return measurement;
}

[[nodiscard]] Measurement run_mandelbrot(
    const Variant &variant,
    luisa::compute::simd::SIMDThreadPool &pool,
    std::string_view dump_path) noexcept {
    auto count = static_cast<size_t>(mandelbrot_width) * mandelbrot_height;
    std::vector<uint32_t> output(count);
    auto dispatch = [&]() noexcept {
        pool.parallel_for(
            mandelbrot_height, 1u,
            [&](uint64_t begin, uint64_t end) noexcept {
                variant.mandelbrot(
                    mandelbrot_width, mandelbrot_height,
                    static_cast<uint32_t>(begin),
                    static_cast<uint32_t>(end), output.data());
            });
    };
    auto result = measure(
        mandelbrot_dispatch_count, count, dispatch);
    result.checksum = hash_values(output);
    if (!dump_values(dump_path, output)) { return {}; }
    return result;
}

[[nodiscard]] Measurement run_masked_stream(
    const Variant &variant,
    luisa::compute::simd::SIMDThreadPool &pool,
    std::string_view dump_path) noexcept {
    std::vector<float> a(stream_element_count);
    std::vector<float> b(stream_element_count);
    std::vector<uint32_t> mask(stream_element_count);
    std::vector<float> output(stream_element_count);
    for (auto i = uint32_t{0u}; i < stream_element_count; i++) {
        a[i] = static_cast<float>(static_cast<int32_t>(i % 257u) - 128) *
               (1.0f / 128.0f);
        b[i] = static_cast<float>(static_cast<int32_t>((i * 17u) % 263u) - 131) *
               (1.0f / 64.0f);
        auto bits = i * 747796405u + 2891336453u;
        mask[i] = ((bits >> 29u) & 3u) != 0u ? 1u : 0u;
    }
    auto dispatch = [&]() noexcept {
        pool.parallel_for(
            stream_element_count, 16384u,
            [&](uint64_t begin, uint64_t end) noexcept {
                variant.masked_stream(
                    a.data(), b.data(), mask.data(), output.data(),
                    static_cast<uint32_t>(begin),
                    static_cast<uint32_t>(end));
            });
    };
    auto result = measure(
        stream_dispatch_count, stream_element_count, dispatch);
    for (auto i = uint32_t{0u}; i < stream_element_count; i++) {
        auto expected = mask[i] != 0u ? b[i] - a[i] : 0.0f;
        if (output[i] != expected) {
            std::cerr << "masked_stream mismatch at " << i << ".\n";
            return {};
        }
    }
    result.checksum = hash_values(output);
    if (!dump_values(dump_path, output)) { return {}; }
    return result;
}

[[nodiscard]] Measurement run_aos_to_soa(
    const Variant &variant,
    luisa::compute::simd::SIMDThreadPool &pool,
    std::string_view dump_path) noexcept {
    std::vector<float> input(static_cast<size_t>(aos_element_count) * 4u);
    std::vector<float> output(static_cast<size_t>(aos_element_count) * 4u);
    for (auto i = size_t{0u}; i < input.size(); i++) {
        input[i] = static_cast<float>(
                       static_cast<int32_t>((i * 29u + 7u) % 509u) - 254) *
                   (1.0f / 256.0f);
    }
    auto dispatch = [&]() noexcept {
        pool.parallel_for(
            aos_element_count, 16384u,
            [&](uint64_t begin, uint64_t end) noexcept {
                variant.aos_to_soa(
                    input.data(), output.data(),
                    output.data() + aos_element_count,
                    output.data() + 2u * aos_element_count,
                    output.data() + 3u * aos_element_count,
                    static_cast<uint32_t>(begin),
                    static_cast<uint32_t>(end));
            });
    };
    auto result = measure(aos_dispatch_count, aos_element_count, dispatch);
    for (auto i = uint32_t{0u}; i < aos_element_count; i++) {
        for (auto component = uint32_t{0u}; component < 4u; component++) {
            auto observed = output[static_cast<size_t>(component) * aos_element_count + i];
            auto expected = input[static_cast<size_t>(i) * 4u + component];
            if (observed != expected) {
                std::cerr << "aos_to_soa mismatch at " << i << ".\n";
                return {};
            }
        }
    }
    result.checksum = hash_values(output);
    if (!dump_values(dump_path, output)) { return {}; }
    return result;
}

[[nodiscard]] Measurement run_gemm(
    const Variant &variant,
    luisa::compute::simd::SIMDThreadPool &pool,
    std::string_view dump_path) noexcept {
    auto element_count = static_cast<size_t>(gemm_matrix_size) *
                         gemm_matrix_size;
    std::vector<float> lhs(element_count);
    std::vector<float> rhs(element_count);
    std::vector<float> output(
        element_count, std::numeric_limits<float>::quiet_NaN());
    for (auto i = size_t{0u}; i < element_count; i++) {
        auto lhs_value = static_cast<int32_t>((i * 17u + 3u) % 29u) - 14;
        auto rhs_value = static_cast<int32_t>((i * 11u + 5u) % 31u) - 15;
        lhs[i] = static_cast<float>(lhs_value) * 0.03125f;
        rhs[i] = static_cast<float>(rhs_value) * 0.025f;
    }
    auto dispatch = [&]() noexcept {
        pool.parallel_for(
            gemm_matrix_size, 1u,
            [&](uint64_t begin, uint64_t end) noexcept {
                variant.gemm(
                    lhs.data(), rhs.data(), output.data(),
                    static_cast<uint32_t>(begin),
                    static_cast<uint32_t>(end));
            });
    };
    auto result = measure(
        gemm_dispatch_count, element_count, dispatch);
    for (auto row = uint32_t{0u}; row < gemm_matrix_size; row++) {
        for (auto column = uint32_t{0u}; column < gemm_matrix_size;
             column++) {
            auto expected = 0.0;
            for (auto inner = uint32_t{0u}; inner < gemm_matrix_size;
                 inner++) {
                expected += static_cast<double>(
                                lhs[row * gemm_matrix_size + inner]) *
                            static_cast<double>(
                                rhs[inner * gemm_matrix_size + column]);
            }
            auto observed = output[row * gemm_matrix_size + column];
            auto tolerance = 2.0e-4 + 2.0e-4 * std::abs(expected);
            if (!std::isfinite(observed) ||
                std::abs(static_cast<double>(observed) - expected) >
                    tolerance) {
                std::cerr << "gemm mismatch at (" << row << ", "
                          << column << ").\n";
                return {};
            }
        }
    }
    result.checksum = hash_values(output);
    result.rate_numerator =
        2.0 * gemm_matrix_size * gemm_matrix_size * gemm_matrix_size *
        gemm_dispatch_count;
    result.rate_scale = 1.0e-9;
    result.rate_unit = "gflop_per_second";
    if (!dump_values(dump_path, output)) { return {}; }
    return result;
}

[[nodiscard]] Measurement run_path_trace(
    const Variant &variant,
    luisa::compute::simd::SIMDThreadPool &pool,
    std::string_view dump_path) noexcept {
    auto count = static_cast<size_t>(path_width) * path_height;
    std::vector<float> output(count * 4u);
    auto dispatch = [&]() noexcept {
        pool.parallel_for(
            path_height, 1u,
            [&](uint64_t begin, uint64_t end) noexcept {
                variant.path_trace(
                    path_width, path_height, path_frame,
                    static_cast<uint32_t>(begin),
                    static_cast<uint32_t>(end), output.data());
            });
    };
    auto result = measure(path_dispatch_count, count, dispatch);
    for (auto i = size_t{0u}; i < count; i++) {
        if (!std::isfinite(output[4u * i + 0u]) ||
            !std::isfinite(output[4u * i + 1u]) ||
            !std::isfinite(output[4u * i + 2u]) ||
            output[4u * i + 3u] != 1.0f) {
            std::cerr << "path_trace produced invalid output at " << i
                      << ".\n";
            return {};
        }
    }
    result.checksum = hash_values(output);
    if (!dump_values(dump_path, output)) { return {}; }
    return result;
}

}// namespace

int main(int argc, char *argv[]) {
    auto options = parse_options(argc, argv);
    if (!options) { return 1; }
    auto variant_iter = std::ranges::find_if(
        variants, [&](auto &&variant) noexcept {
            return variant.name == options->target;
        });
    auto &variant = *variant_iter;
    luisa::compute::simd::SIMDThreadPool pool{options->workers};
    Measurement result{};
    if (options->workload == "mandelbrot") {
        result = run_mandelbrot(variant, pool, options->dump_path);
    } else if (options->workload == "masked_stream") {
        result = run_masked_stream(variant, pool, options->dump_path);
    } else if (options->workload == "aos_to_soa") {
        result = run_aos_to_soa(variant, pool, options->dump_path);
    } else if (options->workload == "gemm") {
        result = run_gemm(variant, pool, options->dump_path);
    } else {
        result = run_path_trace(variant, pool, options->dump_path);
    }
    if (result.item_count == 0u) { return 2; }
    auto seconds = median(result.samples);
    auto rate_numerator = result.rate_numerator == 0.0 ?
                              static_cast<double>(result.item_count) *
                                  result.dispatch_count :
                              result.rate_numerator;
    auto rate = rate_numerator / seconds * result.rate_scale;
    std::cout << "simd_ispc_suite"
              << ",implementation=ispc"
              << ",backend=ispc"
              << ",target=" << variant.name
              << ",width=" << variant.width
              << ",workers=" << options->workers
              << ",workload=" << options->workload
              << ",items=" << result.item_count
              << ",dispatches=" << result.dispatch_count
              << ",median_seconds=" << std::setprecision(12) << seconds
              << ",rate_unit=" << result.rate_unit
              << ",median_rate=" << rate
              << ",checksum=" << std::hex << result.checksum << std::dec
              << ",samples_seconds=";
    for (auto i = size_t{0u}; i < result.samples.size(); i++) {
        if (i != 0u) { std::cout << ';'; }
        std::cout << result.samples[i];
    }
    std::cout << '\n';
    return 0;
}

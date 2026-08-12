// Same-algorithm ISPC control for the Luisa fallback/SIMD GEMM benchmark.

#include "simd_thread_pool.h"

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
#include <thread>
#include <vector>

namespace {

constexpr auto matrix_size = uint32_t{256u};
constexpr auto warmup_dispatch_count = uint32_t{4u};
constexpr auto timed_dispatch_count = uint32_t{128u};
constexpr auto sample_count = size_t{7u};

using Gemm = void(const float *, const float *, float *, uint32_t, uint32_t);

extern "C" {
Gemm luisa_ispc_gemm_avx2_w4;
Gemm luisa_ispc_gemm_avx2_w8;
Gemm luisa_ispc_gemm_avx512_w4;
Gemm luisa_ispc_gemm_avx512_w8;
Gemm luisa_ispc_gemm_avx512_w16;
}

struct Variant {
    std::string_view name;
    Gemm *function;
};

constexpr std::array variants{
    Variant{"avx2-i32x4", &luisa_ispc_gemm_avx2_w4},
    Variant{"avx2-i32x8", &luisa_ispc_gemm_avx2_w8},
    Variant{"avx512skx-x4", &luisa_ispc_gemm_avx512_w4},
    Variant{"avx512skx-x8", &luisa_ispc_gemm_avx512_w8},
    Variant{"avx512skx-x16", &luisa_ispc_gemm_avx512_w16},
};

struct Options {
    uint32_t worker_count;
    std::string_view target;
};

[[nodiscard]] std::optional<Options> parse_options(
    int argc, char *argv[]) noexcept {
    if (argc > 3) {
        std::cerr << "Usage: " << (argc > 0 ? argv[0] : "benchmark")
                  << " [worker-count] [target]\n";
        return std::nullopt;
    }
    auto options = Options{
        .worker_count = std::max(
            std::thread::hardware_concurrency(), 1u),
    };
    if (argc < 2 || argv[1] == nullptr) { return options; }
    auto text = std::string_view{argv[1]};
    auto result = std::from_chars(
        text.data(), text.data() + text.size(), options.worker_count);
    if (result.ec != std::errc{} ||
        result.ptr != text.data() + text.size() ||
        options.worker_count == 0u) {
        std::cerr << "Invalid worker count '" << text << "'\n";
        return std::nullopt;
    }
    if (argc == 3) {
        options.target = argv[2];
        auto found = std::ranges::any_of(
            variants, [&](const Variant &variant) noexcept {
                return variant.name == options.target;
            });
        if (!found) {
            std::cerr << "Unknown ISPC target '" << options.target << "'\n";
            return std::nullopt;
        }
    }
    return options;
}

[[nodiscard]] bool validate(
    const std::vector<float> &lhs, const std::vector<float> &rhs,
    const std::vector<float> &actual) noexcept {
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
            auto tolerance = 2.0e-4 + 2.0e-4 * std::abs(expected);
            if (!std::isfinite(observed) ||
                std::abs(static_cast<double>(observed) - expected) >
                    tolerance) {
                std::cerr << "ISPC GEMM mismatch at (" << row << ", "
                          << column << "): expected " << expected
                          << ", got " << observed << " (tolerance "
                          << tolerance << ")\n";
                return false;
            }
        }
    }
    return true;
}

[[nodiscard]] double median(std::array<double, sample_count> values) noexcept {
    std::ranges::sort(values);
    return values[sample_count / 2u];
}

}// namespace

int main(int argc, char *argv[]) {
    auto options = parse_options(argc, argv);
    if (!options) { return 1; }
    auto element_count = static_cast<size_t>(matrix_size) * matrix_size;
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

    luisa::compute::simd::SIMDThreadPool pool{options->worker_count};
    auto dispatch = [&](Gemm *function) noexcept {
        pool.parallel_for(
            matrix_size, 1u,
            [&](uint64_t begin, uint64_t end) noexcept {
                function(
                    lhs.data(), rhs.data(), output.data(),
                    static_cast<uint32_t>(begin),
                    static_cast<uint32_t>(end));
            });
    };

    for (auto &&variant : variants) {
        if (!options->target.empty() &&
            variant.name != options->target) {
            continue;
        }
        dispatch(variant.function);
        if (!validate(lhs, rhs, output)) { return 2; }
    }
    for (auto i = uint32_t{0u}; i < warmup_dispatch_count; i++) {
        for (auto &&variant : variants) {
            if (options->target.empty() ||
                variant.name == options->target) {
                dispatch(variant.function);
            }
        }
    }

    std::array<std::array<double, sample_count>, variants.size()> samples{};
    // Rotate the first variant in every sample to reduce systematic thermal
    // and shared-host order bias.
    for (auto sample = size_t{0u}; sample < sample_count; sample++) {
        for (auto offset = size_t{0u}; offset < variants.size(); offset++) {
            auto index = (sample + offset) % variants.size();
            if (!options->target.empty() &&
                variants[index].name != options->target) {
                continue;
            }
            auto start = std::chrono::steady_clock::now();
            for (auto i = uint32_t{0u}; i < timed_dispatch_count; i++) {
                dispatch(variants[index].function);
            }
            auto end = std::chrono::steady_clock::now();
            samples[index][sample] =
                std::chrono::duration<double>(end - start).count();
        }
    }

    auto operations = 2.0 * matrix_size * matrix_size * matrix_size *
                      timed_dispatch_count;
    for (auto i = size_t{0u}; i < variants.size(); i++) {
        if (!options->target.empty() &&
            variants[i].name != options->target) {
            continue;
        }
        auto seconds = median(samples[i]);
        std::cout << "ispc_gemm,target=" << variants[i].name
                  << ",math=precise-no-fma"
                  << ",workers=" << options->worker_count
                  << ",size=" << matrix_size
                  << ",dispatches=" << timed_dispatch_count
                  << ",median_seconds=" << seconds
                  << ",median_gflops=" << operations / seconds * 1.0e-9
                  << ",samples_seconds=";
        for (auto sample = size_t{0u}; sample < sample_count; sample++) {
            if (sample != 0u) { std::cout << ';'; }
            std::cout << samples[i][sample];
        }
        std::cout << '\n';
    }
    return 0;
}

// Paired benchmark for the original Metal backend and the Metal4 AIR backend.
//
// Each process compiles exactly one cache-disabled shader. Run this executable
// repeatedly in fresh processes to measure process-cold user-shader JIT time.
// The steady-state section validates the result, warms the pipeline, and then
// batches dispatches between synchronizations to reduce host timing noise.

#include <luisa/luisa-compute.h>

#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string_view>

using namespace luisa;
using namespace luisa::compute;

namespace {

constexpr auto element_count = 1u << 20u;
constexpr auto arithmetic_round_count = 64u;
// A few sub-millisecond dispatches do not reliably raise the Apple GPU out of
// its idle frequency state. Use a bounded ~50-100 ms warmup on the reference
// M1 Max so the timed samples compare steady-state shader throughput.
constexpr auto warmup_dispatch_count = 256u;
constexpr auto sample_count = 9u;

[[nodiscard]] bool parse_positive_uint(
    std::string_view text, uint32_t &value) noexcept {
    auto result = std::from_chars(
        text.data(), text.data() + text.size(), value);
    return result.ec == std::errc{} &&
           result.ptr == text.data() + text.size() &&
           value != 0u;
}

[[nodiscard]] float4 host_round(float4 value) noexcept {
    constexpr float4 factors{0.99991f, 0.99989f, 0.99987f, 0.99983f};
    constexpr float4 bias{0.00017f, -0.00013f, 0.00011f, -0.00007f};
    auto rotated = make_float4(value.y, value.z, value.w, value.x);
    return value * factors + rotated * 0.00019f + bias;
}

[[nodiscard]] double median(
    std::array<double, sample_count> values) noexcept {
    std::ranges::sort(values);
    return values[values.size() / 2u];
}

}// namespace

int main(int argc, char *argv[]) {
    if (argc < 2 || argv[1] == nullptr) {
        std::cerr << "Usage: " << (argc > 0 ? argv[0] : "benchmark_metal4")
                  << " <metal|metal4> [dispatches-per-sample] "
                     "[compile-variant]\n";
        return 1;
    }
    auto backend = std::string_view{argv[1]};
    if (backend != "metal" && backend != "metal4") {
        std::cerr << "Backend must be 'metal' or 'metal4'.\n";
        return 1;
    }
    auto dispatches_per_sample = uint32_t{64u};
    if (argc >= 3 &&
        (argv[2] == nullptr ||
         !parse_positive_uint(argv[2], dispatches_per_sample))) {
        std::cerr << "Invalid dispatch count.\n";
        return 1;
    }
    auto compile_variant = uint32_t{1u};
    if (argc >= 4 &&
        (argv[3] == nullptr ||
         !parse_positive_uint(argv[3], compile_variant))) {
        std::cerr << "Invalid compile variant.\n";
        return 1;
    }
    if (argc > 4) {
        std::cerr << "Too many arguments.\n";
        return 1;
    }

    setenv("LUISA_METAL_SHADER_INFO", "1", 1);

    Context context{argc > 0 ? argv[0] : ""};
    auto device = context.create_device(backend);

    // Keep the generated source/MTLB unique across fresh-process cold-JIT
    // samples. The tiny normal float is far below one ULP for this workload,
    // so it changes the compiler cache key without changing the checked result.
    auto variant_bias = static_cast<float>(compile_variant) * 1.0e-30f;
    Kernel1D workload = [variant_bias](
                            BufferFloat4 input,
                            BufferFloat4 output) noexcept {
        set_block_size(256u);
        auto index = dispatch_x();
        auto value = input.read(index);
        constexpr float4 factors{
            0.99991f, 0.99989f, 0.99987f, 0.99983f};
        constexpr float4 bias{
            0.00017f, -0.00013f, 0.00011f, -0.00007f};
        for (auto round = 0u; round < arithmetic_round_count; round++) {
            auto rotated = make_float4(
                value.y, value.z, value.w, value.x);
            value = value * factors + rotated * 0.00019f + bias;
        }
        value += variant_bias;
        output.write(index, value);
    };

    ShaderOption shader_option{
        .enable_cache = false,
        .enable_fast_math = true};
    auto jit_begin = std::chrono::steady_clock::now();
    auto shader = device.compile(workload, shader_option);
    auto jit_end = std::chrono::steady_clock::now();
    auto jit_ms = std::chrono::duration<double, std::milli>(
                      jit_end - jit_begin)
                      .count();
    shader.set_name("metal-backend-benchmark");

    luisa::vector<float4> input(element_count);
    luisa::vector<float4> output(element_count);
    for (auto i = 0u; i < element_count; i++) {
        auto x = static_cast<float>(i & 1023u) * (1.0f / 1024.0f);
        input[i] = make_float4(x, x * 0.5f + 0.1f,
                              x * 0.25f - 0.2f, 1.0f - x * 0.125f);
    }
    auto input_buffer = device.create_buffer<float4>(element_count);
    auto output_buffer = device.create_buffer<float4>(element_count);
    auto stream = device.create_stream();
    stream << input_buffer.copy_from(luisa::span{input})
           << shader(input_buffer, output_buffer).dispatch(element_count)
           << output_buffer.copy_to(luisa::span{output})
           << synchronize();

    auto checksum = 0.0;
    for (auto i = 0u; i < 16u; i++) {
        auto index = i * (element_count / 16u);
        auto expected = input[index];
        for (auto round = 0u; round < arithmetic_round_count; round++) {
            expected = host_round(expected);
        }
        expected += variant_bias;
        auto actual = output[index];
        auto delta = abs(actual - expected);
        auto error = std::max(
            std::max(delta.x, delta.y),
            std::max(delta.z, delta.w));
        if (!std::isfinite(error) || error > 2.0e-4f) {
            std::cerr << "Validation failed at " << index
                      << ": maximum error " << error << "\n";
            return 2;
        }
        checksum += static_cast<double>(actual.x) + actual.y +
                    actual.z + actual.w;
    }

    for (auto i = 0u; i < warmup_dispatch_count; i++) {
        stream << shader(input_buffer, output_buffer).dispatch(element_count);
    }
    stream << synchronize();

    std::array<double, sample_count> runtime_samples_ms{};
    for (auto sample = 0u; sample < sample_count; sample++) {
        auto begin = std::chrono::steady_clock::now();
        for (auto i = 0u; i < dispatches_per_sample; i++) {
            stream << shader(input_buffer, output_buffer).dispatch(element_count);
        }
        stream << synchronize();
        auto end = std::chrono::steady_clock::now();
        runtime_samples_ms[sample] =
            std::chrono::duration<double, std::milli>(end - begin).count() /
            static_cast<double>(dispatches_per_sample);
    }

    auto runtime_ms = median(runtime_samples_ms);
    std::ranges::sort(runtime_samples_ms);
    std::cout << std::fixed << std::setprecision(6)
              << "METAL_BACKEND_BENCHMARK {\"backend\":\"" << backend
              << "\",\"jit_ms\":" << jit_ms
              << ",\"runtime_ms_median\":" << runtime_ms
              << ",\"runtime_ms_p25\":" << runtime_samples_ms[2u]
              << ",\"runtime_ms_p75\":" << runtime_samples_ms[6u]
              << ",\"dispatches_per_sample\":" << dispatches_per_sample
              << ",\"compile_variant\":" << compile_variant
              << ",\"element_count\":" << element_count
              << ",\"checksum\":" << checksum << "}\n";
    return 0;
}

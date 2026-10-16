// Tile runtime GEMM benchmark with an explicit native or TIRx lowering.
// Host wall time excludes compilation/allocation/upload. Native selects each
// backend's existing route; it does not imply CUDA XIR or a silent fallback.
// CUDA accepts explicit TIRx or the opt-in CUDA Tile IR native route.
#include "ut/ut.hpp"// boost.ut cfg used by test_device.h
#include "test_device.h"
#include "tile_xir_test_utils.h"
#include <luisa/core/logging.h>
#include <luisa/tile/runtime.h>
#include <luisa/runtime/context.h>
#include <luisa/runtime/stream.h>
#include <luisa/core/stl/algorithm.h>
#include <luisa/core/stl/string.h>
#include <algorithm>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>

using namespace luisa;
using namespace luisa::compute;
using Clock = std::chrono::steady_clock;

namespace {

[[nodiscard]] int64_t positive(const char *text) {
    auto input = luisa::string_view{text};
    int64_t n{};
    auto result = std::from_chars(input.data(), input.data() + input.size(), n);
    LUISA_ASSERT(result.ec == std::errc{} && result.ptr == input.data() + input.size() && n > 0, "expected positive integer: {}", text);
    return n;
}

[[nodiscard]] int64_t non_negative(const char *text) {
    auto input = luisa::string_view{text};
    int64_t n{};
    auto result = std::from_chars(input.data(), input.data() + input.size(), n);
    LUISA_ASSERT(result.ec == std::errc{} && result.ptr == input.data() + input.size() && n >= 0, "expected non-negative integer: {}", text);
    return n;
}

[[nodiscard]] double elapsed(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>{Clock::now() - start}.count();
}

[[nodiscard]] vector<float> values(size_t n, int64_t seed) {
    vector<float> result(n);
    for (auto i = size_t{0}; i < n; i++) { result[i] = static_cast<float>((static_cast<int64_t>(i) * seed + 17) % 127 - 63) / 64.0f; }
    return result;
}

void samples(const char *name, span<const double> v) {
    std::cout << std::quoted(name) << ":[";
    auto separator = "";
    for (auto x : v) {
        std::cout << separator << x;
        separator = ",";
    }
    std::cout << ']';
}

}// namespace

// Usage: benchmark_tile_xir_gpu <backend> M N K tile-M tile-N tile-K [samples=10] [sample-ms=50] [warmup-ms=200] [variant=0] [native|tirx=native]
int main(int argc, char *argv[]) {
    if (argc < 8 || argc > 13) {
        std::cerr << "Usage: benchmark_tile_xir_gpu <backend> M N K tile-M tile-N tile-K [samples=10] [sample-ms=50] [warmup-ms=200] [variant=0] [native|tirx=native]\n";
        return 1;
    }
    test::tile_xir::Gemm cfg{positive(argv[2]), positive(argv[3]), positive(argv[4]), positive(argv[5]), positive(argv[6]), positive(argv[7])};
    auto count = argc > 8 ? static_cast<int>(positive(argv[8])) : 10;
    auto target_ms = argc > 9 ? positive(argv[9]) : 50;
    auto warmup_ms = argc > 10 ? positive(argv[10]) : 200;
    auto variant = argc > 11 ? non_negative(argv[11]) : 0;// 0=f32, 1=f16 accumulator, 2=bf16 accumulator
    auto lowering = argc > 12 ? luisa::string_view{argv[12]} : luisa::string_view{"native"};
    LUISA_ASSERT(lowering == "native" || lowering == "tirx", "lowering must be native or tirx");
    tile::CompileOptions compile_options;
    compile_options.lowering = lowering == "tirx" ? tile::Lowering::TIRX : tile::Lowering::NATIVE;
    LUISA_ASSERT(cfg.m <= 16384 && cfg.n <= 16384 && cfg.k <= 16384 && cfg.bm <= 256 && cfg.bn <= 256 && cfg.bk <= 1024 &&
                     count <= 101 && target_ms <= 10000 && warmup_ms <= 60000 && variant >= 0 && variant <= 3,
                 "invalid shape/schedule/timing limits");
    log_level_error();
    auto [context, device] = test::create_device(argc, argv);
    auto stream = device.create_stream(StreamTag::COMPUTE);
    auto start = Clock::now();
    using namespace tile;
    auto kernel = [&]() -> luisa::compute::tile::Kernel {
        if (variant == 0) { return test::tile_xir::gemm(cfg); }
        // Same GEMM shape with a narrow accumulator: buffers stay FP32 and the
        // cast happens in-kernel (wide FP32 accumulation, one rounding back).
        const auto bm = cfg.bm, bn = cfg.bn, bk = cfg.bk;
        auto definition = tile_kernel("xir_gemm_narrow", [=](TensorView<const float, 2> A, TensorView<const float, 2> B, TensorView<float, 2> C) {
            auto gm = axis("gm", ceil_div(cfg.m, bm)), gn = axis("gn", ceil_div(cfg.n, bn));
            auto m = axis("m", bm), n = axis("n", bn), k = axis("k", bk);
            for (auto &nest : parallel(shape(gm, gn))) {
                auto m0 = nest.index(gm) * bm, n0 = nest.index(gn) * bn;
                if (variant == 1 || variant == 3) {
                    MmaPolicy policy{};
                    policy.allow_reassociation = variant != 3;
                    auto acc = zeros<half>(shape(m, n));
                    for (auto &step : nest.pipeline(shape(ceil_div(cfg.k, bk)), {.window = cfg.window, .interval = 1u})) {
                        step.stage("load");
                        auto a = A.tile(coord(m0, step.index() * bk), shape(m, k)).load();
                        auto b = B.tile(coord(step.index() * bk, n0), shape(k, n)).load();
                        step.stage("compute");
                        acc = mma(cast<half>(a), cast<half>(b), acc, policy);
                    }
                    C(coord(m0, n0), shape(m, n)).store(cast<float>(acc));
                } else {
                    auto acc = zeros<bfloat16>(shape(m, n));
                    for (auto &step : nest.pipeline(shape(ceil_div(cfg.k, bk)), {.window = cfg.window, .interval = 1u})) {
                        step.stage("load");
                        auto a = A.tile(coord(m0, step.index() * bk), shape(m, k)).load();
                        auto b = B.tile(coord(step.index() * bk, n0), shape(k, n)).load();
                        step.stage("compute");
                        acc = mma(cast<bfloat16>(a), cast<bfloat16>(b), acc);
                    }
                    C(coord(m0, n0), shape(m, n)).store(cast<float>(acc));
                }
            }
        });
        return definition.capture(tensor_shape(cfg.m, cfg.k), tensor_shape(cfg.k, cfg.n), tensor_shape(cfg.m, cfg.n));
    }();
    auto capture_ms = elapsed(start);
    start = Clock::now();
    auto shader = tile::compile(device, kernel, compile_options, {.enable_fast_math = false});
    auto compile_ms = elapsed(start);
    LUISA_ASSERT(shader, "{}", shader.metadata().error.c_str());
    auto host_a = values(cfg.m * cfg.k, 5), host_b = values(cfg.k * cfg.n, 11);
    vector<float> output(cfg.m * cfg.n, std::numeric_limits<float>::quiet_NaN());
    start = Clock::now();
    auto a = device.create_buffer<float>(host_a.size());
    auto b = device.create_buffer<float>(host_b.size());
    auto c = device.create_buffer<float>(output.size());
    stream << a.copy_from(span{host_a}) << b.copy_from(span{host_b}) << c.copy_from(span{output}) << synchronize();
    auto upload_ms = elapsed(start);
    auto batch = [&](uint64_t repetitions) {
        stream.synchronize();
        auto before = Clock::now();
        CommandList commands;
        for (auto i = uint64_t{0}; i < repetitions; i++) { commands << shader(a, b, c).dispatch(); }
        stream << commands.commit() << synchronize();
        return elapsed(before);
    };
    auto cold_ms = batch(1);
    start = Clock::now();
    while (elapsed(start) < warmup_ms) { static_cast<void>(batch(8)); }
    uint64_t repetitions = 1;
    for (auto attempt = 0; attempt < 8; attempt++) {
        auto ms = batch(repetitions);
        if (ms >= target_ms * .8 || repetitions == 100000) { break; }
        auto estimate = repetitions * static_cast<double>(target_ms) / std::max(ms, 1e-6);
        repetitions = std::clamp<uint64_t>(static_cast<uint64_t>(estimate), repetitions + 1, 100000);
    }
    vector<double> throughput, latency;
    for (auto i = 0; i < count; i++) { throughput.emplace_back(1000.0 * batch(repetitions) / repetitions); }
    for (auto i = 0; i < count; i++) {
        stream.synchronize();
        auto before = Clock::now();
        stream << shader(a, b, c).dispatch() << synchronize();
        latency.emplace_back(1000.0 * elapsed(before));
    }
    luisa::sort(throughput.begin(), throughput.end());
    luisa::sort(latency.begin(), latency.end());
    stream << c.copy_to(span{output}) << synchronize();
    // Verify every output against an independent FP64 dot product. Readback
    // and CPU validation remain outside both GPU timing series.
    start = Clock::now();
    auto max_error = 0.0;
    auto non_finite_outputs = size_t{0};
    for (auto i = int64_t{0}; i < cfg.m; i++) {
        for (auto j = int64_t{0}; j < cfg.n; j++) {
            auto actual = output[static_cast<size_t>(i) * cfg.n + j];
            if (!std::isfinite(actual)) {
                non_finite_outputs++;
                continue;
            }
            double expected = cfg.initial;
            for (auto k = int64_t{0}; k < cfg.k; k++) { expected += static_cast<double>(host_a[static_cast<size_t>(i) * cfg.k + k]) * host_b[static_cast<size_t>(k) * cfg.n + j]; }
            max_error = std::max(max_error, std::abs(expected - actual));
        }
    }
    auto validation_ms = elapsed(start);
    auto median = [](span<const double> v) { return v[v.size() / 2]; };
    auto gflops = 2.0 * cfg.m * cfg.n * cfg.k / (median(span{throughput}) * 1e3);// µs → s, flops → GFLOPS
    // Narrow accumulators round once at write-back, so their error is
    // bounded by storage precision rather than the FP32 oracle tolerance.
    auto tolerance = variant == 0 ? 1e-3 + 1e-3 * static_cast<double>(cfg.k) : 2e-2 + 2e-2 * static_cast<double>(cfg.k);
    std::cout << "{\"backend\":\"" << argv[1] << "\",\"m\":" << cfg.m << ",\"n\":" << cfg.n << ",\"k\":" << cfg.k
              << ",\"tile\":[" << cfg.bm << "," << cfg.bn << "," << cfg.bk << "],\"samples\":" << count
              << ",\"repetitions\":" << repetitions << ",\"capture_ms\":" << capture_ms
              << ",\"compile_ms\":" << compile_ms << ",\"upload_ms\":" << upload_ms
              << ",\"cold_ms\":" << cold_ms << ",";
    samples("throughput_us", span{throughput});
    std::cout << ',';
    samples("latency_us", span{latency});
    std::cout << ",\"throughput_p50_us\":" << median(span{throughput})
              << ",\"latency_p50_us\":" << median(span{latency})
              << ",\"gflops\":" << gflops << ",\"max_error\":" << max_error
              << ",\"validated_elements\":" << output.size() << ",\"non_finite_outputs\":" << non_finite_outputs
              << ",\"validation_ms\":" << validation_ms
              << ",\"variant\":" << variant << ",\"fast_math\":false"
              << ",\"lowering\":" << std::quoted(lowering)
              << ",\"realization\":" << std::quoted(shader.metadata().realization)
              << ",\"dispatch\":" << shader.metadata().dispatch_size.x
              << ",\"block\":" << shader.block_size().x << "}\n";
    return non_finite_outputs == 0u && max_error <= tolerance ? 0 : 2;
}

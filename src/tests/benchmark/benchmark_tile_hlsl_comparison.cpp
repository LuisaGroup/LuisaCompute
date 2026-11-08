// Compare native Tile HLSL on DX/VK with CUDA using the same FP32 GEMM.
// CUDA explicitly selects native Tile IR, TIRx, or an ordinary DSL scalar
// baseline. Compilation/allocation/upload are excluded from synchronized
// host-wall timing, and every output is checked against an FP64 oracle.
#include "tile_native_test_utils.h"

#include <luisa/core/logging.h>
#include <luisa/dsl/sugar.h>
#include <luisa/runtime/command_list.h>
#include <luisa/runtime/context.h>
#include <luisa/runtime/stream.h>
#include <luisa/tile/runtime.h>

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>

using namespace luisa;
using namespace luisa::compute;
using Clock = std::chrono::steady_clock;

namespace {

struct Options {
    string backend;
    string route{"native"};
    test::tile_native::Gemm gemm{512, 512, 64, 32, 32};
    int64_t samples{7};
    int64_t sample_ms{50};
    int64_t warmup_ms{200};
};

struct ShaderDescription {
    string implementation;
    string realization;
    size_t source_hash{0u};
    uint3 dispatch_size{};
    uint3 block_size{};
    double capture_ms{0.0};
    double compile_ms{0.0};
};

[[nodiscard]] int64_t positive(const char *text) {
    auto input = string_view{text};
    int64_t value{};
    auto result = std::from_chars(input.data(), input.data() + input.size(), value);
    LUISA_ASSERT(result.ec == std::errc{} && result.ptr == input.data() + input.size() && value > 0,
                 "Expected a positive integer, got '{}'.", input);
    return value;
}

[[nodiscard]] double elapsed(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>{Clock::now() - start}.count();
}

[[nodiscard]] vector<float> values(size_t count, int64_t seed) {
    vector<float> result(count);
    for (auto i = size_t{0u}; i < count; i++) {
        result[i] = static_cast<float>((static_cast<int64_t>(i) * seed + 17) % 127 - 63) / 64.0f;
    }
    return result;
}

void print_samples(const char *name, span<const double> samples) {
    std::cout << std::quoted(name) << ":[";
    auto separator = "";
    for (auto value : samples) {
        std::cout << separator << value;
        separator = ",";
    }
    std::cout << ']';
}

[[nodiscard]] double median(span<const double> samples) {
    return samples[samples.size() / 2u];
}

[[nodiscard]] Options parse_options(int argc, char *argv[]) {
    LUISA_ASSERT(argc >= 2 && argc <= 11,
                 "Usage: benchmark_tile_hlsl_comparison <cuda|dx|vk> [M=512] [N=512] [K=64] "
                 "[tile-M=32] [tile-N=32] [samples=7] [sample-ms=50] [warmup-ms=200] "
                 "[route=native|tirx|dsl]");
    Options options;
    options.backend = argv[1];
    if (argc > 2) { options.gemm.m = positive(argv[2]); }
    if (argc > 3) { options.gemm.n = positive(argv[3]); }
    if (argc > 4) { options.gemm.k = positive(argv[4]); }
    if (argc > 5) { options.gemm.tile_m = positive(argv[5]); }
    if (argc > 6) { options.gemm.tile_n = positive(argv[6]); }
    if (argc > 7) { options.samples = positive(argv[7]); }
    if (argc > 8) { options.sample_ms = positive(argv[8]); }
    if (argc > 9) { options.warmup_ms = positive(argv[9]); }
    if (argc > 10) { options.route = argv[10]; }
    auto &&cfg = options.gemm;
    LUISA_ASSERT((options.backend == "cuda" || options.backend == "dx" || options.backend == "vk") &&
                     (options.route == "native" || options.route == "tirx" || options.route == "dsl") &&
                     (options.backend == "cuda" || options.route == "native"),
                 "DX/VK require native HLSL; CUDA selects native, tirx, or dsl.");
    LUISA_ASSERT(cfg.m <= 4096 && cfg.n <= 4096 && cfg.k <= 4096 &&
                     cfg.tile_m <= 256 && cfg.tile_n <= 256 &&
                     options.samples <= 101 && options.sample_ms <= 10000 && options.warmup_ms <= 60000,
                 "Invalid shape, tile, or timing limits.");
    return options;
}

}// namespace

int main(int argc, char *argv[]) {
    auto options = parse_options(argc, argv);
    auto &&cfg = options.gemm;
    // Use identical constraint-free TileIR on every backend. HLSL still maps
    // the automatic root to groups; CUDA Tile IR rejects an explicit GROUP.
    cfg.explicit_group = false;
    log_level_error();

    auto start = Clock::now();
    Context context{argv[0]};
    auto device = context.create_device(options.backend);
    auto stream = device.create_stream(StreamTag::COMPUTE);
    auto runtime_ms = elapsed(start);

    auto host_a = values(static_cast<size_t>(cfg.m * cfg.k), 5);
    auto host_b = values(static_cast<size_t>(cfg.k * cfg.n), 11);
    vector<float> output(static_cast<size_t>(cfg.m * cfg.n), std::numeric_limits<float>::quiet_NaN());
    start = Clock::now();
    auto a = device.create_buffer<float>(host_a.size());
    auto b = device.create_buffer<float>(host_b.size());
    auto c = device.create_buffer<float>(output.size());
    stream << a.copy_from(span{host_a}) << b.copy_from(span{host_b})
           << c.copy_from(span{output}) << synchronize();
    auto upload_ms = elapsed(start);

    auto execute = [&](auto launch, ShaderDescription description) {
        auto submit = [&](uint64_t repetitions) {
            CommandList commands;
            for (auto i = uint64_t{0u}; i < repetitions; i++) { commands << launch(); }
            stream << commands.commit() << synchronize();
        };
        auto batch = [&](uint64_t repetitions) {
            stream.synchronize();
            auto before = Clock::now();
            submit(repetitions);
            return elapsed(before);
        };

        auto cold_ms = batch(1u);
        start = Clock::now();
        while (elapsed(start) < options.warmup_ms) { static_cast<void>(batch(8u)); }
        auto actual_warmup_ms = elapsed(start);
        auto repetitions = uint64_t{1u};
        for (auto attempt = 0; attempt < 8; attempt++) {
            auto ms = batch(repetitions);
            if (ms >= options.sample_ms * .8 || repetitions == 100000u) { break; }
            auto estimate = repetitions * static_cast<double>(options.sample_ms) / std::max(ms, 1e-6);
            repetitions = std::clamp<uint64_t>(static_cast<uint64_t>(estimate), repetitions + 1u, 100000u);
        }
        vector<double> throughput, latency;
        for (auto i = int64_t{0}; i < options.samples; i++) {
            throughput.emplace_back(1000.0 * batch(repetitions) / repetitions);
        }
        for (auto i = int64_t{0}; i < options.samples; i++) { latency.emplace_back(1000.0 * batch(1u)); }
        luisa::sort(throughput.begin(), throughput.end());
        luisa::sort(latency.begin(), latency.end());

        stream << c.copy_to(span{output}) << synchronize();
        start = Clock::now();
        auto max_error = 0.0;
        auto non_finite_outputs = size_t{0u};
        for (auto m = int64_t{0}; m < cfg.m; m++) {
            for (auto n = int64_t{0}; n < cfg.n; n++) {
                auto actual = output[static_cast<size_t>(m * cfg.n + n)];
                if (!std::isfinite(actual)) {
                    non_finite_outputs++;
                    continue;
                }
                auto expected = 0.0;
                for (auto k = int64_t{0}; k < cfg.k; k++) {
                    expected += static_cast<double>(host_a[static_cast<size_t>(m * cfg.k + k)]) *
                                host_b[static_cast<size_t>(k * cfg.n + n)];
                }
                max_error = std::max(max_error, std::abs(expected - actual));
            }
        }
        auto validation_ms = elapsed(start);
        auto tolerance = 1e-3 + 1e-3 * static_cast<double>(cfg.k);
        auto passed = non_finite_outputs == 0u && max_error <= tolerance;
        auto throughput_p50 = median(span{throughput});
        auto gflops = 2.0 * cfg.m * cfg.n * cfg.k / (throughput_p50 * 1e3);

        std::cout << std::setprecision(12)
                  << "{\"backend\":" << std::quoted(options.backend)
                  << ",\"route\":" << std::quoted(options.route)
                  << ",\"implementation\":" << std::quoted(description.implementation)
                  << ",\"precision\":\"fp32\",\"fast_math\":false"
                  << ",\"timing\":\"synchronized_host_wall\""
                  << ",\"m\":" << cfg.m << ",\"n\":" << cfg.n << ",\"k\":" << cfg.k
                  << ",\"tile\":[" << cfg.tile_m << ',' << cfg.tile_n << ']'
                  << ",\"dispatch\":[" << description.dispatch_size.x << ',' << description.dispatch_size.y << ',' << description.dispatch_size.z << ']'
                  << ",\"block\":[" << description.block_size.x << ',' << description.block_size.y << ',' << description.block_size.z << ']'
                  << ",\"samples\":" << options.samples << ",\"repetitions\":" << repetitions
                  << ",\"runtime_init_ms\":" << runtime_ms
                  << ",\"capture_ms\":" << description.capture_ms
                  << ",\"compile_ms\":" << description.compile_ms
                  << ",\"allocation_upload_ms\":" << upload_ms
                  << ",\"cold_call_ms\":" << cold_ms << ",\"warmup_ms\":" << actual_warmup_ms
                  << ",\"realization\":" << std::quoted(description.realization)
                  << ",\"source_hash\":" << description.source_hash << ',';
        print_samples("throughput_us", span{throughput});
        std::cout << ',';
        print_samples("latency_us", span{latency});
        std::cout << ",\"throughput_p50_us\":" << throughput_p50
                  << ",\"latency_p50_us\":" << median(span{latency})
                  << ",\"gflops\":" << gflops << ",\"max_error\":" << max_error
                  << ",\"error_tolerance\":" << tolerance
                  << ",\"validated_elements\":" << output.size()
                  << ",\"non_finite_outputs\":" << non_finite_outputs
                  << ",\"validation_ms\":" << validation_ms
                  << ",\"validation_passed\":" << (passed ? "true" : "false") << "}\n";
        return passed ? 0 : 2;
    };

    if (options.route == "dsl") {
        start = Clock::now();
        Kernel2D kernel = [](BufferFloat a, BufferFloat b, BufferFloat c,
                             UInt m, UInt n, UInt k) noexcept {
            set_block_size(16u, 16u, 1u);
            auto row = dispatch_id().x;
            auto column = dispatch_id().y;
            $if ((row < m) & (column < n)) {
                auto sum = def(0.0f);
                $for (index, k) {
                    sum += a.read(row * k + index) * b.read(index * n + column);
                };
                c.write(row * n + column, sum);
            };
        };
        auto capture_ms = elapsed(start);
        ShaderOption shader_option;
        shader_option.enable_cache = false;
        shader_option.enable_fast_math = false;
        start = Clock::now();
        auto shader = device.compile(kernel, shader_option);
        auto compile_ms = elapsed(start);
        auto m = static_cast<uint>(cfg.m);
        auto n = static_cast<uint>(cfg.n);
        auto k = static_cast<uint>(cfg.k);
        auto launch = [&] { return shader(a, b, c, m, n, k).dispatch(m, n); };
        return execute(launch, {.implementation = "cuda_dsl_scalar_gemm",
                                .realization = "CUDA ordinary DSL -> PTX; one scalar output per thread; no TileIR",
                                .dispatch_size = make_uint3(m, n, 1u),
                                .block_size = shader.block_size(),
                                .capture_ms = capture_ms,
                                .compile_ms = compile_ms});
    }

    start = Clock::now();
    auto kernel = test::tile_native::gemm(cfg);
    auto capture_ms = elapsed(start);
    LUISA_ASSERT(kernel.valid(), "Failed to capture benchmark TileIR.");
    tile::CompileOptions compile_options;
    compile_options.lowering = options.route == "tirx" ? tile::Lowering::TIRX : tile::Lowering::NATIVE;
    ShaderOption shader_option;
    shader_option.enable_cache = false;
    shader_option.enable_fast_math = false;
    start = Clock::now();
    auto shader = tile::compile(device, kernel, compile_options, shader_option);
    auto compile_ms = elapsed(start);
    if (!shader) {
        std::cerr << "Tile compilation failed: " << shader.metadata().error << '\n';
        if (options.backend == "cuda" && options.route == "native") {
            std::cerr << "CUDA native Tile IR requires LUISA_CUDA_TILE_IR=1 and a CUDA 13.4+ Tile toolchain; "
                         "use route 'tirx' or the explicit 'dsl' baseline when unavailable.\n";
        }
        return 3;
    }
    auto &&metadata = shader.metadata();
    auto route_ok = true;
    if (options.backend == "dx" || options.backend == "vk") {
        route_ok = metadata.realization.find("native Tile") != string::npos &&
                   metadata.source.find("_lc_tile_native") != string::npos;
    } else if (options.route == "native") {
        route_ok = metadata.realization.starts_with("CUDA Tile C++ -> NVRTC Tile IR -> tileiras -> cubin");
    } else {
        route_ok = metadata.realization.find("TIRx ->") != string::npos &&
                   metadata.realization.find("PTX") != string::npos;
    }
    LUISA_ASSERT(route_ok, "Requested realization was not produced: {}", metadata.realization);
    auto implementation = options.backend == "cuda" ?
                              (options.route == "native" ? "cuda_native_tile" : "cuda_tirx_tile") :
                              (options.backend == "dx" ? "dx_native_tile_hlsl" : "vk_native_tile_hlsl_spirv");
    auto launch = [&] { return shader(a, b, c).dispatch(); };
    return execute(launch, {.implementation = implementation,
                            .realization = metadata.realization,
                            .source_hash = hash<string_view>{}(metadata.source),
                            .dispatch_size = metadata.dispatch_size,
                            .block_size = shader.block_size(),
                            .capture_ms = capture_ms,
                            .compile_ms = compile_ms});
}

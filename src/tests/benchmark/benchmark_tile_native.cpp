// Real TileIR -> Metal backend -> MPP -> ordinary Runtime Stream benchmark.
// Host wall time, NOT GPU event time. Keep the handwritten MPP baseline separate.
#include "tile_native_test_utils.h"
#include <luisa/tile/runtime.h>
#include <luisa/runtime/context.h>
#include <luisa/runtime/stream.h>
#include <algorithm>
#include <charconv>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>

using namespace luisa;
using namespace luisa::compute;
using Clock = std::chrono::steady_clock;

namespace {

[[nodiscard]] int64_t positive(const char *text) {
    auto input = std::string_view{text};
    int64_t n{};
    auto result = std::from_chars(input.data(), input.data() + input.size(), n);
    if (result.ec != std::errc{} || result.ptr != input.data() + input.size() || n <= 0) { throw std::invalid_argument{"expected positive integer"}; }
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

void samples(const char *name, span<const double> values) {
    std::cout << std::quoted(name) << ":[";
    auto separator = "";
    for (auto x : values) {
        std::cout << separator << x;
        separator = ",";
    }
    std::cout << ']';
}

}// namespace

int main(int argc, char *argv[]) {
    if (argc != 14) {
        std::cerr << "Usage: benchmark_tile_native fp32 M N K samples sample-ms warmup-ms output.f32 tile-M tile-N op-subgroups group-subgroups cohort-M\n";
        return 1;
    }
    try {
        if (std::string_view{argv[1]} != "fp32") { throw std::invalid_argument{"only FP32 supported"}; }
        test::tile_native::Gemm cfg{positive(argv[2]), positive(argv[3]), positive(argv[4]), positive(argv[9]), positive(argv[10])};
        auto count = positive(argv[5]), target_ms = positive(argv[6]), warmup_ms = positive(argv[7]);
        auto op_sg = positive(argv[11]), group_sg = positive(argv[12]), cohort_m = positive(argv[13]);
        if (cfg.m > 16384 || cfg.n > 16384 || cfg.k > 16384 || cfg.tile_m > 512 || cfg.tile_n > 512 ||
            count > 101 || group_sg > 32 || (op_sg != 1 && op_sg != group_sg) ||
            group_sg % cohort_m != 0 || (op_sg != 1 && cohort_m != 1)) { throw std::invalid_argument{"invalid shape/schedule/timing limits"}; }
        if (op_sg == 1) {
            cfg.subgroups_m = cohort_m;
            cfg.subgroups_n = group_sg / cohort_m;
        }
        if (std::filesystem::exists(argv[8])) { throw std::invalid_argument{"output path already exists"}; }
        log_level_error();
        auto start = Clock::now();
        Context context{argv[0]};
        auto device = context.create_device("metal");
        auto stream = device.create_stream(StreamTag::COMPUTE);
        auto runtime_ms = elapsed(start);
        start = Clock::now();
        auto kernel = test::tile_native::gemm(cfg);
        auto capture_ms = elapsed(start);
        start = Clock::now();
        auto shader = tile::compile(device, kernel, {static_cast<uint32_t>(group_sg * 32)});
        auto compile_ms = elapsed(start);
        if (!shader) { throw std::runtime_error{shader.metadata().error.c_str()}; }
        if (auto path = std::getenv("LUISA_TILE_BENCH_DUMP_SOURCE")) {
            if (std::filesystem::exists(path)) { throw std::invalid_argument{"source path already exists"}; }
            std::ofstream file{path};
            file << shader.metadata().source;
            if (!file) { throw std::runtime_error{"cannot write generated source"}; }
        }
        auto host_a = values(cfg.m * cfg.k, 5), host_b = values(cfg.k * cfg.n, 11);
        vector<float> output(cfg.m * cfg.n, std::numeric_limits<float>::quiet_NaN());
        start = Clock::now();
        auto a = device.create_buffer<float>(host_a.size());
        auto b = device.create_buffer<float>(host_b.size());
        auto c = device.create_buffer<float>(output.size());
        stream << a.copy_from(host_a.data()) << b.copy_from(host_b.data()) << c.copy_from(output.data()) << synchronize();
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
        auto actual_warmup_ms = elapsed(start);
        uint64_t repetitions = 1;
        for (auto attempt = 0; attempt < 8; attempt++) {
            auto ms = batch(repetitions);
            if (ms >= target_ms * .8 || repetitions == 100000) { break; }
            auto estimate = repetitions * static_cast<double>(target_ms) / std::max(ms, 1e-6);
            repetitions = std::clamp<uint64_t>(static_cast<uint64_t>(estimate), repetitions + 1, 100000);
        }
        vector<double> throughput, latency;
        for (auto i = 0; i < count; i++) { throughput.emplace_back(1000.0 * batch(repetitions) / repetitions); }
        for (auto i = 0; i < count; i++) { latency.emplace_back(1000.0 * batch(1)); }
        stream << c.copy_to(output.data()) << synchronize();
        std::ofstream file{argv[8], std::ios::binary};
        file.write(reinterpret_cast<const char *>(output.data()), static_cast<std::streamsize>(output.size() * sizeof(float)));
        if (!file) { throw std::runtime_error{"cannot write output"}; }
        std::cout << std::setprecision(12)
                  << "{\"implementation\":\"tile_native_mpp\",\"backend\":\"metal\",\"precision\":\"fp32\","
                     "\"fast_math\":false,\"relaxed_precision\":false,\"timing\":\"synchronized_host_wall\","
                     "\"batch_policy\":\"one_runtime_command_list_per_batch\",\"m\":"
                  << cfg.m << ",\"n\":" << cfg.n << ",\"k\":" << cfg.k
                  << ",\"block\":[" << cfg.tile_m << ',' << cfg.tile_n << "],\"execution_simdgroups\":" << op_sg
                  << ",\"group_simdgroups\":" << group_sg << ",\"cohort_rows\":" << cohort_m
                  << ",\"repetitions\":" << repetitions << ",\"runtime_init_ms\":" << runtime_ms
                  << ",\"capture_ms\":" << capture_ms << ",\"compile_ms\":" << compile_ms
                  << ",\"allocation_upload_ms\":" << upload_ms << ",\"cold_call_ms\":" << cold_ms
                  << ",\"warmup_ms\":" << actual_warmup_ms << ",\"realization\":" << std::quoted(shader.metadata().realization)
                  << ",\"source_hash\":" << std::quoted(std::to_string(luisa::hash<luisa::string_view>{}(shader.metadata().source))) << ',';
        samples("throughput_us", throughput);
        std::cout << ',';
        samples("latency_us", latency);
        std::cout << "}\n";
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 2;
    }
}

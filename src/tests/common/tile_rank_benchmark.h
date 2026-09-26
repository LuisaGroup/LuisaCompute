#pragma once

// Ranking coverage for the existing quadratic reference composition. Keep
// Runtime E2E timing separate from optional instrumented Metal GPU intervals.
#include "tile_rank_test_utils.h"
#include "metal_benchmark.h"
#include "metal4_benchmark.h"
#include <luisa/core/logging.h>
#include <luisa/tile/runtime.h>
#include <luisa/runtime/context.h>
#include <luisa/runtime/stream.h>
#include <luisa/core/stl/filesystem.h>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <system_error>

namespace luisa::test::tile_rank {

[[nodiscard]] inline int benchmark(int argc, char *argv[], string_view backend,
                                   const compute::tile::CompileOptions &compile_options = {}) {
    using namespace compute;
    using Clock = std::chrono::steady_clock;
    LUISA_ASSERT(argc == 11, "Usage: benchmark_tile_<xir|native> rank <topk|sort> R N K <ascending|descending> samples sample-ms warmup-ms output-prefix");
    auto positive = [](string_view text) {
        int64_t n{};
        auto parsed = std::from_chars(text.data(), text.data() + text.size(), n);
        LUISA_ASSERT(parsed.ec == std::errc{} && parsed.ptr == text.data() + text.size() && n > 0, "expected positive integer: {}", text);
        return n;
    };
    auto positive_ms = [](const char *text) {
        char *end = nullptr;
        auto value = std::strtod(text, &end);
        LUISA_ASSERT(end != text && *end == '\0' && std::isfinite(value) && value > 0.0,
                     "expected finite positive milliseconds: {}", text);
        return value;
    };
    auto operation = string_view{argv[2]};
    auto row_count = positive(argv[3]), columns = positive(argv[4]), count = positive(argv[5]);
    auto direction = string_view{argv[6]};
    auto samples = positive(argv[7]);
    auto target_ms = positive_ms(argv[8]), warmup_ms = positive_ms(argv[9]);
    LUISA_ASSERT((operation == "topk" || operation == "sort") && (direction == "ascending" || direction == "descending") &&
                     (backend == "simd" || backend == "metal4" || backend == "metal") &&
                     samples <= 101 && target_ms <= 10000 && warmup_ms <= 60000,
                 "Invalid ranking operation, direction, backend or timing limits");
    auto prefix = luisa::filesystem::path{argv[10]};
    auto input_path = prefix.string() + ".input.f32";
    auto values_path = prefix.string() + ".values.f32";
    auto indices_path = prefix.string() + ".indices.i64";
    auto require_missing = [](const luisa::filesystem::path &path) {
        std::error_code error;
        auto exists = luisa::filesystem::exists(path, error);
        LUISA_ASSERT(!error && !exists, "Ranking benchmark path unavailable or already exists: {} ({})", path.string(), error.message());
    };
    for (auto &path : {input_path, values_path, indices_path}) { require_missing(path); }
    auto write = []<typename T>(const luisa::filesystem::path &path, span<const T> data) {
        std::ofstream file{path, std::ios::binary};
        file.write(reinterpret_cast<const char *>(data.data()), static_cast<std::streamsize>(data.size_bytes()));
        file.close();
        LUISA_ASSERT(file, "Cannot write ranking benchmark tensor: {}", path.string());
    };
    auto elapsed = [](Clock::time_point start) { return std::chrono::duration<double, std::milli>{Clock::now() - start}.count(); };
    log_level_error();
    auto start = Clock::now();
    auto fixture = rows(operation == "sort" ? Operation::SORT : Operation::TOPK, row_count, columns, count, direction == "descending");
    auto fixture_ms = elapsed(start);
    // Retain the exact input even if compile rejects this lowering, so an
    // Error row does not prevent independently measuring the Torch baseline.
    write(input_path, span<const float>{fixture.input});
    LUISA_ASSERT(fixture.kernel.valid(), "Ranking fixture capture failed");
    start = Clock::now();
    Context context{argv[0]};
    auto device = context.create_device(backend);
    auto stream = device.create_stream(StreamTag::COMPUTE);
    auto runtime_ms = elapsed(start);
    auto options = compile_options;
    // The legacy "native" executable's ranking mode is explicitly TIRx Metal;
    // native MPP matrix lowering is not a general-purpose ranking backend.
    options.lowering = backend == "metal" ? tile::Lowering::TIRX : tile::Lowering::NATIVE;
    start = Clock::now();
    auto shader = tile::compile(device, fixture.kernel, options);
    auto compile_ms = elapsed(start);
    LUISA_ASSERT(shader, "{}", shader.metadata().error.c_str());
    if (auto path = std::getenv("LUISA_TILE_BENCH_DUMP_SOURCE")) {
        require_missing(path);
        std::ofstream file{path};
        file << shader.metadata().source;
        file.close();
        LUISA_ASSERT(file, "Cannot write ranking source: {}", path);
    }
    start = Clock::now();
    GuardedData data{fixture};
    auto input = device.create_buffer<float>(data.input.size());
    auto values = device.create_buffer<float>(data.values.size());
    auto indices = device.create_buffer<int64_t>(data.indices.size());
    stream << input.copy_from(span{data.input}) << values.copy_from(span{data.values}) << indices.copy_from(span{data.indices}) << synchronize();
    auto upload_ms = elapsed(start);
    auto submit = [&](uint64_t repetitions) {
        CommandList commands;
        for (auto i = uint64_t{0}; i < repetitions; i++) {
            commands << shader(input.view(GuardedData::pad, fixture.input.size()),
                               values.view(GuardedData::pad, fixture.expected_values.size()),
                               indices.view(GuardedData::pad, fixture.expected_indices.size()))
                            .dispatch();
        }
        stream << commands.commit() << synchronize();
    };
    auto batch = [&](uint64_t repetitions) {
        stream.synchronize();
        auto before = Clock::now();
        submit(repetitions);
        return elapsed(before);
    };
    auto check = [&] {
        stream << input.copy_to(span{data.input}) << values.copy_to(span{data.values}) << indices.copy_to(span{data.indices}) << synchronize();
        auto validation = validate(fixture, data);
        LUISA_ASSERT(validation.passed(), "Ranking oracle mismatch: values={}, indices={}, input={}, guards={}",
                     validation.value_mismatches, validation.index_mismatches, validation.input_mismatches, validation.guard_mismatches);
    };
    auto cold_ms = batch(1u);
    check();
    start = Clock::now();
    while (elapsed(start) < warmup_ms) { static_cast<void>(batch(1u)); }
    auto actual_warmup_ms = elapsed(start);
    auto repetitions = uint64_t{1};
    auto fixed_repetitions = std::getenv("LUISA_TILE_BENCH_FIXED_REPETITIONS");
    if (fixed_repetitions != nullptr) {
        repetitions = static_cast<uint64_t>(positive(fixed_repetitions));
        LUISA_ASSERT(repetitions <= 100000u, "Fixed benchmark repetitions exceed limit");
    } else {
        for (auto attempt = 0; attempt < 8; attempt++) {
            auto ms = batch(repetitions);
            if (ms >= target_ms * .8 || repetitions == 100000u) { break; }
            repetitions = std::clamp<uint64_t>(static_cast<uint64_t>(repetitions * target_ms / std::max(ms, 1e-6)), repetitions + 1u, 100000u);
        }
    }
    vector<double> throughput, latency;
    for (auto i = int64_t{0}; i < samples; i++) { throughput.emplace_back(1000.0 * batch(repetitions) / repetitions); }
    for (auto i = int64_t{0}; i < samples; i++) { latency.emplace_back(1000.0 * batch(1u)); }
    MetalBenchmarkTiming metal_timing{backend == "metal"};
    metal_timing.measure([&] { stream.synchronize(); }, submit, repetitions, static_cast<uint32_t>(samples));
    Metal4BenchmarkTiming metal4_timing{device, stream, backend == "metal4"};
    metal4_timing.measure(submit, repetitions, static_cast<uint32_t>(samples));
    check();
    write(values_path, span<const float>{data.values}.subspan(GuardedData::pad, fixture.expected_values.size()));
    write(indices_path, span<const int64_t>{data.indices}.subspan(GuardedData::pad, fixture.expected_indices.size()));
    auto array = [](const auto &items) {
        std::cout << '[';
        auto separator = "";
        for (auto x : items) {
            std::cout << separator << x;
            separator = ",";
        }
        std::cout << ']';
    };
    std::cout << std::setprecision(12)
              << "{\"status\":\"passed\",\"implementation\":"
              << std::quoted(backend == "metal" ? "tile_tirx_metal" : backend == "metal4" ? "tile_xir_metal4" :
                                                                                            "tile_xir_simd")
              << ",\"backend\":" << std::quoted(backend)
              << ",\"operation\":" << std::quoted(operation) << ",\"direction\":" << std::quoted(direction)
              << ",\"dimensions\":[" << row_count << ',' << columns << ',' << count << ']'
              << ",\"precision\":\"fp32\",\"index_dtype\":\"int64\",\"fast_math\":false,\"relaxed_precision\":false"
                 ",\"algorithm\":\"quadratic_rank_reference\",\"stable_ties\":true,\"finite_inputs_only\":true"
                 ",\"runtime\":\"luisa\",\"timing\":\"synchronized_host_wall\",\"batch_policy\":\"one_runtime_command_list_per_batch\""
              << ",\"input_shape\":[" << row_count << ',' << columns << "],\"output_shape\":[" << row_count << ',' << count << ']'
              << ",\"input_path\":" << std::quoted(input_path) << ",\"values_path\":" << std::quoted(values_path)
              << ",\"indices_path\":" << std::quoted(indices_path)
              << ",\"fixture_ms\":" << fixture_ms << ",\"runtime_init_ms\":" << runtime_ms << ",\"compile_ms\":" << compile_ms
              << ",\"allocation_upload_ms\":" << upload_ms << ",\"cold_call_ms\":" << cold_ms << ",\"warmup_ms\":" << actual_warmup_ms
              << ",\"repetitions\":" << repetitions << ",\"repetition_policy\":" << std::quoted(fixed_repetitions == nullptr ? "adaptive_host_wall" : "fixed")
              << ",\"realization\":" << std::quoted(shader.metadata().realization)
              << ",\"correctness\":{\"checks\":2,\"values_per_check\":" << fixture.expected_values.size()
              << ",\"indices_per_check\":" << fixture.expected_indices.size() << ",\"input_elements_per_check\":" << fixture.input.size()
              << ",\"guard_elements_per_check\":" << 6u * GuardedData::pad
              << ",\"values_bitwise_equal\":true,\"indices_exact\":true,\"input_immutable\":true,\"all_guards_intact\":true}"
                 ",\"throughput_us\":";
    array(throughput);
    std::cout << ",\"latency_us\":";
    array(latency);
    metal_timing.print();
    metal4_timing.print();
    std::cout << "}\n";
    return 0;
}

}// namespace luisa::test::tile_rank

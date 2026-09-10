#pragma once

// One capture, input generator, FP64 oracle and Runtime submission protocol for
// XIR/SIMD, XIR/Metal4 and TIRx/Metal. This is a benchmark, not a dispatch policy.
#include "tile_llm_test_utils.h"
#include "metal_benchmark.h"
#include <luisa/core/logging.h>
#include <luisa/tile/runtime.h>
#include <luisa/runtime/context.h>
#include <luisa/runtime/stream.h>
#include <charconv>
#include <chrono>
#include <filesystem>
#include <system_error>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>

namespace luisa::test::tile_llm {

[[nodiscard]] inline int benchmark(int argc, char *argv[], string_view backend,
                                   const compute::tile::CompileOptions &compile_options = {}, bool reduction_tree = false,
                                   bool forward_input_views = false, bool attention_qk_reduction = false) {
    using namespace compute;
    using Clock = std::chrono::steady_clock;
    LUISA_ASSERT(argc == 10, "Usage: benchmark_tile_<xir|native> llm <swiglu|rope|rmsnorm|layernorm|gelu_residual|masked_softmax|attention> dimensions-comma-separated block-Q block-K samples sample-ms warmup-ms output.f32");
    auto integer = [](string_view text) {
        int64_t n{};
        auto parsed = std::from_chars(text.data(), text.data() + text.size(), n);
        LUISA_ASSERT(parsed.ec == std::errc{} && parsed.ptr == text.data() + text.size() && n > 0, "expected positive integer: {}", text);
        return n;
    };
    vector<int64_t> dimensions;
    auto text = string_view{argv[3]};
    for (;;) {
        auto end = text.find(',');
        dimensions.emplace_back(integer(text.substr(0u, end)));
        if (end == string_view::npos) { break; }
        text.remove_prefix(end + 1u);
    }
    auto op = string_view{argv[2]};
    auto bq = integer(argv[4]), bk = integer(argv[5]);
    auto count = integer(argv[6]), target_ms = integer(argv[7]), warmup_ms = integer(argv[8]);
    LUISA_ASSERT(bq <= 128 && bk <= 256 && count <= 101 && target_ms <= 10000 && warmup_ms <= 60000 &&
                     dimensions.size() == (op == "attention" ? 7u : 2u) &&
                     std::all_of(dimensions.begin(), dimensions.end(), [](int64_t d) { return d <= 65536; }),
                 "invalid dimensions, block or timing limits");
    // Bound all products before allocation or capture; do not overflow
    // while validating an adversarial command line.
    auto bounded_product = [](std::initializer_list<int64_t> factors) {
        auto size = int64_t{1};
        for (auto factor : factors) {
            LUISA_ASSERT(size <= (1ll << 26) / factor, "tensor exceeds benchmark element limit");
            size *= factor;
        }
    };
    if (op == "attention") {
        bounded_product({dimensions[0], dimensions[1], dimensions[3], dimensions[5]});
        bounded_product({dimensions[0], dimensions[2], dimensions[4], dimensions[5]});
        bounded_product({dimensions[0], dimensions[2], dimensions[4], dimensions[6]});
        bounded_product({dimensions[0], dimensions[1], dimensions[3], dimensions[6]});
    } else {
        LUISA_ASSERT(bq == 1 && bk == 1, "row kernels do not use attention block parameters");
        bounded_product({dimensions[0], dimensions[1]});
    }
    auto output_path = std::filesystem::path{argv[9]};
    auto require_missing = [](const std::filesystem::path &path) {
        std::error_code error;
        auto exists = std::filesystem::exists(path, error);
        LUISA_ASSERT(!error, "cannot inspect benchmark path {}: {}", path.string(), error.message());
        LUISA_ASSERT(!exists, "benchmark output/input export already exists: {}", path.string());
    };
    for (auto suffix : {"", ".input0.f32", ".input1.f32", ".input2.f32"}) {
        require_missing(output_path.string() + suffix);
    }
    auto elapsed = [](Clock::time_point start) { return std::chrono::duration<double, std::milli>{Clock::now() - start}.count(); };
    log_level_error();
    auto start = Clock::now();
    auto fixture = [&] {
        if (op == "attention") { return attention(dimensions[0], dimensions[1], dimensions[2], dimensions[3], dimensions[4], dimensions[5], dimensions[6], bq, bk, attention_qk_reduction); }
        auto kind = op == "swiglu" ? RowOp::SWIGLU : op == "rope"      ? RowOp::ROPE :
                                                 op == "rmsnorm"       ? RowOp::RMS_NORM :
                                                 op == "layernorm"     ? RowOp::LAYER_NORM :
                                                 op == "gelu_residual" ? RowOp::GELU_RESIDUAL :
                                                                         RowOp::MASKED_SOFTMAX;
        if (op != "swiglu" && op != "rope" && op != "rmsnorm" && op != "layernorm" && op != "gelu_residual" && op != "masked_softmax") {
            LUISA_ERROR("unknown LLM operation: {}", op);
        }
        return rows(kind, dimensions[0], dimensions[1]);
    }();
    auto fixture_ms = elapsed(start);
    auto write = [](const std::filesystem::path &path, span<const float> data) {
        std::ofstream file{path, std::ios::binary};
        file.write(reinterpret_cast<const char *>(data.data()), static_cast<std::streamsize>(data.size_bytes()));
        file.close();
        LUISA_ASSERT(file, "cannot write benchmark tensor: {}", path.string());
    };
    // Export inputs even when a lowering is rejected, so a failed native
    // case does not silently remove the corresponding Torch measurement.
    for (auto i = 0u; i < 3u; i++) { write(output_path.string() + ".input" + std::to_string(i) + ".f32", fixture.inputs[i]); }
    start = Clock::now();
    Context context{argv[0]};
    auto device = context.create_device(backend);
    auto stream = device.create_stream(StreamTag::COMPUTE);
    auto runtime_ms = elapsed(start);
    start = Clock::now();
    auto options = compile_options;
    options.lowering = backend == "metal" ? tile::Lowering::TIRX : tile::Lowering::NATIVE;
    auto shader = tile::compile(device, fixture.kernel, options);
    auto compile_ms = elapsed(start);
    LUISA_ASSERT(shader, "{}", shader.metadata().error.c_str());
    if (auto path = std::getenv("LUISA_TILE_BENCH_DUMP_SOURCE")) {
        require_missing(path);
        std::ofstream file{path};
        file << shader.metadata().source;
        file.close();
        LUISA_ASSERT(file, "cannot write source: {}", path);
    }
    start = Clock::now();
    auto a = device.create_buffer<float>(fixture.inputs[0].size());
    auto b = device.create_buffer<float>(fixture.inputs[1].size());
    auto c = device.create_buffer<float>(fixture.inputs[2].size());
    constexpr size_t pad = 17u;
    constexpr float guard = -719.5f;
    vector<float> output(fixture.expected.size() + 2u * pad, guard);
    std::fill(output.begin() + pad, output.end() - pad, std::numeric_limits<float>::quiet_NaN());
    auto d = device.create_buffer<float>(output.size());
    stream << a.copy_from(span{fixture.inputs[0]}) << b.copy_from(span{fixture.inputs[1]}) << c.copy_from(span{fixture.inputs[2]})
           << d.copy_from(span{output}) << synchronize();
    auto upload_ms = elapsed(start);
    auto submit = [&](uint64_t repetitions) {
        CommandList commands;
        for (auto i = uint64_t{0}; i < repetitions; i++) { commands << shader(a, b, c, d.view(pad, fixture.expected.size())).dispatch(); }
        stream << commands.commit() << synchronize();
    };
    auto batch = [&](uint64_t repetitions) {
        stream.synchronize();
        auto before = Clock::now();
        submit(repetitions);
        return elapsed(before);
    };
    auto max_error = 0.0;
    auto check = [&] {
        stream << d.copy_to(span{output}) << synchronize();
        for (auto i = size_t{0}; i < output.size(); i++) {
            if (i < pad || i >= output.size() - pad) {
                LUISA_ASSERT(output[i] == guard, "output guard overwritten at padded element {}", i);
            } else {
                auto expected = fixture.expected[i - pad];
                auto error = std::abs(output[i] - expected);
                LUISA_ASSERT(std::isfinite(output[i]) && error <= 5e-5 + 5e-5 * std::abs(expected),
                             "complete FP64 oracle mismatch at element {}: {} != {}", i - pad, output[i], expected);
                max_error = std::max(max_error, error);
            }
        }
    };
    auto cold_ms = batch(1u);
    check();// Never time a kernel that has not passed the complete oracle.
    start = Clock::now();
    while (elapsed(start) < warmup_ms) { static_cast<void>(batch(8u)); }
    auto actual_warmup_ms = elapsed(start);
    uint64_t repetitions = 1u;
    for (auto attempt = 0; attempt < 8; attempt++) {
        auto ms = batch(repetitions);
        if (ms >= target_ms * .8 || repetitions == 100000u) { break; }
        repetitions = std::clamp<uint64_t>(static_cast<uint64_t>(repetitions * target_ms / std::max(ms, 1e-6)), repetitions + 1u, 100000u);
    }
    vector<double> throughput, latency;
    for (auto i = 0; i < count; i++) { throughput.emplace_back(1000.0 * batch(repetitions) / repetitions); }
    for (auto i = 0; i < count; i++) { latency.emplace_back(1000.0 * batch(1u)); }
    MetalBenchmarkTiming timing{backend == "metal"};
    timing.measure([&] { stream.synchronize(); }, submit, repetitions, static_cast<uint32_t>(count));
    check();
    write(output_path, span{output}.subspan(pad, fixture.expected.size()));
    auto array = [](const auto &values) {
        std::cout << '[';
        auto separator = "";
        for (auto x : values) {
            std::cout << separator << x;
            separator = ",";
        }
        std::cout << ']';
    };
    std::cout << std::setprecision(12)
              << "{\"implementation\":" << std::quoted(backend == "metal" ? "tile_tirx_metal" : backend == "metal4" ? "tile_xir_metal4" :
                                                                                                                      "tile_xir_simd")
              << ",\"backend\":" << std::quoted(backend == "simd" ? "cpu" : backend)
              << ",\"precision\":\"fp32\",\"fast_math\":false,\"relaxed_precision\":false,\"runtime\":\"luisa\","
                 "\"timing\":\"synchronized_host_wall\",\"batch_policy\":\"one_runtime_command_list_per_batch\",\"operation\":"
              << std::quoted(op)
              << ",\"dimensions\":";
    array(dimensions);
    // These fixtures use the source default on every reduce. Keep the
    // legacy requested candidate flag distinct from numerical permission
    // and from the capability-resolved automatic Runtime choice.
    std::cout << ",\"reduction_tree\":" << (reduction_tree ? "true" : "false")
              << ",\"requested_input_views\":" << (forward_input_views ? "true" : "false")
              << ",\"attention_qk\":" << std::quoted(op != "attention" ? "not_applicable" : attention_qk_reduction ? "reduce" :
                                                                                                                     "mma")
              << ",\"source_reduction_policy\":\"unordered_tree\""
              << ",\"reduction_candidate_setting\":" << std::quoted(backend != "metal" ? "not_applicable" : compile_options.tirx == nullptr ? "automatic" :
                                                                                                        reduction_tree                      ? "enabled" :
                                                                                                                                              "disabled")
              << ",\"requested_group_threads\":" << options.threads_per_group
              << ",\"attention_block\":[" << bq << ',' << bk << "],\"input_shapes\":[";
    for (auto i = 0u; i < 3u; i++) {
        if (i != 0u) { std::cout << ','; }
        array(fixture.shapes[i]);
    }
    std::cout << "],\"output_shape\":";
    array(fixture.shapes[3]);
    std::cout << ",\"fixture_ms\":" << fixture_ms << ",\"runtime_init_ms\":" << runtime_ms << ",\"compile_ms\":" << compile_ms
              << ",\"allocation_upload_ms\":" << upload_ms << ",\"cold_call_ms\":" << cold_ms << ",\"warmup_ms\":" << actual_warmup_ms
              << ",\"repetitions\":" << repetitions << ",\"realization\":" << std::quoted(shader.metadata().realization)
              << ",\"correctness\":{\"checks\":2,\"elements_per_check\":" << fixture.expected.size()
              << ",\"guard_elements_per_check\":34,\"atol\":0.00005,\"rtol\":0.00005,\"max_abs_error\":" << max_error << '}'
              << ",\"throughput_us\":";
    array(throughput);
    std::cout << ",\"latency_us\":";
    array(latency);
    timing.print();
    std::cout << "}\n";
    return 0;
}

}// namespace luisa::test::tile_llm

#include "tile_migrated_test_utils.h"
#include "tile_native_test_utils.h"
#include "metal_benchmark.h"
#include <luisa/ast/ast2json.h>
#include <luisa/tile/runtime.h>
#include <luisa/tile/bridge/xir/planner.h>
#ifdef LUISA_TILE_BENCH_TIRX
#include <luisa/tile/bridge/tirx/compiler.h>
#endif
#include <luisa/runtime/context.h>
#include <luisa/runtime/stream.h>
#include <charconv>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>

using namespace luisa;
using namespace luisa::compute;
namespace migrated = luisa::test::tile_migrated;

template<typename T>
int run(int argc, char *argv[]) {
    // Like the other opt-in benchmark executables, host I/O failures use
    // exceptions. The library/example kernels do not depend on exceptions.
    try {
        if (argc != 13 && argc != 16) {
            throw std::invalid_argument{"Usage: benchmark_tile_migrated <simd|metal|metal-native|metal-tirx-mpp|metal-tirx-reference> <operation> M N K block-M block-N block-K samples sample-ms warmup-ms output-prefix [legacy.ast.json dispatch-X dispatch-Y]"};
        }
        auto route = string_view{argv[1]};
        auto backend = route.starts_with("metal") ? string_view{"metal"} : route;
        auto direct_gemm = string_view{argv[2]} == "gemm_direct";
        auto operation = migrated::parse(std::same_as<T, float> && !direct_gemm ? argv[2] : "gemm");
        constexpr auto precision = std::same_as<T, float> ? "fp32" : std::same_as<T, half> ? "fp16" :
                                                                                             "bf16";
        constexpr auto input0_suffix = std::same_as<T, float> ? ".input0.f32" : std::same_as<T, half> ? ".input0.f16" :
                                                                                                        ".input0.bf16";
        constexpr auto input1_suffix = std::same_as<T, float> ? ".input1.f32" : std::same_as<T, half> ? ".input1.f16" :
                                                                                                        ".input1.bf16";
        constexpr auto output_suffix = std::same_as<T, float> ? ".output.f32" : std::same_as<T, half> ? ".output.f16" :
                                                                                                        ".output.bf16";
        if ((route != "simd" && route != "metal" && route != "metal-native" && route != "metal-tirx-mpp" && route != "metal-tirx-reference") || !operation) { throw std::invalid_argument{"unknown lowering route or operation"}; }
        if (argc == 16 && route != "simd" && route != "metal") { throw std::invalid_argument{"legacy AST replay has no TileIR lowering options"}; }
        auto integer = [](string_view text) {
            int64_t value{};
            auto parsed = std::from_chars(text.data(), text.data() + text.size(), value);
            if (parsed.ec != std::errc{} || parsed.ptr != text.data() + text.size() || value <= 0) {
                throw std::invalid_argument{"expected a positive integer"};
            }
            return value;
        };
        auto rows = integer(argv[3]), columns = integer(argv[4]), depth = integer(argv[5]);
        auto block = example::tile::Block{integer(argv[6]), integer(argv[7]), integer(argv[8])};
        auto samples = integer(argv[9]), sample_ms = integer(argv[10]), warmup_ms = integer(argv[11]);
        if (rows > 16384 || columns > 16384 || depth > 16384 ||
            rows * columns > (1ll << 24) || rows * depth > (1ll << 24) || depth * columns > (1ll << 24) ||
            block.m > 128 || block.n > 128 || block.k > 256 || samples > 101 || sample_ms > 10000 || warmup_ms > 60000 ||
            (*operation != migrated::Operation::GEMM && depth != 1)) {
            throw std::invalid_argument{"invalid shape, block or timing limits"};
        }
        auto prefix = std::filesystem::path{argv[12]};
        for (auto suffix : {input0_suffix, input1_suffix, output_suffix, ".expected.f64", ".source.txt"}) {
            if (std::filesystem::exists(prefix.string() + suffix)) { throw std::invalid_argument{"output prefix already exists"}; }
        }
        auto write = [&](const char *suffix, const auto &values) {
            std::ofstream file{prefix.string() + suffix, std::ios::binary};
            file.write(reinterpret_cast<const char *>(values.data()),
                       static_cast<std::streamsize>(values.size() * sizeof(values[0])));
            if (!file) { throw std::runtime_error{"cannot write benchmark artifact"}; }
        };
        using Clock = std::chrono::steady_clock;
        auto elapsed = [](auto start) { return std::chrono::duration<double, std::milli>{Clock::now() - start}.count(); };
        log_level_error();
        auto start = Clock::now();
        auto fixture = migrated::make(*operation, rows, columns, depth, block);
        if (direct_gemm) {
            // A separately labelled full-K source schedule, not a claim that
            // the planner transformed the pipelined legacy-style capture.
            fixture.kernel = test::tile_native::gemm({rows, columns, depth, block.m, block.n});
        }
        if constexpr (!std::same_as<T, float>) {
            fixture.kernel = example::tile::gemm<T>(rows, columns, depth, block);
            // These /64 inputs are exactly representable in both FP16/BF16.
            // The oracle is the FP64 dot product rounded to output storage.
            for (auto &value : fixture.expected) { value = static_cast<float>(static_cast<T>(static_cast<float>(value))); }
        }
        vector<vector<T>> host_inputs;
        for (auto &input : fixture.inputs) {
            auto &typed = host_inputs.emplace_back();
            typed.reserve(input.size());
            for (auto value : input) { typed.emplace_back(static_cast<T>(value)); }
        }
        auto fixture_ms = elapsed(start);
        if (!fixture.kernel.valid()) { throw std::runtime_error{"kernel capture failed"}; }
        write(input0_suffix, host_inputs[0]);
        if (host_inputs.size() == 2u) { write(input1_suffix, host_inputs[1]); }
        write(".expected.f64", fixture.expected);
        Context context{argv[0]};
        auto device = context.create_device(backend);
        auto stream = device.create_stream(StreamTag::COMPUTE);
        tile::CompileOptions options;
        tile::bridge::xir::PlannerOptions xir_options;
        if (auto value = std::getenv("LUISA_TILE_BENCH_XIR_MAX_UNROLLED")) {
            auto limit = integer(value);
            if (backend != "simd" || argc == 16 || limit > UINT32_MAX) { throw std::invalid_argument{"unroll constraint requires the modern SIMD route"}; }
            xir_options.max_unrolled_tile_elements = static_cast<uint32_t>(limit);
            options.xir = &xir_options;
        }
        options.lowering = backend == "metal" && route != "metal-native" ? tile::Lowering::TIRX : tile::Lowering::NATIVE;
#ifdef LUISA_TILE_BENCH_TIRX
        tile::bridge::tirx::CompileOptions tirx_options;
        if (route == "metal-tirx-mpp" || route == "metal-tirx-reference") {
            tirx_options.cooperative_matrix = route == "metal-tirx-mpp";
            tirx_options.metal_mpp = route == "metal-tirx-mpp";
            tirx_options.forward_readonly_tile_loads = route == "metal-tirx-mpp";
            tirx_options.planner.metal_subgroup_reductions = route == "metal-tirx-mpp";
            options.tirx = &tirx_options;
        }
#else
        if (route == "metal-tirx-mpp" || route == "metal-tirx-reference") {
            throw std::invalid_argument{"this benchmark was built without TIRx policy support"};
        }
#endif
        start = Clock::now();
        auto legacy = argc == 16;
        auto shader = [&] {
            if (!legacy) { return tile::compile(device, fixture.kernel, options, {.enable_fast_math = false}); }
            // Replay old lowering through the SAME current backend/Runtime.
            // The old frontend exists only in an isolated AST exporter.
            auto dx = integer(argv[14]), dy = integer(argv[15]);
            if (dx > UINT32_MAX || dy > UINT32_MAX) { throw std::invalid_argument{"invalid legacy dispatch extent"}; }
            auto bytes = std::filesystem::file_size(argv[13]);
            if (bytes == 0u || bytes > ASTJsonLimits{}.max_document_bytes) { throw std::invalid_argument{"invalid legacy AST document size"}; }
            std::ifstream input{argv[13], std::ios::binary};
            string json(bytes, '\0');
            input.read(json.data(), static_cast<std::streamsize>(bytes));
            if (!input) { throw std::runtime_error{"cannot read legacy AST"}; }
            auto decoded = from_json(json);
            if (!decoded) { throw std::runtime_error{decoded.error.c_str()}; }
            Function function{decoded.function.get()};
            auto arguments = function.arguments();
            if (function.tag() != Function::Tag::KERNEL ||
                arguments.size() != function.unbound_arguments().size() ||
                arguments.size() < 2u || arguments.size() > fixture.inputs.size() + 1u) {
                throw std::invalid_argument{"legacy AST must be an unbound buffer-only compute kernel"};
            }
            tile::KernelMetadata metadata;
            metadata.dispatch_size = make_uint3(static_cast<uint>(dx), static_cast<uint>(dy), 1u);
            metadata.source = std::move(json);
            metadata.realization = "legacy_tile_to_kernel;current_backend;serialized_simt_ast";
            for (auto i = size_t{0}; i < arguments.size(); i++) {
                auto type = arguments[i].type();
                if (!type->is_buffer() || type->element() != Type::of<T>()) { throw std::invalid_argument{"legacy replay buffer precision does not match the requested benchmark"}; }
                auto count = i + 1u == arguments.size() ? fixture.expected.size() : fixture.inputs[i].size();
                metadata.arguments.push_back({tile::scalar_type_v<T>, count * sizeof(T), function.variable_usage(arguments[i].uid())});
            }
            auto info = device.impl()->create_shader({.enable_fast_math = false}, function);
            if (!info.valid()) { throw std::runtime_error{"legacy AST backend compilation failed"}; }
            return tile::Shader{device.impl(), info, std::move(metadata)};
        }();
        auto compile_ms = elapsed(start);
        if (!shader) { throw std::runtime_error{shader.metadata().error.c_str()}; }
        {
            std::ofstream source{prefix.string() + ".source.txt"};
            source << shader.metadata().source;
            if (!source) { throw std::runtime_error{"cannot write lowered source"}; }
        }
        vector<Buffer<T>> inputs;
        for (auto &&data : host_inputs) {
            inputs.emplace_back(device.create_buffer<T>(data.size()));
            stream << inputs.back().copy_from(span{data});
        }
        vector<T> output(fixture.expected.size() + 2u * migrated::padding, static_cast<T>(migrated::canary));
        std::fill(output.begin() + migrated::padding, output.end() - migrated::padding, std::numeric_limits<T>::quiet_NaN());
        auto destination = device.create_buffer<T>(output.size());
        auto view = destination.view(migrated::padding, fixture.expected.size());
        stream << destination.copy_from(span{output}) << synchronize();
        auto submit = [&](uint64_t count) {
            CommandList commands;
            for (auto i = uint64_t{0}; i < count; i++) {
                if (shader.metadata().arguments.size() == 3u) {
                    commands << shader(inputs[0], inputs[1], view).dispatch();
                } else {
                    commands << shader(inputs[0], view).dispatch();
                }
            }
            stream << commands.commit() << synchronize();
        };
        auto batch = [&](uint64_t count) {
            stream.synchronize();
            auto before = Clock::now();
            submit(count);
            return elapsed(before);
        };
        auto maximum_error = 0.0;
        auto check = [&] {
            stream << destination.copy_to(span{output}) << synchronize();
            vector<float> normalized;
            normalized.reserve(output.size());
            for (auto value : output) { normalized.emplace_back(static_cast<float>(value)); }
            auto checked = migrated::validate(normalized, fixture.expected, static_cast<float>(static_cast<T>(migrated::canary)));
            if (!checked.passed) { throw std::runtime_error{"complete oracle/guard mismatch at padded element " + std::to_string(checked.bad_index)}; }
            maximum_error = std::max(maximum_error, checked.max_abs_error);
        };
        auto cold_ms = batch(1u);
        check();
        start = Clock::now();
        while (elapsed(start) < warmup_ms) { static_cast<void>(batch(4u)); }
        auto actual_warmup_ms = elapsed(start);
        auto repetitions = uint64_t{1};
        for (auto attempt = 0; attempt < 8; attempt++) {
            auto ms = batch(repetitions);
            if (ms >= sample_ms * .8 || repetitions == 100000u) { break; }
            repetitions = std::clamp<uint64_t>(static_cast<uint64_t>(repetitions * sample_ms / std::max(ms, 1e-6)), repetitions + 1u, 100000u);
        }
        vector<double> throughput, latency;
        for (auto i = 0; i < samples; i++) { throughput.emplace_back(1000.0 * batch(repetitions) / repetitions); }
        for (auto i = 0; i < samples; i++) { latency.emplace_back(1000.0 * batch(1u)); }
        test::MetalBenchmarkTiming timing{backend == "metal"};
        timing.measure([&] { stream.synchronize(); }, submit, repetitions, static_cast<uint32_t>(samples));
        check();
        write(output_suffix, span{output}.subspan(migrated::padding, fixture.expected.size()));
        auto print_array = [](const auto &values) {
            std::cout << '[';
            auto separator = "";
            for (auto value : values) {
                std::cout << separator << value;
                separator = ",";
            }
            std::cout << ']';
        };
        std::cout << std::setprecision(12)
                  << "{\"suite\":\"legacy_tile_port\",\"implementation\":" << std::quoted(legacy ? (backend == "metal" ? "legacy_tile_ast_current_metal" : "legacy_tile_ast_current_simd") : route == "metal-native" ? "tile_native_mpp_metal" :
                                                                                                                                                                                         backend == "metal"          ? "tile_tirx_metal" :
                                                                                                                                                                                                                       "tile_xir_simd")
                  << ",\"requested_lowering_route\":" << std::quoted(route)
                  << ",\"source_schedule\":" << std::quoted(direct_gemm ? "full_k_direct_mma" : *operation == migrated::Operation::GEMM ? "k_tile_pipeline" :
                                                                                                                                          "migrated_library_operation")
                  << ",\"source_kind\":" << std::quoted(legacy ? "serialized_simt_ast" : "tile_lowering_source")
                  << ",\"operation\":" << std::quoted(argv[2]) << ",\"precision\":" << std::quoted(precision) << ",\"accumulation\":\"fp32\",\"fast_math\":false"
                  << ",\"dimensions\":[" << rows << ',' << columns << ',' << depth << ']'
                  << ",\"dispatch\":[" << shader.metadata().dispatch_size.x << ',' << shader.metadata().dispatch_size.y << ',' << shader.metadata().dispatch_size.z << ']'
                  << ",\"requested_max_unrolled_tile_elements\":" << (options.xir == nullptr ? 0u : xir_options.max_unrolled_tile_elements)
                  << ",\"block\":[" << block.m << ',' << block.n << ',' << block.k << ']'
                  << ",\"block_applies_to\":" << std::quoted(legacy ? "unused_modern_fixture" : "modern_kernel")
                  << ",\"source_reduction_policy\":" << std::quoted(legacy ? "legacy_lowering_order" : "unordered_tree")
                  << ",\"scan_algorithm\":" << std::quoted(*operation == migrated::Operation::CUMSUM || *operation == migrated::Operation::CUMMAX ? (legacy ? "legacy_scan_lowering" : "blocked_inclusive_hillis_steele_32") : "not_applicable")
                  << ",\"rmsnorm_weight\":false,\"rmsnorm_epsilon\":1e-12"
                  << ",\"input_distribution\":\"binary_fraction_periods_97_89\",\"fixture_ms\":" << fixture_ms
                  << ",\"compile_ms\":" << compile_ms << ",\"cold_call_ms\":" << cold_ms << ",\"warmup_ms\":" << actual_warmup_ms
                  << ",\"repetitions\":" << repetitions << ",\"realization\":" << std::quoted(shader.metadata().realization)
                  << ",\"correctness\":{\"checks\":2,\"elements_per_check\":" << fixture.expected.size()
                  << ",\"guard_elements_per_check\":34,\"atol\":" << migrated::atol << ",\"rtol\":" << migrated::rtol
                  << ",\"max_abs_error\":" << maximum_error << '}'
                  << ",\"timing\":\"synchronized_host_wall\",\"batch_policy\":\"one_runtime_command_list_per_batch\",\"throughput_us\":";
        print_array(throughput);
        std::cout << ",\"latency_us\":";
        print_array(latency);
        timing.print();
        std::cout << "}\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 2;
    }
}

int main(int argc, char *argv[]) {
    if (argc > 2) {
        auto operation = string_view{argv[2]};
        if (operation == "gemm_fp16") { return run<half>(argc, argv); }
        if (operation == "gemm_bf16") { return run<tile::bf16>(argc, argv); }
    }
    return run<float>(argc, argv);
}

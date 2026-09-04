// Device-resident native TileIR -> TVMx timing companion to the PyTorch driver.
// This reports synchronized host wall time, including dispatch overhead, NOT
// GPU hardware-event time. Every input/output allocation precedes warm timing.
#include "tile_tirx_test_utils.h"

#include <luisa/core/mathematics.h>
#include <luisa/tile/algorithms.h>

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string_view>
#include <vector>

using namespace luisa::compute::tile;
using luisa::ceil_div;
using luisa::test::tile_tirx::Runtime;

namespace {

using Clock = std::chrono::steady_clock;

#if defined(__clang__)
constexpr std::string_view compiler_version = __clang_version__;
#elif defined(__GNUC__)
constexpr std::string_view compiler_version = __VERSION__;
#else
constexpr std::string_view compiler_version = "unknown";
#endif

struct Configuration {
    int64_t m;
    int64_t n;
    int64_t k;
    int64_t bm;
    int64_t bn;
    int64_t bk;
    exec::Scope execution_scope{exec::Scope::AUTOMATIC};
    uint32_t pipeline_window{2u};
};

[[nodiscard]] exec::Scope parse_execution_scope(std::string_view name) {
    if (name == "auto") { return exec::Scope::AUTOMATIC; }
    if (name == "worker") { return exec::Scope::WORKER; }
    if (name == "group") { return exec::Scope::GROUP; }
    throw std::invalid_argument{"execution scope must be auto, worker, or group"};
}

[[nodiscard]] int64_t positive_integer(const char *text) {
    auto input = std::string_view{text};
    int64_t value = 0;
    auto parsed = std::from_chars(input.data(), input.data() + input.size(), value);
    if (parsed.ec != std::errc{} || parsed.ptr != input.data() + input.size() || value <= 0) {
        throw std::invalid_argument{"expected a positive integer"};
    }
    return value;
}

[[nodiscard]] double milliseconds(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>{Clock::now() - start}.count();
}

void print_plans(luisa::span<const bridge::tirx::GroupPlan> plans) {
    std::cout << "\"execution_plans\":[";
    auto separator = "";
    for (auto &plan : plans) {
        std::cout << separator << "{\"threads\":" << plan.threads
                  << ",\"shared_memory_bytes\":" << plan.shared_memory_bytes
                  << ",\"optimized\":" << (plan.optimized ? "true" : "false")
                  << ",\"candidates_considered\":" << plan.candidates_considered
                  << ",\"candidates_rejected\":" << plan.candidates_rejected
                  << ",\"max_copy_batch\":" << plan.max_copy_batch
                  << ",\"batched_copy_operations\":" << plan.batched_copy_operations
                  << ",\"group_barrier_sites_before\":" << plan.group_barrier_sites_before
                  << ",\"group_barrier_sites_after\":" << plan.group_barrier_sites_after
                  << ",\"normalized_cost\":" << plan.cost.score
                  << ",\"matrix_issues\":" << plan.cost.matrix_issues
                  << ",\"shared_fragment_transfers\":" << plan.cost.shared_fragment_transfers
                  << ",\"direct_fragment_stores\":" << plan.cost.direct_fragment_stores
                  << ",\"fragment_scalars_per_lane\":" << plan.cost.fragment_scalars_per_lane
                  << ",\"matrices\":[";
        auto matrix_separator = "";
        for (auto &matrix : plan.matrices) {
            std::cout << matrix_separator << "{\"subgroups_m\":" << matrix.subgroups_m << ",\"subgroups_n\":" << matrix.subgroups_n
                      << ",\"atom_rows\":" << matrix.atom_rows << ",\"atom_columns\":" << matrix.atom_columns
                      << ",\"persistent_accumulator\":" << (matrix.persistent_accumulator ? "true" : "false")
                      << ",\"direct_accumulator_store\":" << (matrix.direct_accumulator_store ? "true" : "false") << '}';
            matrix_separator = ",";
        }
        std::cout << "]}";
        separator = ",";
    }
    std::cout << ']';
}

// Count actual static call sites in the generated Metal source, not merely
// semantic MMA operations or a requested compiler capability.
[[nodiscard]] size_t matrix_intrinsics(const tvm::ffi::Module &module) {
    auto count = size_t{0u};
    if (std::string_view{module->kind()} == "metal") {
        auto source = module->InspectSource("metal");
        auto code = std::string_view{source.data(), source.size()};
        constexpr auto call = std::string_view{"simdgroup_multiply_accumulate("};
        for (auto position = code.find(call); position != std::string_view::npos; position = code.find(call, position + call.size())) { count++; }
    }
    for (auto &&child : module->imports()) { count += matrix_intrinsics(child.cast<tvm::ffi::Module>()); }
    return count;
}

// Explicit opt-in diagnostics, outside all timed phases. For Metal inspect
// the device module, not the LLVM host launch wrapper.
void dump_source(const tvm::ffi::Module &module, std::string_view kind, const char *path) {
    if (std::string_view{module->kind()} == kind) {
        if (std::filesystem::exists(path)) { throw std::runtime_error{"source dump path already exists"}; }
        auto source = module->InspectSource(kind == "metal" ? "metal" : "ll");
        std::ofstream file{path};
        file.write(source.data(), static_cast<std::streamsize>(source.size()));
        if (!file) { throw std::runtime_error{"failed to write generated source"}; }
    }
    for (auto &&child : module->imports()) { dump_source(child.cast<tvm::ffi::Module>(), kind, path); }
}

[[nodiscard]] Kernel capture(std::string_view operation, Configuration cfg) {
    if (operation == "gemm") {
        auto definition = tile_kernel("benchmark_gemm", [=](TensorView<const float, 2> A,
                                                            TensorView<const float, 2> B,
                                                            TensorView<float, 2> C) {
            auto gm = axis("block_m", ceil_div(A.extent<0>(), cfg.bm));
            auto gn = axis("block_n", ceil_div(B.extent<1>(), cfg.bn));
            auto kt = axis("k_tiles", ceil_div(A.extent<1>(), cfg.bk));
            auto m = axis("m", cfg.bm);
            auto n = axis("n", cfg.bn);
            auto k = axis("k", cfg.bk);
            for (auto &nest : parallel(shape(gm, gn), cfg.execution_scope)) {
                auto m0 = nest.index(gm) * cfg.bm;
                auto n0 = nest.index(gn) * cfg.bn;
                auto acc = zeros<float>(shape(m, n));
                for (auto &step : nest.pipeline(shape(kt), {.stages = cfg.pipeline_window, .initiation_interval = 1})) {
                    auto k0 = step.index() * cfg.bk;
                    step.stage("load");
                    auto a = A[coord(m0, k0), shape(m, k)];
                    auto b = B[coord(k0, n0), shape(k, n)];
                    step.stage("compute");
                    acc = mma(a, b, acc);
                }
                C(coord(m0, n0), shape(m, n)).store(acc);
            }
        });
        return definition.capture(tensor_shape(cfg.m, cfg.k), tensor_shape(cfg.k, cfg.n), tensor_shape(cfg.m, cfg.n));
    }
    if (operation == "add") {
        auto definition = tile_kernel("benchmark_add", [=](TensorView<const float, 2> A,
                                                           TensorView<const float, 2> B,
                                                           TensorView<float, 2> C) {
            auto gm = axis("block_m", ceil_div(A.extent<0>(), cfg.bm));
            auto gn = axis("block_n", ceil_div(A.extent<1>(), cfg.bn));
            auto m = axis("m", cfg.bm);
            auto n = axis("n", cfg.bn);
            for (auto &nest : parallel(shape(gm, gn), cfg.execution_scope)) {
                auto origin = coord(nest.index(gm) * cfg.bm, nest.index(gn) * cfg.bn);
                C(origin, shape(m, n)).store(A[origin, shape(m, n)] + B[origin, shape(m, n)]);
            }
        });
        return definition.capture(tensor_shape(cfg.m, cfg.n), tensor_shape(cfg.m, cfg.n), tensor_shape(cfg.m, cfg.n));
    }
    if (operation == "sum") {
        auto definition = tile_kernel("benchmark_sum", [=](TensorView<const float, 2> A, TensorView<float, 1> C) {
            auto rows = axis("rows", A.extent<0>());
            auto m = axis("m", 1);
            auto n = axis("n", A.extent<1>());
            for (auto &nest : parallel(shape(rows), cfg.execution_scope)) {
                auto value = A[coord(nest.index(), 0), shape(m, n)];
                C(coord(nest.index()), shape(m)).store(reduce(value, n, add));
            }
        });
        return definition.capture(tensor_shape(cfg.m, cfg.n), tensor_shape(cfg.m));
    }
    if (operation == "softmax") {
        auto definition = tile_kernel("benchmark_softmax", [=](TensorView<const float, 2> A, TensorView<float, 2> C) {
            auto rows = axis("rows", A.extent<0>());
            auto m = axis("m", 1);
            auto n = axis("n", A.extent<1>());
            for (auto &nest : parallel(shape(rows), cfg.execution_scope)) {
                auto origin = coord(nest.index(), 0);
                auto value = A[origin, shape(m, n)];
                auto exponentials = exp(value - reduce(value, n, maximum));
                C(origin, shape(m, n)).store(exponentials / reduce(exponentials, n, add));
            }
        });
        return definition.capture(tensor_shape(cfg.m, cfg.n), tensor_shape(cfg.m, cfg.n));
    }
    throw std::invalid_argument{"operation must be gemm, add, sum, or softmax"};
}

[[nodiscard]] luisa::vector<float> input_values(size_t count, uint64_t seed) {
    luisa::vector<float> result(count);
    for (auto i = 0u; i < count; i++) {
        result[i] = static_cast<float>(static_cast<int64_t>((i * seed + 17u) % 127u) - 63) / 64.0f;
    }
    return result;
}

[[nodiscard]] double batch(const Runtime &runtime, const std::function<void()> &invoke, uint64_t repetitions) {
    runtime.synchronize();
    auto start = Clock::now();
    for (auto i = 0u; i < repetitions; i++) { invoke(); }
    runtime.synchronize();
    return milliseconds(start);
}

void print_samples(std::string_view name, const std::vector<double> &samples) {
    std::cout << std::quoted(name) << ":[";
    for (auto i = 0u; i < samples.size(); i++) {
        if (i != 0u) { std::cout << ','; }
        std::cout << samples[i];
    }
    std::cout << ']';
}

}// namespace

int main(int argc, char *argv[]) {
    if (argc < 13 || argc > 19) {
        std::cerr << "Usage: benchmark_tile_tirx <cpu|metal> <gemm|add|sum|softmax> M N K BM BN BK samples sample-ms warmup-ms output.f32 [auto|worker|group] [pipeline-window:1|2] [scalar|matrix] [vectorize|no-vectorize|auto-vectorize] [group-threads:auto|N] [copy-batch:1..16]\n";
        return 1;
    }
    try {
        auto backend = std::string_view{argv[1]};
        auto operation = std::string_view{argv[2]};
        Configuration cfg{positive_integer(argv[3]), positive_integer(argv[4]), positive_integer(argv[5]),
                          positive_integer(argv[6]), positive_integer(argv[7]), positive_integer(argv[8])};
        auto execution_scope = argc >= 14 ? std::string_view{argv[13]} : std::string_view{"auto"};
        cfg.execution_scope = parse_execution_scope(execution_scope);
        auto pipeline_window = argc >= 15 ? positive_integer(argv[14]) : 2;
        if (pipeline_window > 2) { throw std::invalid_argument{"benchmark pipeline window must be 1 or 2"}; }
        cfg.pipeline_window = static_cast<uint32_t>(pipeline_window);
        auto matrix_mode = argc >= 16 ? std::string_view{argv[15]} : std::string_view{"scalar"};
        if (matrix_mode != "scalar" && matrix_mode != "matrix") { throw std::invalid_argument{"matrix mode must be scalar or matrix"}; }
        auto cooperative_matrix = matrix_mode == "matrix";
        auto vector_mode = argc >= 17 ? std::string_view{argv[16]} : std::string_view{"vectorize"};
        if (vector_mode != "vectorize" && vector_mode != "no-vectorize" && vector_mode != "auto-vectorize") { throw std::invalid_argument{"vector mode must be vectorize, no-vectorize, or auto-vectorize"}; }
        auto vectorize = vector_mode != "no-vectorize";
        auto auto_vectorize = vector_mode == "auto-vectorize";
        bridge::tirx::PlannerOptions planner;
        if (argc >= 18 && std::string_view{argv[17]} != "auto") {
            auto requested = positive_integer(argv[17]);
            if (requested > std::numeric_limits<uint32_t>::max() || backend != "metal" || cfg.execution_scope != exec::Scope::GROUP) {
                throw std::invalid_argument{"explicit group threads require a uint32 count and Metal group execution"};
            }
            planner.threads_per_group = static_cast<uint32_t>(requested);
        }
        if (argc == 19) {
            auto requested = positive_integer(argv[18]);
            if (requested > 16u || backend != "metal" || cfg.execution_scope != exec::Scope::GROUP) {
                throw std::invalid_argument{"copy batching requires a value in [1,16] and Metal group execution"};
            }
            planner.max_copy_batch = static_cast<uint32_t>(requested);
        }
        auto sample_count = positive_integer(argv[9]);
        auto target_ms = positive_integer(argv[10]);
        auto warmup_ms = positive_integer(argv[11]);
        if (cfg.m > 16384 || cfg.n > 16384 || cfg.k > 16384 ||
            cfg.bm > 512 || cfg.bn > 16384 || cfg.bk > 512 || sample_count > 101) {
            throw std::invalid_argument{"benchmark dimensions or sample count exceed the supported limit"};
        }
        auto start = Clock::now();
        Runtime runtime{backend};
        auto runtime_init_ms = milliseconds(start);
        start = Clock::now();
        auto kernel = capture(operation, cfg);
        auto capture_ms = milliseconds(start);
        start = Clock::now();
        auto executable = runtime.build(kernel, true, cooperative_matrix, vectorize, auto_vectorize, planner);
        auto compile_ms = milliseconds(start);
        if (!executable.ok()) { throw std::runtime_error{executable.error.c_str()}; }
        auto matrix_calls = matrix_intrinsics(executable.module.value());
        if (auto path = std::getenv("LUISA_TILE_BENCH_DUMP_SOURCE")) {
            dump_source(executable.module.value(), backend == "metal" ? "metal" : "llvm", path);
            if (!std::filesystem::exists(path)) { throw std::runtime_error{"requested generated source is unavailable"}; }
        }
        auto columns_a = operation == "gemm" ? cfg.k : cfg.n;
        auto rows_b = operation == "gemm" ? cfg.k : cfg.m;
        auto binary = operation == "gemm" || operation == "add";
        auto host_a = input_values(static_cast<size_t>(cfg.m * columns_a), 5);
        auto host_b = input_values(binary ? static_cast<size_t>(rows_b * cfg.n) : 0u, 11);
        start = Clock::now();
        auto a = runtime.upload<float>({cfg.m, columns_a}, host_a);
        tvm::runtime::Tensor b;
        if (binary) { b = runtime.upload<float>({rows_b, cfg.n}, host_b); }
        auto out = operation == "sum" ? runtime.allocate<float>({cfg.m}) : runtime.allocate<float>({cfg.m, cfg.n});
        runtime.synchronize();
        auto allocation_upload_ms = milliseconds(start);
        std::function<void()> invoke = [&] {
            if (binary) {
                (*executable.entry)(a, b, out);
            } else {
                (*executable.entry)(a, out);
            }
        };
        auto cold_call_ms = batch(runtime, invoke, 1);
        start = Clock::now();
        while (milliseconds(start) < static_cast<double>(warmup_ms)) { static_cast<void>(batch(runtime, invoke, 8)); }
        auto actual_warmup_ms = milliseconds(start);
        uint64_t repetitions = 1u;
        for (auto attempt = 0u; attempt < 8u; attempt++) {
            auto elapsed = batch(runtime, invoke, repetitions);
            if (elapsed >= target_ms * 0.8 || repetitions == 100000u) { break; }
            auto estimate = repetitions * static_cast<double>(target_ms) / std::max(elapsed, 1e-6);
            repetitions = std::clamp<uint64_t>(static_cast<uint64_t>(estimate), repetitions + 1u, 100000u);
        }
        std::vector<double> throughput_us;
        std::vector<double> latency_us;
        for (auto i = 0; i < sample_count; i++) { throughput_us.emplace_back(1000.0 * batch(runtime, invoke, repetitions) / repetitions); }
        for (auto i = 0; i < sample_count; i++) { latency_us.emplace_back(1000.0 * batch(runtime, invoke, 1)); }
        start = Clock::now();
        auto output_count = static_cast<size_t>(operation == "sum" ? cfg.m : cfg.m * cfg.n);
        auto output = runtime.download<float>(out, output_count);
        auto download_ms = milliseconds(start);
        std::ofstream file{argv[12], std::ios::binary};
        file.write(reinterpret_cast<const char *>(output.data()), static_cast<std::streamsize>(output.size() * sizeof(float)));
        if (!file) { throw std::runtime_error{"failed to write benchmark output"}; }
        std::cout << std::setprecision(12)
                  << "{\"backend\":" << std::quoted(backend) << ",\"operation\":" << std::quoted(operation)
                  << ",\"execution_scope\":" << std::quoted(execution_scope)
                  << ",\"pipeline_window\":" << cfg.pipeline_window
                  << ",\"cooperative_matrix\":" << (cooperative_matrix ? "true" : "false")
                  << ",\"vectorize\":" << (vectorize ? "true" : "false")
                  << ",\"auto_vectorize\":" << (auto_vectorize ? "true" : "false")
                  << ",\"planner_threads\":" << planner.threads_per_group
                  << ",\"copy_batch\":" << planner.max_copy_batch
                  << ",\"matrix_intrinsics\":" << matrix_calls
                  << ",\"runtime_init_ms\":" << runtime_init_ms << ",\"capture_ms\":" << capture_ms
                  << ",\"compile_ms\":" << compile_ms << ",\"allocation_upload_ms\":" << allocation_upload_ms
                  << ",\"cold_call_ms\":" << cold_call_ms << ",\"warmup_ms\":" << actual_warmup_ms
                  << ",\"download_ms\":" << download_ms << ",\"repetitions\":" << repetitions
                  << ",\"output_elements\":" << output_count << ",\"mma_operations\":"
                  << luisa::test::tile_tirx::count_operations(kernel.function().body(), OperationKind::MMA)
                  << ",\"compiler\":" << std::quoted(compiler_version) << ',';
        print_samples("throughput_us", throughput_us);
        std::cout << ',';
        print_samples("latency_us", latency_us);
        std::cout << ',';
        print_plans(executable.plans);
        std::cout << "}\n";
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 2;
    }
}

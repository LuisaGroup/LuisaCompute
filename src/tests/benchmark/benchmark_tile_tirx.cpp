// Device-resident native TileIR -> TVMx timing companion to the PyTorch driver.
// Host-wall samples include dispatch overhead. Optional Metal GPU counters
// run in a separate phase; every input/output allocation precedes warm timing.
#include "tile_tirx_test_utils.h"
#include "metal_benchmark.h"

#include <luisa/core/mathematics.h>
#include <luisa/tile/algorithms.h>
#include <luisa/tile/runtime.h>
#include <luisa/runtime/context.h>
#include <luisa/runtime/stream.h>

#include <tvm/script/printer/printer.h>

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

[[nodiscard]] bool uses_auxiliary_input(std::string_view operation) noexcept {
    return operation == "gemm" || operation == "add" || operation == "gelu_add" ||
           operation == "rmsnorm" || operation == "layernorm" ||
           operation == "residual_layernorm";
}

[[nodiscard]] bool uses_label_input(std::string_view operation) noexcept {
    return operation == "cross_entropy";
}

[[nodiscard]] bool has_row_output(std::string_view operation) noexcept {
    return operation == "sum" || operation == "cross_entropy";
}

[[nodiscard]] int64_t auxiliary_input_rows(
    std::string_view operation, const Configuration &cfg) noexcept {
    if (operation == "gemm") { return cfg.k; }
    if (operation == "rmsnorm") { return 1; }
    if (operation == "layernorm") { return 2; }
    return cfg.m;
}

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
        auto cost_basis = plan.elementwise_elements_per_program != 0u ? "fused_element_grid_v1" :
                          plan.reduction_subgroups_per_program != 0u ?
                                                                        "metal_subgroup_reduction_v1" :
                          plan.cost_basis == bridge::tirx::MatrixCostBasis::METAL_MPP_MEMORY ?
                                                                        "metal_mpp_memory_v2" :
                                                                        "simdgroup_reference_geometry";
        std::cout << separator << "{\"threads\":" << plan.threads
                  << ",\"metal_mpp\":" << (plan.metal_mpp ? "true" : "false")
                  << ",\"cost_basis\":" << std::quoted(cost_basis)
                  << ",\"programs\":" << plan.programs
                  << ",\"shared_memory_bytes\":" << plan.shared_memory_bytes
                  << ",\"optimized\":" << (plan.optimized ? "true" : "false")
                  << ",\"candidates_considered\":" << plan.candidates_considered
                  << ",\"candidates_rejected\":" << plan.candidates_rejected
                  << ",\"max_copy_batch\":" << plan.max_copy_batch
                  << ",\"batched_copy_operations\":" << plan.batched_copy_operations
                  << ",\"prefetched_pipeline_loops\":" << plan.prefetched_pipeline_loops
                  << ",\"prefetch_storage_scalars_per_lane\":" << plan.prefetch_storage_scalars_per_lane
                  << ",\"reduction_subgroups_per_program\":" << plan.reduction_subgroups_per_program
                  << ",\"reduction_programs_per_group\":" << plan.reduction_programs_per_group
                  << ",\"reduction_unroll_factor\":" << plan.reduction_unroll_factor
                  << ",\"reduction_lane_elements\":" << plan.reduction_lane_elements
                  << ",\"reduction_threadgroups\":" << plan.reduction_threadgroups
                  << ",\"reduction_scalar_rounds\":" << plan.reduction_scalar_rounds
                  << ",\"reduction_lane_utilization\":" << plan.reduction_lane_utilization
                  << ",\"striped_storage_scalars_per_worker\":" << plan.striped_storage_scalars_per_worker
                  << ",\"reduction_operations\":" << plan.reduction_operations
                  << ",\"reduction_elements\":" << plan.reduction_elements
                  << ",\"elementwise_elements_per_program\":" << plan.elementwise_elements_per_program
                  << ",\"elementwise_scalar_temporaries\":" << plan.elementwise_scalar_temporaries
                  << ",\"group_barrier_sites_before\":" << plan.group_barrier_sites_before
                  << ",\"group_barrier_sites_after\":" << plan.group_barrier_sites_after
                  << ",\"independent_subgroups\":" << (plan.independent_subgroups ? "true" : "false")
                  << ",\"normalized_cost\":" << plan.cost.score
                  << ",\"normalized_kernel_cost\":" << plan.cost.kernel_score
                  << ",\"matrix_issues\":" << plan.cost.matrix_issues
                  << ",\"shared_fragment_transfers\":" << plan.cost.shared_fragment_transfers
                  << ",\"direct_fragment_stores\":" << plan.cost.direct_fragment_stores
                  << ",\"metal_mpp_operations\":" << plan.cost.metal_mpp_operations
                  << ",\"memory_fragment_reads\":" << plan.cost.memory_fragment_reads
                  << ",\"lhs_footprint_fragments\":" << plan.cost.lhs_footprint_fragments
                  << ",\"rhs_footprint_fragments\":" << plan.cost.rhs_footprint_fragments
                  << ",\"accumulator_initializations\":" << plan.cost.accumulator_initializations
                  << ",\"tile_aspect_fragments\":" << plan.cost.tile_aspect_fragments
                  << ",\"local_row_aspect_issues\":" << plan.cost.local_row_aspect_issues
                  << ",\"local_column_aspect_issues\":" << plan.cost.local_column_aspect_issues
                  << ",\"independent_elements\":" << plan.cost.independent_elements
                  << ",\"fragment_scalars_per_lane\":" << plan.cost.fragment_scalars_per_lane
                  << ",\"concurrent_waves\":" << plan.cost.concurrent_waves
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

void dump_tile_ir(std::ostream &out, const Region &region, uint32_t depth = 0u) {
    auto indent = std::string(depth * 2u, ' ');
    for (auto block : region.blocks()) {
        out << indent << "block(";
        for (auto i = 0u; i < block->argument_count(); i++) {
            if (i != 0u) { out << ','; }
            out << '%' << block->argument(i)->id();
        }
        out << ")\n";
        for (auto operation : block->operations()) {
            out << indent << "  #" << operation->id() << ' ' << operation->name() << " (";
            for (auto i = 0u; i < operation->operand_count(); i++) {
                if (i != 0u) { out << ','; }
                out << '%' << operation->operand(i)->id();
            }
            out << ") -> (";
            for (auto i = 0u; i < operation->result_count(); i++) {
                if (i != 0u) { out << ','; }
                out << '%' << operation->result(i)->id();
            }
            out << ')';
            if (operation->domain()) {
                out << " domain[";
                for (auto i = 0u; i < operation->domain()->rank(); i++) {
                    if (i != 0u) { out << ','; }
                    auto &extent = operation->domain()->axis(i).extent;
                    if (extent.is_constant()) {
                        out << extent.constant_value();
                    } else {
                        out << '?';
                    }
                }
                out << ']';
            }
            out << '\n';
            for (auto &child : operation->regions()) { dump_tile_ir(out, *child, depth + 2u); }
        }
    }
}

// Count actual static call sites in the generated Metal source, not merely
// semantic MMA operations or a requested compiler capability.
[[nodiscard]] size_t matrix_intrinsics(const tvm::ffi::Module &module, bool mpp = false) {
    auto count = size_t{0u};
    if (std::string_view{module->kind()} == "metal") {
        auto source = module->InspectSource("metal");
        auto code = std::string_view{source.data(), source.size()};
        auto call = mpp ? std::string_view{"{}.run("} : std::string_view{"simdgroup_multiply_accumulate("};
        for (auto position = code.find(call); position != std::string_view::npos; position = code.find(call, position + call.size())) { count++; }
    }
    for (auto &&child : module->imports()) { count += matrix_intrinsics(child.cast<tvm::ffi::Module>(), mpp); }
    return count;
}

[[nodiscard]] size_t external_matrix_calls(const tvm::ffi::Module &module) {
    auto count = size_t{0u};
    if (std::string_view{module->kind()} == "llvm") {
        auto source = module->InspectSource("ll");
        auto code = std::string_view{source.data(), source.size()};
        constexpr auto provider = std::string_view{"tvm.contrib.cblas.matmul"};
        // LowerTVMBuiltin emits several references to one cached provider
        // symbol. Report the single semantic call site, not textual uses.
        count += code.find(provider) != std::string_view::npos;
    }
    for (auto &&child : module->imports()) {
        count += external_matrix_calls(child.cast<tvm::ffi::Module>());
    }
    return count;
}

[[nodiscard]] size_t external_vector_math_calls(const tvm::ffi::Module &module) {
    auto count = size_t{0u};
    if (std::string_view{module->kind()} == "llvm") {
        auto source = module->InspectSource("ll");
        auto code = std::string_view{source.data(), source.size()};
        // Report semantic static call sites; declarations do not contain the
        // direct-call spelling used by LLVM instructions.
        constexpr auto call = std::string_view{"call void @luisa_tile_accelerate_"};
        for (auto position = code.find(call); position != std::string_view::npos;
             position = code.find(call, position + call.size())) { count++; }
    }
    for (auto &&child : module->imports()) {
        count += external_vector_math_calls(child.cast<tvm::ffi::Module>());
    }
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
    if (operation == "add" || operation == "gelu_add") {
        auto definition = tile_kernel(operation == "add" ? "benchmark_add" : "benchmark_gelu_add", [=](TensorView<const float, 2> A,
                                                                                                       TensorView<const float, 2> B,
                                                                                                       TensorView<float, 2> C) {
            auto gm = axis("block_m", ceil_div(A.extent<0>(), cfg.bm));
            auto gn = axis("block_n", ceil_div(A.extent<1>(), cfg.bn));
            auto m = axis("m", cfg.bm);
            auto n = axis("n", cfg.bn);
            for (auto &nest : parallel(shape(gm, gn), cfg.execution_scope)) {
                auto origin = coord(nest.index(gm) * cfg.bm, nest.index(gn) * cfg.bn);
                auto value = A[origin, shape(m, n)] + B[origin, shape(m, n)];
                if (operation == "gelu_add") {
                    C(origin, shape(m, n)).store(0.5f * value * (1.0f + tanh(0.7978845608f * (value + 0.044715f * value * value * value))));
                } else {
                    C(origin, shape(m, n)).store(value);
                }
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
    if (operation == "rmsnorm") {
        auto definition = tile_kernel("benchmark_rmsnorm", [=](TensorView<const float, 2> X,
                                                               TensorView<const float, 2> Gamma,
                                                               TensorView<float, 2> Y) {
            auto rows = axis("rows", X.extent<0>());
            auto m = axis("m", 1);
            auto n = axis("n", X.extent<1>());
            for (auto &nest : parallel(shape(rows), cfg.execution_scope)) {
                auto origin = coord(nest.index(), 0);
                auto x = X[origin, shape(m, n)];
                auto variance = reduce(x * x, n, add) / static_cast<float>(cfg.n);
                auto gamma = Gamma[coord(0, 0), shape(m, n)];
                Y(origin, shape(m, n)).store(x / sqrt(variance + 1e-5f) * gamma);
            }
        });
        return definition.capture(tensor_shape(cfg.m, cfg.n), tensor_shape(1, cfg.n), tensor_shape(cfg.m, cfg.n));
    }
    if (operation == "layernorm") {
        auto definition = tile_kernel("benchmark_layernorm", [=](TensorView<const float, 2> X,
                                                                 TensorView<const float, 2> Parameters,
                                                                 TensorView<float, 2> Y) {
            auto rows = axis("rows", X.extent<0>());
            auto m = axis("m", 1);
            auto n = axis("n", X.extent<1>());
            for (auto &nest : parallel(shape(rows), cfg.execution_scope)) {
                auto origin = coord(nest.index(), 0);
                auto x = X[origin, shape(m, n)];
                auto denominator = static_cast<float>(cfg.n);
                auto mean = reduce(x, n, add) / denominator;
                auto centered = x - mean;
                auto variance = reduce(centered * centered, n, add) / denominator;
                auto gamma = Parameters[coord(0, 0), shape(m, n)];
                auto beta = Parameters[coord(1, 0), shape(m, n)];
                Y(origin, shape(m, n)).store(centered / sqrt(variance + 1e-5f) * gamma + beta);
            }
        });
        return definition.capture(tensor_shape(cfg.m, cfg.n), tensor_shape(2, cfg.n), tensor_shape(cfg.m, cfg.n));
    }
    if (operation == "residual_layernorm") {
        auto definition = tile_kernel("benchmark_residual_layernorm", [=](TensorView<const float, 2> X,
                                                                          TensorView<const float, 2> Residual,
                                                                          TensorView<float, 2> Y) {
            auto rows = axis("rows", X.extent<0>());
            auto m = axis("m", 1);
            auto n = axis("n", X.extent<1>());
            for (auto &nest : parallel(shape(rows), cfg.execution_scope)) {
                auto origin = coord(nest.index(), 0);
                auto combined = X[origin, shape(m, n)] + Residual[origin, shape(m, n)];
                auto denominator = static_cast<float>(cfg.n);
                auto mean = reduce(combined, n, add) / denominator;
                auto centered = combined - mean;
                auto variance = reduce(centered * centered, n, add) / denominator;
                Y(origin, shape(m, n)).store(centered / sqrt(variance + 1e-5f));
            }
        });
        return definition.capture(tensor_shape(cfg.m, cfg.n), tensor_shape(cfg.m, cfg.n), tensor_shape(cfg.m, cfg.n));
    }
    if (operation == "cross_entropy") {
        auto definition = tile_kernel("benchmark_cross_entropy", [=](TensorView<const float, 2> Logits,
                                                                     TensorView<const int64_t, 1> Labels,
                                                                     TensorView<float, 1> Losses) {
            auto rows = axis("rows", Logits.extent<0>());
            auto m = axis("m", 1);
            auto n = axis("n", Logits.extent<1>());
            for (auto &nest : parallel(shape(rows), cfg.execution_scope)) {
                auto origin = coord(nest.index(), 0);
                auto logits = Logits[origin, shape(m, n)];
                auto label = Labels[coord(nest.index()), shape(m)];
                auto peak = reduce(logits, n, maximum);
                auto total = reduce(exp(logits - peak), n, add);
                auto selected = gather(logits, label, n);
                Losses(coord(nest.index()), shape(m)).store(luisa::compute::tile::log(total) + peak - selected);
            }
        });
        return definition.capture(tensor_shape(cfg.m, cfg.n), tensor_shape(cfg.m), tensor_shape(cfg.m));
    }
    throw std::invalid_argument{"operation must be gemm, add, gelu_add, sum, softmax, rmsnorm, layernorm, residual_layernorm, or cross_entropy"};
}

[[nodiscard]] luisa::vector<float> input_values(size_t count, uint64_t seed) {
    luisa::vector<float> result(count);
    for (auto i = 0u; i < count; i++) {
        result[i] = static_cast<float>(static_cast<int64_t>((i * seed + 17u) % 127u) - 63) / 64.0f;
    }
    return result;
}

[[nodiscard]] luisa::vector<int64_t> label_values(int64_t rows, int64_t columns) {
    luisa::vector<int64_t> result(static_cast<size_t>(rows));
    for (auto row = int64_t{0}; row < rows; row++) {
        result[static_cast<size_t>(row)] = (row * 13 + 7) % columns;
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

// Same capture() and bridge options as the TVM-runtime path. Only the native
// compilation policy / Runtime binding and submission path change here.
void run_luisa(const char *program, const char *output_path, std::string_view operation, Configuration cfg,
               int64_t sample_count, int64_t target_ms, int64_t warmup_ms,
               const bridge::tirx::CompileOptions &options, bool fast_math) {
    using namespace luisa::compute;
    luisa::log_level_error();
    auto start = Clock::now();
    Context context{program};
    auto device = context.create_device("metal");
    auto stream = device.create_stream(StreamTag::COMPUTE);
    auto runtime_ms = milliseconds(start);
    start = Clock::now();
    auto kernel = capture(operation, cfg);
    auto capture_ms = milliseconds(start);
    start = Clock::now();
    auto shader = tile::compile(device, kernel,
                                {.threads_per_group = options.planner.threads_per_group, .lowering = Lowering::TIRX, .tirx = &options},
                                {.enable_fast_math = fast_math});
    auto compile_ms = milliseconds(start);
    if (!shader) { throw std::runtime_error{shader.metadata().error.c_str()}; }
    if (auto path = std::getenv("LUISA_TILE_BENCH_DUMP_SOURCE")) {
        if (std::filesystem::exists(path)) { throw std::runtime_error{"source dump path already exists"}; }
        std::ofstream file{path};
        file << shader.metadata().source;
        if (!file) { throw std::runtime_error{"cannot write generated source"}; }
    }
    auto columns_a = operation == "gemm" ? cfg.k : cfg.n;
    auto rows_b = auxiliary_input_rows(operation, cfg);
    auto binary = uses_auxiliary_input(operation);
    auto labeled = uses_label_input(operation);
    auto host_a = input_values(cfg.m * columns_a, 5);
    auto host_b = input_values(binary ? rows_b * cfg.n : 1, 11);
    auto host_labels = label_values(cfg.m, cfg.n);
    auto output_count = has_row_output(operation) ? cfg.m : cfg.m * cfg.n;
    luisa::vector<float> output(output_count, std::numeric_limits<float>::quiet_NaN());
    start = Clock::now();
    auto a = device.create_buffer<float>(host_a.size());
    auto b = device.create_buffer<float>(host_b.size());
    auto labels = device.create_buffer<int64_t>(host_labels.size());
    auto c = device.create_buffer<float>(output.size());
    stream << a.copy_from(host_a.data()) << b.copy_from(host_b.data())
           << labels.copy_from(host_labels.data()) << c.copy_from(output.data())
           << synchronize();
    auto upload_ms = milliseconds(start);
    auto submit = [&](uint64_t repetitions) {
        CommandList commands;
        for (auto i = uint64_t{0}; i < repetitions; i++) {
            if (labeled) {
                commands << shader(a, labels, c).dispatch();
            } else if (binary) {
                commands << shader(a, b, c).dispatch();
            } else {
                commands << shader(a, c).dispatch();
            }
        }
        stream << commands.commit() << synchronize();
    };
    auto invoke = [&](uint64_t repetitions) {
        stream.synchronize();
        auto before = Clock::now();
        submit(repetitions);
        return milliseconds(before);
    };
    auto cold_ms = invoke(1);
    start = Clock::now();
    while (milliseconds(start) < warmup_ms) { static_cast<void>(invoke(8)); }
    auto actual_warmup_ms = milliseconds(start);
    uint64_t repetitions = 1;
    for (auto attempt = 0; attempt < 8; attempt++) {
        auto elapsed = invoke(repetitions);
        if (elapsed >= target_ms * .8 || repetitions == 100000u) { break; }
        auto estimate = repetitions * static_cast<double>(target_ms) / std::max(elapsed, 1e-6);
        repetitions = std::clamp<uint64_t>(static_cast<uint64_t>(estimate), repetitions + 1, 100000);
    }
    std::vector<double> throughput, latency;
    for (auto i = 0; i < sample_count; i++) { throughput.emplace_back(1000.0 * invoke(repetitions) / repetitions); }
    for (auto i = 0; i < sample_count; i++) { latency.emplace_back(1000.0 * invoke(1)); }
    luisa::test::MetalBenchmarkTiming device_timing{true};
    device_timing.measure([&] { stream.synchronize(); }, submit, repetitions, static_cast<uint32_t>(sample_count));
    start = Clock::now();
    stream << c.copy_to(output.data()) << synchronize();
    auto download_ms = milliseconds(start);
    std::ofstream file{output_path, std::ios::binary};
    file.write(reinterpret_cast<const char *>(output.data()), static_cast<std::streamsize>(output.size() * sizeof(float)));
    if (!file) { throw std::runtime_error{"cannot write output"}; }
    auto matrix_calls = size_t{0};
    auto mpp_calls = size_t{0};
    constexpr auto call = std::string_view{"simdgroup_multiply_accumulate("};
    auto &source = shader.metadata().source;
    for (auto pos = source.find(call); pos != std::string::npos; pos = source.find(call, pos + call.size())) { matrix_calls++; }
    constexpr auto mpp_call = std::string_view{"{}.run("};
    for (auto pos = source.find(mpp_call); pos != std::string::npos; pos = source.find(mpp_call, pos + mpp_call.size())) { mpp_calls++; }
    std::cout << std::setprecision(12)
              << "{\"backend\":\"metal\",\"runtime\":\"luisa\",\"timing\":\"synchronized_host_wall\","
                 "\"batch_policy\":\"one_runtime_command_list_per_batch\",\"operation\":"
              << std::quoted(operation)
              << ",\"fast_math\":" << (fast_math ? "true" : "false")
              << ",\"execution_scope\":" << std::quoted(cfg.execution_scope == exec::Scope::GROUP ? "group" : cfg.execution_scope == exec::Scope::WORKER ? "worker" :
                                                                                                                                                           "auto")
              << ",\"pipeline_window\":" << cfg.pipeline_window
              << ",\"cooperative_matrix\":" << (options.cooperative_matrix ? "true" : "false")
              << ",\"metal_mpp\":" << (options.metal_mpp ? "true" : "false")
              << ",\"metal_subgroup_reductions\":" << (options.planner.metal_subgroup_reductions ? "true" : "false")
              << ",\"shared_tile_materialization\":\"preserve\""
              << ",\"forward_readonly_tile_loads\":" << (options.forward_readonly_tile_loads ? "true" : "false")
              << ",\"elide_independent_subgroup_barriers\":" << (options.planner.elide_independent_subgroup_barriers ? "true" : "false")
              << ",\"vectorize\":" << (options.vectorize ? "true" : "false")
              << ",\"auto_vectorize\":" << (options.auto_vectorize ? "true" : "false")
              << ",\"max_reduction_striped_scalars_per_worker\":" << options.planner.max_reduction_striped_scalars_per_worker
              << ",\"planner_threads\":" << options.planner.threads_per_group << ",\"copy_batch\":" << options.planner.max_copy_batch
              << ",\"realized_threads\":" << shader.block_size().x * shader.block_size().y * shader.block_size().z
              << ",\"matrix_intrinsics\":" << matrix_calls + mpp_calls
              << ",\"simdgroup_intrinsics\":" << matrix_calls << ",\"mpp_intrinsics\":" << mpp_calls
              << ",\"output_elements\":" << output_count
              << ",\"mma_operations\":" << luisa::test::tile_tirx::count_operations(kernel.function().body(), OperationKind::MMA)
              << ",\"runtime_init_ms\":" << runtime_ms << ",\"capture_ms\":" << capture_ms << ",\"compile_ms\":" << compile_ms
              << ",\"allocation_upload_ms\":" << upload_ms << ",\"cold_call_ms\":" << cold_ms << ",\"warmup_ms\":" << actual_warmup_ms
              << ",\"download_ms\":" << download_ms << ",\"repetitions\":" << repetitions
              << ",\"realization\":" << std::quoted(shader.metadata().realization)
              << ",\"source_hash\":" << std::quoted(std::to_string(luisa::hash<luisa::string_view>{}(source))) << ',';
    print_samples("throughput_us", throughput);
    std::cout << ',';
    print_samples("latency_us", latency);
    device_timing.print();
    std::cout << "}\n";
}

}// namespace

int main(int argc, char *argv[]) {
    if (argc < 13 || argc > 32) {
        std::cerr << "Usage: benchmark_tile_tirx <cpu|metal> <gemm|add|gelu_add|sum|softmax|rmsnorm|layernorm|residual_layernorm|cross_entropy> M N K BM BN BK samples sample-ms warmup-ms output.f32 [auto|worker|group] [pipeline-window:1|2] [scalar|subgroup-reduce|matrix|mpp|mpp-views] [vectorize|no-vectorize|auto-vectorize] [group-threads:auto|N] [copy-batch:1..16] [tvm|luisa|luisa-fast] [retain-subgroup-fences|elide-subgroup-fences] [cpu-stack-bytes:0..65536] [cpu-vector-lanes:16|32|64|128] [retain-input-snapshots|forward-input-views] [cpu-model:generic|native] [cpu-matrix:reference|cblas] [cpu-math:reference|accelerate] [shared-tiles:preserve|expensive-only]\n";
        std::cerr << "Additional mapping options: [reduction-programs:auto|1..8] [element-grid:auto|reference] [reduction-unroll:1..16] [reduction-lane-elements:1|2|4|8]\n";
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
        if (matrix_mode != "scalar" && matrix_mode != "subgroup-reduce" && matrix_mode != "matrix" && matrix_mode != "mpp" && matrix_mode != "mpp-views") { throw std::invalid_argument{"realization mode must be scalar, subgroup-reduce, matrix, mpp, or mpp-views"}; }
        auto metal_subgroup_reductions = matrix_mode == "subgroup-reduce";
        auto cooperative_matrix = matrix_mode == "matrix" || matrix_mode == "mpp" || matrix_mode == "mpp-views";
        auto forward_readonly_tile_loads = matrix_mode == "mpp-views" || metal_subgroup_reductions;
        auto metal_mpp = matrix_mode == "mpp" || matrix_mode == "mpp-views";
        if (argc >= 24) {
            auto policy = std::string_view{argv[23]};
            if (policy != "retain-input-snapshots" && policy != "forward-input-views") {
                throw std::invalid_argument{"unknown input snapshot policy"};
            }
            if (policy == "forward-input-views") {
                forward_readonly_tile_loads = true;
            }
        }
        if (metal_mpp && (backend != "metal" || operation != "gemm" || cfg.execution_scope != exec::Scope::GROUP)) {
            throw std::invalid_argument{"MPP benchmarking requires Metal group GEMM"};
        }
        if (metal_subgroup_reductions &&
            (backend != "metal" || cfg.execution_scope != exec::Scope::AUTOMATIC ||
             (operation != "sum" && operation != "softmax" && operation != "rmsnorm" && operation != "layernorm" && operation != "residual_layernorm" && operation != "cross_entropy"))) {
            throw std::invalid_argument{"SIMD-group reduction benchmarking requires automatic Metal sum, softmax, RMSNorm, residual LayerNorm, LayerNorm, or cross-entropy"};
        }
        auto vector_mode = argc >= 17 ? std::string_view{argv[16]} : std::string_view{"vectorize"};
        if (vector_mode != "vectorize" && vector_mode != "no-vectorize" && vector_mode != "auto-vectorize") { throw std::invalid_argument{"vector mode must be vectorize, no-vectorize, or auto-vectorize"}; }
        auto vectorize = vector_mode != "no-vectorize";
        auto auto_vectorize = vector_mode == "auto-vectorize";
        bridge::tirx::PlannerOptions planner;
        planner.metal_subgroup_reductions = metal_subgroup_reductions;
        if (argc >= 32) {
            auto width = positive_integer(argv[31]);
            if ((width != 1 && width != 2 && width != 4 && width != 8) ||
                (width != 1 && !metal_subgroup_reductions)) {
                throw std::invalid_argument{"reduction lane elements require 1, 2, 4 or 8 and subgroup-reduce when non-default"};
            }
            planner.reduction_lane_elements = static_cast<uint32_t>(width);
        }
        if (argc >= 31) {
            auto factor = positive_integer(argv[30]);
            if (factor > 16 || (factor != 1 && !metal_subgroup_reductions)) {
                throw std::invalid_argument{"reduction unrolling requires a factor in [1,16] and subgroup-reduce when non-default"};
            }
            planner.reduction_unroll_factor = static_cast<uint32_t>(factor);
        }
        if (argc >= 29 && std::string_view{argv[28]} != "auto") {
            auto programs = positive_integer(argv[28]);
            if (!metal_subgroup_reductions || programs > 8) {
                throw std::invalid_argument{"reduction packing requires subgroup-reduce and 1..8 programs"};
            }
            planner.reduction_programs_per_group = static_cast<uint32_t>(programs);
        }
        if (argc >= 30) {
            auto mapping = std::string_view{argv[29]};
            if (mapping != "auto" && mapping != "reference") {
                throw std::invalid_argument{"element grid must be auto or reference"};
            }
            planner.fuse_gpu_elementwise = mapping == "auto";
        }
        if (argc >= 18 && std::string_view{argv[17]} != "auto") {
            auto requested = positive_integer(argv[17]);
            if (requested > std::numeric_limits<uint32_t>::max() || backend != "metal" ||
                (cfg.execution_scope != exec::Scope::GROUP && !metal_subgroup_reductions)) {
                throw std::invalid_argument{"explicit group threads require Metal group execution or the subgroup-reduction planner"};
            }
            planner.threads_per_group = static_cast<uint32_t>(requested);
        }
        if (argc >= 19) {
            auto requested = positive_integer(argv[18]);
            if (requested > 16u || (requested != 1u && (backend != "metal" || cfg.execution_scope != exec::Scope::GROUP))) {
                throw std::invalid_argument{"copy batching requires a value in [1,16] and Metal group execution"};
            }
            planner.max_copy_batch = static_cast<uint32_t>(requested);
        }
        auto sample_count = positive_integer(argv[9]);
        auto target_ms = positive_integer(argv[10]);
        auto warmup_ms = positive_integer(argv[11]);
        if (cfg.m > 16384 || cfg.n > 16384 || cfg.k > 16384 ||
            cfg.bm > 512 || cfg.bn > 16384 || cfg.bk > 16384 || sample_count > 101) {
            throw std::invalid_argument{"benchmark dimensions or sample count exceed the supported limit"};
        }
        if (std::filesystem::exists(argv[12])) { throw std::invalid_argument{"output path already exists"}; }
        auto runtime_choice = argc >= 20 ? std::string_view{argv[19]} : "tvm";
        auto cpu_model = argc >= 25 ? std::string_view{argv[24]} : "generic";
        if ((cpu_model != "generic" && cpu_model != "native") ||
            (cpu_model == "native" && (backend != "cpu" || runtime_choice != "tvm"))) {
            throw std::invalid_argument{"CPU model must be generic; native requires the CPU TVM runtime"};
        }
        auto cpu_matrix_name = argc >= 26 ? std::string_view{argv[25]} : "reference";
        auto cpu_matrix_backend = bridge::tirx::CpuMatrixBackend::REFERENCE;
        if (cpu_matrix_name == "cblas") {
            if (backend != "cpu" || runtime_choice != "tvm") {
                throw std::invalid_argument{"CBLAS realization requires the CPU TVM runtime"};
            }
            cpu_matrix_backend = bridge::tirx::CpuMatrixBackend::CBLAS;
        } else if (cpu_matrix_name != "reference") {
            throw std::invalid_argument{"CPU matrix realization must be reference or cblas"};
        }
        auto cpu_math_name = argc >= 27 ? std::string_view{argv[26]} : "reference";
        auto cpu_math_backend = bridge::tirx::CpuMathBackend::REFERENCE;
        if (cpu_math_name == "accelerate") {
            if (backend != "cpu" || runtime_choice != "tvm") {
                throw std::invalid_argument{"Accelerate array math requires the CPU TVM runtime"};
            }
            cpu_math_backend = bridge::tirx::CpuMathBackend::ACCELERATE;
        } else if (cpu_math_name != "reference") {
            throw std::invalid_argument{"CPU array-math realization must be reference or accelerate"};
        }
        auto shared_tiles_name = argc >= 28 ? std::string_view{argv[27]} : "preserve";
        auto lower_options = bridge::tirx::LowerOptions{};
        if (shared_tiles_name == "expensive-only") {
            lower_options.shared_tiles = bridge::tirx::SharedTileMaterialization::EXPENSIVE_ONLY;
        } else if (shared_tiles_name != "preserve") {
            throw std::invalid_argument{"shared-Tile materialization must be preserve or expensive-only"};
        }
        if (argc >= 23) {
            auto lanes = positive_integer(argv[22]);
            if ((lanes != 16 && lanes != 32 && lanes != 64 && lanes != 128) ||
                (lanes != 16 && (backend != "cpu" || !auto_vectorize))) {
                throw std::invalid_argument{"CPU vector lanes require 16/32/64/128 and CPU auto-vectorization when non-default"};
            }
            planner.max_cpu_vector_lanes = static_cast<uint32_t>(lanes);
        }
        if (argc >= 22) {
            auto budget = std::string_view{argv[21]} == "0" ? 0 : positive_integer(argv[21]);
            if (budget > 65536 || (budget != 0 && backend != "cpu")) {
                throw std::invalid_argument{"CPU stack budget must be in [0,65536] and requires the CPU backend"};
            }
            planner.max_cpu_stack_bytes = static_cast<uint32_t>(budget);
        }
        if (argc >= 21) {
            auto policy = std::string_view{argv[20]};
            if (policy != "retain-subgroup-fences" && policy != "elide-subgroup-fences") {
                throw std::invalid_argument{"unknown subgroup-fence policy"};
            }
            planner.elide_independent_subgroup_barriers = policy == "elide-subgroup-fences";
            if (planner.elide_independent_subgroup_barriers && !forward_readonly_tile_loads) {
                throw std::invalid_argument{"subgroup-fence elision currently requires mpp-views"};
            }
        }
        if (runtime_choice != "tvm") {
            if (backend != "metal" || (runtime_choice != "luisa" && runtime_choice != "luisa-fast")) {
                throw std::invalid_argument{"Runtime must be tvm, or luisa/luisa-fast on Metal"};
            }
            if (lower_options.shared_tiles != bridge::tirx::SharedTileMaterialization::PRESERVE) {
                throw std::invalid_argument{"the Luisa Runtime benchmark currently requires preserved shared Tile SSA"};
            }
            bridge::tirx::CompileOptions options;
            options.noalias = true;
            options.cooperative_matrix = cooperative_matrix;
            options.metal_mpp = metal_mpp;
            options.forward_readonly_tile_loads = forward_readonly_tile_loads;
            options.vectorize = vectorize;
            options.auto_vectorize = auto_vectorize;
            options.planner = planner;
            run_luisa(argv[0], argv[12], operation, cfg, sample_count, target_ms, warmup_ms, options, runtime_choice == "luisa-fast");
            return 0;
        }
        auto start = Clock::now();
        Runtime runtime{backend, cpu_model == "native"};
        auto runtime_init_ms = milliseconds(start);
        start = Clock::now();
        auto kernel = capture(operation, cfg);
        auto capture_ms = milliseconds(start);
        if (auto path = std::getenv("LUISA_TILE_BENCH_DUMP_TILE_IR")) {
            if (std::filesystem::exists(path)) { throw std::runtime_error{"TileIR dump path already exists"}; }
            std::ofstream file{path};
            dump_tile_ir(file, kernel.function().body());
            if (!file) { throw std::runtime_error{"cannot write TileIR dump"}; }
        }
        if (auto path = std::getenv("LUISA_TILE_BENCH_DUMP_TIRX")) {
            if (std::filesystem::exists(path)) { throw std::runtime_error{"TIRx dump path already exists"}; }
            auto native = bridge::tirx::lower(kernel.function(), lower_options);
            if (!native) { throw std::runtime_error{native.error.c_str()}; }
            std::ofstream file{path};
            file << tvm::Script(native.value);
            if (!file) { throw std::runtime_error{"cannot write native TIRx dump"}; }
        }
        start = Clock::now();
        auto executable = runtime.build(kernel, true, cooperative_matrix, vectorize, auto_vectorize, planner, metal_mpp, forward_readonly_tile_loads, cpu_matrix_backend, cpu_math_backend, lower_options);
        auto compile_ms = milliseconds(start);
        if (!executable.ok()) { throw std::runtime_error{executable.error.c_str()}; }
        auto matrix_calls = matrix_intrinsics(executable.module.value());
        auto mpp_calls = matrix_intrinsics(executable.module.value(), true);
        auto library_matrix_calls = external_matrix_calls(executable.module.value());
        auto library_vector_math_calls = external_vector_math_calls(executable.module.value());
        if (auto path = std::getenv("LUISA_TILE_BENCH_DUMP_SOURCE")) {
            dump_source(executable.module.value(), backend == "metal" ? "metal" : "llvm", path);
            if (!std::filesystem::exists(path)) { throw std::runtime_error{"requested generated source is unavailable"}; }
        }
        auto columns_a = operation == "gemm" ? cfg.k : cfg.n;
        auto rows_b = auxiliary_input_rows(operation, cfg);
        auto binary = uses_auxiliary_input(operation);
        auto labeled = uses_label_input(operation);
        auto host_a = input_values(static_cast<size_t>(cfg.m * columns_a), 5);
        auto host_b = input_values(binary ? static_cast<size_t>(rows_b * cfg.n) : 0u, 11);
        auto host_labels = label_values(cfg.m, cfg.n);
        start = Clock::now();
        auto a = runtime.upload<float>({cfg.m, columns_a}, host_a);
        tvm::runtime::Tensor b;
        if (binary) { b = runtime.upload<float>({rows_b, cfg.n}, host_b); }
        tvm::runtime::Tensor labels;
        if (labeled) { labels = runtime.upload<int64_t>({cfg.m}, host_labels); }
        auto out = has_row_output(operation) ? runtime.allocate<float>({cfg.m}) : runtime.allocate<float>({cfg.m, cfg.n});
        runtime.synchronize();
        auto allocation_upload_ms = milliseconds(start);
        std::function<void()> invoke = [&] {
            if (labeled) {
                (*executable.entry)(a, labels, out);
            } else if (binary) {
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
        luisa::test::MetalBenchmarkTiming device_timing{backend == "metal"};
        device_timing.measure([&] { runtime.synchronize(); }, [&](uint64_t count) {
            for (auto i = uint64_t{0u}; i < count; i++) { invoke(); }
            runtime.synchronize(); }, repetitions, static_cast<uint32_t>(sample_count));
        start = Clock::now();
        auto output_count = static_cast<size_t>(has_row_output(operation) ? cfg.m : cfg.m * cfg.n);
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
                  << ",\"metal_mpp\":" << (metal_mpp ? "true" : "false")
                  << ",\"metal_subgroup_reductions\":" << (metal_subgroup_reductions ? "true" : "false")
                  << ",\"metal_max_threads\":" << runtime.metal_max_threads()
                  << ",\"reduction_programs_per_group\":" << planner.reduction_programs_per_group
                  << ",\"reduction_unroll_factor\":" << planner.reduction_unroll_factor
                  << ",\"reduction_lane_elements\":" << planner.reduction_lane_elements
                  << ",\"fuse_gpu_elementwise\":" << (planner.fuse_gpu_elementwise ? "true" : "false")
                  << ",\"shared_tile_materialization\":" << std::quoted(shared_tiles_name)
                  << ",\"forward_readonly_tile_loads\":" << (forward_readonly_tile_loads ? "true" : "false")
                  << ",\"elide_independent_subgroup_barriers\":" << (planner.elide_independent_subgroup_barriers ? "true" : "false")
                  << ",\"vectorize\":" << (vectorize ? "true" : "false")
                  << ",\"auto_vectorize\":" << (auto_vectorize ? "true" : "false")
                  << ",\"cpu_stack_bytes\":" << planner.max_cpu_stack_bytes
                  << ",\"cpu_parallel_task_threshold\":" << planner.min_cpu_parallel_tasks
                  << ",\"cpu_vector_lanes\":" << planner.max_cpu_vector_lanes
                  << ",\"max_reduction_striped_scalars_per_worker\":" << planner.max_reduction_striped_scalars_per_worker
                  << ",\"cpu_target_policy\":" << std::quoted(cpu_model)
                  << ",\"cpu_model\":" << std::quoted(runtime.cpu_model())
                  << ",\"cpu_matrix_backend\":" << std::quoted(cpu_matrix_name)
                  << ",\"cpu_math_backend\":" << std::quoted(cpu_math_name)
                  << ",\"planner_threads\":" << planner.threads_per_group
                  << ",\"copy_batch\":" << planner.max_copy_batch
                  << ",\"matrix_intrinsics\":" << matrix_calls + mpp_calls
                  << ",\"external_matrix_calls\":" << library_matrix_calls
                  << ",\"external_vector_math_calls\":" << library_vector_math_calls
                  << ",\"simdgroup_intrinsics\":" << matrix_calls << ",\"mpp_intrinsics\":" << mpp_calls
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
        device_timing.print();
        std::cout << "}\n";
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 2;
    }
}

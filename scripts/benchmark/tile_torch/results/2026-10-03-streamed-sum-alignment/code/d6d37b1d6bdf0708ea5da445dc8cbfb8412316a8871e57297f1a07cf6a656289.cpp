// Actual Tile workload benchmark: exported fixtures, complete FP64/guard
// checks, explicit routes, host/event timing, and optional CUDA graph replay.
#include "tile_workload_test_utils.h"
#include "cuda_tile_streamed_sum_ir.h"
#include "diagnostic_alignment.h"
#include <luisa/core/platform.h>
#include <luisa/core/stl/filesystem.h>
#include <luisa/runtime/context.h>
#include <luisa/runtime/stream.h>
#include <luisa/tile/runtime.h>
#include <luisa/backends/ext/cuda/cuda_graph_ext.h>
#include <algorithm>
#include <charconv>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iterator>
#include <iostream>
#include <limits>
#include <sstream>
#include <system_error>

#if defined(LUISA_TILE_BENCH_CUDA_EVENTS) && LUISA_TILE_BENCH_CUDA_EVENTS
#include <cuda.h>
#endif

using namespace luisa;
using namespace luisa::compute;
namespace workloads = luisa::test::tile_workloads;
using Clock = std::chrono::steady_clock;

namespace {

void quoted(std::ostream &out, string_view text) {
    out << '"';
    for (auto c : text) {
        if (c == '"' || c == '\\') {
            out << '\\' << c;
        } else if (static_cast<uint8_t>(c) < 32u) {
            out << "\\u00" << "0123456789abcdef"[(c >> 4u) & 15u] << "0123456789abcdef"[c & 15u];
        } else {
            out << c;
        }
    }
    out << '"';
}

template<typename T>
void array(std::ostream &out, const T &values) {
    out << '[';
    auto separator = "";
    for (auto x : values) {
        out << separator << x;
        separator = ",";
    }
    out << ']';
}

[[nodiscard]] double elapsed(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>{Clock::now() - start}.count();
}

[[nodiscard]] double median(vector<double> values) {
    std::sort(values.begin(), values.end());
    return values[values.size() / 2u];
}

template<typename T>
[[nodiscard]] bool write_binary(const luisa::filesystem::path &path, const vector<T> &data) {
    std::ofstream file{path, std::ios::binary};
    file.write(reinterpret_cast<const char *>(data.data()), static_cast<std::streamsize>(data.size() * sizeof(T)));
    file.close();
    return static_cast<bool>(file);
}

[[nodiscard]] bool write_text(const luisa::filesystem::path &path, const std::string &text) {
    std::ofstream file{path, std::ios::binary};
    file.write(text.data(), static_cast<std::streamsize>(text.size()));
    file.close();
    return static_cast<bool>(file);
}

[[nodiscard]] bool integer(string_view text, uint64_t &value) {
    auto result = std::from_chars(text.data(), text.data() + text.size(), value);
    return result.ec == std::errc{} && result.ptr == text.data() + text.size();
}

[[nodiscard]] bool dimensions(string_view text, vector<int64_t> &output) {
    for (;;) {
        auto split = text.find(',');
        uint64_t value{};
        if (!integer(text.substr(0u, split), value) || value == 0u || value > 65536u) { return false; }
        output.emplace_back(static_cast<int64_t>(value));
        if (split == string_view::npos) { return true; }
        text.remove_prefix(split + 1u);
    }
}

void identity(std::ostream &out, const workloads::Options &o) {
    out << "\"schema\":1,\"backend\":";
    quoted(out, o.backend);
    out << ",\"lowering\":";
    quoted(out, o.lowering);
    out << ",\"operation\":";
    quoted(out, o.operation);
    out << ",\"precision\":";
    quoted(out, o.precision);
    out << ",\"seed\":" << o.seed << ",\"pattern\":";
    quoted(out, o.pattern);
    out << ",\"dimensions\":";
    array(out, o.dimensions);
    out << ",\"tile\":";
    array(out, o.tile);
    if (o.operation == "sort" || o.operation == "topk") {
        out << ",\"ranking_algorithm\":";
        quoted(out, o.ranking_algorithm);
    }
    out << ",\"fast_math\":" << (o.fast_math ? "true" : "false") << ",\"graph_batch\":" << o.graph_batch;
}

[[nodiscard]] int finish(const workloads::Options &o, const luisa::filesystem::path &directory,
                         string_view status, string_view reason, int code, double compile_ms = 0.0,
                         string_view realization = {}) {
    std::ostringstream out;
    out << std::setprecision(17) << '{';
    identity(out, o);
    out << ",\"status\":";
    quoted(out, status);
    out << ",\"reason\":";
    quoted(out, reason);
    out << ",\"compile_ms\":" << compile_ms << ",\"realization\":";
    quoted(out, realization);
    out << "}\n";
    if (!directory.empty() && !write_text(directory / "results.json", out.str())) {
        std::cerr << "Cannot write results.json\n";
        return 1;
    }
    std::cout << out.str();
    return code;
}

template<typename T>
[[nodiscard]] constexpr string_view storage_name() {
    if constexpr (std::is_same_v<T, half>) {
        return "float16";
    } else if constexpr (std::is_same_v<T, tile::bfloat16>) {
        return "bfloat16";
    } else {
        return "float32";
    }
}

template<typename T>
[[nodiscard]] constexpr const char *storage_suffix() {
    if constexpr (std::is_same_v<T, half>) {
        return ".f16";
    } else if constexpr (std::is_same_v<T, tile::bfloat16>) {
        return ".bf16";
    } else {
        return ".f32";
    }
}

void pipeline_metadata(std::ostream &out, const workloads::Fixture &f) {
    if (f.pipeline_widths.empty()) { return; }
    out << ",\"pipeline\":{\"kind\":\"chunked_bitonic_whole_tile_merge\",\"chunk\":" << f.pipeline_chunk
        << ",\"stages_per_operation\":" << f.pipeline_widths.size() << ",\"stage_widths\":";
    array(out, f.pipeline_widths);
    out << ",\"scratch_elements_per_plane\":" << f.scratch_elements
        << ",\"scratch_slots\":" << (f.scratch_elements == 0u ? 0u : 2u)
        << ",\"scratch_value_dtype\":\"float32\",\"scratch_index_dtype\":\"int32\","
           "\"scratch_guard_elements_per_plane\":128,\"final_indices_dtype\":\"int64\","
           "\"timing_unit\":\"one complete sort, all stages included\","
           "\"graph_ordering\":\"buffer hazard DAG; independent stages from adjacent sorts may overlap\"}";
}

template<typename T>
[[nodiscard]] bool export_fixture(const luisa::filesystem::path &directory,
                                  const workloads::Options &o, const workloads::Fixture &f) {
    for (size_t i = 0u; i < (f.input_ids ? 1u : 3u); i++) {
        vector<T> stored;
        stored.reserve(f.inputs[i].size());
        for (auto value : f.inputs[i]) { stored.emplace_back(T{value}); }
        if (!write_binary(directory / ("input" + std::to_string(i) + storage_suffix<T>()), stored)) { return false; }
    }
    if (f.input_ids && !write_binary(directory / "input1.i64", *f.input_ids)) { return false; }
    if (!write_binary(directory / "expected.f64", f.expected) || !write_binary(directory / "per_element_bound.f64", f.bound)) { return false; }
    if (f.ranking && !write_binary(directory / "expected_indices.i64", f.expected_indices)) { return false; }
    if (!f.strict_bound.empty() &&
        (!write_binary(directory / "strict_bound.f64", f.strict_bound) ||
         !write_binary(directory / "probability_rounding_bound.f64", f.probability_rounding_bound))) { return false; }
    std::ostringstream out;
    out << std::setprecision(17) << '{';
    identity(out, o);
    out << ",\"algorithm\":";
    quoted(out, f.algorithm);
    pipeline_metadata(out, f);
    out << ",\"endianness\":\"little\",\"inputs\":[";
    for (size_t i = 0; i < (f.input_ids ? 1u : 3u); i++) {
        if (i != 0u) { out << ','; }
        out << "{\"name\":\"input" << i << "\",\"path\":\"input" << i << storage_suffix<T>() << "\",\"storage_dtype\":";
        quoted(out, storage_name<T>());
        out << ",\"shape\":";
        array(out, f.input_shapes[i]);
        out << '}';
    }
    if (f.input_ids) {
        out << ",{\"name\":\"input1\",\"path\":\"input1.i64\",\"storage_dtype\":\"int64\",\"shape\":[" << f.input_ids->size() << "]}";
    }
    out << "],\"output\":{\"path\":\"output" << storage_suffix<T>() << "\",\"storage_dtype\":";
    quoted(out, storage_name<T>());
    out << ",\"shape\":";
    array(out, f.output_shape);
    out << "},\"expected\":{\"path\":\"expected.f64\",\"bound_path\":\"per_element_bound.f64\",\"storage_dtype\":\"float64\"";
    if (!f.strict_bound.empty()) {
        out << ",\"strict_bound_path\":\"strict_bound.f64\",\"probability_rounding_bound_path\":\"probability_rounding_bound.f64\"";
    }
    out << '}';
    if (f.ranking) {
        out << ",\"indices\":{\"path\":\"output_indices.i64\",\"expected_path\":\"expected_indices.i64\",\"storage_dtype\":\"int64\",\"shape\":";
        array(out, f.output_shape);
        out << '}';
    }
    if (!f.strict_bound.empty()) {
        auto unit = static_cast<double>(static_cast<float>(std::numeric_limits<T>::epsilon())) * .5;
        auto eta = static_cast<double>(static_cast<float>(std::numeric_limits<T>::denorm_min()));
        out << ",\"precision_contract\":{\"name\":\"attention_single_narrow_probability_v1\",\"acceptance_kind\":\"predeclared_numerical_envelope\","
               "\"stage\":\"unnormalized_probability_before_pv_per_kv_block\",\"rounding\":\"rne\","
               "\"fp32_base_envelope\":\"5e-5*(1+abs(reference))\","
               "\"probability_bound_formula\":\"((1+gamma_n)/(1-gamma_n))*(u_T*max_abs_V_j+eta_T/2*sum_abs_V_j)\","
               "\"gamma_formula\":\"n*u32/(1-n*u32), n=2*valid_keys+2\","
               "\"primary_bound_formula\":\"strict_bound+(1+u_T)*probability_rounding_bound\","
               "\"valid_keys\":\"K-Q+q+1\",\"u32\":"
            << 0x1p-24
            << ",\"u_T\":" << unit << ",\"eta_T\":" << eta
            << ",\"requirements\":\"One probability narrowing, FP32 scores/accumulation, positive normalization; unspecified extra narrowing or FTZ is not covered\"}";
    }
    out << ",\"semantics\":{\"epsilon\":" << static_cast<double>(1e-5f)
        << ",\"affine\":true,\"rope_pairing\":\"half_split\",\"softmax_axis\":-1,\"mask\":";
    quoted(out, o.operation == "masked_softmax" ? "column<=row%width" : "none");
    out << ",\"scan_policy\":";
    quoted(out, o.operation == "scan_ordered" ? "ordered_fold_left" : "unordered_tree");
    out << ",\"scan\":\"inclusive_sum\",\"descending\":true,"
           "\"stable\":true,\"tie_break\":\"original_index_ascending\",\"accumulation\":";
    quoted(out, f.input_ids ? "none" : "float32");
    out << ","
           "\"gelu_approximation\":\"tanh\",\"causal\":true,\"query_positions\":\"last_Q_in_K\","
           "\"attention_scale\":";
    auto scale = (o.operation == "attention" || o.operation == "attention_tensorcore") ? 1.0f / std::sqrt(static_cast<float>(o.dimensions[5])) : 1.0f;
    out << static_cast<double>(scale) << ",\"input_quantization\":\"round_to_nearest_even_before_fp64_oracle\",\"output_rounding\":\"round_to_nearest_even\",\"contraction\":";
    auto fused_contraction = o.operation == "gemm" || o.operation == "bmm" ||
                             o.operation == "attention" || o.operation == "attention_tensorcore";
    quoted(out, o.operation == "gemv" ? "nonfused_products_unordered_tree_sum" :
                fused_contraction     ? "mma_fused_reassociation_allowed" :
                                        "not_applicable");
    if (f.input_ids) {
        out << ",\"index_dtype\":\"int64\",\"index_bounds\":\"reject_invalid\",\"gather_axis\":0,\"value_preservation\":\"storage_bits\",\"index_view_offset_bytes\":" << 64u * sizeof(int64_t)
            << ",\"index_pattern\":";
        quoted(out, o.pattern == "adversarial" ? "repeated_boundary_rows" : "seeded_uniform_rows");
    }
    if (o.operation == "bmm") { out << ",\"batch_layout\":\"contiguous_bmk_bkn_bmn\",\"batch_broadcast\":false"; }
    out << ",\"logical_view_offset_elements\":64,\"logical_view_offset_bytes\":" << 64u * sizeof(T)
        << ",\"output_preallocated\":true,\"graph_contract\":\"Native: N complete identical workloads including every pipeline stage; same inputs/output/scratch allocations and hazard-DAG ordering. Independent pipeline stages may overlap across calls. R complete graph replays per timed sample, divided by N*R complete workloads; event-primed adaptive_replay_span_v2. Functional Torch output allocation is a separate contract.\"}"
           ",\"validation\":{\"reference\":\"host_fp64\",\"bound\":\"per_element_bound.f64\","
           "\"require_finite\":true,\"output_guard_elements\":128,\"readonly_inputs_checked\":true}}\n";
    return write_text(directory / "manifest.json", out.str());
}

class CudaTiming {
private:
#if defined(LUISA_TILE_BENCH_CUDA_EVENTS) && LUISA_TILE_BENCH_CUDA_EVENTS
    CUcontext _context{};
    CUstream _stream{};
    CUevent _start{}, _end{};
    [[nodiscard]] bool _check(CUresult value) {
        if (value == CUDA_SUCCESS) { return true; }
        const char *message = nullptr;
        static_cast<void>(cuGetErrorName(value, &message));
        error = message == nullptr ? "CUDA event API failed" : message;
        return false;
    }
#endif
public:
    bool enabled{false};
    string error;
    CudaTiming(Device &device, Stream &stream, bool requested) {
        if (!requested) { return; }
#if defined(LUISA_TILE_BENCH_CUDA_EVENTS) && LUISA_TILE_BENCH_CUDA_EVENTS
        _context = static_cast<CUcontext>(device.impl()->native_handle());
        _stream = static_cast<CUstream>(stream.native_handle());
        if (!_check(cuCtxPushCurrent(_context))) { return; }
        auto a = _check(cuEventCreate(&_start, CU_EVENT_DEFAULT));
        auto b = a && _check(cuEventCreate(&_end, CU_EVENT_DEFAULT));
        CUcontext previous{};
        auto c = _check(cuCtxPopCurrent(&previous));
        enabled = a && b && c;
#else
        static_cast<void>(device);
        static_cast<void>(stream);
        error = "benchmark was built without CUDA driver event support";
#endif
    }
    ~CudaTiming() noexcept {
#if defined(LUISA_TILE_BENCH_CUDA_EVENTS) && LUISA_TILE_BENCH_CUDA_EVENTS
        if (_context != nullptr && cuCtxPushCurrent(_context) == CUDA_SUCCESS) {
            if (_start != nullptr) { static_cast<void>(cuEventDestroy(_start)); }
            if (_end != nullptr) { static_cast<void>(cuEventDestroy(_end)); }
            CUcontext previous{};
            static_cast<void>(cuCtxPopCurrent(&previous));
        }
#endif
    }
    [[nodiscard]] bool record(bool start) {
#if defined(LUISA_TILE_BENCH_CUDA_EVENTS) && LUISA_TILE_BENCH_CUDA_EVENTS
        if (!enabled || !_check(cuCtxPushCurrent(_context))) { return false; }
        auto ok = _check(cuEventRecord(start ? _start : _end, _stream));
        CUcontext previous{};
        return _check(cuCtxPopCurrent(&previous)) && ok;
#else
        static_cast<void>(start);
        return false;
#endif
    }
    [[nodiscard]] double milliseconds() {
#if defined(LUISA_TILE_BENCH_CUDA_EVENTS) && LUISA_TILE_BENCH_CUDA_EVENTS
        if (!enabled || !_check(cuCtxPushCurrent(_context))) { return -1.0; }
        float ms{};
        auto ok = _check(cuEventElapsedTime(&ms, _start, _end));
        CUcontext previous{};
        ok = _check(cuCtxPopCurrent(&previous)) && ok;
        return ok ? static_cast<double>(ms) : -1.0;
#else
        return -1.0;
#endif
    }
};

}// namespace

template<typename T>
int run(int argc, char *argv[]) {
    workloads::Options options;
    if (argc < 13 || argc > 19 || (argc - 13) % 2 != 0) {
        std::cerr << "Usage: benchmark_tile_workloads <cuda|simd> <native|tirx> operation <fp32|fp16|bf16> dimensions_csv tile_m,tile_n,tile_k seed <random|cancellation|adversarial> samples sample_ms warmup_ms export_dir [--graph-batch N] [--ranking-algorithm full_sort_prefix|packed_fp32|repeated_extrema|chunked_bitonic_c256|chunked_bitonic_c512] [--fast-math 0|1]\n";
        std::cerr << "BMM operation dimensions are B,M,N,K; tile is BM,BN,BK.\n";
        std::cerr << "Embedding dimensions are V,D,T; tile is 1,feature_width,1, with true INT64 row IDs.\n";
        return finish(options, {}, "failed", "invalid argument count", 1);
    }
    options.backend = argv[1];
    options.lowering = argv[2];
    options.operation = argv[3];
    options.precision = argv[4];
    options.pattern = argv[8];
    vector<int64_t> schedule;
    uint64_t samples{}, sample_ms{}, warmup_ms{}, graph_batch{};
    if ((options.backend != "cuda" && options.backend != "simd") ||
        (options.lowering != "native" && options.lowering != "tirx") ||
        (options.pattern != "random" && options.pattern != "cancellation" && options.pattern != "adversarial") ||
        !dimensions(argv[5], options.dimensions) || !dimensions(argv[6], schedule) || schedule.size() != 3u ||
        !integer(argv[7], options.seed) || !integer(argv[9], samples) || samples < 1u || samples > 101u || samples % 2u == 0u ||
        !integer(argv[10], sample_ms) || sample_ms < 1u || sample_ms > 10000u ||
        !integer(argv[11], warmup_ms) || warmup_ms < 1u || warmup_ms > 60000u) {
        return finish(options, {}, "failed", "invalid operation arguments, dimensions, schedule or timing limits", 1);
    }
    auto graph_seen = false, ranking_seen = false, fast_seen = false;
    for (auto i = 13; i < argc; i += 2) {
        auto flag = string_view{argv[i]};
        if (flag == "--graph-batch" && !graph_seen) {
            graph_seen = true;
            if (!integer(argv[i + 1], graph_batch) || graph_batch == 0u || graph_batch > 100000u) {
                return finish(options, {}, "failed", "invalid --graph-batch", 1);
            }
        } else if (flag == "--ranking-algorithm" && !ranking_seen) {
            ranking_seen = true;
            options.ranking_algorithm = argv[i + 1];
            if (options.ranking_algorithm != "full_sort_prefix" && options.ranking_algorithm != "packed_fp32" && options.ranking_algorithm != "repeated_extrema" &&
                options.ranking_algorithm != "chunked_bitonic_c256" && options.ranking_algorithm != "chunked_bitonic_c512") {
                return finish(options, {}, "failed", "invalid --ranking-algorithm", 1);
            }
        } else if (flag == "--fast-math" && !fast_seen) {
            fast_seen = true;
            auto value = string_view{argv[i + 1]};
            if (value != "0" && value != "1") { return finish(options, {}, "failed", "invalid --fast-math", 1); }
            options.fast_math = value == "1";
        } else {
            return finish(options, {}, "failed", "unknown or duplicate optional argument", 1);
        }
    }
    std::copy(schedule.begin(), schedule.end(), options.tile.begin());
    options.samples = static_cast<uint32_t>(samples);
    options.sample_ms = static_cast<uint32_t>(sample_ms);
    options.warmup_ms = static_cast<uint32_t>(warmup_ms);
    options.graph_batch = static_cast<uint32_t>(graph_batch);
    auto directory = luisa::filesystem::path{argv[12]};
    std::error_code filesystem_error;
    if (!luisa::filesystem::create_directory(directory, filesystem_error) || filesystem_error) {
        return finish(options, {}, "failed", "export_dir must be a new directory under an existing parent", 1);
    }
    log_level_error();
    auto start = Clock::now();
    auto fixture = workloads::make_fixture<T>(options);
    auto fixture_ms = elapsed(start);
    if (!fixture.error.empty()) { return finish(options, directory, "unsupported", fixture.error, 3); }
    if (!export_fixture<T>(directory, options, fixture)) { return finish(options, directory, "failed", "input/oracle export failed", 1); }
    if (!fixture.kernel || !fixture.kernel->valid()) {
        string diagnostics;
        if (fixture.kernel) {
            for (auto &text : fixture.kernel->diagnostics()) {
                diagnostics += text;
                diagnostics += '\n';
            }
        }
        return finish(options, directory, "unsupported", diagnostics.empty() ? string_view{"Tile capture rejected the fixture"} : string_view{diagnostics}, 3);
    }
    for (auto &&kernel : fixture.continuation_kernels) {
        if (!kernel.valid()) {
            string diagnostics;
            for (auto &text : kernel.diagnostics()) {
                diagnostics += text;
                diagnostics += '\n';
            }
            return finish(options, directory, "unsupported", diagnostics.empty() ? string_view{"Tile pipeline capture rejected a stage"} : string_view{diagnostics}, 3);
        }
    }
    auto stage_count = 1u + fixture.continuation_kernels.size();
    auto max_repetitions = uint64_t{100000u} / stage_count;
    if (options.graph_batch > max_repetitions) { return finish(options, directory, "failed", "graph exceeds 100000 total stage dispatches", 1); }
    if (options.backend == "cuda" && options.lowering == "native") {
        auto opt_in = luisa::get_environment_variable("LUISA_CUDA_TILE_IR");
        if (!opt_in || *opt_in != "1") { return finish(options, directory, "unsupported", "CUDA native benchmark requires exact LUISA_CUDA_TILE_IR=1; no fallback", 3); }
    }
    start = Clock::now();
    // Private diagnostic admission; no production entry point or default change.
    uint64_t diagnostic_chunk{};
    auto diagnostic_request = luisa::get_environment_variable("LUISA_DIAGNOSTIC_STREAMED_SUM_CHUNK");
    if (!diagnostic_request || !integer(*diagnostic_request, diagnostic_chunk) ||
        (diagnostic_chunk != 0u && diagnostic_chunk != 1024u && diagnostic_chunk != 2048u) ||
        options.backend != "cuda" || options.lowering != "native" || options.operation != "reduce_sum" ||
        options.fast_math || options.tile[0] != 1 || stage_count != 1u) {
        return finish(options, directory, "failed", "diagnostic requires strict CUDA native BR1 reduce_sum and explicit chunk 0/1024/2048", 1);
    }
    for (auto key : {"LUISA_CUDA_TILE_IR_ALIGNED16", "LUISA_CUDA_TILE_WORKER_WARPS", "LUISA_CUDA_TILE_SCAN_CHUNK",
                     "LUISA_CUDA_TILE_INDEPENDENT_AXIS", "LUISA_CUDA_TILE_STREAMING_SCAN", "LUISA_CUDA_TILE_COLLECTIVE_COST",
                     "LUISA_CUDA_TILE_PROGRAM_ROWS", "LUISA_CUDA_TILE_PARTITION_COST", "LUISA_CUDA_TILE_CUB_SCAN",
                     "LUISA_CUDA_TILE_CUB_SCAN_COST", "LUISA_CUDA_TILE_FORCE_UNSUPPORTED_PTX", "TVM_COMPILE_FORCE_FALLBACK",
                     "LUISA_CUDA_TILE_CUTE"}) {
        if (auto inherited = luisa::get_environment_variable(key); inherited && !inherited->empty()) {
            return finish(options, directory, "failed", "diagnostic forbids inherited schedule experiments", 1);
        }
    }
    uint64_t diagnostic_alignment{};
    auto alignment_request = luisa::get_environment_variable("LUISA_DIAGNOSTIC_STREAMED_SUM_ALIGNED16");
    if (!alignment_request || !integer(*alignment_request, diagnostic_alignment) || diagnostic_alignment > 1u ||
        (options.precision != "fp16" && options.precision != "bf16") ||
        (options.dimensions[0] != 3 && options.dimensions[0] != 128) ||
        (options.dimensions[1] != 8192 && options.dimensions[1] != 16384) || options.graph_batch != 100u) {
        return finish(options, directory, "failed", "V5 requires explicit alignment 0/1 and the declared narrow eight-case cohort", 1);
    }
    vector<tile::DisjointRequirement> diagnostic_ranges(stage_count);
    std::array<uint64_t, 4u> diagnostic_final_pointers{};
    Context context{LUISA_DIAGNOSTIC_RUNTIME_ANCHOR};
    auto device = context.create_device(options.backend);
    auto stream = device.create_stream(StreamTag::COMPUTE);
    auto runtime_ms = elapsed(start);
    tile::CompileOptions compile_options;
    compile_options.lowering = options.lowering == "tirx" ? tile::Lowering::TIRX : tile::Lowering::NATIVE;
    vector<tile::Shader> shaders;
    vector<double> stage_compile_ms;
    shaders.reserve(stage_count);
    auto compile_ms = 0.0;
    for (auto stage = size_t{0u}; stage < stage_count; stage++) {
        auto &kernel = stage == 0u ? *fixture.kernel : fixture.continuation_kernels[stage - 1u];
        start = Clock::now();
        auto shader = [&]() -> tile::Shader {
            namespace native_sum = luisa::compute::cuda::native_tile;
            sum_alignment_diagnostic::ScopedAlignment scoped_alignment{diagnostic_alignment != 0u};
            if (diagnostic_chunk == 0u) {
                if (!write_text(directory / "diagnostic-streamed-sum.json",
                                "{\"schema\":1,\"scope\":\"standalone_actual_fixture\",\"chunk\":0,\"transformed\":false}")) {
                    tile::KernelMetadata metadata;
                    metadata.error = "diagnostic receipt export failed";
                    return {device.impl(), ShaderCreationInfo::make_invalid(), std::move(metadata)};
                }
                return tile::compile(device, kernel, compile_options, {.enable_fast_math = options.fast_math});
            }
            auto candidate = native_sum::build_streamed_sum_ir(kernel.function(), static_cast<uint32_t>(diagnostic_chunk), false);
            tile::KernelMetadata metadata;
            if (!candidate.ok()) {
                metadata.error = "Streamed SUM V4 rejected actual fixture: " + candidate.error;
                return {device.impl(), ShaderCreationInfo::make_invalid(), std::move(metadata)};
            }
            LUISA_ASSERT(candidate.disjoint.input.argument_index == 0u && candidate.disjoint.output.argument_index == 3u &&
                             candidate.disjoint.input.byte_offset == 0u && candidate.disjoint.output.byte_offset == 0u &&
                             candidate.disjoint.input.byte_count == fixture.inputs[0].size() * sizeof(T) &&
                             candidate.disjoint.output.byte_count == fixture.expected.size() * sizeof(T),
                         "Streamed SUM diagnostic requires the actual fixture four-buffer ABI.");
            diagnostic_ranges[stage] = candidate.disjoint;
            auto &facts = candidate.facts;
            std::ostringstream proof;
            proof << "{\"schema\":1,\"scope\":\"standalone_actual_fixture\",\"transformed\":true,\"chunk\":" << diagnostic_chunk
                  << ",\"input_slot\":0,\"output_slot\":3,\"input_bytes\":" << candidate.disjoint.input.byte_count
                  << ",\"output_bytes\":" << candidate.disjoint.output.byte_count
                  << ",\"programs\":" << facts.programs << ",\"serial_iterations\":" << facts.serial_iterations
                  << ",\"collective_invocations_per_program\":" << facts.collective_invocations_per_program
                  << ",\"contribution_elements_per_program\":" << facts.contribution_elements_per_program
                  << ",\"largest_materialized_tile_elements\":" << facts.largest_materialized_tile_elements
                  << ",\"explicit_tile_storage_upper_bound_per_program\":" << facts.explicit_tile_storage_upper_bound_per_program
                  << ",\"discarded_pure_program_operations\":" << facts.discarded_pure_program_operations << '}';
            if (!write_text(directory / "diagnostic-streamed-sum.json", proof.str())) {
                metadata.error = "diagnostic receipt export failed";
                return {device.impl(), ShaderCreationInfo::make_invalid(), std::move(metadata)};
            }
            auto info = device.impl()->create_tile_kernel({.enable_fast_math = false}, *candidate.function, compile_options, metadata);
            if (!info.valid() && metadata.error.empty()) { metadata.error = "Native Tile compilation is unavailable on this device"; }
            return {device.impl(), info, std::move(metadata)};
        }();
        stage_compile_ms.emplace_back(elapsed(start));
        compile_ms += stage_compile_ms.back();
        auto source_name = stage == 0u ? std::string{"source.txt"} : "source-stage" + std::to_string(stage) + ".txt";
        if (!shader.metadata().source.empty() && !write_text(directory / source_name, std::string{shader.metadata().source.data(), shader.metadata().source.size()})) {
            return finish(options, directory, "failed", "stage source export failed", 1, compile_ms);
        }
        // An ordinary CUDA candidate is a different compilation unit. Preserve
        // it separately without changing the original Tile source receipt.
        auto realization = string_view{shader.metadata().realization};
        auto copy_candidate_source = [&](string_view marker, const std::string &name) {
            auto source_position = realization.find(marker);
            if (source_position == string_view::npos) { return true; }
            auto source_path = realization.substr(source_position + marker.size());
            source_path = source_path.substr(0u, source_path.find(';'));
            std::ifstream candidate{std::filesystem::path{std::string{source_path}}, std::ios::binary};
            std::string text{std::istreambuf_iterator<char>{candidate}, std::istreambuf_iterator<char>{}};
            return candidate && !text.empty() && write_text(directory / name, text);
        };
        if (!copy_candidate_source("cub-scan-source-file=", "cub-source-stage" + std::to_string(stage) + ".cu")) {
            return finish(options, directory, "failed", "CUB candidate source export failed", 1, compile_ms);
        }
        // Search attempts have independent compilation units, even when they
        // lose or fail to compile. A missing marker means no source was built.
        // The final winner also keeps the legacy cub-source-stageN.cu receipt.
        for (auto threads : std::array{128u, 256u, 512u, 1024u}) {
            auto marker = "cub-scan-cost-t" + std::to_string(threads) + "-source-file=";
            auto name = "cub-cost-source-stage" + std::to_string(stage) + "-t" + std::to_string(threads) + ".cu";
            if (!copy_candidate_source(marker, name)) {
                return finish(options, directory, "failed", "CUB cost candidate source export failed", 1, compile_ms);
            }
        }
        if (!shader) {
            auto error = string_view{shader.metadata().error};
            auto compiler_failure = error.starts_with("CUDA Tile IR NVRTC failed (") ||
                                    error.starts_with("CUDA Tile IR tileiras failed (") ||
                                    error.starts_with("CUDA Tile IR NVRTC could not start:") ||
                                    error.starts_with("CUDA Tile IR tileiras could not start:");
            return finish(options, directory, compiler_failure ? "compiler_failure" : "unsupported", error,
                          compiler_failure ? 1 : 3, compile_ms, shader.metadata().realization);
        }
        if (options.backend == "cuda" && options.lowering == "native" && !shader.metadata().realization.starts_with("CUDA Tile C++ -> NVRTC Tile IR -> tileiras -> cubin")) {
            return finish(options, directory, "failed", "requested native Tile IR realization was not produced", 1, compile_ms, shader.metadata().realization);
        }
        if (options.backend == "cuda" && options.lowering == "tirx" &&
            (shader.metadata().realization.find("TIRx ->") == string::npos || shader.metadata().realization.find("PTX") == string::npos)) {
            return finish(options, directory, "failed", "requested CUDA TIRx/PTX realization was not produced", 1, compile_ms, shader.metadata().realization);
        }
        shaders.emplace_back(std::move(shader));
    }
    auto &shader = shaders.front();
    CudaTiming events{device, stream, options.backend == "cuda"};
    if (options.backend == "cuda" && !events.enabled) { return finish(options, directory, "unsupported", events.error, 3, compile_ms, shader.metadata().realization); }
    constexpr size_t pad = 64u;
    constexpr float guard = -719.5f;
    constexpr int64_t index_guard = std::numeric_limits<int64_t>::min() + 37;
    std::array<vector<T>, 3u> host_inputs;
    std::array<Buffer<T>, 3u> inputs;
    start = Clock::now();
    for (size_t i = 0; i < 3u; i++) {
        host_inputs[i].assign(fixture.inputs[i].size() + 2u * pad, T{guard});
        for (size_t j = 0; j < fixture.inputs[i].size(); j++) { host_inputs[i][j + pad] = T{fixture.inputs[i][j]}; }
        inputs[i] = device.create_buffer<T>(host_inputs[i].size());
        stream << inputs[i].copy_from(span{host_inputs[i]});
    }
    vector<int64_t> host_input_ids;
    Buffer<int64_t> input_ids;
    if (fixture.input_ids) {
        host_input_ids.assign(fixture.input_ids->size() + 2u * pad, index_guard);
        std::copy(fixture.input_ids->begin(), fixture.input_ids->end(), host_input_ids.begin() + pad);
        input_ids = device.create_buffer<int64_t>(host_input_ids.size());
        stream << input_ids.copy_from(span{host_input_ids});
    }
    vector<T> host_output(fixture.expected.size() + 2u * pad, T{guard});
    std::fill(host_output.begin() + pad, host_output.end() - pad, std::numeric_limits<T>::quiet_NaN());
    auto output = device.create_buffer<T>(host_output.size());
    vector<int64_t> host_indices((fixture.ranking ? fixture.expected_indices.size() : 1u) + 2u * pad, index_guard);
    auto indices = device.create_buffer<int64_t>(host_indices.size());
    stream << output.copy_from(span{host_output}) << indices.copy_from(span{host_indices}) << synchronize();
    std::array<Buffer<float>, 2u> scratch_values;
    std::array<Buffer<int32_t>, 2u> scratch_indices;
    std::array<vector<float>, 2u> host_scratch_values;
    std::array<vector<int32_t>, 2u> host_scratch_indices;
    constexpr int32_t scratch_index_guard = std::numeric_limits<int32_t>::min() + 37;
    if (fixture.scratch_elements != 0u) {
        for (auto slot = size_t{0u}; slot < 2u; slot++) {
            host_scratch_values[slot].assign(fixture.scratch_elements + 2u * pad, guard);
            host_scratch_indices[slot].assign(fixture.scratch_elements + 2u * pad, scratch_index_guard);
            scratch_values[slot] = device.create_buffer<float>(host_scratch_values[slot].size());
            scratch_indices[slot] = device.create_buffer<int32_t>(host_scratch_indices[slot].size());
            stream << scratch_values[slot].copy_from(span{host_scratch_values[slot]})
                   << scratch_indices[slot].copy_from(span{host_scratch_indices[slot]});
        }
        stream << synchronize();
    }
    auto upload_ms = elapsed(start);
    auto make_commands = [&](uint64_t repetitions) {
        CommandList commands;
        for (uint64_t i = 0; i < repetitions; i++) {
            if (fixture.scratch_elements != 0u) {
                // A full operation always starts from the original input. No
                // iteration consumes the previous operation's sorted output.
                commands << shaders[0](inputs[0].view(pad, fixture.inputs[0].size()),
                                       scratch_values[0].view(pad, fixture.scratch_elements),
                                       scratch_indices[0].view(pad, fixture.scratch_elements))
                                .dispatch();
                for (auto stage = size_t{1u}; stage < shaders.size(); stage++) {
                    auto read_slot = (stage - 1u) % 2u, write_slot = stage % 2u;
                    if (stage + 1u == shaders.size()) {
                        commands << shaders[stage](scratch_values[read_slot].view(pad, fixture.scratch_elements),
                                                   scratch_indices[read_slot].view(pad, fixture.scratch_elements),
                                                   output.view(pad, fixture.expected.size()),
                                                   indices.view(pad, fixture.expected_indices.size()))
                                        .dispatch();
                    } else {
                        commands << shaders[stage](scratch_values[read_slot].view(pad, fixture.scratch_elements),
                                                   scratch_indices[read_slot].view(pad, fixture.scratch_elements),
                                                   scratch_values[write_slot].view(pad, fixture.scratch_elements),
                                                   scratch_indices[write_slot].view(pad, fixture.scratch_elements))
                                        .dispatch();
                    }
                }
            } else if (fixture.input_ids) {
                commands << shader(inputs[0].view(pad, fixture.inputs[0].size()),
                                   input_ids.view(pad, fixture.input_ids->size()), output.view(pad, fixture.expected.size()))
                                .dispatch();
            } else if (fixture.ranking) {
                commands << shader(inputs[0].view(pad, fixture.inputs[0].size()), output.view(pad, fixture.expected.size()),
                                   indices.view(pad, fixture.expected_indices.size()))
                                .dispatch();
            } else {
                commands << shader(inputs[0].view(pad, fixture.inputs[0].size()), inputs[1].view(pad, fixture.inputs[1].size()),
                                   inputs[2].view(pad, fixture.inputs[2].size()), output.view(pad, fixture.expected.size()))
                                .dispatch();
            }
        }
        return commands;
    };
    std::string native_alignment_receipts;
    std::string native_streaming_receipts;
    std::string native_partition_receipts;
    std::string native_cub_receipts;
    if (options.backend == "cuda" && options.lowering == "native") {
        // Inspect the actual command binding order, including BufferView byte
        // offsets. Record only pointer residues, never device addresses. This
        // predicts the shared runtime selector, not a device-side trace.
        vector<std::pair<uint64_t, uintptr_t>> buffer_bases;
        auto register_buffer = [&](auto &&buffer) {
            buffer_bases.emplace_back(buffer.handle(), reinterpret_cast<uintptr_t>(buffer.native_handle()));
        };
        for (auto &&input : inputs) { register_buffer(input); }
        register_buffer(output);
        register_buffer(indices);
        if (fixture.input_ids) { register_buffer(input_ids); }
        if (fixture.scratch_elements != 0u) {
            for (auto &&buffer : scratch_values) { register_buffer(buffer); }
            for (auto &&buffer : scratch_indices) { register_buffer(buffer); }
        }
        // The compile-only scope has ended; preserve the original receipt's
        // requested semantics from the explicitly retained compilation policy.
        auto requested = diagnostic_alignment != 0u;
        auto commands = make_commands(1u).steal_commands();
        LUISA_ASSERT(commands.size() == shaders.size(), "Alignment receipts require one command per stage.");
        std::ostringstream receipt;
        std::ostringstream streaming_receipt;
        std::ostringstream partition_receipt;
        std::ostringstream cub_receipt;
        receipt << '[';
        streaming_receipt << '[';
        partition_receipt << '[';
        cub_receipt << '[';
        auto cub_requested = false;
        for (auto stage = size_t{0u}; stage < shaders.size(); stage++) {
            auto command = static_cast<const ShaderDispatchCommand *>(commands[stage].get());
            auto args = command->arguments();
            auto realization = string_view{shaders[stage].metadata().realization};
            constexpr string_view marker = "aligned16-buffer-mask=";
            auto position = realization.find(marker);
            auto mask = uint64_t{0u};
            if (position != string_view::npos) {
                auto text = realization.substr(position + marker.size());
                text = text.substr(0u, text.find(';'));
                LUISA_ASSERT(integer(text, mask), "Invalid alignment mask metadata.");
            }
            LUISA_ASSERT(args.size() <= 31u && (mask >> args.size()) == 0u, "Invalid alignment mask.");
            vector<uint32_t> residues;
            vector<uint64_t> pointers;
            auto aligned = mask != 0u;
            for (auto slot = size_t{0u}; slot < args.size(); slot++) {
                auto &&arg = args[slot];
                LUISA_ASSERT(arg.tag == Argument::Tag::BUFFER, "Tile alignment receipt expects buffers.");
                auto base = std::find_if(buffer_bases.begin(), buffer_bases.end(), [&](auto &&item) { return item.first == arg.buffer.handle; });
                LUISA_ASSERT(base != buffer_bases.end(), "Unknown buffer in alignment receipt.");
                LUISA_ASSERT(arg.buffer.offset <= std::numeric_limits<uint64_t>::max() - base->second, "Buffer address overflow.");
                pointers.emplace_back(base->second + arg.buffer.offset);
                auto residue = static_cast<uint32_t>(((base->second & 15u) + (arg.buffer.offset & 15u)) & 15u);
                residues.emplace_back(residue);
                if ((mask & (uint64_t{1u} << slot)) != 0u && residue != 0u) { aligned = false; }
            }
            LUISA_ASSERT(stage == 0u && pointers.size() == 4u, "V5 single-stage four-pointer ABI required.");
            std::copy(pointers.begin(), pointers.end(), diagnostic_final_pointers.begin());
            // Before any dispatch, prove the V4 conditional rewrite against the
            // actual encoded BufferView pointers and sizes, not base handles.
            if (diagnostic_chunk != 0u) {
                auto &ranges = diagnostic_ranges[stage];
                LUISA_ASSERT(pointers.size() == 4u && args.size() == 4u &&
                                 ranges.input.argument_index == 0u && ranges.output.argument_index == 3u &&
                                 ranges.input.byte_count <= args[0].buffer.size && ranges.output.byte_count <= args[3].buffer.size,
                             "Streamed SUM diagnostic encoded argument bounds differ.");
                auto input = pointers[0], result = pointers[3];
                auto input_bytes = ranges.input.byte_count, output_bytes = ranges.output.byte_count;
                auto disjoint = input != 0u && result != 0u && input_bytes > 0u && output_bytes > 0u &&
                                input_bytes <= std::numeric_limits<uint64_t>::max() - input &&
                                output_bytes <= std::numeric_limits<uint64_t>::max() - result &&
                                (input + input_bytes <= result || result + output_bytes <= input);
                if (!disjoint) { return finish(options, directory, "failed", "streamed SUM actual final-pointer ranges overlap or overflow", 1, compile_ms); }
                if (!write_text(directory / "diagnostic-disjoint.json",
                                "{\"schema\":1,\"actual_encoded_arguments_checked\":true,\"whole_static_ranges_disjoint\":true}")) {
                    return finish(options, directory, "failed", "diagnostic disjoint receipt export failed", 1, compile_ms);
                }
            }
            auto fact = [&](string_view name) {
                auto found = realization.find(name);
                auto value = uint64_t{0u};
                if (found != string_view::npos) {
                    auto text = realization.substr(found + name.size());
                    text = text.substr(0u, text.find(';'));
                    LUISA_ASSERT(integer(text, value), "Invalid streaming metadata.");
                }
                return value;
            };
            auto chunk = fact("streaming-scan-chunk=");
            auto available = realization.find("; streaming-scan-available;") != string_view::npos;
            auto input_slot = fact("streaming-scan-input-slot="), output_slot = fact("streaming-scan-output-slot=");
            auto input_bytes = fact("streaming-scan-input-bytes="), output_bytes = fact("streaming-scan-output-bytes=");
            auto disjoint = false;
            if (available) {
                LUISA_ASSERT(input_slot < pointers.size() && output_slot < pointers.size() && input_slot != output_slot &&
                                 input_bytes > 0u && output_bytes > 0u &&
                                 input_bytes <= args[input_slot].buffer.size && output_bytes <= args[output_slot].buffer.size,
                             "Invalid streaming view metadata.");
                auto input = pointers[input_slot], output_pointer = pointers[output_slot];
                disjoint = input != 0u && output_pointer != 0u &&
                           input_bytes <= std::numeric_limits<uint64_t>::max() - input &&
                           output_bytes <= std::numeric_limits<uint64_t>::max() - output_pointer &&
                           (input + input_bytes <= output_pointer || output_pointer + output_bytes <= input);
            }
            auto selected = available && disjoint ? "luisa_tile_stream_scan" : aligned ? "luisa_tile_aligned16" :
                                                                                         "luisa_tile_main";
            auto partition_rows = fact("program-partition-rows=");
            auto partition_available = realization.find("; program-partition-available;") != string_view::npos;
            auto partition_input_slot = fact("program-partition-input-slot="), partition_output_slot = fact("program-partition-output-slot=");
            auto partition_input_bytes = fact("program-partition-input-bytes="), partition_output_bytes = fact("program-partition-output-bytes=");
            auto partition_original_rows = fact("program-partition-original-rows=");
            auto partition_grid = fact("program-partition-grid-x="), partition_original_grid = fact("program-partition-original-grid-x=");
            auto partition_disjoint = false;
            auto default_grid = shaders[stage].metadata().dispatch_size;
            if (partition_available) {
                LUISA_ASSERT(partition_input_slot < pointers.size() && partition_output_slot < pointers.size() &&
                                 partition_input_slot != partition_output_slot && partition_input_bytes > 0u && partition_output_bytes > 0u &&
                                 partition_input_bytes <= args[partition_input_slot].buffer.size &&
                                 partition_output_bytes <= args[partition_output_slot].buffer.size &&
                                 partition_original_grid == default_grid.x && default_grid.y == 1u && default_grid.z == 1u &&
                                 partition_grid > 0u && partition_grid <= UINT32_MAX,
                             "Invalid program partition view/grid metadata.");
                auto input = pointers[partition_input_slot], output_pointer = pointers[partition_output_slot];
                partition_disjoint = input != 0u && output_pointer != 0u &&
                                     partition_input_bytes <= std::numeric_limits<uint64_t>::max() - input &&
                                     partition_output_bytes <= std::numeric_limits<uint64_t>::max() - output_pointer &&
                                     (input + partition_input_bytes <= output_pointer || output_pointer + partition_output_bytes <= input);
            }
            auto selected_grid = default_grid;
            if (partition_available && partition_disjoint) {
                selected = "luisa_tile_partition";
                selected_grid = make_uint3(static_cast<uint32_t>(partition_grid), 1u, 1u);
            }
            auto cub_threads = fact("cub-scan-threads=");
            auto cub_available = realization.find("; cub-scan-available;") != string_view::npos;
            auto cub_input = fact("cub-scan-input-slot="), cub_output = fact("cub-scan-output-slot=");
            auto cub_input_bytes = fact("cub-scan-input-bytes="), cub_output_bytes = fact("cub-scan-output-bytes=");
            auto cub_grid = fact("cub-scan-grid-x=");
            auto cub_aligned = false, cub_disjoint = false;
            auto selected_block = make_uint3(1u);
            // A cost request can retain the original with threads=0.
            // Preserve its explicit unavailable final receipt as well.
            cub_requested |= cub_threads != 0u || realization.find("; cub-scan-cost-requested=1;") != string_view::npos;
            if (cub_available) {
                LUISA_ASSERT(cub_input < pointers.size() && cub_output < pointers.size() && cub_input != cub_output &&
                                 cub_input_bytes != 0u && cub_output_bytes != 0u &&
                                 cub_input_bytes <= args[cub_input].buffer.size && cub_output_bytes <= args[cub_output].buffer.size &&
                                 cub_grid != 0u && cub_grid <= UINT32_MAX &&
                                 fact("cub-scan-block-x=") == cub_threads &&
                                 fact("cub-scan-alignment-mask=") == ((uint64_t{1u} << cub_input) | (uint64_t{1u} << cub_output)) &&
                                 (cub_threads == 128u || cub_threads == 256u || cub_threads == 512u || cub_threads == 1024u),
                             "Invalid CUB prefix view/launch metadata.");
                auto input = pointers[cub_input], output_pointer = pointers[cub_output];
                cub_aligned = ((input | output_pointer) & 15u) == 0u;
                cub_disjoint = input != 0u && output_pointer != 0u &&
                               cub_input_bytes <= std::numeric_limits<uint64_t>::max() - input &&
                               cub_output_bytes <= std::numeric_limits<uint64_t>::max() - output_pointer &&
                               (input + cub_input_bytes <= output_pointer || output_pointer + cub_output_bytes <= input);
            }
            if (cub_available && cub_aligned && cub_disjoint) {
                selected = "luisa_tile_cub_scan";
                selected_grid = make_uint3(static_cast<uint32_t>(cub_grid), 1u, 1u);
                selected_block = make_uint3(static_cast<uint32_t>(cub_threads), 1u, 1u);
            }
            if (stage != 0u) { cub_receipt << ','; }
            cub_receipt << "{\"stage\":" << stage << ",\"threads_requested\":" << cub_threads
                        << ",\"available\":" << (cub_available ? "true" : "false")
                        << ",\"input_slot\":" << cub_input << ",\"output_slot\":" << cub_output
                        << ",\"input_bytes\":" << cub_input_bytes << ",\"output_bytes\":" << cub_output_bytes
                        << ",\"static_ranges_disjoint\":" << (cub_disjoint ? "true" : "false")
                        << ",\"final_pointers_aligned16\":" << (cub_aligned ? "true" : "false")
                        << ",\"expected_selected_entry\":";
            quoted(cub_receipt, selected);
            cub_receipt << ",\"expected_selected_grid\":";
            array(cub_receipt, std::array{selected_grid.x, selected_grid.y, selected_grid.z});
            cub_receipt << ",\"expected_selected_block\":";
            array(cub_receipt, std::array{selected_block.x, selected_block.y, selected_block.z});
            cub_receipt << '}';
            if (stage != 0u) { partition_receipt << ','; }
            partition_receipt << "{\"stage\":" << stage << ",\"rows_requested\":" << partition_rows
                              << ",\"available\":" << (partition_available ? "true" : "false")
                              << ",\"input_slot\":" << partition_input_slot << ",\"output_slot\":" << partition_output_slot
                              << ",\"input_bytes\":" << partition_input_bytes << ",\"output_bytes\":" << partition_output_bytes
                              << ",\"original_rows\":" << partition_original_rows << ",\"grid_x\":" << partition_grid
                              << ",\"original_grid_x\":" << partition_original_grid
                              << ",\"static_ranges_disjoint\":" << (partition_disjoint ? "true" : "false")
                              << ",\"original_grid\":";
            array(partition_receipt, std::array{default_grid.x, default_grid.y, default_grid.z});
            partition_receipt << ",\"expected_selected_grid\":";
            array(partition_receipt, std::array{selected_grid.x, selected_grid.y, selected_grid.z});
            partition_receipt << ",\"expected_selected_entry\":";
            quoted(partition_receipt, selected);
            partition_receipt << '}';
            if (stage != 0u) { streaming_receipt << ','; }
            streaming_receipt << "{\"stage\":" << stage << ",\"chunk_requested\":" << chunk
                              << ",\"available\":" << (available ? "true" : "false")
                              << ",\"input_slot\":" << input_slot << ",\"output_slot\":" << output_slot
                              << ",\"input_bytes\":" << input_bytes << ",\"output_bytes\":" << output_bytes
                              << ",\"static_ranges_disjoint\":" << (disjoint ? "true" : "false")
                              << ",\"expected_selected_entry\":";
            quoted(streaming_receipt, selected);
            streaming_receipt << '}';
            if (stage != 0u) { receipt << ','; }
            receipt << "{\"stage\":" << stage << ",\"requested\":" << (requested ? "true" : "false")
                    << ",\"eligible_buffer_mask\":" << mask << ",\"final_argument_mod16\":";
            array(receipt, residues);
            receipt << ",\"expected_selected_entry\":";
            quoted(receipt, selected);
            receipt << '}';
        }
        receipt << ']';
        streaming_receipt << ']';
        partition_receipt << ']';
        cub_receipt << ']';
        native_alignment_receipts = receipt.str();
        native_streaming_receipts = streaming_receipt.str();
        native_partition_receipts = partition_receipt.str();
        if (cub_requested) { native_cub_receipts = cub_receipt.str(); }
    }
    auto batch = [&](uint64_t repetitions, bool instrumented, double &device_ms) {
        stream.synchronize();
        auto before = Clock::now();
        auto commands = make_commands(repetitions);
        if (instrumented && !events.record(true)) { return -1.0; }
        stream << commands.commit();
        if (instrumented && !events.record(false)) {
            stream.synchronize();
            return -1.0;
        }
        stream.synchronize();
        auto host_ms = elapsed(before);
        device_ms = instrumented ? events.milliseconds() : 0.0;
        return instrumented && device_ms < 0.0 ? -1.0 : host_ms;
    };
    double max_error = 0.0, max_error_over_bound = 0.0;
    uint64_t checks = 0u, errors = 0u, strict_errors = 0u;
    double strict_max_error_over_bound = 0.0;
    auto check = [&] {
        stream << output.copy_to(span{host_output}) << indices.copy_to(span{host_indices});
        for (size_t i = 0; i < 3u; i++) { stream << inputs[i].copy_to(span{host_inputs[i]}); }
        if (fixture.input_ids) { stream << input_ids.copy_to(span{host_input_ids}); }
        if (fixture.scratch_elements != 0u) {
            for (auto slot = size_t{0u}; slot < 2u; slot++) {
                stream << scratch_values[slot].copy_to(span{host_scratch_values[slot]})
                       << scratch_indices[slot].copy_to(span{host_scratch_indices[slot]});
            }
        }
        stream << synchronize();
        if (fixture.scratch_elements != 0u) {
            for (auto slot = size_t{0u}; slot < 2u; slot++) {
                for (auto i = size_t{0u}; i < pad; i++) {
                    auto tail = fixture.scratch_elements + pad + i;
                    errors += std::bit_cast<uint32_t>(host_scratch_values[slot][i]) != std::bit_cast<uint32_t>(guard);
                    errors += std::bit_cast<uint32_t>(host_scratch_values[slot][tail]) != std::bit_cast<uint32_t>(guard);
                    errors += host_scratch_indices[slot][i] != scratch_index_guard;
                    errors += host_scratch_indices[slot][tail] != scratch_index_guard;
                }
            }
        }
        if (fixture.input_ids) {
            for (size_t i = 0; i < host_input_ids.size(); i++) {
                auto expected = i < pad || i >= host_input_ids.size() - pad ? index_guard : (*fixture.input_ids)[i - pad];
                errors += host_input_ids[i] != expected;
            }
        }
        auto bits = [](T x) { return std::bit_cast<std::array<std::byte, sizeof(T)>>(x); };
        for (size_t i = 0; i < 3u; i++) {
            for (size_t j = 0; j < host_inputs[i].size(); j++) {
                auto expected = T{j < pad || j >= host_inputs[i].size() - pad ? guard : fixture.inputs[i][j - pad]};
                errors += bits(host_inputs[i][j]) != bits(expected);
            }
        }
        for (size_t i = 0; i < host_output.size(); i++) {
            if (i < pad || i >= host_output.size() - pad) {
                errors += bits(host_output[i]) != bits(T{guard});
                continue;
            }
            auto j = i - pad;
            auto error = std::abs(static_cast<double>(static_cast<float>(host_output[i])) - fixture.expected[j]);
            auto finite = std::isfinite(static_cast<float>(host_output[i]));
            errors += !finite || error > fixture.bound[j];
            if (!fixture.strict_bound.empty()) {
                strict_errors += !finite || error > fixture.strict_bound[j];
                if (finite && fixture.strict_bound[j] > 0.0) {
                    strict_max_error_over_bound = std::max(strict_max_error_over_bound, error / fixture.strict_bound[j]);
                }
            }
            if (finite) {
                max_error = std::max(max_error, error);
                if (fixture.bound[j] > 0.0) { max_error_over_bound = std::max(max_error_over_bound, error / fixture.bound[j]); }
            }
            if (fixture.ranking || fixture.input_ids) { errors += bits(host_output[i]) != bits(T{static_cast<float>(fixture.expected[j])}); }
        }
        for (size_t i = 0; i < host_indices.size(); i++) {
            auto expected = fixture.ranking && i >= pad && i < host_indices.size() - pad ? fixture.expected_indices[i - pad] : index_guard;
            errors += host_indices[i] != expected;
        }
        checks++;
        return errors == 0u;
    };
    double ignored{};
    auto cold_ms = batch(1u, false, ignored);
    if (!check()) { return finish(options, directory, "failed", "cold full-output/guard/input/index oracle failed", 2, compile_ms, shader.metadata().realization); }
    start = Clock::now();
    while (elapsed(start) < options.warmup_ms) { static_cast<void>(batch(8u, false, ignored)); }
    auto warmup_actual_ms = elapsed(start);
    uint64_t repetitions = 1u;
    for (uint32_t attempt = 0; attempt < 8u; attempt++) {
        auto ms = batch(repetitions, false, ignored);
        if (ms >= options.sample_ms * .8 || repetitions == max_repetitions) { break; }
        repetitions = std::clamp<uint64_t>(static_cast<uint64_t>(repetitions * options.sample_ms / std::max(ms, 1e-6)), repetitions + 1u, max_repetitions);
    }
    vector<double> throughput, latency, device_span, instrumented_host, graph_host, graph_device;
    for (uint32_t i = 0; i < options.samples; i++) { throughput.emplace_back(1000.0 * batch(repetitions, false, ignored) / repetitions); }
    for (uint32_t i = 0; i < options.samples; i++) { latency.emplace_back(1000.0 * batch(1u, false, ignored)); }
    if (events.enabled) {
        for (uint32_t i = 0; i < options.samples; i++) {
            double device_ms{};
            auto host_ms = batch(repetitions, true, device_ms);
            if (host_ms < 0.0) { return finish(options, directory, "failed", events.error, 1, compile_ms, shader.metadata().realization); }
            instrumented_host.emplace_back(1000.0 * host_ms / repetitions);
            device_span.emplace_back(1000.0 * device_ms / repetitions);
        }
    }
    if (!check()) { return finish(options, directory, "failed", "warm full-output/guard/input/index oracle failed", 2, compile_ms, shader.metadata().realization); }
    double graph_build_ms = 0.0, graph_prime_event_ms = 0.0, graph_prime_host_ms = 0.0;
    double graph_warmup_actual_ms = 0.0;
    uint64_t graph_replays = 0u, graph_replay_cap = 0u, graph_warmup_replays = 0u;
    constexpr uint64_t graph_operation_cap = 10000000u;
    struct GraphSample {
        uint64_t replays{};
        double event_ms{-1.0}, host_ms{-1.0};
    };
    vector<GraphSample> graph_calibration;
    vector<double> graph_event_span_ms, graph_host_span_ms;
    bool graph_calibration_target_reached = false;
    if (options.graph_batch != 0u) {
        if (options.backend != "cuda") { return finish(options, directory, "unsupported", "graph measurement requires CUDA", 3, compile_ms, shader.metadata().realization); }
        auto ext = device.extension<CudaGraphExt>();
        if (ext == nullptr) { return finish(options, directory, "unsupported", "CudaGraphExt is unavailable", 3, compile_ms, shader.metadata().realization); }
        start = Clock::now();
        auto graph = ext->create_graph(make_commands(options.graph_batch));
        if (!graph.handle().valid()) { return finish(options, directory, "unsupported", "actual Tile graph creation rejected", 3, compile_ms, shader.metadata().realization); }
        auto executable = ext->instantiate(graph.handle().handle);
        if (!executable.handle().valid()) { return finish(options, directory, "unsupported", "actual Tile graph instantiation rejected", 3, compile_ms, shader.metadata().realization); }
        graph_build_ms = elapsed(start);
        // Outside every measured interval, inspect all nodes of this exact graph.
        if (!sum_alignment_diagnostic::observe(device, shader, graph, diagnostic_final_pointers,
                static_cast<uint32_t>(options.dimensions[0]), static_cast<uint32_t>(options.graph_batch),
                diagnostic_alignment != 0u, directory)) {
            return finish(options, directory, "failed", "V5 actual graph selection or descriptor check failed", 1, compile_ms);
        }
        auto measure_graph = [&](uint64_t replay_count) {
            stream.synchronize();
            auto before = Clock::now();
            GraphSample sample{.replays = replay_count};
            if (!events.record(true)) { return sample; }
            for (auto replay = uint64_t{0u}; replay < replay_count; replay++) {
                // Launch the complete captured operation batch, including every
                // stage when the fixture is a multi-kernel pipeline.
                ext->launch(executable.handle().handle, stream.handle());
            }
            if (!events.record(false)) {
                stream.synchronize();
                return sample;
            }
            stream.synchronize();
            sample.host_ms = elapsed(before);
            sample.event_ms = events.milliseconds();
            return sample;
        };
        auto valid_graph_sample = [](const GraphSample &sample) noexcept {
            return std::isfinite(sample.event_ms) && sample.event_ms > 0.0 &&
                   std::isfinite(sample.host_ms) && sample.host_ms > 0.0;
        };
        // Reuse the existing first untimed replay to prime this exact event
        // pair before warmup. No official sample is discarded or filtered.
        auto prime = measure_graph(1u);
        if (!valid_graph_sample(prime)) {
            return finish(options, directory, "failed", "graph event priming failed or produced a nonpositive/nonfinite span", 1);
        }
        graph_prime_event_ms = prime.event_ms;
        graph_prime_host_ms = prime.host_ms;
        if (!check()) { return finish(options, directory, "failed", "cold graph oracle failed", 2, compile_ms, shader.metadata().realization); }
        start = Clock::now();
        while (elapsed(start) < options.warmup_ms) {
            ext->launch(executable.handle().handle, stream.handle());
            stream.synchronize();
            graph_warmup_replays++;
        }
        graph_warmup_actual_ms = elapsed(start);
        // Caps count complete logical operations, not driver graph nodes.
        // The same graph and final actually calibrated replay count are used
        // for every official sample; reaching the target is reported, not assumed.
        graph_replay_cap = std::min<uint64_t>(65536u, graph_operation_cap / options.graph_batch);
        graph_replays = 1u;
        for (auto attempt = 0u; attempt < 4u; attempt++) {
            auto sample = measure_graph(graph_replays);
            if (!valid_graph_sample(sample)) {
                return finish(options, directory, "failed", "graph calibration failed or produced a nonpositive/nonfinite span", 1);
            }
            graph_calibration.emplace_back(sample);
            graph_calibration_target_reached = sample.event_ms >= 0.8 * options.sample_ms;
            if (graph_calibration_target_reached || graph_replays == graph_replay_cap || attempt == 3u) { break; }
            auto estimate = std::ceil(static_cast<double>(graph_replays) * options.sample_ms / sample.event_ms);
            // Clamp in floating point before casting; a very short valid event
            // span must not cause an out-of-range integer conversion.
            auto bounded = std::min(estimate, static_cast<double>(graph_replay_cap));
            graph_replays = std::max(graph_replays + 1u, static_cast<uint64_t>(bounded));
        }
        auto operations = graph_replays * options.graph_batch;
        for (uint32_t i = 0; i < options.samples; i++) {
            auto sample = measure_graph(graph_replays);
            if (!valid_graph_sample(sample)) {
                return finish(options, directory, "failed", "graph sample failed or produced a nonpositive/nonfinite span", 1);
            }
            graph_host_span_ms.emplace_back(sample.host_ms);
            graph_event_span_ms.emplace_back(sample.event_ms);
            graph_host.emplace_back(1000.0 * sample.host_ms / operations);
            graph_device.emplace_back(1000.0 * sample.event_ms / operations);
        }
        if (!check()) { return finish(options, directory, "failed", "warm graph oracle failed", 2, compile_ms, shader.metadata().realization); }
    }
    vector<T> final_output(host_output.begin() + pad, host_output.end() - pad);
    if (!write_binary(directory / (std::string{"output"} + storage_suffix<T>()), final_output)) { return finish(options, directory, "failed", "output export failed", 1); }
    if (fixture.ranking) {
        vector<int64_t> final_indices(host_indices.begin() + pad, host_indices.end() - pad);
        if (!write_binary(directory / "output_indices.i64", final_indices)) { return finish(options, directory, "failed", "index export failed", 1); }
    }
    std::ostringstream out;
    out << std::setprecision(17) << '{';
    identity(out, options);
    if (!fixture.strict_bound.empty()) {
        out << ",\"precision_contract\":\"attention_single_narrow_probability_v1\"";
    }
    out << ",\"status\":\"passed\",\"realization\":";
    quoted(out, shader.metadata().realization);
    if (!native_alignment_receipts.empty()) {
        out << ",\"native_alignment\":" << native_alignment_receipts
            << ",\"native_alignment_evidence\":\"actual command BufferView offsets plus CUDA native buffer base modulo 16; expected host selection, not device trace\"";
    }
    if (!native_streaming_receipts.empty()) {
        out << ",\"native_streaming\":" << native_streaming_receipts
            << ",\"native_streaming_evidence\":\"actual command BufferView pointers and proved static touched ranges; expected shared live/graph host selection, not device trace\"";
    }
    if (!native_partition_receipts.empty()) {
        out << ",\"native_program_partition\":" << native_partition_receipts
            << ",\"native_program_partition_evidence\":\"actual command BufferView pointers and proved static ranges; expected shared host entry/grid selection, not device trace\"";
    }
    if (!native_cub_receipts.empty()) {
        out << ",\"native_cub_scan\":" << native_cub_receipts
            << ",\"native_cub_scan_evidence\":\"actual command BufferView pointers and proved static ranges; expected host function/grid/block selection, not device trace\"";
    }
    pipeline_metadata(out, fixture);
    if (!fixture.pipeline_widths.empty()) {
        out << ",\"pipeline_stages\":[";
        for (auto stage = size_t{0u}; stage < shaders.size(); stage++) {
            if (stage != 0u) { out << ','; }
            out << "{\"index\":" << stage << ",\"compile_ms\":" << stage_compile_ms[stage] << ",\"source\":";
            quoted(out, stage == 0u ? std::string{"source.txt"} : "source-stage" + std::to_string(stage) + ".txt");
            out << ",\"realization\":";
            quoted(out, shaders[stage].metadata().realization);
            out << '}';
        }
        out << "],\"graph_stage_dispatches\":" << shaders.size() * options.graph_batch;
    }
    out << ",\"timing_scope\":\"C++ command construction, submission, execution and final synchronization\","
           "\"device_timing_scope\":\"CUDA event stream span after command construction; may include host submission starvation; not isolated kernel time\","
           "\"graph_timing_scope\":\"R complete graph replays per sample; all stages included; event stream span and synchronized host wall divided by graph_batch*R; complete-operation throughput; hazard DAG may overlap independent stages across calls and host submission may starve GPU\","
           "\"fixture_ms\":"
        << fixture_ms << ",\"runtime_ms\":" << runtime_ms << ",\"compile_ms\":" << compile_ms
        << ",\"allocation_upload_ms\":" << upload_ms << ",\"cold_ms\":" << cold_ms << ",\"warmup_actual_ms\":" << warmup_actual_ms
        << ",\"samples\":" << options.samples << ",\"sample_target_ms\":" << options.sample_ms << ",\"repetitions\":" << repetitions
        << ",\"host_wall_us\":";
    array(out, throughput);
    out << ",\"host_wall_p50_us\":" << median(throughput) << ",\"single_sync_us\":";
    array(out, latency);
    out << ",\"cuda_event_stream_span_us\":";
    array(out, device_span);
    out << ",\"event_instrumented_host_wall_us\":";
    array(out, instrumented_host);
    out << ",\"graph_build_ms\":" << graph_build_ms
        << ",\"graph_protocol\":\"adaptive_replay_span_v2\",\"graph_replays_per_sample\":" << graph_replays
        << ",\"graph_operations_per_sample\":" << graph_replays * options.graph_batch
        << ",\"graph_sample_target_ms\":" << options.sample_ms
        << ",\"graph_replay_cap\":" << graph_replay_cap
        << ",\"graph_operation_cap\":" << graph_operation_cap
        << ",\"graph_calibration_target_reached\":" << (graph_calibration_target_reached ? "true" : "false")
        << ",\"graph_prime_event_ms\":" << graph_prime_event_ms
        << ",\"graph_prime_host_ms\":" << graph_prime_host_ms
        << ",\"graph_warmup_replays\":" << graph_warmup_replays
        << ",\"graph_warmup_actual_ms\":" << graph_warmup_actual_ms
        << ",\"graph_calibration\":[";
    for (size_t i = 0u; i < graph_calibration.size(); i++) {
        auto &&sample = graph_calibration[i];
        if (i != 0u) { out << ','; }
        out << "{\"replays\":" << sample.replays << ",\"event_span_ms\":" << sample.event_ms
            << ",\"host_wall_ms\":" << sample.host_ms << '}';
    }
    out << "],\"graph_event_span_ms\":";
    array(out, graph_event_span_ms);
    out << ",\"graph_host_span_ms\":";
    array(out, graph_host_span_ms);
    out << ",\"graph_stage_dispatches_per_sample\":" << graph_replays * options.graph_batch * shaders.size();
    out << ",\"graph_host_wall_us_per_op\":";
    array(out, graph_host);
    out << ",\"graph_event_stream_span_us_per_op\":";
    array(out, graph_device);
    out << ",\"correctness\":{\"checks\":" << checks << ",\"elements_per_check\":" << fixture.expected.size()
        << ",\"errors\":" << errors << ",\"max_abs_error\":" << max_error << ",\"max_error_over_bound\":" << max_error_over_bound
        << ",\"inputs_unchanged\":true,\"guards_unchanged\":true,\"all_outputs_finite\":true}";
    if (!fixture.strict_bound.empty()) {
        out << ",\"strict_correctness\":{\"primary_acceptance\":false,\"checks\":" << checks
            << ",\"failed_element_checks\":" << strict_errors << ",\"max_error_over_bound\":" << strict_max_error_over_bound << '}';
    }
    out << "}\n";
    if (!write_text(directory / "results.json", out.str())) {
        std::cerr << "Cannot write results.json\n";
        return 1;
    }
    std::cout << out.str();
    return 0;
}

int main(int argc, char *argv[]) {
    if (argc > 4) {
        auto precision = string_view{argv[4]};
        if (precision == "fp16") { return run<half>(argc, argv); }
        if (precision == "bf16") { return run<tile::bfloat16>(argc, argv); }
        if (precision != "fp32") {
            std::cerr << "Unsupported precision: expected fp32, fp16 or bf16\n";
            return 3;
        }
    }
    return run<float>(argc, argv);
}

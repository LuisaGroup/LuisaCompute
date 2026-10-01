// Actual Tile workload benchmark: exported fixtures, complete FP64/guard
// checks, explicit routes, host/event timing, and optional CUDA graph replay.
#include "tile_workload_test_utils.h"
#include <luisa/core/platform.h>
#include <luisa/core/stl/filesystem.h>
#include <luisa/runtime/context.h>
#include <luisa/runtime/stream.h>
#include <luisa/tile/runtime.h>
#include <luisa/backends/ext/cuda/cuda_graph_ext.h>
#include <algorithm>
#include <charconv>
#include <chrono>
#include <fstream>
#include <iomanip>
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
    out << ",\"fast_math\":false,\"graph_batch\":" << o.graph_batch;
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

template<typename T>
[[nodiscard]] bool export_fixture(const luisa::filesystem::path &directory,
                                  const workloads::Options &o, const workloads::Fixture &f) {
    for (size_t i = 0u; i < 3u; i++) {
        vector<T> stored;
        stored.reserve(f.inputs[i].size());
        for (auto value : f.inputs[i]) { stored.emplace_back(T{value}); }
        if (!write_binary(directory / ("input" + std::to_string(i) + storage_suffix<T>()), stored)) { return false; }
    }
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
    out << ",\"endianness\":\"little\",\"inputs\":[";
    for (size_t i = 0; i < 3u; i++) {
        if (i != 0u) { out << ','; }
        out << "{\"name\":\"input" << i << "\",\"path\":\"input" << i << storage_suffix<T>() << "\",\"storage_dtype\":";
        quoted(out, storage_name<T>());
        out << ",\"shape\":";
        array(out, f.input_shapes[i]);
        out << '}';
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
           "\"stable\":true,\"tie_break\":\"original_index_ascending\",\"accumulation\":\"float32\","
           "\"gelu_approximation\":\"tanh\",\"causal\":true,\"query_positions\":\"last_Q_in_K\","
           "\"attention_scale\":";
    auto scale = (o.operation == "attention" || o.operation == "attention_tensorcore") ? 1.0f / std::sqrt(static_cast<float>(o.dimensions[5])) : 1.0f;
    out << static_cast<double>(scale) << ",\"input_quantization\":\"round_to_nearest_even_before_fp64_oracle\",\"output_rounding\":\"round_to_nearest_even\",\"contraction\":";
    quoted(out, o.operation == "gemv"                                                                          ? "nonfused_products_unordered_tree_sum" :
                o.operation == "gemm" || (o.operation == "attention" || o.operation == "attention_tensorcore") ? "mma_fused_reassociation_allowed" :
                                                                                                                 "not_applicable");
    out << ",\"logical_view_offset_elements\":64,\"logical_view_offset_bytes\":" << 64u * sizeof(T)
        << ",\"output_preallocated\":true,\"graph_contract\":\"Native: N repeated identical dispatches, same inputs and output allocation, ordered WAW; one graph replay per timed sample. Functional Torch output allocation is a separate contract.\"}"
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
    if (argc != 13 && argc != 15) {
        std::cerr << "Usage: benchmark_tile_workloads <cuda|simd> <native|tirx> operation <fp32|fp16|bf16> dimensions_csv tile_m,tile_n,tile_k seed <random|cancellation|adversarial> samples sample_ms warmup_ms export_dir [--graph-batch N]\n";
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
    if (argc == 15) {
        if (string_view{argv[13]} != "--graph-batch" || !integer(argv[14], graph_batch) || graph_batch == 0u || graph_batch > 100000u) {
            return finish(options, {}, "failed", "invalid --graph-batch", 1);
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
    if (options.backend == "cuda" && options.lowering == "native") {
        auto opt_in = luisa::get_environment_variable("LUISA_CUDA_TILE_IR");
        if (!opt_in || *opt_in != "1") { return finish(options, directory, "unsupported", "CUDA native benchmark requires exact LUISA_CUDA_TILE_IR=1; no fallback", 3); }
    }
    start = Clock::now();
    Context context{argv[0]};
    auto device = context.create_device(options.backend);
    auto stream = device.create_stream(StreamTag::COMPUTE);
    auto runtime_ms = elapsed(start);
    tile::CompileOptions compile_options;
    compile_options.lowering = options.lowering == "tirx" ? tile::Lowering::TIRX : tile::Lowering::NATIVE;
    start = Clock::now();
    auto shader = tile::compile(device, *fixture.kernel, compile_options, {.enable_fast_math = false});
    auto compile_ms = elapsed(start);
    if (!shader.metadata().source.empty() && !write_text(directory / "source.txt", std::string{shader.metadata().source.data(), shader.metadata().source.size()})) {
        return finish(options, directory, "failed", "source export failed", 1, compile_ms);
    }
    if (!shader) { return finish(options, directory, "unsupported", shader.metadata().error, 3, compile_ms, shader.metadata().realization); }
    if (options.backend == "cuda" && options.lowering == "native" && !shader.metadata().realization.starts_with("CUDA Tile C++ -> NVRTC Tile IR -> tileiras -> cubin")) {
        return finish(options, directory, "failed", "requested native Tile IR realization was not produced", 1, compile_ms, shader.metadata().realization);
    }
    if (options.backend == "cuda" && options.lowering == "tirx" &&
        (shader.metadata().realization.find("TIRx ->") == string::npos || shader.metadata().realization.find("PTX") == string::npos)) {
        return finish(options, directory, "failed", "requested CUDA TIRx/PTX realization was not produced", 1, compile_ms, shader.metadata().realization);
    }
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
    vector<T> host_output(fixture.expected.size() + 2u * pad, T{guard});
    std::fill(host_output.begin() + pad, host_output.end() - pad, std::numeric_limits<T>::quiet_NaN());
    auto output = device.create_buffer<T>(host_output.size());
    vector<int64_t> host_indices((fixture.ranking ? fixture.expected_indices.size() : 1u) + 2u * pad, index_guard);
    auto indices = device.create_buffer<int64_t>(host_indices.size());
    stream << output.copy_from(span{host_output}) << indices.copy_from(span{host_indices}) << synchronize();
    auto upload_ms = elapsed(start);
    auto make_commands = [&](uint64_t repetitions) {
        CommandList commands;
        for (uint64_t i = 0; i < repetitions; i++) {
            if (fixture.ranking) {
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
        stream << synchronize();
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
            if (fixture.ranking) { errors += bits(host_output[i]) != bits(T{static_cast<float>(fixture.expected[j])}); }
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
        if (ms >= options.sample_ms * .8 || repetitions == 100000u) { break; }
        repetitions = std::clamp<uint64_t>(static_cast<uint64_t>(repetitions * options.sample_ms / std::max(ms, 1e-6)), repetitions + 1u, 100000u);
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
    double graph_build_ms = 0.0;
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
        ext->launch(executable.handle().handle, stream.handle());
        stream.synchronize();
        if (!check()) { return finish(options, directory, "failed", "cold graph oracle failed", 2, compile_ms, shader.metadata().realization); }
        start = Clock::now();
        while (elapsed(start) < options.warmup_ms) {
            ext->launch(executable.handle().handle, stream.handle());
            stream.synchronize();
        }
        for (uint32_t i = 0; i < options.samples; i++) {
            stream.synchronize();
            auto before = Clock::now();
            if (!events.record(true)) { return finish(options, directory, "failed", events.error, 1); }
            ext->launch(executable.handle().handle, stream.handle());
            if (!events.record(false)) {
                stream.synchronize();
                return finish(options, directory, "failed", events.error, 1);
            }
            stream.synchronize();
            graph_host.emplace_back(1000.0 * elapsed(before) / options.graph_batch);
            auto device_ms = events.milliseconds();
            if (device_ms < 0.0) { return finish(options, directory, "failed", events.error, 1); }
            graph_device.emplace_back(1000.0 * device_ms / options.graph_batch);
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
    out << ",\"timing_scope\":\"C++ command construction, submission, execution and final synchronization\","
           "\"device_timing_scope\":\"CUDA event stream span after command construction; may include host submission starvation; not isolated kernel time\","
           "\"graph_timing_scope\":\"one replay of N ordered same-output dispatches; CUDA event stream span / N and instrumented synchronized host wall / N\","
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
    out << ",\"graph_build_ms\":" << graph_build_ms << ",\"graph_host_wall_us_per_op\":";
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

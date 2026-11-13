// Correctness-gated L1 synthetic operator benchmarks.
#include "benchmark_runners.h"

#include "ocr_test_utils.h"

#include "ocr/ops/cpu_reference.h"
#include "ocr/ops/ops_attention.h"
#include "ocr/ops/ops_gemm.h"
#include "ocr/ops/ops_norm.h"
#include "ocr/runtime/dispatch_adapter.h"

namespace luisa::ocr::bench {
namespace {

using namespace luisa::compute;

[[nodiscard]] Status compile_status(bool ok, luisa::string_view what) {
    if (ok) { return Status::success(); }
    return Status::error(StatusCode::internal_error,
                         luisa::string{"shader compilation failed: "} + luisa::string{what}, "shader");
}

Row run_gemm(Device &device, const BenchmarkOptions &options, uint32_t m, uint32_t n, uint32_t k,
             luisa::string case_name, luisa::string series) {
    Row row;
    row.workload = make_workload(options, BenchmarkSuite::ops, "gemm", std::move(case_name), "reference", std::move(series));
    if (!workload_selected(options, row.workload)) { return {}; }
    row.workload.shape = {{"m", luisa::format("{}", m)}, {"n", luisa::format("{}", n)}, {"k", luisa::format("{}", k)},
                          {"layout", "A[M,K] B[K,N] C[M,N] fp32 row-major"}, {"epilogue", "none"}};
    row.workload.dispatches = options.dispatch_batch;
    row.workload.flops = 2ull * m * n * k * options.dispatch_batch;
    row.workload.bytes = (2ull * m * n * k + m * n) * sizeof(float) * options.dispatch_batch;
    row.workload.weight_bytes = static_cast<uint64_t>(k) * n * sizeof(float);
    row.workload.byte_accounting =
        "Exact algorithmic FP32 A/B element reads plus C writes for the naive kernel (excluding aux); "
        "not measured cache-line or DRAM transactions. FLOPs=2*M*N*K per dispatch.";

    auto stream = device.create_stream();
    luisa::ocr::test::Lcg lcg;
    lcg.state = static_cast<uint32_t>(row.workload.seed);
    luisa::vector<float> a(static_cast<size_t>(m) * k), b(static_cast<size_t>(k) * n);
    lcg.fill(a, -1.0f, 1.0f);
    lcg.fill(b, -1.0f, 1.0f);
    luisa::vector<float> ref;
    if (auto status = ref_gemm(a, b, {}, m, n, k, RefGemmEpilogue::none, ref); !status.ok()) {
        fail_row(row, std::move(status));
        return row;
    }
    auto ba = luisa::ocr::test::upload(device, stream, luisa::span<const float>{a});
    auto bb = luisa::ocr::test::upload(device, stream, luisa::span<const float>{b});
    auto bc = device.create_buffer<float>(static_cast<size_t>(m) * n);
    auto dummy = device.create_buffer<float>(1);
    auto shader = device.compile(make_gemm_kernel(GemmEpilogue::none));
    if (auto status = compile_status(shader.compile_ok(), "gemm.reference"); !status.ok()) {
        fail_row(row, std::move(status));
        return row;
    }
    auto plan = plan_gemm_dispatch(m, n, k);
    if (!plan.ok()) {
        fail_row(row, std::move(plan.status));
        return row;
    }
    stream << synchronize();
    const auto record = [&](CommandList &list, uint32_t repeats) -> Status {
        for (uint32_t i = 0; i < repeats; i++) {
            if (auto command = dispatch_2d(shader(ba, bb, bc, dummy, m, n, k, plan.value->padded_cols), *plan.value)) {
                list << std::move(command);
            }
        }
        return Status::success();
    };
    luisa::vector<float> got;
    const auto gate = [&]() -> Status {
        auto list = CommandList::create(1);
        if (auto status = record(list, 1); !status.ok()) { return status; }
        stream << list.commit();
        luisa::ocr::test::download(stream, bc, got, ref.size());
        stream << synchronize();
        return check_close(got, ref, 2e-3f, 2e-3f, "gemm.reference");
    };
    time_operator(stream, options, row, record, gate);
    return row;
}

Row run_gemv(Device &device, const BenchmarkOptions &options, uint32_t m, uint32_t n, uint32_t k,
             luisa::string case_name, luisa::string series) {
    Row row;
    row.workload = make_workload(options, BenchmarkSuite::ops, "gemv", std::move(case_name), "reference", std::move(series));
    if (!workload_selected(options, row.workload)) { return {}; }
    row.workload.shape = {{"m", luisa::format("{}", m)}, {"n", luisa::format("{}", n)}, {"k", luisa::format("{}", k)},
                          {"layout", "A[M,K] B[K,N] C[M,N] fp32 row-major"}, {"epilogue", "none"}};
    row.workload.dispatches = options.dispatch_batch;
    row.workload.flops = 2ull * m * n * k * options.dispatch_batch;
    row.workload.bytes = (2ull * m * n * k + m * n) * sizeof(float) * options.dispatch_batch;
    row.workload.weight_bytes = static_cast<uint64_t>(k) * n * sizeof(float);
    row.workload.byte_accounting =
        "Exact algorithmic FP32 A/B element reads plus C writes for the one-column-per-thread GEMV kernel; "
        "not measured cache-line or DRAM transactions. FLOPs=2*M*N*K per dispatch.";

    auto stream = device.create_stream();
    luisa::ocr::test::Lcg lcg;
    lcg.state = static_cast<uint32_t>(row.workload.seed) ^ 0xA53Cu;
    luisa::vector<float> a(static_cast<size_t>(m) * k), b(static_cast<size_t>(k) * n);
    lcg.fill(a, -1.0f, 1.0f);
    lcg.fill(b, -1.0f, 1.0f);
    luisa::vector<float> ref;
    if (auto status = ref_gemm(a, b, {}, m, n, k, RefGemmEpilogue::none, ref); !status.ok()) {
        fail_row(row, std::move(status));
        return row;
    }
    auto ba = luisa::ocr::test::upload(device, stream, luisa::span<const float>{a});
    auto bb = luisa::ocr::test::upload(device, stream, luisa::span<const float>{b});
    auto bc = device.create_buffer<float>(static_cast<size_t>(m) * n);
    auto dummy = device.create_buffer<float>(1);
    auto shader = device.compile(make_gemv_kernel(GemmEpilogue::none));
    if (auto status = compile_status(shader.compile_ok(), "gemv.reference"); !status.ok()) {
        fail_row(row, std::move(status));
        return row;
    }
    auto plan = plan_gemv_dispatch(m, n, k);
    if (!plan.ok()) {
        fail_row(row, std::move(plan.status));
        return row;
    }
    stream << synchronize();
    const auto record = [&](CommandList &list, uint32_t repeats) -> Status {
        for (uint32_t i = 0; i < repeats; i++) {
            if (auto command = dispatch_1d(shader(ba, bb, bc, dummy, m, n, k), *plan.value)) {
                list << std::move(command);
            }
        }
        return Status::success();
    };
    luisa::vector<float> got;
    const auto gate = [&]() -> Status {
        auto list = CommandList::create(1);
        if (auto status = record(list, 1); !status.ok()) { return status; }
        stream << list.commit();
        luisa::ocr::test::download(stream, bc, got, ref.size());
        stream << synchronize();
        return check_close(got, ref, 2e-3f, 2e-3f, "gemv.reference");
    };
    time_operator(stream, options, row, record, gate);
    return row;
}

enum class NormKind { layer, rms };

Row run_norm(Device &device, const BenchmarkOptions &options, NormKind kind, luisa::string_view variant,
             uint32_t rows_count, uint32_t hidden, luisa::string case_name, luisa::string series) {
    const bool layer = kind == NormKind::layer;
    const bool cooperative = variant == "cooperative";
    Row row;
    row.workload = make_workload(options, BenchmarkSuite::ops, layer ? "layer_norm" : "rms_norm",
                                 std::move(case_name), luisa::string{variant}, std::move(series));
    if (!workload_selected(options, row.workload)) { return {}; }
    row.workload.shape = {{"rows", luisa::format("{}", rows_count)}, {"hidden", luisa::format("{}", hidden)},
                          {"weighted", "true"}, {"epsilon", layer ? "1e-5" : "1e-6"}};
    row.workload.dispatches = options.dispatch_batch;
    const auto elements = static_cast<uint64_t>(rows_count) * hidden;
    row.workload.flops = elements * (layer ? 6ull : 4ull) * options.dispatch_batch;
    row.workload.bytes = (2ull * elements + 2ull * hidden + (layer ? hidden : 0ull)) * sizeof(float) * options.dispatch_batch;
    row.workload.weight_bytes = (layer ? 2ull : 1ull) * hidden * sizeof(float);
    row.workload.byte_accounting =
        "Idealized unique FP32 input/output and norm-parameter traffic; approximate arithmetic count "
        "(6*rows*H LayerNorm, 4*rows*H RMSNorm), not hardware-counter measurements.";

    auto stream = device.create_stream();
    luisa::ocr::test::Lcg lcg;
    lcg.state = static_cast<uint32_t>(row.workload.seed) ^ (layer ? 0x11u : 0x22u);
    luisa::vector<float> x(static_cast<size_t>(elements)), gamma(hidden), beta(layer ? hidden : 0u);
    lcg.fill(x, -2.0f, 2.0f);
    lcg.fill(gamma, 0.5f, 1.5f);
    if (layer) { lcg.fill(beta, -0.2f, 0.2f); }
    luisa::vector<float> ref;
    auto reference_status = layer ? ref_layer_norm(x, gamma, beta, rows_count, hidden, 1e-5f, ref)
                                  : ref_rms_norm(x, gamma, rows_count, hidden, 1e-6f, ref);
    if (!reference_status.ok()) {
        fail_row(row, std::move(reference_status));
        return row;
    }
    auto bx = luisa::ocr::test::upload(device, stream, luisa::span<const float>{x});
    auto bg = luisa::ocr::test::upload(device, stream, luisa::span<const float>{gamma});
    auto bb = luisa::ocr::test::upload(device, stream, luisa::span<const float>{beta});
    auto by = device.create_buffer<float>(x.size());
    auto layer_shader = layer ? device.compile(cooperative ? make_layer_norm_cooperative_kernel() : make_layer_norm_kernel()) : Shader1D<Buffer<float>, Buffer<float>, Buffer<float>, Buffer<float>, uint32_t, uint32_t, float, uint32_t>{};
    auto rms_shader = !layer ? device.compile(cooperative ? make_rms_norm_cooperative_kernel() : make_rms_norm_kernel()) : Shader1D<Buffer<float>, Buffer<float>, Buffer<float>, uint32_t, uint32_t, float, uint32_t>{};
    const auto variant_name = cooperative ? "cooperative" : "reference";
    if (auto status = compile_status(layer ? layer_shader.compile_ok() : rms_shader.compile_ok(),
                                     layer ? luisa::format("layer_norm.{}", variant_name) : luisa::format("rms_norm.{}", variant_name)); !status.ok()) {
        fail_row(row, std::move(status));
        return row;
    }
    auto plan = cooperative ? plan_norm_cooperative_dispatch(rows_count) : plan_norm_dispatch(rows_count);
    if (!plan.ok()) {
        fail_row(row, std::move(plan.status));
        return row;
    }
    stream << synchronize();
    const auto record = [&](CommandList &list, uint32_t repeats) -> Status {
        for (uint32_t i = 0; i < repeats; i++) {
            unique_ptr<Command> command;
            if (layer) {
                command = dispatch_1d(layer_shader(bx, by, bg, bb, rows_count, hidden, 1e-5f, 1u), *plan.value);
            } else {
                command = dispatch_1d(rms_shader(bx, by, bg, rows_count, hidden, 1e-6f, 1u), *plan.value);
            }
            if (command) { list << std::move(command); }
        }
        return Status::success();
    };
    luisa::vector<float> got;
    const auto gate = [&]() -> Status {
        auto list = CommandList::create(1);
        if (auto status = record(list, 1); !status.ok()) { return status; }
        stream << list.commit();
        luisa::ocr::test::download(stream, by, got, ref.size());
        stream << synchronize();
        return check_close(got, ref, 1e-4f, 1e-4f,
                           layer ? luisa::format("layer_norm.{}", variant_name) : luisa::format("rms_norm.{}", variant_name));
    };
    time_operator(stream, options, row, record, gate);
    return row;
}

Row run_attention(Device &device, const BenchmarkOptions &options, luisa::string_view variant,
                  uint32_t q_len, uint32_t k_len, uint32_t q_heads, uint32_t kv_heads, uint32_t head_dim,
                  luisa::string case_name, luisa::string series) {
    Row row;
    row.workload = make_workload(options, BenchmarkSuite::ops, "attention_prefill", std::move(case_name),
                                 luisa::string{variant}, std::move(series));
    if (!workload_selected(options, row.workload)) { return {}; }
    row.workload.shape = {{"q_len", luisa::format("{}", q_len)}, {"k_len", luisa::format("{}", k_len)},
                          {"q_heads", luisa::format("{}", q_heads)}, {"kv_heads", luisa::format("{}", kv_heads)},
                          {"head_dim", luisa::format("{}", head_dim)}, {"mask", "causal_end_aligned"},
                          {"softcap", "0"}, {"relative_bias", "false"}};
    row.workload.dispatches = options.dispatch_batch;
    row.workload.flops = 4ull * q_len * k_len * q_heads * head_dim * options.dispatch_batch;
    const auto q_elements = static_cast<uint64_t>(q_len) * q_heads * head_dim;
    const auto kv_elements = static_cast<uint64_t>(k_len) * kv_heads * head_dim;
    row.workload.bytes = (q_elements * 2ull + kv_elements * 2ull) * sizeof(float) * options.dispatch_batch;
    row.workload.byte_accounting =
        "Dense upper-bound FLOPs=4*Q*K*heads*D (QK and PV; masking ignored). Bytes are idealized unique "
        "Q/K/V reads plus output writes, NOT the reference kernel's actual repeated global traffic.";

    AttentionMaskDescriptor mask;
    mask.kind = AttentionMaskKind::causal;
    if (auto status = mask.validate(q_len, k_len, q_heads); !status.ok()) {
        fail_row(row, std::move(status));
        return row;
    }
    auto stream = device.create_stream();
    luisa::ocr::test::Lcg lcg;
    lcg.state = static_cast<uint32_t>(row.workload.seed) ^ 0xA77Au;
    luisa::vector<float> q(static_cast<size_t>(q_elements)), k(static_cast<size_t>(kv_elements)), v(k.size());
    lcg.fill(q, -1.0f, 1.0f);
    lcg.fill(k, -1.0f, 1.0f);
    lcg.fill(v, -1.0f, 1.0f);
    RefAttentionMask ref_mask;
    ref_mask.kind = RefMaskKind::causal;
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    luisa::vector<float> ref;
    if (auto status = ref_attention(q, k, v, q_len, k_len, q_heads, kv_heads, head_dim,
                                    scale, 0.0f, ref_mask, ref); !status.ok()) {
        fail_row(row, std::move(status));
        return row;
    }
    auto bq = luisa::ocr::test::upload(device, stream, luisa::span<const float>{q});
    auto bk = luisa::ocr::test::upload(device, stream, luisa::span<const float>{k});
    auto bv = luisa::ocr::test::upload(device, stream, luisa::span<const float>{v});
    auto dummy_u = device.create_buffer<uint32_t>(1);
    auto dummy_f = device.create_buffer<float>(1);
    const bool online = variant == "online";
    auto shader = device.compile(online ? make_attention_prefill_online_kernel()
                                        : make_attention_prefill_reference_kernel());
    if (auto status = compile_status(shader.compile_ok(), online ? "attention.online" : "attention.reference"); !status.ok()) {
        fail_row(row, std::move(status));
        return row;
    }
    auto plan = plan_attention_prefill_dispatch(q_len, q_heads);
    if (!plan.ok()) {
        fail_row(row, std::move(plan.status));
        return row;
    }
    auto bo = device.create_buffer<float>(q.size());
    stream << synchronize();
    const auto record = [&](CommandList &list, uint32_t repeats) -> Status {
        for (uint32_t i = 0; i < repeats; i++) {
            if (auto command = dispatch_1d(shader(bq, bk, bv, bo, dummy_u, dummy_f, q_len, k_len, q_heads, kv_heads,
                                                  head_dim, scale, 0.0f,
                                                  static_cast<uint32_t>(AttentionMaskKind::causal),
                                                  0u, 0u, 0u, 0u), *plan.value)) {
                list << std::move(command);
            }
        }
        return Status::success();
    };
    luisa::vector<float> got;
    const auto gate = [&]() -> Status {
        auto list = CommandList::create(1);
        if (auto status = record(list, 1); !status.ok()) { return status; }
        stream << list.commit();
        luisa::ocr::test::download(stream, bo, got, ref.size());
        stream << synchronize();
        return check_close(got, ref, 2e-4f, 2e-4f, online ? "attention.online" : "attention.reference");
    };
    time_operator(stream, options, row, record, gate);
    return row;
}

void append(luisa::vector<Row> &rows, Row &&row) {
    if (!row.workload.id.empty()) { rows.emplace_back(std::move(row)); }
}

}// namespace

void append_ops_rows(Device &device, const BenchmarkOptions &options, luisa::vector<Row> &rows) {
    append(rows, run_gemm(device, options, 17, 19, 23, "ragged_17x19x23", "smoke"));
    append(rows, run_gemm(device, options, 128, 256, 128, "scalable_128x256x128", "scalable"));
    append(rows, run_gemv(device, options, 1, 257, 129, "decode_1x257x129", "smoke"));
    append(rows, run_gemv(device, options, 1, 4096, 1024, "decode_1x4096x1024", "scalable"));
    for (auto variant : {luisa::string_view{"reference"}, luisa::string_view{"cooperative"}}) {
        append(rows, run_norm(device, options, NormKind::layer, variant, 32, 4, "model_32x4", "model"));
        append(rows, run_norm(device, options, NormKind::layer, variant, 12, 16, "model_12x16", "model"));
        append(rows, run_norm(device, options, NormKind::layer, variant, 7, 33, "ragged_7x33", "smoke"));
        append(rows, run_norm(device, options, NormKind::layer, variant, 7, 128, "model_7x128", "model"));
        append(rows, run_norm(device, options, NormKind::layer, variant, 1024, 1024, "scalable_1024x1024", "scalable"));
        append(rows, run_norm(device, options, NormKind::rms, variant, 32, 4, "model_32x4", "model"));
        append(rows, run_norm(device, options, NormKind::rms, variant, 12, 16, "model_12x16", "model"));
        append(rows, run_norm(device, options, NormKind::rms, variant, 7, 33, "ragged_7x33", "smoke"));
        append(rows, run_norm(device, options, NormKind::rms, variant, 7, 128, "model_7x128", "model"));
        append(rows, run_norm(device, options, NormKind::rms, variant, 1024, 1024, "scalable_1024x1024", "scalable"));
    }
    for (auto variant : {luisa::string_view{"reference"}, luisa::string_view{"online"}}) {
        append(rows, run_attention(device, options, variant, 5, 13, 4, 2, 16, "causal_tail_5x13", "smoke"));
        append(rows, run_attention(device, options, variant, 32, 96, 4, 2, 32, "causal_32x96", "scalable"));
    }
}

}// namespace luisa::ocr::bench

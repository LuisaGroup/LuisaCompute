// Synthetic dense prefill/decode request benchmarks with exact oracle gates.
#include "benchmark_runners.h"

#include "ocr_decode_test_utils.h"

#include "ocr/runtime/dense_vlm_executor.h"

namespace luisa::ocr::bench {
namespace {

using namespace luisa::compute;

[[nodiscard]] luisa::string decode_variant(const BenchmarkOptions &options) {
    luisa::string variant = "reference";
    if (options.graph && options.fusion) { variant = "fused_residual_rms_norm+cuda_graph"; }
    else if (options.graph) { variant = "cuda_graph"; }
    else if (options.fusion) { variant = "fused_residual_rms_norm"; }
    append_norm_variant(options, variant);
    return variant;
}

struct DenseFixture {
    luisa::ocr::test::TempDir tmp;
    luisa::ocr::test::SyntheticWeights weights;
    WeightStore store;
    DenseVlmExecutor executor;
    DenseExecutorConfig config;
    double setup_ms{0.0};

    [[nodiscard]] Status init(Device &device, const BenchmarkOptions &options, uint32_t chunk_rows) {
        Clock clock;
        weights = luisa::ocr::test::generate_synthetic_weights();
        const auto path = tmp.path / "dense_benchmark.lcocr";
        if (auto status = luisa::ocr::test::write_synthetic_package(path, weights); !status.ok()) { return status; }
        auto stream = device.create_stream();
        if (auto status = luisa::ocr::test::upload_synthetic_weights(device, stream, path, store); !status.ok()) { return status; }
        config.decoder = luisa::ocr::test::synthetic_config();
        config.names = DenseDecoderTensorNames{};
        config.tie_word_embeddings = true;
        config.component = "decoder";
        config.prefill_chunk_rows = chunk_rows;
        config.enable_fused_residual_rms_norm = options.fusion;
        config.enable_cooperative_norm = options.cooperative_norm;
        if (auto status = executor.create(device, stream, config, store); !status.ok()) { return status; }
        setup_ms = clock.toc();
        return Status::success();
    }

    [[nodiscard]] Status make_cache(Device &device, Stream &stream, uint32_t max_seq,
                                    ContiguousKvCache &cache) const {
        KvCacheConfig config;
        config.num_layers = luisa::ocr::test::kSynLayers;
        config.num_kv_heads = luisa::ocr::test::kSynKvHeads;
        config.head_dim = luisa::ocr::test::kSynHeadDim;
        config.max_seq_len = max_seq;
        config.batch = 1;
        config.dtype_bytes = 4;
        return cache.create(device, stream, config);
    }
};

[[nodiscard]] Result<luisa::vector<uint32_t>> oracle_greedy(const luisa::ocr::test::SyntheticWeights &weights,
                                                            luisa::span<const uint32_t> prompt,
                                                            uint32_t max_new) {
    using R = Result<luisa::vector<uint32_t>>;
    luisa::vector<luisa::vector<float>> k_cache, v_cache;
    luisa::ocr::test::OracleResult current;
    if (auto status = luisa::ocr::test::oracle_forward(weights, prompt, {}, 0, 0, 0, k_cache, v_cache, current); !status.ok()) {
        return R::failure(std::move(status));
    }
    luisa::vector<uint32_t> tokens;
    const auto vocab = weights.vocab == 0 ? luisa::ocr::test::kSynVocab : weights.vocab;
    // The initial oracle returns one logits row PER PROMPT TOKEN. Only the
    // final row selects the first generated token; flattened argmax is invalid.
    current.logits = luisa::vector<float>(current.logits.end() - vocab, current.logits.end());
    for (uint32_t step = 0; step < max_new; step++) {
        const auto token = luisa::ocr::test::oracle_argmax(current.logits);
        tokens.push_back(token);
        if (token == luisa::ocr::test::kSynEos || step + 1u == max_new) { break; }
        luisa::vector<uint32_t> input{token};
        luisa::ocr::test::OracleResult next;
        if (auto status = luisa::ocr::test::oracle_forward(weights, input, {}, 0, 0,
                                                    static_cast<uint32_t>(prompt.size()) + step,
                                                    k_cache, v_cache, next); !status.ok()) {
            return R::failure(std::move(status));
        }
        current = std::move(next);
        (void)vocab;
    }
    return R::success(std::move(tokens));
}

Row run_generate(Device &device, const BenchmarkOptions &options, uint32_t max_new,
                 luisa::string case_name, luisa::string series) {
    Row row;
    row.workload = make_workload(options, BenchmarkSuite::decode, "dense_synthetic", std::move(case_name),
                                 decode_variant(options), std::move(series));
    if (!workload_selected(options, row.workload)) { return {}; }
    DenseFixture fixture;
    if (auto status = fixture.init(device, options, 4); !status.ok()) {
        fail_row(row, std::move(status));
        return row;
    }
    add_counter(row, "synthetic_setup_ms_including_package_upload_and_shader_compile", fixture.setup_ms);
    const luisa::vector<uint32_t> prompt = {3, 7, 11, 5, 9, 2, 14, 1, 4, 6, 8, 10};
    auto expected = oracle_greedy(fixture.weights, prompt, max_new);
    if (!expected.ok()) {
        fail_row(row, std::move(expected.status));
        return row;
    }
    row.workload.prompt_tokens = prompt.size();
    row.workload.output_tokens = expected.value->size();
    row.workload.weight_bytes = fixture.executor.weight_bytes();
    row.workload.shape = {{"prompt_tokens", luisa::format("{}", prompt.size())},
                          {"max_new_tokens", luisa::format("{}", max_new)},
                          {"prefill_chunk_rows", "4"}, {"sampling", "greedy"},
                          {"cache", "fresh_contiguous_fp32_per_request"}};
    row.workload.byte_accounting = "Resident bytes come from executor/cache accounting; no universal free-VRAM query is used.";
    add_counter(row, "resident_weight_bytes", fixture.executor.weight_bytes());
    add_counter(row, "workspace_bytes", fixture.executor.workspace_bytes());
    add_counter(row, "shader_compiles", fixture.executor.shader_registry().total_compiles());
    add_counter(row, "expected_generated_steps", static_cast<uint64_t>(expected.value->size()));
    add_counter(row, "expected_eos", expected.value->back() == luisa::ocr::test::kSynEos ? "true" : "false");
    add_counter(row, "expected_truncation", expected.value->back() == luisa::ocr::test::kSynEos ? "false" : "true");

    auto stream = device.create_stream();
    luisa::optional<DenseDecodeResult> latest;
    uint64_t latest_cache_bytes = 0;
    const auto request = [&]() -> Status {
        ContiguousKvCache cache;
        if (auto status = fixture.make_cache(device, stream,
                                             static_cast<uint32_t>(prompt.size()) + max_new + 1u, cache); !status.ok()) {
            return status;
        }
        latest_cache_bytes = cache.device_bytes();
        DenseDecodeOptions decode_options;
        decode_options.sampling.greedy = true;
        decode_options.max_new_tokens = max_new;
        decode_options.enable_cuda_graph = options.graph;
        DenseDecodeController controller;
        if (auto status = controller.create(fixture.executor, nullptr, decode_options); !status.ok()) { return status; }
        DecodeRequest decode_request;
        decode_request.prompt_tokens = prompt;
        auto result = controller.generate(stream, decode_request, cache);
        if (!result.ok()) { return result.status; }
        latest = std::move(*result.value);
        return Status::success();
    };
    const auto gate = [&]() -> Status {
        if (!latest) {
            return Status::error(StatusCode::internal_error, "decode produced no result", "correctness");
        }
        if (latest->tokens != *expected.value) {
            return Status::error(StatusCode::internal_error,
                                 luisa::format("greedy token mismatch (got {}, expected {})",
                                               latest->tokens.size(), expected.value->size()),
                                 "correctness");
        }
        if (latest->steps != latest->tokens.size()) {
            return Status::error(StatusCode::internal_error, "decode step count differs from emitted token count", "steps");
        }
        const bool expect_eos = expected.value->back() == luisa::ocr::test::kSynEos;
        if (latest->hit_eos != expect_eos || latest->truncated == expect_eos || latest->hit_stop_string) {
            return Status::error(StatusCode::internal_error, "EOS/truncation/stop flags differ from oracle", "termination");
        }
        row.truncated = latest->truncated;
        row.fallback_used = latest->cuda_graph_fallbacks != 0u;
        if (!options.graph && (latest->cuda_graph_steps != 0u || latest->cuda_graph_updates != 0u)) {
            return Status::error(StatusCode::internal_error, "graph counters changed in the non-graph arm", "cuda_graph");
        }
        if (options.graph && latest->cuda_graph_fallbacks == 0u && latest->cuda_graph_steps == 0u) {
            return Status::error(StatusCode::internal_error, "graph arm executed no captured graph step", "cuda_graph");
        }
        return Status::success();
    };
    time_requests(stream, options, row, request, gate);
    add_counter(row, "kv_cache_bytes", latest_cache_bytes);
    if (latest) {
        add_counter(row, "cuda_graph_steps", latest->cuda_graph_steps);
        add_counter(row, "cuda_graph_updates", latest->cuda_graph_updates);
        add_counter(row, "cuda_graph_fallbacks", latest->cuda_graph_fallbacks);
        add_counter(row, "generated_steps", latest->steps);
    }
    return row;
}

Row run_prefill(Device &device, const BenchmarkOptions &options, uint32_t chunk_rows,
                luisa::string case_name, luisa::string series) {
    Row row;
    auto variant = luisa::string{options.fusion ? "fused_residual_rms_norm" : "reference"};
    append_norm_variant(options, variant);
    row.workload = make_workload(options, BenchmarkSuite::decode, "dense_synthetic", std::move(case_name),
                                 std::move(variant), std::move(series));
    if (!workload_selected(options, row.workload)) { return {}; }
    DenseFixture fixture;
    if (auto status = fixture.init(device, options, chunk_rows); !status.ok()) {
        fail_row(row, std::move(status));
        return row;
    }
    add_counter(row, "synthetic_setup_ms_including_package_upload_and_shader_compile", fixture.setup_ms);
    add_counter(row, "graph_scope", options.graph ? "decode_only_prefill_ungraphed" : "disabled");
    const luisa::vector<uint32_t> prompt = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 1, 4};
    luisa::vector<luisa::vector<float>> k_cache, v_cache;
    luisa::ocr::test::OracleResult oracle;
    if (auto status = luisa::ocr::test::oracle_forward(fixture.weights, prompt, {}, 0, 0, 0, k_cache, v_cache, oracle); !status.ok()) {
        fail_row(row, std::move(status));
        return row;
    }
    luisa::vector<float> expected(oracle.logits.end() - luisa::ocr::test::kSynVocab, oracle.logits.end());
    row.workload.prompt_tokens = prompt.size();
    row.workload.weight_bytes = fixture.executor.weight_bytes();
    row.workload.shape = {{"prompt_tokens", luisa::format("{}", prompt.size())},
                          {"prefill_chunk_rows", luisa::format("{}", chunk_rows)},
                          {"cache", "fresh_contiguous_fp32_per_request"}};
    row.workload.byte_accounting = "Resident bytes come from executor/cache accounting; prefill has no generated-token throughput claim.";
    add_counter(row, "resident_weight_bytes", fixture.executor.weight_bytes());
    add_counter(row, "workspace_bytes", fixture.executor.workspace_bytes());
    add_counter(row, "shader_compiles", fixture.executor.shader_registry().total_compiles());

    auto stream = device.create_stream();
    luisa::vector<float> latest_logits;
    uint64_t latest_cache_bytes = 0;
    const auto request = [&]() -> Status {
        ContiguousKvCache cache;
        if (auto status = fixture.make_cache(device, stream,
                                             static_cast<uint32_t>(prompt.size()) + 1u, cache); !status.ok()) {
            return status;
        }
        latest_cache_bytes = cache.device_bytes();
        latest_logits.assign(luisa::ocr::test::kSynVocab, 0.0f);
        DecodeRequest decode_request;
        decode_request.prompt_tokens = prompt;
        auto list = CommandList::create();
        if (auto status = fixture.executor.record_prefill(list, decode_request, cache, latest_logits); !status.ok()) {
            return status;
        }
        stream << list.commit() << synchronize();
        return Status::success();
    };
    const auto gate = [&]() -> Status {
        return check_close(latest_logits, expected, 3e-4f, 3e-4f, "dense prefill last-row logits");
    };
    time_requests(stream, options, row, request, gate);
    add_counter(row, "kv_cache_bytes", latest_cache_bytes);
    return row;
}

void append(luisa::vector<Row> &rows, Row &&row) {
    if (!row.workload.id.empty()) { rows.emplace_back(std::move(row)); }
}

}// namespace

void append_decode_rows(Device &device, const BenchmarkOptions &options, luisa::vector<Row> &rows) {
    append(rows, run_prefill(device, options, 4, "prefill_chunked_12", "smoke"));
    append(rows, run_generate(device, options, 4, "generate_greedy_4", "smoke"));
    append(rows, run_generate(device, options, 8, "generate_greedy_8", "scalable"));
}

}// namespace luisa::ocr::bench

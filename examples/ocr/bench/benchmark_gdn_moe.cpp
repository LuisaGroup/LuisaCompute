// Complete synthetic hybrid-GDN and MoE request benchmarks.
#include "benchmark_runners.h"

#include "ocr_gdn_test_utils.h"
#include "ocr_deepseek_test_utils.h"

#include "ocr/models/ovis_ocr2.h"
#include "ocr/runtime/hybrid_gdn_executor.h"
#include "ocr/runtime/moe_vlm_executor.h"

namespace luisa::ocr::bench {
namespace {

using namespace luisa::compute;

[[nodiscard]] uint32_t greedy_token(luisa::span<const float> logits) {
    LUISA_ASSERT(!logits.empty(), "argmax requires a nonempty logits row");
    uint32_t best = 0;
    for (uint32_t i = 1; i < logits.size(); i++) {
        if (logits[i] > logits[best]) { best = i; }
    }
    return best;
}

[[nodiscard]] P7ImageInput p7_image(uint32_t width, uint32_t height, const P7SliceConfig &config,
                                    uint32_t seed = 7u) {
    auto crop = luisa::ocr::test::p7_make_crop(width, height, 0, seed);
    P7ImageInput image;
    image.pixels = std::move(crop.pixels);
    image.source_width = width;
    image.source_height = height;
    auto packed = p7_pack_image(width, height, config.packing, config.connector);
    LUISA_ASSERT(packed.ok(), "synthetic P7 image packing must succeed");
    image.packed = *packed.value;
    return image;
}

[[nodiscard]] P7ImageInput deepseek_image(const DeepSeekOcr2Config &config, uint32_t width, uint32_t height,
                                          uint32_t seed = 7u) {
    auto crop = luisa::ocr::test::p7_make_crop(width, height, 0, seed);
    P7ImageInput image;
    image.pixels = std::move(crop.pixels);
    image.source_width = width;
    image.source_height = height;
    P7ConnectorConfig connector;
    connector.kind = P7ConnectorKind::mlp;
    connector.intermediate_size = config.vision.connector_intermediate_size;
    auto packed = p7_pack_image(width, height, config.packing, connector);
    LUISA_ASSERT(packed.ok(), "synthetic DeepSeek image packing must succeed");
    image.packed = *packed.value;
    return image;
}

struct OvisFixture {
    luisa::ocr::test::P7TempDir tmp;
    OvisOcr2Config cfg;
    luisa::ocr::test::P7SyntheticWeights vision_weights;
    luisa::ocr::test::HybridSyntheticWeights decoder_weights;
    WeightStore store;
    HybridVlmGraph graph;
    double setup_ms{0.0};

    [[nodiscard]] Status init(Device &device) {
        Clock clock;
        cfg = ovis_ocr2_synthetic_config();
        if (auto status = cfg.validate(); !status.ok()) { return status; }
        vision_weights = luisa::ocr::test::generate_p7_weights(cfg.vision);
        decoder_weights = luisa::ocr::test::generate_hybrid_weights(cfg.decoder);
        const auto vision_path = tmp.path / "vision.lcocr";
        const auto decoder_path = tmp.path / "decoder.lcocr";
        if (auto status = luisa::ocr::test::write_p7_packages(vision_path, tmp.path / "dense_unused.lcocr",
                                                       cfg.vision, cfg.vision_names, vision_weights,
                                                       "ovis_ocr2"); !status.ok()) { return status; }
        if (auto status = luisa::ocr::test::write_hybrid_decoder_package(decoder_path, cfg.decoder,
                                                                  cfg.decoder_names, decoder_weights); !status.ok()) { return status; }
        {
            auto vision = PackageReader::open(vision_path);
            auto decoder = PackageReader::open(decoder_path);
            if (!vision.ok()) { return vision.status; }
            if (!decoder.ok()) { return decoder.status; }
            if (auto status = ovis_ocr2_validate_tensors((*vision.value)->tensors(), (*decoder.value)->tensors(), cfg);
                !status.ok()) { return status; }
        }
        auto stream = device.create_stream();
        {
            auto reader = PackageReader::open(vision_path);
            if (!reader.ok()) { return reader.status; }
            if (auto status = store.upload_component_sync(device, stream, **reader.value, "vision"); !status.ok()) { return status; }
        }
        {
            auto reader = PackageReader::open(decoder_path);
            if (!reader.ok()) { return reader.status; }
            if (auto status = store.upload_component_sync(device, stream, **reader.value, "decoder"); !status.ok()) { return status; }
        }
        HybridVlmGraphConfig graph_config;
        graph_config.vision = cfg.vision;
        graph_config.vision_names = cfg.vision_names;
        graph_config.decoder = cfg.decoder;
        graph_config.decoder_names = cfg.decoder_names;
        graph_config.max_seq_len = cfg.max_seq_len;
        if (auto status = graph.create(device, stream, graph_config, store); !status.ok()) { return status; }
        setup_ms = clock.toc();
        return Status::success();
    }
};

struct DeepSeekFixture {
    luisa::ocr::test::P7TempDir tmp;
    DeepSeekOcr2Config cfg;
    luisa::ocr::test::DeepSeekSyntheticWeights weights;
    WeightStore store;
    MoeVlmExecutor executor;
    double setup_ms{0.0};

    [[nodiscard]] Status init(Device &device) {
        Clock clock;
        cfg = deepseek_ocr2_synthetic_config();
        if (auto status = cfg.validate(); !status.ok()) { return status; }
        weights = luisa::ocr::test::generate_deepseek_weights(cfg);
        const auto vision_path = tmp.path / "vision.lcocr";
        const auto decoder_path = tmp.path / "decoder.lcocr";
        if (auto status = luisa::ocr::test::write_deepseek_vision_package(vision_path, cfg, weights); !status.ok()) { return status; }
        if (auto status = luisa::ocr::test::write_deepseek_decoder_package(decoder_path, cfg, weights); !status.ok()) { return status; }
        {
            auto vision = PackageReader::open(vision_path);
            auto decoder = PackageReader::open(decoder_path);
            if (!vision.ok()) { return vision.status; }
            if (!decoder.ok()) { return decoder.status; }
            if (auto status = deepseek_ocr2_validate_tensors((*vision.value)->tensors(), (*decoder.value)->tensors(), cfg);
                !status.ok()) { return status; }
        }
        auto stream = device.create_stream();
        if (auto status = luisa::ocr::test::upload_deepseek_weights(device, stream, vision_path, decoder_path, store); !status.ok()) {
            return status;
        }
        if (auto status = executor.create(device, stream, cfg, store); !status.ok()) { return status; }
        setup_ms = clock.toc();
        return Status::success();
    }
};

void append(luisa::vector<Row> &rows, Row &&row) {
    if (!row.workload.id.empty()) { rows.emplace_back(std::move(row)); }
}

}// namespace

Row run_ovis_row(Device &device, const BenchmarkOptions &options, BenchmarkSuite suite,
                 luisa::string case_name, luisa::string series) {
    const bool chunked = options.variant == "chunked";
    Row row;
    row.workload = make_workload(options, suite, "ovis_ocr2", std::move(case_name),
                                 chunked ? "chunked" : "recurrent", std::move(series));
    row.workload.options.push_back({"tier", "1"});
    if (!workload_selected(options, row.workload)) { return {}; }
    if (options.graph || options.fusion) {
        fail_row(row, Status::error(StatusCode::unsupported,
                                    "GDN/Ovis supports only recurrent/chunked prefill A/B; CUDA graph and dense fusion are unsupported",
                                    options.graph ? "graph" : "fusion"),
                 RowStatus::refused);
        return row;
    }
    OvisFixture fixture;
    if (auto status = fixture.init(device); !status.ok()) {
        fail_row(row, std::move(status));
        return row;
    }
    add_counter(row, "synthetic_setup_ms_including_package_upload_and_shader_compile", fixture.setup_ms);
    auto image = p7_image(16, 16, fixture.cfg.vision);
    const auto visual_tokens = static_cast<uint32_t>(image.packed.visual_tokens);
    luisa::vector<uint32_t> prompt = {1u, 2u};
    const uint32_t mm_start = static_cast<uint32_t>(prompt.size());
    prompt.insert(prompt.end(), visual_tokens, 0u);
    prompt.push_back(3u);
    prompt.push_back(4u);
    constexpr uint32_t max_new = 4u;
    constexpr uint32_t eos = 23u;

    auto pixels = luisa::ocr::test::p7_oracle_preprocess(image.pixels, image.source_width, image.source_height,
                                                  image.packed.resized_width, image.packed.resized_height,
                                                  fixture.cfg.vision.mean, fixture.cfg.vision.std_dev);
    luisa::ocr::test::P7VisionOracle vision_oracle;
    if (auto status = luisa::ocr::test::p7_oracle_vision(fixture.cfg.vision, fixture.vision_weights,
                                                  pixels, image.packed, vision_oracle); !status.ok()) {
        fail_row(row, std::move(status));
        return row;
    }
    luisa::ocr::test::HybridCpuOracle oracle(fixture.cfg.decoder, fixture.decoder_weights);
    luisa::vector<float> logits;
    for (const auto token : prompt) { logits = oracle.step(token, vision_oracle.features, mm_start, visual_tokens); }
    luisa::vector<uint32_t> expected;
    for (uint32_t step = 0; step < max_new; step++) {
        const auto token = greedy_token(logits);
        expected.push_back(token);
        if (token == eos || step + 1u == max_new) { break; }
        logits = oracle.step(token);
    }
    row.workload.pages = 1;
    row.workload.patches = image.packed.patch_count;
    row.workload.prompt_tokens = prompt.size();
    row.workload.output_tokens = expected.size();
    row.workload.weight_bytes = fixture.graph.weight_bytes();
    row.workload.shape = {{"source_width", "16"}, {"source_height", "16"},
                          {"visual_tokens", luisa::format("{}", visual_tokens)},
                          {"prompt_tokens", luisa::format("{}", prompt.size())},
                          {"max_new_tokens", luisa::format("{}", max_new)},
                          {"gdn_prefill_chunk", chunked ? "4" : "0_recurrent"},
                          {"sampling", "greedy"}, {"cache_reset", "full_hybrid_cache_each_request"}};
    row.workload.byte_accounting = "Actual graph weight/workspace/hybrid-cache bytes; GDN state includes recurrent and Conv ring plus full-layer KV and metadata.";
    add_counter(row, "resident_weight_bytes", fixture.graph.weight_bytes());
    add_counter(row, "synthetic_setup_ms_including_package_upload_and_shader_compile", fixture.setup_ms);

    auto stream = device.create_stream();
    luisa::optional<HybridVlmRunResult> latest;
    const auto request = [&]() -> Status {
        HybridDecodeOptions decode_options;
        decode_options.sampling.greedy = true;
        decode_options.max_new_tokens = max_new;
        decode_options.prefill_chunk_size = chunked ? 4u : 0u;
        decode_options.eos_ids = {eos};
        auto result = fixture.graph.run(stream, luisa::span<const P7ImageInput>{&image, 1},
                                        prompt, mm_start, decode_options);
        if (!result.ok()) { return result.status; }
        latest = std::move(*result.value);
        return Status::success();
    };
    const auto gate = [&]() -> Status {
        if (!latest || latest->visual_tokens != visual_tokens || latest->decode.tokens != expected) {
            return Status::error(StatusCode::internal_error, "Ovis visual-token count or greedy tokens differ from independent oracle", "correctness");
        }
        if (latest->decode.steps != latest->decode.tokens.size() || latest->decode.hit_stop_string) {
            return Status::error(StatusCode::internal_error, "Ovis decode step/termination counters mismatch", "correctness");
        }
        const bool expect_eos = expected.back() == eos;
        if (latest->decode.hit_eos != expect_eos || latest->decode.truncated == expect_eos) {
            return Status::error(StatusCode::internal_error, "Ovis EOS/truncation mismatch", "correctness");
        }
        row.truncated = latest->decode.truncated;
        luisa::vector<float> features;
        if (auto status = fixture.graph.download_vision_features(stream, features); !status.ok()) { return status; }
        stream << synchronize();
        return check_close(features, vision_oracle.features, 5e-3f, 5e-3f, "Ovis vision merger features");
    };
    time_requests(stream, options, row, request, gate);
    if (latest) {
        add_counter(row, "workspace_bytes", latest->workspace_bytes);
        add_counter(row, "hybrid_cache_bytes", latest->cache_bytes);
        add_counter(row, "visual_tokens", latest->visual_tokens);
        add_counter(row, "generated_steps", latest->decode.steps);
    }
    return row;
}

Row run_deepseek_row(Device &device, const BenchmarkOptions &options, BenchmarkSuite suite,
                     luisa::string case_name, luisa::string series) {
    Row row;
    row.workload = make_workload(options, suite, "deepseek_ocr_2", std::move(case_name),
                                 "reference", std::move(series));
    row.workload.options.push_back({"tier", "1"});
    if (!workload_selected(options, row.workload)) { return {}; }
    if (options.graph || options.fusion) {
        fail_row(row, Status::error(StatusCode::unsupported,
                                    "DeepSeek MoE has no CUDA-graph or dense fused-norm benchmark path",
                                    options.graph ? "graph" : "fusion"),
                 RowStatus::refused);
        return row;
    }
    DeepSeekFixture fixture;
    if (auto status = fixture.init(device); !status.ok()) {
        fail_row(row, std::move(status));
        return row;
    }
    add_counter(row, "synthetic_setup_ms_including_package_upload_and_shader_compile", fixture.setup_ms);
    auto image = deepseek_image(fixture.cfg, 16, 16);
    const auto visual_tokens = fixture.cfg.vision.num_queries;
    luisa::vector<uint32_t> prompt = {1u, 2u};
    const uint32_t mm_start = static_cast<uint32_t>(prompt.size());
    prompt.insert(prompt.end(), visual_tokens, 0u);
    prompt.push_back(3u);
    prompt.push_back(4u);
    constexpr uint32_t max_new = 3u;
    constexpr uint32_t eos = 23u;

    auto pixels = luisa::ocr::test::p7_oracle_preprocess(image.pixels, image.source_width, image.source_height,
                                                  image.packed.resized_width, image.packed.resized_height,
                                                  fixture.cfg.vision.mean, fixture.cfg.vision.std_dev);
    luisa::ocr::test::DeepSeekVisionOracle vision_oracle;
    if (auto status = luisa::ocr::test::ds_oracle_vision(fixture.cfg, fixture.weights, pixels, image.packed, vision_oracle);
        !status.ok()) {
        fail_row(row, std::move(status));
        return row;
    }
    const auto &decoder = fixture.cfg.decoder.decoder;
    const auto hidden = decoder.hidden_size;
    luisa::vector<float> embeddings(static_cast<size_t>(prompt.size()) * hidden);
    for (size_t i = 0; i < prompt.size(); i++) {
        for (uint32_t h = 0; h < hidden; h++) {
            embeddings[i * hidden + h] = fixture.weights.embed[static_cast<size_t>(prompt[i]) * hidden + h];
        }
    }
    for (uint32_t i = 0; i < visual_tokens; i++) {
        for (uint32_t h = 0; h < hidden; h++) {
            embeddings[static_cast<size_t>(mm_start + i) * hidden + h] = vision_oracle.features[static_cast<size_t>(i) * hidden + h];
        }
    }
    luisa::ocr::test::DeepSeekDecoderOracle oracle;
    luisa::vector<float> logits;
    if (auto status = luisa::ocr::test::ds_oracle_decoder_step(fixture.cfg, fixture.weights, oracle, embeddings,
                                                        static_cast<uint32_t>(prompt.size()), 0, logits); !status.ok()) {
        fail_row(row, std::move(status));
        return row;
    }
    luisa::vector<uint32_t> expected;
    for (uint32_t step = 0; step < max_new; step++) {
        const auto token = greedy_token(logits);
        expected.push_back(token);
        if (token == eos || step + 1u == max_new) { break; }
        luisa::vector<float> embedding(hidden);
        for (uint32_t h = 0; h < hidden; h++) { embedding[h] = fixture.weights.embed[static_cast<size_t>(token) * hidden + h]; }
        if (auto status = luisa::ocr::test::ds_oracle_decoder_step(fixture.cfg, fixture.weights, oracle, embedding, 1,
                                                            static_cast<uint32_t>(prompt.size()) + step, logits); !status.ok()) {
            fail_row(row, std::move(status));
            return row;
        }
    }
    row.workload.pages = 1;
    row.workload.patches = image.packed.patch_count;
    row.workload.prompt_tokens = prompt.size();
    row.workload.output_tokens = expected.size();
    row.workload.weight_bytes = fixture.executor.weight_bytes();
    row.workload.shape = {{"source_width", "16"}, {"source_height", "16"},
                          {"visual_queries", luisa::format("{}", visual_tokens)},
                          {"prompt_tokens", luisa::format("{}", prompt.size())},
                          {"max_new_tokens", luisa::format("{}", max_new)},
                          {"experts", "64_top6_two_shared"}, {"routing", "actual_device_routing"},
                          {"visual_mask", "custom_visual_query_block_size1"}, {"sampling", "greedy"}};
    row.workload.byte_accounting = "Actual resident weights/workspace/KV from executor result; active-expert counts are measured per layer, never inferred from requested top-k alone.";
    add_counter(row, "resident_weight_bytes", fixture.executor.weight_bytes());

    auto stream = device.create_stream();
    luisa::optional<MoeRunResult> latest;
    const auto request = [&]() -> Status {
        MoeDecodeOptions decode_options;
        decode_options.sampling.greedy = true;
        decode_options.max_new_tokens = max_new;
        decode_options.eos_ids = {eos};
        auto result = fixture.executor.run(stream, image, prompt, mm_start, decode_options);
        if (!result.ok()) { return result.status; }
        latest = std::move(*result.value);
        return Status::success();
    };
    const auto gate = [&]() -> Status {
        if (!latest || latest->visual_tokens != visual_tokens || latest->decode.tokens != expected) {
            return Status::error(StatusCode::internal_error, "DeepSeek visual count or greedy tokens differ from independent oracle", "correctness");
        }
        const bool expect_eos = expected.back() == eos;
        if (latest->decode.hit_eos != expect_eos || latest->decode.truncated == expect_eos ||
            latest->decode.steps != latest->decode.tokens.size()) {
            return Status::error(StatusCode::internal_error, "DeepSeek termination/step counters mismatch", "correctness");
        }
        row.truncated = latest->decode.truncated;
        luisa::vector<float> features;
        if (auto status = fixture.executor.download_vision_features(stream, features); !status.ok()) { return status; }
        stream << synchronize();
        if (auto status = check_close(features, vision_oracle.features, 5e-3f, 5e-3f, "DeepSeek connector features"); !status.ok()) {
            return status;
        }
        for (uint32_t layer = 0; layer < decoder.num_layers; layer++) {
            luisa::vector<uint32_t> counts, perm_token, slot_pos;
            if (auto status = fixture.executor.download_permutation(stream, layer, counts, perm_token, slot_pos); !status.ok()) {
                return status;
            }
            stream << synchronize();
            uint64_t active = 0;
            for (const auto count : counts) { active += count != 0u ? 1u : 0u; }
            add_counter(row, luisa::format("active_experts_layer_{}", layer), active);
            add_counter(row, luisa::format("permutation_entries_layer_{}", layer), static_cast<uint64_t>(perm_token.size()));
        }
        return Status::success();
    };
    time_requests(stream, options, row, request, gate);
    if (latest) {
        add_counter(row, "workspace_bytes", latest->workspace_bytes);
        add_counter(row, "kv_cache_bytes", latest->cache_bytes);
        add_counter(row, "visual_tokens", latest->visual_tokens);
        add_counter(row, "generated_steps", latest->decode.steps);
    }
    return row;
}

void append_gdn_rows(Device &device, const BenchmarkOptions &options, luisa::vector<Row> &rows) {
    append(rows, run_ovis_row(device, options, BenchmarkSuite::gdn, "full_recurrent_or_chunked", "smoke"));
}

void append_moe_rows(Device &device, const BenchmarkOptions &options, luisa::vector<Row> &rows) {
    append(rows, run_deepseek_row(device, options, BenchmarkSuite::moe, "full_custom_mask_moe", "smoke"));
}

}// namespace luisa::ocr::bench

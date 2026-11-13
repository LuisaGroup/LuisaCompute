// Complete Qwen3-VL and FireRed synthetic request benchmarks.
#include "benchmark_runners.h"

#include "ocr_qwen3_test_utils.h"

#include "ocr/models/fire_red_ocr.h"

namespace luisa::ocr::bench {
namespace {

using namespace luisa::compute;

[[nodiscard]] Qwen3VlImageInput make_image(uint32_t width, uint32_t height, uint32_t seed) {
    Qwen3VlPackingSpec spec;
    spec.patch_size = luisa::ocr::test::kQ3Patch;
    spec.spatial_merge_size = luisa::ocr::test::kQ3Merge;
    spec.min_pixels = 16;
    spec.max_pixels = 4096;
    auto packed = qwen3_vl_pack_image(width, height, spec);
    LUISA_ASSERT(packed.ok(), "synthetic Qwen3 image packing must succeed");
    Qwen3VlImageInput image;
    image.source_width = width;
    image.source_height = height;
    image.packed = *packed.value;
    image.pixels.resize(static_cast<size_t>(width) * height * 3u);
    luisa::ocr::test::Lcg lcg;
    lcg.state = seed;
    for (auto &pixel : image.pixels) { pixel = static_cast<uint8_t>(lcg.next() % 256u); }
    return image;
}

struct QwenFixture {
    luisa::ocr::test::Qwen3TempDir tmp;
    luisa::ocr::test::Qwen3SyntheticWeights weights;
    WeightStore store;
    Qwen3VlExecutor executor;
    Qwen3VlExecutorConfig config;
    double setup_ms{0.0};

    [[nodiscard]] Status init(Device &device, const BenchmarkOptions &options, bool fire_red) {
        Clock clock;
        weights = luisa::ocr::test::generate_qwen3_synthetic_weights();
        if (fire_red) {
            const auto model = fire_red_ocr_synthetic_config();
            config.model = model.graph;
            config.names = model.names;
        } else {
            config.model = qwen3_vl_synthetic_config();
            config.names = Qwen3VlTensorNames{};
        }
        const auto vision_path = tmp.path / "vision.lcocr";
        const auto decoder_path = tmp.path / "decoder.lcocr";
        if (auto status = luisa::ocr::test::write_qwen3_synthetic_packages(vision_path, decoder_path,
                                                                    weights, config.names); !status.ok()) { return status; }
        auto stream = device.create_stream();
        if (auto status = luisa::ocr::test::upload_qwen3_synthetic_weights(device, stream, vision_path,
                                                                    decoder_path, store); !status.ok()) { return status; }
        config.vision_component = "vision";
        config.decoder_component = "decoder";
        config.max_vision_patches = 4096;
        config.enable_fused_residual_rms_norm = options.fusion;
        config.enable_cooperative_norm = options.cooperative_norm;
        if (auto status = executor.create(device, stream, config, store); !status.ok()) { return status; }
        setup_ms = clock.toc();
        return Status::success();
    }

    [[nodiscard]] Status make_cache(Device &device, Stream &stream, uint32_t max_seq,
                                    ContiguousKvCache &cache) const {
        KvCacheConfig config;
        config.num_layers = luisa::ocr::test::kQ3Layers;
        config.num_kv_heads = luisa::ocr::test::kQ3KvHeads;
        config.head_dim = luisa::ocr::test::kQ3HeadDim;
        config.max_seq_len = max_seq;
        config.batch = 1;
        config.dtype_bytes = 4;
        return cache.create(device, stream, config);
    }
};

Row run_qwen(Device &device, const BenchmarkOptions &options, bool fire_red,
             luisa::string case_name, luisa::string series, uint32_t tier) {
    luisa::string variant = "reference";
    if (options.graph && options.fusion) { variant = "fused_residual_rms_norm+cuda_graph"; }
    else if (options.graph) { variant = "cuda_graph"; }
    else if (options.fusion) { variant = "fused_residual_rms_norm"; }
    append_norm_variant(options, variant);
    Row row;
    row.workload = make_workload(options, BenchmarkSuite::model,
                                 fire_red ? "fire_red_ocr" : "qwen3_vl_2b_instruct",
                                 std::move(case_name), std::move(variant), std::move(series));
    row.workload.options.push_back({"tier", luisa::format("{}", tier)});
    if (!workload_selected(options, row.workload)) { return {}; }
    QwenFixture fixture;
    if (auto status = fixture.init(device, options, fire_red); !status.ok()) {
        fail_row(row, std::move(status));
        return row;
    }
    add_counter(row, "synthetic_setup_ms_including_package_upload_and_shader_compile", fixture.setup_ms);
    add_counter(row, "resident_weight_bytes", fixture.executor.weight_bytes());
    add_counter(row, "vision_workspace_bytes", fixture.executor.vision_workspace_bytes());
    add_counter(row, "decoder_workspace_bytes", fixture.executor.decoder_workspace_bytes());
    add_counter(row, "vision_shader_compiles", fixture.executor.shader_registry().total_compiles());
    add_counter(row, "decoder_shader_compiles", fixture.executor.text_executor().shader_registry().total_compiles());
    const auto image = make_image(fire_red ? 12u : 8u, fire_red ? 6u : 8u, fire_red ? 11u : 7u);
    const auto merged = static_cast<uint32_t>(image.packed.merged_count);
    Qwen3VlPrompt prompt;
    prompt.token_ids = {1u, 2u};
    const uint32_t image_start = static_cast<uint32_t>(prompt.token_ids.size());
    prompt.token_ids.insert(prompt.token_ids.end(), merged, 3u);
    prompt.token_ids.push_back(4u);
    prompt.token_ids.push_back(5u);
    prompt.image_ranges.push_back({image_start, merged});
    constexpr uint32_t max_new = 4u;

    auto preprocessed = qwen3_vl_preprocess_image_host(image, fixture.config.mean, fixture.config.std_dev);
    if (!preprocessed.ok()) {
        fail_row(row, std::move(preprocessed.status));
        return row;
    }
    luisa::ocr::test::Qwen3VisionOracle vision_oracle;
    if (auto status = luisa::ocr::test::oracle_vision(fixture.weights, preprocessed.value->pixels,
                                               image.packed, vision_oracle); !status.ok()) {
        fail_row(row, std::move(status));
        return row;
    }
    Qwen3VlPromptSegment segments[3];
    segments[0].text_tokens = 2;
    segments[1].image = &image.packed;
    segments[2].text_tokens = 2;
    auto positions = qwen3_vl_build_position_ids(segments);
    if (!positions.ok()) {
        fail_row(row, std::move(positions.status));
        return row;
    }
    luisa::vector<luisa::vector<float>> k_cache, v_cache;
    luisa::vector<luisa::vector<float>> deepstack_host;
    for (const auto &features : vision_oracle.ds_features) { deepstack_host.push_back(features); }
    luisa::ocr::test::Qwen3TextOracle text_oracle;
    if (auto status = luisa::ocr::test::oracle_qwen3_forward(fixture.weights, prompt.token_ids, positions.value->ids,
                                                      vision_oracle.features, image_start, merged, deepstack_host,
                                                      0, k_cache, v_cache, text_oracle); !status.ok()) {
        fail_row(row, std::move(status));
        return row;
    }
    luisa::vector<uint32_t> expected;
    auto decode_position = positions.value->next_text_position;
    uint32_t input = luisa::ocr::test::oracle_argmax(luisa::span<const float>{
        text_oracle.logits.data() + static_cast<size_t>(prompt.token_ids.size() - 1u) * luisa::ocr::test::kQ3Vocab,
        luisa::ocr::test::kQ3Vocab});
    for (uint32_t step = 0; step < max_new; step++) {
        expected.push_back(input);
        if (input == luisa::ocr::test::kQ3Eos || step + 1u == max_new) { break; }
        const luisa::vector<int32_t> step_position = {decode_position, decode_position, decode_position};
        const luisa::vector<uint32_t> step_tokens = {input};
        luisa::ocr::test::Qwen3TextOracle next;
        if (auto status = luisa::ocr::test::oracle_qwen3_forward(fixture.weights, step_tokens, step_position,
                                                          {}, 0, 0, {}, static_cast<uint32_t>(prompt.token_ids.size()) + step,
                                                          k_cache, v_cache, next); !status.ok()) {
            fail_row(row, std::move(status));
            return row;
        }
        input = luisa::ocr::test::oracle_argmax(next.logits);
        decode_position++;
    }
    row.workload.pages = 1;
    row.workload.patches = image.packed.patch_count;
    row.workload.prompt_tokens = prompt.token_ids.size();
    row.workload.output_tokens = expected.size();
    row.workload.weight_bytes = fixture.executor.weight_bytes();
    row.workload.shape = {{"source_width", luisa::format("{}", image.source_width)},
                          {"source_height", luisa::format("{}", image.source_height)},
                          {"merged_visual_tokens", luisa::format("{}", merged)},
                          {"prompt_tokens", luisa::format("{}", prompt.token_ids.size())},
                          {"max_new_tokens", luisa::format("{}", max_new)},
                          {"sampling", "greedy"}, {"deepstack_injections", luisa::format("{}", vision_oracle.ds_features.size())},
                          {"rope", "interleaved_mrope_3_axes"}, {"cache", "fresh_contiguous_fp32_per_request"}};
    row.workload.byte_accounting = "Actual executor-owned and borrowed weights, vision/decoder workspaces, and fresh KV cache; synthetic setup time is reported separately from warm requests.";

    auto stream = device.create_stream();
    luisa::optional<DenseDecodeResult> latest;
    uint64_t latest_cache_bytes = 0;
    StageDurations stage;
    Clock stage_clock;
    const auto request = [&]() -> Status {
        if (options.instrument) { stage_clock.tic(); }
        fixture.executor.reset_vision_workspace();
        Qwen3VlVisionOutput vision;
        {
            auto list = CommandList::create();
            auto recorded = fixture.executor.record_vision(list, luisa::span<const Qwen3VlImageInput>{&image, 1});
            if (!recorded.ok()) { return recorded.status; }
            vision = *recorded.value;
            stream << list.commit() << synchronize();
        }
        if (options.instrument) {
            stage.vision_ms = stage_clock.toc();
            stage_clock.tic();
        }
        luisa::vector<int32_t> position_ids;
        auto decode_request = fixture.executor.build_decode_request(prompt, vision,
                                                                    luisa::span<const Qwen3VlImageInput>{&image, 1},
                                                                    position_ids);
        if (!decode_request.ok()) { return decode_request.status; }
        if (options.instrument) {
            stage.preprocess_ms = stage_clock.toc();
            stage_clock.tic();
        }
        ContiguousKvCache cache;
        if (auto status = fixture.make_cache(device, stream,
                                             static_cast<uint32_t>(prompt.token_ids.size()) + max_new + 1u,
                                             cache); !status.ok()) { return status; }
        latest_cache_bytes = cache.device_bytes();
        DenseDecodeOptions decode_options;
        decode_options.sampling.greedy = true;
        decode_options.max_new_tokens = max_new;
        decode_options.prompt_position_ids = std::move(position_ids);
        decode_options.next_position = {positions.value->next_text_position,
                                        positions.value->next_text_position,
                                        positions.value->next_text_position};
        decode_options.deepstack_features = vision.deepstack_features;
        decode_options.enable_cuda_graph = options.graph;
        DenseDecodeController controller;
        if (auto status = controller.create(fixture.executor.text_executor(), nullptr, decode_options); !status.ok()) {
            return status;
        }
        auto result = controller.generate(stream, *decode_request.value, cache);
        if (!result.ok()) { return result.status; }
        latest = std::move(*result.value);
        if (options.instrument) { stage.decode_ms = stage_clock.toc(); }
        return Status::success();
    };
    const auto gate = [&]() -> Status {
        if (!latest || latest->tokens != expected || latest->steps != latest->tokens.size()) {
            return Status::error(StatusCode::internal_error, "Qwen3 greedy tokens/steps differ from independent oracle", "correctness");
        }
        const bool expect_eos = expected.back() == luisa::ocr::test::kQ3Eos;
        if (latest->hit_eos != expect_eos || latest->truncated == expect_eos || latest->hit_stop_string) {
            return Status::error(StatusCode::internal_error, "Qwen3 termination flags mismatch", "correctness");
        }
        row.truncated = latest->truncated;
        row.fallback_used = latest->cuda_graph_fallbacks != 0u;
        if (!options.graph && (latest->cuda_graph_steps != 0u || latest->cuda_graph_updates != 0u)) {
            return Status::error(StatusCode::internal_error, "Qwen3 non-graph arm used graph steps", "cuda_graph");
        }
        if (options.graph && latest->cuda_graph_fallbacks == 0u && latest->cuda_graph_steps == 0u) {
            return Status::error(StatusCode::internal_error, "Qwen3 graph arm executed no graph step", "cuda_graph");
        }
        luisa::vector<float> features;
        if (auto status = fixture.executor.download_vision_features(stream, features); !status.ok()) { return status; }
        stream << synchronize();
        if (auto status = check_close(features, vision_oracle.features, 3e-3f, 3e-3f, "Qwen3 merger features"); !status.ok()) {
            return status;
        }
        for (size_t i = 0; i < vision_oracle.ds_features.size(); i++) {
            luisa::vector<float> deepstack;
            if (auto status = fixture.executor.download_deepstack_features(stream, static_cast<uint32_t>(i), deepstack);
                !status.ok()) { return status; }
            stream << synchronize();
            if (auto status = check_close(deepstack, vision_oracle.ds_features[i], 3e-3f, 3e-3f,
                                          "Qwen3 DeepStack features"); !status.ok()) { return status; }
        }
        return Status::success();
    };
    time_requests(stream, options, row, request, gate,
                  options.instrument ? &stage : nullptr);
    add_counter(row, "kv_cache_bytes", latest_cache_bytes);
    if (latest) {
        add_counter(row, "cuda_graph_steps", latest->cuda_graph_steps);
        add_counter(row, "cuda_graph_updates", latest->cuda_graph_updates);
        add_counter(row, "cuda_graph_fallbacks", latest->cuda_graph_fallbacks);
        add_counter(row, "generated_steps", latest->steps);
    }
    return row;
}

void append(luisa::vector<Row> &rows, Row &&row) {
    if (!row.workload.id.empty()) { rows.emplace_back(std::move(row)); }
}

}// namespace

void append_model_rows(Device &device, const BenchmarkOptions &options, luisa::vector<Row> &rows) {
    append(rows, run_qwen(device, options, false, "full_vision_deepstack_decode", "smoke", 1));
    append(rows, run_pp_row(device, options, BenchmarkSuite::model, PpOcrSize::medium, "tier1", 1));
    append(rows, run_pp_row(device, options, BenchmarkSuite::model, PpOcrSize::tiny, "tier2", 2));
    append(rows, run_pp_row(device, options, BenchmarkSuite::model, PpOcrSize::small, "tier2", 2));
    append_paddle_model_rows(device, options, rows);
    append_opendoc_model_rows(device, options, rows);
    append(rows, run_deepseek_row(device, options, BenchmarkSuite::model, "full_vision_moe_decode", "tier1"));
    append(rows, run_ovis_row(device, options, BenchmarkSuite::model, "full_vision_hybrid_decode", "tier1"));
    append_p7_model_rows(device, options, rows);
    append_surya_model_rows(device, options, rows);
    append(rows, run_qwen(device, options, true, "full_shared_qwen_graph", "tier2", 2));
}

}// namespace luisa::ocr::bench

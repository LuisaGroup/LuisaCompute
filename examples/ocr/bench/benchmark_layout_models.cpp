// Synthetic PaddleOCR-VL and OpenDoc region/model request benchmarks.
#include "benchmark_runners.h"

#include "ocr_p6_test_utils.h"
#include "ocr_p7_test_utils.h"

namespace luisa::ocr::bench {
namespace {

using namespace luisa::compute;

[[nodiscard]] luisa::string dense_variant(const BenchmarkOptions &options) {
    luisa::string variant = "reference";
    if (options.graph && options.fusion) { variant = "fused_residual_rms_norm+cuda_graph"; }
    else if (options.graph) { variant = "cuda_graph"; }
    else if (options.fusion) { variant = "fused_residual_rms_norm"; }
    append_norm_variant(options, variant);
    return variant;
}

struct PaddleFixture {
    luisa::ocr::test::P6TempDir tmp;
    PaddleOcrVlConfig cfg;
    PaddleOcrVlTensorNames names;
    luisa::ocr::test::PaddleSyntheticWeights weights;
    WeightStore store;
    PaddleOcrVlRecognizer recognizer;
    PaddleOcrVlRecognizerConfig config;
    double setup_ms{0.0};

    [[nodiscard]] Status init(Device &device, const BenchmarkOptions &options) {
        Clock clock;
        cfg = paddle_ocr_vl_synthetic_config();
        if (auto status = cfg.validate(); !status.ok()) { return status; }
        names = PaddleOcrVlTensorNames{};
        weights = luisa::ocr::test::generate_paddle_weights(cfg);
        const auto vision_path = tmp.path / "vision.lcocr";
        const auto decoder_path = tmp.path / "decoder.lcocr";
        if (auto status = luisa::ocr::test::write_paddle_packages(vision_path, decoder_path, cfg, names, weights); !status.ok()) {
            return status;
        }
        auto stream = device.create_stream();
        if (auto status = luisa::ocr::test::upload_paddle_weights(device, stream, vision_path, decoder_path, store); !status.ok()) {
            return status;
        }
        config.model = cfg;
        config.names = names;
        config.max_vision_patches = 4096;
        config.enable_fused_residual_rms_norm = options.fusion;
        config.enable_cooperative_norm = options.cooperative_norm;
        if (auto status = recognizer.create(device, stream, config, store); !status.ok()) { return status; }
        setup_ms = clock.toc();
        return Status::success();
    }
};

struct OpenDocFixture {
    luisa::ocr::test::P6TempDir tmp;
    luisa::ocr::test::OpenDocFixtureWeights weights;
    WeightStore store;
    OpenDocRecognizer recognizer;
    OpenDocRecognizerConfig config;
    double setup_ms{0.0};

    [[nodiscard]] Status init(Device &device) {
        Clock clock;
        if (auto status = luisa::ocr::test::generate_opendoc_weights(weights); !status.ok()) { return status; }
        const auto layout_path = tmp.path / "layout.lcocr";
        const auto recognizer_path = tmp.path / "unirec.lcocr";
        if (auto status = luisa::ocr::test::write_opendoc_packages(layout_path, recognizer_path, weights); !status.ok()) {
            return status;
        }
        auto stream = device.create_stream();
        if (auto status = luisa::ocr::test::upload_opendoc_weights(device, stream, layout_path, recognizer_path, store); !status.ok()) {
            return status;
        }
        config.model.layout = weights.layout_cfg;
        config.model.layout_names = weights.layout_names;
        config.model.recognizer.recognizer = weights.rec_cfg.recognizer;
        config.model.recognizer.dictionary = luisa::ocr::test::pp_synthetic_dictionary();
        config.model.recognizer_names = UniRecTensorNames{};
        config.recognizer_component = "unirec";
        if (auto status = recognizer.create(device, stream, config, store); !status.ok()) { return status; }
        setup_ms = clock.toc();
        return Status::success();
    }
};

Row run_paddle(Device &device, const BenchmarkOptions &options) {
    Row row;
    row.workload = make_workload(options, BenchmarkSuite::model, "paddle_ocr_vl_1_6",
                                 "region_vision_connector_ernie", dense_variant(options), "tier1");
    row.workload.options.push_back({"tier", "1"});
    if (!workload_selected(options, row.workload)) { return {}; }
    PaddleFixture fixture;
    if (auto status = fixture.init(device, options); !status.ok()) {
        fail_row(row, std::move(status));
        return row;
    }
    add_counter(row, "synthetic_setup_ms_including_package_upload_and_shader_compile", fixture.setup_ms);
    add_counter(row, "resident_weight_bytes", fixture.recognizer.weight_bytes());
    add_counter(row, "workspace_bytes", fixture.recognizer.workspace_bytes());
    auto crop = luisa::ocr::test::p7_make_crop(8, 8, 0, 11u);
    auto packed = paddle_pack_image(8, 8, fixture.cfg.packing, fixture.cfg.connector.max_visual_tokens);
    if (!packed.ok()) {
        fail_row(row, std::move(packed.status));
        return row;
    }
    auto pixels = luisa::ocr::test::p6_oracle_preprocess(crop.pixels, 8, 8,
                                                  packed.value->resized_width, packed.value->resized_height,
                                                  fixture.config.mean, fixture.config.std_dev);
    luisa::ocr::test::PaddleVisionOracle vision_oracle;
    if (auto status = luisa::ocr::test::paddle_oracle_vision(fixture.cfg, fixture.weights, pixels, *packed.value, vision_oracle);
        !status.ok()) {
        fail_row(row, std::move(status));
        return row;
    }
    const auto visual_tokens = static_cast<uint32_t>(packed.value->visual_tokens);
    luisa::vector<uint32_t> prompt = fixture.config.prompt_prefix;
    const uint32_t mm_start = static_cast<uint32_t>(prompt.size());
    prompt.insert(prompt.end(), visual_tokens, 0u);
    prompt.insert(prompt.end(), fixture.config.prompt_suffix.begin(), fixture.config.prompt_suffix.end());
    luisa::vector<luisa::vector<float>> k_cache, v_cache;
    luisa::ocr::test::PaddleTextOracle text_oracle;
    if (auto status = luisa::ocr::test::paddle_oracle_forward(fixture.cfg, fixture.weights, prompt,
                                                       vision_oracle.features, mm_start, visual_tokens,
                                                       0, k_cache, v_cache, text_oracle); !status.ok()) {
        fail_row(row, std::move(status));
        return row;
    }
    constexpr uint32_t max_new = 4u;
    constexpr uint32_t eos = 31u;
    luisa::vector<uint32_t> expected;
    uint32_t input = luisa::ocr::test::oracle_argmax(luisa::span<const float>{
        text_oracle.logits.data() + static_cast<size_t>(prompt.size() - 1u) * fixture.cfg.text.vocab_size,
        fixture.cfg.text.vocab_size});
    for (uint32_t step = 0; step < max_new; step++) {
        expected.push_back(input);
        if (input == eos || step + 1u == max_new) { break; }
        const luisa::vector<uint32_t> step_tokens{input};
        luisa::ocr::test::PaddleTextOracle next;
        if (auto status = luisa::ocr::test::paddle_oracle_forward(fixture.cfg, fixture.weights, step_tokens,
                                                           {}, 0, 0, static_cast<uint32_t>(prompt.size()) + step,
                                                           k_cache, v_cache, next); !status.ok()) {
            fail_row(row, std::move(status));
            return row;
        }
        input = luisa::ocr::test::oracle_argmax(next.logits);
    }
    row.workload.pages = 1;
    row.workload.regions = 1;
    row.workload.patches = packed.value->patch_count;
    row.workload.prompt_tokens = prompt.size();
    row.workload.output_tokens = expected.size();
    row.workload.weight_bytes = fixture.recognizer.weight_bytes();
    row.workload.shape = {{"crop_width", "8"}, {"crop_height", "8"},
                          {"native_patches", luisa::format("{}", packed.value->patch_count)},
                          {"reduced_visual_tokens", luisa::format("{}", visual_tokens)},
                          {"prompt_tokens", luisa::format("{}", prompt.size())},
                          {"max_new_tokens", luisa::format("{}", max_new)},
                          {"sampling", "greedy"}, {"region_kv", "fresh_contiguous_per_region"}};
    row.workload.byte_accounting = "Actual recognizer weight/workspace/KV high-water bytes and actual native patch/token counts.";

    auto stream = device.create_stream();
    luisa::optional<RawRegionOutput> latest;
    uint64_t graph_steps = 0, graph_updates = 0, graph_fallbacks = 0;
    const auto request = [&]() -> Status {
        const auto old_steps = fixture.recognizer.cuda_graph_steps();
        const auto old_updates = fixture.recognizer.cuda_graph_updates();
        const auto old_fallbacks = fixture.recognizer.cuda_graph_fallbacks();
        RegionRecognizeOptions recognize_options;
        recognize_options.max_output_tokens = max_new;
        recognize_options.enable_cuda_graph = options.graph;
        auto result = fixture.recognizer.recognize_batch(stream, luisa::span<const RegionCropInput>{&crop, 1},
                                                         recognize_options);
        if (!result.ok()) { return result.status; }
        if (result.value->size() != 1u) {
            return Status::error(StatusCode::internal_error, "Paddle recognizer returned an unexpected region count", "regions");
        }
        graph_steps = fixture.recognizer.cuda_graph_steps() - old_steps;
        graph_updates = fixture.recognizer.cuda_graph_updates() - old_updates;
        graph_fallbacks = fixture.recognizer.cuda_graph_fallbacks() - old_fallbacks;
        latest = std::move((*result.value)[0]);
        return Status::success();
    };
    const auto gate = [&]() -> Status {
        if (!latest || latest->tokens != expected) {
            return Status::error(StatusCode::internal_error, "Paddle greedy tokens differ from independent oracle", "correctness");
        }
        const bool expect_eos = expected.back() == eos;
        if (latest->hit_eos != expect_eos || latest->truncated == expect_eos) {
            return Status::error(StatusCode::internal_error, "Paddle EOS/truncation mismatch", "correctness");
        }
        row.truncated = latest->truncated;
        row.fallback_used = graph_fallbacks != 0u;
        if (!options.graph && (graph_steps != 0u || graph_updates != 0u)) {
            return Status::error(StatusCode::internal_error, "Paddle non-graph arm used graph steps", "cuda_graph");
        }
        if (options.graph && graph_fallbacks == 0u && graph_steps == 0u) {
            return Status::error(StatusCode::internal_error, "Paddle graph arm executed no graph step", "cuda_graph");
        }
        luisa::vector<float> features;
        if (auto status = fixture.recognizer.download_connector_features(stream, features); !status.ok()) { return status; }
        stream << synchronize();
        return check_close(features, vision_oracle.features, 3e-3f, 3e-3f, "Paddle connector features");
    };
    time_requests(stream, options, row, request, gate);
    add_counter(row, "kv_cache_peak_bytes", fixture.recognizer.kv_cache_peak_bytes());
    add_counter(row, "cuda_graph_steps", graph_steps);
    add_counter(row, "cuda_graph_updates", graph_updates);
    add_counter(row, "cuda_graph_fallbacks", graph_fallbacks);
    if (latest) {
        add_counter(row, "actual_native_patches", latest->native_patches);
        add_counter(row, "generated_steps", latest->tokens.size());
    }
    return row;
}

Row run_opendoc(Device &device, const BenchmarkOptions &options) {
    Row row;
    row.workload = make_workload(options, BenchmarkSuite::model, "open_doc_0_1b",
                                 "unirec_region_ctc", "reference", "tier1");
    row.workload.options.push_back({"tier", "1"});
    if (!workload_selected(options, row.workload)) { return {}; }
    if (options.graph || options.fusion) {
        fail_row(row, Status::error(StatusCode::unsupported,
                                    "OpenDoc UniRec is CNN+CTC; dense graph/fusion switches do not apply",
                                    options.graph ? "graph" : "fusion"),
                 RowStatus::refused);
        return row;
    }
    OpenDocFixture fixture;
    if (auto status = fixture.init(device); !status.ok()) {
        fail_row(row, std::move(status));
        return row;
    }
    add_counter(row, "synthetic_setup_ms_including_package_upload_and_shader_compile", fixture.setup_ms);
    add_counter(row, "resident_weight_bytes", fixture.recognizer.weight_bytes());
    add_counter(row, "workspace_bytes", fixture.recognizer.workspace_bytes());
    const auto &recognizer_config = fixture.weights.rec_cfg.recognizer;
    auto crop = luisa::ocr::test::p7_make_crop(recognizer_config.input_width, recognizer_config.input_height, 0, 11u);
    auto pixels = luisa::ocr::test::p6_oracle_preprocess(crop.pixels, recognizer_config.input_width, recognizer_config.input_height,
                                                  recognizer_config.input_width, recognizer_config.input_height,
                                                  fixture.config.mean, fixture.config.std_dev);
    luisa::ocr::test::PpOracleRecognizer oracle;
    if (auto status = luisa::ocr::test::opendoc_oracle_recognizer(fixture.weights.rec_cfg, fixture.weights.rec_weights,
                                                           pixels, fixture.config.ctc.temperature, oracle); !status.ok()) {
        fail_row(row, std::move(status));
        return row;
    }
    auto decoded = output::ctc_decode_logits(oracle.logits, oracle.frames,
                                             fixture.config.model.recognizer.dictionary, fixture.config.ctc);
    if (!decoded.ok()) {
        fail_row(row, std::move(decoded.status));
        return row;
    }
    row.workload.pages = 1;
    row.workload.regions = 1;
    row.workload.output_tokens = decoded.value->ids.size();
    row.workload.weight_bytes = fixture.recognizer.weight_bytes();
    row.workload.shape = {{"crop_width", luisa::format("{}", recognizer_config.input_width)},
                          {"crop_height", luisa::format("{}", recognizer_config.input_height)},
                          {"frames", luisa::format("{}", oracle.frames)},
                          {"decoder", "cnn_svtr_ctc_no_autoregressive_kv"}};
    row.workload.byte_accounting = "Actual UniRec recognizer weight/workspace bytes; output tokens are exact collapsed CTC class ids.";

    auto stream = device.create_stream();
    luisa::optional<RawRegionOutput> latest;
    const auto request = [&]() -> Status {
        RegionRecognizeOptions recognize_options;
        recognize_options.max_output_tokens = 64;
        auto result = fixture.recognizer.recognize_batch(stream, luisa::span<const RegionCropInput>{&crop, 1},
                                                         recognize_options);
        if (!result.ok()) { return result.status; }
        if (result.value->size() != 1u) {
            return Status::error(StatusCode::internal_error, "OpenDoc recognizer returned an unexpected region count", "regions");
        }
        latest = std::move((*result.value)[0]);
        return Status::success();
    };
    const auto gate = [&]() -> Status {
        if (!latest || latest->tokens != decoded.value->ids || latest->text != decoded.value->text) {
            return Status::error(StatusCode::internal_error, "OpenDoc collapsed CTC ids/text differ from independent oracle", "correctness");
        }
        if (latest->has_confidence != decoded.value->has_confidence ||
            (decoded.value->has_confidence && std::abs(latest->confidence - decoded.value->confidence) >= 2e-3f)) {
            return Status::error(StatusCode::internal_error, "OpenDoc CTC confidence mismatch", "correctness");
        }
        return Status::success();
    };
    time_requests(stream, options, row, request, gate);
    return row;
}

void append(luisa::vector<Row> &rows, Row &&row) {
    if (!row.workload.id.empty()) { rows.emplace_back(std::move(row)); }
}

}// namespace

void append_paddle_model_rows(Device &device, const BenchmarkOptions &options, luisa::vector<Row> &rows) {
    append(rows, run_paddle(device, options));
}

void append_opendoc_model_rows(Device &device, const BenchmarkOptions &options, luisa::vector<Row> &rows) {
    append(rows, run_opendoc(device, options));
}

}// namespace luisa::ocr::bench

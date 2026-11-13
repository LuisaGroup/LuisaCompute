// Parameterized P7/P8 synthetic model-slice benchmarks. Each model's declared
// config/tensor map and independent P7/P8 oracle remain distinct.
#include "benchmark_runners.h"

#include "ocr_p7_test_utils.h"
#include "ocr_p8_test_utils.h"

#include "ocr/models/dots_mocr.h"
#include "ocr/models/glm_ocr.h"
#include "ocr/models/got_ocr2.h"
#include "ocr/models/hunyuan_ocr.h"
#include "ocr/models/mineru.h"
#include "ocr/models/olm_ocr2.h"
#include "ocr/models/tele_ocr.h"

namespace luisa::ocr::bench {
namespace {

using namespace luisa::compute;

enum class ModelKind { tele, glm, mineru, hunyuan, dots, got, olm };

struct ModelSpec {
    P7SliceConfig graph;
    P7TensorNames names;
    luisa::string target;
    luisa::string series;
    uint32_t tier{1};
    bool p8{false};
    DenseLinearStorage storage{DenseLinearStorage::fp32};
    uint32_t group_size{0};
    bool quantized{false};
};

[[nodiscard]] ModelSpec make_spec(ModelKind kind, bool quantized) {
    ModelSpec spec;
    spec.quantized = quantized;
    switch (kind) {
        case ModelKind::tele: {
            const auto cfg = tele_ocr_synthetic_config();
            spec.graph = cfg.graph;
            spec.names = cfg.names;
            spec.target = "tele_ocr";
            spec.series = "tier2";
            spec.tier = 2;
            break;
        }
        case ModelKind::glm: {
            const auto cfg = glm_ocr_synthetic_config();
            spec.graph = cfg.graph;
            spec.names = cfg.names;
            spec.target = "glm_ocr";
            spec.series = "tier2";
            spec.tier = 2;
            break;
        }
        case ModelKind::mineru: {
            const auto cfg = mineru_synthetic_config();
            spec.graph = cfg.graph;
            spec.names = cfg.names;
            spec.target = "mineru_2_5_pro_2605_1_2b";
            spec.series = "tier2";
            spec.tier = 2;
            break;
        }
        case ModelKind::hunyuan: {
            const auto cfg = hunyuan_ocr_synthetic_config();
            spec.graph = cfg.graph;
            spec.names = cfg.names;
            spec.target = "hunyuan_ocr_1_5";
            spec.series = "smoke";
            spec.tier = 1;
            break;
        }
        case ModelKind::dots: {
            const auto cfg = dots_mocr_synthetic_config();
            spec.graph = cfg.graph;
            spec.names = cfg.names;
            spec.target = "dots_mocr";
            spec.series = "tier1";
            spec.tier = 1;
            spec.p8 = true;
            if (quantized) {
                spec.storage = DenseLinearStorage::int4_sym;
                spec.group_size = 64;
            }
            break;
        }
        case ModelKind::got: {
            const auto cfg = got_ocr2_synthetic_config();
            spec.graph = cfg.graph;
            spec.names = cfg.names;
            spec.target = "got_ocr2_0";
            spec.series = "tier1";
            spec.tier = 1;
            spec.p8 = true;
            break;
        }
        case ModelKind::olm: {
            const auto cfg = olm_ocr2_synthetic_config();
            spec.graph = cfg.graph;
            spec.names = cfg.names;
            spec.target = "olm_ocr_2";
            spec.series = "tier1";
            spec.tier = 1;
            spec.p8 = true;
            if (quantized) {
                spec.storage = DenseLinearStorage::int4_sym;
                spec.group_size = 128;
            }
            break;
        }
    }
    spec.graph.linear_storage = spec.storage;
    spec.graph.linear_group_size = spec.group_size;
    if (spec.storage != DenseLinearStorage::fp32) { spec.graph.tie_word_embeddings = false; }
    return spec;
}

struct P7Fixture {
    luisa::ocr::test::P7TempDir tmp;
    ModelSpec spec;
    luisa::ocr::test::P7SyntheticWeights weights7;
    luisa::ocr::test::P8SyntheticWeights weights8;
    WeightStore store;
    P7VlmExecutor executor;
    double setup_ms{0.0};

    [[nodiscard]] Status init(Device &device, const BenchmarkOptions &options, ModelSpec model_spec) {
        Clock clock;
        spec = std::move(model_spec);
        if (auto status = spec.graph.validate(); !status.ok()) { return status; }
        spec.graph.enable_fused_residual_rms_norm = options.fusion;
        spec.graph.enable_cooperative_norm = options.cooperative_norm;
        const auto vision_path = tmp.path / "vision.lcocr";
        const auto decoder_path = tmp.path / "decoder.lcocr";
        if (spec.p8) {
            weights8 = luisa::ocr::test::generate_p8_weights(spec.graph, spec.storage, spec.group_size);
            if (auto status = luisa::ocr::test::write_p8_packages(vision_path, decoder_path, spec.graph,
                                                           spec.names, weights8, spec.target); !status.ok()) { return status; }
        } else {
            weights7 = luisa::ocr::test::generate_p7_weights(spec.graph);
            if (auto status = luisa::ocr::test::write_p7_packages(vision_path, decoder_path, spec.graph,
                                                           spec.names, weights7, spec.target); !status.ok()) { return status; }
        }
        // Preserve the package-level storage gates for the quantized families.
        if (spec.target == "dots_mocr" || spec.target == "olm_ocr_2") {
            auto vision = PackageReader::open(vision_path);
            auto decoder = PackageReader::open(decoder_path);
            if (!vision.ok()) { return vision.status; }
            if (!decoder.ok()) { return decoder.status; }
            if (spec.target == "dots_mocr") {
                DotsMocrConfig cfg = dots_mocr_synthetic_config();
                cfg.graph = spec.graph;
                cfg.names = spec.names;
                cfg.linear_storage = spec.storage;
                cfg.linear_group_size = spec.group_size;
                if (auto status = dots_mocr_validate_packages(cfg, (*vision.value)->tensors(), (*decoder.value)->tensors());
                    !status.ok()) { return status; }
            } else {
                OlmOcr2Config cfg = olm_ocr2_synthetic_config();
                cfg.graph = spec.graph;
                cfg.names = spec.names;
                cfg.linear_storage = spec.storage;
                cfg.linear_group_size = spec.group_size;
                if (auto status = olm_ocr2_validate_packages(cfg, (*vision.value)->tensors(), (*decoder.value)->tensors());
                    !status.ok()) { return status; }
            }
        }
        auto stream = device.create_stream();
        if (auto status = luisa::ocr::test::upload_p7_weights(device, stream, vision_path, decoder_path, store); !status.ok()) {
            return status;
        }
        if (auto status = executor.create(device, stream, spec.graph, spec.names, store); !status.ok()) { return status; }
        setup_ms = clock.toc();
        return Status::success();
    }
};

[[nodiscard]] luisa::string model_variant(const BenchmarkOptions &options, const ModelSpec &spec) {
    luisa::string variant = spec.quantized ?
                                luisa::format("w4a32_int4_sym_g{}", spec.group_size) :
                                luisa::string{"reference"};
    if (options.graph && options.fusion) { variant += "+fused_residual_rms_norm+cuda_graph"; }
    else if (options.graph) { variant += "+cuda_graph"; }
    else if (options.fusion) { variant += "+fused_residual_rms_norm"; }
    append_norm_variant(options, variant);
    return variant;
}

Row run_p7(Device &device, const BenchmarkOptions &options, ModelKind kind, bool quantized) {
    auto spec = make_spec(kind, quantized);
    Row row;
    row.workload = make_workload(options, BenchmarkSuite::model, spec.target,
                                 "full_vision_dense_decode", model_variant(options, spec), spec.series);
    row.workload.options.push_back({"tier", luisa::format("{}", spec.tier)});
    if (!workload_selected(options, row.workload)) { return {}; }
    P7Fixture fixture;
    if (auto status = fixture.init(device, options, std::move(spec)); !status.ok()) {
        fail_row(row, std::move(status));
        return row;
    }
    const auto &graph = fixture.spec.graph;
    add_counter(row, "synthetic_setup_ms_including_package_upload_and_shader_compile", fixture.setup_ms);
    add_counter(row, "resident_weight_bytes", fixture.executor.weight_bytes());
    add_counter(row, "vision_workspace_bytes", fixture.executor.vision_workspace_bytes());
    add_counter(row, "decoder_workspace_bytes", fixture.executor.decoder_workspace_bytes());
    auto crop = luisa::ocr::test::p7_make_crop(8, 8, 0, 11u);
    P7ImageInput image;
    image.pixels = std::move(crop.pixels);
    image.source_width = 8;
    image.source_height = 8;
    auto packed = p7_pack_image(8, 8, graph.packing, graph.connector);
    if (!packed.ok()) {
        fail_row(row, std::move(packed.status));
        return row;
    }
    image.packed = *packed.value;
    const auto visual_tokens = static_cast<uint32_t>(image.packed.visual_tokens);
    luisa::vector<uint32_t> prompt = {1u};
    const uint32_t mm_start = 1;
    prompt.insert(prompt.end(), visual_tokens, 0u);
    prompt.push_back(2u);
    auto positions = p7_build_position_ids(graph, static_cast<uint32_t>(prompt.size()), mm_start,
                                           visual_tokens, image.packed);
    if (!positions.ok()) {
        fail_row(row, std::move(positions.status));
        return row;
    }
    auto pixels = luisa::ocr::test::p7_oracle_preprocess(image.pixels, 8, 8,
                                                  image.packed.resized_width, image.packed.resized_height,
                                                  graph.mean, graph.std_dev);
    luisa::vector<float> vision_features;
    luisa::vector<luisa::vector<float>> k_cache, v_cache;
    luisa::vector<float> logits;
    if (fixture.spec.p8) {
        luisa::ocr::test::P8VisionOracle vision;
        if (auto status = luisa::ocr::test::p8_oracle_vision(graph, fixture.weights8, pixels, image.packed, vision); !status.ok()) {
            fail_row(row, std::move(status));
            return row;
        }
        vision_features = vision.features;
        luisa::ocr::test::P8TextOracle text;
        if (auto status = luisa::ocr::test::p8_oracle_text_forward(graph, fixture.weights8, prompt, positions.value->ids,
                                                            vision.features, mm_start, visual_tokens, 0,
                                                            k_cache, v_cache, text); !status.ok()) {
            fail_row(row, std::move(status));
            return row;
        }
        logits.assign(text.logits.end() - graph.text.vocab_size, text.logits.end());
    } else {
        luisa::ocr::test::P7VisionOracle vision;
        if (auto status = luisa::ocr::test::p7_oracle_vision(graph, fixture.weights7, pixels, image.packed, vision); !status.ok()) {
            fail_row(row, std::move(status));
            return row;
        }
        vision_features = vision.features;
        luisa::ocr::test::P7TextOracle text;
        if (auto status = luisa::ocr::test::p7_oracle_text_forward(graph, fixture.weights7, prompt, positions.value->ids,
                                                            vision.features, mm_start, visual_tokens, 0,
                                                            k_cache, v_cache, text); !status.ok()) {
            fail_row(row, std::move(status));
            return row;
        }
        logits.assign(text.logits.end() - graph.text.vocab_size, text.logits.end());
    }
    constexpr uint32_t max_new = 4u;
    const auto eos = graph.text.eos_token_ids.empty() ? 31u : graph.text.eos_token_ids.front();
    luisa::vector<uint32_t> expected;
    luisa::vector<luisa::vector<float>> expected_logits_trace;
    auto next_position = positions.value->next_position;
    for (uint32_t step = 0; step < max_new; step++) {
        if (step == 0u) { expected_logits_trace.push_back(logits); }
        const auto token = luisa::ocr::test::p7_oracle_argmax(logits);
        expected.push_back(token);
        if (token == eos || step + 1u == max_new) { break; }
        const luisa::vector<uint32_t> step_tokens{token};
        if (fixture.spec.p8) {
            luisa::ocr::test::P8TextOracle next;
            if (auto status = luisa::ocr::test::p8_oracle_text_forward(graph, fixture.weights8, step_tokens, next_position,
                                                                {}, 0, 0, static_cast<uint32_t>(prompt.size()) + step,
                                                                k_cache, v_cache, next); !status.ok()) {
                fail_row(row, std::move(status));
                return row;
            }
            logits = std::move(next.logits);
        } else {
            ocr::test::P7TextOracle next;
            if (auto status = luisa::ocr::test::p7_oracle_text_forward(graph, fixture.weights7, step_tokens, next_position,
                                                                {}, 0, 0, static_cast<uint32_t>(prompt.size()) + step,
                                                                k_cache, v_cache, next); !status.ok()) {
                fail_row(row, std::move(status));
                return row;
            }
            logits = std::move(next.logits);
        }
        expected_logits_trace.push_back(logits);
        for (auto &position : next_position) { position++; }
    }
    if (expected_logits_trace.size() != expected.size()) {
        fail_row(row, Status::error(StatusCode::internal_error, "P7/P8 oracle logits trace length mismatch", "correctness"));
        return row;
    }
    row.workload.pages = 1;
    row.workload.patches = image.packed.patch_count;
    row.workload.prompt_tokens = prompt.size();
    row.workload.output_tokens = expected.size();
    row.workload.weight_bytes = fixture.executor.weight_bytes();
    row.workload.shape = {{"source_width", "8"}, {"source_height", "8"},
                          {"visual_tokens", luisa::format("{}", visual_tokens)},
                          {"prompt_tokens", luisa::format("{}", prompt.size())},
                          {"max_new_tokens", luisa::format("{}", max_new)},
                          {"sampling", "greedy"}, {"rope_axes", luisa::format("{}", graph.text_rope_axes())},
                          {"linear_storage", fixture.spec.quantized ? luisa::format("int4_sym_group_{}_w4a32", fixture.spec.group_size) : luisa::string{"fp32"}},
                          {"activation_precision", "fp32"}, {"cache", "fresh_contiguous_fp32_per_request"}};
    row.workload.byte_accounting = fixture.spec.p8 && fixture.spec.quantized ?
        "Actual packed weight/workspace/KV bytes; oracle consumes exact dequantized stored codes. W4A32 uses FP32 activations, never W4A16." :
        "Actual resident weight/workspace/KV bytes for the declared synthetic FP32 slice.";

    auto stream = device.create_stream();
    luisa::optional<DenseDecodeResult> latest;
    uint64_t latest_cache_bytes = 0;
    StageDurations stage;
    Clock stage_clock;
    const auto request = [&]() -> Status {
        if (options.instrument) { stage_clock.tic(); }
        fixture.executor.reset_vision_workspace();
        P7VisionOutput vision;
        {
            auto list = CommandList::create();
            auto recorded = fixture.executor.record_vision(list, luisa::span<const P7ImageInput>{&image, 1});
            if (!recorded.ok()) { return recorded.status; }
            vision = *recorded.value;
            stream << list.commit() << synchronize();
        }
        if (options.instrument) {
            stage.vision_ms = stage_clock.toc();
            stage_clock.tic();
        }
        luisa::vector<int32_t> position_ids, next;
        auto decode_request = fixture.executor.build_decode_request(prompt, mm_start, visual_tokens, vision,
                                                                    luisa::span<const P7ImageInput>{&image, 1},
                                                                    position_ids, next);
        if (!decode_request.ok()) { return decode_request.status; }
        if (options.instrument) {
            stage.preprocess_ms = stage_clock.toc();
            stage_clock.tic();
        }
        ContiguousKvCache cache;
        KvCacheConfig cache_config;
        cache_config.num_layers = graph.text.num_layers;
        cache_config.num_kv_heads = graph.text.num_kv_heads;
        cache_config.head_dim = graph.text.head_dim;
        cache_config.max_seq_len = static_cast<uint32_t>(prompt.size()) + max_new + 1u;
        cache_config.batch = 1;
        cache_config.dtype_bytes = 4;
        if (auto status = cache.create(device, stream, cache_config); !status.ok()) { return status; }
        latest_cache_bytes = cache.device_bytes();
        DenseDecodeOptions decode_options;
        decode_options.sampling.greedy = true;
        decode_options.max_new_tokens = max_new;
        if (graph.text_rope_axes() >= 3u) {
            decode_options.prompt_position_ids = std::move(position_ids);
            decode_options.next_position = std::move(next);
        }
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
            const auto sequence = [](luisa::span<const uint32_t> tokens) {
                luisa::string text{"["};
                for (size_t i = 0; i < tokens.size(); i++) {
                    if (i != 0u) { text += ","; }
                    text += luisa::format("{}", tokens[i]);
                }
                text += "]";
                return text;
            };
            return Status::error(StatusCode::internal_error,
                                 luisa::format("P7/P8 greedy tokens/steps differ from independent oracle: expected {}, got {} (steps {})",
                                               sequence(expected), latest ? sequence(latest->tokens) : luisa::string{"<none>"},
                                               latest ? latest->steps : 0u),
                                 "correctness");
        }
        const bool expect_eos = expected.back() == eos;
        if (latest->hit_eos != expect_eos || latest->truncated == expect_eos || latest->hit_stop_string) {
            return Status::error(StatusCode::internal_error, "P7/P8 termination flags mismatch", "correctness");
        }
        row.truncated = latest->truncated;
        row.fallback_used = latest->cuda_graph_fallbacks != 0u;
        if (!options.graph && (latest->cuda_graph_steps != 0u || latest->cuda_graph_updates != 0u)) {
            return Status::error(StatusCode::internal_error, "P7/P8 non-graph arm used graph steps", "cuda_graph");
        }
        if (options.graph && latest->cuda_graph_fallbacks == 0u && latest->cuda_graph_steps == 0u) {
            return Status::error(StatusCode::internal_error, "P7/P8 graph arm executed no graph step", "cuda_graph");
        }
        luisa::vector<float> features;
        if (auto status = fixture.executor.download_vision_features(stream, features); !status.ok()) { return status; }
        stream << synchronize();
        return check_close(features, vision_features, 3e-3f, 3e-3f, "P7/P8 vision features");
    };
    // Layered gate: independently verify prefill and EVERY decode-step logits
    // row before any request is timed (the controller exposes tokens, not logits).
    const auto verify_logits_trace = [&]() -> Status {
        fixture.executor.reset_vision_workspace();
        P7VisionOutput vision;
        {
            auto list = CommandList::create();
            auto recorded = fixture.executor.record_vision(list, luisa::span<const P7ImageInput>{&image, 1});
            if (!recorded.ok()) { return recorded.status; }
            vision = *recorded.value;
            stream << list.commit() << synchronize();
        }
        luisa::vector<int32_t> position_ids, next;
        auto decode_request = fixture.executor.build_decode_request(prompt, mm_start, visual_tokens, vision,
                                                                    luisa::span<const P7ImageInput>{&image, 1},
                                                                    position_ids, next);
        if (!decode_request.ok()) { return decode_request.status; }
        ContiguousKvCache cache;
        KvCacheConfig cache_config;
        cache_config.num_layers = graph.text.num_layers;
        cache_config.num_kv_heads = graph.text.num_kv_heads;
        cache_config.head_dim = graph.text.head_dim;
        cache_config.max_seq_len = static_cast<uint32_t>(prompt.size()) + max_new + 1u;
        cache_config.batch = 1;
        cache_config.dtype_bytes = 4;
        if (auto status = cache.create(device, stream, cache_config); !status.ok()) { return status; }
        luisa::vector<float> device_logits(graph.text.vocab_size);
        {
            auto list = CommandList::create();
            Status status;
            if (graph.text_rope_axes() >= 3u) {
                status = fixture.executor.text_executor().record_prefill_positions(
                    list, *decode_request.value, position_ids, {}, cache, false, device_logits, {});
            } else {
                status = fixture.executor.text_executor().record_prefill(list, *decode_request.value, cache, device_logits);
            }
            if (!status.ok()) { return status; }
            stream << list.commit() << synchronize();
        }
        if (auto status = check_close(device_logits, expected_logits_trace.front(), 3e-3f, 3e-3f,
                                      "P7/P8 prefill last-row logits"); !status.ok()) { return status; }
        if (ocr::test::p7_oracle_argmax(device_logits) != expected.front()) {
            return Status::error(StatusCode::internal_error, "P7/P8 prefill argmax differs from oracle", "correctness");
        }
        auto step_position = positions.value->next_position;
        for (size_t step = 0; step + 1u < expected.size(); step++) {
            device_logits.assign(graph.text.vocab_size, 0.0f);
            {
                auto list = CommandList::create();
                Status status;
                if (graph.text_rope_axes() >= 3u) {
                    status = fixture.executor.text_executor().record_decode_step_positions(
                        list, expected[step], step_position, cache, device_logits);
                } else {
                    status = fixture.executor.text_executor().record_decode_step(list, expected[step], cache, device_logits);
                }
                if (!status.ok()) { return status; }
                stream << list.commit() << synchronize();
            }
            if (auto status = check_close(device_logits, expected_logits_trace[step + 1u], 3e-3f, 3e-3f,
                                          luisa::format("P7/P8 decode step {} logits", step)); !status.ok()) { return status; }
            if (ocr::test::p7_oracle_argmax(device_logits) != expected[step + 1u]) {
                return Status::error(StatusCode::internal_error,
                                     luisa::format("P7/P8 decode step {} argmax differs from oracle", step),
                                     "correctness");
            }
            for (auto &position : step_position) { position++; }
        }
        cache.release(0);
        return Status::success();
    };
    if (auto status = verify_logits_trace(); !status.ok()) {
        fail_row(row, std::move(status));
        return row;
    }
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

void append_p7_model_rows(Device &device, const BenchmarkOptions &options, luisa::vector<Row> &rows) {
    append(rows, run_p7(device, options, ModelKind::hunyuan, false));
    append(rows, run_p7(device, options, ModelKind::dots, false));
    append(rows, run_p7(device, options, ModelKind::dots, true));
    append(rows, run_p7(device, options, ModelKind::olm, false));
    append(rows, run_p7(device, options, ModelKind::olm, true));
    append(rows, run_p7(device, options, ModelKind::got, false));
    append(rows, run_p7(device, options, ModelKind::tele, false));
    append(rows, run_p7(device, options, ModelKind::glm, false));
    append(rows, run_p7(device, options, ModelKind::mineru, false));
}

}// namespace luisa::ocr::bench

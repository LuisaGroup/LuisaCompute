// Complete Surya detector -> crop -> vision -> hybrid-GDN page benchmark.
#include "benchmark_runners.h"

#include "ocr_gdn_test_utils.h"

#include "ocr/runtime/surya_executor.h"

namespace luisa::ocr::bench {
namespace {

using namespace luisa::compute;

struct SuryaFixture {
    luisa::ocr::test::P7TempDir tmp;
    SuryaConfig cfg;
    luisa::ocr::test::SuryaDetectorWeights detector_weights;
    luisa::ocr::test::P7SyntheticWeights vision_weights;
    luisa::ocr::test::HybridSyntheticWeights decoder_weights;
    WeightStore store;
    luisa::unique_ptr<Tokenizer, TokenizerDeleter> tokenizer;
    SuryaExecutor executor;
    double setup_ms{0.0};

    [[nodiscard]] Status init(Device &device) {
        Clock clock;
        cfg = surya_synthetic_config();
        cfg.detector.stem.act = 0;
        cfg.detector.stage1.act = 0;
        cfg.detector.stage2.act = 0;
        if (auto status = cfg.validate(); !status.ok()) { return status; }
        detector_weights = luisa::ocr::test::make_surya_detector_weights(cfg.detector);
        vision_weights = luisa::ocr::test::generate_p7_weights(cfg.vision);
        decoder_weights = luisa::ocr::test::generate_hybrid_weights(cfg.decoder);
        tokenizer = luisa::ocr::test::load_hybrid_tokenizer();
        if (!tokenizer) { return Status::error(StatusCode::internal_error, "failed to create synthetic Surya tokenizer", "tokenizer"); }
        cfg.tokenizer = tokenizer.get();
        const auto detector_path = tmp.path / "detector.lcocr";
        const auto vision_path = tmp.path / "vision.lcocr";
        const auto decoder_path = tmp.path / "decoder.lcocr";
        if (auto status = luisa::ocr::test::write_surya_detector_package(detector_path, cfg.detector,
                                                                  cfg.detector_names, detector_weights); !status.ok()) { return status; }
        if (auto status = luisa::ocr::test::write_p7_packages(vision_path, tmp.path / "dense_unused.lcocr",
                                                       cfg.vision, cfg.vision_names, vision_weights,
                                                       "surya_2_5"); !status.ok()) { return status; }
        if (auto status = luisa::ocr::test::write_hybrid_decoder_package(decoder_path, cfg.decoder,
                                                                  cfg.decoder_names, decoder_weights); !status.ok()) { return status; }
        {
            auto detector = PackageReader::open(detector_path);
            auto vision = PackageReader::open(vision_path);
            auto decoder = PackageReader::open(decoder_path);
            if (!detector.ok()) { return detector.status; }
            if (!vision.ok()) { return vision.status; }
            if (!decoder.ok()) { return decoder.status; }
            if (auto status = surya_validate_tensors((*detector.value)->tensors(), (*vision.value)->tensors(),
                                                     (*decoder.value)->tensors(), cfg); !status.ok()) { return status; }
        }
        auto stream = device.create_stream();
        for (const auto &[path, component] : {
                 std::pair<const luisa::filesystem::path *, const char *>{&detector_path, "detector"},
                 {&vision_path, "vision"}, {&decoder_path, "decoder"}}) {
            auto reader = PackageReader::open(*path);
            if (!reader.ok()) { return reader.status; }
            if (auto status = store.upload_component_sync(device, stream, **reader.value, component); !status.ok()) {
                return status;
            }
        }
        if (auto status = executor.create(device, stream, cfg, store); !status.ok()) { return status; }
        setup_ms = clock.toc();
        return Status::success();
    }
};

struct Crop {
    luisa::vector<uint8_t> pixels;
    uint32_t width{0};
    uint32_t height{0};
};

[[nodiscard]] Crop crop_line(const SuryaPageInput &page, const output::DbRect &bounds) {
    const auto sw = static_cast<int32_t>(page.source_width);
    const auto sh = static_cast<int32_t>(page.source_height);
    auto x0 = std::clamp(static_cast<int32_t>(std::floor(bounds.x0)), 0, sw - 1);
    auto y0 = std::clamp(static_cast<int32_t>(std::floor(bounds.y0)), 0, sh - 1);
    auto x1 = std::clamp(static_cast<int32_t>(std::ceil(bounds.x1)), x0 + 1, sw);
    auto y1 = std::clamp(static_cast<int32_t>(std::ceil(bounds.y1)), y0 + 1, sh);
    constexpr int32_t min_side = 4;
    if (x1 - x0 < min_side) {
        const auto need = min_side - (x1 - x0);
        x0 = std::max(0, x0 - need / 2);
        x1 = std::min(sw, x0 + min_side);
        x0 = std::max(0, x1 - min_side);
    }
    if (y1 - y0 < min_side) {
        const auto need = min_side - (y1 - y0);
        y0 = std::max(0, y0 - need / 2);
        y1 = std::min(sh, y0 + min_side);
        y0 = std::max(0, y1 - min_side);
    }
    Crop crop;
    crop.width = static_cast<uint32_t>(x1 - x0);
    crop.height = static_cast<uint32_t>(y1 - y0);
    crop.pixels.resize(static_cast<size_t>(crop.width) * crop.height * 3u);
    for (uint32_t y = 0; y < crop.height; y++) {
        for (uint32_t x = 0; x < crop.width; x++) {
            const auto src = (static_cast<size_t>(y0 + y) * page.source_width + x0 + x) * 3u;
            const auto dst = (static_cast<size_t>(y) * crop.width + x) * 3u;
            crop.pixels[dst] = page.pixels[src];
            crop.pixels[dst + 1] = page.pixels[src + 1];
            crop.pixels[dst + 2] = page.pixels[src + 2];
        }
    }
    return crop;
}

[[nodiscard]] uint32_t greedy(luisa::span<const float> logits) {
    uint32_t best = 0;
    for (uint32_t i = 1; i < logits.size(); i++) {
        if (logits[i] > logits[best]) { best = i; }
    }
    return best;
}

Row run_surya(Device &device, const BenchmarkOptions &options) {
    Row row;
    row.workload = make_workload(options, BenchmarkSuite::model, "surya_2_5",
                                 "page_detector_crops_hybrid_decode", "recurrent", "tier1");
    row.workload.options.push_back({"tier", "1"});
    if (!workload_selected(options, row.workload)) { return {}; }
    if (options.graph || options.fusion || options.variant == "chunked") {
        fail_row(row, Status::error(StatusCode::unsupported,
                                    "Surya benchmark currently supports only its real recurrent detector/crop/GDN path; graph/fusion/chunked switch not exposed",
                                    options.graph ? "graph" : (options.fusion ? "fusion" : "variant")),
                 RowStatus::refused);
        return row;
    }
    SuryaFixture fixture;
    if (auto status = fixture.init(device); !status.ok()) {
        fail_row(row, std::move(status));
        return row;
    }
    add_counter(row, "synthetic_setup_ms_including_package_upload_and_shader_compile", fixture.setup_ms);
    add_counter(row, "resident_weight_bytes", fixture.executor.weight_bytes());
    add_counter(row, "detector_workspace_bytes", fixture.executor.workspace_bytes());
    add_counter(row, "detector_shader_compiles", fixture.executor.shader_registry().total_compiles());
    SuryaPageInput page;
    page.source_width = 32;
    page.source_height = 32;
    page.page_index = 0;
    const std::pair<uint32_t, uint32_t> bars[] = {{8u, 16u}, {20u, 28u}};
    page.pixels = luisa::ocr::test::make_surya_page_with_bars(32, 32, bars);
    row.workload.pages = 1;
    row.workload.weight_bytes = fixture.executor.weight_bytes();
    row.workload.shape = {{"source_width", "32"}, {"source_height", "32"},
                          {"bars", "2_dark_horizontal_bars"},
                          {"max_new_tokens_per_line", luisa::format("{}", fixture.cfg.max_new_tokens)},
                          {"gdn_prefill", "recurrent"}, {"tokenizer", "synthetic_unigram_24_pieces"},
                          {"cache", "hybrid_cache_reset_per_line"}};
    row.workload.byte_accounting = "Actual detector/graph resident weights, detector workspace, and peak per-line hybrid cache; output tokens are generated token ids, separate from detokenized UTF-8 text.";

    auto stream = device.create_stream();
    luisa::optional<SuryaRunResult> latest;
    const auto request = [&]() -> Status {
        auto result = fixture.executor.recognize(stream, luisa::span<const SuryaPageInput>{&page, 1});
        if (!result.ok()) { return result.status; }
        latest = std::move(*result.value);
        return Status::success();
    };
    const auto gate = [&]() -> Status {
        if (!latest || latest->pages.size() != 1u) {
            return Status::error(StatusCode::internal_error, "Surya returned no single-page result", "correctness");
        }
        const auto &actual = latest->pages[0];
        auto pixels = luisa::ocr::test::p7_oracle_preprocess(page.pixels, 32, 32, 32, 32,
                                                      fixture.cfg.detector.mean, fixture.cfg.detector.std_dev);
        uint32_t map_h = 0, map_w = 0;
        auto probability = luisa::ocr::test::surya_detector_oracle(fixture.cfg.detector, fixture.detector_weights,
                                                            pixels, map_h, map_w);
        if (actual.det_map_height != map_h || actual.det_map_width != map_w) {
            return Status::error(StatusCode::internal_error, "Surya detector map geometry mismatch", "correctness");
        }
        if (auto status = check_close(actual.det_probabilities, probability, 5e-3f, 5e-3f,
                                      "Surya detector probabilities"); !status.ok()) { return status; }
        output::DbSourceTransform transform;
        transform.scale_x = static_cast<float>(page.source_width) / static_cast<float>(map_w);
        transform.scale_y = static_cast<float>(page.source_height) / static_cast<float>(map_h);
        auto boxes = output::db_postprocess(probability, map_w, map_h, transform, fixture.cfg.db);
        if (!boxes.ok()) { return boxes.status; }
        if (boxes.value->boxes.size() > fixture.cfg.max_lines_per_page) {
            boxes.value->boxes.resize(fixture.cfg.max_lines_per_page);
        }
        if (actual.boxes.size() != boxes.value->boxes.size() || actual.boxes.empty() || actual.lines.empty()) {
            return Status::error(StatusCode::internal_error, "Surya DB boxes/lines missing or count mismatch", "correctness");
        }
        row.workload.regions = actual.lines.size();
        row.workload.output_tokens = 0;
        row.truncated = false;
        for (size_t line_index = 0; line_index < actual.lines.size(); line_index++) {
            const auto &line = actual.lines[line_index];
            const auto &expected_box = boxes.value->boxes[line_index];
            if (std::abs(line.confidence - expected_box.score) >= 5e-3f || !line.has_confidence) {
                return Status::error(StatusCode::internal_error, "Surya detector confidence mismatch", "correctness");
            }
            auto crop = crop_line(page, line.bounds);
            P7ImageInput image;
            image.pixels = std::move(crop.pixels);
            image.source_width = crop.width;
            image.source_height = crop.height;
            auto packed = p7_pack_image(crop.width, crop.height, fixture.cfg.vision.packing,
                                        fixture.cfg.vision.connector);
            if (!packed.ok()) { return packed.status; }
            image.packed = *packed.value;
            const auto visual_tokens = static_cast<uint32_t>(image.packed.visual_tokens);
            luisa::vector<uint32_t> prompt = {1u};
            constexpr uint32_t mm_start = 1;
            prompt.insert(prompt.end(), visual_tokens, 0u);
            prompt.push_back(2u);
            auto crop_pixels = luisa::ocr::test::p7_oracle_preprocess(image.pixels, crop.width, crop.height,
                                                               image.packed.resized_width, image.packed.resized_height,
                                                               fixture.cfg.vision.mean, fixture.cfg.vision.std_dev);
            luisa::ocr::test::P7VisionOracle vision;
            if (auto status = luisa::ocr::test::p7_oracle_vision(fixture.cfg.vision, fixture.vision_weights,
                                                          crop_pixels, image.packed, vision); !status.ok()) { return status; }
            luisa::ocr::test::HybridCpuOracle oracle(fixture.cfg.decoder, fixture.decoder_weights);
            luisa::vector<float> logits;
            for (const auto token : prompt) { logits = oracle.step(token, vision.features, mm_start, visual_tokens); }
            luisa::vector<uint32_t> expected_tokens;
            for (uint32_t step = 0; step < fixture.cfg.max_new_tokens; step++) {
                const auto token = greedy(logits);
                expected_tokens.push_back(token);
                if (token == 23u || step + 1u == fixture.cfg.max_new_tokens) { break; }
                logits = oracle.step(token);
            }
            if (line.tokens != expected_tokens) {
                return Status::error(StatusCode::internal_error,
                                     luisa::format("Surya line {} greedy tokens differ from independent hybrid oracle", line_index),
                                     "correctness");
            }
            luisa::string expected_text;
            for (const auto token : line.tokens) {
                if (token < 23u) { expected_text += static_cast<char>('a' + token); }
            }
            if (line.text != expected_text) {
                return Status::error(StatusCode::internal_error, "Surya detokenized line text mismatch", "correctness");
            }
            const bool expect_eos = expected_tokens.back() == 23u;
            if (line.hit_eos != expect_eos || line.truncated == expect_eos) {
                return Status::error(StatusCode::internal_error, "Surya line termination flags mismatch", "correctness");
            }
            row.truncated = row.truncated || line.truncated;
            row.workload.output_tokens += line.tokens.size();
            for (const auto &point : line.polygon) {
                if (!std::isfinite(point.x) || !std::isfinite(point.y) ||
                    point.x < -16.0f || point.x > 48.0f || point.y < -16.0f || point.y > 48.0f) {
                    return Status::error(StatusCode::internal_error, "Surya source polygon is nonfinite or on the wrong scale", "correctness");
                }
            }
        }
        luisa::string joined;
        for (size_t i = 0; i < actual.lines.size(); i++) {
            if (i != 0u) { joined += "\n"; }
            joined += actual.lines[i].text;
        }
        if (actual.text != joined) {
            return Status::error(StatusCode::internal_error, "Surya page text is not the canonical line join", "correctness");
        }
        return Status::success();
    };
    time_requests(stream, options, row, request, gate);
    if (latest) {
        add_counter(row, "workspace_bytes", latest->workspace_bytes);
        add_counter(row, "peak_line_hybrid_cache_bytes", latest->cache_bytes);
    }
    return row;
}

}// namespace

void append_surya_model_rows(Device &device, const BenchmarkOptions &options, luisa::vector<Row> &rows) {
    auto row = run_surya(device, options);
    if (!row.workload.id.empty()) { rows.emplace_back(std::move(row)); }
}

}// namespace luisa::ocr::bench

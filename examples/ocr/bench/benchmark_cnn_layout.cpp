// Synthetic CNN page and layout-detector request benchmarks.
#include "benchmark_runners.h"

#include "ocr_pp_test_utils.h"
#include "ocr_p6_test_utils.h"

#include "ocr/runtime/cnn_executor.h"
#include "ocr/runtime/layout_vlm_executor.h"

namespace luisa::ocr::bench {
namespace {

using namespace luisa::compute;

[[nodiscard]] CnnPageInput cnn_page(uint32_t width, uint32_t height, uint32_t seed = 7u) {
    const auto source = luisa::ocr::test::p6_make_page(width, height, seed);
    CnnPageInput page;
    page.pixels = source.pixels;
    page.source_width = source.source_width;
    page.source_height = source.source_height;
    page.page_index = source.page_index;
    return page;
}

[[nodiscard]] Status reject_unsupported_switches(const BenchmarkOptions &options, Row &row,
                                                 luisa::string_view family) {
    if (options.graph) {
        return Status::error(StatusCode::unsupported,
                             luisa::string{"CUDA graph benchmark requested for unsupported family: "} + luisa::string{family},
                             "graph");
    }
    if (options.fusion) {
        return Status::error(StatusCode::unsupported,
                             luisa::string{"dense residual+RMSNorm fusion is not a path for family: "} + luisa::string{family},
                             "fusion");
    }
    return Status::success();
}

struct PpFixture {
    luisa::ocr::test::PpTempDir tmp;
    PpOcrv6Config cfg;
    PpOcrv6TensorNames names;
    luisa::ocr::test::PpSyntheticWeights weights;
    luisa::ocr::test::PpFoldedWeights folded;
    WeightStore store;
    CnnExecutor executor;
    CnnExecutorConfig config;
    double setup_ms{0.0};

    [[nodiscard]] Status init(Device &device, PpOcrSize size) {
        Clock clock;
        cfg = pp_ocrv6_synthetic_config(size);
        if (auto status = cfg.validate(); !status.ok()) { return status; }
        names = PpOcrv6TensorNames{};
        weights = luisa::ocr::test::generate_pp_synthetic_weights(cfg);
        if (auto status = luisa::ocr::test::fold_pp_weights(cfg, weights, folded); !status.ok()) { return status; }
        if (auto status = luisa::ocr::test::pp_calibrate_db_head(cfg, weights, folded); !status.ok()) { return status; }
        const auto detector_path = tmp.path / "detector.lcocr";
        const auto recognizer_path = tmp.path / "recognizer.lcocr";
        if (auto status = luisa::ocr::test::write_pp_synthetic_packages(detector_path, recognizer_path,
                                                                 cfg, names, weights, folded); !status.ok()) { return status; }
        auto stream = device.create_stream();
        if (auto status = luisa::ocr::test::upload_pp_synthetic_weights(device, stream, detector_path,
                                                                 recognizer_path, store); !status.ok()) { return status; }
        config.model = cfg;
        config.names = names;
        config.dictionary = luisa::ocr::test::pp_synthetic_dictionary();
        config.max_lines_per_page = 16;
        if (auto status = executor.create(device, stream, config, store); !status.ok()) { return status; }
        setup_ms = clock.toc();
        return Status::success();
    }
};

Row run_pp_row_impl(Device &device, const BenchmarkOptions &options, BenchmarkSuite suite,
                    PpOcrSize size, luisa::string series, uint32_t tier) {
    const auto target = luisa::string{"pp_ocrv6:"} + luisa::string{pp_ocr_size_name(size)};
    Row row;
    row.workload = make_workload(options, suite, target, "full_page_bars", "reference", std::move(series));
    row.workload.options.push_back({"tier", luisa::format("{}", tier)});
    if (!workload_selected(options, row.workload)) { return {}; }
    if (auto status = reject_unsupported_switches(options, row, "cnn"); !status.ok()) {
        fail_row(row, std::move(status), RowStatus::refused);
        return row;
    }
    PpFixture fixture;
    if (auto status = fixture.init(device, size); !status.ok()) {
        fail_row(row, std::move(status));
        return row;
    }
    add_counter(row, "synthetic_setup_ms_including_package_upload_and_shader_compile", fixture.setup_ms);
    add_counter(row, "resident_weight_bytes", fixture.executor.weight_bytes());
    add_counter(row, "workspace_bytes", fixture.executor.workspace_bytes());
    add_counter(row, "shader_compiles", fixture.executor.shader_registry().total_compiles());
    const auto page = cnn_page(fixture.cfg.detector.input_width * 2u, fixture.cfg.detector.input_height * 2u);
    row.workload.pages = 1;
    row.workload.weight_bytes = fixture.executor.weight_bytes();
    row.workload.shape = {{"source_width", luisa::format("{}", page.source_width)},
                          {"source_height", luisa::format("{}", page.source_height)},
                          {"detector_input", luisa::format("{}x{}", fixture.cfg.detector.input_width, fixture.cfg.detector.input_height)},
                          {"recognizer_input", luisa::format("{}x{}", fixture.cfg.recognizer.input_width, fixture.cfg.recognizer.input_height)},
                          {"page_content", "deterministic_bright_bars_dark_noise"},
                          {"precision", "fp32_folded_conv_bn_replk_w4a32_not_applicable"}};
    row.workload.byte_accounting = "Actual resident weights and grow-only workspace from executor accounting; CTC tokens are collapsed class tokens, not UTF-8 bytes.";

    auto stream = device.create_stream();
    luisa::optional<CnnRunResult> latest;
    const auto request = [&]() -> Status {
        auto result = fixture.executor.recognize(stream, luisa::span<const CnnPageInput>{&page, 1});
        if (!result.ok()) { return result.status; }
        latest = std::move(*result.value);
        return Status::success();
    };
    const auto gate = [&]() -> Status {
        if (!latest || latest->pages.size() != 1u) {
            return Status::error(StatusCode::internal_error, "CNN run returned no single-page result", "correctness");
        }
        const auto &actual = latest->pages[0];
        auto pixels = luisa::ocr::test::pp_oracle_preprocess(page.pixels, page.source_width, page.source_height,
                                                      fixture.cfg.detector.input_width, fixture.cfg.detector.input_height,
                                                      fixture.config.mean, fixture.config.std_dev);
        luisa::ocr::test::PpOracleDetector detector;
        if (auto status = luisa::ocr::test::pp_oracle_detector(fixture.cfg, fixture.weights, pixels, detector); !status.ok()) {
            return status;
        }
        if (actual.det_map_width != detector.map_w || actual.det_map_height != detector.map_h) {
            return Status::error(StatusCode::internal_error, "CNN detector map geometry mismatch", "correctness");
        }
        if (auto status = check_close(actual.det_probabilities, detector.prob, 3e-3f, 3e-3f, "CNN detector probability"); !status.ok()) {
            return status;
        }
        output::DbSourceTransform transform;
        transform.scale_x = static_cast<float>(page.source_width) / static_cast<float>(detector.map_w);
        transform.scale_y = static_cast<float>(page.source_height) / static_cast<float>(detector.map_h);
        auto db_options = fixture.config.db;
        db_options.max_boxes = fixture.config.max_lines_per_page;
        auto boxes = output::db_postprocess(detector.prob, detector.map_w, detector.map_h, transform, db_options);
        if (!boxes.ok()) { return boxes.status; }
        if (actual.boxes.size() != boxes.value->boxes.size() || actual.boxes.empty()) {
            return Status::error(StatusCode::internal_error, "CNN DB box count mismatch or calibrated page produced no boxes", "correctness");
        }
        for (size_t i = 0; i < actual.boxes.size(); i++) {
            const auto &a = actual.boxes[i];
            const auto &b = boxes.value->boxes[i];
            if (std::abs(a.score - b.score) >= 5e-3f || a.polygon.size() != 4u || b.polygon.size() != 4u) {
                return Status::error(StatusCode::internal_error, "CNN DB score/polygon mismatch", "correctness");
            }
            for (size_t p = 0; p < 4u; p++) {
                if (std::abs(a.polygon[p].x - b.polygon[p].x) >= 0.75f ||
                    std::abs(a.polygon[p].y - b.polygon[p].y) >= 0.75f) {
                    return Status::error(StatusCode::internal_error, "CNN DB coordinate mismatch", "correctness");
                }
            }
        }
        luisa::vector<float> crop, logits;
        if (auto status = fixture.executor.download_line_crop(stream, crop); !status.ok()) { return status; }
        if (auto status = fixture.executor.download_recognizer_logits(stream, logits); !status.ok()) { return status; }
        stream << synchronize();
        luisa::ocr::test::PpOracleRecognizer recognizer;
        if (auto status = luisa::ocr::test::pp_oracle_recognizer(fixture.cfg, fixture.weights, crop,
                                                          fixture.config.ctc.temperature, recognizer); !status.ok()) { return status; }
        if (auto status = check_close(logits, recognizer.logits, 3e-3f, 3e-3f, "CNN recognizer CTC logits"); !status.ok()) {
            return status;
        }
        auto decoded = output::ctc_decode_logits(recognizer.logits, recognizer.frames,
                                                 fixture.config.dictionary, fixture.config.ctc);
        if (!decoded.ok()) { return decoded.status; }
        if (actual.lines.empty() || actual.lines.back().text != decoded.value->text ||
            actual.lines.back().has_confidence != decoded.value->has_confidence) {
            return Status::error(StatusCode::internal_error, "CNN collapsed CTC text/confidence mismatch", "correctness");
        }
        if (decoded.value->has_confidence && std::abs(actual.lines.back().confidence - decoded.value->confidence) >= 2e-3f) {
            return Status::error(StatusCode::internal_error, "CNN CTC confidence mismatch", "correctness");
        }
        row.workload.regions = actual.lines.size();
        row.workload.output_tokens = 0;
        for (const auto &line : actual.lines) { row.workload.output_tokens += line.ctc_token_count; }
        return Status::success();
    };
    time_requests(stream, options, row, request, gate);
    add_counter(row, "actual_regions", row.workload.regions);
    add_counter(row, "collapsed_ctc_tokens", row.workload.output_tokens);
    return row;
}

struct LayoutFixture {
    luisa::ocr::test::P6TempDir tmp;
    PpDocLayoutConfig cfg;
    PpDocLayoutTensorNames names;
    luisa::ocr::test::LayoutSyntheticWeights weights;
    WeightStore store;
    PpDocLayoutExecutor executor;
    PpDocLayoutExecutorConfig config;
    double setup_ms{0.0};

    [[nodiscard]] Status init(Device &device, PpDocLayoutVersion version) {
        Clock clock;
        cfg = pp_doc_layout_synthetic_config(version);
        if (auto status = cfg.validate(); !status.ok()) { return status; }
        names = PpDocLayoutTensorNames{};
        weights = luisa::ocr::test::generate_layout_weights(cfg);
        const auto path = tmp.path / "layout.lcocr";
        if (auto status = luisa::ocr::test::write_layout_package(path, cfg, names, weights); !status.ok()) { return status; }
        auto stream = device.create_stream();
        auto reader = PackageReader::open(path);
        if (!reader.ok()) { return reader.status; }
        if (auto status = store.upload_component_sync(device, stream, **reader.value, "layout"); !status.ok()) { return status; }
        config.model = cfg;
        config.names = names;
        config.component = "layout";
        if (auto status = executor.create_from_packages(device, stream, config, store, **reader.value); !status.ok()) { return status; }
        setup_ms = clock.toc();
        return Status::success();
    }
};

Row run_layout(Device &device, const BenchmarkOptions &options, PpDocLayoutVersion version,
               luisa::string series, uint32_t tier) {
    const auto target = luisa::string{"pp_doc_layout_"} + luisa::string{pp_doc_layout_version_name(version)};
    Row row;
    row.workload = make_workload(options, BenchmarkSuite::layout, target, "detect_bars", "reference", std::move(series));
    row.workload.options.push_back({"tier", luisa::format("{}", tier)});
    if (!workload_selected(options, row.workload)) { return {}; }
    if (auto status = reject_unsupported_switches(options, row, "layout_detector"); !status.ok()) {
        fail_row(row, std::move(status), RowStatus::refused);
        return row;
    }
    LayoutFixture fixture;
    if (auto status = fixture.init(device, version); !status.ok()) {
        fail_row(row, std::move(status));
        return row;
    }
    add_counter(row, "synthetic_setup_ms_including_package_upload_and_shader_compile", fixture.setup_ms);
    add_counter(row, "resident_weight_bytes", fixture.executor.weight_bytes());
    add_counter(row, "workspace_bytes", fixture.executor.workspace_bytes());
    const auto page = luisa::ocr::test::p6_make_page(64, 64);
    row.workload.pages = 1;
    row.workload.weight_bytes = fixture.executor.weight_bytes();
    row.workload.shape = {{"source_width", "64"}, {"source_height", "64"},
                          {"detector_input", luisa::format("{}x{}", fixture.cfg.input_width, fixture.cfg.input_height)},
                          {"page_content", "deterministic_bright_bars_dark_noise"},
                          {"layout_version", luisa::string{pp_doc_layout_version_name(version)}}};
    row.workload.byte_accounting = "Actual resident detector weights and grow-only workspace from executor accounting.";

    auto stream = device.create_stream();
    luisa::optional<LayoutHeadMaps> latest;
    const auto request = [&]() -> Status {
        auto result = fixture.executor.detect(stream, page);
        if (!result.ok()) { return result.status; }
        latest = std::move(*result.value);
        return Status::success();
    };
    const auto gate = [&]() -> Status {
        if (!latest) { return Status::error(StatusCode::internal_error, "layout detect returned no maps", "correctness"); }
        auto pixels = luisa::ocr::test::p6_oracle_preprocess(page.pixels, page.source_width, page.source_height,
                                                      fixture.cfg.input_width, fixture.cfg.input_height,
                                                      fixture.config.mean, fixture.config.std_dev);
        luisa::ocr::test::LayoutOracleMaps oracle;
        if (auto status = luisa::ocr::test::oracle_layout_detect(fixture.cfg, fixture.weights, pixels, oracle); !status.ok()) {
            return status;
        }
        if (latest->map_width != oracle.map_w || latest->map_height != oracle.map_h ||
            latest->num_classes != oracle.num_classes) {
            return Status::error(StatusCode::internal_error, "layout map geometry mismatch", "correctness");
        }
        if (auto status = check_close(latest->class_probs, oracle.class_probs, 3e-3f, 3e-3f, "layout class probabilities"); !status.ok()) {
            return status;
        }
        return check_close(latest->box_deltas, oracle.box_deltas, 3e-3f, 3e-3f, "layout box deltas");
    };
    time_requests(stream, options, row, request, gate);
    return row;
}

void append(luisa::vector<Row> &rows, Row &&row) {
    if (!row.workload.id.empty()) { rows.emplace_back(std::move(row)); }
}

}// namespace

Row run_pp_row(Device &device, const BenchmarkOptions &options, BenchmarkSuite suite,
               PpOcrSize size, luisa::string series, uint32_t tier) {
    return run_pp_row_impl(device, options, suite, size, std::move(series), tier);
}

void append_cnn_rows(Device &device, const BenchmarkOptions &options, luisa::vector<Row> &rows) {
    append(rows, run_pp_row(device, options, BenchmarkSuite::cnn, PpOcrSize::tiny, "smoke", 2));
    append(rows, run_pp_row(device, options, BenchmarkSuite::cnn, PpOcrSize::medium, "scalable", 1));
    append(rows, run_pp_row(device, options, BenchmarkSuite::cnn, PpOcrSize::small, "tier2", 2));
}

void append_layout_rows(Device &device, const BenchmarkOptions &options, luisa::vector<Row> &rows) {
    append(rows, run_layout(device, options, PpDocLayoutVersion::v2, "smoke", 1));
    append(rows, run_layout(device, options, PpDocLayoutVersion::v3, "tier2", 2));
}

}// namespace luisa::ocr::bench

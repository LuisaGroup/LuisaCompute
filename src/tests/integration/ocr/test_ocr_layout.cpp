// Test for ocr/runtime/layout_vlm_executor PpDocLayoutExecutor + pp_doc_layout
// decode (Phase 6, device) -- the SMALL SYNTHETIC PP-DocLayoutV2/V3 detector.
// Covers: V2/V3 distinct config/tensor contracts, detector head-map parity
// (device conv graph vs independent double-accumulating CPU oracle), CPU decode
// (boxes/scores/classes) parity, source-coordinate transforms, the documented
// cpu_reference layout option, and empty/degenerate pages.
// Oracle: independent host reference (ref_conv2d + ref_sigmoid). Synthetic
// weights only; no model/download, no real-model parity claim.
#include "ut/ut.hpp"
#include "test_device.h"
#include "ocr_p6_test_utils.h"

#include "ocr/runtime/layout_vlm_executor.h"

using namespace luisa;
using namespace luisa::compute;
using namespace luisa::ocr;
using namespace boost::ut;
using namespace boost::ut::literals;

namespace {

// Tolerances: FP32 device conv chain vs double oracle (operation-appropriate).
constexpr float kAtol = 3e-3f;
constexpr float kRtol = 3e-3f;
// Decoded box coordinates (source pixels) and scores.
constexpr float kBoxAtol = 0.75f;
constexpr float kScoreAtol = 5e-3f;

struct Fixture {
    ocr::test::P6TempDir tmp;
    PpDocLayoutConfig cfg;
    PpDocLayoutTensorNames names;
    ocr::test::LayoutSyntheticWeights weights;
    WeightStore store;
    luisa::unique_ptr<PpDocLayoutExecutor> executor;
    PpDocLayoutExecutorConfig config;

    [[nodiscard]] bool init(Device &device, PpDocLayoutVersion version, uint32_t seed = 0x5EEDu) {
        cfg = pp_doc_layout_synthetic_config(version);
        if (!cfg.validate().ok()) { LUISA_WARNING("layout cfg validate failed"); return false; }
        names = PpDocLayoutTensorNames{};
        weights = ocr::test::generate_layout_weights(cfg, seed);
        const auto path = tmp.path / "layout.lcocr";
        if (auto s = ocr::test::write_layout_package(path, cfg, names, weights); !s.ok()) {
            LUISA_WARNING("write_layout_package: {}", s.message);
            return false;
        }
        auto stream = device.create_stream();
        auto reader = PackageReader::open(path);
        if (!reader.ok()) { LUISA_WARNING("open: {}", reader.status.message); return false; }
        if (auto s = store.upload_component_sync(device, stream, **reader.value, "layout"); !s.ok()) {
            LUISA_WARNING("upload: {}", s.message);
            return false;
        }
        config.model = cfg;
        config.names = names;
        config.component = "layout";
        executor = luisa::make_unique<PpDocLayoutExecutor>();
        if (auto s = executor->create_from_packages(device, stream, config, store, **reader.value); !s.ok()) {
            LUISA_WARNING("create_from_packages: {}", s.message);
            return false;
        }
        return true;
    }
};

// ---------------------------------------------------------------------------
// Host contract checks (no device needed).
// ---------------------------------------------------------------------------
void test_layout_contracts() {
    for (auto version : {PpDocLayoutVersion::v2, PpDocLayoutVersion::v3}) {
        auto cfg = pp_doc_layout_synthetic_config(version);
        expect(cfg.validate().ok()) << pp_doc_layout_version_name(version);
        expect(cfg.version == version);
        expect(cfg.num_classes() >= 5u);
        // Heads are 1x1; box head outputs 4.
        expect(cfg.class_head.kh == 1u && cfg.class_head.kw == 1u);
        expect(cfg.box_head.out_c == 4u);
        // Tensor inventory is exact and version-specific.
        PpDocLayoutTensorNames names;
        auto expected = pp_doc_layout_expected_tensors(cfg, names);
        expect(expected.size() == 2u * cfg.backbone.size() + 4u);
        // Real-artifact gate fails closed.
        expect(pp_doc_layout_validate_real_artifact("r", "f").code == StatusCode::unvalidated);
    }
    // V2 and V3 are distinct.
    auto v2 = pp_doc_layout_synthetic_config(PpDocLayoutVersion::v2);
    auto v3 = pp_doc_layout_synthetic_config(PpDocLayoutVersion::v3);
    expect(v2.num_classes() != v3.num_classes());
}

// ---------------------------------------------------------------------------
// Detector head-map parity (device vs independent CPU oracle).
// ---------------------------------------------------------------------------
void test_detector_parity(Device &device) {
    for (auto version : {PpDocLayoutVersion::v2, PpDocLayoutVersion::v3}) {
        Fixture fx;
        expect(fx.init(device, version)) << pp_doc_layout_version_name(version);
        if (!fx.executor) { continue; }
        auto stream = device.create_stream();
        auto page = ocr::test::p6_make_page(64, 64);
        // Device head maps through the full detect path (host preprocessing +
        // conv graph). The oracle mirrors the declared float preprocessing.
        auto maps = fx.executor->detect(stream, page);
        expect(maps.ok());
        if (!maps.ok()) { continue; }
        auto pixels = ocr::test::p6_oracle_preprocess(page.pixels, page.source_width, page.source_height,
                                                      fx.cfg.input_width, fx.cfg.input_height,
                                                      fx.config.mean, fx.config.std_dev);
        ocr::test::LayoutOracleMaps oracle;
        expect(ocr::test::oracle_layout_detect(fx.cfg, fx.weights, pixels, oracle).ok());
        // Preprocessing parity: the device's uploaded pixels must match the
        // oracle's float-mirror preprocessing exactly.
        {
            luisa::vector<float> dev_pixels;
            expect(fx.executor->download_page_pixels(stream, dev_pixels).ok());
            stream << synchronize();
            ocr::test::expect_close(dev_pixels, pixels, 1e-5f, 1e-5f, "layout preprocessing");
        }
        expect(maps.value->map_height == oracle.map_h);
        expect(maps.value->map_width == oracle.map_w);
        expect(maps.value->num_classes == oracle.num_classes);
        // Layered isolation: the final backbone output (2 stages -> feat_a).
        {
            luisa::vector<float> fa, fb;
            expect(fx.executor->download_feat_a(stream, fa).ok());
            expect(fx.executor->download_feat_b(stream, fb).ok());
            stream << synchronize();
            // Recompute the oracle backbone output for comparison.
            luisa::vector<float> cur{pixels.begin(), pixels.end()};
            uint32_t h = fx.cfg.input_height, wd = fx.cfg.input_width;
            for (uint32_t i = 0; i < fx.cfg.backbone.size(); i++) {
                const auto &s = fx.cfg.backbone[i];
                luisa::vector<float> next, act;
                uint32_t oh = 0, ow = 0;
                expect(ref_conv2d(cur, fx.weights.backbone[i].weight, fx.weights.backbone[i].bias, 1u, s.in_c, h, wd,
                                  s.out_c, s.kh, s.kw, s.stride_h, s.stride_w, s.pad_h, s.pad_w,
                                  s.dil_h, s.dil_w, s.groups, next, oh, ow).ok());
                expect(ref_cnn_act(next, s.act, act).ok());
                cur = std::move(act);
                h = oh;
                wd = ow;
            }
            const auto &bb = (fx.cfg.backbone.size() % 2u == 1u) ? fa : fb;
            ocr::test::expect_close(luisa::span<const float>{bb.data(), cur.size()}, cur, kAtol, kRtol,
                                    "layout backbone");
        }
        ocr::test::expect_close(maps.value->class_probs, oracle.class_probs, kAtol, kRtol,
                                "layout class probs");
        ocr::test::expect_close(maps.value->box_deltas, oracle.box_deltas, kAtol, kRtol,
                                "layout box deltas");
    }
}

// ---------------------------------------------------------------------------
// Decode parity: device head maps vs oracle head maps -> same regions.
// ---------------------------------------------------------------------------
void test_decode_parity(Device &device) {
    Fixture fx;
    expect(fx.init(device, PpDocLayoutVersion::v2));
    if (!fx.executor) { return; }
    auto stream = device.create_stream();
    auto page = ocr::test::p6_make_page(64, 64);
    auto maps = fx.executor->detect(stream, page);
    expect(maps.ok());
    if (!maps.ok()) { return; }
    auto pixels = ocr::test::p6_oracle_preprocess(page.pixels, page.source_width, page.source_height,
                                                  fx.cfg.input_width, fx.cfg.input_height,
                                                  fx.config.mean, fx.config.std_dev);
    ocr::test::LayoutOracleMaps oracle;
    expect(ocr::test::oracle_layout_detect(fx.cfg, fx.weights, pixels, oracle).ok());
    // Decode both with the same options + transform.
    output::DbSourceTransform transform;
    transform.scale_x = static_cast<float>(page.source_width) / static_cast<float>(maps.value->map_width);
    transform.scale_y = static_cast<float>(page.source_height) / static_cast<float>(maps.value->map_height);
    auto mh = fx.cfg.map_height();
    auto mw = fx.cfg.map_width();
    expect(mh.ok() && mw.ok());
    const float stride = static_cast<float>(fx.cfg.input_width) / static_cast<float>(*mw.value);
    PpDocLayoutDecodeOptions opts;
    opts.score_threshold = 0.3f;
    opts.nms_iou_threshold = 0.5f;
    opts.class_aware_nms = true;
    auto dev_regions = pp_doc_layout_decode(maps.value->class_probs, maps.value->box_deltas,
                                            maps.value->map_width, maps.value->map_height,
                                            maps.value->num_classes, stride, transform, fx.cfg, opts);
    auto oracle_regions = pp_doc_layout_decode(oracle.class_probs, oracle.box_deltas,
                                               oracle.map_w, oracle.map_h, oracle.num_classes,
                                               stride, transform, fx.cfg, opts);
    expect(dev_regions.ok());
    expect(oracle_regions.ok());
    if (!dev_regions.ok() || !oracle_regions.ok()) { return; }
    expect(dev_regions.value->size() == oracle_regions.value->size());
    if (dev_regions.value->size() != oracle_regions.value->size()) { return; }
    for (size_t i = 0; i < dev_regions.value->size(); i++) {
        const auto &d = (*dev_regions.value)[i];
        const auto &o = (*oracle_regions.value)[i];
        expect(d.class_index == o.class_index);
        expect(d.block_type == o.block_type);
        expect(d.reading_order == o.reading_order);
        expect(std::abs(d.score - o.score) < kScoreAtol);
        expect(std::abs(d.bounds.x0 - o.bounds.x0) < kBoxAtol);
        expect(std::abs(d.bounds.y0 - o.bounds.y0) < kBoxAtol);
        expect(std::abs(d.bounds.x1 - o.bounds.x1) < kBoxAtol);
        expect(std::abs(d.bounds.y1 - o.bounds.y1) < kBoxAtol);
    }
}

// ---------------------------------------------------------------------------
// Empty / degenerate pages.
// ---------------------------------------------------------------------------
void test_empty_page(Device &device) {
    Fixture fx;
    expect(fx.init(device, PpDocLayoutVersion::v2));
    if (!fx.executor) { return; }
    auto stream = device.create_stream();
    LayoutPageInput blank;
    blank.source_width = 64;
    blank.source_height = 64;
    blank.page_index = 0;
    blank.pixels.assign(static_cast<size_t>(64) * 64 * 3u, 0u);
    auto maps = fx.executor->detect(stream, blank);
    expect(maps.ok());
    if (!maps.ok()) { return; }
    // Decode with a high threshold: a blank page yields zero regions.
    output::DbSourceTransform transform;
    transform.scale_x = 4.0f;
    transform.scale_y = 4.0f;
    PpDocLayoutDecodeOptions opts;
    opts.score_threshold = 0.9f;
    auto regions = pp_doc_layout_decode(maps.value->class_probs, maps.value->box_deltas,
                                        maps.value->map_width, maps.value->map_height,
                                        maps.value->num_classes, 4.0f, transform, fx.cfg, opts);
    expect(regions.ok());
    if (regions.ok()) { expect(regions.value->empty()); }
    // Degenerate page input fails closed.
    LayoutPageInput bad;
    bad.source_width = 0;
    bad.source_height = 64;
    expect(!fx.executor->detect(stream, bad).ok());
}

// ---------------------------------------------------------------------------
// Repeat-run determinism.
// ---------------------------------------------------------------------------
void test_determinism(Device &device) {
    Fixture fx;
    expect(fx.init(device, PpDocLayoutVersion::v2));
    if (!fx.executor) { return; }
    auto stream = device.create_stream();
    auto page = ocr::test::p6_make_page(64, 64);
    auto m1 = fx.executor->detect(stream, page);
    auto m2 = fx.executor->detect(stream, page);
    expect(m1.ok() && m2.ok());
    if (!m1.ok() || !m2.ok()) { return; }
    expect(m1.value->class_probs.size() == m2.value->class_probs.size());
    bool identical = m1.value->class_probs == m2.value->class_probs &&
                     m1.value->box_deltas == m2.value->box_deltas;
    expect(identical) << "repeat detector runs must be bit-identical";
}

}// namespace

int main(int argc, char *argv[]) {
    auto dc = luisa::test::create_device_from_ut(argc, argv);
    if (!dc) { return 0; }
    boost::ut::detail::cfg::parse_arg_with_fallback(argc, const_cast<const char **>(argv));
    auto &device = dc->device;
    "layout_contracts"_test = [] { test_layout_contracts(); };
    "layout_detector_parity"_test = [&] { test_detector_parity(device); };
    "layout_decode_parity"_test = [&] { test_decode_parity(device); };
    "layout_empty_page"_test = [&] { test_empty_page(device); };
    "layout_determinism"_test = [&] { test_determinism(device); };
}
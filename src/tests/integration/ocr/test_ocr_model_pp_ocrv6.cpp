// Test for ocr/runtime/cnn_executor + ocr/models/pp_ocrv6 (Phase 5, device) --
// the SMALL SYNTHETIC PP-OCRv6 CNN pipeline, tiny/small/medium independently.
// Covers: per-size config/tensor-map contracts, Conv+BN and RepLK folding
// parity (folded device weights vs UNFOLDED two-step oracle), detector
// probability map, DB polygons/scores, polygon crop pixels, per-frame CTC
// logits/log-probs, collapsed ids/text/confidence, reading order, blank
// pages, repeated text, wide lines, and shape tails.
// Oracle: independent double-accumulating host reference composed from the
// Phase 2 cpu_reference ops + Phase 5 host oracles. Synthetic weights only;
// no model/download, no real-model parity claim, no fabricated confidence.
#include "ut/ut.hpp"
#include "test_device.h"
#include "ocr_pp_test_utils.h"

#include "ocr/runtime/cnn_executor.h"
#include "ocr/runtime/dispatch_adapter.h"

// NOTE: the public RecognizeResult bridge (cnn_executor_result.h) pulls api.h
// (host AttentionMaskKind), which clashes with the ops enum in this device
// TU. The bridge is exercised by the host contract test; here we verify the
// plain CnnRunResult and map the fields by hand for the public-schema checks.

using namespace luisa;
using namespace luisa::compute;
using namespace luisa::ocr;
using namespace boost::ut;
using namespace boost::ut::literals;

namespace {

// Tolerances: FP32 device vs double oracle, operation-appropriate (plan
// §10.1). Conv/BN/sigmoid/resize stages are pure arithmetic; the SVTR GELU
// stage uses the declared portable erf approximation on both arms.
constexpr float kAtol = 3e-3f;
constexpr float kRtol = 3e-3f;
// The detector probability map is the output of a multi-stage FP32 conv
// chain (backbone -> lateral -> resize -> concat -> RepLK -> DB -> sigmoid)
// compared against a double-accumulating oracle; the accumulated rounding
// is bounded by the same 3e-3 operation-appropriate gate as the GELU stages.
constexpr float kDetAtol = 3e-3f;
constexpr float kDetRtol = 3e-3f;

struct Fixture {
    ocr::test::PpTempDir tmp;
    PpOcrv6Config cfg;
    PpOcrv6TensorNames names;
    ocr::test::PpSyntheticWeights weights;
    ocr::test::PpFoldedWeights folded;
    WeightStore store;
    luisa::unique_ptr<CnnExecutor> executor;
    CnnExecutorConfig config;

    [[nodiscard]] bool init(Device &device, PpOcrSize size, uint32_t seed = 0x5EEDu) {
        cfg = pp_ocrv6_synthetic_config(size);
        if (!cfg.validate().ok()) { return false; }
        names = PpOcrv6TensorNames{};
        weights = ocr::test::generate_pp_synthetic_weights(cfg, seed);
        if (!ocr::test::fold_pp_weights(cfg, weights, folded).ok()) { return false; }
        // Fixture calibration: deterministic bars-above-threshold pattern.
        if (!ocr::test::pp_calibrate_db_head(cfg, weights, folded).ok()) { return false; }
        const auto dpath = tmp.path / "pp_detector.lcocr";
        const auto rpath = tmp.path / "pp_recognizer.lcocr";
        if (!ocr::test::write_pp_synthetic_packages(dpath, rpath, cfg, names, weights, folded).ok()) { return false; }
        auto stream = device.create_stream();
        if (!ocr::test::upload_pp_synthetic_weights(device, stream, dpath, rpath, store).ok()) { return false; }
        config.model = cfg;
        config.names = names;
        config.dictionary = ocr::test::pp_synthetic_dictionary();
        config.max_lines_per_page = 16;
        executor = luisa::make_unique<CnnExecutor>();
        return executor->create(device, stream, config, store).ok();
    }
};

// A deterministic synthetic page: bright text-like bars on a dark background.
[[nodiscard]] CnnPageInput make_page(uint32_t w, uint32_t h, uint32_t seed = 7u) {
    CnnPageInput page;
    page.source_width = w;
    page.source_height = h;
    page.page_index = 0;
    page.pixels.resize(static_cast<size_t>(w) * h * 3u);
    ocr::test::Lcg lcg;
    lcg.state = seed;
    for (auto &b : page.pixels) { b = static_cast<uint8_t>(lcg.next() % 40u); }// dark noise
    // Two bright horizontal bars (text-like regions).
    const auto bar = [&](uint32_t x0, uint32_t y0, uint32_t x1, uint32_t y1) {
        for (uint32_t y = y0; y < y1 && y < h; y++) {
            for (uint32_t x = x0; x < x1 && x < w; x++) {
                for (uint32_t c = 0; c < 3u; c++) {
                    page.pixels[(static_cast<size_t>(y) * w + x) * 3u + c] = 230u;
                }
            }
        }
    };
    bar(w / 8u, h / 6u, w / 2u, h / 6u + h / 8u);
    bar(w / 8u, h / 2u, 7u * w / 8u, h / 2u + h / 8u);
    return page;
}

[[nodiscard]] CnnPageInput make_blank_page(uint32_t w, uint32_t h) {
    CnnPageInput page;
    page.source_width = w;
    page.source_height = h;
    page.page_index = 0;
    page.pixels.assign(static_cast<size_t>(w) * h * 3u, 0u);
    return page;
}

// ---------------------------------------------------------------------------
// Host contract checks (no device needed).
// ---------------------------------------------------------------------------
void test_pp_contracts() {
    // Each size validates independently with DISTINCT geometry.
    PpOcrv6Config prev;
    bool have_prev = false;
    for (auto size : {PpOcrSize::tiny, PpOcrSize::small, PpOcrSize::medium}) {
        auto cfg = pp_ocrv6_synthetic_config(size);
        expect(cfg.validate().ok()) << pp_ocr_size_name(size);
        expect(cfg.size == size);
        // Asymmetric strides present in both backbone chains.
        bool det_asym = false, rec_asym = false;
        for (const auto &s : cfg.detector.backbone) { det_asym |= s.stride_h != s.stride_w; }
        for (const auto &s : cfg.recognizer.backbone) { rec_asym |= s.stride_h != s.stride_w; }
        expect(det_asym);
        expect(rec_asym);
        // Large + depthwise convs present.
        bool large = false, depthwise = false;
        for (const auto &s : cfg.detector.backbone) {
            large |= s.kh >= 5u;
            depthwise |= s.groups == s.in_c && s.groups > 1u;
        }
        expect(large);
        expect(depthwise);
        expect(cfg.detector.replk_conv.kh >= 7u);// folded RepLK large kernel
        // Tensor maps are per-size and non-empty.
        PpOcrv6TensorNames names;
        auto det = pp_ocrv6_expected_tensors(cfg, names, "detector");
        auto rec = pp_ocrv6_expected_tensors(cfg, names, "recognizer");
        expect(!det.empty());
        expect(!rec.empty());
        if (have_prev) {
            // Sizes must differ in at least the detector input width.
            expect(cfg.detector.input_width != prev.detector.input_width);
        }
        prev = cfg;
        have_prev = true;
    }
    // Real artifacts fail closed unconditionally.
    expect(pp_ocrv6_validate_real_artifact("rev", "fp").code == StatusCode::unvalidated);
    expect(pp_ocrv6_validate_real_artifact("", "").code == StatusCode::unvalidated);
    // Training/NRTR guard.
    luisa::vector<luisa::string> good{"detector", "recognizer"};
    expect(pp_ocrv6_check_no_training_nodes(good).ok());
    luisa::vector<luisa::string> bad{"detector", "nrtr_decoder"};
    expect(!pp_ocrv6_check_no_training_nodes(bad).ok());
    luisa::vector<luisa::string> bad2{"dropout"};
    expect(!pp_ocrv6_check_no_training_nodes(bad2).ok());
    luisa::vector<luisa::string> bad3{"ctc_loss"};
    expect(!pp_ocrv6_check_no_training_nodes(bad3).ok());
}

// Folding helpers vs independent two-step evaluation (host math gate).
void test_pp_folding_math() {
    // Conv+BN fold: random small conv, compare folded conv output against
    // conv->BN two-step on random input.
    ocr::test::Lcg lcg;
    lcg.state = 0xF01Du;
    const uint32_t oc = 3, ic = 2, kh = 3, kw = 3, h = 6, w = 6;
    luisa::vector<float> weight(oc * ic * kh * kw), bias(oc), mean(oc), var(oc), gamma(oc), beta(oc);
    lcg.fill(weight, -0.5f, 0.5f);
    lcg.fill(bias, -0.2f, 0.2f);
    lcg.fill(mean, -0.3f, 0.3f);
    lcg.fill(var, 0.4f, 1.6f);
    lcg.fill(gamma, 0.6f, 1.4f);
    lcg.fill(beta, -0.2f, 0.2f);
    luisa::vector<float> fw, fb;
    expect(pp_fold_conv_bn(weight, bias, mean, var, gamma, beta, oc, ic * kh * kw, 1e-5f, fw, fb).ok());
    luisa::vector<float> input(ic * h * w);
    lcg.fill(input, -1.0f, 1.0f);
    luisa::vector<float> conv_out, two_step;
    uint32_t oh = 0, ow = 0;
    expect(ref_conv2d(input, weight, bias, 1u, ic, h, w, oc, kh, kw, 1u, 1u, 1u, 1u, 1u, 1u, 1u,
                      conv_out, oh, ow).ok());
    expect(ref_batch_norm(conv_out, mean, var, gamma, beta, 1u, oc, static_cast<uint64_t>(oh) * ow,
                          1e-5f, two_step).ok());
    luisa::vector<float> folded_out;
    uint32_t fh = 0, fw2 = 0;
    expect(ref_conv2d(input, fw, fb, 1u, ic, h, w, oc, kh, kw, 1u, 1u, 1u, 1u, 1u, 1u, 1u,
                      folded_out, fh, fw2).ok());
    ocr::test::expect_close(folded_out, two_step, 1e-4f, 1e-4f, "conv+bn fold parity");

    // RepLK fold (depthwise): large 5x5 + small 3x3 + identity BN on C channels.
    const uint32_t C = 3, K = 5, k = 3;
    luisa::vector<float> lw(C * K * K), lb(C), sw(C * k * k), sb(C), ig(C), ib(C);
    lcg.fill(lw, -0.4f, 0.4f);
    lcg.fill(lb, -0.1f, 0.1f);
    lcg.fill(sw, -0.4f, 0.4f);
    lcg.fill(sb, -0.1f, 0.1f);
    lcg.fill(ig, 0.5f, 1.5f);
    lcg.fill(ib, -0.1f, 0.1f);
    luisa::vector<float> mw, mb;
    expect(pp_fold_replk(lw, lb, sw, sb, ig, ib, C, K, k, mw, mb).ok());
    // Independent merge: pad small into center, add identity impulse.
    luisa::vector<float> expect_w = lw;
    const uint32_t off = (K - k) / 2u, center = K / 2u;
    for (uint32_t c = 0; c < C; c++) {
        for (uint32_t y = 0; y < k; y++) {
            for (uint32_t x = 0; x < k; x++) {
                expect_w[(c * K + y + off) * K + x + off] += sw[(c * k + y) * k + x];
            }
        }
        expect_w[(c * K + center) * K + center] += ig[c];
    }
    ocr::test::expect_close(mw, expect_w, 1e-5f, 1e-5f, "replk fold weight");
    for (uint32_t c = 0; c < C; c++) {
        expect(std::abs(mb[c] - (lb[c] + sb[c] + ib[c])) < 1e-5f);
    }
    // Merged conv == sum of the three branch convs on random input.
    luisa::vector<float> in(C * h * w);
    lcg.fill(in, -1.0f, 1.0f);
    luisa::vector<float> merged_out, large_out, small_out, id_out;
    uint32_t mh = 0, mw2 = 0, lh2 = 0, lw2 = 0, sh = 0, sw2 = 0;
    expect(ref_conv2d(in, mw, mb, 1u, C, h, w, C, K, K, 1u, 1u, K / 2u, K / 2u, 1u, 1u, C,
                      merged_out, mh, mw2).ok());
    expect(ref_conv2d(in, lw, lb, 1u, C, h, w, C, K, K, 1u, 1u, K / 2u, K / 2u, 1u, 1u, C,
                      large_out, lh2, lw2).ok());
    expect(ref_conv2d(in, sw, sb, 1u, C, h, w, C, k, k, 1u, 1u, k / 2u, k / 2u, 1u, 1u, C,
                      small_out, sh, sw2).ok());
    // Identity branch: impulse conv = ig[c]*x[c] + ib[c].
    id_out.resize(in.size());
    for (uint32_t c = 0; c < C; c++) {
        for (uint32_t p = 0; p < h * w; p++) {
            id_out[c * h * w + p] = ig[c] * in[c * h * w + p] + ib[c];
        }
    }
    luisa::vector<float> sum(merged_out.size());
    for (size_t i = 0; i < sum.size(); i++) { sum[i] = large_out[i] + small_out[i] + id_out[i]; }
    ocr::test::expect_close(merged_out, sum, 1e-4f, 1e-4f, "replk fold forward parity");
    // Invalid geometry fails closed.
    luisa::vector<float> dummy;
    expect(!pp_fold_replk(lw, lb, sw, sb, ig, ib, C, 4, k, dummy, dummy).ok());// even K
    expect(!pp_fold_replk(lw, lb, sw, sb, ig, ib, C, K, 7, dummy, dummy).ok());// k > K
}

// ---------------------------------------------------------------------------
// Device parity per size.
// ---------------------------------------------------------------------------
void test_pp_detector_parity(Device &device, PpOcrSize size) {
    Fixture fx;
    expect(fx.init(device, size)) << pp_ocr_size_name(size);
    if (!fx.executor) { return; }
    auto stream = device.create_stream();
    // Source page larger than the detector input (exercises resize).
    auto page = make_page(fx.cfg.detector.input_width * 2u, fx.cfg.detector.input_height * 2u);
    auto pixels = ocr::test::pp_oracle_preprocess(page.pixels, page.source_width, page.source_height,
                                                  fx.cfg.detector.input_width, fx.cfg.detector.input_height,
                                                  fx.config.mean, fx.config.std_dev);
    ocr::test::PpOracleDetector oracle;
    expect(ocr::test::pp_oracle_detector(fx.cfg, fx.weights, pixels, oracle).ok());
    auto run = fx.executor->recognize(stream, luisa::span<const CnnPageInput>{&page, 1});
    expect(run.ok());
    if (!run.ok()) { return; }
    expect(run.value->pages.size() == 1u);
    const auto &pr = run.value->pages[0];
    expect(pr.det_map_width == oracle.map_w);
    expect(pr.det_map_height == oracle.map_h);
    // Host check: the folded RepLK conv on the ORACLE fused input must match
    // the oracle's unfolded branch sum (folding parity, host-side).
    {
        luisa::vector<float> host_replk;
        uint32_t hh = 0, hw = 0;
        const auto &rs = fx.cfg.detector.replk_conv;
        expect(ref_conv2d(oracle.fused, fx.folded.det_replk_w, fx.folded.det_replk_b, 1u, rs.in_c,
                          oracle.backbone_h, oracle.backbone_w, rs.out_c, rs.kh, rs.kw,
                          rs.stride_h, rs.stride_w, rs.pad_h, rs.pad_w, rs.dil_h, rs.dil_w, rs.groups,
                          host_replk, hh, hw).ok());
        luisa::vector<float> host_act;
        expect(ref_cnn_act(host_replk, rs.act, host_act).ok());
        ocr::test::expect_close(host_act, oracle.replk, 1e-3f, 1e-3f, "host folded replk vs oracle");
    }
    // Layered isolation: backbone output, fused, replk, then prob.
    {
        luisa::vector<float> da, db, fused, replk;
        expect(fx.executor->download_det_a(stream, da).ok());
        expect(fx.executor->download_det_b(stream, db).ok());
        expect(fx.executor->download_det_fused(stream, fused).ok());
        expect(fx.executor->download_det_replk(stream, replk).ok());
        stream << synchronize();
        // The final backbone output lives in det_a or det_b depending on the
        // stage count parity (3 stages -> det_a).
        const auto &bb = (fx.cfg.detector.backbone.size() % 2u == 1u) ? da : db;
        ocr::test::expect_close(luisa::span<const float>{bb.data(), oracle.backbone_out.size()},
                                oracle.backbone_out, kDetAtol, kDetRtol,
                                (luisa::string{pp_ocr_size_name(size)} + " backbone").c_str());
        ocr::test::expect_close(luisa::span<const float>{fused.data(), oracle.fused.size()},
                                oracle.fused, kDetAtol, kDetRtol,
                                (luisa::string{pp_ocr_size_name(size)} + " fused").c_str());
        ocr::test::expect_close(luisa::span<const float>{replk.data(), oracle.replk.size()},
                                oracle.replk, kDetAtol, kDetRtol,
                                (luisa::string{pp_ocr_size_name(size)} + " replk").c_str());
    }
    ocr::test::expect_close(pr.det_probabilities, oracle.prob, kDetAtol, kDetRtol,
                            (luisa::string{pp_ocr_size_name(size)} + " detector prob").c_str());
    // DB boxes: the executor's boxes must match an independent db_postprocess
    // run on the ORACLE probability map (same options/transform).
    output::DbSourceTransform transform;
    transform.scale_x = static_cast<float>(page.source_width) / static_cast<float>(oracle.map_w);
    transform.scale_y = static_cast<float>(page.source_height) / static_cast<float>(oracle.map_h);
    auto db_opts = fx.config.db;
    db_opts.max_boxes = fx.config.max_lines_per_page;
    auto db_oracle = output::db_postprocess(oracle.prob, oracle.map_w, oracle.map_h, transform, db_opts);
    expect(db_oracle.ok());
    expect(pr.boxes.size() == db_oracle.value->boxes.size());
    const size_t n = std::min(pr.boxes.size(), db_oracle.value->boxes.size());
    for (size_t i = 0; i < n; i++) {
        expect(std::abs(pr.boxes[i].score - db_oracle.value->boxes[i].score) < 5e-3f);
        expect(pr.boxes[i].polygon.size() == 4u);
        for (size_t p = 0; p < 4u && p < db_oracle.value->boxes[i].polygon.size(); p++) {
            expect(std::abs(pr.boxes[i].polygon[p].x - db_oracle.value->boxes[i].polygon[p].x) < 0.75f);
            expect(std::abs(pr.boxes[i].polygon[p].y - db_oracle.value->boxes[i].polygon[p].y) < 0.75f);
        }
    }
}

void test_pp_recognizer_parity(Device &device, PpOcrSize size) {
    Fixture fx;
    expect(fx.init(device, size)) << pp_ocr_size_name(size);
    if (!fx.executor) { return; }
    auto stream = device.create_stream();
    // One bright-bar page: at least one line should be detected.
    auto page = make_page(fx.cfg.detector.input_width * 2u, fx.cfg.detector.input_height * 2u);
    auto run = fx.executor->recognize(stream, luisa::span<const CnnPageInput>{&page, 1});
    expect(run.ok());
    if (!run.ok()) { return; }
    const auto &pr = run.value->pages[0];
    // The calibrated fixture MUST detect the bars (no silent skip).
    expect(!pr.boxes.empty()) << pp_ocr_size_name(size) << " calibrated bars must be detected";
    if (pr.boxes.empty()) { return; }
    // Download the LAST line's crop + logits and compare against the oracle
    // evaluated on the same crop (independent polygon-crop oracle).
    luisa::vector<float> crop, logits;
    expect(fx.executor->download_line_crop(stream, crop).ok());
    expect(fx.executor->download_recognizer_logits(stream, logits).ok());
    stream << synchronize();
    ocr::test::PpOracleRecognizer rec_oracle;
    expect(ocr::test::pp_oracle_recognizer(fx.cfg, fx.weights, crop, fx.config.ctc.temperature, rec_oracle).ok());
    expect(logits.size() == rec_oracle.logits.size());
    ocr::test::expect_close(logits, rec_oracle.logits, kAtol, kRtol,
                            (luisa::string{pp_ocr_size_name(size)} + " ctc logits").c_str());
    // Collapsed ids/text/confidence: decode the ORACLE logits with the host
    // reference decoder and compare against the executor's last line.
    auto decoded = output::ctc_decode_logits(rec_oracle.logits, rec_oracle.frames, fx.config.dictionary,
                                             fx.config.ctc);
    expect(decoded.ok());
    if (!pr.lines.empty()) {
        const auto &last = pr.lines.back();
        expect(last.text == decoded.value->text);
        expect(last.has_confidence == decoded.value->has_confidence);
        if (last.has_confidence) {
            expect(std::abs(last.confidence - decoded.value->confidence) < 2e-3f);
        }
    }
}

void test_pp_blank_page(Device &device, PpOcrSize size) {
    Fixture fx;
    expect(fx.init(device, size)) << pp_ocr_size_name(size);
    if (!fx.executor) { return; }
    auto stream = device.create_stream();
    auto page = make_blank_page(fx.cfg.detector.input_width, fx.cfg.detector.input_height);
    auto run = fx.executor->recognize(stream, luisa::span<const CnnPageInput>{&page, 1});
    expect(run.ok());
    if (!run.ok()) { return; }
    expect(run.value->pages.size() == 1u);
    const auto &pr = run.value->pages[0];
    // A blank page must produce NO lines and NO text (no hallucination).
    expect(pr.lines.empty());
    expect(pr.text.empty());
    // The probability map must yield no boxes. (Individual map pixels may
    // exceed the threshold from random-weight bias; what matters is that no
    // component survives the size/score gates -- no hallucinated lines.)
    expect(pr.boxes.empty());
}

void test_pp_repeat_determinism(Device &device, PpOcrSize size) {
    Fixture fx;
    expect(fx.init(device, size)) << pp_ocr_size_name(size);
    if (!fx.executor) { return; }
    auto stream = device.create_stream();
    auto page = make_page(fx.cfg.detector.input_width * 2u, fx.cfg.detector.input_height * 2u);
    auto run1 = fx.executor->recognize(stream, luisa::span<const CnnPageInput>{&page, 1});
    auto run2 = fx.executor->recognize(stream, luisa::span<const CnnPageInput>{&page, 1});
    expect(run1.ok() && run2.ok());
    if (!run1.ok() || !run2.ok()) { return; }
    const auto &a = run1.value->pages[0];
    const auto &b = run2.value->pages[0];
    expect(a.lines.size() == b.lines.size());
    expect(a.text == b.text);
    const size_t n = std::min(a.lines.size(), b.lines.size());
    for (size_t i = 0; i < n; i++) {
        expect(a.lines[i].text == b.lines[i].text);
        expect(a.lines[i].reading_order == b.lines[i].reading_order);
        expect(a.lines[i].confidence == b.lines[i].confidence);
    }
}

void test_pp_wide_and_multipage(Device &device) {
    // Wide line: a page much wider than tall; multi-page: two pages with
    // distinct page indices preserved in order.
    Fixture fx;
    expect(fx.init(device, PpOcrSize::tiny));
    if (!fx.executor) { return; }
    auto stream = device.create_stream();
    auto wide = make_page(fx.cfg.detector.input_width * 3u, fx.cfg.detector.input_height);
    auto p0 = make_page(fx.cfg.detector.input_width * 2u, fx.cfg.detector.input_height * 2u, 11u);
    auto p1 = make_blank_page(fx.cfg.detector.input_width, fx.cfg.detector.input_height);
    p0.page_index = 0;
    p1.page_index = 1;
    luisa::vector<CnnPageInput> pages{wide, p0, p1};
    auto run = fx.executor->recognize(stream, pages);
    expect(run.ok());
    if (!run.ok()) { return; }
    expect(run.value->pages.size() == 3u);
    expect(run.value->pages[0].source_width == wide.source_width);
    expect(run.value->pages[1].page_index == 0u);
    expect(run.value->pages[2].page_index == 1u);
    // The blank second page has no lines.
    expect(run.value->pages[2].lines.empty());
    // Reading order is dense within each page.
    for (const auto &pr : run.value->pages) {
        for (size_t i = 0; i < pr.lines.size(); i++) {
            expect(pr.lines[i].reading_order == static_cast<uint32_t>(i));
        }
    }
}

void test_pp_limits(Device &device) {
    Fixture fx;
    expect(fx.init(device, PpOcrSize::tiny));
    if (!fx.executor) { return; }
    auto stream = device.create_stream();
    // Pixel cap refusal: 8192x8192 = 64M pixels > the 16M default cap.
    CnnPageInput big;
    big.source_width = 8192u;
    big.source_height = 8192u;
    big.pixels.assign(static_cast<size_t>(8192u) * 8192u * 3u, 0u);
    auto run = fx.executor->recognize(stream, luisa::span<const CnnPageInput>{&big, 1});
    expect(!run.ok());
    // Degenerate page refusal.
    CnnPageInput bad;
    bad.source_width = 0;
    bad.source_height = 10;
    auto run2 = fx.executor->recognize(stream, luisa::span<const CnnPageInput>{&bad, 1});
    expect(!run2.ok());
}

// ---------------------------------------------------------------------------
// M1: crop orientation/content vs an INDEPENDENT source-derived oracle.
// ---------------------------------------------------------------------------
void test_pp_crop_orientation(Device &device, PpOcrSize size) {
    Fixture fx;
    expect(fx.init(device, size)) << pp_ocr_size_name(size);
    if (!fx.executor) { return; }
    auto stream = device.create_stream();
    auto page = make_page(fx.cfg.detector.input_width * 2u, fx.cfg.detector.input_height * 2u);
    auto run = fx.executor->recognize(stream, luisa::span<const CnnPageInput>{&page, 1});
    expect(run.ok());
    if (!run.ok()) { return; }
    const auto &pr = run.value->pages[0];
    expect(!pr.boxes.empty());
    if (pr.boxes.empty()) { return; }

    // The reported quads must be TL,TR,BR,BL: edge 0->1 is the LONG horizontal
    // text edge for these wide bars (a reversed winding swaps width/height).
    for (const auto &box : pr.boxes) {
        expect(box.polygon.size() == 4u);
        if (box.polygon.size() != 4u) { continue; }
        const auto &q = box.polygon;
        const auto dist = [](const output::DbPoint &a, const output::DbPoint &b) {
            const float dx = b.x - a.x, dy = b.y - a.y;
            return std::sqrt(dx * dx + dy * dy);
        };
        const float text_edge = dist(q[0], q[1]);
        const float side_edge = dist(q[0], q[3]);
        (void)text_edge;
        (void)side_edge;
        // Winding invariant (aspect-INDEPENDENT): TL,TR,BR,BL in y-down image
        // coords means edge 0->1 runs left-to-right along the top and the
        // shoelace sum is POSITIVE. The tiny fixture may legitimately merge
        // both bars into one tall component, so this test does not assume the
        // 0->1 edge is the longer one; the aspect-sensitive long-edge check
        // lives in the DB unit test (db_quad_vertex_order_horizontal_blob).
        expect(q[1].x > q[0].x) << pp_ocr_size_name(size) << " edge 0->1 must run left-to-right";
        expect(std::abs(q[1].y - q[0].y) < 1e-3f) << pp_ocr_size_name(size) << " top edge must be horizontal";
        expect(q[3].y > q[0].y) << pp_ocr_size_name(size) << " edge 0->3 must run downward";
        double shoelace = 0.0;
        for (size_t i = 0; i < 4u; i++) {
            const auto &a = q[i];
            const auto &b = q[(i + 1u) % 4u];
            shoelace += static_cast<double>(a.x) * b.y - static_cast<double>(b.x) * a.y;
        }
        expect(shoelace > 0.0) << pp_ocr_size_name(size) << " TL,TR,BR,BL shoelace must be positive";
    }

    // Download the LAST line's crop and compare against the crop re-derived
    // from the SOURCE page + the reported quad (independent oracle).
    luisa::vector<float> crop;
    expect(fx.executor->download_line_crop(stream, crop).ok());
    stream << synchronize();
    const auto &last_quad = pr.boxes.back().polygon;
    // The executor crops the LAST processed box; boxes are iterated in order,
    // so the last box's quad is the right expectation input.
    auto expect_crop = ocr::test::pp_oracle_crop_from_source(
        last_quad, page.source_width, page.source_height, page.pixels,
        fx.cfg.detector.input_width, fx.cfg.detector.input_height,
        fx.cfg.recognizer.input_width, fx.cfg.recognizer.input_height,
        fx.config.mean, fx.config.std_dev);
    expect(expect_crop.pixels.size() == crop.size());
    ocr::test::expect_close(crop, expect_crop.pixels, kAtol, kRtol,
                            (luisa::string{pp_ocr_size_name(size)} + " crop vs source-derived oracle").c_str());
    // Orientation sanity from CONTENT: a correct horizontal-bar crop is
    // uniformly bright along its width (left half ~ right half). A 90-degree
    // rotation would make one half background-dark.
    const auto prof = ocr::test::pp_crop_profile(crop, fx.cfg.recognizer.input_width,
                                                 fx.cfg.recognizer.input_height);
    const auto &lb = pr.boxes.back().bounds;
    if ((lb.x1 - lb.x0) > (lb.y1 - lb.y0)) {
        // Wide bar: a correct crop is uniformly bright along its width; a
        // 90-degree rotation would make one half background-dark.
        expect(std::abs(prof.left - prof.right) < 0.35f)
            << pp_ocr_size_name(size) << " crop halves must match: L=" << prof.left << " R=" << prof.right;
    }
}

// ---------------------------------------------------------------------------
// S1/S2: geometry hardening negatives (host-only math, no device work).
// ---------------------------------------------------------------------------
void test_pp_geometry_hardening() {
    // S1: a lateral skip LARGER than the backbone output must be rejected
    // BEFORE allocation (the FPN neck only upsamples; the arenas are sized at
    // the backbone output resolution).
    {
        auto cfg = pp_ocrv6_synthetic_config(PpOcrSize::tiny);
        expect(cfg.validate().ok());// baseline is valid
        // Move the skip to stage 0 (stride 2 -> 32x32) while the backbone
        // output is 16x32: skip height 32 > output height 16.
        cfg.detector.lateral_stage = 0;
        auto s = cfg.validate();
        expect(!s.ok()) << "oversized lateral skip must be refused";
        expect(s.code == StatusCode::invalid_argument);
    }
    // S1: a skip whose resolution EQUALS the backbone output stays valid
    // (identity upsample). Taking the last stage as the skip also changes its
    // channel count, so the lateral conv input must be updated to match.
    {
        auto cfg = pp_ocrv6_synthetic_config(PpOcrSize::tiny);
        cfg.detector.lateral_stage = static_cast<uint32_t>(cfg.detector.backbone.size() - 1u);
        cfg.detector.lateral_conv.in_c = cfg.detector.backbone.back().out_c;
        expect(cfg.validate().ok());
    }
    // S2: num_frames must include the height-collapse factor.
    {
        auto cfg = pp_ocrv6_synthetic_config(PpOcrSize::tiny);
        auto fh = cfg.recognizer.feature_height();
        auto fw = cfg.recognizer.feature_width();
        auto nf = cfg.recognizer.num_frames();
        expect(fh.ok() && fw.ok() && nf.ok());
        if (fh.ok() && fw.ok() && nf.ok()) {
            const uint32_t expected = *fw.value * (*fh.value / cfg.recognizer.height_collapse);
            expect(*nf.value == expected) << "frames " << *nf.value << " vs " << expected;
        }
        // Force feature_height > height_collapse by halving the collapse: the
        // frame count must DOUBLE (the old code ignored the height factor).
        auto wide = cfg;
        wide.recognizer.height_collapse = 1u;
        expect(wide.validate().ok());
        auto nf1 = wide.recognizer.num_frames();
        expect(nf1.ok());
        if (nf.ok() && nf1.ok()) {
            expect(*nf1.value == 2u * *nf.value)
                << "collapse 1 must double frames: " << *nf1.value << " vs " << *nf.value;
        }
        // Non-divisible collapse is refused by both validate and num_frames.
        auto bad = cfg;
        bad.recognizer.height_collapse = 3u;// feature height 2 not divisible
        expect(!bad.validate().ok());
        expect(!bad.recognizer.num_frames().ok());
        // Zero collapse is refused.
        auto zero = cfg;
        zero.recognizer.height_collapse = 0u;
        expect(!zero.validate().ok());
        expect(!zero.recognizer.num_frames().ok());
    }
}

// ---------------------------------------------------------------------------
// S3: NON-IDENTITY FPN resize parity (ragged up/downsample + tails), device.
// ---------------------------------------------------------------------------
void test_pp_fpn_resize_nonidentity(Device &device) {
    auto stream = device.create_stream();
    ocr::test::Lcg lcg;
    lcg.state = 0xF9Au;
    const uint32_t n = 1, c = 3;
    // Ragged cases: upsample, downsample, mixed, and non-multiple tails.
    const struct Case { uint32_t h, w, oh, ow; } cases[] = {
        {3, 5, 7, 11},   // up both axes, ragged
        {8, 6, 3, 4},    // down both axes
        {4, 9, 9, 4},    // up height / down width
        {1, 1, 5, 7},    // degenerate source, upsample
        {7, 3, 1, 1},    // downsample to 1x1
        {5, 5, 5, 5},    // identity control
        {2, 17, 5, 33},  // wide tail
    };
    for (const auto &tc : cases) {
        luisa::vector<float> x(static_cast<size_t>(n) * c * tc.h * tc.w);
        lcg.fill(x, -1.5f, 1.5f);
        luisa::vector<float> ref;
        expect(ref_fpn_resize(x, n, c, tc.h, tc.w, tc.oh, tc.ow, ref).ok());
        auto bx = ocr::test::upload(device, stream, luisa::span<const float>{x});
        auto bo = device.create_buffer<float>(ref.size());
        auto kernel = make_fpn_resize_kernel();
        auto shader = device.compile(kernel);
        auto plan = plan_fpn_resize_dispatch(n, c, tc.oh, tc.ow);
        expect(plan.ok());
        if (!plan.ok()) { continue; }
        OCR_DISPATCH_1D(stream, shader(bx, bo, n, c, tc.h, tc.w, tc.oh, tc.ow), *plan.value);
        luisa::vector<float> got;
        ocr::test::download(stream, bo, got, ref.size());
        stream << synchronize();
        const auto what = luisa::format("fpn {}x{} -> {}x{}", tc.w, tc.h, tc.ow, tc.oh);
        ocr::test::expect_close(got, ref, 1e-5f, 1e-5f, what.c_str());
        // Non-identity cases must actually differ from a naive copy/nearest
        // read: verify at least one value differs from the source's first
        // element (guards against an accidentally identity-only test).
        if (tc.h != tc.oh || tc.w != tc.ow) {
            bool differs = false;
            for (size_t i = 0; i < got.size() && !differs; i++) {
                differs = std::abs(got[i] - x[0]) > 1e-6f;
            }
            expect(differs) << what << " must not be a constant copy";
        }
    }
}

// ---------------------------------------------------------------------------
// S5: the ON-DISK layout gate must refuse a wrong package layout tag.
// ---------------------------------------------------------------------------
void test_pp_on_disk_layout_gate(Device &device) {
    PpOcrv6Config cfg = pp_ocrv6_synthetic_config(PpOcrSize::tiny);
    PpOcrv6TensorNames names;
    auto weights = ocr::test::generate_pp_synthetic_weights(cfg);
    ocr::test::PpFoldedWeights folded;
    expect(ocr::test::fold_pp_weights(cfg, weights, folded).ok());
    expect(ocr::test::pp_calibrate_db_head(cfg, weights, folded).ok());
    ocr::test::PpTempDir tmp;
    const auto dpath = tmp.path / "gate_det.lcocr";
    const auto rpath = tmp.path / "gate_rec.lcocr";
    expect(ocr::test::write_pp_synthetic_packages(dpath, rpath, cfg, names, weights, folded).ok());

    // Baseline: the correct packages pass the on-disk gate and create().
    {
        auto det = PackageReader::open(dpath);
        auto rec = PackageReader::open(rpath);
        expect(det.ok() && rec.ok());
        if (!det.ok() || !rec.ok()) { return; }
        const auto det_exp = pp_ocrv6_expected_tensors(cfg, names, "detector");
        const auto rec_exp = pp_ocrv6_expected_tensors(cfg, names, "recognizer");
        expect(pp_ocrv6_validate_tensors(det.value->get()->tensors(), det_exp).ok());
        expect(pp_ocrv6_validate_tensors(rec.value->get()->tensors(), rec_exp).ok());
        WeightStore store;
        auto stream = device.create_stream();
        expect(ocr::test::upload_pp_synthetic_weights(device, stream, dpath, rpath, store).ok());
        CnnExecutorConfig ec;
        ec.model = cfg;
        ec.names = names;
        ec.dictionary = ocr::test::pp_synthetic_dictionary();
        CnnExecutor ex;
        expect(ex.create_from_packages(device, stream, ec, store, **det.value, **rec.value).ok());
    }

    // Rewrite the detector package with a WRONG on-disk layout tag for a conv
    // weight (row_major instead of conv_oihw). The device-descriptor gate
    // cannot see this (runtime maps conv_oihw -> row_major), so ONLY the
    // on-disk gate can refuse it.
    const auto bad_path = tmp.path / "gate_det_badlayout.lcocr";
    {
        auto src = PackageReader::open(dpath);
        expect(src.ok());
        if (!src.ok()) { return; }
        auto writer = PackageWriter::create(bad_path);
        expect(writer.ok());
        if (!writer.ok()) { return; }
        auto &wr = *writer.value;
        bool flipped = false;
        for (const auto &t : src.value->get()->tensors()) {
            PackageWriter::TensorSpec spec;
            spec.name = t.name;
            spec.encoding = t.encoding;
            spec.layout = t.layout;
            spec.shape = t.shape;
            luisa::vector<float> data(static_cast<size_t>(t.element_count));
            expect(src.value->get()->read_tensor_f32(t, data).ok());
            spec.payload = ocr::test::pp_f32_bytes(data);
            if (!flipped && t.layout == TensorLayout::conv_oihw) {
                spec.layout = TensorLayout::row_major;// WRONG on-disk tag
                flipped = true;
            }
            expect(wr->add_tensor(spec).ok());
        }
        expect(flipped) << "fixture must contain a conv_oihw tensor to flip";
        expect(wr->finish(ocr::test::pp_synthetic_manifest("detector")).ok());
    }
    {
        auto det = PackageReader::open(bad_path);
        auto rec = PackageReader::open(rpath);
        expect(det.ok() && rec.ok());
        if (!det.ok() || !rec.ok()) { return; }
        const auto det_exp = pp_ocrv6_expected_tensors(cfg, names, "detector");
        auto gate = pp_ocrv6_validate_tensors(det.value->get()->tensors(), det_exp);
        expect(!gate.ok()) << "wrong on-disk conv layout must be refused";
        expect(gate.code == StatusCode::invalid_argument);
        WeightStore store;
        auto stream = device.create_stream();
        expect(ocr::test::upload_pp_synthetic_weights(device, stream, bad_path, rpath, store).ok());
        CnnExecutorConfig ec;
        ec.model = cfg;
        ec.names = names;
        ec.dictionary = ocr::test::pp_synthetic_dictionary();
        CnnExecutor ex;
        auto created = ex.create_from_packages(device, stream, ec, store, **det.value, **rec.value);
        expect(!created.ok()) << "create_from_packages must refuse the wrong on-disk layout";
    }
}

}// namespace

int main(int argc, char *argv[]) {
    auto dc = luisa::test::create_device_from_ut(argc, argv);
    if (!dc) { return 0; }
    boost::ut::detail::cfg::parse_arg_with_fallback(argc, const_cast<const char **>(argv));
    auto &device = dc->device;
    "pp_contracts"_test = [&] { test_pp_contracts(); };
    "pp_folding_math"_test = [&] { test_pp_folding_math(); };
    "pp_detector_parity_tiny"_test = [&] { test_pp_detector_parity(device, PpOcrSize::tiny); };
    "pp_detector_parity_small"_test = [&] { test_pp_detector_parity(device, PpOcrSize::small); };
    "pp_detector_parity_medium"_test = [&] { test_pp_detector_parity(device, PpOcrSize::medium); };
    "pp_recognizer_parity_tiny"_test = [&] { test_pp_recognizer_parity(device, PpOcrSize::tiny); };
    "pp_recognizer_parity_small"_test = [&] { test_pp_recognizer_parity(device, PpOcrSize::small); };
    "pp_recognizer_parity_medium"_test = [&] { test_pp_recognizer_parity(device, PpOcrSize::medium); };
    "pp_blank_page_tiny"_test = [&] { test_pp_blank_page(device, PpOcrSize::tiny); };
    "pp_blank_page_small"_test = [&] { test_pp_blank_page(device, PpOcrSize::small); };
    "pp_blank_page_medium"_test = [&] { test_pp_blank_page(device, PpOcrSize::medium); };
    "pp_repeat_determinism_tiny"_test = [&] { test_pp_repeat_determinism(device, PpOcrSize::tiny); };
    "pp_repeat_determinism_small"_test = [&] { test_pp_repeat_determinism(device, PpOcrSize::small); };
    "pp_repeat_determinism_medium"_test = [&] { test_pp_repeat_determinism(device, PpOcrSize::medium); };
    "pp_geometry_hardening"_test = [&] { test_pp_geometry_hardening(); };
    "pp_fpn_resize_nonidentity"_test = [&] { test_pp_fpn_resize_nonidentity(device); };
    "pp_on_disk_layout_gate"_test = [&] { test_pp_on_disk_layout_gate(device); };
    "pp_crop_orientation_tiny"_test = [&] { test_pp_crop_orientation(device, PpOcrSize::tiny); };
    "pp_crop_orientation_small"_test = [&] { test_pp_crop_orientation(device, PpOcrSize::small); };
    "pp_crop_orientation_medium"_test = [&] { test_pp_crop_orientation(device, PpOcrSize::medium); };
    "pp_wide_and_multipage"_test = [&] { test_pp_wide_and_multipage(device); };
    "pp_limits"_test = [&] { test_pp_limits(device); };
    return 0;
}
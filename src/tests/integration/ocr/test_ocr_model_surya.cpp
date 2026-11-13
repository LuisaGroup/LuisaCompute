// Test for ocr/models/surya + runtime/surya_executor (device) -- the SMALL
// SYNTHETIC Surya vertical slice: EfficientViT-Segformer-style line detector +
// line crops + vision merger + hybrid GDN decoder (explicit GDN + periodic
// full-attention schedule).
// Covers: the declared 24-layer / 18-GDN / 6-full schedule contract, config/
// tensor/signature contracts, real-artifact fail-closed gate, memory preflight,
// detector probability-map parity vs an independent CPU oracle, the executing
// page->detector->crops->vision->hybrid-decode pipeline, SOURCE-coordinate
// polygons, detector-score confidence (no generative pseudo-confidence), blank
// pages (no hallucinated lines), deterministic repeat-run, and plain/spotting/
// markdown composition (extract fails closed). Synthetic weights only; no
// model/download, no real-model parity claim.
#include "ut/ut.hpp"
#include "test_device.h"
#include "ocr_gdn_test_utils.h"
#include "ocr_decode_test_utils.h"// compact tokenizer (undersized-vocab gate test)

#include <cmath>
#include <vector>

#include <luisa/runtime/buffer.h>
#include <luisa/runtime/command_list.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/stream.h>

#include "ocr/models/surya.h"
#include "ocr/runtime/surya_executor.h"

using namespace luisa;
using namespace luisa::compute;
using namespace luisa::ocr;
using namespace boost::ut;
using namespace boost::ut::literals;

namespace {

constexpr float kAtol = 5e-3f;
constexpr float kRtol = 5e-3f;

// A controllable synthetic detector: identity-center backbone convs (act 0) so
// the probability map is a downsampled "darkness" signal, and a DB head that
// negates the channel mean (dark -> high probability). This makes detection
// deterministic for drawn dark bars on a white page. The conv2d + sigmoid math
// is still fully validated against the independent CPU oracle.
struct SuryaDetectorWeights {
    luisa::vector<float> stem_w, stem_b, s1_w, s1_b, s2_w, s2_b, db_w, db_b;
};

SuryaDetectorWeights make_detector_weights(const SuryaDetectorConfig &det) {
    SuryaDetectorWeights w;
    const auto center_conv = [](uint32_t oc, uint32_t ic, uint32_t k, bool identity) {
        luisa::vector<float> v(static_cast<size_t>(oc) * ic * k * k, 0.0f);
        const uint32_t c = k / 2u;
        for (uint32_t o = 0; o < oc; o++) {
            for (uint32_t i = 0; i < ic; i++) {
                float val = 0.0f;
                if (identity) { val = (o == i) ? 1.0f : 0.0f; }
                else { val = (i < 3u && o == i) ? 1.0f : (o >= 3u ? 1.0f / 3.0f : 0.0f); }
                v[((static_cast<size_t>(o) * ic + i) * k + c) * k + c] = val;
            }
        }
        return v;
    };
    // stem 3->4: channels 0..2 copy RGB, channel 3 = mean RGB.
    w.stem_w = center_conv(det.stem.out_c, det.stem.in_c, det.stem.k, false);
    w.stem_b.assign(det.stem.out_c, 0.0f);
    w.s1_w = center_conv(det.stage1.out_c, det.stage1.in_c, det.stage1.k, true);
    w.s1_b.assign(det.stage1.out_c, 0.0f);
    w.s2_w = center_conv(det.stage2.out_c, det.stage2.in_c, det.stage2.k, true);
    w.s2_b.assign(det.stage2.out_c, 0.0f);
    // DB head 4->1 1x1: negative mean (dark -> high logit).
    w.db_w.assign(static_cast<size_t>(det.db_head.out_c) * det.db_head.in_c, -1.0f / static_cast<float>(det.db_head.in_c));
    w.db_b.assign(det.db_head.out_c, 0.0f);
    return w;
}

Status write_detector_package(const luisa::filesystem::path &path, const SuryaDetectorConfig &det,
                              const SuryaDetectorTensorNames &names, const SuryaDetectorWeights &w) {
    auto writer = PackageWriter::create(path);
    if (!writer.ok()) { return writer.status; }
    auto &wr = *writer.value;
    const auto add = [&wr](const luisa::string &name, luisa::vector<uint64_t> shape,
                           luisa::span<const float> data) -> Status {
        PackageWriter::TensorSpec spec;
        spec.name = name;
        spec.encoding = TensorEncoding::fp32;
        spec.layout = TensorLayout::row_major;
        spec.shape = std::move(shape);
        spec.payload = ocr::test::p7_f32_bytes(data);
        return wr->add_tensor(spec);
    };
    const auto cs = [](const SuryaConvStage &s) { return luisa::vector<uint64_t>{s.out_c, s.in_c, s.k, s.k}; };
    if (auto s = add(names.stem_weight(), cs(det.stem), w.stem_w); !s.ok()) { return s; }
    if (auto s = add(names.stem_bias(), {det.stem.out_c}, w.stem_b); !s.ok()) { return s; }
    if (auto s = add(names.stage1_weight(), cs(det.stage1), w.s1_w); !s.ok()) { return s; }
    if (auto s = add(names.stage1_bias(), {det.stage1.out_c}, w.s1_b); !s.ok()) { return s; }
    if (auto s = add(names.stage2_weight(), cs(det.stage2), w.s2_w); !s.ok()) { return s; }
    if (auto s = add(names.stage2_bias(), {det.stage2.out_c}, w.s2_b); !s.ok()) { return s; }
    if (auto s = add(names.db_weight(), cs(det.db_head), w.db_w); !s.ok()) { return s; }
    if (auto s = add(names.db_bias(), {det.db_head.out_c}, w.db_b); !s.ok()) { return s; }
    return wr->finish(ocr::test::p7_manifest("detector", "surya_2_5"));
}

// Independent CPU detector oracle: ref_conv2d + ref_cnn_act + ref_sigmoid.
luisa::vector<float> detector_oracle(const SuryaDetectorConfig &det, const SuryaDetectorWeights &w,
                                     luisa::span<const float> pixels_chw, uint32_t &map_h, uint32_t &map_w) {
    uint32_t h = det.input_height, ww = det.input_width;
    luisa::vector<float> cur(pixels_chw.begin(), pixels_chw.end());
    const auto stage = [&](const SuryaConvStage &s, luisa::span<const float> weight, luisa::span<const float> bias) {
        luisa::vector<float> out;
        uint32_t oh = 0, ow = 0;
        LUISA_ASSERT(ref_conv2d(cur, weight, bias, 1, s.in_c, h, ww, s.out_c, s.k, s.k,
                                s.stride, s.stride, s.pad, s.pad, 1, 1, 1, out, oh, ow).ok(), "oracle conv");
        if (s.act != 0u) {
            luisa::vector<float> act;
            LUISA_ASSERT(ref_cnn_act(out, s.act, act).ok(), "oracle act");
            out = act;
        }
        cur = out;
        h = oh;
        ww = ow;
    };
    stage(det.stem, w.stem_w, w.stem_b);
    stage(det.stage1, w.s1_w, w.s1_b);
    stage(det.stage2, w.s2_w, w.s2_b);
    stage(det.db_head, w.db_w, w.db_b);
    luisa::vector<float> prob;
    LUISA_ASSERT(ref_sigmoid(cur, prob).ok(), "oracle sigmoid");
    map_h = h;
    map_w = ww;
    return prob;
}

// Draw a white page with dark horizontal bars (text lines).
luisa::vector<uint8_t> make_page_with_bars(uint32_t w, uint32_t h, luisa::span<const std::pair<uint32_t, uint32_t>> bars) {
    luisa::vector<uint8_t> px(static_cast<size_t>(w) * h * 3u, 255u);
    for (const auto &[y0, y1] : bars) {
        for (uint32_t y = y0; y < y1 && y < h; y++) {
            for (uint32_t x = 0; x < w; x++) {
                const size_t i = (static_cast<size_t>(y) * w + x) * 3u;
                px[i] = 0;
                px[i + 1] = 0;
                px[i + 2] = 0;
            }
        }
    }
    return px;
}

struct SuryaFixture {
    ocr::test::P7TempDir tmp;
    SuryaConfig cfg;
    SuryaDetectorWeights det_w;
    ocr::test::P7SyntheticWeights vision_w;
    ocr::test::HybridSyntheticWeights decoder_w;
    WeightStore store;
    // REAL synthetic Unigram tokenizer (24 pieces: 'a'..'w' + </s> EOS). Owned by
    // the fixture, which outlives exec + every recognize() call, satisfying the
    // borrowed-pointer lifetime contract.
    luisa::unique_ptr<Tokenizer, TokenizerDeleter> tokenizer;
    luisa::unique_ptr<SuryaExecutor> exec;
    [[nodiscard]] bool init(Device &device) {
        cfg = surya_synthetic_config();
        // Controllable detector: identity backbone (act 0) for deterministic bars.
        cfg.detector.stem.act = 0;
        cfg.detector.stage1.act = 0;
        cfg.detector.stage2.act = 0;
        if (!cfg.validate().ok()) { return false; }
        det_w = make_detector_weights(cfg.detector);
        vision_w = ocr::test::generate_p7_weights(cfg.vision);
        decoder_w = ocr::test::generate_hybrid_weights(cfg.decoder);
        // S2: a text recognizer needs a tokenizer. Load the real synthetic
        // Unigram tokenizer and hand the executor a BORROWED pointer.
        tokenizer = ocr::test::load_hybrid_tokenizer();
        if (tokenizer == nullptr) { LUISA_WARNING("surya tokenizer load failed"); return false; }
        cfg.tokenizer = tokenizer.get();
        const auto detpath = tmp.path / "surya_detector.lcocr";
        const auto vpath = tmp.path / "surya_vision.lcocr";
        const auto dpath = tmp.path / "surya_decoder.lcocr";
        if (auto s = write_detector_package(detpath, cfg.detector, cfg.detector_names, det_w); !s.ok()) { return false; }
        if (auto s = ocr::test::write_p7_packages(vpath, tmp.path / "surya_dense_unused.lcocr",
                                                  cfg.vision, cfg.vision_names, vision_w, "surya_2_5"); !s.ok()) { return false; }
        if (auto s = ocr::test::write_hybrid_decoder_package(dpath, cfg.decoder, cfg.decoder_names, decoder_w); !s.ok()) { return false; }
        {
            auto detr = PackageReader::open(detpath);
            auto vr = PackageReader::open(vpath);
            auto dr = PackageReader::open(dpath);
            if (!detr.ok() || !vr.ok() || !dr.ok()) { return false; }
            if (auto s = surya_validate_tensors((*detr.value)->tensors(), (*vr.value)->tensors(), (*dr.value)->tensors(), cfg); !s.ok()) {
                LUISA_WARNING("surya tensor gate: {}", s.message);
                return false;
            }
        }
        auto stream = device.create_stream();
        {
            auto detr = PackageReader::open(detpath);
            if (!detr.ok()) { return false; }
            if (auto s = store.upload_component_sync(device, stream, **detr.value, "detector"); !s.ok()) { return false; }
        }
        {
            auto vr = PackageReader::open(vpath);
            if (!vr.ok()) { return false; }
            if (auto s = store.upload_component_sync(device, stream, **vr.value, "vision"); !s.ok()) { return false; }
        }
        {
            auto dr = PackageReader::open(dpath);
            if (!dr.ok()) { return false; }
            if (auto s = store.upload_component_sync(device, stream, **dr.value, "decoder"); !s.ok()) { return false; }
        }
        exec = luisa::make_unique<SuryaExecutor>();
        if (auto s = exec->create(device, stream, cfg, store); !s.ok()) {
            LUISA_WARNING("surya create: {}", s.message);
            return false;
        }
        return true;
    }
};

// ---------------------------------------------------------------------------
// Host-only contract tests.
// ---------------------------------------------------------------------------
void test_declared_schedule() {
    const auto declared = surya_declared_decoder_config();
    expect(declared.validate().ok());
    expect(eq(declared.decoder.num_layers, 24u));
    expect(eq(declared.num_gdn_layers(), 18u));
    expect(eq(declared.num_full_layers(), 6u));
    expect(surya_validate_schedule(declared.layer_kinds).ok()) << "declared 24/18/6 schedule";
    auto bad = declared.layer_kinds;
    bad[7] = HybridLayerKind::gdn;
    expect(!surya_validate_schedule(bad).ok()) << "wrong full position refused";
    auto bad2 = declared.layer_kinds;
    bad2.pop_back();
    expect(!surya_validate_schedule(bad2).ok()) << "wrong layer count refused";
    const auto synth = surya_synthetic_config();
    expect(synth.validate().ok());
    expect(!surya_validate_schedule(synth.decoder.layer_kinds).ok()) << "scaled slice != full contract";
}

void test_signature_and_real_gate() {
    const auto cfg = surya_synthetic_config();
    auto real = surya_validate_real_artifact("rev", "fp");
    expect(!real.ok());
    expect(real.code == StatusCode::unvalidated);
    SuryaSignature empty;
    expect(!surya_validate_signature(cfg, empty).ok());
    SuryaSignature good;
    good.checkpoint_revision = "synthetic-rev";
    good.config_fingerprint = "synthetic-fp";
    for (const auto &e : surya_expected_tensors(cfg)) { good.tensor_names.push_back(e.name); }
    expect(surya_validate_signature(cfg, good).ok()) << "exact signature accepted";
    auto bad = good;
    bad.tensor_names.push_back("detector.moe.weight");
    expect(!surya_validate_signature(cfg, bad).ok()) << "unexpected tensor refused";
    auto missing = good;
    missing.tensor_names.pop_back();
    expect(!surya_validate_signature(cfg, missing).ok()) << "missing tensor refused";
}

void test_config_parse() {
    const luisa::string json = R"({
      "model_type": "surya_2_5",
      "decoder_config": {
        "vocab_size": 24, "hidden_size": 16, "num_hidden_layers": 6,
        "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 4,
        "intermediate_size": 32, "max_position_embeddings": 64,
        "layer_kinds": ["gdn","gdn","gdn","full_attention","gdn","full_attention"]
      }
    })";
    auto parsed = parse_surya_config(json);
    expect(parsed.ok()) << "valid config parses";
    const luisa::string moe = R"({"model_type":"surya","num_experts":8,"decoder_config":{"vocab_size":24,"hidden_size":16,"num_hidden_layers":6,"num_attention_heads":4,"num_key_value_heads":2,"intermediate_size":32,"max_position_embeddings":64,"layer_kinds":["gdn","gdn","gdn","full_attention","gdn","full_attention"]}})";
    auto m = parse_surya_config(moe);
    expect(!m.ok());
    expect(m.status.code == StatusCode::unsupported);
}

void test_memory_preflight() {
    const auto cfg = surya_synthetic_config();
    auto ok = surya_preflight_decoder(cfg, 64, MemoryLimits{64ull * 1024 * 1024, 0});
    expect(ok.ok()) << "sufficient budget fits";
    auto tight = surya_preflight_decoder(cfg, 64, MemoryLimits{1024, 0});
    expect(!tight.ok());
    expect(tight.status.code == StatusCode::out_of_memory);
}

// ---------------------------------------------------------------------------
// Device: detector parity + full pipeline + source coordinates.
// ---------------------------------------------------------------------------
void test_detector_parity(Device &device) {
    SuryaFixture f;
    LUISA_ASSERT(f.init(device), "surya fixture init");
    auto stream = device.create_stream();
    SuryaPageInput page;
    page.source_width = 32;
    page.source_height = 32;
    page.page_index = 0;
    const std::vector<std::pair<uint32_t, uint32_t>> bars = {{8u, 16u}, {20u, 28u}};
    page.pixels = make_page_with_bars(32, 32, bars);
    auto run = f.exec->recognize(stream, luisa::span<const SuryaPageInput>{&page, 1});
    expect(run.ok()) << "detector pipeline runs";
    if (!run.ok()) { LUISA_WARNING("recognize: {}", run.status.message); return; }
    // Independent CPU detector oracle.
    auto pixels_chw = ocr::test::p7_oracle_preprocess(page.pixels, 32, 32, 32, 32,
                                                      f.cfg.detector.mean, f.cfg.detector.std_dev);
    uint32_t mh = 0, mw = 0;
    auto prob_ref = detector_oracle(f.cfg.detector, f.det_w, pixels_chw, mh, mw);
    luisa::vector<float> prob_dev;
    LUISA_ASSERT(f.exec->download_detector_prob(stream, prob_dev).ok(), "dl prob");
    stream << synchronize();
    expect(eq(f.exec->last_map_height(), mh));
    expect(eq(f.exec->last_map_width(), mw));
    ocr::test::expect_close(prob_dev, prob_ref, kAtol, kRtol, "surya detector probability map");
}

void test_full_pipeline_source_coords(Device &device) {
    SuryaFixture f;
    LUISA_ASSERT(f.init(device), "surya fixture init");
    auto stream = device.create_stream();
    SuryaPageInput page;
    page.source_width = 32;
    page.source_height = 32;
    page.page_index = 0;
    const std::vector<std::pair<uint32_t, uint32_t>> bars = {{8u, 16u}, {20u, 28u}};
    page.pixels = make_page_with_bars(32, 32, bars);
    auto run = f.exec->recognize(stream, luisa::span<const SuryaPageInput>{&page, 1});
    expect(run.ok());
    if (!run.ok()) { return; }
    expect(eq(run.value->pages.size(), 1u));
    const auto &pr = run.value->pages[0];
    expect(!pr.boxes.empty()) << "dark bars detected as boxes";
    expect(!pr.lines.empty()) << "lines recognized";
    // Source-coordinate checks: polygons are in ORIGINAL source coordinates
    // (source scale ~32, NOT the 8x8 detector-map scale). DB unclip legitimately
    // expands a box outward, so a full-width line can extend slightly past the
    // page edge; allow an unclip margin rather than a strict page bound.
    bool coords_ok = true;
    for (const auto &line : pr.lines) {
        for (const auto &p : line.polygon) {
            if (!std::isfinite(p.x) || !std::isfinite(p.y) ||
                p.x < -16.0f || p.x > 48.0f || p.y < -16.0f || p.y > 48.0f) { coords_ok = false; }
        }
        // The box must be on the SOURCE scale (a full-width unclipped line is
        // ~33 wide; a map-scale box would be <= 8). This proves the detector-map
        // -> source transform was applied.
        if (line.bounds.x1 - line.bounds.x0 < 8.0f) { coords_ok = false; }
        // Confidence is the DETECTOR score in [0,1], not a generative probability.
        if (!line.has_confidence || line.confidence < 0.0f || line.confidence > 1.0f) { coords_ok = false; }
    }
    expect(coords_ok) << "source-coordinate polygons + detector-score confidence";
    // Reading order is deterministic (top-to-bottom).
    bool ordered = true;
    for (size_t i = 1; i < pr.lines.size(); i++) {
        if (pr.lines[i].bounds.y0 < pr.lines[i - 1].bounds.y0) { ordered = false; }
    }
    expect(ordered) << "lines in top-to-bottom reading order";
    // Compose to markdown / spotting / plain.
    auto md = surya_compose(*run.value, LayoutOutputMode::markdown);
    expect(md.ok()) << "markdown compose";
    auto spot = surya_compose(*run.value, LayoutOutputMode::spotting);
    expect(spot.ok()) << "spotting compose";
    auto plain = surya_compose(*run.value, LayoutOutputMode::plain);
    expect(plain.ok()) << "plain compose";
    auto ext = surya_compose(*run.value, LayoutOutputMode::extract);
    expect(!ext.ok()) << "extract fails closed";
}

void test_blank_page(Device &device) {
    SuryaFixture f;
    LUISA_ASSERT(f.init(device), "surya fixture init");
    auto stream = device.create_stream();
    SuryaPageInput page;
    page.source_width = 32;
    page.source_height = 32;
    page.page_index = 0;
    page.pixels.assign(static_cast<size_t>(32) * 32 * 3u, 255u);// all white
    auto run = f.exec->recognize(stream, luisa::span<const SuryaPageInput>{&page, 1});
    expect(run.ok());
    if (run.ok()) {
        expect(run.value->pages[0].lines.empty()) << "blank page: no hallucinated lines";
        expect(run.value->pages[0].text.empty()) << "blank page: empty text";
    }
}

void test_repeat_determinism(Device &device) {
    SuryaFixture f;
    LUISA_ASSERT(f.init(device), "surya fixture init");
    auto stream = device.create_stream();
    SuryaPageInput page;
    page.source_width = 32;
    page.source_height = 32;
    page.page_index = 0;
    const std::vector<std::pair<uint32_t, uint32_t>> bars = {{8u, 16u}, {20u, 28u}};
    page.pixels = make_page_with_bars(32, 32, bars);
    auto r1 = f.exec->recognize(stream, luisa::span<const SuryaPageInput>{&page, 1});
    auto r2 = f.exec->recognize(stream, luisa::span<const SuryaPageInput>{&page, 1});
    expect(r1.ok() && r2.ok());
    if (r1.ok() && r2.ok()) {
        const auto &p1 = r1.value->pages[0];
        const auto &p2 = r2.value->pages[0];
        expect(eq(p1.lines.size(), p2.lines.size()));
        bool same = p1.lines.size() == p2.lines.size();
        for (size_t i = 0; same && i < p1.lines.size(); i++) {
            if (p1.lines[i].tokens != p2.lines[i].tokens) { same = false; }
        }
        expect(same) << "repeat-run exact line tokens (cache reset per line, no leak)";
    }
}

// ---------------------------------------------------------------------------
// S2 REGRESSION (Phase 9 review): Surya must actually DETOKENIZE recognized
// lines. Before the fix no tokenizer ever reached HybridVlmGraph::run, so every
// line.text / page.text (and therefore all composed content) was silently empty
// even though tokens were generated. This test uses the REAL synthetic Unigram
// tokenizer (24 pieces 'a'..'w' + </s>) and asserts non-empty line text, page
// text, and composed content -- plus the null-tokenizer fail-closed gates.
// ---------------------------------------------------------------------------
void test_line_text_detokenized(Device &device) {
    SuryaFixture f;
    LUISA_ASSERT(f.init(device), "surya fixture init (with tokenizer)");
    auto stream = device.create_stream();
    SuryaPageInput page;
    page.source_width = 32;
    page.source_height = 32;
    page.page_index = 0;
    const std::vector<std::pair<uint32_t, uint32_t>> bars = {{8u, 16u}, {20u, 28u}};
    page.pixels = make_page_with_bars(32, 32, bars);
    auto run = f.exec->recognize(stream, luisa::span<const SuryaPageInput>{&page, 1});
    expect(run.ok()) << "surya text pipeline runs";
    if (!run.ok()) { LUISA_WARNING("surya text: {}", run.status.message); return; }
    const auto &pr = run.value->pages[0];
    expect(!pr.lines.empty()) << "lines detected";
    // EVERY line must carry generated tokens AND non-empty detokenized text.
    bool all_text = !pr.lines.empty();
    bool all_tokens = !pr.lines.empty();
    for (const auto &line : pr.lines) {
        if (line.tokens.empty()) { all_tokens = false; }
        if (line.text.empty()) { all_text = false; }
        // The synthetic Unigram pieces are lowercase letters: the text must be
        // exactly the concatenation of the emitted pieces (id -> 'a'+id).
        luisa::string expect_text;
        for (auto t : line.tokens) {
            if (t < 23u) { expect_text += static_cast<char>('a' + t); }
        }
        if (line.text != expect_text) { all_text = false; }
    }
    expect(all_tokens) << "every line has generated tokens";
    expect(all_text) << "every line has NON-EMPTY text matching its token pieces";
    // Page text is the canonical newline join of the (non-empty) line texts.
    expect(!pr.text.empty()) << "page text NON-EMPTY";
    luisa::string joined;
    for (size_t i = 0; i < pr.lines.size(); i++) {
        if (i != 0u) { joined += "\n"; }
        joined += pr.lines[i].text;
    }
    expect(pr.text == joined) << "page text == canonical join of line texts";
    // Composed content must be non-empty in every supported mode.
    for (auto mode : {LayoutOutputMode::plain, LayoutOutputMode::spotting, LayoutOutputMode::markdown}) {
        auto doc = surya_compose(*run.value, mode);
        expect(doc.ok()) << "compose ok";
        if (doc.ok()) {
            expect(!doc.value->text.empty()) << "composed document text NON-EMPTY";
            bool block_content = !doc.value->blocks.empty();
            for (const auto &b : doc.value->blocks) {
                if (b.content.empty()) { block_content = false; }
            }
            expect(block_content) << "composed blocks carry NON-EMPTY content";
        }
    }
}

// S2: the tokenizer is REQUIRED and its vocabulary must cover the decoder; a
// text-capable run must fail closed rather than silently emit empty content.
void test_tokenizer_required(Device &device) {
    SuryaFixture f;
    LUISA_ASSERT(f.init(device), "surya fixture init");
    auto stream = device.create_stream();
    // Null tokenizer: create() refuses.
    {
        auto cfg = f.cfg;
        cfg.tokenizer = nullptr;
        SuryaExecutor ex;
        auto s = ex.create(device, stream, cfg, f.store);
        expect(!s.ok()) << "null tokenizer refused at create";
        expect(s.code == StatusCode::invalid_argument);
    }
    // Undersized vocabulary: create() refuses.
    {
        auto small = ocr::test::load_compact_tokenizer();// vocab 10 < 24
        LUISA_ASSERT(small != nullptr, "compact tokenizer");
        auto cfg = f.cfg;
        cfg.tokenizer = small.get();
        SuryaExecutor ex;
        auto s = ex.create(device, stream, cfg, f.store);
        expect(!s.ok()) << "undersized tokenizer vocabulary refused";
        expect(s.code == StatusCode::invalid_argument);
    }
    // config.validate() stays a pure contract check (no tokenizer needed), so
    // parse / signature / preflight keep working without one.
    {
        auto cfg = f.cfg;
        cfg.tokenizer = nullptr;
        expect(cfg.validate().ok()) << "contract validate does not require a tokenizer";
    }
}

}// namespace

int main(int argc, char *argv[]) {
    auto dc = luisa::test::create_device_from_ut(argc, argv);
    if (!dc) { return 0; }
    boost::ut::detail::cfg::parse_arg_with_fallback(argc, const_cast<const char **>(argv));
    auto &device = dc->device;
    "surya_declared_schedule"_test = [&] { test_declared_schedule(); };
    "surya_signature_and_real_gate"_test = [&] { test_signature_and_real_gate(); };
    "surya_config_parse"_test = [&] { test_config_parse(); };
    "surya_memory_preflight"_test = [&] { test_memory_preflight(); };
    "surya_detector_parity"_test = [&] { test_detector_parity(device); };
    "surya_full_pipeline_source_coords"_test = [&] { test_full_pipeline_source_coords(device); };
    "surya_blank_page"_test = [&] { test_blank_page(device); };
    "surya_repeat_determinism"_test = [&] { test_repeat_determinism(device); };
    "surya_line_text_detokenized"_test = [&] { test_line_text_detokenized(device); };
    "surya_tokenizer_required"_test = [&] { test_tokenizer_required(device); };
}

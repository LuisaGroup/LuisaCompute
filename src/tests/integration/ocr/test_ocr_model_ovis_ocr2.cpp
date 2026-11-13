// Test for ocr/models/ovis_ocr2 + the hybrid GDN VLM graph (device) -- the SMALL
// SYNTHETIC OvisOCR2 vertical slice: ViT + spatial-merge patch merger + hybrid
// GDN decoder (explicit GDN + periodic full-attention schedule).
// Covers: the declared 24-layer / 18-GDN / 6-full schedule contract, config/
// tensor/signature contracts, real-artifact fail-closed gate, memory preflight,
// the executing image->vision->hybrid-decode->Markdown graph, vision-merger
// parity vs the independent P7 oracle, deterministic tokens, repeat-run, and
// plain/markdown composition (spotting/extract fail closed). Synthetic weights
// only; no model/download, no real-model parity claim.
#include "ut/ut.hpp"
#include "test_device.h"
#include "ocr_gdn_test_utils.h"

#include <cmath>

#include <luisa/runtime/buffer.h>
#include <luisa/runtime/command_list.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/stream.h>

#include "ocr/models/ovis_ocr2.h"
#include "ocr/runtime/hybrid_gdn_executor.h"

using namespace luisa;
using namespace luisa::compute;
using namespace luisa::ocr;
using namespace boost::ut;
using namespace boost::ut::literals;

namespace {

constexpr float kAtol = 5e-3f;
constexpr float kRtol = 5e-3f;

// Build a deterministic synthetic RGB page (HWC uint8).
luisa::vector<uint8_t> make_page(uint32_t w, uint32_t h, uint32_t seed = 7u) {
    luisa::vector<uint8_t> px(static_cast<size_t>(w) * h * 3u);
    ocr::test::Lcg lcg;
    lcg.state = seed;
    for (auto &v : px) { v = static_cast<uint8_t>(lcg.next() % 256u); }
    return px;
}

struct OvisFixture {
    ocr::test::P7TempDir tmp;
    OvisOcr2Config cfg;
    ocr::test::P7SyntheticWeights vision_w;
    ocr::test::HybridSyntheticWeights decoder_w;
    WeightStore store;
    luisa::unique_ptr<HybridVlmGraph> graph;
    [[nodiscard]] bool init(Device &device, bool self_loop = false) {
        cfg = ovis_ocr2_synthetic_config();
        if (!cfg.validate().ok()) { return false; }
        vision_w = ocr::test::generate_p7_weights(cfg.vision);
        decoder_w = self_loop ? ocr::test::make_self_loop_weights(cfg.decoder)
                              : ocr::test::generate_hybrid_weights(cfg.decoder);
        const auto vpath = tmp.path / "ovis_vision.lcocr";
        const auto dpath = tmp.path / "ovis_decoder.lcocr";
        // Vision package: reuse the P7 writer's vision component by writing a
        // full P7 package then a hybrid decoder package. The P7 writer writes
        // BOTH vision and a dense decoder; we only upload the vision component
        // from it and the hybrid decoder from our own writer.
        if (auto s = ocr::test::write_p7_packages(vpath, tmp.path / "ovis_dense_unused.lcocr",
                                                  cfg.vision, cfg.vision_names, vision_w, "ovis_ocr2"); !s.ok()) {
            LUISA_WARNING("write vision: {}", s.message);
            return false;
        }
        if (auto s = ocr::test::write_hybrid_decoder_package(dpath, cfg.decoder, cfg.decoder_names, decoder_w); !s.ok()) {
            LUISA_WARNING("write decoder: {}", s.message);
            return false;
        }
        // Package-load gate: validate both component tables on disk.
        {
            auto vr = PackageReader::open(vpath);
            auto dr = PackageReader::open(dpath);
            if (!vr.ok() || !dr.ok()) { return false; }
            if (auto s = ovis_ocr2_validate_tensors((*vr.value)->tensors(), (*dr.value)->tensors(), cfg); !s.ok()) {
                LUISA_WARNING("ovis tensor gate: {}", s.message);
                return false;
            }
        }
        auto stream = device.create_stream();
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
        HybridVlmGraphConfig gc;
        gc.vision = cfg.vision;
        gc.vision_names = cfg.vision_names;
        gc.decoder = cfg.decoder;
        gc.decoder_names = cfg.decoder_names;
        gc.max_seq_len = cfg.max_seq_len;
        graph = luisa::make_unique<HybridVlmGraph>();
        if (auto s = graph->create(device, stream, gc, store); !s.ok()) {
            LUISA_WARNING("ovis graph create: {}", s.message);
            return false;
        }
        return true;
    }
};

// ---------------------------------------------------------------------------
// Host-only contract tests (no device work).
// ---------------------------------------------------------------------------
void test_declared_schedule() {
    const auto declared = ovis_ocr2_declared_decoder_config();
    expect(declared.validate().ok()) << "declared config valid";
    expect(eq(declared.decoder.num_layers, 24u));
    expect(eq(declared.num_gdn_layers(), 18u));
    expect(eq(declared.num_full_layers(), 6u));
    expect(ovis_ocr2_validate_schedule(declared.layer_kinds).ok()) << "declared 24/18/6 schedule";
    // A wrong schedule fails closed.
    auto bad = declared.layer_kinds;
    bad[3] = HybridLayerKind::gdn;// remove a full layer
    expect(!ovis_ocr2_validate_schedule(bad).ok()) << "wrong full position refused";
    auto bad2 = declared.layer_kinds;
    bad2.pop_back();
    expect(!ovis_ocr2_validate_schedule(bad2).ok()) << "wrong layer count refused";
    // The synthetic slice is a SCALED schedule (not 24 layers) and must NOT
    // pass the full 24/18/6 gate (it is validated by its own config).
    const auto synth = ovis_ocr2_synthetic_config();
    expect(synth.validate().ok());
    expect(!ovis_ocr2_validate_schedule(synth.decoder.layer_kinds).ok()) << "scaled slice != full contract";
    expect(eq(synth.decoder.num_gdn_layers(), 4u));
    expect(eq(synth.decoder.num_full_layers(), 2u));
}

void test_signature_and_real_gate() {
    const auto cfg = ovis_ocr2_synthetic_config();
    // Real artifacts ALWAYS fail closed.
    auto real = ovis_ocr2_validate_real_artifact("some-rev", "some-fp");
    expect(!real.ok());
    expect(real.code == StatusCode::unvalidated);
    // Signature: empty identity refused.
    OvisOcr2Signature empty;
    expect(!ovis_ocr2_validate_signature(cfg, empty).ok());
    // Signature: exact tensor inventory accepted.
    OvisOcr2Signature good;
    good.checkpoint_revision = "synthetic-rev";
    good.config_fingerprint = "synthetic-fp";
    for (const auto &e : ovis_ocr2_expected_tensors(cfg)) { good.tensor_names.push_back(e.name); }
    expect(ovis_ocr2_validate_signature(cfg, good).ok()) << "exact signature accepted";
    // Signature: an unexpected tensor refused.
    auto bad = good;
    bad.tensor_names.push_back("model.layers.0.moe.experts.0.weight");
    expect(!ovis_ocr2_validate_signature(cfg, bad).ok()) << "unexpected tensor refused";
    // Signature: a missing required tensor refused.
    auto missing = good;
    missing.tensor_names.pop_back();
    expect(!ovis_ocr2_validate_signature(cfg, missing).ok()) << "missing tensor refused";
}

void test_config_parse() {
    // A minimal valid synthetic config parses.
    const luisa::string json = R"({
      "model_type": "ovis_ocr2",
      "decoder_config": {
        "vocab_size": 24, "hidden_size": 16, "num_hidden_layers": 6,
        "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 4,
        "intermediate_size": 32, "max_position_embeddings": 64,
        "layer_kinds": ["gdn","gdn","gdn","full_attention","gdn","full_attention"]
      }
    })";
    auto parsed = parse_ovis_ocr2_config(json);
    expect(parsed.ok()) << "valid config parses";
    if (parsed.ok()) {
        expect(eq(parsed.value->decoder.decoder.num_layers, 6u));
        expect(eq(parsed.value->decoder.num_gdn_layers(), 4u));
    }
    // MoE marker fails closed.
    const luisa::string moe = R"({"model_type":"ovis_ocr2","num_experts":8,"decoder_config":{"vocab_size":24,"hidden_size":16,"num_hidden_layers":6,"num_attention_heads":4,"num_key_value_heads":2,"intermediate_size":32,"max_position_embeddings":64,"layer_kinds":["gdn","gdn","gdn","full_attention","gdn","full_attention"]}})";
    auto m = parse_ovis_ocr2_config(moe);
    expect(!m.ok());
    expect(m.status.code == StatusCode::unsupported);
    // Missing layer_kinds fails closed (schedule never inferred).
    const luisa::string no_sched = R"({"decoder_config":{"vocab_size":24,"hidden_size":16,"num_hidden_layers":6,"num_attention_heads":4,"num_key_value_heads":2,"intermediate_size":32,"max_position_embeddings":64}})";
    expect(!parse_ovis_ocr2_config(no_sched).ok());
}

void test_memory_preflight() {
    const auto cfg = ovis_ocr2_synthetic_config();
    // Sufficient budget fits.
    auto ok = ovis_ocr2_preflight_decoder(cfg, 64, MemoryLimits{64ull * 1024 * 1024, 0});
    expect(ok.ok()) << "sufficient budget fits";
    if (ok.ok()) {
        expect(ok.value->fits);
        expect(ok.value->cache_bytes > 0u);
        expect(ok.value->total_device_bytes > 0u);
    }
    // A tiny budget refuses BEFORE allocation with a useful report.
    auto tight = ovis_ocr2_preflight_decoder(cfg, 64, MemoryLimits{1024, 0});
    expect(!tight.ok());
    expect(tight.status.code == StatusCode::out_of_memory);
}

// ---------------------------------------------------------------------------
// Device: full synthetic image -> vision -> hybrid decode -> Markdown graph.
// ---------------------------------------------------------------------------
void test_full_pipeline(Device &device) {
    OvisFixture f;
    LUISA_ASSERT(f.init(device), "ovis fixture init");
    auto stream = device.create_stream();
    // Pack a small page through the dynamic packing contract.
    const uint32_t sw = 16, sh = 16;
    P7ImageInput img;
    img.pixels = make_page(sw, sh);
    img.source_width = sw;
    img.source_height = sh;
    auto packed = p7_pack_image(sw, sh, f.cfg.vision.packing, f.cfg.vision.connector);
    LUISA_ASSERT(packed.ok(), "pack image");
    img.packed = *packed.value;
    const uint32_t vt = static_cast<uint32_t>(img.packed.visual_tokens);
    expect(vt > 0u);
    // Prompt: [prefix tokens][visual rows][suffix tokens].
    luisa::vector<uint32_t> prompt = {1u, 2u};
    const uint32_t mm_start = static_cast<uint32_t>(prompt.size());
    for (uint32_t i = 0; i < vt; i++) { prompt.push_back(0u); }// visual placeholder rows
    prompt.push_back(3u);
    prompt.push_back(4u);
    HybridDecodeOptions opts;
    opts.sampling.greedy = true;
    opts.max_new_tokens = 6;
    opts.eos_ids = {23u};
    auto run = f.graph->run(stream, luisa::span<const P7ImageInput>{&img, 1}, prompt, mm_start, opts);
    expect(run.ok()) << "full pipeline runs";
    if (!run.ok()) { LUISA_WARNING("pipeline: {}", run.status.message); return; }
    expect(eq(run.value->visual_tokens, vt));
    expect(!run.value->decode.tokens.empty()) << "generated tokens";
    expect(run.value->cache_bytes > 0u);
    // Vision-merger parity vs the independent P7 oracle.
    auto pixels_chw = ocr::test::p7_oracle_preprocess(img.pixels, sw, sh, img.packed.resized_width,
                                                      img.packed.resized_height, f.cfg.vision.mean, f.cfg.vision.std_dev);
    ocr::test::P7VisionOracle vor;
    LUISA_ASSERT(ocr::test::p7_oracle_vision(f.cfg.vision, f.vision_w, pixels_chw, img.packed, vor).ok(), "vision oracle");
    luisa::vector<float> feat_dev;
    LUISA_ASSERT(f.graph->download_vision_features(stream, feat_dev).ok(), "dl features");
    stream << synchronize();
    ocr::test::expect_close(feat_dev, vor.features, kAtol, kRtol, "ovis vision merger features");
}

void test_determinism_and_repeat(Device &device) {
    OvisFixture f;
    LUISA_ASSERT(f.init(device), "ovis fixture init");
    auto stream = device.create_stream();
    const uint32_t sw = 16, sh = 16;
    P7ImageInput img;
    img.pixels = make_page(sw, sh, 11u);
    img.source_width = sw;
    img.source_height = sh;
    auto packed = p7_pack_image(sw, sh, f.cfg.vision.packing, f.cfg.vision.connector);
    LUISA_ASSERT(packed.ok(), "pack");
    img.packed = *packed.value;
    const uint32_t vt = static_cast<uint32_t>(img.packed.visual_tokens);
    luisa::vector<uint32_t> prompt = {5u};
    const uint32_t mm_start = 1u;
    for (uint32_t i = 0; i < vt; i++) { prompt.push_back(0u); }
    prompt.push_back(6u);
    HybridDecodeOptions opts;
    opts.sampling.greedy = true;
    opts.max_new_tokens = 8;
    opts.eos_ids = {23u};
    auto r1 = f.graph->run(stream, luisa::span<const P7ImageInput>{&img, 1}, prompt, mm_start, opts);
    auto r2 = f.graph->run(stream, luisa::span<const P7ImageInput>{&img, 1}, prompt, mm_start, opts);
    expect(r1.ok() && r2.ok());
    if (r1.ok() && r2.ok()) {
        expect(r1.value->decode.tokens == r2.value->decode.tokens) << "repeat-run exact tokens (cache reset, no leak)";
    }
}

void test_compose_modes() {
    // Build a tiny LayoutRunResult with one text region.
    LayoutRunResult run;
    LayoutPageResult page;
    page.page_index = 0;
    page.source_width = 16;
    page.source_height = 16;
    LayoutRegionResult region;
    region.region_id = 0;
    region.page_index = 0;
    region.reading_order = 0;
    region.block_type = LayoutBlockType::text;
    region.content = "hello world";
    page.regions.push_back(region);
    run.pages.push_back(page);
    auto plain = ovis_ocr2_compose(run, LayoutOutputMode::plain);
    expect(plain.ok()) << "plain compose";
    auto md = ovis_ocr2_compose(run, LayoutOutputMode::markdown);
    expect(md.ok()) << "markdown compose";
    // spotting / extract fail closed.
    auto spot = ovis_ocr2_compose(run, LayoutOutputMode::spotting);
    expect(!spot.ok());
    expect(spot.status.code == StatusCode::unsupported);
    auto ext = ovis_ocr2_compose(run, LayoutOutputMode::extract);
    expect(!ext.ok());
}

// Ragged non-square image through the full graph + end-to-end Markdown from a
// REAL pipeline run (image -> vision -> hybrid decode -> page Markdown). The
// dynamic packing must handle a wide (non-square) source, and the composed
// Markdown must carry the generated text.
void test_ragged_image_end_to_end_markdown(Device &device) {
    OvisFixture f;
    // Self-loop weights: greedy decode repeats the last prompt token (3 -> 'd'),
    // giving deterministic NON-EMPTY text for the end-to-end Markdown path.
    LUISA_ASSERT(f.init(device, true), "ovis self-loop fixture init");
    auto tokenizer = ocr::test::load_hybrid_tokenizer();
    LUISA_ASSERT(tokenizer != nullptr, "hybrid tokenizer");
    auto stream = device.create_stream();
    // Wide, non-square source (ragged aspect ratio).
    const uint32_t sw = 32, sh = 8;
    P7ImageInput img;
    img.pixels = make_page(sw, sh, 21u);
    img.source_width = sw;
    img.source_height = sh;
    auto packed = p7_pack_image(sw, sh, f.cfg.vision.packing, f.cfg.vision.connector);
    expect(packed.ok()) << "ragged non-square packing";
    if (!packed.ok()) { return; }
    img.packed = *packed.value;
    expect(img.packed.grid_w > img.packed.grid_h) << "wide grid for a wide source";
    const uint32_t vt = static_cast<uint32_t>(img.packed.visual_tokens);
    luisa::vector<uint32_t> prompt = {1u, 2u};
    const uint32_t mm_start = static_cast<uint32_t>(prompt.size());
    for (uint32_t i = 0; i < vt; i++) { prompt.push_back(0u); }
    prompt.push_back(3u);
    HybridDecodeOptions opts;
    opts.sampling.greedy = true;
    opts.max_new_tokens = 6;
    opts.eos_ids = {23u};
    auto run = f.graph->run(stream, luisa::span<const P7ImageInput>{&img, 1}, prompt, mm_start, opts, tokenizer.get());
    expect(run.ok()) << "ragged pipeline runs";
    if (!run.ok()) { LUISA_WARNING("ragged: {}", run.status.message); return; }
    expect(eq(run.value->visual_tokens, vt));
    // Self-loop on the last prompt token (3 -> 'd'): deterministic non-empty text.
    expect(!run.value->decode.text.empty()) << "detokenized text present";
    // End-to-end: compose the REAL generated text into a page Markdown document.
    LayoutRunResult lr;
    LayoutPageResult page;
    page.page_index = 0;
    page.source_width = sw;
    page.source_height = sh;
    LayoutRegionResult region;
    region.region_id = 0;
    region.page_index = 0;
    region.reading_order = 0;
    region.block_type = LayoutBlockType::text;
    region.content = run.value->decode.text;
    page.regions.push_back(region);
    lr.pages.push_back(page);
    auto md = ovis_ocr2_compose(lr, LayoutOutputMode::markdown);
    expect(md.ok()) << "end-to-end Markdown compose";
    if (md.ok()) {
        // The composed document carries the generated text (non-empty pipeline).
        expect(!md.value->text.empty() || run.value->decode.tokens.empty())
            << "Markdown reflects generated content";
    }
}

}// namespace

int main(int argc, char *argv[]) {
    auto dc = luisa::test::create_device_from_ut(argc, argv);
    if (!dc) { return 0; }
    boost::ut::detail::cfg::parse_arg_with_fallback(argc, const_cast<const char **>(argv));
    auto &device = dc->device;
    "ovis_declared_schedule"_test = [&] { test_declared_schedule(); };
    "ovis_signature_and_real_gate"_test = [&] { test_signature_and_real_gate(); };
    "ovis_config_parse"_test = [&] { test_config_parse(); };
    "ovis_memory_preflight"_test = [&] { test_memory_preflight(); };
    "ovis_full_pipeline"_test = [&] { test_full_pipeline(device); };
    "ovis_determinism_and_repeat"_test = [&] { test_determinism_and_repeat(device); };
    "ovis_compose_modes"_test = [&] { test_compose_modes(); };
    "ovis_ragged_image_end_to_end_markdown"_test = [&] { test_ragged_image_end_to_end_markdown(device); };
}

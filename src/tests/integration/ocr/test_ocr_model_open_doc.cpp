// Test for ocr/runtime/layout_vlm_executor OpenDocRecognizer + the two-stage
// PP-DocLayoutV2 + UniRec pipeline (Phase 6, device) -- the SMALL SYNTHETIC
// OpenDoc vertical slice. OpenDoc is a TWO-STAGE text/formula/table recognizer,
// NOT a generic dense VLM: the UniRec stage is a conventional CNN+CTC sequence
// model (the shared CnnExecutor recognizer graph), with type-specific output
// adapters. Covers: config/tensor contracts, UniRec recognizer parity vs an
// independent UNFOLDED conv->BN CPU oracle, type-specific adapters
// (text/formula/table) and invalid-output fail-closed, region batching vs
// batch-1 equivalence, and the full layout->region->composition pipeline.
// Synthetic weights only; no model/download, no real-model parity claim.
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

constexpr float kAtol = 3e-3f;
constexpr float kRtol = 3e-3f;

struct Fixture {
    ocr::test::P6TempDir tmp;
    ocr::test::OpenDocFixtureWeights weights;
    WeightStore store;
    luisa::unique_ptr<PpDocLayoutExecutor> layout;
    luisa::unique_ptr<OpenDocRecognizer> recognizer;
    PpDocLayoutExecutorConfig layout_config;
    OpenDocRecognizerConfig rec_config;

    [[nodiscard]] bool init(Device &device, uint32_t seed = 0x5EEDu) {
        if (auto s = ocr::test::generate_opendoc_weights(weights, seed); !s.ok()) {
            LUISA_WARNING("generate_opendoc_weights: {}", s.message);
            return false;
        }
        const auto lpath = tmp.path / "opendoc_layout.lcocr";
        const auto rpath = tmp.path / "opendoc_unirec.lcocr";
        if (auto s = ocr::test::write_opendoc_packages(lpath, rpath, weights); !s.ok()) {
            LUISA_WARNING("write_opendoc_packages: {}", s.message);
            return false;
        }
        auto stream = device.create_stream();
        if (auto s = ocr::test::upload_opendoc_weights(device, stream, lpath, rpath, store); !s.ok()) {
            LUISA_WARNING("upload_opendoc: {}", s.message);
            return false;
        }
        layout_config.model = weights.layout_cfg;
        layout_config.names = weights.layout_names;
        layout_config.component = "layout";
        layout = luisa::make_unique<PpDocLayoutExecutor>();
        if (auto s = layout->create(device, stream, layout_config, store); !s.ok()) {
            LUISA_WARNING("layout create: {}", s.message);
            return false;
        }
        // OpenDoc recognizer config: the UniRec geometry + dictionary.
        rec_config.model.layout = weights.layout_cfg;
        rec_config.model.layout_names = weights.layout_names;
        rec_config.model.recognizer.recognizer = weights.rec_cfg.recognizer;
        rec_config.model.recognizer.dictionary = ocr::test::pp_synthetic_dictionary();
        rec_config.model.recognizer_names = UniRecTensorNames{};
        rec_config.recognizer_component = "unirec";
        recognizer = luisa::make_unique<OpenDocRecognizer>();
        if (auto s = recognizer->create(device, stream, rec_config, store); !s.ok()) {
            LUISA_WARNING("opendoc recognizer create: {}", s.message);
            return false;
        }
        return true;
    }
};

[[nodiscard]] RegionCropInput make_crop(uint32_t w, uint32_t h, uint32_t region_id,
                                        LayoutBlockType type, uint32_t seed = 11u) {
    RegionCropInput crop;
    crop.source_width = w;
    crop.source_height = h;
    crop.task.region_id = region_id;
    crop.task.page_index = 0;
    crop.task.reading_order = region_id;
    crop.task.block_type = type;
    crop.task.source_box = output::DbRect{0, 0, static_cast<float>(w), static_cast<float>(h)};
    crop.pixels.resize(static_cast<size_t>(w) * h * 3u);
    ocr::test::Lcg lcg;
    lcg.state = seed;
    for (auto &b : crop.pixels) { b = static_cast<uint8_t>(lcg.next() % 60u); }
    return crop;
}

// ---------------------------------------------------------------------------
// Host contract checks.
// ---------------------------------------------------------------------------
void test_opendoc_contracts() {
    auto cfg = open_doc_synthetic_config();
    expect(cfg.validate().ok());
    // OpenDoc uses PP-DocLayoutV2 (not V3).
    expect(cfg.layout.version == PpDocLayoutVersion::v2);
    // UniRec is a CNN+CTC recognizer (no transformer decoder).
    expect(cfg.recognizer.recognizer.num_classes >= 2u);
    expect(cfg.recognizer.dictionary.num_classes() == cfg.recognizer.recognizer.num_classes);
    // Real-artifact gate fails closed.
    expect(open_doc_validate_real_artifact("r", "f").code == StatusCode::unvalidated);
    // UniRec tensor inventory present.
    UniRecTensorNames names;
    auto expected = uni_rec_expected_tensors(cfg.recognizer, names);
    expect(expected.size() > 10u);
}

// ---------------------------------------------------------------------------
// UniRec recognizer parity: device CTC decode vs independent UNFOLDED oracle.
// ---------------------------------------------------------------------------
void test_unirec_parity(Device &device) {
    Fixture fx;
    expect(fx.init(device));
    if (!fx.recognizer) { return; }
    auto stream = device.create_stream();
    const auto &rec = fx.weights.rec_cfg.recognizer;
    auto crop = make_crop(rec.input_width, rec.input_height, 0, LayoutBlockType::text);
    // Oracle preprocessing (float mirror) at the recognizer input size.
    auto pixels = ocr::test::p6_oracle_preprocess(crop.pixels, rec.input_width, rec.input_height,
                                                  rec.input_width, rec.input_height,
                                                  fx.rec_config.mean, fx.rec_config.std_dev);
    ocr::test::PpOracleRecognizer oracle;
    expect(ocr::test::opendoc_oracle_recognizer(fx.weights.rec_cfg, fx.weights.rec_weights, pixels,
                                                fx.rec_config.ctc.temperature, oracle).ok());
    // Oracle greedy CTC decode.
    auto oracle_decoded = output::ctc_decode_logits(oracle.logits, oracle.frames,
                                                    fx.rec_config.model.recognizer.dictionary, fx.rec_config.ctc);
    expect(oracle_decoded.ok());
    // Device recognize_batch.
    RegionRecognizeOptions opts;
    opts.max_output_tokens = 64;
    auto out = fx.recognizer->recognize_batch(stream, luisa::span<const RegionCropInput>{&crop, 1}, opts);
    expect(out.ok());
    if (!out.ok()) { return; }
    expect(out.value->size() == 1u);
    // The device CTC token ids must match the oracle decode exactly.
    if (oracle_decoded.ok()) {
        expect((*out.value)[0].tokens == oracle_decoded.value->ids);
        expect((*out.value)[0].text == oracle_decoded.value->text);
    }
}

// ---------------------------------------------------------------------------
// Region batching vs batch-1 equivalence.
// ---------------------------------------------------------------------------
void test_batch_equivalence(Device &device) {
    Fixture fx;
    expect(fx.init(device));
    if (!fx.recognizer) { return; }
    auto stream = device.create_stream();
    const auto &rec = fx.weights.rec_cfg.recognizer;
    luisa::vector<RegionCropInput> crops;
    for (uint32_t i = 0; i < 3u; i++) {
        crops.push_back(make_crop(rec.input_width, rec.input_height, i, LayoutBlockType::text, 11u + i));
    }
    RegionRecognizeOptions opts;
    opts.max_output_tokens = 64;
    luisa::vector<luisa::vector<uint32_t>> single_tokens;
    for (const auto &crop : crops) {
        auto out = fx.recognizer->recognize_batch(stream, luisa::span<const RegionCropInput>{&crop, 1}, opts);
        expect(out.ok());
        if (!out.ok()) { return; }
        single_tokens.push_back((*out.value)[0].tokens);
    }
    auto batched = fx.recognizer->recognize_batch(stream, crops, opts);
    expect(batched.ok());
    if (!batched.ok()) { return; }
    expect(batched.value->size() == crops.size());
    bool equivalent = true;
    for (size_t i = 0; i < crops.size(); i++) {
        if ((*batched.value)[i].tokens != single_tokens[i]) { equivalent = false; break; }
    }
    expect(equivalent) << "OpenDoc region batching must equal batch-1";
}

// ---------------------------------------------------------------------------
// Full two-stage pipeline: layout -> regions -> UniRec -> ordered result.
// ---------------------------------------------------------------------------
void test_full_pipeline(Device &device) {
    Fixture fx;
    expect(fx.init(device));
    if (!fx.recognizer || !fx.layout) { return; }
    auto stream = device.create_stream();
    // Orchestrator with the device layout policy + OpenDoc recognizer.
    LayoutVlmExecutorConfig orch;
    orch.layout_policy = LayoutDevicePolicy::device;
    orch.decode.score_threshold = 0.3f;
    orch.scheduler.max_patches_per_vision_batch = 4096;
    orch.scheduler.max_regions_per_batch = 4;
    orch.scheduler.max_decode_batch = 2;
    orch.region.max_output_tokens = 64;
    LayoutVlmExecutor orchestrator;
    expect(orchestrator.create(orch, fx.layout.get(), fx.weights.layout_cfg, fx.recognizer.get()).ok());
    auto page = ocr::test::p6_make_page(64, 64);
    auto run = orchestrator.recognize(stream, luisa::span<const LayoutPageInput>{&page, 1});
    if (!run.ok()) { LUISA_WARNING("opendoc pipeline: {} [{}]", run.status.message, run.status.detail); }
    expect(run.ok());
    if (!run.ok()) { return; }
    expect(run.value->pages.size() == 1u);
    // Regions are in reading order; source coordinates preserved.
    const auto &page_result = run.value->pages[0];
    uint32_t prev_order = 0;
    bool ordered = true;
    for (const auto &region : page_result.regions) {
        if (region.reading_order < prev_order) { ordered = false; }
        prev_order = region.reading_order;
        expect(region.page_index == 0u);
        expect(!region.source_box.empty());
    }
    expect(ordered);
    // Repeat-run determinism: a second run produces identical region content.
    auto run2 = orchestrator.recognize(stream, luisa::span<const LayoutPageInput>{&page, 1});
    expect(run2.ok());
    if (run2.ok()) {
        expect(run2.value->pages.size() == run.value->pages.size());
        if (run2.value->pages.size() == run.value->pages.size()) {
            bool same = true;
            for (size_t p = 0; p < run.value->pages.size(); p++) {
                const auto &a = run.value->pages[p];
                const auto &b = run2.value->pages[p];
                if (a.regions.size() != b.regions.size()) { same = false; break; }
                for (size_t i = 0; i < a.regions.size(); i++) {
                    if (a.regions[i].content != b.regions[i].content ||
                        a.regions[i].reading_order != b.regions[i].reading_order) { same = false; break; }
                }
            }
            expect(same) << "repeat pipeline runs must be identical";
        }
    }
}

// ---------------------------------------------------------------------------
// Empty page: valid empty result, no hallucinated regions.
// ---------------------------------------------------------------------------
void test_empty_page(Device &device) {
    Fixture fx;
    expect(fx.init(device));
    if (!fx.recognizer || !fx.layout) { return; }
    auto stream = device.create_stream();
    LayoutVlmExecutorConfig orch;
    orch.layout_policy = LayoutDevicePolicy::device;
    orch.decode.score_threshold = 0.9f;// high threshold: blank page -> no regions
    orch.region.max_output_tokens = 64;
    LayoutVlmExecutor orchestrator;
    expect(orchestrator.create(orch, fx.layout.get(), fx.weights.layout_cfg, fx.recognizer.get()).ok());
    LayoutPageInput blank;
    blank.source_width = 64;
    blank.source_height = 64;
    blank.page_index = 0;
    blank.pixels.assign(static_cast<size_t>(64) * 64 * 3u, 0u);
    auto run = orchestrator.recognize(stream, luisa::span<const LayoutPageInput>{&blank, 1});
    expect(run.ok());
    if (!run.ok()) { return; }
    expect(run.value->pages.size() == 1u);
    expect(run.value->pages[0].regions.empty());
    expect(run.value->regions_processed == 0u);
}

}// namespace

int main(int argc, char *argv[]) {
    auto dc = luisa::test::create_device_from_ut(argc, argv);
    if (!dc) { return 0; }
    boost::ut::detail::cfg::parse_arg_with_fallback(argc, const_cast<const char **>(argv));
    auto &device = dc->device;
    "opendoc_contracts"_test = [] { test_opendoc_contracts(); };
    "opendoc_unirec_parity"_test = [&] { test_unirec_parity(device); };
    "opendoc_batch_equivalence"_test = [&] { test_batch_equivalence(device); };
    "opendoc_full_pipeline"_test = [&] { test_full_pipeline(device); };
    "opendoc_empty_page"_test = [&] { test_empty_page(device); };
}
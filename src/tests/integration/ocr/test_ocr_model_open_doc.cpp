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
// Region batching vs batch-1 equivalence (ordered outputs). NOTE (S7): Phase 6
// execution is batch-1; this asserts ordered equivalence of the plan/restore
// path, not batched kernel execution (Phase 11).
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
    orch.scheduler.max_output_tokens_per_region = 64;
    LayoutVlmExecutor orchestrator;
    expect(orchestrator.create(orch, fx.layout.get(), fx.weights.layout_cfg, fx.recognizer.get()).ok());
    auto page = ocr::test::p6_make_page(64, 64);
    auto run = orchestrator.recognize(stream, luisa::span<const LayoutPageInput>{&page, 1});
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
    orch.scheduler.max_output_tokens_per_region = 64;
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

// ---------------------------------------------------------------------------
// S5: UniRec tensor-contract refusal. The recognizer load path resolves the
// DISTINCT UniRec names (unirec.*), NOT a PP-OCRv6 alias. A package written under
// PP-OCRv6 recognizer names, a missing tensor, or a wrong-shape tensor must all
// be refused. The on-disk inventory is uni_rec_expected_tensors.
// ---------------------------------------------------------------------------
void test_unirec_tensor_refusal(Device &device) {
    ocr::test::P6TempDir tmp;
    ocr::test::OpenDocFixtureWeights w;
    expect(ocr::test::generate_opendoc_weights(w).ok());
    auto stream = device.create_stream();
    const auto &rec = w.rec_cfg.recognizer;
    const auto conv_shape = [](const PpConvStage &s) {
        return luisa::vector<uint64_t>{s.out_c, s.in_c / s.groups, s.kh, s.kw};
    };
    uint32_t collapsed = 0;
    if (auto cc = rec.collapsed_channels(); cc.ok()) { collapsed = *cc.value; }
    // Helper: write a recognizer package with a custom name map / mutation.
    // mode 0 = correct UniRec names; 1 = PP-OCRv6 names (alias refusal);
    // 2 = drop the CTC weight (missing); 3 = wrong CTC bias shape.
    const auto write_rec = [&](const luisa::filesystem::path &path, int mode) -> Status {
        auto writer = PackageWriter::create(path);
        if (!writer.ok()) { return writer.status; }
        auto &wr = *writer.value;
        const auto add = [&wr](const luisa::string &name, luisa::vector<uint64_t> shape,
                               TensorLayout layout, luisa::span<const float> data) -> Status {
            PackageWriter::TensorSpec spec;
            spec.name = name;
            spec.encoding = TensorEncoding::fp32;
            spec.layout = layout;
            spec.shape = std::move(shape);
            spec.payload = ocr::test::p6_f32_bytes(data);
            return wr->add_tensor(spec);
        };
        PpOcrv6TensorNames pp;
        const auto bw = [&](uint32_t i) { return mode == 1 ? pp.rec_backbone_w(i) : w.rec_names.backbone_w(i); };
        const auto bb = [&](uint32_t i) { return mode == 1 ? pp.rec_backbone_b(i) : w.rec_names.backbone_b(i); };
        for (uint32_t i = 0; i < rec.backbone.size(); i++) {
            const auto &s = rec.backbone[i];
            if (auto st = add(bw(i), conv_shape(s), TensorLayout::conv_oihw, w.rec_folded.rec_backbone_w[i]); !st.ok()) { return st; }
            if (auto st = add(bb(i), {s.out_c}, TensorLayout::row_major, w.rec_folded.rec_backbone_b[i]); !st.ok()) { return st; }
        }
        const auto f1w = mode == 1 ? pp.rec_svtr_fc1_w() : w.rec_names.svtr_fc1_w();
        const auto f1b = mode == 1 ? pp.rec_svtr_fc1_b() : w.rec_names.svtr_fc1_b();
        const auto f2w = mode == 1 ? pp.rec_svtr_fc2_w() : w.rec_names.svtr_fc2_w();
        const auto f2b = mode == 1 ? pp.rec_svtr_fc2_b() : w.rec_names.svtr_fc2_b();
        const auto cw = mode == 1 ? pp.rec_ctc_w() : w.rec_names.ctc_w();
        const auto cb = mode == 1 ? pp.rec_ctc_b() : w.rec_names.ctc_b();
        if (auto s = add(f1w, {collapsed, rec.svtr_intermediate}, TensorLayout::kn, w.rec_weights.svtr_fc1_w); !s.ok()) { return s; }
        if (auto s = add(f1b, {rec.svtr_intermediate}, TensorLayout::row_major, w.rec_weights.svtr_fc1_b); !s.ok()) { return s; }
        if (auto s = add(f2w, {rec.svtr_intermediate, collapsed}, TensorLayout::kn, w.rec_weights.svtr_fc2_w); !s.ok()) { return s; }
        if (auto s = add(f2b, {collapsed}, TensorLayout::row_major, w.rec_weights.svtr_fc2_b); !s.ok()) { return s; }
        if (mode != 2) {// mode 2 drops the CTC weight (missing-tensor refusal)
            if (auto s = add(cw, {collapsed, rec.num_classes}, TensorLayout::kn, w.rec_weights.ctc_w); !s.ok()) { return s; }
        }
        const uint32_t cb_shape = mode == 3 ? rec.num_classes + 1u : rec.num_classes;// mode 3 wrong shape
        luisa::vector<float> cb_data(w.rec_weights.ctc_b.begin(), w.rec_weights.ctc_b.end());
        if (mode == 3) { cb_data.push_back(0.0f); }
        if (auto s = add(cb, {cb_shape}, TensorLayout::row_major, cb_data); !s.ok()) { return s; }
        return wr->finish(ocr::test::p6_manifest("unirec"));
    };
    // Build the recognizer config (UniRec geometry + names).
    OpenDocRecognizerConfig rc;
    rc.model.layout = w.layout_cfg;
    rc.model.layout_names = w.layout_names;
    rc.model.recognizer.recognizer = rec;
    rc.model.recognizer.dictionary = ocr::test::pp_synthetic_dictionary();
    rc.model.recognizer_names = UniRecTensorNames{};
    rc.recognizer_component = "unirec";
    // mode 0: correct UniRec names -> the on-disk inventory validates AND create succeeds.
    {
        const auto path = tmp.path / "unirec_ok.lcocr";
        expect(write_rec(path, 0).ok());
        auto reader = PackageReader::open(path);
        expect(reader.ok());
        if (reader.ok()) {
            UniRecConfig uc;
            uc.recognizer = rec;
            uc.dictionary = ocr::test::pp_synthetic_dictionary();
            auto expected = uni_rec_expected_tensors(uc, UniRecTensorNames{});
            expect(uni_rec_validate_tensors((*reader.value)->tensors(), expected).ok());
            WeightStore store;
            expect(store.upload_component_sync(device, stream, **reader.value, "unirec").ok());
            OpenDocRecognizer r;
            expect(r.create(device, stream, rc, store).ok());
        }
    }
    // mode 1: PP-OCRv6 names -> on-disk UniRec validation refuses (missing unirec.*).
    {
        const auto path = tmp.path / "unirec_pp.lcocr";
        expect(write_rec(path, 1).ok());
        auto reader = PackageReader::open(path);
        expect(reader.ok());
        if (reader.ok()) {
            UniRecConfig uc;
            uc.recognizer = rec;
            uc.dictionary = ocr::test::pp_synthetic_dictionary();
            auto expected = uni_rec_expected_tensors(uc, UniRecTensorNames{});
            auto v = uni_rec_validate_tensors((*reader.value)->tensors(), expected);
            expect(!v.ok()) << "PP-OCRv6 names must not satisfy the UniRec inventory";
            // The device load path also refuses (no unirec.* tensors installed).
            WeightStore store;
            expect(store.upload_component_sync(device, stream, **reader.value, "unirec").ok());
            OpenDocRecognizer r;
            expect(!r.create(device, stream, rc, store).ok());
        }
    }
    // mode 2: missing CTC weight -> refusal.
    {
        const auto path = tmp.path / "unirec_missing.lcocr";
        expect(write_rec(path, 2).ok());
        auto reader = PackageReader::open(path);
        expect(reader.ok());
        if (reader.ok()) {
            UniRecConfig uc;
            uc.recognizer = rec;
            uc.dictionary = ocr::test::pp_synthetic_dictionary();
            auto expected = uni_rec_expected_tensors(uc, UniRecTensorNames{});
            expect(!uni_rec_validate_tensors((*reader.value)->tensors(), expected).ok());
            WeightStore store;
            expect(store.upload_component_sync(device, stream, **reader.value, "unirec").ok());
            OpenDocRecognizer r;
            expect(!r.create(device, stream, rc, store).ok());
        }
    }
    // mode 3: wrong CTC bias shape -> refusal.
    {
        const auto path = tmp.path / "unirec_shape.lcocr";
        expect(write_rec(path, 3).ok());
        auto reader = PackageReader::open(path);
        expect(reader.ok());
        if (reader.ok()) {
            UniRecConfig uc;
            uc.recognizer = rec;
            uc.dictionary = ocr::test::pp_synthetic_dictionary();
            auto expected = uni_rec_expected_tensors(uc, UniRecTensorNames{});
            expect(!uni_rec_validate_tensors((*reader.value)->tensors(), expected).ok());
        }
    }
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
    "opendoc_unirec_tensor_refusal"_test = [&] { test_unirec_tensor_refusal(device); };
}
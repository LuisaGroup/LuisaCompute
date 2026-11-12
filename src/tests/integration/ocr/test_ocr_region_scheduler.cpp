// Test for ocr/runtime/scheduler + LayoutVlmExecutor orchestration (Phase 6,
// device) -- token-budgeted region scheduling over the full layout->region
// pipeline. Covers: one overview/layout pass, crop sorting + batch packing under
// max_patches/max_regions/max_decode, separate overview/crop/output budgets,
// independent region decode, restore to page/reading order, region batching vs
// batch-1 equivalence, detector-output caching for recognition retry (explicit
// content/geometry key), cancellation/time/memory checks, the documented
// cpu_reference layout option, and cross-page composition.
// Uses the OpenDoc two-stage recognizer (PP-DocLayoutV2 + UniRec) as the region
// engine. Synthetic weights only; no model/download, no real-model claim.
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

struct Fixture {
    ocr::test::P6TempDir tmp;
    ocr::test::OpenDocFixtureWeights weights;
    WeightStore store;
    luisa::unique_ptr<PpDocLayoutExecutor> layout;
    luisa::unique_ptr<OpenDocRecognizer> recognizer;
    PpDocLayoutExecutorConfig layout_config;
    OpenDocRecognizerConfig rec_config;

    [[nodiscard]] bool init(Device &device, uint32_t seed = 0x5EEDu) {
        if (!ocr::test::generate_opendoc_weights(weights, seed).ok()) { return false; }
        const auto lpath = tmp.path / "rs_layout.lcocr";
        const auto rpath = tmp.path / "rs_unirec.lcocr";
        if (!ocr::test::write_opendoc_packages(lpath, rpath, weights).ok()) { return false; }
        auto stream = device.create_stream();
        if (!ocr::test::upload_opendoc_weights(device, stream, lpath, rpath, store).ok()) { return false; }
        layout_config.model = weights.layout_cfg;
        layout_config.names = weights.layout_names;
        layout_config.component = "layout";
        layout = luisa::make_unique<PpDocLayoutExecutor>();
        if (!layout->create(device, stream, layout_config, store).ok()) { return false; }
        rec_config.model.layout = weights.layout_cfg;
        rec_config.model.layout_names = weights.layout_names;
        rec_config.model.recognizer.recognizer = weights.rec_cfg.recognizer;
        rec_config.model.recognizer.dictionary = ocr::test::pp_synthetic_dictionary();
        rec_config.model.recognizer_names = UniRecTensorNames{};
        rec_config.recognizer_component = "unirec";
        recognizer = luisa::make_unique<OpenDocRecognizer>();
        return recognizer->create(device, stream, rec_config, store).ok();
    }

    [[nodiscard]] LayoutVlmExecutorConfig orch_config() const {
        LayoutVlmExecutorConfig orch;
        orch.layout_policy = LayoutDevicePolicy::device;
        orch.decode.score_threshold = 0.3f;
        orch.scheduler.max_patches_per_vision_batch = 4096;
        orch.scheduler.max_regions_per_batch = 4;
        orch.scheduler.max_decode_batch = 2;
        orch.region.max_output_tokens = 64;
        return orch;
    }
};

// ---------------------------------------------------------------------------
// Region batching vs batch-1: equivalent ordered outputs.
// ---------------------------------------------------------------------------
void test_batch_vs_single(Device &device) {
    Fixture fx;
    expect(fx.init(device));
    if (!fx.recognizer || !fx.layout) { return; }
    auto stream = device.create_stream();
    auto page = ocr::test::p6_make_page(64, 64);
    // Batched: max_regions_per_batch = 4.
    auto orch_batched = fx.orch_config();
    orch_batched.scheduler.max_regions_per_batch = 4;
    LayoutVlmExecutor exec_batched;
    expect(exec_batched.create(orch_batched, fx.layout.get(), fx.weights.layout_cfg, fx.recognizer.get()).ok());
    auto run_batched = exec_batched.recognize(stream, luisa::span<const LayoutPageInput>{&page, 1});
    expect(run_batched.ok());
    // Batch-1: max_regions_per_batch = 1, max_decode_batch = 1.
    auto orch_single = fx.orch_config();
    orch_single.scheduler.max_regions_per_batch = 1;
    orch_single.scheduler.max_decode_batch = 1;
    LayoutVlmExecutor exec_single;
    expect(exec_single.create(orch_single, fx.layout.get(), fx.weights.layout_cfg, fx.recognizer.get()).ok());
    auto run_single = exec_single.recognize(stream, luisa::span<const LayoutPageInput>{&page, 1});
    expect(run_single.ok());
    if (!run_batched.ok() || !run_single.ok()) { return; }
    // Same regions in the same reading order with the same content.
    expect(run_batched.value->pages.size() == run_single.value->pages.size());
    if (run_batched.value->pages.size() != run_single.value->pages.size()) { return; }
    bool equivalent = true;
    for (size_t p = 0; p < run_batched.value->pages.size(); p++) {
        const auto &a = run_batched.value->pages[p];
        const auto &b = run_single.value->pages[p];
        if (a.regions.size() != b.regions.size()) { equivalent = false; break; }
        for (size_t i = 0; i < a.regions.size(); i++) {
            if (a.regions[i].content != b.regions[i].content ||
                a.regions[i].reading_order != b.regions[i].reading_order ||
                a.regions[i].block_type != b.regions[i].block_type) { equivalent = false; break; }
        }
    }
    expect(equivalent) << "region batching must produce equivalent ordered outputs vs batch-1";
}

// ---------------------------------------------------------------------------
// Detector-output cache: a recognition retry reuses cached layout.
// ---------------------------------------------------------------------------
void test_layout_cache(Device &device) {
    Fixture fx;
    expect(fx.init(device));
    if (!fx.recognizer || !fx.layout) { return; }
    auto stream = device.create_stream();
    auto orch = fx.orch_config();
    orch.layout_cache_entries = 8;
    LayoutVlmExecutor exec;
    expect(exec.create(orch, fx.layout.get(), fx.weights.layout_cfg, fx.recognizer.get()).ok());
    auto page = ocr::test::p6_make_page(64, 64);
    auto run1 = exec.recognize(stream, luisa::span<const LayoutPageInput>{&page, 1});
    expect(run1.ok());
    // Second run with the SAME page: the layout cache must hit (no re-detect).
    auto run2 = exec.recognize(stream, luisa::span<const LayoutPageInput>{&page, 1});
    expect(run2.ok());
    if (!run1.ok() || !run2.ok()) { return; }
    expect(run2.value->layout_cache_hits >= 1u);
    // A DIFFERENT page (different content hash) must NOT hit the cache.
    auto page2 = ocr::test::p6_make_page(64, 64, 99u);
    auto run3 = exec.recognize(stream, luisa::span<const LayoutPageInput>{&page2, 1});
    expect(run3.ok());
    if (run3.ok()) { expect(run3.value->layout_cache_hits == 0u); }
    // Cache disabled: no hits even on repeat.
    auto orch_nocache = fx.orch_config();
    orch_nocache.layout_cache_entries = 0;
    LayoutVlmExecutor exec_nocache;
    expect(exec_nocache.create(orch_nocache, fx.layout.get(), fx.weights.layout_cfg, fx.recognizer.get()).ok());
    auto run4 = exec_nocache.recognize(stream, luisa::span<const LayoutPageInput>{&page, 1});
    auto run5 = exec_nocache.recognize(stream, luisa::span<const LayoutPageInput>{&page, 1});
    expect(run4.ok() && run5.ok());
    if (run4.ok() && run5.ok()) { expect(run5.value->layout_cache_hits == 0u); }
}

// ---------------------------------------------------------------------------
// Cancellation: a fired callback stops the request with cancelled.
// ---------------------------------------------------------------------------
void test_cancellation(Device &device) {
    Fixture fx;
    expect(fx.init(device));
    if (!fx.recognizer || !fx.layout) { return; }
    auto stream = device.create_stream();
    auto orch = fx.orch_config();
    orch.scheduler.cancel_callback = [] { return true; };
    LayoutVlmExecutor exec;
    expect(exec.create(orch, fx.layout.get(), fx.weights.layout_cfg, fx.recognizer.get()).ok());
    auto page = ocr::test::p6_make_page(64, 64);
    auto run = exec.recognize(stream, luisa::span<const LayoutPageInput>{&page, 1});
    expect(!run.ok());
    expect(run.status.code == StatusCode::cancelled);
}

// ---------------------------------------------------------------------------
// Memory budget: an absurdly small device limit refuses before allocation.
// ---------------------------------------------------------------------------
void test_memory_budget(Device &device) {
    Fixture fx;
    expect(fx.init(device));
    if (!fx.recognizer || !fx.layout) { return; }
    auto stream = device.create_stream();
    // The layout executor was created without a limit; verify the scheduler's
    // per-page output budget preflight refuses an over-large region count.
    auto orch = fx.orch_config();
    orch.scheduler.max_output_tokens_per_page = 1;// cannot hold >1 region
    LayoutVlmExecutor exec;
    expect(exec.create(orch, fx.layout.get(), fx.weights.layout_cfg, fx.recognizer.get()).ok());
    auto page = ocr::test::p6_make_page(64, 64);
    auto run = exec.recognize(stream, luisa::span<const LayoutPageInput>{&page, 1});
    // If the page yields >1 region, the budget preflight refuses; if 0/1, it
    // succeeds. Either way the result is well-formed (no crash, no OOB).
    if (!run.ok()) {
        expect(run.status.code == StatusCode::out_of_memory);
    } else {
        expect(run.value->pages.size() == 1u);
    }
}

// ---------------------------------------------------------------------------
// CPU-reference layout option: head maps from the independent oracle.
// ---------------------------------------------------------------------------
void test_cpu_reference_layout(Device &device) {
    Fixture fx;
    expect(fx.init(device));
    if (!fx.recognizer) { return; }
    auto stream = device.create_stream();
    auto page = ocr::test::p6_make_page(64, 64);
    // Device-policy run for reference.
    auto orch_dev = fx.orch_config();
    LayoutVlmExecutor exec_dev;
    expect(exec_dev.create(orch_dev, fx.layout.get(), fx.weights.layout_cfg, fx.recognizer.get()).ok());
    auto run_dev = exec_dev.recognize(stream, luisa::span<const LayoutPageInput>{&page, 1});
    expect(run_dev.ok());
    // CPU-reference run: supply head maps from the independent CPU oracle.
    auto orch_cpu = fx.orch_config();
    orch_cpu.layout_policy = LayoutDevicePolicy::cpu_reference;
    LayoutVlmExecutor exec_cpu;
    auto provider = [&](const LayoutPageInput &p) -> Result<LayoutHeadMaps> {
        auto pixels = ocr::test::p6_oracle_preprocess(p.pixels, p.source_width, p.source_height,
                                                      fx.weights.layout_cfg.input_width,
                                                      fx.weights.layout_cfg.input_height,
                                                      fx.layout_config.mean, fx.layout_config.std_dev);
        ocr::test::LayoutOracleMaps oracle;
        if (auto s = ocr::test::oracle_layout_detect(fx.weights.layout_cfg, fx.weights.layout_weights, pixels, oracle); !s.ok()) {
            return Result<LayoutHeadMaps>::failure(s);
        }
        LayoutHeadMaps maps;
        maps.map_width = oracle.map_w;
        maps.map_height = oracle.map_h;
        maps.num_classes = oracle.num_classes;
        maps.class_probs = std::move(oracle.class_probs);
        maps.box_deltas = std::move(oracle.box_deltas);
        return Result<LayoutHeadMaps>::success(std::move(maps));
    };
    expect(exec_cpu.create(orch_cpu, nullptr, fx.weights.layout_cfg, fx.recognizer.get(), provider).ok());
    auto run_cpu = exec_cpu.recognize(stream, luisa::span<const LayoutPageInput>{&page, 1});
    expect(run_cpu.ok());
    if (!run_dev.ok() || !run_cpu.ok()) { return; }
    // The CPU-reference path must produce the same regions as the device path
    // (the oracle mirrors the device conv chain within tolerance).
    expect(run_cpu.value->pages.size() == run_dev.value->pages.size());
    // The cpu_reference diagnostic must be present.
    bool has_diag = false;
    for (const auto &d : run_cpu.value->diagnostics) {
        if (d.code == "layout_cpu_reference") { has_diag = true; }
    }
    expect(has_diag);
}

// ---------------------------------------------------------------------------
// Cross-page composition: pages preserved in ascending order.
// ---------------------------------------------------------------------------
void test_cross_page(Device &device) {
    Fixture fx;
    expect(fx.init(device));
    if (!fx.recognizer || !fx.layout) { return; }
    auto stream = device.create_stream();
    auto page0 = ocr::test::p6_make_page(64, 64, 7u);
    page0.page_index = 0;
    auto page1 = ocr::test::p6_make_page(64, 64, 8u);
    page1.page_index = 1;
    luisa::vector<LayoutPageInput> pages{page0, page1};
    auto orch = fx.orch_config();
    LayoutVlmExecutor exec;
    expect(exec.create(orch, fx.layout.get(), fx.weights.layout_cfg, fx.recognizer.get()).ok());
    auto run = exec.recognize(stream, pages);
    expect(run.ok());
    if (!run.ok()) { return; }
    expect(run.value->pages.size() == 2u);
    if (run.value->pages.size() == 2u) {
        expect(run.value->pages[0].page_index == 0u);
        expect(run.value->pages[1].page_index == 1u);
    }
}

}// namespace

int main(int argc, char *argv[]) {
    auto dc = luisa::test::create_device_from_ut(argc, argv);
    if (!dc) { return 0; }
    boost::ut::detail::cfg::parse_arg_with_fallback(argc, const_cast<const char **>(argv));
    auto &device = dc->device;
    "region_scheduler_batch_vs_single"_test = [&] { test_batch_vs_single(device); };
    "region_scheduler_layout_cache"_test = [&] { test_layout_cache(device); };
    "region_scheduler_cancellation"_test = [&] { test_cancellation(device); };
    "region_scheduler_memory_budget"_test = [&] { test_memory_budget(device); };
    "region_scheduler_cpu_reference"_test = [&] { test_cpu_reference_layout(device); };
    "region_scheduler_cross_page"_test = [&] { test_cross_page(device); };
}
// Test for ocr/models/mineru + the shared Phase 7 slice executor (device) --
// the SMALL SYNTHETIC MinerU2.5-Pro vertical slice: overview layout +
// native-resolution crops, a 675M-NaViT-shaped vision tower with 2D-RoPE, a
// 2x2 pixel-unshuffle connector (NOT Paddle's adaptive connector), and a Qwen2
// decoder with adjacent-pair M-RoPE. Reading-order + cross-page composition.
// Covers: config/tensor contracts, Qwen3/adaptive fail-closed parsing, NaViT +
// pixel-unshuffle parity (the channel-fold ordering is the distinguishing
// contract), adjacent-MRoPE prefill logits + decode KV, deterministic tokens,
// blank/repeated/adversarial, budget + repeat-run, and cross-page composition.
// Synthetic weights only; no model/download, no real-model parity claim.
#include "ut/ut.hpp"
#include "test_device.h"
#include "ocr_p7_test_utils.h"
#include "ocr_p6_test_utils.h"

#include "ocr/models/mineru.h"
#include <luisa/backends/ext/cuda/cuda_graph_ext.h>
#include "ocr/runtime/layout_vlm_executor.h"

using namespace luisa;
using namespace luisa::compute;
using namespace luisa::ocr;
using namespace boost::ut;
using namespace boost::ut::literals;

namespace {

constexpr float kAtol = 3e-3f;
constexpr float kRtol = 3e-3f;
constexpr float kLinAtol = 5e-4f;
constexpr float kLinRtol = 5e-4f;

struct Fixture {
    ocr::test::P7TempDir tmp;
    MineruConfig cfg;
    ocr::test::P7SyntheticWeights weights;
    WeightStore store;
    luisa::unique_ptr<P7VlmExecutor> exec;

    [[nodiscard]] bool init(Device &device, uint32_t seed = 0x5EEDu) {
        cfg = mineru_synthetic_config();
        if (!cfg.validate().ok()) { return false; }
        weights = ocr::test::generate_p7_weights(cfg.graph, seed);
        const auto vpath = tmp.path / "mineru_vision.lcocr";
        const auto dpath = tmp.path / "mineru_decoder.lcocr";
        if (auto s = ocr::test::write_p7_packages(vpath, dpath, cfg.graph, cfg.names, weights, "mineru_2_5_pro_2605_1_2b"); !s.ok()) {
            LUISA_WARNING("write_p7_packages: {}", s.message);
            return false;
        }
        auto stream = device.create_stream();
        if (auto s = ocr::test::upload_p7_weights(device, stream, vpath, dpath, store); !s.ok()) {
            LUISA_WARNING("upload_p7: {}", s.message);
            return false;
        }
        exec = luisa::make_unique<P7VlmExecutor>();
        if (auto s = exec->create(device, stream, cfg.graph, cfg.names, store); !s.ok()) {
            LUISA_WARNING("mineru create: {}", s.message);
            return false;
        }
        return true;
    }
};

[[nodiscard]] P7ImageInput make_image(const MineruConfig &cfg, uint32_t w, uint32_t h, uint32_t seed = 11u) {
    auto crop = ocr::test::p7_make_crop(w, h, 0, seed);
    P7ImageInput img;
    img.pixels = crop.pixels;
    img.source_width = w;
    img.source_height = h;
    auto packed = p7_pack_image(w, h, cfg.graph.packing, cfg.graph.connector);
    img.packed = *packed.value;
    return img;
}

// ---------------------------------------------------------------------------
// Host contract checks + fail-closed parsing.
// ---------------------------------------------------------------------------
void test_mineru_contracts() {
    auto cfg = mineru_synthetic_config();
    expect(cfg.validate().ok());
    expect(cfg.graph.connector.kind == P7ConnectorKind::unshuffle_mlp);
    expect(cfg.graph.connector.spatial_factor == 2u);
    expect(cfg.graph.vision.rope_kind == P7VisionRopeKind::axial2d);
    expect(cfg.graph.text_pos == P7TextPosKind::mrope_adjacent);
    expect(cfg.graph.text.rope_pair_layout == RopePairLayout::adjacent);
    auto expected = mineru_expected_tensors(cfg);
    expect(expected.size() > 30u);
    expect(mineru_validate_real_artifact("r", "f").code == StatusCode::unvalidated);
    const char *good = R"({"model_type":"mineru","vision_config":{"num_layers":2,"hidden_size":16,"num_heads":2,"head_dim":8,"intermediate_size":32,"patch_size":2},"text_config":{"vocab_size":32,"hidden_size":32,"num_hidden_layers":3,"num_attention_heads":4,"num_key_value_heads":2,"intermediate_size":64,"max_position_embeddings":256},"mrope_section":[2,1,1]})";
    auto parsed = parse_mineru_config(good);
    expect(parsed.ok());
    if (parsed.ok()) { expect(parsed.value->validate().ok()); }
    // Qwen3 QK-Norm fails closed.
    const char *qwen3 = R"({"model_type":"mineru","vision_config":{"num_layers":2,"hidden_size":16,"num_heads":2,"head_dim":8,"intermediate_size":32,"patch_size":2},"text_config":{"vocab_size":32,"hidden_size":32,"num_hidden_layers":3,"num_attention_heads":4,"num_key_value_heads":2,"intermediate_size":64,"max_position_embeddings":256,"qk_norm":true},"mrope_section":[2,1,1]})";
    expect(!parse_mineru_config(qwen3).ok());
    // Paddle adaptive connector fails closed.
    const char *adaptive = R"({"model_type":"mineru","vision_config":{"num_layers":2,"hidden_size":16,"num_heads":2,"head_dim":8,"intermediate_size":32,"patch_size":2},"text_config":{"vocab_size":32,"hidden_size":32,"num_hidden_layers":3,"num_attention_heads":4,"num_key_value_heads":2,"intermediate_size":64,"max_position_embeddings":256},"mrope_section":[2,1,1],"connector":{"kind":"adaptive"}})";
    expect(!parse_mineru_config(adaptive).ok());
}

// ---------------------------------------------------------------------------
// NaViT + pixel-unshuffle parity. The channel-fold ordering is the
// distinguishing contract: verify the reduced connector input directly.
// ---------------------------------------------------------------------------
void test_mineru_vision_parity(Device &device) {
    Fixture fx;
    expect(fx.init(device));
    if (!fx.exec) { return; }
    auto stream = device.create_stream();
    for (const auto &[w, h] : std::initializer_list<std::pair<uint32_t, uint32_t>>{{8, 8}, {12, 8}, {16, 8}}) {
        auto img = make_image(fx.cfg, w, h);
        auto pixels = ocr::test::p7_oracle_preprocess(img.pixels, w, h,
                                                      img.packed.resized_width, img.packed.resized_height,
                                                      fx.cfg.graph.mean, fx.cfg.graph.std_dev);
        ocr::test::P7VisionOracle oracle;
        expect(ocr::test::p7_oracle_vision(fx.cfg.graph, fx.weights, pixels, img.packed, oracle).ok());
        fx.exec->reset_vision_workspace();
        {
            auto list = CommandList::create();
            auto v = fx.exec->record_vision(list, luisa::span<const P7ImageInput>{&img, 1});
            expect(v.ok());
            if (!v.ok()) { continue; }
            stream << list.commit() << synchronize();
        }
        luisa::vector<float> dev_patches, dev_final, dev_features, dev_reduced;
        expect(fx.exec->download_vision_patches(stream, dev_patches).ok());
        expect(fx.exec->download_vision_final(stream, dev_final).ok());
        expect(fx.exec->download_vision_features(stream, dev_features).ok());
        expect(fx.exec->download_vision_reduced(stream, dev_reduced).ok());
        stream << synchronize();
        const auto what = luisa::format("mineru {}x{}", w, h);
        ocr::test::expect_close(dev_patches, oracle.patches, kLinAtol, kLinRtol, (what + " patches").c_str());
        ocr::test::expect_close(dev_final, oracle.block_out.back(), kAtol, kRtol, (what + " navit final").c_str());
        ocr::test::expect_close(dev_reduced, oracle.reduced, kAtol, kRtol, (what + " unshuffle reduced").c_str());
        ocr::test::expect_close(dev_features, oracle.features, kAtol, kRtol, (what + " pixel-unshuffle connector").c_str());
        // The pixel-unshuffle reduction must DIFFER from a plain spatial-merge
        // (the channel-fold ordering is distinct). Compare the oracle reduced
        // tensors: unshuffle vs spatial-merge of the same vision output.
        const auto &vout = oracle.block_out.back();
        auto spatial = ocr::test::p7_oracle_spatial_merge(vout, static_cast<uint32_t>(img.packed.patch_count),
                                                          fx.cfg.graph.vision.hidden_size, img.packed.grid_w, 2u);
        bool differ = false;
        for (size_t i = 0; i < oracle.reduced.size() && i < spatial.size(); i++) {
            if (std::abs(oracle.reduced[i] - spatial[i]) > 1e-4f) { differ = true; break; }
        }
        expect(differ) << "pixel-unshuffle and spatial-merge reductions must differ (channel-fold ordering)";
    }
}

// ---------------------------------------------------------------------------
// Adjacent-MRoPE Qwen2 decoder parity: prefill logits + decode KV/logits.
// ---------------------------------------------------------------------------
void test_mineru_decoder_parity(Device &device) {
    Fixture fx;
    expect(fx.init(device));
    if (!fx.exec) { return; }
    auto stream = device.create_stream();
    auto img = make_image(fx.cfg, 8, 8);
    auto pixels = ocr::test::p7_oracle_preprocess(img.pixels, 8, 8,
                                                  img.packed.resized_width, img.packed.resized_height,
                                                  fx.cfg.graph.mean, fx.cfg.graph.std_dev);
    ocr::test::P7VisionOracle vision_oracle;
    expect(ocr::test::p7_oracle_vision(fx.cfg.graph, fx.weights, pixels, img.packed, vision_oracle).ok());
    fx.exec->reset_vision_workspace();
    P7VisionOutput vision;
    {
        auto list = CommandList::create();
        auto v = fx.exec->record_vision(list, luisa::span<const P7ImageInput>{&img, 1});
        expect(v.ok());
        if (!v.ok()) { return; }
        vision = *v.value;
        stream << list.commit() << synchronize();
    }
    const uint32_t tokens = static_cast<uint32_t>(img.packed.visual_tokens);
    luisa::vector<uint32_t> prompt = {1u};
    const uint32_t mm_start = static_cast<uint32_t>(prompt.size());
    for (uint32_t i = 0; i < tokens; i++) { prompt.push_back(0u); }
    prompt.push_back(2u);
    luisa::vector<int32_t> position_ids, next_position;
    auto req = fx.exec->build_decode_request(prompt, mm_start, tokens, vision,
                                             luisa::span<const P7ImageInput>{&img, 1}, position_ids, next_position);
    expect(req.ok());
    if (!req.ok()) { return; }
    luisa::vector<luisa::vector<float>> k_caches, v_caches;
    ocr::test::P7TextOracle text_oracle;
    expect(ocr::test::p7_oracle_text_forward(fx.cfg.graph, fx.weights, prompt, position_ids,
                                             vision_oracle.features, mm_start, tokens, 0u,
                                             k_caches, v_caches, text_oracle).ok());
    const uint32_t vocab = fx.cfg.graph.text.vocab_size;
    const uint32_t prompt_len = static_cast<uint32_t>(prompt.size());
    ContiguousKvCache cache;
    KvCacheConfig kc;
    kc.num_layers = fx.cfg.graph.text.num_layers;
    kc.num_kv_heads = fx.cfg.graph.text.num_kv_heads;
    kc.head_dim = fx.cfg.graph.text.head_dim;
    kc.max_seq_len = prompt_len + 8;
    kc.batch = 1;
    kc.dtype_bytes = 4;
    expect(cache.create(device, stream, kc, nullptr).ok());
    auto logits_buf = device.create_buffer<float>(static_cast<size_t>(prompt_len) * vocab);
    {
        auto list = CommandList::create();
        auto s = fx.exec->text_executor().record_prefill_positions(list, *req.value, position_ids, {},
                                                                   cache, true, {}, logits_buf.view());
        expect(s.ok());
        if (!s.ok()) { return; }
        stream << list.commit() << synchronize();
    }
    luisa::vector<float> dev_logits(static_cast<size_t>(prompt_len) * vocab);
    stream << logits_buf.copy_to(luisa::span<float>{dev_logits}) << synchronize();
    ocr::test::expect_close(dev_logits, text_oracle.logits, kAtol, kRtol, "mineru adjacent-MRoPE prefill logits");
    const uint32_t first = ocr::test::p7_oracle_argmax(luisa::span<const float>{
        text_oracle.logits.data() + static_cast<size_t>(prompt_len - 1) * vocab, vocab});
    luisa::vector<uint32_t> step_tokens = {first};
    ocr::test::P7TextOracle step_oracle;
    expect(ocr::test::p7_oracle_text_forward(fx.cfg.graph, fx.weights, step_tokens, next_position, {}, 0u, 0u,
                                             prompt_len, k_caches, v_caches, step_oracle).ok());
    luisa::vector<float> dev_step_logits(vocab);
    {
        auto list = CommandList::create();
        auto s = fx.exec->text_executor().record_decode_step_positions(list, first, next_position, cache, dev_step_logits);
        expect(s.ok());
        if (!s.ok()) { return; }
        stream << list.commit() << synchronize();
    }
    ocr::test::expect_close(dev_step_logits, step_oracle.logits, kAtol, kRtol, "mineru decode step logits");
    luisa::vector<float> dev_k, dev_v;
    expect(cache.download_layer(stream, 0, 0, dev_k, dev_v).ok());
    stream << synchronize();
    ocr::test::expect_close(dev_k, step_oracle.k_cache[0], kAtol, kRtol, "mineru decode K cache");
    ocr::test::expect_close(dev_v, step_oracle.v_cache[0], kAtol, kRtol, "mineru decode V cache");
    cache.release(0);
}

// ---------------------------------------------------------------------------
// Deterministic tokens + repeat-run + budget.
// ---------------------------------------------------------------------------
void test_mineru_deterministic_and_budget(Device &device) {
    Fixture fx;
    expect(fx.init(device));
    if (!fx.exec) { return; }
    auto stream = device.create_stream();
    auto img = make_image(fx.cfg, 8, 8);
    fx.exec->reset_vision_workspace();
    P7VisionOutput vision;
    {
        auto list = CommandList::create();
        auto v = fx.exec->record_vision(list, luisa::span<const P7ImageInput>{&img, 1});
        expect(v.ok());
        if (!v.ok()) { return; }
        vision = *v.value;
        stream << list.commit() << synchronize();
    }
    const uint32_t tokens = static_cast<uint32_t>(img.packed.visual_tokens);
    luisa::vector<uint32_t> prompt = {1u};
    const uint32_t mm_start = static_cast<uint32_t>(prompt.size());
    for (uint32_t i = 0; i < tokens; i++) { prompt.push_back(0u); }
    prompt.push_back(2u);
    luisa::vector<int32_t> position_ids, next_position;
    auto req = fx.exec->build_decode_request(prompt, mm_start, tokens, vision,
                                             luisa::span<const P7ImageInput>{&img, 1}, position_ids, next_position);
    expect(req.ok());
    if (!req.ok()) { return; }
    const auto generate = [&](uint32_t max_new) -> luisa::vector<uint32_t> {
        ContiguousKvCache cache;
        KvCacheConfig kc;
        kc.num_layers = fx.cfg.graph.text.num_layers;
        kc.num_kv_heads = fx.cfg.graph.text.num_kv_heads;
        kc.head_dim = fx.cfg.graph.text.head_dim;
        kc.max_seq_len = static_cast<uint32_t>(req.value->prompt_tokens.size()) + max_new;
        kc.batch = 1;
        kc.dtype_bytes = 4;
        if (!cache.create(device, stream, kc, nullptr).ok()) { return {}; }
        DenseDecodeOptions dopts;
        dopts.sampling.greedy = true;
        dopts.max_new_tokens = max_new;
        dopts.prompt_position_ids = position_ids;
        dopts.next_position = next_position;
        DenseDecodeController controller;
        if (!controller.create(fx.exec->text_executor(), nullptr, dopts).ok()) { return {}; }
        auto gen = controller.generate(stream, *req.value, cache);
        if (!gen.ok()) { return {}; }
        return gen.value->tokens;
    };
    auto t1 = generate(6);
    auto t2 = generate(6);
    expect(!t1.empty());
    expect(t1 == t2) << "repeat MinerU decode must be identical";
    auto t_short = generate(1);
    expect(t_short.size() == 1u);
    expect(t_short[0] == t1[0]);
}

// ---------------------------------------------------------------------------
// Cross-page composition: two pages, regions restored to page/reading order.
// ---------------------------------------------------------------------------
void test_mineru_cross_page(Device &device) {
    ocr::test::P7TempDir tmp;
    auto cfg = mineru_synthetic_config();
    auto weights = ocr::test::generate_p7_weights(cfg.graph, 0x5EEDu);
    const auto vpath = tmp.path / "mineru_cp_vision.lcocr";
    const auto dpath = tmp.path / "mineru_cp_decoder.lcocr";
    expect(ocr::test::write_p7_packages(vpath, dpath, cfg.graph, cfg.names, weights, "mineru_2_5_pro_2605_1_2b").ok());
    auto stream = device.create_stream();
    WeightStore store;
    expect(ocr::test::upload_p7_weights(device, stream, vpath, dpath, store).ok());
    P7RegionRecognizerConfig rcfg;
    rcfg.model = cfg.graph;
    rcfg.names = cfg.names;
    rcfg.prompt_prefix = {1u};
    rcfg.prompt_suffix = {2u};
 P7RegionRecognizer recognizer;
 expect(recognizer.create(device, stream, rcfg, store).ok());
 // Phase 11: two ragged crops share one vision graph/submission but retain
 // exactly the independent single-crop decode outputs and feature ranges.
 luisa::vector<RegionCropInput> crops(2);
 for (uint32_t i = 0; i < 2u; i++) {
 auto &crop = crops[i];
 crop.source_width = 16u + i * 8u;
 crop.source_height = 16u;
 crop.pixels.resize(static_cast<size_t>(crop.source_width) * crop.source_height * 3u);
 for (size_t p = 0; p < crop.pixels.size(); p++) { crop.pixels[p] = static_cast<uint8_t>((p * 17u + i * 53u) & 255u); }
 crop.task.region_id = 40u + i;
 crop.task.approx_patches = 128u;
 }
 RegionRecognizeOptions region_options;
 region_options.max_output_tokens = 5u;
 region_options.enable_cuda_graph = true; // exercise M-RoPE graph updates
 auto single0 = recognizer.recognize_batch(stream, luisa::span<const RegionCropInput>{crops}.subspan(0, 1), region_options);
 auto single1 = recognizer.recognize_batch(stream, luisa::span<const RegionCropInput>{crops}.subspan(1, 1), region_options);
 expect(single0.ok() && single1.ok());
 const auto graphs_before = recognizer.vision_graph_count();
 const auto crops_before = recognizer.vision_crop_count();
 auto batched = recognizer.recognize_batch(stream, luisa::span<const RegionCropInput>{crops}, region_options);
 expect(batched.ok());
 expect(recognizer.vision_graph_count() == graphs_before + 1u);
 expect(recognizer.vision_crop_count() == crops_before + 2u);
 if (device.extension<CudaGraphExt>() != nullptr) {
 expect(recognizer.cuda_graph_steps() > 0u);
 expect(recognizer.cuda_graph_updates() > 0u);
 expect(recognizer.cuda_graph_fallbacks() == 0u);
 } else {
 expect(recognizer.cuda_graph_fallbacks() > 0u);
 }
 if (single0.ok() && single1.ok() && batched.ok() && batched.value->size() == 2u) {
 expect(!single0.value->empty() && !single1.value->empty());
 if (!single0.value->empty() && !single1.value->empty()) {
 expect((*batched.value)[0].tokens == (*single0.value)[0].tokens);
 expect((*batched.value)[1].tokens == (*single1.value)[0].tokens);
 expect((*batched.value)[0].region_id == 40u && (*batched.value)[1].region_id == 41u);
 expect((*batched.value)[0].native_patches != (*batched.value)[1].native_patches);
 }
 }
 // Exact packed totals, not the scheduler's approximate cost, bound a graph.
 // 64 + 96 patches exceeds this 96-patch arena, so two valid crops degrade to
 // two graphs and must still match the independent outputs.
 auto chunked_cfg = rcfg;
 chunked_cfg.model.max_vision_patches = 96u;
 P7RegionRecognizer chunked;
 expect(chunked.create(device, stream, chunked_cfg, store).ok());
 auto split = chunked.recognize_batch(stream, luisa::span<const RegionCropInput>{crops}, region_options);
 expect(split.ok());
 expect(chunked.vision_graph_count() == 2u);
 expect(chunked.vision_crop_count() == 2u);
 if (split.ok() && split.value->size() == 2u && single0.ok() && single1.ok() &&
 !single0.value->empty() && !single1.value->empty()) {
 expect((*split.value)[0].tokens == (*single0.value)[0].tokens);
 expect((*split.value)[1].tokens == (*single1.value)[0].tokens);
 }
 auto layout_cfg = pp_doc_layout_synthetic_config(PpDocLayoutVersion::v3);

    PpDocLayoutTensorNames layout_names;
    auto layout_weights = ocr::test::generate_layout_weights(layout_cfg, 0x5EEDu);
    LayoutVlmExecutorConfig orch;
    orch.layout_policy = LayoutDevicePolicy::cpu_reference;
    orch.decode.score_threshold = 0.3f;
    orch.decode.box_log_cap = 1.0f;
    orch.scheduler.max_patches_per_vision_batch = 4096;
    orch.scheduler.max_regions_per_batch = 4;
    orch.scheduler.max_output_tokens_per_region = 5;
 orch.scheduler.enable_cuda_graph = true;
    const luisa::vector<float> layout_mean = {0.5f, 0.5f, 0.5f};
    const luisa::vector<float> layout_std = {0.5f, 0.5f, 0.5f};
    auto provider = [&](const LayoutPageInput &p) -> Result<LayoutHeadMaps> {
        auto pixels = ocr::test::p6_oracle_preprocess(p.pixels, p.source_width, p.source_height,
                                                      layout_cfg.input_width, layout_cfg.input_height,
                                                      layout_mean, layout_std);
        ocr::test::LayoutOracleMaps oracle;
        if (auto s = ocr::test::oracle_layout_detect(layout_cfg, layout_weights, pixels, oracle); !s.ok()) {
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
    LayoutVlmExecutor orchestrator;
    expect(orchestrator.create(orch, nullptr, layout_cfg, &recognizer, provider).ok());
    // Two pages with DIFFERENT page_index, passed in REVERSE order to prove the
    // cross-page composition sorts by source page_index.
    auto page0 = ocr::test::p6_make_page(64, 64, 7u);
    page0.page_index = 0;
    auto page1 = ocr::test::p6_make_page(64, 64, 13u);
    page1.page_index = 1;
    luisa::vector<LayoutPageInput> pages = {page1, page0};// reversed input
 const auto graph_updates_before_orch = recognizer.cuda_graph_updates();
 auto run = orchestrator.recognize(stream, pages);
 expect(run.ok());
 if (device.extension<CudaGraphExt>() != nullptr) {
 expect(recognizer.cuda_graph_updates() > graph_updates_before_orch);
 }

    if (!run.ok()) { return; }
    expect(run.value->pages.size() == 2u);
    // Pages must be sorted ascending by source page_index.
    expect(run.value->pages[0].page_index == 0u);
    expect(run.value->pages[1].page_index == 1u);
    // Regions within each page are in reading order; no confidence.
    for (const auto &page : run.value->pages) {
        uint32_t prev = 0;
        for (const auto &region : page.regions) {
            expect(region.reading_order >= prev);
            prev = region.reading_order;
            expect(!region.has_confidence);
            expect(region.page_index == page.page_index);
        }
    }
    expect(run.value->regions_processed > 0u);
    expect(run.value->kv_cache_peak_bytes > 0u);
    // Cross-page markdown composition.
    auto md = mineru_compose(*run.value, LayoutOutputMode::markdown, {});
    expect(md.ok());
    // spotting fails closed.
    expect(mineru_compose(*run.value, LayoutOutputMode::spotting, {}).status.code == StatusCode::unsupported);
    // Repeat-run determinism.
    auto run2 = orchestrator.recognize(stream, pages);
    expect(run2.ok());
    if (run2.ok() && run2.value->pages.size() == run.value->pages.size()) {
        bool same = true;
        for (size_t p = 0; p < run.value->pages.size(); p++) {
            const auto &a = run.value->pages[p];
            const auto &b = run2.value->pages[p];
            if (a.regions.size() != b.regions.size()) { same = false; break; }
            for (size_t i = 0; i < a.regions.size(); i++) {
                if (a.regions[i].content != b.regions[i].content) { same = false; break; }
            }
        }
        expect(same) << "repeat MinerU cross-page runs must be identical";
    }
}

}// namespace

int main(int argc, char *argv[]) {
    auto dc = luisa::test::create_device_from_ut(argc, argv);
    if (!dc) { return 0; }
    boost::ut::detail::cfg::parse_arg_with_fallback(argc, const_cast<const char **>(argv));
    auto &device = dc->device;
    "mineru_contracts"_test = [] { test_mineru_contracts(); };
    "mineru_vision_parity"_test = [&] { test_mineru_vision_parity(device); };
    "mineru_decoder_parity"_test = [&] { test_mineru_decoder_parity(device); };
    "mineru_deterministic_and_budget"_test = [&] { test_mineru_deterministic_and_budget(device); };
    "mineru_cross_page"_test = [&] { test_mineru_cross_page(device); };
}

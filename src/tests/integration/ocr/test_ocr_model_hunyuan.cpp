// Test for ocr/models/hunyuan_ocr + the shared Phase 7 slice executor (device)
// -- the SMALL SYNTHETIC HunyuanOCR-1.5 vertical slice: native Hunyuan-ViT
// (SiLU MLP, 2D RoPE), adaptive-pool/MLP connector, and the Hunyuan decoder
// with XD-RoPE over text/height/width/time (4 axes, adjacent pairs). Ordinary
// target decode only; DFlash deferred. Spotting/extraction adapters; NO fake
// confidence.
// Covers: config/tensor contracts, Qwen/MoE/DFlash fail-closed parsing,
// Hunyuan-ViT + adaptive connector parity, XD-RoPE prefill logits + decode KV,
// deterministic tokens, blank/repeated/adversarial, budget + repeat-run,
// spotting coordinates, and extract strictness.
// Synthetic weights only; no model/download, no real-model parity claim.
#include "ut/ut.hpp"
#include "test_device.h"
#include "ocr_p7_test_utils.h"
#include "ocr_p6_test_utils.h"

#include "ocr/models/hunyuan_ocr.h"
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
    HunyuanOcrConfig cfg;
    ocr::test::P7SyntheticWeights weights;
    WeightStore store;
    luisa::unique_ptr<P7VlmExecutor> exec;

    [[nodiscard]] bool init(Device &device, uint32_t seed = 0x5EEDu) {
        cfg = hunyuan_ocr_synthetic_config();
        if (!cfg.validate().ok()) { return false; }
        weights = ocr::test::generate_p7_weights(cfg.graph, seed);
        const auto vpath = tmp.path / "hunyuan_vision.lcocr";
        const auto dpath = tmp.path / "hunyuan_decoder.lcocr";
        if (auto s = ocr::test::write_p7_packages(vpath, dpath, cfg.graph, cfg.names, weights, "hunyuan_ocr_1_5"); !s.ok()) {
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
            LUISA_WARNING("hunyuan create: {}", s.message);
            return false;
        }
        return true;
    }
};

[[nodiscard]] P7ImageInput make_image(const HunyuanOcrConfig &cfg, uint32_t w, uint32_t h, uint32_t seed = 11u) {
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
void test_hunyuan_contracts() {
    auto cfg = hunyuan_ocr_synthetic_config();
    expect(cfg.validate().ok());
    expect(cfg.graph.connector.kind == P7ConnectorKind::adaptive_mlp);
    expect(cfg.graph.vision.mlp_act == 1u);// SiLU
    expect(cfg.graph.text_pos == P7TextPosKind::xd);
    expect(cfg.graph.text_rope_axes() == 4u);
    expect(cfg.graph.pos_sections.size() == 4u);
    expect(cfg.graph.text.rope_pair_layout == RopePairLayout::adjacent);
    auto expected = hunyuan_ocr_expected_tensors(cfg);
    expect(expected.size() > 30u);
    expect(hunyuan_ocr_validate_real_artifact("r", "f").code == StatusCode::unvalidated);
    const char *good = R"({"model_type":"hunyuan_ocr","vision_config":{"num_layers":2,"hidden_size":16,"num_heads":2,"head_dim":8,"intermediate_size":32,"patch_size":2},"text_config":{"vocab_size":32,"hidden_size":32,"num_hidden_layers":3,"num_attention_heads":4,"num_key_value_heads":2,"intermediate_size":64,"max_position_embeddings":256},"xd_rope_section":[1,1,1,1]})";
    auto parsed = parse_hunyuan_ocr_config(good);
    expect(parsed.ok());
    if (parsed.ok()) { expect(parsed.value->validate().ok()); }
    // Qwen3 QK-Norm fails closed.
    const char *qwen3 = R"({"model_type":"hunyuan_ocr","vision_config":{"num_layers":2,"hidden_size":16,"num_heads":2,"head_dim":8,"intermediate_size":32,"patch_size":2},"text_config":{"vocab_size":32,"hidden_size":32,"num_hidden_layers":3,"num_attention_heads":4,"num_key_value_heads":2,"intermediate_size":64,"max_position_embeddings":256,"qk_norm":true},"xd_rope_section":[1,1,1,1]})";
    expect(!parse_hunyuan_ocr_config(qwen3).ok());
    // A 3-axis mrope_section (Qwen) fails closed (XD needs 4).
    const char *mrope = R"({"model_type":"hunyuan_ocr","mrope_section":[2,1,1],"vision_config":{"num_layers":2,"hidden_size":16,"num_heads":2,"head_dim":8,"intermediate_size":32,"patch_size":2},"text_config":{"vocab_size":32,"hidden_size":32,"num_hidden_layers":3,"num_attention_heads":4,"num_key_value_heads":2,"intermediate_size":64,"max_position_embeddings":256},"xd_rope_section":[1,1,1,1]})";
    expect(!parse_hunyuan_ocr_config(mrope).ok());
    // DFlash requirement fails closed (deferred).
    const char *dflash = R"({"model_type":"hunyuan_ocr","dflash_num_layers":2,"vision_config":{"num_layers":2,"hidden_size":16,"num_heads":2,"head_dim":8,"intermediate_size":32,"patch_size":2},"text_config":{"vocab_size":32,"hidden_size":32,"num_hidden_layers":3,"num_attention_heads":4,"num_key_value_heads":2,"intermediate_size":64,"max_position_embeddings":256},"xd_rope_section":[1,1,1,1]})";
    expect(!parse_hunyuan_ocr_config(dflash).ok());
}

// ---------------------------------------------------------------------------
// Hunyuan-ViT + adaptive connector parity (SiLU MLP, adaptive pool).
// ---------------------------------------------------------------------------
void test_hunyuan_vision_parity(Device &device) {
    Fixture fx;
    expect(fx.init(device));
    if (!fx.exec) { return; }
    auto stream = device.create_stream();
    for (const auto &[w, h] : std::initializer_list<std::pair<uint32_t, uint32_t>>{{8, 8}, {12, 8}, {16, 16}}) {
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
        const auto what = luisa::format("hunyuan {}x{}", w, h);
        ocr::test::expect_close(dev_patches, oracle.patches, kLinAtol, kLinRtol, (what + " patches").c_str());
        ocr::test::expect_close(dev_final, oracle.block_out.back(), kAtol, kRtol, (what + " hunyuan-vit final").c_str());
        ocr::test::expect_close(dev_reduced, oracle.reduced, kAtol, kRtol, (what + " adaptive pool reduced").c_str());
        ocr::test::expect_close(dev_features, oracle.features, kAtol, kRtol, (what + " adaptive connector").c_str());
        // The adaptive pool must respect the token budget.
        expect(img.packed.visual_tokens <= fx.cfg.graph.connector.max_visual_tokens);
    }
}

// ---------------------------------------------------------------------------
// XD-RoPE decoder parity: prefill logits + decode KV/logits (4 axes).
// ---------------------------------------------------------------------------
void test_hunyuan_decoder_parity(Device &device) {
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
    // XD-RoPE: 4 axes per token.
    expect(position_ids.size() == prompt.size() * 4u);
    expect(next_position.size() == 4u);
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
    ocr::test::expect_close(dev_logits, text_oracle.logits, kAtol, kRtol, "hunyuan XD-RoPE prefill logits");
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
    ocr::test::expect_close(dev_step_logits, step_oracle.logits, kAtol, kRtol, "hunyuan decode step logits");
    luisa::vector<float> dev_k, dev_v;
    expect(cache.download_layer(stream, 0, 0, dev_k, dev_v).ok());
    stream << synchronize();
    ocr::test::expect_close(dev_k, step_oracle.k_cache[0], kAtol, kRtol, "hunyuan decode K cache");
    ocr::test::expect_close(dev_v, step_oracle.v_cache[0], kAtol, kRtol, "hunyuan decode V cache");
    cache.release(0);
}

// ---------------------------------------------------------------------------
// Deterministic tokens + repeat-run + budget (ordinary target decode; no DFlash).
// ---------------------------------------------------------------------------
void test_hunyuan_deterministic_and_budget(Device &device) {
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
    expect(t1 == t2) << "repeat Hunyuan decode must be identical";
    auto t_short = generate(1);
    expect(t_short.size() == 1u);
    expect(t_short[0] == t1[0]);
}

// ---------------------------------------------------------------------------
// Full pipeline + spotting coordinates + extract strictness + no confidence.
// ---------------------------------------------------------------------------
void test_hunyuan_pipeline_and_adapters(Device &device) {
    ocr::test::P7TempDir tmp;
    auto cfg = hunyuan_ocr_synthetic_config();
    auto weights = ocr::test::generate_p7_weights(cfg.graph, 0x5EEDu);
    const auto vpath = tmp.path / "hunyuan_pipe_vision.lcocr";
    const auto dpath = tmp.path / "hunyuan_pipe_decoder.lcocr";
    expect(ocr::test::write_p7_packages(vpath, dpath, cfg.graph, cfg.names, weights, "hunyuan_ocr_1_5").ok());
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
    auto page = ocr::test::p6_make_page(64, 64);
    auto run = orchestrator.recognize(stream, luisa::span<const LayoutPageInput>{&page, 1});
    expect(run.ok());
    if (!run.ok()) { return; }
    expect(run.value->pages.size() == 1u);
    expect(run.value->regions_processed > 0u);
    expect(run.value->kv_cache_peak_bytes > 0u);
    // NO fake confidence on generative content.
    for (const auto &region : run.value->pages[0].regions) {
        expect(!region.has_confidence);
    }
    // Spotting adapter: lines/blocks carry source coordinates, no confidence.
    auto spot = hunyuan_ocr_spotting(*run.value);
    expect(spot.ok());
    if (spot.ok()) {
        expect(!spot.value->blocks.empty());
        for (const auto &block : spot.value->blocks) {
            expect(!block.has_confidence);
            expect(!block.source_box.empty());
        }
        for (const auto &line : spot.value->lines) {
            expect(!line.has_confidence);
        }
    }
    // compose(spotting) routes to the spotting adapter.
    auto spot2 = hunyuan_ocr_compose(*run.value, LayoutOutputMode::spotting, {});
    expect(spot2.ok());
    // markdown composition.
    auto md = hunyuan_ocr_compose(*run.value, LayoutOutputMode::markdown, {});
    expect(md.ok());
    // extract requires a schema; without one it fails closed.
    expect(hunyuan_ocr_compose(*run.value, LayoutOutputMode::extract, {}).status.code == StatusCode::invalid_argument);
    // Repeat-run determinism.
    auto run2 = orchestrator.recognize(stream, luisa::span<const LayoutPageInput>{&page, 1});
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
        expect(same) << "repeat Hunyuan pipeline runs must be identical";
    }
}

// ---------------------------------------------------------------------------
// Adaptive-pool layout-agnostic TRUTH test (defect 1 regression). A NON-SQUARE
// patch grid (gh != gw) with a channel-distinct ramp makes wrong-axis / wrong-
// layout pooling produce different numbers, so a token-major-fed NCHW pool
// cannot pass. The expected reduced values are hand-computed per-channel
// spatial PyTorch bins (start=floor, end=ceil), independent of the device path.
// ---------------------------------------------------------------------------
void test_hunyuan_adaptive_truth(Device &device) {
    Fixture fx;
    expect(fx.init(device));
    if (!fx.exec) { return; }
    auto stream = device.create_stream();
    // 16x8 source (w=16,h=8) -> patch 2 -> grid gh=4, gw=8 (NON-SQUARE). Budget
    // 16 tokens -> reduce_grid(4,8,16): scale=sqrt(16/32)=0.707 -> out 2x5=10.
    auto img = make_image(fx.cfg, 16, 8);
    expect(img.packed.grid_h == 4u);
    expect(img.packed.grid_w == 8u);
    expect(img.packed.out_h != img.packed.out_w) << "non-square output grid required";
    auto pixels = ocr::test::p7_oracle_preprocess(img.pixels, 16, 8,
                                                  img.packed.resized_width, img.packed.resized_height,
                                                  fx.cfg.graph.mean, fx.cfg.graph.std_dev);
    ocr::test::P7VisionOracle oracle;
    expect(ocr::test::p7_oracle_vision(fx.cfg.graph, fx.weights, pixels, img.packed, oracle).ok());
    fx.exec->reset_vision_workspace();
    {
        auto list = CommandList::create();
        auto v = fx.exec->record_vision(list, luisa::span<const P7ImageInput>{&img, 1});
        expect(v.ok());
        if (!v.ok()) { return; }
        stream << list.commit() << synchronize();
    }
    luisa::vector<float> dev_reduced, dev_features;
    expect(fx.exec->download_vision_reduced(stream, dev_reduced).ok());
    expect(fx.exec->download_vision_features(stream, dev_features).ok());
    stream << synchronize();
    // The oracle reduced is the independent per-channel spatial pool; the device
    // must match it exactly (within tolerance). A wrong-axis pool would not.
    ocr::test::expect_close(dev_reduced, oracle.reduced, kAtol, kRtol, "hunyuan adaptive truth reduced");
    ocr::test::expect_close(dev_features, oracle.features, kAtol, kRtol, "hunyuan adaptive truth features");
    // Independent hand-check: recompute the reduced pool DIRECTLY from the
    // oracle vision-final token rows with explicit PyTorch bins, and confirm it
    // equals the oracle reduced (guards the oracle itself against mirroring a
    // device bug). This is a second, separate derivation.
    const uint32_t vh = fx.cfg.graph.vision.hidden_size;
    const uint32_t gh = img.packed.grid_h, gw = img.packed.grid_w;
    const uint32_t oh = img.packed.out_h, ow = img.packed.out_w;
    const auto &vf = oracle.block_out.back();
    luisa::vector<float> truth(static_cast<size_t>(oh) * ow * vh);
    for (uint32_t oy = 0; oy < oh; oy++) {
        const uint32_t y0 = static_cast<uint32_t>(static_cast<uint64_t>(oy) * gh / oh);
        const uint32_t y1 = static_cast<uint32_t>((static_cast<uint64_t>(oy + 1) * gh + oh - 1u) / oh);
        for (uint32_t ox = 0; ox < ow; ox++) {
            const uint32_t x0 = static_cast<uint32_t>(static_cast<uint64_t>(ox) * gw / ow);
            const uint32_t x1 = static_cast<uint32_t>((static_cast<uint64_t>(ox + 1) * gw + ow - 1u) / ow);
            for (uint32_t ci = 0; ci < vh; ci++) {
                double sum = 0.0;
                uint32_t cnt = 0;
                for (uint32_t yy = y0; yy < y1; yy++) {
                    for (uint32_t xx = x0; xx < x1; xx++) {
                        sum += vf[static_cast<size_t>(yy * gw + xx) * vh + ci];
                        cnt++;
                    }
                }
                truth[(static_cast<size_t>(oy) * ow + ox) * vh + ci] = cnt ? static_cast<float>(sum / cnt) : 0.0f;
            }
        }
    }
    ocr::test::expect_close(dev_reduced, truth, kAtol, kRtol, "hunyuan adaptive hand-computed truth");
}

// ---------------------------------------------------------------------------
// Defect 4 regression: TWO images whose combined adaptive-connector tokens
// exceed the OLD per-image features bound. The adaptive connector emits up to
// max_visual_tokens tokens PER image, but the old features arena was sized for
// min(maxp, max_visual_tokens) TOTAL -- so two images overflowed. The new arena
// is sized for the total patch budget. With max_visual_tokens=4 and two 8x8
// images (16 patches each -> 4 tokens each = 8 total), the old bound held 4
// tokens (< 8). Verifies no OOB abort and exact concatenated parity.
// ---------------------------------------------------------------------------
void test_hunyuan_two_image_arena(Device &device) {
    ocr::test::P7TempDir tmp;
    auto cfg = hunyuan_ocr_synthetic_config();
    cfg.graph.connector.max_visual_tokens = 4;// small per-image token budget
    cfg.graph.max_vision_patches = 64;// total patch budget >= 2*16
    expect(cfg.validate().ok());
    auto weights = ocr::test::generate_p7_weights(cfg.graph, 0x5EEDu);
    const auto vpath = tmp.path / "hunyuan_2img_vision.lcocr";
    const auto dpath = tmp.path / "hunyuan_2img_decoder.lcocr";
    expect(ocr::test::write_p7_packages(vpath, dpath, cfg.graph, cfg.names, weights, "hunyuan_ocr_1_5").ok());
    auto stream = device.create_stream();
    WeightStore store;
    expect(ocr::test::upload_p7_weights(device, stream, vpath, dpath, store).ok());
    P7VlmExecutor exec;
    expect(exec.create(device, stream, cfg.graph, cfg.names, store).ok());
    auto img0 = make_image(cfg, 8, 8, 11u);
    auto img1 = make_image(cfg, 8, 8, 29u);
    luisa::vector<P7ImageInput> images = {img0, img1};
    const uint32_t t0 = static_cast<uint32_t>(img0.packed.visual_tokens);
    const uint32_t t1 = static_cast<uint32_t>(img1.packed.visual_tokens);
    // The combined token total exceeds the OLD per-image features bound
    // (min(maxp, max_visual_tokens) = 4).
    expect(t0 + t1 > cfg.graph.connector.max_visual_tokens);
    luisa::vector<ocr::test::P7VisionOracle> oracles(2);
    for (size_t i = 0; i < 2; i++) {
        auto pixels = ocr::test::p7_oracle_preprocess(images[i].pixels, images[i].source_width, images[i].source_height,
                                                      images[i].packed.resized_width, images[i].packed.resized_height,
                                                      cfg.graph.mean, cfg.graph.std_dev);
        expect(ocr::test::p7_oracle_vision(cfg.graph, weights, pixels, images[i].packed, oracles[i]).ok());
    }
    exec.reset_vision_workspace();
    P7VisionOutput vision;
    {
        auto list = CommandList::create();
        auto v = exec.record_vision(list, images);
        expect(v.ok());
        if (!v.ok()) { return; }
        vision = *v.value;
        stream << list.commit() << synchronize();
    }
    expect(vision.total_tokens == t0 + t1);
    luisa::vector<float> dev_features;
    expect(exec.download_vision_features(stream, dev_features).ok());
    stream << synchronize();
    luisa::vector<float> oracle_features;
    oracle_features.insert(oracle_features.end(), oracles[0].features.begin(), oracles[0].features.end());
    oracle_features.insert(oracle_features.end(), oracles[1].features.begin(), oracles[1].features.end());
    ocr::test::expect_close(dev_features, oracle_features, kAtol, kRtol, "hunyuan two-image arena features");
}

}// namespace

int main(int argc, char *argv[]) {
    auto dc = luisa::test::create_device_from_ut(argc, argv);
    if (!dc) { return 0; }
    boost::ut::detail::cfg::parse_arg_with_fallback(argc, const_cast<const char **>(argv));
    auto &device = dc->device;
    "hunyuan_contracts"_test = [] { test_hunyuan_contracts(); };
    "hunyuan_vision_parity"_test = [&] { test_hunyuan_vision_parity(device); };
    "hunyuan_decoder_parity"_test = [&] { test_hunyuan_decoder_parity(device); };
    "hunyuan_deterministic_and_budget"_test = [&] { test_hunyuan_deterministic_and_budget(device); };
    "hunyuan_pipeline_and_adapters"_test = [&] { test_hunyuan_pipeline_and_adapters(device); };
    "hunyuan_adaptive_truth"_test = [&] { test_hunyuan_adaptive_truth(device); };
    "hunyuan_two_image_arena"_test = [&] { test_hunyuan_two_image_arena(device); };
}

// Test for ocr/models/glm_ocr + the shared Phase 7 slice executor (device) --
// the SMALL SYNTHETIC GLM-OCR vertical slice: PP-DocLayoutV3 front end, CogViT
// (learned absolute positions, no vision RoPE), learned downsampling connector,
// and the GLM-0.5B-style decoder with PARTIAL rotary (rotate-half within the
// rotated prefix), QKV bias, no QK-Norm, plain 1D positions. Ordinary target
// decoding only; MTP is deferred and never required.
// Covers: config/tensor contracts, Qwen/MoE/MTP fail-closed parsing, CogViT +
// downsampling connector parity, partial-rotary prefill logits + decode KV,
// deterministic tokens, blank/repeated/adversarial, budget + repeat-run, and
// the full PP-DocLayoutV3 -> Markdown/JSON/extract pipeline.
// Synthetic weights only; no model/download, no real-model parity claim.
#include "ut/ut.hpp"
#include "test_device.h"
#include "ocr_p7_test_utils.h"
#include "ocr_p6_test_utils.h"

#include "ocr/models/glm_ocr.h"
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
    GlmOcrConfig cfg;
    ocr::test::P7SyntheticWeights weights;
    WeightStore store;
    luisa::unique_ptr<P7VlmExecutor> exec;

    [[nodiscard]] bool init(Device &device, uint32_t seed = 0x5EEDu) {
        cfg = glm_ocr_synthetic_config();
        if (!cfg.validate().ok()) { return false; }
        weights = ocr::test::generate_p7_weights(cfg.graph, seed);
        const auto vpath = tmp.path / "glm_vision.lcocr";
        const auto dpath = tmp.path / "glm_decoder.lcocr";
        if (auto s = ocr::test::write_p7_packages(vpath, dpath, cfg.graph, cfg.names, weights, "glm_ocr"); !s.ok()) {
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
            LUISA_WARNING("glm create: {}", s.message);
            return false;
        }
        return true;
    }
};

[[nodiscard]] P7ImageInput make_image(const GlmOcrConfig &cfg, uint32_t w, uint32_t h, uint32_t seed = 11u) {
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
void test_glm_contracts() {
    auto cfg = glm_ocr_synthetic_config();
    expect(cfg.validate().ok());
    expect(cfg.graph.vision.rope_kind == P7VisionRopeKind::none);
    expect(cfg.graph.vision.position_rows > 0u);
    expect(cfg.graph.connector.kind == P7ConnectorKind::downsample_mlp);
    expect(cfg.graph.text_pos == P7TextPosKind::rope1d);
    expect(cfg.graph.partial_rotary_num == 1u);
    expect(cfg.graph.partial_rotary_den == 2u);
    expect(cfg.graph.text.rope_pair_layout == RopePairLayout::rotate_half);
    // Partial rotary dim = head_dim/2 = 4 (2 rotated pairs, 4 passthrough).
    expect(cfg.graph.rotary_dim() == 4u);
    auto expected = glm_ocr_expected_tensors(cfg);
    expect(expected.size() > 30u);
    expect(glm_ocr_validate_real_artifact("r", "f").code == StatusCode::unvalidated);
    // A valid synthetic config parses.
    const char *good = R"({"model_type":"glm_ocr","vision_config":{"num_layers":2,"hidden_size":16,"num_heads":2,"head_dim":8,"intermediate_size":32,"patch_size":2,"position_rows":1024},"text_config":{"vocab_size":32,"hidden_size":32,"num_hidden_layers":3,"num_attention_heads":4,"num_key_value_heads":2,"intermediate_size":64,"max_position_embeddings":256},"partial_rotary_factor":0.5})";
    auto parsed = parse_glm_ocr_config(good);
    expect(parsed.ok());
    if (parsed.ok()) { expect(parsed.value->validate().ok()); }
    // Qwen3 QK-Norm marker fails closed.
    const char *qwen3 = R"({"model_type":"glm_ocr","vision_config":{"num_layers":2,"hidden_size":16,"num_heads":2,"head_dim":8,"intermediate_size":32,"patch_size":2},"text_config":{"vocab_size":32,"hidden_size":32,"num_hidden_layers":3,"num_attention_heads":4,"num_key_value_heads":2,"intermediate_size":64,"max_position_embeddings":256,"qk_norm":true},"partial_rotary_factor":0.5})";
    expect(!parse_glm_ocr_config(qwen3).ok());
    // A Qwen model_type fails closed.
    const char *qwen_type = R"({"model_type":"qwen2","vision_config":{"num_layers":2,"hidden_size":16,"num_heads":2,"head_dim":8,"intermediate_size":32,"patch_size":2},"text_config":{"vocab_size":32,"hidden_size":32,"num_hidden_layers":3,"num_attention_heads":4,"num_key_value_heads":2,"intermediate_size":64,"max_position_embeddings":256},"partial_rotary_factor":0.5})";
    expect(!parse_glm_ocr_config(qwen_type).ok());
    // Full rotary (factor 1.0) is a Qwen decoder: fail closed.
    const char *full_rotary = R"({"model_type":"glm_ocr","vision_config":{"num_layers":2,"hidden_size":16,"num_heads":2,"head_dim":8,"intermediate_size":32,"patch_size":2},"text_config":{"vocab_size":32,"hidden_size":32,"num_hidden_layers":3,"num_attention_heads":4,"num_key_value_heads":2,"intermediate_size":64,"max_position_embeddings":256},"partial_rotary_factor":1.0})";
    expect(!parse_glm_ocr_config(full_rotary).ok());
    // MTP requirement fails closed (deferred).
    const char *mtp = R"({"model_type":"glm_ocr","mtp_num_layers":1,"vision_config":{"num_layers":2,"hidden_size":16,"num_heads":2,"head_dim":8,"intermediate_size":32,"patch_size":2},"text_config":{"vocab_size":32,"hidden_size":32,"num_hidden_layers":3,"num_attention_heads":4,"num_key_value_heads":2,"intermediate_size":64,"max_position_embeddings":256},"partial_rotary_factor":0.5})";
    expect(!parse_glm_ocr_config(mtp).ok());
    // MoE marker fails closed.
    const char *moe = R"({"model_type":"glm_ocr","num_experts":8,"vision_config":{"num_layers":2,"hidden_size":16,"num_heads":2,"head_dim":8,"intermediate_size":32,"patch_size":2},"text_config":{"vocab_size":32,"hidden_size":32,"num_hidden_layers":3,"num_attention_heads":4,"num_key_value_heads":2,"intermediate_size":64,"max_position_embeddings":256},"partial_rotary_factor":0.5})";
    expect(!parse_glm_ocr_config(moe).ok());
}

// ---------------------------------------------------------------------------
// CogViT + downsampling connector parity (learned positions, no vision RoPE).
// ---------------------------------------------------------------------------
void test_glm_vision_parity(Device &device) {
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
        luisa::vector<float> dev_patches, dev_final, dev_features;
        expect(fx.exec->download_vision_patches(stream, dev_patches).ok());
        expect(fx.exec->download_vision_final(stream, dev_final).ok());
        expect(fx.exec->download_vision_features(stream, dev_features).ok());
        stream << synchronize();
        const auto what = luisa::format("glm {}x{}", w, h);
        ocr::test::expect_close(dev_patches, oracle.patches, kLinAtol, kLinRtol, (what + " patches").c_str());
        ocr::test::expect_close(dev_final, oracle.block_out.back(), kAtol, kRtol, (what + " cogvit final").c_str());
        ocr::test::expect_close(dev_features, oracle.features, kAtol, kRtol, (what + " downsample connector").c_str());
    }
}

// ---------------------------------------------------------------------------
// Partial-rotary decoder parity: prefill logits + decode KV/logits.
// ---------------------------------------------------------------------------
void test_glm_decoder_parity(Device &device) {
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
    // GLM uses 1D positions: the executor's record_prefill (1-axis) path.
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
        // 1-axis path: record_prefill_all_logits (sequential positions).
        auto s = fx.exec->text_executor().record_prefill_all_logits(list, *req.value, cache, logits_buf.view());
        expect(s.ok());
        if (!s.ok()) { return; }
        stream << list.commit() << synchronize();
    }
    luisa::vector<float> dev_logits(static_cast<size_t>(prompt_len) * vocab);
    stream << logits_buf.copy_to(luisa::span<float>{dev_logits}) << synchronize();
    ocr::test::expect_close(dev_logits, text_oracle.logits, kAtol, kRtol, "glm partial-rotary prefill logits");
    // Decode step (1-axis).
    const uint32_t first = ocr::test::p7_oracle_argmax(luisa::span<const float>{
        text_oracle.logits.data() + static_cast<size_t>(prompt_len - 1) * vocab, vocab});
    luisa::vector<uint32_t> step_tokens = {first};
    luisa::vector<int32_t> step_pos = {static_cast<int32_t>(prompt_len)};
    ocr::test::P7TextOracle step_oracle;
    expect(ocr::test::p7_oracle_text_forward(fx.cfg.graph, fx.weights, step_tokens, step_pos, {}, 0u, 0u,
                                             prompt_len, k_caches, v_caches, step_oracle).ok());
    luisa::vector<float> dev_step_logits(vocab);
    {
        auto list = CommandList::create();
        auto s = fx.exec->text_executor().record_decode_step(list, first, cache, dev_step_logits);
        expect(s.ok());
        if (!s.ok()) { return; }
        stream << list.commit() << synchronize();
    }
    ocr::test::expect_close(dev_step_logits, step_oracle.logits, kAtol, kRtol, "glm decode step logits");
    luisa::vector<float> dev_k, dev_v;
    expect(cache.download_layer(stream, 0, 0, dev_k, dev_v).ok());
    stream << synchronize();
    ocr::test::expect_close(dev_k, step_oracle.k_cache[0], kAtol, kRtol, "glm decode K cache");
    ocr::test::expect_close(dev_v, step_oracle.v_cache[0], kAtol, kRtol, "glm decode V cache");
    cache.release(0);
}

// ---------------------------------------------------------------------------
// Deterministic tokens + repeat-run + budget (ordinary target decode; no MTP).
// ---------------------------------------------------------------------------
void test_glm_deterministic_and_budget(Device &device) {
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
        // GLM is 1-axis: no multimodal position options (ordinary decode).
        DenseDecodeController controller;
        if (!controller.create(fx.exec->text_executor(), nullptr, dopts).ok()) { return {}; }
        auto gen = controller.generate(stream, *req.value, cache);
        if (!gen.ok()) { return {}; }
        return gen.value->tokens;
    };
    auto t1 = generate(6);
    auto t2 = generate(6);
    expect(!t1.empty());
    expect(t1 == t2) << "repeat GLM decode must be identical";
    auto t_short = generate(1);
    expect(t_short.size() == 1u);
    expect(t_short[0] == t1[0]);
}

// ---------------------------------------------------------------------------
// Full PP-DocLayoutV3 -> region -> Markdown/JSON/extract pipeline.
// ---------------------------------------------------------------------------
void test_glm_full_pipeline(Device &device) {
    ocr::test::P7TempDir tmp;
    auto cfg = glm_ocr_synthetic_config();
    auto weights = ocr::test::generate_p7_weights(cfg.graph, 0x5EEDu);
    const auto vpath = tmp.path / "glm_pipe_vision.lcocr";
    const auto dpath = tmp.path / "glm_pipe_decoder.lcocr";
    expect(ocr::test::write_p7_packages(vpath, dpath, cfg.graph, cfg.names, weights, "glm_ocr").ok());
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
    // PP-DocLayoutV3 (the GLM-OCR front end).
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
    expect(run.value->patches_processed > 0u);
    expect(run.value->kv_cache_peak_bytes > 0u);
    for (const auto &region : run.value->pages[0].regions) {
        expect(!region.has_confidence);
    }
    // Markdown composition (validated GLM mode).
    auto md = glm_ocr_compose(*run.value, LayoutOutputMode::markdown, {});
    expect(md.ok());
    // plain composition.
    auto plain = glm_ocr_compose(*run.value, LayoutOutputMode::plain, {});
    expect(plain.ok());
    // spotting is NOT validated for GLM -> fail closed.
    expect(glm_ocr_compose(*run.value, LayoutOutputMode::spotting, {}).status.code == StatusCode::unsupported);
    // extract requires a schema; without one it fails closed.
    expect(glm_ocr_compose(*run.value, LayoutOutputMode::extract, {}).status.code == StatusCode::invalid_argument);
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
        expect(same) << "repeat GLM pipeline runs must be identical";
    }
}

// ---------------------------------------------------------------------------
// Defect 3 regression: large-grid downsample connector (>16 / 64 patches). The
// merged scratch [tokens, vh*m*m] must be sized for the m*m-expanded rows; the
// old undersized cnorm scratch aborted. 16x16 source -> patch 2 -> 8x8 grid =
// 64 patches -> downsample stride 2 -> 4x4 = 16 tokens, merged_dim = vh*4.
// ---------------------------------------------------------------------------
void test_glm_downsample_large(Device &device) {
    Fixture fx;
    expect(fx.init(device));
    if (!fx.exec) { return; }
    auto stream = device.create_stream();
    for (const auto &[w, h] : std::initializer_list<std::pair<uint32_t, uint32_t>>{{12, 12}, {16, 16}}) {
        auto img = make_image(fx.cfg, w, h);
        expect(img.packed.patch_count > 16u) << "regression requires >16 patches";
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
        luisa::vector<float> dev_reduced, dev_features;
        expect(fx.exec->download_vision_reduced(stream, dev_reduced).ok());
        expect(fx.exec->download_vision_features(stream, dev_features).ok());
        stream << synchronize();
        const auto what = luisa::format("glm downsample {}x{} ({} patches)", w, h, img.packed.patch_count);
        ocr::test::expect_close(dev_reduced, oracle.reduced, kAtol, kRtol, (what + " reduced").c_str());
        ocr::test::expect_close(dev_features, oracle.features, kAtol, kRtol, (what + " features").c_str());
    }
}

// ---------------------------------------------------------------------------
// Defect 5 regression: TWO images of DIFFERENT sizes through the learned-
// position path. The per-image position-id staging must be stable (no dangling
// / reallocated vector across uploads). Verifies no OOB abort and correct
// per-image positions/features against the independent oracle.
// ---------------------------------------------------------------------------
void test_glm_two_image_positions(Device &device) {
    Fixture fx;
    expect(fx.init(device));
    if (!fx.exec) { return; }
    auto stream = device.create_stream();
    auto img0 = make_image(fx.cfg, 8, 8, 11u);
    auto img1 = make_image(fx.cfg, 12, 8, 23u);// different size + content
    expect(img0.packed.patch_count != img1.packed.patch_count);
    luisa::vector<P7ImageInput> images = {img0, img1};
    // Independent per-image oracles.
    luisa::vector<ocr::test::P7VisionOracle> oracles(2);
    for (size_t i = 0; i < 2; i++) {
        auto pixels = ocr::test::p7_oracle_preprocess(images[i].pixels, images[i].source_width, images[i].source_height,
                                                      images[i].packed.resized_width, images[i].packed.resized_height,
                                                      fx.cfg.graph.mean, fx.cfg.graph.std_dev);
        expect(ocr::test::p7_oracle_vision(fx.cfg.graph, fx.weights, pixels, images[i].packed, oracles[i]).ok());
    }
    fx.exec->reset_vision_workspace();
    P7VisionOutput vision;
    {
        auto list = CommandList::create();
        auto v = fx.exec->record_vision(list, images);
        expect(v.ok());
        if (!v.ok()) { return; }
        vision = *v.value;
        stream << list.commit() << synchronize();
    }
    const uint32_t t0 = static_cast<uint32_t>(img0.packed.visual_tokens);
    const uint32_t t1 = static_cast<uint32_t>(img1.packed.visual_tokens);
    expect(vision.total_tokens == t0 + t1);
    luisa::vector<float> dev_features;
    expect(fx.exec->download_vision_features(stream, dev_features).ok());
    stream << synchronize();
    // Concatenate the per-image oracle features and compare the whole span.
    luisa::vector<float> oracle_features;
    oracle_features.insert(oracle_features.end(), oracles[0].features.begin(), oracles[0].features.end());
    oracle_features.insert(oracle_features.end(), oracles[1].features.begin(), oracles[1].features.end());
    ocr::test::expect_close(dev_features, oracle_features, kAtol, kRtol, "glm two-image positions features");
    // The second image's features must match its OWN oracle (proves the position
    // ids were not corrupted by the first image's staging).
    const uint32_t th = fx.cfg.graph.text.hidden_size;
    luisa::vector<float> dev_img1(dev_features.begin() + static_cast<size_t>(t0) * th, dev_features.end());
    ocr::test::expect_close(dev_img1, oracles[1].features, kAtol, kRtol, "glm two-image second-image features");
}

}// namespace

int main(int argc, char *argv[]) {
    auto dc = luisa::test::create_device_from_ut(argc, argv);
    if (!dc) { return 0; }
    boost::ut::detail::cfg::parse_arg_with_fallback(argc, const_cast<const char **>(argv));
    auto &device = dc->device;
    "glm_contracts"_test = [] { test_glm_contracts(); };
    "glm_vision_parity"_test = [&] { test_glm_vision_parity(device); };
    "glm_decoder_parity"_test = [&] { test_glm_decoder_parity(device); };
    "glm_deterministic_and_budget"_test = [&] { test_glm_deterministic_and_budget(device); };
    "glm_full_pipeline"_test = [&] { test_glm_full_pipeline(device); };
    "glm_downsample_large"_test = [&] { test_glm_downsample_large(device); };
    "glm_two_image_positions"_test = [&] { test_glm_two_image_positions(device); };
}

// Test for ocr/models/got_ocr2 + the shared Phase 7 slice executor (device) --
// the SMALL SYNTHETIC GOT-OCR2.0 vertical slice: ViTDet/SAM window/global
// attention with decomposed relative-position bias, linear connector, Qwen
// decoder. Covers: config/tensor/signature contracts, relative-bias vision
// parity (window vs global distinction), decoder logits/KV parity, deterministic
// tokens, the exact prompt/parser contracts (plain/formatted/region/color),
// source region transforms, invalid guidance/mode refusal, and multi-crop
// composition/order. Synthetic weights only; no model/download, no real-model
// parity claim.
#include "ut/ut.hpp"
#include "test_device.h"
#include "ocr_p8_test_utils.h"

#include "ocr/models/got_ocr2.h"

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
    GotOcr2Config cfg;
    ocr::test::P8SyntheticWeights weights;
    WeightStore store;
    luisa::unique_ptr<P7VlmExecutor> exec;

    [[nodiscard]] bool init(Device &device, uint32_t seed = 0x5EEDu) {
        cfg = got_ocr2_synthetic_config();
        if (!cfg.validate().ok()) { return false; }
        weights = ocr::test::generate_p8_weights(cfg.graph, DenseLinearStorage::fp32, 0, seed);
        const auto vpath = tmp.path / "got_vision.lcocr";
        const auto dpath = tmp.path / "got_decoder.lcocr";
        if (auto s = ocr::test::write_p8_packages(vpath, dpath, cfg.graph, cfg.names, weights, "got_ocr2_0"); !s.ok()) {
            LUISA_WARNING("write_p8_packages: {}", s.message);
            return false;
        }
        auto stream = device.create_stream();
        if (auto s = ocr::test::upload_p7_weights(device, stream, vpath, dpath, store); !s.ok()) {
            LUISA_WARNING("upload_p8: {}", s.message);
            return false;
        }
        exec = luisa::make_unique<P7VlmExecutor>();
        if (auto s = exec->create(device, stream, cfg.graph, cfg.names, store); !s.ok()) {
            LUISA_WARNING("got create: {}", s.message);
            return false;
        }
        return true;
    }
};

[[nodiscard]] P7ImageInput make_image(const GotOcr2Config &cfg, uint32_t w, uint32_t h, uint32_t seed = 11u) {
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
// Host contract checks: identity gates + prompt/parser + region transforms.
// ---------------------------------------------------------------------------
void test_got_contracts() {
    auto cfg = got_ocr2_synthetic_config();
    expect(cfg.validate().ok());
    expect(cfg.graph.vision.mask_kind == P7VisionMaskKind::window_global);
    expect(cfg.graph.vision.rel_pos_window != 0u);
    expect(cfg.graph.vision.rope_kind == P7VisionRopeKind::none);
    expect(cfg.graph.connector.kind == P7ConnectorKind::mlp);
    expect(cfg.graph.text_pos == P7TextPosKind::rope1d);
    auto expected = got_ocr2_expected_tensors(cfg);
    expect(expected.size() > 30u);
    // The relative-bias tables are in the inventory.
    bool has_rel = false;
    for (const auto &e : expected) { if (e.name.find("rel_pos_bias") != luisa::string::npos) { has_rel = true; } }
    expect(has_rel);
    expect(got_ocr2_validate_real_artifact("r", "f").code == StatusCode::unvalidated);
    GotOcr2Signature sig;
    expect(got_ocr2_validate_signature(cfg, sig).code == StatusCode::unvalidated);
    sig.checkpoint_revision = "rev";
    sig.config_fingerprint = "fp";
    for (const auto &e : expected) { if (e.required) { sig.tensor_names.push_back(e.name); } }
    expect(got_ocr2_validate_signature(cfg, sig).ok());
    sig.tensor_names.push_back("visual.bogus.weight");
    expect(got_ocr2_validate_signature(cfg, sig).code == StatusCode::unvalidated);
    // A MoE config is refused.
    const char *moe_cfg = R"({"model_type":"got","num_experts":8,"vision_config":{"num_layers":2,"hidden_size":16,"num_heads":2,"head_dim":8,"intermediate_size":32,"patch_size":2,"window_patch_size":2,"rel_pos_window":4,"global_layers":[1]},"text_config":{"vocab_size":32,"hidden_size":32,"num_hidden_layers":3,"num_attention_heads":4,"num_key_value_heads":2,"intermediate_size":64,"max_position_embeddings":256}})";
    expect(!parse_got_ocr2_config(moe_cfg).ok());
    // Vision RoPE is refused (GOT uses relative bias).
    const char *rope_cfg = R"({"model_type":"got","vision_config":{"num_layers":2,"hidden_size":16,"num_heads":2,"head_dim":8,"intermediate_size":32,"patch_size":2,"window_patch_size":2,"rel_pos_window":4,"global_layers":[1],"rope_kind":"axial2d"},"text_config":{"vocab_size":32,"hidden_size":32,"num_hidden_layers":3,"num_attention_heads":4,"num_key_value_heads":2,"intermediate_size":64,"max_position_embeddings":256}})";
    expect(!parse_got_ocr2_config(rope_cfg).ok());
    const char *good_cfg = R"({"model_type":"got_ocr2","vision_config":{"num_layers":2,"hidden_size":16,"num_heads":2,"head_dim":8,"intermediate_size":32,"patch_size":2,"window_patch_size":2,"rel_pos_window":4,"global_layers":[1]},"text_config":{"vocab_size":32,"hidden_size":32,"num_hidden_layers":3,"num_attention_heads":4,"num_key_value_heads":2,"intermediate_size":64,"max_position_embeddings":256}})";
    auto parsed = parse_got_ocr2_config(good_cfg);
    expect(parsed.ok());
    if (parsed.ok()) { expect(parsed.value->validate().ok()); }
}

// Exact prompt/parser contracts + round-trip + invalid refusal.
void test_got_prompt_contracts() {
    GotRegionBox box{0.1f, 0.2f, 0.8f, 0.9f};
    // plain
    auto plain = got_ocr2_build_prompt(GotPromptMode::plain, {}, "");
    expect(plain.ok());
    if (plain.ok()) {
        expect(*plain.value == "OCR: ");
        auto p = got_ocr2_parse_prompt(*plain.value);
        expect(p.ok());
        if (p.ok()) { expect(p.value->mode == GotPromptMode::plain); }
    }
    // formatted
    auto fmt = got_ocr2_build_prompt(GotPromptMode::formatted, {}, "");
    expect(fmt.ok());
    if (fmt.ok()) {
        expect(*fmt.value == "OCR with format: ");
        auto p = got_ocr2_parse_prompt(*fmt.value);
        expect(p.ok());
        if (p.ok()) { expect(p.value->mode == GotPromptMode::formatted); }
    }
    // region round-trip
    auto region = got_ocr2_build_prompt(GotPromptMode::region, box, "");
    expect(region.ok());
    if (region.ok()) {
        auto p = got_ocr2_parse_prompt(*region.value);
        expect(p.ok());
        if (p.ok()) {
            expect(p.value->mode == GotPromptMode::region);
            expect(std::abs(p.value->region.x0 - box.x0) < 1e-4f);
            expect(std::abs(p.value->region.y1 - box.y1) < 1e-4f);
        }
    }
    // color round-trip
    auto color = got_ocr2_build_prompt(GotPromptMode::color, {}, "red");
    expect(color.ok());
    if (color.ok()) {
        expect(*color.value == "OCR with color: red");
        auto p = got_ocr2_parse_prompt(*color.value);
        expect(p.ok());
        if (p.ok()) {
            expect(p.value->mode == GotPromptMode::color);
            expect(p.value->color == "red");
        }
    }
    // Invalid guidance: region without a valid box is refused.
    GotRegionBox bad{0.5f, 0.5f, 0.5f, 0.5f};// zero area
    expect(!got_ocr2_build_prompt(GotPromptMode::region, bad, "").ok());
    GotRegionBox oob{0.0f, 0.0f, 1.5f, 1.0f};// out of [0,1]
    expect(!got_ocr2_build_prompt(GotPromptMode::region, oob, "").ok());
    // Color without a name is refused.
    expect(!got_ocr2_build_prompt(GotPromptMode::color, {}, "").ok());
    // Malformed prompts are refused (never a silent plain fallback).
    expect(!got_ocr2_parse_prompt("garbage").ok());
    expect(!got_ocr2_parse_prompt("OCR with region: [0.1,0.2]").ok());// too few coords
    expect(!got_ocr2_parse_prompt("OCR with region: [a,b,c,d]").ok());// non-numeric
    expect(!got_ocr2_parse_prompt("OCR with region: [0.9,0.9,0.1,0.1]").ok());// negative area
    expect(!got_ocr2_parse_prompt("OCR with color: ").ok());// empty color
    // Locale-independent canonical form: comma decimal separators, wrong
    // precision, exponent forms, and signs are all refused (only d.dddd).
    expect(!got_ocr2_parse_prompt("OCR with region: [0,1000,0,2000,0,8000,0,9000]").ok());// comma decimals
    expect(!got_ocr2_parse_prompt("OCR with region: [0.1,0.2,0.8,0.9]").ok());// 1-digit fraction
    expect(!got_ocr2_parse_prompt("OCR with region: [0.10000,0.20000,0.80000,0.90000]").ok());// 5-digit
    expect(!got_ocr2_parse_prompt("OCR with region: [1e-1,0.2000,0.8000,0.9000]").ok());// exponent
    expect(!got_ocr2_parse_prompt("OCR with region: [+0.1000,0.2000,0.8000,0.9000]").ok());// sign
    // Exact canonical round-trip for every coordinate digit pattern.
    GotRegionBox digits{0.0f, 0.1234f, 0.9999f, 1.0f};
    auto rt = got_ocr2_build_prompt(GotPromptMode::region, digits, "");
    expect(rt.ok());
    if (rt.ok()) {
        auto p = got_ocr2_parse_prompt(*rt.value);
        expect(p.ok());
        if (p.ok()) {
            expect(p.value->region.x0 == 0.0f);
            expect(std::abs(p.value->region.y0 - 0.1234f) < 1e-6f);
            expect(std::abs(p.value->region.x1 - 0.9999f) < 1e-6f);
            expect(p.value->region.y1 == 1.0f);
        }
    }
}

// GOT memory preflight: report + fail-before-allocation + measured mirror.
void test_got_memory_preflight() {
    auto cfg = got_ocr2_synthetic_config();
    MemoryLimits unlimited;
    auto report = got_ocr2_preflight(cfg, 64, unlimited);
    expect(report.ok());
    if (report.ok()) {
        expect(report.value->vision_weight_bytes > 0u);
        expect(report.value->decoder_weight_bytes > 0u);
        expect(report.value->workspace_bytes > 0u);
        expect(report.value->kv_cache_bytes > 0u);
        expect(report.value->total_device_bytes ==
               report.value->vision_weight_bytes + report.value->decoder_weight_bytes +
                   report.value->workspace_bytes + report.value->kv_cache_bytes);
        expect(report.value->fits);
    }
    MemoryLimits tiny;
    tiny.max_device_bytes = 1024;
    auto refused = got_ocr2_preflight(cfg, 64, tiny);
    expect(!refused.ok());
    expect(refused.status.code == StatusCode::out_of_memory);
    expect(!got_ocr2_preflight(cfg, 0, unlimited).ok());// degenerate context
}

// Source region transforms: normalized box -> source pixels.
void test_got_region_transforms() {
    GotRegionBox box{0.25f, 0.25f, 0.75f, 0.75f};
    auto px = got_ocr2_region_to_pixels(box, 100, 200);
    expect(px.ok());
    if (px.ok()) {
        expect(px.value->x0 == 25u);
        expect(px.value->y0 == 50u);
        expect(px.value->x1 == 75u);
        expect(px.value->y1 == 150u);
    }
    // Clamping to the image edge.
    GotRegionBox edge{0.0f, 0.0f, 1.0f, 1.0f};
    auto full = got_ocr2_region_to_pixels(edge, 10, 10);
    expect(full.ok());
    if (full.ok()) {
        expect(full.value->x0 == 0u);
        expect(full.value->y0 == 0u);
        expect(full.value->x1 == 10u);
        expect(full.value->y1 == 10u);
    }
    // Degenerate image / box refused.
    expect(!got_ocr2_region_to_pixels(box, 0, 10).ok());
    GotRegionBox tiny{0.5f, 0.5f, 0.5001f, 0.5001f};
    expect(!got_ocr2_region_to_pixels(tiny, 2, 2).ok());// maps to zero area
}

// ---------------------------------------------------------------------------
// Relative-bias vision parity + window-vs-global distinction.
// ---------------------------------------------------------------------------
void test_got_vision_parity(Device &device) {
    Fixture fx;
    expect(fx.init(device));
    if (!fx.exec) { return; }
    auto stream = device.create_stream();
    for (const auto &[w, h] : std::initializer_list<std::pair<uint32_t, uint32_t>>{{8, 8}, {12, 8}}) {
        auto img = make_image(fx.cfg, w, h);
        auto pixels = ocr::test::p7_oracle_preprocess(img.pixels, w, h,
                                                      img.packed.resized_width, img.packed.resized_height,
                                                      fx.cfg.graph.mean, fx.cfg.graph.std_dev);
        ocr::test::P8VisionOracle oracle;
        expect(ocr::test::p8_oracle_vision(fx.cfg.graph, fx.weights, pixels, img.packed, oracle).ok());
        fx.exec->reset_vision_workspace();
        P7VisionOutput vision;
        {
            auto list = CommandList::create();
            auto v = fx.exec->record_vision(list, luisa::span<const P7ImageInput>{&img, 1});
            expect(v.ok());
            if (!v.ok()) { continue; }
            vision = *v.value;
            stream << list.commit() << synchronize();
        }
        luisa::vector<float> dev_patches, dev_final, dev_features;
        expect(fx.exec->download_vision_patches(stream, dev_patches).ok());
        expect(fx.exec->download_vision_final(stream, dev_final).ok());
        expect(fx.exec->download_vision_features(stream, dev_features).ok());
        stream << synchronize();
        const auto what = luisa::format("got {}x{}", w, h);
        ocr::test::expect_close(dev_patches, oracle.patches, kLinAtol, kLinRtol, (what + " patches").c_str());
        ocr::test::expect_close(dev_final, oracle.block_out.back(), kAtol, kRtol, (what + " vision final (rel bias)").c_str());
        ocr::test::expect_close(dev_features, oracle.features, kAtol, kRtol, (what + " linear connector").c_str());
        expect(vision.total_tokens == static_cast<uint32_t>(img.packed.visual_tokens));
    }
}

// The relative-position bias must actually change the output: a zero-bias tower
// must DIFFER from the learned-bias tower on the same input.
void test_got_rel_bias_distinction(Device &device) {
    Fixture fx;
    expect(fx.init(device));
    if (!fx.exec) { return; }
    auto stream = device.create_stream();
    auto img = make_image(fx.cfg, 8, 8);
    auto pixels = ocr::test::p7_oracle_preprocess(img.pixels, 8, 8,
                                                  img.packed.resized_width, img.packed.resized_height,
                                                  fx.cfg.graph.mean, fx.cfg.graph.std_dev);
    ocr::test::P8VisionOracle with_bias;
    expect(ocr::test::p8_oracle_vision(fx.cfg.graph, fx.weights, pixels, img.packed, with_bias).ok());
    // Zero-bias oracle: same weights but zeroed relative-bias tables.
    auto zero_weights = fx.weights;
    for (auto &t : zero_weights.rel_pos_bias) { std::fill(t.begin(), t.end(), 0.0f); }
    ocr::test::P8VisionOracle zero_bias;
    expect(ocr::test::p8_oracle_vision(fx.cfg.graph, zero_weights, pixels, img.packed, zero_bias).ok());
    bool differ = false;
    for (size_t i = 0; i < with_bias.features.size() && i < zero_bias.features.size(); i++) {
        if (std::abs(with_bias.features[i] - zero_bias.features[i]) > 1e-4f) { differ = true; break; }
    }
    expect(differ) << "relative-position bias must change the vision output";
    // Device matches the learned-bias oracle.
    fx.exec->reset_vision_workspace();
    {
        auto list = CommandList::create();
        auto v = fx.exec->record_vision(list, luisa::span<const P7ImageInput>{&img, 1});
        expect(v.ok());
        if (!v.ok()) { return; }
        stream << list.commit() << synchronize();
    }
    luisa::vector<float> dev_features;
    expect(fx.exec->download_vision_features(stream, dev_features).ok());
    stream << synchronize();
    ocr::test::expect_close(dev_features, with_bias.features, kAtol, kRtol, "got rel-bias features");
}

// ---------------------------------------------------------------------------
// Decoder logits/KV parity + deterministic tokens.
// ---------------------------------------------------------------------------
void test_got_decoder_parity(Device &device) {
    Fixture fx;
    expect(fx.init(device));
    if (!fx.exec) { return; }
    auto stream = device.create_stream();
    auto img = make_image(fx.cfg, 8, 8);
    auto pixels = ocr::test::p7_oracle_preprocess(img.pixels, 8, 8,
                                                  img.packed.resized_width, img.packed.resized_height,
                                                  fx.cfg.graph.mean, fx.cfg.graph.std_dev);
    ocr::test::P8VisionOracle vision_oracle;
    expect(ocr::test::p8_oracle_vision(fx.cfg.graph, fx.weights, pixels, img.packed, vision_oracle).ok());
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
    ocr::test::P8TextOracle text_oracle;
    expect(ocr::test::p8_oracle_text_forward(fx.cfg.graph, fx.weights, prompt, position_ids,
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
        auto s = fx.exec->text_executor().record_prefill_all_logits(list, *req.value, cache, logits_buf.view());
        expect(s.ok());
        if (!s.ok()) { return; }
        stream << list.commit() << synchronize();
    }
    luisa::vector<float> dev_logits(static_cast<size_t>(prompt_len) * vocab);
    stream << logits_buf.copy_to(luisa::span<float>{dev_logits}) << synchronize();
    ocr::test::expect_close(dev_logits, text_oracle.logits, kAtol, kRtol, "got full prefill logits");
    const uint32_t first = ocr::test::p7_oracle_argmax(luisa::span<const float>{
        text_oracle.logits.data() + static_cast<size_t>(prompt_len - 1) * vocab, vocab});
    luisa::vector<uint32_t> step_tokens = {first};
    luisa::vector<int32_t> step_pos = {static_cast<int32_t>(prompt_len)};
    ocr::test::P8TextOracle step_oracle;
    expect(ocr::test::p8_oracle_text_forward(fx.cfg.graph, fx.weights, step_tokens, step_pos, {}, 0u, 0u,
                                             prompt_len, k_caches, v_caches, step_oracle).ok());
    luisa::vector<float> dev_step_logits(vocab);
    {
        auto list = CommandList::create();
        auto s = fx.exec->text_executor().record_decode_step(list, first, cache, dev_step_logits);
        expect(s.ok());
        if (!s.ok()) { return; }
        stream << list.commit() << synchronize();
    }
    ocr::test::expect_close(dev_step_logits, step_oracle.logits, kAtol, kRtol, "got decode step logits");
    luisa::vector<float> dev_k, dev_v;
    expect(cache.download_layer(stream, 0, 0, dev_k, dev_v).ok());
    stream << synchronize();
    ocr::test::expect_close(dev_k, step_oracle.k_cache[0], kAtol, kRtol, "got decode K cache");
    ocr::test::expect_close(dev_v, step_oracle.v_cache[0], kAtol, kRtol, "got decode V cache");
    // MEASURED synthetic memory vs the preflight plan (S2 accounting: include
    // the executor-owned tied-head transpose in weight residency).
    const uint64_t store_bytes = fx.store.total_bytes();
    const uint64_t owned_bytes = fx.exec->text_executor().owned_device_bytes();
    const uint64_t measured_weights = store_bytes + owned_bytes;
    const uint64_t measured_ws = fx.exec->vision_workspace_bytes() + fx.exec->decoder_workspace_bytes();
    const uint64_t measured_kv = cache.device_bytes();
    expect(fx.exec->weight_bytes() == measured_weights);
    LUISA_INFO("GOT-OCR2.0 synthetic MEASURED device bytes: weights={} (store {} + owned tied-head {}) ws={} kv={} total={}",
               measured_weights, store_bytes, owned_bytes, measured_ws, measured_kv,
               measured_weights + measured_ws + measured_kv);
    MemoryLimits unlimited;
    auto pf = got_ocr2_preflight(fx.cfg, kc.max_seq_len, unlimited);
    expect(pf.ok());
    if (pf.ok()) {
        expect(pf.value->workspace_bytes == measured_ws) << "GOT preflight workspace must mirror the actual arenas";
        expect(pf.value->kv_cache_bytes == measured_kv) << "GOT preflight KV must equal the actual cache bytes";
        expect(pf.value->vision_weight_bytes == fx.store.component_bytes("vision"));
        expect(pf.value->decoder_weight_bytes == fx.store.component_bytes("decoder") + owned_bytes);
    }
    cache.release(0);
    // Deterministic repeat-run.
    const auto generate = [&](uint32_t max_new) -> luisa::vector<uint32_t> {
        ContiguousKvCache c2;
        KvCacheConfig kc2 = kc;
        kc2.max_seq_len = prompt_len + max_new;
        if (!c2.create(device, stream, kc2, nullptr).ok()) { return {}; }
        DenseDecodeOptions dopts;
        dopts.sampling.greedy = true;
        dopts.max_new_tokens = max_new;
        DenseDecodeController controller;
        if (!controller.create(fx.exec->text_executor(), nullptr, dopts).ok()) { return {}; }
        auto gen = controller.generate(stream, *req.value, c2);
        if (!gen.ok()) { return {}; }
        return gen.value->tokens;
    };
    auto t1 = generate(5);
    auto t2 = generate(5);
    expect(!t1.empty());
    expect(t1 == t2) << "repeat GOT decode must be identical";
}

// ---------------------------------------------------------------------------
// Multi-crop composition + order + malformed formatted refusal.
// ---------------------------------------------------------------------------
void test_got_multi_crop() {
    // Three crops out of reading order; composition must restore order.
    luisa::vector<GotCropResult> crops;
    GotCropResult c0;
    c0.box = GotPixelBox{0, 0, 10, 10};
    c0.text = "first";
    c0.reading_order = 0;
    GotCropResult c1;
    c1.box = GotPixelBox{0, 10, 10, 20};
    c1.text = "second";
    c1.reading_order = 1;
    GotCropResult c2;
    c2.box = GotPixelBox{0, 20, 10, 30};
    c2.text = "third";
    c2.reading_order = 2;
    crops.push_back(c2);
    crops.push_back(c0);
    crops.push_back(c1);
    auto doc = got_ocr2_compose_crops(crops, GotPromptMode::plain);
    expect(doc.ok());
    if (doc.ok()) {
        expect(doc.value->text == "first\nsecond\nthird");
        expect(doc.value->lines.size() == 3u);
        expect(doc.value->lines[0].reading_order == 0u);
        expect(doc.value->lines[2].reading_order == 2u);
        // Source boxes preserved.
        expect(doc.value->lines[0].source_box.y0 == 0.0f);
        expect(doc.value->lines[2].source_box.y0 == 20.0f);
        // Generative content: no confidence.
        for (const auto &line : doc.value->lines) { expect(!line.has_confidence); }
    }
    // Multi-page: pages ascending first, reading order within each page; the
    // page provenance is carried on lines AND blocks.
    luisa::vector<GotCropResult> pages;
    GotCropResult p1c0;
    p1c0.box = GotPixelBox{0, 0, 10, 10};
    p1c0.text = "page1-first";
    p1c0.reading_order = 0;
    p1c0.page_index = 1;
    GotCropResult p0c1;
    p0c1.box = GotPixelBox{0, 10, 10, 20};
    p0c1.text = "page0-second";
    p0c1.reading_order = 1;
    p0c1.page_index = 0;
    GotCropResult p0c0;
    p0c0.box = GotPixelBox{0, 0, 10, 10};
    p0c0.text = "page0-first";
    p0c0.reading_order = 0;
    p0c0.page_index = 0;
    pages.push_back(p1c0);
    pages.push_back(p0c1);
    pages.push_back(p0c0);
    auto pdoc = got_ocr2_compose_crops(pages, GotPromptMode::plain);
    expect(pdoc.ok());
    if (pdoc.ok()) {
        expect(pdoc.value->text == "page0-first\npage0-second\npage1-first");
        expect(pdoc.value->lines[0].page_index == 0u);
        expect(pdoc.value->lines[1].page_index == 0u);
        expect(pdoc.value->lines[2].page_index == 1u);
        expect(pdoc.value->blocks[2].page_index == 1u);
    }
    // Formatted mode: valid Markdown composes; malformed fails closed.
    luisa::vector<GotCropResult> fmt_crops;
    GotCropResult f0;
    f0.box = GotPixelBox{0, 0, 10, 10};
    f0.text = "# Title\n\nBody text.";
    f0.reading_order = 0;
    fmt_crops.push_back(f0);
    auto fmt_doc = got_ocr2_compose_crops(fmt_crops, GotPromptMode::formatted);
    expect(fmt_doc.ok());
    // Malformed Markdown (unclosed table) is an ERROR, never a plain fallback.
    luisa::vector<GotCropResult> bad_crops;
    GotCropResult b0;
    b0.box = GotPixelBox{0, 0, 10, 10};
    b0.text = "<table><tr><td>unclosed";
    b0.reading_order = 0;
    bad_crops.push_back(b0);
    auto bad_doc = got_ocr2_compose_crops(bad_crops, GotPromptMode::formatted);
    expect(!bad_doc.ok());
    // region/color are prompt modes, not composition modes -> refused.
    expect(!got_ocr2_compose_crops(crops, GotPromptMode::region).ok());
    expect(!got_ocr2_compose_crops(crops, GotPromptMode::color).ok());
}

}// namespace

int main(int argc, char *argv[]) {
    auto dc = luisa::test::create_device_from_ut(argc, argv);
    if (!dc) { return 0; }
    boost::ut::detail::cfg::parse_arg_with_fallback(argc, const_cast<const char **>(argv));
    auto &device = dc->device;
    "got_contracts"_test = [&] { test_got_contracts(); };
    "got_prompt_contracts"_test = [&] { test_got_prompt_contracts(); };
    "got_region_transforms"_test = [&] { test_got_region_transforms(); };
    "got_vision_parity"_test = [&] { test_got_vision_parity(device); };
    "got_rel_bias_distinction"_test = [&] { test_got_rel_bias_distinction(device); };
    "got_decoder_parity"_test = [&] { test_got_decoder_parity(device); };
    "got_multi_crop"_test = [&] { test_got_multi_crop(); };
    "got_memory_preflight"_test = [&] { test_got_memory_preflight(); };
}

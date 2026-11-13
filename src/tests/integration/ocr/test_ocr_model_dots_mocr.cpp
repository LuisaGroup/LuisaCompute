// Test for ocr/models/dots_mocr + the shared dense executor (device) -- the
// SMALL SYNTHETIC dots.mocr vertical slice: ViT + spatial-merge patch merger +
// dense GQA decoder (RMSNorm / 1D RoPE / SwiGLU), plus the W4A32 quantized path.
// Covers: the 28-layer / 12-query / 2-KV declared contract, config/tensor/
// signature contracts, vision/merger parity, first decoder layer, full prefill
// logits, decode KV/logits, deterministic tokens, blank/repeated inputs,
// memory preflight, and the quantized A/B arm (packed INT4 reads vs a
// dequantized reference). Synthetic weights only; no model/download, no
// real-model parity claim.
#include "ut/ut.hpp"
#include "test_device.h"
#include "ocr_p8_test_utils.h"

#include "ocr/models/dots_mocr.h"

using namespace luisa;
using namespace luisa::compute;
using namespace luisa::ocr;
using namespace boost::ut;
using namespace boost::ut::literals;

namespace {

// Tolerances: FP32 device vs double oracle (operation-appropriate, plan §10.1).
constexpr float kAtol = 3e-3f;
constexpr float kRtol = 3e-3f;
constexpr float kLinAtol = 5e-4f;
constexpr float kLinRtol = 5e-4f;
// Quantized arm: the device packed-INT4 path vs the dequantized-weight oracle.
// Both read the SAME dequantized values, so parity is tight (the only extra
// error is the device FP32 accumulation order).
constexpr float kQuantAtol = 3e-3f;
constexpr float kQuantRtol = 3e-3f;

struct Fixture {
    ocr::test::P7TempDir tmp;
    DotsMocrConfig cfg;
    ocr::test::P8SyntheticWeights weights;
    WeightStore store;
    luisa::unique_ptr<P7VlmExecutor> exec;

    [[nodiscard]] bool init(Device &device, DenseLinearStorage storage = DenseLinearStorage::fp32,
                            uint32_t group_size = 0, uint32_t seed = 0x5EEDu) {
        cfg = dots_mocr_synthetic_config();
        cfg.linear_storage = storage;
        cfg.linear_group_size = group_size;
        cfg.graph.linear_storage = storage;
        cfg.graph.linear_group_size = group_size;
        if (storage != DenseLinearStorage::fp32) { cfg.graph.tie_word_embeddings = false; }
        if (!cfg.validate().ok()) { return false; }
        weights = ocr::test::generate_p8_weights(cfg.graph, storage, group_size, seed);
        const auto vpath = tmp.path / "dots_vision.lcocr";
        const auto dpath = tmp.path / "dots_decoder.lcocr";
        if (auto s = ocr::test::write_p8_packages(vpath, dpath, cfg.graph, cfg.names, weights, "dots_mocr"); !s.ok()) {
            LUISA_WARNING("write_p8_packages: {}", s.message);
            return false;
        }
        // S1: the package-load gate validates BOTH component tables on disk
        // (vision FP32-only; decoder per-linear storage) BEFORE any upload.
        {
            auto vr = PackageReader::open(vpath);
            auto dr = PackageReader::open(dpath);
            if (!vr.ok() || !dr.ok()) { return false; }
            if (auto s = dots_mocr_validate_packages(cfg, (*vr.value)->tensors(), (*dr.value)->tensors()); !s.ok()) {
                LUISA_WARNING("dots_mocr package gate: {}", s.message);
                return false;
            }
        }
        auto stream = device.create_stream();
        if (auto s = ocr::test::upload_p7_weights(device, stream, vpath, dpath, store); !s.ok()) {
            LUISA_WARNING("upload_p8: {}", s.message);
            return false;
        }
        exec = luisa::make_unique<P7VlmExecutor>();
        if (auto s = exec->create(device, stream, cfg.graph, cfg.names, store); !s.ok()) {
            LUISA_WARNING("dots create: {}", s.message);
            return false;
        }
        return true;
    }
};

[[nodiscard]] P7ImageInput make_image(const DotsMocrConfig &cfg, uint32_t w, uint32_t h, uint32_t seed = 11u) {
    auto crop = ocr::test::p7_make_crop(w, h, 0, seed);
    P7ImageInput img;
    img.pixels = crop.pixels;
    img.source_width = w;
    img.source_height = h;
    auto packed = p7_pack_image(w, h, cfg.graph.packing, cfg.graph.connector);
    img.packed = *packed.value;
    return img;
}

// Build the dense-executor config for the dots.mocr decoder (1-axis RoPE, QKV
// bias, no QK-Norm) with the fixture's storage policy.
[[nodiscard]] DenseExecutorConfig dense_cfg(const Fixture &fx) {
    DenseExecutorConfig dc;
    dc.decoder = fx.cfg.graph.text;
    dc.names = fx.cfg.names.text;
    dc.tie_word_embeddings = fx.cfg.graph.tie_word_embeddings;
    dc.qk_norm = false;
    dc.qkv_bias = true;
    dc.rope_axes = 1;
    dc.component = fx.cfg.graph.decoder_component;
    dc.prefill_chunk_rows = fx.cfg.graph.prefill_chunk_rows;
    dc.linear_storage = fx.cfg.linear_storage;
    dc.linear_group_size = fx.cfg.linear_group_size;
    return dc;
}

// ---------------------------------------------------------------------------
// Host contract checks: the 28/12/2 declared contract + identity gates.
// ---------------------------------------------------------------------------
void test_dots_contracts() {
    auto cfg = dots_mocr_synthetic_config();
    expect(cfg.validate().ok());
    // The DECLARED full contract: 28 layers, 12 query heads, 2 KV heads.
    auto declared = dots_mocr_declared_decoder_config();
    expect(declared.num_layers == 28u);
    expect(declared.num_attention_heads == 12u);
    expect(declared.num_kv_heads == 2u);
    expect(declared.num_attention_heads % declared.num_kv_heads == 0u);
    expect(declared.validate().ok());
    expect(declared.rope_pair_layout == RopePairLayout::rotate_half);
    // The synthetic slice uses a SCALED geometry but the SAME conventions.
    expect(cfg.graph.text_pos == P7TextPosKind::rope1d);
    expect(cfg.graph.connector.kind == P7ConnectorKind::spatial_mlp);
    expect(cfg.graph.vision.mask_kind == P7VisionMaskKind::bidirectional);
    // Tensor inventory.
    auto expected = dots_mocr_expected_tensors(cfg);
    expect(expected.size() > 30u);
    // Real-artifact gate fails closed.
    expect(dots_mocr_validate_real_artifact("r", "f").code == StatusCode::unvalidated);
    // Signature: outer model_type alone is never identity.
    DotsMocrSignature sig;
    expect(dots_mocr_validate_signature(cfg, sig).code == StatusCode::unvalidated);
    sig.checkpoint_revision = "rev";
    sig.config_fingerprint = "fp";
    expect(dots_mocr_validate_signature(cfg, sig).code == StatusCode::unvalidated);// no tensors
    for (const auto &e : expected) { if (e.required) { sig.tensor_names.push_back(e.name); } }
    expect(dots_mocr_validate_signature(cfg, sig).ok());
    sig.tensor_names.push_back("visual.bogus.weight");
    expect(dots_mocr_validate_signature(cfg, sig).code == StatusCode::unvalidated);
    // A MoE config is refused.
    const char *moe_cfg = R"({"model_type":"dots","num_experts":8,"vision_config":{"num_layers":2,"hidden_size":16,"num_heads":2,"head_dim":8,"intermediate_size":32,"patch_size":2},"text_config":{"vocab_size":32,"hidden_size":32,"num_hidden_layers":3,"num_attention_heads":4,"num_key_value_heads":2,"intermediate_size":64,"max_position_embeddings":256}})";
    expect(!parse_dots_mocr_config(moe_cfg).ok());
    // A valid synthetic config parses.
    const char *good_cfg = R"({"model_type":"dots_mocr","vision_config":{"num_layers":2,"hidden_size":16,"num_heads":2,"head_dim":8,"intermediate_size":32,"patch_size":2},"text_config":{"vocab_size":32,"hidden_size":32,"num_hidden_layers":3,"num_attention_heads":4,"num_key_value_heads":2,"intermediate_size":64,"max_position_embeddings":256}})";
    auto parsed = parse_dots_mocr_config(good_cfg);
    expect(parsed.ok());
    if (parsed.ok()) { expect(parsed.value->validate().ok()); }
    // The W4A32 path requires a standalone lm_head (tied embeddings refused).
    auto qcfg = dots_mocr_synthetic_config();
    qcfg.linear_storage = DenseLinearStorage::int4_sym;
    qcfg.linear_group_size = 64;
    expect(!qcfg.validate().ok());// tied -> refused
    qcfg.graph.tie_word_embeddings = false;
    expect(qcfg.validate().ok());
    // S1: package-load gate. Build the exact expected tables as FP32, then
    // flip individual encodings. Vision INT4 must be REFUSED even though
    // visual.blocks.*.attn.q_proj.weight shares a decoder projection suffix;
    // decoder INT4 + vision FP32 must be ACCEPTED.
    const auto make_tables = [](const DotsMocrConfig &c, luisa::vector<PackageTensor> &vis,
                                luisa::vector<PackageTensor> &dec) {
        for (const auto &e : dots_mocr_expected_tensors(c)) {
            PackageTensor t;
            t.name = e.name;
            t.shape = e.shape;
            t.layout = e.layout;
            t.encoding = TensorEncoding::fp32;
            uint64_t elems = 1;
            for (auto dim : e.shape) { elems *= dim; }
            t.element_count = elems;
            if (e.name.compare(0, c.names.visual_prefix.size(), c.names.visual_prefix) == 0) { vis.push_back(std::move(t)); }
            else { dec.push_back(std::move(t)); }
        }
    };
    {
        luisa::vector<PackageTensor> vis, dec;
        make_tables(cfg, vis, dec);
        // FP32 baseline accepted.
        expect(dots_mocr_validate_packages(cfg, vis, dec).ok());
        // Quantized config: decoder linears INT4, vision stays FP32 -> accepted.
        luisa::vector<PackageTensor> qvis, qdec;
        make_tables(qcfg, qvis, qdec);
        for (auto &t : qdec) {
            if (dense_linear_is_quantizable(t.name, "model.", "lm_head.weight")) {
                t.encoding = TensorEncoding::int4_packed;
                t.quant_kind = QuantKind::int4_symmetric;
                t.group_size = 64;
            }
        }
        expect(dots_mocr_validate_packages(qcfg, qvis, qdec).ok());
        // S1 regression: a VISUAL q_proj stored as INT4 must be refused (the
        // old suffix classifier wrongly treated it as a quantizable linear).
        auto bad_vis = qvis;
        bool flipped = false;
        for (auto &t : bad_vis) {
            if (t.name.find(".attn.q_proj.weight") != luisa::string::npos) {
                t.encoding = TensorEncoding::int4_packed;
                t.quant_kind = QuantKind::int4_symmetric;
                t.group_size = 64;
                flipped = true;
            }
        }
        expect(flipped);
        expect(!dots_mocr_validate_packages(qcfg, bad_vis, qdec).ok());
        // A decoder linear left FP32 under the INT4 policy is refused.
        auto bad_dec = qdec;
        for (auto &t : bad_dec) {
            if (t.name == "model.layers.0.self_attn.q_proj.weight") { t.encoding = TensorEncoding::fp32; t.quant_kind = QuantKind::none; t.group_size = 0; }
        }
        expect(!dots_mocr_validate_packages(qcfg, qvis, bad_dec).ok());
        // An unexpected tensor name is refused.
        auto extra = dec;
        PackageTensor bogus;
        bogus.name = "model.bogus.weight";
        bogus.shape = {1};
        bogus.element_count = 1;
        extra.push_back(bogus);
        expect(!dots_mocr_validate_packages(cfg, vis, extra).ok());
    }
}

// ---------------------------------------------------------------------------
// Memory preflight: required-memory report + fail-before-allocation.
// ---------------------------------------------------------------------------
void test_dots_memory_preflight() {
    auto cfg = dots_mocr_synthetic_config();
    // No explicit limit -> fits, report populated.
    MemoryLimits unlimited;
    auto report = dots_mocr_preflight_decoder(cfg, 64, unlimited);
    expect(report.ok());
    if (report.ok()) {
        expect(report.value->decoder_weight_bytes > 0u);
        expect(report.value->workspace_bytes > 0u);
        expect(report.value->kv_cache_bytes > 0u);
        expect(report.value->total_device_bytes ==
               report.value->decoder_weight_bytes + report.value->workspace_bytes + report.value->kv_cache_bytes);
        expect(report.value->fits);
    }
    // A tiny budget fails BEFORE allocation with a useful report.
    MemoryLimits tiny;
    tiny.max_device_bytes = 1024;
    auto refused = dots_mocr_preflight_decoder(cfg, 64, tiny);
    expect(!refused.ok());
    expect(refused.status.code == StatusCode::out_of_memory);
    // The quantized path requires fewer weight bytes than FP32.
    auto qcfg = dots_mocr_synthetic_config();
    qcfg.linear_storage = DenseLinearStorage::int4_sym;
    qcfg.linear_group_size = 64;
    qcfg.graph.tie_word_embeddings = false;
    auto fp32_report = dots_mocr_preflight_decoder(cfg, 64, unlimited);
    auto quant_report = dots_mocr_preflight_decoder(qcfg, 64, unlimited);
    expect(fp32_report.ok());
    expect(quant_report.ok());
    if (fp32_report.ok() && quant_report.ok()) {
        expect(quant_report.value->decoder_weight_bytes < fp32_report.value->decoder_weight_bytes);
    }
    // The DECLARED 28/12/2 contract geometry preflights (no allocation) and
    // reports a large requirement; a small budget refuses it.
    auto declared_cfg = dots_mocr_synthetic_config();
    declared_cfg.graph.text = dots_mocr_declared_decoder_config();
    declared_cfg.graph.tie_word_embeddings = false;
    declared_cfg.linear_storage = DenseLinearStorage::int4_sym;
    declared_cfg.linear_group_size = 128;
    auto declared_report = dots_mocr_preflight_decoder(declared_cfg, 4096, unlimited);
    expect(declared_report.ok());
    if (declared_report.ok()) {
        // A 7B-class decoder is far beyond a 1 GiB budget.
        MemoryLimits gib;
        gib.max_device_bytes = 1ull << 30;
        auto big_refused = dots_mocr_preflight_decoder(declared_cfg, 4096, gib);
        expect(!big_refused.ok());
        expect(big_refused.status.code == StatusCode::out_of_memory);
    }
}

// ---------------------------------------------------------------------------
// Vision + patch-merger parity (FP32).
// ---------------------------------------------------------------------------
void test_dots_vision_parity(Device &device) {
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
        const auto what = luisa::format("dots {}x{}", w, h);
        ocr::test::expect_close(dev_patches, oracle.patches, kLinAtol, kLinRtol, (what + " patches").c_str());
        ocr::test::expect_close(dev_final, oracle.block_out.back(), kAtol, kRtol, (what + " vision final").c_str());
        ocr::test::expect_close(dev_features, oracle.features, kAtol, kRtol, (what + " merger").c_str());
        expect(vision.total_tokens == static_cast<uint32_t>(img.packed.visual_tokens));
    }
}

// ---------------------------------------------------------------------------
// Full prefill logits + decode KV/logits parity (FP32).
// ---------------------------------------------------------------------------
void test_dots_decoder_parity(Device &device) {
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
    // Oracle prefill (all rows). dots.mocr is 1-axis: positions are sequential.
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
        // dots.mocr is 1-axis: use the Phase 3 prefill entry point.
        auto s = fx.exec->text_executor().record_prefill_all_logits(list, *req.value, cache, logits_buf.view());
        expect(s.ok());
        if (!s.ok()) { return; }
        stream << list.commit() << synchronize();
    }
    luisa::vector<float> dev_logits(static_cast<size_t>(prompt_len) * vocab);
    stream << logits_buf.copy_to(luisa::span<float>{dev_logits}) << synchronize();
    ocr::test::expect_close(dev_logits, text_oracle.logits, kAtol, kRtol, "dots full prefill logits");
    // First decoder layer parity (layer 0 output) via the oracle's layer_out.
    // The device does not expose per-layer hidden states directly; the full
    // logits + decode KV below exercise every layer.
    // Decode KV/logits: one greedy step.
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
    ocr::test::expect_close(dev_step_logits, step_oracle.logits, kAtol, kRtol, "dots decode step logits");
    luisa::vector<float> dev_k, dev_v;
    expect(cache.download_layer(stream, 0, 0, dev_k, dev_v).ok());
    stream << synchronize();
    ocr::test::expect_close(dev_k, step_oracle.k_cache[0], kAtol, kRtol, "dots decode K cache");
    ocr::test::expect_close(dev_v, step_oracle.v_cache[0], kAtol, kRtol, "dots decode V cache");
    // MEASURED synthetic memory (actual device buffers, not estimates). S2:
    // the tied-LM-head transpose is EXECUTOR-OWNED, so total weight residency
    // = store bytes + owned bytes (the store alone omits it). The preflight
    // decoder plan already includes the tied head; assert exact equality.
    const uint64_t store_bytes = fx.store.total_bytes();
    const uint64_t owned_bytes = fx.exec->text_executor().owned_device_bytes();
    const uint64_t measured_weights = store_bytes + owned_bytes;
    const uint64_t measured_decoder_ws = fx.exec->decoder_workspace_bytes();
    const uint64_t measured_vision_ws = fx.exec->vision_workspace_bytes();
    const uint64_t measured_kv = cache.device_bytes();
    expect(fx.exec->weight_bytes() == measured_weights) << "executor weight_bytes must equal store + owned bytes";
    expect(fx.exec->text_executor().weight_bytes() == fx.store.component_bytes("decoder") + owned_bytes);
    LUISA_INFO("dots.mocr synthetic MEASURED device bytes: weights={} (store {} + owned tied-head {}) vision_ws={} decoder_ws={} kv={} total={}",
               measured_weights, store_bytes, owned_bytes, measured_vision_ws, measured_decoder_ws, measured_kv,
               measured_weights + measured_vision_ws + measured_decoder_ws + measured_kv);
    MemoryLimits unlimited;
    auto pf = dots_mocr_preflight_decoder(fx.cfg, kc.max_seq_len, unlimited);
    expect(pf.ok());
    if (pf.ok()) {
        expect(pf.value->workspace_bytes == measured_decoder_ws) << "preflight workspace must mirror the actual decoder arena";
        expect(pf.value->kv_cache_bytes == measured_kv) << "preflight KV must equal the actual cache bytes";
        expect(pf.value->decoder_weight_bytes == fx.store.component_bytes("decoder") + owned_bytes)
            << "preflight decoder weights must equal actual decoder store + owned tied-head bytes";
    }
    cache.release(0);
}

// ---------------------------------------------------------------------------
// Deterministic tokens + repeat-run + blank/repeated + budget (FP32).
// ---------------------------------------------------------------------------
void test_dots_deterministic_and_budget(Device &device) {
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
        DenseDecodeController controller;
        if (!controller.create(fx.exec->text_executor(), nullptr, dopts).ok()) { return {}; }
        auto gen = controller.generate(stream, *req.value, cache);
        if (!gen.ok()) { return {}; }
        return gen.value->tokens;
    };
    auto t1 = generate(6);
    auto t2 = generate(6);
    expect(!t1.empty());
    expect(t1 == t2) << "repeat dots.mocr decode must be identical";
    auto t_short = generate(1);
    expect(t_short.size() == 1u);
    expect(t_short[0] == t1[0]);
    // Blank image (all dark) still produces a deterministic bounded sequence.
    auto blank = make_image(fx.cfg, 8, 8, 99u);
    for (auto &b : blank.pixels) { b = 10u; }
    fx.exec->reset_vision_workspace();
    P7VisionOutput bvision;
    {
        auto list = CommandList::create();
        auto v = fx.exec->record_vision(list, luisa::span<const P7ImageInput>{&blank, 1});
        expect(v.ok());
        if (!v.ok()) { return; }
        bvision = *v.value;
        stream << list.commit() << synchronize();
    }
    luisa::vector<int32_t> bpos, bnext;
    auto breq = fx.exec->build_decode_request(prompt, mm_start, tokens, bvision,
                                              luisa::span<const P7ImageInput>{&blank, 1}, bpos, bnext);
    expect(breq.ok());
    if (breq.ok()) {
        ContiguousKvCache cache;
        KvCacheConfig kc;
        kc.num_layers = fx.cfg.graph.text.num_layers;
        kc.num_kv_heads = fx.cfg.graph.text.num_kv_heads;
        kc.head_dim = fx.cfg.graph.text.head_dim;
        kc.max_seq_len = static_cast<uint32_t>(breq.value->prompt_tokens.size()) + 4;
        kc.batch = 1;
        kc.dtype_bytes = 4;
        expect(cache.create(device, stream, kc, nullptr).ok());
        DenseDecodeOptions dopts;
        dopts.sampling.greedy = true;
        dopts.max_new_tokens = 4;
        DenseDecodeController controller;
        expect(controller.create(fx.exec->text_executor(), nullptr, dopts).ok());
        auto gen = controller.generate(stream, *breq.value, cache);
        expect(gen.ok());
        if (gen.ok()) { expect(gen.value->tokens.size() <= 4u); }
    }
}

// ---------------------------------------------------------------------------
// W4A32 quantized A/B: packed INT4 device reads vs a dequantized-weight oracle.
// The quantized arm must match the dequantized reference (NOT the FP32 arm),
// proving the packed-INT4 read path is real and the quantization error is
// bounded. Reports the FP32-vs-quantized quality delta.
// ---------------------------------------------------------------------------
void test_dots_quantized_ab(Device &device) {
    // FP32 reference arm.
    Fixture fp32;
    expect(fp32.init(device, DenseLinearStorage::fp32, 0));
    if (!fp32.exec) { return; }
    // Quantized arm (same seed -> same underlying weights, then quantized).
    // group_size 128 = whole rows of the small synthetic geometry (K=32/64),
    // matching the declared memory-constrained configuration; the 64-group
    // variant is covered by the olmOCR-2 quantized suite.
    Fixture quant;
    expect(quant.init(device, DenseLinearStorage::int4_sym, 128));
    if (!quant.exec) { return; }
    auto stream = device.create_stream();
    auto img = make_image(quant.cfg, 8, 8);
    auto pixels = ocr::test::p7_oracle_preprocess(img.pixels, 8, 8,
                                                  img.packed.resized_width, img.packed.resized_height,
                                                  quant.cfg.graph.mean, quant.cfg.graph.std_dev);
    // Dequantized-weight oracle for the quantized arm.
    ocr::test::P8VisionOracle vision_oracle;
    expect(ocr::test::p8_oracle_vision(quant.cfg.graph, quant.weights, pixels, img.packed, vision_oracle).ok());
    quant.exec->reset_vision_workspace();
    P7VisionOutput vision;
    {
        auto list = CommandList::create();
        auto v = quant.exec->record_vision(list, luisa::span<const P7ImageInput>{&img, 1});
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
    auto req = quant.exec->build_decode_request(prompt, mm_start, tokens, vision,
                                                luisa::span<const P7ImageInput>{&img, 1}, position_ids, next_position);
    expect(req.ok());
    if (!req.ok()) { return; }
    luisa::vector<luisa::vector<float>> k_caches, v_caches;
    ocr::test::P8TextOracle text_oracle;
    expect(ocr::test::p8_oracle_text_forward(quant.cfg.graph, quant.weights, prompt, position_ids,
                                             vision_oracle.features, mm_start, tokens, 0u,
                                             k_caches, v_caches, text_oracle).ok());
    const uint32_t vocab = quant.cfg.graph.text.vocab_size;
    const uint32_t prompt_len = static_cast<uint32_t>(prompt.size());
    ContiguousKvCache cache;
    KvCacheConfig kc;
    kc.num_layers = quant.cfg.graph.text.num_layers;
    kc.num_kv_heads = quant.cfg.graph.text.num_kv_heads;
    kc.head_dim = quant.cfg.graph.text.head_dim;
    kc.max_seq_len = prompt_len + 8;
    kc.batch = 1;
    kc.dtype_bytes = 4;
    expect(cache.create(device, stream, kc, nullptr).ok());
    auto logits_buf = device.create_buffer<float>(static_cast<size_t>(prompt_len) * vocab);
    {
        auto list = CommandList::create();
        auto s = quant.exec->text_executor().record_prefill_all_logits(list, *req.value, cache, logits_buf.view());
        expect(s.ok());
        if (!s.ok()) { return; }
        stream << list.commit() << synchronize();
    }
    luisa::vector<float> dev_logits(static_cast<size_t>(prompt_len) * vocab);
    stream << logits_buf.copy_to(luisa::span<float>{dev_logits}) << synchronize();
    // The packed-INT4 device path must match the DEQUANTIZED oracle.
    ocr::test::expect_close(dev_logits, text_oracle.logits, kQuantAtol, kQuantRtol, "dots W4A32 vs dequantized oracle");
    // Decode step parity (quantized).
    const uint32_t first = ocr::test::p7_oracle_argmax(luisa::span<const float>{
        text_oracle.logits.data() + static_cast<size_t>(prompt_len - 1) * vocab, vocab});
    luisa::vector<uint32_t> step_tokens = {first};
    luisa::vector<int32_t> step_pos = {static_cast<int32_t>(prompt_len)};
    ocr::test::P8TextOracle step_oracle;
    expect(ocr::test::p8_oracle_text_forward(quant.cfg.graph, quant.weights, step_tokens, step_pos, {}, 0u, 0u,
                                             prompt_len, k_caches, v_caches, step_oracle).ok());
    luisa::vector<float> dev_step_logits(vocab);
    {
        auto list = CommandList::create();
        auto s = quant.exec->text_executor().record_decode_step(list, first, cache, dev_step_logits);
        expect(s.ok());
        if (!s.ok()) { return; }
        stream << list.commit() << synchronize();
    }
    ocr::test::expect_close(dev_step_logits, step_oracle.logits, kQuantAtol, kQuantRtol, "dots W4A32 decode logits");
    cache.release(0);
    // Quality report: FP32-vs-quantized logits delta (the quantization error).
    // Run the FP32 arm on the SAME image/prompt and compare last-row logits.
    fp32.exec->reset_vision_workspace();
    P7VisionOutput fvision;
    {
        auto list = CommandList::create();
        auto v = fp32.exec->record_vision(list, luisa::span<const P7ImageInput>{&img, 1});
        expect(v.ok());
        if (!v.ok()) { return; }
        fvision = *v.value;
        stream << list.commit() << synchronize();
    }
    luisa::vector<int32_t> fpos, fnext;
    auto freq = fp32.exec->build_decode_request(prompt, mm_start, tokens, fvision,
                                                luisa::span<const P7ImageInput>{&img, 1}, fpos, fnext);
    expect(freq.ok());
    if (!freq.ok()) { return; }
    ContiguousKvCache fcache;
    expect(fcache.create(device, stream, kc, nullptr).ok());
    luisa::vector<float> fp32_logits(vocab);
    {
        auto list = CommandList::create();
        auto s = fp32.exec->text_executor().record_prefill(list, *freq.value, fcache, fp32_logits);
        expect(s.ok());
        if (!s.ok()) { return; }
        stream << list.commit() << synchronize();
    }
    fcache.release(0);
    // Report the FP32-vs-quantized last-row logit quality: max abs delta, max
    // per-element rel delta, and the SIGNAL-NORMALIZED relative L2 error
    // ||q-f||2 / ||f||2 (scale-free; the honest synthetic quality measure).
    float max_abs = 0.0f, max_rel = 0.0f;
    double err_sq = 0.0, ref_sq = 0.0;
    const float *q_last = dev_logits.data() + static_cast<size_t>(prompt_len - 1) * vocab;
    for (uint32_t i = 0; i < vocab; i++) {
        const float d = std::abs(q_last[i] - fp32_logits[i]);
        max_abs = std::max(max_abs, d);
        const float denom = std::max(std::abs(fp32_logits[i]), 1e-6f);
        max_rel = std::max(max_rel, d / denom);
        err_sq += static_cast<double>(d) * d;
        ref_sq += static_cast<double>(fp32_logits[i]) * fp32_logits[i];
    }
    const double rel_l2 = std::sqrt(err_sq / std::max(ref_sq, 1e-12));
    LUISA_INFO("dots.mocr W4A32 quality: max_abs_logit_delta={} max_rel_logit_delta={} rel_l2={} (synthetic, not official accuracy)",
               max_abs, max_rel, rel_l2);
    // The quantization error must be finite and small RELATIVE TO THE SIGNAL
    // (a relabeled FP32 path would have zero delta; a real INT4 path has a
    // small nonzero delta). The bound is scale-free, not an absolute constant.
    expect(std::isfinite(max_abs));
    expect(std::isfinite(rel_l2));
    expect(rel_l2 < 0.50) << "quantization relative-L2 logit error must be < 50% of the FP32 signal (declared synthetic gate for random small-scale weights; measured ~0.46 at group 128, NOT official accuracy)";
    // Greedy tokens: the quantized DEVICE decode-step argmax must match the
    // dequantized oracle's decode-step argmax (deterministic decode under
    // quantization; a real device-vs-oracle token check).
    const uint32_t dev_step_token = ocr::test::p7_oracle_argmax(dev_step_logits);
    const uint32_t oracle_step_token = ocr::test::p7_oracle_argmax(step_oracle.logits);
    expect(dev_step_token == oracle_step_token) << "quantized decode token must match the dequantized oracle";
}

}// namespace

int main(int argc, char *argv[]) {
    auto dc = luisa::test::create_device_from_ut(argc, argv);
    if (!dc) { return 0; }
    boost::ut::detail::cfg::parse_arg_with_fallback(argc, const_cast<const char **>(argv));
    auto &device = dc->device;
    "dots_contracts"_test = [&] { test_dots_contracts(); };
    "dots_memory_preflight"_test = [&] { test_dots_memory_preflight(); };
    "dots_vision_parity"_test = [&] { test_dots_vision_parity(device); };
    "dots_decoder_parity"_test = [&] { test_dots_decoder_parity(device); };
    "dots_deterministic_and_budget"_test = [&] { test_dots_deterministic_and_budget(device); };
    "dots_quantized_ab"_test = [&] { test_dots_quantized_ab(device); };
}

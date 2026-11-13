// Test for ocr/models/olm_ocr2 + the shared dense executor (device) -- the
// SMALL SYNTHETIC olmOCR-2 vertical slice: Qwen2.5-VL window/global vision +
// spatial-merge RMSNorm merger + Qwen2.5-VL decoder (QKV bias, no QK-Norm,
// rotate-half M-RoPE). Covers: config/tensor/signature contracts, the declared
// 7B geometry, English-only enforcement, INT4 requirement + strict memory
// preflight (fail BEFORE allocation), host-offload refusal, vision/merger
// parity, decoder logits/KV parity, deterministic tokens, and the quantized
// decode A/B arm. Synthetic weights only; no model/download, no real-model
// parity claim, no claim that a real 7B model fits.
#include "ut/ut.hpp"
#include "test_device.h"
#include "ocr_p8_test_utils.h"

#include "ocr/models/olm_ocr2.h"

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
constexpr float kQuantAtol = 3e-3f;
constexpr float kQuantRtol = 3e-3f;

struct Fixture {
    ocr::test::P7TempDir tmp;
    OlmOcr2Config cfg;
    ocr::test::P8SyntheticWeights weights;
    WeightStore store;
    luisa::unique_ptr<P7VlmExecutor> exec;

    [[nodiscard]] bool init(Device &device, DenseLinearStorage storage = DenseLinearStorage::fp32,
                            uint32_t group_size = 0, uint32_t seed = 0x5EEDu) {
        cfg = olm_ocr2_synthetic_config();
        cfg.linear_storage = storage;
        cfg.linear_group_size = group_size;
        cfg.graph.linear_storage = storage;
        cfg.graph.linear_group_size = group_size;
        if (storage != DenseLinearStorage::fp32) { cfg.graph.tie_word_embeddings = false; }
        if (!cfg.validate().ok()) { return false; }
        weights = ocr::test::generate_p8_weights(cfg.graph, storage, group_size, seed);
        const auto vpath = tmp.path / "olm_vision.lcocr";
        const auto dpath = tmp.path / "olm_decoder.lcocr";
        if (auto s = ocr::test::write_p8_packages(vpath, dpath, cfg.graph, cfg.names, weights, "olm_ocr_2"); !s.ok()) {
            LUISA_WARNING("write_p8_packages: {}", s.message);
            return false;
        }
        // S1: the package-load gate validates BOTH component tables on disk
        // (vision FP32-only; decoder per-linear storage) BEFORE any upload.
        {
            auto vr = PackageReader::open(vpath);
            auto dr = PackageReader::open(dpath);
            if (!vr.ok() || !dr.ok()) { return false; }
            if (auto s = olm_ocr2_validate_packages(cfg, (*vr.value)->tensors(), (*dr.value)->tensors()); !s.ok()) {
                LUISA_WARNING("olm_ocr2 package gate: {}", s.message);
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
            LUISA_WARNING("olm create: {}", s.message);
            return false;
        }
        return true;
    }
};

[[nodiscard]] P7ImageInput make_image(const OlmOcr2Config &cfg, uint32_t w, uint32_t h, uint32_t seed = 11u) {
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
// Host contract checks: identity, declared 7B geometry, English-only, INT4.
// ---------------------------------------------------------------------------
void test_olm_contracts() {
    auto cfg = olm_ocr2_synthetic_config();
    expect(cfg.validate().ok());
    expect(cfg.graph.vision.mask_kind == P7VisionMaskKind::window_global);
    expect(cfg.graph.connector.kind == P7ConnectorKind::spatial_mlp);
    expect(cfg.graph.connector.rms_norm);
    expect(cfg.graph.text_pos == P7TextPosKind::mrope_interleaved);
    expect(cfg.graph.text_qwen2_5_contract);
    expect(cfg.graph.text.rope_pair_layout == RopePairLayout::rotate_half);
    // The DECLARED 7B geometry (a contract, NOT a fit claim).
    auto declared = olm_ocr2_declared_decoder_config();
    expect(declared.num_layers == 28u);
    expect(declared.num_attention_heads == 28u);
    expect(declared.num_kv_heads == 4u);
    expect(declared.hidden_size == 3584u);
    expect(declared.validate().ok());
    auto expected = olm_ocr2_expected_tensors(cfg);
    expect(expected.size() > 30u);
    expect(olm_ocr2_validate_real_artifact("r", "f").code == StatusCode::unvalidated);
    OlmOcr2Signature sig;
    expect(olm_ocr2_validate_signature(cfg, sig).code == StatusCode::unvalidated);
    sig.checkpoint_revision = "rev";
    sig.config_fingerprint = "fp";
    for (const auto &e : expected) { if (e.required) { sig.tensor_names.push_back(e.name); } }
    expect(olm_ocr2_validate_signature(cfg, sig).ok());
    sig.tensor_names.push_back("visual.bogus.weight");
    expect(olm_ocr2_validate_signature(cfg, sig).code == StatusCode::unvalidated);
    // English-only enforcement.
    expect(olm_ocr2_check_language("en").ok());
    expect(olm_ocr2_check_language("").ok());
    expect(!olm_ocr2_check_language("zh").ok());
    expect(!olm_ocr2_check_language("multi").ok());
    expect(!olm_ocr2_check_language("xx").ok());
    // The declared 8 GB path REQUIRES INT4: FP32 with require_int4 is refused.
    auto int4_cfg = olm_ocr2_synthetic_config();
    int4_cfg.require_int4 = true;
    int4_cfg.linear_storage = DenseLinearStorage::fp32;
    expect(!int4_cfg.validate().ok());
    int4_cfg.linear_storage = DenseLinearStorage::int4_sym;
    int4_cfg.linear_group_size = 128;
    int4_cfg.graph.tie_word_embeddings = false;
    expect(int4_cfg.validate().ok());
    // Host offload fails closed.
    auto offload_cfg = olm_ocr2_synthetic_config();
    offload_cfg.offload = OlmOffloadPolicy::explicit_host;
    expect(!offload_cfg.validate().ok());
    // A Qwen3 config is refused.
    const char *qwen3_cfg = R"({"model_type":"qwen3_vl","vision_config":{"num_layers":2,"hidden_size":16,"num_heads":2,"head_dim":8,"intermediate_size":32,"patch_size":2,"window_patch_size":2,"global_layers":[1]},"text_config":{"vocab_size":32,"hidden_size":32,"num_hidden_layers":3,"num_attention_heads":4,"num_key_value_heads":2,"intermediate_size":64,"max_position_embeddings":256},"mrope_section":[2,1,1]})";
    expect(!parse_olm_ocr2_config(qwen3_cfg).ok());
    // A non-English language declaration is refused.
    const char *zh_cfg = R"({"model_type":"qwen2_5_vl","languages":["zh"],"vision_config":{"num_layers":2,"hidden_size":16,"num_heads":2,"head_dim":8,"intermediate_size":32,"patch_size":2,"window_patch_size":2,"global_layers":[1]},"text_config":{"vocab_size":32,"hidden_size":32,"num_hidden_layers":3,"num_attention_heads":4,"num_key_value_heads":2,"intermediate_size":64,"max_position_embeddings":256},"mrope_section":[2,1,1]})";
    expect(!parse_olm_ocr2_config(zh_cfg).ok());
    const char *good_cfg = R"({"model_type":"qwen2_5_vl","languages":["en"],"vision_config":{"num_layers":2,"hidden_size":16,"num_heads":2,"head_dim":8,"intermediate_size":32,"patch_size":2,"window_patch_size":2,"global_layers":[1]},"text_config":{"vocab_size":32,"hidden_size":32,"num_hidden_layers":3,"num_attention_heads":4,"num_key_value_heads":2,"intermediate_size":64,"max_position_embeddings":256},"mrope_section":[2,1,1],"quantization":{"linear_storage":"int4_sym","group_size":128}})";
    auto parsed = parse_olm_ocr2_config(good_cfg);
    expect(parsed.ok());
    if (parsed.ok()) { expect(parsed.value->validate().ok()); }
    // S4: languages present but NOT an array is refused (scalar/object/null).
    const char *scalar_lang = R"({"model_type":"qwen2_5_vl","languages":"en","vision_config":{"num_layers":2,"hidden_size":16,"num_heads":2,"head_dim":8,"intermediate_size":32,"patch_size":2,"window_patch_size":2,"global_layers":[1]},"text_config":{"vocab_size":32,"hidden_size":32,"num_hidden_layers":3,"num_attention_heads":4,"num_key_value_heads":2,"intermediate_size":64,"max_position_embeddings":256},"mrope_section":[2,1,1]})";
    expect(!parse_olm_ocr2_config(scalar_lang).ok());
    const char *obj_lang = R"({"model_type":"qwen2_5_vl","languages":{"lang":"en"},"vision_config":{"num_layers":2,"hidden_size":16,"num_heads":2,"head_dim":8,"intermediate_size":32,"patch_size":2,"window_patch_size":2,"global_layers":[1]},"text_config":{"vocab_size":32,"hidden_size":32,"num_hidden_layers":3,"num_attention_heads":4,"num_key_value_heads":2,"intermediate_size":64,"max_position_embeddings":256},"mrope_section":[2,1,1]})";
    expect(!parse_olm_ocr2_config(obj_lang).ok());
    const char *null_lang = R"({"model_type":"qwen2_5_vl","languages":null,"vision_config":{"num_layers":2,"hidden_size":16,"num_heads":2,"head_dim":8,"intermediate_size":32,"patch_size":2,"window_patch_size":2,"global_layers":[1]},"text_config":{"vocab_size":32,"hidden_size":32,"num_hidden_layers":3,"num_attention_heads":4,"num_key_value_heads":2,"intermediate_size":64,"max_position_embeddings":256},"mrope_section":[2,1,1]})";
    expect(!parse_olm_ocr2_config(null_lang).ok());
    const char *num_entry = R"({"model_type":"qwen2_5_vl","languages":[42],"vision_config":{"num_layers":2,"hidden_size":16,"num_heads":2,"head_dim":8,"intermediate_size":32,"patch_size":2,"window_patch_size":2,"global_layers":[1]},"text_config":{"vocab_size":32,"hidden_size":32,"num_hidden_layers":3,"num_attention_heads":4,"num_key_value_heads":2,"intermediate_size":64,"max_position_embeddings":256},"mrope_section":[2,1,1]})";
    expect(!parse_olm_ocr2_config(num_entry).ok());
    // S1: package-load gate (vision FP32 + decoder INT4 accepted; visual INT4
    // refused; decoder linear left FP32 under INT4 refused).
    auto qcfg = olm_ocr2_synthetic_config();
    qcfg.linear_storage = DenseLinearStorage::int4_sym;
    qcfg.linear_group_size = 64;
    qcfg.graph.tie_word_embeddings = false;
    expect(qcfg.validate().ok());
    const auto make_tables = [](const OlmOcr2Config &c, luisa::vector<PackageTensor> &vis,
                                luisa::vector<PackageTensor> &dec) {
        for (const auto &e : olm_ocr2_expected_tensors(c)) {
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
    luisa::vector<PackageTensor> qvis, qdec;
    make_tables(qcfg, qvis, qdec);
    for (auto &t : qdec) {
        if (dense_linear_is_quantizable(t.name, "model.", "lm_head.weight")) {
            t.encoding = TensorEncoding::int4_packed;
            t.quant_kind = QuantKind::int4_symmetric;
            t.group_size = 64;
        }
    }
    expect(olm_ocr2_validate_packages(qcfg, qvis, qdec).ok());
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
    expect(!olm_ocr2_validate_packages(qcfg, bad_vis, qdec).ok());
    auto bad_dec = qdec;
    for (auto &t : bad_dec) {
        if (t.name == "model.layers.0.self_attn.q_proj.weight") { t.encoding = TensorEncoding::fp32; t.quant_kind = QuantKind::none; t.group_size = 0; }
    }
    expect(!olm_ocr2_validate_packages(qcfg, qvis, bad_dec).ok());
}

// ---------------------------------------------------------------------------
// Strict memory preflight: fail BEFORE allocation + host-offload refusal.
// ---------------------------------------------------------------------------
void test_olm_memory_preflight() {
    auto cfg = olm_ocr2_synthetic_config();
    MemoryLimits unlimited;
    auto report = olm_ocr2_preflight(cfg, 64, unlimited);
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
    // A tiny budget fails BEFORE allocation with a useful report.
    MemoryLimits tiny;
    tiny.max_device_bytes = 1024;
    auto refused = olm_ocr2_preflight(cfg, 64, tiny);
    expect(!refused.ok());
    expect(refused.status.code == StatusCode::out_of_memory);
    // Host offload fails closed BEFORE any accounting/allocation.
    auto offload_cfg = olm_ocr2_synthetic_config();
    offload_cfg.offload = OlmOffloadPolicy::explicit_host;
    auto offload_refused = olm_ocr2_preflight(offload_cfg, 64, unlimited);
    expect(!offload_refused.ok());
    expect(offload_refused.status.code == StatusCode::unsupported);
    // The DECLARED 7B geometry: INT4 preflight reports a large requirement and
    // refuses an 8 GiB budget at a long context (NO claim that 7B fits).
    auto declared_cfg = olm_ocr2_synthetic_config();
    declared_cfg.graph.text = olm_ocr2_declared_decoder_config();
    // M-RoPE sections must sum to head_dim/2 = 64 for the declared geometry.
    declared_cfg.graph.pos_sections = {32u, 16u, 16u};
    declared_cfg.graph.tie_word_embeddings = false;
    declared_cfg.linear_storage = DenseLinearStorage::int4_sym;
    declared_cfg.linear_group_size = 128;
    declared_cfg.require_int4 = true;
    auto declared_report = olm_ocr2_preflight(declared_cfg, 32768, unlimited);
    expect(declared_report.ok());
    if (declared_report.ok()) {
        LUISA_INFO("olmOCR-2 declared 7B INT4 preflight: total_device_bytes={} (weights vision {} + decoder {}, workspace {}, KV {})",
                   declared_report.value->total_device_bytes, declared_report.value->vision_weight_bytes,
                   declared_report.value->decoder_weight_bytes, declared_report.value->workspace_bytes,
                   declared_report.value->kv_cache_bytes);
        MemoryLimits eight_gib;
        eight_gib.max_device_bytes = 8ull << 30;
        auto eight_refused = olm_ocr2_preflight(declared_cfg, 32768, eight_gib);
        // At 32768 context the 7B KV + weights exceed 8 GiB -> refused.
        expect(!eight_refused.ok());
        expect(eight_refused.status.code == StatusCode::out_of_memory);
    }
}

// ---------------------------------------------------------------------------
// Vision + RMSNorm merger parity (FP32, Qwen2.5-VL contract).
// ---------------------------------------------------------------------------
void test_olm_vision_parity(Device &device) {
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
        const auto what = luisa::format("olm {}x{}", w, h);
        ocr::test::expect_close(dev_patches, oracle.patches, kLinAtol, kLinRtol, (what + " patches").c_str());
        ocr::test::expect_close(dev_final, oracle.block_out.back(), kAtol, kRtol, (what + " vision final").c_str());
        ocr::test::expect_close(dev_features, oracle.features, kAtol, kRtol, (what + " RMSNorm merger").c_str());
        expect(vision.total_tokens == static_cast<uint32_t>(img.packed.visual_tokens));
    }
}

// ---------------------------------------------------------------------------
// Decoder logits/KV parity + deterministic tokens (FP32, rotate-half M-RoPE).
// ---------------------------------------------------------------------------
void test_olm_decoder_parity(Device &device) {
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
    // The Qwen2.5-VL contract uses rotate-half M-RoPE with QKV bias (no QK-Norm).
    expect(fx.exec->text_executor().config().qkv_bias);
    expect(!fx.exec->text_executor().config().qk_norm);
    expect(fx.exec->text_executor().config().rope_axes == 3u);
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
        auto s = fx.exec->text_executor().record_prefill_positions(list, *req.value, position_ids, {},
                                                                   cache, true, {}, logits_buf.view());
        expect(s.ok());
        if (!s.ok()) { return; }
        stream << list.commit() << synchronize();
    }
    luisa::vector<float> dev_logits(static_cast<size_t>(prompt_len) * vocab);
    stream << logits_buf.copy_to(luisa::span<float>{dev_logits}) << synchronize();
    ocr::test::expect_close(dev_logits, text_oracle.logits, kAtol, kRtol, "olm full prefill logits");
    const uint32_t first = ocr::test::p7_oracle_argmax(luisa::span<const float>{
        text_oracle.logits.data() + static_cast<size_t>(prompt_len - 1) * vocab, vocab});
    luisa::vector<uint32_t> step_tokens = {first};
    ocr::test::P8TextOracle step_oracle;
    expect(ocr::test::p8_oracle_text_forward(fx.cfg.graph, fx.weights, step_tokens, next_position, {}, 0u, 0u,
                                             prompt_len, k_caches, v_caches, step_oracle).ok());
    luisa::vector<float> dev_step_logits(vocab);
    {
        auto list = CommandList::create();
        auto s = fx.exec->text_executor().record_decode_step_positions(list, first, next_position, cache, dev_step_logits);
        expect(s.ok());
        if (!s.ok()) { return; }
        stream << list.commit() << synchronize();
    }
    ocr::test::expect_close(dev_step_logits, step_oracle.logits, kAtol, kRtol, "olm decode step logits");
    luisa::vector<float> dev_k, dev_v;
    expect(cache.download_layer(stream, 0, 0, dev_k, dev_v).ok());
    stream << synchronize();
    ocr::test::expect_close(dev_k, step_oracle.k_cache[0], kAtol, kRtol, "olm decode K cache");
    ocr::test::expect_close(dev_v, step_oracle.v_cache[0], kAtol, kRtol, "olm decode V cache");
    // MEASURED synthetic memory (actual device buffers, not estimates). S2:
    // include the EXECUTOR-OWNED tied-head transpose in weight residency.
    const uint64_t store_bytes = fx.store.total_bytes();
    const uint64_t owned_bytes = fx.exec->text_executor().owned_device_bytes();
    const uint64_t measured_weights = store_bytes + owned_bytes;
    const uint64_t measured_vision_ws = fx.exec->vision_workspace_bytes();
    const uint64_t measured_decoder_ws = fx.exec->decoder_workspace_bytes();
    const uint64_t measured_kv = cache.device_bytes();
    expect(fx.exec->weight_bytes() == measured_weights) << "executor weight_bytes must equal store + owned bytes";
    LUISA_INFO("olmOCR-2 synthetic MEASURED device bytes: weights={} (store {} + owned tied-head {}) vision_ws={} decoder_ws={} kv={} total={}",
               measured_weights, store_bytes, owned_bytes, measured_vision_ws, measured_decoder_ws, measured_kv,
               measured_weights + measured_vision_ws + measured_decoder_ws + measured_kv);
    MemoryLimits unlimited;
    auto pf = olm_ocr2_preflight(fx.cfg, kc.max_seq_len, unlimited);
    expect(pf.ok());
    if (pf.ok()) {
        expect(pf.value->workspace_bytes == measured_vision_ws + measured_decoder_ws)
            << "preflight workspace must mirror the actual vision+decoder arenas";
        expect(pf.value->kv_cache_bytes == measured_kv) << "preflight KV must equal the actual cache bytes";
        expect(pf.value->decoder_weight_bytes == fx.store.component_bytes("decoder") + owned_bytes)
            << "preflight decoder weights must equal actual decoder store + owned tied-head bytes";
        expect(pf.value->vision_weight_bytes == fx.store.component_bytes("vision"))
            << "preflight vision weights must equal the actual vision store bytes";
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
        dopts.prompt_position_ids = position_ids;
        dopts.next_position = next_position;
        DenseDecodeController controller;
        if (!controller.create(fx.exec->text_executor(), nullptr, dopts).ok()) { return {}; }
        auto gen = controller.generate(stream, *req.value, c2);
        if (!gen.ok()) { return {}; }
        return gen.value->tokens;
    };
    auto t1 = generate(5);
    auto t2 = generate(5);
    expect(!t1.empty());
    expect(t1 == t2) << "repeat olmOCR-2 decode must be identical";
}

// ---------------------------------------------------------------------------
// Quantized decode A/B: packed INT4 device reads vs a dequantized oracle.
// ---------------------------------------------------------------------------
void test_olm_quantized_decode(Device &device) {
    Fixture quant;
    expect(quant.init(device, DenseLinearStorage::int4_sym, 64));
    if (!quant.exec) { return; }
    auto stream = device.create_stream();
    auto img = make_image(quant.cfg, 8, 8);
    auto pixels = ocr::test::p7_oracle_preprocess(img.pixels, 8, 8,
                                                  img.packed.resized_width, img.packed.resized_height,
                                                  quant.cfg.graph.mean, quant.cfg.graph.std_dev);
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
        auto s = quant.exec->text_executor().record_prefill_positions(list, *req.value, position_ids, {},
                                                                      cache, true, {}, logits_buf.view());
        expect(s.ok());
        if (!s.ok()) { return; }
        stream << list.commit() << synchronize();
    }
    luisa::vector<float> dev_logits(static_cast<size_t>(prompt_len) * vocab);
    stream << logits_buf.copy_to(luisa::span<float>{dev_logits}) << synchronize();
    ocr::test::expect_close(dev_logits, text_oracle.logits, kQuantAtol, kQuantRtol, "olm W4A32 vs dequantized oracle");
    // Quantized decode step.
    const uint32_t first = ocr::test::p7_oracle_argmax(luisa::span<const float>{
        text_oracle.logits.data() + static_cast<size_t>(prompt_len - 1) * vocab, vocab});
    luisa::vector<uint32_t> step_tokens = {first};
    ocr::test::P8TextOracle step_oracle;
    expect(ocr::test::p8_oracle_text_forward(quant.cfg.graph, quant.weights, step_tokens, next_position, {}, 0u, 0u,
                                             prompt_len, k_caches, v_caches, step_oracle).ok());
    luisa::vector<float> dev_step_logits(vocab);
    {
        auto list = CommandList::create();
        auto s = quant.exec->text_executor().record_decode_step_positions(list, first, next_position, cache, dev_step_logits);
        expect(s.ok());
        if (!s.ok()) { return; }
        stream << list.commit() << synchronize();
    }
    ocr::test::expect_close(dev_step_logits, step_oracle.logits, kQuantAtol, kQuantRtol, "olm W4A32 decode logits");
    cache.release(0);
    // Deterministic quantized greedy tokens.
    const auto generate = [&](uint32_t max_new) -> luisa::vector<uint32_t> {
        ContiguousKvCache c2;
        KvCacheConfig kc2 = kc;
        kc2.max_seq_len = prompt_len + max_new;
        if (!c2.create(device, stream, kc2, nullptr).ok()) { return {}; }
        DenseDecodeOptions dopts;
        dopts.sampling.greedy = true;
        dopts.max_new_tokens = max_new;
        dopts.prompt_position_ids = position_ids;
        dopts.next_position = next_position;
        DenseDecodeController controller;
        if (!controller.create(quant.exec->text_executor(), nullptr, dopts).ok()) { return {}; }
        auto gen = controller.generate(stream, *req.value, c2);
        if (!gen.ok()) { return {}; }
        return gen.value->tokens;
    };
    auto t1 = generate(5);
    auto t2 = generate(5);
    expect(!t1.empty());
    expect(t1 == t2) << "repeat quantized olmOCR-2 decode must be identical";
}

}// namespace

int main(int argc, char *argv[]) {
    auto dc = luisa::test::create_device_from_ut(argc, argv);
    if (!dc) { return 0; }
    boost::ut::detail::cfg::parse_arg_with_fallback(argc, const_cast<const char **>(argv));
    auto &device = dc->device;
    "olm_contracts"_test = [&] { test_olm_contracts(); };
    "olm_memory_preflight"_test = [&] { test_olm_memory_preflight(); };
    "olm_vision_parity"_test = [&] { test_olm_vision_parity(device); };
    "olm_decoder_parity"_test = [&] { test_olm_decoder_parity(device); };
    "olm_quantized_decode"_test = [&] { test_olm_quantized_decode(device); };
}

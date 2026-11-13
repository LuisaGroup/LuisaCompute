// OCR engine -- OvisOCR2 graph contract (Phase 9). Host-only TU.
// Synthetic recipe-driven contract; real artifacts fail closed. The executing
// graph is runtime/hybrid_gdn_executor.* (HybridVlmGraph). See MODELS.md.
#include "ocr/models/ovis_ocr2.h"

#include <utility>

#include <luisa/core/stl/format.h>
#include <yyjson.h>

#include "ocr/quantization.h"
#include "ocr/runtime/layout_result.h"

namespace luisa::ocr {

namespace {
struct JsonDoc {
    yyjson_doc *doc{nullptr};
    ~JsonDoc() { if (doc != nullptr) { yyjson_doc_free(doc); } }
};
[[nodiscard]] Result<uint32_t> get_u32(yyjson_val *obj, const char *key, bool required, uint32_t fallback) {
    using R = Result<uint32_t>;
    auto *v = yyjson_obj_get(obj, key);
    if (v == nullptr || yyjson_is_null(v)) {
        if (required) { return R::failure(StatusCode::invalid_argument, luisa::format("config key '{}' is required", key), key); }
        return R::success(fallback);
    }
    if (!yyjson_is_uint(v)) { return R::failure(StatusCode::invalid_argument, luisa::format("config key '{}' must be a nonnegative integer", key), key); }
    const auto value = yyjson_get_uint(v);
    if (value > 0xFFFFFFFFull) { return R::failure(StatusCode::invalid_argument, luisa::format("config key '{}' exceeds uint32", key), key); }
    return R::success(static_cast<uint32_t>(value));
}
}// namespace

// ---------------------------------------------------------------------------
// Config validation
// ---------------------------------------------------------------------------

[[nodiscard]] Status OvisOcr2Config::validate() const {
    auto s = vision.validate();
    if (!s.ok()) { return s; }
    s = decoder.validate();
    if (!s.ok()) { return s; }
    if (vision.text_pos != P7TextPosKind::rope1d) {
        return Status::error(StatusCode::invalid_argument,
                             "the OvisOCR2 hybrid text decoder is 1-axis; vision.text_pos must be rope1d", "text_pos");
    }
    if (vision.text.hidden_size != decoder.decoder.hidden_size ||
        vision.text.vocab_size != decoder.decoder.vocab_size) {
        return Status::error(StatusCode::invalid_argument,
                             "vision.text hidden/vocab must match the hybrid decoder (connector output)", "hidden/vocab");
    }
    if (max_seq_len == 0u) {
        return Status::error(StatusCode::invalid_argument, "max_seq_len must be nonzero", "max_seq_len");
    }
    return Status::success();
}

// ---------------------------------------------------------------------------
// Declared full contract + schedule validation
// ---------------------------------------------------------------------------

[[nodiscard]] HybridGdnExecutorConfig ovis_ocr2_declared_decoder_config() {
    // DECLARED full Qwen3.5-0.8B-style hybrid geometry (NOT a verified
    // checkpoint): 24 layers, 18 GDN + 6 full (full at every 4th layer). The
    // hidden/head/intermediate values are a consistent declared geometry used
    // to validate the 24/18/6 schedule explicitly.
    HybridGdnExecutorConfig cfg;
    auto &d = cfg.decoder;
    d.vocab_size = 151936;
    d.hidden_size = 1024;
    d.num_layers = 24;
    d.num_attention_heads = 8;
    d.num_kv_heads = 2;
    d.head_dim = 128;
    d.intermediate_size = 3072;
    d.rms_norm_eps = 1e-6f;
    d.rope_theta = 1000000.0f;
    d.rope_pair_layout = RopePairLayout::rotate_half;
    d.max_position_embeddings = 32768;
    d.eos_token_ids = {151645u};
    cfg.gdn.num_heads = 8;
    cfg.gdn.head_dim = 128;
    cfg.gdn.conv_kernel_size = 4;
    cfg.gdn.l2_eps = 1e-6f;
    cfg.gdn.norm_eps = 1e-6f;
    cfg.gdn.softplus_threshold = 20.0f;
    cfg.layer_kinds.assign(24, HybridLayerKind::gdn);
    for (uint32_t l = 3; l < 24; l += 4) { cfg.layer_kinds[l] = HybridLayerKind::full_attention; }
    cfg.tie_word_embeddings = true;
    cfg.qkv_bias = true;
    cfg.qk_norm = false;
    cfg.component = "decoder";
    return cfg;
}

[[nodiscard]] Status ovis_ocr2_validate_schedule(const luisa::vector<HybridLayerKind> &kinds) {
    if (kinds.size() != 24u) {
        return Status::error(StatusCode::invalid_argument,
                             luisa::format("OvisOCR2 schedule must have exactly 24 layers (got {})", kinds.size()), "layer_kinds");
    }
    uint32_t gdn = 0, full = 0;
    for (uint32_t l = 0; l < 24u; l++) {
        const bool want_full = (l % 4u == 3u);// full at 3,7,11,15,19,23
        const bool is_full = kinds[l] == HybridLayerKind::full_attention;
        if (is_full != want_full) {
            return Status::error(StatusCode::invalid_argument,
                                 luisa::format("OvisOCR2 layer {} must be {} (3:1 GDN:full schedule)", l,
                                               want_full ? "full_attention" : "gdn"),
                                 "layer_kinds");
        }
        if (is_full) { full++; } else { gdn++; }
    }
    if (gdn != 18u || full != 6u) {
        return Status::error(StatusCode::invalid_argument,
                             luisa::format("OvisOCR2 schedule must be 18 GDN + 6 full (got {} GDN + {} full)", gdn, full),
                             "layer_kinds");
    }
    return Status::success();
}

// ---------------------------------------------------------------------------
// Small synthetic slice
// ---------------------------------------------------------------------------

[[nodiscard]] OvisOcr2Config ovis_ocr2_synthetic_config() {
    OvisOcr2Config cfg;
    auto &g = cfg.vision;
    // ViT vision (small synthetic): 2D axial RoPE, bidirectional, GELU-erf.
    g.vision.num_layers = 2;
    g.vision.hidden_size = 16;
    g.vision.num_heads = 2;
    g.vision.head_dim = 8;
    g.vision.intermediate_size = 32;
    g.vision.patch_size = 2;
    g.vision.channels = 3;
    g.vision.layer_norm_eps = 1e-5f;
    g.vision.qkv_bias = true;
    g.vision.qk_norm = false;
    g.vision.mlp_act = 0;// GELU-erf
    g.vision.rope_kind = P7VisionRopeKind::axial2d;
    g.vision.rope_theta = 10000.0f;
    g.vision.rope_height_pairs = 2;
    g.vision.mask_kind = P7VisionMaskKind::bidirectional;
    g.vision.position_rows = 0;
    // Spatial-merge patch merger + LN + 2-layer GELU MLP.
    g.connector.kind = P7ConnectorKind::spatial_mlp;
    g.connector.intermediate_size = 32;
    g.connector.layer_norm_eps = 1e-5f;
    g.connector.spatial_factor = 2;
    // The vision.text fields describe the connector OUTPUT (must match decoder).
    g.text.vocab_size = 24;
    g.text.hidden_size = 16;
    g.text.num_layers = 6;
    g.text.num_attention_heads = 4;
    g.text.num_kv_heads = 2;
    g.text.head_dim = 4;
    g.text.intermediate_size = 32;
    g.text.rms_norm_eps = 1e-5f;
    g.text.rope_theta = 10000.0f;
    g.text.rope_pair_layout = RopePairLayout::adjacent;
    g.text.max_position_embeddings = 64;
    g.text.eos_token_ids = {23u};
    g.tie_word_embeddings = true;
    g.text_pos = P7TextPosKind::rope1d;
    g.pos_sections.clear();
    g.packing.patch_size = 2;
    g.packing.min_pixels = 0;
    g.packing.max_pixels = 4096;
    g.packing.grid_multiple = 2;
    g.vision_component = "vision";
    g.decoder_component = "decoder";
    g.max_vision_patches = 4096;
    g.prefill_chunk_rows = 64;
    cfg.vision_names.text.prefix = "model.";
    cfg.vision_names.visual_prefix = "visual.";
    // SCALED hybrid decoder: 6 layers = 4 GDN + 2 full (full at 3,5 -- a 3:1-ish
    // scaled schedule; the FULL 24/18/6 contract is validated separately).
    auto &d = cfg.decoder.decoder;
    d.vocab_size = 24;
    d.hidden_size = 16;
    d.num_layers = 6;
    d.num_attention_heads = 4;
    d.num_kv_heads = 2;
    d.head_dim = 4;
    d.intermediate_size = 32;
    d.rms_norm_eps = 1e-5f;
    d.rope_theta = 10000.0f;
    d.rope_pair_layout = RopePairLayout::adjacent;
    d.max_position_embeddings = 64;
    d.eos_token_ids = {23u};
    cfg.decoder.gdn.num_heads = 2;
    cfg.decoder.gdn.head_dim = 4;
    cfg.decoder.gdn.conv_kernel_size = 4;
    cfg.decoder.gdn.l2_eps = 1e-6f;
    cfg.decoder.gdn.norm_eps = 1e-5f;
    cfg.decoder.gdn.softplus_threshold = 20.0f;
    cfg.decoder.layer_kinds = {HybridLayerKind::gdn, HybridLayerKind::gdn, HybridLayerKind::gdn,
                               HybridLayerKind::full_attention, HybridLayerKind::gdn, HybridLayerKind::full_attention};
    cfg.decoder.tie_word_embeddings = true;
    cfg.decoder.qkv_bias = true;
    cfg.decoder.qk_norm = false;
    cfg.decoder.component = "decoder";
    cfg.decoder.prefill_chunk_rows = 64;
    cfg.decoder_names.dense.prefix = "model.";
    cfg.max_seq_len = 64;
    return cfg;
}

// ---------------------------------------------------------------------------
// Strict config parsing (synthetic schema)
// ---------------------------------------------------------------------------

[[nodiscard]] Result<OvisOcr2Config> parse_ovis_ocr2_config(luisa::string_view json) {
    using R = Result<OvisOcr2Config>;
    JsonDoc doc;
    doc.doc = yyjson_read(json.data(), json.size(), 0);
    if (doc.doc == nullptr) { return R::failure(StatusCode::invalid_argument, "ovis_ocr2 config is not valid JSON", "config"); }
    auto *root = yyjson_doc_get_root(doc.doc);
    if (root == nullptr || !yyjson_is_obj(root)) { return R::failure(StatusCode::invalid_argument, "ovis_ocr2 config must be a JSON object", "config"); }
    if (auto *mt = yyjson_obj_get(root, "model_type"); mt != nullptr && yyjson_is_str(mt)) {
        const luisa::string_view type{yyjson_get_str(mt), yyjson_get_len(mt)};
        if (type.find("ovis") == luisa::string_view::npos) {
            return R::failure(StatusCode::unsupported, "outer model_type does not name OvisOCR2 (and is never sufficient for identity)", "model_type");
        }
    }
    for (const char *marker : {"num_experts", "moe", "grpo", "mtp_num_layers"}) {
        if (auto *v = yyjson_obj_get(root, marker); v != nullptr && !yyjson_is_null(v) && !yyjson_is_false(v)) {
            return R::failure(StatusCode::unsupported, luisa::format("ovis_ocr2 config field '{}' is outside the hybrid GDN contract", marker), marker);
        }
    }
    OvisOcr2Config cfg = ovis_ocr2_synthetic_config();
    auto *dc = yyjson_obj_get(root, "decoder_config");
    if (dc == nullptr || !yyjson_is_obj(dc)) { return R::failure(StatusCode::invalid_argument, "decoder_config object is required", "decoder_config"); }
    auto u = [&](const char *k, bool req, uint32_t fb, uint32_t &out) -> Status {
        auto r = get_u32(dc, k, req, fb);
        if (!r.ok()) { return r.status; }
        out = *r.value;
        return Status::success();
    };
    if (auto s = u("vocab_size", true, 0, cfg.decoder.decoder.vocab_size); !s.ok()) { return R::failure(s); }
    if (auto s = u("hidden_size", true, 0, cfg.decoder.decoder.hidden_size); !s.ok()) { return R::failure(s); }
    if (auto s = u("num_hidden_layers", true, 0, cfg.decoder.decoder.num_layers); !s.ok()) { return R::failure(s); }
    if (auto s = u("num_attention_heads", true, 0, cfg.decoder.decoder.num_attention_heads); !s.ok()) { return R::failure(s); }
    if (auto s = u("num_key_value_heads", true, 0, cfg.decoder.decoder.num_kv_heads); !s.ok()) { return R::failure(s); }
    if (auto s = u("intermediate_size", true, 0, cfg.decoder.decoder.intermediate_size); !s.ok()) { return R::failure(s); }
    if (auto s = u("max_position_embeddings", true, 0, cfg.decoder.decoder.max_position_embeddings); !s.ok()) { return R::failure(s); }
    const uint32_t def_hd = cfg.decoder.decoder.num_attention_heads == 0u ? 0u : cfg.decoder.decoder.hidden_size / cfg.decoder.decoder.num_attention_heads;
    if (auto s = u("head_dim", false, def_hd, cfg.decoder.decoder.head_dim); !s.ok()) { return R::failure(s); }
    // The layer schedule is REQUIRED and must be explicit (never inferred).
    auto *sched = yyjson_obj_get(dc, "layer_kinds");
    if (sched == nullptr || !yyjson_is_arr(sched)) { return R::failure(StatusCode::invalid_argument, "decoder_config.layer_kinds array is required", "layer_kinds"); }
    cfg.decoder.layer_kinds.clear();
    size_t idx, max;
    yyjson_val *elem;
    yyjson_arr_foreach(sched, idx, max, elem) {
        if (!yyjson_is_str(elem)) { return R::failure(StatusCode::invalid_argument, "layer_kinds entries must be strings", "layer_kinds"); }
        const luisa::string_view kind{yyjson_get_str(elem), yyjson_get_len(elem)};
        if (kind == "gdn") { cfg.decoder.layer_kinds.push_back(HybridLayerKind::gdn); }
        else if (kind == "full_attention") { cfg.decoder.layer_kinds.push_back(HybridLayerKind::full_attention); }
        else { return R::failure(StatusCode::invalid_argument, luisa::format("unknown layer kind '{}'", kind), "layer_kinds"); }
    }
    // Keep vision.text consistent with the parsed decoder geometry.
    cfg.vision.text.vocab_size = cfg.decoder.decoder.vocab_size;
    cfg.vision.text.hidden_size = cfg.decoder.decoder.hidden_size;
    cfg.vision.text.num_layers = cfg.decoder.decoder.num_layers;
    if (auto s = cfg.validate(); !s.ok()) { return R::failure(s); }
    return R::success(std::move(cfg));
}

// ---------------------------------------------------------------------------
// Tensor inventory + signature + real-artifact gates
// ---------------------------------------------------------------------------

[[nodiscard]] luisa::vector<ExpectedTensor> ovis_ocr2_expected_tensors(const OvisOcr2Config &config) {
    // Vision component = P7 vision tower + connector ONLY (the visual.* prefix).
    // The P7 inventory also lists a dense TEXT decoder (model.layers.*); that is
    // NOT part of OvisOCR2 -- the text decoder is the hybrid GDN inventory below.
    // Including the P7 text tensors would duplicate model.layers.* names.
    luisa::vector<ExpectedTensor> out;
    for (auto &e : p7_expected_tensors(config.vision, config.vision_names)) {
        if (e.name.compare(0, config.vision_names.visual_prefix.size(), config.vision_names.visual_prefix) == 0) {
            out.push_back(std::move(e));
        }
    }
    auto dec = hybrid_gdn_expected_tensors(config.decoder, config.decoder_names);
    out.insert(out.end(), dec.begin(), dec.end());
    return out;
}

[[nodiscard]] Status ovis_ocr2_validate_tensors(luisa::span<const PackageTensor> vision_tensors,
                                                luisa::span<const PackageTensor> decoder_tensors,
                                                const OvisOcr2Config &config) {
    // Vision component: P7 vision + connector inventory (fp32).
    auto full = p7_expected_tensors(config.vision, config.vision_names);
    luisa::vector<ExpectedTensor> vision_expected;
    for (auto &e : full) {
        if (e.name.compare(0, config.vision_names.visual_prefix.size(), config.vision_names.visual_prefix) == 0) {
            vision_expected.push_back(std::move(e));
        }
    }
    if (auto s = p7_validate_tensors(vision_tensors, vision_expected); !s.ok()) { return s; }
    // Decoder component: hybrid GDN inventory (fp32).
    auto dec_expected = hybrid_gdn_expected_tensors(config.decoder, config.decoder_names);
    return hybrid_gdn_validate_tensors(decoder_tensors, dec_expected);
}

[[nodiscard]] Status ovis_ocr2_validate_signature(const OvisOcr2Config &config, const OvisOcr2Signature &signature) {
    if (signature.checkpoint_revision.empty() || signature.config_fingerprint.empty()) {
        return Status::error(StatusCode::unvalidated,
                             "OvisOCR2 identity requires a checkpoint revision AND config fingerprint; outer model_type alone is never sufficient",
                             "signature");
    }
    if (signature.tensor_names.empty()) {
        return Status::error(StatusCode::unvalidated, "OvisOCR2 identity requires the exact tensor signature", "tensor_names");
    }
    const auto expected = ovis_ocr2_expected_tensors(config);
    for (size_t i = 0; i < signature.tensor_names.size(); i++) {
        for (size_t j = i + 1; j < signature.tensor_names.size(); j++) {
            if (signature.tensor_names[i] == signature.tensor_names[j]) {
                return Status::error(StatusCode::unvalidated, "duplicate artifact tensor name", "tensor_names");
            }
        }
    }
    const auto known = [&expected](const luisa::string &name) {
        for (const auto &e : expected) { if (e.name == name) { return true; } }
        return false;
    };
    const auto observed = [&signature](const luisa::string &name) {
        for (const auto &n : signature.tensor_names) { if (n == name) { return true; } }
        return false;
    };
    for (const auto &name : signature.tensor_names) {
        if (!known(name)) {
            return Status::error(StatusCode::unvalidated, luisa::format("unexpected artifact tensor '{}'", name), "tensor_names");
        }
    }
    for (const auto &e : expected) {
        if (e.required && !observed(e.name)) {
            return Status::error(StatusCode::unvalidated, luisa::format("missing required tensor '{}'", e.name), "tensor_names");
        }
    }
    return Status::success();
}

[[nodiscard]] Status ovis_ocr2_validate_real_artifact(luisa::string_view checkpoint_revision,
                                                      luisa::string_view config_fingerprint) {
    (void)checkpoint_revision;
    (void)config_fingerprint;
    return Status::error(StatusCode::unvalidated,
                         "OvisOCR2 has no qualified official artifact: real checkpoints are refused "
                         "(synthetic recipe-driven path only; see examples/ocr/MODELS.md)",
                         "ovis_ocr2");
}

// ---------------------------------------------------------------------------
// Memory preflight
// ---------------------------------------------------------------------------

[[nodiscard]] Result<OvisOcr2MemoryReport> ovis_ocr2_preflight_decoder(
    const OvisOcr2Config &config, uint32_t max_seq_len, MemoryLimits limits) {
    using R = Result<OvisOcr2MemoryReport>;
    if (auto s = config.validate(); !s.ok()) { return R::failure(s); }
    if (max_seq_len == 0u) {
        return R::failure(StatusCode::invalid_argument, "max_seq_len must be nonzero", "max_seq_len");
    }
    OvisOcr2MemoryReport report;
    report.available_device_bytes = limits.max_device_bytes;
    // Decoder weight bytes: sum of the expected inventory element counts (fp32).
    // A tied LM head adds an executor-owned transposed embedding buffer.
    const auto expected = hybrid_gdn_expected_tensors(config.decoder, config.decoder_names);
    uint64_t weight_bytes = 0;
    for (const auto &e : expected) {
        uint64_t elems = 1;
        for (auto dim : e.shape) {
            if (!checked_mul(elems, dim, elems)) { return R::failure(StatusCode::out_of_memory, "weight element count overflows", "weights"); }
        }
        uint64_t bytes = 0;
        if (!checked_mul(elems, 4ull, bytes) || !checked_add(weight_bytes, bytes, weight_bytes)) {
            return R::failure(StatusCode::out_of_memory, "weight byte count overflows", "weights");
        }
    }
    if (config.decoder.tie_word_embeddings) {
        uint64_t head = 0;
        if (!checked_mul(config.decoder.decoder.hidden_size, config.decoder.decoder.vocab_size, head) ||
            !checked_mul(head, 4ull, head) || !checked_add(weight_bytes, head, weight_bytes)) {
            return R::failure(StatusCode::out_of_memory, "tied LM head byte count overflows", "weights");
        }
    }
    report.decoder_weight_bytes = weight_bytes;
    // Workspace bound: the hybrid arena is sized for prefill_chunk_rows rows.
    // Use a conservative checked bound (the executor reserves the exact arena).
    const auto &d = config.decoder.decoder;
    const auto &g = config.decoder.gdn;
    const uint64_t mr = config.decoder.prefill_chunk_rows;
    uint64_t ws_elems = 0;
    const uint64_t per_row = 8ull * d.hidden_size + 4ull * d.q_dim() + 2ull * d.kv_dim() +
                             3ull * d.intermediate_size + 14ull * g.qk_dim() + 4ull * g.num_heads;
    if (!checked_mul(mr, per_row, ws_elems) || !checked_add(ws_elems, d.vocab_size, ws_elems)) {
        return R::failure(StatusCode::out_of_memory, "workspace element count overflows", "workspace");
    }
    uint64_t ws_bytes = 0;
    if (!checked_mul(ws_elems, 4ull, ws_bytes)) { return R::failure(StatusCode::out_of_memory, "workspace byte count overflows", "workspace"); }
    // Round up to the arena alignment per region is bounded by adding one
    // alignment slack per region; use a generous +64KiB slack for the bound.
    if (!checked_add(ws_bytes, 65536ull, ws_bytes)) { return R::failure(StatusCode::out_of_memory, "workspace slack overflows", "workspace"); }
    report.workspace_bytes = ws_bytes;
    // Hybrid cache: KV (full layers) + GDN recurrent + Conv ring + metadata.
    HybridCacheConfig cc;
    cc.num_full_layers = config.decoder.num_full_layers();
    cc.num_gdn_layers = config.decoder.num_gdn_layers();
    cc.batch = 1;
    cc.num_kv_heads = d.num_kv_heads;
    cc.head_dim = d.head_dim;
    cc.max_seq_len = max_seq_len;
    cc.gdn_heads = g.num_heads;
    cc.gdn_head_dim = g.head_dim;
    cc.conv_kernel_size = g.conv_kernel_size;
    cc.dtype_bytes = 4;
    auto cache = cc.total_bytes();
    if (!cache.ok()) { return R::failure(cache.status); }
    report.cache_bytes = *cache.value;
    uint64_t total = 0;
    if (!checked_add(report.decoder_weight_bytes, report.workspace_bytes, total) ||
        !checked_add(total, report.cache_bytes, total)) {
        return R::failure(StatusCode::out_of_memory, "total device byte requirement overflows", "total");
    }
    report.total_device_bytes = total;
    report.fits = limits.max_device_bytes == 0u || total <= limits.max_device_bytes;
    if (!report.fits) {
        return R::failure(StatusCode::out_of_memory,
                          luisa::format("OvisOCR2 decoder requires {} device bytes (weights {} + workspace {} + cache {}) "
                                        "but only {} bytes are available; reduce context or raise the limit",
                                        total, report.decoder_weight_bytes, report.workspace_bytes, report.cache_bytes,
                                        limits.max_device_bytes),
                          "max_device_bytes");
    }
    return R::success(std::move(report));
}

// ---------------------------------------------------------------------------
// Host composition
// ---------------------------------------------------------------------------

[[nodiscard]] Result<LayoutDocument> ovis_ocr2_compose(const LayoutRunResult &run, LayoutOutputMode mode) {
    using R = Result<LayoutDocument>;
    if (mode != LayoutOutputMode::plain && mode != LayoutOutputMode::markdown) {
        return R::failure(StatusCode::unsupported,
                          "OvisOCR2 supports plain and markdown output only; spotting/extract are not validated",
                          "mode");
    }
    return layout_compose_strict(run, mode, {});
}

}// namespace luisa::ocr

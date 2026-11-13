// OCR engine -- Surya graph contract (Phase 9). Host-only TU.
// Synthetic recipe-driven contract; real artifacts fail closed. The executing
// graph is runtime/surya_executor.* (detector -> crops -> HybridVlmGraph). See
// MODELS.md.
#include "ocr/models/surya.h"

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
// Detector contract validation
// ---------------------------------------------------------------------------

[[nodiscard]] Status SuryaConvStage::validate() const {
    if (in_c == 0u || out_c == 0u || k == 0u || stride == 0u) {
        return Status::error(StatusCode::invalid_argument, "detector conv stage dims must be nonzero", "stage");
    }
    if (act > 3u) {
        return Status::error(StatusCode::invalid_argument, "detector conv act tag must be 0..3", "act");
    }
    return Status::success();
}

[[nodiscard]] Status SuryaDetectorConfig::validate() const {
    if (input_height == 0u || input_width == 0u) {
        return Status::error(StatusCode::invalid_argument, "detector input size must be nonzero", "input");
    }
    for (const auto *s : {&stem, &stage1, &stage2, &db_head}) {
        if (auto v = s->validate(); !v.ok()) { return v; }
    }
    if (stem.in_c != 3u) {
        return Status::error(StatusCode::invalid_argument, "detector stem must take 3 RGB channels", "stem.in_c");
    }
    if (stage1.in_c != stem.out_c || stage2.in_c != stage1.out_c || db_head.in_c != stage2.out_c) {
        return Status::error(StatusCode::invalid_argument, "detector stage channels must chain", "channels");
    }
    if (db_head.out_c != 1u || db_head.k != 1u) {
        return Status::error(StatusCode::invalid_argument, "DB head must be a 1x1 conv to 1 channel", "db_head");
    }
    if (mean.size() != 3u || std_dev.size() != 3u) {
        return Status::error(StatusCode::invalid_argument, "detector mean/std must have 3 entries", "norm");
    }
    return Status::success();
}

[[nodiscard]] Status SuryaDetectorTensorNames::validate() const {
    if (prefix.empty()) {
        return Status::error(StatusCode::invalid_argument, "detector prefix must not be empty", "prefix");
    }
    return Status::success();
}
[[nodiscard]] luisa::string SuryaDetectorTensorNames::stem_weight() const { return prefix + "stem.weight"; }
[[nodiscard]] luisa::string SuryaDetectorTensorNames::stem_bias() const { return prefix + "stem.bias"; }
[[nodiscard]] luisa::string SuryaDetectorTensorNames::stage1_weight() const { return prefix + "stage1.weight"; }
[[nodiscard]] luisa::string SuryaDetectorTensorNames::stage1_bias() const { return prefix + "stage1.bias"; }
[[nodiscard]] luisa::string SuryaDetectorTensorNames::stage2_weight() const { return prefix + "stage2.weight"; }
[[nodiscard]] luisa::string SuryaDetectorTensorNames::stage2_bias() const { return prefix + "stage2.bias"; }
[[nodiscard]] luisa::string SuryaDetectorTensorNames::db_weight() const { return prefix + "db.weight"; }
[[nodiscard]] luisa::string SuryaDetectorTensorNames::db_bias() const { return prefix + "db.bias"; }

// ---------------------------------------------------------------------------
// Config validation
// ---------------------------------------------------------------------------

[[nodiscard]] Status SuryaConfig::validate() const {
    auto s = detector.validate();
    if (!s.ok()) { return s; }
    s = detector_names.validate();
    if (!s.ok()) { return s; }
    s = vision.validate();
    if (!s.ok()) { return s; }
    s = decoder.validate();
    if (!s.ok()) { return s; }
    if (vision.text_pos != P7TextPosKind::rope1d) {
        return Status::error(StatusCode::invalid_argument,
                             "the Surya hybrid text decoder is 1-axis; vision.text_pos must be rope1d", "text_pos");
    }
    if (vision.text.hidden_size != decoder.decoder.hidden_size ||
        vision.text.vocab_size != decoder.decoder.vocab_size) {
        return Status::error(StatusCode::invalid_argument,
                             "vision.text hidden/vocab must match the hybrid decoder (connector output)", "hidden/vocab");
    }
    if (max_lines_per_page == 0u || max_new_tokens == 0u || max_seq_len == 0u) {
        return Status::error(StatusCode::invalid_argument, "Surya caps must be nonzero", "caps");
    }
    return db.validate();
}

// ---------------------------------------------------------------------------
// Declared full contract + schedule validation
// ---------------------------------------------------------------------------

[[nodiscard]] HybridGdnExecutorConfig surya_declared_decoder_config() {
    // DECLARED full Surya hybrid geometry (NOT a verified checkpoint): 24
    // layers, 18 GDN + 6 full (full at every 4th layer). Consistent declared
    // geometry used to validate the 24/18/6 schedule explicitly.
    HybridGdnExecutorConfig cfg;
    auto &d = cfg.decoder;
    d.vocab_size = 65536;
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
    d.eos_token_ids = {2u};
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

[[nodiscard]] Status surya_validate_schedule(const luisa::vector<HybridLayerKind> &kinds) {
    if (kinds.size() != 24u) {
        return Status::error(StatusCode::invalid_argument,
                             luisa::format("Surya schedule must have exactly 24 layers (got {})", kinds.size()), "layer_kinds");
    }
    uint32_t gdn = 0, full = 0;
    for (uint32_t l = 0; l < 24u; l++) {
        const bool want_full = (l % 4u == 3u);// full at 3,7,11,15,19,23
        const bool is_full = kinds[l] == HybridLayerKind::full_attention;
        if (is_full != want_full) {
            return Status::error(StatusCode::invalid_argument,
                                 luisa::format("Surya layer {} must be {} (3:1 GDN:full schedule)", l,
                                               want_full ? "full_attention" : "gdn"),
                                 "layer_kinds");
        }
        if (is_full) { full++; } else { gdn++; }
    }
    if (gdn != 18u || full != 6u) {
        return Status::error(StatusCode::invalid_argument,
                             luisa::format("Surya schedule must be 18 GDN + 6 full (got {} GDN + {} full)", gdn, full),
                             "layer_kinds");
    }
    return Status::success();
}

// ---------------------------------------------------------------------------
// Small synthetic slice
// ---------------------------------------------------------------------------

[[nodiscard]] SuryaConfig surya_synthetic_config() {
    SuryaConfig cfg;
    // Detector: scaled 4-stage CNN at 32x32 input (fewer channels). stem 3->4
    // stride 2 (16x16), stage1 4->4 stride 2 (8x8), stage2 4->4 stride 1 (8x8),
    // db_head 4->1 1x1 (8x8 probability map). hard_swish backbone (EfficientViT).
    auto &det = cfg.detector;
    det.input_height = 32;
    det.input_width = 32;
    det.stem = SuryaConvStage{3, 4, 3, 2, 1, 3};
    det.stage1 = SuryaConvStage{4, 4, 3, 2, 1, 3};
    det.stage2 = SuryaConvStage{4, 4, 3, 1, 1, 3};
    det.db_head = SuryaConvStage{4, 1, 1, 1, 0, 0};
    det.mean = {0.5f, 0.5f, 0.5f};
    det.std_dev = {0.5f, 0.5f, 0.5f};
    cfg.detector_names.prefix = "detector.";
    // Vision tower + spatial-merge merger (small synthetic).
    auto &g = cfg.vision;
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
    g.vision.mlp_act = 0;
    g.vision.rope_kind = P7VisionRopeKind::axial2d;
    g.vision.rope_theta = 10000.0f;
    g.vision.rope_height_pairs = 2;
    g.vision.mask_kind = P7VisionMaskKind::bidirectional;
    g.vision.position_rows = 0;
    g.connector.kind = P7ConnectorKind::spatial_mlp;
    g.connector.intermediate_size = 32;
    g.connector.layer_norm_eps = 1e-5f;
    g.connector.spatial_factor = 2;
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
    // SCALED hybrid decoder: 6 layers = 4 GDN + 2 full.
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
    cfg.db = output::DbPostprocessOptions{};
    cfg.db.binarize_threshold = 0.3f;
    cfg.db.box_threshold = 0.4f;
    cfg.db.unclip_ratio = 1.4f;
    cfg.db.min_box_pixels = 2;
    cfg.max_lines_per_page = 16;
    cfg.max_new_tokens = 8;
    cfg.max_seq_len = 64;
    return cfg;
}

// ---------------------------------------------------------------------------
// Strict config parsing (synthetic schema)
// ---------------------------------------------------------------------------

[[nodiscard]] Result<SuryaConfig> parse_surya_config(luisa::string_view json) {
    using R = Result<SuryaConfig>;
    JsonDoc doc;
    doc.doc = yyjson_read(json.data(), json.size(), 0);
    if (doc.doc == nullptr) { return R::failure(StatusCode::invalid_argument, "surya config is not valid JSON", "config"); }
    auto *root = yyjson_doc_get_root(doc.doc);
    if (root == nullptr || !yyjson_is_obj(root)) { return R::failure(StatusCode::invalid_argument, "surya config must be a JSON object", "config"); }
    if (auto *mt = yyjson_obj_get(root, "model_type"); mt != nullptr && yyjson_is_str(mt)) {
        const luisa::string_view type{yyjson_get_str(mt), yyjson_get_len(mt)};
        if (type.find("surya") == luisa::string_view::npos) {
            return R::failure(StatusCode::unsupported, "outer model_type does not name Surya (and is never sufficient for identity)", "model_type");
        }
    }
    for (const char *marker : {"num_experts", "moe", "grpo", "mtp_num_layers"}) {
        if (auto *v = yyjson_obj_get(root, marker); v != nullptr && !yyjson_is_null(v) && !yyjson_is_false(v)) {
            return R::failure(StatusCode::unsupported, luisa::format("surya config field '{}' is outside the hybrid GDN contract", marker), marker);
        }
    }
    SuryaConfig cfg = surya_synthetic_config();
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
    cfg.vision.text.vocab_size = cfg.decoder.decoder.vocab_size;
    cfg.vision.text.hidden_size = cfg.decoder.decoder.hidden_size;
    cfg.vision.text.num_layers = cfg.decoder.decoder.num_layers;
    if (auto s = cfg.validate(); !s.ok()) { return R::failure(s); }
    return R::success(std::move(cfg));
}

// ---------------------------------------------------------------------------
// Tensor inventory + signature + real-artifact gates
// ---------------------------------------------------------------------------

[[nodiscard]] luisa::vector<ExpectedTensor> surya_expected_tensors(const SuryaConfig &config) {
    luisa::vector<ExpectedTensor> out;
    const auto &dn = config.detector_names;
    const auto &det = config.detector;
    const auto rm = TensorLayout::row_major;
    const auto conv_shape = [](const SuryaConvStage &s) {
        return luisa::vector<uint64_t>{s.out_c, s.in_c, s.k, s.k};
    };
    out.emplace_back(ExpectedTensor{dn.stem_weight(), conv_shape(det.stem), rm, true});
    out.emplace_back(ExpectedTensor{dn.stem_bias(), {det.stem.out_c}, rm, true});
    out.emplace_back(ExpectedTensor{dn.stage1_weight(), conv_shape(det.stage1), rm, true});
    out.emplace_back(ExpectedTensor{dn.stage1_bias(), {det.stage1.out_c}, rm, true});
    out.emplace_back(ExpectedTensor{dn.stage2_weight(), conv_shape(det.stage2), rm, true});
    out.emplace_back(ExpectedTensor{dn.stage2_bias(), {det.stage2.out_c}, rm, true});
    out.emplace_back(ExpectedTensor{dn.db_weight(), conv_shape(det.db_head), rm, true});
    out.emplace_back(ExpectedTensor{dn.db_bias(), {det.db_head.out_c}, rm, true});
    // Vision component = P7 vision tower + connector ONLY (visual.* prefix).
    for (auto &e : p7_expected_tensors(config.vision, config.vision_names)) {
        if (e.name.compare(0, config.vision_names.visual_prefix.size(), config.vision_names.visual_prefix) == 0) {
            out.push_back(std::move(e));
        }
    }
    auto dec = hybrid_gdn_expected_tensors(config.decoder, config.decoder_names);
    out.insert(out.end(), dec.begin(), dec.end());
    return out;
}

[[nodiscard]] Status surya_validate_tensors(luisa::span<const PackageTensor> detector_tensors,
                                            luisa::span<const PackageTensor> vision_tensors,
                                            luisa::span<const PackageTensor> decoder_tensors,
                                            const SuryaConfig &config) {
    const auto expected = surya_expected_tensors(config);
    const auto &dn = config.detector_names;
    // Split the inventory by component prefix.
    luisa::vector<ExpectedTensor> det_exp, vis_exp, dec_exp;
    for (const auto &e : expected) {
        if (e.name.compare(0, dn.prefix.size(), dn.prefix) == 0) { det_exp.push_back(e); }
        else if (e.name.compare(0, config.vision_names.visual_prefix.size(), config.vision_names.visual_prefix) == 0) { vis_exp.push_back(e); }
        else { dec_exp.push_back(e); }
    }
    if (auto s = validate_dense_decoder_tensors(detector_tensors, det_exp); !s.ok()) { return s; }
    if (auto s = p7_validate_tensors(vision_tensors, vis_exp); !s.ok()) { return s; }
    return hybrid_gdn_validate_tensors(decoder_tensors, dec_exp);
}

[[nodiscard]] Status surya_validate_signature(const SuryaConfig &config, const SuryaSignature &signature) {
    if (signature.checkpoint_revision.empty() || signature.config_fingerprint.empty()) {
        return Status::error(StatusCode::unvalidated,
                             "Surya identity requires a checkpoint revision AND config fingerprint; outer model_type alone is never sufficient",
                             "signature");
    }
    if (signature.tensor_names.empty()) {
        return Status::error(StatusCode::unvalidated, "Surya identity requires the exact tensor signature", "tensor_names");
    }
    const auto expected = surya_expected_tensors(config);
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

[[nodiscard]] Status surya_validate_real_artifact(luisa::string_view checkpoint_revision,
                                                  luisa::string_view config_fingerprint) {
    (void)checkpoint_revision;
    (void)config_fingerprint;
    return Status::error(StatusCode::unvalidated,
                         "Surya has no qualified official artifact: real checkpoints are refused "
                         "(synthetic recipe-driven path only; see examples/ocr/MODELS.md)",
                         "surya");
}

// ---------------------------------------------------------------------------
// Memory preflight
// ---------------------------------------------------------------------------

[[nodiscard]] Result<SuryaMemoryReport> surya_preflight_decoder(
    const SuryaConfig &config, uint32_t max_seq_len, MemoryLimits limits) {
    using R = Result<SuryaMemoryReport>;
    if (auto s = config.validate(); !s.ok()) { return R::failure(s); }
    if (max_seq_len == 0u) {
        return R::failure(StatusCode::invalid_argument, "max_seq_len must be nonzero", "max_seq_len");
    }
    SuryaMemoryReport report;
    report.available_device_bytes = limits.max_device_bytes;
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
    if (!checked_mul(ws_elems, 4ull, ws_bytes) || !checked_add(ws_bytes, 65536ull, ws_bytes)) {
        return R::failure(StatusCode::out_of_memory, "workspace byte count overflows", "workspace");
    }
    report.workspace_bytes = ws_bytes;
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
                          luisa::format("Surya decoder requires {} device bytes (weights {} + workspace {} + cache {}) "
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

[[nodiscard]] Result<LayoutDocument> surya_compose(const SuryaRunResult &run, LayoutOutputMode mode) {
    using R = Result<LayoutDocument>;
    if (mode != LayoutOutputMode::plain && mode != LayoutOutputMode::spotting && mode != LayoutOutputMode::markdown) {
        return R::failure(StatusCode::unsupported,
                          "Surya supports plain, spotting, and markdown output; extract is not validated",
                          "mode");
    }
    LayoutDocument doc;
    // Build a LayoutRunResult mirror so the shared strict composer applies the
    // same per-type Markdown / reading-order rules. Line confidence is carried
    // as the DETECTOR score (labeled internal/test by the public bridge).
    LayoutRunResult mirror;
    for (const auto &page : run.pages) {
        LayoutPageResult lp;
        lp.page_index = page.page_index;
        lp.source_width = page.source_width;
        lp.source_height = page.source_height;
        for (const auto &line : page.lines) {
            LayoutRegionResult region;
            region.region_id = line.reading_order;
            region.page_index = line.page_index;
            region.reading_order = line.reading_order;
            region.block_type = LayoutBlockType::text;
            region.content = line.text;
            region.source_box = line.bounds;
            region.polygon = line.polygon;
            region.has_confidence = line.has_confidence;
            region.confidence = line.confidence;
            lp.regions.push_back(std::move(region));
        }
        mirror.pages.push_back(std::move(lp));
    }
    mirror.weight_bytes = run.weight_bytes;
    mirror.workspace_bytes = run.workspace_bytes;
    mirror.kv_cache_peak_bytes = run.cache_bytes;
    mirror.peak_device_bytes = run.weight_bytes + run.workspace_bytes + run.cache_bytes;
    return layout_compose_strict(mirror, mode, {});
}

}// namespace luisa::ocr

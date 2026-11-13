// OCR engine -- GLM-OCR graph contract (Phase 7). Host-only TU.
// Synthetic recipe-driven path over the shared Phase 7 slice; real artifacts
// fail closed. See glm_ocr.h.
#include "ocr/models/glm_ocr.h"

#include <utility>

#include <luisa/core/stl/format.h>
#include <yyjson.h>

#include "ocr/runtime/layout_result.h"

namespace luisa::ocr {

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

[[nodiscard]] Status GlmOcrConfig::validate() const {
    if (auto s = graph.validate(); !s.ok()) { return s; }
    if (auto s = names.validate(); !s.ok()) { return s; }
    // GLM-OCR identity boundaries (never inferred from model_type):
    //  * CogViT uses learned absolute positions and NO vision RoPE;
    //  * the connector MUST be the learned downsampling MLP;
    //  * the text decoder MUST be GLM: partial rotary, 1D positions, QKV bias.
    if (graph.vision.rope_kind != P7VisionRopeKind::none) {
        return Status::error(StatusCode::invalid_argument, "GLM-OCR CogViT uses learned absolute positions, not vision RoPE", "vision.rope_kind");
    }
    if (graph.vision.position_rows == 0u) {
        return Status::error(StatusCode::invalid_argument, "GLM-OCR CogViT requires a learned absolute position table", "vision.position_rows");
    }
    if (graph.vision.mask_kind != P7VisionMaskKind::bidirectional) {
        return Status::error(StatusCode::invalid_argument, "GLM-OCR CogViT uses bidirectional attention", "vision.mask_kind");
    }
    if (graph.connector.kind != P7ConnectorKind::downsample_mlp) {
        return Status::error(StatusCode::invalid_argument, "GLM-OCR uses the learned downsampling connector", "connector.kind");
    }
    if (graph.text_pos != P7TextPosKind::rope1d) {
        return Status::error(StatusCode::invalid_argument, "GLM-OCR uses plain 1D text positions (not M-RoPE/XD-RoPE)", "text_pos");
    }
    if (graph.partial_rotary_num == graph.partial_rotary_den) {
        return Status::error(StatusCode::invalid_argument, "GLM-OCR requires PARTIAL rotary (a Qwen full-rotary decoder is never substituted)", "partial_rotary");
    }
    if (graph.text.rope_pair_layout != RopePairLayout::rotate_half) {
        return Status::error(StatusCode::invalid_argument, "GLM-OCR partial rotary uses rotate-half pairs within the rotated prefix", "rope_pair_layout");
    }
    return Status::success();
}

[[nodiscard]] GlmOcrConfig glm_ocr_synthetic_config() {
    GlmOcrConfig cfg;
    auto &g = cfg.graph;
    // CogViT vision (small synthetic): learned absolute positions, no RoPE.
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
    g.vision.rope_kind = P7VisionRopeKind::none;
    g.vision.mask_kind = P7VisionMaskKind::bidirectional;
    g.vision.position_rows = 1024;
    // Downsampling connector: stride 2 gather + learned linear + LN + MLP.
    g.connector.kind = P7ConnectorKind::downsample_mlp;
    g.connector.intermediate_size = 32;
    g.connector.layer_norm_eps = 1e-5f;
    g.connector.spatial_factor = 2;
    // GLM-0.5B-style text decoder (small synthetic): partial rotary 1/2.
    g.text.vocab_size = 32;
    g.text.hidden_size = 32;
    g.text.num_layers = 3;
    g.text.num_attention_heads = 4;
    g.text.num_kv_heads = 2;
    g.text.head_dim = 8;
    g.text.intermediate_size = 64;
    g.text.rms_norm_eps = 1e-5f;
    g.text.rope_theta = 10000.0f;
    g.text.rope_pair_layout = RopePairLayout::rotate_half;
    g.text.max_position_embeddings = 256;
    g.text.eos_token_ids = {31u};
    g.tie_word_embeddings = true;
    g.text_pos = P7TextPosKind::rope1d;
    g.partial_rotary_num = 1;
    g.partial_rotary_den = 2;
    g.packing.patch_size = 2;
    g.packing.min_pixels = 0;
    g.packing.max_pixels = 4096;
    g.packing.grid_multiple = 2;
    g.vision_component = "vision";
    g.decoder_component = "decoder";
    g.max_vision_patches = 4096;
    g.prefill_chunk_rows = 256;
    // GLM-OCR canonical prefixes (synthetic contract).
    cfg.names.text.prefix = "model.";
    cfg.names.visual_prefix = "visual.";
    return cfg;
}

// ---------------------------------------------------------------------------
// Config parsing (strict; fail-closed on Qwen / MoE / GDN / MTP)
// ---------------------------------------------------------------------------

namespace {

struct JsonDoc {
    yyjson_doc *doc{nullptr};
    ~JsonDoc() { if (doc != nullptr) { yyjson_doc_free(doc); } }
};

[[nodiscard]] Result<uint32_t> get_u32(yyjson_val *obj, const char *key, bool required, uint32_t fallback) {
    using R = Result<uint32_t>;
    auto *v = yyjson_obj_get(obj, key);
    if (v == nullptr || yyjson_is_null(v)) {
        if (required) { return R::failure(StatusCode::invalid_argument, luisa::format("missing config key '{}'", key), key); }
        return R::success(fallback);
    }
    if (!yyjson_is_uint(v)) { return R::failure(StatusCode::invalid_argument, luisa::format("config key '{}' must be a nonnegative integer", key), key); }
    const auto value = yyjson_get_uint(v);
    if (value > UINT32_MAX) { return R::failure(StatusCode::invalid_argument, luisa::format("config key '{}' exceeds uint32", key), key); }
    return R::success(static_cast<uint32_t>(value));
}

[[nodiscard]] Result<float> get_f32(yyjson_val *obj, const char *key, float fallback) {
    using R = Result<float>;
    auto *v = yyjson_obj_get(obj, key);
    if (v == nullptr || yyjson_is_null(v)) { return R::success(fallback); }
    if (!yyjson_is_real(v) && !yyjson_is_uint(v) && !yyjson_is_sint(v)) {
        return R::failure(StatusCode::invalid_argument, luisa::format("config key '{}' must be a number", key), key);
    }
    return R::success(static_cast<float>(yyjson_get_num(v)));
}

}// namespace

[[nodiscard]] Result<GlmOcrConfig> parse_glm_ocr_config(luisa::string_view json) {
    using R = Result<GlmOcrConfig>;
    JsonDoc doc;
    doc.doc = yyjson_read(json.data(), json.size(), 0);
    if (doc.doc == nullptr) { return R::failure(StatusCode::invalid_argument, "GLM-OCR config is not valid JSON", "config"); }
    auto *root = yyjson_doc_get_root(doc.doc);
    if (root == nullptr || !yyjson_is_obj(root)) { return R::failure(StatusCode::invalid_argument, "GLM-OCR config must be a JSON object", "config"); }
    if (auto *mt = yyjson_obj_get(root, "model_type"); mt != nullptr && yyjson_is_str(mt)) {
        const luisa::string_view type{yyjson_get_str(mt), yyjson_get_len(mt)};
        if (type.find("glm") == luisa::string_view::npos) {
            return R::failure(StatusCode::unsupported, "outer model_type does not name GLM (and is never sufficient for identity)", "model_type");
        }
        // A Qwen model_type is a different decoder contract: fail closed.
        if (type.find("qwen") != luisa::string_view::npos) {
            return R::failure(StatusCode::unsupported, "a Qwen model_type is not GLM-OCR; real Qwen configs fail closed", "model_type");
        }
    }
    // Fail closed on Qwen3 / MoE / GDN / MTP markers anywhere at the top level.
    for (const char *marker : {"qk_norm", "mrope_section", "num_experts", "linear_num_key_heads", "linear_key_head_dim"}) {
        if (auto *v = yyjson_obj_get(root, marker); v != nullptr && !yyjson_is_null(v) && !yyjson_is_false(v)) {
            return R::failure(StatusCode::unsupported, luisa::format("GLM-OCR config field '{}' is outside the GLM dense contract", marker), marker);
        }
    }
    // MTP is DEFERRED: a positive mtp_num_layers requirement fails closed.
    if (auto *mtp = yyjson_obj_get(root, "mtp_num_layers"); mtp != nullptr && !yyjson_is_null(mtp)) {
        if (!yyjson_is_uint(mtp)) { return R::failure(StatusCode::invalid_argument, "mtp_num_layers must be a nonnegative integer", "mtp_num_layers"); }
        if (yyjson_get_uint(mtp) > 0u) {
            return R::failure(StatusCode::unsupported, "GLM-OCR MTP is deferred; ordinary target decoding only (mtp_num_layers must be 0/absent)", "mtp_num_layers");
        }
    }
    GlmOcrConfig cfg = glm_ocr_synthetic_config();
    auto &g = cfg.graph;
    auto *vc = yyjson_obj_get(root, "vision_config");
    if (vc == nullptr || !yyjson_is_obj(vc)) { return R::failure(StatusCode::invalid_argument, "vision_config object is required", "vision_config"); }
    {
        auto u = [&](const char *k, bool req, uint32_t fb, uint32_t &out) -> Status {
            auto r = get_u32(vc, k, req, fb);
            if (!r.ok()) { return r.status; }
            out = *r.value;
            return Status::success();
        };
        if (auto s = u("num_layers", true, 0, g.vision.num_layers); !s.ok()) { return R::failure(s); }
        if (auto s = u("hidden_size", true, 0, g.vision.hidden_size); !s.ok()) { return R::failure(s); }
        if (auto s = u("num_heads", true, 0, g.vision.num_heads); !s.ok()) { return R::failure(s); }
        if (auto s = u("head_dim", true, 0, g.vision.head_dim); !s.ok()) { return R::failure(s); }
        if (auto s = u("intermediate_size", true, 0, g.vision.intermediate_size); !s.ok()) { return R::failure(s); }
        if (auto s = u("patch_size", true, 0, g.vision.patch_size); !s.ok()) { return R::failure(s); }
        if (auto s = u("position_rows", false, g.vision.position_rows, g.vision.position_rows); !s.ok()) { return R::failure(s); }
        auto eps = get_f32(vc, "layer_norm_eps", g.vision.layer_norm_eps);
        if (!eps.ok()) { return R::failure(eps.status); }
        g.vision.layer_norm_eps = *eps.value;
        // CogViT has no vision RoPE; a rope_theta/rope request fails closed.
        if (auto *vr = yyjson_obj_get(vc, "rope"); vr != nullptr && yyjson_is_true(vr)) {
            return R::failure(StatusCode::unsupported, "GLM-OCR CogViT uses learned absolute positions, not vision RoPE", "vision.rope");
        }
    }
    auto *tc = yyjson_obj_get(root, "text_config");
    if (tc == nullptr || !yyjson_is_obj(tc)) { return R::failure(StatusCode::invalid_argument, "text_config object is required", "text_config"); }
    {
        // A Qwen3 QK-Norm decoder is refused.
        if (auto *qn = yyjson_obj_get(tc, "qk_norm"); qn != nullptr && yyjson_is_true(qn)) {
            return R::failure(StatusCode::unsupported, "GLM-OCR has no QK-Norm (Qwen3 convention); real Qwen configs fail closed", "text_config.qk_norm");
        }
        auto u = [&](const char *k, bool req, uint32_t fb, uint32_t &out) -> Status {
            auto r = get_u32(tc, k, req, fb);
            if (!r.ok()) { return r.status; }
            out = *r.value;
            return Status::success();
        };
        if (auto s = u("vocab_size", true, 0, g.text.vocab_size); !s.ok()) { return R::failure(s); }
        if (auto s = u("hidden_size", true, 0, g.text.hidden_size); !s.ok()) { return R::failure(s); }
        if (auto s = u("num_hidden_layers", true, 0, g.text.num_layers); !s.ok()) { return R::failure(s); }
        if (auto s = u("num_attention_heads", true, 0, g.text.num_attention_heads); !s.ok()) { return R::failure(s); }
        if (auto s = u("num_key_value_heads", true, 0, g.text.num_kv_heads); !s.ok()) { return R::failure(s); }
        if (auto s = u("intermediate_size", true, 0, g.text.intermediate_size); !s.ok()) { return R::failure(s); }
        if (auto s = u("max_position_embeddings", true, 0, g.text.max_position_embeddings); !s.ok()) { return R::failure(s); }
        const uint32_t def_hd = g.text.num_attention_heads == 0u ? 0u : g.text.hidden_size / g.text.num_attention_heads;
        if (auto s = u("head_dim", false, def_hd, g.text.head_dim); !s.ok()) { return R::failure(s); }
        auto eps = get_f32(tc, "rms_norm_eps", g.text.rms_norm_eps);
        if (!eps.ok()) { return R::failure(eps.status); }
        g.text.rms_norm_eps = *eps.value;
        auto theta = get_f32(tc, "rope_theta", g.text.rope_theta);
        if (!theta.ok()) { return R::failure(theta.status); }
        g.text.rope_theta = *theta.value;
        g.text.rope_pair_layout = RopePairLayout::rotate_half;
    }
    // partial_rotary_factor in (0, 1]. Represent as num/den with den 1000.
    {
        auto f = get_f32(root, "partial_rotary_factor", 0.5f);
        if (!f.ok()) { return R::failure(f.status); }
        const float factor = *f.value;
        if (!(factor > 0.0f) || factor > 1.0f) {
            return R::failure(StatusCode::invalid_argument, "partial_rotary_factor must be in (0, 1]", "partial_rotary_factor");
        }
        if (factor == 1.0f) {
            return R::failure(StatusCode::unsupported, "GLM-OCR requires PARTIAL rotary; a full-rotary (Qwen) decoder fails closed", "partial_rotary_factor");
        }
        g.partial_rotary_den = 1000u;
        g.partial_rotary_num = static_cast<uint32_t>(factor * 1000.0f + 0.5f);
        if (g.partial_rotary_num == 0u) { g.partial_rotary_num = 1u; }
    }
    auto *cc = yyjson_obj_get(root, "connector");
    if (cc != nullptr && yyjson_is_obj(cc)) {
        auto r = get_u32(cc, "intermediate_size", false, g.connector.intermediate_size);
        if (!r.ok()) { return R::failure(r.status); }
        g.connector.intermediate_size = *r.value;
        auto m = get_u32(cc, "downsample_stride", false, g.connector.spatial_factor);
        if (!m.ok()) { return R::failure(m.status); }
        g.connector.spatial_factor = *m.value;
    }
    g.packing.patch_size = g.vision.patch_size;
    g.packing.grid_multiple = g.connector.spatial_factor;
    auto *pk = yyjson_obj_get(root, "packing");
    if (pk != nullptr && yyjson_is_obj(pk)) {
        auto mx = get_u32(pk, "max_pixels", false, static_cast<uint32_t>(g.packing.max_pixels));
        if (!mx.ok()) { return R::failure(mx.status); }
        g.packing.max_pixels = *mx.value;
    }
    if (auto s = cfg.validate(); !s.ok()) { return R::failure(s); }
    return R::success(std::move(cfg));
}

// ---------------------------------------------------------------------------
// Tensor inventory + real-artifact gate + composition
// ---------------------------------------------------------------------------

[[nodiscard]] luisa::vector<ExpectedTensor> glm_ocr_expected_tensors(const GlmOcrConfig &config) {
    return p7_expected_tensors(config.graph, config.names);
}

[[nodiscard]] Status glm_ocr_validate_tensors(luisa::span<const PackageTensor> tensors,
                                              luisa::span<const ExpectedTensor> expected) {
    return p7_validate_tensors(tensors, expected);
}

[[nodiscard]] Status glm_ocr_validate_real_artifact(luisa::string_view checkpoint_revision,
                                                    luisa::string_view config_fingerprint) {
    (void)checkpoint_revision;
    (void)config_fingerprint;
    return Status::error(StatusCode::unvalidated,
                         "GLM-OCR has no qualified official artifact: real checkpoints are refused "
                         "(synthetic recipe-driven path only; see examples/ocr/MODELS.md)",
                         "glm_ocr");
}

[[nodiscard]] Result<LayoutDocument> glm_ocr_compose(const LayoutRunResult &run, LayoutOutputMode mode,
                                                     luisa::string_view extract_schema) {
    using R = Result<LayoutDocument>;
    // GLM-OCR validates plain + markdown + extract. spotting is not qualified.
    if (mode == LayoutOutputMode::spotting) {
        return R::failure(StatusCode::unsupported,
                          "GLM-OCR supports plain, markdown, and extract output; spotting is not validated",
                          "mode");
    }
    return layout_compose_strict(run, mode, extract_schema);
}

}// namespace luisa::ocr

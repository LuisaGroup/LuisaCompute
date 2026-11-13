// OCR engine -- MinerU2.5-Pro graph contract (Phase 7). Host-only TU.
// Synthetic recipe-driven path over the shared Phase 7 slice; real artifacts
// fail closed. See mineru.h.
#include "ocr/models/mineru.h"

#include <utility>

#include <luisa/core/stl/format.h>
#include <yyjson.h>

#include "ocr/runtime/layout_result.h"

namespace luisa::ocr {

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

[[nodiscard]] Status MineruConfig::validate() const {
    if (auto s = graph.validate(); !s.ok()) { return s; }
    if (auto s = names.validate(); !s.ok()) { return s; }
    // MinerU identity boundaries (never inferred from model_type):
    //  * the connector MUST be the 2x2 pixel-unshuffle MLP (not adaptive pool);
    //  * the vision tower uses 2D axial RoPE (NaViT), bidirectional;
    //  * the text decoder MUST be Qwen2 adjacent-pair M-RoPE (not Qwen3).
    if (graph.connector.kind != P7ConnectorKind::unshuffle_mlp) {
        return Status::error(StatusCode::invalid_argument,
                             "MinerU uses the 2x2 pixel-unshuffle connector; Paddle's adaptive connector is never substituted",
                             "connector.kind");
    }
    if (graph.connector.spatial_factor != 2u) {
        return Status::error(StatusCode::invalid_argument, "MinerU pixel-unshuffle is fixed 2x2", "connector.spatial_factor");
    }
    if (graph.vision.rope_kind != P7VisionRopeKind::axial2d) {
        return Status::error(StatusCode::invalid_argument, "MinerU NaViT uses 2D axial RoPE", "vision.rope_kind");
    }
    if (graph.text_pos != P7TextPosKind::mrope_adjacent) {
        return Status::error(StatusCode::invalid_argument,
                             "MinerU requires the Qwen2 adjacent-pair M-RoPE decoder; the Qwen3 interleaved convention is never substituted",
                             "text_pos");
    }
    if (graph.text.rope_pair_layout != RopePairLayout::adjacent) {
        return Status::error(StatusCode::invalid_argument, "MinerU Qwen2 M-RoPE uses adjacent pairs", "rope_pair_layout");
    }
    return Status::success();
}

[[nodiscard]] MineruConfig mineru_synthetic_config() {
    MineruConfig cfg;
    auto &g = cfg.graph;
    // NaViT-shaped vision (small synthetic): 2D RoPE, bidirectional.
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
    // 2x2 pixel-unshuffle connector + LN + 2-layer GELU MLP.
    g.connector.kind = P7ConnectorKind::unshuffle_mlp;
    g.connector.intermediate_size = 32;
    g.connector.layer_norm_eps = 1e-5f;
    g.connector.spatial_factor = 2;
    // Qwen2-0.5B-style text decoder (small synthetic): adjacent M-RoPE.
    g.text.vocab_size = 32;
    g.text.hidden_size = 32;
    g.text.num_layers = 3;
    g.text.num_attention_heads = 4;
    g.text.num_kv_heads = 2;
    g.text.head_dim = 8;
    g.text.intermediate_size = 64;
    g.text.rms_norm_eps = 1e-5f;
    g.text.rope_theta = 10000.0f;
    g.text.rope_pair_layout = RopePairLayout::adjacent;
    g.text.max_position_embeddings = 256;
    g.text.eos_token_ids = {31u};
    g.tie_word_embeddings = true;
    g.text_pos = P7TextPosKind::mrope_adjacent;
    g.pos_sections = {2u, 1u, 1u};
    g.packing.patch_size = 2;
    g.packing.min_pixels = 0;
    g.packing.max_pixels = 4096;
    g.packing.grid_multiple = 2;
    g.vision_component = "vision";
    g.decoder_component = "decoder";
    g.max_vision_patches = 4096;
    g.prefill_chunk_rows = 256;
    // MinerU canonical prefixes (synthetic contract).
    cfg.names.text.prefix = "model.";
    cfg.names.visual_prefix = "visual.";
    return cfg;
}

// ---------------------------------------------------------------------------
// Config parsing (strict; fail-closed on Qwen3 / adaptive / MoE / GDN)
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

[[nodiscard]] Result<MineruConfig> parse_mineru_config(luisa::string_view json) {
    using R = Result<MineruConfig>;
    JsonDoc doc;
    doc.doc = yyjson_read(json.data(), json.size(), 0);
    if (doc.doc == nullptr) { return R::failure(StatusCode::invalid_argument, "MinerU config is not valid JSON", "config"); }
    auto *root = yyjson_doc_get_root(doc.doc);
    if (root == nullptr || !yyjson_is_obj(root)) { return R::failure(StatusCode::invalid_argument, "MinerU config must be a JSON object", "config"); }
    if (auto *mt = yyjson_obj_get(root, "model_type"); mt != nullptr && yyjson_is_str(mt)) {
        const luisa::string_view type{yyjson_get_str(mt), yyjson_get_len(mt)};
        if (type.find("mineru") == luisa::string_view::npos) {
            return R::failure(StatusCode::unsupported, "outer model_type does not name MinerU (and is never sufficient for identity)", "model_type");
        }
    }
    for (const char *marker : {"qk_norm", "num_experts", "linear_num_key_heads", "linear_key_head_dim"}) {
        if (auto *v = yyjson_obj_get(root, marker); v != nullptr && !yyjson_is_null(v) && !yyjson_is_false(v)) {
            return R::failure(StatusCode::unsupported, luisa::format("MinerU config field '{}' is outside the dense contract", marker), marker);
        }
    }
    MineruConfig cfg = mineru_synthetic_config();
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
        if (auto s = u("rope_height_pairs", false, g.vision.rope_height_pairs, g.vision.rope_height_pairs); !s.ok()) { return R::failure(s); }
        auto eps = get_f32(vc, "layer_norm_eps", g.vision.layer_norm_eps);
        if (!eps.ok()) { return R::failure(eps.status); }
        g.vision.layer_norm_eps = *eps.value;
    }
    auto *tc = yyjson_obj_get(root, "text_config");
    if (tc == nullptr || !yyjson_is_obj(tc)) { return R::failure(StatusCode::invalid_argument, "text_config object is required", "text_config"); }
    {
        if (auto *qn = yyjson_obj_get(tc, "qk_norm"); qn != nullptr && yyjson_is_true(qn)) {
            return R::failure(StatusCode::unsupported, "MinerU uses the Qwen2 decoder (no QK-Norm); a Qwen3 decoder fails closed", "text_config.qk_norm");
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
        g.text.rope_pair_layout = RopePairLayout::adjacent;
    }
    auto *ms = yyjson_obj_get(root, "mrope_section");
    if (ms == nullptr || !yyjson_is_arr(ms) || yyjson_arr_size(ms) != 3u) {
        return R::failure(StatusCode::invalid_argument, "mrope_section must be a 3-element array [t,h,w]", "mrope_section");
    }
    g.pos_sections.clear();
    for (size_t i = 0; i < 3u; i++) {
        auto *item = yyjson_arr_get(ms, i);
        if (!yyjson_is_uint(item) || yyjson_get_uint(item) > UINT32_MAX) {
            return R::failure(StatusCode::invalid_argument, "mrope_section entries must be nonnegative integers", "mrope_section");
        }
        g.pos_sections.push_back(static_cast<uint32_t>(yyjson_get_uint(item)));
    }
    auto *cc = yyjson_obj_get(root, "connector");
    if (cc != nullptr && yyjson_is_obj(cc)) {
        // An adaptive-pool connector request is the Paddle convention: refuse.
        if (auto *kind = yyjson_obj_get(cc, "kind"); kind != nullptr && yyjson_is_str(kind)) {
            const luisa::string_view k{yyjson_get_str(kind), yyjson_get_len(kind)};
            if (k == "adaptive" || k == "adaptive_pool") {
                return R::failure(StatusCode::unsupported, "MinerU uses pixel-unshuffle, not Paddle's adaptive connector", "connector.kind");
            }
        }
        auto r = get_u32(cc, "intermediate_size", false, g.connector.intermediate_size);
        if (!r.ok()) { return R::failure(r.status); }
        g.connector.intermediate_size = *r.value;
    }
    g.packing.patch_size = g.vision.patch_size;
    g.packing.grid_multiple = 2;
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

[[nodiscard]] luisa::vector<ExpectedTensor> mineru_expected_tensors(const MineruConfig &config) {
    return p7_expected_tensors(config.graph, config.names);
}

[[nodiscard]] Status mineru_validate_tensors(luisa::span<const PackageTensor> tensors,
                                             luisa::span<const ExpectedTensor> expected) {
    return p7_validate_tensors(tensors, expected);
}

[[nodiscard]] Status mineru_validate_real_artifact(luisa::string_view checkpoint_revision,
                                                   luisa::string_view config_fingerprint) {
    (void)checkpoint_revision;
    (void)config_fingerprint;
    return Status::error(StatusCode::unvalidated,
                         "MinerU has no qualified official artifact: real checkpoints are refused "
                         "(synthetic recipe-driven path only; see examples/ocr/MODELS.md)",
                         "mineru");
}

[[nodiscard]] Result<LayoutDocument> mineru_compose(const LayoutRunResult &run, LayoutOutputMode mode,
                                                    luisa::string_view extract_schema) {
    using R = Result<LayoutDocument>;
    // MinerU validates plain + markdown + extract with cross-page composition.
    // spotting is not qualified.
    if (mode == LayoutOutputMode::spotting) {
        return R::failure(StatusCode::unsupported,
                          "MinerU supports plain, markdown, and extract output; spotting is not validated",
                          "mode");
    }
    // layout_compose_strict already orders pages ascending + reading order
    // within each page (cross-page composition).
    return layout_compose_strict(run, mode, extract_schema);
}

}// namespace luisa::ocr

// OCR engine -- HunyuanOCR-1.5 graph contract (Phase 7). Host-only TU.
// Synthetic recipe-driven path over the shared Phase 7 slice; real artifacts
// fail closed. See hunyuan_ocr.h.
#include "ocr/models/hunyuan_ocr.h"

#include <utility>

#include <luisa/core/stl/format.h>
#include <yyjson.h>

#include "ocr/runtime/layout_result.h"

namespace luisa::ocr {

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

[[nodiscard]] Status HunyuanOcrConfig::validate() const {
    if (auto s = graph.validate(); !s.ok()) { return s; }
    if (auto s = names.validate(); !s.ok()) { return s; }
    // HunyuanOCR identity boundaries (never inferred from model_type):
    //  * the connector MUST be the adaptive-pool MLP;
    //  * the vision tower uses SiLU MLP + 2D RoPE (Hunyuan-ViT);
    //  * the text decoder MUST be XD-RoPE (4 axes, adjacent pairs).
    if (graph.connector.kind != P7ConnectorKind::adaptive_mlp) {
        return Status::error(StatusCode::invalid_argument, "HunyuanOCR uses the adaptive-pool MLP connector", "connector.kind");
    }
    if (graph.vision.mlp_act != 1u) {
        return Status::error(StatusCode::invalid_argument, "HunyuanOCR Hunyuan-ViT uses the SiLU MLP activation", "vision.mlp_act");
    }
    if (graph.text_pos != P7TextPosKind::xd) {
        return Status::error(StatusCode::invalid_argument,
                             "HunyuanOCR requires XD-RoPE over text/height/width/time; a Qwen M-RoPE decoder is never substituted",
                             "text_pos");
    }
    if (graph.text.rope_pair_layout != RopePairLayout::adjacent) {
        return Status::error(StatusCode::invalid_argument, "HunyuanOCR XD-RoPE uses adjacent pairs", "rope_pair_layout");
    }
    if (graph.pos_sections.size() != 4u) {
        return Status::error(StatusCode::invalid_argument, "XD-RoPE requires 4 position sections [text,h,w,time]", "pos_sections");
    }
    return Status::success();
}

[[nodiscard]] HunyuanOcrConfig hunyuan_ocr_synthetic_config() {
    HunyuanOcrConfig cfg;
    auto &g = cfg.graph;
    // Hunyuan-ViT vision (small synthetic): SiLU MLP, 2D RoPE.
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
    g.vision.mlp_act = 1;// SiLU
    g.vision.rope_kind = P7VisionRopeKind::axial2d;
    g.vision.rope_theta = 10000.0f;
    g.vision.rope_height_pairs = 2;
    g.vision.mask_kind = P7VisionMaskKind::bidirectional;
    g.vision.position_rows = 0;
    // Adaptive-pool connector + LN + 2-layer GELU MLP.
    g.connector.kind = P7ConnectorKind::adaptive_mlp;
    g.connector.intermediate_size = 32;
    g.connector.layer_norm_eps = 1e-5f;
    g.connector.max_visual_tokens = 16;
    // Hunyuan-0.5B-style text decoder (small synthetic): XD-RoPE 4 axes.
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
    g.text_pos = P7TextPosKind::xd;
    g.pos_sections = {1u, 1u, 1u, 1u};// text/height/width/time
    g.packing.patch_size = 2;
    g.packing.min_pixels = 0;
    g.packing.max_pixels = 4096;
    g.packing.grid_multiple = 1;
    g.vision_component = "vision";
    g.decoder_component = "decoder";
    g.max_vision_patches = 4096;
    g.prefill_chunk_rows = 256;
    // HunyuanOCR canonical prefixes (synthetic contract).
    cfg.names.text.prefix = "model.";
    cfg.names.visual_prefix = "visual.";
    return cfg;
}

// ---------------------------------------------------------------------------
// Config parsing (strict; fail-closed on Qwen / MoE / GDN / DFlash)
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

[[nodiscard]] Result<HunyuanOcrConfig> parse_hunyuan_ocr_config(luisa::string_view json) {
    using R = Result<HunyuanOcrConfig>;
    JsonDoc doc;
    doc.doc = yyjson_read(json.data(), json.size(), 0);
    if (doc.doc == nullptr) { return R::failure(StatusCode::invalid_argument, "HunyuanOCR config is not valid JSON", "config"); }
    auto *root = yyjson_doc_get_root(doc.doc);
    if (root == nullptr || !yyjson_is_obj(root)) { return R::failure(StatusCode::invalid_argument, "HunyuanOCR config must be a JSON object", "config"); }
    if (auto *mt = yyjson_obj_get(root, "model_type"); mt != nullptr && yyjson_is_str(mt)) {
        const luisa::string_view type{yyjson_get_str(mt), yyjson_get_len(mt)};
        if (type.find("hunyuan") == luisa::string_view::npos) {
            return R::failure(StatusCode::unsupported, "outer model_type does not name Hunyuan (and is never sufficient for identity)", "model_type");
        }
    }
    for (const char *marker : {"qk_norm", "mrope_section", "num_experts", "linear_num_key_heads", "linear_key_head_dim"}) {
        if (auto *v = yyjson_obj_get(root, marker); v != nullptr && !yyjson_is_null(v) && !yyjson_is_false(v)) {
            return R::failure(StatusCode::unsupported, luisa::format("HunyuanOCR config field '{}' is outside the dense contract", marker), marker);
        }
    }
    // DFlash is DEFERRED: a positive dflash requirement fails closed.
    if (auto *df = yyjson_obj_get(root, "dflash_num_layers"); df != nullptr && !yyjson_is_null(df)) {
        if (!yyjson_is_uint(df)) { return R::failure(StatusCode::invalid_argument, "dflash_num_layers must be a nonnegative integer", "dflash_num_layers"); }
        if (yyjson_get_uint(df) > 0u) {
            return R::failure(StatusCode::unsupported, "HunyuanOCR DFlash is deferred; ordinary target decoding only", "dflash_num_layers");
        }
    }
    HunyuanOcrConfig cfg = hunyuan_ocr_synthetic_config();
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
            return R::failure(StatusCode::unsupported, "HunyuanOCR has no QK-Norm (Qwen3 convention); real Qwen configs fail closed", "text_config.qk_norm");
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
    auto *xs = yyjson_obj_get(root, "xd_rope_section");
    if (xs == nullptr || !yyjson_is_arr(xs) || yyjson_arr_size(xs) != 4u) {
        return R::failure(StatusCode::invalid_argument, "xd_rope_section must be a 4-element array [text,h,w,time]", "xd_rope_section");
    }
    g.pos_sections.clear();
    for (size_t i = 0; i < 4u; i++) {
        auto *item = yyjson_arr_get(xs, i);
        if (!yyjson_is_uint(item) || yyjson_get_uint(item) > UINT32_MAX) {
            return R::failure(StatusCode::invalid_argument, "xd_rope_section entries must be nonnegative integers", "xd_rope_section");
        }
        g.pos_sections.push_back(static_cast<uint32_t>(yyjson_get_uint(item)));
    }
    auto *cc = yyjson_obj_get(root, "connector");
    if (cc != nullptr && yyjson_is_obj(cc)) {
        auto r = get_u32(cc, "intermediate_size", false, g.connector.intermediate_size);
        if (!r.ok()) { return R::failure(r.status); }
        g.connector.intermediate_size = *r.value;
        auto mt = get_u32(cc, "max_visual_tokens", false, g.connector.max_visual_tokens);
        if (!mt.ok()) { return R::failure(mt.status); }
        g.connector.max_visual_tokens = *mt.value;
    }
    g.packing.patch_size = g.vision.patch_size;
    g.packing.grid_multiple = 1;
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
// Tensor inventory + real-artifact gate + composition / adapters
// ---------------------------------------------------------------------------

[[nodiscard]] luisa::vector<ExpectedTensor> hunyuan_ocr_expected_tensors(const HunyuanOcrConfig &config) {
    return p7_expected_tensors(config.graph, config.names);
}

[[nodiscard]] Status hunyuan_ocr_validate_tensors(luisa::span<const PackageTensor> tensors,
                                                  luisa::span<const ExpectedTensor> expected) {
    return p7_validate_tensors(tensors, expected);
}

[[nodiscard]] Status hunyuan_ocr_validate_real_artifact(luisa::string_view checkpoint_revision,
                                                        luisa::string_view config_fingerprint) {
    (void)checkpoint_revision;
    (void)config_fingerprint;
    return Status::error(StatusCode::unvalidated,
                         "HunyuanOCR has no qualified official artifact: real checkpoints are refused "
                         "(synthetic recipe-driven path only; see examples/ocr/MODELS.md)",
                         "hunyuan_ocr");
}

[[nodiscard]] Result<LayoutDocument> hunyuan_ocr_compose(const LayoutRunResult &run, LayoutOutputMode mode,
                                                         luisa::string_view extract_schema) {
    using R = Result<LayoutDocument>;
    // HunyuanOCR validates plain + markdown + extract + spotting. Translation
    // is NOT implemented and is not a LayoutOutputMode; it is refused at the
    // session/prompt layer (no translation adapter exists).
    if (mode == LayoutOutputMode::spotting) {
        return hunyuan_ocr_spotting(run);
    }
    return layout_compose_strict(run, mode, extract_schema);
}

[[nodiscard]] Result<LayoutDocument> hunyuan_ocr_spotting(const LayoutRunResult &run) {
    using R = Result<LayoutDocument>;
    // Spotting carries source coordinates from the layout regions. The
    // canonical text is the region content joined in reading order. NO
    // confidence is fabricated (generative content).
    LayoutDocument doc;
    doc.diagnostics = run.diagnostics;
    luisa::string joined;
    bool first = true;
    uint32_t line_order = 0;
    for (const auto &page : run.pages) {
        for (const auto &region : page.regions) {
            LayoutBlock block;
            block.type = region.block_type;
            block.content = region.content;
            block.page_index = region.page_index;
            block.reading_order = region.reading_order;
            block.source_box = region.source_box;
            block.polygon = region.polygon;
            block.has_confidence = false;// NO fake confidence
            block.confidence = 0.0f;
            doc.blocks.push_back(block);
            if (!region.content.empty()) {
                LayoutLine line;
                line.text = region.content;
                line.page_index = region.page_index;
                line.reading_order = line_order++;
                line.source_box = region.source_box;
                line.polygon = region.polygon;
                line.has_confidence = false;
                doc.lines.push_back(std::move(line));
                if (!first) { joined += '\n'; }
                first = false;
                joined += region.content;
            }
        }
    }
    doc.text = std::move(joined);
    return R::success(std::move(doc));
}

}// namespace luisa::ocr

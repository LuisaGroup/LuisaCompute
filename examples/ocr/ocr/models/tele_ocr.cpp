// OCR engine -- TeleOCR graph contract (Phase 7). Host-only TU.
// Synthetic recipe-driven path over the shared Phase 7 slice; real artifacts
// fail closed. See tele_ocr.h.
#include "ocr/models/tele_ocr.h"

#include <algorithm>
#include <cmath>
#include <utility>

#include <luisa/core/stl/format.h>
#include <yyjson.h>

#include "ocr/runtime/layout_result.h"

namespace luisa::ocr {

// ---------------------------------------------------------------------------
// Deformation metadata + deterministic warp
// ---------------------------------------------------------------------------

[[nodiscard]] luisa::string_view tele_deform_kind_name(TeleDeformKind kind) noexcept {
    switch (kind) {
        case TeleDeformKind::flat: return "flat";
        case TeleDeformKind::curved: return "curved";
        case TeleDeformKind::folded: return "folded";
        case TeleDeformKind::crumpled: return "crumpled";
        default: return "unknown";
    }
}

[[nodiscard]] Status TeleDeformSpec::validate() const {
    if (kind != TeleDeformKind::flat && kind != TeleDeformKind::curved &&
        kind != TeleDeformKind::folded && kind != TeleDeformKind::crumpled) {
        return Status::error(StatusCode::invalid_argument, "unknown deformation kind", "kind");
    }
    if (!std::isfinite(amplitude) || amplitude < 0.0f || amplitude > 64.0f) {
        return Status::error(StatusCode::invalid_argument, "deformation amplitude must be in [0, 64]", "amplitude");
    }
    return Status::success();
}

namespace {

// Deterministic forward warp: for each OUTPUT pixel (x, y) sample the SOURCE
// at (x + dx(y), y + dy(x)) with a kind-specific displacement. Nearest-neighbor
// sampling with clamped edges (the synthetic convention; bilinear dewarping is
// an official gate). flat is identity.
[[nodiscard]] void tele_displacement(TeleDeformKind kind, float amp, uint32_t x, uint32_t y,
                                     uint32_t w, uint32_t h, float &dx, float &dy) noexcept {
    const float fx = static_cast<float>(x);
    const float fy = static_cast<float>(y);
    const float fw = static_cast<float>(w);
    const float fh = static_cast<float>(h);
    dx = 0.0f;
    dy = 0.0f;
    switch (kind) {
        case TeleDeformKind::curved:
            // Horizontal sinusoidal bow across the page height.
            dx = amp * std::sin(3.14159265f * fy / (fh == 0.0f ? 1.0f : fh));
            break;
        case TeleDeformKind::folded:
            // A vertical crease: pixels right of the midline shift by a ramp.
            dx = (fx > fw * 0.5f) ? amp * 0.5f : 0.0f;
            dy = amp * 0.25f * std::sin(6.2831853f * fx / (fw == 0.0f ? 1.0f : fw));
            break;
        case TeleDeformKind::crumpled:
            // Two-axis high-frequency ripple.
            dx = amp * 0.5f * std::sin(0.7f * fy);
            dy = amp * 0.5f * std::cos(0.6f * fx);
            break;
        case TeleDeformKind::flat:
        default:
            break;
    }
}

}// namespace

[[nodiscard]] Result<TeleWarpedImage> tele_warp_image(luisa::span<const uint8_t> pixels,
                                                      uint32_t width, uint32_t height,
                                                      const TeleDeformSpec &spec) {
    using R = Result<TeleWarpedImage>;
    if (auto s = spec.validate(); !s.ok()) { return R::failure(s); }
    if (width == 0u || height == 0u || pixels.size() != static_cast<size_t>(width) * height * 3u) {
        return R::failure(StatusCode::invalid_argument, "degenerate source image", "pixels");
    }
    TeleWarpedImage out;
    out.width = width;
    out.height = height;
    out.spec = spec;
    out.pixels.resize(pixels.size());
    if (spec.kind == TeleDeformKind::flat || spec.amplitude == 0.0f) {
        std::copy(pixels.begin(), pixels.end(), out.pixels.begin());
        return R::success(std::move(out));
    }
    const auto sample = [&](int32_t sx, int32_t sy, uint32_t c) -> uint8_t {
        const int32_t cx = std::clamp(sx, 0, static_cast<int32_t>(width) - 1);
        const int32_t cy = std::clamp(sy, 0, static_cast<int32_t>(height) - 1);
        return pixels[(static_cast<size_t>(cy) * width + cx) * 3u + c];
    };
    for (uint32_t y = 0; y < height; y++) {
        for (uint32_t x = 0; x < width; x++) {
            float dx = 0.0f, dy = 0.0f;
            tele_displacement(spec.kind, spec.amplitude, x, y, width, height, dx, dy);
            const int32_t sx = static_cast<int32_t>(std::lround(static_cast<float>(x) + dx));
            const int32_t sy = static_cast<int32_t>(std::lround(static_cast<float>(y) + dy));
            for (uint32_t c = 0; c < 3u; c++) {
                out.pixels[(static_cast<size_t>(y) * width + x) * 3u + c] = sample(sx, sy, c);
            }
        }
    }
    return R::success(std::move(out));
}

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

[[nodiscard]] Status TeleOcrConfig::validate() const {
    if (auto s = graph.validate(); !s.ok()) { return s; }
    if (auto s = names.validate(); !s.ok()) { return s; }
    // TeleOCR identity boundaries (never inferred from model_type):
    //  * vision MUST be window/global (the Qwen2.5-VL-derived convention);
    //  * the aligner MUST be the spatial-merge MLP (not adaptive/unshuffle);
    //  * the text decoder MUST be Qwen3 (interleaved MRoPE, QK-Norm, no bias).
    if (graph.vision.mask_kind != P7VisionMaskKind::window_global) {
        return Status::error(StatusCode::invalid_argument, "TeleOCR vision must use window/global attention", "vision.mask_kind");
    }
    if (graph.vision.qk_norm) {
        return Status::error(StatusCode::invalid_argument, "TeleOCR vision has no QK-Norm (that is the Qwen3-VL convention)", "vision.qk_norm");
    }
    if (graph.connector.kind != P7ConnectorKind::spatial_mlp) {
        return Status::error(StatusCode::invalid_argument, "TeleOCR uses the spatial-merge MLP aligner", "connector.kind");
    }
    if (graph.text_pos != P7TextPosKind::mrope_interleaved) {
        return Status::error(StatusCode::invalid_argument,
                             "TeleOCR requires the Qwen3 interleaved-MRoPE decoder; a Qwen2 decoder is never substituted", "text_pos");
    }
    if (graph.text.rope_pair_layout != RopePairLayout::rotate_half) {
        return Status::error(StatusCode::invalid_argument, "TeleOCR Qwen3 decoder uses rotate-half interleaved pairs", "rope_pair_layout");
    }
    return Status::success();
}

[[nodiscard]] TeleOcrConfig tele_ocr_synthetic_config() {
    TeleOcrConfig cfg;
    auto &g = cfg.graph;
    // Vision: Qwen2.5-VL-derived window/global tower (small synthetic).
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
    g.vision.mask_kind = P7VisionMaskKind::window_global;
    g.vision.window_patch_size = 2;
    g.vision.global_layers = {1u};// layer 1 is global; layer 0 windowed
    g.vision.position_rows = 0;
    // Aligner: spatial merge 2 + LN + 2-layer GELU MLP.
    g.connector.kind = P7ConnectorKind::spatial_mlp;
    g.connector.intermediate_size = 32;
    g.connector.layer_norm_eps = 1e-5f;
    g.connector.spatial_factor = 2;
    // Qwen3-0.6B-style text decoder (small synthetic).
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
    g.text_pos = P7TextPosKind::mrope_interleaved;
    g.pos_sections = {2u, 1u, 1u};
    g.packing.patch_size = 2;
    g.packing.min_pixels = 0;
    g.packing.max_pixels = 4096;
    g.packing.grid_multiple = 2;
    g.vision_component = "vision";
    g.decoder_component = "decoder";
    g.max_vision_patches = 4096;
    g.prefill_chunk_rows = 256;
    // TeleOCR canonical prefixes (synthetic contract).
    cfg.names.text.prefix = "model.";
    cfg.names.visual_prefix = "visual.";
    cfg.layout_route = false;
    return cfg;
}

// ---------------------------------------------------------------------------
// Config parsing (strict; fail-closed on non-TeleOCR contracts)
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

[[nodiscard]] Result<TeleOcrConfig> parse_tele_ocr_config(luisa::string_view json) {
    using R = Result<TeleOcrConfig>;
    JsonDoc doc;
    doc.doc = yyjson_read(json.data(), json.size(), 0);
    if (doc.doc == nullptr) { return R::failure(StatusCode::invalid_argument, "TeleOCR config is not valid JSON", "config"); }
    auto *root = yyjson_doc_get_root(doc.doc);
    if (root == nullptr || !yyjson_is_obj(root)) { return R::failure(StatusCode::invalid_argument, "TeleOCR config must be a JSON object", "config"); }
    // The outer model_type is NEVER identity; when present it must at least
    // mention tele, but a matching string alone never selects the recipe.
    if (auto *mt = yyjson_obj_get(root, "model_type"); mt != nullptr && yyjson_is_str(mt)) {
        const luisa::string_view type{yyjson_get_str(mt), yyjson_get_len(mt)};
        if (type.find("tele") == luisa::string_view::npos) {
            return R::failure(StatusCode::unsupported, "outer model_type does not name TeleOCR (and is never sufficient for identity)", "model_type");
        }
    }
    // Fail closed on MoE/GDN/training markers.
    for (const char *marker : {"num_experts", "linear_num_key_heads", "linear_key_head_dim", "grpo", "mtp_num_layers"}) {
        if (auto *v = yyjson_obj_get(root, marker); v != nullptr && !yyjson_is_null(v) && !yyjson_is_false(v)) {
            return R::failure(StatusCode::unsupported, luisa::format("TeleOCR config field '{}' is outside the dense contract", marker), marker);
        }
    }
    TeleOcrConfig cfg = tele_ocr_synthetic_config();
    auto &g = cfg.graph;
    // vision_config
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
        if (auto s = u("window_patch_size", true, 0, g.vision.window_patch_size); !s.ok()) { return R::failure(s); }
        if (auto s = u("rope_height_pairs", false, g.vision.rope_height_pairs, g.vision.rope_height_pairs); !s.ok()) { return R::failure(s); }
        auto eps = get_f32(vc, "layer_norm_eps", g.vision.layer_norm_eps);
        if (!eps.ok()) { return R::failure(eps.status); }
        g.vision.layer_norm_eps = *eps.value;
        // global_layers array (required for the window/global contract).
        auto *gl = yyjson_obj_get(vc, "global_layers");
        if (gl == nullptr || !yyjson_is_arr(gl)) { return R::failure(StatusCode::invalid_argument, "vision global_layers array is required", "global_layers"); }
        g.vision.global_layers.clear();
        size_t idx = 0, max = 0;
        yyjson_val *item = nullptr;
        yyjson_arr_foreach(gl, idx, max, item) {
            if (!yyjson_is_uint(item) || yyjson_get_uint(item) > UINT32_MAX) {
                return R::failure(StatusCode::invalid_argument, "global_layers entries must be nonnegative integers", "global_layers");
            }
            g.vision.global_layers.push_back(static_cast<uint32_t>(yyjson_get_uint(item)));
        }
        // Vision QK-Norm is the Qwen3-VL convention; TeleOCR vision refuses it.
        if (auto *qn = yyjson_obj_get(vc, "qk_norm"); qn != nullptr && yyjson_is_true(qn)) {
            return R::failure(StatusCode::unsupported, "TeleOCR vision has no QK-Norm (Qwen3-VL convention)", "vision.qk_norm");
        }
    }
    // text_config: reuse the shared dense geometry parser, then enforce Qwen3.
    auto *tc = yyjson_obj_get(root, "text_config");
    if (tc == nullptr || !yyjson_is_obj(tc)) { return R::failure(StatusCode::invalid_argument, "text_config object is required", "text_config"); }
    {
        // A Qwen2 decoder (qkv_bias true / explicit no-qk_norm) is refused.
        if (auto *qb = yyjson_obj_get(tc, "qkv_bias"); qb != nullptr && yyjson_is_true(qb)) {
            return R::failure(StatusCode::unsupported, "TeleOCR requires the Qwen3 decoder (no QKV bias); a Qwen2 decoder is never substituted", "text_config.qkv_bias");
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
    // mrope_section [t,h,w]
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
    // connector
    auto *cc = yyjson_obj_get(root, "connector");
    if (cc != nullptr && yyjson_is_obj(cc)) {
        auto r = get_u32(cc, "intermediate_size", false, g.connector.intermediate_size);
        if (!r.ok()) { return R::failure(r.status); }
        g.connector.intermediate_size = *r.value;
        auto m = get_u32(cc, "spatial_merge_size", false, g.connector.spatial_factor);
        if (!m.ok()) { return R::failure(m.status); }
        g.connector.spatial_factor = *m.value;
    }
    // packing
    g.packing.patch_size = g.vision.patch_size;
    g.packing.grid_multiple = g.connector.spatial_factor;
    auto *pk = yyjson_obj_get(root, "packing");
    if (pk != nullptr && yyjson_is_obj(pk)) {
        auto mx = get_u32(pk, "max_pixels", false, static_cast<uint32_t>(g.packing.max_pixels));
        if (!mx.ok()) { return R::failure(mx.status); }
        g.packing.max_pixels = *mx.value;
    }
    // layout_route
    if (auto *lr = yyjson_obj_get(root, "layout_route"); lr != nullptr) {
        if (!yyjson_is_bool(lr)) { return R::failure(StatusCode::invalid_argument, "layout_route must be a boolean", "layout_route"); }
        cfg.layout_route = yyjson_is_true(lr);
    }
    if (auto s = cfg.validate(); !s.ok()) { return R::failure(s); }
    return R::success(std::move(cfg));
}

// ---------------------------------------------------------------------------
// Tensor inventory + signature + real-artifact gate
// ---------------------------------------------------------------------------

[[nodiscard]] luisa::vector<ExpectedTensor> tele_ocr_expected_tensors(const TeleOcrConfig &config) {
    return p7_expected_tensors(config.graph, config.names);
}

[[nodiscard]] Status tele_ocr_validate_tensors(luisa::span<const PackageTensor> tensors,
                                               luisa::span<const ExpectedTensor> expected) {
    return p7_validate_tensors(tensors, expected);
}

[[nodiscard]] Status tele_ocr_validate_signature(const TeleOcrConfig &config, const TeleOcrSignature &signature) {
    // The outer model_type is NEVER identity: require non-empty revision AND
    // fingerprint AND an exact canonical tensor-name set.
    if (signature.checkpoint_revision.empty() || signature.config_fingerprint.empty()) {
        return Status::error(StatusCode::unvalidated,
                             "TeleOCR identity requires a checkpoint revision AND config fingerprint; outer model_type alone is never sufficient",
                             "signature");
    }
    if (signature.tensor_names.empty()) {
        return Status::error(StatusCode::unvalidated, "TeleOCR identity requires the exact tensor signature", "tensor_names");
    }
    const auto expected = tele_ocr_expected_tensors(config);
    // No duplicate names.
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

[[nodiscard]] Status tele_ocr_validate_real_artifact(luisa::string_view checkpoint_revision,
                                                     luisa::string_view config_fingerprint) {
    (void)checkpoint_revision;
    (void)config_fingerprint;
    return Status::error(StatusCode::unvalidated,
                         "TeleOCR has no qualified official artifact: real checkpoints are refused "
                         "(synthetic recipe-driven path only; see examples/ocr/MODELS.md)",
                         "tele_ocr");
}

[[nodiscard]] Result<LayoutDocument> tele_ocr_compose(const LayoutRunResult &run, LayoutOutputMode mode) {
    using R = Result<LayoutDocument>;
    // TeleOCR validates plain + markdown only. spotting/extract are not
    // qualified for TeleOCR and fail closed (never a silent plain fallback).
    if (mode != LayoutOutputMode::plain && mode != LayoutOutputMode::markdown) {
        return R::failure(StatusCode::unsupported,
                          "TeleOCR supports plain and markdown output only; spotting/extract are not validated",
                          "mode");
    }
    return layout_compose_strict(run, mode, {});
}

}// namespace luisa::ocr

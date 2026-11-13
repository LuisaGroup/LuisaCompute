// OCR engine -- GOT-OCR2.0 graph contract (Phase 8). Host-only TU.
// Synthetic recipe-driven path over the shared Phase 7 slice; real artifacts
// fail closed. See got_ocr2.h.
#include "ocr/models/got_ocr2.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <utility>

#include <luisa/core/stl/format.h>
#include <yyjson.h>

#include "ocr/output/markdown.h"

namespace luisa::ocr {

// ---------------------------------------------------------------------------
// Prompt modes
// ---------------------------------------------------------------------------

[[nodiscard]] luisa::string_view got_prompt_mode_name(GotPromptMode mode) noexcept {
    switch (mode) {
        case GotPromptMode::plain: return "plain";
        case GotPromptMode::formatted: return "formatted";
        case GotPromptMode::region: return "region";
        case GotPromptMode::color: return "color";
        default: return "unknown";
    }
}

[[nodiscard]] Status GotRegionBox::validate() const {
    const auto finite = [](float v) { return std::isfinite(v); };
    if (!finite(x0) || !finite(y0) || !finite(x1) || !finite(y1)) {
        return Status::error(StatusCode::invalid_argument, "region box coordinates must be finite", "region");
    }
    if (x0 < 0.0f || y0 < 0.0f || x1 > 1.0f || y1 > 1.0f) {
        return Status::error(StatusCode::invalid_argument, "region box must be within normalized [0,1]", "region");
    }
    if (x1 <= x0 || y1 <= y0) {
        return Status::error(StatusCode::invalid_argument, "region box must have positive area", "region");
    }
    return Status::success();
}

namespace {

[[nodiscard]] constexpr uint32_t pow10(uint32_t e) {
    uint32_t r = 1;
    for (uint32_t i = 0; i < e; i++) { r *= 10u; }
    return r;
}

[[nodiscard]] luisa::string format_coord(float v) {
    // Fixed 4-decimal normalized coordinate, LOCALE-INDEPENDENT: integer math
    // on the rounded ten-thousandths (no snprintf/strtod locale dependence).
    // Deterministic and exactly round-trippable with parse_coord.
    const auto units = static_cast<uint32_t>(std::lround(static_cast<double>(v) * 10000.0));
    const uint32_t whole = units / 10000u;
    const uint32_t frac = units % 10000u;
    char buf[32];
    buf[0] = static_cast<char>('0' + whole);
    buf[1] = '.';
    for (uint32_t i = 0; i < 4u; i++) {
        buf[2 + i] = static_cast<char>('0' + (frac / pow10(3u - i)) % 10u);
    }
    buf[6] = '\0';
    return luisa::string{buf};
}

// Locale-independent strict parse of a fixed 4-decimal normalized coordinate
// "d.dddd" (digits only, exactly one '.', exactly 4 fraction digits). No
// strtof/strtod: the C locale's decimal separator can never change semantics.
[[nodiscard]] bool parse_coord(luisa::string_view token, float &out) {
    if (token.size() != 6u || token[1] != '.') { return false; }
    uint32_t units = 0;
    for (size_t i = 0; i < token.size(); i++) {
        if (i == 1u) { continue; }
        const char c = token[i];
        if (c < '0' || c > '9') { return false; }
        units = units * 10u + static_cast<uint32_t>(c - '0');
    }
    out = static_cast<float>(units) / 10000.0f;
    return true;
}


}// namespace

[[nodiscard]] Result<luisa::string> got_ocr2_build_prompt(GotPromptMode mode,
                                                         const GotRegionBox &region,
                                                         luisa::string_view color) {
    using R = Result<luisa::string>;
    switch (mode) {
        case GotPromptMode::plain:
            return R::success(luisa::string{"OCR: "});
        case GotPromptMode::formatted:
            return R::success(luisa::string{"OCR with format: "});
        case GotPromptMode::region: {
            if (auto s = region.validate(); !s.ok()) { return R::failure(s); }
            return R::success(luisa::format("OCR with region: [{},{},{},{}]",
                                            format_coord(region.x0), format_coord(region.y0),
                                            format_coord(region.x1), format_coord(region.y1)));
        }
        case GotPromptMode::color: {
            if (color.empty()) {
                return R::failure(StatusCode::invalid_argument, "color guidance requires a non-empty color name", "color");
            }
            return R::success(luisa::format("OCR with color: {}", color));
        }
        default:
            return R::failure(StatusCode::unsupported, "unknown GOT prompt mode", "mode");
    }
}

[[nodiscard]] Result<GotParsedPrompt> got_ocr2_parse_prompt(luisa::string_view prompt) {
    using R = Result<GotParsedPrompt>;
    GotParsedPrompt out;
    constexpr std::string_view kPlain = "OCR: ";
    constexpr std::string_view kFormatted = "OCR with format: ";
    constexpr std::string_view kRegion = "OCR with region: ";
    constexpr std::string_view kColor = "OCR with color: ";
    const auto starts = [&prompt](std::string_view prefix) {
        return prompt.size() >= prefix.size() && prompt.compare(0, prefix.size(), prefix) == 0;
    };
    if (starts(kPlain)) {
        out.mode = GotPromptMode::plain;
        return R::success(std::move(out));
    }
    if (starts(kFormatted)) {
        out.mode = GotPromptMode::formatted;
        return R::success(std::move(out));
    }
    if (starts(kRegion)) {
        out.mode = GotPromptMode::region;
        const auto body = prompt.substr(kRegion.size());
        // Expect "[x0,y0,x1,y1]".
        if (body.size() < 2u || body.front() != '[' || body.back() != ']') {
            return R::failure(StatusCode::invalid_argument, "malformed GOT region prompt (expected [x0,y0,x1,y1])", "prompt");
        }
        const auto inner = body.substr(1, body.size() - 2u);
        luisa::vector<float> coords;
        size_t start = 0;
        while (start <= inner.size()) {
            const auto comma = inner.find(',', start);
            const auto token = inner.substr(start, comma == luisa::string_view::npos ? inner.size() - start : comma - start);
            // Strict LOCALE-INDEPENDENT parse: exactly "d.dddd" (the canonical
            // format produced by got_ocr2_build_prompt). No strtof: a comma
            // decimal separator or locale-specific form can never be accepted.
            if (token.empty()) { return R::failure(StatusCode::invalid_argument, "empty region coordinate", "prompt"); }
            float v = 0.0f;
            if (!parse_coord(token, v) || !std::isfinite(v)) {
                return R::failure(StatusCode::invalid_argument,
                                  "invalid region coordinate (expected d.dddd, locale-independent)", "prompt");
            }
            coords.push_back(v);
            if (comma == luisa::string_view::npos) { break; }
            start = comma + 1u;
        }
        if (coords.size() != 4u) {
            return R::failure(StatusCode::invalid_argument, "GOT region prompt must have exactly 4 coordinates", "prompt");
        }
        out.region = GotRegionBox{coords[0], coords[1], coords[2], coords[3]};
        if (auto s = out.region.validate(); !s.ok()) { return R::failure(s); }
        return R::success(std::move(out));
    }
    if (starts(kColor)) {
        out.mode = GotPromptMode::color;
        out.color = luisa::string{prompt.substr(kColor.size())};
        if (out.color.empty()) {
            return R::failure(StatusCode::invalid_argument, "GOT color prompt requires a color name", "prompt");
        }
        return R::success(std::move(out));
    }
    return R::failure(StatusCode::invalid_argument, "unknown GOT prompt (never a silent plain fallback)", "prompt");
}

// ---------------------------------------------------------------------------
// Source region transforms
// ---------------------------------------------------------------------------

[[nodiscard]] Result<GotPixelBox> got_ocr2_region_to_pixels(const GotRegionBox &region,
                                                           uint32_t width, uint32_t height) {
    using R = Result<GotPixelBox>;
    if (auto s = region.validate(); !s.ok()) { return R::failure(s); }
    if (width == 0u || height == 0u) {
        return R::failure(StatusCode::invalid_argument, "degenerate source image size", "image");
    }
    const auto to_px = [](float v, uint32_t extent) -> uint32_t {
        const double scaled = static_cast<double>(v) * static_cast<double>(extent);
        const auto clamped = std::clamp(scaled, 0.0, static_cast<double>(extent));
        return static_cast<uint32_t>(std::lround(clamped));
    };
    GotPixelBox box;
    box.x0 = to_px(region.x0, width);
    box.y0 = to_px(region.y0, height);
    box.x1 = to_px(region.x1, width);
    box.y1 = to_px(region.y1, height);
    if (box.x1 <= box.x0 || box.y1 <= box.y0) {
        return R::failure(StatusCode::invalid_argument, "region maps to a degenerate pixel box", "region");
    }
    return R::success(std::move(box));
}

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

[[nodiscard]] Status GotOcr2Config::validate() const {
    if (auto s = graph.validate(); !s.ok()) { return s; }
    if (auto s = names.validate(); !s.ok()) { return s; }
    // GOT identity boundaries (never inferred from model_type):
    //  * vision uses window/global attention + decomposed relative-position
    //    bias (NOT RoPE, NOT absolute positions);
    //  * the connector is a plain linear projection (mlp kind, no reduction);
    //  * the decoder is a Qwen-style dense GQA transformer with QKV bias, no
    //    QK-Norm, and plain 1D RoPE.
    if (graph.vision.mask_kind != P7VisionMaskKind::window_global) {
        return Status::error(StatusCode::invalid_argument, "GOT ViTDet/SAM uses window/global attention", "vision.mask_kind");
    }
    if (graph.vision.rel_pos_window == 0u) {
        return Status::error(StatusCode::invalid_argument, "GOT requires a decomposed relative-position bias", "vision.rel_pos_window");
    }
    if (graph.vision.rope_kind != P7VisionRopeKind::none) {
        return Status::error(StatusCode::invalid_argument, "GOT vision uses relative bias, not RoPE", "vision.rope_kind");
    }
    if (graph.connector.kind != P7ConnectorKind::mlp) {
        return Status::error(StatusCode::invalid_argument, "GOT uses a linear connector (no spatial reduction)", "connector.kind");
    }
    if (graph.text_pos != P7TextPosKind::rope1d) {
        return Status::error(StatusCode::invalid_argument, "GOT Qwen decoder uses plain 1D RoPE", "text_pos");
    }
    return Status::success();
}

[[nodiscard]] GotOcr2Config got_ocr2_synthetic_config() {
    GotOcr2Config cfg;
    auto &g = cfg.graph;
    // ViTDet/SAM vision (small synthetic): window/global + relative bias.
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
    g.vision.mask_kind = P7VisionMaskKind::window_global;
    g.vision.window_patch_size = 2;
    g.vision.global_layers = {1u};
    g.vision.position_rows = 0;
    g.vision.rel_pos_window = 4;
    // Linear connector (no reduction): LN + 2-layer GELU MLP, 16 -> 32.
    g.connector.kind = P7ConnectorKind::mlp;
    g.connector.intermediate_size = 32;
    g.connector.layer_norm_eps = 1e-5f;
    g.connector.spatial_factor = 1;
    // Qwen-0.5B-style decoder (small synthetic): QKV bias, no QK-Norm, 1D RoPE.
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
    g.pos_sections.clear();
    g.packing.patch_size = 2;
    g.packing.min_pixels = 0;
    g.packing.max_pixels = 4096;
    g.packing.grid_multiple = 1;
    g.vision_component = "vision";
    g.decoder_component = "decoder";
    g.max_vision_patches = 4096;
    g.prefill_chunk_rows = 256;
    cfg.names.text.prefix = "model.";
    cfg.names.visual_prefix = "visual.";
    return cfg;
}

// ---------------------------------------------------------------------------
// Config parsing
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

}// namespace

[[nodiscard]] Result<GotOcr2Config> parse_got_ocr2_config(luisa::string_view json) {
    using R = Result<GotOcr2Config>;
    JsonDoc doc;
    doc.doc = yyjson_read(json.data(), json.size(), 0);
    if (doc.doc == nullptr) { return R::failure(StatusCode::invalid_argument, "GOT config is not valid JSON", "config"); }
    auto *root = yyjson_doc_get_root(doc.doc);
    if (root == nullptr || !yyjson_is_obj(root)) { return R::failure(StatusCode::invalid_argument, "GOT config must be a JSON object", "config"); }
    if (auto *mt = yyjson_obj_get(root, "model_type"); mt != nullptr && yyjson_is_str(mt)) {
        const luisa::string_view type{yyjson_get_str(mt), yyjson_get_len(mt)};
        if (type.find("got") == luisa::string_view::npos) {
            return R::failure(StatusCode::unsupported, "outer model_type does not name GOT (and is never sufficient for identity)", "model_type");
        }
    }
    for (const char *marker : {"num_experts", "linear_num_key_heads", "linear_key_head_dim", "grpo", "mtp_num_layers"}) {
        if (auto *v = yyjson_obj_get(root, marker); v != nullptr && !yyjson_is_null(v) && !yyjson_is_false(v)) {
            return R::failure(StatusCode::unsupported, luisa::format("GOT config field '{}' is outside the dense contract", marker), marker);
        }
    }
    GotOcr2Config cfg = got_ocr2_synthetic_config();
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
        if (auto s = u("window_patch_size", true, 0, g.vision.window_patch_size); !s.ok()) { return R::failure(s); }
        if (auto s = u("rel_pos_window", true, 0, g.vision.rel_pos_window); !s.ok()) { return R::failure(s); }
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
        // GOT vision uses relative bias, never RoPE.
        if (auto *rk = yyjson_obj_get(vc, "rope_kind"); rk != nullptr && yyjson_is_str(rk)) {
            const luisa::string_view rks{yyjson_get_str(rk), yyjson_get_len(rk)};
            if (rks != "none") {
                return R::failure(StatusCode::unsupported, "GOT vision uses relative-position bias, not RoPE", "vision.rope_kind");
            }
        }
    }
    auto *tc = yyjson_obj_get(root, "text_config");
    if (tc == nullptr || !yyjson_is_obj(tc)) { return R::failure(StatusCode::invalid_argument, "text_config object is required", "text_config"); }
    {
        if (auto *qn = yyjson_obj_get(tc, "qk_norm"); qn != nullptr && yyjson_is_true(qn)) {
            return R::failure(StatusCode::unsupported, "GOT Qwen decoder has no QK-Norm", "text_config.qk_norm");
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
        g.text.rope_pair_layout = RopePairLayout::rotate_half;
    }
    g.packing.patch_size = g.vision.patch_size;
    g.packing.grid_multiple = 1;
    if (auto s = cfg.validate(); !s.ok()) { return R::failure(s); }
    return R::success(std::move(cfg));
}

// ---------------------------------------------------------------------------
// Tensor inventory + signature + real-artifact gate
// ---------------------------------------------------------------------------

[[nodiscard]] luisa::vector<ExpectedTensor> got_ocr2_expected_tensors(const GotOcr2Config &config) {
    return p7_expected_tensors(config.graph, config.names);
}

[[nodiscard]] Status got_ocr2_validate_tensors(luisa::span<const PackageTensor> tensors,
                                               luisa::span<const ExpectedTensor> expected) {
    return p7_validate_tensors(tensors, expected);
}

[[nodiscard]] Status got_ocr2_validate_signature(const GotOcr2Config &config, const GotOcr2Signature &signature) {
    if (signature.checkpoint_revision.empty() || signature.config_fingerprint.empty()) {
        return Status::error(StatusCode::unvalidated,
                             "GOT identity requires a checkpoint revision AND config fingerprint; outer model_type alone is never sufficient",
                             "signature");
    }
    if (signature.tensor_names.empty()) {
        return Status::error(StatusCode::unvalidated, "GOT identity requires the exact tensor signature", "tensor_names");
    }
    const auto expected = got_ocr2_expected_tensors(config);
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

[[nodiscard]] Status got_ocr2_validate_real_artifact(luisa::string_view checkpoint_revision,
                                                     luisa::string_view config_fingerprint) {
    (void)checkpoint_revision;
    (void)config_fingerprint;
    return Status::error(StatusCode::unvalidated,
                         "GOT-OCR2.0 has no qualified official artifact: real checkpoints are refused "
                         "(synthetic recipe-driven path only; see examples/ocr/MODELS.md)",
                         "got_ocr2");
}

// ---------------------------------------------------------------------------
// Memory preflight (host; no allocation)
// ---------------------------------------------------------------------------

[[nodiscard]] Result<GotOcr2MemoryReport> got_ocr2_preflight(
    const GotOcr2Config &config, uint32_t max_seq_len, MemoryLimits limits) {
    using R = Result<GotOcr2MemoryReport>;
    if (auto s = config.validate(); !s.ok()) { return R::failure(s); }
    auto plan = p7_plan_slice_footprint(config.graph, max_seq_len);
    if (!plan.ok()) { return R::failure(plan.status); }
    GotOcr2MemoryReport report;
    report.vision_weight_bytes = plan.value->vision_weight_bytes;
    report.decoder_weight_bytes = plan.value->decoder_weight_bytes;
    report.workspace_bytes = plan.value->workspace_bytes;
    report.kv_cache_bytes = plan.value->kv_cache_bytes;
    report.total_device_bytes = plan.value->total_device_bytes;
    report.available_device_bytes = limits.max_device_bytes;
    report.fits = limits.max_device_bytes == 0u || report.total_device_bytes <= limits.max_device_bytes;
    if (!report.fits) {
        return R::failure(StatusCode::out_of_memory,
                          luisa::format("GOT-OCR2.0 requires {} device bytes (vision {} + decoder {} + workspace {} + KV {}) "
                                        "but only {} bytes are available; reduce context/pixels or raise the limit",
                                        report.total_device_bytes, report.vision_weight_bytes, report.decoder_weight_bytes,
                                        report.workspace_bytes, report.kv_cache_bytes, limits.max_device_bytes),
                          "max_device_bytes");
    }
    return R::success(std::move(report));
}

// ---------------------------------------------------------------------------
// Multi-crop composition
// ---------------------------------------------------------------------------

[[nodiscard]] Result<LayoutDocument> got_ocr2_compose_crops(luisa::span<const GotCropResult> crops,
                                                           GotPromptMode mode) {
    using R = Result<LayoutDocument>;
    if (mode != GotPromptMode::plain && mode != GotPromptMode::formatted) {
        return R::failure(StatusCode::unsupported,
                          "GOT multi-crop composition supports plain and formatted output; region/color are prompt modes, not composition modes",
                          "mode");
    }
    // Sort by (page_index, reading_order) — pages ascending first, reading
    // order within each page (stable: equal keys keep input order).
    luisa::vector<size_t> order(crops.size());
    for (size_t i = 0; i < order.size(); i++) { order[i] = i; }
    std::stable_sort(order.begin(), order.end(), [&crops](size_t a, size_t b) {
        if (crops[a].page_index != crops[b].page_index) { return crops[a].page_index < crops[b].page_index; }
        return crops[a].reading_order < crops[b].reading_order;
    });
    LayoutDocument doc;
    luisa::string canonical;
    for (size_t idx : order) {
        const auto &crop = crops[idx];
        luisa::string content = crop.text;
        if (mode == GotPromptMode::formatted) {
            // Strict Markdown validation; malformed output is an ERROR.
            auto parsed = output::parse_markdown(crop.text);
            if (!parsed.ok()) { return R::failure(parsed.status); }
            auto composed = output::compose_markdown(parsed.value->blocks);
            if (!composed.ok()) { return R::failure(composed.status); }
            content = std::move(*composed.value);
        }
        LayoutLine line;
        line.text = content;
        line.page_index = crop.page_index;
        line.reading_order = crop.reading_order;
        line.source_box = output::DbRect{static_cast<float>(crop.box.x0), static_cast<float>(crop.box.y0),
                                         static_cast<float>(crop.box.x1), static_cast<float>(crop.box.y1)};
        line.has_confidence = false;
        doc.lines.push_back(std::move(line));
        LayoutBlock block;
        block.type = LayoutBlockType::text;
        block.content = content;
        block.page_index = crop.page_index;
        block.reading_order = crop.reading_order;
        block.source_box = output::DbRect{static_cast<float>(crop.box.x0), static_cast<float>(crop.box.y0),
                                          static_cast<float>(crop.box.x1), static_cast<float>(crop.box.y1)};
        block.has_confidence = false;
        doc.blocks.push_back(std::move(block));
        if (!canonical.empty()) { canonical += "\n"; }
        canonical += content;
    }
    doc.text = std::move(canonical);
    return R::success(std::move(doc));
}

}// namespace luisa::ocr

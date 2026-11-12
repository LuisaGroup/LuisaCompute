// OCR engine -- Qwen3-VL graph contract (Phase 4). Host-only.
// Synthetic declared contract + strict config/tensor validation + host image
// packing / interleaved-MRoPE position semantics. See qwen3_vl.h.
#include "ocr/models/qwen3_vl.h"

#include <algorithm>
#include <cmath>
#include <utility>

#include <luisa/core/stl/format.h>
#include <yyjson.h>

namespace luisa::ocr {

[[nodiscard]] luisa::string_view qwen3_vl_variant_name(Qwen3VlVariant variant) noexcept {
    switch (variant) {
        case Qwen3VlVariant::instruct: return "instruct";
        case Qwen3VlVariant::thinking: return "thinking";
    }
    return "unknown";
}

[[nodiscard]] Result<Qwen3VlVariant> qwen3_vl_variant_from_name(luisa::string_view name) {
    using R = Result<Qwen3VlVariant>;
    if (name == "instruct") { return R::success(Qwen3VlVariant::instruct); }
    if (name == "thinking") { return R::success(Qwen3VlVariant::thinking); }
    return R::failure(StatusCode::invalid_argument,
                      luisa::format("unknown Qwen3-VL variant '{}'", name), "variant");
}

// ---------------------------------------------------------------------------
// Geometry validation
// ---------------------------------------------------------------------------

[[nodiscard]] Status Qwen3VlVisionConfig::validate() const {
    const auto nonzero = [](uint32_t v, const char *what) -> Status {
        if (v == 0u) {
            return Status::error(StatusCode::invalid_argument,
                                 luisa::format("vision config field '{}' must be nonzero", what), what);
        }
        return Status::success();
    };
    for (const auto &[v, name] : std::initializer_list<std::pair<uint32_t, const char *>>{
             {num_layers, "num_layers"},
             {hidden_size, "hidden_size"},
             {num_heads, "num_heads"},
             {head_dim, "head_dim"},
             {intermediate_size, "intermediate_size"},
             {patch_size, "patch_size"},
             {spatial_merge_size, "spatial_merge_size"},
             {merger_intermediate_size, "merger_intermediate_size"}}) {
        if (auto s = nonzero(v, name); !s.ok()) { return s; }
    }
    if (channels != 3u) {
        return Status::error(StatusCode::unsupported, "Qwen3-VL vision uses 3-channel RGB input", "channels");
    }
    if (temporal_patch_size != 1u) {
        return Status::error(StatusCode::unsupported,
                             "temporal_patch_size > 1 (video packing) is outside the Phase 4 slice", "temporal_patch_size");
    }
    if (static_cast<uint64_t>(num_heads) * head_dim != hidden_size) {
        return Status::error(StatusCode::invalid_argument,
                             "vision num_heads * head_dim must equal hidden_size", "head_dim");
    }
    if (head_dim % 2u != 0u) {
        return Status::error(StatusCode::invalid_argument, "vision head_dim must be even (RoPE pairs)", "head_dim");
    }
    if (rope_height_pairs == 0u || rope_height_pairs >= head_dim / 2u) {
        return Status::error(StatusCode::invalid_argument,
                             "rope_height_pairs must be in (0, head_dim/2): both height and width axes need pairs",
                             "rope_height_pairs");
    }
    if (!std::isfinite(layer_norm_eps) || layer_norm_eps <= 0.0f) {
        return Status::error(StatusCode::invalid_argument, "layer_norm_eps must be finite and positive", "layer_norm_eps");
    }
    if (!std::isfinite(rope_theta) || rope_theta <= 0.0f) {
        return Status::error(StatusCode::invalid_argument, "rope_theta must be finite and positive", "rope_theta");
    }
    return Status::success();
}

[[nodiscard]] Status Qwen3VlDeepStackConfig::validate(uint32_t num_decoder_layers, uint32_t num_vision_layers) const {
    if (decoder_layers.size() != vision_layers.size()) {
        return Status::error(StatusCode::invalid_argument,
                             "deepstack decoder_layers and vision_layers must have equal length", "deepstack");
    }
    for (uint32_t i = 0; i < decoder_layers.size(); i++) {
        if (decoder_layers[i] >= num_decoder_layers) {
            return Status::error(StatusCode::invalid_argument,
                                 luisa::format("deepstack decoder layer {} is outside the decoder ({} layers)",
                                               decoder_layers[i], num_decoder_layers),
                                 "deepstack.decoder_layers");
        }
        if (vision_layers[i] >= num_vision_layers) {
            return Status::error(StatusCode::invalid_argument,
                                 luisa::format("deepstack vision layer {} is outside the vision tower ({} layers)",
                                               vision_layers[i], num_vision_layers),
                                 "deepstack.vision_layers");
        }
        if (i > 0u && decoder_layers[i] <= decoder_layers[i - 1u]) {
            return Status::error(StatusCode::invalid_argument,
                                 "deepstack decoder layers must be strictly increasing (exact injection order)",
                                 "deepstack.decoder_layers");
        }
        // S7: vision layers must also be strictly increasing (no duplicates,
        // exact source order) so each injection maps to a distinct block.
        if (i > 0u && vision_layers[i] <= vision_layers[i - 1u]) {
            return Status::error(StatusCode::invalid_argument,
                                 "deepstack vision layers must be strictly increasing (no duplicates)",
                                 "deepstack.vision_layers");
        }
    }
    return Status::success();
}

[[nodiscard]] luisa::vector<uint32_t> Qwen3VlMropeSections::axis_of_pair() const {
    luisa::vector<uint32_t> out;
    out.reserve(total());
    for (uint32_t i = 0; i < temporal; i++) { out.push_back(0u); }
    for (uint32_t i = 0; i < height; i++) { out.push_back(1u); }
    for (uint32_t i = 0; i < width; i++) { out.push_back(2u); }
    return out;
}

[[nodiscard]] Status Qwen3VlMropeSections::validate(uint32_t head_dim) const {
    if (head_dim % 2u != 0u) {
        return Status::error(StatusCode::invalid_argument, "mrope head_dim must be even", "head_dim");
    }
    if (temporal == 0u || height == 0u || width == 0u) {
        return Status::error(StatusCode::invalid_argument,
                             "mrope sections must assign at least one pair to every axis", "mrope_section");
    }
    if (total() != head_dim / 2u) {
        return Status::error(StatusCode::invalid_argument,
                             luisa::format("mrope sections sum {} must equal head_dim/2 ({})", total(), head_dim / 2u),
                             "mrope_section");
    }
    return Status::success();
}

[[nodiscard]] Status Qwen3VlConfig::validate() const {
    if (variant != Qwen3VlVariant::instruct) {
        return Status::error(StatusCode::unsupported,
                             "only the Qwen3-VL Instruct recipe is contracted; Thinking fails closed", "variant");
    }
    if (auto s = text.validate(); !s.ok()) { return s; }
    if (auto s = vision.validate(); !s.ok()) { return s; }
    if (auto s = deepstack.validate(text.num_layers, vision.num_layers); !s.ok()) { return s; }
    if (auto s = mrope.validate(text.head_dim); !s.ok()) { return s; }
    // The Qwen3 text contract uses interleaved MRoPE. The shared geometry
    // struct's 1D pair-layout field is unused by this graph (the executor
    // binds the interleaved 3-axis kernel); require the explicit Qwen3
    // rotate-half pair family so a config can never carry the adjacent
    // synthetic 1D convention by accident.
    if (text.rope_pair_layout != RopePairLayout::rotate_half) {
        return Status::error(StatusCode::invalid_argument,
                             "Qwen3-VL text RoPE uses the interleaved (rotate-half family) pair layout",
                             "rope_pair_layout");
    }
    if (vision.merged_dim() == 0u) {
        return Status::error(StatusCode::invalid_argument, "vision merged_dim must be nonzero", "spatial_merge_size");
    }
    return Status::success();
}

// ---------------------------------------------------------------------------
// Config parsing (strict, synthetic contract schema)
// ---------------------------------------------------------------------------

namespace {

struct JsonDoc {
    yyjson_doc *doc{nullptr};
    ~JsonDoc() noexcept {
        if (doc != nullptr) { yyjson_doc_free(doc); }
    }
};

[[nodiscard]] Result<uint32_t> q3_get_u32(yyjson_val *obj, const char *key, bool required, uint32_t fallback) {
    using R = Result<uint32_t>;
    auto *v = yyjson_obj_get(obj, key);
    if (v == nullptr || yyjson_is_null(v)) {
        if (required) {
            return R::failure(StatusCode::invalid_argument,
                              luisa::format("config is missing required field '{}'", key), key);
        }
        return R::success(fallback);
    }
    if (!yyjson_is_uint(v)) {
        return R::failure(StatusCode::invalid_argument,
                          luisa::format("config field '{}' must be a nonnegative integer", key), key);
    }
    if (yyjson_get_uint(v) > UINT32_MAX) {
        return R::failure(StatusCode::invalid_argument,
                          luisa::format("config field '{}' exceeds uint32", key), key);
    }
    return R::success(static_cast<uint32_t>(yyjson_get_uint(v)));
}

[[nodiscard]] Result<float> q3_get_f32(yyjson_val *obj, const char *key, float fallback) {
    using R = Result<float>;
    auto *v = yyjson_obj_get(obj, key);
    if (v == nullptr || yyjson_is_null(v)) { return R::success(fallback); }
    if (!yyjson_is_real(v) && !yyjson_is_uint(v) && !yyjson_is_sint(v)) {
        return R::failure(StatusCode::invalid_argument,
                          luisa::format("config field '{}' must be a number", key), key);
    }
    const auto value = static_cast<float>(yyjson_get_num(v));
    if (!std::isfinite(value)) {
        return R::failure(StatusCode::invalid_argument,
                          luisa::format("config field '{}' must be finite", key), key);
    }
    return R::success(value);
}

[[nodiscard]] Result<luisa::vector<uint32_t>> q3_get_u32_array(yyjson_val *obj, const char *key, size_t exact_size) {
    using R = Result<luisa::vector<uint32_t>>;
    auto *v = yyjson_obj_get(obj, key);
    if (v == nullptr || !yyjson_is_arr(v)) {
        return R::failure(StatusCode::invalid_argument,
                          luisa::format("config field '{}' must be an array", key), key);
    }
    if (yyjson_arr_size(v) != exact_size) {
        return R::failure(StatusCode::invalid_argument,
                          luisa::format("config field '{}' must have exactly {} entries", key, exact_size), key);
    }
    luisa::vector<uint32_t> out;
    size_t idx = 0, max = 0;
    yyjson_val *item = nullptr;
    yyjson_arr_foreach(v, idx, max, item) {
        if (!yyjson_is_uint(item) || yyjson_get_uint(item) > UINT32_MAX) {
            return R::failure(StatusCode::invalid_argument,
                              luisa::format("config field '{}' entries must be nonnegative integers", key), key);
        }
        out.push_back(static_cast<uint32_t>(yyjson_get_uint(item)));
    }
    return R::success(std::move(out));
}

}// namespace

// S8: reject any structural field outside the declared whitelist. This makes
// the contract fail-closed on unknown keys instead of silently ignoring them.
[[nodiscard]] Status q3_check_known_keys(yyjson_val *obj, luisa::span<const char *const> known,
                                         luisa::string_view where) {
    size_t idx = 0, max = 0;
    yyjson_val *key = nullptr, *val = nullptr;
    yyjson_obj_foreach(obj, idx, max, key, val) {
        const luisa::string_view k{yyjson_get_str(key), yyjson_get_len(key)};
        bool found = false;
        for (const char *kn : known) {
            if (k == kn) { found = true; break; }
        }
        if (!found) {
            return Status::error(StatusCode::unsupported,
                                 luisa::format("{} contains unknown structural field '{}' (fail-closed)", where, k),
                                 luisa::string{k});
        }
    }
    return Status::success();
}

// S8: a merely PRESENT but DISABLED sliding window is the ordinary dense path
// (exact) and is accepted; an ENABLED window changes the attention mask and is
// refused. Aligned with the shared Qwen2.5 parser: a positive window with
// use_sliding_window absent defaults to enabled (HF semantics).
[[nodiscard]] Status q3_check_sliding_window(yyjson_val *obj, luisa::string_view where) {
    auto *use = yyjson_obj_get(obj, "use_sliding_window");
    const bool use_present = use != nullptr && !yyjson_is_null(use);
    const bool use_true = use_present && yyjson_is_true(use);
    const bool use_false = use_present && yyjson_is_false(use);
    if (use_present && !use_true && !use_false) {
        return Status::error(StatusCode::invalid_argument,
                             luisa::format("{} use_sliding_window must be a boolean", where), "use_sliding_window");
    }
    auto *win = yyjson_obj_get(obj, "sliding_window");
    const bool win_present = win != nullptr && !yyjson_is_null(win);
    if (win_present) {
        if (!yyjson_is_uint(win)) {
            return Status::error(StatusCode::invalid_argument,
                                 luisa::format("{} sliding_window must be a positive integer or null", where), "sliding_window");
        }
        const bool window_enabled = use_true || !use_present;
        if (window_enabled && yyjson_get_uint(win) > 0u) {
            return Status::error(StatusCode::unsupported,
                                 luisa::format("{} an enabled sliding_window is outside the dense Qwen3-VL contract", where),
                                 "sliding_window");
        }
    } else if (use_true) {
        return Status::error(StatusCode::unsupported,
                             luisa::format("{} use_sliding_window=true without a window is outside the dense Qwen3-VL contract", where),
                             "use_sliding_window");
    }
    return Status::success();
}

[[nodiscard]] Result<Qwen3VlConfig> parse_qwen3_vl_config(luisa::string_view json) {
    using R = Result<Qwen3VlConfig>;
    JsonDoc doc;
    doc.doc = yyjson_read(json.data(), json.size(), 0);
    if (doc.doc == nullptr) {
        return R::failure(StatusCode::invalid_argument, "Qwen3-VL config is not valid JSON", "config");
    }
    auto *root = yyjson_doc_get_root(doc.doc);
    if (root == nullptr || !yyjson_is_obj(root)) {
        return R::failure(StatusCode::invalid_argument, "Qwen3-VL config must be a JSON object", "config");
    }
    // Identity boundary: EXACT model_type identity (not a substring match), so
    // a look-alike like "qwen3_vl_moe" or "not_qwen3_vl" is refused. Qwen2.5
    // markers and Thinking are refused explicitly.
    if (auto *mt = yyjson_obj_get(root, "model_type"); mt != nullptr && !yyjson_is_null(mt)) {
        if (!yyjson_is_str(mt)) {
            return R::failure(StatusCode::invalid_argument, "model_type must be a string", "model_type");
        }
        const luisa::string_view type{yyjson_get_str(mt), yyjson_get_len(mt)};
        if (type != "qwen3_vl") {
            return R::failure(StatusCode::unsupported,
                              luisa::format("model_type '{}' is not the exact qwen3_vl contract (no silent substitution)", type),
                              "model_type");
        }
    }
    Qwen3VlConfig config;
    if (auto *v = yyjson_obj_get(root, "variant"); v != nullptr && !yyjson_is_null(v)) {
        if (!yyjson_is_str(v)) {
            return R::failure(StatusCode::invalid_argument, "variant must be a string", "variant");
        }
        auto vr = qwen3_vl_variant_from_name({yyjson_get_str(v), yyjson_get_len(v)});
        if (!vr.ok()) { return R::failure(vr.status); }
        config.variant = *vr.value;
        if (config.variant == Qwen3VlVariant::thinking) {
            return R::failure(StatusCode::unsupported,
                              "Qwen3-VL-Thinking is NOT implemented: only the Instruct recipe is contracted",
                              "variant");
        }
    }
    // S8: root-level structural whitelist (unknown fields fail closed).
    {
        static constexpr const char *root_keys[] = {
            "model_type", "variant", "text_config", "vision_config", "deepstack",
            "mrope_section", "tie_word_embeddings", "num_experts", "num_experts_per_tok",
            "linear_num_key_heads", "linear_key_head_dim", "sliding_window", "use_sliding_window"};
        if (auto s = q3_check_known_keys(root, root_keys, "config"); !s.ok()) { return R::failure(s); }
    }
    // MoE / GDN markers are outside this dense contract.
    for (const char *marker : {"num_experts", "num_experts_per_tok", "linear_num_key_heads",
                               "linear_key_head_dim"}) {
        if (auto *v = yyjson_obj_get(root, marker); v != nullptr && !yyjson_is_null(v) && !yyjson_is_false(v)) {
            return R::failure(StatusCode::unsupported,
                              luisa::format("config field '{}' is outside the dense Qwen3-VL contract", marker),
                              marker);
        }
    }
    // Sliding window: disabled accepted, enabled refused (aligned with Qwen2.5).
    if (auto s = q3_check_sliding_window(root, "config"); !s.ok()) { return R::failure(s); }
    // Text decoder geometry: reuse the shared dense geometry parser but with
    // the qwen3 identity boundary INVERTED -- parse_qwen2_5_text_config
    // refuses qwen3 markers, so parse the geometry keys here directly.
    auto *text = yyjson_obj_get(root, "text_config");
    if (text == nullptr || !yyjson_is_obj(text)) {
        return R::failure(StatusCode::invalid_argument, "text_config must be an object", "text_config");
    }
    // S8: text_config structural whitelist.
    {
        static constexpr const char *text_keys[] = {
            "vocab_size", "hidden_size", "num_hidden_layers", "num_attention_heads",
            "num_key_value_heads", "intermediate_size", "max_position_embeddings",
            "head_dim", "rms_norm_eps", "rope_theta", "eos_token_id", "attention_bias",
            "num_experts", "linear_num_key_heads", "linear_key_head_dim",
            "sliding_window", "use_sliding_window"};
        if (auto s = q3_check_known_keys(text, text_keys, "text_config"); !s.ok()) { return R::failure(s); }
    }
    for (const char *marker : {"num_experts", "linear_num_key_heads", "linear_key_head_dim"}) {
        if (auto *v = yyjson_obj_get(text, marker); v != nullptr && !yyjson_is_null(v) && !yyjson_is_false(v)) {
            return R::failure(StatusCode::unsupported,
                              luisa::format("text_config field '{}' is outside the dense Qwen3-VL contract", marker),
                              marker);
        }
    }
    // S8: the Qwen3 text contract has NO attention (QKV) bias. attention_bias
    // true is refused; false/absent is the contract.
    if (auto *ab = yyjson_obj_get(text, "attention_bias"); ab != nullptr && !yyjson_is_null(ab)) {
        if (!yyjson_is_bool(ab)) {
            return R::failure(StatusCode::invalid_argument, "attention_bias must be a boolean", "attention_bias");
        }
        if (yyjson_is_true(ab)) {
            return R::failure(StatusCode::unsupported,
                              "attention_bias=true is outside the Qwen3-VL contract (no QKV bias)", "attention_bias");
        }
    }
    if (auto s = q3_check_sliding_window(text, "text_config"); !s.ok()) { return R::failure(s); }
    auto tu32 = [&](const char *key, bool required, uint32_t fallback, uint32_t &out) -> Status {
        auto r = q3_get_u32(text, key, required, fallback);
        if (!r.ok()) { return r.status; }
        out = *r.value;
        return Status::success();
    };
    if (auto s = tu32("vocab_size", true, 0, config.text.vocab_size); !s.ok()) { return R::failure(s); }
    if (auto s = tu32("hidden_size", true, 0, config.text.hidden_size); !s.ok()) { return R::failure(s); }
    if (auto s = tu32("num_hidden_layers", true, 0, config.text.num_layers); !s.ok()) { return R::failure(s); }
    if (auto s = tu32("num_attention_heads", true, 0, config.text.num_attention_heads); !s.ok()) { return R::failure(s); }
    if (auto s = tu32("num_key_value_heads", true, 0, config.text.num_kv_heads); !s.ok()) { return R::failure(s); }
    if (auto s = tu32("intermediate_size", true, 0, config.text.intermediate_size); !s.ok()) { return R::failure(s); }
    if (auto s = tu32("max_position_embeddings", true, 0, config.text.max_position_embeddings); !s.ok()) { return R::failure(s); }
    const uint32_t default_head_dim = config.text.num_attention_heads == 0u
                                          ? 0u
                                          : config.text.hidden_size / config.text.num_attention_heads;
    if (auto s = tu32("head_dim", false, default_head_dim, config.text.head_dim); !s.ok()) { return R::failure(s); }
    {
        auto eps = q3_get_f32(text, "rms_norm_eps", 1e-6f);
        if (!eps.ok()) { return R::failure(eps.status); }
        config.text.rms_norm_eps = *eps.value;
        auto theta = q3_get_f32(text, "rope_theta", 10000.0f);
        if (!theta.ok()) { return R::failure(theta.status); }
        config.text.rope_theta = *theta.value;
    }
    config.text.rope_pair_layout = RopePairLayout::rotate_half;
    if (auto *eos = yyjson_obj_get(text, "eos_token_id"); eos != nullptr && !yyjson_is_null(eos)) {
        if (yyjson_is_uint(eos)) {
            if (yyjson_get_uint(eos) > UINT32_MAX) {
                return R::failure(StatusCode::invalid_argument, "eos_token_id exceeds uint32", "eos_token_id");
            }
            config.text.eos_token_ids.push_back(static_cast<uint32_t>(yyjson_get_uint(eos)));
        } else if (yyjson_is_arr(eos)) {
            size_t idx = 0, max = 0;
            yyjson_val *item = nullptr;
            yyjson_arr_foreach(eos, idx, max, item) {
                if (!yyjson_is_uint(item) || yyjson_get_uint(item) > UINT32_MAX) {
                    return R::failure(StatusCode::invalid_argument,
                                      "eos_token_id entries must be nonnegative integers", "eos_token_id");
                }
                config.text.eos_token_ids.push_back(static_cast<uint32_t>(yyjson_get_uint(item)));
            }
        } else {
            return R::failure(StatusCode::invalid_argument,
                              "eos_token_id must be an integer or a list of integers", "eos_token_id");
        }
    }
    // Vision config.
    auto *vis = yyjson_obj_get(root, "vision_config");
    if (vis == nullptr || !yyjson_is_obj(vis)) {
        return R::failure(StatusCode::invalid_argument, "vision_config must be an object", "vision_config");
    }
    // S8: vision_config structural whitelist.
    {
        static constexpr const char *vision_keys[] = {
            "num_layers", "hidden_size", "num_heads", "head_dim", "intermediate_size",
            "patch_size", "temporal_patch_size", "spatial_merge_size",
            "merger_intermediate_size", "rope_height_pairs", "layer_norm_eps", "rope_theta"};
        if (auto s = q3_check_known_keys(vis, vision_keys, "vision_config"); !s.ok()) { return R::failure(s); }
    }
    auto vu32 = [&](const char *key, bool required, uint32_t fallback, uint32_t &out) -> Status {
        auto r = q3_get_u32(vis, key, required, fallback);
        if (!r.ok()) { return r.status; }
        out = *r.value;
        return Status::success();
    };
    if (auto s = vu32("num_layers", true, 0, config.vision.num_layers); !s.ok()) { return R::failure(s); }
    if (auto s = vu32("hidden_size", true, 0, config.vision.hidden_size); !s.ok()) { return R::failure(s); }
    if (auto s = vu32("num_heads", true, 0, config.vision.num_heads); !s.ok()) { return R::failure(s); }
    if (auto s = vu32("head_dim", true, 0, config.vision.head_dim); !s.ok()) { return R::failure(s); }
    if (auto s = vu32("intermediate_size", true, 0, config.vision.intermediate_size); !s.ok()) { return R::failure(s); }
    if (auto s = vu32("patch_size", true, 0, config.vision.patch_size); !s.ok()) { return R::failure(s); }
    if (auto s = vu32("temporal_patch_size", false, 1, config.vision.temporal_patch_size); !s.ok()) { return R::failure(s); }
    if (auto s = vu32("spatial_merge_size", true, 0, config.vision.spatial_merge_size); !s.ok()) { return R::failure(s); }
    if (auto s = vu32("merger_intermediate_size", true, 0, config.vision.merger_intermediate_size); !s.ok()) { return R::failure(s); }
    if (auto s = vu32("rope_height_pairs", true, 0, config.vision.rope_height_pairs); !s.ok()) { return R::failure(s); }
    {
        auto eps = q3_get_f32(vis, "layer_norm_eps", 1e-6f);
        if (!eps.ok()) { return R::failure(eps.status); }
        config.vision.layer_norm_eps = *eps.value;
        auto theta = q3_get_f32(vis, "rope_theta", 10000.0f);
        if (!theta.ok()) { return R::failure(theta.status); }
        config.vision.rope_theta = *theta.value;
    }
    // DeepStack (optional: absent = no DeepStack branches).
    if (auto *ds = yyjson_obj_get(root, "deepstack"); ds != nullptr && !yyjson_is_null(ds)) {
        if (!yyjson_is_obj(ds)) {
            return R::failure(StatusCode::invalid_argument, "deepstack must be an object", "deepstack");
        }
        // S8: deepstack structural whitelist.
        static constexpr const char *ds_keys[] = {"decoder_layers", "vision_layers"};
        if (auto s = q3_check_known_keys(ds, ds_keys, "deepstack"); !s.ok()) { return R::failure(s); }
        auto *dl = yyjson_obj_get(ds, "decoder_layers");
        auto *vl = yyjson_obj_get(ds, "vision_layers");
        if (dl == nullptr || vl == nullptr || !yyjson_is_arr(dl) || !yyjson_is_arr(vl)) {
            return R::failure(StatusCode::invalid_argument,
                              "deepstack requires decoder_layers and vision_layers arrays", "deepstack");
        }
        auto dr = q3_get_u32_array(ds, "decoder_layers", yyjson_arr_size(dl));
        if (!dr.ok()) { return R::failure(dr.status); }
        auto vr = q3_get_u32_array(ds, "vision_layers", yyjson_arr_size(vl));
        if (!vr.ok()) { return R::failure(vr.status); }
        config.deepstack.decoder_layers = std::move(*dr.value);
        config.deepstack.vision_layers = std::move(*vr.value);
    }
    // MRoPE sections [t, h, w] (required: the axis semantics are explicit).
    {
        auto r = q3_get_u32_array(root, "mrope_section", 3);
        if (!r.ok()) { return R::failure(r.status); }
        config.mrope.temporal = (*r.value)[0];
        config.mrope.height = (*r.value)[1];
        config.mrope.width = (*r.value)[2];
    }
    if (auto *tie = yyjson_obj_get(root, "tie_word_embeddings"); tie != nullptr && !yyjson_is_null(tie)) {
        if (!yyjson_is_bool(tie)) {
            return R::failure(StatusCode::invalid_argument, "tie_word_embeddings must be a boolean", "tie_word_embeddings");
        }
        config.tie_word_embeddings = yyjson_is_true(tie);
    }
    if (auto s = config.validate(); !s.ok()) { return R::failure(s); }
    return R::success(std::move(config));
}

[[nodiscard]] Qwen3VlConfig qwen3_vl_synthetic_config() {
    Qwen3VlConfig c;
    c.variant = Qwen3VlVariant::instruct;
    c.text.vocab_size = 32;
    c.text.hidden_size = 32;
    c.text.num_layers = 3;
    c.text.num_attention_heads = 4;
    c.text.num_kv_heads = 2;
    c.text.head_dim = 8;
    c.text.intermediate_size = 64;
    c.text.rms_norm_eps = 1e-5f;
    c.text.rope_theta = 10000.0f;
    c.text.rope_pair_layout = RopePairLayout::rotate_half;
    c.text.max_position_embeddings = 128;
    c.text.eos_token_ids = {31};
    c.vision.num_layers = 2;
    c.vision.hidden_size = 16;
    c.vision.num_heads = 2;
    c.vision.head_dim = 8;
    c.vision.intermediate_size = 32;
    c.vision.patch_size = 2;
    c.vision.temporal_patch_size = 1;
    c.vision.channels = 3;
    c.vision.spatial_merge_size = 2;
    c.vision.merger_intermediate_size = 32;
    c.vision.layer_norm_eps = 1e-5f;
    c.vision.rope_theta = 10000.0f;
    c.vision.rope_height_pairs = 2;
    c.deepstack.decoder_layers = {0u, 1u};
    c.deepstack.vision_layers = {0u, 1u};
    c.mrope.temporal = 2;
    c.mrope.height = 1;
    c.mrope.width = 1;
    c.tie_word_embeddings = true;
    return c;
}

// ---------------------------------------------------------------------------
// Tensor names + inventory
// ---------------------------------------------------------------------------

[[nodiscard]] Status Qwen3VlTensorNames::validate() const {
    if (auto s = text.validate(); !s.ok()) { return s; }
    if (visual_prefix.empty() || visual_prefix.back() != '.') {
        return Status::error(StatusCode::invalid_argument,
                             "visual_prefix must be non-empty and end with '.'", "visual_prefix");
    }
    return Status::success();
}

[[nodiscard]] luisa::string Qwen3VlTensorNames::patch_embed_weight() const { return visual_prefix + "patch_embed.weight"; }
[[nodiscard]] luisa::string Qwen3VlTensorNames::patch_embed_bias() const { return visual_prefix + "patch_embed.bias"; }
[[nodiscard]] luisa::string Qwen3VlTensorNames::block_prefix(uint32_t layer) const {
    return visual_prefix + "blocks." + luisa::format("{}", layer) + ".";
}
[[nodiscard]] luisa::string Qwen3VlTensorNames::block_norm1_weight(uint32_t layer) const { return block_prefix(layer) + "norm1.weight"; }
[[nodiscard]] luisa::string Qwen3VlTensorNames::block_norm1_bias(uint32_t layer) const { return block_prefix(layer) + "norm1.bias"; }
[[nodiscard]] luisa::string Qwen3VlTensorNames::block_norm2_weight(uint32_t layer) const { return block_prefix(layer) + "norm2.weight"; }
[[nodiscard]] luisa::string Qwen3VlTensorNames::block_norm2_bias(uint32_t layer) const { return block_prefix(layer) + "norm2.bias"; }
[[nodiscard]] luisa::string Qwen3VlTensorNames::block_q_proj(uint32_t layer) const { return block_prefix(layer) + "attn.q_proj.weight"; }
[[nodiscard]] luisa::string Qwen3VlTensorNames::block_k_proj(uint32_t layer) const { return block_prefix(layer) + "attn.k_proj.weight"; }
[[nodiscard]] luisa::string Qwen3VlTensorNames::block_v_proj(uint32_t layer) const { return block_prefix(layer) + "attn.v_proj.weight"; }
[[nodiscard]] luisa::string Qwen3VlTensorNames::block_o_proj(uint32_t layer) const { return block_prefix(layer) + "attn.o_proj.weight"; }
[[nodiscard]] luisa::string Qwen3VlTensorNames::block_q_bias(uint32_t layer) const { return block_prefix(layer) + "attn.q_proj.bias"; }
[[nodiscard]] luisa::string Qwen3VlTensorNames::block_k_bias(uint32_t layer) const { return block_prefix(layer) + "attn.k_proj.bias"; }
[[nodiscard]] luisa::string Qwen3VlTensorNames::block_v_bias(uint32_t layer) const { return block_prefix(layer) + "attn.v_proj.bias"; }
[[nodiscard]] luisa::string Qwen3VlTensorNames::block_o_bias(uint32_t layer) const { return block_prefix(layer) + "attn.o_proj.bias"; }
[[nodiscard]] luisa::string Qwen3VlTensorNames::block_q_norm(uint32_t layer) const { return block_prefix(layer) + "attn.q_norm.weight"; }
[[nodiscard]] luisa::string Qwen3VlTensorNames::block_k_norm(uint32_t layer) const { return block_prefix(layer) + "attn.k_norm.weight"; }
[[nodiscard]] luisa::string Qwen3VlTensorNames::block_fc1(uint32_t layer) const { return block_prefix(layer) + "mlp.fc1.weight"; }
[[nodiscard]] luisa::string Qwen3VlTensorNames::block_fc1_bias(uint32_t layer) const { return block_prefix(layer) + "mlp.fc1.bias"; }
[[nodiscard]] luisa::string Qwen3VlTensorNames::block_fc2(uint32_t layer) const { return block_prefix(layer) + "mlp.fc2.weight"; }
[[nodiscard]] luisa::string Qwen3VlTensorNames::block_fc2_bias(uint32_t layer) const { return block_prefix(layer) + "mlp.fc2.bias"; }
[[nodiscard]] luisa::string Qwen3VlTensorNames::merger_prefix() const { return visual_prefix + "merger."; }
[[nodiscard]] luisa::string Qwen3VlTensorNames::deepstack_prefix(uint32_t index) const {
    return visual_prefix + "deepstack_merger." + luisa::format("{}", index) + ".";
}
[[nodiscard]] luisa::string Qwen3VlTensorNames::merger_ln_weight(const luisa::string &prefix) { return prefix + "ln.weight"; }
[[nodiscard]] luisa::string Qwen3VlTensorNames::merger_ln_bias(const luisa::string &prefix) { return prefix + "ln.bias"; }
[[nodiscard]] luisa::string Qwen3VlTensorNames::merger_fc1(const luisa::string &prefix) { return prefix + "fc1.weight"; }
[[nodiscard]] luisa::string Qwen3VlTensorNames::merger_fc1_bias(const luisa::string &prefix) { return prefix + "fc1.bias"; }
[[nodiscard]] luisa::string Qwen3VlTensorNames::merger_fc2(const luisa::string &prefix) { return prefix + "fc2.weight"; }
[[nodiscard]] luisa::string Qwen3VlTensorNames::merger_fc2_bias(const luisa::string &prefix) { return prefix + "fc2.bias"; }

[[nodiscard]] luisa::vector<ExpectedTensor> qwen3_vl_expected_tensors(
    const Qwen3VlConfig &config, const Qwen3VlTensorNames &names) {
    luisa::vector<ExpectedTensor> out;
    const auto add = [&out](luisa::string name, luisa::vector<uint64_t> shape,
                            TensorLayout layout, bool required = true) {
        out.emplace_back(ExpectedTensor{std::move(name), std::move(shape), layout, required});
    };
    const auto &v = config.vision;
    const uint64_t vh = v.hidden_size, vi = v.intermediate_size;
    const uint64_t merged = v.merged_dim();
    const uint64_t mi = v.merger_intermediate_size;
    // Patch embed: linear over flattened (C, patch, patch) rows -> hidden.
    add(names.patch_embed_weight(), {v.patch_dim(), vh}, TensorLayout::kn);
    add(names.patch_embed_bias(), {vh}, TensorLayout::row_major);
    for (uint32_t l = 0; l < v.num_layers; l++) {
        add(names.block_norm1_weight(l), {vh}, TensorLayout::row_major);
        add(names.block_norm1_bias(l), {vh}, TensorLayout::row_major);
        add(names.block_norm2_weight(l), {vh}, TensorLayout::row_major);
        add(names.block_norm2_bias(l), {vh}, TensorLayout::row_major);
        add(names.block_q_proj(l), {vh, vh}, TensorLayout::kn);
        add(names.block_k_proj(l), {vh, vh}, TensorLayout::kn);
        add(names.block_v_proj(l), {vh, vh}, TensorLayout::kn);
        add(names.block_o_proj(l), {vh, vh}, TensorLayout::kn);
        add(names.block_q_bias(l), {vh}, TensorLayout::row_major);
        add(names.block_k_bias(l), {vh}, TensorLayout::row_major);
        add(names.block_v_bias(l), {vh}, TensorLayout::row_major);
        add(names.block_o_bias(l), {vh}, TensorLayout::row_major);
        add(names.block_q_norm(l), {v.head_dim}, TensorLayout::row_major);
        add(names.block_k_norm(l), {v.head_dim}, TensorLayout::row_major);
        add(names.block_fc1(l), {vh, vi}, TensorLayout::kn);
        add(names.block_fc1_bias(l), {vi}, TensorLayout::row_major);
        add(names.block_fc2(l), {vi, vh}, TensorLayout::kn);
        add(names.block_fc2_bias(l), {vh}, TensorLayout::row_major);
    }
    // Main merger: LayerNorm over the merged dim, then fc1/fc2 to text hidden.
    const auto add_merger = [&](const luisa::string &prefix, uint64_t out_dim) {
        add(Qwen3VlTensorNames::merger_ln_weight(prefix), {merged}, TensorLayout::row_major);
        add(Qwen3VlTensorNames::merger_ln_bias(prefix), {merged}, TensorLayout::row_major);
        add(Qwen3VlTensorNames::merger_fc1(prefix), {merged, mi}, TensorLayout::kn);
        add(Qwen3VlTensorNames::merger_fc1_bias(prefix), {mi}, TensorLayout::row_major);
        add(Qwen3VlTensorNames::merger_fc2(prefix), {mi, out_dim}, TensorLayout::kn);
        add(Qwen3VlTensorNames::merger_fc2_bias(prefix), {out_dim}, TensorLayout::row_major);
    };
    add_merger(names.merger_prefix(), config.text.hidden_size);
    for (uint32_t i = 0; i < config.deepstack.count(); i++) {
        add_merger(names.deepstack_prefix(i), config.text.hidden_size);
    }
    // Text decoder: Qwen3 contract (QK-Norm required, NO QKV bias).
    const auto &t = config.text;
    const auto &tn = names.text;
    const uint64_t hidden = t.hidden_size, inter = t.intermediate_size;
    const uint64_t qd = t.q_dim(), kvd = t.kv_dim();
    add(tn.embed_tokens(), {t.vocab_size, hidden}, TensorLayout::row_major);
    add(tn.final_norm(), {hidden}, TensorLayout::row_major);
    add(tn.lm_head(), {hidden, t.vocab_size}, TensorLayout::kn, !config.tie_word_embeddings);
    for (uint32_t l = 0; l < t.num_layers; l++) {
        add(tn.input_layernorm(l), {hidden}, TensorLayout::row_major);
        add(tn.post_attention_layernorm(l), {hidden}, TensorLayout::row_major);
        add(tn.q_proj(l), {hidden, qd}, TensorLayout::kn);
        add(tn.k_proj(l), {hidden, kvd}, TensorLayout::kn);
        add(tn.v_proj(l), {hidden, kvd}, TensorLayout::kn);
        add(tn.o_proj(l), {qd, hidden}, TensorLayout::kn);
        add(tn.gate_proj(l), {hidden, inter}, TensorLayout::kn);
        add(tn.up_proj(l), {hidden, inter}, TensorLayout::kn);
        add(tn.down_proj(l), {inter, hidden}, TensorLayout::kn);
        // Qwen3 QK-Norm weights (per-head RMSNorm over head_dim).
        add(tn.layer_prefix(l) + "self_attn.q_norm.weight", {t.head_dim}, TensorLayout::row_major);
        add(tn.layer_prefix(l) + "self_attn.k_norm.weight", {t.head_dim}, TensorLayout::row_major);
    }
    return out;
}

[[nodiscard]] Status validate_qwen3_vl_tensors(luisa::span<const PackageTensor> tensors,
                                               luisa::span<const ExpectedTensor> expected) {
    // Reuse the shared exact-shape/layout/fp32 checker (it is contract-agnostic).
    return validate_dense_decoder_tensors(tensors, expected);
}

// ---------------------------------------------------------------------------
// Image packing
// ---------------------------------------------------------------------------

[[nodiscard]] Status Qwen3VlPackingSpec::validate() const {
    if (patch_size == 0u || spatial_merge_size == 0u) {
        return Status::error(StatusCode::invalid_argument, "packing patch_size/spatial_merge_size must be nonzero", "packing");
    }
    if (min_pixels == 0u || max_pixels == 0u || min_pixels > max_pixels) {
        return Status::error(StatusCode::invalid_argument,
                             "packing pixel bounds must satisfy 0 < min_pixels <= max_pixels", "packing");
    }
    const uint64_t factor = static_cast<uint64_t>(patch_size) * spatial_merge_size;
    if (factor > UINT32_MAX) {
        return Status::error(StatusCode::invalid_argument, "packing factor overflow", "packing");
    }
    return Status::success();
}

[[nodiscard]] Result<Qwen3VlPackedImage> qwen3_vl_pack_image(uint32_t source_width, uint32_t source_height,
                                                             const Qwen3VlPackingSpec &spec) {
    using R = Result<Qwen3VlPackedImage>;
    if (auto s = spec.validate(); !s.ok()) { return R::failure(s); }
    if (source_width == 0u || source_height == 0u) {
        return R::failure(StatusCode::invalid_argument, "degenerate source image size", "image");
    }
    const uint32_t factor = spec.patch_size * spec.spatial_merge_size;
    const uint64_t pixels = static_cast<uint64_t>(source_width) * source_height;
    // Uniform aspect-preserving scale into the pixel budget.
    double scale = 1.0;
    if (pixels > spec.max_pixels) {
        scale = std::sqrt(static_cast<double>(spec.max_pixels) / static_cast<double>(pixels));
    } else if (pixels < spec.min_pixels) {
        scale = std::sqrt(static_cast<double>(spec.min_pixels) / static_cast<double>(pixels));
    }
    const auto round_to = [factor](double v) -> uint32_t {
        // Round to the NEAREST positive multiple of factor (>= factor).
        const double units = v / static_cast<double>(factor);
        uint64_t n = static_cast<uint64_t>(units + 0.5);
        if (n == 0u) { n = 1u; }
        if (n > UINT32_MAX / factor) { n = UINT32_MAX / factor; }
        return static_cast<uint32_t>(n * factor);
    };
    // S6: rounding to the patch*merge grid can push the pixel count back over
    // the declared cap (e.g. a very wide 4000x1 image). After the initial
    // rounding, re-check the budget and shrink the grid (floor, never below
    // one factor per edge) until it fits. If even the minimal grid exceeds the
    // cap, fail closed — the result NEVER exceeds max_pixels.
    uint32_t rw = round_to(static_cast<double>(source_width) * scale);
    uint32_t rh = round_to(static_cast<double>(source_height) * scale);
    const auto area = [](uint32_t w, uint32_t h) -> uint64_t {
        return static_cast<uint64_t>(w) * h;
    };
    if (area(rw, rh) > spec.max_pixels) {
        // Reduce the longer edge by one factor step at a time, keeping the
        // aspect ratio as close as possible, until the area fits.
        while (area(rw, rh) > spec.max_pixels) {
            if (rw >= rh && rw > factor) {
                rw -= factor;
            } else if (rh > factor) {
                rh -= factor;
            } else {
                // Both edges are at the minimum and still over budget.
                return R::failure(StatusCode::invalid_argument,
                                  "image cannot fit the declared max_pixels budget at the patch/merge grid",
                                  "max_pixels");
            }
        }
    }
    // The min_pixels bound is a soft lower target: rounding/shrinking for the
    // cap takes precedence, but we never silently exceed the cap. (A result
    // below min_pixels is acceptable when the cap forces it; the cap is the
    // hard contract.)
    Qwen3VlPackedImage packed;
    packed.source_width = source_width;
    packed.source_height = source_height;
    packed.resized_width = rw;
    packed.resized_height = rh;
    packed.grid_w = packed.resized_width / spec.patch_size;
    packed.grid_h = packed.resized_height / spec.patch_size;
    packed.grid_t = 1u;
    packed.spatial_merge_size = spec.spatial_merge_size;
    packed.patch_count = static_cast<uint64_t>(packed.grid_t) * packed.grid_h * packed.grid_w;
    const uint64_t mm = static_cast<uint64_t>(spec.spatial_merge_size) * spec.spatial_merge_size;
    packed.merged_count = packed.patch_count / mm;
    packed.transform.scale_x = static_cast<float>(source_width) / static_cast<float>(packed.resized_width);
    packed.transform.scale_y = static_cast<float>(source_height) / static_cast<float>(packed.resized_height);
    return R::success(std::move(packed));
}

// ---------------------------------------------------------------------------
// Interleaved M-RoPE positions
// ---------------------------------------------------------------------------

[[nodiscard]] Result<Qwen3VlPositionIds> qwen3_vl_build_position_ids(
    luisa::span<const Qwen3VlPromptSegment> segments) {
    using R = Result<Qwen3VlPositionIds>;
    Qwen3VlPositionIds out;
    int32_t pos = 0;
    for (const auto &seg : segments) {
        if (seg.image != nullptr) {
            if (seg.text_tokens != 0u) {
                return R::failure(StatusCode::invalid_argument,
                                  "a prompt segment is either text or one image, not both", "segments");
            }
            const auto &img = *seg.image;
            if (img.patch_count == 0u || img.merged_count == 0u || img.spatial_merge_size == 0u) {
                return R::failure(StatusCode::invalid_argument, "image segment has no packed patches", "segments");
            }
            const int32_t start = pos;
            const uint32_t mgh = img.merged_grid_h(), mgw = img.merged_grid_w();
            for (uint32_t ti = 0; ti < img.grid_t; ti++) {
                for (uint32_t hi = 0; hi < mgh; hi++) {
                    for (uint32_t wi = 0; wi < mgw; wi++) {
                        out.ids.push_back(start + static_cast<int32_t>(ti));
                        out.ids.push_back(start + static_cast<int32_t>(hi));
                        out.ids.push_back(start + static_cast<int32_t>(wi));
                        out.seq_len++;
                    }
                }
            }
            const uint32_t extent = std::max({img.grid_t, mgh, mgw});
            pos = start + static_cast<int32_t>(extent);
        } else {
            for (uint32_t i = 0; i < seg.text_tokens; i++) {
                out.ids.push_back(pos);
                out.ids.push_back(pos);
                out.ids.push_back(pos);
                out.seq_len++;
                pos++;
            }
        }
    }
    out.next_text_position = pos;
    return R::success(std::move(out));
}

[[nodiscard]] luisa::vector<int32_t> qwen3_vl_vision_positions(const Qwen3VlPackedImage &packed) {
    luisa::vector<int32_t> out;
    out.reserve(static_cast<size_t>(packed.patch_count) * 2u);
    for (uint32_t h = 0; h < packed.grid_h; h++) {
        for (uint32_t w = 0; w < packed.grid_w; w++) {
            out.push_back(static_cast<int32_t>(h));
            out.push_back(static_cast<int32_t>(w));
        }
    }
    return out;
}

[[nodiscard]] luisa::vector<uint32_t> qwen3_vl_vision_axis_of_pair(const Qwen3VlVisionConfig &vision) {
    luisa::vector<uint32_t> out;
    const uint32_t pairs = vision.head_dim / 2u;
    out.reserve(pairs);
    for (uint32_t d = 0; d < pairs; d++) {
        out.push_back(d < vision.rope_height_pairs ? 0u : 1u);
    }
    return out;
}

}// namespace luisa::ocr

// OCR engine -- SHARED dense decoder components (Phase 3). Host-only.
// Recipe-driven geometry + exact tensor-name/layout validation. NOT a model.
#include "ocr/models/qwen2_5_vl.h"

#include <cmath>

#include <luisa/core/stl/format.h>

#include <yyjson.h>

namespace luisa::ocr {

[[nodiscard]] luisa::string_view rope_pair_layout_name(RopePairLayout layout) noexcept {
    switch (layout) {
        case RopePairLayout::adjacent: return "adjacent";
        case RopePairLayout::rotate_half: return "rotate_half";
    }
    return "unknown";
}

[[nodiscard]] luisa::string_view dense_linear_storage_name(DenseLinearStorage storage) noexcept {
    switch (storage) {
        case DenseLinearStorage::fp32: return "fp32";
        case DenseLinearStorage::int4_sym: return "int4_sym";
        case DenseLinearStorage::int4_asym: return "int4_asym";
    }
    return "unknown";
}

[[nodiscard]] bool dense_linear_is_quantizable(luisa::string_view name,
                                               luisa::string_view decoder_prefix,
                                               luisa::string_view lm_head_name) noexcept {
    // The standalone lm_head is a decoder linear B operand.
    if (!lm_head_name.empty() && name == lm_head_name) { return true; }
    // Only tensors under the decoder prefix can be decoder linears. Vision /
    // connector tensors never share this prefix, so their projection suffixes
    // (visual.blocks.*.attn.q_proj.weight, ...) can never qualify.
    if (decoder_prefix.empty() || name.size() < decoder_prefix.size() ||
        name.compare(0, decoder_prefix.size(), decoder_prefix) != 0) {
        return false;
    }
    for (const char *suffix : {".q_proj.weight", ".k_proj.weight", ".v_proj.weight", ".o_proj.weight",
                               ".gate_proj.weight", ".up_proj.weight", ".down_proj.weight"}) {
        const std::string_view s{suffix};
        if (name.size() >= s.size() && name.compare(name.size() - s.size(), s.size(), s) == 0) { return true; }
    }
    return false;
}

[[nodiscard]] Status validate_dense_decoder_tensors_storage(
    luisa::span<const PackageTensor> tensors,
    luisa::span<const ExpectedTensor> expected,
    DenseLinearStorage linear_storage,
    luisa::string_view decoder_prefix,
    luisa::string_view lm_head_name) {
    // Reject unexpected tensor names (the inventory is exact).
    for (const auto &t : tensors) {
        bool known = false;
        for (const auto &e : expected) { if (e.name == t.name) { known = true; break; } }
        if (!known) {
            return Status::error(StatusCode::invalid_argument,
                                 luisa::format("unexpected tensor '{}' in the package table", t.name), t.name);
        }
    }
    for (const auto &exp : expected) {
        const PackageTensor *found = nullptr;
        for (const auto &t : tensors) {
            if (t.name == exp.name) { found = &t; break; }
        }
        if (found == nullptr) {
            if (exp.required) {
                return Status::error(StatusCode::not_found, luisa::format("required tensor '{}' is missing", exp.name), exp.name);
            }
            continue;
        }
        if (found->shape != exp.shape) {
            return Status::error(StatusCode::invalid_argument, luisa::format("tensor '{}' has the wrong shape", exp.name), exp.name);
        }
        if (found->layout != exp.layout) {
            return Status::error(StatusCode::invalid_argument, luisa::format("tensor '{}' has the wrong layout", exp.name), exp.name);
        }
        const bool quantizable = linear_storage != DenseLinearStorage::fp32 &&
                                 dense_linear_is_quantizable(exp.name, decoder_prefix, lm_head_name);
        if (quantizable) {
            const auto want_kind = linear_storage == DenseLinearStorage::int4_sym
                                       ? QuantKind::int4_symmetric
                                       : QuantKind::int4_asymmetric;
            if (found->encoding != TensorEncoding::int4_packed || found->quant_kind != want_kind) {
                return Status::error(StatusCode::unsupported,
                                     luisa::format("tensor '{}' must be packed {} for the W4A32 path", exp.name, quant_kind_name(want_kind)),
                                     exp.name);
            }
            if (!is_supported_group_size(found->group_size)) {
                return Status::error(StatusCode::invalid_argument, luisa::format("tensor '{}' has an unsupported group size", exp.name), exp.name);
            }
        } else if (found->encoding != TensorEncoding::fp32) {
            return Status::error(StatusCode::unsupported,
                                 luisa::format("tensor '{}' must be dense fp32 (only decoder linears are quantizable)", exp.name), exp.name);
        }
    }
    return Status::success();
}

[[nodiscard]] Status DenseDecoderConfig::validate() const {
    const auto nonzero = [](uint32_t v, const char *what) -> Status {
        if (v == 0u) {
            return Status::error(StatusCode::invalid_argument,
                                 luisa::format("decoder config field '{}' must be nonzero", what), what);
        }
        return Status::success();
    };
    for (const auto &[v, name] : std::initializer_list<std::pair<uint32_t, const char *>>{
             {vocab_size, "vocab_size"},
             {hidden_size, "hidden_size"},
             {num_layers, "num_layers"},
             {num_attention_heads, "num_attention_heads"},
             {num_kv_heads, "num_kv_heads"},
             {head_dim, "head_dim"},
             {intermediate_size, "intermediate_size"},
             {max_position_embeddings, "max_position_embeddings"}}) {
        auto s = nonzero(v, name);
        if (!s.ok()) { return s; }
    }
    if (num_attention_heads % num_kv_heads != 0u) {
        return Status::error(StatusCode::invalid_argument,
                             "num_attention_heads must be a multiple of num_kv_heads (GQA)", "num_kv_heads");
    }
    if (static_cast<uint64_t>(num_attention_heads) * head_dim != hidden_size) {
        return Status::error(StatusCode::invalid_argument,
                             "num_attention_heads * head_dim must equal hidden_size", "head_dim");
    }
    if (head_dim % 2u != 0u) {
        return Status::error(StatusCode::invalid_argument,
                             "head_dim must be even (RoPE pairs)", "head_dim");
    }
    if (!std::isfinite(rms_norm_eps) || rms_norm_eps <= 0.0f) {
        return Status::error(StatusCode::invalid_argument,
                             "rms_norm_eps must be finite and positive", "rms_norm_eps");
    }
    if (!std::isfinite(rope_theta) || rope_theta <= 0.0f) {
        return Status::error(StatusCode::invalid_argument,
                             "rope_theta must be finite and positive", "rope_theta");
    }
    for (auto id : eos_token_ids) {
        if (id >= vocab_size) {
            return Status::error(StatusCode::invalid_argument,
                                 "eos_token_id outside the vocabulary", "eos_token_id");
        }
    }
    if (has_bos && bos_token_id >= vocab_size) {
        return Status::error(StatusCode::invalid_argument,
                             "bos_token_id outside the vocabulary", "bos_token_id");
    }
    return Status::success();
}

[[nodiscard]] Status DenseDecoderTensorNames::validate() const {
    if (prefix != "model." && prefix != "language_model.") {
        return Status::error(StatusCode::invalid_argument,
                             "decoder tensor prefix must be 'model.' or 'language_model.'", "prefix");
    }
    return Status::success();
}

[[nodiscard]] luisa::string DenseDecoderTensorNames::embed_tokens() const { return prefix + "embed_tokens.weight"; }
[[nodiscard]] luisa::string DenseDecoderTensorNames::final_norm() const { return prefix + "norm.weight"; }
[[nodiscard]] luisa::string DenseDecoderTensorNames::lm_head() const { return "lm_head.weight"; }
[[nodiscard]] luisa::string DenseDecoderTensorNames::layer_prefix(uint32_t layer) const {
    return prefix + "layers." + luisa::format("{}", layer) + ".";
}
[[nodiscard]] luisa::string DenseDecoderTensorNames::input_layernorm(uint32_t layer) const {
    return layer_prefix(layer) + "input_layernorm.weight";
}
[[nodiscard]] luisa::string DenseDecoderTensorNames::post_attention_layernorm(uint32_t layer) const {
    return layer_prefix(layer) + "post_attention_layernorm.weight";
}
[[nodiscard]] luisa::string DenseDecoderTensorNames::q_proj(uint32_t layer) const {
    return layer_prefix(layer) + "self_attn.q_proj.weight";
}
[[nodiscard]] luisa::string DenseDecoderTensorNames::k_proj(uint32_t layer) const {
    return layer_prefix(layer) + "self_attn.k_proj.weight";
}
[[nodiscard]] luisa::string DenseDecoderTensorNames::v_proj(uint32_t layer) const {
    return layer_prefix(layer) + "self_attn.v_proj.weight";
}
[[nodiscard]] luisa::string DenseDecoderTensorNames::q_bias(uint32_t layer) const {
    return layer_prefix(layer) + "self_attn.q_proj.bias";
}
[[nodiscard]] luisa::string DenseDecoderTensorNames::k_bias(uint32_t layer) const {
    return layer_prefix(layer) + "self_attn.k_proj.bias";
}
[[nodiscard]] luisa::string DenseDecoderTensorNames::v_bias(uint32_t layer) const {
    return layer_prefix(layer) + "self_attn.v_proj.bias";
}
[[nodiscard]] luisa::string DenseDecoderTensorNames::o_proj(uint32_t layer) const {
    return layer_prefix(layer) + "self_attn.o_proj.weight";
}
[[nodiscard]] luisa::string DenseDecoderTensorNames::gate_proj(uint32_t layer) const {
    return layer_prefix(layer) + "mlp.gate_proj.weight";
}
[[nodiscard]] luisa::string DenseDecoderTensorNames::up_proj(uint32_t layer) const {
    return layer_prefix(layer) + "mlp.up_proj.weight";
}
[[nodiscard]] luisa::string DenseDecoderTensorNames::down_proj(uint32_t layer) const {
    return layer_prefix(layer) + "mlp.down_proj.weight";
}
[[nodiscard]] luisa::string DenseDecoderTensorNames::q_norm(uint32_t layer) const {
    return layer_prefix(layer) + "self_attn.q_norm.weight";
}
[[nodiscard]] luisa::string DenseDecoderTensorNames::k_norm(uint32_t layer) const {
    return layer_prefix(layer) + "self_attn.k_norm.weight";
}

[[nodiscard]] luisa::vector<ExpectedTensor> dense_decoder_expected_tensors(
    const DenseDecoderConfig &config, const DenseDecoderTensorNames &names,
    bool tie_word_embeddings) {
    luisa::vector<ExpectedTensor> out;
    const auto hidden = static_cast<uint64_t>(config.hidden_size);
    const auto inter = static_cast<uint64_t>(config.intermediate_size);
    const auto q_dim = static_cast<uint64_t>(config.q_dim());
    const auto kv_dim = static_cast<uint64_t>(config.kv_dim());
    const auto add = [&out](luisa::string name, luisa::vector<uint64_t> shape,
                            TensorLayout layout, bool required = true) {
        out.emplace_back(ExpectedTensor{std::move(name), std::move(shape), layout, required});
    };
    add(names.embed_tokens(), {config.vocab_size, hidden}, TensorLayout::row_major);
    add(names.final_norm(), {hidden}, TensorLayout::row_major);
    // Standalone LM head is the packed linear B operand [hidden, vocab]
    // (K=hidden, N=vocab), layout kn. Tied: the embedding table is reused; the
    // standalone tensor is optional but must match this shape when present.
    add(names.lm_head(), {hidden, config.vocab_size}, TensorLayout::kn, !tie_word_embeddings);
    for (uint32_t l = 0; l < config.num_layers; l++) {
        add(names.input_layernorm(l), {hidden}, TensorLayout::row_major);
        add(names.post_attention_layernorm(l), {hidden}, TensorLayout::row_major);
        // Linear weights are PACKED [K, N] (packer pre-transposes HF [N, K]).
        add(names.q_proj(l), {hidden, q_dim}, TensorLayout::kn);
        add(names.k_proj(l), {hidden, kv_dim}, TensorLayout::kn);
        add(names.v_proj(l), {hidden, kv_dim}, TensorLayout::kn);
        add(names.q_bias(l), {q_dim}, TensorLayout::row_major);
        add(names.k_bias(l), {kv_dim}, TensorLayout::row_major);
        add(names.v_bias(l), {kv_dim}, TensorLayout::row_major);
        add(names.o_proj(l), {q_dim, hidden}, TensorLayout::kn);
        add(names.gate_proj(l), {hidden, inter}, TensorLayout::kn);
        add(names.up_proj(l), {hidden, inter}, TensorLayout::kn);
        add(names.down_proj(l), {inter, hidden}, TensorLayout::kn);
    }
    return out;
}

[[nodiscard]] Status validate_dense_decoder_tensors(
    luisa::span<const PackageTensor> tensors,
    luisa::span<const ExpectedTensor> expected) {
    for (const auto &exp : expected) {
        const PackageTensor *found = nullptr;
        for (const auto &t : tensors) {
            if (t.name == exp.name) {
                found = &t;
                break;
            }
        }
        if (found == nullptr) {
            if (exp.required) {
                return Status::error(StatusCode::not_found,
                                     luisa::format("required decoder tensor '{}' is missing", exp.name), exp.name);
            }
            continue;
        }
        if (found->shape != exp.shape) {
            return Status::error(StatusCode::invalid_argument,
                                 luisa::format("decoder tensor '{}' has the wrong shape for the config", exp.name),
                                 exp.name);
        }
        if (found->layout != exp.layout) {
            return Status::error(StatusCode::invalid_argument,
                                 luisa::format("decoder tensor '{}' must use layout '{}' (packed [K,N] for linears)",
                                               exp.name, tensor_layout_name(exp.layout)),
                                 exp.name);
        }
        if (found->encoding != TensorEncoding::fp32) {
            return Status::error(StatusCode::unsupported,
                                 luisa::format("decoder tensor '{}' uses encoding '{}'; the Phase 3 dense baseline "
                                               "requires fp32 (packed/quantized decoder weights are a later matrix)",
                                               exp.name, tensor_encoding_name(found->encoding)),
                                 exp.name);
        }
    }
    return Status::success();
}

[[nodiscard]] Status validate_decoder_weight_desc(const TensorDesc &logical,
                                                  uint64_t element_count,
                                                  const ExpectedTensor &expected) {
    // Exact shape match (rank + every dimension).
    if (logical.shape != expected.shape) {
        return Status::error(StatusCode::invalid_argument,
                             luisa::format("decoder weight '{}' has shape rank/dims that do not match the config "
                                           "(expected {} dims)", expected.name, expected.shape.size()),
                             expected.name);
    }
    // Exact layout tag match (kn vs row_major vs nk ...).
    const auto expected_tag = static_cast<TensorLayoutTag>(static_cast<uint32_t>(expected.layout));
    if (logical.layout != expected_tag) {
        return Status::error(StatusCode::invalid_argument,
                             luisa::format("decoder weight '{}' has layout '{}'; expected '{}' "
                                           "(packed [K,N] for linears)", expected.name,
                                           tensor_layout_tag_name(logical.layout),
                                           tensor_layout_tag_name(expected_tag)),
                             expected.name);
    }
    // Element count must equal the product of the expected shape AND the
    // descriptor's own checked element count (catches undersized buffers).
    uint64_t want = 1u;
    for (auto dim : expected.shape) {
        if (!checked_mul(want, dim, want)) {
            return Status::error(StatusCode::invalid_argument,
                                 luisa::format("decoder weight '{}' expected element count overflows", expected.name),
                                 expected.name);
        }
    }
    auto desc_count = logical.element_count();
    if (!desc_count.ok()) { return desc_count.status; }
    if (element_count != want || *desc_count.value != want) {
        return Status::error(StatusCode::invalid_argument,
                             luisa::format("decoder weight '{}' element count {} does not match the expected {}",
                                           expected.name, element_count, want),
                             expected.name);
    }
    return Status::success();
}

namespace {

struct JsonDoc {
    yyjson_doc *doc{nullptr};
    ~JsonDoc() noexcept {
        if (doc != nullptr) { yyjson_doc_free(doc); }
    }
};

[[nodiscard]] Result<uint32_t> json_get_u32(yyjson_val *obj, const char *key, bool required, uint32_t fallback) {
    using R = Result<uint32_t>;
    auto *v = yyjson_obj_get(obj, key);
    if (v == nullptr || yyjson_is_null(v)) {
        if (required) {
            return R::failure(StatusCode::invalid_argument,
                              luisa::format("decoder config is missing required field '{}'", key), key);
        }
        return R::success(fallback);
    }
    if (!yyjson_is_uint(v)) {
        return R::failure(StatusCode::invalid_argument,
                          luisa::format("decoder config field '{}' must be a nonnegative integer", key), key);
    }
    const auto value = yyjson_get_uint(v);
    if (value > UINT32_MAX) {
        return R::failure(StatusCode::invalid_argument,
                          luisa::format("decoder config field '{}' exceeds uint32", key), key);
    }
    return R::success(static_cast<uint32_t>(value));
}

[[nodiscard]] Result<float> json_get_f32(yyjson_val *obj, const char *key, float fallback) {
    using R = Result<float>;
    auto *v = yyjson_obj_get(obj, key);
    if (v == nullptr || yyjson_is_null(v)) { return R::success(fallback); }
    if (!yyjson_is_real(v) && !yyjson_is_uint(v) && !yyjson_is_sint(v)) {
        return R::failure(StatusCode::invalid_argument,
                          luisa::format("decoder config field '{}' must be a number", key), key);
    }
    // yyjson_get_num handles integer-valued numbers (e.g. rope_theta: 1000000)
    // as well as reals; yyjson_get_real would return 0 for an integer node.
    const auto value = static_cast<float>(yyjson_get_num(v));
    if (!std::isfinite(value)) {
        return R::failure(StatusCode::invalid_argument,
                          luisa::format("decoder config field '{}' must be finite", key), key);
    }
    return R::success(value);
}

}// namespace

[[nodiscard]] Result<DenseDecoderConfig> parse_qwen2_5_text_config(luisa::string_view json) {
    using R = Result<DenseDecoderConfig>;
    JsonDoc doc;
    doc.doc = yyjson_read(json.data(), json.size(), 0);
    if (doc.doc == nullptr) {
        return R::failure(StatusCode::invalid_argument, "decoder config is not valid JSON", "config");
    }
    auto *root = yyjson_doc_get_root(doc.doc);
    if (root == nullptr || !yyjson_is_obj(root)) {
        return R::failure(StatusCode::invalid_argument, "decoder config must be a JSON object", "config");
    }
    // Accept either a flat decoder config or a wrapper with text_config.
    auto *cfg = root;
    if (auto *text = yyjson_obj_get(root, "text_config"); text != nullptr && !yyjson_is_null(text)) {
        if (!yyjson_is_obj(text)) {
            return R::failure(StatusCode::invalid_argument, "text_config must be an object", "text_config");
        }
        cfg = text;
    }
    // Identity boundary: refuse Qwen3 markers instead of silently accepting a
    // different decoder contract (QK-Norm / no-QKV-bias / interleaved MRoPE).
    if (auto *mt = yyjson_obj_get(cfg, "model_type"); mt != nullptr && yyjson_is_str(mt)) {
        const luisa::string_view type{yyjson_get_str(mt), yyjson_get_len(mt)};
        if (type.find("qwen3") != luisa::string_view::npos) {
            return R::failure(StatusCode::unsupported,
                              "Qwen3-VL text configs are NOT Qwen2.5-VL: use the qwen3_vl contract (Phase 4)",
                              "model_type");
        }
    }
    for (const char *marker : {"qk_norm", "q_norm", "k_norm", "num_experts", "linear_num_key_heads",
                               "linear_key_head_dim"}) {
        if (auto *v = yyjson_obj_get(cfg, marker); v != nullptr && !yyjson_is_null(v) && !yyjson_is_false(v)) {
            return R::failure(StatusCode::unsupported,
                              luisa::format("decoder config field '{}' is outside the shared dense Qwen2.5 contract",
                                            marker),
                              marker);
        }
    }
    // Sliding window: a merely PRESENT but DISABLED window is the ordinary
    // dense path (exact), so it is accepted. An ENABLED window changes the
    // attention mask and is unsupported here. "use_sliding_window": false (or
    // absent) disables it; a positive "sliding_window" with use_sliding_window
    // true (or absent, HF defaults to enabled when a window is set) is refused.
    {
        auto *use = yyjson_obj_get(cfg, "use_sliding_window");
        const bool use_present = use != nullptr && !yyjson_is_null(use);
        const bool use_true = use_present && yyjson_is_true(use);
        const bool use_false = use_present && yyjson_is_false(use);
        if (use_present && !use_true && !use_false) {
            return R::failure(StatusCode::invalid_argument,
                              "use_sliding_window must be a boolean", "use_sliding_window");
        }
        auto *win = yyjson_obj_get(cfg, "sliding_window");
        const bool win_present = win != nullptr && !yyjson_is_null(win);
        if (win_present) {
            if (!yyjson_is_uint(win)) {
                return R::failure(StatusCode::invalid_argument,
                                  "sliding_window must be a positive integer or null", "sliding_window");
            }
            const bool window_enabled = use_true || !use_present;// HF: window set => enabled unless explicitly false
            if (window_enabled && yyjson_get_uint(win) > 0u) {
                return R::failure(StatusCode::unsupported,
                                  "an enabled sliding_window is outside the shared dense Qwen2.5 contract",
                                  "sliding_window");
            }
        } else if (use_true) {
            return R::failure(StatusCode::unsupported,
                              "use_sliding_window=true without a window is outside the shared dense Qwen2.5 contract",
                              "use_sliding_window");
        }
    }
    DenseDecoderConfig config;
    // Qwen2.5-VL HF text decoders use rotate-half RoPE pairs (d, d+head_dim/2).
    config.rope_pair_layout = RopePairLayout::rotate_half;
    auto u32 = [&cfg](const char *key, bool required, uint32_t fallback, uint32_t &out) -> Status {
        auto r = json_get_u32(cfg, key, required, fallback);
        if (!r.ok()) { return r.status; }
        out = *r.value;
        return Status::success();
    };
    if (auto s = u32("vocab_size", true, 0, config.vocab_size); !s.ok()) { return R::failure(s); }
    if (auto s = u32("hidden_size", true, 0, config.hidden_size); !s.ok()) { return R::failure(s); }
    if (auto s = u32("num_hidden_layers", true, 0, config.num_layers); !s.ok()) { return R::failure(s); }
    if (auto s = u32("num_attention_heads", true, 0, config.num_attention_heads); !s.ok()) { return R::failure(s); }
    if (auto s = u32("num_key_value_heads", true, 0, config.num_kv_heads); !s.ok()) { return R::failure(s); }
    if (auto s = u32("intermediate_size", true, 0, config.intermediate_size); !s.ok()) { return R::failure(s); }
    if (auto s = u32("max_position_embeddings", true, 0, config.max_position_embeddings); !s.ok()) { return R::failure(s); }
    const uint32_t default_head_dim = config.num_attention_heads == 0u
                                          ? 0u
                                          : config.hidden_size / config.num_attention_heads;
    if (auto s = u32("head_dim", false, default_head_dim, config.head_dim); !s.ok()) { return R::failure(s); }
    auto eps = json_get_f32(cfg, "rms_norm_eps", 1e-6f);
    if (!eps.ok()) { return R::failure(eps.status); }
    config.rms_norm_eps = *eps.value;
    auto theta = json_get_f32(cfg, "rope_theta", 10000.0f);
    if (!theta.ok()) { return R::failure(theta.status); }
    config.rope_theta = *theta.value;
    // EOS: scalar or list. Absent = none declared.
    if (auto *eos = yyjson_obj_get(cfg, "eos_token_id"); eos != nullptr && !yyjson_is_null(eos)) {
        if (yyjson_is_uint(eos)) {
            if (yyjson_get_uint(eos) > UINT32_MAX) {
                return R::failure(StatusCode::invalid_argument, "eos_token_id exceeds uint32", "eos_token_id");
            }
            config.eos_token_ids.push_back(static_cast<uint32_t>(yyjson_get_uint(eos)));
        } else if (yyjson_is_arr(eos)) {
            size_t idx = 0, max = 0;
            yyjson_val *item = nullptr;
            yyjson_arr_foreach(eos, idx, max, item) {
                if (!yyjson_is_uint(item) || yyjson_get_uint(item) > UINT32_MAX) {
                    return R::failure(StatusCode::invalid_argument,
                                      "eos_token_id entries must be nonnegative integers", "eos_token_id");
                }
                config.eos_token_ids.push_back(static_cast<uint32_t>(yyjson_get_uint(item)));
            }
        } else {
            return R::failure(StatusCode::invalid_argument,
                              "eos_token_id must be an integer or a list of integers", "eos_token_id");
        }
    }
    if (auto *bos = yyjson_obj_get(cfg, "bos_token_id"); bos != nullptr && !yyjson_is_null(bos)) {
        if (!yyjson_is_uint(bos) || yyjson_get_uint(bos) > UINT32_MAX) {
            return R::failure(StatusCode::invalid_argument,
                              "bos_token_id must be a nonnegative integer", "bos_token_id");
        }
        config.bos_token_id = static_cast<uint32_t>(yyjson_get_uint(bos));
        config.has_bos = true;
    }
    auto valid = config.validate();
    if (!valid.ok()) { return R::failure(valid); }
    return R::success(std::move(config));
}

}// namespace luisa::ocr
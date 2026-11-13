// OCR engine -- DeepSeek-OCR-2 MoE graph contract (Phase 10). Host-only TU.
// Synthetic recipe-driven contract; real artifacts fail closed. The executing
// graph is runtime/moe_vlm_executor.*. See MODELS.md for the unmet official gates.
#include "ocr/models/deepseek_ocr2.h"

#include <cmath>
#include <utility>

#include <luisa/core/stl/format.h>
#include <yyjson.h>

#include "ocr/quantization.h"
#include "ocr/runtime/layout_result.h"

namespace luisa::ocr {

namespace {

struct JsonDoc {
    yyjson_doc *doc{nullptr};
    ~JsonDoc() {
        if (doc != nullptr) { yyjson_doc_free(doc); }
    }
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

// Checked fp32 byte sum of an expected-tensor inventory.
[[nodiscard]] Result<uint64_t> inventory_bytes(luisa::span<const ExpectedTensor> expected) {
    using R = Result<uint64_t>;
    uint64_t total = 0;
    for (const auto &e : expected) {
        uint64_t elems = 1;
        for (auto dim : e.shape) {
            if (!checked_mul(elems, dim, elems)) { return R::failure(StatusCode::out_of_memory, "weight element count overflows", "weights"); }
        }
        uint64_t bytes = 0;
        if (!checked_mul(elems, 4ull, bytes) || !checked_add(total, bytes, total)) {
            return R::failure(StatusCode::out_of_memory, "weight byte count overflows", "weights");
        }
    }
    return R::success(total);
}

}// namespace

// ---------------------------------------------------------------------------
// Config validation
// ---------------------------------------------------------------------------

[[nodiscard]] Status MoeDecoderConfig::validate() const {
    auto s = decoder.validate();
    if (!s.ok()) { return s; }
    if (num_experts == 0u || top_k == 0u || top_k > num_experts) {
        return Status::error(StatusCode::invalid_argument, "MoE expert geometry invalid (0 < top_k <= num_experts)", "experts");
    }
    if (num_shared_experts == 0u) {
        return Status::error(StatusCode::invalid_argument, "MoE num_shared_experts must be nonzero", "num_shared_experts");
    }
    if (moe_intermediate_size == 0u || shared_intermediate_size == 0u) {
        return Status::error(StatusCode::invalid_argument, "MoE intermediate sizes must be nonzero", "intermediate");
    }
    if (component.empty()) {
        return Status::error(StatusCode::invalid_argument, "weight component tag must not be empty", "component");
    }
    if (prefill_chunk_rows == 0u) {
        return Status::error(StatusCode::invalid_argument, "prefill_chunk_rows must be nonzero", "prefill_chunk_rows");
    }
    // S1: the MoE decoder ships a STANDALONE lm_head. A tied head would need an
    // executor-owned transpose buffer that this executor never creates, so a
    // tied configuration is refused instead of silently binding a missing head.
    if (tie_word_embeddings) {
        return Status::error(StatusCode::unsupported,
                             "the MoE decoder requires a standalone lm_head; tied word embeddings are not implemented",
                             "tie_word_embeddings");
    }
    return Status::success();
}

[[nodiscard]] Status MoeDecoderTensorNames::validate() const { return dense.validate(); }
[[nodiscard]] luisa::string MoeDecoderTensorNames::layer_prefix(uint32_t layer) const { return dense.layer_prefix(layer); }
[[nodiscard]] luisa::string MoeDecoderTensorNames::router_weight(uint32_t layer) const {
    return layer_prefix(layer) + "mlp.gate.weight";
}
[[nodiscard]] luisa::string MoeDecoderTensorNames::expert_gate(uint32_t layer, uint32_t e) const {
    return layer_prefix(layer) + "mlp.experts." + luisa::format("{}", e) + ".gate_proj.weight";
}
[[nodiscard]] luisa::string MoeDecoderTensorNames::expert_up(uint32_t layer, uint32_t e) const {
    return layer_prefix(layer) + "mlp.experts." + luisa::format("{}", e) + ".up_proj.weight";
}
[[nodiscard]] luisa::string MoeDecoderTensorNames::expert_down(uint32_t layer, uint32_t e) const {
    return layer_prefix(layer) + "mlp.experts." + luisa::format("{}", e) + ".down_proj.weight";
}
[[nodiscard]] luisa::string MoeDecoderTensorNames::shared_gate(uint32_t layer, uint32_t s) const {
    return layer_prefix(layer) + "mlp.shared_experts." + luisa::format("{}", s) + ".gate_proj.weight";
}
[[nodiscard]] luisa::string MoeDecoderTensorNames::shared_up(uint32_t layer, uint32_t s) const {
    return layer_prefix(layer) + "mlp.shared_experts." + luisa::format("{}", s) + ".up_proj.weight";
}
[[nodiscard]] luisa::string MoeDecoderTensorNames::shared_down(uint32_t layer, uint32_t s) const {
    return layer_prefix(layer) + "mlp.shared_experts." + luisa::format("{}", s) + ".down_proj.weight";
}

[[nodiscard]] Status DeepSeekConvDownsampler::validate() const {
    if (out_channels == 0u || kernel == 0u || stride == 0u) {
        return Status::error(StatusCode::invalid_argument, "conv downsampler fields must be nonzero", "downsampler");
    }
    // M4: a DOWNSAMPLER must never enlarge the spatial grid, otherwise the
    // executor's `enc_rows_max = max_vision_patches + num_queries` workspace
    // bound (and the preflight built from it) would be too small. Mirror the
    // ops_conv out-dim formula (dilation 1) with checked host math:
    //   eff = kernel; out(in) = (in + 2*pad - eff) / stride + 1.
    // * stride == 1: out - in is the CONSTANT 2*pad - eff + 1, so require <= 0.
    // * stride >= 2: out - in is non-increasing in `in`, so its maximum is at
    //   the smallest legal input in_min = max(1, eff - 2*pad); check there.
    const uint64_t eff = kernel;
    const uint64_t pad2 = 2ull * pad;
    if (stride == 1u) {
        if (pad2 + 1ull > eff) {
            return Status::error(StatusCode::invalid_argument,
                                 "a stride-1 conv downsampler would enlarge the grid (2*pad + 1 must be <= kernel)",
                                 "downsampler");
        }
    } else {
        const uint64_t in_min = eff > pad2 ? eff - pad2 : 1ull;
        const uint64_t out_min = (in_min + pad2 - eff) / stride + 1ull;
        if (out_min > in_min) {
            return Status::error(StatusCode::invalid_argument,
                                 "conv downsampler would enlarge the grid at the smallest legal input",
                                 "downsampler");
        }
    }
    return Status::success();
}

[[nodiscard]] Status DeepSeekVisionConfig::validate() const {
    auto s = perception.validate();
    if (!s.ok()) { return s; }
    // SAM local perception MUST be windowed (never the custom visual/query mask,
    // which belongs to the visual LM encoder).
    if (perception.mask_kind != P7VisionMaskKind::window_global) {
        return Status::error(StatusCode::invalid_argument,
                             "SAM local perception must use the windowed (window_global) mask", "perception.mask_kind");
    }
    s = down1.validate();
    if (!s.ok()) { return s; }
    s = down2.validate();
    if (!s.ok()) { return s; }
    // The first downsampler consumes the perception channels.
    if (down1.out_channels == 0u) {
        return Status::error(StatusCode::invalid_argument, "down1 out_channels must be nonzero", "down1");
    }
    if (encoder_layers == 0u || encoder_intermediate_size == 0u || num_queries == 0u) {
        return Status::error(StatusCode::invalid_argument, "visual LM encoder fields must be nonzero", "encoder");
    }
    if (!std::isfinite(encoder_norm_eps) || encoder_norm_eps <= 0.0f) {
        return Status::error(StatusCode::invalid_argument, "encoder_norm_eps must be > 0", "encoder_norm_eps");
    }
    if (connector_intermediate_size == 0u) {
        return Status::error(StatusCode::invalid_argument, "connector intermediate_size must be nonzero", "connector");
    }
    if (!std::isfinite(connector_norm_eps) || connector_norm_eps <= 0.0f) {
        return Status::error(StatusCode::invalid_argument, "connector_norm_eps must be > 0", "connector_norm_eps");
    }
    if (mean.size() != 3u || std_dev.size() != 3u) {
        return Status::error(StatusCode::invalid_argument, "mean/std must have 3 channels", "mean/std");
    }
    // The encoder hidden (= down2.out_channels) must split into whole heads.
    const uint32_t eh = encoder_hidden();
    if (perception.num_heads == 0u || eh % perception.num_heads != 0u) {
        return Status::error(StatusCode::invalid_argument,
                             "encoder hidden must be divisible by the perception head count", "encoder_hidden");
    }
    const uint32_t ehd = eh / perception.num_heads;
    if (ehd % 2u != 0u) {
        return Status::error(StatusCode::invalid_argument, "encoder head_dim must be even (RoPE pairs)", "encoder_head_dim");
    }
    return Status::success();
}

[[nodiscard]] Status DeepSeekVisionTensorNames::validate() const { return perception.validate(); }
[[nodiscard]] luisa::string DeepSeekVisionTensorNames::down_weight(uint32_t which) const {
    return perception.visual_prefix + "downsamplers." + luisa::format("{}", which - 1u) + ".weight";
}
[[nodiscard]] luisa::string DeepSeekVisionTensorNames::down_bias(uint32_t which) const {
    return perception.visual_prefix + "downsamplers." + luisa::format("{}", which - 1u) + ".bias";
}
[[nodiscard]] luisa::string DeepSeekVisionTensorNames::encoder_prefix(uint32_t layer) const {
    return perception.visual_prefix + "encoder.layers." + luisa::format("{}", layer) + ".";
}
[[nodiscard]] luisa::string DeepSeekVisionTensorNames::encoder_norm1_weight(uint32_t layer) const { return encoder_prefix(layer) + "norm1.weight"; }
[[nodiscard]] luisa::string DeepSeekVisionTensorNames::encoder_norm1_bias(uint32_t layer) const { return encoder_prefix(layer) + "norm1.bias"; }
[[nodiscard]] luisa::string DeepSeekVisionTensorNames::encoder_norm2_weight(uint32_t layer) const { return encoder_prefix(layer) + "norm2.weight"; }
[[nodiscard]] luisa::string DeepSeekVisionTensorNames::encoder_norm2_bias(uint32_t layer) const { return encoder_prefix(layer) + "norm2.bias"; }
[[nodiscard]] luisa::string DeepSeekVisionTensorNames::encoder_q_proj(uint32_t layer) const { return encoder_prefix(layer) + "attn.q_proj.weight"; }
[[nodiscard]] luisa::string DeepSeekVisionTensorNames::encoder_k_proj(uint32_t layer) const { return encoder_prefix(layer) + "attn.k_proj.weight"; }
[[nodiscard]] luisa::string DeepSeekVisionTensorNames::encoder_v_proj(uint32_t layer) const { return encoder_prefix(layer) + "attn.v_proj.weight"; }
[[nodiscard]] luisa::string DeepSeekVisionTensorNames::encoder_o_proj(uint32_t layer) const { return encoder_prefix(layer) + "attn.o_proj.weight"; }
[[nodiscard]] luisa::string DeepSeekVisionTensorNames::encoder_q_bias(uint32_t layer) const { return encoder_prefix(layer) + "attn.q_proj.bias"; }
[[nodiscard]] luisa::string DeepSeekVisionTensorNames::encoder_k_bias(uint32_t layer) const { return encoder_prefix(layer) + "attn.k_proj.bias"; }
[[nodiscard]] luisa::string DeepSeekVisionTensorNames::encoder_v_bias(uint32_t layer) const { return encoder_prefix(layer) + "attn.v_proj.bias"; }
[[nodiscard]] luisa::string DeepSeekVisionTensorNames::encoder_o_bias(uint32_t layer) const { return encoder_prefix(layer) + "attn.o_proj.bias"; }
[[nodiscard]] luisa::string DeepSeekVisionTensorNames::encoder_fc1(uint32_t layer) const { return encoder_prefix(layer) + "mlp.fc1.weight"; }
[[nodiscard]] luisa::string DeepSeekVisionTensorNames::encoder_fc1_bias(uint32_t layer) const { return encoder_prefix(layer) + "mlp.fc1.bias"; }
[[nodiscard]] luisa::string DeepSeekVisionTensorNames::encoder_fc2(uint32_t layer) const { return encoder_prefix(layer) + "mlp.fc2.weight"; }
[[nodiscard]] luisa::string DeepSeekVisionTensorNames::encoder_fc2_bias(uint32_t layer) const { return encoder_prefix(layer) + "mlp.fc2.bias"; }
[[nodiscard]] luisa::string DeepSeekVisionTensorNames::query_embed() const { return perception.visual_prefix + "query_embed.weight"; }
[[nodiscard]] luisa::string DeepSeekVisionTensorNames::connector_ln_weight() const { return perception.visual_prefix + "connector.ln.weight"; }
[[nodiscard]] luisa::string DeepSeekVisionTensorNames::connector_ln_bias() const { return perception.visual_prefix + "connector.ln.bias"; }
[[nodiscard]] luisa::string DeepSeekVisionTensorNames::connector_fc1() const { return perception.visual_prefix + "connector.fc1.weight"; }
[[nodiscard]] luisa::string DeepSeekVisionTensorNames::connector_fc1_bias() const { return perception.visual_prefix + "connector.fc1.bias"; }
[[nodiscard]] luisa::string DeepSeekVisionTensorNames::connector_fc2() const { return perception.visual_prefix + "connector.fc2.weight"; }
[[nodiscard]] luisa::string DeepSeekVisionTensorNames::connector_fc2_bias() const { return perception.visual_prefix + "connector.fc2.bias"; }

[[nodiscard]] Status DeepSeekOcr2Config::validate() const {
    auto s = vision.validate();
    if (!s.ok()) { return s; }
    s = decoder.validate();
    if (!s.ok()) { return s; }
    s = vision_names.validate();
    if (!s.ok()) { return s; }
    s = decoder_names.validate();
    if (!s.ok()) { return s; }
    s = packing.validate();
    if (!s.ok()) { return s; }
    // The connector emits text-hidden features -> decoder hidden.
    if (max_seq_len == 0u) {
        return Status::error(StatusCode::invalid_argument, "max_seq_len must be nonzero", "max_seq_len");
    }
    // The exact expert contract is enforced for BOTH the full and synthetic config.
    return deepseek_ocr2_validate_expert_contract(decoder);
}

// ---------------------------------------------------------------------------
// Declared full contract + expert-contract validation
// ---------------------------------------------------------------------------

[[nodiscard]] MoeDecoderConfig deepseek_ocr2_declared_decoder_config() {
    // DECLARED full DeepSeek MoE geometry (NOT a verified checkpoint). The
    // 64 routed / top-6 / two-shared contract is EXACT; hidden/head/intermediate
    // are a consistent declared geometry used to validate the contract.
    MoeDecoderConfig cfg;
    auto &d = cfg.decoder;
    d.vocab_size = 129280;
    d.hidden_size = 2048;
    d.num_layers = 12;
    d.num_attention_heads = 16;
    d.num_kv_heads = 16;
    d.head_dim = 128;
    d.intermediate_size = 2048;// dense-attention MLP slot (unused by MoE layers)
    d.rms_norm_eps = 1e-6f;
    d.rope_theta = 10000.0f;
    d.rope_pair_layout = RopePairLayout::rotate_half;
    d.max_position_embeddings = 8192;
    d.eos_token_ids = {100001u};
    cfg.num_experts = kDeepSeekNumRoutedExperts;
    cfg.top_k = kDeepSeekTopK;
    cfg.num_shared_experts = kDeepSeekNumSharedExperts;
    cfg.moe_intermediate_size = 1408;
    cfg.shared_intermediate_size = 1408;
    cfg.tie_word_embeddings = false;
    cfg.component = "decoder";
    return cfg;
}

[[nodiscard]] Status deepseek_ocr2_validate_expert_contract(const MoeDecoderConfig &config) {
    if (config.num_experts != kDeepSeekNumRoutedExperts) {
        return Status::error(StatusCode::invalid_argument,
                             luisa::format("DeepSeek-OCR-2 requires exactly {} routed experts (got {}); the expert count may not scale",
                                           kDeepSeekNumRoutedExperts, config.num_experts),
                             "num_experts");
    }
    if (config.top_k != kDeepSeekTopK) {
        return Status::error(StatusCode::invalid_argument,
                             luisa::format("DeepSeek-OCR-2 requires exactly top-{} routing (got {}); top-k may not scale",
                                           kDeepSeekTopK, config.top_k),
                             "top_k");
    }
    if (config.num_shared_experts != kDeepSeekNumSharedExperts) {
        return Status::error(StatusCode::invalid_argument,
                             luisa::format("DeepSeek-OCR-2 requires exactly {} shared experts (got {})",
                                           kDeepSeekNumSharedExperts, config.num_shared_experts),
                             "num_shared_experts");
    }
    return Status::success();
}

// ---------------------------------------------------------------------------
// Small synthetic slice
// ---------------------------------------------------------------------------

[[nodiscard]] DeepSeekOcr2Config deepseek_ocr2_synthetic_config() {
    DeepSeekOcr2Config cfg;
    auto &v = cfg.vision;
    // SAM-style local perception: windowed ViT, hidden 16, 2 heads, head_dim 8,
    // patch 2, 2D axial RoPE, GELU-erf MLP. window_global mask (local windows).
    v.perception.num_layers = 2;
    v.perception.hidden_size = 16;
    v.perception.num_heads = 2;
    v.perception.head_dim = 8;
    v.perception.intermediate_size = 32;
    v.perception.patch_size = 2;
    v.perception.channels = 3;
    v.perception.layer_norm_eps = 1e-5f;
    v.perception.qkv_bias = true;
    v.perception.qk_norm = false;
    v.perception.mlp_act = 0;// GELU-erf
    v.perception.rope_kind = P7VisionRopeKind::axial2d;
    v.perception.rope_theta = 10000.0f;
    v.perception.rope_height_pairs = 2;
    v.perception.mask_kind = P7VisionMaskKind::window_global;
    v.perception.window_patch_size = 2;
    v.perception.global_layers = {1u};// layer 1 global, layer 0 windowed
    v.perception.position_rows = 0;
    v.perception.rel_pos_window = 0;
    // BOTH convolutional downsamplers: 16->16 (stride 2), 16->16 (stride 2).
    // The encoder hidden stays 16 (= down2.out_channels), 2 heads, head_dim 8.
    v.down1 = DeepSeekConvDownsampler{16u, 3u, 2u, 1u};
    v.down2 = DeepSeekConvDownsampler{16u, 3u, 2u, 1u};
    v.encoder_layers = 2;
    v.encoder_intermediate_size = 32;
    v.encoder_norm_eps = 1e-5f;
    v.encoder_qkv_bias = true;
    v.num_queries = 4;
    v.connector_intermediate_size = 32;
    v.connector_norm_eps = 1e-5f;
    v.mean = {0.5f, 0.5f, 0.5f};
    v.std_dev = {0.5f, 0.5f, 0.5f};
    v.max_vision_patches = 256;
    // Packing: patch 2, grid multiple 4 (two stride-2 downsamplers => /4).
    cfg.packing.patch_size = 2;
    cfg.packing.min_pixels = 0;
    cfg.packing.max_pixels = 32u * 32u;
    cfg.packing.grid_multiple = 4;
    // MoE decoder: SCALED hidden dims, EXACT 64/6/2 expert contract.
    auto &d = cfg.decoder.decoder;
    d.vocab_size = 24;
    d.hidden_size = 16;
    d.num_layers = 2;
    d.num_attention_heads = 4;
    d.num_kv_heads = 2;
    d.head_dim = 4;
    d.intermediate_size = 32;
    d.rms_norm_eps = 1e-6f;
    d.rope_theta = 10000.0f;
    d.rope_pair_layout = RopePairLayout::rotate_half;
    d.max_position_embeddings = 128;
    d.eos_token_ids = {23u};
    cfg.decoder.num_experts = kDeepSeekNumRoutedExperts;
    cfg.decoder.top_k = kDeepSeekTopK;
    cfg.decoder.num_shared_experts = kDeepSeekNumSharedExperts;
    cfg.decoder.moe_intermediate_size = 16;
    cfg.decoder.shared_intermediate_size = 16;
    cfg.decoder.tie_word_embeddings = false;
    cfg.decoder.component = "decoder";
    cfg.decoder.prefill_chunk_rows = 64;
    cfg.max_seq_len = 128;
    // The connector output must equal the decoder hidden (16). The connector
    // input is the encoder hidden (16); fc2 maps encoder_hidden -> text hidden.
    return cfg;
}

// ---------------------------------------------------------------------------
// Config parsing (synthetic schema)
// ---------------------------------------------------------------------------

[[nodiscard]] Result<DeepSeekOcr2Config> parse_deepseek_ocr2_config(luisa::string_view json) {
    using R = Result<DeepSeekOcr2Config>;
    JsonDoc doc{yyjson_read(json.data(), json.size(), 0)};
    if (doc.doc == nullptr) { return R::failure(StatusCode::invalid_argument, "config is not valid JSON", "json"); }
    auto *root = yyjson_doc_get_root(doc.doc);
    if (!yyjson_is_obj(root)) { return R::failure(StatusCode::invalid_argument, "config root must be an object", "json"); }
    // The outer model_type is NEVER identity; require the explicit marker but do
    // not treat it as sufficient (the signature gate is the real identity).
    auto *mt = yyjson_obj_get(root, "model_type");
    if (mt != nullptr && yyjson_is_str(mt)) {
        const luisa::string_view type{yyjson_get_str(mt), yyjson_get_len(mt)};
        if (type != "deepseek_ocr_2" && type != "deepseek_ocr2") {
            return R::failure(StatusCode::invalid_argument, luisa::format("unexpected model_type '{}'", type), "model_type");
        }
    }
    auto cfg = deepseek_ocr2_synthetic_config();
    auto *dc = yyjson_obj_get(root, "decoder_config");
    if (dc == nullptr || !yyjson_is_obj(dc)) {
        return R::failure(StatusCode::invalid_argument, "decoder_config object is required", "decoder_config");
    }
    auto &d = cfg.decoder.decoder;
    auto parse = [&](const char *key, uint32_t &field) -> Status {
        auto r = get_u32(dc, key, true, 0);
        if (!r.ok()) { return r.status; }
        field = *r.value;
        return Status::success();
    };
    if (auto s = parse("vocab_size", d.vocab_size); !s.ok()) { return R::failure(s); }
    if (auto s = parse("hidden_size", d.hidden_size); !s.ok()) { return R::failure(s); }
    if (auto s = parse("num_hidden_layers", d.num_layers); !s.ok()) { return R::failure(s); }
    if (auto s = parse("num_attention_heads", d.num_attention_heads); !s.ok()) { return R::failure(s); }
    if (auto s = parse("num_key_value_heads", d.num_kv_heads); !s.ok()) { return R::failure(s); }
    if (auto s = parse("intermediate_size", d.intermediate_size); !s.ok()) { return R::failure(s); }
    if (auto s = parse("max_position_embeddings", d.max_position_embeddings); !s.ok()) { return R::failure(s); }
    {
        // M1: validate the divisor BEFORE deriving the default head_dim. A
        // num_attention_heads of 0 (or a non-dividing hidden_size) must fail
        // closed with a Status, never divide by zero.
        if (d.num_attention_heads == 0u) {
            return R::failure(StatusCode::invalid_argument,
                              "decoder_config.num_attention_heads must be nonzero", "num_attention_heads");
        }
        if (d.hidden_size % d.num_attention_heads != 0u) {
            return R::failure(StatusCode::invalid_argument,
                              "decoder_config.hidden_size must be divisible by num_attention_heads to derive head_dim",
                              "head_dim");
        }
        auto r = get_u32(dc, "head_dim", false, d.hidden_size / d.num_attention_heads);
        if (!r.ok()) { return R::failure(r.status); }
        d.head_dim = *r.value;
    }
    {
        auto r = get_u32(dc, "moe_intermediate_size", false, cfg.decoder.moe_intermediate_size);
        if (!r.ok()) { return R::failure(r.status); }
        cfg.decoder.moe_intermediate_size = *r.value;
    }
    {
        auto r = get_u32(dc, "shared_intermediate_size", false, cfg.decoder.shared_intermediate_size);
        if (!r.ok()) { return R::failure(r.status); }
        cfg.decoder.shared_intermediate_size = *r.value;
    }
    // Expert contract fields: if present they MUST match the exact contract.
    {
        auto r = get_u32(dc, "num_experts", false, kDeepSeekNumRoutedExperts);
        if (!r.ok()) { return R::failure(r.status); }
        cfg.decoder.num_experts = *r.value;
    }
    {
        auto r = get_u32(dc, "top_k", false, kDeepSeekTopK);
        if (!r.ok()) { return R::failure(r.status); }
        cfg.decoder.top_k = *r.value;
    }
    {
        auto r = get_u32(dc, "num_shared_experts", false, kDeepSeekNumSharedExperts);
        if (!r.ok()) { return R::failure(r.status); }
        cfg.decoder.num_shared_experts = *r.value;
    }
    // Keep the connector output consistent with the parsed decoder hidden.
    if (auto s = cfg.validate(); !s.ok()) { return R::failure(s); }
    return R::success(std::move(cfg));
}

// ---------------------------------------------------------------------------
// Tensor inventory + signature + real-artifact gates
// ---------------------------------------------------------------------------

[[nodiscard]] luisa::vector<ExpectedTensor> deepseek_ocr2_vision_expected_tensors(const DeepSeekOcr2Config &config) {
    luisa::vector<ExpectedTensor> out;
    const auto &v = config.vision;
    const auto &names = config.vision_names;
    const auto kn = TensorLayout::kn;
    const auto rm = TensorLayout::row_major;
    // Conv downsampler weights are stored row-major with an OIHW shape (the
    // shared WeightStore maps every non-kn/nk package layout to the row_major
    // compute tag, exactly like the Phase 5/9 CNN conv weights). The conv kernel
    // reads them in OIHW order regardless of the semantic layout tag.
    const auto oi = TensorLayout::row_major;
    // Perception tower = the P7 vision inventory (visual.* prefix ONLY; the P7
    // dense text decoder is NOT part of DeepSeek vision).
    P7SliceConfig p7;
    p7.vision = v.perception;
    // A minimal connector/packing so p7_expected_tensors emits the perception
    // blocks; we only keep the visual.* perception tensors (patch embed + blocks).
    p7.connector.kind = P7ConnectorKind::mlp;
    p7.connector.intermediate_size = v.connector_intermediate_size;
    p7.packing = config.packing;
    for (auto &e : p7_expected_tensors(p7, names.perception)) {
        // Keep patch embed + perception blocks; drop the P7 connector (DeepSeek
        // has its own connector below) and any text tensors.
        const auto &n = e.name;
        const auto vp = names.perception.visual_prefix;
        if (n.compare(0, vp.size(), vp) != 0) { continue; }
        if (n.find("connector.") != luisa::string::npos) { continue; }
        out.push_back(std::move(e));
    }
    // BOTH conv downsamplers. down1: perception.hidden -> down1.out_channels;
    // down2: down1.out_channels -> down2.out_channels.
    {
        const uint32_t in1 = v.perception.hidden_size;
        out.emplace_back(ExpectedTensor{names.down_weight(1), {v.down1.out_channels, in1, v.down1.kernel, v.down1.kernel}, oi, true});
        out.emplace_back(ExpectedTensor{names.down_bias(1), {v.down1.out_channels}, rm, true});
        const uint32_t in2 = v.down1.out_channels;
        out.emplace_back(ExpectedTensor{names.down_weight(2), {v.down2.out_channels, in2, v.down2.kernel, v.down2.kernel}, oi, true});
        out.emplace_back(ExpectedTensor{names.down_bias(2), {v.down2.out_channels}, rm, true});
    }
    // Visual LM encoder (Qwen2-style: QKV bias, no QK-Norm).
    const uint32_t eh = v.encoder_hidden();
    const uint32_t ei = v.encoder_intermediate_size;
    for (uint32_t l = 0; l < v.encoder_layers; l++) {
        out.emplace_back(ExpectedTensor{names.encoder_norm1_weight(l), {eh}, rm, true});
        out.emplace_back(ExpectedTensor{names.encoder_norm1_bias(l), {eh}, rm, true});
        out.emplace_back(ExpectedTensor{names.encoder_norm2_weight(l), {eh}, rm, true});
        out.emplace_back(ExpectedTensor{names.encoder_norm2_bias(l), {eh}, rm, true});
        out.emplace_back(ExpectedTensor{names.encoder_q_proj(l), {eh, eh}, kn, true});
        out.emplace_back(ExpectedTensor{names.encoder_k_proj(l), {eh, eh}, kn, true});
        out.emplace_back(ExpectedTensor{names.encoder_v_proj(l), {eh, eh}, kn, true});
        out.emplace_back(ExpectedTensor{names.encoder_o_proj(l), {eh, eh}, kn, true});
        if (v.encoder_qkv_bias) {
            out.emplace_back(ExpectedTensor{names.encoder_q_bias(l), {eh}, rm, true});
            out.emplace_back(ExpectedTensor{names.encoder_k_bias(l), {eh}, rm, true});
            out.emplace_back(ExpectedTensor{names.encoder_v_bias(l), {eh}, rm, true});
            out.emplace_back(ExpectedTensor{names.encoder_o_bias(l), {eh}, rm, true});
        }
        out.emplace_back(ExpectedTensor{names.encoder_fc1(l), {eh, ei}, kn, true});
        out.emplace_back(ExpectedTensor{names.encoder_fc1_bias(l), {ei}, rm, true});
        out.emplace_back(ExpectedTensor{names.encoder_fc2(l), {ei, eh}, kn, true});
        out.emplace_back(ExpectedTensor{names.encoder_fc2_bias(l), {eh}, rm, true});
    }
    // Causal-flow query embeddings [num_queries, encoder_hidden].
    out.emplace_back(ExpectedTensor{names.query_embed(), {v.num_queries, eh}, rm, true});
    // Connector: LayerNorm + 2-layer MLP (encoder_hidden -> inter -> text hidden).
    const uint32_t th = config.decoder.decoder.hidden_size;
    out.emplace_back(ExpectedTensor{names.connector_ln_weight(), {eh}, rm, true});
    out.emplace_back(ExpectedTensor{names.connector_ln_bias(), {eh}, rm, true});
    out.emplace_back(ExpectedTensor{names.connector_fc1(), {eh, v.connector_intermediate_size}, kn, true});
    out.emplace_back(ExpectedTensor{names.connector_fc1_bias(), {v.connector_intermediate_size}, rm, true});
    out.emplace_back(ExpectedTensor{names.connector_fc2(), {v.connector_intermediate_size, th}, kn, true});
    out.emplace_back(ExpectedTensor{names.connector_fc2_bias(), {th}, rm, true});
    return out;
}

[[nodiscard]] luisa::vector<ExpectedTensor> deepseek_ocr2_decoder_expected_tensors(const DeepSeekOcr2Config &config) {
    luisa::vector<ExpectedTensor> out;
    const auto &cfg = config.decoder;
    const auto &d = cfg.decoder;
    const auto &names = config.decoder_names;
    const auto kn = TensorLayout::kn;
    const auto rm = TensorLayout::row_major;
    const uint32_t hidden = d.hidden_size, qd = d.q_dim(), kvd = d.kv_dim();
    const uint32_t mi = cfg.moe_intermediate_size, si = cfg.shared_intermediate_size;
    out.emplace_back(ExpectedTensor{names.dense.embed_tokens(), {d.vocab_size, hidden}, rm, true});
    out.emplace_back(ExpectedTensor{names.dense.final_norm(), {hidden}, rm, true});
    if (!cfg.tie_word_embeddings) {
        out.emplace_back(ExpectedTensor{names.dense.lm_head(), {hidden, d.vocab_size}, kn, true});
    }
    for (uint32_t l = 0; l < d.num_layers; l++) {
        out.emplace_back(ExpectedTensor{names.dense.input_layernorm(l), {hidden}, rm, true});
        out.emplace_back(ExpectedTensor{names.dense.post_attention_layernorm(l), {hidden}, rm, true});
        out.emplace_back(ExpectedTensor{names.dense.q_proj(l), {hidden, qd}, kn, true});
        out.emplace_back(ExpectedTensor{names.dense.k_proj(l), {hidden, kvd}, kn, true});
        out.emplace_back(ExpectedTensor{names.dense.v_proj(l), {hidden, kvd}, kn, true});
        out.emplace_back(ExpectedTensor{names.dense.o_proj(l), {qd, hidden}, kn, true});
        out.emplace_back(ExpectedTensor{names.dense.q_bias(l), {qd}, rm, true});
        out.emplace_back(ExpectedTensor{names.dense.k_bias(l), {kvd}, rm, true});
        out.emplace_back(ExpectedTensor{names.dense.v_bias(l), {kvd}, rm, true});
        // FP32 router [hidden, num_experts].
        out.emplace_back(ExpectedTensor{names.router_weight(l), {hidden, cfg.num_experts}, kn, true});
        // Routed experts.
        for (uint32_t e = 0; e < cfg.num_experts; e++) {
            out.emplace_back(ExpectedTensor{names.expert_gate(l, e), {hidden, mi}, kn, true});
            out.emplace_back(ExpectedTensor{names.expert_up(l, e), {hidden, mi}, kn, true});
            out.emplace_back(ExpectedTensor{names.expert_down(l, e), {mi, hidden}, kn, true});
        }
        // Two shared dense experts.
        for (uint32_t s = 0; s < cfg.num_shared_experts; s++) {
            out.emplace_back(ExpectedTensor{names.shared_gate(l, s), {hidden, si}, kn, true});
            out.emplace_back(ExpectedTensor{names.shared_up(l, s), {hidden, si}, kn, true});
            out.emplace_back(ExpectedTensor{names.shared_down(l, s), {si, hidden}, kn, true});
        }
    }
    return out;
}

[[nodiscard]] Status deepseek_ocr2_validate_tensors(luisa::span<const PackageTensor> vision_tensors,
                                                    luisa::span<const PackageTensor> decoder_tensors,
                                                    const DeepSeekOcr2Config &config) {
    auto vision_expected = deepseek_ocr2_vision_expected_tensors(config);
    if (auto s = p7_validate_tensors(vision_tensors, vision_expected); !s.ok()) { return s; }
    auto decoder_expected = deepseek_ocr2_decoder_expected_tensors(config);
    return p7_validate_tensors(decoder_tensors, decoder_expected);
}

[[nodiscard]] Status deepseek_ocr2_validate_signature(const DeepSeekOcr2Config &config,
                                                      const DeepSeekOcr2Signature &signature) {
    if (signature.checkpoint_revision.empty() || signature.config_fingerprint.empty()) {
        return Status::error(StatusCode::unvalidated,
                             "DeepSeek-OCR-2 identity requires a checkpoint revision AND config fingerprint; outer model_type alone is never sufficient",
                             "signature");
    }
    if (signature.tensor_names.empty()) {
        return Status::error(StatusCode::unvalidated, "DeepSeek-OCR-2 identity requires the exact tensor signature", "tensor_names");
    }
    auto expected = deepseek_ocr2_vision_expected_tensors(config);
    auto dec = deepseek_ocr2_decoder_expected_tensors(config);
    expected.insert(expected.end(), dec.begin(), dec.end());
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

[[nodiscard]] Status deepseek_ocr2_validate_real_artifact(luisa::string_view checkpoint_revision,
                                                          luisa::string_view config_fingerprint) {
    (void)checkpoint_revision;
    (void)config_fingerprint;
    return Status::error(StatusCode::unvalidated,
                         "DeepSeek-OCR-2 has no qualified official artifact: real checkpoints are refused "
                         "(synthetic recipe-driven path only; see examples/ocr/MODELS.md)",
                         "deepseek_ocr_2");
}

// ---------------------------------------------------------------------------
// Memory preflight (host; no allocation)
// ---------------------------------------------------------------------------

// EXACT workspace arena plan shared by the executor and the preflight. The
// region names/order/sizes mirror runtime/moe_vlm_executor.cpp reserve_workspace
// precisely, so the preflighted workspace bytes EQUAL the measured arena.
[[nodiscard]] Result<WorkspacePlan> deepseek_ocr2_workspace_plan(const DeepSeekOcr2Config &config) {
    using R = Result<WorkspacePlan>;
    const auto &v = config.vision;
    const auto &d = config.decoder.decoder;
    const uint64_t maxp = v.max_vision_patches;
    const uint64_t vh = v.perception.hidden_size, vi = v.perception.intermediate_size;
    const uint64_t patch_dim = v.perception.patch_dim();
    const uint64_t eh = v.encoder_hidden(), ei = v.encoder_intermediate_size;
    const uint64_t nq = v.num_queries;
    const uint64_t enc_rows_max = maxp + nq;
    const uint64_t th = d.hidden_size;
    const uint64_t mr = config.decoder.prefill_chunk_rows;
    const uint64_t qd = d.q_dim(), kvd = d.kv_dim(), vocab = d.vocab_size;
    const uint64_t top_k = config.decoder.top_k;
    const uint64_t slots_max = mr * top_k;
    const uint64_t mi = config.decoder.moe_intermediate_size;
    const uint64_t si = config.decoder.shared_intermediate_size;
    const uint64_t ne = config.decoder.num_experts;
    const uint64_t act_max = std::max(slots_max * mi, mr * si);
    const uint64_t max_channels = std::max({vh, static_cast<uint64_t>(v.down1.out_channels),
                                            static_cast<uint64_t>(v.down2.out_channels)});
    const uint64_t mask_words_max = (enc_rows_max * enc_rows_max + 3ull) / 4ull + 1ull;
    const auto bytes_of = [](uint64_t count) -> Result<uint64_t> {
        return tensor_byte_size(TensorDType::f32, count);
    };
    struct RegionSpec { const char *name; uint64_t elements; };
    const RegionSpec specs[] = {
        {"pixels", maxp * patch_dim},
        {"patches", maxp * patch_dim},
        {"px", maxp * vh}, {"pnorm", maxp * vh}, {"pq", maxp * vh}, {"pk", maxp * vh},
        {"pv", maxp * vh}, {"pqr", maxp * vh}, {"pkr", maxp * vh}, {"pattn", maxp * vh},
        {"pfc1", maxp * vi}, {"paux", vh + maxp * vh},
        {"conv_a", maxp * max_channels}, {"conv_b", maxp * max_channels}, {"conv_tokens", maxp * max_channels},
        {"ex", enc_rows_max * eh}, {"enorm", enc_rows_max * eh}, {"eq", enc_rows_max * eh},
        {"ek", enc_rows_max * eh}, {"ev", enc_rows_max * eh}, {"eattn", enc_rows_max * eh},
        {"efc1", enc_rows_max * ei}, {"eaux", eh + enc_rows_max * eh},
        {"enc_mask", mask_words_max}, {"enc_pos", enc_rows_max}, {"enc_axis", v.perception.head_dim / 2u}, {"p_pos", maxp * 2u},
        {"conn_norm", nq * eh}, {"conn_fc1", nq * v.connector_intermediate_size}, {"features", nq * th},
        {"emb", mr * th}, {"x", mr * th}, {"norm", mr * th}, {"norm2", mr * th},
        {"q", mr * qd}, {"k", mr * kvd}, {"v", mr * kvd}, {"q_rope", mr * qd}, {"k_rope", mr * kvd},
        {"attn", mr * qd}, {"logits", vocab}, {"ids_tok", mr}, {"positions", mr}, {"axis", d.head_dim / 2u},
        {"dummy_f", 1}, {"dummy_u", 1},
    };
    WorkspacePlan plan;
    for (const auto &spec : specs) {
        auto bytes = bytes_of(spec.elements);
        if (!bytes.ok()) { return R::failure(bytes.status); }
        if (auto s = plan.add_bytes(spec.name, *bytes.value); !s.ok()) { return R::failure(s); }
    }
    const uint32_t nlayers = d.num_layers;
    for (uint32_t l = 0; l < nlayers; l++) {
        const RegionSpec moe_specs[] = {
            {"router_logits", mr * ne}, {"probs", mr * ne}, {"ids", slots_max}, {"weights", slots_max},
            {"counts", ne}, {"offsets", ne}, {"perm_slot", slots_max}, {"slot_pos", slots_max},
            {"perm_token", slots_max}, {"gathered", slots_max * th}, {"gate", act_max},
            {"up", act_max}, {"act", act_max}, {"expert_out", slots_max * th},
            {"shared0", mr * th}, {"shared1", mr * th}, {"moe_out", mr * th},
        };
        for (const auto &spec : moe_specs) {
            auto bytes = bytes_of(spec.elements);
            if (!bytes.ok()) { return R::failure(bytes.status); }
            if (auto s = plan.add_bytes(luisa::format("moe{}_{}", l, spec.name), *bytes.value); !s.ok()) { return R::failure(s); }
        }
    }
    for (uint32_t l = 0; l < v.encoder_layers; l++) {
        auto bytes = bytes_of(enc_rows_max * eh);
        if (!bytes.ok()) { return R::failure(bytes.status); }
        if (auto s = plan.add_bytes(luisa::format("enc_attn_{}", l), *bytes.value); !s.ok()) { return R::failure(s); }
    }
    return R::success(std::move(plan));
}

[[nodiscard]] Result<DeepSeekOcr2MemoryReport> deepseek_ocr2_preflight(
    const DeepSeekOcr2Config &config, uint32_t max_seq_len, MemoryLimits limits) {
    using R = Result<DeepSeekOcr2MemoryReport>;
    if (auto s = config.validate(); !s.ok()) { return R::failure(s); }
    if (max_seq_len == 0u) {
        return R::failure(StatusCode::invalid_argument, "max_seq_len must be nonzero", "max_seq_len");
    }
    DeepSeekOcr2MemoryReport report;
    report.available_device_bytes = limits.max_device_bytes;
    // Vision weights.
    auto vision_expected = deepseek_ocr2_vision_expected_tensors(config);
    auto vb = inventory_bytes(vision_expected);
    if (!vb.ok()) { return R::failure(vb.status); }
    report.vision_weight_bytes = *vb.value;
    // Decoder weights. S1: the MoE decoder always ships a STANDALONE lm_head
    // (validate() refuses tie_word_embeddings), so there is NO executor-owned
    // tied-head transpose buffer to account for here.
    auto decoder_expected = deepseek_ocr2_decoder_expected_tensors(config);
    auto db = inventory_bytes(decoder_expected);
    if (!db.ok()) { return R::failure(db.status); }
    report.decoder_weight_bytes = *db.value;
    // Workspace: the EXACT shared arena plan (same regions/order/sizes as the
    // executor's reserve_workspace), so the preflighted bytes EQUAL the measured
    // arena capacity. No estimate.
    const auto &d = config.decoder.decoder;
    auto ws_plan = deepseek_ocr2_workspace_plan(config);
    if (!ws_plan.ok()) { return R::failure(ws_plan.status); }
    report.workspace_bytes = ws_plan.value->total_bytes();
    // KV cache: all decoder layers are full-attention (contiguous KV). Use the
    // host-safe plan_kv_cache_bytes (no device decode_cache.h dependency).
    auto kv = plan_kv_cache_bytes(d.num_layers, 1u, max_seq_len, d.num_kv_heads, d.head_dim, 4u);
    if (!kv.ok()) { return R::failure(kv.status); }
    // DecodeState metadata: 4 buffers x batch(1) x 4 bytes = 16 bytes.
    uint64_t kv_bytes = 0;
    if (!checked_add(*kv.value, 16ull, kv_bytes)) {
        return R::failure(StatusCode::out_of_memory, "KV cache byte count overflows", "kv_cache");
    }
    report.kv_cache_bytes = kv_bytes;
    uint64_t total = 0;
    if (!checked_add(report.vision_weight_bytes, report.decoder_weight_bytes, total) ||
        !checked_add(total, report.workspace_bytes, total) ||
        !checked_add(total, report.kv_cache_bytes, total)) {
        return R::failure(StatusCode::out_of_memory, "total device byte requirement overflows", "total");
    }
    report.total_device_bytes = total;
    report.fits = limits.max_device_bytes == 0u || total <= limits.max_device_bytes;
    if (!report.fits) {
        return R::failure(StatusCode::out_of_memory,
                          luisa::format("DeepSeek-OCR-2 requires {} device bytes (vision {} + decoder {} + workspace {} + KV {}) "
                                        "but only {} bytes are available; reduce context or raise the limit",
                                        total, report.vision_weight_bytes, report.decoder_weight_bytes,
                                        report.workspace_bytes, report.kv_cache_bytes, limits.max_device_bytes),
                          "max_device_bytes");
    }
    return R::success(std::move(report));
}

// ---------------------------------------------------------------------------
// Grounded Markdown adapter
// ---------------------------------------------------------------------------

[[nodiscard]] Result<LayoutDocument> deepseek_ocr2_compose(const LayoutRunResult &run, LayoutOutputMode mode) {
    using R = Result<LayoutDocument>;
    if (mode != LayoutOutputMode::plain && mode != LayoutOutputMode::markdown) {
        return R::failure(StatusCode::unsupported,
                          "DeepSeek-OCR-2 supports plain and grounded markdown output only; spotting/extract are not validated",
                          "mode");
    }
    // M8: GROUND every region BEFORE composition. The raw generative content may
    // carry <|ref|>/<|det|> markers; they must be parsed into source coordinates
    // and stripped from the canonical content. Malformed or out-of-range
    // grounding fails closed instead of leaking markers with an empty box.
    LayoutRunResult grounded;
    grounded.diagnostics = run.diagnostics;
    grounded.regions_processed = run.regions_processed;
    grounded.patches_processed = run.patches_processed;
    grounded.output_tokens = run.output_tokens;
    grounded.layout_cache_hits = run.layout_cache_hits;
    grounded.weight_bytes = run.weight_bytes;
    grounded.workspace_bytes = run.workspace_bytes;
    grounded.kv_cache_peak_bytes = run.kv_cache_peak_bytes;
    grounded.peak_device_bytes = run.peak_device_bytes;
    for (const auto &page : run.pages) {
        if (page.source_width == 0u || page.source_height == 0u) {
            return R::failure(StatusCode::invalid_argument,
                              "grounded composition requires nonzero page source dimensions", "page");
        }
        LayoutPageResult out_page;
        out_page.source_width = page.source_width;
        out_page.source_height = page.source_height;
        for (const auto &region : page.regions) {
            auto parsed = deepseek_ocr2_parse_grounded(region.content, page.source_width, page.source_height);
            if (!parsed.ok()) { return R::failure(parsed.status); }
            // Canonical content = the marker-free text of every block, in order.
            luisa::string canonical;
            luisa::vector<output::DbRect> boxes;
            for (const auto &b : *parsed.value) {
                if (!canonical.empty() && !b.text.empty()) { canonical += "\n"; }
                canonical += b.text;
                for (const auto &box : b.boxes) { boxes.push_back(box); }
            }
            LayoutRegionResult out_region = region;
            out_region.content = std::move(canonical);
            if (!boxes.empty()) {
                // The region's source box is the union of its grounding boxes;
                // every box is also preserved as polygon corners so no
                // coordinate is silently dropped.
                output::DbRect u = boxes.front();
                for (const auto &b : boxes) {
                    u.x0 = std::min(u.x0, b.x0);
                    u.y0 = std::min(u.y0, b.y0);
                    u.x1 = std::max(u.x1, b.x1);
                    u.y1 = std::max(u.y1, b.y1);
                }
                out_region.source_box = u;
                out_region.polygon.clear();
                for (const auto &b : boxes) {
                    out_region.polygon.push_back(output::DbPoint{b.x0, b.y0});
                    out_region.polygon.push_back(output::DbPoint{b.x1, b.y0});
                    out_region.polygon.push_back(output::DbPoint{b.x1, b.y1});
                    out_region.polygon.push_back(output::DbPoint{b.x0, b.y1});
                }
            }
            out_page.regions.push_back(std::move(out_region));
        }
        grounded.pages.push_back(std::move(out_page));
    }
    return layout_compose_strict(grounded, mode, {});
}

namespace {

// Skip ASCII whitespace.
[[nodiscard]] size_t skip_ws(luisa::string_view s, size_t i) noexcept {
    while (i < s.size() && (s[i] == ' ' || s[i] == '\t' || s[i] == '\r' || s[i] == '\n')) { i++; }
    return i;
}

// Parse one nonnegative integer at s[i..]; advances i. Fails on overflow/none.
[[nodiscard]] bool parse_uint(luisa::string_view s, size_t &i, uint64_t &out) noexcept {
    const size_t start = i;
    uint64_t v = 0;
    while (i < s.size() && s[i] >= '0' && s[i] <= '9') {
        const uint64_t d = static_cast<uint64_t>(s[i] - '0');
        if (v > (UINT64_MAX - d) / 10ull) { return false; }
        v = v * 10ull + d;
        i++;
    }
    if (i == start) { return false; }
    out = v;
    return true;
}

}// namespace

[[nodiscard]] Result<luisa::vector<DeepSeekGroundedBlock>> deepseek_ocr2_parse_grounded(
    luisa::string_view text, uint32_t page_width, uint32_t page_height) {
    using R = Result<luisa::vector<DeepSeekGroundedBlock>>;
    if (page_width == 0u || page_height == 0u) {
        return R::failure(StatusCode::invalid_argument, "page dimensions must be nonzero", "page");
    }
    constexpr luisa::string_view kRefOpen = "<|ref|>";
    constexpr luisa::string_view kRefClose = "<|/ref|>";
    constexpr luisa::string_view kDetOpen = "<|det|>";
    constexpr luisa::string_view kDetClose = "<|/det|>";
    luisa::vector<DeepSeekGroundedBlock> blocks;
    size_t i = 0;
    luisa::string plain_accum;
    const auto flush_plain = [&]() {
        if (!plain_accum.empty()) {
            DeepSeekGroundedBlock b;
            b.text = plain_accum;
            b.grounded = false;
            blocks.push_back(std::move(b));
            plain_accum.clear();
        }
    };
    while (i < text.size()) {
        const size_t ref = text.find(kRefOpen, i);
        if (ref == luisa::string_view::npos) {
            plain_accum.append(text.substr(i));
            break;
        }
        plain_accum.append(text.substr(i, ref - i));
        const size_t text_begin = ref + kRefOpen.size();
        const size_t ref_close = text.find(kRefClose, text_begin);
        if (ref_close == luisa::string_view::npos) {
            return R::failure(StatusCode::invalid_argument, "unclosed <|ref|> grounding token", "grounding");
        }
        flush_plain();
        DeepSeekGroundedBlock b;
        b.text = luisa::string{text.substr(text_begin, ref_close - text_begin)};
        b.grounded = true;
        // The detection block MUST immediately follow the ref close.
        size_t j = skip_ws(text, ref_close + kRefClose.size());
        if (text.compare(j, kDetOpen.size(), kDetOpen) != 0) {
            return R::failure(StatusCode::invalid_argument, "grounded <|ref|> must be followed by <|det|> coordinates", "grounding");
        }
        j += kDetOpen.size();
        const size_t det_close = text.find(kDetClose, j);
        if (det_close == luisa::string_view::npos) {
            return R::failure(StatusCode::invalid_argument, "unclosed <|det|> grounding token", "grounding");
        }
        const luisa::string_view det = text.substr(j, det_close - j);
        // Parse [[x0,y0,x1,y1], ...] in the CLOSED [0,1000] normalized space
        // (S7: the header/docs declare [0,1000] inclusive; 1000 maps to the full
        // page edge). Trailing commas are rejected.
        size_t k = skip_ws(det, 0);
        if (k >= det.size() || det[k] != '[') {
            return R::failure(StatusCode::invalid_argument, "grounding coordinates must be a [[...]] list", "grounding");
        }
        k++;
        bool any_box = false;
        while (true) {
            k = skip_ws(det, k);
            if (k < det.size() && det[k] == ']') { k++; break; }// end of outer list
            if (k >= det.size() || det[k] != '[') {
                return R::failure(StatusCode::invalid_argument, "each grounding box must be [x0,y0,x1,y1]", "grounding");
            }
            k++;
            uint64_t c[4];
            for (int ci = 0; ci < 4; ci++) {
                k = skip_ws(det, k);
                if (!parse_uint(det, k, c[ci])) {
                    return R::failure(StatusCode::invalid_argument, "grounding coordinate must be a nonnegative integer", "grounding");
                }
                if (c[ci] > 1000ull) {
                    return R::failure(StatusCode::invalid_argument, "grounding coordinate out of [0,1000] range", "grounding");
                }
                k = skip_ws(det, k);
                if (ci < 3) {
                    if (k >= det.size() || det[k] != ',') {
                        return R::failure(StatusCode::invalid_argument, "grounding box needs 4 comma-separated coordinates", "grounding");
                    }
                    k++;
                }
            }
            k = skip_ws(det, k);
            if (k >= det.size() || det[k] != ']') {
                return R::failure(StatusCode::invalid_argument, "unterminated grounding box", "grounding");
            }
            k++;
            if (c[2] <= c[0] || c[3] <= c[1]) {
                return R::failure(StatusCode::invalid_argument, "grounding box must have x1>x0 and y1>y0", "grounding");
            }
            output::DbRect box;
            box.x0 = static_cast<float>(c[0]) / 1000.0f * static_cast<float>(page_width);
            box.y0 = static_cast<float>(c[1]) / 1000.0f * static_cast<float>(page_height);
            box.x1 = static_cast<float>(c[2]) / 1000.0f * static_cast<float>(page_width);
            box.y1 = static_cast<float>(c[3]) / 1000.0f * static_cast<float>(page_height);
            b.boxes.push_back(box);
            any_box = true;
            k = skip_ws(det, k);
            if (k < det.size() && det[k] == ',') {
                k++;
                // S7: a trailing comma before the closing ']' is malformed.
                const size_t after = skip_ws(det, k);
                if (after < det.size() && det[after] == ']') {
                    return R::failure(StatusCode::invalid_argument,
                                      "trailing comma in the grounding coordinate list", "grounding");
                }
                continue;
            }
            if (k < det.size() && det[k] == ']') { k++; break; }
            return R::failure(StatusCode::invalid_argument, "expected ',' or ']' after a grounding box", "grounding");
        }
        if (!any_box) {
            return R::failure(StatusCode::invalid_argument, "grounded block must carry at least one box", "grounding");
        }
        k = skip_ws(det, k);
        if (k != det.size()) {
            return R::failure(StatusCode::invalid_argument, "trailing data after grounding coordinate list", "grounding");
        }
        blocks.push_back(std::move(b));
        i = det_close + kDetClose.size();
    }
    flush_plain();
    return R::success(std::move(blocks));
}

}// namespace luisa::ocr

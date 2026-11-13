// OCR engine -- dots.mocr graph contract (Phase 8). Host-only TU.
// Synthetic recipe-driven path over the shared dense executor; real artifacts
// fail closed. See dots_mocr.h.
#include "ocr/models/dots_mocr.h"

#include <utility>

#include <luisa/core/stl/format.h>
#include <yyjson.h>

#include "ocr/runtime/layout_result.h"

namespace luisa::ocr {

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

[[nodiscard]] Status DotsMocrConfig::validate() const {
    if (auto s = graph.validate(); !s.ok()) { return s; }
    if (auto s = names.validate(); !s.ok()) { return s; }
    // dots.mocr identity boundaries (never inferred from model_type):
    //  * vision is a bidirectional ViT with 2D axial RoPE (no QK-Norm);
    //  * the connector is the spatial-merge patch merger;
    //  * the decoder is a dense GQA RMSNorm / 1D-RoPE / SwiGLU transformer
    //    (NO QK-Norm, NO M-RoPE -- that is the Qwen-VL convention).
    if (graph.vision.qk_norm) {
        return Status::error(StatusCode::invalid_argument, "dots.mocr vision has no QK-Norm", "vision.qk_norm");
    }
    if (graph.vision.mask_kind != P7VisionMaskKind::bidirectional) {
        return Status::error(StatusCode::invalid_argument, "dots.mocr ViT uses bidirectional attention", "vision.mask_kind");
    }
    if (graph.connector.kind != P7ConnectorKind::spatial_mlp) {
        return Status::error(StatusCode::invalid_argument, "dots.mocr uses the spatial-merge patch merger", "connector.kind");
    }
    if (graph.text_pos != P7TextPosKind::rope1d) {
        return Status::error(StatusCode::invalid_argument,
                             "dots.mocr uses plain 1D text RoPE; M-RoPE is the Qwen-VL convention and is never substituted",
                             "text_pos");
    }
    if (graph.text.num_attention_heads % graph.text.num_kv_heads != 0u) {
        return Status::error(StatusCode::invalid_argument, "dots.mocr decoder requires GQA divisibility", "num_kv_heads");
    }
    // Storage policy.
    if (linear_storage != DenseLinearStorage::fp32) {
        if (linear_storage != DenseLinearStorage::int4_sym && linear_storage != DenseLinearStorage::int4_asym) {
            return Status::error(StatusCode::invalid_argument, "unknown dots.mocr linear storage", "linear_storage");
        }
        if (!is_supported_group_size(linear_group_size)) {
            return Status::error(StatusCode::invalid_argument, "dots.mocr W4A32 requires group size 64 or 128", "linear_group_size");
        }
        if (graph.tie_word_embeddings) {
            return Status::error(StatusCode::unsupported,
                                 "the dots.mocr W4A32 path requires a standalone packed lm_head", "tie_word_embeddings");
        }
    } else if (linear_group_size != 0u) {
        return Status::error(StatusCode::invalid_argument, "linear_group_size must be 0 for fp32", "linear_group_size");
    }
    return Status::success();
}

[[nodiscard]] DenseDecoderConfig dots_mocr_declared_decoder_config() {
    // The DECLARED full contract geometry (28 layers / 12 Q / 2 KV). The hidden
    // / head_dim / intermediate values are the declared contract, NOT a verified
    // checkpoint. head_dim 128 * 12 heads = 1536 hidden. This is a consistent
    // declared geometry used to validate the 28/12/2 contract explicitly.
    DenseDecoderConfig d;
    d.vocab_size = 151936;
    d.hidden_size = 1536;
    d.num_layers = 28;
    d.num_attention_heads = 12;
    d.num_kv_heads = 2;
    d.head_dim = 128;
    d.intermediate_size = 8960;
    d.rms_norm_eps = 1e-6f;
    d.rope_theta = 1000000.0f;
    d.rope_pair_layout = RopePairLayout::rotate_half;
    d.max_position_embeddings = 32768;
    d.eos_token_ids = {151645u};
    return d;
}

[[nodiscard]] DotsMocrConfig dots_mocr_synthetic_config() {
    DotsMocrConfig cfg;
    auto &g = cfg.graph;
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
    // SCALED dense GQA decoder (small synthetic): SAME RMSNorm / 1D-RoPE /
    // SwiGLU conventions as the full 28/12/2 contract. 4 Q / 2 KV (GQA group 2,
    // the same ratio as 12/2 = 6 is NOT preserved here; the full contract is
    // validated separately). rotate-half pairs match the declared contract.
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
    g.packing.grid_multiple = 2;
    g.vision_component = "vision";
    g.decoder_component = "decoder";
    g.max_vision_patches = 4096;
    g.prefill_chunk_rows = 256;
    cfg.names.text.prefix = "model.";
    cfg.names.visual_prefix = "visual.";
    cfg.linear_storage = DenseLinearStorage::fp32;
    cfg.linear_group_size = 0;
    return cfg;
}

// ---------------------------------------------------------------------------
// Config parsing (strict; fail-closed on MoE / GDN / Qwen3 markers)
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

[[nodiscard]] Result<DotsMocrConfig> parse_dots_mocr_config(luisa::string_view json) {
    using R = Result<DotsMocrConfig>;
    JsonDoc doc;
    doc.doc = yyjson_read(json.data(), json.size(), 0);
    if (doc.doc == nullptr) { return R::failure(StatusCode::invalid_argument, "dots.mocr config is not valid JSON", "config"); }
    auto *root = yyjson_doc_get_root(doc.doc);
    if (root == nullptr || !yyjson_is_obj(root)) { return R::failure(StatusCode::invalid_argument, "dots.mocr config must be a JSON object", "config"); }
    if (auto *mt = yyjson_obj_get(root, "model_type"); mt != nullptr && yyjson_is_str(mt)) {
        const luisa::string_view type{yyjson_get_str(mt), yyjson_get_len(mt)};
        if (type.find("dots") == luisa::string_view::npos) {
            return R::failure(StatusCode::unsupported, "outer model_type does not name dots.mocr (and is never sufficient for identity)", "model_type");
        }
    }
    for (const char *marker : {"num_experts", "linear_num_key_heads", "linear_key_head_dim", "grpo", "mtp_num_layers", "qk_norm"}) {
        if (auto *v = yyjson_obj_get(root, marker); v != nullptr && !yyjson_is_null(v) && !yyjson_is_false(v)) {
            return R::failure(StatusCode::unsupported, luisa::format("dots.mocr config field '{}' is outside the dense contract", marker), marker);
        }
    }
    DotsMocrConfig cfg = dots_mocr_synthetic_config();
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
            return R::failure(StatusCode::unsupported, "dots.mocr decoder has no QK-Norm (that is the Qwen3 convention)", "text_config.qk_norm");
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
    auto *cc = yyjson_obj_get(root, "connector");
    if (cc != nullptr && yyjson_is_obj(cc)) {
        auto r = get_u32(cc, "intermediate_size", false, g.connector.intermediate_size);
        if (!r.ok()) { return R::failure(r.status); }
        g.connector.intermediate_size = *r.value;
        auto m = get_u32(cc, "spatial_merge_size", false, g.connector.spatial_factor);
        if (!m.ok()) { return R::failure(m.status); }
        g.connector.spatial_factor = *m.value;
    }
    g.packing.patch_size = g.vision.patch_size;
    g.packing.grid_multiple = g.connector.spatial_factor;
    // Optional quantization policy.
    if (auto *q = yyjson_obj_get(root, "quantization"); q != nullptr && yyjson_is_obj(q)) {
        auto *kind = yyjson_obj_get(q, "linear_storage");
        if (kind != nullptr && yyjson_is_str(kind)) {
            const luisa::string_view ks{yyjson_get_str(kind), yyjson_get_len(kind)};
            if (ks == "int4_sym") { cfg.linear_storage = DenseLinearStorage::int4_sym; }
            else if (ks == "int4_asym") { cfg.linear_storage = DenseLinearStorage::int4_asym; }
            else if (ks != "fp32") { return R::failure(StatusCode::unsupported, "unknown dots.mocr linear_storage", "quantization.linear_storage"); }
        }
        auto gs = get_u32(q, "group_size", false, 0);
        if (!gs.ok()) { return R::failure(gs.status); }
        cfg.linear_group_size = *gs.value;
        if (cfg.linear_storage != DenseLinearStorage::fp32) { cfg.graph.tie_word_embeddings = false; }
    }
    if (auto s = cfg.validate(); !s.ok()) { return R::failure(s); }
    return R::success(std::move(cfg));
}

// ---------------------------------------------------------------------------
// Tensor inventory + signature + real-artifact gate
// ---------------------------------------------------------------------------

[[nodiscard]] luisa::vector<ExpectedTensor> dots_mocr_expected_tensors(const DotsMocrConfig &config) {
    return p7_expected_tensors(config.graph, config.names);
}

[[nodiscard]] Status dots_mocr_validate_tensors(luisa::span<const PackageTensor> tensors,
                                                luisa::span<const ExpectedTensor> expected,
                                                DenseLinearStorage linear_storage) {
    // S1: only DECODER linears (model. prefix projections + lm_head) are
    // quantizable; vision/connector tensors (visual.*) must stay dense FP32
    // even though they share projection suffixes. The shared gate enforces
    // the prefix restriction and rejects unexpected names.
    return validate_dense_decoder_tensors_storage(tensors, expected, linear_storage,
                                                  "model.", "lm_head.weight");
}


[[nodiscard]] Status dots_mocr_validate_packages(const DotsMocrConfig &config,
                                                luisa::span<const PackageTensor> vision_tensors,
                                                luisa::span<const PackageTensor> decoder_tensors) {
    if (auto s = config.validate(); !s.ok()) { return s; }
    const auto expected = dots_mocr_expected_tensors(config);
    // Split the inventory: vision/connector tensors carry the visual prefix;
    // decoder tensors carry the text prefix. Each component table is validated
    // against its OWN slice (unexpected names in either component are refused).
    luisa::vector<ExpectedTensor> vision_expected, decoder_expected;
    for (const auto &e : expected) {
        if (e.name.compare(0, config.names.visual_prefix.size(), config.names.visual_prefix) == 0) {
            vision_expected.push_back(e);
        } else {
            decoder_expected.push_back(e);
        }
    }
    // Vision/connector: ALWAYS dense FP32 (fp32 policy for the whole slice).
    if (auto s = validate_dense_decoder_tensors_storage(vision_tensors, vision_expected,
                                                        DenseLinearStorage::fp32, "", ""); !s.ok()) {
        return s;
    }
    // Decoder: the configured per-linear storage policy.
    return validate_dense_decoder_tensors_storage(decoder_tensors, decoder_expected,
                                                  config.linear_storage,
                                                  config.names.text.prefix, config.names.text.lm_head());
}

[[nodiscard]] Status dots_mocr_validate_signature(const DotsMocrConfig &config, const DotsMocrSignature &signature) {
    if (signature.checkpoint_revision.empty() || signature.config_fingerprint.empty()) {
        return Status::error(StatusCode::unvalidated,
                             "dots.mocr identity requires a checkpoint revision AND config fingerprint; outer model_type alone is never sufficient",
                             "signature");
    }
    if (signature.tensor_names.empty()) {
        return Status::error(StatusCode::unvalidated, "dots.mocr identity requires the exact tensor signature", "tensor_names");
    }
    const auto expected = dots_mocr_expected_tensors(config);
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

[[nodiscard]] Status dots_mocr_validate_real_artifact(luisa::string_view checkpoint_revision,
                                                     luisa::string_view config_fingerprint) {
    (void)checkpoint_revision;
    (void)config_fingerprint;
    return Status::error(StatusCode::unvalidated,
                         "dots.mocr has no qualified official artifact: real checkpoints are refused "
                         "(synthetic recipe-driven path only; see examples/ocr/MODELS.md)",
                         "dots_mocr");
}

// ---------------------------------------------------------------------------
// Memory preflight (host; no allocation)
// ---------------------------------------------------------------------------

namespace {

// Workspace region alignment. MUST match runtime/workspace.h kWorkspaceAlignment
// (the host preflight mirrors the device arena layout without including the
// device header, which would clash with the host composition API).
constexpr uint64_t kDotsWorkspaceAlignment = 256u;

// Checked bytes for `count` fp32 elements.
[[nodiscard]] Result<uint64_t> f32_bytes(uint64_t count) {
    return tensor_byte_size(TensorDType::f32, count);
}

// Decoder weight bytes for the declared geometry + storage. Embeddings, norms,
// and QKV biases are FP32; the linear projections (and standalone lm_head) are
// FP32 or packed INT4 (codes + per-group scales [+ zero points]).
[[nodiscard]] Result<uint64_t> decoder_weight_bytes(const DenseDecoderConfig &d, bool tied,
                                                    DenseLinearStorage storage, uint32_t group_size) {
    using R = Result<uint64_t>;
    uint64_t total = 0;
    const auto add = [&total](uint64_t bytes) -> Status {
        uint64_t next = 0;
        if (!checked_add(total, bytes, next)) {
            return Status::error(StatusCode::out_of_memory, "decoder weight byte total overflows", "weights");
        }
        total = next;
        return Status::success();
    };
    const auto fp32 = [&add](uint64_t elems) -> Status {
        auto b = f32_bytes(elems);
        if (!b.ok()) { return b.status; }
        return add(*b.value);
    };
    // One linear B operand [K, N].
    const auto linear = [&](uint64_t k, uint64_t n) -> Status {
        uint64_t elems = 0;
        if (!checked_mul(k, n, elems)) {
            return Status::error(StatusCode::out_of_memory, "linear element count overflows", "weights");
        }
        if (storage == DenseLinearStorage::fp32) { return fp32(elems); }
        // Packed INT4: ceil(elems/8) uint32 code words + groups FP32 scales
        // (+ groups uint32 zero points for asymmetric).
        uint64_t words = (elems + 7u) / 8u;
        auto wb = tensor_byte_size(TensorDType::u32, words);
        if (!wb.ok()) { return wb.status; }
        if (auto s = add(*wb.value); !s.ok()) { return s; }
        uint64_t groups = 0;
        if (!group_count_for(elems, group_size, groups)) {
            return Status::error(StatusCode::invalid_argument, "group count overflow", "weights");
        }
        auto sb = f32_bytes(groups);
        if (!sb.ok()) { return sb.status; }
        if (auto s = add(*sb.value); !s.ok()) { return s; }
        if (storage == DenseLinearStorage::int4_asym) {
            auto zb = tensor_byte_size(TensorDType::u32, groups);
            if (!zb.ok()) { return zb.status; }
            if (auto s = add(*zb.value); !s.ok()) { return s; }
        }
        return Status::success();
    };
    const uint64_t hidden = d.hidden_size, qd = d.q_dim(), kvd = d.kv_dim(), inter = d.intermediate_size;
    // Embedding table + final norm (FP32).
    if (auto s = fp32(static_cast<uint64_t>(d.vocab_size) * hidden); !s.ok()) { return R::failure(s); }
    if (auto s = fp32(hidden); !s.ok()) { return R::failure(s); }
    // LM head: standalone (quantizable) or tied (FP32, transposed embedding).
    if (!tied) {
        if (auto s = linear(hidden, d.vocab_size); !s.ok()) { return R::failure(s); }
    } else {
        if (auto s = fp32(static_cast<uint64_t>(d.vocab_size) * hidden); !s.ok()) { return R::failure(s); }
    }
    for (uint32_t l = 0; l < d.num_layers; l++) {
        if (auto s = fp32(hidden); !s.ok()) { return R::failure(s); }// input norm
        if (auto s = fp32(hidden); !s.ok()) { return R::failure(s); }// post norm
        if (auto s = linear(hidden, qd); !s.ok()) { return R::failure(s); }// q
        if (auto s = linear(hidden, kvd); !s.ok()) { return R::failure(s); }// k
        if (auto s = linear(hidden, kvd); !s.ok()) { return R::failure(s); }// v
        if (auto s = linear(qd, hidden); !s.ok()) { return R::failure(s); }// o
        if (auto s = fp32(qd); !s.ok()) { return R::failure(s); }// q bias
        if (auto s = fp32(kvd); !s.ok()) { return R::failure(s); }// k bias
        if (auto s = fp32(kvd); !s.ok()) { return R::failure(s); }// v bias
        if (auto s = linear(hidden, inter); !s.ok()) { return R::failure(s); }// gate
        if (auto s = linear(hidden, inter); !s.ok()) { return R::failure(s); }// up
        if (auto s = linear(inter, hidden); !s.ok()) { return R::failure(s); }// down
    }
    return R::success(total);
}

// Dense executor workspace bytes (mirrors DenseVlmExecutor::reserve_workspace
// for the dots.mocr contract: no QK-Norm, no DeepStack, 1-axis RoPE). Each
// region is aligned up to kWorkspaceAlignment.
[[nodiscard]] Result<uint64_t> dense_workspace_bytes(const DenseDecoderConfig &d, uint32_t chunk_rows) {
    using R = Result<uint64_t>;
    const uint64_t chunk = chunk_rows;
    const uint64_t hidden = d.hidden_size, qd = d.q_dim(), kvd = d.kv_dim();
    const uint64_t inter = d.intermediate_size, vocab = d.vocab_size;
    const uint64_t elems[] = {
        chunk * hidden,              // emb
        chunk * hidden,              // x
        chunk * hidden,              // norm
        chunk * hidden,              // norm2
        chunk * qd,                  // q
        chunk * kvd,                 // k
        chunk * kvd,                 // v
        chunk * qd,                  // q_rope
        chunk * kvd,                 // k_rope
        chunk * qd,                  // attn
        chunk * inter,               // gate
        chunk * inter,               // up
        chunk * inter,               // mlp
        vocab,                       // logits
        chunk,                       // ids
        chunk * 1u,                  // positions (1-axis)
        d.head_dim / 2u,             // axis_of_pair
        1,                           // dummy_f
        1,                           // dummy_u
        1,                           // q_normed (unused)
        1,                           // k_normed (unused)
        1,                           // ds_scratch (unused)
    };
    uint64_t total = 0;
    for (auto e : elems) {
        auto b = f32_bytes(e);
        if (!b.ok()) { return R::failure(b.status); }
        uint64_t aligned = 0;
        if (!checked_align_up(*b.value, kDotsWorkspaceAlignment, aligned)) {
            return R::failure(StatusCode::out_of_memory, "workspace alignment overflow", "workspace");
        }
        uint64_t next = 0;
        if (!checked_add(total, aligned, next)) {
            return R::failure(StatusCode::out_of_memory, "workspace byte total overflows", "workspace");
        }
        total = next;
    }
    return R::success(total);
}

}// namespace

[[nodiscard]] Result<DotsMocrMemoryReport> dots_mocr_preflight_decoder(
    const DotsMocrConfig &config, uint32_t max_seq_len, MemoryLimits limits) {
    using R = Result<DotsMocrMemoryReport>;
    if (auto s = config.validate(); !s.ok()) { return R::failure(s); }
    if (max_seq_len == 0u) {
        return R::failure(StatusCode::invalid_argument, "max_seq_len must be nonzero", "max_seq_len");
    }
    const auto &d = config.graph.text;
    DotsMocrMemoryReport report;
    report.available_device_bytes = limits.max_device_bytes;
    auto weights = decoder_weight_bytes(d, config.graph.tie_word_embeddings, config.linear_storage, config.linear_group_size);
    if (!weights.ok()) { return R::failure(weights.status); }
    report.decoder_weight_bytes = *weights.value;
    auto ws = dense_workspace_bytes(d, config.graph.prefill_chunk_rows);
    if (!ws.ok()) { return R::failure(ws.status); }
    report.workspace_bytes = *ws.value;
    // KV cache: layers * tokens * kv_heads * head_dim * 2 (K+V) * 4 bytes, plus
    // the DecodeState metadata (4 buffers * batch 1 * 4 bytes = 16 bytes).
    auto kv = plan_kv_cache_bytes(d.num_layers, 1u, max_seq_len, d.num_kv_heads, d.head_dim, 4u);
    if (!kv.ok()) { return R::failure(kv.status); }
    report.kv_cache_bytes = *kv.value + 16u;
    uint64_t total = 0;
    if (!checked_add(report.decoder_weight_bytes, report.workspace_bytes, total) ||
        !checked_add(total, report.kv_cache_bytes, total)) {
        return R::failure(StatusCode::out_of_memory, "total device byte requirement overflows", "total");
    }
    report.total_device_bytes = total;
    report.fits = limits.max_device_bytes == 0u || total <= limits.max_device_bytes;
    if (!report.fits) {
        return R::failure(StatusCode::out_of_memory,
                          luisa::format("dots.mocr decoder requires {} device bytes (weights {} + workspace {} + KV {}) "
                                        "but only {} bytes are available; reduce context, quantize, or raise the limit",
                                        total, report.decoder_weight_bytes, report.workspace_bytes, report.kv_cache_bytes,
                                        limits.max_device_bytes),
                          "max_device_bytes");
    }
    return R::success(std::move(report));
}

// ---------------------------------------------------------------------------
// Host composition
// ---------------------------------------------------------------------------

[[nodiscard]] Result<LayoutDocument> dots_mocr_compose(const LayoutRunResult &run, LayoutOutputMode mode) {
    using R = Result<LayoutDocument>;
    if (mode != LayoutOutputMode::plain && mode != LayoutOutputMode::markdown) {
        return R::failure(StatusCode::unsupported,
                          "dots.mocr supports plain and markdown output only; spotting/extract are not validated",
                          "mode");
    }
    return layout_compose_strict(run, mode, {});
}

}// namespace luisa::ocr

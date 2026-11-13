// OCR engine -- olmOCR-2 graph contract (Phase 8). Host-only TU.
// Synthetic recipe-driven path over the shared dense executor; real artifacts
// fail closed. See olm_ocr2.h.
#include "ocr/models/olm_ocr2.h"

#include <algorithm>
#include <utility>

#include <luisa/core/stl/format.h>
#include <yyjson.h>

#include "ocr/runtime/layout_result.h"

namespace luisa::ocr {

[[nodiscard]] luisa::string_view olm_offload_policy_name(OlmOffloadPolicy policy) noexcept {
    switch (policy) {
        case OlmOffloadPolicy::none: return "none";
        case OlmOffloadPolicy::explicit_host: return "explicit_host";
        default: return "unknown";
    }
}

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

[[nodiscard]] Status OlmOcr2Config::validate() const {
    if (auto s = graph.validate(); !s.ok()) { return s; }
    if (auto s = names.validate(); !s.ok()) { return s; }
    // olmOCR-2 identity boundaries (Qwen2.5-VL-derived, never Qwen3):
    //  * vision is window/global with 2D axial RoPE, NO QK-Norm;
    //  * the merger is spatial-merge with RMSNorm (Qwen2.5-VL convention);
    //  * the decoder is Qwen2.5-VL: QKV bias, NO QK-Norm, rotate-half M-RoPE.
    if (graph.vision.mask_kind != P7VisionMaskKind::window_global) {
        return Status::error(StatusCode::invalid_argument, "olmOCR-2 Qwen2.5-VL vision uses window/global attention", "vision.mask_kind");
    }
    if (graph.vision.qk_norm) {
        return Status::error(StatusCode::invalid_argument, "olmOCR-2 vision has no QK-Norm (that is the Qwen3-VL convention)", "vision.qk_norm");
    }
    if (graph.connector.kind != P7ConnectorKind::spatial_mlp) {
        return Status::error(StatusCode::invalid_argument, "olmOCR-2 uses the spatial-merge merger", "connector.kind");
    }
    if (!graph.connector.rms_norm) {
        return Status::error(StatusCode::invalid_argument, "olmOCR-2 Qwen2.5-VL merger uses RMSNorm (not LayerNorm)", "connector.rms_norm");
    }
    if (graph.text_pos != P7TextPosKind::mrope_interleaved) {
        return Status::error(StatusCode::invalid_argument,
                             "olmOCR-2 requires the Qwen2.5-VL rotate-half M-RoPE decoder", "text_pos");
    }
    if (graph.text.rope_pair_layout != RopePairLayout::rotate_half) {
        return Status::error(StatusCode::invalid_argument, "olmOCR-2 Qwen2.5-VL M-RoPE uses rotate-half pairs", "rope_pair_layout");
    }
    // Offload: only `none` is implemented.
    if (offload != OlmOffloadPolicy::none) {
        return Status::error(StatusCode::unsupported,
                             "olmOCR-2 host offload is not implemented; it fails closed as unsupported (no performance claim)",
                             "offload");
    }
    // Storage policy.
    if (linear_storage != DenseLinearStorage::fp32 &&
        linear_storage != DenseLinearStorage::int4_sym && linear_storage != DenseLinearStorage::int4_asym) {
        return Status::error(StatusCode::invalid_argument, "unknown olmOCR-2 linear storage", "linear_storage");
    }
    if (linear_storage != DenseLinearStorage::fp32) {
        if (!is_supported_group_size(linear_group_size)) {
            return Status::error(StatusCode::invalid_argument, "olmOCR-2 W4A32 requires group size 64 or 128", "linear_group_size");
        }
        if (graph.tie_word_embeddings) {
            return Status::error(StatusCode::unsupported,
                                 "the olmOCR-2 W4A32 path requires a standalone packed lm_head", "tie_word_embeddings");
        }
    } else if (linear_group_size != 0u) {
        return Status::error(StatusCode::invalid_argument, "linear_group_size must be 0 for fp32", "linear_group_size");
    }
    // The declared 8 GB path REQUIRES INT4.
    if (require_int4 && linear_storage == DenseLinearStorage::fp32) {
        return Status::error(StatusCode::unsupported,
                             "the declared olmOCR-2 8 GB path requires INT4 (W4A32); FP32 is only for the small synthetic parity slice",
                             "linear_storage");
    }
    return Status::success();
}

[[nodiscard]] DenseDecoderConfig olm_ocr2_declared_decoder_config() {
    // DECLARED Qwen2.5-VL-7B text geometry (28 layers, 28 Q / 4 KV, hidden 3584,
    // head_dim 128). A declared contract, NOT a verified checkpoint and NOT a
    // claim that a real 7B model fits in 8 GB.
    DenseDecoderConfig d;
    d.vocab_size = 152064;
    d.hidden_size = 3584;
    d.num_layers = 28;
    d.num_attention_heads = 28;
    d.num_kv_heads = 4;
    d.head_dim = 128;
    d.intermediate_size = 18944;
    d.rms_norm_eps = 1e-6f;
    d.rope_theta = 1000000.0f;
    d.rope_pair_layout = RopePairLayout::rotate_half;
    d.max_position_embeddings = 32768;
    d.eos_token_ids = {151645u};
    return d;
}

[[nodiscard]] OlmOcr2Config olm_ocr2_synthetic_config() {
    OlmOcr2Config cfg;
    auto &g = cfg.graph;
    // Qwen2.5-VL-style vision (small synthetic): window/global + 2D RoPE.
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
    g.vision.global_layers = {1u};
    g.vision.position_rows = 0;
    // Spatial-merge RMSNorm merger (Qwen2.5-VL convention).
    g.connector.kind = P7ConnectorKind::spatial_mlp;
    g.connector.intermediate_size = 32;
    g.connector.layer_norm_eps = 1e-5f;
    g.connector.spatial_factor = 2;
    g.connector.rms_norm = true;
    // Qwen2.5-VL-style decoder (small synthetic): QKV bias, no QK-Norm,
    // rotate-half M-RoPE. The P7 slice selects the Qwen3-style QK-Norm text
    // contract for mrope_interleaved, so the synthetic slice instead uses the
    // shared dense executor directly with the Qwen2.5 contract (see the test).
    // Here we declare the geometry; the executor contract is built by the test.
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
    g.text_qwen2_5_contract = true;
    g.pos_sections = {2u, 1u, 1u};
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
    // The small synthetic parity slice uses FP32 (require_int4 false).
    cfg.linear_storage = DenseLinearStorage::fp32;
    cfg.linear_group_size = 0;
    cfg.offload = OlmOffloadPolicy::none;
    cfg.require_int4 = false;
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

[[nodiscard]] Result<OlmOcr2Config> parse_olm_ocr2_config(luisa::string_view json) {
    using R = Result<OlmOcr2Config>;
    JsonDoc doc;
    doc.doc = yyjson_read(json.data(), json.size(), 0);
    if (doc.doc == nullptr) { return R::failure(StatusCode::invalid_argument, "olmOCR-2 config is not valid JSON", "config"); }
    auto *root = yyjson_doc_get_root(doc.doc);
    if (root == nullptr || !yyjson_is_obj(root)) { return R::failure(StatusCode::invalid_argument, "olmOCR-2 config must be a JSON object", "config"); }
    if (auto *mt = yyjson_obj_get(root, "model_type"); mt != nullptr && yyjson_is_str(mt)) {
        const luisa::string_view type{yyjson_get_str(mt), yyjson_get_len(mt)};
        if (type.find("qwen3") != luisa::string_view::npos) {
            return R::failure(StatusCode::unsupported, "olmOCR-2 is Qwen2.5-VL-derived, never Qwen3", "model_type");
        }
    }
    for (const char *marker : {"num_experts", "linear_num_key_heads", "linear_key_head_dim", "grpo", "mtp_num_layers", "qk_norm"}) {
        if (auto *v = yyjson_obj_get(root, marker); v != nullptr && !yyjson_is_null(v) && !yyjson_is_false(v)) {
            return R::failure(StatusCode::unsupported, luisa::format("olmOCR-2 config field '{}' is outside the dense contract", marker), marker);
        }
    }
    // English-only: a declared language list must be an ARRAY of strings, each
    // exactly "en". S4: a present-but-non-array `languages` (scalar, object,
    // null) is REFUSED, never silently ignored; a scalar is not an accepted
    // alternative form.
    if (auto *langs = yyjson_obj_get(root, "languages"); langs != nullptr) {
        if (!yyjson_is_arr(langs)) {
            return R::failure(StatusCode::invalid_argument,
                              "olmOCR-2 languages must be an array of strings (a scalar/object/null is not accepted)",
                              "languages");
        }
        size_t idx = 0, max = 0;
        yyjson_val *item = nullptr;
        yyjson_arr_foreach(langs, idx, max, item) {
            if (!yyjson_is_str(item)) { return R::failure(StatusCode::invalid_argument, "languages entries must be strings", "languages"); }
            const luisa::string_view lang{yyjson_get_str(item), yyjson_get_len(item)};
            if (auto s = olm_ocr2_check_language(lang); !s.ok()) { return R::failure(s); }
        }
    }
    OlmOcr2Config cfg = olm_ocr2_synthetic_config();
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
        if (auto s = u("rope_height_pairs", false, g.vision.rope_height_pairs, g.vision.rope_height_pairs); !s.ok()) { return R::failure(s); }
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
    }
    auto *tc = yyjson_obj_get(root, "text_config");
    if (tc == nullptr || !yyjson_is_obj(tc)) { return R::failure(StatusCode::invalid_argument, "text_config object is required", "text_config"); }
    {
        if (auto *qn = yyjson_obj_get(tc, "qk_norm"); qn != nullptr && yyjson_is_true(qn)) {
            return R::failure(StatusCode::unsupported, "olmOCR-2 Qwen2.5-VL decoder has no QK-Norm", "text_config.qk_norm");
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
    g.packing.patch_size = g.vision.patch_size;
    g.packing.grid_multiple = g.connector.spatial_factor;
    // Quantization policy: the declared 8 GB path requires INT4.
    if (auto *q = yyjson_obj_get(root, "quantization"); q != nullptr && yyjson_is_obj(q)) {
        auto *kind = yyjson_obj_get(q, "linear_storage");
        if (kind != nullptr && yyjson_is_str(kind)) {
            const luisa::string_view ks{yyjson_get_str(kind), yyjson_get_len(kind)};
            if (ks == "int4_sym") { cfg.linear_storage = DenseLinearStorage::int4_sym; }
            else if (ks == "int4_asym") { cfg.linear_storage = DenseLinearStorage::int4_asym; }
            else if (ks == "fp32") { cfg.linear_storage = DenseLinearStorage::fp32; }
            else { return R::failure(StatusCode::unsupported, "unknown olmOCR-2 linear_storage", "quantization.linear_storage"); }
        }
        auto gs = get_u32(q, "group_size", false, cfg.linear_group_size);
        if (!gs.ok()) { return R::failure(gs.status); }
        cfg.linear_group_size = *gs.value;
    }
    if (cfg.linear_storage != DenseLinearStorage::fp32) { cfg.graph.tie_word_embeddings = false; }
    // Offload policy.
    if (auto *off = yyjson_obj_get(root, "offload"); off != nullptr && yyjson_is_str(off)) {
        const luisa::string_view os{yyjson_get_str(off), yyjson_get_len(off)};
        if (os == "explicit_host") { cfg.offload = OlmOffloadPolicy::explicit_host; }
        else if (os != "none") { return R::failure(StatusCode::unsupported, "unknown olmOCR-2 offload policy", "offload"); }
    }
    if (auto s = cfg.validate(); !s.ok()) { return R::failure(s); }
    return R::success(std::move(cfg));
}

// ---------------------------------------------------------------------------
// Tensor inventory + signature + real-artifact gate
// ---------------------------------------------------------------------------

[[nodiscard]] luisa::vector<ExpectedTensor> olm_ocr2_expected_tensors(const OlmOcr2Config &config) {
    return p7_expected_tensors(config.graph, config.names);
}

[[nodiscard]] Status olm_ocr2_validate_tensors(luisa::span<const PackageTensor> tensors,
                                               luisa::span<const ExpectedTensor> expected,
                                               DenseLinearStorage linear_storage) {
    // S1: only DECODER linears (model. prefix projections + lm_head) are
    // quantizable; vision/merger tensors (visual.*) must stay dense FP32.
    return validate_dense_decoder_tensors_storage(tensors, expected, linear_storage,
                                                  "model.", "lm_head.weight");
}


[[nodiscard]] Status olm_ocr2_validate_packages(const OlmOcr2Config &config,
                                                luisa::span<const PackageTensor> vision_tensors,
                                                luisa::span<const PackageTensor> decoder_tensors) {
    if (auto s = config.validate(); !s.ok()) { return s; }
    const auto expected = olm_ocr2_expected_tensors(config);
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

[[nodiscard]] Status olm_ocr2_validate_signature(const OlmOcr2Config &config, const OlmOcr2Signature &signature) {
    if (signature.checkpoint_revision.empty() || signature.config_fingerprint.empty()) {
        return Status::error(StatusCode::unvalidated,
                             "olmOCR-2 identity requires a checkpoint revision AND config fingerprint; outer model_type alone is never sufficient",
                             "signature");
    }
    if (signature.tensor_names.empty()) {
        return Status::error(StatusCode::unvalidated, "olmOCR-2 identity requires the exact tensor signature", "tensor_names");
    }
    const auto expected = olm_ocr2_expected_tensors(config);
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

[[nodiscard]] Status olm_ocr2_validate_real_artifact(luisa::string_view checkpoint_revision,
                                                     luisa::string_view config_fingerprint) {
    (void)checkpoint_revision;
    (void)config_fingerprint;
    return Status::error(StatusCode::unvalidated,
                         "olmOCR-2 has no qualified official artifact: real checkpoints are refused "
                         "(synthetic recipe-driven path only; see examples/ocr/MODELS.md)",
                         "olm_ocr2");
}

[[nodiscard]] Status olm_ocr2_check_language(luisa::string_view language) {
    if (language.empty() || language == "en") { return Status::success(); }
    return Status::error(StatusCode::unsupported,
                         luisa::format("olmOCR-2 is English-only; language '{}' is not supported (no bilingual claim)", language),
                         "language");
}

// ---------------------------------------------------------------------------
// Strict memory preflight (host; no allocation)
// ---------------------------------------------------------------------------

namespace {

// Workspace region alignment. MUST match runtime/workspace.h kWorkspaceAlignment.
constexpr uint64_t kOlmWorkspaceAlignment = 256u;

[[nodiscard]] Result<uint64_t> f32_bytes(uint64_t count) {
    return tensor_byte_size(TensorDType::f32, count);
}

[[nodiscard]] Result<uint64_t> align_add(uint64_t total, uint64_t bytes) {
    using R = Result<uint64_t>;
    uint64_t aligned = 0, next = 0;
    if (!checked_align_up(bytes, kOlmWorkspaceAlignment, aligned)) {
        return R::failure(StatusCode::out_of_memory, "workspace alignment overflow", "workspace");
    }
    if (!checked_add(total, aligned, next)) {
        return R::failure(StatusCode::out_of_memory, "workspace byte total overflows", "workspace");
    }
    return R::success(next);
}

// Vision + merger weight bytes (always FP32). Mirrors p7_expected_tensors.
[[nodiscard]] Result<uint64_t> vision_weight_bytes(const P7VisionConfig &v, const P7ConnectorConfig &c,
                                                   uint32_t text_hidden) {
    using R = Result<uint64_t>;
    uint64_t total = 0;
    const auto fp32 = [&total](uint64_t elems) -> Status {
        auto b = f32_bytes(elems);
        if (!b.ok()) { return b.status; }
        uint64_t next = 0;
        if (!checked_add(total, *b.value, next)) {
            return Status::error(StatusCode::out_of_memory, "vision weight total overflows", "weights");
        }
        total = next;
        return Status::success();
    };
    const uint64_t vh = v.hidden_size, vi = v.intermediate_size;
    if (auto s = fp32(static_cast<uint64_t>(v.patch_dim()) * vh); !s.ok()) { return R::failure(s); }// patch embed w
    if (auto s = fp32(vh); !s.ok()) { return R::failure(s); }// patch embed b
    if (v.position_rows != 0u) {
        if (auto s = fp32(static_cast<uint64_t>(v.position_rows) * vh); !s.ok()) { return R::failure(s); }
    }
    if (v.rel_pos_window != 0u) {
        const uint64_t side = 2ull * v.rel_pos_window - 1ull;
        for (uint32_t l = 0; l < v.num_layers; l++) {
            if (auto s = fp32(static_cast<uint64_t>(v.num_heads) * side * side); !s.ok()) { return R::failure(s); }
        }
    }
    for (uint32_t l = 0; l < v.num_layers; l++) {
        if (auto s = fp32(vh); !s.ok()) { return R::failure(s); }// norm1 w
        if (auto s = fp32(vh); !s.ok()) { return R::failure(s); }// norm1 b
        if (auto s = fp32(vh); !s.ok()) { return R::failure(s); }// norm2 w
        if (auto s = fp32(vh); !s.ok()) { return R::failure(s); }// norm2 b
        for (int i = 0; i < 4; i++) {
            if (auto s = fp32(static_cast<uint64_t>(vh) * vh); !s.ok()) { return R::failure(s); }// q/k/v/o w
        }
        if (v.qkv_bias) {
            for (int i = 0; i < 4; i++) {
                if (auto s = fp32(vh); !s.ok()) { return R::failure(s); }// q/k/v/o b
            }
        }
        if (v.qk_norm) {
            if (auto s = fp32(v.head_dim); !s.ok()) { return R::failure(s); }
            if (auto s = fp32(v.head_dim); !s.ok()) { return R::failure(s); }
        }
        if (auto s = fp32(static_cast<uint64_t>(vh) * vi); !s.ok()) { return R::failure(s); }// fc1 w
        if (auto s = fp32(vi); !s.ok()) { return R::failure(s); }// fc1 b
        if (auto s = fp32(static_cast<uint64_t>(vi) * vh); !s.ok()) { return R::failure(s); }// fc2 w
        if (auto s = fp32(vh); !s.ok()) { return R::failure(s); }// fc2 b
    }
    // Connector.
    uint64_t cin = vh;
    if (c.kind == P7ConnectorKind::spatial_mlp || c.kind == P7ConnectorKind::downsample_mlp) {
        cin = vh * c.spatial_factor * c.spatial_factor;
    } else if (c.kind == P7ConnectorKind::unshuffle_mlp) {
        cin = vh * 4u;
    }
    if (c.kind == P7ConnectorKind::downsample_mlp) {
        if (auto s = fp32(vh * c.spatial_factor * c.spatial_factor * vh); !s.ok()) { return R::failure(s); }
        if (auto s = fp32(vh); !s.ok()) { return R::failure(s); }
        cin = vh;
    }
    if (auto s = fp32(cin); !s.ok()) { return R::failure(s); }// ln w
    if (!c.rms_norm) {
        if (auto s = fp32(cin); !s.ok()) { return R::failure(s); }// ln b
    }
    if (auto s = fp32(cin * c.intermediate_size); !s.ok()) { return R::failure(s); }// fc1 w
    if (auto s = fp32(c.intermediate_size); !s.ok()) { return R::failure(s); }// fc1 b
    if (auto s = fp32(static_cast<uint64_t>(c.intermediate_size) * text_hidden); !s.ok()) { return R::failure(s); }// fc2 w
    if (auto s = fp32(text_hidden); !s.ok()) { return R::failure(s); }// fc2 b
    return R::success(total);
}

// Decoder weight bytes (FP32 or packed INT4). Mirrors the dots.mocr accounting.
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
    const auto linear = [&](uint64_t k, uint64_t n) -> Status {
        uint64_t elems = 0;
        if (!checked_mul(k, n, elems)) {
            return Status::error(StatusCode::out_of_memory, "linear element count overflows", "weights");
        }
        if (storage == DenseLinearStorage::fp32) { return fp32(elems); }
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
    if (auto s = fp32(static_cast<uint64_t>(d.vocab_size) * hidden); !s.ok()) { return R::failure(s); }// embed
    if (auto s = fp32(hidden); !s.ok()) { return R::failure(s); }// final norm
    if (!tied) {
        if (auto s = linear(hidden, d.vocab_size); !s.ok()) { return R::failure(s); }
    } else {
        if (auto s = fp32(static_cast<uint64_t>(d.vocab_size) * hidden); !s.ok()) { return R::failure(s); }
    }
    for (uint32_t l = 0; l < d.num_layers; l++) {
        if (auto s = fp32(hidden); !s.ok()) { return R::failure(s); }
        if (auto s = fp32(hidden); !s.ok()) { return R::failure(s); }
        if (auto s = linear(hidden, qd); !s.ok()) { return R::failure(s); }
        if (auto s = linear(hidden, kvd); !s.ok()) { return R::failure(s); }
        if (auto s = linear(hidden, kvd); !s.ok()) { return R::failure(s); }
        if (auto s = linear(qd, hidden); !s.ok()) { return R::failure(s); }
        if (auto s = fp32(qd); !s.ok()) { return R::failure(s); }
        if (auto s = fp32(kvd); !s.ok()) { return R::failure(s); }
        if (auto s = fp32(kvd); !s.ok()) { return R::failure(s); }
        if (auto s = linear(hidden, inter); !s.ok()) { return R::failure(s); }
        if (auto s = linear(hidden, inter); !s.ok()) { return R::failure(s); }
        if (auto s = linear(inter, hidden); !s.ok()) { return R::failure(s); }
    }
    return R::success(total);
}

// Combined vision + decoder workspace bytes (mirrors the P7 vision arena and
// the dense decoder arena for the olmOCR-2 contract).
[[nodiscard]] Result<uint64_t> workspace_bytes(const P7SliceConfig &g, uint32_t max_patches) {
    using R = Result<uint64_t>;
    const auto &v = g.vision;
    const auto &c = g.connector;
    const auto &d = g.text;
    const uint64_t maxp = max_patches;
    const uint64_t vh = v.hidden_size, vi = v.intermediate_size;
    const uint64_t text_hidden = d.hidden_size;
    uint64_t max_tokens = maxp, reduced_dim = vh, merged_dim = vh;
    if (c.kind == P7ConnectorKind::spatial_mlp || c.kind == P7ConnectorKind::downsample_mlp) {
        const uint64_t mm = static_cast<uint64_t>(c.spatial_factor) * c.spatial_factor;
        max_tokens = maxp / mm;
        reduced_dim = vh * mm;
        merged_dim = vh * mm;
    } else if (c.kind == P7ConnectorKind::unshuffle_mlp) {
        max_tokens = maxp / 4u;
        reduced_dim = vh * 4u;
    } else if (c.kind == P7ConnectorKind::adaptive_mlp) {
        max_tokens = std::min<uint64_t>(maxp, c.max_visual_tokens);
    }
    if (c.kind == P7ConnectorKind::downsample_mlp) { reduced_dim = vh; }
    if (max_tokens == 0u) { max_tokens = 1u; }
    const uint64_t mask_words_bound = (maxp * maxp + 3u) / 4u + 1u;
    const uint64_t rel_bias_elems = v.rel_pos_window != 0u ? static_cast<uint64_t>(v.num_heads) * maxp * maxp : 1ull;
    // Vision arena regions (element counts; each *4 bytes, aligned).
    const uint64_t vision_elems[] = {
        maxp * v.patch_dim(), maxp * v.patch_dim(), maxp * vh, maxp * vh, maxp * vh, maxp * vh,
        maxp * vh, maxp * vh, maxp * vh, maxp * vh, maxp * vh, maxp * vh, maxp * vi,
        max_tokens * reduced_dim, max_tokens * reduced_dim, max_tokens * c.intermediate_size,
        max_tokens * merged_dim, maxp * text_hidden, maxp * 2u, v.head_dim / 2u,
        mask_words_bound, 1, 1, vh + maxp * vh, vh * maxp, rel_bias_elems,
    };
    uint64_t total = 0;
    for (auto e : vision_elems) {
        auto b = f32_bytes(e);
        if (!b.ok()) { return R::failure(b.status); }
        auto next = align_add(total, *b.value);
        if (!next.ok()) { return R::failure(next.status); }
        total = *next.value;
    }
    // Decoder arena regions.
    const uint64_t chunk = g.prefill_chunk_rows;
    const uint64_t hidden = d.hidden_size, qd = d.q_dim(), kvd = d.kv_dim(), inter = d.intermediate_size;
    const uint64_t decoder_elems[] = {
        chunk * hidden, chunk * hidden, chunk * hidden, chunk * hidden, chunk * qd, chunk * kvd,
        chunk * kvd, chunk * qd, chunk * kvd, chunk * qd, chunk * inter, chunk * inter, chunk * inter,
        d.vocab_size, chunk, chunk * 3u, d.head_dim / 2u, 1, 1, 1, 1, 1,
    };
    for (auto e : decoder_elems) {
        auto b = f32_bytes(e);
        if (!b.ok()) { return R::failure(b.status); }
        auto next = align_add(total, *b.value);
        if (!next.ok()) { return R::failure(next.status); }
        total = *next.value;
    }
    return R::success(total);
}

}// namespace

[[nodiscard]] Result<OlmOcr2MemoryReport> olm_ocr2_preflight(
    const OlmOcr2Config &config, uint32_t max_seq_len, MemoryLimits limits) {
    using R = Result<OlmOcr2MemoryReport>;
    // Host offload fails closed BEFORE any allocation or accounting.
    if (config.offload == OlmOffloadPolicy::explicit_host) {
        return R::failure(StatusCode::unsupported,
                          "olmOCR-2 host offload is not implemented; it fails closed as unsupported with no performance claim",
                          "offload");
    }
    if (auto s = config.validate(); !s.ok()) { return R::failure(s); }
    if (max_seq_len == 0u) {
        return R::failure(StatusCode::invalid_argument, "max_seq_len must be nonzero", "max_seq_len");
    }
    const auto &d = config.graph.text;
    OlmOcr2MemoryReport report;
    report.available_device_bytes = limits.max_device_bytes;
    auto vision = vision_weight_bytes(config.graph.vision, config.graph.connector, d.hidden_size);
    if (!vision.ok()) { return R::failure(vision.status); }
    report.vision_weight_bytes = *vision.value;
    auto decoder = decoder_weight_bytes(d, config.graph.tie_word_embeddings, config.linear_storage, config.linear_group_size);
    if (!decoder.ok()) { return R::failure(decoder.status); }
    report.decoder_weight_bytes = *decoder.value;
    auto ws = workspace_bytes(config.graph, config.graph.max_vision_patches);
    if (!ws.ok()) { return R::failure(ws.status); }
    report.workspace_bytes = *ws.value;
    auto kv = plan_kv_cache_bytes(d.num_layers, 1u, max_seq_len, d.num_kv_heads, d.head_dim, 4u);
    if (!kv.ok()) { return R::failure(kv.status); }
    report.kv_cache_bytes = *kv.value + 16u;
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
                          luisa::format("olmOCR-2 requires {} device bytes (vision {} + decoder {} + workspace {} + KV {}) "
                                        "but only {} bytes are available; the declared 8 GB path needs INT4 and a reduced "
                                        "context/pixel budget, or it is unsupported on this device",
                                        total, report.vision_weight_bytes, report.decoder_weight_bytes,
                                        report.workspace_bytes, report.kv_cache_bytes, limits.max_device_bytes),
                          "max_device_bytes");
    }
    return R::success(std::move(report));
}

// ---------------------------------------------------------------------------
// Host composition
// ---------------------------------------------------------------------------

[[nodiscard]] Result<LayoutDocument> olm_ocr2_compose(const LayoutRunResult &run, LayoutOutputMode mode) {
    using R = Result<LayoutDocument>;
    if (mode != LayoutOutputMode::plain && mode != LayoutOutputMode::markdown) {
        return R::failure(StatusCode::unsupported,
                          "olmOCR-2 supports plain and markdown output only (English-only); spotting/extract are not validated",
                          "mode");
    }
    return layout_compose_strict(run, mode, {});
}

}// namespace luisa::ocr

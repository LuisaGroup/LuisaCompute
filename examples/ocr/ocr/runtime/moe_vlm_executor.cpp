// OCR engine -- DeepSeek-OCR-2 MoE VLM executor (Phase 10).
// Correctness-first FP32 MoE decoder + SAM/downsampler/visual-LM-encoder vision.
// See moe_vlm_executor.h. SYNTHETIC ONLY; real artifacts fail closed in the model
// module. The MoE reference loop runs over ACTIVE experts (host reads counts).
#include "ocr/runtime/moe_vlm_executor.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <utility>

#include <luisa/core/logging.h>
#include <luisa/core/stl/format.h>
#include <luisa/runtime/stream.h>

namespace luisa::ocr {

using namespace luisa::compute;

namespace {

[[nodiscard]] Result<uint64_t> f32_bytes(uint64_t count) {
    return tensor_byte_size(TensorDType::f32, count);
}

[[nodiscard]] Status add_region(DeviceWorkspace &ws, WorkspaceAllocation &out, uint64_t elements) {
    auto bytes = f32_bytes(elements);
    if (!bytes.ok()) { return bytes.status; }
    auto alloc = ws.allocate(*bytes.value);
    if (!alloc.ok()) { return alloc.status; }
    out = *alloc.value;
    return Status::success();
}

// Checked KV-cache bytes for this decoder at `max_seq_len` (K+V for every layer
// plus the 16-byte batch-1 DecodeState metadata). Mirrors the model preflight so
// the end-to-end budget check and the plan agree exactly.
[[nodiscard]] Result<uint64_t> moe_kv_bytes(const DenseDecoderConfig &d, uint32_t max_seq_len) {
    using R = Result<uint64_t>;
    auto kv = plan_kv_cache_bytes(d.num_layers, 1u, max_seq_len, d.num_kv_heads, d.head_dim, 4u);
    if (!kv.ok()) { return R::failure(kv.status); }
    uint64_t total = 0;
    if (!checked_add(*kv.value, 16ull, total)) {
        return R::failure(StatusCode::out_of_memory, "KV cache byte count overflows", "kv_cache");
    }
    return R::success(total);
}

// Host bilinear resize + normalize matching the Phase 1 convention (half-pixel
// centers, clamp edges), identical to p7_oracle_preprocess. Kept local so the
// executor TU does not pull the host-only preprocess.h -> api.h chain.
[[nodiscard]] luisa::vector<float> preprocess_chw(luisa::span<const uint8_t> pixels, uint32_t sw, uint32_t sh,
                                                  uint32_t dw, uint32_t dh, luisa::span<const float> mean,
                                                  luisa::span<const float> std_dev) {
    luisa::vector<float> out(static_cast<size_t>(3) * dw * dh);
    const float scale_x = static_cast<float>(sw) / static_cast<float>(dw);
    const float scale_y = static_cast<float>(sh) / static_cast<float>(dh);
    for (uint32_t y = 0; y < dh; y++) {
        for (uint32_t x = 0; x < dw; x++) {
            const float sx = (static_cast<float>(x) + 0.5f) * scale_x - 0.5f;
            const float sy = (static_cast<float>(y) + 0.5f) * scale_y - 0.5f;
            const float x0f = std::floor(sx), y0f = std::floor(sy);
            const int32_t x0 = static_cast<int32_t>(x0f), y0 = static_cast<int32_t>(y0f);
            const float fx = sx - x0f, fy = sy - y0f;
            for (uint32_t c = 0; c < 3u; c++) {
                const auto at = [&](int32_t px, int32_t py) -> float {
                    const int32_t cx = std::clamp(px, 0, static_cast<int32_t>(sw) - 1);
                    const int32_t cy = std::clamp(py, 0, static_cast<int32_t>(sh) - 1);
                    return static_cast<float>(pixels[(static_cast<size_t>(cy) * sw + cx) * 3u + c]);
                };
                const float v00 = at(x0, y0), v10 = at(x0 + 1, y0);
                const float v01 = at(x0, y0 + 1), v11 = at(x0 + 1, y0 + 1);
                const float top = v00 + (v10 - v00) * fx;
                const float bottom = v01 + (v11 - v01) * fy;
                const float v = (top + (bottom - top) * fy) / 255.0f;
                out[static_cast<size_t>(c) * dw * dh + static_cast<size_t>(y) * dw + x] = (v - mean[c]) / std_dev[c];
            }
        }
    }
    return out;
}

// Per-record heap-owned upload staging (retained through completion). Buffer
// copy commands borrow host pointers until EXECUTION, so every uploaded host
// array (pixels, positions, axis maps, per-layer attention masks) lives here
// and is released only by a completion callback.
struct RecordUploadStaging {
    luisa::vector<int32_t> ids;
    luisa::vector<int32_t> positions;
    luisa::vector<uint32_t> axis;
    luisa::vector<float> pixels;
    luisa::vector<luisa::vector<uint32_t>> masks;
};

// S4: keep upload staging alive until command completion on EVERY exit path.
//
// Two ownership modes:
// * CALLER-OWNED list (record_vision): attach the completion callback to the
//   caller's list, including on an error return after an upload was recorded, so
//   the caller may still commit it safely.
// * EXECUTOR-OWNED list (prefill / decode_step): the caller never sees this list,
//   so on an error return the guard COMMITS AND DRAINS it. That both releases the
//   staging through the callback and leaves the list committed, which the
//   runtime requires before destruction (a non-empty uncommitted list aborts).
struct UploadStagingGuard {
    CommandList *list;
    luisa::unique_ptr<RecordUploadStaging> staging;
    Stream *drain_stream{nullptr};// non-null => executor-owned list
    ~UploadStagingGuard() {
        if (!staging) { return; }
        list->add_callback([s = std::move(staging)]() mutable { s.reset(); });
        if (drain_stream != nullptr && !list->empty()) {
            *drain_stream << list->commit() << synchronize();
        }
    }
};

}// namespace

// ---------------------------------------------------------------------------
// Options
// ---------------------------------------------------------------------------

[[nodiscard]] Status MoeDecodeOptions::validate(uint32_t vocab_size) const {
    if (max_new_tokens == 0u) {
        return Status::error(StatusCode::invalid_argument, "max_new_tokens must be nonzero", "max_new_tokens");
    }
    auto s = validate_decode_sampling(sampling, vocab_size);
    if (!s.ok()) { return s; }
    for (auto id : eos_ids) {
        if (id >= vocab_size) {
            return Status::error(StatusCode::invalid_argument, "EOS id outside the vocabulary", "eos_ids");
        }
    }
    for (const auto &stop : stop_strings) {
        if (stop.empty()) {
            return Status::error(StatusCode::invalid_argument, "stop strings must not be empty", "stop_strings");
        }
    }
    return Status::success();
}

// ---------------------------------------------------------------------------
// create / resolve / reserve / compile
// ---------------------------------------------------------------------------

[[nodiscard]] Status MoeVlmExecutor::create(Device &device, Stream &stream,
                                            const DeepSeekOcr2Config &config, const WeightStore &weights) {
    auto valid = config.validate();
    if (!valid.ok()) { return valid; }
    _config = config;
    _device = &device;
    _workspace = DeviceWorkspace{device};
    _patchify_cache = TypedShaderCache<ImagePatchifyKernel>{device, _shader_registry};
    _layer_norm_cache = TypedShaderCache<LayerNormKernel>{device, _shader_registry};
    _rms_norm_cache = TypedShaderCache<RmsNormKernel>{device, _shader_registry};
    _gemm_cache = TypedShaderCache<GemmKernel>{device, _shader_registry};
    _gemv_cache = TypedShaderCache<GemvKernel>{device, _shader_registry};
    _gated_act_cache = TypedShaderCache<GatedActKernel>{device, _shader_registry};
    _rope2d_cache = TypedShaderCache<RopeAxialKernel>{device, _shader_registry};
    _rope_rh_cache = TypedShaderCache<RopeAxialKernel>{device, _shader_registry};
    _attn_prefill_cache = TypedShaderCache<AttentionPrefillKernel>{device, _shader_registry};
    _attn_decode_cache = TypedShaderCache<AttentionDecodeKernel>{device, _shader_registry};
    _kv_append_cache = TypedShaderCache<KvAppendKernel>{device, _shader_registry};
    _embedding_cache = TypedShaderCache<EmbeddingLookupKernel>{device, _shader_registry};
    _mm_scatter_cache = TypedShaderCache<MmScatterKernel>{device, _shader_registry};
    _conv_cache = TypedShaderCache<Conv2dKernel>{device, _shader_registry};
    _transpose_cache = TypedShaderCache<Transpose2dKernel>{device, _shader_registry};
    _merge_cache = TypedShaderCache<SpatialMergeKernel>{device, _shader_registry};
    _moe_softmax_cache = TypedShaderCache<MoeRouterSoftmaxKernel>{device, _shader_registry};
    _moe_topk_cache = TypedShaderCache<MoeTopKKernel>{device, _shader_registry};
    _moe_hist_cache = TypedShaderCache<MoeHistogramKernel>{device, _shader_registry};
    _moe_offsets_cache = TypedShaderCache<MoeOffsetsKernel>{device, _shader_registry};
    _moe_perm_cache = TypedShaderCache<MoePermutationKernel>{device, _shader_registry};
    _moe_gather_cache = TypedShaderCache<MoeGatherKernel>{device, _shader_registry};
    _moe_scatter_cache = TypedShaderCache<MoeScatterKernel>{device, _shader_registry};
    _residual_cache = TypedShaderCache<ResidualAddKernel>{device, _shader_registry};
    (void)stream;
    if (auto s = resolve_weights(weights); !s.ok()) { return s; }
    if (auto s = reserve_workspace(); !s.ok()) { return s; }
    // S5: preflight the FULL footprint (borrowed weights + the exact workspace
    // arena + the KV cache at the configured max sequence) BEFORE any cache is
    // allocated, so an over-budget configuration fails with required-vs-available.
    {
        auto kv_bytes = moe_kv_bytes(_config.decoder.decoder, _config.max_seq_len);
        if (!kv_bytes.ok()) { return kv_bytes.status; }
        MemoryBudget total{_config.memory};
        if (auto s = total.require("moe_weights", MemoryCategory::weights, _weight_bytes); !s.ok()) { return s; }
        if (auto s = total.require("moe_workspace", MemoryCategory::workspace, _workspace.capacity_bytes()); !s.ok()) { return s; }
        if (auto s = total.require("moe_kv_cache", MemoryCategory::kv_cache, *kv_bytes.value); !s.ok()) { return s; }
        if (auto s = total.preflight(); !s.ok()) { return s; }
    }
    return compile_shaders();
}

[[nodiscard]] Status MoeVlmExecutor::resolve_weights(const WeightStore &weights) {
    const auto &vcomp = _config.vision_names.perception.visual_prefix;// component tag "vision"
    (void)vcomp;
    const luisa::string vision_comp = "vision";
    const luisa::string decoder_comp = _config.decoder.component;
    const auto &vn = _config.vision_names;
    const auto &dn = _config.decoder_names;
    const auto &v = _config.vision;
    const auto &d = _config.decoder.decoder;

    auto vision_expected = deepseek_ocr2_vision_expected_tensors(_config);
    auto decoder_expected = deepseek_ocr2_decoder_expected_tensors(_config);
    luisa::unordered_map<luisa::string, const ExpectedTensor *> vmap, dmap;
    for (const auto &e : vision_expected) { vmap.emplace(e.name, &e); }
    for (const auto &e : decoder_expected) { dmap.emplace(e.name, &e); }
    for (const auto &name : weights.tensor_names(vision_comp)) {
        if (vmap.find(name) == vmap.end()) {
            return Status::error(StatusCode::invalid_argument,
                                 luisa::format("unexpected vision tensor '{}' in component '{}'", name, vision_comp), name);
        }
    }
    for (const auto &name : weights.tensor_names(decoder_comp)) {
        if (dmap.find(name) == dmap.end()) {
            return Status::error(StatusCode::invalid_argument,
                                 luisa::format("unexpected decoder tensor '{}' in component '{}'", name, decoder_comp), name);
        }
    }
    auto view_of = [&](const luisa::string &comp, const luisa::unordered_map<luisa::string, const ExpectedTensor *> &map,
                       const luisa::string &name, BufferView<float> &out) -> Status {
        const auto *w = weights.find(comp, name);
        if (w == nullptr) {
            return Status::error(StatusCode::not_found,
                                 luisa::format("weight '{}' not found in component '{}'", name, comp), name);
        }
        if (w->encoding != TensorEncoding::fp32 || !w->dense) {
            return Status::error(StatusCode::unsupported,
                                 luisa::format("weight '{}' must be dense fp32 in the Phase 10 slice", name), name);
        }
        const auto it = map.find(name);
        if (it == map.end()) {
            return Status::error(StatusCode::invalid_argument,
                                 luisa::format("weight '{}' has no expected-tensor contract", name), name);
        }
        if (auto s = validate_decoder_weight_desc(w->logical, w->element_count, *it->second); !s.ok()) { return s; }
        if (w->dense.size() < w->element_count) {
            return Status::error(StatusCode::invalid_argument,
                                 luisa::format("weight '{}' device buffer smaller than element count", name), name);
        }
        out = w->dense.view();
        return Status::success();
    };
    const auto vv = [&](const luisa::string &name, BufferView<float> &out) { return view_of(vision_comp, vmap, name, out); };
    const auto dv = [&](const luisa::string &name, BufferView<float> &out) { return view_of(decoder_comp, dmap, name, out); };

    // Perception (P7 names).
    if (auto s = vv(vn.perception.patch_embed_weight(), _patch_embed_w); !s.ok()) { return s; }
    if (auto s = vv(vn.perception.patch_embed_bias(), _patch_embed_b); !s.ok()) { return s; }
    _perception.resize(v.perception.num_layers);
    for (uint32_t l = 0; l < v.perception.num_layers; l++) {
        auto &B = _perception[l];
        if (auto s = vv(vn.perception.block_norm1_weight(l), B.norm1_w); !s.ok()) { return s; }
        if (auto s = vv(vn.perception.block_norm1_bias(l), B.norm1_b); !s.ok()) { return s; }
        if (auto s = vv(vn.perception.block_norm2_weight(l), B.norm2_w); !s.ok()) { return s; }
        if (auto s = vv(vn.perception.block_norm2_bias(l), B.norm2_b); !s.ok()) { return s; }
        if (auto s = vv(vn.perception.block_q_proj(l), B.q_w); !s.ok()) { return s; }
        if (auto s = vv(vn.perception.block_k_proj(l), B.k_w); !s.ok()) { return s; }
        if (auto s = vv(vn.perception.block_v_proj(l), B.v_w); !s.ok()) { return s; }
        if (auto s = vv(vn.perception.block_o_proj(l), B.o_w); !s.ok()) { return s; }
        if (auto s = vv(vn.perception.block_q_bias(l), B.q_b); !s.ok()) { return s; }
        if (auto s = vv(vn.perception.block_k_bias(l), B.k_b); !s.ok()) { return s; }
        if (auto s = vv(vn.perception.block_v_bias(l), B.v_b); !s.ok()) { return s; }
        if (auto s = vv(vn.perception.block_o_bias(l), B.o_b); !s.ok()) { return s; }
        if (auto s = vv(vn.perception.block_fc1(l), B.fc1_w); !s.ok()) { return s; }
        if (auto s = vv(vn.perception.block_fc1_bias(l), B.fc1_b); !s.ok()) { return s; }
        if (auto s = vv(vn.perception.block_fc2(l), B.fc2_w); !s.ok()) { return s; }
        if (auto s = vv(vn.perception.block_fc2_bias(l), B.fc2_b); !s.ok()) { return s; }
    }
    // BOTH conv downsamplers.
    if (auto s = vv(vn.down_weight(1), _down1.weight); !s.ok()) { return s; }
    if (auto s = vv(vn.down_bias(1), _down1.bias); !s.ok()) { return s; }
    if (auto s = vv(vn.down_weight(2), _down2.weight); !s.ok()) { return s; }
    if (auto s = vv(vn.down_bias(2), _down2.bias); !s.ok()) { return s; }
    // Visual LM encoder.
    _encoder.resize(v.encoder_layers);
    for (uint32_t l = 0; l < v.encoder_layers; l++) {
        auto &B = _encoder[l];
        if (auto s = vv(vn.encoder_norm1_weight(l), B.norm1_w); !s.ok()) { return s; }
        if (auto s = vv(vn.encoder_norm1_bias(l), B.norm1_b); !s.ok()) { return s; }
        if (auto s = vv(vn.encoder_norm2_weight(l), B.norm2_w); !s.ok()) { return s; }
        if (auto s = vv(vn.encoder_norm2_bias(l), B.norm2_b); !s.ok()) { return s; }
        if (auto s = vv(vn.encoder_q_proj(l), B.q_w); !s.ok()) { return s; }
        if (auto s = vv(vn.encoder_k_proj(l), B.k_w); !s.ok()) { return s; }
        if (auto s = vv(vn.encoder_v_proj(l), B.v_w); !s.ok()) { return s; }
        if (auto s = vv(vn.encoder_o_proj(l), B.o_w); !s.ok()) { return s; }
        if (auto s = vv(vn.encoder_q_bias(l), B.q_b); !s.ok()) { return s; }
        if (auto s = vv(vn.encoder_k_bias(l), B.k_b); !s.ok()) { return s; }
        if (auto s = vv(vn.encoder_v_bias(l), B.v_b); !s.ok()) { return s; }
        if (auto s = vv(vn.encoder_o_bias(l), B.o_b); !s.ok()) { return s; }
        if (auto s = vv(vn.encoder_fc1(l), B.fc1_w); !s.ok()) { return s; }
        if (auto s = vv(vn.encoder_fc1_bias(l), B.fc1_b); !s.ok()) { return s; }
        if (auto s = vv(vn.encoder_fc2(l), B.fc2_w); !s.ok()) { return s; }
        if (auto s = vv(vn.encoder_fc2_bias(l), B.fc2_b); !s.ok()) { return s; }
    }
    if (auto s = vv(vn.query_embed(), _query_embed); !s.ok()) { return s; }
    if (auto s = vv(vn.connector_ln_weight(), _conn_ln_w); !s.ok()) { return s; }
    if (auto s = vv(vn.connector_ln_bias(), _conn_ln_b); !s.ok()) { return s; }
    if (auto s = vv(vn.connector_fc1(), _conn_fc1_w); !s.ok()) { return s; }
    if (auto s = vv(vn.connector_fc1_bias(), _conn_fc1_b); !s.ok()) { return s; }
    if (auto s = vv(vn.connector_fc2(), _conn_fc2_w); !s.ok()) { return s; }
    if (auto s = vv(vn.connector_fc2_bias(), _conn_fc2_b); !s.ok()) { return s; }
    // Decoder.
    if (auto s = dv(dn.dense.embed_tokens(), _embed); !s.ok()) { return s; }
    if (auto s = dv(dn.dense.final_norm(), _final_norm); !s.ok()) { return s; }
    if (auto s = dv(dn.dense.lm_head(), _lm_head); !s.ok()) { return s; }
    _layers.resize(d.num_layers);
    for (uint32_t l = 0; l < d.num_layers; l++) {
        auto &L = _layers[l];
        if (auto s = dv(dn.dense.input_layernorm(l), L.input_norm); !s.ok()) { return s; }
        if (auto s = dv(dn.dense.post_attention_layernorm(l), L.post_norm); !s.ok()) { return s; }
        if (auto s = dv(dn.dense.q_proj(l), L.q_w); !s.ok()) { return s; }
        if (auto s = dv(dn.dense.k_proj(l), L.k_w); !s.ok()) { return s; }
        if (auto s = dv(dn.dense.v_proj(l), L.v_w); !s.ok()) { return s; }
        if (auto s = dv(dn.dense.o_proj(l), L.o_w); !s.ok()) { return s; }
        if (auto s = dv(dn.dense.q_bias(l), L.q_b); !s.ok()) { return s; }
        if (auto s = dv(dn.dense.k_bias(l), L.k_b); !s.ok()) { return s; }
        if (auto s = dv(dn.dense.v_bias(l), L.v_b); !s.ok()) { return s; }
        if (auto s = dv(dn.router_weight(l), L.router_w); !s.ok()) { return s; }
        L.expert_gate.resize(_config.decoder.num_experts);
        L.expert_up.resize(_config.decoder.num_experts);
        L.expert_down.resize(_config.decoder.num_experts);
        for (uint32_t e = 0; e < _config.decoder.num_experts; e++) {
            if (auto s = dv(dn.expert_gate(l, e), L.expert_gate[e]); !s.ok()) { return s; }
            if (auto s = dv(dn.expert_up(l, e), L.expert_up[e]); !s.ok()) { return s; }
            if (auto s = dv(dn.expert_down(l, e), L.expert_down[e]); !s.ok()) { return s; }
        }
        if (auto s = dv(dn.shared_gate(l, 0), L.shared0_gate); !s.ok()) { return s; }
        if (auto s = dv(dn.shared_up(l, 0), L.shared0_up); !s.ok()) { return s; }
        if (auto s = dv(dn.shared_down(l, 0), L.shared0_down); !s.ok()) { return s; }
        if (auto s = dv(dn.shared_gate(l, 1), L.shared1_gate); !s.ok()) { return s; }
        if (auto s = dv(dn.shared_up(l, 1), L.shared1_up); !s.ok()) { return s; }
        if (auto s = dv(dn.shared_down(l, 1), L.shared1_down); !s.ok()) { return s; }
    }
    _weight_bytes = weights.component_bytes(vision_comp) + weights.component_bytes(decoder_comp);
    return Status::success();
}

[[nodiscard]] Status MoeVlmExecutor::reserve_workspace() {
    const auto &v = _config.vision;
    const auto &d = _config.decoder.decoder;
    const uint64_t maxp = v.max_vision_patches;
    const uint64_t vh = v.perception.hidden_size, vi = v.perception.intermediate_size;
    const uint64_t patch_dim = v.perception.patch_dim();
    const uint64_t eh = v.encoder_hidden(), ei = v.encoder_intermediate_size;
    const uint64_t nq = v.num_queries;
    const uint64_t enc_rows_max = maxp + nq;// downsamplers only reduce
    const uint64_t th = d.hidden_size;
    const uint64_t mr = _config.decoder.prefill_chunk_rows;
    const uint64_t qd = d.q_dim(), kvd = d.kv_dim(), vocab = d.vocab_size;
    const uint64_t top_k = _config.decoder.top_k;
    const uint64_t slots_max = mr * top_k;
    const uint64_t mi = _config.decoder.moe_intermediate_size;
    const uint64_t si = _config.decoder.shared_intermediate_size;
    const uint64_t ne = _config.decoder.num_experts;
    // The shared experts reuse the gate/up/act scratch AFTER the routed experts;
    // size those regions for the larger of (slots*mi) and (rows*si).
    const uint64_t act_max = std::max(slots_max * mi, mr * si);
    const uint64_t max_channels = std::max({vh, static_cast<uint64_t>(v.down1.out_channels),
                                            static_cast<uint64_t>(v.down2.out_channels)});
    // Custom-mask word bound (block_size 1 => enc_rows^2 bytes => /4 words).
    const uint64_t mask_words_max = (enc_rows_max * enc_rows_max + 3ull) / 4ull + 1ull;

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
    // EXACT shared arena plan (identical regions/order/sizes to the host memory
    // preflight), so the reserved capacity EQUALS the preflighted workspace bytes.
    auto plan_result = deepseek_ocr2_workspace_plan(_config);
    if (!plan_result.ok()) { return plan_result.status; }
    const auto &plan = *plan_result.value;
    const uint32_t nlayers = d.num_layers;
    MemoryBudget budget{_config.memory};
    if (auto s = budget.require("moe_workspace", MemoryCategory::workspace, plan.total_bytes()); !s.ok()) { return s; }
    if (auto s = budget.preflight(); !s.ok()) { return s; }
    if (auto s = _workspace.reserve(plan.total_bytes()); !s.ok()) { return s; }

    WorkspaceAllocation *outs[] = {
        &_pixels, &_patches, &_px, &_pnorm, &_pq, &_pk, &_pv, &_pqr, &_pkr, &_pattn, &_pfc1, &_paux,
        &_conv_a, &_conv_b, &_conv_tokens, &_ex, &_enorm, &_eq, &_ek, &_ev, &_eattn, &_efc1, &_eaux,
        &_enc_mask, &_enc_pos, &_enc_axis, &_p_pos, &_conn_norm, &_conn_fc1, &_features,
        &_emb, &_x, &_norm, &_norm2, &_q, &_k, &_v, &_q_rope, &_k_rope, &_attn, &_logits, &_ids_tok,
        &_positions, &_axis, &_dummy_f, &_dummy_u,
    };
    for (size_t i = 0; i < std::size(specs); i++) {
        if (auto s = add_region(_workspace, *outs[i], specs[i].elements); !s.ok()) { return s; }
    }
    _moe.resize(nlayers);
    _layer_counts.assign(nlayers, luisa::vector<uint32_t>(static_cast<size_t>(ne), 0u));
    _layer_offsets.assign(nlayers, luisa::vector<uint32_t>(static_cast<size_t>(ne), 0u));
    for (uint32_t l = 0; l < nlayers; l++) {
        auto &M = _moe[l];
        WorkspaceAllocation *mouts[] = {
            &M.router_logits, &M.probs, &M.ids, &M.weights, &M.counts, &M.offsets,
            &M.perm_slot, &M.slot_pos, &M.perm_token, &M.gathered, &M.gate, &M.up, &M.act,
            &M.expert_out, &M.shared0, &M.shared1, &M.moe_out,
        };
        const uint64_t melems[] = {
            mr * ne, mr * ne, slots_max, slots_max, ne, ne, slots_max, slots_max, slots_max,
            slots_max * th, act_max, act_max, act_max, slots_max * th,
            mr * th, mr * th, mr * th,
        };
        for (size_t i = 0; i < std::size(mouts); i++) {
            if (auto s = add_region(_workspace, *mouts[i], melems[i]); !s.ok()) { return s; }
        }
    }
    _enc_attn_out.resize(v.encoder_layers);
    for (uint32_t l = 0; l < v.encoder_layers; l++) {
        if (auto s = add_region(_workspace, _enc_attn_out[l], enc_rows_max * eh); !s.ok()) { return s; }
    }
    // 1-axis rotate-half RoPE axis map (all zeros) for the text decoder.
    _axis_pending.assign(static_cast<size_t>(d.head_dim / 2u), 0u);
    return Status::success();
}

[[nodiscard]] Status MoeVlmExecutor::compile_shaders() {
    const auto get = [](auto &cache, const luisa::string &key, auto factory, auto *&out) -> Status {
        auto r = cache.get_or_compile(key, factory);
        if (!r.ok()) { return r.status; }
        out = *r.value;
        return Status::success();
    };
    if (auto s = get(_patchify_cache, image_variant_key("patchify"), [] { return make_image_patchify_kernel(); }, _sh_patchify); !s.ok()) { return s; }
    if (auto s = get(_layer_norm_cache, norm_variant_key("layer_norm"), [] { return make_layer_norm_kernel(); }, _sh_layer_norm); !s.ok()) { return s; }
    if (auto s = get(_rms_norm_cache, norm_variant_key("rms_norm"), [] { return make_rms_norm_kernel(); }, _sh_rms_norm); !s.ok()) { return s; }
    if (auto s = get(_gemm_cache, gemm_variant_key(GemmEpilogue::none, GemmWeightFormat::fp32, false), [] { return make_gemm_kernel(GemmEpilogue::none); }, _sh_gemm_none); !s.ok()) { return s; }
    if (auto s = get(_gemm_cache, gemm_variant_key(GemmEpilogue::bias, GemmWeightFormat::fp32, false), [] { return make_gemm_kernel(GemmEpilogue::bias); }, _sh_gemm_bias); !s.ok()) { return s; }
    if (auto s = get(_gemm_cache, gemm_variant_key(GemmEpilogue::residual, GemmWeightFormat::fp32, false), [] { return make_gemm_kernel(GemmEpilogue::residual); }, _sh_gemm_residual); !s.ok()) { return s; }
    if (auto s = get(_gemm_cache, gemm_variant_key(GemmEpilogue::bias_residual, GemmWeightFormat::fp32, false), [] { return make_gemm_kernel(GemmEpilogue::bias_residual); }, _sh_gemm_bias_residual); !s.ok()) { return s; }
    if (auto s = get(_gemm_cache, gemm_variant_key(GemmEpilogue::bias_gelu_erf, GemmWeightFormat::fp32, false), [] { return make_gemm_kernel(GemmEpilogue::bias_gelu_erf); }, _sh_gemm_bias_gelu); !s.ok()) { return s; }
    if (auto s = get(_gemv_cache, gemm_variant_key(GemmEpilogue::none, GemmWeightFormat::fp32, true), [] { return make_gemv_kernel(GemmEpilogue::none); }, _sh_gemv_none); !s.ok()) { return s; }
    if (auto s = get(_gemv_cache, gemm_variant_key(GemmEpilogue::bias, GemmWeightFormat::fp32, true), [] { return make_gemv_kernel(GemmEpilogue::bias); }, _sh_gemv_bias); !s.ok()) { return s; }
    if (auto s = get(_gemv_cache, gemm_variant_key(GemmEpilogue::residual, GemmWeightFormat::fp32, true), [] { return make_gemv_kernel(GemmEpilogue::residual); }, _sh_gemv_residual); !s.ok()) { return s; }
    if (auto s = get(_gated_act_cache, gemm_variant_key(GemmEpilogue::none, GemmWeightFormat::fp32, false) + "_silu", [] { return make_gated_act_kernel(0u); }, _sh_gated_silu); !s.ok()) { return s; }
    if (auto s = get(_rope2d_cache, rope_variant_key("2d"), [] { return make_rope_2d_kernel(); }, _sh_rope2d); !s.ok()) { return s; }
    if (auto s = get(_rope_rh_cache, rope_variant_key("rotate_half"), [] { return make_rope_rotate_half_kernel(); }, _sh_rope_rh); !s.ok()) { return s; }
    if (auto s = get(_attn_prefill_cache, attention_variant_key("prefill_reference"), [] { return make_attention_prefill_reference_kernel(); }, _sh_attn_prefill); !s.ok()) { return s; }
    if (auto s = get(_attn_decode_cache, attention_variant_key("decode"), [] { return make_attention_decode_kernel(); }, _sh_attn_decode); !s.ok()) { return s; }
    if (auto s = get(_kv_append_cache, decode_variant_key("kv_append"), [] { return make_kv_append_kernel(); }, _sh_kv_append); !s.ok()) { return s; }
    if (auto s = get(_embedding_cache, decode_variant_key("embedding_lookup"), [] { return make_embedding_lookup_kernel(); }, _sh_embedding); !s.ok()) { return s; }
    if (auto s = get(_mm_scatter_cache, decode_variant_key("mm_scatter"), [] { return make_mm_scatter_kernel(); }, _sh_mm_scatter); !s.ok()) { return s; }
    if (auto s = get(_conv_cache, conv_variant_key("conv2d"), [] { return make_conv2d_kernel(); }, _sh_conv); !s.ok()) { return s; }
    if (auto s = get(_transpose_cache, vision_variant_key("transpose_2d"), [] { return make_transpose_2d_kernel(); }, _sh_transpose); !s.ok()) { return s; }
    if (auto s = get(_merge_cache, vision_variant_key("spatial_merge"), [] { return make_spatial_merge_kernel(); }, _sh_merge); !s.ok()) { return s; }
    if (auto s = get(_moe_softmax_cache, moe_variant_key("router_softmax"), [] { return make_moe_router_softmax_kernel(); }, _sh_moe_softmax); !s.ok()) { return s; }
    if (auto s = get(_moe_topk_cache, moe_variant_key("topk"), [] { return make_moe_topk_kernel(); }, _sh_moe_topk); !s.ok()) { return s; }
    if (auto s = get(_moe_hist_cache, moe_variant_key("histogram"), [] { return make_moe_histogram_kernel(); }, _sh_moe_hist); !s.ok()) { return s; }
    if (auto s = get(_moe_offsets_cache, moe_variant_key("offsets"), [] { return make_moe_offsets_kernel(); }, _sh_moe_offsets); !s.ok()) { return s; }
    if (auto s = get(_moe_perm_cache, moe_variant_key("permutation"), [] { return make_moe_permutation_kernel(); }, _sh_moe_perm); !s.ok()) { return s; }
    if (auto s = get(_moe_gather_cache, moe_variant_key("gather"), [] { return make_moe_gather_kernel(); }, _sh_moe_gather); !s.ok()) { return s; }
    if (auto s = get(_moe_scatter_cache, moe_variant_key("scatter"), [] { return make_moe_scatter_kernel(); }, _sh_moe_scatter); !s.ok()) { return s; }
    if (auto s = get(_residual_cache, norm_variant_key("residual_add"), [] { return make_residual_add_kernel(); }, _sh_residual); !s.ok()) { return s; }
    return Status::success();
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

[[nodiscard]] Status MoeVlmExecutor::flush(CommandList &list, Stream &stream) {
    // commit() moves the list into the Commit (leaving it moved-from); clear()
    // resets it for reuse (CommandList move-assignment is deleted).
    stream << list.commit() << synchronize();
    list.clear();
    return Status::success();
}

[[nodiscard]] Status MoeVlmExecutor::copy_rows(CommandList &list, BufferView<float> src,
                                               BufferView<float> dst, uint32_t rows, uint32_t hidden) {
    auto plan = plan_spatial_merge_dispatch(rows, hidden);
    if (!plan.ok()) { return plan.status; }
    if (auto cmd = dispatch_1d((*_sh_merge)(src, dst, rows, hidden, hidden, 1u), *plan.value)) {
        list << std::move(cmd);
    }
    return Status::success();
}

[[nodiscard]] Status MoeVlmExecutor::record_linear(CommandList &list, BufferView<float> A,
                                                   BufferView<float> B, BufferView<float> C,
                                                   BufferView<float> aux, uint32_t m, uint32_t n, uint32_t k,
                                                   GemmEpilogue epilogue) {
    if (m == 0u || n == 0u || k == 0u) {
        return Status::error(StatusCode::invalid_argument, "linear dimensions must be nonzero", "m/n/k");
    }
    const auto aux_view = epilogue == GemmEpilogue::none ? ws(_dummy_f) : aux;
    // The GEMV/small-M path only implements the none/bias/residual epilogues.
    // The gated-activation epilogues (bias_gelu_erf, used by the connector and
    // vision MLPs) MUST use the GEMM path even for small M, otherwise the bias
    // and activation would be silently dropped. Correctness over speed here.
    const bool gemv_ok = epilogue == GemmEpilogue::none || epilogue == GemmEpilogue::bias ||
                         epilogue == GemmEpilogue::residual;
    if (m <= kGemvSmallM && gemv_ok) {
        auto plan = plan_gemv_dispatch(m, n, k);
        if (!plan.ok()) { return plan.status; }
        const auto *shader = epilogue == GemmEpilogue::bias ? _sh_gemv_bias
                             : epilogue == GemmEpilogue::residual ? _sh_gemv_residual
                                                                  : _sh_gemv_none;
        if (auto cmd = dispatch_1d((*shader)(A, B, C, aux_view, m, n, k), *plan.value)) {
            list << std::move(cmd);
        }
        return Status::success();
    }
    auto plan = plan_gemm_dispatch_epilogue(m, n, k, epilogue);
    if (!plan.ok()) { return plan.status; }
    const auto *shader = epilogue == GemmEpilogue::bias ? _sh_gemm_bias
                         : epilogue == GemmEpilogue::residual ? _sh_gemm_residual
                         : epilogue == GemmEpilogue::bias_residual ? _sh_gemm_bias_residual
                         : epilogue == GemmEpilogue::bias_gelu_erf ? _sh_gemm_bias_gelu
                                                                   : _sh_gemm_none;
    if (auto cmd = dispatch_2d((*shader)(A, B, C, aux_view, m, n, k, plan.value->padded_cols), *plan.value)) {
        list << std::move(cmd);
    }
    return Status::success();
}

// ---------------------------------------------------------------------------
// Vision: SAM perception block (windowed/global, 2D RoPE, GELU MLP)
// ---------------------------------------------------------------------------

[[nodiscard]] Status MoeVlmExecutor::record_perception_block(CommandList &list, uint32_t layer,
                                                             BufferView<float> x, uint32_t rows,
                                                             const P7PackedImage &packed,
                                                             luisa::vector<luisa::vector<uint32_t>> &mask_staging) {
    const auto &v = _config.vision.perception;
    const auto &B = _perception[layer];
    const uint32_t vh = v.hidden_size, vi = v.intermediate_size;
    const uint32_t heads = v.num_heads, hd = v.head_dim;
    const float eps = v.layer_norm_eps, theta = v.rope_theta;
    const float scale = 1.0f / std::sqrt(static_cast<float>(hd));
    auto norm = ws(_pnorm).subview(0, static_cast<size_t>(rows) * vh);
    auto q = ws(_pq).subview(0, static_cast<size_t>(rows) * vh);
    auto k = ws(_pk).subview(0, static_cast<size_t>(rows) * vh);
    auto vv = ws(_pv).subview(0, static_cast<size_t>(rows) * vh);
    auto qr = ws(_pqr).subview(0, static_cast<size_t>(rows) * vh);
    auto kr = ws(_pkr).subview(0, static_cast<size_t>(rows) * vh);
    auto attn = ws(_pattn).subview(0, static_cast<size_t>(rows) * vh);
    auto fc1 = ws(_pfc1).subview(0, static_cast<size_t>(rows) * vi);
    auto dummy_f = ws(_dummy_f);
    auto dummy_u = ws(_dummy_u).as<uint32_t>();
    // 1. Pre-LN.
    {
        auto plan = plan_norm_dispatch(rows);
        if (!plan.ok()) { return plan.status; }
        if (auto cmd = dispatch_1d((*_sh_layer_norm)(x, norm, B.norm1_w, B.norm1_b, rows, vh, eps, 1u), *plan.value)) { list << std::move(cmd); }
    }
    // 2. QKV (bias).
    if (auto s = record_linear(list, norm, B.q_w, q, B.q_b, rows, vh, vh, GemmEpilogue::bias); !s.ok()) { return s; }
    if (auto s = record_linear(list, norm, B.k_w, k, B.k_b, rows, vh, vh, GemmEpilogue::bias); !s.ok()) { return s; }
    if (auto s = record_linear(list, norm, B.v_w, vv, B.v_b, rows, vh, vh, GemmEpilogue::bias); !s.ok()) { return s; }
    // 3. 2D axial RoPE. Perception positions live in the dedicated _p_pos region
    // (uploaded by record_vision); the axis map is in _enc_axis.
    auto positions = ws(_p_pos).as<int32_t>().subview(0, static_cast<size_t>(rows) * 2u);
    auto axis = ws(_enc_axis).as<uint32_t>();
    {
        auto plan_q = plan_rope_dispatch(rows, heads);
        if (!plan_q.ok()) { return plan_q.status; }
        if (auto cmd = dispatch_1d((*_sh_rope2d)(q, qr, positions, axis, rows, heads, hd, 2u, theta), *plan_q.value)) { list << std::move(cmd); }
        auto plan_k = plan_rope_dispatch(rows, heads);
        if (!plan_k.ok()) { return plan_k.status; }
        if (auto cmd = dispatch_1d((*_sh_rope2d)(k, kr, positions, axis, rows, heads, hd, 2u, theta), *plan_k.value)) { list << std::move(cmd); }
    }
    // 4. Windowed/global attention. Build the mask host-side per layer.
    const bool global = std::find(v.global_layers.begin(), v.global_layers.end(), layer) != v.global_layers.end();
    uint32_t mask_kind = static_cast<uint32_t>(AttentionMaskKind::custom_block_mask);
    uint32_t block_size = global ? rows : 1u;
    luisa::vector<uint8_t> mask;
    if (global) {
        mask.assign(1u, 1u);
    } else {
        const uint32_t window = v.window_patch_size;
        const uint32_t gw = packed.grid_w;
        const uint32_t tiles_w = gw == 0u ? 1u : (gw + window - 1u) / window;
        luisa::vector<uint32_t> tile_of(rows);
        for (uint32_t t = 0; t < rows; t++) {
            const uint32_t ph = gw == 0u ? 0u : t / gw;
            const uint32_t pw = gw == 0u ? 0u : t % gw;
            tile_of[t] = (ph / window) * tiles_w + (pw / window);
        }
        mask.resize(static_cast<size_t>(rows) * rows);
        for (uint32_t qi = 0; qi < rows; qi++) {
            for (uint32_t ki = 0; ki < rows; ki++) {
                mask[static_cast<size_t>(qi) * rows + ki] = (tile_of[qi] == tile_of[ki]) ? 1u : 0u;
            }
        }
    }
    // Retain the packed mask in the per-record staging (the copy command borrows
    // it until execution). Upload into _enc_mask (reused; perception runs before
    // the encoder, which overwrites it with the visual/query mask).
    mask_staging.push_back(pack_block_mask_words(mask));
    const auto &mask_words = mask_staging.back();
    auto mask_view = ws(_enc_mask).as<uint32_t>().subview(0, mask_words.size());
    list << mask_view.copy_from(luisa::span<const uint32_t>{mask_words});
    {
        auto plan = plan_attention_prefill_dispatch(rows, heads);
        if (!plan.ok()) { return plan.status; }
        if (auto cmd = dispatch_1d((*_sh_attn_prefill)(qr, kr, vv, attn, mask_view, dummy_f,
                                                       rows, rows, heads, heads, hd, scale, 0.0f,
                                                       mask_kind, 0u, 0u, block_size, 0u), *plan.value)) { list << std::move(cmd); }
    }
    // 5. o-proj (+bias) + residual.
    {
        auto aux = ws(_paux).subview(0, static_cast<size_t>(vh) + static_cast<size_t>(rows) * vh);
        if (auto s = copy_rows(list, B.o_b, aux.subview(0, vh), 1u, vh); !s.ok()) { return s; }
        if (auto s = copy_rows(list, x, aux.subview(vh, static_cast<size_t>(rows) * vh), rows, vh); !s.ok()) { return s; }
        if (auto s = record_linear(list, attn, B.o_w, x, aux, rows, vh, vh, GemmEpilogue::bias_residual); !s.ok()) { return s; }
    }
    // 6. Post-LN.
    {
        auto plan = plan_norm_dispatch(rows);
        if (!plan.ok()) { return plan.status; }
        if (auto cmd = dispatch_1d((*_sh_layer_norm)(x, norm, B.norm2_w, B.norm2_b, rows, vh, eps, 1u), *plan.value)) { list << std::move(cmd); }
    }
    // 7. MLP fc1 (bias+gelu), fc2 (bias) + residual.
    if (auto s = record_linear(list, norm, B.fc1_w, fc1, B.fc1_b, rows, vi, vh, GemmEpilogue::bias_gelu_erf); !s.ok()) { return s; }
    {
        auto aux = ws(_paux).subview(0, static_cast<size_t>(vh) + static_cast<size_t>(rows) * vh);
        if (auto s = copy_rows(list, B.fc2_b, aux.subview(0, vh), 1u, vh); !s.ok()) { return s; }
        if (auto s = copy_rows(list, x, aux.subview(vh, static_cast<size_t>(rows) * vh), rows, vh); !s.ok()) { return s; }
        if (auto s = record_linear(list, fc1, B.fc2_w, x, aux, rows, vh, vi, GemmEpilogue::bias_residual); !s.ok()) { return s; }
    }
    return Status::success();
}

// ---------------------------------------------------------------------------
// Vision: one strided Conv2D downsampler (token <-> NCHW via transpose)
// ---------------------------------------------------------------------------

[[nodiscard]] Status MoeVlmExecutor::record_conv_down(CommandList &list, const ConvWeights &w,
                                                      const DeepSeekConvDownsampler &spec,
                                                      BufferView<float> tokens_in, uint32_t in_c,
                                                      uint32_t gh, uint32_t gw, BufferView<float> tokens_out,
                                                      uint32_t &out_gh, uint32_t &out_gw) {
    const uint32_t rows = gh * gw;
    auto oh = conv2d_out_dim(gh, spec.kernel, spec.stride, spec.pad, 1u);
    auto ow = conv2d_out_dim(gw, spec.kernel, spec.stride, spec.pad, 1u);
    if (!oh.ok()) { return oh.status; }
    if (!ow.ok()) { return ow.status; }
    out_gh = *oh.value;
    out_gw = *ow.value;
    const uint32_t out_rows = out_gh * out_gw;
    if (auto s = validate_conv2d_inputs(1u, in_c, gh, gw, spec.out_channels, spec.kernel, spec.kernel,
                                        spec.stride, spec.stride, 1u, 1u, 1u); !s.ok()) { return s; }
    auto nchw_in = ws(_conv_a).subview(0, static_cast<size_t>(in_c) * rows);
    auto nchw_out = ws(_conv_b).subview(0, static_cast<size_t>(spec.out_channels) * out_rows);
    // tokens [rows, in_c] -> NCHW [in_c, rows].
    {
        auto plan = plan_transpose_2d_dispatch(rows, in_c);
        if (!plan.ok()) { return plan.status; }
        if (auto cmd = dispatch_1d((*_sh_transpose)(tokens_in, nchw_in, rows, in_c), *plan.value)) { list << std::move(cmd); }
    }
    // Conv2D.
    {
        auto plan = plan_conv2d_dispatch(1u, spec.out_channels, out_gh, out_gw);
        if (!plan.ok()) { return plan.status; }
        if (auto cmd = dispatch_1d((*_sh_conv)(nchw_in, w.weight, w.bias, nchw_out, 1u, in_c, gh, gw,
                                               spec.out_channels, spec.kernel, spec.kernel, spec.stride, spec.stride,
                                               spec.pad, spec.pad, 1u, 1u, 1u, out_gh, out_gw, 1u), *plan.value)) { list << std::move(cmd); }
    }
    // NCHW [out_c, out_rows] -> tokens [out_rows, out_c].
    {
        auto plan = plan_transpose_2d_dispatch(spec.out_channels, out_rows);
        if (!plan.ok()) { return plan.status; }
        if (auto cmd = dispatch_1d((*_sh_transpose)(nchw_out, tokens_out, spec.out_channels, out_rows), *plan.value)) { list << std::move(cmd); }
    }
    return Status::success();
}

// ---------------------------------------------------------------------------
// Vision: Qwen2-style visual LM encoder block with the EXACT custom mask.
// Rows = visual_after_down + num_queries. Visual rows bidirectional; each query
// sees all visual rows + previous query rows (incl. itself). The mask is built
// host-side as DATA (block_size 1) and uploaded by record_vision into _enc_mask.
// ---------------------------------------------------------------------------

[[nodiscard]] Status MoeVlmExecutor::record_encoder_block(CommandList &list, uint32_t layer,
                                                          BufferView<float> x, uint32_t rows) {
    const auto &v = _config.vision;
    const auto &B = _encoder[layer];
    const uint32_t eh = v.encoder_hidden(), ei = v.encoder_intermediate_size;
    const uint32_t heads = v.perception.num_heads;
    const uint32_t hd = eh / heads;
    const float eps = v.encoder_norm_eps;
    const float scale = 1.0f / std::sqrt(static_cast<float>(hd));
    auto norm = ws(_enorm).subview(0, static_cast<size_t>(rows) * eh);
    auto q = ws(_eq).subview(0, static_cast<size_t>(rows) * eh);
    auto k = ws(_ek).subview(0, static_cast<size_t>(rows) * eh);
    auto vv = ws(_ev).subview(0, static_cast<size_t>(rows) * eh);
    auto attn = ws(_eattn).subview(0, static_cast<size_t>(rows) * eh);
    auto fc1 = ws(_efc1).subview(0, static_cast<size_t>(rows) * ei);
    auto dummy_f = ws(_dummy_f);
    auto golden = ws(_enc_attn_out[layer]).subview(0, static_cast<size_t>(rows) * eh);
    // 1. Pre-LN.
    {
        auto plan = plan_norm_dispatch(rows);
        if (!plan.ok()) { return plan.status; }
        if (auto cmd = dispatch_1d((*_sh_layer_norm)(x, norm, B.norm1_w, B.norm1_b, rows, eh, eps, 1u), *plan.value)) { list << std::move(cmd); }
    }
    // 2. QKV (bias, Qwen2-style).
    if (auto s = record_linear(list, norm, B.q_w, q, B.q_b, rows, eh, eh, GemmEpilogue::bias); !s.ok()) { return s; }
    if (auto s = record_linear(list, norm, B.k_w, k, B.k_b, rows, eh, eh, GemmEpilogue::bias); !s.ok()) { return s; }
    if (auto s = record_linear(list, norm, B.v_w, vv, B.v_b, rows, eh, eh, GemmEpilogue::bias); !s.ok()) { return s; }
    // 3. Custom visual/query block mask attention (NO RoPE in the encoder; the
    // mask carries the causal-flow structure). block_size 1, mask uploaded by
    // record_vision. The mask is IDENTICAL across encoder layers.
    auto mask_view = ws(_enc_mask).as<uint32_t>().subview(0, static_cast<size_t>((_enc_mask_bytes + 3u) / 4u));
    {
        auto plan = plan_attention_prefill_dispatch(rows, heads);
        if (!plan.ok()) { return plan.status; }
        if (auto cmd = dispatch_1d((*_sh_attn_prefill)(q, k, vv, attn, mask_view, dummy_f,
                                                       rows, rows, heads, heads, hd, scale, 0.0f,
                                                       static_cast<uint32_t>(AttentionMaskKind::custom_block_mask),
                                                       0u, 0u, 1u, 0u), *plan.value)) { list << std::move(cmd); }
    }
    // Save the attention output for layered-golden comparison.
    if (auto s = copy_rows(list, attn, golden, rows, eh); !s.ok()) { return s; }
    // 4. o-proj (+bias) + residual.
    {
        auto aux = ws(_eaux).subview(0, static_cast<size_t>(eh) + static_cast<size_t>(rows) * eh);
        if (auto s = copy_rows(list, B.o_b, aux.subview(0, eh), 1u, eh); !s.ok()) { return s; }
        if (auto s = copy_rows(list, x, aux.subview(eh, static_cast<size_t>(rows) * eh), rows, eh); !s.ok()) { return s; }
        if (auto s = record_linear(list, attn, B.o_w, x, aux, rows, eh, eh, GemmEpilogue::bias_residual); !s.ok()) { return s; }
    }
    // 5. Post-LN.
    {
        auto plan = plan_norm_dispatch(rows);
        if (!plan.ok()) { return plan.status; }
        if (auto cmd = dispatch_1d((*_sh_layer_norm)(x, norm, B.norm2_w, B.norm2_b, rows, eh, eps, 1u), *plan.value)) { list << std::move(cmd); }
    }
    // 6. MLP fc1 (bias+gelu), fc2 (bias) + residual.
    if (auto s = record_linear(list, norm, B.fc1_w, fc1, B.fc1_b, rows, ei, eh, GemmEpilogue::bias_gelu_erf); !s.ok()) { return s; }
    {
        auto aux = ws(_eaux).subview(0, static_cast<size_t>(eh) + static_cast<size_t>(rows) * eh);
        if (auto s = copy_rows(list, B.fc2_b, aux.subview(0, eh), 1u, eh); !s.ok()) { return s; }
        if (auto s = copy_rows(list, x, aux.subview(eh, static_cast<size_t>(rows) * eh), rows, eh); !s.ok()) { return s; }
        if (auto s = record_linear(list, fc1, B.fc2_w, x, aux, rows, eh, ei, GemmEpilogue::bias_residual); !s.ok()) { return s; }
    }
    return Status::success();
}

// ---------------------------------------------------------------------------
// Vision: full graph for ONE image.
// ---------------------------------------------------------------------------

// S9: ONE canonical custom visual/query mask builder, shared by the executor and
// the tests (block_size 1 => enc_rows*enc_rows bytes).
[[nodiscard]] luisa::vector<uint8_t> MoeVlmExecutor::build_visual_query_mask(uint32_t visual_rows, uint32_t num_queries) {
    const uint32_t rows = visual_rows + num_queries;
    luisa::vector<uint8_t> mask(static_cast<size_t>(rows) * rows, 0u);
    for (uint32_t qi = 0; qi < rows; qi++) {
        for (uint32_t ki = 0; ki < rows; ki++) {
            bool allow;
            if (qi < visual_rows) {
                allow = (ki < visual_rows);// visual: bidirectional among visual only
            } else {
                const uint32_t q = qi - visual_rows;
                allow = (ki < visual_rows) || (ki >= visual_rows && (ki - visual_rows) <= q);
            }
            mask[static_cast<size_t>(qi) * rows + ki] = allow ? 1u : 0u;
        }
    }
    return mask;
}

// S4: record the real pixel/position/axis uploads, then return an error. The
// StagingGuard must still attach its completion callback on this error path, so
// committing the list afterwards cannot read freed staging.
[[nodiscard]] Status MoeVlmExecutor::record_vision_uploads_then_fail_for_test(CommandList &list, const P7ImageInput &image) {
    if (_device == nullptr) { return Status::error(StatusCode::invalid_argument, "moe executor not created", "device"); }
    const auto &v = _config.vision;
    const auto &p = image.packed;
    if (image.source_width == 0u || image.source_height == 0u || p.grid_h == 0u || p.grid_w == 0u ||
        p.resized_width == 0u || p.resized_height == 0u) {
        return Status::error(StatusCode::invalid_argument, "degenerate image for the staging test", "image");
    }
    uint64_t grid_patches = 0;
    if (!checked_mul(p.grid_h, p.grid_w, grid_patches) || grid_patches != p.patch_count ||
        grid_patches > v.max_vision_patches) {
        return Status::error(StatusCode::invalid_argument, "inconsistent packed grid for the staging test", "packed");
    }
    const uint32_t rows = static_cast<uint32_t>(grid_patches);
    UploadStagingGuard guard{&list, luisa::make_unique<RecordUploadStaging>()};
    auto *stage = guard.staging.get();
    stage->pixels = preprocess_chw(image.pixels, image.source_width, image.source_height,
                                   p.resized_width, p.resized_height, v.mean, v.std_dev);
    stage->positions = p7_vision_positions(p);
    stage->axis = p7_vision_axis_of_pair(v.perception);
    _last_pixel_elems = stage->pixels.size();
    list << ws(_pixels).subview(0, stage->pixels.size()).copy_from(luisa::span<const float>{stage->pixels})
         << ws(_p_pos).as<int32_t>().subview(0, static_cast<size_t>(rows) * 2u)
                .copy_from(luisa::span<const int32_t>{stage->positions.data(), static_cast<size_t>(rows) * 2u})
         << ws(_enc_axis).as<uint32_t>().subview(0, stage->axis.size())
                .copy_from(luisa::span<const uint32_t>{stage->axis});
    // Deliberate failure AFTER the uploads were recorded.
    return Status::error(StatusCode::internal_error, "injected post-upload failure (staging lifetime test)", "staging_test");
}

[[nodiscard]] Result<MoeVisionOutput> MoeVlmExecutor::record_vision(CommandList &list, const P7ImageInput &image) {
    using R = Result<MoeVisionOutput>;
    if (_device == nullptr) { return R::failure(StatusCode::invalid_argument, "moe executor not created", "device"); }
    const auto &v = _config.vision;
    const auto &p = image.packed;
    // M3: reject a degenerate SOURCE before any preprocess clamp math (a zero
    // edge would divide by zero in the bilinear scale and clamp to -1).
    if (image.source_width == 0u || image.source_height == 0u) {
        return R::failure(StatusCode::invalid_argument, "image source dimensions must be nonzero", "source");
    }
    if (image.pixels.empty()) {
        return R::failure(StatusCode::invalid_argument, "image pixel buffer is empty", "pixels");
    }
    uint64_t want_pixels = 0;
    if (!checked_mul(static_cast<uint64_t>(image.source_width) * image.source_height, 3ull, want_pixels) ||
        image.pixels.size() != want_pixels) {
        return R::failure(StatusCode::invalid_argument, "image pixel buffer size mismatch", "pixels");
    }
    if (p.resized_width == 0u || p.resized_height == 0u || p.grid_h == 0u || p.grid_w == 0u) {
        return R::failure(StatusCode::invalid_argument, "degenerate packed image", "packed");
    }
    if (p.patch_count == 0u) { return R::failure(StatusCode::invalid_argument, "degenerate packed image", "packed"); }
    // M2: patch_count MUST equal grid_h*grid_w (checked). Deriving `rows` from a
    // forged patch_count would over- or under-run every vision region.
    uint64_t grid_patches = 0;
    if (!checked_mul(p.grid_h, p.grid_w, grid_patches) || grid_patches != p.patch_count) {
        return R::failure(StatusCode::invalid_argument,
                          "packed patch_count does not equal grid_h*grid_w", "patch_count");
    }
    if (p.patch_count > v.max_vision_patches) {
        return R::failure(StatusCode::out_of_memory, "vision patches exceed the workspace capacity", "max_vision_patches");
    }
    uint64_t want_h = 0, want_w = 0;
    if (!checked_mul(p.grid_h, v.perception.patch_size, want_h) || !checked_mul(p.grid_w, v.perception.patch_size, want_w) ||
        want_h != p.resized_height || want_w != p.resized_width) {
        return R::failure(StatusCode::invalid_argument, "packing grid does not match the resized size", "packed");
    }
    // Both downsamplers must fit the actual grid (config validation already
    // guarantees they never ENLARGE it; this checks the concrete input).
    {
        uint32_t gh = p.grid_h, gw = p.grid_w;
        for (const auto *spec : {&v.down1, &v.down2}) {
            auto oh = conv2d_out_dim(gh, spec->kernel, spec->stride, spec->pad, 1u);
            auto ow = conv2d_out_dim(gw, spec->kernel, spec->stride, spec->pad, 1u);
            if (!oh.ok()) { return R::failure(oh.status); }
            if (!ow.ok()) { return R::failure(ow.status); }
            if (*oh.value == 0u || *ow.value == 0u || *oh.value > gh || *ow.value > gw) {
                return R::failure(StatusCode::invalid_argument,
                                  "conv downsampler output grid is degenerate or enlarges the input", "downsampler");
            }
            gh = *oh.value;
            gw = *ow.value;
        }
    }
    const uint32_t rows = static_cast<uint32_t>(grid_patches);
    const uint32_t vh = v.perception.hidden_size, patch_dim = v.perception.patch_dim();
    // Host preprocess (bilinear + normalize). All uploaded host arrays live in
    // the per-record staging (retained through completion).
    // S4: the completion callback is attached by a RAII guard on EVERY exit
    // path, including an error return AFTER an upload was recorded, so a caller
    // that commits the list can never read freed staging.
    UploadStagingGuard guard{&list, luisa::make_unique<RecordUploadStaging>()};
    auto *stage = guard.staging.get();
    stage->pixels = preprocess_chw(image.pixels, image.source_width, image.source_height,
                                   p.resized_width, p.resized_height, v.mean, v.std_dev);
    stage->positions = p7_vision_positions(p);
    stage->axis = p7_vision_axis_of_pair(v.perception);
    stage->masks.reserve(v.perception.num_layers + 1u);
    _last_pixel_elems = stage->pixels.size();
    auto pixels = ws(_pixels).subview(0, stage->pixels.size());
    list << pixels.copy_from(luisa::span<const float>{stage->pixels})
         << ws(_p_pos).as<int32_t>().subview(0, static_cast<size_t>(rows) * 2u)
                .copy_from(luisa::span<const int32_t>{stage->positions.data(), static_cast<size_t>(rows) * 2u})
         << ws(_enc_axis).as<uint32_t>().subview(0, stage->axis.size())
                .copy_from(luisa::span<const uint32_t>{stage->axis});
    // Patchify.
    auto patches = ws(_patches).subview(0, static_cast<size_t>(rows) * patch_dim);
    {
        auto plan = plan_image_patchify_dispatch(rows, patch_dim);
        if (!plan.ok()) { return R::failure(plan.status); }
        if (auto cmd = dispatch_1d((*_sh_patchify)(pixels, patches, 3u, p.resized_height, p.resized_width, v.perception.patch_size), *plan.value)) { list << std::move(cmd); }
    }
    // Patch embed (bias).
    auto px = ws(_px).subview(0, static_cast<size_t>(rows) * vh);
    if (auto s = record_linear(list, patches, _patch_embed_w, px, _patch_embed_b, rows, vh, patch_dim, GemmEpilogue::bias); !s.ok()) { return R::failure(s); }
    // SAM perception blocks (masks retained in stage->masks).
    for (uint32_t l = 0; l < v.perception.num_layers; l++) {
        if (auto s = record_perception_block(list, l, px, rows, p, stage->masks); !s.ok()) { return R::failure(s); }
    }
    // BOTH conv downsamplers. px [rows, vh] -> conv_tokens -> down1 -> down2.
    uint32_t gh = p.grid_h, gw = p.grid_w;
    uint32_t cur_c = vh;
    auto cur_tokens = px;
    // down1: px -> _conv_tokens.
    {
        uint32_t ogh = 0, ogw = 0;
        auto out = ws(_conv_tokens).subview(0, static_cast<size_t>(v.down1.out_channels) * (rows));
        if (auto s = record_conv_down(list, _down1, v.down1, cur_tokens, cur_c, gh, gw, out, ogh, ogw); !s.ok()) { return R::failure(s); }
        gh = ogh; gw = ogw; cur_c = v.down1.out_channels;
        cur_tokens = ws(_conv_tokens).subview(0, static_cast<size_t>(cur_c) * gh * gw);
    }
    // down2: _conv_tokens -> _ex (the encoder residual stream base).
    {
        uint32_t ogh = 0, ogw = 0;
        const uint32_t visual_after = gh * gw;// before down2
        (void)visual_after;
        auto out = ws(_ex).subview(0, static_cast<size_t>(v.down2.out_channels) * (gh * gw));
        if (auto s = record_conv_down(list, _down2, v.down2, cur_tokens, cur_c, gh, gw, out, ogh, ogw); !s.ok()) { return R::failure(s); }
        gh = ogh; gw = ogw; cur_c = v.down2.out_channels;
    }
    const uint32_t visual_rows = gh * gw;
    const uint32_t nq = v.num_queries;
    const uint32_t enc_rows = visual_rows + nq;
    const uint32_t eh = v.encoder_hidden();
    _last_encoder_rows = enc_rows;
    // Append the causal-flow query embeddings to the visual rows in _ex.
    auto ex = ws(_ex).subview(0, static_cast<size_t>(enc_rows) * eh);
    {
        auto query_dst = ex.subview(static_cast<size_t>(visual_rows) * eh, static_cast<size_t>(nq) * eh);
        if (auto s = copy_rows(list, _query_embed, query_dst, nq, eh); !s.ok()) { return R::failure(s); }
    }
    // Build the EXACT custom visual/query block mask (block_size 1):
    //   visual row i: sees all visual rows [0, visual_rows), no queries.
    //   query row (visual_rows + q): sees all visual rows + queries [0, q] (incl self).
    {
        const auto mask = build_visual_query_mask(visual_rows, nq);
        // Retain the packed mask in the per-record staging (the copy command
        // borrows it until execution; a later record_vision must never dangle it).
        stage->masks.push_back(pack_block_mask_words(mask));
        _enc_mask_bytes = static_cast<uint32_t>(mask.size());
        const auto &enc_words = stage->masks.back();
        auto mask_view = ws(_enc_mask).as<uint32_t>().subview(0, enc_words.size());
        list << mask_view.copy_from(luisa::span<const uint32_t>{enc_words});
    }
    // Visual LM encoder blocks (custom mask).
    for (uint32_t l = 0; l < v.encoder_layers; l++) {
        if (auto s = record_encoder_block(list, l, ex, enc_rows); !s.ok()) { return R::failure(s); }
    }
    // Connector over the QUERY rows only: LayerNorm + fc1(gelu) + fc2 -> text hidden.
    const uint32_t th = _config.decoder.decoder.hidden_size;
    auto query_rows = ex.subview(static_cast<size_t>(visual_rows) * eh, static_cast<size_t>(nq) * eh);
    auto conn_norm = ws(_conn_norm).subview(0, static_cast<size_t>(nq) * eh);
    {
        auto plan = plan_norm_dispatch(nq);
        if (!plan.ok()) { return R::failure(plan.status); }
        if (auto cmd = dispatch_1d((*_sh_layer_norm)(query_rows, conn_norm, _conn_ln_w, _conn_ln_b, nq, eh, v.connector_norm_eps, 1u), *plan.value)) { list << std::move(cmd); }
    }
    auto conn_fc1 = ws(_conn_fc1).subview(0, static_cast<size_t>(nq) * v.connector_intermediate_size);
    if (auto s = record_linear(list, conn_norm, _conn_fc1_w, conn_fc1, _conn_fc1_b, nq, v.connector_intermediate_size, eh, GemmEpilogue::bias_gelu_erf); !s.ok()) { return R::failure(s); }
    auto features = ws(_features).subview(0, static_cast<size_t>(nq) * th);
    if (auto s = record_linear(list, conn_fc1, _conn_fc2_w, features, _conn_fc2_b, nq, th, v.connector_intermediate_size, GemmEpilogue::bias); !s.ok()) { return R::failure(s); }
    // S4: the StagingGuard destructor attaches the completion callback on this
    // (and every error) return path, so no explicit add_callback is needed here.
    MoeVisionOutput out;
    out.features = features;
    out.visual_tokens = nq;
    out.encoder_rows = enc_rows;
    return R::success(std::move(out));
}

// ---------------------------------------------------------------------------
// MoE decoder block: attention + router + active-expert loop + shared + scatter.
// Drives the stream: records attention+router+permutation, flushes, reads the
// routing counts/offsets on the host, then records per-ACTIVE-expert GEMMs and
// the weighted scatter, flushes. NO grouped persistent GEMM.
// ---------------------------------------------------------------------------

[[nodiscard]] Status MoeVlmExecutor::record_moe_block(Stream &stream, uint32_t layer,
                                                      BufferView<float> x, uint32_t rows, uint32_t pos0,
                                                      ContiguousKvCache &cache, bool decode) {
    const auto &d = _config.decoder.decoder;
    const auto &L = _layers[layer];
    auto &M = _moe[layer];
    const uint32_t hidden = d.hidden_size, qd = d.q_dim(), kvd = d.kv_dim();
    const uint32_t qh = d.num_attention_heads, kvh = d.num_kv_heads, hd = d.head_dim;
    const float eps = d.rms_norm_eps, theta = d.rope_theta;
    const float scale = 1.0f / std::sqrt(static_cast<float>(hd));
    const uint32_t top_k = _config.decoder.top_k, ne = _config.decoder.num_experts;
    const uint32_t mi = _config.decoder.moe_intermediate_size, si = _config.decoder.shared_intermediate_size;
    const uint32_t slots = rows * top_k;
    // S6: run the MoE input validators on the ACTUAL execution path (not only in
    // tests), so a malformed geometry/envelope fails closed before any launch.
    const MoeRouterConfig router_cfg{ne, top_k};
    if (auto s = validate_moe_router_inputs(rows, hidden, router_cfg); !s.ok()) { return s; }
    if (auto s = validate_moe_topk_inputs(rows, router_cfg); !s.ok()) { return s; }
    auto list = CommandList::create();
    auto norm = ws(_norm).subview(0, static_cast<size_t>(rows) * hidden);
    auto norm2 = ws(_norm2).subview(0, static_cast<size_t>(rows) * hidden);
    auto q = ws(_q).subview(0, static_cast<size_t>(rows) * qd);
    auto k = ws(_k).subview(0, static_cast<size_t>(rows) * kvd);
    auto v = ws(_v).subview(0, static_cast<size_t>(rows) * kvd);
    auto q_rope = ws(_q_rope).subview(0, static_cast<size_t>(rows) * qd);
    auto k_rope = ws(_k_rope).subview(0, static_cast<size_t>(rows) * kvd);
    auto attn = ws(_attn).subview(0, static_cast<size_t>(rows) * qd);
    auto pos_view = ws(_positions).as<int32_t>().subview(0, rows);
    auto axis_view = ws(_axis).as<uint32_t>();
    auto dummy_f = ws(_dummy_f);
    auto dummy_u = ws(_dummy_u).as<uint32_t>();
    // 1. input RMSNorm.
    {
        auto plan = plan_norm_dispatch(rows);
        if (!plan.ok()) { return plan.status; }
        if (auto cmd = dispatch_1d((*_sh_rms_norm)(x, norm, L.input_norm, rows, hidden, eps, 1u), *plan.value)) { list << std::move(cmd); }
    }
    // 2. QKV (bias).
    if (auto s = record_linear(list, norm, L.q_w, q, L.q_b, rows, qd, hidden, GemmEpilogue::bias); !s.ok()) { return s; }
    if (auto s = record_linear(list, norm, L.k_w, k, L.k_b, rows, kvd, hidden, GemmEpilogue::bias); !s.ok()) { return s; }
    if (auto s = record_linear(list, norm, L.v_w, v, L.v_b, rows, kvd, hidden, GemmEpilogue::bias); !s.ok()) { return s; }
    // 3. 1D rotate-half RoPE on Q/K.
    {
        auto plan_q = plan_rope_dispatch(rows, qh);
        if (!plan_q.ok()) { return plan_q.status; }
        if (auto cmd = dispatch_1d((*_sh_rope_rh)(q, q_rope, pos_view, axis_view, rows, qh, hd, 1u, theta), *plan_q.value)) { list << std::move(cmd); }
        auto plan_k = plan_rope_dispatch(rows, kvh);
        if (!plan_k.ok()) { return plan_k.status; }
        if (auto cmd = dispatch_1d((*_sh_rope_rh)(k, k_rope, pos_view, axis_view, rows, kvh, hd, 1u, theta), *plan_k.value)) { list << std::move(cmd); }
    }
    // 4. KV append at pos0.
    {
        if (auto s = validate_kv_append(pos0, rows, cache.config().max_seq_len); !s.ok()) { return s; }
        auto plan = plan_kv_append_dispatch(rows, kvd);
        if (!plan.ok()) { return plan.status; }
        if (auto cmd = dispatch_1d((*_sh_kv_append)(k_rope, v, cache.k_layer(layer), cache.v_layer(layer), rows, kvd, pos0), *plan.value)) { list << std::move(cmd); }
    }
    // 5. Causal attention (text decoder IS causal; the custom mask is vision-only).
    if (decode) {
        auto plan = plan_attention_decode_dispatch(1, qh);
        if (!plan.ok()) { return plan.status; }
        if (auto cmd = dispatch_1d((*_sh_attn_decode)(q_rope, cache.k_layer(layer), cache.v_layer(layer),
                                                      attn, cache.state().kv_len(), cache.state().positions(), dummy_u,
                                                      1u, cache.config().max_seq_len, qh, kvh, hd, scale, 0.0f,
                                                      static_cast<uint32_t>(AttentionMaskKind::causal), 0u, 0u, 0u, 0u), *plan.value)) { list << std::move(cmd); }
    } else {
        const uint32_t k_len = pos0 + rows;
        auto plan = plan_attention_prefill_dispatch(rows, qh);
        if (!plan.ok()) { return plan.status; }
        if (auto cmd = dispatch_1d((*_sh_attn_prefill)(q_rope, cache.k_layer(layer), cache.v_layer(layer),
                                                       attn, dummy_u, dummy_f, rows, k_len, qh, kvh, hd, scale, 0.0f,
                                                       static_cast<uint32_t>(AttentionMaskKind::causal), 0u, 0u, 0u, 0u), *plan.value)) { list << std::move(cmd); }
    }
    // 6. o-proj + residual.
    if (auto s = record_linear(list, attn, L.o_w, x, x, rows, hidden, qd, GemmEpilogue::residual); !s.ok()) { return s; }
    // 7. post-attention RMSNorm.
    {
        auto plan = plan_norm_dispatch(rows);
        if (!plan.ok()) { return plan.status; }
        if (auto cmd = dispatch_1d((*_sh_rms_norm)(x, norm2, L.post_norm, rows, hidden, eps, 1u), *plan.value)) { list << std::move(cmd); }
    }
    // 8. MoE router: FP32 projection -> softmax -> top-k -> histogram -> offsets
    //    -> permutation -> gather. Record + flush, then read counts on the host.
    auto router_logits = ws(M.router_logits).subview(0, static_cast<size_t>(rows) * ne);
    auto probs = ws(M.probs).subview(0, static_cast<size_t>(rows) * ne);
    auto ids = ws(M.ids).as<uint32_t>().subview(0, slots);
    auto weights = ws(M.weights).subview(0, slots);
    auto counts = ws(M.counts).as<uint32_t>().subview(0, ne);
    auto offsets = ws(M.offsets).as<uint32_t>().subview(0, ne);
    auto perm_slot = ws(M.perm_slot).as<uint32_t>().subview(0, slots);
    auto slot_pos = ws(M.slot_pos).as<uint32_t>().subview(0, slots);
    auto perm_token = ws(M.perm_token).as<uint32_t>().subview(0, slots);
    auto gathered = ws(M.gathered).subview(0, static_cast<size_t>(slots) * hidden);
    if (auto s = record_linear(list, norm2, L.router_w, router_logits, dummy_f, rows, ne, hidden, GemmEpilogue::none); !s.ok()) { return s; }
    {
        auto plan = plan_moe_router_dispatch(rows);
        if (!plan.ok()) { return plan.status; }
        if (auto cmd = dispatch_1d((*_sh_moe_softmax)(router_logits, probs, rows, ne), *plan.value)) { list << std::move(cmd); }
    }
    {
        auto plan = plan_moe_topk_dispatch(rows);
        if (!plan.ok()) { return plan.status; }
        if (auto cmd = dispatch_1d((*_sh_moe_topk)(probs, ids, weights, rows, ne, top_k), *plan.value)) { list << std::move(cmd); }
    }
    {
        auto plan = plan_moe_histogram_dispatch(ne);
        if (!plan.ok()) { return plan.status; }
        if (auto cmd = dispatch_1d((*_sh_moe_hist)(ids, counts, slots, ne), *plan.value)) { list << std::move(cmd); }
    }
    {
        auto plan = plan_moe_offsets_dispatch(ne);
        if (!plan.ok()) { return plan.status; }
        if (auto cmd = dispatch_1d((*_sh_moe_offsets)(counts, offsets, ne), *plan.value)) { list << std::move(cmd); }
    }
    {
        auto plan = plan_moe_permutation_dispatch(ne);
        if (!plan.ok()) { return plan.status; }
        if (auto cmd = dispatch_1d((*_sh_moe_perm)(ids, offsets, perm_slot, slot_pos, perm_token, slots, ne, top_k), *plan.value)) { list << std::move(cmd); }
    }
    {
        auto plan = plan_moe_gather_dispatch(slots, hidden);
        if (!plan.ok()) { return plan.status; }
        if (auto cmd = dispatch_1d((*_sh_moe_gather)(norm2, perm_token, gathered, slots, hidden), *plan.value)) { list << std::move(cmd); }
    }
    // Flush + read the routing counts/offsets on the host (active-expert loop).
    if (auto s = flush(list, stream); !s.ok()) { return s; }
    stream << counts.copy_to(luisa::span<uint32_t>{_layer_counts[layer]})
           << offsets.copy_to(luisa::span<uint32_t>{_layer_offsets[layer]})
           << synchronize();
    // S3: validate the readback BEFORE it drives any subview (Status, not abort).
    if (auto s = validate_moe_routing_tables(_layer_counts[layer], _layer_offsets[layer], slots); !s.ok()) { return s; }
    // 9. Per-ACTIVE-expert FFN (gate/up/SwiGLU/down) over its gathered rows.
    auto gate = ws(M.gate);
    auto up = ws(M.up);
    auto act = ws(M.act);
    auto expert_out = ws(M.expert_out);
    for (uint32_t e = 0; e < ne; e++) {
        const uint32_t count = _layer_counts[layer][e];
        if (count == 0u) { continue; }// inactive expert: skipped entirely
        const uint32_t off = _layer_offsets[layer][e];
        auto g_rows = gathered.subview(static_cast<size_t>(off) * hidden, static_cast<size_t>(count) * hidden);
        auto gate_e = gate.subview(0, static_cast<size_t>(count) * mi);
        auto up_e = up.subview(0, static_cast<size_t>(count) * mi);
        auto act_e = act.subview(0, static_cast<size_t>(count) * mi);
        auto out_e = expert_out.subview(static_cast<size_t>(off) * hidden, static_cast<size_t>(count) * hidden);
        if (auto s = record_linear(list, g_rows, L.expert_gate[e], gate_e, dummy_f, count, mi, hidden, GemmEpilogue::none); !s.ok()) { return s; }
        if (auto s = record_linear(list, g_rows, L.expert_up[e], up_e, dummy_f, count, mi, hidden, GemmEpilogue::none); !s.ok()) { return s; }
        {
            const uint64_t cnt = static_cast<uint64_t>(count) * mi;
            auto plan = plan_gated_act_dispatch(cnt);
            if (!plan.ok()) { return plan.status; }
            auto cnt32 = checked_u32(cnt, "moe act count");
            if (!cnt32.ok()) { return cnt32.status; }
            if (auto cmd = dispatch_1d((*_sh_gated_silu)(gate_e, up_e, act_e, *cnt32.value), *plan.value)) { list << std::move(cmd); }
        }
        if (auto s = record_linear(list, act_e, L.expert_down[e], out_e, dummy_f, count, hidden, mi, GemmEpilogue::none); !s.ok()) { return s; }
    }
    // 10. Two shared dense experts over ALL rows.
    auto shared0 = ws(M.shared0).subview(0, static_cast<size_t>(rows) * hidden);
    auto shared1 = ws(M.shared1).subview(0, static_cast<size_t>(rows) * hidden);
    const auto shared_ffn = [&](BufferView<float> gw, BufferView<float> uw, BufferView<float> dw,
                                BufferView<float> out) -> Status {
        auto gate_s = gate.subview(0, static_cast<size_t>(rows) * si);
        auto up_s = up.subview(0, static_cast<size_t>(rows) * si);
        auto act_s = act.subview(0, static_cast<size_t>(rows) * si);
        if (auto s = record_linear(list, norm2, gw, gate_s, dummy_f, rows, si, hidden, GemmEpilogue::none); !s.ok()) { return s; }
        if (auto s = record_linear(list, norm2, uw, up_s, dummy_f, rows, si, hidden, GemmEpilogue::none); !s.ok()) { return s; }
        {
            const uint64_t cnt = static_cast<uint64_t>(rows) * si;
            auto plan = plan_gated_act_dispatch(cnt);
            if (!plan.ok()) { return plan.status; }
            auto cnt32 = checked_u32(cnt, "moe shared act count");
            if (!cnt32.ok()) { return cnt32.status; }
            if (auto cmd = dispatch_1d((*_sh_gated_silu)(gate_s, up_s, act_s, *cnt32.value), *plan.value)) { list << std::move(cmd); }
        }
        return record_linear(list, act_s, dw, out, dummy_f, rows, hidden, si, GemmEpilogue::none);
    };
    if (auto s = shared_ffn(L.shared0_gate, L.shared0_up, L.shared0_down, shared0); !s.ok()) { return s; }
    if (auto s = shared_ffn(L.shared1_gate, L.shared1_up, L.shared1_down, shared1); !s.ok()) { return s; }
    // 11. Weighted restore to token order + shared -> moe_out.
    auto moe_out = ws(M.moe_out).subview(0, static_cast<size_t>(rows) * hidden);
    {
        auto plan = plan_moe_scatter_dispatch(rows, hidden);
        if (!plan.ok()) { return plan.status; }
        if (auto cmd = dispatch_1d((*_sh_moe_scatter)(expert_out, slot_pos, weights, shared0, shared1, moe_out, rows, hidden, top_k), *plan.value)) { list << std::move(cmd); }
    }
    // 12. Residual: x = x + moe_out.
    {
        const uint64_t cnt = static_cast<uint64_t>(rows) * hidden;
        auto plan = plan_residual_add_dispatch(cnt);
        if (!plan.ok()) { return plan.status; }
        auto cnt32 = checked_u32(cnt, "moe residual count");
        if (!cnt32.ok()) { return cnt32.status; }
        if (auto cmd = dispatch_1d((*_sh_residual)(x, moe_out, x, *cnt32.value), *plan.value)) { list << std::move(cmd); }
    }
    return flush(list, stream);
}

// ---------------------------------------------------------------------------
// final norm + LM head
// ---------------------------------------------------------------------------

[[nodiscard]] Status MoeVlmExecutor::record_lm_head(CommandList &list, BufferView<float> x,
                                                    uint32_t rows, BufferView<float> C) {
    const auto &d = _config.decoder.decoder;
    const uint32_t hidden = d.hidden_size, vocab = d.vocab_size;
    auto norm = ws(_norm).subview(0, static_cast<size_t>(rows) * hidden);
    {
        auto plan = plan_norm_dispatch(rows);
        if (!plan.ok()) { return plan.status; }
        if (auto cmd = dispatch_1d((*_sh_rms_norm)(x, norm, _final_norm, rows, hidden, d.rms_norm_eps, 1u), *plan.value)) { list << std::move(cmd); }
    }
    return record_linear(list, norm, _lm_head, C, ws(_dummy_f), rows, vocab, hidden, GemmEpilogue::none);
}

// ---------------------------------------------------------------------------
// KV cache creation
// ---------------------------------------------------------------------------

[[nodiscard]] Status MoeVlmExecutor::create_cache(Device &device, Stream &stream, uint32_t max_seq_len,
                                                  ContiguousKvCache &cache, MemoryBudget *budget) const {
    const auto &d = _config.decoder.decoder;
    if (max_seq_len == 0u) {
        return Status::error(StatusCode::invalid_argument, "max_seq_len must be nonzero", "max_seq_len");
    }
    // S2: the KV capacity may never exceed the decoder's declared context
    // ceiling (RoPE positions beyond it are undefined for this contract).
    if (max_seq_len > d.max_position_embeddings) {
        return Status::error(StatusCode::invalid_argument,
                             luisa::format("max_seq_len {} exceeds the decoder max_position_embeddings {}",
                                           max_seq_len, d.max_position_embeddings),
                             "max_seq_len");
    }
    // S5: preflight the FULL footprint (borrowed weights + the actual workspace
    // arena + this KV capacity) against the configured limits BEFORE allocating
    // the cache, so an over-budget request fails with required-vs-available.
    auto kv_bytes = moe_kv_bytes(d, max_seq_len);
    if (!kv_bytes.ok()) { return kv_bytes.status; }
    MemoryBudget total{_config.memory};
    if (auto s = total.require("moe_weights", MemoryCategory::weights, _weight_bytes); !s.ok()) { return s; }
    if (auto s = total.require("moe_workspace", MemoryCategory::workspace, _workspace.capacity_bytes()); !s.ok()) { return s; }
    if (auto s = total.require("moe_kv_cache", MemoryCategory::kv_cache, *kv_bytes.value); !s.ok()) { return s; }
    if (auto s = total.preflight(); !s.ok()) { return s; }
    KvCacheConfig cc;
    cc.num_layers = d.num_layers;
    cc.num_kv_heads = d.num_kv_heads;
    cc.head_dim = d.head_dim;
    cc.max_seq_len = max_seq_len;
    cc.batch = 1;
    cc.dtype_bytes = 4;
    return cache.create(device, stream, cc, budget);
}

// ---------------------------------------------------------------------------
// prefill (single chunk; drives the stream, syncs at each MoE layer)
// ---------------------------------------------------------------------------

[[nodiscard]] Status MoeVlmExecutor::prefill(Stream &stream, const DecodeRequest &request,
                                             ContiguousKvCache &cache, luisa::span<float> logits_out) {
    auto valid = request.validate();
    if (!valid.ok()) { return valid; }
    const auto &d = _config.decoder.decoder;
    const uint32_t prompt_len = static_cast<uint32_t>(request.prompt_tokens.size());
    const uint32_t hidden = d.hidden_size, vocab = d.vocab_size;
    if (prompt_len == 0u) {
        return Status::error(StatusCode::invalid_argument, "empty prompt", "prompt_tokens");
    }
    if (prompt_len > _config.decoder.prefill_chunk_rows) {
        return Status::error(StatusCode::unsupported,
                             "MoE prefill is single-chunk; prompt exceeds prefill_chunk_rows", "prompt_len");
    }
    if (prompt_len > cache.config().max_seq_len) {
        return Status::error(StatusCode::out_of_memory, "prompt exceeds KV capacity", "prompt_len");
    }
    if (cache.config().num_layers != d.num_layers || cache.config().num_kv_heads != d.num_kv_heads ||
        cache.config().head_dim != d.head_dim || cache.config().batch != 1u) {
        return Status::error(StatusCode::invalid_argument, "KV cache geometry mismatch", "cache");
    }
    if (logits_out.size() != vocab) {
        return Status::error(StatusCode::invalid_argument, "logits_out size must equal vocab_size", "logits_out");
    }
    if (request.has_multimodal() && request.mm_features.size() < static_cast<size_t>(request.mm_count) * hidden) {
        return Status::error(StatusCode::invalid_argument, "mm_features too small", "mm_features");
    }
    for (size_t i = 0; i < request.prompt_tokens.size(); i++) {
        if (request.prompt_tokens[i] >= vocab) {
            return Status::error(StatusCode::invalid_argument, "prompt token outside the vocabulary", "prompt_tokens");
        }
    }
    if (auto s = cache.bind_sequence(0, prompt_len); !s.ok()) { return s; }
    _decode_step = 0;
    _last_rows = prompt_len;
    _last_slots = prompt_len * _config.decoder.top_k;

    auto list = CommandList::create();
    UploadStagingGuard guard{&list, luisa::make_unique<RecordUploadStaging>(), &stream};
    auto *stage = guard.staging.get();
    stage->ids.assign(request.prompt_tokens.begin(), request.prompt_tokens.end());
    stage->positions.resize(prompt_len);
    for (uint32_t i = 0; i < prompt_len; i++) { stage->positions[i] = static_cast<int32_t>(i); }
    stage->axis = _axis_pending;
    auto ids_view = ws(_ids_tok).as<int32_t>().subview(0, prompt_len);
    auto pos_view = ws(_positions).as<int32_t>().subview(0, prompt_len);
    list << ids_view.copy_from(luisa::span<const int32_t>{stage->ids})
         << pos_view.copy_from(luisa::span<const int32_t>{stage->positions})
         << ws(_axis).as<uint32_t>().copy_from(luisa::span<const uint32_t>{stage->axis});
    auto emb = ws(_emb).subview(0, static_cast<size_t>(prompt_len) * hidden);
    auto x = ws(_x).subview(0, static_cast<size_t>(prompt_len) * hidden);
    {
        auto plan = plan_embedding_dispatch(prompt_len, hidden);
        if (!plan.ok()) { return plan.status; }
        if (auto cmd = dispatch_1d((*_sh_embedding)(_embed, ids_view, emb, prompt_len, hidden), *plan.value)) { list << std::move(cmd); }
    }
    {
        auto plan = plan_mm_scatter_dispatch(prompt_len, hidden);
        if (!plan.ok()) { return plan.status; }
        auto feats = request.has_multimodal() ? request.mm_features : ws(_dummy_f);
        if (auto cmd = dispatch_1d((*_sh_mm_scatter)(emb, feats, x, prompt_len, hidden, request.mm_start, request.mm_count), *plan.value)) { list << std::move(cmd); }
    }
    // Record device metadata: prefill state, kv_len = prompt_len, next pos = prompt_len.
    const luisa::vector<uint32_t> meta_kv_len = {prompt_len};
    const luisa::vector<int32_t> meta_positions = {static_cast<int32_t>(prompt_len)};
    const luisa::vector<uint32_t> meta_step = {0u};
    const luisa::vector<uint32_t> meta_state = {static_cast<uint32_t>(DecodeSeqState::prefill)};
    if (auto s = cache.state().record(list, meta_kv_len, meta_positions, meta_step, meta_state); !s.ok()) { return s; }
    if (auto s = flush(list, stream); !s.ok()) { return s; }
    // Decoder layers (each drives the stream internally for the MoE loop).
    for (uint32_t l = 0; l < d.num_layers; l++) {
        if (auto s = record_moe_block(stream, l, x, prompt_len, 0u, cache, false); !s.ok()) { return s; }
    }
    // LM head over the LAST row.
    {
        auto last_row = x.subview(static_cast<size_t>(prompt_len - 1u) * hidden, hidden);
        auto logits = ws(_logits).subview(0, vocab);
        if (auto s = record_lm_head(list, last_row, 1u, logits); !s.ok()) { return s; }
        list << logits.copy_to(logits_out);
        if (auto s = flush(list, stream); !s.ok()) { return s; }
    }
    return Status::success();
}

// ---------------------------------------------------------------------------
// single decode step
// ---------------------------------------------------------------------------

[[nodiscard]] Status MoeVlmExecutor::decode_step(Stream &stream, uint32_t input_token,
                                                 ContiguousKvCache &cache, luisa::span<float> logits_out) {
    const auto &d = _config.decoder.decoder;
    const uint32_t hidden = d.hidden_size, vocab = d.vocab_size;
    if (logits_out.size() != vocab) {
        return Status::error(StatusCode::invalid_argument, "logits_out size must equal vocab_size", "logits_out");
    }
    if (cache.config().batch != 1u) {
        return Status::error(StatusCode::unsupported, "MoE decode is batch 1 only", "batch");
    }
    const uint32_t pos = cache.length(0);
    if (pos == 0u) {
        return Status::error(StatusCode::invalid_argument, "decode step before prefill", "position");
    }
    if (pos >= cache.config().max_seq_len) {
        return Status::error(StatusCode::out_of_memory, "decode position reached KV capacity", "position");
    }
    if (input_token >= vocab) {
        return Status::error(StatusCode::invalid_argument, "decode input token outside the vocabulary", "input_token");
    }
    _last_rows = 1;
    _last_slots = _config.decoder.top_k;
    // Device metadata BEFORE the blocks: kv_len = pos+1, position = pos.
    const luisa::vector<uint32_t> meta_kv_len = {pos + 1u};
    const luisa::vector<int32_t> meta_positions = {static_cast<int32_t>(pos)};
    const luisa::vector<uint32_t> meta_step = {_decode_step + 1u};
    const luisa::vector<uint32_t> meta_state = {static_cast<uint32_t>(DecodeSeqState::decoding)};
    auto list = CommandList::create();
    if (auto s = cache.state().record(list, meta_kv_len, meta_positions, meta_step, meta_state); !s.ok()) { return s; }
    UploadStagingGuard guard{&list, luisa::make_unique<RecordUploadStaging>(), &stream};
    auto *stage = guard.staging.get();
    stage->ids = {static_cast<int32_t>(input_token)};
    stage->positions = {static_cast<int32_t>(pos)};
    stage->axis = _axis_pending;
    auto ids_view = ws(_ids_tok).as<int32_t>().subview(0, 1);
    auto pos_view = ws(_positions).as<int32_t>().subview(0, 1);
    list << ids_view.copy_from(luisa::span<const int32_t>{stage->ids})
         << pos_view.copy_from(luisa::span<const int32_t>{stage->positions})
         << ws(_axis).as<uint32_t>().copy_from(luisa::span<const uint32_t>{stage->axis});
    auto emb = ws(_emb).subview(0, hidden);
    auto x = ws(_x).subview(0, hidden);
    {
        auto plan = plan_embedding_dispatch(1, hidden);
        if (!plan.ok()) { return plan.status; }
        if (auto cmd = dispatch_1d((*_sh_embedding)(_embed, ids_view, emb, 1u, hidden), *plan.value)) { list << std::move(cmd); }
    }
    {
        auto plan = plan_mm_scatter_dispatch(1, hidden);
        if (!plan.ok()) { return plan.status; }
        if (auto cmd = dispatch_1d((*_sh_mm_scatter)(emb, ws(_dummy_f), x, 1u, hidden, 0u, 0u), *plan.value)) { list << std::move(cmd); }
    }
    if (auto s = flush(list, stream); !s.ok()) { return s; }
    for (uint32_t l = 0; l < d.num_layers; l++) {
        if (auto s = record_moe_block(stream, l, x, 1u, pos, cache, true); !s.ok()) { return s; }
    }
    {
        auto logits = ws(_logits).subview(0, vocab);
        if (auto s = record_lm_head(list, x, 1u, logits); !s.ok()) { return s; }
        list << logits.copy_to(logits_out);
        if (auto s = flush(list, stream); !s.ok()) { return s; }
    }
    if (auto s = cache.append_tokens(0, 1u); !s.ok()) { return s; }
    _decode_step++;
    return Status::success();
}

// ---------------------------------------------------------------------------
// Full image -> vision -> MoE decode request.
// ---------------------------------------------------------------------------

[[nodiscard]] Result<MoeRunResult> MoeVlmExecutor::run(Stream &stream, const P7ImageInput &image,
                                                       luisa::span<const uint32_t> prompt_tokens, uint32_t mm_start,
                                                       const MoeDecodeOptions &options, const Tokenizer *tokenizer) {
    using R = Result<MoeRunResult>;
    if (_device == nullptr) { return R::failure(StatusCode::invalid_argument, "moe executor not created", "device"); }
    const auto &d = _config.decoder.decoder;
    const uint32_t vocab = d.vocab_size;
    auto valid = options.validate(vocab);
    if (!valid.ok()) { return R::failure(valid); }
    const uint32_t nq = _config.vision.num_queries;
    if (mm_start + nq > prompt_tokens.size()) {
        return R::failure(StatusCode::invalid_argument, "visual rows exceed the prompt", "mm_start");
    }
    // Vision.
    MoeVisionOutput vision;
    {
        auto list = CommandList::create();
        auto vr = record_vision(list, image);
        if (!vr.ok()) { return R::failure(vr.status); }
        vision = *vr.value;
        stream << list.commit() << synchronize();
    }
    // Build the decode request with the visual features replacing the mm rows.
    DecodeRequest request;
    request.prompt_tokens.assign(prompt_tokens.begin(), prompt_tokens.end());
    request.mm_features = vision.features;
    request.mm_start = mm_start;
    request.mm_count = nq;
    // Cache.
    ContiguousKvCache cache;
    if (auto s = create_cache(*_device, stream, _config.max_seq_len, cache); !s.ok()) { return R::failure(s); }
    MoeRunResult result;
    result.visual_tokens = nq;
    result.weight_bytes = _weight_bytes;
    result.workspace_bytes = workspace_bytes();
    result.cache_bytes = cache.device_bytes();
    // Prefill.
    luisa::vector<float> logits(vocab);
    {
        auto s = prefill(stream, request, cache, logits);
        if (!s.ok()) { cache.release(0); return R::failure(s); }
    }
    // EOS ids = decoder config + options + tokenizer eos (deduplicated).
    luisa::vector<uint32_t> eos_ids;
    const auto add_eos = [&eos_ids](uint32_t id) {
        if (std::find(eos_ids.begin(), eos_ids.end(), id) == eos_ids.end()) { eos_ids.push_back(id); }
    };
    for (auto id : d.eos_token_ids) { add_eos(id); }
    for (auto id : options.eos_ids) { add_eos(id); }
    if (tokenizer != nullptr && tokenizer->eos_id().has_value()) { add_eos(*tokenizer->eos_id()); }
    const auto is_eos = [&eos_ids](uint32_t id) {
        return std::find(eos_ids.begin(), eos_ids.end(), id) != eos_ids.end();
    };
    luisa::optional<IncrementalDetokenizer> detok;
    if (tokenizer != nullptr) {
        DetokenizerOptions dopts;
        dopts.skip_special_tokens = options.skip_special_tokens;
        dopts.stop_strings = options.stop_strings;
        dopts.eos_ids = eos_ids;
        detok.emplace(*tokenizer, std::move(dopts));
    }
    DeterministicSampler sampler{options.sampling, vocab, prompt_tokens};
    for (uint32_t step = 0; step < options.max_new_tokens; step++) {
        auto picked = sampler.sample(logits);
        if (!picked.ok()) { cache.release(0); return R::failure(picked.status); }
        const uint32_t token = *picked.value;
        result.decode.tokens.push_back(token);
        result.decode.steps = step + 1u;
        if (is_eos(token)) { result.decode.hit_eos = true; break; }
        if (detok.has_value()) {
            auto text = detok->push(token);
            if (!text.ok()) { cache.release(0); return R::failure(text.status); }
            result.decode.text += *text.value;
            if (detok->stopped()) { result.decode.hit_stop_string = true; break; }
        }
        if (cache.length(0) >= cache.config().max_seq_len) { result.decode.truncated = true; break; }
        if (step + 1u == options.max_new_tokens) { result.decode.truncated = true; break; }
        if (auto s = decode_step(stream, token, cache, logits); !s.ok()) { cache.release(0); return R::failure(s); }
    }
    if (detok.has_value() && !detok->finished()) {
        auto tail = detok->finish();
        if (!tail.ok()) { cache.release(0); return R::failure(tail.status); }
        result.decode.text += *tail.value;
    }
    if (result.decode.truncated) {
        result.decode.diagnostics.push_back(Diagnostic{
            DiagnosticSeverity::warning, "output_truncated",
            luisa::format("generation stopped at the {}-token output budget before EOS/stop", options.max_new_tokens)});
    }
    // Record finished device state + release the slot.
    {
        auto list = CommandList::create();
        if (auto s = cache.record_finished(list, 0); !s.ok()) { cache.release(0); return R::failure(s); }
        stream << list.commit() << synchronize();
    }
    cache.release(0);
    return R::success(std::move(result));
}

// ---------------------------------------------------------------------------
// Layered-golden download helpers (caller drains the stream afterwards).
// ---------------------------------------------------------------------------

[[nodiscard]] Status MoeVlmExecutor::download_vision_pixels(Stream &stream, luisa::vector<float> &out) const {
    if (_last_pixel_elems == 0u) {
        return Status::error(StatusCode::invalid_argument, "no vision pixels uploaded yet", "pixels");
    }
    out.resize(static_cast<size_t>(_last_pixel_elems));
    stream << ws(_pixels).subview(0, static_cast<size_t>(_last_pixel_elems)).copy_to(luisa::span<float>{out});
    return Status::success();
}

[[nodiscard]] Status MoeVlmExecutor::download_vision_features(Stream &stream, luisa::vector<float> &out) const {
    const uint32_t nq = _config.vision.num_queries, th = _config.decoder.decoder.hidden_size;
    out.resize(static_cast<size_t>(nq) * th);
    stream << ws(_features).subview(0, out.size()).copy_to(luisa::span<float>{out});
    return Status::success();
}
[[nodiscard]] Status MoeVlmExecutor::download_router_logits(Stream &stream, uint32_t layer, luisa::vector<float> &out) const {
    if (layer >= _moe.size()) { return Status::error(StatusCode::invalid_argument, "layer out of range", "layer"); }
    const uint64_t n = static_cast<uint64_t>(_last_rows) * _config.decoder.num_experts;
    out.resize(static_cast<size_t>(n));
    stream << ws(_moe[layer].router_logits).subview(0, static_cast<size_t>(n)).copy_to(luisa::span<float>{out});
    return Status::success();
}
[[nodiscard]] Status MoeVlmExecutor::download_router_probs(Stream &stream, uint32_t layer, luisa::vector<float> &out) const {
    if (layer >= _moe.size()) { return Status::error(StatusCode::invalid_argument, "layer out of range", "layer"); }
    const uint64_t n = static_cast<uint64_t>(_last_rows) * _config.decoder.num_experts;
    out.resize(static_cast<size_t>(n));
    stream << ws(_moe[layer].probs).subview(0, static_cast<size_t>(n)).copy_to(luisa::span<float>{out});
    return Status::success();
}
[[nodiscard]] Status MoeVlmExecutor::download_selected(Stream &stream, uint32_t layer,
                                                       luisa::vector<uint32_t> &ids, luisa::vector<float> &weights) const {
    if (layer >= _moe.size()) { return Status::error(StatusCode::invalid_argument, "layer out of range", "layer"); }
    const uint64_t n = static_cast<uint64_t>(_last_rows) * _config.decoder.top_k;
    ids.resize(static_cast<size_t>(n));
    weights.resize(static_cast<size_t>(n));
    stream << ws(_moe[layer].ids).as<uint32_t>().subview(0, static_cast<size_t>(n)).copy_to(luisa::span<uint32_t>{ids})
           << ws(_moe[layer].weights).subview(0, static_cast<size_t>(n)).copy_to(luisa::span<float>{weights});
    return Status::success();
}
[[nodiscard]] Status MoeVlmExecutor::download_permutation(Stream &stream, uint32_t layer,
                                                          luisa::vector<uint32_t> &counts, luisa::vector<uint32_t> &perm_token,
                                                          luisa::vector<uint32_t> &slot_pos) const {
    if (layer >= _moe.size()) { return Status::error(StatusCode::invalid_argument, "layer out of range", "layer"); }
    const uint32_t ne = _config.decoder.num_experts;
    const uint64_t n = static_cast<uint64_t>(_last_rows) * _config.decoder.top_k;
    counts.resize(ne);
    perm_token.resize(static_cast<size_t>(n));
    slot_pos.resize(static_cast<size_t>(n));
    stream << ws(_moe[layer].counts).as<uint32_t>().subview(0, ne).copy_to(luisa::span<uint32_t>{counts})
           << ws(_moe[layer].perm_token).as<uint32_t>().subview(0, static_cast<size_t>(n)).copy_to(luisa::span<uint32_t>{perm_token})
           << ws(_moe[layer].slot_pos).as<uint32_t>().subview(0, static_cast<size_t>(n)).copy_to(luisa::span<uint32_t>{slot_pos});
    return Status::success();
}
[[nodiscard]] Status MoeVlmExecutor::download_expert_out(Stream &stream, uint32_t layer, luisa::vector<float> &out) const {
    if (layer >= _moe.size()) { return Status::error(StatusCode::invalid_argument, "layer out of range", "layer"); }
    const uint64_t n = static_cast<uint64_t>(_last_rows) * _config.decoder.top_k * _config.decoder.decoder.hidden_size;
    out.resize(static_cast<size_t>(n));
    stream << ws(_moe[layer].expert_out).subview(0, static_cast<size_t>(n)).copy_to(luisa::span<float>{out});
    return Status::success();
}
[[nodiscard]] Status MoeVlmExecutor::download_shared(Stream &stream, uint32_t layer,
                                                     luisa::vector<float> &shared0, luisa::vector<float> &shared1) const {
    if (layer >= _moe.size()) { return Status::error(StatusCode::invalid_argument, "layer out of range", "layer"); }
    const uint64_t n = static_cast<uint64_t>(_last_rows) * _config.decoder.decoder.hidden_size;
    shared0.resize(static_cast<size_t>(n));
    shared1.resize(static_cast<size_t>(n));
    stream << ws(_moe[layer].shared0).subview(0, static_cast<size_t>(n)).copy_to(luisa::span<float>{shared0})
           << ws(_moe[layer].shared1).subview(0, static_cast<size_t>(n)).copy_to(luisa::span<float>{shared1});
    return Status::success();
}
[[nodiscard]] Status MoeVlmExecutor::download_moe_out(Stream &stream, uint32_t layer, luisa::vector<float> &out) const {
    if (layer >= _moe.size()) { return Status::error(StatusCode::invalid_argument, "layer out of range", "layer"); }
    const uint64_t n = static_cast<uint64_t>(_last_rows) * _config.decoder.decoder.hidden_size;
    out.resize(static_cast<size_t>(n));
    stream << ws(_moe[layer].moe_out).subview(0, static_cast<size_t>(n)).copy_to(luisa::span<float>{out});
    return Status::success();
}
[[nodiscard]] Status MoeVlmExecutor::download_visual_query_mask(Stream &stream, luisa::vector<uint8_t> &out) const {
    if (_last_encoder_rows == 0u || _enc_mask_bytes == 0u) {
        return Status::error(StatusCode::invalid_argument, "no vision run recorded a visual/query mask", "mask");
    }
    // The device buffer holds packed uint32 words (4 mask bytes per word, low
    // byte first). Download the words and unpack exactly _enc_mask_bytes bytes.
    const uint32_t words = (_enc_mask_bytes + 3u) / 4u;
    luisa::vector<uint32_t> packed(words, 0u);
    stream << ws(_enc_mask).as<uint32_t>().subview(0, words).copy_to(luisa::span<uint32_t>{packed});
    out.resize(_enc_mask_bytes);
    for (uint32_t i = 0; i < _enc_mask_bytes; i++) {
        out[i] = static_cast<uint8_t>((packed[i / 4u] >> (8u * (i % 4u))) & 0xFFu);
    }
    return Status::success();
}

[[nodiscard]] Status MoeVlmExecutor::download_encoder_attn(Stream &stream, uint32_t layer, luisa::vector<float> &out) const {
    if (layer >= _enc_attn_out.size()) { return Status::error(StatusCode::invalid_argument, "encoder layer out of range", "layer"); }
    // Encoder rows = visual_after_down + num_queries; recovered from the last
    // vision run via the feature count is not stored, so size by the max bound
    // and let the caller trim. We store the actual encoder row count separately.
    const uint64_t n = static_cast<uint64_t>(_last_encoder_rows) * _config.vision.encoder_hidden();
    out.resize(static_cast<size_t>(n));
    stream << ws(_enc_attn_out[layer]).subview(0, static_cast<size_t>(n)).copy_to(luisa::span<float>{out});
    return Status::success();
}

}// namespace luisa::ocr

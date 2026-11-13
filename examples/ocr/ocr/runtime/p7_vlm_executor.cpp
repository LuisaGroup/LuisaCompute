// OCR engine -- shared Phase 7 vertical-slice executor. See p7_vlm_executor.h.
// Correctness-first FP32 graph; SYNTHETIC contracts only.
#include "ocr/runtime/p7_vlm_executor.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <utility>

#include <luisa/core/clock.h>
#include <luisa/core/logging.h>
#include <luisa/core/stl/format.h>
#include <luisa/core/stl/unordered_map.h>

#include "ocr/runtime/dispatch_adapter.h"

namespace luisa::ocr {

using namespace luisa::compute;

// Host preprocessing forward declaration (the host-only bridge lives in
// p7_vlm_preprocess.cpp; this TU never includes preprocess.h/api.h).
[[nodiscard]] Result<luisa::vector<float>> p7_preprocess_image_host(
    const P7ImageInput &image, luisa::span<const float> mean, luisa::span<const float> std_dev);

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

// Per-record heap-owned upload staging (pixels / positions / axis / mask).
// BufferUploadCommand borrows the host pointer, so the staging must outlive
// command EXECUTION; a completion callback releases it.
struct VisionUploadStaging {
    luisa::vector<float> pixels;
    luisa::vector<int32_t> positions;
    luisa::vector<uint32_t> axis;
    // Stable per-image window/global masks (one borrowed buffer each; the
    // per-layer uploads reference these, never a reused scratch vector).
    luisa::vector<luisa::vector<uint32_t>> window_masks;
    luisa::vector<luisa::vector<uint32_t>> global_masks;
    // Stable PER-IMAGE learned absolute position ids. Each image gets its own
    // vector retained through completion; a single reused vector would dangle
    // / reallocate across the per-image uploads (defect: multi-image OOB).
    luisa::vector<luisa::vector<int32_t>> pos_ids;
    // Additive Phase 8: stable PER-IMAGE PER-LAYER expanded relative-position
    // bias buffers [heads, rows, rows]. Each upload borrows its own buffer.
    luisa::vector<luisa::vector<luisa::vector<float>>> rel_bias;
};

// Build the window/global custom block mask for ONE image. GLOBAL layers use
// a single all-ones block (every query sees every key). WINDOW layers use
// block_size 1 (per-patch granularity) so a query attends EXACTLY the patches
// of its own spatial window x window tile -- no cross-window leakage at block
// edges. Returns the packed uint32 words + the block size.
struct WindowMask {
    luisa::vector<uint32_t> words;
    uint32_t block_size{0};
};
[[nodiscard]] WindowMask build_window_mask(uint32_t rows, uint32_t grid_w, uint32_t window, bool global) {
    WindowMask m;
    if (global) {
        m.block_size = rows == 0u ? 1u : rows;
        luisa::vector<uint8_t> mask(1u, 1u);
        m.words = pack_block_mask_words(mask);
        return m;
    }
    m.block_size = 1u;
    const uint32_t tiles_w = grid_w == 0u ? 1u : (grid_w + window - 1u) / window;
    luisa::vector<uint32_t> tile_of(rows);
    for (uint32_t t = 0; t < rows; t++) {
        const uint32_t ph = grid_w == 0u ? 0u : t / grid_w;
        const uint32_t pw = grid_w == 0u ? 0u : t % grid_w;
        tile_of[t] = (ph / window) * tiles_w + (pw / window);
    }
    luisa::vector<uint8_t> mask(static_cast<size_t>(rows) * rows, 0u);
    for (uint32_t q = 0; q < rows; q++) {
        for (uint32_t k = 0; k < rows; k++) {
            mask[static_cast<size_t>(q) * rows + k] = (tile_of[q] == tile_of[k]) ? 1u : 0u;
        }
    }
    m.words = pack_block_mask_words(mask);
    return m;
}

// Additive Phase 8: expand a decomposed relative-position bias table
// [heads, side, side] (side = 2*window-1) into the full per-head bias
// [heads, rows, rows] indexed by the patch grid delta. This is the ViTDet/SAM
// convention: bias[h, q, k] = table[h, clamp(qh-kh)+window-1, clamp(qw-kw)+window-1].
// Independent host math; the device attention adds it to the scores.
[[nodiscard]] luisa::vector<float> expand_rel_pos_bias(luisa::span<const float> table, uint32_t heads,
                                                       uint32_t window, uint32_t grid_h, uint32_t grid_w) {
    const uint32_t rows = grid_h * grid_w;
    const uint32_t side = 2u * window - 1u;
    luisa::vector<float> out(static_cast<size_t>(heads) * rows * rows, 0.0f);
    const auto clamp_idx = [side, window](int32_t d) -> uint32_t {
        const int32_t v = d + static_cast<int32_t>(window) - 1;
        return static_cast<uint32_t>(std::clamp(v, 0, static_cast<int32_t>(side) - 1));
    };
    for (uint32_t h = 0; h < heads; h++) {
        for (uint32_t q = 0; q < rows; q++) {
            const int32_t qh = static_cast<int32_t>(grid_w == 0u ? 0u : q / grid_w);
            const int32_t qw = static_cast<int32_t>(grid_w == 0u ? 0u : q % grid_w);
            for (uint32_t k = 0; k < rows; k++) {
                const int32_t kh = static_cast<int32_t>(grid_w == 0u ? 0u : k / grid_w);
                const int32_t kw = static_cast<int32_t>(grid_w == 0u ? 0u : k % grid_w);
                const uint32_t dy = clamp_idx(qh - kh);
                const uint32_t dx = clamp_idx(qw - kw);
                out[(static_cast<size_t>(h) * rows + q) * rows + k] =
                    table[(static_cast<size_t>(h) * side + dy) * side + dx];
            }
        }
    }
    return out;
}

}// namespace

// ---------------------------------------------------------------------------
// create / resolve / reserve / compile
// ---------------------------------------------------------------------------

[[nodiscard]] Status P7VlmExecutor::create(Device &device, Stream &stream,
                                           const P7SliceConfig &config, const P7TensorNames &names,
                                           const WeightStore &weights) {
    return create_impl(device, stream, config, names, weights, true);
}

// Additive Phase 9: vision-only create for the hybrid GDN model slices. The
// vision tower + connector setup is IDENTICAL to create(); only the embedded
// dense text executor is skipped (the slice drives a HybridGdnExecutor instead).
[[nodiscard]] Status P7VlmExecutor::create_vision_only(Device &device, Stream &stream,
                                                       const P7SliceConfig &config, const P7TensorNames &names,
                                                       const WeightStore &weights) {
    return create_impl(device, stream, config, names, weights, false);
}

[[nodiscard]] Status P7VlmExecutor::create_impl(Device &device, Stream &stream,
                                                const P7SliceConfig &config, const P7TensorNames &names,
                                                const WeightStore &weights, bool with_text) {
    auto valid = config.validate();
    if (!valid.ok()) { return valid; }
    valid = names.validate();
    if (!valid.ok()) { return valid; }
    _config = config;
    _names = names;
    _device = &device;
    _vision_ws = DeviceWorkspace{device};
    _patchify_cache = TypedShaderCache<ImagePatchifyKernel>{device, _shader_registry};
    _layer_norm_cache = TypedShaderCache<LayerNormKernel>{device, _shader_registry};
    _rms_norm_cache = TypedShaderCache<RmsNormKernel>{device, _shader_registry};
    _gemm_cache = TypedShaderCache<GemmKernel>{device, _shader_registry};
    _rope2d_cache = TypedShaderCache<RopeAxialKernel>{device, _shader_registry};
    _attn_cache = TypedShaderCache<AttentionPrefillKernel>{device, _shader_registry};
    _merge_cache = TypedShaderCache<SpatialMergeKernel>{device, _shader_registry};
    _unshuffle_cache = TypedShaderCache<PixelUnshuffleKernel>{device, _shader_registry};
    _transpose_cache = TypedShaderCache<Transpose2dKernel>{device, _shader_registry};
    _pool_cache = TypedShaderCache<AdaptiveAvgPool2dKernel>{device, _shader_registry};
    _add_pos_cache = TypedShaderCache<AddPositionKernel>{device, _shader_registry};
    _residual_cache = TypedShaderCache<ResidualAddKernel>{device, _shader_registry};

    if (auto s = resolve_weights(weights); !s.ok()) { return s; }
    // Additive Phase 8: download the immutable relative-position bias tables
    // ONCE here (the create-time stream is alive and drained synchronously).
    // record_vision then expands them host-side per image grid; no borrowed
    // stream is retained past create.
    if (config.vision.rel_pos_window != 0u) {
        const uint64_t side = 2ull * config.vision.rel_pos_window - 1ull;
        const uint64_t table_elems = static_cast<uint64_t>(config.vision.num_heads) * side * side;
        _rel_pos_tables.resize(config.vision.num_layers);
        for (uint32_t l = 0; l < config.vision.num_layers; l++) {
            _rel_pos_tables[l].resize(static_cast<size_t>(table_elems));
            stream << _blocks[l].rel_pos_bias.copy_to(luisa::span<float>{_rel_pos_tables[l]}) << synchronize();
        }
    }
    if (auto s = reserve_workspace(); !s.ok()) { return s; }
    if (auto s = compile_shaders(); !s.ok()) { return s; }

    if (!with_text) {
        // Vision-only mode (Phase 9 hybrid slices): no dense text executor.
        _vision_axis_pending = p7_vision_axis_of_pair(config.vision);
        _weight_bytes = weights.component_bytes(config.vision_component);
        return Status::success();
    }

    // Embedded text executor with the model's explicit contract.
    DenseExecutorConfig text_config;
    text_config.decoder = config.text;
    text_config.names = names.text;
    text_config.tie_word_embeddings = config.tie_word_embeddings;
    text_config.rope_axes = config.text_rope_axes();
    text_config.mrope_axis_of_pair = config.text_axis_of_pair();
    text_config.partial_rotary_num = config.partial_rotary_num;
    text_config.partial_rotary_den = config.partial_rotary_den;
    // Text contract selection (explicit, never inferred from model_type):
    //  * Qwen3 (mrope_interleaved WITHOUT the Qwen2.5 switch): QK-Norm, no bias;
    //  * Qwen2.5-VL (mrope_interleaved WITH text_qwen2_5_contract, olmOCR-2):
    //    QKV bias, no QK-Norm, rotate-half M-RoPE;
    //  * every other Phase 7 contract: QKV bias, no QK-Norm.
    const bool qwen3_text = config.text_pos == P7TextPosKind::mrope_interleaved && !config.text_qwen2_5_contract;
    text_config.qk_norm = qwen3_text;
    text_config.qkv_bias = !qwen3_text;
    text_config.component = config.decoder_component;
    text_config.prefill_chunk_rows = config.prefill_chunk_rows;
    text_config.enable_fused_residual_rms_norm = config.enable_fused_residual_rms_norm;
    text_config.enable_cooperative_norm = config.enable_cooperative_norm;
    text_config.memory = config.memory;
    text_config.linear_storage = config.linear_storage;
    text_config.linear_group_size = config.linear_group_size;
    if (auto s = _text.create(device, stream, text_config, weights); !s.ok()) { return s; }

    _vision_axis_pending = p7_vision_axis_of_pair(config.vision);
    _weight_bytes = weights.component_bytes(config.vision_component) + _text.weight_bytes();
    return Status::success();
}

[[nodiscard]] Status P7VlmExecutor::resolve_weights(const WeightStore &weights) {
    const auto &comp = _config.vision_component;
    const auto &v = _config.vision;
    const auto full_expected = p7_expected_tensors(_config, _names);
    luisa::unordered_map<luisa::string, const ExpectedTensor *> expected_by_name;
    for (const auto &e : full_expected) {
        if (e.name.compare(0, _names.visual_prefix.size(), _names.visual_prefix) == 0) {
            expected_by_name.emplace(e.name, &e);
        }
    }
    for (const auto &name : weights.tensor_names(comp)) {
        if (expected_by_name.find(name) == expected_by_name.end()) {
            return Status::error(StatusCode::invalid_argument,
                                 luisa::format("unexpected vision tensor '{}' in component '{}'", name, comp), name);
        }
    }
    auto view_of = [&](const luisa::string &name, BufferView<float> &out) -> Status {
        const auto *w = weights.find(comp, name);
        if (w == nullptr) {
            return Status::error(StatusCode::not_found,
                                 luisa::format("vision weight '{}' not found in component '{}'", name, comp), name);
        }
        if (w->encoding != TensorEncoding::fp32 || !w->dense) {
            return Status::error(StatusCode::unsupported,
                                 luisa::format("vision weight '{}' must be dense fp32 in the Phase 7 slice", name), name);
        }
        const auto it = expected_by_name.find(name);
        if (it == expected_by_name.end()) {
            return Status::error(StatusCode::invalid_argument,
                                 luisa::format("vision weight '{}' has no expected-tensor contract", name), name);
        }
        if (auto s = validate_decoder_weight_desc(w->logical, w->element_count, *it->second); !s.ok()) { return s; }
        if (w->dense.size() < w->element_count) {
            return Status::error(StatusCode::invalid_argument,
                                 luisa::format("vision weight '{}' device buffer is smaller than its element count", name), name);
        }
        out = w->dense.view();
        return Status::success();
    };
    if (auto s = view_of(_names.patch_embed_weight(), _patch_embed_w); !s.ok()) { return s; }
    if (auto s = view_of(_names.patch_embed_bias(), _patch_embed_b); !s.ok()) { return s; }
    if (v.position_rows != 0u) {
        if (auto s = view_of(_names.position_embed(), _position_embed); !s.ok()) { return s; }
    }
    _blocks.resize(v.num_layers);
    for (uint32_t l = 0; l < v.num_layers; l++) {
        auto &B = _blocks[l];
        if (auto s = view_of(_names.block_norm1_weight(l), B.norm1_w); !s.ok()) { return s; }
        if (auto s = view_of(_names.block_norm1_bias(l), B.norm1_b); !s.ok()) { return s; }
        if (auto s = view_of(_names.block_norm2_weight(l), B.norm2_w); !s.ok()) { return s; }
        if (auto s = view_of(_names.block_norm2_bias(l), B.norm2_b); !s.ok()) { return s; }
        if (auto s = view_of(_names.block_q_proj(l), B.q_w); !s.ok()) { return s; }
        if (auto s = view_of(_names.block_k_proj(l), B.k_w); !s.ok()) { return s; }
        if (auto s = view_of(_names.block_v_proj(l), B.v_w); !s.ok()) { return s; }
        if (auto s = view_of(_names.block_o_proj(l), B.o_w); !s.ok()) { return s; }
        if (v.qkv_bias) {
            if (auto s = view_of(_names.block_q_bias(l), B.q_b); !s.ok()) { return s; }
            if (auto s = view_of(_names.block_k_bias(l), B.k_b); !s.ok()) { return s; }
            if (auto s = view_of(_names.block_v_bias(l), B.v_b); !s.ok()) { return s; }
            if (auto s = view_of(_names.block_o_bias(l), B.o_b); !s.ok()) { return s; }
        }
        if (v.qk_norm) {
            if (auto s = view_of(_names.block_q_norm(l), B.q_norm); !s.ok()) { return s; }
            if (auto s = view_of(_names.block_k_norm(l), B.k_norm); !s.ok()) { return s; }
        }
        if (auto s = view_of(_names.block_fc1(l), B.fc1_w); !s.ok()) { return s; }
        if (auto s = view_of(_names.block_fc1_bias(l), B.fc1_b); !s.ok()) { return s; }
        if (auto s = view_of(_names.block_fc2(l), B.fc2_w); !s.ok()) { return s; }
        if (auto s = view_of(_names.block_fc2_bias(l), B.fc2_b); !s.ok()) { return s; }
        if (v.rel_pos_window != 0u) {
            if (auto s = view_of(_names.block_rel_pos_bias(l), B.rel_pos_bias); !s.ok()) { return s; }
        }
    }
    if (auto s = view_of(_names.connector_ln_weight(), _connector.ln_w); !s.ok()) { return s; }
    if (!_config.connector.rms_norm) {
        if (auto s = view_of(_names.connector_ln_bias(), _connector.ln_b); !s.ok()) { return s; }
    }
    if (auto s = view_of(_names.connector_fc1(), _connector.fc1_w); !s.ok()) { return s; }
    if (auto s = view_of(_names.connector_fc1_bias(), _connector.fc1_b); !s.ok()) { return s; }
    if (auto s = view_of(_names.connector_fc2(), _connector.fc2_w); !s.ok()) { return s; }
    if (auto s = view_of(_names.connector_fc2_bias(), _connector.fc2_b); !s.ok()) { return s; }
    if (_config.connector.kind == P7ConnectorKind::downsample_mlp) {
        if (auto s = view_of(_names.connector_down_weight(), _connector.down_w); !s.ok()) { return s; }
        if (auto s = view_of(_names.connector_down_bias(), _connector.down_b); !s.ok()) { return s; }
    }
    return Status::success();
}

[[nodiscard]] Status P7VlmExecutor::reserve_workspace() {
    const auto &v = _config.vision;
    const auto &c = _config.connector;
    const uint64_t maxp = _config.max_vision_patches;
    const uint64_t vh = v.hidden_size, vi = v.intermediate_size;
    const uint64_t text_hidden = _config.text.hidden_size;
    // Per-image connector scratch bound (reduced/cnorm/cfc1/merged are reused
    // per image, so they are sized for the LARGEST SINGLE image, not the sum).
    uint64_t max_tokens = maxp;
    uint64_t reduced_dim = vh;
    uint64_t merged_dim = vh;// downsample_mlp merged scratch [tokens, vh*m*m]
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
    // The FEATURES arena accumulates EVERY image's connector output, so it is
    // sized for the TOTAL token bound. Each image emits <= its patch count
    // tokens (every connector reduces or preserves), so the sum over images is
    // bounded by the total patch budget maxp. This prevents a multi-image
    // record_vision from overflowing the features arena (defect: per-image
    // bound previously undersized it for total_tokens > per-image bound).
    const uint64_t max_total_tokens = maxp;
    const uint64_t aux_elems = vh + maxp * vh;
    // Window/global block mask bound: the mask is block-level. The largest
    // grid is at block_size 1 (window=1): maxp*maxp bytes -> ceil(/4) words.
    // Global masks are 1 word. Bound by the block_size-1 case.
    const uint64_t mask_words_bound = (maxp * maxp + 3u) / 4u + 1u;
    // Additive Phase 8: expanded relative-position bias [heads, maxp, maxp].
    // Only used when rel_pos_window != 0; sized for the worst-case patch grid.
    const uint64_t rel_bias_elems = v.rel_pos_window != 0u ? static_cast<uint64_t>(v.num_heads) * maxp * maxp : 1ull;

    struct RegionSpec { const char *name; uint64_t elements; };
    const RegionSpec specs[] = {
        {"pixels", maxp * v.patch_dim()},
        {"patches", maxp * v.patch_dim()},
        {"x", maxp * vh},
        {"norm", maxp * vh},
        {"q", maxp * vh},
        {"k", maxp * vh},
        {"v", maxp * vh},
        {"qn", maxp * vh},
        {"kn", maxp * vh},
        {"qr", maxp * vh},
        {"kr", maxp * vh},
        {"attn", maxp * vh},
        {"fc1", maxp * vi},
        {"reduced", max_tokens * reduced_dim},
        {"cnorm", max_tokens * reduced_dim},
        {"cfc1", max_tokens * c.intermediate_size},
        // Dedicated checked merged scratch for downsample_mlp [tokens, vh*m*m]
        // (the cnorm region is only vh wide for downsample, so it cannot hold
        // the m*m-expanded merged rows; defect: undersized by m*m).
        {"merged", max_tokens * merged_dim},
        {"features", max_total_tokens * text_hidden},
        {"positions", maxp * 2u},
        {"axis", v.head_dim / 2u},
        {"block_mask", mask_words_bound},
        {"dummy_f", 1},
        {"dummy_u", 1},
        {"aux", aux_elems},
        // Pixel-unshuffle input transpose scratch [vh, maxp] (channel-major).
        {"unshuffle_in", vh * maxp},
        // Additive Phase 8: expanded relative-position bias scratch.
        {"rel_bias", rel_bias_elems},
    };
    WorkspacePlan plan;
    for (const auto &spec : specs) {
        auto bytes = f32_bytes(spec.elements);
        if (!bytes.ok()) { return bytes.status; }
        if (auto s = plan.add_bytes(spec.name, *bytes.value); !s.ok()) { return s; }
    }
    MemoryBudget budget{_config.memory};
    if (auto s = budget.require("p7_vision_workspace", MemoryCategory::workspace, plan.total_bytes()); !s.ok()) { return s; }
    if (auto s = budget.preflight(); !s.ok()) { return s; }
    if (auto s = _vision_ws.reserve(plan.total_bytes()); !s.ok()) { return s; }

    WorkspaceAllocation *outs[] = {
        &_v_pixels, &_v_patches, &_v_x, &_v_norm, &_v_q, &_v_k, &_v_v, &_v_qn, &_v_kn,
        &_v_qr, &_v_kr, &_v_attn, &_v_fc1, &_v_reduced, &_v_cnorm, &_v_cfc1, &_v_merged, &_v_features,
        &_v_positions, &_v_axis, &_v_block_mask, &_v_dummy_f, &_v_dummy_u, &_v_aux,
        &_v_unshuffle, &_v_rel_bias,
    };
    for (size_t i = 0; i < std::size(specs); i++) {
        if (auto s = add_region(_vision_ws, *outs[i], specs[i].elements); !s.ok()) { return s; }
    }
    return Status::success();
}

[[nodiscard]] Status P7VlmExecutor::compile_shaders() {
    const auto get = [](auto &cache, const luisa::string &key, auto factory, auto *&out) -> Status {
        auto r = cache.get_or_compile(key, factory);
        if (!r.ok()) { return r.status; }
        out = *r.value;
        return Status::success();
    };
    if (auto s = get(_patchify_cache, image_variant_key("patchify"),
                     [] { return make_image_patchify_kernel(); }, _sh_patchify); !s.ok()) { return s; }
    if (auto s = get(_layer_norm_cache, norm_variant_key("layer_norm"),
                     [] { return make_layer_norm_kernel(); }, _sh_layer_norm); !s.ok()) { return s; }
    if (auto s = get(_rms_norm_cache, norm_variant_key("rms_norm"),
                     [] { return make_rms_norm_kernel(); }, _sh_rms_norm); !s.ok()) { return s; }
    if (auto s = get(_gemm_cache, gemm_variant_key(GemmEpilogue::none, GemmWeightFormat::fp32, false),
                     [] { return make_gemm_kernel(GemmEpilogue::none); }, _sh_gemm_none); !s.ok()) { return s; }
    if (auto s = get(_gemm_cache, gemm_variant_key(GemmEpilogue::residual, GemmWeightFormat::fp32, false),
                     [] { return make_gemm_kernel(GemmEpilogue::residual); }, _sh_gemm_residual); !s.ok()) { return s; }
    if (auto s = get(_gemm_cache, gemm_variant_key(GemmEpilogue::bias, GemmWeightFormat::fp32, false),
                     [] { return make_gemm_kernel(GemmEpilogue::bias); }, _sh_gemm_bias); !s.ok()) { return s; }
    if (auto s = get(_gemm_cache, gemm_variant_key(GemmEpilogue::bias_gelu_erf, GemmWeightFormat::fp32, false),
                     [] { return make_gemm_kernel(GemmEpilogue::bias_gelu_erf); }, _sh_gemm_bias_gelu); !s.ok()) { return s; }
    if (auto s = get(_gemm_cache, gemm_variant_key(GemmEpilogue::bias_silu, GemmWeightFormat::fp32, false),
                     [] { return make_gemm_kernel(GemmEpilogue::bias_silu); }, _sh_gemm_bias_silu); !s.ok()) { return s; }
    if (auto s = get(_gemm_cache, gemm_variant_key(GemmEpilogue::bias_residual, GemmWeightFormat::fp32, false),
                     [] { return make_gemm_kernel(GemmEpilogue::bias_residual); }, _sh_gemm_bias_residual); !s.ok()) { return s; }
    if (auto s = get(_rope2d_cache, rope_variant_key("2d"),
                     [] { return make_rope_2d_kernel(); }, _sh_rope2d); !s.ok()) { return s; }
    if (auto s = get(_attn_cache, attention_variant_key("prefill_reference"),
                     [] { return make_attention_prefill_reference_kernel(); }, _sh_attn); !s.ok()) { return s; }
    if (auto s = get(_merge_cache, vision_variant_key("spatial_merge"),
                     [] { return make_spatial_merge_kernel(); }, _sh_merge); !s.ok()) { return s; }
    if (auto s = get(_unshuffle_cache, vision_variant_key("pixel_unshuffle"),
                     [] { return make_pixel_unshuffle_kernel(); }, _sh_unshuffle); !s.ok()) { return s; }
    if (auto s = get(_transpose_cache, vision_variant_key("transpose_2d"),
                     [] { return make_transpose_2d_kernel(); }, _sh_transpose); !s.ok()) { return s; }
    if (auto s = get(_pool_cache, "ocr_conv_adaptive_avg_pool2d",
                     [] { return make_adaptive_avg_pool2d_kernel(); }, _sh_pool); !s.ok()) { return s; }
    if (auto s = get(_add_pos_cache, vision_variant_key("add_position"),
                     [] { return make_add_position_kernel(); }, _sh_add_pos); !s.ok()) { return s; }
    if (auto s = get(_residual_cache, norm_variant_key("residual_add"),
                     [] { return make_residual_add_kernel(); }, _sh_residual); !s.ok()) { return s; }
    return Status::success();
}

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

[[nodiscard]] Status P7VlmExecutor::record_linear(CommandList &list, BufferView<float> A,
                                                  BufferView<float> B, BufferView<float> C,
                                                  BufferView<float> aux, uint32_t m, uint32_t n, uint32_t k,
                                                  GemmEpilogue epilogue) {
    if (m == 0u || n == 0u || k == 0u) {
        return Status::error(StatusCode::invalid_argument, "linear dimensions must be nonzero", "m/n/k");
    }
    const auto aux_view = epilogue == GemmEpilogue::none ? vws(_v_dummy_f) : aux;
    auto plan = plan_gemm_dispatch_epilogue(m, n, k, epilogue);
    if (!plan.ok()) { return plan.status; }
    const auto *shader = epilogue == GemmEpilogue::bias ? _sh_gemm_bias
                         : epilogue == GemmEpilogue::residual ? _sh_gemm_residual
                         : epilogue == GemmEpilogue::bias_gelu_erf ? _sh_gemm_bias_gelu
                         : epilogue == GemmEpilogue::bias_silu ? _sh_gemm_bias_silu
                         : epilogue == GemmEpilogue::bias_residual ? _sh_gemm_bias_residual
                                                                   : _sh_gemm_none;
    if (auto cmd = dispatch_2d((*shader)(A, B, C, aux_view, m, n, k, plan.value->padded_cols), *plan.value)) {
        list << std::move(cmd);
    }
    return Status::success();
}

[[nodiscard]] Status P7VlmExecutor::copy_rows(CommandList &list, BufferView<float> src,
                                              BufferView<float> dst, uint32_t rows, uint32_t hidden) {
    auto plan = plan_spatial_merge_dispatch(rows, hidden);
    if (!plan.ok()) { return plan.status; }
    if (auto cmd = dispatch_1d((*_sh_merge)(src, dst, rows, hidden, hidden, 1u), *plan.value)) {
        list << std::move(cmd);
    }
    return Status::success();
}

// ---------------------------------------------------------------------------
// vision block
// ---------------------------------------------------------------------------

[[nodiscard]] Status P7VlmExecutor::record_vision_block(CommandList &list, uint32_t layer,
                                                        BufferView<float> x, BufferView<int32_t> positions,
                                                        uint32_t rows, const P7PackedImage &packed) {
    const auto &v = _config.vision;
    const auto &B = _blocks[layer];
    const uint32_t vh = v.hidden_size, vi = v.intermediate_size;
    const uint32_t heads = v.num_heads, hd = v.head_dim;
    const float eps = v.layer_norm_eps;
    const float theta = v.rope_theta;
    const float scale = 1.0f / std::sqrt(static_cast<float>(hd));
    auto norm = vws(_v_norm).subview(0, static_cast<size_t>(rows) * vh);
    auto q = vws(_v_q).subview(0, static_cast<size_t>(rows) * vh);
    auto k = vws(_v_k).subview(0, static_cast<size_t>(rows) * vh);
    auto vv = vws(_v_v).subview(0, static_cast<size_t>(rows) * vh);
    auto qr = vws(_v_qr).subview(0, static_cast<size_t>(rows) * vh);
    auto kr = vws(_v_kr).subview(0, static_cast<size_t>(rows) * vh);
    auto attn = vws(_v_attn).subview(0, static_cast<size_t>(rows) * vh);
    auto fc1 = vws(_v_fc1).subview(0, static_cast<size_t>(rows) * vi);
    auto axis = vws(_v_axis).as<uint32_t>();
    auto dummy_f = vws(_v_dummy_f);
    auto dummy_u = vws(_v_dummy_u).as<uint32_t>();

    // 1. Pre-attention LayerNorm.
    {
        auto plan = plan_norm_dispatch(rows);
        if (!plan.ok()) { return plan.status; }
        if (auto cmd = dispatch_1d((*_sh_layer_norm)(x, norm, B.norm1_w, B.norm1_b, rows, vh, eps, 1u), *plan.value)) {
            list << std::move(cmd);
        }
    }
    // 2. QKV projections (bias per contract).
    const auto qkv_epi = v.qkv_bias ? GemmEpilogue::bias : GemmEpilogue::none;
    if (auto s = record_linear(list, norm, B.q_w, q, B.q_b, rows, vh, vh, qkv_epi); !s.ok()) { return s; }
    if (auto s = record_linear(list, norm, B.k_w, k, B.k_b, rows, vh, vh, qkv_epi); !s.ok()) { return s; }
    if (auto s = record_linear(list, norm, B.v_w, vv, B.v_b, rows, vh, vh, qkv_epi); !s.ok()) { return s; }
    // 3. Optional QK-Norm (per-head RMSNorm over head_dim).
    auto q_for_rope = q;
    auto k_for_rope = k;
    if (v.qk_norm) {
        auto qn = vws(_v_qn).subview(0, static_cast<size_t>(rows) * vh);
        auto kn = vws(_v_kn).subview(0, static_cast<size_t>(rows) * vh);
        auto plan_q = plan_norm_dispatch(static_cast<uint64_t>(rows) * heads);
        if (!plan_q.ok()) { return plan_q.status; }
        if (auto cmd = dispatch_1d((*_sh_rms_norm)(q, qn, B.q_norm, rows * heads, hd, eps, 1u), *plan_q.value)) {
            list << std::move(cmd);
        }
        auto plan_k = plan_norm_dispatch(static_cast<uint64_t>(rows) * heads);
        if (!plan_k.ok()) { return plan_k.status; }
        if (auto cmd = dispatch_1d((*_sh_rms_norm)(k, kn, B.k_norm, rows * heads, hd, eps, 1u), *plan_k.value)) {
            list << std::move(cmd);
        }
        q_for_rope = qn;
        k_for_rope = kn;
    }
    // 4. Optional 2D axial vision RoPE.
    auto q_for_attn = q_for_rope;
    auto k_for_attn = k_for_rope;
    if (v.rope_kind == P7VisionRopeKind::axial2d) {
        auto plan_q = plan_rope_dispatch(rows, heads);
        if (!plan_q.ok()) { return plan_q.status; }
        if (auto cmd = dispatch_1d((*_sh_rope2d)(q_for_rope, qr, positions, axis, rows, heads, hd, 2u, theta), *plan_q.value)) {
            list << std::move(cmd);
        }
        auto plan_k = plan_rope_dispatch(rows, heads);
        if (!plan_k.ok()) { return plan_k.status; }
        if (auto cmd = dispatch_1d((*_sh_rope2d)(k_for_rope, kr, positions, axis, rows, heads, hd, 2u, theta), *plan_k.value)) {
            list << std::move(cmd);
        }
        q_for_attn = qr;
        k_for_attn = kr;
    }
    // 5. Masked bidirectional attention WITHIN this image.
    {
        uint32_t mask_kind = static_cast<uint32_t>(AttentionMaskKind::bidirectional);
        uint32_t block_size = 0u;
        auto mask_view = dummy_u;
        if (v.mask_kind == P7VisionMaskKind::window_global) {
            const bool global = std::find(v.global_layers.begin(), v.global_layers.end(), layer) != v.global_layers.end();
            // The mask was uploaded by record_vision for THIS layer; the view
            // is bound there. Here we only select the kind/granularity.
            mask_kind = static_cast<uint32_t>(AttentionMaskKind::custom_block_mask);
            // Window layers use per-patch (block_size 1) masks; global layers
            // use a single all-ones block (block_size rows).
            block_size = global ? rows : 1u;
            mask_view = vws(_v_block_mask).as<uint32_t>();
        }
        // Additive Phase 8: decomposed relative-position bias. record_vision
        // expanded the per-layer table into _v_rel_bias [heads, rows, rows].
        auto rel_bias_view = dummy_f;
        uint32_t has_rel_bias = 0u;
        if (v.rel_pos_window != 0u) {
            rel_bias_view = vws(_v_rel_bias).subview(0, static_cast<size_t>(heads) * rows * rows);
            has_rel_bias = 1u;
        }
        auto plan = plan_attention_prefill_dispatch(rows, heads);
        if (!plan.ok()) { return plan.status; }
        if (auto cmd = dispatch_1d((*_sh_attn)(q_for_attn, k_for_attn, vv, attn, mask_view, rel_bias_view,
                                               rows, rows, heads, heads, hd, scale, 0.0f,
                                               mask_kind, 0u, 0u, block_size, has_rel_bias), *plan.value)) {
            list << std::move(cmd);
        }
    }
    // 6. Output projection (+bias) + residual.
    if (v.qkv_bias) {
        auto aux = vws(_v_aux).subview(0, static_cast<size_t>(vh) + static_cast<size_t>(rows) * vh);
        if (auto s = copy_rows(list, B.o_b, aux.subview(0, vh), 1u, vh); !s.ok()) { return s; }
        if (auto s = copy_rows(list, x, aux.subview(vh, static_cast<size_t>(rows) * vh), rows, vh); !s.ok()) { return s; }
        if (auto s = record_linear(list, attn, B.o_w, x, aux, rows, vh, vh, GemmEpilogue::bias_residual); !s.ok()) { return s; }
    } else {
        if (auto s = record_linear(list, attn, B.o_w, x, x, rows, vh, vh, GemmEpilogue::residual); !s.ok()) { return s; }
    }
    // 7. Post-attention LayerNorm.
    {
        auto plan = plan_norm_dispatch(rows);
        if (!plan.ok()) { return plan.status; }
        if (auto cmd = dispatch_1d((*_sh_layer_norm)(x, norm, B.norm2_w, B.norm2_b, rows, vh, eps, 1u), *plan.value)) {
            list << std::move(cmd);
        }
    }
    // 8. MLP: fc1 (+bias, activation), fc2 (+bias) + residual.
    const auto act_epi = v.mlp_act == 1u ? GemmEpilogue::bias_silu : GemmEpilogue::bias_gelu_erf;
    if (auto s = record_linear(list, norm, B.fc1_w, fc1, B.fc1_b, rows, vi, vh, act_epi); !s.ok()) { return s; }
    {
        auto aux = vws(_v_aux).subview(0, static_cast<size_t>(vh) + static_cast<size_t>(rows) * vh);
        if (auto s = copy_rows(list, B.fc2_b, aux.subview(0, vh), 1u, vh); !s.ok()) { return s; }
        if (auto s = copy_rows(list, x, aux.subview(vh, static_cast<size_t>(rows) * vh), rows, vh); !s.ok()) { return s; }
        if (auto s = record_linear(list, fc1, B.fc2_w, x, aux, rows, vh, vi, GemmEpilogue::bias_residual); !s.ok()) { return s; }
    }
    return Status::success();
}

// ---------------------------------------------------------------------------
// connector
// ---------------------------------------------------------------------------

[[nodiscard]] Status P7VlmExecutor::record_connector(CommandList &list, BufferView<float> x,
                                                     const P7PackedImage &packed, BufferView<float> out) {
    const auto &v = _config.vision;
    const auto &c = _config.connector;
    const uint32_t vh = v.hidden_size;
    const uint32_t text_hidden = _config.text.hidden_size;
    const uint32_t ci = c.intermediate_size;
    const float eps = c.layer_norm_eps;
    const uint32_t tokens = static_cast<uint32_t>(packed.visual_tokens);
    const uint32_t patch_rows = static_cast<uint32_t>(packed.patch_count);

    // 1. Reduction to [tokens, cin] in _v_reduced. The arena is sized for the
    // connector's actual reduced dim; the initial view uses vh (the minimum)
    // and each branch re-views to its exact cin before use.
    uint32_t cin = vh;
    auto reduced = vws(_v_reduced).subview(0, static_cast<size_t>(tokens) * vh);
    switch (c.kind) {
        case P7ConnectorKind::spatial_mlp: {
            const uint32_t m = c.spatial_factor;
            cin = vh * m * m;
            reduced = vws(_v_reduced).subview(0, static_cast<size_t>(tokens) * cin);
            auto plan = plan_spatial_merge_dispatch(tokens, cin);
            if (!plan.ok()) { return plan.status; }
            if (auto cmd = dispatch_1d((*_sh_merge)(x, reduced, patch_rows, vh, packed.grid_w, m), *plan.value)) {
                list << std::move(cmd);
            }
            break;
        }
        case P7ConnectorKind::unshuffle_mlp: {
            cin = vh * 4u;
            reduced = vws(_v_reduced).subview(0, static_cast<size_t>(tokens) * cin);
            // The vision output x is TOKEN-major [patch_rows, vh]. The Phase 2
            // pixel_unshuffle kernel expects channel-major NCHW [1, vh, gh, gw],
            // so transpose x -> [vh, patch_rows] first, unshuffle to
            // [vh*4, tokens] (channel index ci*4 + ry*2 + rx), then transpose
            // back to token rows [tokens, cin]. The token channel order is
            // (ci, ry, rx), matching ref_pixel_unshuffle.
            auto nchw = vws(_v_unshuffle).subview(0, static_cast<size_t>(vh) * patch_rows);
            auto plan_in = plan_transpose_2d_dispatch(patch_rows, vh);
            if (!plan_in.ok()) { return plan_in.status; }
            if (auto cmd = dispatch_1d((*_sh_transpose)(x, nchw, patch_rows, vh), *plan_in.value)) {
                list << std::move(cmd);
            }
            auto unsh = vws(_v_cnorm).subview(0, static_cast<size_t>(tokens) * cin);
            auto plan = plan_pixel_unshuffle_dispatch(1u, vh, packed.grid_h, packed.grid_w);
            if (!plan.ok()) { return plan.status; }
            if (auto cmd = dispatch_1d((*_sh_unshuffle)(nchw, unsh, 1u, vh, packed.grid_h, packed.grid_w, 2u), *plan.value)) {
                list << std::move(cmd);
            }
            auto plan_t = plan_transpose_2d_dispatch(cin, tokens);
            if (!plan_t.ok()) { return plan_t.status; }
            if (auto cmd = dispatch_1d((*_sh_transpose)(unsh, reduced, cin, tokens), *plan_t.value)) {
                list << std::move(cmd);
            }
            break;
        }
        case P7ConnectorKind::adaptive_mlp: {
            cin = vh;
            reduced = vws(_v_reduced).subview(0, static_cast<size_t>(tokens) * cin);
            // The vision output x is TOKEN-major [patch_rows, vh]. The adaptive
            // pool kernel expects channel-major NCHW [1, vh, gh, gw], so
            // transpose x -> [vh, patch_rows] first, pool per-channel spatially
            // to [vh, out_h*out_w], then transpose back to token rows
            // [tokens, vh]. (Defect: feeding token-major x directly to the NCHW
            // pool pooled across the wrong axis.)
            auto nchw = vws(_v_unshuffle).subview(0, static_cast<size_t>(vh) * patch_rows);
            auto plan_in = plan_transpose_2d_dispatch(patch_rows, vh);
            if (!plan_in.ok()) { return plan_in.status; }
            if (auto cmd = dispatch_1d((*_sh_transpose)(x, nchw, patch_rows, vh), *plan_in.value)) {
                list << std::move(cmd);
            }
            auto pooled = vws(_v_cnorm).subview(0, static_cast<size_t>(vh) * tokens);
            const uint64_t count = static_cast<uint64_t>(vh) * packed.out_h * packed.out_w;
            auto plan = plan_elementwise_dispatch(count);
            if (!plan.ok()) { return plan.status; }
            if (auto cmd = dispatch_1d((*_sh_pool)(nchw, pooled, 1u, vh, packed.grid_h, packed.grid_w,
                                                   packed.out_h, packed.out_w), *plan.value)) {
                list << std::move(cmd);
            }
            auto plan_t = plan_transpose_2d_dispatch(vh, tokens);
            if (!plan_t.ok()) { return plan_t.status; }
            if (auto cmd = dispatch_1d((*_sh_transpose)(pooled, reduced, vh, tokens), *plan_t.value)) {
                list << std::move(cmd);
            }
            break;
        }
        case P7ConnectorKind::downsample_mlp: {
            // Learned strided downsample: spatial_merge m x m -> [tokens, vh*m*m]
            // then a learned linear projection -> [tokens, vh]. The merged rows
            // live in the DEDICATED _v_merged region (sized max_tokens*vh*m*m);
            // the cnorm region is only vh wide here and cannot hold them.
            const uint32_t m = c.spatial_factor;
            const uint32_t merged_dim = vh * m * m;
            auto merged = vws(_v_merged).subview(0, static_cast<size_t>(tokens) * merged_dim);
            auto plan = plan_spatial_merge_dispatch(tokens, merged_dim);
            if (!plan.ok()) { return plan.status; }
            if (auto cmd = dispatch_1d((*_sh_merge)(x, merged, patch_rows, vh, packed.grid_w, m), *plan.value)) {
                list << std::move(cmd);
            }
            cin = vh;
            reduced = vws(_v_reduced).subview(0, static_cast<size_t>(tokens) * cin);
            if (auto s = record_linear(list, merged, _connector.down_w, reduced, _connector.down_b,
                                       tokens, vh, merged_dim, GemmEpilogue::bias); !s.ok()) { return s; }
            break;
        }
        case P7ConnectorKind::mlp:
        default:
            cin = vh;
            reduced = vws(_v_reduced).subview(0, static_cast<size_t>(tokens) * cin);
            if (auto s = copy_rows(list, x, reduced, tokens, cin); !s.ok()) { return s; }
            break;
    }
    // 2. Normalization over cin: LayerNorm (weight+bias) by default, or RMSNorm
    //    (weight only) for the Qwen2.5-VL merger convention (olmOCR-2).
    auto cnorm = vws(_v_cnorm).subview(0, static_cast<size_t>(tokens) * cin);
    {
        auto plan = plan_norm_dispatch(tokens);
        if (!plan.ok()) { return plan.status; }
        if (c.rms_norm) {
            if (auto cmd = dispatch_1d((*_sh_rms_norm)(reduced, cnorm, _connector.ln_w, tokens, cin, eps, 1u), *plan.value)) {
                list << std::move(cmd);
            }
        } else {
            if (auto cmd = dispatch_1d((*_sh_layer_norm)(reduced, cnorm, _connector.ln_w, _connector.ln_b, tokens, cin, eps, 1u), *plan.value)) {
                list << std::move(cmd);
            }
        }
    }
    // 3. fc1 (+bias, GELU-erf).
    auto cfc1 = vws(_v_cfc1).subview(0, static_cast<size_t>(tokens) * ci);
    if (auto s = record_linear(list, cnorm, _connector.fc1_w, cfc1, _connector.fc1_b, tokens, ci, cin, GemmEpilogue::bias_gelu_erf); !s.ok()) { return s; }
    // 4. fc2 (+bias) -> out [tokens, text_hidden].
    return record_linear(list, cfc1, _connector.fc2_w, out, _connector.fc2_b, tokens, text_hidden, ci, GemmEpilogue::bias);
}

// ---------------------------------------------------------------------------
// record_vision
// ---------------------------------------------------------------------------

[[nodiscard]] Result<P7VisionOutput> P7VlmExecutor::record_vision(CommandList &list,
                                                                  luisa::span<const P7ImageInput> images) {
    using R = Result<P7VisionOutput>;
    if (_device == nullptr) { return R::failure(StatusCode::invalid_argument, "p7 executor not created", "device"); }
    if (images.empty()) { return R::failure(StatusCode::invalid_argument, "no images", "images"); }
    const auto &v = _config.vision;
    const uint32_t text_hidden = _config.text.hidden_size;
    // Validate + preprocess every image first (checked totals).
    uint64_t total_patches = 0, total_tokens = 0;
    luisa::vector<luisa::vector<float>> preprocessed;
    preprocessed.reserve(images.size());
    for (const auto &img : images) {
        if (img.pixels.size() != static_cast<size_t>(img.source_width) * img.source_height * 3u) {
            return R::failure(StatusCode::invalid_argument, "image pixel buffer size mismatch", "pixels");
        }
        const auto &p = img.packed;
        if (p.patch_count == 0u || p.visual_tokens == 0u) {
            return R::failure(StatusCode::invalid_argument, "degenerate packed image", "packed");
        }
        uint64_t want_h = 0, want_w = 0;
        if (!checked_mul(p.grid_h, v.patch_size, want_h) || !checked_mul(p.grid_w, v.patch_size, want_w) ||
            want_h != p.resized_height || want_w != p.resized_width) {
            return R::failure(StatusCode::invalid_argument, "packing grid does not match the resized size", "packed");
        }
        if (v.position_rows != 0u && p.patch_count > v.position_rows) {
            return R::failure(StatusCode::invalid_argument, "patch grid exceeds the learned position table", "position_rows");
        }
        auto pre = p7_preprocess_image_host(img, _config.mean, _config.std_dev);
        if (!pre.ok()) { return R::failure(pre.status); }
        if (pre.value->size() != static_cast<size_t>(3) * p.resized_width * p.resized_height) {
            return R::failure(StatusCode::internal_error, "preprocessed size mismatch", "preprocess");
        }
        uint64_t np = 0, nt = 0;
        if (!checked_add(total_patches, p.patch_count, np) || !checked_add(total_tokens, p.visual_tokens, nt)) {
            return R::failure(StatusCode::out_of_memory, "vision totals overflow", "packed");
        }
        if (np > _config.max_vision_patches) {
            return R::failure(StatusCode::out_of_memory, "total vision patches exceed the workspace capacity", "max_vision_patches");
        }
        total_patches = np;
        total_tokens = nt;
        preprocessed.push_back(std::move(*pre.value));
    }

    auto staging = luisa::make_unique<VisionUploadStaging>();
    staging->axis = _vision_axis_pending;
    for (const auto &img : images) {
        auto pos = p7_vision_positions(img.packed);
        staging->positions.insert(staging->positions.end(), pos.begin(), pos.end());
    }
    for (const auto &pre : preprocessed) {
        staging->pixels.insert(staging->pixels.end(), pre.begin(), pre.end());
    }
    // Pre-build stable per-image learned position ids BEFORE any upload so the
    // pos_ids vector-of-vectors never reallocates while a copy command borrows
    // an element (defect: a single reused vector dangled across images).
    if (v.position_rows != 0u) {
        staging->pos_ids.reserve(images.size());
        for (const auto &img : images) {
            const uint32_t rows = static_cast<uint32_t>(img.packed.patch_count);
            luisa::vector<int32_t> ids(rows);
            for (uint32_t t = 0; t < rows; t++) { ids[t] = static_cast<int32_t>(t); }
            staging->pos_ids.push_back(std::move(ids));
        }
    }
    auto *stage = staging.get();
    list << vws(_v_axis).as<uint32_t>().copy_from(luisa::span<const uint32_t>{stage->axis});

    P7VisionOutput output;
    output.total_tokens = static_cast<uint32_t>(total_tokens);
    uint64_t patch_offset = 0, token_offset = 0, pixel_offset = 0, pos_offset = 0;
    for (size_t i = 0; i < images.size(); i++) {
        const auto &img = images[i];
        const auto &p = img.packed;
        const uint32_t rows = static_cast<uint32_t>(p.patch_count);
        const uint32_t tokens = static_cast<uint32_t>(p.visual_tokens);
        const uint64_t pixel_elems = preprocessed[i].size();
        auto pixels = vws(_v_pixels).subview(0, static_cast<size_t>(pixel_elems));
        auto positions = vws(_v_positions).as<int32_t>().subview(0, static_cast<size_t>(rows) * 2u);
        list << pixels.copy_from(luisa::span<const float>{stage->pixels.data() + pixel_offset, static_cast<size_t>(pixel_elems)})
             << positions.copy_from(luisa::span<const int32_t>{stage->positions.data() + pos_offset, static_cast<size_t>(rows) * 2u});
        // Patchify.
        auto patches = vws(_v_patches).subview(0, static_cast<size_t>(rows) * v.patch_dim());
        {
            auto plan = plan_image_patchify_dispatch(rows, v.patch_dim());
            if (!plan.ok()) { return R::failure(plan.status); }
            if (auto cmd = dispatch_1d((*_sh_patchify)(pixels, patches, v.channels, p.resized_height, p.resized_width, v.patch_size), *plan.value)) {
                list << std::move(cmd);
            }
        }
        // Patch embed.
        auto x = vws(_v_x).subview(0, static_cast<size_t>(rows) * v.hidden_size);
        if (auto s = record_linear(list, patches, _patch_embed_w, x, _patch_embed_b, rows, v.hidden_size, v.patch_dim(), GemmEpilogue::bias); !s.ok()) {
            return R::failure(s);
        }
        // Optional learned absolute position embedding. The position ids are
        // uploaded into the aux scratch (the positions buffer is still needed
        // by RoPE below); the add writes a separate norm scratch then copies.
        // The ids come from the stable PER-IMAGE staging vector (built before
        // any upload), so multi-image uploads never dangle / reallocate.
        if (v.position_rows != 0u) {
            const auto &ids = stage->pos_ids[i];
            auto id_view = vws(_v_aux).as<int32_t>().subview(0, rows);
            list << id_view.copy_from(luisa::span<const int32_t>{ids});
            auto added = vws(_v_norm).subview(0, static_cast<size_t>(rows) * v.hidden_size);
            auto plan = plan_add_position_dispatch(rows, v.hidden_size);
            if (!plan.ok()) { return R::failure(plan.status); }
            if (auto cmd = dispatch_1d((*_sh_add_pos)(x, _position_embed, id_view, added, rows, v.hidden_size), *plan.value)) {
                list << std::move(cmd);
            }
            if (auto s = copy_rows(list, added, x, rows, v.hidden_size); !s.ok()) { return R::failure(s); }
        }
        // Vision blocks. For window/global, precompute the two distinct masks
        // (windowed + global) ONCE per image into stable staging buffers; each
        // layer's upload borrows the appropriate one (never a reused scratch).
        if (v.mask_kind == P7VisionMaskKind::window_global) {
            auto wm = build_window_mask(rows, p.grid_w, v.window_patch_size, false);
            auto gm = build_window_mask(rows, p.grid_w, v.window_patch_size, true);
            staging->window_masks.push_back(std::move(wm.words));
            staging->global_masks.push_back(std::move(gm.words));
        }
        const size_t mask_idx = staging->window_masks.empty() ? 0u : staging->window_masks.size() - 1u;
        // Additive Phase 8: download + expand the per-layer relative-position
        // bias tables ONCE per image into stable staging buffers (the weight
        // buffers are device-resident; the expansion is host math). Each layer's
        // upload below borrows its own buffer, never a reused scratch.
        const size_t rel_img_idx = staging->rel_bias.size();
        if (v.rel_pos_window != 0u) {
            luisa::vector<luisa::vector<float>> per_layer;
            per_layer.reserve(v.num_layers);
            for (uint32_t l = 0; l < v.num_layers; l++) {
                per_layer.push_back(expand_rel_pos_bias(_rel_pos_tables[l], v.num_heads, v.rel_pos_window, p.grid_h, p.grid_w));
            }
            staging->rel_bias.push_back(std::move(per_layer));
        }
        for (uint32_t l = 0; l < v.num_layers; l++) {
            if (v.mask_kind == P7VisionMaskKind::window_global) {
                const bool global = std::find(v.global_layers.begin(), v.global_layers.end(), l) != v.global_layers.end();
                const auto &words = global ? staging->global_masks[mask_idx] : staging->window_masks[mask_idx];
                auto mask_view = vws(_v_block_mask).as<uint32_t>().subview(0, words.size());
                list << mask_view.copy_from(luisa::span<const uint32_t>{words});
            }
            if (v.rel_pos_window != 0u) {
                const auto &expanded = staging->rel_bias[rel_img_idx][l];
                auto rel_view = vws(_v_rel_bias).subview(0, expanded.size());
                list << rel_view.copy_from(luisa::span<const float>{expanded});
            }
            if (auto s = record_vision_block(list, l, x, positions, rows, p); !s.ok()) { return R::failure(s); }
        }
        // Connector -> features.
        auto feat = vws(_v_features).subview(static_cast<size_t>(token_offset) * text_hidden,
                                             static_cast<size_t>(tokens) * text_hidden);
        if (auto s = record_connector(list, x, p, feat); !s.ok()) { return R::failure(s); }
        patch_offset += rows;
        token_offset += tokens;
        pixel_offset += pixel_elems;
        pos_offset += static_cast<uint64_t>(rows) * 2u;
    }
    output.features = vws(_v_features).subview(0, static_cast<size_t>(total_tokens) * text_hidden);
    _last_total_patches = static_cast<uint32_t>(total_patches);
    _last_total_tokens = static_cast<uint32_t>(total_tokens);
    list.add_callback([staging = std::move(staging)]() mutable { staging.reset(); });
    return R::success(std::move(output));
}

// ---------------------------------------------------------------------------
// prompt assembly
// ---------------------------------------------------------------------------

[[nodiscard]] Result<DecodeRequest> P7VlmExecutor::build_decode_request(
    luisa::span<const uint32_t> token_ids, uint32_t mm_start, uint32_t mm_count,
    const P7VisionOutput &vision, luisa::span<const P7ImageInput> images,
    luisa::vector<int32_t> &position_ids_out, luisa::vector<int32_t> &next_position_out) const {
    using R = Result<DecodeRequest>;
    const auto &t = _config.text;
    if (token_ids.empty()) { return R::failure(StatusCode::invalid_argument, "empty prompt", "token_ids"); }
    for (auto id : token_ids) {
        if (id >= t.vocab_size) { return R::failure(StatusCode::invalid_argument, "prompt token outside the vocabulary", "token_ids"); }
    }
    if (images.size() > 1u) {
        return R::failure(StatusCode::unsupported, "the Phase 7 slice supports one image per prompt", "images");
    }
    if (mm_count != 0u && images.empty()) {
        return R::failure(StatusCode::invalid_argument, "image rows require an image", "mm_count");
    }
    if (!images.empty() && mm_count != images[0].packed.visual_tokens) {
        return R::failure(StatusCode::invalid_argument, "image row count does not match the packed tokens", "mm_count");
    }
    P7PackedImage packed;
    if (!images.empty()) { packed = images[0].packed; }
    auto pos = p7_build_position_ids(_config, static_cast<uint32_t>(token_ids.size()), mm_start, mm_count, packed);
    if (!pos.ok()) { return R::failure(pos.status); }
    position_ids_out = std::move(pos.value->ids);
    next_position_out = std::move(pos.value->next_position);
    DecodeRequest request;
    request.prompt_tokens.assign(token_ids.begin(), token_ids.end());
    if (mm_count != 0u) {
        request.mm_features = vision.features;
        request.mm_start = mm_start;
        request.mm_count = mm_count;
    }
    return R::success(std::move(request));
}

// ---------------------------------------------------------------------------
// download helpers
// ---------------------------------------------------------------------------

[[nodiscard]] Status P7VlmExecutor::download_vision_patches(Stream &stream, luisa::vector<float> &out) const {
    const uint64_t count = static_cast<uint64_t>(_last_total_patches) * _config.vision.patch_dim();
    out.resize(static_cast<size_t>(count));
    if (count != 0u) {
        stream << vws(_v_patches).subview(0, static_cast<size_t>(count)).copy_to(luisa::span<float>{out});
    }
    return Status::success();
}
[[nodiscard]] Status P7VlmExecutor::download_vision_final(Stream &stream, luisa::vector<float> &out) const {
    const uint64_t count = static_cast<uint64_t>(_last_total_patches) * _config.vision.hidden_size;
    out.resize(static_cast<size_t>(count));
    if (count != 0u) {
        stream << vws(_v_x).subview(0, static_cast<size_t>(count)).copy_to(luisa::span<float>{out});
    }
    return Status::success();
}
[[nodiscard]] Status P7VlmExecutor::download_vision_features(Stream &stream, luisa::vector<float> &out) const {
    const uint64_t count = static_cast<uint64_t>(_last_total_tokens) * _config.text.hidden_size;
    out.resize(static_cast<size_t>(count));
    if (count != 0u) {
        stream << vws(_v_features).subview(0, static_cast<size_t>(count)).copy_to(luisa::span<float>{out});
    }
    return Status::success();
}
[[nodiscard]] uint32_t P7VlmExecutor::connector_input_dim() const noexcept {
    const uint32_t vh = _config.vision.hidden_size;
    switch (_config.connector.kind) {
        case P7ConnectorKind::spatial_mlp:
            return vh * _config.connector.spatial_factor * _config.connector.spatial_factor;
        case P7ConnectorKind::unshuffle_mlp:
            return vh * 4u;
        default:
            return vh;
    }
}
[[nodiscard]] Status P7VlmExecutor::download_vision_reduced(Stream &stream, luisa::vector<float> &out) const {
    const uint64_t count = static_cast<uint64_t>(_last_total_tokens) * connector_input_dim();
    out.resize(static_cast<size_t>(count));
    if (count != 0u) {
        stream << vws(_v_reduced).subview(0, static_cast<size_t>(count)).copy_to(luisa::span<float>{out});
    }
    return Status::success();
}

// ---------------------------------------------------------------------------
// P7RegionRecognizer
// ---------------------------------------------------------------------------

[[nodiscard]] Status P7RegionRecognizerConfig::validate() const {
    if (auto s = model.validate(); !s.ok()) { return s; }
    if (auto s = names.validate(); !s.ok()) { return s; }
    if (prompt_prefix.empty() && prompt_suffix.empty()) {
        return Status::error(StatusCode::invalid_argument, "prompt must have a prefix or suffix", "prompt");
    }
    for (auto id : prompt_prefix) {
        if (id >= model.text.vocab_size) { return Status::error(StatusCode::invalid_argument, "prompt prefix token outside the vocabulary", "prompt_prefix"); }
    }
    for (auto id : prompt_suffix) {
        if (id >= model.text.vocab_size) { return Status::error(StatusCode::invalid_argument, "prompt suffix token outside the vocabulary", "prompt_suffix"); }
    }
    return Status::success();
}

[[nodiscard]] Status P7RegionRecognizer::create(Device &device, Stream &stream,
                                                const P7RegionRecognizerConfig &config, const WeightStore &weights) {
    auto valid = config.validate();
    if (!valid.ok()) { return valid; }
    _config = config;
    _device = &device;
    return _slice.create(device, stream, config.model, config.names, weights);
}

[[nodiscard]] Result<luisa::vector<RawRegionOutput>> P7RegionRecognizer::recognize_batch(
    Stream &stream, luisa::span<const RegionCropInput> crops, const RegionRecognizeOptions &options) {
    using R = Result<luisa::vector<RawRegionOutput>>;
    if (_device == nullptr) { return R::failure(StatusCode::invalid_argument, "p7 recognizer not created", "device"); }
    if (options.max_output_tokens == 0u) {
        return R::failure(StatusCode::invalid_argument, "max_output_tokens must be nonzero", "max_output_tokens");
    }
    const auto &model = _config.model;
    Clock clock;
    const auto cancelled = [&]() { return options.cancel_callback && options.cancel_callback(); };
    const auto timed_out = [&]() { return options.max_wall_time_ms > 0.0 && clock.toc() > options.max_wall_time_ms; };

    // Phase 11: pack and validate the whole crop batch before recording. The
    // shared P7 graph keeps attention ragged per image and writes connector
    // features into disjoint token ranges. Exact packed-patch totals split an
    // oversized caller batch into multiple submissions; valid crops are never
    // rejected merely because the scheduler used an approximate cost. Text
    // decoding remains independent per crop.
    luisa::vector<P7ImageInput> images;
    images.reserve(crops.size());
    for (const auto &crop : crops) {
        if (cancelled()) { return R::failure(StatusCode::cancelled, "region recognition cancelled", "cancel"); }
        if (timed_out()) { return R::failure(StatusCode::timeout, "region recognition wall-time budget exceeded", "max_wall_time_ms"); }
        if (crop.source_width == 0u || crop.source_height == 0u ||
            crop.pixels.size() != static_cast<size_t>(crop.source_width) * crop.source_height * 3u) {
            return R::failure(StatusCode::invalid_argument, "degenerate region crop", "crop");
        }
        auto packed = p7_pack_image(crop.source_width, crop.source_height, model.packing, model.connector);
        if (!packed.ok()) { return R::failure(packed.status); }
        if (packed.value->patch_count > model.max_vision_patches) {
            return R::failure(StatusCode::out_of_memory, "region patch count exceeds the vision workspace capacity", "max_vision_patches");
        }
        P7ImageInput img;
        img.pixels = crop.pixels;
        img.source_width = crop.source_width;
        img.source_height = crop.source_height;
        img.packed = *packed.value;
        images.push_back(std::move(img));
    }
    luisa::vector<RawRegionOutput> outputs;
    outputs.reserve(crops.size());
    for (size_t batch_begin = 0; batch_begin < crops.size();) {
        size_t batch_end = batch_begin;
        uint64_t batch_patches = 0;
        while (batch_end < crops.size()) {
            const uint64_t next = images[batch_end].packed.patch_count;
            if (next > static_cast<uint64_t>(model.max_vision_patches) - batch_patches) { break; }
            batch_patches += next;
            batch_end++;
        }
        if (batch_end == batch_begin) {
            return R::failure(StatusCode::internal_error, "failed to place a validated crop in a vision batch", "max_vision_patches");
        }
        const size_t batch_count = batch_end - batch_begin;
        _slice.reset_vision_workspace();
        P7VisionOutput batch_vision;
        {
            auto list = CommandList::create();
            auto v = _slice.record_vision(list, luisa::span<const P7ImageInput>{images}.subspan(batch_begin, batch_count));
            if (!v.ok()) { return R::failure(v.status); }
            batch_vision = *v.value;
            stream << list.commit() << synchronize();
        }
        _vision_graph_count++;
        _vision_crop_count += batch_count;
        size_t token_offset = 0;
        for (size_t crop_index = batch_begin; crop_index < batch_end; crop_index++) {
        if (cancelled()) { return R::failure(StatusCode::cancelled, "region recognition cancelled", "cancel"); }
        if (timed_out()) { return R::failure(StatusCode::timeout, "region recognition wall-time budget exceeded", "max_wall_time_ms"); }
        const auto &crop = crops[crop_index];
        const auto &img = images[crop_index];
        const uint32_t rows = static_cast<uint32_t>(img.packed.patch_count);
        const uint32_t tokens = static_cast<uint32_t>(img.packed.visual_tokens);
        P7VisionOutput vision;
        vision.total_tokens = tokens;
        vision.features = batch_vision.features.subview(
            token_offset * model.text.hidden_size,
            static_cast<size_t>(tokens) * model.text.hidden_size);
        token_offset += tokens;

        luisa::vector<uint32_t> prompt = _config.prompt_prefix;
        const uint32_t mm_start = static_cast<uint32_t>(prompt.size());
        for (uint32_t i = 0; i < tokens; i++) { prompt.push_back(0u); }
        prompt.insert(prompt.end(), _config.prompt_suffix.begin(), _config.prompt_suffix.end());
        luisa::vector<int32_t> position_ids, next_position;
        auto req = _slice.build_decode_request(prompt, mm_start, tokens, vision,
                                               luisa::span<const P7ImageInput>{&img, 1}, position_ids, next_position);
        if (!req.ok()) { return R::failure(req.status); }
        ContiguousKvCache cache;
        KvCacheConfig kc;
        kc.num_layers = model.text.num_layers;
        kc.num_kv_heads = model.text.num_kv_heads;
        kc.head_dim = model.text.head_dim;
        kc.max_seq_len = static_cast<uint32_t>(req.value->prompt_tokens.size()) + options.max_output_tokens;
        kc.batch = 1;
        kc.dtype_bytes = 4;
        if (kc.max_seq_len > model.text.max_position_embeddings) {
            return R::failure(StatusCode::out_of_memory, "region prompt + output exceeds the decoder context ceiling", "max_position_embeddings");
        }
        MemoryBudget kv_budget{_config.memory};
        if (auto s = kv_budget.set_requirement("p7_region", MemoryCategory::weights, _slice.weight_bytes()); !s.ok()) { return R::failure(s); }
        if (auto s = kv_budget.set_requirement("p7_region", MemoryCategory::workspace, workspace_bytes()); !s.ok()) { return R::failure(s); }
        if (auto s = cache.create(*_device, stream, kc, &kv_budget); !s.ok()) { return R::failure(s); }
        _kv_peak_bytes = std::max(_kv_peak_bytes, cache.device_bytes());
        DenseDecodeOptions dopts;
        dopts.sampling.greedy = true;
        dopts.max_new_tokens = options.max_output_tokens;
        dopts.max_wall_time_ms = options.max_wall_time_ms;
        dopts.enable_cuda_graph = options.enable_cuda_graph;
        dopts.cancel_callback = options.cancel_callback;
        if (model.text_rope_axes() >= 3u) {
            dopts.prompt_position_ids = std::move(position_ids);
            dopts.next_position = std::move(next_position);
        }
        DenseDecodeController controller;
        if (auto s = controller.create(_slice.text_executor(), nullptr, dopts); !s.ok()) { return R::failure(s); }
        auto gen = controller.generate(stream, *req.value, cache);
        if (!gen.ok()) { return R::failure(gen.status); }
        _cuda_graph_steps += gen.value->cuda_graph_steps;
        _cuda_graph_updates += gen.value->cuda_graph_updates;
        _cuda_graph_fallbacks += gen.value->cuda_graph_fallbacks;
        RawRegionOutput out;
        out.region_id = crop.task.region_id;
        out.text = std::move(gen.value->text);
        out.tokens = std::move(gen.value->tokens);
        out.truncated = gen.value->truncated;
        out.hit_eos = gen.value->hit_eos;
        out.native_patches = rows;
        out.output_tokens = static_cast<uint32_t>(out.tokens.size());
        out.has_confidence = false;
        outputs.push_back(std::move(out));
        }
        batch_begin = batch_end;
    }
    return R::success(std::move(outputs));
}

}// namespace luisa::ocr

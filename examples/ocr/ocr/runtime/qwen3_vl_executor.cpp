// OCR engine -- Qwen3-VL vertical-slice executor (Phase 4).
// Vision tower + mergers from Phase 2 operators; the text decoder is the
// shared DenseVlmExecutor with the additive Qwen3 contract. See the header.
#include "ocr/runtime/qwen3_vl_executor.h"

#include <cmath>
#include <cstring>
#include <utility>

#include <luisa/core/logging.h>
#include <luisa/core/stl/format.h>
#include <luisa/runtime/command_list.h>
#include <luisa/runtime/stream.h>

#include "ocr/runtime/dispatch_adapter.h"

namespace luisa::ocr {

using namespace luisa::compute;

// Host preprocessing bridge (runtime/qwen3_vl_preprocess.cpp). The bridge
// header pulls preprocess.h -> api.h -> model_recipe.h, whose host-only
// AttentionMaskKind must never meet the ops enum in this device TU, so the
// function is forward-declared here over the shared plain data types
// (runtime/qwen3_vl_image.h); the bridge TU defines the same signature.
[[nodiscard]] Result<Qwen3VlPreprocessedImage> qwen3_vl_preprocess_image_host(
    const Qwen3VlImageInput &image, luisa::span<const float> mean, luisa::span<const float> std_dev);

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

// Per-record heap-owned upload staging: the pixel/position uploads borrow
// host pointers that must outlive command EXECUTION. One block per
// record_vision call, retained by a completion callback.
struct VisionUploadStaging {
    luisa::vector<float> pixels;     // normalized CHW of every image, concatenated
    luisa::vector<int32_t> positions;// vision 2D positions of every image
    luisa::vector<uint32_t> axis;    // vision RoPE axis map
    luisa::vector<float> mean, std_dev;
};

}// namespace

// ---------------------------------------------------------------------------
// Config / prompt validation
// ---------------------------------------------------------------------------

[[nodiscard]] Status Qwen3VlExecutorConfig::validate() const {
    if (auto s = model.validate(); !s.ok()) { return s; }
    if (auto s = names.validate(); !s.ok()) { return s; }
    if (vision_component.empty() || decoder_component.empty()) {
        return Status::error(StatusCode::invalid_argument, "weight component tags must not be empty", "component");
    }
    if (vision_component == decoder_component) {
        return Status::error(StatusCode::invalid_argument,
                             "vision and decoder weight components must be distinct", "component");
    }
    if (mean.size() != 3u || std_dev.size() != 3u) {
        return Status::error(StatusCode::invalid_argument, "mean/std_dev must have 3 channels", "mean/std_dev");
    }
    for (auto v : mean) {
        if (!std::isfinite(v)) {
            return Status::error(StatusCode::invalid_argument, "mean must be finite", "mean");
        }
    }
    for (auto v : std_dev) {
        if (!std::isfinite(v) || v == 0.0f) {
            return Status::error(StatusCode::invalid_argument, "std_dev must be finite and nonzero", "std_dev");
        }
    }
    if (max_vision_patches == 0u) {
        return Status::error(StatusCode::invalid_argument,
                             "max_vision_patches must be nonzero", "max_vision_patches");
    }
    if (prefill_chunk_rows == 0u) {
        return Status::error(StatusCode::invalid_argument, "prefill_chunk_rows must be nonzero", "prefill_chunk_rows");
    }
    return Status::success();
}

[[nodiscard]] Status Qwen3VlPrompt::validate(uint32_t vocab_size) const {
    if (token_ids.empty()) {
        return Status::error(StatusCode::invalid_argument, "prompt must not be empty", "token_ids");
    }
    for (auto id : token_ids) {
        if (id >= vocab_size) {
            return Status::error(StatusCode::invalid_argument, "prompt token outside the vocabulary", "token_ids");
        }
    }
    uint32_t prev_end = 0u;
    for (const auto &r : image_ranges) {
        if (r.count == 0u) {
            return Status::error(StatusCode::invalid_argument, "image range must be non-empty", "image_ranges");
        }
        if (r.start < prev_end) {
            return Status::error(StatusCode::invalid_argument, "image ranges must be ordered and disjoint", "image_ranges");
        }
        if (static_cast<uint64_t>(r.start) + r.count > token_ids.size()) {
            return Status::error(StatusCode::invalid_argument, "image range exceeds the prompt", "image_ranges");
        }
        prev_end = r.start + r.count;
    }
    return Status::success();
}

// ---------------------------------------------------------------------------
// create / resolve / reserve / compile
// ---------------------------------------------------------------------------

[[nodiscard]] Status Qwen3VlExecutor::create(Device &device, Stream &stream,
                                             const Qwen3VlExecutorConfig &config,
                                             const WeightStore &weights) {
    auto valid = config.validate();
    if (!valid.ok()) { return valid; }
    _config = config;
    _device = &device;
    _vision_ws = DeviceWorkspace{device};
    _resize_cache = TypedShaderCache<ImageResizeNormalizeKernel>{device, _shader_registry};
    _patchify_cache = TypedShaderCache<ImagePatchifyKernel>{device, _shader_registry};
    _layer_norm_cache = TypedShaderCache<LayerNormKernel>{device, _shader_registry};
    _rms_norm_cache = TypedShaderCache<RmsNormKernel>{device, _shader_registry};
    _gemm_cache = TypedShaderCache<GemmKernel>{device, _shader_registry};
    _rope2d_cache = TypedShaderCache<RopeAxialKernel>{device, _shader_registry};
    _attn_cache = TypedShaderCache<AttentionPrefillKernel>{device, _shader_registry};
    _merge_cache = TypedShaderCache<SpatialMergeKernel>{device, _shader_registry};
    _residual_cache = TypedShaderCache<ResidualAddKernel>{device, _shader_registry};

    if (auto s = resolve_weights(weights); !s.ok()) { return s; }
    if (auto s = reserve_workspace(); !s.ok()) { return s; }
    if (auto s = compile_shaders(); !s.ok()) { return s; }

    // Normalization constants (immutable after create).
    _mean_buf = device.create_buffer<float>(3);
    _std_buf = device.create_buffer<float>(3);
    if (!_mean_buf || !_std_buf) {
        return Status::error(StatusCode::out_of_memory, "failed to allocate normalization constants", "mean/std");
    }
    stream << _mean_buf.copy_from(luisa::span<const float>{_config.mean})
           << _std_buf.copy_from(luisa::span<const float>{_config.std_dev})
           << synchronize();

    // Embedded text executor with the additive Qwen3 contract.
    DenseExecutorConfig text_config;
    text_config.decoder = config.model.text;
    text_config.names = config.names.text;
    text_config.tie_word_embeddings = config.model.tie_word_embeddings;
    text_config.qk_norm = true;
    text_config.qkv_bias = false;
    text_config.rope_axes = 3u;
    text_config.mrope_axis_of_pair = config.model.mrope.axis_of_pair();
    text_config.deepstack_layers = config.model.deepstack.decoder_layers;
    text_config.component = config.decoder_component;
    text_config.prefill_chunk_rows = config.prefill_chunk_rows;
    text_config.enable_fused_residual_rms_norm = config.enable_fused_residual_rms_norm;
    text_config.enable_cooperative_norm = config.enable_cooperative_norm;
    text_config.memory = config.memory;
    if (auto s = _text.create(device, stream, text_config, weights); !s.ok()) { return s; }

    _vision_axis_pending = qwen3_vl_vision_axis_of_pair(config.model.vision);
    _weight_bytes = weights.component_bytes(config.vision_component) + _text.weight_bytes();
    return Status::success();
}

[[nodiscard]] Status Qwen3VlExecutor::resolve_weights(const WeightStore &weights) {
    const auto &names = _config.names;
    const auto &comp = _config.vision_component;
    const auto &v = _config.model.vision;

    // Exact expected inventory for the VISION component (the text component
    // is validated by the embedded text executor). Every installed vision
    // tensor must be part of the declared inventory.
    const auto full_expected = qwen3_vl_expected_tensors(_config.model, names);
    luisa::unordered_map<luisa::string, const ExpectedTensor *> expected_by_name;
    for (const auto &e : full_expected) {
        // Vision-side entries are exactly those under the visual prefix.
        if (e.name.compare(0, names.visual_prefix.size(), names.visual_prefix) == 0) {
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
                                 luisa::format("vision weight '{}' must be dense fp32 in the Phase 4 slice", name), name);
        }
        const auto it = expected_by_name.find(name);
        if (it == expected_by_name.end()) {
            return Status::error(StatusCode::invalid_argument,
                                 luisa::format("vision weight '{}' has no expected-tensor contract", name), name);
        }
        if (auto s = validate_decoder_weight_desc(w->logical, w->element_count, *it->second); !s.ok()) { return s; }
        if (w->dense.size() < w->element_count) {
            return Status::error(StatusCode::invalid_argument,
                                 luisa::format("vision weight '{}' device buffer is smaller than its element count", name),
                                 name);
        }
        out = w->dense.view();
        return Status::success();
    };
    if (auto s = view_of(names.patch_embed_weight(), _patch_embed_w); !s.ok()) { return s; }
    if (auto s = view_of(names.patch_embed_bias(), _patch_embed_b); !s.ok()) { return s; }
    _blocks.resize(v.num_layers);
    for (uint32_t l = 0; l < v.num_layers; l++) {
        auto &B = _blocks[l];
        if (auto s = view_of(names.block_norm1_weight(l), B.norm1_w); !s.ok()) { return s; }
        if (auto s = view_of(names.block_norm1_bias(l), B.norm1_b); !s.ok()) { return s; }
        if (auto s = view_of(names.block_norm2_weight(l), B.norm2_w); !s.ok()) { return s; }
        if (auto s = view_of(names.block_norm2_bias(l), B.norm2_b); !s.ok()) { return s; }
        if (auto s = view_of(names.block_q_proj(l), B.q_w); !s.ok()) { return s; }
        if (auto s = view_of(names.block_k_proj(l), B.k_w); !s.ok()) { return s; }
        if (auto s = view_of(names.block_v_proj(l), B.v_w); !s.ok()) { return s; }
        if (auto s = view_of(names.block_o_proj(l), B.o_w); !s.ok()) { return s; }
        if (auto s = view_of(names.block_q_bias(l), B.q_b); !s.ok()) { return s; }
        if (auto s = view_of(names.block_k_bias(l), B.k_b); !s.ok()) { return s; }
        if (auto s = view_of(names.block_v_bias(l), B.v_b); !s.ok()) { return s; }
        if (auto s = view_of(names.block_o_bias(l), B.o_b); !s.ok()) { return s; }
        if (auto s = view_of(names.block_q_norm(l), B.q_norm); !s.ok()) { return s; }
        if (auto s = view_of(names.block_k_norm(l), B.k_norm); !s.ok()) { return s; }
        if (auto s = view_of(names.block_fc1(l), B.fc1_w); !s.ok()) { return s; }
        if (auto s = view_of(names.block_fc1_bias(l), B.fc1_b); !s.ok()) { return s; }
        if (auto s = view_of(names.block_fc2(l), B.fc2_w); !s.ok()) { return s; }
        if (auto s = view_of(names.block_fc2_bias(l), B.fc2_b); !s.ok()) { return s; }
    }
    const auto resolve_merger = [&](const luisa::string &prefix, MergerWeights &M) -> Status {
        if (auto s = view_of(Qwen3VlTensorNames::merger_ln_weight(prefix), M.ln_w); !s.ok()) { return s; }
        if (auto s = view_of(Qwen3VlTensorNames::merger_ln_bias(prefix), M.ln_b); !s.ok()) { return s; }
        if (auto s = view_of(Qwen3VlTensorNames::merger_fc1(prefix), M.fc1_w); !s.ok()) { return s; }
        if (auto s = view_of(Qwen3VlTensorNames::merger_fc1_bias(prefix), M.fc1_b); !s.ok()) { return s; }
        if (auto s = view_of(Qwen3VlTensorNames::merger_fc2(prefix), M.fc2_w); !s.ok()) { return s; }
        if (auto s = view_of(Qwen3VlTensorNames::merger_fc2_bias(prefix), M.fc2_b); !s.ok()) { return s; }
        return Status::success();
    };
    if (auto s = resolve_merger(names.merger_prefix(), _merger); !s.ok()) { return s; }
    _ds_mergers.resize(_config.model.deepstack.count());
    for (uint32_t i = 0; i < _ds_mergers.size(); i++) {
        if (auto s = resolve_merger(names.deepstack_prefix(i), _ds_mergers[i]); !s.ok()) { return s; }
    }
    return Status::success();
}

[[nodiscard]] Status Qwen3VlExecutor::reserve_workspace() {
    const auto &v = _config.model.vision;
    const uint64_t maxp = _config.max_vision_patches;
    const uint64_t vh = v.hidden_size, vi = v.intermediate_size;
    const uint64_t merged = v.merged_dim();
    const uint64_t mi = v.merger_intermediate_size;
    const uint64_t text_hidden = _config.model.text.hidden_size;
    const uint64_t max_merged = maxp / (static_cast<uint64_t>(v.spatial_merge_size) * v.spatial_merge_size);
    const uint64_t ds_count = _config.model.deepstack.count();
    // The largest aux is the o-proj bias_residual: [bias vh | residual maxp*vh].
    const uint64_t aux_elems = vh + maxp * vh;

    struct RegionSpec {
        const char *name;
        uint64_t elements;
    };
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
        {"fc2", maxp * vh},
        {"merged", maxp * merged},
        {"mnorm", max_merged * merged},
        {"mfc1", max_merged * mi},
        {"features", max_merged * text_hidden},
        {"positions", maxp * 2u},
        {"axis", v.head_dim / 2u},
        {"dummy_f", 1},
        {"dummy_u", 1},
        {"aux", aux_elems},
    };
    WorkspacePlan plan;
    for (const auto &spec : specs) {
        auto bytes = f32_bytes(spec.elements);
        if (!bytes.ok()) { return bytes.status; }
        if (auto s = plan.add_bytes(spec.name, *bytes.value); !s.ok()) { return s; }
    }
    // DeepStack feature arenas (one per injection).
    for (uint32_t i = 0; i < ds_count; i++) {
        auto bytes = f32_bytes(max_merged * text_hidden);
        if (!bytes.ok()) { return bytes.status; }
        if (auto s = plan.add_bytes(luisa::format("ds_features_{}", i), *bytes.value); !s.ok()) { return s; }
    }
    MemoryBudget budget{_config.memory};
    if (auto s = budget.require("qwen3_vl_vision_workspace", MemoryCategory::workspace, plan.total_bytes()); !s.ok()) { return s; }
    if (auto s = budget.preflight(); !s.ok()) { return s; }
    if (auto s = _vision_ws.reserve(plan.total_bytes()); !s.ok()) { return s; }

    WorkspaceAllocation *outs[] = {
        &_v_pixels, &_v_patches, &_v_x, &_v_norm, &_v_q, &_v_k, &_v_v,
        &_v_qn, &_v_kn, &_v_qr, &_v_kr, &_v_attn, &_v_fc1, &_v_fc2, &_v_merged,
        &_v_mnorm, &_v_mfc1, &_v_features, &_v_positions, &_v_axis, &_v_dummy_f, &_v_dummy_u,
        &_v_aux,
    };
    for (size_t i = 0; i < std::size(specs); i++) {
        if (auto s = add_region(_vision_ws, *outs[i], specs[i].elements); !s.ok()) { return s; }
    }
    _v_ds_features.resize(static_cast<size_t>(ds_count));
    for (uint32_t i = 0; i < ds_count; i++) {
        if (auto s = add_region(_vision_ws, _v_ds_features[i], max_merged * text_hidden); !s.ok()) { return s; }
    }
    return Status::success();
}

[[nodiscard]] Status Qwen3VlExecutor::compile_shaders() {
    const auto get = [](auto &cache, const luisa::string &key, auto factory, auto *&out) -> Status {
        auto r = cache.get_or_compile(key, factory);
        if (!r.ok()) { return r.status; }
        out = *r.value;
        return Status::success();
    };
    if (auto s = get(_resize_cache, image_variant_key("resize_normalize"),
                     [] { return make_image_resize_normalize_kernel(); }, _sh_resize); !s.ok()) { return s; }
    if (auto s = get(_patchify_cache, image_variant_key("patchify"),
                     [] { return make_image_patchify_kernel(); }, _sh_patchify); !s.ok()) { return s; }
    if (auto s = get(_layer_norm_cache, norm_variant_key("layer_norm"),
                     [] { return make_layer_norm_kernel(); }, _sh_layer_norm); !s.ok()) { return s; }
    if (auto s = get(_rms_norm_cache, norm_variant_key("rms_norm"),
                     [] { return make_rms_norm_kernel(); }, _sh_rms_norm); !s.ok()) { return s; }
    if (auto s = get(_gemm_cache, gemm_variant_key(GemmEpilogue::none, GemmWeightFormat::fp32, false),
                     [] { return make_gemm_kernel(GemmEpilogue::none); }, _sh_gemm_none); !s.ok()) { return s; }
    if (auto s = get(_gemm_cache, gemm_variant_key(GemmEpilogue::bias, GemmWeightFormat::fp32, false),
                     [] { return make_gemm_kernel(GemmEpilogue::bias); }, _sh_gemm_bias); !s.ok()) { return s; }
    if (auto s = get(_gemm_cache, gemm_variant_key(GemmEpilogue::bias_gelu_erf, GemmWeightFormat::fp32, false),
                     [] { return make_gemm_kernel(GemmEpilogue::bias_gelu_erf); }, _sh_gemm_bias_gelu); !s.ok()) { return s; }
    if (auto s = get(_gemm_cache, gemm_variant_key(GemmEpilogue::bias_residual, GemmWeightFormat::fp32, false),
                     [] { return make_gemm_kernel(GemmEpilogue::bias_residual); }, _sh_gemm_bias_residual); !s.ok()) { return s; }
    if (auto s = get(_rope2d_cache, rope_variant_key("2d"),
                     [] { return make_rope_2d_kernel(); }, _sh_rope2d); !s.ok()) { return s; }
    if (auto s = get(_attn_cache, attention_variant_key("prefill_reference"),
                     [] { return make_attention_prefill_reference_kernel(); }, _sh_attn); !s.ok()) { return s; }
    if (auto s = get(_merge_cache, vision_variant_key("spatial_merge"),
                     [] { return make_spatial_merge_kernel(); }, _sh_merge); !s.ok()) { return s; }
    if (auto s = get(_residual_cache, norm_variant_key("residual_add"),
                     [] { return make_residual_add_kernel(); }, _sh_residual); !s.ok()) { return s; }
    return Status::success();
}

// ---------------------------------------------------------------------------
// linear helper (vision path)
// ---------------------------------------------------------------------------

[[nodiscard]] Status Qwen3VlExecutor::record_linear(CommandList &list,
                                                    BufferView<float> A, BufferView<float> B,
                                                    BufferView<float> C, BufferView<float> aux,
                                                    uint32_t m, uint32_t n, uint32_t k,
                                                    GemmEpilogue epilogue) {
    if (m == 0u || n == 0u || k == 0u) {
        return Status::error(StatusCode::invalid_argument, "linear dimensions must be nonzero", "m/n/k");
    }
    const auto aux_view = epilogue == GemmEpilogue::none ? vws(_v_dummy_f) : aux;
    auto plan = plan_gemm_dispatch_epilogue(m, n, k, epilogue);
    if (!plan.ok()) { return plan.status; }
    const auto *shader = epilogue == GemmEpilogue::bias ? _sh_gemm_bias
                         : epilogue == GemmEpilogue::bias_gelu_erf ? _sh_gemm_bias_gelu
                         : epilogue == GemmEpilogue::bias_residual ? _sh_gemm_bias_residual
                                                                   : _sh_gemm_none;
    if (auto cmd = dispatch_2d((*shader)(A, B, C, aux_view, m, n, k, plan.value->padded_cols), *plan.value)) {
        list << std::move(cmd);
    }
    return Status::success();
}

// ---------------------------------------------------------------------------
// vision block
// ---------------------------------------------------------------------------

[[nodiscard]] Status Qwen3VlExecutor::record_vision_block(CommandList &list, uint32_t layer,
                                                          BufferView<float> x,
                                                          BufferView<int32_t> positions,
                                                          uint32_t rows) {
    const auto &v = _config.model.vision;
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
    auto qn = vws(_v_qn).subview(0, static_cast<size_t>(rows) * vh);
    auto kn = vws(_v_kn).subview(0, static_cast<size_t>(rows) * vh);
    auto qr = vws(_v_qr).subview(0, static_cast<size_t>(rows) * vh);
    auto kr = vws(_v_kr).subview(0, static_cast<size_t>(rows) * vh);
    auto attn = vws(_v_attn).subview(0, static_cast<size_t>(rows) * vh);
    auto fc1 = vws(_v_fc1).subview(0, static_cast<size_t>(rows) * vi);
    auto fc2 = vws(_v_fc2).subview(0, static_cast<size_t>(rows) * vh);
    auto axis = vws(_v_axis).as<uint32_t>();
    auto dummy_f = vws(_v_dummy_f);
    auto dummy_u = vws(_v_dummy_u).as<uint32_t>();

    // 1. Pre-attention LayerNorm (weight + bias).
    {
        auto plan = plan_norm_dispatch(rows);
        if (!plan.ok()) { return plan.status; }
        if (auto cmd = dispatch_1d((*_sh_layer_norm)(x, norm, B.norm1_w, B.norm1_b, rows, vh, eps, 1u), *plan.value)) {
            list << std::move(cmd);
        }
    }
    // 2. QKV projections with bias.
    if (auto s = record_linear(list, norm, B.q_w, q, B.q_b, rows, vh, vh, GemmEpilogue::bias); !s.ok()) { return s; }
    if (auto s = record_linear(list, norm, B.k_w, k, B.k_b, rows, vh, vh, GemmEpilogue::bias); !s.ok()) { return s; }
    if (auto s = record_linear(list, norm, B.v_w, vv, B.v_b, rows, vh, vh, GemmEpilogue::bias); !s.ok()) { return s; }
    // 3. QK-Norm: per-head RMSNorm over head_dim.
    {
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
    }
    // 4. 2D axial vision RoPE (adjacent pairs; height/width axis map).
    {
        auto plan_q = plan_rope_dispatch(rows, heads);
        if (!plan_q.ok()) { return plan_q.status; }
        if (auto cmd = dispatch_1d((*_sh_rope2d)(qn, qr, positions, axis, rows, heads, hd, 2u, theta), *plan_q.value)) {
            list << std::move(cmd);
        }
        auto plan_k = plan_rope_dispatch(rows, heads);
        if (!plan_k.ok()) { return plan_k.status; }
        if (auto cmd = dispatch_1d((*_sh_rope2d)(kn, kr, positions, axis, rows, heads, hd, 2u, theta), *plan_k.value)) {
            list << std::move(cmd);
        }
    }
    // 5. Bidirectional attention WITHIN this image (MHA: kv_heads == heads).
    {
        auto plan = plan_attention_prefill_dispatch(rows, heads);
        if (!plan.ok()) { return plan.status; }
        if (auto cmd = dispatch_1d((*_sh_attn)(qr, kr, vv, attn, dummy_u, dummy_f,
                                               rows, rows, heads, heads, hd, scale, 0.0f,
                                               static_cast<uint32_t>(AttentionMaskKind::bidirectional),
                                               0u, 0u, 0u, 0u), *plan.value)) {
            list << std::move(cmd);
        }
    }
    // 6. Output projection + bias + residual: x = x + attn @ Wo + bo.
    //    bias_residual aux = [bias vh | residual rows*vh]; stage the residual
    //    (current x) after the bias into the aux scratch.
    {
        auto aux = vws(_v_aux).subview(0, static_cast<size_t>(vh) + static_cast<size_t>(rows) * vh);
        // Copy bias into aux[0..vh) and x into aux[vh..). Both are device
        // copies; use mm_scatter-free plain copies via residual_add with a
        // zero source is NOT a copy, so record two explicit copy commands
        // through the patchify-free path: buffer views support copy_from
        // only host->device. Use the residual_add kernel with a zero buffer
        // is wrong; instead the GEMM reads the residual from aux, so we fill
        // aux with a dedicated copy kernel: reuse mm_scatter (count=0 copy).
        // The text executor's mm_scatter shader is not compiled here; use
        // the spatial_merge kernel with m=1 (identity token packing) which
        // copies [rows, H] verbatim.
        auto copy_rows = [&](BufferView<float> src, BufferView<float> dst, uint32_t r, uint32_t h) -> Status {
            auto plan = plan_spatial_merge_dispatch(r, h);
            if (!plan.ok()) { return plan.status; }
            if (auto cmd = dispatch_1d((*_sh_merge)(src, dst, r, h, h, 1u), *plan.value)) {
                list << std::move(cmd);
            }
            return Status::success();
        };
        if (auto s = copy_rows(B.o_b, aux.subview(0, vh), 1u, vh); !s.ok()) { return s; }
        if (auto s = copy_rows(x, aux.subview(vh, static_cast<size_t>(rows) * vh), rows, vh); !s.ok()) { return s; }
        if (auto s = record_linear(list, attn, B.o_w, x, aux, rows, vh, vh, GemmEpilogue::bias_residual); !s.ok()) { return s; }
    }
    // 7. Post-attention LayerNorm.
    {
        auto plan = plan_norm_dispatch(rows);
        if (!plan.ok()) { return plan.status; }
        if (auto cmd = dispatch_1d((*_sh_layer_norm)(x, norm, B.norm2_w, B.norm2_b, rows, vh, eps, 1u), *plan.value)) {
            list << std::move(cmd);
        }
    }
    // 8. MLP: fc1 + bias + GELU-erf, fc2 + bias + residual.
    if (auto s = record_linear(list, norm, B.fc1_w, fc1, B.fc1_b, rows, vi, vh, GemmEpilogue::bias_gelu_erf); !s.ok()) { return s; }
    {
        auto aux = vws(_v_aux).subview(0, static_cast<size_t>(vh) + static_cast<size_t>(rows) * vh);
        auto copy_rows = [&](BufferView<float> src, BufferView<float> dst, uint32_t r, uint32_t h) -> Status {
            auto plan = plan_spatial_merge_dispatch(r, h);
            if (!plan.ok()) { return plan.status; }
            if (auto cmd = dispatch_1d((*_sh_merge)(src, dst, r, h, h, 1u), *plan.value)) {
                list << std::move(cmd);
            }
            return Status::success();
        };
        if (auto s = copy_rows(B.fc2_b, aux.subview(0, vh), 1u, vh); !s.ok()) { return s; }
        if (auto s = copy_rows(x, aux.subview(vh, static_cast<size_t>(rows) * vh), rows, vh); !s.ok()) { return s; }
        if (auto s = record_linear(list, fc1, B.fc2_w, x, aux, rows, vh, vi, GemmEpilogue::bias_residual); !s.ok()) { return s; }
    }
    return Status::success();
}

// ---------------------------------------------------------------------------
// merger
// ---------------------------------------------------------------------------

[[nodiscard]] Status Qwen3VlExecutor::record_merger(CommandList &list, const MergerWeights &m,
                                                    BufferView<float> x, uint32_t patch_rows,
                                                    uint32_t grid_w, uint32_t out_dim,
                                                    BufferView<float> out) {
    const auto &v = _config.model.vision;
    const uint32_t merge = v.spatial_merge_size;
    const uint32_t merged_rows = patch_rows / (merge * merge);
    const uint32_t merged_dim = v.merged_dim();
    const uint32_t mi = v.merger_intermediate_size;
    const float eps = v.layer_norm_eps;

    auto merged = vws(_v_merged).subview(0, static_cast<size_t>(patch_rows) * v.hidden_size);
    auto mnorm = vws(_v_mnorm).subview(0, static_cast<size_t>(merged_rows) * merged_dim);
    auto mfc1 = vws(_v_mfc1).subview(0, static_cast<size_t>(merged_rows) * mi);

    // 1. Spatial merge: [patch_rows, vh] -> [merged_rows, vh*m*m].
    {
        auto plan = plan_spatial_merge_dispatch(merged_rows, merged_dim);
        if (!plan.ok()) { return plan.status; }
        if (auto cmd = dispatch_1d((*_sh_merge)(x, merged, patch_rows, v.hidden_size, grid_w, merge), *plan.value)) {
            list << std::move(cmd);
        }
    }
    // 2. LayerNorm over the merged dim.
    {
        auto plan = plan_norm_dispatch(merged_rows);
        if (!plan.ok()) { return plan.status; }
        if (auto cmd = dispatch_1d((*_sh_layer_norm)(merged, mnorm, m.ln_w, m.ln_b, merged_rows, merged_dim, eps, 1u), *plan.value)) {
            list << std::move(cmd);
        }
    }
    // 3. fc1 + bias + GELU-erf.
    if (auto s = record_linear(list, mnorm, m.fc1_w, mfc1, m.fc1_b, merged_rows, mi, merged_dim, GemmEpilogue::bias_gelu_erf); !s.ok()) { return s; }
    // 4. fc2 + bias -> out.
    return record_linear(list, mfc1, m.fc2_w, out, m.fc2_b, merged_rows, out_dim, mi, GemmEpilogue::bias);
}

// ---------------------------------------------------------------------------
// vision graph
// ---------------------------------------------------------------------------

[[nodiscard]] Result<Qwen3VlVisionOutput> Qwen3VlExecutor::record_vision(CommandList &list,
                                                                         luisa::span<const Qwen3VlImageInput> images) {
    using R = Result<Qwen3VlVisionOutput>;
    if (_device == nullptr) {
        return R::failure(StatusCode::invalid_argument, "executor is not created", "executor");
    }
    if (images.empty()) {
        return R::failure(StatusCode::invalid_argument, "vision requires at least one image", "images");
    }
    const auto &v = _config.model.vision;
    const uint32_t text_hidden = _config.model.text.hidden_size;

    // Validate + preprocess every image on the host (Phase 1 reference: the
    // exact declared bilinear convention), and check the total patch budget.
    // M2: the FULL packing contract is validated with checked math BEFORE any
    // upload/dispatch, so forged metadata can never reach the GPU.
    uint64_t total_patches = 0, total_merged = 0;
    luisa::vector<Qwen3VlPreprocessedImage> preprocessed;
    preprocessed.reserve(images.size());
    const uint64_t merge = v.spatial_merge_size;
    const uint64_t merge_sq = merge * merge;
    for (const auto &img : images) {
        if (img.pixels.size() != static_cast<size_t>(img.source_width) * img.source_height * 3u) {
            return R::failure(StatusCode::invalid_argument, "image pixel buffer size mismatch", "pixels");
        }
        const auto &p = img.packed;
        if (p.spatial_merge_size != v.spatial_merge_size) {
            return R::failure(StatusCode::invalid_argument, "image packing merge size does not match the executor", "packed");
        }
        // Still-image contract: exactly one temporal frame.
        if (p.grid_t != 1u) {
            return R::failure(StatusCode::invalid_argument, "image packing grid_t must be 1 (still image)", "packed.grid_t");
        }
        if (p.grid_h == 0u || p.grid_w == 0u) {
            return R::failure(StatusCode::invalid_argument, "image packing grid must be nonzero", "packed.grid");
        }
        // patch_count == grid_t * grid_h * grid_w (checked).
        uint64_t want_patches = 0;
        if (!checked_mul(p.grid_h, p.grid_w, want_patches) ||
            !checked_mul(want_patches, p.grid_t, want_patches)) {
            return R::failure(StatusCode::invalid_argument, "image packing patch count overflows", "packed.patch_count");
        }
        if (p.patch_count != want_patches) {
            return R::failure(StatusCode::invalid_argument,
                              luisa::format("image packing patch_count {} does not equal grid_h*grid_w*grid_t ({})",
                                            p.patch_count, want_patches),
                              "packed.patch_count");
        }
        // Grid divisible by the merge size (both axes).
        if (p.grid_h % merge != 0u || p.grid_w % merge != 0u) {
            return R::failure(StatusCode::invalid_argument,
                              "image packing grid is not divisible by the spatial merge size", "packed.grid");
        }
        // merged_count == patch_count / merge^2 (checked; exact division).
        if (p.patch_count % merge_sq != 0u || p.merged_count != p.patch_count / merge_sq) {
            return R::failure(StatusCode::invalid_argument,
                              luisa::format("image packing merged_count {} does not equal patch_count/merge^2 ({})",
                                            p.merged_count, p.patch_count / merge_sq),
                              "packed.merged_count");
        }
        if (p.merged_count == 0u) {
            return R::failure(StatusCode::invalid_argument, "image packing merged_count must be nonzero", "packed.merged_count");
        }
        // Grid <-> resized size consistency (checked).
        uint64_t want_h = 0, want_w = 0;
        if (!checked_mul(p.grid_h, v.patch_size, want_h) || !checked_mul(p.grid_w, v.patch_size, want_w) ||
            want_h != p.resized_height || want_w != p.resized_width) {
            return R::failure(StatusCode::invalid_argument, "image packing grid does not match the resized size", "packed");
        }
        auto pre = qwen3_vl_preprocess_image_host(img, _config.mean, _config.std_dev);
        if (!pre.ok()) { return R::failure(pre.status); }
        if (pre.value->width != p.resized_width || pre.value->height != p.resized_height) {
            return R::failure(StatusCode::internal_error, "preprocessed size does not match the packing contract", "preprocess");
        }
        // Checked accumulation + arena bounds (patch and merged arenas).
        uint64_t new_patches = 0, new_merged = 0;
        if (!checked_add(total_patches, p.patch_count, new_patches) ||
            !checked_add(total_merged, p.merged_count, new_merged)) {
            return R::failure(StatusCode::out_of_memory, "vision patch/merged totals overflow", "packed");
        }
        if (new_patches > _config.max_vision_patches) {
            return R::failure(StatusCode::out_of_memory,
                              luisa::format("total vision patches {} exceed the workspace capacity {}",
                                            new_patches, _config.max_vision_patches),
                              "max_vision_patches");
        }
        const uint64_t max_merged = static_cast<uint64_t>(_config.max_vision_patches) / merge_sq;
        if (new_merged > max_merged) {
            return R::failure(StatusCode::out_of_memory,
                              luisa::format("total merged tokens {} exceed the feature arena capacity {}",
                                            new_merged, max_merged),
                              "max_vision_patches");
        }
        total_patches = new_patches;
        total_merged = new_merged;
        preprocessed.push_back(std::move(*pre.value));
    }

    // Per-record heap-owned staging (pixels + positions + axis + norm consts).
    auto staging = luisa::make_unique<VisionUploadStaging>();
    staging->axis = _vision_axis_pending;
    staging->mean = _config.mean;
    staging->std_dev = _config.std_dev;
    for (const auto &img : images) {
        auto pos = qwen3_vl_vision_positions(img.packed);
        staging->positions.insert(staging->positions.end(), pos.begin(), pos.end());
    }
    for (const auto &pre : preprocessed) {
        staging->pixels.insert(staging->pixels.end(), pre.pixels.begin(), pre.pixels.end());
    }
    auto *stage = staging.get();

    // Upload the constant axis map once per record.
    list << vws(_v_axis).as<uint32_t>().copy_from(luisa::span<const uint32_t>{stage->axis});

    Qwen3VlVisionOutput output;
    output.total_merged = static_cast<uint32_t>(total_merged);
    uint64_t patch_offset = 0, merged_offset = 0, pixel_offset = 0, pos_offset = 0;
    for (size_t i = 0; i < images.size(); i++) {
        const auto &img = images[i];
        const auto &pre = preprocessed[i];
        const uint32_t rows = static_cast<uint32_t>(img.packed.patch_count);
        const uint32_t merged_rows = static_cast<uint32_t>(img.packed.merged_count);
        const uint64_t pixel_elems = static_cast<uint64_t>(pre.pixels.size());

        // Upload this image's normalized CHW pixels + vision positions.
        auto pixels = vws(_v_pixels).subview(0, static_cast<size_t>(pixel_elems));
        auto positions = vws(_v_positions).as<int32_t>().subview(0, static_cast<size_t>(rows) * 2u);
        list << pixels.copy_from(luisa::span<const float>{stage->pixels.data() + pixel_offset, static_cast<size_t>(pixel_elems)})
             << positions.copy_from(luisa::span<const int32_t>{stage->positions.data() + pos_offset, static_cast<size_t>(rows) * 2u});

        // Patchify: CHW -> [rows, patch_dim].
        auto patches = vws(_v_patches).subview(0, static_cast<size_t>(rows) * v.patch_dim());
        {
            auto plan = plan_image_patchify_dispatch(rows, v.patch_dim());
            if (!plan.ok()) { return R::failure(plan.status); }
            if (auto cmd = dispatch_1d((*_sh_patchify)(pixels, patches, v.channels, pre.height, pre.width, v.patch_size), *plan.value)) {
                list << std::move(cmd);
            }
        }
        // Patch embed: [rows, patch_dim] @ [patch_dim, vh] + bias.
        auto x = vws(_v_x).subview(0, static_cast<size_t>(rows) * v.hidden_size);
        if (auto s = record_linear(list, patches, _patch_embed_w, x, _patch_embed_b, rows, v.hidden_size, v.patch_dim(), GemmEpilogue::bias); !s.ok()) {
            return R::failure(s);
        }
        // Vision blocks (bidirectional within this image only).
        for (uint32_t l = 0; l < v.num_layers; l++) {
            if (auto s = record_vision_block(list, l, x, positions, rows); !s.ok()) { return R::failure(s); }
            // DeepStack: capture the output of the selected vision block into
            // the corresponding merger input. The block output is x; the
            // DeepStack merger consumes it BEFORE the next block overwrites x.
            for (uint32_t di = 0; di < _config.model.deepstack.count(); di++) {
                if (_config.model.deepstack.vision_layers[di] != l) { continue; }
                auto ds_out = vws(_v_ds_features[di]).subview(static_cast<size_t>(merged_offset) * text_hidden,
                                                              static_cast<size_t>(merged_rows) * text_hidden);
                if (auto s = record_merger(list, _ds_mergers[di], x, rows, img.packed.grid_w, text_hidden, ds_out); !s.ok()) {
                    return R::failure(s);
                }
            }
        }
        // Main merger -> features.
        auto feat = vws(_v_features).subview(static_cast<size_t>(merged_offset) * text_hidden,
                                             static_cast<size_t>(merged_rows) * text_hidden);
        if (auto s = record_merger(list, _merger, x, rows, img.packed.grid_w, text_hidden, feat); !s.ok()) {
            return R::failure(s);
        }

        patch_offset += rows;
        merged_offset += merged_rows;
        pixel_offset += pixel_elems;
        pos_offset += static_cast<uint64_t>(rows) * 2u;
    }

    output.features = vws(_v_features).subview(0, static_cast<size_t>(total_merged) * text_hidden);
    for (uint32_t di = 0; di < _config.model.deepstack.count(); di++) {
        output.deepstack_features.push_back(
            vws(_v_ds_features[di]).subview(0, static_cast<size_t>(total_merged) * text_hidden));
    }
    _last_total_patches = static_cast<uint32_t>(total_patches);
    _last_total_merged = static_cast<uint32_t>(total_merged);

    // Retain the upload staging through command completion.
    list.add_callback([staging = std::move(staging)]() mutable { staging.reset(); });
    return R::success(std::move(output));
}

// ---------------------------------------------------------------------------
// prompt assembly
// ---------------------------------------------------------------------------

[[nodiscard]] Result<DecodeRequest> Qwen3VlExecutor::build_decode_request(
    const Qwen3VlPrompt &prompt, const Qwen3VlVisionOutput &vision,
    luisa::span<const Qwen3VlImageInput> images,
    luisa::vector<int32_t> &position_ids_out) const {
    using R = Result<DecodeRequest>;
    const auto &t = _config.model.text;
    if (auto s = prompt.validate(t.vocab_size); !s.ok()) { return R::failure(s); }
    if (prompt.image_ranges.size() != images.size()) {
        return R::failure(StatusCode::invalid_argument,
                          "prompt image ranges must match the image count", "image_ranges");
    }
    if (images.size() > 1u) {
        return R::failure(StatusCode::unsupported,
                          "the Phase 4 slice supports one image per prompt (single multimodal range)", "images");
    }
    // Validate each range against its image's merged token count.
    luisa::vector<Qwen3VlPromptSegment> segments;
    uint32_t cursor = 0u;
    for (size_t i = 0; i < images.size(); i++) {
        const auto &r = prompt.image_ranges[i];
        if (r.count != images[i].packed.merged_count) {
            return R::failure(StatusCode::invalid_argument,
                              "image placeholder count does not match the packed merged token count", "image_ranges");
        }
        if (r.start > cursor) {
            Qwen3VlPromptSegment text_seg;
            text_seg.text_tokens = r.start - cursor;
            segments.push_back(text_seg);
        }
        Qwen3VlPromptSegment img_seg;
        img_seg.image = &images[i].packed;
        segments.push_back(img_seg);
        cursor = r.start + r.count;
    }
    if (cursor < prompt.token_ids.size()) {
        Qwen3VlPromptSegment text_seg;
        text_seg.text_tokens = static_cast<uint32_t>(prompt.token_ids.size()) - cursor;
        segments.push_back(text_seg);
    }
    auto pos = qwen3_vl_build_position_ids(segments);
    if (!pos.ok()) { return R::failure(pos.status); }
    if (pos.value->seq_len != prompt.token_ids.size()) {
        return R::failure(StatusCode::internal_error, "position id count does not match the prompt", "position_ids");
    }
    position_ids_out = std::move(pos.value->ids);

    DecodeRequest request;
    request.prompt_tokens = prompt.token_ids;
    if (!images.empty()) {
        request.mm_features = vision.features;
        request.mm_start = prompt.image_ranges[0].start;
        request.mm_count = prompt.image_ranges[0].count;
    }
    return R::success(std::move(request));
}

// ---------------------------------------------------------------------------
// download helpers (layered golden tests)
// ---------------------------------------------------------------------------

[[nodiscard]] Status Qwen3VlExecutor::download_vision_patches(Stream &stream, luisa::vector<float> &out) const {
    const uint64_t count = static_cast<uint64_t>(_last_total_patches) * _config.model.vision.patch_dim();
    out.resize(static_cast<size_t>(count));
    if (count != 0u) {
        stream << vws(_v_patches).subview(0, static_cast<size_t>(count)).copy_to(luisa::span<float>{out});
    }
    return Status::success();
}

[[nodiscard]] Status Qwen3VlExecutor::download_vision_final(Stream &stream, luisa::vector<float> &out) const {
    // S11: the FINAL vision residual stream (after ALL vision blocks) remains
    // in _v_x. For an isolated first-block golden the test runs a 1-block
    // executor, so this stream IS the block-0 output.
    const uint64_t count = static_cast<uint64_t>(_last_total_patches) * _config.model.vision.hidden_size;
    out.resize(static_cast<size_t>(count));
    if (count != 0u) {
        stream << vws(_v_x).subview(0, static_cast<size_t>(count)).copy_to(luisa::span<float>{out});
    }
    return Status::success();
}

[[nodiscard]] Status Qwen3VlExecutor::download_vision_features(Stream &stream, luisa::vector<float> &out) const {
    const uint64_t count = static_cast<uint64_t>(_last_total_merged) * _config.model.text.hidden_size;
    out.resize(static_cast<size_t>(count));
    if (count != 0u) {
        stream << vws(_v_features).subview(0, static_cast<size_t>(count)).copy_to(luisa::span<float>{out});
    }
    return Status::success();
}

[[nodiscard]] Status Qwen3VlExecutor::download_deepstack_features(Stream &stream, uint32_t index,
                                                                  luisa::vector<float> &out) const {
    if (index >= _v_ds_features.size()) {
        return Status::error(StatusCode::invalid_argument, "deepstack index out of range", "index");
    }
    const uint64_t count = static_cast<uint64_t>(_last_total_merged) * _config.model.text.hidden_size;
    out.resize(static_cast<size_t>(count));
    if (count != 0u) {
        stream << vws(_v_ds_features[index]).subview(0, static_cast<size_t>(count)).copy_to(luisa::span<float>{out});
    }
    return Status::success();
}

}// namespace luisa::ocr

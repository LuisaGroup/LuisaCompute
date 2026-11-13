// OCR engine -- layout/region orchestration implementation (Phase 6).
// See the header for the contract. Device-safe TU (no api.h/model_recipe.h).
#include "ocr/runtime/layout_vlm_executor.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <utility>

#include <luisa/core/clock.h>
#include <luisa/core/logging.h>
#include <luisa/core/stl/format.h>
#include <luisa/runtime/stream.h>

#include "ocr/runtime/dispatch_adapter.h"
#include "ocr/runtime/qwen3_vl_image.h"

namespace luisa::ocr {

using namespace luisa::compute;

// Host preprocessing bridge (runtime/qwen3_vl_preprocess.cpp). Forward-declared
// over the shared plain types so this device TU never includes preprocess.h ->
// api.h -> model_recipe.h (host AttentionMaskKind clash). The bridge runs the
// Phase 1 bilinear stretch-resize + normalize reference.
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

// Host STRETCH-resize + normalize of an HWC uint8 image to CHW float at a fixed
// target size (independent x/y scales, half-pixel centers, clamp edges). This is
// the declared synthetic convention shared with the CNN executor; it is NOT
// official preprocessing parity. Used by the layout detector overview pass.
[[nodiscard]] luisa::vector<float> stretch_normalize_host(luisa::span<const uint8_t> pixels,
                                                          uint32_t sw, uint32_t sh, uint32_t dw, uint32_t dh,
                                                          luisa::span<const float> mean, luisa::span<const float> std_dev) {
    luisa::vector<float> out(static_cast<size_t>(3) * dw * dh);
    const float scale_x = static_cast<float>(sw) / static_cast<float>(dw);
    const float scale_y = static_cast<float>(sh) / static_cast<float>(dh);
    const auto sample = [&](uint32_t c, float sx, float sy) -> float {
        const float x0f = std::floor(sx);
        const float y0f = std::floor(sy);
        const int32_t x0 = static_cast<int32_t>(x0f);
        const int32_t y0 = static_cast<int32_t>(y0f);
        const float fx = sx - x0f;
        const float fy = sy - y0f;
        const auto at = [&](int32_t px, int32_t py) -> float {
            const int32_t cx = std::clamp(px, 0, static_cast<int32_t>(sw) - 1);
            const int32_t cy = std::clamp(py, 0, static_cast<int32_t>(sh) - 1);
            return static_cast<float>(pixels[(static_cast<size_t>(cy) * sw + cx) * 3u + c]);
        };
        const float v00 = at(x0, y0);
        const float v10 = at(x0 + 1, y0);
        const float v01 = at(x0, y0 + 1);
        const float v11 = at(x0 + 1, y0 + 1);
        const float top = v00 + (v10 - v00) * fx;
        const float bottom = v01 + (v11 - v01) * fx;
        return top + (bottom - top) * fy;
    };
    for (uint32_t y = 0; y < dh; y++) {
        for (uint32_t x = 0; x < dw; x++) {
            const float sx = (static_cast<float>(x) + 0.5f) * scale_x - 0.5f;
            const float sy = (static_cast<float>(y) + 0.5f) * scale_y - 0.5f;
            for (uint32_t c = 0; c < 3u; c++) {
                const float v = sample(c, sx, sy) / 255.0f;
                out[static_cast<size_t>(c) * dw * dh + static_cast<size_t>(y) * dw + x] = (v - mean[c]) / std_dev[c];
            }
        }
    }
    return out;
}

// Per-record heap-owned upload staging (pixels + positions + axis). Retained by
// a completion callback so borrowed host spans outlive command execution.
struct VisionUploadStaging {
    luisa::vector<float> pixels;
    luisa::vector<int32_t> positions;
    luisa::vector<uint32_t> axis;
};

}// namespace

// ---------------------------------------------------------------------------
// PpDocLayoutExecutorConfig
// ---------------------------------------------------------------------------

[[nodiscard]] Status PpDocLayoutExecutorConfig::validate() const {
    if (auto s = model.validate(); !s.ok()) { return s; }
    if (auto s = names.validate(); !s.ok()) { return s; }
    if (component.empty()) {
        return Status::error(StatusCode::invalid_argument, "weight component tag must not be empty", "component");
    }
    if (mean.size() != 3u || std_dev.size() != 3u) {
        return Status::error(StatusCode::invalid_argument, "mean/std_dev must have 3 channels", "mean/std_dev");
    }
    for (auto v : std_dev) {
        if (!std::isfinite(v) || v == 0.0f) {
            return Status::error(StatusCode::invalid_argument, "std_dev must be finite and nonzero", "std_dev");
        }
    }
    if (max_source_pixels == 0u) {
        return Status::error(StatusCode::invalid_argument, "max_source_pixels must be nonzero", "max_source_pixels");
    }
    return Status::success();
}

// ---------------------------------------------------------------------------
// PpDocLayoutExecutor: create / resolve / reserve / compile
// ---------------------------------------------------------------------------

[[nodiscard]] Status PpDocLayoutExecutor::create(Device &device, Stream &stream,
                                                 const PpDocLayoutExecutorConfig &config,
                                                 const WeightStore &weights) {
    auto valid = config.validate();
    if (!valid.ok()) { return valid; }
    _config = config;
    _device = &device;
    _ws = DeviceWorkspace{device};
    _conv_cache = TypedShaderCache<Conv2dKernel>{device, _shader_registry};
    _act_cache = TypedShaderCache<CnnActKernel>{device, _shader_registry};
    _sigmoid_cache = TypedShaderCache<SigmoidKernel>{device, _shader_registry};
    auto mh = config.model.map_height();
    auto mw = config.model.map_width();
    if (!mh.ok() || !mw.ok()) { return Status::error(StatusCode::internal_error, "layout geometry", "map"); }
    _map_h = *mh.value;
    _map_w = *mw.value;
    if (auto s = resolve_weights(weights); !s.ok()) { return s; }
    if (auto s = reserve_workspace(); !s.ok()) { return s; }
    if (auto s = compile_shaders(); !s.ok()) { return s; }
    _weight_bytes = weights.component_bytes(config.component);
    (void)stream;
    return Status::success();
}

[[nodiscard]] Status PpDocLayoutExecutor::create_from_packages(Device &device, Stream &stream,
                                                               const PpDocLayoutExecutorConfig &config,
                                                               const WeightStore &weights,
                                                               const PackageReader &layout_package) {
    if (auto s = config.model.validate(); !s.ok()) { return s; }
    if (auto s = config.names.validate(); !s.ok()) { return s; }
    const auto expected = pp_doc_layout_expected_tensors(config.model, config.names);
    if (auto s = pp_doc_layout_validate_tensors(layout_package.tensors(), expected); !s.ok()) { return s; }
    return create(device, stream, config, weights);
}

[[nodiscard]] Status PpDocLayoutExecutor::resolve_weights(const WeightStore &weights) {
    const auto &names = _config.names;
    const auto &comp = _config.component;
    const auto expected = pp_doc_layout_device_expected_tensors(_config.model, names);
    luisa::unordered_map<luisa::string, const ExpectedTensor *> by_name;
    for (const auto &e : expected) { by_name.emplace(e.name, &e); }
    for (const auto &name : weights.tensor_names(comp)) {
        if (by_name.find(name) == by_name.end()) {
            return Status::error(StatusCode::invalid_argument,
                                 luisa::format("unexpected layout tensor '{}' in component '{}'", name, comp), name);
        }
    }
    const auto view_of = [&](const luisa::string &name, BufferView<float> &out) -> Status {
        const auto *w = weights.find(comp, name);
        if (w == nullptr) {
            return Status::error(StatusCode::not_found,
                                 luisa::format("layout weight '{}' not found in component '{}'", name, comp), name);
        }
        if (w->encoding != TensorEncoding::fp32 || !w->dense) {
            return Status::error(StatusCode::unsupported,
                                 luisa::format("layout weight '{}' must be dense fp32 in the Phase 6 baseline", name), name);
        }
        const auto it = by_name.find(name);
        if (it == by_name.end()) {
            return Status::error(StatusCode::invalid_argument,
                                 luisa::format("layout weight '{}' has no expected-tensor contract", name), name);
        }
        if (auto s = validate_decoder_weight_desc(w->logical, w->element_count, *it->second); !s.ok()) { return s; }
        if (w->dense.size() < w->element_count) {
            return Status::error(StatusCode::invalid_argument,
                                 luisa::format("layout weight '{}' buffer smaller than element count", name), name);
        }
        out = w->dense.view();
        return Status::success();
    };
    _backbone.resize(_config.model.backbone.size());
    for (uint32_t i = 0; i < _config.model.backbone.size(); i++) {
        if (auto s = view_of(names.backbone_w(i), _backbone[i].weight); !s.ok()) { return s; }
        if (auto s = view_of(names.backbone_b(i), _backbone[i].bias); !s.ok()) { return s; }
    }
    if (auto s = view_of(names.class_head_w(), _class_head.weight); !s.ok()) { return s; }
    if (auto s = view_of(names.class_head_b(), _class_head.bias); !s.ok()) { return s; }
    if (auto s = view_of(names.box_head_w(), _box_head.weight); !s.ok()) { return s; }
    if (auto s = view_of(names.box_head_b(), _box_head.bias); !s.ok()) { return s; }
    return Status::success();
}

[[nodiscard]] Status PpDocLayoutExecutor::reserve_workspace() {
    const auto &model = _config.model;
    const uint64_t dp = static_cast<uint64_t>(model.input_width) * model.input_height;
    uint64_t feat_max = 0;
    {
        uint32_t h = model.input_height, w = model.input_width, c = model.in_channels;
        for (const auto &s : model.backbone) {
            auto oh = conv2d_out_dim(h, s.kh, s.stride_h, s.pad_h, s.dil_h);
            auto ow = conv2d_out_dim(w, s.kw, s.stride_w, s.pad_w, s.dil_w);
            if (!oh.ok() || !ow.ok()) { return oh.status; }
            h = *oh.value;
            w = *ow.value;
            c = s.out_c;
            feat_max = std::max(feat_max, static_cast<uint64_t>(c) * h * w);
        }
    }
    const uint64_t plane = static_cast<uint64_t>(_map_h) * _map_w;
    const uint64_t nc = model.num_classes();
    struct RegionSpec {
        const char *name;
        uint64_t elements;
    };
    const RegionSpec specs[] = {
        {"page_pixels", 3u * dp},
        {"feat_a", feat_max},
        {"feat_b", feat_max},
        {"class_logits", nc * plane},
        {"class_probs", nc * plane},
        {"box_deltas", 4u * plane},
        {"dummy_f", 1},
    };
    WorkspacePlan plan;
    for (const auto &spec : specs) {
        auto bytes = f32_bytes(spec.elements);
        if (!bytes.ok()) { return bytes.status; }
        if (auto s = plan.add_bytes(spec.name, *bytes.value); !s.ok()) { return s; }
    }
    MemoryBudget budget{_config.memory};
    if (auto s = budget.require("layout_workspace", MemoryCategory::workspace, plan.total_bytes()); !s.ok()) { return s; }
    if (auto s = budget.preflight(); !s.ok()) { return s; }
    if (auto s = _ws.reserve(plan.total_bytes()); !s.ok()) { return s; }
    WsRegion *outs[] = {&_page_pixels, &_feat_a, &_feat_b, &_class_logits, &_class_probs, &_box_deltas, &_dummy_f};
    for (size_t i = 0; i < std::size(specs); i++) {
        auto bytes = f32_bytes(specs[i].elements);
        auto alloc = _ws.allocate(*bytes.value);
        if (!alloc.ok()) { return alloc.status; }
        outs[i]->offset = alloc.value->offset;
        outs[i]->bytes = alloc.value->bytes;
    }
    _ws.reset();
    return Status::success();
}

[[nodiscard]] Status PpDocLayoutExecutor::compile_shaders() {
    const auto get = [](auto &cache, const luisa::string &key, auto factory, auto *&out) -> Status {
        auto r = cache.get_or_compile(key, factory);
        if (!r.ok()) { return r.status; }
        out = *r.value;
        return Status::success();
    };
    if (auto s = get(_conv_cache, conv_variant_key("conv2d"), [] { return make_conv2d_kernel(); }, _sh_conv); !s.ok()) { return s; }
    for (uint32_t act = 0; act < 4u; act++) {
        if (auto s = get(_act_cache, ctc_variant_key(luisa::format("act_{}", act)),
                         [act] { return make_cnn_act_kernel(act); }, _sh_act[act]); !s.ok()) { return s; }
    }
    if (auto s = get(_sigmoid_cache, ctc_variant_key("sigmoid"), [] { return make_sigmoid_kernel(); }, _sh_sigmoid); !s.ok()) { return s; }
    return Status::success();
}

[[nodiscard]] Status PpDocLayoutExecutor::record_conv(CommandList &list, const PpConvStage &stage,
                                                      const ConvWeights &w, BufferView<float> in,
                                                      BufferView<float> out, uint32_t n,
                                                      uint32_t in_h, uint32_t in_w,
                                                      uint32_t &out_h, uint32_t &out_w) {
    auto oh = conv2d_out_dim(in_h, stage.kh, stage.stride_h, stage.pad_h, stage.dil_h);
    auto ow = conv2d_out_dim(in_w, stage.kw, stage.stride_w, stage.pad_w, stage.dil_w);
    if (!oh.ok()) { return oh.status; }
    if (!ow.ok()) { return ow.status; }
    out_h = *oh.value;
    out_w = *ow.value;
    if (auto s = validate_conv2d_inputs(n, stage.in_c, in_h, in_w, stage.out_c, stage.kh, stage.kw,
                                        stage.stride_h, stage.stride_w, stage.dil_h, stage.dil_w, stage.groups); !s.ok()) {
        return s;
    }
    auto plan = plan_conv2d_dispatch(n, stage.out_c, out_h, out_w);
    if (!plan.ok()) { return plan.status; }
    if (auto cmd = dispatch_1d((*_sh_conv)(in, w.weight, w.bias, out, n, stage.in_c, in_h, in_w,
                                           stage.out_c, stage.kh, stage.kw, stage.stride_h, stage.stride_w,
                                           stage.pad_h, stage.pad_w, stage.dil_h, stage.dil_w, stage.groups,
                                           out_h, out_w, 1u), *plan.value)) {
        list << std::move(cmd);
    }
    if (stage.act != 0u) {
        const uint64_t count = static_cast<uint64_t>(n) * stage.out_c * out_h * out_w;
        auto act_plan = plan_cnn_act_dispatch(count);
        if (!act_plan.ok()) { return act_plan.status; }
        auto count32 = checked_u32(count, "layout act count");
        if (!count32.ok()) { return count32.status; }
        if (auto cmd = dispatch_1d((*_sh_act[stage.act])(out, out, *count32.value), *act_plan.value)) {
            list << std::move(cmd);
        }
    }
    return Status::success();
}

[[nodiscard]] Result<LayoutHeadMaps> PpDocLayoutExecutor::detect(Stream &stream, const LayoutPageInput &page) {
    using R = Result<LayoutHeadMaps>;
    if (_device == nullptr) { return R::failure(StatusCode::invalid_argument, "layout executor not created", "device"); }
    const auto &model = _config.model;
    if (page.source_width == 0u || page.source_height == 0u ||
        page.pixels.size() != static_cast<size_t>(page.source_width) * page.source_height * 3u) {
        return R::failure(StatusCode::invalid_argument, "degenerate layout page input", "page");
    }
    const uint64_t source_pixels = static_cast<uint64_t>(page.source_width) * page.source_height;
    if (source_pixels > _config.max_source_pixels) {
        return R::failure(StatusCode::invalid_argument, "page exceeds max_source_pixels", "page");
    }
    auto pixels_host = stretch_normalize_host(page.pixels, page.source_width, page.source_height,
                                              model.input_width, model.input_height, _config.mean, _config.std_dev);
    return detect_from_pixels(stream, pixels_host);
}

[[nodiscard]] Result<LayoutHeadMaps> PpDocLayoutExecutor::detect_from_pixels(Stream &stream,
                                                                             luisa::span<const float> pixels_chw) {
    using R = Result<LayoutHeadMaps>;
    if (_device == nullptr) { return R::failure(StatusCode::invalid_argument, "layout executor not created", "device"); }
    const auto &model = _config.model;
    const size_t expect = static_cast<size_t>(3) * model.input_width * model.input_height;
    if (pixels_chw.size() != expect) {
        return R::failure(StatusCode::invalid_argument, "pixel buffer does not match the detector input", "pixels");
    }
    _ws.reset();
    auto page_pixels = ws(_page_pixels);
    stream << page_pixels.copy_from(pixels_chw);
    // Backbone (ping-pong feat_a/feat_b).
    BufferView<float> cur = page_pixels;
    uint32_t h = model.input_height, w = model.input_width;
    bool use_a = true;
    {
        auto list = CommandList::create();
        for (uint32_t i = 0; i < model.backbone.size(); i++) {
            auto out = use_a ? ws(_feat_a) : ws(_feat_b);
            uint32_t oh = 0, ow = 0;
            if (auto s = record_conv(list, model.backbone[i], _backbone[i], cur, out, 1u, h, w, oh, ow); !s.ok()) {
                return R::failure(s);
            }
            cur = out;
            h = oh;
            w = ow;
            use_a = !use_a;
        }
        // Class head 1x1 -> num_classes logits, then sigmoid.
        auto class_logits = ws(_class_logits);
        uint32_t ch = 0, cw = 0;
        if (auto s = record_conv(list, model.class_head, _class_head, cur, class_logits, 1u, h, w, ch, cw); !s.ok()) {
            return R::failure(s);
        }
        const uint64_t class_count = static_cast<uint64_t>(model.num_classes()) * ch * cw;
        {
            auto plan = plan_sigmoid_dispatch(class_count);
            if (!plan.ok()) { return R::failure(plan.status); }
            auto count32 = checked_u32(class_count, "layout sigmoid count");
            if (!count32.ok()) { return R::failure(count32.status); }
            if (auto cmd = dispatch_1d((*_sh_sigmoid)(class_logits, ws(_class_probs), *count32.value), *plan.value)) {
                list << std::move(cmd);
            }
        }
        // Box head 1x1 -> 4 deltas.
        uint32_t bh = 0, bw = 0;
        if (auto s = record_conv(list, model.box_head, _box_head, cur, ws(_box_deltas), 1u, h, w, bh, bw); !s.ok()) {
            return R::failure(s);
        }
        stream << list.commit() << synchronize();
    }
    // Download the head maps.
    LayoutHeadMaps maps;
    maps.map_width = _map_w;
    maps.map_height = _map_h;
    maps.num_classes = model.num_classes();
    const size_t plane = static_cast<size_t>(_map_h) * _map_w;
    maps.class_probs.resize(static_cast<size_t>(model.num_classes()) * plane);
    maps.box_deltas.resize(4u * plane);
    stream << ws(_class_probs).subview(0, maps.class_probs.size()).copy_to(luisa::span<float>{maps.class_probs})
           << ws(_box_deltas).subview(0, maps.box_deltas.size()).copy_to(luisa::span<float>{maps.box_deltas})
           << synchronize();
    return R::success(std::move(maps));
}

[[nodiscard]] Status PpDocLayoutExecutor::download_feat_a(Stream &stream, luisa::vector<float> &out) const {
    out.resize(static_cast<size_t>(_feat_a.bytes / 4u));
    stream << ws(_feat_a).copy_to(luisa::span<float>{out});
    return Status::success();
}

[[nodiscard]] Status PpDocLayoutExecutor::download_feat_b(Stream &stream, luisa::vector<float> &out) const {
    out.resize(static_cast<size_t>(_feat_b.bytes / 4u));
    stream << ws(_feat_b).copy_to(luisa::span<float>{out});
    return Status::success();
}

[[nodiscard]] Status PpDocLayoutExecutor::download_page_pixels(Stream &stream, luisa::vector<float> &out) const {
    out.resize(static_cast<size_t>(_page_pixels.bytes / 4u));
    stream << ws(_page_pixels).copy_to(luisa::span<float>{out});
    return Status::success();
}

// ---------------------------------------------------------------------------
// PaddleOcrVlRecognizerConfig
// ---------------------------------------------------------------------------

[[nodiscard]] Status PaddleOcrVlRecognizerConfig::validate() const {
    if (auto s = model.validate(); !s.ok()) { return s; }
    if (auto s = names.validate(); !s.ok()) { return s; }
    if (vision_component.empty() || decoder_component.empty()) {
        return Status::error(StatusCode::invalid_argument, "weight component tags must not be empty", "component");
    }
    if (vision_component == decoder_component) {
        return Status::error(StatusCode::invalid_argument, "vision and decoder components must be distinct", "component");
    }
    if (mean.size() != 3u || std_dev.size() != 3u) {
        return Status::error(StatusCode::invalid_argument, "mean/std_dev must have 3 channels", "mean/std_dev");
    }
    for (auto v : std_dev) {
        if (!std::isfinite(v) || v == 0.0f) {
            return Status::error(StatusCode::invalid_argument, "std_dev must be finite and nonzero", "std_dev");
        }
    }
    if (max_vision_patches == 0u) {
        return Status::error(StatusCode::invalid_argument, "max_vision_patches must be nonzero", "max_vision_patches");
    }
    if (prefill_chunk_rows == 0u) {
        return Status::error(StatusCode::invalid_argument, "prefill_chunk_rows must be nonzero", "prefill_chunk_rows");
    }
    if (prompt_prefix.empty() && prompt_suffix.empty()) {
        return Status::error(StatusCode::invalid_argument, "prompt must have at least one text token", "prompt");
    }
    for (auto id : prompt_prefix) {
        if (id >= model.text.vocab_size) {
            return Status::error(StatusCode::invalid_argument, "prompt prefix token outside the vocabulary", "prompt_prefix");
        }
    }
    for (auto id : prompt_suffix) {
        if (id >= model.text.vocab_size) {
            return Status::error(StatusCode::invalid_argument, "prompt suffix token outside the vocabulary", "prompt_suffix");
        }
    }
    return Status::success();
}

// ---------------------------------------------------------------------------
// PaddleOcrVlRecognizer: create / resolve / reserve / compile
// ---------------------------------------------------------------------------

[[nodiscard]] Status PaddleOcrVlRecognizer::create(Device &device, Stream &stream,
                                                   const PaddleOcrVlRecognizerConfig &config,
                                                   const WeightStore &weights) {
    auto valid = config.validate();
    if (!valid.ok()) { return valid; }
    _config = config;
    _device = &device;
    _vision_ws = DeviceWorkspace{device};
    _patchify_cache = TypedShaderCache<ImagePatchifyKernel>{device, _shader_registry};
    _layer_norm_cache = TypedShaderCache<LayerNormKernel>{device, _shader_registry};
    _gemm_cache = TypedShaderCache<GemmKernel>{device, _shader_registry};
    _rope2d_cache = TypedShaderCache<RopeAxialKernel>{device, _shader_registry};
    _attn_cache = TypedShaderCache<AttentionPrefillKernel>{device, _shader_registry};
    _pool_cache = TypedShaderCache<AdaptiveAvgPool2dKernel>{device, _shader_registry};
    _residual_cache = TypedShaderCache<ResidualAddKernel>{device, _shader_registry};
    _copy_cache = TypedShaderCache<SpatialMergeKernel>{device, _shader_registry};
    if (auto s = resolve_weights(weights); !s.ok()) { return s; }
    if (auto s = reserve_workspace(); !s.ok()) { return s; }
    if (auto s = compile_shaders(); !s.ok()) { return s; }
    _vision_axis_pending = paddle_vision_axis_of_pair(config.model.vision);

    // Embedded ERNIE text executor: the shared Phase 3 dense contract
    // (qk_norm=false, qkv_bias=true, rope_axes=1, rotate_half). No DeepStack.
    DenseExecutorConfig text_config;
    text_config.decoder = config.model.text;
    text_config.names = config.names.text;
    text_config.tie_word_embeddings = config.model.tie_word_embeddings;
    text_config.qk_norm = false;
    text_config.qkv_bias = true;
    text_config.rope_axes = 1u;
    text_config.component = config.decoder_component;
    text_config.prefill_chunk_rows = config.prefill_chunk_rows;
    text_config.memory = config.memory;
    if (auto s = _text.create(device, stream, text_config, weights); !s.ok()) { return s; }
    _weight_bytes = weights.component_bytes(config.vision_component) + _text.weight_bytes();
    return Status::success();
}

[[nodiscard]] Status PaddleOcrVlRecognizer::resolve_weights(const WeightStore &weights) {
    const auto &names = _config.names;
    const auto &comp = _config.vision_component;
    const auto &v = _config.model.vision;
    const auto full_expected = paddle_ocr_vl_expected_tensors(_config.model, names);
    luisa::unordered_map<luisa::string, const ExpectedTensor *> expected_by_name;
    for (const auto &e : full_expected) {
        if (e.name.compare(0, names.visual_prefix.size(), names.visual_prefix) == 0) {
            expected_by_name.emplace(e.name, &e);
        }
    }
    for (const auto &name : weights.tensor_names(comp)) {
        if (expected_by_name.find(name) == expected_by_name.end()) {
            return Status::error(StatusCode::invalid_argument,
                                 luisa::format("unexpected paddle vision tensor '{}' in component '{}'", name, comp), name);
        }
    }
    const auto view_of = [&](const luisa::string &name, BufferView<float> &out) -> Status {
        const auto *w = weights.find(comp, name);
        if (w == nullptr) {
            return Status::error(StatusCode::not_found,
                                 luisa::format("paddle vision weight '{}' not found in component '{}'", name, comp), name);
        }
        if (w->encoding != TensorEncoding::fp32 || !w->dense) {
            return Status::error(StatusCode::unsupported,
                                 luisa::format("paddle vision weight '{}' must be dense fp32 in the Phase 6 slice", name), name);
        }
        const auto it = expected_by_name.find(name);
        if (it == expected_by_name.end()) {
            return Status::error(StatusCode::invalid_argument,
                                 luisa::format("paddle vision weight '{}' has no expected-tensor contract", name), name);
        }
        if (auto s = validate_decoder_weight_desc(w->logical, w->element_count, *it->second); !s.ok()) { return s; }
        if (w->dense.size() < w->element_count) {
            return Status::error(StatusCode::invalid_argument,
                                 luisa::format("paddle vision weight '{}' buffer smaller than element count", name), name);
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
        if (auto s = view_of(names.block_fc1(l), B.fc1_w); !s.ok()) { return s; }
        if (auto s = view_of(names.block_fc1_bias(l), B.fc1_b); !s.ok()) { return s; }
        if (auto s = view_of(names.block_fc2(l), B.fc2_w); !s.ok()) { return s; }
        if (auto s = view_of(names.block_fc2_bias(l), B.fc2_b); !s.ok()) { return s; }
    }
    if (auto s = view_of(names.connector_ln_weight(), _connector.ln_w); !s.ok()) { return s; }
    if (auto s = view_of(names.connector_ln_bias(), _connector.ln_b); !s.ok()) { return s; }
    if (auto s = view_of(names.connector_fc1(), _connector.fc1_w); !s.ok()) { return s; }
    if (auto s = view_of(names.connector_fc1_bias(), _connector.fc1_b); !s.ok()) { return s; }
    if (auto s = view_of(names.connector_fc2(), _connector.fc2_w); !s.ok()) { return s; }
    if (auto s = view_of(names.connector_fc2_bias(), _connector.fc2_b); !s.ok()) { return s; }
    return Status::success();
}

[[nodiscard]] Status PaddleOcrVlRecognizer::reserve_workspace() {
    const auto &v = _config.model.vision;
    const uint64_t maxp = _config.max_vision_patches;
    const uint64_t vh = v.hidden_size, vi = v.intermediate_size;
    const uint64_t text_hidden = _config.model.text.hidden_size;
    const uint64_t max_tokens = _config.model.connector.max_visual_tokens;
    const uint64_t ci = _config.model.connector.connector_intermediate;
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
        {"qr", maxp * vh},
        {"kr", maxp * vh},
        {"attn", maxp * vh},
        {"fc1", maxp * vi},
        {"fc2", maxp * vh},
        {"pooled", max_tokens * vh},
        {"cnorm", max_tokens * vh},
        {"cfc1", max_tokens * ci},
        {"features", max_tokens * text_hidden},
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
    MemoryBudget budget{_config.memory};
    if (auto s = budget.require("paddle_vision_workspace", MemoryCategory::workspace, plan.total_bytes()); !s.ok()) { return s; }
    if (auto s = budget.preflight(); !s.ok()) { return s; }
    if (auto s = _vision_ws.reserve(plan.total_bytes()); !s.ok()) { return s; }
    WorkspaceAllocation *outs[] = {
        &_v_pixels, &_v_patches, &_v_x, &_v_norm, &_v_q, &_v_k, &_v_v, &_v_qr, &_v_kr,
        &_v_attn, &_v_fc1, &_v_fc2, &_v_pooled, &_v_cnorm, &_v_cfc1, &_v_features,
        &_v_positions, &_v_axis, &_v_dummy_f, &_v_dummy_u, &_v_aux,
    };
    for (size_t i = 0; i < std::size(specs); i++) {
        if (auto s = add_region(_vision_ws, *outs[i], specs[i].elements); !s.ok()) { return s; }
    }
    return Status::success();
}

[[nodiscard]] Status PaddleOcrVlRecognizer::compile_shaders() {
    const auto get = [](auto &cache, const luisa::string &key, auto factory, auto *&out) -> Status {
        auto r = cache.get_or_compile(key, factory);
        if (!r.ok()) { return r.status; }
        out = *r.value;
        return Status::success();
    };
    if (auto s = get(_patchify_cache, image_variant_key("patchify"), [] { return make_image_patchify_kernel(); }, _sh_patchify); !s.ok()) { return s; }
    if (auto s = get(_layer_norm_cache, norm_variant_key("layer_norm"), [] { return make_layer_norm_kernel(); }, _sh_layer_norm); !s.ok()) { return s; }
    if (auto s = get(_gemm_cache, gemm_variant_key(GemmEpilogue::none, GemmWeightFormat::fp32, false),
                     [] { return make_gemm_kernel(GemmEpilogue::none); }, _sh_gemm_none); !s.ok()) { return s; }
    if (auto s = get(_gemm_cache, gemm_variant_key(GemmEpilogue::bias, GemmWeightFormat::fp32, false),
                     [] { return make_gemm_kernel(GemmEpilogue::bias); }, _sh_gemm_bias); !s.ok()) { return s; }
    if (auto s = get(_gemm_cache, gemm_variant_key(GemmEpilogue::bias_gelu_erf, GemmWeightFormat::fp32, false),
                     [] { return make_gemm_kernel(GemmEpilogue::bias_gelu_erf); }, _sh_gemm_bias_gelu); !s.ok()) { return s; }
    if (auto s = get(_gemm_cache, gemm_variant_key(GemmEpilogue::bias_residual, GemmWeightFormat::fp32, false),
                     [] { return make_gemm_kernel(GemmEpilogue::bias_residual); }, _sh_gemm_bias_residual); !s.ok()) { return s; }
    if (auto s = get(_rope2d_cache, rope_variant_key("2d"), [] { return make_rope_2d_kernel(); }, _sh_rope2d); !s.ok()) { return s; }
    if (auto s = get(_attn_cache, attention_variant_key("prefill_reference"),
                     [] { return make_attention_prefill_reference_kernel(); }, _sh_attn); !s.ok()) { return s; }
    if (auto s = get(_pool_cache, conv_variant_key("adaptive_avg_pool2d"),
                     [] { return make_adaptive_avg_pool2d_kernel(); }, _sh_pool); !s.ok()) { return s; }
    if (auto s = get(_residual_cache, norm_variant_key("residual_add"), [] { return make_residual_add_kernel(); }, _sh_residual); !s.ok()) { return s; }
    if (auto s = get(_copy_cache, vision_variant_key("spatial_merge"), [] { return make_spatial_merge_kernel(); }, _sh_copy); !s.ok()) { return s; }
    return Status::success();
}

// ---------------------------------------------------------------------------
// PaddleOcrVlRecognizer: linear helper + vision block + connector
// ---------------------------------------------------------------------------

[[nodiscard]] Status PaddleOcrVlRecognizer::record_linear(CommandList &list, BufferView<float> A,
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
                         : epilogue == GemmEpilogue::bias_gelu_erf ? _sh_gemm_bias_gelu
                         : epilogue == GemmEpilogue::bias_residual ? _sh_gemm_bias_residual
                                                                   : _sh_gemm_none;
    if (auto cmd = dispatch_2d((*shader)(A, B, C, aux_view, m, n, k, plan.value->padded_cols), *plan.value)) {
        list << std::move(cmd);
    }
    return Status::success();
}

[[nodiscard]] Status PaddleOcrVlRecognizer::record_vision_block(CommandList &list, uint32_t layer,
                                                                BufferView<float> x,
                                                                BufferView<int32_t> positions, uint32_t rows) {
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
    auto qr = vws(_v_qr).subview(0, static_cast<size_t>(rows) * vh);
    auto kr = vws(_v_kr).subview(0, static_cast<size_t>(rows) * vh);
    auto attn = vws(_v_attn).subview(0, static_cast<size_t>(rows) * vh);
    auto fc1 = vws(_v_fc1).subview(0, static_cast<size_t>(rows) * vi);
    auto axis = vws(_v_axis).as<uint32_t>();
    auto dummy_u = vws(_v_dummy_u).as<uint32_t>();
    auto dummy_f = vws(_v_dummy_f);
    // 1. Pre-attention LayerNorm.
    {
        auto plan = plan_norm_dispatch(rows);
        if (!plan.ok()) { return plan.status; }
        if (auto cmd = dispatch_1d((*_sh_layer_norm)(x, norm, B.norm1_w, B.norm1_b, rows, vh, eps, 1u), *plan.value)) {
            list << std::move(cmd);
        }
    }
    // 2. QKV projections WITH bias (NaViT contract; no QK-Norm).
    if (auto s = record_linear(list, norm, B.q_w, q, B.q_b, rows, vh, vh, GemmEpilogue::bias); !s.ok()) { return s; }
    if (auto s = record_linear(list, norm, B.k_w, k, B.k_b, rows, vh, vh, GemmEpilogue::bias); !s.ok()) { return s; }
    if (auto s = record_linear(list, norm, B.v_w, vv, B.v_b, rows, vh, vh, GemmEpilogue::bias); !s.ok()) { return s; }
    // 3. 2D axial vision RoPE (adjacent pairs; height/width axis map).
    {
        auto plan_q = plan_rope_dispatch(rows, heads);
        if (!plan_q.ok()) { return plan_q.status; }
        if (auto cmd = dispatch_1d((*_sh_rope2d)(q, qr, positions, axis, rows, heads, hd, 2u, theta), *plan_q.value)) {
            list << std::move(cmd);
        }
        auto plan_k = plan_rope_dispatch(rows, heads);
        if (!plan_k.ok()) { return plan_k.status; }
        if (auto cmd = dispatch_1d((*_sh_rope2d)(k, kr, positions, axis, rows, heads, hd, 2u, theta), *plan_k.value)) {
            list << std::move(cmd);
        }
    }
    // 4. Bidirectional attention WITHIN this image (MHA: kv_heads == heads).
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
    // 5. Output projection + bias + residual: x = x + attn @ Wo + bo.
    {
        auto aux = vws(_v_aux).subview(0, static_cast<size_t>(vh) + static_cast<size_t>(rows) * vh);
        const auto copy_rows = [&](BufferView<float> src, BufferView<float> dst, uint32_t r, uint32_t h) -> Status {
            auto plan = plan_spatial_merge_dispatch(r, h);
            if (!plan.ok()) { return plan.status; }
            if (auto cmd = dispatch_1d((*_sh_copy)(src, dst, r, h, h, 1u), *plan.value)) {
                list << std::move(cmd);
            }
            return Status::success();
        };
        if (auto s = copy_rows(B.o_b, aux.subview(0, vh), 1u, vh); !s.ok()) { return s; }
        if (auto s = copy_rows(x, aux.subview(vh, static_cast<size_t>(rows) * vh), rows, vh); !s.ok()) { return s; }
        if (auto s = record_linear(list, attn, B.o_w, x, aux, rows, vh, vh, GemmEpilogue::bias_residual); !s.ok()) { return s; }
    }
    // 6. Post-attention LayerNorm.
    {
        auto plan = plan_norm_dispatch(rows);
        if (!plan.ok()) { return plan.status; }
        if (auto cmd = dispatch_1d((*_sh_layer_norm)(x, norm, B.norm2_w, B.norm2_b, rows, vh, eps, 1u), *plan.value)) {
            list << std::move(cmd);
        }
    }
    // 7. MLP: fc1 + bias + GELU-erf, fc2 + bias + residual.
    if (auto s = record_linear(list, norm, B.fc1_w, fc1, B.fc1_b, rows, vi, vh, GemmEpilogue::bias_gelu_erf); !s.ok()) { return s; }
    {
        auto aux = vws(_v_aux).subview(0, static_cast<size_t>(vh) + static_cast<size_t>(rows) * vh);
        const auto copy_rows = [&](BufferView<float> src, BufferView<float> dst, uint32_t r, uint32_t h) -> Status {
            auto plan = plan_spatial_merge_dispatch(r, h);
            if (!plan.ok()) { return plan.status; }
            if (auto cmd = dispatch_1d((*_sh_copy)(src, dst, r, h, h, 1u), *plan.value)) {
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

[[nodiscard]] Status PaddleOcrVlRecognizer::record_vision(CommandList &list, BufferView<float> pixels,
                                                          BufferView<int32_t> positions, uint32_t rows) {
    const auto &v = _config.model.vision;
    (void)pixels;// patchify is recorded by the caller (needs explicit H/W)
    // Patch embed over the patchified rows, then the vision blocks.
    auto patches = vws(_v_patches).subview(0, static_cast<size_t>(rows) * v.patch_dim());
    auto x = vws(_v_x).subview(0, static_cast<size_t>(rows) * v.hidden_size);
    if (auto s = record_linear(list, patches, _patch_embed_w, x, _patch_embed_b, rows, v.hidden_size, v.patch_dim(), GemmEpilogue::bias); !s.ok()) {
        return s;
    }
    for (uint32_t l = 0; l < v.num_layers; l++) {
        if (auto s = record_vision_block(list, l, x, positions, rows); !s.ok()) { return s; }
    }
    return Status::success();
}

[[nodiscard]] Status PaddleOcrVlRecognizer::record_connector(CommandList &list, BufferView<float> x,
                                                             uint32_t grid_h, uint32_t grid_w,
                                                             uint32_t out_h, uint32_t out_w,
                                                             BufferView<float> out) {
    const auto &v = _config.model.vision;
    const uint32_t vh = v.hidden_size;
    const uint32_t text_hidden = _config.model.text.hidden_size;
    const uint32_t ci = _config.model.connector.connector_intermediate;
    const float eps = _config.model.connector.layer_norm_eps;
    const uint32_t tokens = out_h * out_w;
    // 1. Adaptive average pool over the patch grid: x is [grid_h*grid_w, vh] in
    //    row-major spatial order == NCHW [1, vh, grid_h, grid_w]. Pool to
    //    [1, vh, out_h, out_w] == [tokens, vh].
    auto pooled = vws(_v_pooled).subview(0, static_cast<size_t>(tokens) * vh);
    {
        const uint64_t count = static_cast<uint64_t>(vh) * out_h * out_w;
        auto plan = plan_elementwise_dispatch(count);
        if (!plan.ok()) { return plan.status; }
        if (auto cmd = dispatch_1d((*_sh_pool)(x, pooled, 1u, vh, grid_h, grid_w, out_h, out_w), *plan.value)) {
            list << std::move(cmd);
        }
    }
    // 2. LayerNorm over vision hidden.
    auto cnorm = vws(_v_cnorm).subview(0, static_cast<size_t>(tokens) * vh);
    {
        auto plan = plan_norm_dispatch(tokens);
        if (!plan.ok()) { return plan.status; }
        if (auto cmd = dispatch_1d((*_sh_layer_norm)(pooled, cnorm, _connector.ln_w, _connector.ln_b, tokens, vh, eps, 1u), *plan.value)) {
            list << std::move(cmd);
        }
    }
    // 3. fc1 + bias + GELU-erf.
    auto cfc1 = vws(_v_cfc1).subview(0, static_cast<size_t>(tokens) * ci);
    if (auto s = record_linear(list, cnorm, _connector.fc1_w, cfc1, _connector.fc1_b, tokens, ci, vh, GemmEpilogue::bias_gelu_erf); !s.ok()) { return s; }
    // 4. fc2 + bias -> out [tokens, text_hidden].
    return record_linear(list, cfc1, _connector.fc2_w, out, _connector.fc2_b, tokens, text_hidden, ci, GemmEpilogue::bias);
}

// ---------------------------------------------------------------------------
// PaddleOcrVlRecognizer: recognize_batch (per-region vision -> connector -> ERNIE)
// ---------------------------------------------------------------------------

[[nodiscard]] Result<luisa::vector<RawRegionOutput>> PaddleOcrVlRecognizer::recognize_batch(
    Stream &stream, luisa::span<const RegionCropInput> crops, const RegionRecognizeOptions &options) {
    using R = Result<luisa::vector<RawRegionOutput>>;
    if (_device == nullptr) { return R::failure(StatusCode::invalid_argument, "paddle recognizer not created", "device"); }
    if (options.max_output_tokens == 0u) {
        return R::failure(StatusCode::invalid_argument, "max_output_tokens must be nonzero", "max_output_tokens");
    }
    const auto &model = _config.model;
    const auto &v = model.vision;
    const uint32_t text_hidden = model.text.hidden_size;
    Clock clock;
    const auto cancelled = [&]() { return options.cancel_callback && options.cancel_callback(); };
    const auto timed_out = [&]() { return options.max_wall_time_ms > 0.0 && clock.toc() > options.max_wall_time_ms; };

    luisa::vector<RawRegionOutput> outputs;
    outputs.reserve(crops.size());
    for (const auto &crop : crops) {
        if (cancelled()) { return R::failure(StatusCode::cancelled, "region recognition cancelled", "cancel"); }
        if (timed_out()) { return R::failure(StatusCode::timeout, "region recognition wall-time budget exceeded", "max_wall_time_ms"); }
        if (crop.source_width == 0u || crop.source_height == 0u ||
            crop.pixels.size() != static_cast<size_t>(crop.source_width) * crop.source_height * 3u) {
            return R::failure(StatusCode::invalid_argument, "degenerate region crop", "crop");
        }
        // Pack the crop to the native-resolution patch grid + reduced token grid.
        auto packed = paddle_pack_image(crop.source_width, crop.source_height, model.packing,
                                        model.connector.max_visual_tokens);
        if (!packed.ok()) { return R::failure(packed.status); }
        const uint32_t rows = static_cast<uint32_t>(packed.value->patch_count);
        const uint32_t tokens = static_cast<uint32_t>(packed.value->visual_tokens);
        if (rows > _config.max_vision_patches) {
            return R::failure(StatusCode::out_of_memory,
                              "region patch count exceeds the vision workspace capacity", "max_vision_patches");
        }
        // Host preprocessing (Phase 1 bilinear stretch + normalize) via the bridge.
        Qwen3VlImageInput bridge_input;
        bridge_input.pixels = crop.pixels;
        bridge_input.source_width = crop.source_width;
        bridge_input.source_height = crop.source_height;
        bridge_input.packed.resized_width = packed.value->resized_width;
        bridge_input.packed.resized_height = packed.value->resized_height;
        auto pre = qwen3_vl_preprocess_image_host(bridge_input, _config.mean, _config.std_dev);
        if (!pre.ok()) { return R::failure(pre.status); }
        if (pre.value->width != packed.value->resized_width || pre.value->height != packed.value->resized_height) {
            return R::failure(StatusCode::internal_error, "preprocessed size does not match the packing contract", "preprocess");
        }
        // Per-region heap-owned staging (pixels + positions + axis).
        auto staging = luisa::make_unique<VisionUploadStaging>();
        staging->pixels = std::move(pre.value->pixels);
        staging->positions = paddle_vision_positions(*packed.value);
        staging->axis = _vision_axis_pending;
        auto *stage = staging.get();

        _vision_ws.reset();
        auto pixels = vws(_v_pixels).subview(0, staging->pixels.size());
        auto positions = vws(_v_positions).as<int32_t>().subview(0, static_cast<size_t>(rows) * 2u);
        auto patches = vws(_v_patches).subview(0, static_cast<size_t>(rows) * v.patch_dim());
        auto x = vws(_v_x).subview(0, static_cast<size_t>(rows) * v.hidden_size);
        auto features = vws(_v_features).subview(0, static_cast<size_t>(tokens) * text_hidden);
        {
            auto list = CommandList::create();
            list << pixels.copy_from(luisa::span<const float>{stage->pixels})
                 << positions.copy_from(luisa::span<const int32_t>{stage->positions})
                 << vws(_v_axis).as<uint32_t>().copy_from(luisa::span<const uint32_t>{stage->axis});
            // Patchify CHW -> [rows, patch_dim].
            {
                auto plan = plan_image_patchify_dispatch(rows, v.patch_dim());
                if (!plan.ok()) { return R::failure(plan.status); }
                if (auto cmd = dispatch_1d((*_sh_patchify)(pixels, patches, v.channels, packed.value->resized_height,
                                                           packed.value->resized_width, v.patch_size), *plan.value)) {
                    list << std::move(cmd);
                }
            }
            // Vision blocks (patch embed + N blocks).
            if (auto s = record_vision(list, pixels, positions, rows); !s.ok()) { return R::failure(s); }
            // Adaptive reduction + MLP connector -> [tokens, text_hidden].
            if (auto s = record_connector(list, x, packed.value->grid_h, packed.value->grid_w,
                                          packed.value->reduced_h, packed.value->reduced_w, features); !s.ok()) {
                return R::failure(s);
            }
            list.add_callback([staging = std::move(staging)]() mutable { staging.reset(); });
            stream << list.commit() << synchronize();
        }
        _last_patches = rows;
        _last_tokens = tokens;

        // Build the multimodal prompt: prefix + visual placeholders + suffix.
        DecodeRequest request;
        request.prompt_tokens = _config.prompt_prefix;
        const uint32_t mm_start = static_cast<uint32_t>(request.prompt_tokens.size());
        for (uint32_t i = 0; i < tokens; i++) { request.prompt_tokens.push_back(0u); }// placeholder id
        request.prompt_tokens.insert(request.prompt_tokens.end(), _config.prompt_suffix.begin(), _config.prompt_suffix.end());
        request.mm_features = features;
        request.mm_start = mm_start;
        request.mm_count = tokens;

        // ERNIE decode via the shared controller (greedy, bounded, immediate
        // sequence release). A fresh KV cache per region: no prefix sharing.
        ContiguousKvCache cache;
        KvCacheConfig kc;
        kc.num_layers = model.text.num_layers;
        kc.num_kv_heads = model.text.num_kv_heads;
        kc.head_dim = model.text.head_dim;
        kc.max_seq_len = static_cast<uint32_t>(request.prompt_tokens.size()) + options.max_output_tokens;
        kc.batch = 1;
        kc.dtype_bytes = 4;
        if (kc.max_seq_len > model.text.max_position_embeddings) {
            return R::failure(StatusCode::out_of_memory,
                              "region prompt + output exceeds the decoder context ceiling", "max_position_embeddings");
        }
        // Preflight the per-region KV allocation against the caller's device
        // memory limit, accounting for the resident weights + fixed workspace.
        // The cache is released at the end of this region (fresh per region), so
        // the budget records it transiently and drops it before the next region.
        MemoryBudget kv_budget{_config.memory};
        if (auto s = kv_budget.set_requirement("paddle_ocr_vl", MemoryCategory::weights, _weight_bytes); !s.ok()) { return R::failure(s); }
        if (auto s = kv_budget.set_requirement("paddle_ocr_vl", MemoryCategory::workspace, workspace_bytes()); !s.ok()) { return R::failure(s); }
        if (auto s = cache.create(*_device, stream, kc, &kv_budget); !s.ok()) { return R::failure(s); }
        _kv_peak_bytes = std::max(_kv_peak_bytes, cache.device_bytes());
        DenseDecodeOptions dopts;
        dopts.sampling.greedy = true;
        dopts.max_new_tokens = options.max_output_tokens;
        dopts.max_wall_time_ms = options.max_wall_time_ms;
        dopts.enable_cuda_graph = options.enable_cuda_graph;
        dopts.cancel_callback = options.cancel_callback;
        DenseDecodeController controller;
        if (auto s = controller.create(_text, nullptr, dopts); !s.ok()) { return R::failure(s); }
        auto gen = controller.generate(stream, request, cache);
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
        // Generative content: NO confidence (never a pseudo-probability).
        out.has_confidence = false;
        out.confidence = 0.0f;
        outputs.push_back(std::move(out));
    }
    return R::success(std::move(outputs));
}

[[nodiscard]] Status PaddleOcrVlRecognizer::download_vision_patches(Stream &stream, luisa::vector<float> &out) const {
    const uint64_t count = static_cast<uint64_t>(_last_patches) * _config.model.vision.patch_dim();
    out.resize(static_cast<size_t>(count));
    if (count != 0u) {
        stream << vws(_v_patches).subview(0, static_cast<size_t>(count)).copy_to(luisa::span<float>{out});
    }
    return Status::success();
}

[[nodiscard]] Status PaddleOcrVlRecognizer::download_connector_features(Stream &stream, luisa::vector<float> &out) const {
    const uint64_t count = static_cast<uint64_t>(_last_tokens) * _config.model.text.hidden_size;
    out.resize(static_cast<size_t>(count));
    if (count != 0u) {
        stream << vws(_v_features).subview(0, static_cast<size_t>(count)).copy_to(luisa::span<float>{out});
    }
    return Status::success();
}

// ---------------------------------------------------------------------------
// OpenDocRecognizerConfig
// ---------------------------------------------------------------------------

[[nodiscard]] Status OpenDocRecognizerConfig::validate() const {
    if (auto s = model.validate(); !s.ok()) { return s; }
    if (recognizer_component.empty()) {
        return Status::error(StatusCode::invalid_argument, "recognizer component tag must not be empty", "component");
    }
    if (mean.size() != 3u || std_dev.size() != 3u) {
        return Status::error(StatusCode::invalid_argument, "mean/std_dev must have 3 channels", "mean/std_dev");
    }
    for (auto v : std_dev) {
        if (!std::isfinite(v) || v == 0.0f) {
            return Status::error(StatusCode::invalid_argument, "std_dev must be finite and nonzero", "std_dev");
        }
    }
    if (auto s = ctc.validate(); !s.ok()) { return s; }
    return Status::success();
}

// ---------------------------------------------------------------------------
// OpenDocRecognizer: create (UniRec CNN+CTC via the shared CnnExecutor)
// ---------------------------------------------------------------------------

[[nodiscard]] Status OpenDocRecognizer::create(Device &device, Stream &stream,
                                               const OpenDocRecognizerConfig &config, const WeightStore &weights) {
    auto valid = config.validate();
    if (!valid.ok()) { return valid; }
    _config = config;
    _device = &device;
    // Build a CnnExecutor config whose recognizer is the UniRec geometry. The
    // detector part is a minimal VALID placeholder (never run: the executor is
    // created in recognizer-only mode and only recognize_line() is used). The
    // layout stage is owned by the orchestrator's PpDocLayoutExecutor, not here.
    CnnExecutorConfig cnn;
    cnn.model.size = PpOcrSize::tiny;// identity tag only; recognizer geometry is UniRec
    cnn.model.recognizer = config.model.recognizer.recognizer;
    // Minimal valid detector placeholder (1-stage backbone + neck + DB head).
    PpConvStage d0;
    d0.in_c = 3;
    d0.out_c = 4;
    d0.kh = d0.kw = 3;
    d0.stride_h = d0.stride_w = 2;
    d0.pad_h = d0.pad_w = 1;
    d0.act = 1;
    cnn.model.detector.in_channels = 3;
    cnn.model.detector.input_width = 16;
    cnn.model.detector.input_height = 16;
    cnn.model.detector.backbone = {d0};
    cnn.model.detector.lateral_stage = 0;
    cnn.model.detector.lateral_conv.in_c = 4;
    cnn.model.detector.lateral_conv.out_c = 4;
    cnn.model.detector.lateral_conv.kh = cnn.model.detector.lateral_conv.kw = 1;
    cnn.model.detector.lateral_conv.stride_h = cnn.model.detector.lateral_conv.stride_w = 1;
    cnn.model.detector.replk_conv.in_c = 8;
    cnn.model.detector.replk_conv.out_c = 4;
    cnn.model.detector.replk_conv.kh = cnn.model.detector.replk_conv.kw = 3;
    cnn.model.detector.replk_conv.stride_h = cnn.model.detector.replk_conv.stride_w = 1;
    cnn.model.detector.replk_conv.pad_h = cnn.model.detector.replk_conv.pad_w = 1;
    cnn.model.detector.replk_conv.act = 1;
    cnn.model.detector.db_conv.in_c = 4;
    cnn.model.detector.db_conv.out_c = 1;
    cnn.model.detector.db_conv.kh = cnn.model.detector.db_conv.kw = 1;
    cnn.model.detector.db_conv.stride_h = cnn.model.detector.db_conv.stride_w = 1;
    // Tensor names: the UniRec recognizer has a DISTINCT tensor contract
    // (UniRecTensorNames, "unirec." prefix). Install that name map into the
    // CnnExecutor so the load path resolves/validates the UniRec names -- NOT a
    // silent PP-OCRv6 alias. The on-disk/device inventory is uni_rec_*; the
    // fixture/package writes tensors under exactly these names.
    cnn.names = PpOcrv6TensorNames{};// detector placeholder names (never resolved)
    const auto &un = config.model.recognizer_names;
    cnn.recognizer_names.backbone_w = [un](uint32_t i) { return un.backbone_w(i); };
    cnn.recognizer_names.backbone_b = [un](uint32_t i) { return un.backbone_b(i); };
    cnn.recognizer_names.svtr_fc1_w = [un] { return un.svtr_fc1_w(); };
    cnn.recognizer_names.svtr_fc1_b = [un] { return un.svtr_fc1_b(); };
    cnn.recognizer_names.svtr_fc2_w = [un] { return un.svtr_fc2_w(); };
    cnn.recognizer_names.svtr_fc2_b = [un] { return un.svtr_fc2_b(); };
    cnn.recognizer_names.ctc_w = [un] { return un.ctc_w(); };
    cnn.recognizer_names.ctc_b = [un] { return un.ctc_b(); };
    cnn.detector_component = "layout_placeholder";// never resolved in recognizer-only mode
    cnn.recognizer_component = config.recognizer_component;
    cnn.mean = config.mean;
    cnn.std_dev = config.std_dev;
    cnn.ctc = config.ctc;
    cnn.dictionary = config.model.recognizer.dictionary;
    cnn.memory = config.memory;
    return _cnn.create_recognizer_only(device, stream, cnn, weights);
}

[[nodiscard]] Result<luisa::vector<RawRegionOutput>> OpenDocRecognizer::recognize_batch(
    Stream &stream, luisa::span<const RegionCropInput> crops, const RegionRecognizeOptions &options) {
    using R = Result<luisa::vector<RawRegionOutput>>;
    if (_device == nullptr) { return R::failure(StatusCode::invalid_argument, "open_doc recognizer not created", "device"); }
    (void)options;// UniRec is CTC (no autoregressive output budget); cancellation
                  // is polled by the orchestrator between regions.
    const auto &rec = _config.model.recognizer.recognizer;
    luisa::vector<RawRegionOutput> outputs;
    outputs.reserve(crops.size());
    for (const auto &crop : crops) {
        if (options.cancel_callback && options.cancel_callback()) {
            return R::failure(StatusCode::cancelled, "open_doc region recognition cancelled", "cancel");
        }
        if (crop.source_width == 0u || crop.source_height == 0u ||
            crop.pixels.size() != static_cast<size_t>(crop.source_width) * crop.source_height * 3u) {
            return R::failure(StatusCode::invalid_argument, "degenerate region crop", "crop");
        }
        // Resize the crop to the recognizer input (stretch + normalize) via the
        // Phase 1 bridge, then run the UniRec CNN+CTC graph.
        Qwen3VlImageInput bridge_input;
        bridge_input.pixels = crop.pixels;
        bridge_input.source_width = crop.source_width;
        bridge_input.source_height = crop.source_height;
        bridge_input.packed.resized_width = rec.input_width;
        bridge_input.packed.resized_height = rec.input_height;
        auto pre = qwen3_vl_preprocess_image_host(bridge_input, _config.mean, _config.std_dev);
        if (!pre.ok()) { return R::failure(pre.status); }
        auto decoded = _cnn.recognize_line(stream, pre.value->pixels);
        if (!decoded.ok()) { return R::failure(decoded.status); }
        RawRegionOutput out;
        out.region_id = crop.task.region_id;
        out.text = std::move(decoded.value->text);
        // UniRec CTC: the kept class ids are the "tokens" (one per emitted char).
        out.tokens = decoded.value->ids;
        out.truncated = false;
        out.hit_eos = false;
        out.native_patches = 0;// CNN path: no vision patches
        out.output_tokens = static_cast<uint32_t>(decoded.value->ids.size());
        // CTC confidence is an explicitly validated detector/CTC score source;
        // labeled internal/test by the host bridge. Empty decode -> no confidence.
        out.has_confidence = decoded.value->has_confidence;
        out.confidence = decoded.value->confidence;
        outputs.push_back(std::move(out));
    }
    return R::success(std::move(outputs));
}

// ---------------------------------------------------------------------------
// LayoutVlmExecutorConfig
// ---------------------------------------------------------------------------

[[nodiscard]] Status LayoutVlmExecutorConfig::validate() const {
    if (auto s = decode.validate(); !s.ok()) { return s; }
    if (auto s = scheduler.validate(); !s.ok()) { return s; }
    if (max_overview_pixels == 0u) {
        return Status::error(StatusCode::invalid_argument, "max_overview_pixels must be nonzero", "max_overview_pixels");
    }
    if (max_crop_pixels == 0u) {
        return Status::error(StatusCode::invalid_argument, "max_crop_pixels must be nonzero", "max_crop_pixels");
    }
    return Status::success();
}

// ---------------------------------------------------------------------------
// LayoutVlmExecutor: create
// ---------------------------------------------------------------------------

[[nodiscard]] Status LayoutVlmExecutor::create(const LayoutVlmExecutorConfig &config,
                                               PpDocLayoutExecutor *layout,
                                               const PpDocLayoutConfig &layout_config,
                                               RegionRecognizer *recognizer,
                                               LayoutHeadMapProvider cpu_head_maps) {
    auto valid = config.validate();
    if (!valid.ok()) { return valid; }
    if (auto s = layout_config.validate(); !s.ok()) { return s; }
    if (recognizer == nullptr) {
        return Status::error(StatusCode::invalid_argument, "region recognizer must not be null", "recognizer");
    }
    if (config.layout_policy == LayoutDevicePolicy::device) {
        if (layout == nullptr) {
            return Status::error(StatusCode::invalid_argument,
                                 "device layout policy requires a created layout executor", "layout");
        }
    } else {
        if (!cpu_head_maps) {
            return Status::error(StatusCode::invalid_argument,
                                 "cpu_reference layout policy requires a head-map provider", "cpu_head_maps");
        }
    }
    _config = config;
    _layout = layout;
    _layout_config = layout_config;
    _recognizer = recognizer;
    _cpu_head_maps = std::move(cpu_head_maps);
    _scheduler = RegionScheduler{config.scheduler};
    _cache = LayoutCache{config.layout_cache_entries == 0u ? 1u : config.layout_cache_entries};
    return Status::success();
}

// ---------------------------------------------------------------------------
// LayoutVlmExecutor: layout pass (device or cpu_reference) + cache
// ---------------------------------------------------------------------------

[[nodiscard]] Result<luisa::vector<RegionTask>> LayoutVlmExecutor::detect_regions(Stream &stream,
                                                                                  const LayoutPageInput &page,
                                                                                  uint32_t region_id_base,
                                                                                  uint64_t &cache_hits) {
    using R = Result<luisa::vector<RegionTask>>;
    // Explicit content/geometry cache key. A hit returns the cached regions
    // WITHOUT re-running layout (recognition retry).
    LayoutCacheKey key;
    key.content_hash = layout_hash_bytes(page.pixels);
    key.geometry_hash = layout_hash_geometry(page.source_width, page.source_height,
                                             _layout_config.input_width, _layout_config.input_height, page.page_index);
    if (_config.layout_cache_entries != 0u) {
        if (const auto *cached = _cache.find(key)) {
            cache_hits++;
            return R::success(cached->regions);
        }
    }
    // Run the layout detector (device graph or the cpu_reference head maps).
    LayoutHeadMaps maps;
    if (_config.layout_policy == LayoutDevicePolicy::device) {
        auto detected = _layout->detect(stream, page);
        if (!detected.ok()) { return R::failure(detected.status); }
        maps = std::move(*detected.value);
    } else {
        auto provided = _cpu_head_maps(page);
        if (!provided.ok()) { return R::failure(provided.status); }
        maps = std::move(*provided.value);
    }
    if (maps.num_classes != _layout_config.num_classes()) {
        return R::failure(StatusCode::invalid_argument, "head-map class count does not match the layout config", "num_classes");
    }
    // Stride: the backbone total stride (map -> DETECTOR INPUT). The decode
    // multiplies the decoded box by this stride exactly once, producing boxes in
    // detector-input pixels.
    auto mh = _layout_config.map_height();
    auto mw = _layout_config.map_width();
    if (!mh.ok() || !mw.ok()) { return R::failure(mh.status); }
    if (*mh.value == 0u || *mw.value == 0u ||
        _layout_config.input_width == 0u || _layout_config.input_height == 0u) {
        return R::failure(StatusCode::invalid_argument, "degenerate layout detector geometry", "input/map");
    }
    const float stride_h = static_cast<float>(_layout_config.input_height) / static_cast<float>(*mh.value);
    const float stride_w = static_cast<float>(_layout_config.input_width) / static_cast<float>(*mw.value);
    // Detector-input -> source transform: the page was stretch-resized to the
    // detector input. The decoded boxes are ALREADY in detector-input pixels
    // (stride applied by the decode), so the scale is source/input -- NOT
    // source/map, which would double-count the stride.
    output::DbSourceTransform transform;
    transform.scale_x = static_cast<float>(page.source_width) / static_cast<float>(_layout_config.input_width);
    transform.scale_y = static_cast<float>(page.source_height) / static_cast<float>(_layout_config.input_height);
    // The decode uses a single isotropic stride; the synthetic backbone has
    // equal h/w strides. Refuse anisotropic strides rather than silently skew.
    if (std::abs(stride_h - stride_w) > 1e-3f) {
        return R::failure(StatusCode::unsupported, "anisotropic layout stride is not supported by the decode", "stride");
    }
    auto regions = pp_doc_layout_decode(maps.class_probs, maps.box_deltas, maps.map_width, maps.map_height,
                                        maps.num_classes, stride_w, transform, _layout_config, _config.decode);
    if (!regions.ok()) { return R::failure(regions.status); }
    // Map decoded regions to RegionTasks (document-global region_id, approx
    // costs). region_id_base makes ids unique across pages in input order.
    luisa::vector<RegionTask> tasks;
    tasks.reserve(regions.value->size());
    uint32_t next_id = region_id_base;
    for (const auto &region : *regions.value) {
        RegionTask task;
        task.region_id = next_id++;
        task.page_index = page.page_index;
        task.reading_order = region.reading_order;
        task.block_type = region.block_type;
        task.source_box = region.bounds;
        task.polygon = region.polygon;
        // Approximate patch/token cost from the crop geometry (host estimate).
        const float w = std::max(0.0f, region.bounds.x1 - region.bounds.x0);
        const float h = std::max(0.0f, region.bounds.y1 - region.bounds.y0);
        const uint64_t crop_pixels = static_cast<uint64_t>(w) * static_cast<uint64_t>(h);
        // Patch estimate at a nominal 16x16 patch (scheduler cost model only).
        task.approx_patches = static_cast<uint32_t>(std::min<uint64_t>(crop_pixels / 256u + 1u, 1u << 20));
        task.approx_tokens = task.approx_patches;
        tasks.push_back(std::move(task));
    }
    if (_config.layout_cache_entries != 0u) {
        _cache.store(key, CachedLayout{tasks});
    }
    return R::success(std::move(tasks));
}

// ---------------------------------------------------------------------------
// LayoutVlmExecutor: crop extraction (source coordinates preserved)
// ---------------------------------------------------------------------------

[[nodiscard]] Result<RegionCropInput> LayoutVlmExecutor::make_crop(const LayoutPageInput &page,
                                                                   const RegionTask &task) const {
    using R = Result<RegionCropInput>;
    // Axis-aligned crop of the source box, clamped to the page. The crop keeps
    // the ORIGINAL source pixels (no resize here; the recognizer packs/resizes).
    const int32_t sw = static_cast<int32_t>(page.source_width);
    const int32_t sh = static_cast<int32_t>(page.source_height);
    const int32_t x0 = std::clamp(static_cast<int32_t>(std::floor(task.source_box.x0)), 0, sw);
    const int32_t y0 = std::clamp(static_cast<int32_t>(std::floor(task.source_box.y0)), 0, sh);
    const int32_t x1 = std::clamp(static_cast<int32_t>(std::ceil(task.source_box.x1)), 0, sw);
    const int32_t y1 = std::clamp(static_cast<int32_t>(std::ceil(task.source_box.y1)), 0, sh);
    if (x1 <= x0 || y1 <= y0) {
        return R::failure(StatusCode::invalid_argument, "degenerate region crop box", "source_box");
    }
    const uint32_t cw = static_cast<uint32_t>(x1 - x0);
    const uint32_t ch = static_cast<uint32_t>(y1 - y0);
    const uint64_t crop_pixels = static_cast<uint64_t>(cw) * ch;
    if (crop_pixels > _config.max_crop_pixels) {
        return R::failure(StatusCode::out_of_memory, "region crop exceeds max_crop_pixels", "max_crop_pixels");
    }
    RegionCropInput crop;
    crop.source_width = cw;
    crop.source_height = ch;
    crop.task = task;
    crop.pixels.resize(static_cast<size_t>(cw) * ch * 3u);
    for (uint32_t y = 0; y < ch; y++) {
        const uint8_t *src_row = page.pixels.data() + (static_cast<size_t>(y0 + y) * page.source_width + x0) * 3u;
        uint8_t *dst_row = crop.pixels.data() + static_cast<size_t>(y) * cw * 3u;
        std::memcpy(dst_row, src_row, static_cast<size_t>(cw) * 3u);
    }
    return R::success(std::move(crop));
}

// ---------------------------------------------------------------------------
// LayoutVlmExecutor: full orchestration
// ---------------------------------------------------------------------------

[[nodiscard]] Result<LayoutRunResult> LayoutVlmExecutor::recognize(Stream &stream,
                                                                   luisa::span<const LayoutPageInput> pages) {
    using R = Result<LayoutRunResult>;
    if (_recognizer == nullptr) { return R::failure(StatusCode::invalid_argument, "orchestrator not created", "recognizer"); }
    Clock clock;
    const auto &limits = _config.scheduler;
    const auto cancelled = [&]() { return limits.cancel_callback && limits.cancel_callback(); };
    const auto timed_out = [&]() { return limits.max_wall_time_ms > 0.0 && clock.toc() > limits.max_wall_time_ms; };

    LayoutRunResult run;
    run.weight_bytes = _recognizer->weight_bytes() + (_layout != nullptr ? _layout->weight_bytes() : 0u);
    if (_config.layout_policy == LayoutDevicePolicy::cpu_reference) {
        run.diagnostics.push_back(Diagnostic{DiagnosticSeverity::info, "layout_cpu_reference",
                                             "layout ran through the documented CPU reference head-map path"});
    }
    // Document-global region_id offset: the count of regions detected on all
    // previously processed pages (input order), so ids are unique across pages.
    uint32_t region_id_base = 0u;
    for (const auto &page : pages) {
        if (cancelled()) { return R::failure(StatusCode::cancelled, "request cancelled before a page", "cancel"); }
        if (timed_out()) { return R::failure(StatusCode::timeout, "wall-time budget exceeded before a page", "max_wall_time_ms"); }
        if (page.source_width == 0u || page.source_height == 0u ||
            page.pixels.size() != static_cast<size_t>(page.source_width) * page.source_height * 3u) {
            return R::failure(StatusCode::invalid_argument, "degenerate page input", "page");
        }
        const uint64_t overview_pixels = static_cast<uint64_t>(page.source_width) * page.source_height;
        if (overview_pixels > _config.max_overview_pixels) {
            return R::failure(StatusCode::out_of_memory, "page exceeds max_overview_pixels", "max_overview_pixels");
        }
        // ---- Stage 1: ONE overview/layout pass -> regions (source coords) ----
        uint64_t cache_hits = 0;
        auto tasks = detect_regions(stream, page, region_id_base, cache_hits);
        if (!tasks.ok()) { return R::failure(tasks.status); }
        run.layout_cache_hits += cache_hits;
        region_id_base += static_cast<uint32_t>(tasks.value->size());

        LayoutPageResult page_result;
        page_result.page_index = page.page_index;
        page_result.source_width = page.source_width;
        page_result.source_height = page.source_height;

        if (tasks.value->empty()) {
            // Empty page: valid empty result, no hallucinated regions.
            run.pages.push_back(std::move(page_result));
            continue;
        }
        if (auto s = _scheduler.preflight_page_output_budget(static_cast<uint32_t>(tasks.value->size())); !s.ok()) {
            return R::failure(s);
        }
        // ---- Stage 2: crops (source pixels preserved) ----
        // A region whose source box does not intersect the page (a degenerate or
        // out-of-page detection) is skipped with a visible diagnostic, never a
        // silent drop or a whole-page failure. Skipped regions keep their task
        // index so batch indices stay aligned; they are marked filled with empty
        // content and excluded from the composed result below.
        luisa::vector<RegionCropInput> crops;
        crops.reserve(tasks.value->size());
        luisa::vector<uint8_t> skipped(tasks.value->size(), 0u);
        for (size_t i = 0; i < tasks.value->size(); i++) {
            const auto &task = (*tasks.value)[i];
            auto crop = make_crop(page, task);
            if (!crop.ok()) {
                skipped[i] = 1u;
                crops.emplace_back();// placeholder; never recognized
                run.diagnostics.push_back(Diagnostic{
                    DiagnosticSeverity::warning, "region_crop_skipped",
                    luisa::format("region {} on page {} has a degenerate/out-of-page source box and was skipped",
                                  task.region_id, task.page_index)});
                continue;
            }
            crops.push_back(std::move(*crop.value));
        }
        // ---- Stage 3: token-budgeted vision/prefill batches ----
        // plan_vision_batches packs crops under approximate patch/region caps.
        // Phase 11 P7 recognizers execute ragged multi-crop vision graphs (and
        // split by exact packed capacity); Paddle/OpenDoc still loop per crop.
        // Text decode always uses a fresh batch-1 KV cache per region, and
        // max_decode_batch remains planning-only.
        auto batches = _scheduler.plan_vision_batches(*tasks.value);
        if (!batches.ok()) { return R::failure(batches.status); }
        // Recognize each batch independently; collect raw outputs by task index.
        luisa::vector<RawRegionOutput> raw_by_id(tasks.value->size());
        luisa::vector<uint8_t> filled(tasks.value->size(), 0u);
        // Per-page output-token budget, enforced DURING accumulation. Once the
        // page's accumulated output tokens reach the cap, later regions are not
        // recognized and a visible output_truncated diagnostic is emitted.
        uint64_t page_output_tokens = 0;
        bool page_budget_hit = false;
        const uint64_t page_budget = limits.max_output_tokens_per_page;
        for (const auto &batch : *batches.value) {
            if (cancelled()) { return R::failure(StatusCode::cancelled, "request cancelled before a region batch", "cancel"); }
            if (timed_out()) { return R::failure(StatusCode::timeout, "wall-time budget exceeded before a region batch", "max_wall_time_ms"); }
            luisa::vector<RegionCropInput> batch_crops;
            luisa::vector<uint32_t> batch_indices;
            batch_crops.reserve(batch.region_indices.size());
            for (auto idx : batch.region_indices) {
                if (skipped[idx]) { continue; }// degenerate region: not recognized
                if (page_budget_hit) { continue; }// page output budget exhausted
                // Stop once the page output budget is exhausted; the per-region
                // cap below is clamped to the remaining budget so accumulation
                // cannot overshoot it.
                if (page_output_tokens >= page_budget) { page_budget_hit = true; continue; }
                batch_crops.push_back(crops[idx]);
                batch_indices.push_back(idx);
            }
            if (batch_crops.empty()) { continue; }
            RegionRecognizeOptions ropts;
            // Single source of truth for the per-region output cap: the scheduler
            // limit (forwarded to the recognizer). Clamp to the remaining page
            // budget so accumulation cannot overshoot it.
            const uint64_t remaining = page_budget - page_output_tokens;
            ropts.max_output_tokens = static_cast<uint32_t>(std::min<uint64_t>(
                limits.max_output_tokens_per_region, remaining));
        ropts.max_wall_time_ms = limits.max_wall_time_ms;
        ropts.enable_cuda_graph = limits.enable_cuda_graph;
        ropts.cancel_callback = limits.cancel_callback;

            auto raw = _recognizer->recognize_batch(stream, batch_crops, ropts);
            if (!raw.ok()) { return R::failure(raw.status); }
            if (raw.value->size() != batch_crops.size()) {
                return R::failure(StatusCode::internal_error, "recognizer returned a wrong output count", "recognize_batch");
            }
            for (size_t i = 0; i < raw.value->size(); i++) {
                const uint32_t idx = batch_indices[i];
                raw_by_id[idx] = std::move((*raw.value)[i]);
                filled[idx] = 1u;
                run.regions_processed++;
                run.patches_processed += raw_by_id[idx].native_patches;
                run.output_tokens += raw_by_id[idx].output_tokens;
                page_output_tokens += raw_by_id[idx].output_tokens;
                if (page_output_tokens >= page_budget) { page_budget_hit = true; }
            }
        }
        if (page_budget_hit) {
            run.diagnostics.push_back(Diagnostic{
                DiagnosticSeverity::warning, "output_truncated",
                luisa::format("page {} reached max_output_tokens_per_page ({}); later regions were not recognized",
                              page.page_index, page_budget)});
        }
        // ---- Stage 4: restore original page/reading order ----
        auto order = RegionScheduler::restore_order(*tasks.value);
        if (!order.ok()) { return R::failure(order.status); }
        for (auto idx : *order.value) {
            if (skipped[idx]) { continue; }// degenerate region: excluded from the result
            if (!filled[idx]) {
                // A region left unfilled because the page output budget was
                // exhausted is a deliberate, diagnosed truncation -- not an error.
                if (page_budget_hit) { continue; }
                return R::failure(StatusCode::internal_error, "a region was not recognized", "restore_order");
            }
            const auto &task = (*tasks.value)[idx];
            const auto &raw = raw_by_id[idx];
            LayoutRegionResult region;
            region.region_id = task.region_id;
            region.page_index = task.page_index;
            region.reading_order = task.reading_order;
            region.block_type = task.block_type;
            region.content = raw.text;// raw content; host compose adapts per type
            region.source_box = task.source_box;
            region.polygon = task.polygon;
            region.has_confidence = raw.has_confidence;
            region.confidence = raw.confidence;
            page_result.regions.push_back(std::move(region));
        }
        run.pages.push_back(std::move(page_result));
    }
    // Sort returned pages by source page_index (stable) before composition, so a
    // shuffled input page span still yields document-ordered pages.
    {
        const auto order = layout_sort_pages_by_index(run.pages);
        luisa::vector<LayoutPageResult> sorted;
        sorted.reserve(run.pages.size());
        for (auto idx : order) { sorted.push_back(std::move(run.pages[idx])); }
        run.pages = std::move(sorted);
    }
    run.workspace_bytes = _recognizer->workspace_bytes() + (_layout != nullptr ? _layout->workspace_bytes() : 0u);
    // Honest peak/resident device accounting: weights + fixed workspace + the
    // peak per-region KV cache actually allocated during the run.
    run.kv_cache_peak_bytes = _recognizer->kv_cache_peak_bytes();
    run.peak_device_bytes = run.weight_bytes + run.workspace_bytes + run.kv_cache_peak_bytes;
    return R::success(std::move(run));
}

}// namespace luisa::ocr
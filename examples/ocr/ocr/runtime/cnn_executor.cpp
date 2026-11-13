// OCR engine -- conventional CNN OCR executor implementation (Phase 5).
// Real detector -> crop -> recognizer -> CTC pipeline. See the header.
#include "ocr/runtime/cnn_executor.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <utility>

#include <luisa/core/logging.h>
#include <luisa/core/stl/format.h>
#include <luisa/runtime/stream.h>

#include "ocr/runtime/dispatch_adapter.h"

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

// Host STRETCH-resize + normalize of an HWC uint8 page to CHW float at the
// detector input size (independent x/y scales, half-pixel centers, clamp
// edges; NO aspect preservation, NO padding). This is the declared SYNTHETIC
// convention -- it uses the same bilinear sampling math as preprocess_image's
// bilinear arm, but it is NOT official PP-OCRv6 preprocessing parity.
[[nodiscard]] luisa::vector<float> preprocess_page_host(const CnnPageInput &page, uint32_t dw, uint32_t dh,
                                                        luisa::span<const float> mean, luisa::span<const float> std_dev) {
    const uint32_t sw = page.source_width, sh = page.source_height;
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
            return static_cast<float>(page.pixels[(static_cast<size_t>(cy) * sw + cx) * 3u + c]);
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
                const float normalized = (v - mean[c]) / std_dev[c];
                out[static_cast<size_t>(c) * dw * dh + static_cast<size_t>(y) * dw + x] = normalized;
            }
        }
    }
    return out;
}

// Compute the rotated-rectangle crop parameters (center, size, angle) from a
// DB quad polygon (source coordinates). The quad is ordered TL,TR,BR,BL.
struct CropRect {
    float cx{0}, cy{0}, rw{0}, rh{0}, angle{0};
};
[[nodiscard]] CropRect crop_rect_from_quad(luisa::span<const output::DbPoint> quad) noexcept {
    CropRect r;
    if (quad.size() < 4u) { return r; }
    // Center = mean of corners.
    for (const auto &p : quad) {
        r.cx += p.x;
        r.cy += p.y;
    }
    r.cx *= 0.25f;
    r.cy *= 0.25f;
    // Width = mean of top and bottom edge lengths; height = mean of left/right.
    const auto dist = [](const output::DbPoint &a, const output::DbPoint &b) {
        const float dx = b.x - a.x, dy = b.y - a.y;
        return std::sqrt(dx * dx + dy * dy);
    };
    r.rw = 0.5f * (dist(quad[0], quad[1]) + dist(quad[3], quad[2]));
    r.rh = 0.5f * (dist(quad[0], quad[3]) + dist(quad[1], quad[2]));
    // Angle from the top edge (TL->TR).
    r.angle = std::atan2(quad[1].y - quad[0].y, quad[1].x - quad[0].x);
    return r;
}

}// namespace

// ---------------------------------------------------------------------------
// Config validation
// ---------------------------------------------------------------------------

[[nodiscard]] Status CnnExecutorConfig::validate() const {
    if (auto s = model.validate(); !s.ok()) { return s; }
    if (auto s = names.validate(); !s.ok()) { return s; }
    if (detector_component.empty() || recognizer_component.empty()) {
        return Status::error(StatusCode::invalid_argument, "weight component tags must not be empty", "component");
    }
    if (detector_component == recognizer_component) {
        return Status::error(StatusCode::invalid_argument,
                             "detector and recognizer components must be distinct", "component");
    }
    if (mean.size() != 3u || std_dev.size() != 3u) {
        return Status::error(StatusCode::invalid_argument, "mean/std_dev must have 3 channels", "mean/std_dev");
    }
    for (auto v : mean) {
        if (!std::isfinite(v)) { return Status::error(StatusCode::invalid_argument, "mean must be finite", "mean"); }
    }
    for (auto v : std_dev) {
        if (!std::isfinite(v) || v == 0.0f) {
            return Status::error(StatusCode::invalid_argument, "std_dev must be finite and nonzero", "std_dev");
        }
    }
    if (auto s = db.validate(); !s.ok()) { return s; }
    if (auto s = ctc.validate(); !s.ok()) { return s; }
    if (auto s = reading_order.validate(); !s.ok()) { return s; }
    if (auto s = dictionary.validate(); !s.ok()) { return s; }
    if (dictionary.num_classes() != model.recognizer.num_classes) {
        return Status::error(StatusCode::invalid_argument,
                             "dictionary class count must match the recognizer CTC head", "dictionary");
    }
    if (dictionary.blank_id != model.recognizer.blank_id) {
        return Status::error(StatusCode::invalid_argument,
                             "dictionary blank id must match the recognizer", "dictionary");
    }
    if (max_lines_per_page == 0u) {
        return Status::error(StatusCode::invalid_argument, "max_lines_per_page must be nonzero", "max_lines_per_page");
    }
    if (max_source_pixels == 0u) {
        return Status::error(StatusCode::invalid_argument, "max_source_pixels must be nonzero", "max_source_pixels");
    }
    return Status::success();
}

// ---------------------------------------------------------------------------
// create / resolve / reserve / compile
// ---------------------------------------------------------------------------

[[nodiscard]] Status CnnExecutor::create(Device &device, Stream &stream,
                                         const CnnExecutorConfig &config, const WeightStore &weights) {
    _recognizer_only = false;
    return create_impl(device, stream, config, weights);
}

[[nodiscard]] Status CnnExecutor::create_recognizer_only(Device &device, Stream &stream,
                                                         const CnnExecutorConfig &config, const WeightStore &weights) {
    _recognizer_only = true;
    return create_impl(device, stream, config, weights);
}

[[nodiscard]] Status CnnExecutor::create_impl(Device &device, Stream &stream,
                                              const CnnExecutorConfig &config, const WeightStore &weights) {
    auto valid = config.validate();
    if (!valid.ok()) { return valid; }
    _config = config;
    _device = &device;
    _ws = DeviceWorkspace{device};
    _conv_cache = TypedShaderCache<Conv2dKernel>{device, _shader_registry};
    _act_cache = TypedShaderCache<CnnActKernel>{device, _shader_registry};
    _fpn_cache = TypedShaderCache<FpnResizeKernel>{device, _shader_registry};
    _concat_cache = TypedShaderCache<ConcatChannelsKernel>{device, _shader_registry};
    _sigmoid_cache = TypedShaderCache<SigmoidKernel>{device, _shader_registry};
    _crop_cache = TypedShaderCache<PolygonCropKernel>{device, _shader_registry};
    _resize_cache = TypedShaderCache<ImageResizeNormalizeKernel>{device, _shader_registry};
    _collapse_cache = TypedShaderCache<HeightCollapseKernel>{device, _shader_registry};
    _gemm_cache = TypedShaderCache<GemmKernel>{device, _shader_registry};
    _logsoftmax_cache = TypedShaderCache<CtcLogSoftmaxKernel>{device, _shader_registry};
    _argmax_cache = TypedShaderCache<CtcArgmaxProbsKernel>{device, _shader_registry};

    // Geometry caches.
    const auto &det = _config.model.detector;
    const auto &rec = _config.model.recognizer;
    if (!_recognizer_only) {
        uint32_t h = det.input_height, w = det.input_width;
        for (const auto &s : det.backbone) {
            auto oh = conv2d_out_dim(h, s.kh, s.stride_h, s.pad_h, s.dil_h);
            auto ow = conv2d_out_dim(w, s.kw, s.stride_w, s.pad_w, s.dil_w);
            if (!oh.ok() || !ow.ok()) { return Status::error(StatusCode::internal_error, "detector geometry", "det"); }
            h = *oh.value;
            w = *ow.value;
        }
        _det_map_h = h;
        _det_map_w = w;
    }
    {
        auto fh = rec.feature_height();
        auto fw = rec.feature_width();
        auto cc = rec.collapsed_channels();
        auto nf = rec.num_frames();
        if (!fh.ok() || !fw.ok() || !cc.ok() || !nf.ok()) {
            return Status::error(StatusCode::internal_error, "recognizer geometry", "rec");
        }
        _rec_feat_h = *fh.value;
        _rec_feat_w = *fw.value;
        _rec_collapsed = *cc.value;
        // Frames include the height-collapse factor (matches the device
        // height-collapse kernel's w*(h/r) output rows and the arena/GEMM
        // sizing below).
        _rec_frames = *nf.value;
    }

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

    _weight_bytes = weights.component_bytes(config.recognizer_component);
    if (!_recognizer_only) { _weight_bytes += weights.component_bytes(config.detector_component); }
    return Status::success();
}

[[nodiscard]] Status CnnExecutor::create_from_packages(Device &device, Stream &stream,
                                                       const CnnExecutorConfig &config,
                                                       const WeightStore &weights,
                                                       const PackageReader &detector_package,
                                                       const PackageReader &recognizer_package) {
    // Active ON-DISK gate: validate both package tables against the declared
    // per-size inventory before trusting any uploaded weight.
    if (auto s = config.model.validate(); !s.ok()) { return s; }
    if (auto s = config.names.validate(); !s.ok()) { return s; }
    const auto det_expected = pp_ocrv6_expected_tensors(config.model, config.names, "detector");
    const auto rec_expected = pp_ocrv6_expected_tensors(config.model, config.names, "recognizer");
    if (auto s = pp_ocrv6_validate_tensors(detector_package.tensors(), det_expected); !s.ok()) { return s; }
    if (auto s = pp_ocrv6_validate_tensors(recognizer_package.tensors(), rec_expected); !s.ok()) { return s; }
    return create(device, stream, config, weights);
}

[[nodiscard]] Status CnnExecutor::resolve_weights(const WeightStore &weights) {
    const auto &names = _config.names;
    const auto resolve_component = [&](const luisa::string &comp,
                                       const luisa::vector<ExpectedTensor> &expected) -> Status {
        luisa::unordered_map<luisa::string, const ExpectedTensor *> by_name;
        for (const auto &e : expected) { by_name.emplace(e.name, &e); }
        for (const auto &name : weights.tensor_names(comp)) {
            if (by_name.find(name) == by_name.end()) {
                return Status::error(StatusCode::invalid_argument,
                                     luisa::format("unexpected pp_ocrv6 tensor '{}' in component '{}'", name, comp), name);
            }
        }
        return Status::success();
    };
    // Installed-weight checks use the DEVICE-descriptor view (conv weights
    // read row_major); the ON-DISK conv_oihw gate runs in
    // create_from_packages via pp_ocrv6_validate_tensors.
    const auto det_expected = pp_ocrv6_device_expected_tensors(_config.model, names, "detector");
    // Recognizer name resolution: a DISTINCT name map (UniRec) overrides the
    // PP-OCRv6 recognizer names. The expected inventory is rebuilt from that map
    // so the load path validates the caller's actual tensor contract -- never a
    // silent PP alias. The geometry (shapes) is the shared recognizer config.
    const bool custom_rec_names = static_cast<bool>(_config.recognizer_names.backbone_w);
    if (custom_rec_names && !_config.recognizer_names.complete()) {
        return Status::error(StatusCode::invalid_argument,
                             "recognizer name map is incomplete", "recognizer_names");
    }
    const auto rec_name = [&](auto pp_fn, auto custom_fn) -> luisa::string {
        return custom_rec_names ? custom_fn() : (names.*pp_fn)();
    };
    const auto rec_name_i = [&](auto pp_fn, auto custom_fn, uint32_t i) -> luisa::string {
        return custom_rec_names ? custom_fn(i) : (names.*pp_fn)(i);
    };
    luisa::vector<ExpectedTensor> rec_expected;
    if (custom_rec_names) {
        const auto &rec = _config.model.recognizer;
        const auto &nm = _config.recognizer_names;
        const auto conv_w = [](const luisa::string &n, const PpConvStage &s) {
            ExpectedTensor t;
            t.name = n;
            t.shape = {s.out_c, s.in_c / s.groups, s.kh, s.kw};
            t.layout = TensorLayout::row_major;// device view
            return t;
        };
        const auto bias_t = [](const luisa::string &n, uint32_t c) {
            ExpectedTensor t;
            t.name = n;
            t.shape = {c};
            t.layout = TensorLayout::row_major;
            return t;
        };
        const auto kn_w = [](const luisa::string &n, uint32_t k, uint32_t nn) {
            ExpectedTensor t;
            t.name = n;
            t.shape = {k, nn};
            t.layout = TensorLayout::kn;
            return t;
        };
        for (uint32_t i = 0; i < rec.backbone.size(); i++) {
            rec_expected.push_back(conv_w(nm.backbone_w(i), rec.backbone[i]));
            rec_expected.push_back(bias_t(nm.backbone_b(i), rec.backbone[i].out_c));
        }
        uint32_t collapsed = 0;
        if (auto cc = rec.collapsed_channels(); cc.ok()) { collapsed = *cc.value; }
        rec_expected.push_back(kn_w(nm.svtr_fc1_w(), collapsed, rec.svtr_intermediate));
        rec_expected.push_back(bias_t(nm.svtr_fc1_b(), rec.svtr_intermediate));
        rec_expected.push_back(kn_w(nm.svtr_fc2_w(), rec.svtr_intermediate, collapsed));
        rec_expected.push_back(bias_t(nm.svtr_fc2_b(), collapsed));
        rec_expected.push_back(kn_w(nm.ctc_w(), collapsed, rec.num_classes));
        rec_expected.push_back(bias_t(nm.ctc_b(), rec.num_classes));
    } else {
        rec_expected = pp_ocrv6_device_expected_tensors(_config.model, names, "recognizer");
    }
    if (!_recognizer_only) {
        if (auto s = resolve_component(_config.detector_component, det_expected); !s.ok()) { return s; }
    }
    if (auto s = resolve_component(_config.recognizer_component, rec_expected); !s.ok()) { return s; }

    const auto view_of = [&](const luisa::string &comp, const luisa::string &name,
                             const luisa::vector<ExpectedTensor> &expected,
                             BufferView<float> &out) -> Status {
        const auto *w = weights.find(comp, name);
        if (w == nullptr) {
            return Status::error(StatusCode::not_found,
                                 luisa::format("pp_ocrv6 weight '{}' not found in component '{}'", name, comp), name);
        }
        if (w->encoding != TensorEncoding::fp32 || !w->dense) {
            return Status::error(StatusCode::unsupported,
                                 luisa::format("pp_ocrv6 weight '{}' must be dense fp32 in the Phase 5 baseline", name), name);
        }
        const ExpectedTensor *exp = nullptr;
        for (const auto &e : expected) {
            if (e.name == name) {
                exp = &e;
                break;
            }
        }
        if (exp == nullptr) {
            return Status::error(StatusCode::invalid_argument,
                                 luisa::format("pp_ocrv6 weight '{}' has no expected-tensor contract", name), name);
        }
        if (auto s = validate_decoder_weight_desc(w->logical, w->element_count, *exp); !s.ok()) { return s; }
        if (w->dense.size() < w->element_count) {
            return Status::error(StatusCode::invalid_argument,
                                 luisa::format("pp_ocrv6 weight '{}' buffer smaller than element count", name), name);
        }
        out = w->dense.view();
        return Status::success();
    };

    const auto &dc = _config.detector_component;
    const auto &rc = _config.recognizer_component;
    const auto &det = _config.model.detector;
    const auto &rec = _config.model.recognizer;
    if (!_recognizer_only) {
        _det_backbone.resize(det.backbone.size());
        for (uint32_t i = 0; i < det.backbone.size(); i++) {
            if (auto s = view_of(dc, names.det_backbone_w(i), det_expected, _det_backbone[i].weight); !s.ok()) { return s; }
            if (auto s = view_of(dc, names.det_backbone_b(i), det_expected, _det_backbone[i].bias); !s.ok()) { return s; }
        }
        if (auto s = view_of(dc, names.det_lateral_w(), det_expected, _det_lateral.weight); !s.ok()) { return s; }
        if (auto s = view_of(dc, names.det_lateral_b(), det_expected, _det_lateral.bias); !s.ok()) { return s; }
        if (auto s = view_of(dc, names.det_replk_w(), det_expected, _det_replk.weight); !s.ok()) { return s; }
        if (auto s = view_of(dc, names.det_replk_b(), det_expected, _det_replk.bias); !s.ok()) { return s; }
        if (auto s = view_of(dc, names.det_db_w(), det_expected, _det_db.weight); !s.ok()) { return s; }
        if (auto s = view_of(dc, names.det_db_b(), det_expected, _det_db.bias); !s.ok()) { return s; }
    }

    const auto &rnm = _config.recognizer_names;
    _rec_backbone.resize(rec.backbone.size());
    for (uint32_t i = 0; i < rec.backbone.size(); i++) {
        if (auto s = view_of(rc, rec_name_i(&PpOcrv6TensorNames::rec_backbone_w, rnm.backbone_w, i), rec_expected, _rec_backbone[i].weight); !s.ok()) { return s; }
        if (auto s = view_of(rc, rec_name_i(&PpOcrv6TensorNames::rec_backbone_b, rnm.backbone_b, i), rec_expected, _rec_backbone[i].bias); !s.ok()) { return s; }
    }
    if (auto s = view_of(rc, rec_name(&PpOcrv6TensorNames::rec_svtr_fc1_w, rnm.svtr_fc1_w), rec_expected, _rec_svtr_fc1.weight); !s.ok()) { return s; }
    if (auto s = view_of(rc, rec_name(&PpOcrv6TensorNames::rec_svtr_fc1_b, rnm.svtr_fc1_b), rec_expected, _rec_svtr_fc1.bias); !s.ok()) { return s; }
    if (auto s = view_of(rc, rec_name(&PpOcrv6TensorNames::rec_svtr_fc2_w, rnm.svtr_fc2_w), rec_expected, _rec_svtr_fc2.weight); !s.ok()) { return s; }
    if (auto s = view_of(rc, rec_name(&PpOcrv6TensorNames::rec_svtr_fc2_b, rnm.svtr_fc2_b), rec_expected, _rec_svtr_fc2.bias); !s.ok()) { return s; }
    if (auto s = view_of(rc, rec_name(&PpOcrv6TensorNames::rec_ctc_w, rnm.ctc_w), rec_expected, _rec_ctc.weight); !s.ok()) { return s; }
    if (auto s = view_of(rc, rec_name(&PpOcrv6TensorNames::rec_ctc_b, rnm.ctc_b), rec_expected, _rec_ctc.bias); !s.ok()) { return s; }
    return Status::success();
}

[[nodiscard]] Status CnnExecutor::reserve_workspace() {
    const auto &det = _config.model.detector;
    const auto &rec = _config.model.recognizer;
    const uint64_t dp = static_cast<uint64_t>(det.input_width) * det.input_height;
    const uint64_t rp = static_cast<uint64_t>(rec.input_width) * rec.input_height;

    // Detector intermediate sizes (worst-case over the backbone).
    uint64_t det_max = 0;
    {
        uint32_t h = det.input_height, w = det.input_width, c = det.in_channels;
        for (const auto &s : det.backbone) {
            auto oh = conv2d_out_dim(h, s.kh, s.stride_h, s.pad_h, s.dil_h);
            auto ow = conv2d_out_dim(w, s.kw, s.stride_w, s.pad_w, s.dil_w);
            if (!oh.ok() || !ow.ok()) { return oh.status; }
            h = *oh.value;
            w = *ow.value;
            c = s.out_c;
            det_max = std::max(det_max, static_cast<uint64_t>(c) * h * w);
        }
    }
    const uint64_t lat_c = det.lateral_conv.out_c;
    const uint64_t fused_c = det.replk_conv.in_c;
    const uint64_t replk_c = det.replk_conv.out_c;
    const uint64_t map_plane = static_cast<uint64_t>(_det_map_h) * _det_map_w;
    // Lateral/up/fused operate at the backbone output resolution.
    const uint64_t lat_elems = lat_c * map_plane;
    const uint64_t fused_elems = fused_c * map_plane;
    const uint64_t replk_elems = replk_c * map_plane;

    // Recognizer intermediate sizes.
    uint64_t rec_max = 0;
    {
        uint32_t h = rec.input_height, w = rec.input_width, c = rec.in_channels;
        for (const auto &s : rec.backbone) {
            auto oh = conv2d_out_dim(h, s.kh, s.stride_h, s.pad_h, s.dil_h);
            auto ow = conv2d_out_dim(w, s.kw, s.stride_w, s.pad_w, s.dil_w);
            if (!oh.ok() || !ow.ok()) { return oh.status; }
            h = *oh.value;
            w = *ow.value;
            c = s.out_c;
            rec_max = std::max(rec_max, static_cast<uint64_t>(c) * h * w);
        }
    }
    const uint64_t frames = _rec_frames;
    const uint64_t collapsed = _rec_collapsed;
    const uint64_t inter = rec.svtr_intermediate;
    const uint64_t classes = rec.num_classes;

    struct RegionSpec {
        const char *name;
        uint64_t elements;
    };
    // In recognizer-only mode the detector arenas are unused; zero them so no
    // detector workspace is allocated (the region layout stays consistent).
    const uint64_t z = _recognizer_only ? 0u : 1u;
    const RegionSpec specs[] = {
        {"page_pixels", 3u * dp * z},
        {"det_a", det_max * z},
        {"det_b", det_max * z},
        {"det_lateral", lat_elems * z},
        {"det_up", lat_elems * z},
        {"det_fused", fused_elems * z},
        {"det_replk", replk_elems * z},
        {"det_prob", map_plane * z},
        {"crop_norm", 3u * rp},
        {"crop", 3u * rp},
        {"rec_a", rec_max},
        {"rec_b", rec_max},
        {"rec_seq", frames * collapsed},
        {"rec_svtr1", frames * inter},
        {"rec_svtr2", frames * collapsed},
        {"rec_logits", frames * classes},
        {"rec_logprob", frames * classes},
        {"rec_ids", frames},
        {"rec_probs", frames},
        {"svtr_aux", std::max(inter, collapsed)},
        {"dummy_f", 1},
    };
    WorkspacePlan plan;
    for (const auto &spec : specs) {
        if (spec.elements == 0u) { continue; }// unused region (recognizer-only detector arenas)
        auto bytes = f32_bytes(spec.elements);
        if (!bytes.ok()) { return bytes.status; }
        if (auto s = plan.add_bytes(spec.name, *bytes.value); !s.ok()) { return s; }
    }
    MemoryBudget budget{_config.memory};
    if (auto s = budget.require("cnn_workspace", MemoryCategory::workspace, plan.total_bytes()); !s.ok()) { return s; }
    if (auto s = budget.preflight(); !s.ok()) { return s; }
    if (auto s = _ws.reserve(plan.total_bytes()); !s.ok()) { return s; }

    WsRegion *outs[] = {
        &_page_pixels, &_det_a, &_det_b, &_det_lat_ws, &_det_up, &_det_fused,
        &_det_replk_ws, &_det_prob, &_crop_norm, &_crop, &_rec_a, &_rec_b, &_rec_seq,
        &_rec_svtr1, &_rec_svtr2, &_rec_logits, &_rec_logprob, &_rec_ids, &_rec_probs,
        &_svtr_aux, &_dummy_f,
    };
    for (size_t i = 0; i < std::size(specs); i++) {
        if (specs[i].elements == 0u) {
            *outs[i] = WsRegion{};// empty region; never bound on this path
            continue;
        }
        auto bytes = f32_bytes(specs[i].elements);
        auto alloc = _ws.allocate(*bytes.value);
        if (!alloc.ok()) { return alloc.status; }
        outs[i]->offset = alloc.value->offset;
        outs[i]->bytes = alloc.value->bytes;
    }
    // Rewind the cursor so the first request starts at offset 0; the regions
    // above keep their fixed offsets for the executor's lifetime.
    _ws.reset();
    return Status::success();
}

[[nodiscard]] Status CnnExecutor::compile_shaders() {
    const auto get = [](auto &cache, const luisa::string &key, auto factory, auto *&out) -> Status {
        auto r = cache.get_or_compile(key, factory);
        if (!r.ok()) { return r.status; }
        out = *r.value;
        return Status::success();
    };
    if (auto s = get(_conv_cache, conv_variant_key("conv2d"),
                     [] { return make_conv2d_kernel(); }, _sh_conv); !s.ok()) { return s; }
    for (uint32_t act = 0; act < 4u; act++) {
        if (auto s = get(_act_cache, ctc_variant_key(luisa::format("act_{}", act)),
                         [act] { return make_cnn_act_kernel(act); }, _sh_act[act]); !s.ok()) { return s; }
    }
    if (auto s = get(_fpn_cache, ctc_variant_key("fpn_resize"),
                     [] { return make_fpn_resize_kernel(); }, _sh_fpn); !s.ok()) { return s; }
    if (auto s = get(_concat_cache, conv_variant_key("concat_channels"),
                     [] { return make_concat_channels_kernel(); }, _sh_concat); !s.ok()) { return s; }
    if (auto s = get(_sigmoid_cache, ctc_variant_key("sigmoid"),
                     [] { return make_sigmoid_kernel(); }, _sh_sigmoid); !s.ok()) { return s; }
    if (auto s = get(_crop_cache, ctc_variant_key("polygon_crop"),
                     [] { return make_polygon_crop_kernel(); }, _sh_crop); !s.ok()) { return s; }
    if (auto s = get(_resize_cache, image_variant_key("resize_normalize"),
                     [] { return make_image_resize_normalize_kernel(); }, _sh_resize); !s.ok()) { return s; }
    if (auto s = get(_collapse_cache, ctc_variant_key("height_collapse"),
                     [] { return make_height_collapse_kernel(); }, _sh_collapse); !s.ok()) { return s; }
    if (auto s = get(_gemm_cache, gemm_variant_key(GemmEpilogue::bias, GemmWeightFormat::fp32, false),
                     [] { return make_gemm_kernel(GemmEpilogue::bias); }, _sh_gemm_bias); !s.ok()) { return s; }
    if (auto s = get(_gemm_cache, gemm_variant_key(GemmEpilogue::bias_gelu_erf, GemmWeightFormat::fp32, false),
                     [] { return make_gemm_kernel(GemmEpilogue::bias_gelu_erf); }, _sh_gemm_gelu); !s.ok()) { return s; }
    if (auto s = get(_logsoftmax_cache, ctc_variant_key("log_softmax"),
                     [] { return make_ctc_log_softmax_kernel(); }, _sh_logsoftmax); !s.ok()) { return s; }
    if (auto s = get(_argmax_cache, ctc_variant_key("argmax_probs"),
                     [] { return make_ctc_argmax_probs_kernel(); }, _sh_argmax); !s.ok()) { return s; }
    return Status::success();
}

// ---------------------------------------------------------------------------
// conv helper
// ---------------------------------------------------------------------------

[[nodiscard]] Status CnnExecutor::record_conv(CommandList &list, const PpConvStage &stage,
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
                                        stage.stride_h, stage.stride_w, stage.dil_h, stage.dil_w, stage.groups);
        !s.ok()) {
        return s;
    }
    auto plan = plan_conv2d_dispatch(n, stage.out_c, out_h, out_w);
    if (!plan.ok()) { return plan.status; }
    if (auto cmd = dispatch_1d((*_sh_conv)(in, w.weight, w.bias, out, n, stage.in_c, in_h, in_w,
                                           stage.out_c, stage.kh, stage.kw, stage.stride_h, stage.stride_w,
                                           stage.pad_h, stage.pad_w, stage.dil_h, stage.dil_w, stage.groups,
                                           out_h, out_w, 1u),
                               *plan.value)) {
        list << std::move(cmd);
    }
    // Activation (in-place on out via a second pass is not aliasing-safe for
    // the conv output, so apply elementwise out->out; the act kernel reads and
    // writes the same index, which is safe).
    if (stage.act != 0u) {
        const uint64_t count = static_cast<uint64_t>(n) * stage.out_c * out_h * out_w;
        auto act_plan = plan_cnn_act_dispatch(count);
        if (!act_plan.ok()) { return act_plan.status; }
        auto count32 = checked_u32(count, "act count");
        if (!count32.ok()) { return count32.status; }
        if (auto cmd = dispatch_1d((*_sh_act[stage.act])(out, out, *count32.value), *act_plan.value)) {
            list << std::move(cmd);
        }
    }
    return Status::success();
}

// ---------------------------------------------------------------------------
// detector graph
// ---------------------------------------------------------------------------

[[nodiscard]] Status CnnExecutor::record_detector(CommandList &list, BufferView<float> pixels,
                                                  uint32_t &map_h, uint32_t &map_w) {
    const auto &det = _config.model.detector;
    // Backbone: ping-pong between det_a and det_b. Track the skip feature.
    BufferView<float> cur = pixels;
    uint32_t h = det.input_height, w = det.input_width;
    uint32_t skip_h = 0, skip_w = 0;
    BufferView<float> skip;
    bool use_a = true;
    for (uint32_t i = 0; i < det.backbone.size(); i++) {
        auto out = use_a ? ws(_det_a) : ws(_det_b);
        uint32_t oh = 0, ow = 0;
        if (auto s = record_conv(list, det.backbone[i], _det_backbone[i], cur, out, 1u, h, w, oh, ow); !s.ok()) {
            return s;
        }
        cur = out;
        h = oh;
        w = ow;
        use_a = !use_a;
        if (i == det.lateral_stage) {
            skip = cur;
            skip_h = h;
            skip_w = w;
        }
    }
    // `cur` now holds the backbone output at (h,w) == (_det_map_h,_det_map_w).
    // Lateral 1x1 on the skip.
    auto lateral = ws(_det_lat_ws);
    uint32_t lh = 0, lw = 0;
    if (auto s = record_conv(list, det.lateral_conv, _det_lateral, skip, lateral, 1u, skip_h, skip_w, lh, lw); !s.ok()) {
        return s;
    }
    // Upsample the lateral to the backbone output resolution.
    auto up = ws(_det_up);
    {
        auto plan = plan_fpn_resize_dispatch(1u, det.lateral_conv.out_c, h, w);
        if (!plan.ok()) { return plan.status; }
        if (auto cmd = dispatch_1d((*_sh_fpn)(lateral, up, 1u, det.lateral_conv.out_c, lh, lw, h, w), *plan.value)) {
            list << std::move(cmd);
        }
    }
    // Fuse: concat backbone output (cur) + upsampled lateral along channels.
    auto fused = ws(_det_fused);
    {
        const uint32_t ca = det.backbone.back().out_c;
        const uint32_t cb = det.lateral_conv.out_c;
        const uint32_t spatial = h * w;
        auto plan = plan_elementwise_dispatch(static_cast<uint64_t>(ca + cb) * spatial);
        if (!plan.ok()) { return plan.status; }
        if (auto cmd = dispatch_1d((*_sh_concat)(cur, up, fused, 1u, ca, cb, spatial), *plan.value)) {
            list << std::move(cmd);
        }
    }
    // RepLK folded large-kernel conv.
    auto replk = ws(_det_replk_ws);
    uint32_t rh = 0, rw = 0;
    if (auto s = record_conv(list, det.replk_conv, _det_replk, fused, replk, 1u, h, w, rh, rw); !s.ok()) {
        return s;
    }
    // DB head 1x1 -> 1 channel logit, then sigmoid.
    auto prob = ws(_det_prob);
    uint32_t ph = 0, pw = 0;
    if (auto s = record_conv(list, det.db_conv, _det_db, replk, prob, 1u, rh, rw, ph, pw); !s.ok()) {
        return s;
    }
    {
        const uint64_t count = static_cast<uint64_t>(ph) * pw;
        auto plan = plan_sigmoid_dispatch(count);
        if (!plan.ok()) { return plan.status; }
        auto count32 = checked_u32(count, "sigmoid count");
        if (!count32.ok()) { return count32.status; }
        if (auto cmd = dispatch_1d((*_sh_sigmoid)(prob, prob, *count32.value), *plan.value)) {
            list << std::move(cmd);
        }
    }
    map_h = ph;
    map_w = pw;
    return Status::success();
}

// ---------------------------------------------------------------------------
// recognizer graph
// ---------------------------------------------------------------------------

[[nodiscard]] Status CnnExecutor::record_recognizer(CommandList &list, BufferView<float> crop,
                                                    uint32_t &frames) {
    const auto &rec = _config.model.recognizer;
    BufferView<float> cur = crop;
    uint32_t h = rec.input_height, w = rec.input_width;
    bool use_a = true;
    for (uint32_t i = 0; i < rec.backbone.size(); i++) {
        auto out = use_a ? ws(_rec_a) : ws(_rec_b);
        uint32_t oh = 0, ow = 0;
        if (auto s = record_conv(list, rec.backbone[i], _rec_backbone[i], cur, out, 1u, h, w, oh, ow); !s.ok()) {
            return s;
        }
        cur = out;
        h = oh;
        w = ow;
        use_a = !use_a;
    }
    // LightSVTR height-collapse: [1, c, h, w] -> [w*(h/r), c*r].
    auto seq = ws(_rec_seq);
    {
        const uint32_t c = rec.backbone.back().out_c;
        const uint32_t r = rec.height_collapse;
        const uint64_t out_frames = static_cast<uint64_t>(w) * (h / r);
        const uint32_t channels = c * r;
        auto plan = plan_height_collapse_dispatch(out_frames, channels);
        if (!plan.ok()) { return plan.status; }
        if (auto cmd = dispatch_1d((*_sh_collapse)(cur, seq, 1u, c, h, w, r), *plan.value)) {
            list << std::move(cmd);
        }
    }
    const uint32_t f = _rec_frames;
    const uint32_t collapsed = _rec_collapsed;
    const uint32_t inter = rec.svtr_intermediate;
    const uint32_t classes = rec.num_classes;
    // SVTR per-frame MLP: fc1 (GELU) then fc2 (bias). A [f, collapsed].
    auto svtr1 = ws(_rec_svtr1);
    auto svtr2 = ws(_rec_svtr2);
    auto aux = ws(_svtr_aux);
    const auto record_linear = [&](BufferView<float> A, const ConvWeights &W, BufferView<float> C,
                                   uint32_t m, uint32_t n, uint32_t k, bool gelu) -> Status {
        auto plan = plan_gemm_dispatch_epilogue(m, n, k, gelu ? GemmEpilogue::bias_gelu_erf : GemmEpilogue::bias);
        if (!plan.ok()) { return plan.status; }
        const auto *shader = gelu ? _sh_gemm_gelu : _sh_gemm_bias;
        if (auto cmd = dispatch_2d((*shader)(A, W.weight, C, W.bias, m, n, k, plan.value->padded_cols), *plan.value)) {
            list << std::move(cmd);
        }
        return Status::success();
    };
    if (auto s = record_linear(seq, _rec_svtr_fc1, svtr1, f, inter, collapsed, true); !s.ok()) { return s; }
    if (auto s = record_linear(svtr1, _rec_svtr_fc2, svtr2, f, collapsed, inter, false); !s.ok()) { return s; }
    // CTC head: [f, collapsed] -> [f, classes].
    auto logits = ws(_rec_logits);
    if (auto s = record_linear(svtr2, _rec_ctc, logits, f, classes, collapsed, false); !s.ok()) { return s; }
    frames = f;
    (void)aux;
    return Status::success();
}

// ---------------------------------------------------------------------------
// recognize (full pipeline)
// ---------------------------------------------------------------------------

[[nodiscard]] Result<CnnRunResult> CnnExecutor::recognize(Stream &stream, luisa::span<const CnnPageInput> pages) {
    using R = Result<CnnRunResult>;
    if (_device == nullptr) {
        return R::failure(StatusCode::invalid_argument, "executor not created", "device");
    }
    if (_recognizer_only) {
        return R::failure(StatusCode::unsupported,
                          "the full detector->recognizer pipeline is unavailable in recognizer-only mode; "
                          "use recognize_line() per region",
                          "recognizer_only");
    }
    CnnRunResult run;
    run.weight_bytes = _weight_bytes;
    const auto &det = _config.model.detector;
    const auto &rec = _config.model.recognizer;

    for (const auto &page : pages) {
        if (page.source_width == 0u || page.source_height == 0u ||
            page.pixels.size() != static_cast<size_t>(page.source_width) * page.source_height * 3u) {
            return R::failure(StatusCode::invalid_argument, "degenerate page input", "page");
        }
        const uint64_t source_pixels = static_cast<uint64_t>(page.source_width) * page.source_height;
        if (source_pixels > _config.max_source_pixels) {
            return R::failure(StatusCode::invalid_argument, "page exceeds max_source_pixels", "page");
        }
        CnnPageResult page_result;
        page_result.page_index = page.page_index;
        page_result.source_width = page.source_width;
        page_result.source_height = page.source_height;

        // ---- Stage 1: preprocessing (host) + upload ----
        _ws.reset();
        auto pixels_host = preprocess_page_host(page, det.input_width, det.input_height,
                                                _config.mean, _config.std_dev);
        auto page_pixels = ws(_page_pixels);
        stream << page_pixels.copy_from(luisa::span<const float>{pixels_host});

        // ---- Stage 2: detector ----
        uint32_t map_h = 0, map_w = 0;
        {
            auto list = CommandList::create();
            if (auto s = record_detector(list, page_pixels, map_h, map_w); !s.ok()) { return R::failure(s); }
            stream << list.commit() << synchronize();
        }
        // Download the probability map.
        page_result.det_map_width = map_w;
        page_result.det_map_height = map_h;
        page_result.det_probabilities.resize(static_cast<size_t>(map_h) * map_w);
        stream << ws(_det_prob).subview(0, page_result.det_probabilities.size())
                      .copy_to(luisa::span<float>{page_result.det_probabilities})
               << synchronize();

        // ---- Stage 3: DB postprocess (CPU) -> source polygons ----
        // Detector-map -> source transform: the page was resized to the
        // detector input, and the map is the backbone-strided detector input.
        output::DbSourceTransform transform;
        transform.scale_x = static_cast<float>(page.source_width) / static_cast<float>(map_w);
        transform.scale_y = static_cast<float>(page.source_height) / static_cast<float>(map_h);
        auto db_options = _config.db;
        if (db_options.max_boxes == 0u) { db_options.max_boxes = _config.max_lines_per_page; }
        auto db = output::db_postprocess(page_result.det_probabilities, map_w, map_h, transform, db_options);
        if (!db.ok()) { return R::failure(db.status); }
        page_result.boxes = std::move(db.value->boxes);

        // ---- Stages 4-6: per-line crop + recognizer + CTC ----
        luisa::vector<output::ReadingOrderLine> ro_lines;
        for (const auto &box : page_result.boxes) {
            if (ro_lines.size() >= _config.max_lines_per_page) { break; }
            const auto rect = crop_rect_from_quad(box.polygon);
            if (rect.rw <= 0.0f || rect.rh <= 0.0f) { continue; }
            // Crop from the preprocessed page (detector-input CHW), mapping
            // the source-coordinate rect into detector-input coordinates.
            const float to_det_x = static_cast<float>(det.input_width) / static_cast<float>(page.source_width);
            const float to_det_y = static_cast<float>(det.input_height) / static_cast<float>(page.source_height);
            // Uniform scale for the rotated rect (average of axis scales).
            const float to_det = 0.5f * (to_det_x + to_det_y);
            const float cx = rect.cx * to_det_x;
            const float cy = rect.cy * to_det_y;
            const float rw = rect.rw * to_det;
            const float rh = rect.rh * to_det;
            // Crop to the recognizer input size.
            auto crop_norm = ws(_crop_norm);
            {
                auto plan = plan_polygon_crop_dispatch(rec.input_width, rec.input_height);
                if (!plan.ok()) { return R::failure(plan.status); }
                if (auto cmd = dispatch_1d((*_sh_crop)(page_pixels, crop_norm, det.input_width, det.input_height,
                                                       rec.input_width, rec.input_height, cx, cy, rw, rh, rect.angle),
                                           *plan.value)) {
                    stream << std::move(cmd);
                }
            }
            // Recognizer.
            uint32_t frames = 0;
            {
                auto list = CommandList::create();
                if (auto s = record_recognizer(list, crop_norm, frames); !s.ok()) { return R::failure(s); }
                stream << list.commit() << synchronize();
            }
            // CTC log-softmax + argmax/probs (device).
            auto logprob = ws(_rec_logprob);
            auto ids_view = ws_i32(_rec_ids);
            auto probs_view = ws(_rec_probs);
            {
                auto plan = plan_ctc_log_softmax_dispatch(frames);
                if (!plan.ok()) { return R::failure(plan.status); }
                if (auto cmd = dispatch_1d((*_sh_logsoftmax)(ws(_rec_logits), logprob, frames, rec.num_classes,
                                                             _config.ctc.temperature),
                                           *plan.value)) {
                    stream << std::move(cmd);
                }
            }
            {
                auto plan = plan_ctc_argmax_dispatch(frames);
                if (!plan.ok()) { return R::failure(plan.status); }
                if (auto cmd = dispatch_1d((*_sh_argmax)(logprob, ids_view, probs_view, frames, rec.num_classes),
                                           *plan.value)) {
                    stream << std::move(cmd);
                }
            }
            stream << synchronize();
            // Download ids/probs and decode on the host.
            luisa::vector<int32_t> ids(frames);
            luisa::vector<float> probs(frames);
            stream << ids_view.subview(0, frames).copy_to(luisa::span<int32_t>{ids})
                   << probs_view.subview(0, frames).copy_to(luisa::span<float>{probs})
                   << synchronize();
            auto decoded = output::ctc_decode_greedy(ids, probs, _config.dictionary);
            if (!decoded.ok()) { return R::failure(decoded.status); }
            if (decoded.value->empty()) { continue; }// blank line: no hallucination

            output::ReadingOrderLine line;
            line.text = std::move(decoded.value->text);
            line.page_index = page.page_index;
            line.x0 = box.bounds.x0;
            line.y0 = box.bounds.y0;
            line.x1 = box.bounds.x1;
            line.y1 = box.bounds.y1;
            for (const auto &p : box.polygon) { line.polygon.emplace_back(p.x, p.y); }
            line.has_confidence = decoded.value->has_confidence;
            line.confidence = decoded.value->confidence;
            line.ctc_token_count = static_cast<uint32_t>(decoded.value->ids.size());
            ro_lines.push_back(std::move(line));
        }

        // ---- Stage 7: reading order + canonical join ----
        auto ordered = output::order_lines(ro_lines, _config.reading_order);
        if (!ordered.ok()) { return R::failure(ordered.status); }
        for (const auto &line : *ordered.value) {
            CnnLine out;
            out.text = line.text;
            out.page_index = line.page_index;
            out.reading_order = line.reading_order;
            out.bounds = output::DbRect{line.x0, line.y0, line.x1, line.y1};
            for (const auto &p : line.polygon) { out.polygon.push_back(output::DbPoint{p.first, p.second}); }
            out.has_confidence = line.has_confidence;
            out.confidence = line.confidence;
            out.ctc_token_count = line.ctc_token_count;
            page_result.lines.push_back(std::move(out));
        }
        page_result.text = output::join_ordered_text(*ordered.value);
        run.pages.push_back(std::move(page_result));
    }
    run.workspace_bytes = _ws.capacity_bytes();
    return R::success(std::move(run));
}

// ---------------------------------------------------------------------------
// recognize_line (per-region CNN+CTC entry point; reused by OpenDoc UniRec)
// ---------------------------------------------------------------------------

[[nodiscard]] Result<output::CtcDecodeResult> CnnExecutor::recognize_line(Stream &stream,
                                                                          luisa::span<const float> crop_norm) {
    using R = Result<output::CtcDecodeResult>;
    if (_device == nullptr) {
        return R::failure(StatusCode::invalid_argument, "executor not created", "device");
    }
    const auto &rec = _config.model.recognizer;
    const size_t expect = static_cast<size_t>(3) * rec.input_width * rec.input_height;
    if (crop_norm.size() != expect) {
        return R::failure(StatusCode::invalid_argument,
                          "line crop size does not match the recognizer input", "crop_norm");
    }
    // Upload the normalized crop into the recognizer crop arena. The detector
    // arenas are unused on this path; the recognizer arenas keep their fixed
    // offsets from create(), so no re-allocation happens here.
    auto crop_view = ws(_crop_norm);
    stream << crop_view.subview(0, expect).copy_from(crop_norm);
    uint32_t frames = 0;
    {
        auto list = CommandList::create();
        if (auto s = record_recognizer(list, crop_view, frames); !s.ok()) { return R::failure(s); }
        stream << list.commit() << synchronize();
    }
    // CTC log-softmax + argmax/probs (device), then host greedy decode.
    auto logprob = ws(_rec_logprob);
    auto ids_view = ws_i32(_rec_ids);
    auto probs_view = ws(_rec_probs);
    {
        auto plan = plan_ctc_log_softmax_dispatch(frames);
        if (!plan.ok()) { return R::failure(plan.status); }
        if (auto cmd = dispatch_1d((*_sh_logsoftmax)(ws(_rec_logits), logprob, frames, rec.num_classes,
                                                     _config.ctc.temperature), *plan.value)) {
            stream << std::move(cmd);
        }
    }
    {
        auto plan = plan_ctc_argmax_dispatch(frames);
        if (!plan.ok()) { return R::failure(plan.status); }
        if (auto cmd = dispatch_1d((*_sh_argmax)(logprob, ids_view, probs_view, frames, rec.num_classes), *plan.value)) {
            stream << std::move(cmd);
        }
    }
    stream << synchronize();
    luisa::vector<int32_t> ids(frames);
    luisa::vector<float> probs(frames);
    stream << ids_view.subview(0, frames).copy_to(luisa::span<int32_t>{ids})
           << probs_view.subview(0, frames).copy_to(luisa::span<float>{probs})
           << synchronize();
    return output::ctc_decode_greedy(ids, probs, _config.dictionary);
}

// ---------------------------------------------------------------------------
// download helpers (layered golden tests)
// ---------------------------------------------------------------------------

[[nodiscard]] Status CnnExecutor::download_detector_prob(Stream &stream, luisa::vector<float> &out) const {
    const size_t count = static_cast<size_t>(_det_map_h) * _det_map_w;
    out.resize(count);
    if (count != 0) {
        stream << ws(_det_prob).subview(0, count).copy_to(luisa::span<float>{out});
    }
    return Status::success();
}

[[nodiscard]] Status CnnExecutor::download_det_a(Stream &stream, luisa::vector<float> &out) const {
    out.resize(static_cast<size_t>(_det_a.bytes / 4u));
    stream << ws(_det_a).copy_to(luisa::span<float>{out});
    return Status::success();
}
[[nodiscard]] Status CnnExecutor::download_det_b(Stream &stream, luisa::vector<float> &out) const {
    out.resize(static_cast<size_t>(_det_b.bytes / 4u));
    stream << ws(_det_b).copy_to(luisa::span<float>{out});
    return Status::success();
}
[[nodiscard]] Status CnnExecutor::download_det_fused(Stream &stream, luisa::vector<float> &out) const {
    out.resize(static_cast<size_t>(_det_fused.bytes / 4u));
    stream << ws(_det_fused).copy_to(luisa::span<float>{out});
    return Status::success();
}
[[nodiscard]] Status CnnExecutor::download_det_replk(Stream &stream, luisa::vector<float> &out) const {
    out.resize(static_cast<size_t>(_det_replk_ws.bytes / 4u));
    stream << ws(_det_replk_ws).copy_to(luisa::span<float>{out});
    return Status::success();
}
[[nodiscard]] Status CnnExecutor::download_line_crop(Stream &stream, luisa::vector<float> &out) const {
    const size_t count = static_cast<size_t>(3) * _config.model.recognizer.input_width *
                         _config.model.recognizer.input_height;
    out.resize(count);
    stream << ws(_crop_norm).subview(0, count).copy_to(luisa::span<float>{out});
    return Status::success();
}

[[nodiscard]] Status CnnExecutor::download_recognizer_logits(Stream &stream, luisa::vector<float> &out) const {
    const size_t count = static_cast<size_t>(_rec_frames) * _config.model.recognizer.num_classes;
    out.resize(count);
    if (count != 0) {
        stream << ws(_rec_logits).subview(0, count).copy_to(luisa::span<float>{out});
    }
    return Status::success();
}

}// namespace luisa::ocr
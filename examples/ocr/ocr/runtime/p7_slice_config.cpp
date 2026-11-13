// OCR engine -- shared Phase 7 vertical-slice configuration (device-safe).
// Host math + validation for the Phase 7 custom dense/layout variants. See
// p7_slice_config.h. SYNTHETIC contracts only; real artifacts fail closed.
#include "ocr/runtime/p7_slice_config.h"

#include <algorithm>
#include <cmath>
#include <utility>

#include <luisa/core/stl/format.h>

namespace luisa::ocr {

// ---------------------------------------------------------------------------
// Vision config
// ---------------------------------------------------------------------------

[[nodiscard]] Status P7VisionConfig::validate() const {
    const auto nonzero = [](uint32_t v, const char *what) -> Status {
        if (v == 0u) {
            return Status::error(StatusCode::invalid_argument,
                                 luisa::format("p7 vision field '{}' must be nonzero", what), what);
        }
        return Status::success();
    };
    for (const auto &[v, name] : std::initializer_list<std::pair<uint32_t, const char *>>{
             {num_layers, "num_layers"}, {hidden_size, "hidden_size"}, {num_heads, "num_heads"},
             {head_dim, "head_dim"}, {intermediate_size, "intermediate_size"}, {patch_size, "patch_size"}}) {
        if (auto s = nonzero(v, name); !s.ok()) { return s; }
    }
    if (static_cast<uint64_t>(num_heads) * head_dim != hidden_size) {
        return Status::error(StatusCode::invalid_argument, "p7 vision heads*head_dim must equal hidden_size", "head_dim");
    }
    if (head_dim % 2u != 0u) {
        return Status::error(StatusCode::invalid_argument, "p7 vision head_dim must be even", "head_dim");
    }
    if (channels != 3u) {
        return Status::error(StatusCode::invalid_argument, "p7 vision expects 3 channels", "channels");
    }
    if (mlp_act > 1u) {
        return Status::error(StatusCode::invalid_argument, "p7 vision mlp_act must be 0 (gelu-erf) or 1 (silu)", "mlp_act");
    }
    if (!std::isfinite(layer_norm_eps) || layer_norm_eps <= 0.0f) {
        return Status::error(StatusCode::invalid_argument, "p7 vision layer_norm_eps must be > 0", "layer_norm_eps");
    }
    if (rope_kind == P7VisionRopeKind::axial2d) {
        if (rope_height_pairs > head_dim / 2u) {
            return Status::error(StatusCode::invalid_argument, "rope_height_pairs exceeds head_dim/2", "rope_height_pairs");
        }
        if (!std::isfinite(rope_theta) || rope_theta <= 0.0f) {
            return Status::error(StatusCode::invalid_argument, "p7 vision rope_theta must be > 0", "rope_theta");
        }
    }
    if (mask_kind == P7VisionMaskKind::window_global) {
        if (window_patch_size == 0u) {
            return Status::error(StatusCode::invalid_argument, "window_global requires window_patch_size > 0", "window_patch_size");
        }
        for (auto l : global_layers) {
            if (l >= num_layers) {
                return Status::error(StatusCode::invalid_argument, "global layer index outside the vision tower", "global_layers");
            }
        }
    }
    // Additive Phase 8: decomposed relative-position bias (ViTDet/SAM).
    if (rel_pos_window != 0u) {
        if (rel_pos_window < 2u) {
            return Status::error(StatusCode::invalid_argument, "rel_pos_window must be >= 2 (or 0 to disable)", "rel_pos_window");
        }
        if (rope_kind != P7VisionRopeKind::none) {
            return Status::error(StatusCode::invalid_argument,
                                 "relative-position bias and vision RoPE are distinct conventions; enable only one", "rel_pos_window");
        }
    }
    return Status::success();
}

// ---------------------------------------------------------------------------
// Connector config
// ---------------------------------------------------------------------------

[[nodiscard]] Status P7ConnectorConfig::validate() const {
    if (intermediate_size == 0u) {
        return Status::error(StatusCode::invalid_argument, "connector intermediate_size must be nonzero", "intermediate_size");
    }
    if (!std::isfinite(layer_norm_eps) || layer_norm_eps <= 0.0f) {
        return Status::error(StatusCode::invalid_argument, "connector layer_norm_eps must be > 0", "layer_norm_eps");
    }
    switch (kind) {
        case P7ConnectorKind::spatial_mlp:
        case P7ConnectorKind::downsample_mlp:
            if (spatial_factor == 0u) {
                return Status::error(StatusCode::invalid_argument, "spatial/downsample connector requires spatial_factor > 0", "spatial_factor");
            }
            break;
        case P7ConnectorKind::unshuffle_mlp:
            if (spatial_factor != 2u) {
                return Status::error(StatusCode::invalid_argument, "the MinerU pixel-unshuffle connector is fixed 2x2", "spatial_factor");
            }
            break;
        case P7ConnectorKind::adaptive_mlp:
            if (max_visual_tokens == 0u) {
                return Status::error(StatusCode::invalid_argument, "adaptive connector requires max_visual_tokens > 0", "max_visual_tokens");
            }
            break;
        case P7ConnectorKind::mlp:
        default:
            break;
    }
    return Status::success();
}

// ---------------------------------------------------------------------------
// Packing
// ---------------------------------------------------------------------------

[[nodiscard]] Status P7PackingSpec::validate() const {
    if (patch_size == 0u) {
        return Status::error(StatusCode::invalid_argument, "packing patch_size must be nonzero", "patch_size");
    }
    if (max_pixels == 0u) {
        return Status::error(StatusCode::invalid_argument, "packing max_pixels must be nonzero", "max_pixels");
    }
    if (min_pixels > max_pixels) {
        return Status::error(StatusCode::invalid_argument, "packing min_pixels exceeds max_pixels", "min_pixels");
    }
    if (grid_multiple == 0u) {
        return Status::error(StatusCode::invalid_argument, "packing grid_multiple must be nonzero", "grid_multiple");
    }
    return Status::success();
}

namespace {

// Aspect-preserving grid reduction to a token budget (floor, >= 1 per axis).
[[nodiscard]] Result<std::pair<uint32_t, uint32_t>> reduce_grid(uint32_t gh, uint32_t gw, uint32_t max_tokens) {
    using R = Result<std::pair<uint32_t, uint32_t>>;
    if (gh == 0u || gw == 0u || max_tokens == 0u) {
        return R::failure(StatusCode::invalid_argument, "degenerate grid/token budget", "grid");
    }
    const uint64_t tokens = static_cast<uint64_t>(gh) * gw;
    if (tokens <= max_tokens) { return R::success(std::make_pair(gh, gw)); }
    const double scale = std::sqrt(static_cast<double>(max_tokens) / static_cast<double>(tokens));
    uint32_t oh = static_cast<uint32_t>(static_cast<double>(gh) * scale);
    uint32_t ow = static_cast<uint32_t>(static_cast<double>(gw) * scale);
    if (oh == 0u) { oh = 1u; }
    if (ow == 0u) { ow = 1u; }
    while (static_cast<uint64_t>(oh) * ow > max_tokens) {
        if (oh >= ow && oh > 1u) { oh--; }
        else if (ow > 1u) { ow--; }
        else { break; }
    }
    return R::success(std::make_pair(oh, ow));
}

}// namespace

[[nodiscard]] Result<P7PackedImage> p7_pack_image(uint32_t source_width, uint32_t source_height,
                                                  const P7PackingSpec &spec, const P7ConnectorConfig &connector) {
    using R = Result<P7PackedImage>;
    if (auto s = spec.validate(); !s.ok()) { return R::failure(s); }
    if (auto s = connector.validate(); !s.ok()) { return R::failure(s); }
    if (source_width == 0u || source_height == 0u) {
        return R::failure(StatusCode::invalid_argument, "degenerate source image size", "image");
    }
    // The patch grid must be a multiple of grid_multiple per axis (the
    // connector's spatial factor). Round the resized edge to patch*grid_multiple.
    const uint32_t factor = spec.patch_size * spec.grid_multiple;
    const uint64_t pixels = static_cast<uint64_t>(source_width) * source_height;
    double scale = 1.0;
    if (pixels > spec.max_pixels) {
        scale = std::sqrt(static_cast<double>(spec.max_pixels) / static_cast<double>(pixels));
    } else if (pixels < spec.min_pixels) {
        scale = std::sqrt(static_cast<double>(spec.min_pixels) / static_cast<double>(pixels));
    }
    const auto round_to = [factor](double v) -> uint32_t {
        const double units = v / static_cast<double>(factor);
        uint64_t n = static_cast<uint64_t>(units + 0.5);
        if (n == 0u) { n = 1u; }
        if (n > UINT32_MAX / factor) { n = UINT32_MAX / factor; }
        return static_cast<uint32_t>(n * factor);
    };
    uint32_t rw = round_to(static_cast<double>(source_width) * scale);
    uint32_t rh = round_to(static_cast<double>(source_height) * scale);
    const auto area = [](uint32_t w, uint32_t h) -> uint64_t { return static_cast<uint64_t>(w) * h; };
    if (area(rw, rh) > spec.max_pixels) {
        while (area(rw, rh) > spec.max_pixels) {
            if (rw >= rh && rw > factor) { rw -= factor; }
            else if (rh > factor) { rh -= factor; }
            else {
                return R::failure(StatusCode::invalid_argument,
                                  "image cannot fit the declared max_pixels budget at the patch grid", "max_pixels");
            }
        }
    }
    P7PackedImage packed;
    packed.source_width = source_width;
    packed.source_height = source_height;
    packed.resized_width = rw;
    packed.resized_height = rh;
    packed.grid_w = rw / spec.patch_size;
    packed.grid_h = rh / spec.patch_size;
    packed.patch_count = static_cast<uint64_t>(packed.grid_h) * packed.grid_w;
    // Connector output grid.
    switch (connector.kind) {
        case P7ConnectorKind::spatial_mlp:
        case P7ConnectorKind::unshuffle_mlp:
        case P7ConnectorKind::downsample_mlp: {
            const uint32_t m = connector.spatial_factor;
            packed.out_h = packed.grid_h / m;
            packed.out_w = packed.grid_w / m;
            break;
        }
        case P7ConnectorKind::adaptive_mlp: {
            auto reduced = reduce_grid(packed.grid_h, packed.grid_w, connector.max_visual_tokens);
            if (!reduced.ok()) { return R::failure(reduced.status); }
            packed.out_h = reduced.value->first;
            packed.out_w = reduced.value->second;
            break;
        }
        case P7ConnectorKind::mlp:
        default:
            packed.out_h = packed.grid_h;
            packed.out_w = packed.grid_w;
            break;
    }
    if (packed.out_h == 0u || packed.out_w == 0u) {
        return R::failure(StatusCode::invalid_argument, "connector output grid is degenerate", "connector");
    }
    packed.visual_tokens = static_cast<uint64_t>(packed.out_h) * packed.out_w;
    packed.scale_x = static_cast<float>(source_width) / static_cast<float>(rw);
    packed.scale_y = static_cast<float>(source_height) / static_cast<float>(rh);
    return R::success(std::move(packed));
}

[[nodiscard]] luisa::vector<int32_t> p7_vision_positions(const P7PackedImage &packed) {
    luisa::vector<int32_t> positions;
    positions.reserve(static_cast<size_t>(packed.patch_count) * 2u);
    for (uint32_t h = 0; h < packed.grid_h; h++) {
        for (uint32_t w = 0; w < packed.grid_w; w++) {
            positions.push_back(static_cast<int32_t>(h));
            positions.push_back(static_cast<int32_t>(w));
        }
    }
    return positions;
}

[[nodiscard]] luisa::vector<uint32_t> p7_vision_axis_of_pair(const P7VisionConfig &vision) {
    const uint32_t pairs = vision.head_dim / 2u;
    luisa::vector<uint32_t> axis(pairs, 1u);// width by default
    for (uint32_t d = 0; d < vision.rope_height_pairs && d < pairs; d++) { axis[d] = 0u; }// height
    return axis;
}

// ---------------------------------------------------------------------------
// Full slice config
// ---------------------------------------------------------------------------

[[nodiscard]] luisa::vector<uint32_t> P7SliceConfig::text_axis_of_pair() const {
    const uint32_t pairs = text.head_dim / 2u;
    luisa::vector<uint32_t> axis;
    axis.reserve(pairs);
    if (text_pos == P7TextPosKind::rope1d) {
        axis.assign(pairs, 0u);
        return axis;
    }
    // pos_sections: per-axis pair counts in axis order.
    uint32_t d = 0;
    for (uint32_t a = 0; a < pos_sections.size(); a++) {
        for (uint32_t i = 0; i < pos_sections[a] && d < pairs; i++, d++) {
            axis.push_back(a);
        }
    }
    // Any remaining pairs (should not happen after validate) map to axis 0.
    while (d < pairs) { axis.push_back(0u); d++; }
    return axis;
}

[[nodiscard]] Status P7SliceConfig::validate() const {
    if (auto s = vision.validate(); !s.ok()) { return s; }
    if (auto s = connector.validate(); !s.ok()) { return s; }
    if (auto s = text.validate(); !s.ok()) { return s; }
    if (auto s = packing.validate(); !s.ok()) { return s; }
    if (packing.patch_size != vision.patch_size) {
        return Status::error(StatusCode::invalid_argument, "packing patch_size must match the vision patch_size", "patch_size");
    }
    if (vision_component.empty() || decoder_component.empty() || vision_component == decoder_component) {
        return Status::error(StatusCode::invalid_argument, "weight component tags must be non-empty and distinct", "component");
    }
    if (mean.size() != 3u || std_dev.size() != 3u) {
        return Status::error(StatusCode::invalid_argument, "mean/std_dev must have 3 channels", "mean/std_dev");
    }
    for (auto v : mean) {
        if (!std::isfinite(v)) { return Status::error(StatusCode::invalid_argument, "mean must be finite", "mean"); }
    }
    for (auto v : std_dev) {
        if (!std::isfinite(v) || v == 0.0f) { return Status::error(StatusCode::invalid_argument, "std_dev must be finite and nonzero", "std_dev"); }
    }
    if (max_vision_patches == 0u || prefill_chunk_rows == 0u) {
        return Status::error(StatusCode::invalid_argument, "max_vision_patches/prefill_chunk_rows must be nonzero", "budget");
    }
    // Grid multiple must match the connector's spatial factor.
    uint32_t want_multiple = 1u;
    if (connector.kind == P7ConnectorKind::spatial_mlp || connector.kind == P7ConnectorKind::unshuffle_mlp ||
        connector.kind == P7ConnectorKind::downsample_mlp) {
        want_multiple = connector.spatial_factor;
    }
    if (packing.grid_multiple != want_multiple) {
        return Status::error(StatusCode::invalid_argument, "packing grid_multiple must match the connector spatial factor", "grid_multiple");
    }
    // Text position convention.
    const uint32_t pairs = text.head_dim / 2u;
    switch (text_pos) {
        case P7TextPosKind::rope1d:
            if (text.rope_pair_layout != RopePairLayout::adjacent && text.rope_pair_layout != RopePairLayout::rotate_half) {
                return Status::error(StatusCode::invalid_argument, "unknown text rope pair layout", "rope_pair_layout");
            }
            break;
        case P7TextPosKind::mrope_adjacent:
            if (text.rope_pair_layout != RopePairLayout::adjacent) {
                return Status::error(StatusCode::invalid_argument, "adjacent M-RoPE requires the adjacent pair family", "rope_pair_layout");
            }
            if (pos_sections.size() != 3u) {
                return Status::error(StatusCode::invalid_argument, "M-RoPE requires 3 position sections [t,h,w]", "pos_sections");
            }
            break;
        case P7TextPosKind::mrope_interleaved:
            if (text.rope_pair_layout != RopePairLayout::rotate_half) {
                return Status::error(StatusCode::invalid_argument, "interleaved M-RoPE requires the rotate-half pair family", "rope_pair_layout");
            }
            if (pos_sections.size() != 3u) {
                return Status::error(StatusCode::invalid_argument, "M-RoPE requires 3 position sections [t,h,w]", "pos_sections");
            }
            break;
        case P7TextPosKind::xd:
            if (text.rope_pair_layout != RopePairLayout::adjacent) {
                return Status::error(StatusCode::invalid_argument, "XD-RoPE requires the adjacent pair family", "rope_pair_layout");
            }
            if (pos_sections.size() != 4u) {
                return Status::error(StatusCode::invalid_argument, "XD-RoPE requires 4 position sections [text,h,w,time]", "pos_sections");
            }
            break;
        default:
            return Status::error(StatusCode::invalid_argument, "unknown text position convention", "text_pos");
    }
    if (text_pos != P7TextPosKind::rope1d) {
        uint32_t sum = 0;
        for (auto s : pos_sections) { sum += s; }
        if (sum != pairs) {
            return Status::error(StatusCode::invalid_argument, "position sections must sum to head_dim/2", "pos_sections");
        }
        if (partial_rotary_num != partial_rotary_den) {
            return Status::error(StatusCode::invalid_argument, "partial rotary is a 1-axis text convention", "partial_rotary");
        }
    }
    // Additive Phase 8: the explicit Qwen2.5-VL text contract requires
    // rotate-half M-RoPE (mrope_interleaved). It is never combined with the
    // Qwen3 QK-Norm contract or with adjacent/1D positions.
    if (text_qwen2_5_contract && text_pos != P7TextPosKind::mrope_interleaved) {
        return Status::error(StatusCode::invalid_argument,
                             "the Qwen2.5-VL text contract requires rotate-half M-RoPE (mrope_interleaved)", "text_qwen2_5_contract");
    }
    // Additive Phase 8: per-linear storage policy.
    if (linear_storage != DenseLinearStorage::fp32) {
        if (linear_storage != DenseLinearStorage::int4_sym && linear_storage != DenseLinearStorage::int4_asym) {
            return Status::error(StatusCode::invalid_argument, "unknown slice linear storage", "linear_storage");
        }
        if (!is_supported_group_size(linear_group_size)) {
            return Status::error(StatusCode::invalid_argument, "W4A32 slice requires group size 64 or 128", "linear_group_size");
        }
        if (tie_word_embeddings) {
            return Status::error(StatusCode::unsupported,
                                 "the W4A32 slice requires a standalone packed lm_head; tied embeddings stay FP32",
                                 "tie_word_embeddings");
        }
    } else if (linear_group_size != 0u) {
        return Status::error(StatusCode::invalid_argument, "linear_group_size must be 0 for fp32", "linear_group_size");
    }
    return Status::success();
}

// ---------------------------------------------------------------------------
// Position ids (explicit per-convention host math)
// ---------------------------------------------------------------------------

[[nodiscard]] Result<P7PositionIds> p7_build_position_ids(
    const P7SliceConfig &config, uint32_t prompt_len, uint32_t mm_start, uint32_t mm_count,
    const P7PackedImage &packed) {
    using R = Result<P7PositionIds>;
    const uint32_t axes = config.text_rope_axes();
    P7PositionIds out;
    out.seq_len = prompt_len;
    if (prompt_len == 0u) {
        return R::failure(StatusCode::invalid_argument, "empty prompt", "prompt_len");
    }
    if (static_cast<uint64_t>(mm_start) + mm_count > prompt_len) {
        return R::failure(StatusCode::invalid_argument, "image range exceeds the prompt", "mm");
    }
    if (mm_count != 0u && mm_count != packed.visual_tokens) {
        return R::failure(StatusCode::invalid_argument, "image row count does not match the packed visual tokens", "mm_count");
    }
    out.ids.reserve(static_cast<size_t>(prompt_len) * axes);

    if (config.text_pos == P7TextPosKind::rope1d) {
        // Plain sequential 1D positions (image rows consume positions too).
        for (uint32_t i = 0; i < prompt_len; i++) { out.ids.push_back(static_cast<int32_t>(i)); }
        out.next_position = {static_cast<int32_t>(prompt_len)};
        return R::success(std::move(out));
    }

    // Multimodal conventions. Text rows advance a running scalar position on
    // every axis; image rows lay out the connector output grid.
    const bool xd = config.text_pos == P7TextPosKind::xd;
    int32_t pos = 0;
    uint32_t row = 0;
    while (row < prompt_len) {
        if (mm_count != 0u && row == mm_start) {
            const int32_t start = pos;
            const uint32_t oh = packed.out_h, ow = packed.out_w;
            for (uint32_t hi = 0; hi < oh; hi++) {
                for (uint32_t wi = 0; wi < ow; wi++) {
                    if (xd) {
                        // XD: (text, height, width, time). The still image is
                        // anchored at time = start (single frame).
                        out.ids.push_back(start);
                        out.ids.push_back(start + static_cast<int32_t>(hi));
                        out.ids.push_back(start + static_cast<int32_t>(wi));
                        out.ids.push_back(start);
                    } else {
                        // M-RoPE: (temporal, height, width); still image t=0.
                        out.ids.push_back(start);
                        out.ids.push_back(start + static_cast<int32_t>(hi));
                        out.ids.push_back(start + static_cast<int32_t>(wi));
                    }
                }
            }
            const uint32_t extent = std::max({1u, oh, ow});
            pos = start + static_cast<int32_t>(extent);
            row += mm_count;
        } else {
            for (uint32_t a = 0; a < axes; a++) { out.ids.push_back(pos); }
            pos++;
            row++;
        }
    }
    out.next_position.assign(axes, pos);
    return R::success(std::move(out));
}


// ---------------------------------------------------------------------------
// Shared slice footprint plan (additive Phase 8 review; host math, no alloc)
// ---------------------------------------------------------------------------

namespace {

constexpr uint64_t kP7FootprintAlignment = 256u;// == workspace.h kWorkspaceAlignment

[[nodiscard]] Result<uint64_t> fp_f32_bytes(uint64_t count) {
    return tensor_byte_size(TensorDType::f32, count);
}

[[nodiscard]] Result<uint64_t> fp_align_add(uint64_t total, uint64_t bytes) {
    using R = Result<uint64_t>;
    uint64_t aligned = 0, next = 0;
    if (!checked_align_up(bytes, kP7FootprintAlignment, aligned)) {
        return R::failure(StatusCode::out_of_memory, "footprint alignment overflow", "workspace");
    }
    if (!checked_add(total, aligned, next)) {
        return R::failure(StatusCode::out_of_memory, "footprint byte total overflows", "workspace");
    }
    return R::success(next);
}

[[nodiscard]] Result<uint64_t> fp_vision_weights(const P7VisionConfig &v, const P7ConnectorConfig &c,
                                                 uint32_t text_hidden) {
    using R = Result<uint64_t>;
    uint64_t total = 0;
    const auto fp32 = [&total](uint64_t elems) -> Status {
        auto b = fp_f32_bytes(elems);
        if (!b.ok()) { return b.status; }
        uint64_t next = 0;
        if (!checked_add(total, *b.value, next)) {
            return Status::error(StatusCode::out_of_memory, "vision weight total overflows", "weights");
        }
        total = next;
        return Status::success();
    };
    const uint64_t vh = v.hidden_size, vi = v.intermediate_size;
    if (auto s = fp32(static_cast<uint64_t>(v.patch_dim()) * vh); !s.ok()) { return R::failure(s); }
    if (auto s = fp32(vh); !s.ok()) { return R::failure(s); }
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
        for (int i = 0; i < 4; i++) {
            if (auto s = fp32(vh); !s.ok()) { return R::failure(s); }// norm1/2 w+b
        }
        for (int i = 0; i < 4; i++) {
            if (auto s = fp32(static_cast<uint64_t>(vh) * vh); !s.ok()) { return R::failure(s); }
        }
        if (v.qkv_bias) {
            for (int i = 0; i < 4; i++) {
                if (auto s = fp32(vh); !s.ok()) { return R::failure(s); }
            }
        }
        if (v.qk_norm) {
            if (auto s = fp32(v.head_dim); !s.ok()) { return R::failure(s); }
            if (auto s = fp32(v.head_dim); !s.ok()) { return R::failure(s); }
        }
        if (auto s = fp32(static_cast<uint64_t>(vh) * vi); !s.ok()) { return R::failure(s); }
        if (auto s = fp32(vi); !s.ok()) { return R::failure(s); }
        if (auto s = fp32(static_cast<uint64_t>(vi) * vh); !s.ok()) { return R::failure(s); }
        if (auto s = fp32(vh); !s.ok()) { return R::failure(s); }
    }
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
    if (auto s = fp32(cin); !s.ok()) { return R::failure(s); }
    if (!c.rms_norm) {
        if (auto s = fp32(cin); !s.ok()) { return R::failure(s); }
    }
    if (auto s = fp32(cin * c.intermediate_size); !s.ok()) { return R::failure(s); }
    if (auto s = fp32(c.intermediate_size); !s.ok()) { return R::failure(s); }
    if (auto s = fp32(static_cast<uint64_t>(c.intermediate_size) * text_hidden); !s.ok()) { return R::failure(s); }
    if (auto s = fp32(text_hidden); !s.ok()) { return R::failure(s); }
    return R::success(total);
}

[[nodiscard]] Result<uint64_t> fp_decoder_weights(const DenseDecoderConfig &d, bool tied,
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
        auto b = fp_f32_bytes(elems);
        if (!b.ok()) { return b.status; }
        return add(*b.value);
    };
    const auto linear = [&](uint64_t k, uint64_t n) -> Status {
        uint64_t elems = 0;
        if (!checked_mul(k, n, elems)) {
            return Status::error(StatusCode::out_of_memory, "linear element count overflows", "weights");
        }
        if (storage == DenseLinearStorage::fp32) { return fp32(elems); }
        const uint64_t words = (elems + 7u) / 8u;
        auto wb = tensor_byte_size(TensorDType::u32, words);
        if (!wb.ok()) { return wb.status; }
        if (auto s = add(*wb.value); !s.ok()) { return s; }
        uint64_t groups = 0;
        if (!group_count_for(elems, group_size, groups)) {
            return Status::error(StatusCode::invalid_argument, "group count overflow", "weights");
        }
        auto sb = fp_f32_bytes(groups);
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
    if (auto s = fp32(static_cast<uint64_t>(d.vocab_size) * hidden); !s.ok()) { return R::failure(s); }
    if (auto s = fp32(hidden); !s.ok()) { return R::failure(s); }
    if (!tied) {
        if (auto s = linear(hidden, d.vocab_size); !s.ok()) { return R::failure(s); }
    } else {
        // Tied head: executor-owned FP32 transpose of the embedding table.
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

}// namespace

[[nodiscard]] Result<P7SliceFootprint> p7_plan_slice_footprint(const P7SliceConfig &config, uint32_t max_seq_len) {
    using R = Result<P7SliceFootprint>;
    if (auto s = config.validate(); !s.ok()) { return R::failure(s); }
    if (max_seq_len == 0u) {
        return R::failure(StatusCode::invalid_argument, "max_seq_len must be nonzero", "max_seq_len");
    }
    const auto &v = config.vision;
    const auto &c = config.connector;
    const auto &d = config.text;
    P7SliceFootprint out;
    auto vision = fp_vision_weights(v, c, d.hidden_size);
    if (!vision.ok()) { return R::failure(vision.status); }
    out.vision_weight_bytes = *vision.value;
    auto decoder = fp_decoder_weights(d, config.tie_word_embeddings, config.linear_storage, config.linear_group_size);
    if (!decoder.ok()) { return R::failure(decoder.status); }
    out.decoder_weight_bytes = *decoder.value;
    // Combined workspace: vision arena then decoder arena (region order and
    // alignment mirror P7VlmExecutor::reserve_workspace + DenseVlmExecutor).
    const uint64_t maxp = config.max_vision_patches;
    const uint64_t vh = v.hidden_size, vi = v.intermediate_size;
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
    const uint64_t vision_elems[] = {
        maxp * v.patch_dim(), maxp * v.patch_dim(), maxp * vh, maxp * vh, maxp * vh, maxp * vh,
        maxp * vh, maxp * vh, maxp * vh, maxp * vh, maxp * vh, maxp * vh, maxp * vi,
        max_tokens * reduced_dim, max_tokens * reduced_dim, max_tokens * c.intermediate_size,
        max_tokens * merged_dim, maxp * d.hidden_size, maxp * 2u, v.head_dim / 2u,
        mask_words_bound, 1, 1, vh + maxp * vh, vh * maxp, rel_bias_elems,
    };
    uint64_t total_ws = 0;
    for (auto e : vision_elems) {
        auto b = fp_f32_bytes(e);
        if (!b.ok()) { return R::failure(b.status); }
        auto next = fp_align_add(total_ws, *b.value);
        if (!next.ok()) { return R::failure(next.status); }
        total_ws = *next.value;
    }
    const uint64_t chunk = config.prefill_chunk_rows;
    const uint64_t hidden = d.hidden_size, qd = d.q_dim(), kvd = d.kv_dim(), inter = d.intermediate_size;
    const uint32_t axes = config.text_rope_axes();
    const bool qwen3_text = config.text_pos == P7TextPosKind::mrope_interleaved && !config.text_qwen2_5_contract;
    const uint64_t qk_norm_q = qwen3_text ? chunk * qd : 1ull;
    const uint64_t qk_norm_k = qwen3_text ? chunk * kvd : 1ull;
    const uint64_t decoder_elems[] = {
        chunk * hidden, chunk * hidden, chunk * hidden, chunk * hidden, chunk * qd, chunk * kvd,
        chunk * kvd, chunk * qd, chunk * kvd, chunk * qd, chunk * inter, chunk * inter, chunk * inter,
        d.vocab_size, chunk, chunk * axes, d.head_dim / 2u, 1, 1, qk_norm_q, qk_norm_k, 1,
    };
    for (auto e : decoder_elems) {
        auto b = fp_f32_bytes(e);
        if (!b.ok()) { return R::failure(b.status); }
        auto next = fp_align_add(total_ws, *b.value);
        if (!next.ok()) { return R::failure(next.status); }
        total_ws = *next.value;
    }
    out.workspace_bytes = total_ws;
    auto kv = plan_kv_cache_bytes(d.num_layers, 1u, max_seq_len, d.num_kv_heads, d.head_dim, 4u);
    if (!kv.ok()) { return R::failure(kv.status); }
    out.kv_cache_bytes = *kv.value + 16u;// DecodeState metadata (batch 1)
    uint64_t total = 0;
    if (!checked_add(out.vision_weight_bytes, out.decoder_weight_bytes, total) ||
        !checked_add(total, out.workspace_bytes, total) ||
        !checked_add(total, out.kv_cache_bytes, total)) {
        return R::failure(StatusCode::out_of_memory, "total device byte requirement overflows", "total");
    }
    out.total_device_bytes = total;
    return R::success(std::move(out));
}

// ---------------------------------------------------------------------------
// Tensor names
// ---------------------------------------------------------------------------

[[nodiscard]] Status P7TensorNames::validate() const {
    if (auto s = text.validate(); !s.ok()) { return s; }
    if (visual_prefix.empty() || visual_prefix.back() != '.') {
        return Status::error(StatusCode::invalid_argument, "visual_prefix must be non-empty and end with '.'", "visual_prefix");
    }
    return Status::success();
}

[[nodiscard]] luisa::string P7TensorNames::patch_embed_weight() const { return visual_prefix + "patch_embed.weight"; }
[[nodiscard]] luisa::string P7TensorNames::patch_embed_bias() const { return visual_prefix + "patch_embed.bias"; }
[[nodiscard]] luisa::string P7TensorNames::position_embed() const { return visual_prefix + "position_embed.weight"; }
[[nodiscard]] luisa::string P7TensorNames::block_rel_pos_bias(uint32_t l) const {
    return block_prefix(l) + "attn.rel_pos_bias";
}
[[nodiscard]] luisa::string P7TensorNames::block_prefix(uint32_t layer) const {
    return visual_prefix + "blocks." + luisa::format("{}", layer) + ".";
}
[[nodiscard]] luisa::string P7TensorNames::block_norm1_weight(uint32_t l) const { return block_prefix(l) + "norm1.weight"; }
[[nodiscard]] luisa::string P7TensorNames::block_norm1_bias(uint32_t l) const { return block_prefix(l) + "norm1.bias"; }
[[nodiscard]] luisa::string P7TensorNames::block_norm2_weight(uint32_t l) const { return block_prefix(l) + "norm2.weight"; }
[[nodiscard]] luisa::string P7TensorNames::block_norm2_bias(uint32_t l) const { return block_prefix(l) + "norm2.bias"; }
[[nodiscard]] luisa::string P7TensorNames::block_q_proj(uint32_t l) const { return block_prefix(l) + "attn.q_proj.weight"; }
[[nodiscard]] luisa::string P7TensorNames::block_k_proj(uint32_t l) const { return block_prefix(l) + "attn.k_proj.weight"; }
[[nodiscard]] luisa::string P7TensorNames::block_v_proj(uint32_t l) const { return block_prefix(l) + "attn.v_proj.weight"; }
[[nodiscard]] luisa::string P7TensorNames::block_o_proj(uint32_t l) const { return block_prefix(l) + "attn.o_proj.weight"; }
[[nodiscard]] luisa::string P7TensorNames::block_q_bias(uint32_t l) const { return block_prefix(l) + "attn.q_proj.bias"; }
[[nodiscard]] luisa::string P7TensorNames::block_k_bias(uint32_t l) const { return block_prefix(l) + "attn.k_proj.bias"; }
[[nodiscard]] luisa::string P7TensorNames::block_v_bias(uint32_t l) const { return block_prefix(l) + "attn.v_proj.bias"; }
[[nodiscard]] luisa::string P7TensorNames::block_o_bias(uint32_t l) const { return block_prefix(l) + "attn.o_proj.bias"; }
[[nodiscard]] luisa::string P7TensorNames::block_q_norm(uint32_t l) const { return block_prefix(l) + "attn.q_norm.weight"; }
[[nodiscard]] luisa::string P7TensorNames::block_k_norm(uint32_t l) const { return block_prefix(l) + "attn.k_norm.weight"; }
[[nodiscard]] luisa::string P7TensorNames::block_fc1(uint32_t l) const { return block_prefix(l) + "mlp.fc1.weight"; }
[[nodiscard]] luisa::string P7TensorNames::block_fc1_bias(uint32_t l) const { return block_prefix(l) + "mlp.fc1.bias"; }
[[nodiscard]] luisa::string P7TensorNames::block_fc2(uint32_t l) const { return block_prefix(l) + "mlp.fc2.weight"; }
[[nodiscard]] luisa::string P7TensorNames::block_fc2_bias(uint32_t l) const { return block_prefix(l) + "mlp.fc2.bias"; }
[[nodiscard]] luisa::string P7TensorNames::connector_prefix() const { return visual_prefix + "connector."; }
[[nodiscard]] luisa::string P7TensorNames::connector_ln_weight() const { return connector_prefix() + "ln.weight"; }
[[nodiscard]] luisa::string P7TensorNames::connector_ln_bias() const { return connector_prefix() + "ln.bias"; }
[[nodiscard]] luisa::string P7TensorNames::connector_fc1() const { return connector_prefix() + "fc1.weight"; }
[[nodiscard]] luisa::string P7TensorNames::connector_fc1_bias() const { return connector_prefix() + "fc1.bias"; }
[[nodiscard]] luisa::string P7TensorNames::connector_fc2() const { return connector_prefix() + "fc2.weight"; }
[[nodiscard]] luisa::string P7TensorNames::connector_fc2_bias() const { return connector_prefix() + "fc2.bias"; }
[[nodiscard]] luisa::string P7TensorNames::connector_down_weight() const { return connector_prefix() + "down.weight"; }
[[nodiscard]] luisa::string P7TensorNames::connector_down_bias() const { return connector_prefix() + "down.bias"; }

// ---------------------------------------------------------------------------
// Expected tensor inventory
// ---------------------------------------------------------------------------

[[nodiscard]] luisa::vector<ExpectedTensor> p7_expected_tensors(const P7SliceConfig &config,
                                                               const P7TensorNames &names) {
    luisa::vector<ExpectedTensor> out;
    const auto add = [&out](luisa::string name, luisa::vector<uint64_t> shape,
                            TensorLayout layout, bool required = true) {
        out.emplace_back(ExpectedTensor{std::move(name), std::move(shape), layout, required});
    };
    const auto &v = config.vision;
    const uint64_t vh = v.hidden_size, vi = v.intermediate_size;
    add(names.patch_embed_weight(), {v.patch_dim(), vh}, TensorLayout::kn);
    add(names.patch_embed_bias(), {vh}, TensorLayout::row_major);
    if (v.position_rows != 0u) {
        add(names.position_embed(), {v.position_rows, vh}, TensorLayout::row_major);
    }
    // Additive Phase 8: per-layer decomposed relative-position bias table
    // [num_heads, 2*window-1, 2*window-1] (flattened row-major).
    if (v.rel_pos_window != 0u) {
        const uint64_t side = 2ull * v.rel_pos_window - 1ull;
        for (uint32_t l = 0; l < v.num_layers; l++) {
            add(names.block_rel_pos_bias(l), {v.num_heads, side, side}, TensorLayout::row_major);
        }
    }
    for (uint32_t l = 0; l < v.num_layers; l++) {
        add(names.block_norm1_weight(l), {vh}, TensorLayout::row_major);
        add(names.block_norm1_bias(l), {vh}, TensorLayout::row_major);
        add(names.block_norm2_weight(l), {vh}, TensorLayout::row_major);
        add(names.block_norm2_bias(l), {vh}, TensorLayout::row_major);
        add(names.block_q_proj(l), {vh, vh}, TensorLayout::kn);
        add(names.block_k_proj(l), {vh, vh}, TensorLayout::kn);
        add(names.block_v_proj(l), {vh, vh}, TensorLayout::kn);
        add(names.block_o_proj(l), {vh, vh}, TensorLayout::kn);
        if (v.qkv_bias) {
            add(names.block_q_bias(l), {vh}, TensorLayout::row_major);
            add(names.block_k_bias(l), {vh}, TensorLayout::row_major);
            add(names.block_v_bias(l), {vh}, TensorLayout::row_major);
            add(names.block_o_bias(l), {vh}, TensorLayout::row_major);
        }
        if (v.qk_norm) {
            add(names.block_q_norm(l), {v.head_dim}, TensorLayout::row_major);
            add(names.block_k_norm(l), {v.head_dim}, TensorLayout::row_major);
        }
        add(names.block_fc1(l), {vh, vi}, TensorLayout::kn);
        add(names.block_fc1_bias(l), {vi}, TensorLayout::row_major);
        add(names.block_fc2(l), {vi, vh}, TensorLayout::kn);
        add(names.block_fc2_bias(l), {vh}, TensorLayout::row_major);
    }
    // Connector. The LayerNorm/MLP input dim depends on the reduction.
    const auto &c = config.connector;
    uint64_t cin = vh;
    if (c.kind == P7ConnectorKind::spatial_mlp || c.kind == P7ConnectorKind::downsample_mlp) {
        cin = vh * c.spatial_factor * c.spatial_factor;
    } else if (c.kind == P7ConnectorKind::unshuffle_mlp) {
        cin = vh * 4u;
    }
    if (c.kind == P7ConnectorKind::downsample_mlp) {
        // Learned downsample projection: [vh*m*m, vh] applied before the LN.
        add(names.connector_down_weight(), {vh * c.spatial_factor * c.spatial_factor, vh}, TensorLayout::kn);
        add(names.connector_down_bias(), {vh}, TensorLayout::row_major);
        cin = vh;
    }
    add(names.connector_ln_weight(), {cin}, TensorLayout::row_major);
    // The RMSNorm merger (Qwen2.5-VL / olmOCR-2) has no bias; LayerNorm does.
    if (!c.rms_norm) {
        add(names.connector_ln_bias(), {cin}, TensorLayout::row_major);
    }
    add(names.connector_fc1(), {cin, c.intermediate_size}, TensorLayout::kn);
    add(names.connector_fc1_bias(), {c.intermediate_size}, TensorLayout::row_major);
    add(names.connector_fc2(), {c.intermediate_size, config.text.hidden_size}, TensorLayout::kn);
    add(names.connector_fc2_bias(), {config.text.hidden_size}, TensorLayout::row_major);
    // Text decoder inventory (shared contract; QK-Norm vs QKV-bias selected).
    auto text_expected = dense_decoder_expected_tensors(config.text, names.text, config.tie_word_embeddings);
    const bool qwen3_text = config.text_pos == P7TextPosKind::mrope_interleaved && !config.text_qwen2_5_contract;
    if (qwen3_text) {
        luisa::vector<ExpectedTensor> filtered;
        for (auto &e : text_expected) {
            bool is_qkv_bias = false;
            for (uint32_t l = 0; l < config.text.num_layers && !is_qkv_bias; l++) {
                is_qkv_bias = e.name == names.text.q_bias(l) || e.name == names.text.k_bias(l) || e.name == names.text.v_bias(l);
            }
            if (!is_qkv_bias) { filtered.push_back(std::move(e)); }
        }
        for (uint32_t l = 0; l < config.text.num_layers; l++) {
            filtered.emplace_back(ExpectedTensor{names.text.q_norm(l), {config.text.head_dim}, TensorLayout::row_major, true});
            filtered.emplace_back(ExpectedTensor{names.text.k_norm(l), {config.text.head_dim}, TensorLayout::row_major, true});
        }
        text_expected = std::move(filtered);
    }
    for (auto &e : text_expected) { out.push_back(std::move(e)); }
    return out;
}

[[nodiscard]] Status p7_validate_tensors(luisa::span<const PackageTensor> tensors,
                                         luisa::span<const ExpectedTensor> expected) {
    return validate_dense_decoder_tensors(tensors, expected);
}


}// namespace luisa::ocr

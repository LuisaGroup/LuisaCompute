// OCR engine -- PaddleOCR-VL-1.6 contract (Phase 6). Device-safe TU.
// Config/packing/tensor-inventory/gate only; the device graph lives in
// runtime/layout_vlm_executor.* and host composition in paddle_ocr_vl_result.*.
#include "ocr/models/paddle_ocr_vl.h"

#include <algorithm>
#include <cmath>
#include <utility>

#include <luisa/core/stl/format.h>

namespace luisa::ocr {

// ---------------------------------------------------------------------------
// NaViT vision config
// ---------------------------------------------------------------------------

[[nodiscard]] Status PaddleNavitConfig::validate() const {
    const auto nonzero = [](uint32_t v, const char *what) -> Status {
        if (v == 0u) {
            return Status::error(StatusCode::invalid_argument,
                                 luisa::format("paddle vision field '{}' must be nonzero", what), what);
        }
        return Status::success();
    };
    for (const auto &[v, name] : std::initializer_list<std::pair<uint32_t, const char *>>{
             {num_layers, "num_layers"}, {hidden_size, "hidden_size"}, {num_heads, "num_heads"},
             {head_dim, "head_dim"}, {intermediate_size, "intermediate_size"}, {patch_size, "patch_size"}}) {
        if (auto s = nonzero(v, name); !s.ok()) { return s; }
    }
    if (static_cast<uint64_t>(num_heads) * head_dim != hidden_size) {
        return Status::error(StatusCode::invalid_argument, "paddle vision heads*head_dim must equal hidden_size", "head_dim");
    }
    if (head_dim % 2u != 0u) {
        return Status::error(StatusCode::invalid_argument, "paddle vision head_dim must be even", "head_dim");
    }
    if (channels != 3u) {
        return Status::error(StatusCode::invalid_argument, "paddle vision expects 3 channels", "channels");
    }
    if (rope_height_pairs > head_dim / 2u) {
        return Status::error(StatusCode::invalid_argument, "rope_height_pairs exceeds head_dim/2", "rope_height_pairs");
    }
    if (!std::isfinite(layer_norm_eps) || layer_norm_eps <= 0.0f) {
        return Status::error(StatusCode::invalid_argument, "layer_norm_eps must be > 0", "layer_norm_eps");
    }
    if (!std::isfinite(rope_theta) || rope_theta <= 0.0f) {
        return Status::error(StatusCode::invalid_argument, "rope_theta must be > 0", "rope_theta");
    }
    return Status::success();
}

// ---------------------------------------------------------------------------
// Connector config + grid reduction
// ---------------------------------------------------------------------------

[[nodiscard]] Status PaddleConnectorConfig::validate() const {
    if (max_visual_tokens == 0u) {
        return Status::error(StatusCode::invalid_argument, "max_visual_tokens must be nonzero", "max_visual_tokens");
    }
    if (connector_intermediate == 0u) {
        return Status::error(StatusCode::invalid_argument, "connector_intermediate must be nonzero", "connector_intermediate");
    }
    if (!std::isfinite(layer_norm_eps) || layer_norm_eps <= 0.0f) {
        return Status::error(StatusCode::invalid_argument, "connector layer_norm_eps must be > 0", "layer_norm_eps");
    }
    return Status::success();
}

[[nodiscard]] Result<std::pair<uint32_t, uint32_t>> paddle_reduce_grid(
    uint32_t grid_h, uint32_t grid_w, uint32_t max_visual_tokens) {
    using R = Result<std::pair<uint32_t, uint32_t>>;
    if (grid_h == 0u || grid_w == 0u || max_visual_tokens == 0u) {
        return R::failure(StatusCode::invalid_argument, "degenerate grid/token budget", "grid");
    }
    const uint64_t tokens = static_cast<uint64_t>(grid_h) * grid_w;
    if (tokens <= max_visual_tokens) {
        return R::success(std::make_pair(grid_h, grid_w));
    }
    // Aspect-preserving scale, then floor each axis to >= 1.
    const double scale = std::sqrt(static_cast<double>(max_visual_tokens) / static_cast<double>(tokens));
    uint32_t out_h = static_cast<uint32_t>(static_cast<double>(grid_h) * scale);
    uint32_t out_w = static_cast<uint32_t>(static_cast<double>(grid_w) * scale);
    if (out_h == 0u) { out_h = 1u; }
    if (out_w == 0u) { out_w = 1u; }
    // Flooring can still leave out_h*out_w slightly over budget for extreme
    // aspect ratios; shrink the longer axis until it fits (never below 1).
    while (static_cast<uint64_t>(out_h) * out_w > max_visual_tokens) {
        if (out_h >= out_w && out_h > 1u) { out_h--; }
        else if (out_w > 1u) { out_w--; }
        else { break; }// 1x1 still over budget only when max_visual_tokens==0 (rejected)
    }
    return R::success(std::make_pair(out_h, out_w));
}

// ---------------------------------------------------------------------------
// Packing
// ---------------------------------------------------------------------------

[[nodiscard]] Status PaddlePackingSpec::validate() const {
    if (patch_size == 0u) {
        return Status::error(StatusCode::invalid_argument, "patch_size must be nonzero", "patch_size");
    }
    if (max_pixels == 0u) {
        return Status::error(StatusCode::invalid_argument, "max_pixels must be nonzero", "max_pixels");
    }
    if (min_pixels > max_pixels) {
        return Status::error(StatusCode::invalid_argument, "min_pixels exceeds max_pixels", "min_pixels");
    }
    return Status::success();
}

[[nodiscard]] Result<PaddlePackedImage> paddle_pack_image(uint32_t source_width, uint32_t source_height,
                                                          const PaddlePackingSpec &spec, uint32_t max_visual_tokens) {
    using R = Result<PaddlePackedImage>;
    if (auto s = spec.validate(); !s.ok()) { return R::failure(s); }
    if (source_width == 0u || source_height == 0u) {
        return R::failure(StatusCode::invalid_argument, "degenerate source image size", "image");
    }
    const uint32_t factor = spec.patch_size;
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
    // Rounding to the patch grid can push the area back over the cap; shrink
    // the longer edge by one factor step until it fits (never below one factor).
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
    PaddlePackedImage packed;
    packed.source_width = source_width;
    packed.source_height = source_height;
    packed.resized_width = rw;
    packed.resized_height = rh;
    packed.grid_w = rw / spec.patch_size;
    packed.grid_h = rh / spec.patch_size;
    packed.patch_count = static_cast<uint64_t>(packed.grid_h) * packed.grid_w;
    auto reduced = paddle_reduce_grid(packed.grid_h, packed.grid_w, max_visual_tokens);
    if (!reduced.ok()) { return R::failure(reduced.status); }
    packed.reduced_h = reduced.value->first;
    packed.reduced_w = reduced.value->second;
    packed.visual_tokens = static_cast<uint64_t>(packed.reduced_h) * packed.reduced_w;
    packed.scale_x = static_cast<float>(source_width) / static_cast<float>(rw);
    packed.scale_y = static_cast<float>(source_height) / static_cast<float>(rh);
    return R::success(std::move(packed));
}

[[nodiscard]] luisa::vector<int32_t> paddle_vision_positions(const PaddlePackedImage &packed) {
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

[[nodiscard]] luisa::vector<uint32_t> paddle_vision_axis_of_pair(const PaddleNavitConfig &vision) {
    const uint32_t pairs = vision.head_dim / 2u;
    luisa::vector<uint32_t> axis(pairs, 1u);// width by default
    for (uint32_t d = 0; d < vision.rope_height_pairs && d < pairs; d++) { axis[d] = 0u; }// height
    return axis;
}

// ---------------------------------------------------------------------------
// Full config
// ---------------------------------------------------------------------------

[[nodiscard]] Status PaddleOcrVlConfig::validate() const {
    if (auto s = vision.validate(); !s.ok()) { return s; }
    if (auto s = connector.validate(); !s.ok()) { return s; }
    if (auto s = text.validate(); !s.ok()) { return s; }
    if (auto s = packing.validate(); !s.ok()) { return s; }
    if (packing.patch_size != vision.patch_size) {
        return Status::error(StatusCode::invalid_argument, "packing patch_size must match the vision patch_size", "patch_size");
    }
    // ERNIE contract: rotate-half 1D RoPE (the shared Phase 3 dense convention).
    if (text.rope_pair_layout != RopePairLayout::rotate_half) {
        return Status::error(StatusCode::invalid_argument,
                             "the ERNIE decoder contract requires rotate-half RoPE", "rope_pair_layout");
    }
    return Status::success();
}

[[nodiscard]] PaddleOcrVlConfig paddle_ocr_vl_synthetic_config() {
    PaddleOcrVlConfig cfg;
    // NaViT vision: 2 layers, hidden 16, 2 heads, head_dim 8, inter 32, patch 2.
    cfg.vision.num_layers = 2;
    cfg.vision.hidden_size = 16;
    cfg.vision.num_heads = 2;
    cfg.vision.head_dim = 8;
    cfg.vision.intermediate_size = 32;
    cfg.vision.patch_size = 2;
    cfg.vision.channels = 3;
    cfg.vision.layer_norm_eps = 1e-5f;
    cfg.vision.rope_theta = 10000.0f;
    cfg.vision.rope_height_pairs = 2;// 2 height + 2 width pairs
    // Connector: reduce to <= 16 visual tokens, MLP inter 32.
    cfg.connector.max_visual_tokens = 16;
    cfg.connector.connector_intermediate = 32;
    cfg.connector.layer_norm_eps = 1e-5f;
    // ERNIE-4.5-0.3B-style decoder (small synthetic): vocab 32, hidden 32,
    // 3 layers, 4 Q / 2 KV heads, head_dim 8, inter 64, rotate-half RoPE.
    cfg.text.vocab_size = 32;
    cfg.text.hidden_size = 32;
    cfg.text.num_layers = 3;
    cfg.text.num_attention_heads = 4;
    cfg.text.num_kv_heads = 2;
    cfg.text.head_dim = 8;
    cfg.text.intermediate_size = 64;
    cfg.text.rms_norm_eps = 1e-5f;
    cfg.text.rope_theta = 10000.0f;
    cfg.text.rope_pair_layout = RopePairLayout::rotate_half;
    cfg.text.max_position_embeddings = 256;
    cfg.text.eos_token_ids = {31};
    cfg.tie_word_embeddings = true;
    // Native-resolution packing: patch 2, [16, 4096] pixels.
    cfg.packing.patch_size = 2;
    cfg.packing.min_pixels = 16;
    cfg.packing.max_pixels = 4096;
    return cfg;
}

// ---------------------------------------------------------------------------
// Tensor names
// ---------------------------------------------------------------------------

[[nodiscard]] Status PaddleOcrVlTensorNames::validate() const {
    if (auto s = text.validate(); !s.ok()) { return s; }
    if (visual_prefix.empty()) {
        return Status::error(StatusCode::invalid_argument, "visual_prefix must not be empty", "visual_prefix");
    }
    return Status::success();
}

[[nodiscard]] luisa::string PaddleOcrVlTensorNames::patch_embed_weight() const { return visual_prefix + "patch_embed.weight"; }
[[nodiscard]] luisa::string PaddleOcrVlTensorNames::patch_embed_bias() const { return visual_prefix + "patch_embed.bias"; }
[[nodiscard]] luisa::string PaddleOcrVlTensorNames::block_norm1_weight(uint32_t i) const {
    return visual_prefix + "blocks." + luisa::format("{}", i) + ".norm1.weight";
}
[[nodiscard]] luisa::string PaddleOcrVlTensorNames::block_norm1_bias(uint32_t i) const {
    return visual_prefix + "blocks." + luisa::format("{}", i) + ".norm1.bias";
}
[[nodiscard]] luisa::string PaddleOcrVlTensorNames::block_norm2_weight(uint32_t i) const {
    return visual_prefix + "blocks." + luisa::format("{}", i) + ".norm2.weight";
}
[[nodiscard]] luisa::string PaddleOcrVlTensorNames::block_norm2_bias(uint32_t i) const {
    return visual_prefix + "blocks." + luisa::format("{}", i) + ".norm2.bias";
}
[[nodiscard]] luisa::string PaddleOcrVlTensorNames::block_q_proj(uint32_t i) const {
    return visual_prefix + "blocks." + luisa::format("{}", i) + ".q_proj.weight";
}
[[nodiscard]] luisa::string PaddleOcrVlTensorNames::block_k_proj(uint32_t i) const {
    return visual_prefix + "blocks." + luisa::format("{}", i) + ".k_proj.weight";
}
[[nodiscard]] luisa::string PaddleOcrVlTensorNames::block_v_proj(uint32_t i) const {
    return visual_prefix + "blocks." + luisa::format("{}", i) + ".v_proj.weight";
}
[[nodiscard]] luisa::string PaddleOcrVlTensorNames::block_o_proj(uint32_t i) const {
    return visual_prefix + "blocks." + luisa::format("{}", i) + ".o_proj.weight";
}
[[nodiscard]] luisa::string PaddleOcrVlTensorNames::block_q_bias(uint32_t i) const {
    return visual_prefix + "blocks." + luisa::format("{}", i) + ".q_proj.bias";
}
[[nodiscard]] luisa::string PaddleOcrVlTensorNames::block_k_bias(uint32_t i) const {
    return visual_prefix + "blocks." + luisa::format("{}", i) + ".k_proj.bias";
}
[[nodiscard]] luisa::string PaddleOcrVlTensorNames::block_v_bias(uint32_t i) const {
    return visual_prefix + "blocks." + luisa::format("{}", i) + ".v_proj.bias";
}
[[nodiscard]] luisa::string PaddleOcrVlTensorNames::block_o_bias(uint32_t i) const {
    return visual_prefix + "blocks." + luisa::format("{}", i) + ".o_proj.bias";
}
[[nodiscard]] luisa::string PaddleOcrVlTensorNames::block_fc1(uint32_t i) const {
    return visual_prefix + "blocks." + luisa::format("{}", i) + ".fc1.weight";
}
[[nodiscard]] luisa::string PaddleOcrVlTensorNames::block_fc1_bias(uint32_t i) const {
    return visual_prefix + "blocks." + luisa::format("{}", i) + ".fc1.bias";
}
[[nodiscard]] luisa::string PaddleOcrVlTensorNames::block_fc2(uint32_t i) const {
    return visual_prefix + "blocks." + luisa::format("{}", i) + ".fc2.weight";
}
[[nodiscard]] luisa::string PaddleOcrVlTensorNames::block_fc2_bias(uint32_t i) const {
    return visual_prefix + "blocks." + luisa::format("{}", i) + ".fc2.bias";
}
[[nodiscard]] luisa::string PaddleOcrVlTensorNames::connector_ln_weight() const { return visual_prefix + "connector.ln.weight"; }
[[nodiscard]] luisa::string PaddleOcrVlTensorNames::connector_ln_bias() const { return visual_prefix + "connector.ln.bias"; }
[[nodiscard]] luisa::string PaddleOcrVlTensorNames::connector_fc1() const { return visual_prefix + "connector.fc1.weight"; }
[[nodiscard]] luisa::string PaddleOcrVlTensorNames::connector_fc1_bias() const { return visual_prefix + "connector.fc1.bias"; }
[[nodiscard]] luisa::string PaddleOcrVlTensorNames::connector_fc2() const { return visual_prefix + "connector.fc2.weight"; }
[[nodiscard]] luisa::string PaddleOcrVlTensorNames::connector_fc2_bias() const { return visual_prefix + "connector.fc2.bias"; }

// ---------------------------------------------------------------------------
// Tensor inventory
// ---------------------------------------------------------------------------

[[nodiscard]] luisa::vector<ExpectedTensor> paddle_ocr_vl_expected_tensors(
    const PaddleOcrVlConfig &config, const PaddleOcrVlTensorNames &names) {
    luisa::vector<ExpectedTensor> out;
    const auto &v = config.vision;
    const uint64_t vh = v.hidden_size, vi = v.intermediate_size;
    const uint64_t patch_dim = v.patch_dim();
    const uint64_t text_hidden = config.text.hidden_size;
    const uint64_t ci = config.connector.connector_intermediate;
    const auto add = [&out](luisa::string name, luisa::vector<uint64_t> shape, TensorLayout layout) {
        out.emplace_back(ExpectedTensor{std::move(name), std::move(shape), layout, true});
    };
    // Vision patch embed.
    add(names.patch_embed_weight(), {patch_dim, vh}, TensorLayout::kn);
    add(names.patch_embed_bias(), {vh}, TensorLayout::row_major);
    for (uint32_t l = 0; l < v.num_layers; l++) {
        add(names.block_norm1_weight(l), {vh}, TensorLayout::row_major);
        add(names.block_norm1_bias(l), {vh}, TensorLayout::row_major);
        add(names.block_norm2_weight(l), {vh}, TensorLayout::row_major);
        add(names.block_norm2_bias(l), {vh}, TensorLayout::row_major);
        add(names.block_q_proj(l), {vh, vh}, TensorLayout::kn);
        add(names.block_k_proj(l), {vh, vh}, TensorLayout::kn);
        add(names.block_v_proj(l), {vh, vh}, TensorLayout::kn);
        add(names.block_o_proj(l), {vh, vh}, TensorLayout::kn);
        add(names.block_q_bias(l), {vh}, TensorLayout::row_major);
        add(names.block_k_bias(l), {vh}, TensorLayout::row_major);
        add(names.block_v_bias(l), {vh}, TensorLayout::row_major);
        add(names.block_o_bias(l), {vh}, TensorLayout::row_major);
        add(names.block_fc1(l), {vh, vi}, TensorLayout::kn);
        add(names.block_fc1_bias(l), {vi}, TensorLayout::row_major);
        add(names.block_fc2(l), {vi, vh}, TensorLayout::kn);
        add(names.block_fc2_bias(l), {vh}, TensorLayout::row_major);
    }
    // Connector: LayerNorm over vision hidden + 2-layer MLP to text hidden.
    add(names.connector_ln_weight(), {vh}, TensorLayout::row_major);
    add(names.connector_ln_bias(), {vh}, TensorLayout::row_major);
    add(names.connector_fc1(), {vh, ci}, TensorLayout::kn);
    add(names.connector_fc1_bias(), {ci}, TensorLayout::row_major);
    add(names.connector_fc2(), {ci, text_hidden}, TensorLayout::kn);
    add(names.connector_fc2_bias(), {text_hidden}, TensorLayout::row_major);
    // Text decoder: the shared Phase 3 ERNIE inventory (QKV bias, no QK-Norm).
    auto text = dense_decoder_expected_tensors(config.text, names.text, config.tie_word_embeddings);
    for (auto &t : text) { out.push_back(std::move(t)); }
    return out;
}

[[nodiscard]] Status paddle_ocr_vl_validate_tensors(luisa::span<const PackageTensor> tensors,
                                                    luisa::span<const ExpectedTensor> expected) {
    for (const auto &exp : expected) {
        const PackageTensor *found = nullptr;
        for (const auto &t : tensors) {
            if (t.name == exp.name) { found = &t; break; }
        }
        if (found == nullptr) {
            if (exp.required) {
                return Status::error(StatusCode::not_found,
                                     luisa::format("required paddle_ocr_vl tensor '{}' is missing", exp.name), exp.name);
            }
            continue;
        }
        if (found->shape != exp.shape) {
            return Status::error(StatusCode::invalid_argument,
                                 luisa::format("paddle_ocr_vl tensor '{}' has the wrong shape", exp.name), exp.name);
        }
        if (found->layout != exp.layout) {
            return Status::error(StatusCode::invalid_argument,
                                 luisa::format("paddle_ocr_vl tensor '{}' must use layout '{}'", exp.name,
                                               tensor_layout_name(exp.layout)), exp.name);
        }
        if (found->encoding != TensorEncoding::fp32) {
            return Status::error(StatusCode::unsupported,
                                 luisa::format("paddle_ocr_vl tensor '{}' must be fp32 in the Phase 6 baseline", exp.name),
                                 exp.name);
        }
    }
    return Status::success();
}

// ---------------------------------------------------------------------------
// Real-artifact gate
// ---------------------------------------------------------------------------

[[nodiscard]] Status paddle_ocr_vl_validate_real_artifact(luisa::string_view checkpoint_revision,
                                                          luisa::string_view config_fingerprint) {
    (void)checkpoint_revision;
    (void)config_fingerprint;
    return Status::error(StatusCode::unvalidated,
                         "no official PaddleOCR-VL checkpoint/config has been qualified; "
                         "real artifacts fail closed (synthetic path only). PaddleOCR-VL is NOT a "
                         "validated complex-document default.",
                         "paddle_ocr_vl");
}

}// namespace luisa::ocr
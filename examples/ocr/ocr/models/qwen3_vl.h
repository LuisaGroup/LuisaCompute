#pragma once
// OCR engine -- Qwen3-VL graph contract (Phase 4). Host-only.
//
// This module owns the Qwen3-VL-specific recipe/config/tensor contracts and
// the host-side image-packing / interleaved-MRoPE position semantics. It is
// deliberately DISTINCT from models/qwen2_5_vl.h (the shared Qwen2.5-style
// dense decoder contract):
//   * Qwen3 text decoder: QK-Norm (per-head RMSNorm on Q/K), NO QKV bias,
//     GQA, interleaved M-RoPE (pairs (d, d+head_dim/2), three position axes
//     with an explicit per-axis pair-section map).
//   * Vision: patch embed, bidirectional per-image attention with QK-Norm,
//     2D axial vision RoPE, vision MLP, spatial merger, and DeepStack
//     feature projections injected at EXACT configured early decoder layers.
//   * Instruct vs Thinking is an explicit identity boundary: only the
//     Instruct variant is accepted here; a Thinking config fails closed.
//
// IMPORTANT (plan §1/§6 honesty rules): no official Qwen3-VL checkpoint has
// been pinned or inspected. Every tensor name / dimension / convention in
// this header is a DECLARED SYNTHETIC CONTRACT exercised by the synthetic
// vertical slice (runtime/qwen3_vl_executor.*). Nothing here validates a
// real artifact; real packages fail closed through the registry/Session.
// The official gates are recorded in MODELS.md.
//
// Host-only header: no DSL/runtime device includes.
#include <cstdint>

#include <luisa/core/stl/string.h>
#include <luisa/core/stl/vector.h>

#include "ocr/models/qwen2_5_vl.h"
#include "ocr/package.h"
#include "ocr/status.h"

namespace luisa::ocr {

// The packing transform is a plain scale/offset map back to source pixels.
// It is structurally identical to the Phase 1 CoordinateTransform but declared
// here so this header stays free of api.h/model_recipe.h (their host-only
// AttentionMaskKind would clash with the ops enum in device TUs). The host
// preprocessing bridge (runtime/qwen3_vl_preprocess.*) converts to the Phase
// 1 PreprocessSpec where preprocessing is invoked.
struct Qwen3VlSourceTransform {
    float scale_x{1.0f};
    float scale_y{1.0f};
    float offset_x{0.0f};
    float offset_y{0.0f};
};

// ---------------------------------------------------------------------------
// Identity boundary
// ---------------------------------------------------------------------------

// Explicit Qwen3-VL product variant. Instruct and Thinking are different
// recipes (different chat templates / generation semantics); Thinking is NOT
// implemented and fails closed everywhere in Phase 4.
enum class Qwen3VlVariant : uint32_t {
    instruct = 0,
    thinking = 1,
};

[[nodiscard]] luisa::string_view qwen3_vl_variant_name(Qwen3VlVariant variant) noexcept;
[[nodiscard]] Result<Qwen3VlVariant> qwen3_vl_variant_from_name(luisa::string_view name);

// ---------------------------------------------------------------------------
// Vision / merger / DeepStack geometry
// ---------------------------------------------------------------------------

// Vision tower geometry. All fields concrete before use. The synthetic slice
// uses: patch-linear embed over flattened (C, patch, patch) rows, LayerNorm
// (weight+bias) pre-norms, per-head QK-Norm (RMSNorm over head_dim), 2D axial
// RoPE (adjacent pairs, height/width axes), full bidirectional attention
// WITHIN one image (ragged images never attend across each other), GELU-erf
// MLP, and a spatial merger (m x m token packing + LayerNorm + 2-layer MLP).
struct Qwen3VlVisionConfig {
    uint32_t num_layers{0};
    uint32_t hidden_size{0};
    uint32_t num_heads{0};
    uint32_t head_dim{0};// explicit; heads * head_dim == hidden_size
    uint32_t intermediate_size{0};
    uint32_t patch_size{0};       // spatial patch edge (pixels)
    uint32_t temporal_patch_size{0};// 1 for still images (declared; >1 refused)
    uint32_t channels{3};
    uint32_t spatial_merge_size{0};// m; merger groups m*m patches
    uint32_t merger_intermediate_size{0};// merger MLP hidden width
    float layer_norm_eps{1e-6f};
    float rope_theta{10000.0f};
    // 2D vision RoPE pair sections: how many of the head_dim/2 adjacent pairs
    // use the HEIGHT axis; the rest use WIDTH. Explicit, never guessed.
    uint32_t rope_height_pairs{0};

    [[nodiscard]] uint32_t patch_dim() const noexcept { return channels * patch_size * patch_size * temporal_patch_size; }
    [[nodiscard]] uint32_t merged_dim() const noexcept { return hidden_size * spatial_merge_size * spatial_merge_size; }
    [[nodiscard]] Status validate() const;
};

// DeepStack contract: `decoder_layers[i]` is the EXACT decoder layer after
// whose residual stream the i-th projected visual feature is added (image
// token rows only); `vision_layers[i]` is the vision block whose output feeds
// the i-th DeepStack merger. Both lists are caller data, strictly increasing,
// and validated against the decoder/vision layer counts. The injection point
// (AFTER the selected decoder layer) is the declared synthetic convention;
// the official placement is an unmet gate (MODELS.md).
struct Qwen3VlDeepStackConfig {
    luisa::vector<uint32_t> decoder_layers;
    luisa::vector<uint32_t> vision_layers;

    [[nodiscard]] uint32_t count() const noexcept { return static_cast<uint32_t>(decoder_layers.size()); }
    [[nodiscard]] Status validate(uint32_t num_decoder_layers, uint32_t num_vision_layers) const;
};

// Interleaved M-RoPE section map: number of RoPE pairs assigned to the
// temporal / height / width position axes. Sum must equal head_dim/2. The
// pair layout is interleaved (d, d+head_dim/2) -- NOT the adjacent-pair
// Qwen2-VL M-RoPE and NOT the 1-axis Qwen2.5 text RoPE.
struct Qwen3VlMropeSections {
    uint32_t temporal{0};
    uint32_t height{0};
    uint32_t width{0};

    [[nodiscard]] uint32_t total() const noexcept { return temporal + height + width; }
    // axis_of_pair[d] for d in [0, head_dim/2): 0=temporal, 1=height, 2=width.
    [[nodiscard]] luisa::vector<uint32_t> axis_of_pair() const;
    [[nodiscard]] Status validate(uint32_t head_dim) const;
};

// ---------------------------------------------------------------------------
// Full Qwen3-VL config
// ---------------------------------------------------------------------------

struct Qwen3VlConfig {
    Qwen3VlVariant variant{Qwen3VlVariant::instruct};
    // Text decoder geometry reuses the shared dense geometry struct, but the
    // Qwen3 CONTRACT fields below (qk_norm/no-bias/interleaved MRoPE) are
    // fixed by this module -- a Qwen3 config can never be routed through the
    // Qwen2.5 contract (parse_qwen2_5_text_config refuses qwen3 markers, and
    // this parser refuses qwen2_5 markers).
    DenseDecoderConfig text;
    Qwen3VlVisionConfig vision;
    Qwen3VlDeepStackConfig deepstack;
    Qwen3VlMropeSections mrope;
    bool tie_word_embeddings{true};

    [[nodiscard]] Status validate() const;
};

// Strict Qwen3-VL config parsing (synthetic contract schema). Accepts a JSON
// object with: optional "model_type" (must contain "qwen3_vl" when present),
// "variant" ("instruct" required; "thinking" fails closed), "text_config"
// (shared dense geometry keys), "vision_config" (keys below), "deepstack"
// ({decoder_layers, vision_layers}), "mrope_section" ([t,h,w]),
// "tie_word_embeddings" (bool, default true).
// vision_config keys: num_layers, hidden_size, num_heads, head_dim,
// intermediate_size, patch_size, temporal_patch_size (must be 1),
// spatial_merge_size, merger_intermediate_size, layer_norm_eps, rope_theta,
// rope_height_pairs. Fails closed on Qwen2.5 markers, MoE/GDN fields, and
// unknown structural fields it does not implement.
[[nodiscard]] Result<Qwen3VlConfig> parse_qwen3_vl_config(luisa::string_view json);

// The declared SMALL SYNTHETIC Qwen3-like geometry used by the vertical-slice
// tests. NOT a real checkpoint: vocab 32, hidden 32, 3 decoder layers, 4 Q /
// 2 KV heads, head_dim 8, inter 64; vision 16 hidden / 2 heads / head_dim 8 /
// 2 layers / inter 32, patch 2, merge 2; DeepStack at decoder layers {0,1}
// from vision layers {0,1}; MRoPE sections {2,1,1}.
[[nodiscard]] Qwen3VlConfig qwen3_vl_synthetic_config();

// ---------------------------------------------------------------------------
// Tensor-name map + expected inventory (synthetic canonical contract)
// ---------------------------------------------------------------------------

// Canonical tensor names for the Qwen3-VL graph. The text decoder reuses the
// shared DenseDecoderTensorNames prefix contract ("model." or
// "language_model."); vision/merger/DeepStack names live under `visual_prefix`.
// These are the SYNTHETIC canonical names; the official checkpoint names are
// an unmet gate and real artifacts fail closed before any name is trusted.
struct Qwen3VlTensorNames {
    DenseDecoderTensorNames text;
    luisa::string visual_prefix{"visual."};

    [[nodiscard]] Status validate() const;
    [[nodiscard]] luisa::string patch_embed_weight() const;
    [[nodiscard]] luisa::string patch_embed_bias() const;
    [[nodiscard]] luisa::string block_prefix(uint32_t layer) const;
    [[nodiscard]] luisa::string block_norm1_weight(uint32_t layer) const;
    [[nodiscard]] luisa::string block_norm1_bias(uint32_t layer) const;
    [[nodiscard]] luisa::string block_norm2_weight(uint32_t layer) const;
    [[nodiscard]] luisa::string block_norm2_bias(uint32_t layer) const;
    [[nodiscard]] luisa::string block_q_proj(uint32_t layer) const;
    [[nodiscard]] luisa::string block_k_proj(uint32_t layer) const;
    [[nodiscard]] luisa::string block_v_proj(uint32_t layer) const;
    [[nodiscard]] luisa::string block_o_proj(uint32_t layer) const;
    [[nodiscard]] luisa::string block_q_bias(uint32_t layer) const;
    [[nodiscard]] luisa::string block_k_bias(uint32_t layer) const;
    [[nodiscard]] luisa::string block_v_bias(uint32_t layer) const;
    [[nodiscard]] luisa::string block_o_bias(uint32_t layer) const;
    [[nodiscard]] luisa::string block_q_norm(uint32_t layer) const;
    [[nodiscard]] luisa::string block_k_norm(uint32_t layer) const;
    [[nodiscard]] luisa::string block_fc1(uint32_t layer) const;
    [[nodiscard]] luisa::string block_fc1_bias(uint32_t layer) const;
    [[nodiscard]] luisa::string block_fc2(uint32_t layer) const;
    [[nodiscard]] luisa::string block_fc2_bias(uint32_t layer) const;
    [[nodiscard]] luisa::string merger_prefix() const;// visual.merger.
    [[nodiscard]] luisa::string deepstack_prefix(uint32_t index) const;// visual.deepstack_merger.{i}.
    // Merger (main or DeepStack) tensor names under a given prefix.
    [[nodiscard]] static luisa::string merger_ln_weight(const luisa::string &prefix);
    [[nodiscard]] static luisa::string merger_ln_bias(const luisa::string &prefix);
    [[nodiscard]] static luisa::string merger_fc1(const luisa::string &prefix);
    [[nodiscard]] static luisa::string merger_fc1_bias(const luisa::string &prefix);
    [[nodiscard]] static luisa::string merger_fc2(const luisa::string &prefix);
    [[nodiscard]] static luisa::string merger_fc2_bias(const luisa::string &prefix);
};

// Full expected tensor inventory (vision + merger + DeepStack + text decoder)
// for `config` under `names`. Linear weights are packed [K,N] (`kn`). The
// text-decoder part is the Qwen3 inventory: QK-Norm weights REQUIRED, QKV
// biases ABSENT (a package carrying them is rejected as a contract mismatch).
[[nodiscard]] luisa::vector<ExpectedTensor> qwen3_vl_expected_tensors(
    const Qwen3VlConfig &config, const Qwen3VlTensorNames &names);

// Validate a package tensor table against the full expected inventory: every
// required tensor present with exact shape/layout/fp32 encoding, no missing
// required entry. (Unexpected-tensor rejection is the executor's job per
// component; this checks presence/shape/layout of the declared inventory.)
[[nodiscard]] Status validate_qwen3_vl_tensors(luisa::span<const PackageTensor> tensors,
                                               luisa::span<const ExpectedTensor> expected);

// ---------------------------------------------------------------------------
// Native/dynamic image packing + source transforms (host metadata)
// ---------------------------------------------------------------------------

// Declared synthetic packing convention (contract to verify against the
// official processor): each source edge is scaled by a uniform factor so the
// pixel count fits [min_pixels, max_pixels], then rounded to a positive
// multiple of patch_size * spatial_merge_size. The resize is aspect-preserving
// up to the independent per-edge rounding. The CoordinateTransform maps
// preprocessed pixels back to source pixels.
struct Qwen3VlPackingSpec {
    uint32_t patch_size{0};
    uint32_t spatial_merge_size{0};
    uint64_t min_pixels{0};
    uint64_t max_pixels{0};

    [[nodiscard]] Status validate() const;
};

struct Qwen3VlPackedImage {
    uint32_t source_width{0};
    uint32_t source_height{0};
    uint32_t resized_width{0};  // multiple of patch*merge
    uint32_t resized_height{0};
    uint32_t grid_h{0};// resized_height / patch (patch units)
    uint32_t grid_w{0};
    uint32_t grid_t{1};// still image
    uint32_t spatial_merge_size{0};
    uint64_t patch_count{0};// grid_t * grid_h * grid_w
    uint64_t merged_count{0};// patch_count / (merge*merge)
    Qwen3VlSourceTransform transform;// preprocessed -> source

    [[nodiscard]] uint32_t merged_grid_h() const noexcept { return spatial_merge_size == 0u ? 0u : grid_h / spatial_merge_size; }
    [[nodiscard]] uint32_t merged_grid_w() const noexcept { return spatial_merge_size == 0u ? 0u : grid_w / spatial_merge_size; }
};

// Pack one image size (no pixels touched). Fails closed on degenerate sizes,
// invalid spec, or a pixel budget that cannot be satisfied.
[[nodiscard]] Result<Qwen3VlPackedImage> qwen3_vl_pack_image(uint32_t source_width, uint32_t source_height,
                                                             const Qwen3VlPackingSpec &spec);

// ---------------------------------------------------------------------------
// Interleaved M-RoPE position semantics (explicit axis/position contract)
// ---------------------------------------------------------------------------

// One prompt segment: either `text_tokens` text tokens or one packed image.
// Exactly one of the two is active (image != nullptr).
struct Qwen3VlPromptSegment {
    uint32_t text_tokens{0};
    const Qwen3VlPackedImage *image{nullptr};
};

// Position ids for the whole prompt, [seq_len * 3] int32, per token
// (temporal, height, width). Declared semantics (synthetic contract):
//   * text token at running position p -> (p, p, p); p advances by 1.
//   * image token (ti, hi, wi) with start s = previous_max + 1 ->
//     (s + ti, s + hi, s + wi); after the image the next text position is
//     s + max(grid_t, grid_h, grid_w).
//   * the first segment starts at 0.
// `next_text_position` is the position the FIRST decode step must use (all
// three axes advance together for generated text).
struct Qwen3VlPositionIds {
    luisa::vector<int32_t> ids;// [seq_len * 3]
    uint32_t seq_len{0};
    int32_t next_text_position{0};
};

[[nodiscard]] Result<Qwen3VlPositionIds> qwen3_vl_build_position_ids(
    luisa::span<const Qwen3VlPromptSegment> segments);

// Vision 2D RoPE position ids for one packed image: [patch_count * 2] int32
// (height, width) per patch in row-major (h, w) order; temporal axis absent.
[[nodiscard]] luisa::vector<int32_t> qwen3_vl_vision_positions(const Qwen3VlPackedImage &packed);

// Vision 2D RoPE axis map: first rope_height_pairs pairs use axis 0 (height),
// the rest axis 1 (width). Length head_dim/2.
[[nodiscard]] luisa::vector<uint32_t> qwen3_vl_vision_axis_of_pair(const Qwen3VlVisionConfig &vision);

}// namespace luisa::ocr

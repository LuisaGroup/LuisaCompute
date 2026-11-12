#pragma once
// OCR engine -- PaddleOCR-VL-1.6 graph contract (Phase 6). Host-safe header.
//
// DEVICE-SAFE header: the PaddleOCR-VL region recognizer is a DISTINCT graph
// from Qwen3-VL:
//   * native-resolution NaViT vision tower (patch embed + bidirectional
//     attention with QKV bias + 2D axial RoPE + GELU MLP; NO QK-Norm, NO
//     DeepStack -- those are Qwen3 conventions and are absent here),
//   * adaptive visual-token reduction (adaptive average pool over the patch
//     grid to a token-budget grid) + a LayerNorm/MLP connector,
//   * an ERNIE-4.5-0.3B-style dense decoder (QKV bias, rotate-half 1D RoPE,
//     RMSNorm, SwiGLU; the shared DenseVlmExecutor Phase 3 contract).
// The full pipeline is layout (PP-DocLayoutV3) -> crops -> this region engine ->
// Markdown/JSON/spotting composition. Running only the VLM is insufficient.
//
// This header pulls NO api.h/model_recipe.h (host AttentionMaskKind clash) and
// NO ops_attention.h, so both host and device TUs may include it. The device
// graph lives in runtime/layout_vlm_executor.*; the host composition + public
// result bridge live in this module's .cpp (host-only).
//
// HONESTY (plan §1/§6): NO official PaddleOCR-VL checkpoint has been pinned,
// inspected, or run. Every dimension / tensor name here is a DECLARED SYNTHETIC
// CONTRACT exercised by the synthetic vertical slice. Real packages fail closed
// through paddle_ocr_vl_validate_real_artifact(). PaddleOCR-VL is NOT a validated
// complex-document default; the official gates are recorded in MODELS.md.
#include <cstdint>
#include <utility>

#include <luisa/core/stl/string.h>
#include <luisa/core/stl/vector.h>

#include "ocr/models/qwen2_5_vl.h"// DenseDecoderConfig / DenseDecoderTensorNames / ExpectedTensor
#include "ocr/package.h"
#include "ocr/runtime/scheduler.h"
#include "ocr/status.h"

namespace luisa::ocr {

// ---------------------------------------------------------------------------
// Native-resolution NaViT vision geometry (declared synthetic contract)
// ---------------------------------------------------------------------------

// NaViT vision tower. Distinct from Qwen3-VL: LayerNorm pre-norms, QKV WITH
// bias, NO QK-Norm, 2D axial RoPE (adjacent pairs, height/width axes), GELU-erf
// MLP, bidirectional attention WITHIN one image. No spatial merger / DeepStack.
struct PaddleNavitConfig {
    uint32_t num_layers{0};
    uint32_t hidden_size{0};
    uint32_t num_heads{0};
    uint32_t head_dim{0};// heads * head_dim == hidden_size
    uint32_t intermediate_size{0};
    uint32_t patch_size{0};
    uint32_t channels{3};
    float layer_norm_eps{1e-6f};
    float rope_theta{10000.0f};
    // 2D vision RoPE pair sections: how many of head_dim/2 adjacent pairs use
    // the HEIGHT axis; the rest use WIDTH. Explicit, never guessed.
    uint32_t rope_height_pairs{0};

    [[nodiscard]] uint32_t patch_dim() const noexcept { return channels * patch_size * patch_size; }
    [[nodiscard]] Status validate() const;
};

// Adaptive visual-token reduction + MLP connector. The patch grid (grid_h,
// grid_w) is adaptive-average-pooled to a target grid whose token count fits
// max_visual_tokens (aspect-preserving, floor, >= 1 per axis). The pooled
// features are flattened to [tokens, hidden], LayerNorm'd, then passed through
// a 2-layer GELU MLP connector to the text hidden size.
struct PaddleConnectorConfig {
    uint32_t max_visual_tokens{0};// token budget for the reduction (0 invalid)
    uint32_t connector_intermediate{0};
    float layer_norm_eps{1e-6f};

    [[nodiscard]] Status validate() const;
};

// Compute the reduced token grid for a patch grid under the token budget.
// Aspect-preserving: scale = sqrt(max_tokens / (gh*gw)) when over budget, then
// floor each axis to >= 1. Returns (out_h, out_w). Pure host math.
[[nodiscard]] Result<std::pair<uint32_t, uint32_t>> paddle_reduce_grid(
    uint32_t grid_h, uint32_t grid_w, uint32_t max_visual_tokens);

// ---------------------------------------------------------------------------
// Native-resolution packing (declared synthetic convention)
// ---------------------------------------------------------------------------

// Packing spec: each source edge is scaled uniformly into [min_pixels,
// max_pixels] then rounded to a positive multiple of patch_size. Unlike the
// Qwen3 spec there is NO spatial-merge factor (the NaViT tower here has no
// merger; token reduction happens in the connector).
struct PaddlePackingSpec {
    uint32_t patch_size{0};
    uint64_t min_pixels{0};
    uint64_t max_pixels{0};

    [[nodiscard]] Status validate() const;
};

struct PaddlePackedImage {
    uint32_t source_width{0};
    uint32_t source_height{0};
    uint32_t resized_width{0};  // multiple of patch
    uint32_t resized_height{0};
    uint32_t grid_h{0};
    uint32_t grid_w{0};
    uint64_t patch_count{0};
    // Reduced token grid (connector output) + count.
    uint32_t reduced_h{0};
    uint32_t reduced_w{0};
    uint64_t visual_tokens{0};
    // Preprocessed -> source transform (scale only; no offset).
    float scale_x{1.0f};
    float scale_y{1.0f};
};

// Pack one image size (no pixels touched). Fails closed on degenerate sizes,
// invalid spec, or a pixel budget that cannot be satisfied.
[[nodiscard]] Result<PaddlePackedImage> paddle_pack_image(uint32_t source_width, uint32_t source_height,
                                                          const PaddlePackingSpec &spec, uint32_t max_visual_tokens);

// Vision 2D RoPE positions for one packed image: [patch_count * 2] int32
// (height, width) per patch, row-major. Axis map: first rope_height_pairs pairs
// use axis 0 (height), the rest axis 1 (width).
[[nodiscard]] luisa::vector<int32_t> paddle_vision_positions(const PaddlePackedImage &packed);
[[nodiscard]] luisa::vector<uint32_t> paddle_vision_axis_of_pair(const PaddleNavitConfig &vision);

// ---------------------------------------------------------------------------
// Full PaddleOCR-VL config
// ---------------------------------------------------------------------------

struct PaddleOcrVlConfig {
    PaddleNavitConfig vision;
    PaddleConnectorConfig connector;
    // ERNIE-4.5-0.3B-style decoder geometry reuses the shared dense contract.
    // The Phase 3 switches (qk_norm=false, qkv_bias=true, rope_axes=1,
    // rotate_half) are FIXED by this module -- a Paddle config can never be
    // routed through the Qwen3 contract.
    DenseDecoderConfig text;
    bool tie_word_embeddings{true};
    PaddlePackingSpec packing;

    [[nodiscard]] Status validate() const;
};

// The declared SMALL SYNTHETIC PaddleOCR-VL geometry used by the vertical-slice
// tests. NOT a real checkpoint.
[[nodiscard]] PaddleOcrVlConfig paddle_ocr_vl_synthetic_config();

// ---------------------------------------------------------------------------
// Tensor-name map + expected inventory (synthetic canonical contract)
// ---------------------------------------------------------------------------

struct PaddleOcrVlTensorNames {
    DenseDecoderTensorNames text;
    luisa::string visual_prefix{"visual."};

    [[nodiscard]] Status validate() const;
    // Vision patch embed.
    [[nodiscard]] luisa::string patch_embed_weight() const;
    [[nodiscard]] luisa::string patch_embed_bias() const;
    // Vision block i.
    [[nodiscard]] luisa::string block_norm1_weight(uint32_t i) const;
    [[nodiscard]] luisa::string block_norm1_bias(uint32_t i) const;
    [[nodiscard]] luisa::string block_norm2_weight(uint32_t i) const;
    [[nodiscard]] luisa::string block_norm2_bias(uint32_t i) const;
    [[nodiscard]] luisa::string block_q_proj(uint32_t i) const;
    [[nodiscard]] luisa::string block_k_proj(uint32_t i) const;
    [[nodiscard]] luisa::string block_v_proj(uint32_t i) const;
    [[nodiscard]] luisa::string block_o_proj(uint32_t i) const;
    [[nodiscard]] luisa::string block_q_bias(uint32_t i) const;
    [[nodiscard]] luisa::string block_k_bias(uint32_t i) const;
    [[nodiscard]] luisa::string block_v_bias(uint32_t i) const;
    [[nodiscard]] luisa::string block_o_bias(uint32_t i) const;
    [[nodiscard]] luisa::string block_fc1(uint32_t i) const;
    [[nodiscard]] luisa::string block_fc1_bias(uint32_t i) const;
    [[nodiscard]] luisa::string block_fc2(uint32_t i) const;
    [[nodiscard]] luisa::string block_fc2_bias(uint32_t i) const;
    // Connector (LayerNorm + 2-layer MLP).
    [[nodiscard]] luisa::string connector_ln_weight() const;
    [[nodiscard]] luisa::string connector_ln_bias() const;
    [[nodiscard]] luisa::string connector_fc1() const;
    [[nodiscard]] luisa::string connector_fc1_bias() const;
    [[nodiscard]] luisa::string connector_fc2() const;
    [[nodiscard]] luisa::string connector_fc2_bias() const;
};

// Full expected tensor inventory (vision + connector + text decoder). Linear
// weights packed [K,N] (`kn`). The text part is the Phase 3 ERNIE inventory
// (QKV biases REQUIRED, no QK-Norm).
[[nodiscard]] luisa::vector<ExpectedTensor> paddle_ocr_vl_expected_tensors(
    const PaddleOcrVlConfig &config, const PaddleOcrVlTensorNames &names);

// Validate a package tensor table against the full expected inventory.
[[nodiscard]] Status paddle_ocr_vl_validate_tensors(luisa::span<const PackageTensor> tensors,
                                                    luisa::span<const ExpectedTensor> expected);

// ---------------------------------------------------------------------------
// Real-artifact gate (host-only composition lives in paddle_ocr_vl_result.h)
// ---------------------------------------------------------------------------

// Real-artifact gate: ALWAYS fail-closed in Phase 6.
[[nodiscard]] Status paddle_ocr_vl_validate_real_artifact(luisa::string_view checkpoint_revision,
                                                          luisa::string_view config_fingerprint);

}// namespace luisa::ocr
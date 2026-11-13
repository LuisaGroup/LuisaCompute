#pragma once
// OCR engine -- GLM-OCR graph contract (Phase 7). Host-safe header.
//
// GLM-OCR is a DISTINCT layout VLM, not a Qwen alias:
//   * PP-DocLayoutV3 front end (the shared layout detector + reading order);
//   * a CogViT vision tower: learned ABSOLUTE position embeddings (NO vision
//     RoPE), QKV bias, NO QK-Norm, bidirectional attention, GELU-erf MLP;
//   * a DOWNSAMPLING connector: learned strided token downsampling (spatial
//     m x m gather -> learned linear projection) + LayerNorm + 2-layer GELU
//     MLP -- NOT Paddle's adaptive pool, NOT MinerU's pixel-unshuffle, NOT
//     TeleOCR's plain spatial-merge aligner;
//   * a GLM-0.5B-style text decoder with the EXACT declared GLM conventions:
//     PARTIAL rotary (only the first head_dim/2 elements rotate, rotate-half
//     pairs within the rotated prefix, inv_freq over the rotated dim), QKV
//     bias, NO QK-Norm, GQA, RMSNorm, SwiGLU, plain 1D text positions.
//   * Ordinary target decoding ONLY. MTP (multi-token prediction) draft/target
//     verification is DEFERRED and is never required for correct support.
//
// Real config fields that differ from Qwen FAIL CLOSED: a Qwen3 marker
// (qk_norm / interleaved mrope), a Qwen2.5 full-rotary rotate-half text config
// without the GLM partial factor, MoE/GDN fields, or an MTP requirement are
// all refused by parse_glm_ocr_config.
//
// This header pulls NO api.h/model_recipe.h and NO ops_attention.h, so both
// host and device TUs may include it. The device graph is the shared
// runtime/p7_vlm_executor.*; host composition lives in this module's .cpp.
//
// HONESTY (plan §1/§6): NO official GLM-OCR checkpoint has been pinned,
// inspected, or run. Every dimension / tensor name here is a DECLARED SYNTHETIC
// CONTRACT. Real packages fail closed through glm_ocr_validate_real_artifact().
#include <cstdint>

#include <luisa/core/stl/string.h>
#include <luisa/core/stl/vector.h>

#include "ocr/models/qwen2_5_vl.h"
#include "ocr/package.h"
#include "ocr/runtime/p7_slice_config.h"
#include "ocr/runtime/scheduler.h"
#include "ocr/status.h"

namespace luisa::ocr {

struct GlmOcrConfig {
    P7SliceConfig graph;
    P7TensorNames names;

    [[nodiscard]] Status validate() const;
};

// The declared SMALL SYNTHETIC GLM-OCR geometry used by the vertical-slice
// tests. NOT a real checkpoint. Vision (CogViT): 2 layers, hidden 16, 2 heads,
// head_dim 8, patch 2, learned absolute positions (1024 rows), NO vision RoPE;
// downsampling connector stride 2; GLM text: vocab 32, hidden 32, 3 layers,
// 4 Q / 2 KV heads, head_dim 8, PARTIAL rotary 1/2 (rotate-half), QKV bias.
[[nodiscard]] GlmOcrConfig glm_ocr_synthetic_config();

// Strict GLM-OCR config parsing (synthetic contract schema). Accepts a JSON
// object with "model_type" (must contain "glm" when present), "vision_config",
// "text_config", "connector", "partial_rotary_factor" (a number in (0,1]).
// Fails closed on Qwen markers (qk_norm, mrope_section, interleaved rope),
// MoE/GDN fields, and an MTP requirement (mtp_num_layers > 0). The resulting
// text contract is ALWAYS GLM (partial rotary, QKV bias, no QK-Norm, 1D).
[[nodiscard]] Result<GlmOcrConfig> parse_glm_ocr_config(luisa::string_view json);

// Full expected tensor inventory (CogViT + downsampling connector + GLM text).
[[nodiscard]] luisa::vector<ExpectedTensor> glm_ocr_expected_tensors(const GlmOcrConfig &config);
[[nodiscard]] Status glm_ocr_validate_tensors(luisa::span<const PackageTensor> tensors,
                                              luisa::span<const ExpectedTensor> expected);

// Real-artifact gate: ALWAYS fail-closed in Phase 7.
[[nodiscard]] Status glm_ocr_validate_real_artifact(luisa::string_view checkpoint_revision,
                                                    luisa::string_view config_fingerprint);

// ---------------------------------------------------------------------------
// Host composition + public bridge (host-only; defined in the .cpp).
// GLM-OCR supports plain + markdown + extract (strict). spotting is NOT
// validated for GLM-OCR and fails closed. Malformed structured output is an
// error, never a plain-text fallback. Generative content: no confidence.
// ---------------------------------------------------------------------------
[[nodiscard]] Result<LayoutDocument> glm_ocr_compose(const LayoutRunResult &run, LayoutOutputMode mode,
                                                     luisa::string_view extract_schema);

}// namespace luisa::ocr

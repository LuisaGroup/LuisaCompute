#pragma once
// OCR engine -- HunyuanOCR-1.5 graph contract (Phase 7). Host-safe header.
//
// HunyuanOCR is a DISTINCT custom VLM, not a Qwen/GLM alias:
//   * a native Hunyuan-ViT vision tower: QKV bias, NO QK-Norm, 2D axial RoPE,
//     SiLU MLP (the Hunyuan vision activation, distinct from the GELU-erf
//     towers), bidirectional attention;
//   * an ADAPTIVE-POOL + MLP connector (adaptive average pool to a token-budget
//     grid + LayerNorm + 2-layer GELU MLP). This is the Hunyuan connector; it
//     is NOT MinerU's pixel-unshuffle and NOT GLM's learned downsampling;
//   * a Hunyuan-0.5B-style text decoder with XD-RoPE over FOUR position axes
//     (text, height, width, time), adjacent pairs, QKV bias, NO QK-Norm, GQA,
//     RMSNorm, SwiGLU;
//   * spotting / extraction / translation output adapters ONLY where
//     implemented and tested (see hunyuan_ocr_compose / the spotting adapter).
//   * Ordinary target decoding ONLY. DFlash is DEFERRED and never required.
//   * NO fake confidence: generative content carries has_confidence == false.
//
// This header pulls NO api.h/model_recipe.h and NO ops_attention.h, so both
// host and device TUs may include it. The device graph is the shared
// runtime/p7_vlm_executor.*; host composition lives in this module's .cpp.
//
// HONESTY (plan §1/§6): NO official HunyuanOCR checkpoint has been pinned,
// inspected, or run. Every dimension / tensor name here is a DECLARED SYNTHETIC
// CONTRACT. Real packages fail closed through hunyuan_ocr_validate_real_artifact().
#include <cstdint>

#include <luisa/core/stl/string.h>
#include <luisa/core/stl/vector.h>

#include "ocr/models/qwen2_5_vl.h"
#include "ocr/package.h"
#include "ocr/runtime/p7_slice_config.h"
#include "ocr/runtime/scheduler.h"
#include "ocr/status.h"

namespace luisa::ocr {

struct HunyuanOcrConfig {
    P7SliceConfig graph;
    P7TensorNames names;

    [[nodiscard]] Status validate() const;
};

// The declared SMALL SYNTHETIC HunyuanOCR geometry used by the vertical-slice
// tests. NOT a real checkpoint. Vision (Hunyuan-ViT): 2 layers, hidden 16,
// 2 heads, head_dim 8, patch 2, 2D RoPE, SiLU MLP; adaptive-pool connector
// (<= 16 visual tokens); Hunyuan text: vocab 32, hidden 32, 3 layers, 4 Q /
// 2 KV heads, head_dim 8, XD-RoPE sections {1,1,1,1} (text/height/width/time).
[[nodiscard]] HunyuanOcrConfig hunyuan_ocr_synthetic_config();

// Strict HunyuanOCR config parsing (synthetic contract schema). Accepts a JSON
// object with "model_type" (must contain "hunyuan" when present),
// "vision_config", "text_config", "connector", "xd_rope_section" ([t,h,w,time]).
// Fails closed on Qwen markers (qk_norm / mrope_section / interleaved rope),
// MoE/GDN fields, and a DFlash requirement. The resulting text contract is
// ALWAYS Hunyuan XD-RoPE (4 axes, adjacent pairs).
[[nodiscard]] Result<HunyuanOcrConfig> parse_hunyuan_ocr_config(luisa::string_view json);

// Full expected tensor inventory (Hunyuan-ViT + adaptive connector + Hunyuan text).
[[nodiscard]] luisa::vector<ExpectedTensor> hunyuan_ocr_expected_tensors(const HunyuanOcrConfig &config);
[[nodiscard]] Status hunyuan_ocr_validate_tensors(luisa::span<const PackageTensor> tensors,
                                                  luisa::span<const ExpectedTensor> expected);

// Real-artifact gate: ALWAYS fail-closed in Phase 7.
[[nodiscard]] Status hunyuan_ocr_validate_real_artifact(luisa::string_view checkpoint_revision,
                                                        luisa::string_view config_fingerprint);

// ---------------------------------------------------------------------------
// Host composition + output adapters (host-only; defined in the .cpp).
// HunyuanOCR supports plain + markdown + extract (strict) and a spotting
// adapter that carries source coordinates from the layout regions. Translation
// is NOT implemented and fails closed. Malformed structured output is an error,
// never a plain-text fallback. Generative content: NO confidence.
// ---------------------------------------------------------------------------
[[nodiscard]] Result<LayoutDocument> hunyuan_ocr_compose(const LayoutRunResult &run, LayoutOutputMode mode,
                                                         luisa::string_view extract_schema);

// Spotting adapter: produces lines/blocks with source coordinates from the
// layout regions (the validated spotting source). The canonical text is the
// region content joined in reading order. NO confidence is fabricated. This is
// the only spotting path; it requires layout regions with source boxes.
[[nodiscard]] Result<LayoutDocument> hunyuan_ocr_spotting(const LayoutRunResult &run);

}// namespace luisa::ocr

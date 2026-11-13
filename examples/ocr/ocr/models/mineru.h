#pragma once
// OCR engine -- MinerU2.5-Pro graph contract (Phase 7). Host-safe header.
//
// MinerU is a DISTINCT layout VLM, not a Paddle/Qwen alias:
//   * an overview page-layout pass + native-resolution region crops (the
//     shared LayoutVlmExecutor front end);
//   * a 675M-NaViT-SHAPED synthetic vision tower: native-resolution patches,
//     QKV bias, NO QK-Norm, 2D axial RoPE (height/width), GELU-erf MLP,
//     bidirectional attention within one image;
//   * a 2x2 PIXEL-UNSHUFFLE connector (NCHW channel-fold) + LayerNorm +
//     2-layer GELU MLP. This is NOT Paddle's adaptive-pool connector and NOT
//     TeleOCR's spatial-merge aligner -- the channel-fold ordering is the
//     distinguishing contract and is device-tested;
//   * a Qwen2-0.5B-style text decoder: QKV bias, NO QK-Norm, GQA, RMSNorm,
//     SwiGLU, ADJACENT-pair M-RoPE (Qwen2-VL style, NOT the Qwen3 interleaved
//     convention);
//   * reading-order restoration and CROSS-PAGE document composition.
//
// This header pulls NO api.h/model_recipe.h and NO ops_attention.h, so both
// host and device TUs may include it. The device graph is the shared
// runtime/p7_vlm_executor.*; host composition lives in this module's .cpp.
//
// HONESTY (plan §1/§6): NO official MinerU checkpoint has been pinned,
// inspected, or run. Every dimension / tensor name here is a DECLARED SYNTHETIC
// CONTRACT (the "675M NaViT" / "Qwen2-0.5B" labels name the intended real
// components, not a qualified checkpoint). Real packages fail closed through
// mineru_validate_real_artifact().
#include <cstdint>

#include <luisa/core/stl/string.h>
#include <luisa/core/stl/vector.h>

#include "ocr/models/qwen2_5_vl.h"
#include "ocr/package.h"
#include "ocr/runtime/p7_slice_config.h"
#include "ocr/runtime/scheduler.h"
#include "ocr/status.h"

namespace luisa::ocr {

struct MineruConfig {
    P7SliceConfig graph;
    P7TensorNames names;

    [[nodiscard]] Status validate() const;
};

// The declared SMALL SYNTHETIC MinerU geometry used by the vertical-slice
// tests. NOT a real checkpoint. Vision (NaViT-shaped): 2 layers, hidden 16,
// 2 heads, head_dim 8, patch 2, 2D RoPE (2 height + 2 width pairs); 2x2
// pixel-unshuffle connector; Qwen2 text: vocab 32, hidden 32, 3 layers, 4 Q /
// 2 KV heads, head_dim 8, adjacent-pair M-RoPE sections {2,1,1}.
[[nodiscard]] MineruConfig mineru_synthetic_config();

// Strict MinerU config parsing (synthetic contract schema). Accepts a JSON
// object with "model_type" (must contain "mineru" when present),
// "vision_config", "text_config", "connector", "mrope_section" ([t,h,w]).
// Fails closed on Qwen3 markers (qk_norm / interleaved rope), an adaptive-pool
// connector request (Paddle convention), MoE/GDN fields. The resulting text
// contract is ALWAYS Qwen2 adjacent-pair M-RoPE (never Qwen3 interleaved).
[[nodiscard]] Result<MineruConfig> parse_mineru_config(luisa::string_view json);

// Full expected tensor inventory (NaViT + pixel-unshuffle connector + Qwen2 text).
[[nodiscard]] luisa::vector<ExpectedTensor> mineru_expected_tensors(const MineruConfig &config);
[[nodiscard]] Status mineru_validate_tensors(luisa::span<const PackageTensor> tensors,
                                             luisa::span<const ExpectedTensor> expected);

// Real-artifact gate: ALWAYS fail-closed in Phase 7.
[[nodiscard]] Status mineru_validate_real_artifact(luisa::string_view checkpoint_revision,
                                                   luisa::string_view config_fingerprint);

// ---------------------------------------------------------------------------
// Host composition + public bridge (host-only; defined in the .cpp).
// MinerU supports plain + markdown + extract (strict) with CROSS-PAGE
// composition (pages ascending, reading order within each page). spotting is
// NOT validated for MinerU and fails closed. Malformed structured output is an
// error, never a plain-text fallback. Generative content: no confidence.
// ---------------------------------------------------------------------------
[[nodiscard]] Result<LayoutDocument> mineru_compose(const LayoutRunResult &run, LayoutOutputMode mode,
                                                    luisa::string_view extract_schema);

}// namespace luisa::ocr

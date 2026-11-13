#pragma once
// OCR engine -- TeleOCR graph contract (Phase 7). Host-safe header.
//
// TeleOCR is a DISTINCT graph, not a Qwen2.5-VL or Qwen3-VL alias:
//   * Qwen2.5-VL-DERIVED vision tower with WINDOW/GLOBAL attention (most
//     layers attend within a spatial window; selected global layers attend
//     the whole image), QKV bias, NO vision QK-Norm, 2D axial vision RoPE,
//     GELU-erf MLP;
//   * an MLP ALIGNER connector (spatial m x m token packing + LayerNorm +
//     2-layer GELU MLP) -- NOT Paddle's adaptive pool, NOT MinerU's
//     pixel-unshuffle;
//   * a Qwen3-0.6B-style text decoder: QK-Norm, NO QKV bias, GQA, INTERLEAVED
//     M-RoPE. A Qwen2 decoder is NEVER substituted for the Qwen3 decoder.
//   * an OPTIONAL layout/crop front end (the shared LayoutVlmExecutor) and a
//     camera/deformed-document input path (curved/folded/crumpled metadata +
//     preprocessing).
//
// IDENTITY (plan §6.1): the outer `model_type` string is NEVER identity.
// TeleOCR's outer type collides with other checkpoints, so a recipe match
// requires the EXACT tensor/config signature (tele_ocr_validate_signature).
//
// This header pulls NO api.h/model_recipe.h (host AttentionMaskKind clash) and
// NO ops_attention.h, so both host and device TUs may include it. The device
// graph is the shared runtime/p7_vlm_executor.*; host composition + the public
// bridge live in this module's .cpp (host-only).
//
// HONESTY (plan §1/§6): NO official TeleOCR checkpoint has been pinned,
// inspected, or run. Every dimension / tensor name here is a DECLARED SYNTHETIC
// CONTRACT exercised by the synthetic vertical slice. Real packages fail closed
// through tele_ocr_validate_real_artifact(). The official gates are in MODELS.md.
#include <cstdint>
#include <utility>

#include <luisa/core/stl/string.h>
#include <luisa/core/stl/vector.h>

#include "ocr/models/qwen2_5_vl.h"
#include "ocr/package.h"
#include "ocr/runtime/p7_slice_config.h"
#include "ocr/runtime/scheduler.h"
#include "ocr/status.h"

namespace luisa::ocr {

// ---------------------------------------------------------------------------
// Camera / deformed-document input contract
// ---------------------------------------------------------------------------

// Explicit deformation kind for the camera/deformed document path. This is
// METADATA carried through preprocessing; the synthetic slice applies a
// deterministic geometric warp so the curved/folded/crumpled paths are
// exercised and tested. It is NOT a claim of official dewarping parity.
enum class TeleDeformKind : uint32_t {
    flat = 0,
    curved = 1,
    folded = 2,
    crumpled = 3,
};

[[nodiscard]] luisa::string_view tele_deform_kind_name(TeleDeformKind kind) noexcept;

// Deformation metadata + the deterministic warp parameters. The warp is a
// host-side geometric remap applied to the source pixels BEFORE packing; the
// source-coordinate transform is preserved so output coordinates map back to
// the ORIGINAL (pre-warp) page.
struct TeleDeformSpec {
    TeleDeformKind kind{TeleDeformKind::flat};
    // Amplitude in source pixels (0 = identity). Bounded by validate().
    float amplitude{0.0f};

    [[nodiscard]] Status validate() const;
};

// Apply the declared deterministic warp to HWC uint8 source pixels. Returns
// the warped pixels (same size) plus the forward map provenance. flat is an
// identity copy. The warp is invertible-by-construction metadata; the synthetic
// tests verify the pixel remap against an independent host oracle.
struct TeleWarpedImage {
    luisa::vector<uint8_t> pixels;// HWC uint8, same size as source
    uint32_t width{0};
    uint32_t height{0};
    TeleDeformSpec spec;
};
[[nodiscard]] Result<TeleWarpedImage> tele_warp_image(luisa::span<const uint8_t> pixels,
                                                      uint32_t width, uint32_t height,
                                                      const TeleDeformSpec &spec);

// ---------------------------------------------------------------------------
// Full TeleOCR config
// ---------------------------------------------------------------------------

struct TeleOcrConfig {
    P7SliceConfig graph;
    P7TensorNames names;
    // Optional layout/crop front end. When false the model runs the direct
    // dense VLM path (one image -> one decode); when true the shared
    // LayoutVlmExecutor orchestrates overview -> crops -> region decode.
    bool layout_route{false};

    [[nodiscard]] Status validate() const;
};

// The declared SMALL SYNTHETIC TeleOCR geometry used by the vertical-slice
// tests. NOT a real checkpoint. Vision: window/global, 2 layers, hidden 16,
// 2 heads, head_dim 8, patch 2, window 2 patches, global layer {1}; aligner
// spatial merge 2; Qwen3 text: vocab 32, hidden 32, 3 layers, 4 Q / 2 KV heads,
// head_dim 8, interleaved MRoPE sections {2,1,1}.
[[nodiscard]] TeleOcrConfig tele_ocr_synthetic_config();

// Strict TeleOCR config parsing (synthetic contract schema). Accepts a JSON
// object with "model_type" (must contain "tele" when present -- but the outer
// type is NEVER sufficient for identity), "vision_config", "text_config",
// "connector", "mrope_section" ([t,h,w]), "layout_route" (bool). Fails closed
// on Qwen2.5 text markers (qkv_bias true / no qk_norm), MoE/GDN fields, and
// unknown structural fields. The resulting text contract is ALWAYS Qwen3
// (QK-Norm, no QKV bias, interleaved MRoPE).
[[nodiscard]] Result<TeleOcrConfig> parse_tele_ocr_config(luisa::string_view json);

// Full expected tensor inventory (vision + aligner + Qwen3 text decoder).
[[nodiscard]] luisa::vector<ExpectedTensor> tele_ocr_expected_tensors(const TeleOcrConfig &config);
[[nodiscard]] Status tele_ocr_validate_tensors(luisa::span<const PackageTensor> tensors,
                                               luisa::span<const ExpectedTensor> expected);

// Exact tensor/config signature gate. The outer model_type alone is NEVER
// identity: a signature must carry the checkpoint revision, config fingerprint,
// AND the exact canonical tensor-name set matching the declared inventory.
// Returns success only when every required name is present with no unexpected
// name and the identity fields are non-empty. (This is the synthetic contract
// gate; real artifacts still fail closed through the real-artifact gate.)
struct TeleOcrSignature {
    luisa::string checkpoint_revision;
    luisa::string config_fingerprint;
    luisa::vector<luisa::string> tensor_names;
};
[[nodiscard]] Status tele_ocr_validate_signature(const TeleOcrConfig &config, const TeleOcrSignature &signature);

// Real-artifact gate: ALWAYS fail-closed in Phase 7. No official TeleOCR
// checkpoint/config/reference has been qualified.
[[nodiscard]] Status tele_ocr_validate_real_artifact(luisa::string_view checkpoint_revision,
                                                     luisa::string_view config_fingerprint);

// ---------------------------------------------------------------------------
// Host composition + public bridge (host-only; declared here, defined in the
// .cpp which includes api.h). The device graph never includes api.h.
// ---------------------------------------------------------------------------

// Compose a layout run (the optional layout/crop route) into a LayoutDocument
// for the requested mode. TeleOCR supports plain + markdown; spotting/extract
// are NOT validated for TeleOCR and fail closed. Malformed structured output
// is an error, never a plain-text fallback. Generative content: no confidence.
[[nodiscard]] Result<LayoutDocument> tele_ocr_compose(const LayoutRunResult &run, LayoutOutputMode mode);

}// namespace luisa::ocr

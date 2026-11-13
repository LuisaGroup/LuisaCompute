#pragma once
// OCR engine -- GOT-OCR2.0 graph contract (Phase 8). Host-safe header.
//
// GOT-OCR2.0 is a DISTINCT dense VLM:
//   * a ViTDet/SAM vision encoder with WINDOWED/GLOBAL attention and a learned
//     decomposed RELATIVE-POSITION bias (NOT RoPE, NOT absolute positions);
//   * a LINEAR connector projecting vision tokens to the decoder hidden size;
//   * a Qwen-0.5B-style dense decoder (QKV bias, no QK-Norm, 1D RoPE, SwiGLU);
//   * plain / formatted / region / color-guided PROMPT modes and MULTI-CROP
//     composition with source-coordinate preservation.
//
// IDENTITY (plan §6.1): the outer `model_type` string is NEVER identity. A
// recipe match requires the EXACT tensor/config signature
// (got_ocr2_validate_signature). Real artifacts fail closed through
// got_ocr2_validate_real_artifact().
//
// This header pulls NO api.h/model_recipe.h and NO ops_attention.h, so both
// host and device TUs may include it. The device graph is the shared
// runtime/p7_vlm_executor.*; host prompt/composition lives in the .cpp.
//
// HONESTY (plan §1/§6): NO official GOT-OCR2.0 checkpoint has been pinned,
// inspected, or run. Every dimension / tensor name / prompt template here is a
// DECLARED SYNTHETIC CONTRACT. The official gates are in MODELS.md. No RTX 4060
// performance claim is made.
#include <cstdint>
#include <utility>

#include <luisa/core/stl/string.h>
#include <luisa/core/stl/vector.h>

#include "ocr/models/qwen2_5_vl.h"
#include "ocr/package.h"
#include "ocr/runtime/memory_budget.h"
#include "ocr/runtime/p7_slice_config.h"
#include "ocr/runtime/scheduler.h"
#include "ocr/status.h"

namespace luisa::ocr {

// ---------------------------------------------------------------------------
// Prompt modes (the GOT-OCR2.0 contract)
// ---------------------------------------------------------------------------

enum class GotPromptMode : uint32_t {
    plain = 0,     // "OCR: " -- raw text
    formatted = 1, // "OCR with format: " -- Markdown/LaTeX formatted text
    region = 2,    // "OCR with region: [x0,y0,x1,y1]" -- region-guided OCR
    color = 3,     // "OCR with color: <color>" -- color-guided OCR
};

[[nodiscard]] luisa::string_view got_prompt_mode_name(GotPromptMode mode) noexcept;

// A region box in NORMALIZED [0,1] source coordinates (x0,y0,x1,y1).
struct GotRegionBox {
    float x0{0.0f};
    float y0{0.0f};
    float x1{0.0f};
    float y1{0.0f};
    [[nodiscard]] Status validate() const;
};

// Build the EXACT GOT prompt string for a mode. Region requires a valid box;
// color requires a non-empty color name. Invalid guidance/mode is refused
// (never silently downgraded to plain). This is the declared synthetic prompt
// contract; the official apply_chat_template parity is an unmet gate.
[[nodiscard]] Result<luisa::string> got_ocr2_build_prompt(GotPromptMode mode,
                                                         const GotRegionBox &region,
                                                         luisa::string_view color);

// Parse a GOT prompt string back into its mode + guidance. Fails closed on a
// malformed/unknown prompt (never a silent plain fallback). Round-trips with
// got_ocr2_build_prompt for valid inputs.
struct GotParsedPrompt {
    GotPromptMode mode{GotPromptMode::plain};
    GotRegionBox region;
    luisa::string color;
};
[[nodiscard]] Result<GotParsedPrompt> got_ocr2_parse_prompt(luisa::string_view prompt);

// ---------------------------------------------------------------------------
// Source region transforms (normalized box <-> source pixels)
// ---------------------------------------------------------------------------

// Map a normalized [0,1] region box to integer source-pixel coordinates for an
// image of (width, height). The box is clamped to the image; a degenerate
// (zero-area) result is refused. Coordinates are preserved for composition.
struct GotPixelBox {
    uint32_t x0{0};
    uint32_t y0{0};
    uint32_t x1{0};
    uint32_t y1{0};
};
[[nodiscard]] Result<GotPixelBox> got_ocr2_region_to_pixels(const GotRegionBox &region,
                                                           uint32_t width, uint32_t height);

// ---------------------------------------------------------------------------
// Full GOT-OCR2.0 config
// ---------------------------------------------------------------------------

struct GotOcr2Config {
    P7SliceConfig graph;
    P7TensorNames names;
    // The decoder hidden size the linear connector projects to (must equal
    // graph.text.hidden_size). Kept explicit for the connector contract.
    [[nodiscard]] Status validate() const;
};

// The SMALL SYNTHETIC GOT-OCR2.0 slice. NOT a real checkpoint. Vision: 2-layer
// ViTDet/SAM, hidden 16, 2 heads, head_dim 8, patch 2, window 2 patches with a
// global layer, decomposed relative-position bias (window 4), NO vision RoPE,
// GELU-erf MLP. Connector: a plain linear projection (16 -> 32). Decoder: a
// small Qwen-style dense GQA geometry (3 layers, 4 Q / 2 KV, head_dim 8,
// hidden 32), QKV bias, no QK-Norm, 1D RoPE, SwiGLU.
[[nodiscard]] GotOcr2Config got_ocr2_synthetic_config();

// Strict GOT config parsing. Fails closed on MoE/GDN markers, vision RoPE
// (GOT uses relative bias), and unknown structural fields.
[[nodiscard]] Result<GotOcr2Config> parse_got_ocr2_config(luisa::string_view json);

// Full expected tensor inventory (vision + relative-bias tables + linear
// connector + Qwen decoder).
[[nodiscard]] luisa::vector<ExpectedTensor> got_ocr2_expected_tensors(const GotOcr2Config &config);
[[nodiscard]] Status got_ocr2_validate_tensors(luisa::span<const PackageTensor> tensors,
                                               luisa::span<const ExpectedTensor> expected);

// Exact tensor/config signature gate.
struct GotOcr2Signature {
    luisa::string checkpoint_revision;
    luisa::string config_fingerprint;
    luisa::vector<luisa::string> tensor_names;
};
[[nodiscard]] Status got_ocr2_validate_signature(const GotOcr2Config &config, const GotOcr2Signature &signature);

// Real-artifact gate: ALWAYS fail-closed in Phase 8.
[[nodiscard]] Status got_ocr2_validate_real_artifact(luisa::string_view checkpoint_revision,
                                                     luisa::string_view config_fingerprint);

// ---------------------------------------------------------------------------
// Memory preflight (host; no allocation). Reports the checked vision+decoder
// weight / workspace / KV requirement and fails BEFORE allocation when the
// budget is exceeded. GOT's synthetic slice is FP32-only (no W4A32 claim).
// ---------------------------------------------------------------------------

struct GotOcr2MemoryReport {
    uint64_t vision_weight_bytes{0};
    uint64_t decoder_weight_bytes{0};
    uint64_t workspace_bytes{0};
    uint64_t kv_cache_bytes{0};
    uint64_t total_device_bytes{0};
    uint64_t available_device_bytes{0};
    bool fits{false};
};

[[nodiscard]] Result<GotOcr2MemoryReport> got_ocr2_preflight(
    const GotOcr2Config &config, uint32_t max_seq_len, MemoryLimits limits);

// ---------------------------------------------------------------------------
// Multi-crop composition (host-only; defined in the .cpp).
// ---------------------------------------------------------------------------

// One crop's recognized text plus its source geometry (pixel box on the page).
struct GotCropResult {
    GotPixelBox box;
    luisa::string text;
    uint32_t reading_order{0};
    // Source page of the crop (multi-page composition keeps page provenance).
    uint32_t page_index{0};
};

// Compose multiple crop results into a single document in reading order. For
// plain mode the texts are joined in reading order; for formatted mode each
// crop's text is validated as Markdown (strict) before composition. Malformed
// formatted output is an ERROR, never a plain-text fallback. Source boxes are
// preserved on the composed lines/blocks. Generative content: no confidence.
[[nodiscard]] Result<LayoutDocument> got_ocr2_compose_crops(luisa::span<const GotCropResult> crops,
                                                           GotPromptMode mode);

}// namespace luisa::ocr

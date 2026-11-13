#pragma once
// OCR engine -- OvisOCR2 graph contract (Phase 9). Host-safe header.
//
// OvisOCR2 is a HYBRID GDN VLM, not a dense ViT+LLM alias:
//   * a ViT vision tower + spatial patch-merger connector (dynamic preprocessing);
//   * a Qwen3.5-0.8B-style HYBRID decoder whose layer schedule EXPLICITLY mixes
//     GDN layers (causal Conv1D + delta-rule recurrent state) and periodic full
//     GQA attention layers (contiguous KV). It is NEVER a KV-only transformer.
//
// The full 24-layer / 18-GDN / 6-full schedule and the hidden/head geometry are
// a RECIPE-DRIVEN DECLARED CONTRACT: a small synthetic slice may scale the
// hidden / channel dimensions for testing, but the full schedule is validated
// explicitly (ovis_ocr2_declared_decoder_config + ovis_ocr2_validate_schedule +
// signature gate). The executing graph is runtime/hybrid_gdn_executor.* (the
// HybridVlmGraph pairs the P7 vision tower with the HybridGdnExecutor).
//
// IDENTITY (plan §6.1): the outer `model_type` string is NEVER identity. A
// recipe match requires the EXACT tensor/config signature
// (ovis_ocr2_validate_signature). Real artifacts fail closed through
// ovis_ocr2_validate_real_artifact().
//
// This header pulls NO api.h/model_recipe.h (host AttentionMaskKind clash) and
// NO ops_attention.h / DSL headers, so both host and device TUs may include it.
//
// HONESTY (plan §1/§6): NO official OvisOCR2 / Qwen3.5 checkpoint has been
// pinned, inspected, or run. Every dimension / tensor name here is a DECLARED
// SYNTHETIC CONTRACT exercised by the synthetic vertical slice. The official
// gates are in MODELS.md. No RTX 4060 performance claim is made.
#include <cstdint>
#include <utility>

#include <luisa/core/stl/string.h>
#include <luisa/core/stl/vector.h>

#include "ocr/models/qwen2_5_vl.h"
#include "ocr/package.h"
#include "ocr/runtime/hybrid_gdn_config.h"
#include "ocr/runtime/memory_budget.h"
#include "ocr/runtime/p7_slice_config.h"
#include "ocr/runtime/scheduler.h"
#include "ocr/status.h"

namespace luisa::ocr {

// ---------------------------------------------------------------------------
// Full OvisOCR2 config
// ---------------------------------------------------------------------------
struct OvisOcr2Config {
    // Vision tower + connector + packing (P7 conventions). vision.text must
    // match the hybrid decoder hidden/vocab and use rope1d.
    P7SliceConfig vision;
    P7TensorNames vision_names;
    // Hybrid GDN decoder (explicit layer schedule).
    HybridGdnExecutorConfig decoder;
    HybridGdnTensorNames decoder_names;
    uint32_t max_seq_len{256};
    MemoryLimits memory;

    [[nodiscard]] Status validate() const;
};

// The DECLARED full Qwen3.5-0.8B-style hybrid decoder contract: 24 layers with
// a 3:1 GDN:full schedule (18 GDN + 6 full; full attention at every 4th layer,
// 0-based indices 3,7,11,15,19,23). The hidden/head/intermediate values are the
// declared contract geometry (NOT a verified checkpoint). Used to validate the
// 24/18/6 schedule explicitly even when the executing slice is scaled.
[[nodiscard]] HybridGdnExecutorConfig ovis_ocr2_declared_decoder_config();

// Validate a hybrid layer schedule against the declared 24/18/6 contract:
// exactly 24 layers, 18 GDN + 6 full, full at indices {3,7,11,15,19,23}.
[[nodiscard]] Status ovis_ocr2_validate_schedule(const luisa::vector<HybridLayerKind> &kinds);

// The SMALL SYNTHETIC OvisOCR2 slice used by the vertical-slice tests. NOT a
// real checkpoint. Vision: 2-layer ViT, hidden 16, 2 heads, head_dim 8, patch 2,
// 2D axial RoPE, bidirectional, GELU-erf MLP; spatial-merge 2 patch merger.
// Decoder: a SCALED hybrid geometry (6 layers = 4 GDN + 2 full, hidden 16) with
// the SAME GDN/full conventions as the full 24/18/6 contract. The full schedule
// is checked separately.
[[nodiscard]] OvisOcr2Config ovis_ocr2_synthetic_config();

// Strict OvisOCR2 config parsing (synthetic contract schema). Fails closed on
// MoE markers and unknown structural fields.
[[nodiscard]] Result<OvisOcr2Config> parse_ovis_ocr2_config(luisa::string_view json);

// Full expected tensor inventory (vision + connector + hybrid decoder).
[[nodiscard]] luisa::vector<ExpectedTensor> ovis_ocr2_expected_tensors(const OvisOcr2Config &config);
// Validate a package tensor table against the inventory (fp32 baseline).
[[nodiscard]] Status ovis_ocr2_validate_tensors(luisa::span<const PackageTensor> vision_tensors,
                                                luisa::span<const PackageTensor> decoder_tensors,
                                                const OvisOcr2Config &config);

// Exact tensor/config signature gate (outer model_type alone is never identity).
struct OvisOcr2Signature {
    luisa::string checkpoint_revision;
    luisa::string config_fingerprint;
    luisa::vector<luisa::string> tensor_names;
};
[[nodiscard]] Status ovis_ocr2_validate_signature(const OvisOcr2Config &config, const OvisOcr2Signature &signature);

// Real-artifact gate: ALWAYS fail-closed in Phase 9.
[[nodiscard]] Status ovis_ocr2_validate_real_artifact(luisa::string_view checkpoint_revision,
                                                      luisa::string_view config_fingerprint);

// ---------------------------------------------------------------------------
// Memory preflight (host; no allocation). Checked device-byte requirement for
// the hybrid decoder weights + workspace + hybrid cache (KV + GDN + Conv).
// ---------------------------------------------------------------------------
struct OvisOcr2MemoryReport {
    uint64_t decoder_weight_bytes{0};
    uint64_t workspace_bytes{0};
    uint64_t cache_bytes{0};
    uint64_t total_device_bytes{0};
    uint64_t available_device_bytes{0};
    bool fits{false};
};
[[nodiscard]] Result<OvisOcr2MemoryReport> ovis_ocr2_preflight_decoder(
    const OvisOcr2Config &config, uint32_t max_seq_len, MemoryLimits limits);

// ---------------------------------------------------------------------------
// Host composition (host-only; defined in the .cpp). OvisOCR2 supports plain +
// markdown; spotting/extract fail closed. Generative content carries NO
// confidence.
// ---------------------------------------------------------------------------
[[nodiscard]] Result<LayoutDocument> ovis_ocr2_compose(const LayoutRunResult &run, LayoutOutputMode mode);

}// namespace luisa::ocr

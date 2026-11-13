#pragma once
// OCR engine -- dots.mocr graph contract (Phase 8). Host-safe header.
//
// dots.mocr is a DISTINCT dense VLM, not a Qwen alias:
//   * a ViT vision tower + spatial patch-merger connector;
//   * a 28-layer dense GQA decoder with 12 query / 2 KV heads, RMSNorm, 1D
//     RoPE, and SwiGLU MLP. The 28-layer / 12-query / 2-KV geometry is a
//     RECIPE-DRIVEN contract: a small synthetic slice may scale the hidden /
//     layer dimensions for testing, but the full 28/12/2 contract is validated
//     explicitly (dots_mocr_declared_decoder_config + signature gate).
//   * a W4A32 quantized path (packed INT4 weights, FP32 activations and FP32
//     accumulation) for the memory-constrained 8 GB configuration. This is the
//     real packed-INT4 read path, never a relabeled FP32 path.
//
// IDENTITY (plan §6.1): the outer `model_type` string is NEVER identity. A
// recipe match requires the EXACT tensor/config signature
// (dots_mocr_validate_signature). Real artifacts fail closed through
// dots_mocr_validate_real_artifact().
//
// This header pulls NO api.h/model_recipe.h (host AttentionMaskKind clash) and
// NO ops_attention.h, so both host and device TUs may include it. The device
// graph is the shared runtime/p7_vlm_executor.* + dense_vlm_executor.*; host
// composition lives in this module's .cpp (host-only).
//
// HONESTY (plan §1/§6): NO official dots.mocr checkpoint has been pinned,
// inspected, or run. Every dimension / tensor name here is a DECLARED SYNTHETIC
// CONTRACT exercised by the synthetic vertical slice. The official gates are in
// MODELS.md. No RTX 4060 performance claim is made.
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
// Full dots.mocr config
// ---------------------------------------------------------------------------

struct DotsMocrConfig {
    P7SliceConfig graph;
    P7TensorNames names;
    // Per-linear decoder weight storage. fp32 = the portable correctness
    // baseline; int4_* = the W4A32 memory-constrained path. The embedding
    // table, norms, and QKV biases stay FP32; a quantized recipe ships a
    // standalone packed lm_head (tied embeddings are refused for W4A32).
    DenseLinearStorage linear_storage{DenseLinearStorage::fp32};
    uint32_t linear_group_size{0};// 64 or 128 when quantized

    [[nodiscard]] Status validate() const;
};

// The DECLARED full dots.mocr decoder contract: 28 layers, 12 query heads,
// 2 KV heads, RMSNorm / 1D RoPE / SwiGLU. The hidden / head_dim / intermediate
// dimensions are the declared contract geometry (NOT a verified checkpoint).
// This is used to validate the 28/12/2 contract explicitly even when the
// executing synthetic slice uses a scaled geometry.
[[nodiscard]] DenseDecoderConfig dots_mocr_declared_decoder_config();

// The SMALL SYNTHETIC dots.mocr slice used by the vertical-slice tests. NOT a
// real checkpoint. Vision: 2-layer ViT, hidden 16, 2 heads, head_dim 8, patch 2,
// 2D axial RoPE, bidirectional, GELU-erf MLP; spatial-merge 2 patch merger.
// Decoder: a SCALED dense GQA geometry (3 layers, 4 Q / 2 KV heads, head_dim 8,
// hidden 32) with the SAME RMSNorm / 1D-RoPE / SwiGLU conventions as the full
// 28/12/2 contract. The full contract is checked separately.
[[nodiscard]] DotsMocrConfig dots_mocr_synthetic_config();

// Strict dots.mocr config parsing (synthetic contract schema). Fails closed on
// MoE/GDN markers, Qwen3 QK-Norm text markers, and unknown structural fields.
[[nodiscard]] Result<DotsMocrConfig> parse_dots_mocr_config(luisa::string_view json);

// Full expected tensor inventory (vision + patch merger + dense decoder).
[[nodiscard]] luisa::vector<ExpectedTensor> dots_mocr_expected_tensors(const DotsMocrConfig &config);
// Validate a package tensor table against the inventory. `linear_storage`
// selects the allowed per-linear encoding (fp32 baseline vs packed INT4).
[[nodiscard]] Status dots_mocr_validate_tensors(luisa::span<const PackageTensor> tensors,
                                                luisa::span<const ExpectedTensor> expected,
                                                DenseLinearStorage linear_storage);

// Package-load gate (S1): validate BOTH component package tables on disk
// against the exact inventory + storage policy BEFORE any weight upload.
// `vision_tensors` must be entirely dense FP32; `decoder_tensors` follows the
// configured per-linear storage (only decoder linears may be packed INT4).
// This is called by the load path; it never uploads or allocates device memory.
[[nodiscard]] Status dots_mocr_validate_packages(const DotsMocrConfig &config,
                                                 luisa::span<const PackageTensor> vision_tensors,
                                                 luisa::span<const PackageTensor> decoder_tensors);

// Exact tensor/config signature gate (outer model_type alone is never identity).
struct DotsMocrSignature {
    luisa::string checkpoint_revision;
    luisa::string config_fingerprint;
    luisa::vector<luisa::string> tensor_names;
};
[[nodiscard]] Status dots_mocr_validate_signature(const DotsMocrConfig &config, const DotsMocrSignature &signature);

// Real-artifact gate: ALWAYS fail-closed in Phase 8.
[[nodiscard]] Status dots_mocr_validate_real_artifact(luisa::string_view checkpoint_revision,
                                                     luisa::string_view config_fingerprint);

// ---------------------------------------------------------------------------
// Memory preflight (host; no allocation). Computes the checked device-byte
// requirement for the decoder weights (FP32 or packed INT4) + workspace + KV
// cache and reports required vs available BEFORE any allocation. The vision
// tower / connector weights are accounted separately by the caller.
// ---------------------------------------------------------------------------

struct DotsMocrMemoryReport {
    uint64_t decoder_weight_bytes{0};
    uint64_t workspace_bytes{0};
    uint64_t kv_cache_bytes{0};
    uint64_t total_device_bytes{0};
    uint64_t available_device_bytes{0};
    bool fits{false};
};

// Preflight the dense decoder footprint for `config` at `max_seq_len` context.
// `limits.max_device_bytes` is the available budget (0 = no explicit limit).
// Returns a report; `fits` is false (and the Status is out_of_memory) when the
// requirement exceeds the budget. No device allocation occurs here.
[[nodiscard]] Result<DotsMocrMemoryReport> dots_mocr_preflight_decoder(
    const DotsMocrConfig &config, uint32_t max_seq_len, MemoryLimits limits);

// ---------------------------------------------------------------------------
// Host composition (host-only; defined in the .cpp which includes api.h).
// dots.mocr supports plain + markdown; spotting/extract fail closed.
// ---------------------------------------------------------------------------
[[nodiscard]] Result<LayoutDocument> dots_mocr_compose(const LayoutRunResult &run, LayoutOutputMode mode);

}// namespace luisa::ocr

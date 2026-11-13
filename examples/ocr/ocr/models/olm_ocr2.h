#pragma once
// OCR engine -- olmOCR-2 graph contract (Phase 8). Host-safe header.
//
// olmOCR-2 is a Qwen2.5-VL-7B-derived dense VLM:
//   * the COMPLETE Qwen2.5-VL encoder (window/global vision attention, 2D RoPE)
//     + spatial-merge RMSNorm merger + Qwen2.5-VL-7B decoder (QKV bias, no
//     QK-Norm, rotate-half M-RoPE, SwiGLU);
//   * ENGLISH-ONLY (no bilingual claim);
//   * INT4 (W4A32) is REQUIRED for the declared 8 GB path, with a STRICT memory
//     preflight that fails BEFORE allocation with a useful required-memory
//     report. Host offload is NOT implemented and fails closed as unsupported
//     (no performance claim).
//
// IDENTITY (plan §6.1): the outer `model_type` string is NEVER identity. A
// recipe match requires the EXACT tensor/config signature
// (olm_ocr2_validate_signature). Real artifacts fail closed through
// olm_ocr2_validate_real_artifact().
//
// This header pulls NO api.h/model_recipe.h and NO ops_attention.h, so both
// host and device TUs may include it. The device graph is the shared
// runtime/p7_vlm_executor.* + dense_vlm_executor.*; host composition lives in
// the .cpp.
//
// HONESTY (plan §1/§6): NO official olmOCR-2 checkpoint has been pinned,
// inspected, or run. Every dimension / tensor name here is a DECLARED SYNTHETIC
// CONTRACT. The 7B geometry is a declared contract, NOT a claim that a real 7B
// model fits or runs. The official gates are in MODELS.md. No RTX 4060
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

// Offload policy. Only `none` is implemented; `explicit_host` fails closed as
// unsupported (correct stream/event lifetime + accounting for a measured host
// offload is a later, separately gated arm). No performance claim is made.
enum class OlmOffloadPolicy : uint32_t {
    none = 0,
    explicit_host = 1,
};

[[nodiscard]] luisa::string_view olm_offload_policy_name(OlmOffloadPolicy policy) noexcept;

// ---------------------------------------------------------------------------
// Full olmOCR-2 config
// ---------------------------------------------------------------------------

struct OlmOcr2Config {
    P7SliceConfig graph;
    P7TensorNames names;
    // The declared 8 GB path REQUIRES INT4 (W4A32). FP32 is allowed only for the
    // small synthetic parity slice (never for the declared 7B 8 GB path).
    DenseLinearStorage linear_storage{DenseLinearStorage::int4_sym};
    uint32_t linear_group_size{128};
    OlmOffloadPolicy offload{OlmOffloadPolicy::none};
    // When true, validate() enforces the INT4 requirement (the declared 8 GB
    // path). The small synthetic parity slice sets this false to exercise FP32.
    bool require_int4{true};

    [[nodiscard]] Status validate() const;
};

// The DECLARED full Qwen2.5-VL-7B decoder contract geometry (28 layers, 28 Q /
// 4 KV heads, hidden 3584, head_dim 128). This is a DECLARED contract, NOT a
// verified checkpoint, and NOT a claim that a real 7B model fits in 8 GB.
[[nodiscard]] DenseDecoderConfig olm_ocr2_declared_decoder_config();

// The SMALL SYNTHETIC olmOCR-2 slice used by the vertical-slice tests. NOT a
// real checkpoint. Vision: 2-layer Qwen2.5-VL-style window/global tower, hidden
// 16, 2 heads, head_dim 8, patch 2, 2D axial RoPE, GELU-erf MLP; spatial-merge 2
// RMSNorm merger. Decoder: a small Qwen2.5-VL-style dense GQA geometry (3
// layers, 4 Q / 2 KV, head_dim 8, hidden 32), QKV bias, no QK-Norm, rotate-half
// M-RoPE (sections {2,1,1}), SwiGLU. require_int4 is false so the FP32 parity
// path is exercised; the INT4 path is tested separately.
[[nodiscard]] OlmOcr2Config olm_ocr2_synthetic_config();

// Strict olmOCR-2 config parsing. Fails closed on MoE/GDN markers, Qwen3
// QK-Norm markers, non-English language declarations, and unknown fields.
[[nodiscard]] Result<OlmOcr2Config> parse_olm_ocr2_config(luisa::string_view json);

// Full expected tensor inventory (vision + RMSNorm merger + Qwen2.5 decoder).
[[nodiscard]] luisa::vector<ExpectedTensor> olm_ocr2_expected_tensors(const OlmOcr2Config &config);
[[nodiscard]] Status olm_ocr2_validate_tensors(luisa::span<const PackageTensor> tensors,
                                               luisa::span<const ExpectedTensor> expected,
                                               DenseLinearStorage linear_storage);

// Package-load gate (S1): validate BOTH component package tables on disk
// against the exact inventory + storage policy BEFORE any weight upload.
// Vision/merger tensors must be entirely dense FP32; only decoder linears may
// be packed INT4. Called by the load path; never uploads/allocates.
[[nodiscard]] Status olm_ocr2_validate_packages(const OlmOcr2Config &config,
                                                luisa::span<const PackageTensor> vision_tensors,
                                                luisa::span<const PackageTensor> decoder_tensors);

// Exact tensor/config signature gate.
struct OlmOcr2Signature {
    luisa::string checkpoint_revision;
    luisa::string config_fingerprint;
    luisa::vector<luisa::string> tensor_names;
};
[[nodiscard]] Status olm_ocr2_validate_signature(const OlmOcr2Config &config, const OlmOcr2Signature &signature);

// Real-artifact gate: ALWAYS fail-closed in Phase 8.
[[nodiscard]] Status olm_ocr2_validate_real_artifact(luisa::string_view checkpoint_revision,
                                                     luisa::string_view config_fingerprint);

// English-only enforcement. Returns success for "en" or an empty hint; any
// other language (including "multi") is refused. This is the declared
// English-only contract; it never claims bilingual support.
[[nodiscard]] Status olm_ocr2_check_language(luisa::string_view language);

// ---------------------------------------------------------------------------
// Strict memory preflight (host; no allocation). The declared 8 GB path
// requires INT4 and fails BEFORE allocation with a useful required-memory
// report when the footprint exceeds the budget. Host offload is unsupported.
// ---------------------------------------------------------------------------

struct OlmOcr2MemoryReport {
    uint64_t vision_weight_bytes{0};
    uint64_t decoder_weight_bytes{0};
    uint64_t workspace_bytes{0};
    uint64_t kv_cache_bytes{0};
    uint64_t total_device_bytes{0};
    uint64_t available_device_bytes{0};
    bool fits{false};
};

// Preflight the FULL olmOCR-2 footprint (vision + merger + decoder weights +
// workspace + KV) for `config` at `max_seq_len` context. `limits` carries the
// available budget. Returns a report; the Status is out_of_memory when the
// requirement exceeds the budget (with required vs available bytes). When
// offload == explicit_host the call fails closed (unsupported) BEFORE any
// allocation. No device allocation occurs here.
[[nodiscard]] Result<OlmOcr2MemoryReport> olm_ocr2_preflight(
    const OlmOcr2Config &config, uint32_t max_seq_len, MemoryLimits limits);

// ---------------------------------------------------------------------------
// Host composition (host-only; defined in the .cpp). olmOCR-2 supports plain +
// markdown (English-only); spotting/extract fail closed.
// ---------------------------------------------------------------------------
[[nodiscard]] Result<LayoutDocument> olm_ocr2_compose(const LayoutRunResult &run, LayoutOutputMode mode);

}// namespace luisa::ocr

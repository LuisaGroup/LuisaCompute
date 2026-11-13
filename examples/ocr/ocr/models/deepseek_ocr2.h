#pragma once
// OCR engine -- DeepSeek-OCR-2 MoE graph contract (Phase 10). Host-safe header.
//
// DeepSeek-OCR-2 is a MoE VLM, NOT a dense ViT+LLM alias and NOT a generic
// causal transformer:
//   * SAM-style LOCAL perception (windowed vision blocks) -- a SEPARATE stage
//     from the visual LM encoder;
//   * BOTH convolutional downsamplers (two strided Conv2D stages);
//   * a Qwen2-style visual LM encoder carrying causal-flow QUERIES and the
//     EXACT custom visual/query block mask: visual tokens are bidirectional
//     among themselves; each causal-flow query sees ALL visual tokens and only
//     previous queries (including itself). This is NEVER a generic causal mask;
//   * a DeepSeek MoE text decoder: FP32 router, 64 routed experts / top-6 / two
//     shared dense experts, contiguous-KV causal attention.
//
// The 64 / top-6 / two-shared expert contract is EXACT in the full config and
// in the synthetic slice (hidden dims may scale; the expert count and top-k may
// NOT). The executing graph is runtime/moe_vlm_executor.*.
//
// IDENTITY (plan §6.1): the outer `model_type` string is NEVER identity. A
// recipe match requires the EXACT tensor/config signature
// (deepseek_ocr2_validate_signature). Real artifacts fail closed through
// deepseek_ocr2_validate_real_artifact().
//
// This header pulls NO api.h/model_recipe.h (host AttentionMaskKind clash) and
// NO ops_attention.h / DSL headers, so both host and device TUs may include it.
//
// HONESTY (plan §1/§6): NO official DeepSeek-OCR-2 checkpoint has been pinned,
// inspected, or run. Every dimension / tensor name here is a DECLARED SYNTHETIC
// CONTRACT exercised by the synthetic vertical slice. The official gates are in
// MODELS.md. No RTX 4060 performance or 8 GB-fit claim is made.
#include <cstdint>
#include <utility>

#include <luisa/core/stl/string.h>
#include <luisa/core/stl/vector.h>

#include "ocr/models/qwen2_5_vl.h"
#include "ocr/package.h"
#include "ocr/runtime/memory_budget.h"
#include "ocr/runtime/p7_slice_config.h"
#include "ocr/runtime/scheduler.h"// device-safe LayoutRunResult/LayoutDocument/output::DbRect
#include "ocr/runtime/workspace.h"// device-safe WorkspacePlan (no AttentionMaskKind)
#include "ocr/status.h"

namespace luisa::ocr {

// Exact DeepSeek MoE routing contract (full config AND synthetic slice).
inline constexpr uint32_t kDeepSeekNumRoutedExperts = 64u;
inline constexpr uint32_t kDeepSeekTopK = 6u;
inline constexpr uint32_t kDeepSeekNumSharedExperts = 2u;

// ---------------------------------------------------------------------------
// MoE text decoder config (attention geometry reused from the dense contract).
// ---------------------------------------------------------------------------
struct MoeDecoderConfig {
    DenseDecoderConfig decoder;// vocab/hidden/layers/GQA/head_dim/rope/norm
    uint32_t num_experts{kDeepSeekNumRoutedExperts};
    uint32_t top_k{kDeepSeekTopK};
    uint32_t num_shared_experts{kDeepSeekNumSharedExperts};
    uint32_t moe_intermediate_size{0};    // routed expert SwiGLU inter
    uint32_t shared_intermediate_size{0};// shared expert SwiGLU inter
    bool tie_word_embeddings{false};
    luisa::string component{"decoder"};
    uint32_t prefill_chunk_rows{256};
    MemoryLimits memory;

    [[nodiscard]] uint32_t num_layers() const noexcept { return decoder.num_layers; }
    [[nodiscard]] Status validate() const;
};

// Tensor-name contract for the MoE decoder. Attention/norm/embed names reuse the
// dense contract; the MLP is replaced by router + routed experts + shared
// experts. Linear weights are packed [K,N] (`kn`).
struct MoeDecoderTensorNames {
    DenseDecoderTensorNames dense;
    [[nodiscard]] Status validate() const;
    [[nodiscard]] luisa::string layer_prefix(uint32_t layer) const;
    // FP32 router projection [hidden, num_experts] (kn).
    [[nodiscard]] luisa::string router_weight(uint32_t layer) const;
    // Routed expert e: gate/up [hidden, moe_inter], down [moe_inter, hidden].
    [[nodiscard]] luisa::string expert_gate(uint32_t layer, uint32_t e) const;
    [[nodiscard]] luisa::string expert_up(uint32_t layer, uint32_t e) const;
    [[nodiscard]] luisa::string expert_down(uint32_t layer, uint32_t e) const;
    // Shared expert s (dense, applied to every token).
    [[nodiscard]] luisa::string shared_gate(uint32_t layer, uint32_t s) const;
    [[nodiscard]] luisa::string shared_up(uint32_t layer, uint32_t s) const;
    [[nodiscard]] luisa::string shared_down(uint32_t layer, uint32_t s) const;
};

// ---------------------------------------------------------------------------
// Visual path config (SAM perception + two conv downsamplers + visual LM
// encoder with causal-flow queries + connector). Explicit, never inferred.
// ---------------------------------------------------------------------------

// One strided Conv2D downsampler (NCHW). Reduces the spatial grid by `stride`.
struct DeepSeekConvDownsampler {
    uint32_t out_channels{0};
    uint32_t kernel{3};
    uint32_t stride{2};
    uint32_t pad{1};
    [[nodiscard]] Status validate() const;
};

struct DeepSeekVisionConfig {
    // SAM-style local perception: windowed ViT blocks (P7 window_global mask).
    P7VisionConfig perception;
    // BOTH convolutional downsamplers, applied in order after perception.
    DeepSeekConvDownsampler down1;
    DeepSeekConvDownsampler down2;
    // Qwen2-style visual LM encoder (pre-LN, QKV bias, custom visual/query mask).
    uint32_t encoder_layers{0};
    uint32_t encoder_intermediate_size{0};
    float encoder_norm_eps{1e-6f};
    bool encoder_qkv_bias{true};
    // Causal-flow query count (learned query embeddings appended to visual rows).
    uint32_t num_queries{0};
    // Connector: LayerNorm + 2-layer GELU MLP -> text hidden.
    uint32_t connector_intermediate_size{0};
    float connector_norm_eps{1e-6f};
    // Normalization (RGB, (pixel/255 - mean)/std).
    luisa::vector<float> mean{0.5f, 0.5f, 0.5f};
    luisa::vector<float> std_dev{0.5f, 0.5f, 0.5f};
    uint32_t max_vision_patches{4096};

    // The visual LM encoder hidden == perception hidden (the downsamplers may
    // change channels; the encoder operates on down2.out_channels).
    [[nodiscard]] uint32_t encoder_hidden() const noexcept { return down2.out_channels; }
    [[nodiscard]] Status validate() const;
};

// Visual tensor-name contract. Perception reuses P7 visual names; the
// downsamplers / encoder / queries / connector add DeepSeek-specific names.
struct DeepSeekVisionTensorNames {
    P7TensorNames perception;// visual_prefix "visual."
    [[nodiscard]] Status validate() const;
    [[nodiscard]] luisa::string down_weight(uint32_t which) const;// which = 1 or 2
    [[nodiscard]] luisa::string down_bias(uint32_t which) const;
    [[nodiscard]] luisa::string encoder_prefix(uint32_t layer) const;
    [[nodiscard]] luisa::string encoder_norm1_weight(uint32_t layer) const;
    [[nodiscard]] luisa::string encoder_norm1_bias(uint32_t layer) const;
    [[nodiscard]] luisa::string encoder_norm2_weight(uint32_t layer) const;
    [[nodiscard]] luisa::string encoder_norm2_bias(uint32_t layer) const;
    [[nodiscard]] luisa::string encoder_q_proj(uint32_t layer) const;
    [[nodiscard]] luisa::string encoder_k_proj(uint32_t layer) const;
    [[nodiscard]] luisa::string encoder_v_proj(uint32_t layer) const;
    [[nodiscard]] luisa::string encoder_o_proj(uint32_t layer) const;
    [[nodiscard]] luisa::string encoder_q_bias(uint32_t layer) const;
    [[nodiscard]] luisa::string encoder_k_bias(uint32_t layer) const;
    [[nodiscard]] luisa::string encoder_v_bias(uint32_t layer) const;
    [[nodiscard]] luisa::string encoder_o_bias(uint32_t layer) const;
    [[nodiscard]] luisa::string encoder_fc1(uint32_t layer) const;
    [[nodiscard]] luisa::string encoder_fc1_bias(uint32_t layer) const;
    [[nodiscard]] luisa::string encoder_fc2(uint32_t layer) const;
    [[nodiscard]] luisa::string encoder_fc2_bias(uint32_t layer) const;
    [[nodiscard]] luisa::string query_embed() const;
    [[nodiscard]] luisa::string connector_ln_weight() const;
    [[nodiscard]] luisa::string connector_ln_bias() const;
    [[nodiscard]] luisa::string connector_fc1() const;
    [[nodiscard]] luisa::string connector_fc1_bias() const;
    [[nodiscard]] luisa::string connector_fc2() const;
    [[nodiscard]] luisa::string connector_fc2_bias() const;
};

// ---------------------------------------------------------------------------
// Full DeepSeek-OCR-2 config.
// ---------------------------------------------------------------------------
struct DeepSeekOcr2Config {
    DeepSeekVisionConfig vision;
    DeepSeekVisionTensorNames vision_names;
    MoeDecoderConfig decoder;
    MoeDecoderTensorNames decoder_names;
    // Packing for the perception patch grid (native-resolution, aspect-preserving).
    P7PackingSpec packing;
    uint32_t max_seq_len{256};
    MemoryLimits memory;
    [[nodiscard]] Status validate() const;
};

// The DECLARED full MoE decoder contract: exact 64 routed / top-6 / two-shared.
// The hidden/head/intermediate values are a consistent DECLARED geometry (NOT a
// verified checkpoint) used to validate the expert contract explicitly.
[[nodiscard]] MoeDecoderConfig deepseek_ocr2_declared_decoder_config();

// Validate the exact expert contract: 64 routed, top-6, two shared. Scaled
// hidden dims are allowed; a scaled expert count / top-k is REFUSED.
[[nodiscard]] Status deepseek_ocr2_validate_expert_contract(const MoeDecoderConfig &config);

// The SMALL SYNTHETIC DeepSeek-OCR-2 slice used by the vertical-slice tests. NOT
// a real checkpoint. Scaled hidden dims, but the EXACT 64/6/2 expert contract.
[[nodiscard]] DeepSeekOcr2Config deepseek_ocr2_synthetic_config();

// Strict synthetic config parsing. Fails closed on unknown structural fields.
[[nodiscard]] Result<DeepSeekOcr2Config> parse_deepseek_ocr2_config(luisa::string_view json);

// Full expected tensor inventory (vision + MoE decoder), fp32 baseline.
[[nodiscard]] luisa::vector<ExpectedTensor> deepseek_ocr2_vision_expected_tensors(const DeepSeekOcr2Config &config);
[[nodiscard]] luisa::vector<ExpectedTensor> deepseek_ocr2_decoder_expected_tensors(const DeepSeekOcr2Config &config);
[[nodiscard]] Status deepseek_ocr2_validate_tensors(luisa::span<const PackageTensor> vision_tensors,
                                                    luisa::span<const PackageTensor> decoder_tensors,
                                                    const DeepSeekOcr2Config &config);

// Exact tensor/config signature gate (outer model_type alone is never identity).
struct DeepSeekOcr2Signature {
    luisa::string checkpoint_revision;
    luisa::string config_fingerprint;
    luisa::vector<luisa::string> tensor_names;
};
[[nodiscard]] Status deepseek_ocr2_validate_signature(const DeepSeekOcr2Config &config,
                                                      const DeepSeekOcr2Signature &signature);

// Real-artifact gate: ALWAYS fail-closed in Phase 10 (unvalidated/planned).
[[nodiscard]] Status deepseek_ocr2_validate_real_artifact(luisa::string_view checkpoint_revision,
                                                          luisa::string_view config_fingerprint);

// ---------------------------------------------------------------------------
// Memory preflight (host; no allocation). Checked device-byte requirement for
// vision + MoE decoder weights + workspaces + KV cache. Either supports a
// measured synthetic configuration or fails BEFORE allocation with
// required-vs-available details. No 8 GB claim based on estimates.
// ---------------------------------------------------------------------------
struct DeepSeekOcr2MemoryReport {
    uint64_t vision_weight_bytes{0};
    uint64_t decoder_weight_bytes{0};
    uint64_t workspace_bytes{0};
    uint64_t kv_cache_bytes{0};
    uint64_t total_device_bytes{0};
    uint64_t available_device_bytes{0};
    bool fits{false};
};
[[nodiscard]] Result<DeepSeekOcr2MemoryReport> deepseek_ocr2_preflight(
    const DeepSeekOcr2Config &config, uint32_t max_seq_len, MemoryLimits limits);

// EXACT workspace arena plan shared by the executor's reserve_workspace and the
// memory preflight, so the preflighted workspace bytes EQUAL the measured arena
// capacity (no estimate). Host-safe; no allocation. Region order/sizes mirror
// runtime/moe_vlm_executor.cpp exactly.
[[nodiscard]] Result<WorkspacePlan> deepseek_ocr2_workspace_plan(const DeepSeekOcr2Config &config);

// ---------------------------------------------------------------------------
// Grounded Markdown adapter (host-only; defined in the .cpp). DeepSeek-OCR-2
// supports plain + grounded markdown; spotting/extract fail closed. Generative
// content carries NO confidence. Grounding tokens (<|ref|>...<|/ref|> and
// <|det|>...<|/det|>) are parsed into source coordinates; invalid grounding
// fails closed (never silently dropped).
// ---------------------------------------------------------------------------
// Grounded Markdown adapter (host-only; defined in the .cpp). DeepSeek-OCR-2
// supports plain + grounded markdown; spotting/extract fail closed. Generative
// content carries NO confidence. Grounding tokens (<|ref|>...<|/ref|> and
// <|det|>...<|/det|>) are parsed into source coordinates; invalid grounding
// fails closed (never silently dropped). LayoutRunResult / LayoutOutputMode /
// LayoutDocument are the device-safe scheduler.h types (included above).
[[nodiscard]] Result<LayoutDocument> deepseek_ocr2_compose(const LayoutRunResult &run, LayoutOutputMode mode);

// Parse a grounded-Markdown string into blocks with source coordinates. The
// declared synthetic grounding format is:
//   <|ref|>text<|/ref|><|det|>[[x0,y0,x1,y1], ...]<|/det|>
// where coordinates are integers in the CLOSED [0, 1000] normalized page space
// (1000 maps to the full page edge; trailing commas are rejected). `page_width`/
// `page_height` scale them to source pixels. Invalid/unclosed grounding or
// out-of-range coordinates fail closed (invalid_argument). Plain Markdown
// (no grounding) is accepted and parsed by the strict Markdown parser.
struct DeepSeekGroundedBlock {
    luisa::string text;
    // Source boxes in page PIXEL space (empty = ungrounded text block).
    luisa::vector<output::DbRect> boxes;
    bool grounded{false};
};
[[nodiscard]] Result<luisa::vector<DeepSeekGroundedBlock>> deepseek_ocr2_parse_grounded(
    luisa::string_view text, uint32_t page_width, uint32_t page_height);

}// namespace luisa::ocr

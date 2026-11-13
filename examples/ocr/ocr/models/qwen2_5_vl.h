#pragma once
// OCR engine -- SHARED dense decoder components (Phase 3). Host-only.
//
// This module owns the recipe-driven dense-decoder geometry, config parsing,
// and exact tensor-name/layout validation shared by Qwen2.5-VL-derived OCR
// models (and the synthetic test decoder). It is NOT a Qwen2.5-VL model:
// there is no vision tower, no M-RoPE wiring, and no qualification here.
//
// Identity boundary (plan §3/§12): Qwen2.5-VL and Qwen3-VL are DIFFERENT
// decoders. Qwen2.5-VL text decoders use bias-bearing QKV projections, plain
// 1D-RoPE text positions (M-RoPE is a multimodal concern owned by the model
// phase), no QK-Norm, and `model.layers.*` tensor names. Qwen3-VL adds
// QK-Norm, drops QKV bias, and uses interleaved M-RoPE + DeepStack; its
// contract lives in models/qwen3_vl.* (Phase 4) and must NEVER be routed
// through this module. parse_qwen2_5_text_config() fails closed on Qwen3
// markers instead of silently accepting them.
//
// Two tensor-name prefixes are supported explicitly:
//   "model."        -- Qwen2.5-VL checkpoints (language tensors under model.*)
//   "language_model." -- wrappers that nest the text decoder (e.g. the
//                        Qwen2.5-VL HF `model.language_model.*` layout).
// The prefix is a declared contract field, never guessed from `model_type`.
//
// Everything here is host metadata/validation. Actual dimensions always come
// from the config/package and are checked against the tensors; no dimension
// is hardcoded to one checkpoint, and no claim is made about olmOCR/TeleOCR
// or any other real model.
#include <cstdint>

#include <luisa/core/stl/string.h>
#include <luisa/core/stl/vector.h>

#include "ocr/package.h"
#include "ocr/runtime/tensor.h"
#include "ocr/status.h"

namespace luisa::ocr {

// RoPE pair layout of the dense decoder's 1D text positions. This is an
// EXPLICIT contract field, never inferred: the two conventions rotate
// different element pairs and are NOT interchangeable.
//   adjacent    : pairs (2d, 2d+1) -- GPT-NeoX-"interleaved"/synthetic style.
//   rotate_half : pairs (d, d+head_dim/2) -- the Qwen2.5-VL HF text-decoder
//                 convention (HF rotate_half). parse_qwen2_5_text_config
//                 selects this; a synthetic adjacent-pair decoder must NOT be
//                 described as Qwen2.5.
enum class RopePairLayout : uint32_t {
    adjacent = 0,
    rotate_half = 1,
};

[[nodiscard]] luisa::string_view rope_pair_layout_name(RopePairLayout layout) noexcept;

// Additive Phase 8: explicit per-linear weight STORAGE policy for the shared
// dense decoder. The FP32 baseline (fp32) is the default and reproduces the
// exact Phase 3/4/7 behavior. int4_sym / int4_asym are the grouped W4A32 path:
// PACKED INT4 code reads with FP32 activations and FP32 accumulation (Phase 1/2
// quantization semantics, dequantization inside the operator kernels). This is
// NEVER a relabeled FP32 path. Embeddings, norms, and QKV biases stay FP32.
// FP16/BF16 arithmetic and W4A16 are separate later arms and are refused.
// Defined here (device-safe) so both the device executor and the host model
// modules can reference it without the host AttentionMaskKind clash.
enum class DenseLinearStorage : uint32_t {
    fp32 = 0,
    int4_sym = 1,
    int4_asym = 2,
};

[[nodiscard]] luisa::string_view dense_linear_storage_name(DenseLinearStorage storage) noexcept;

// Additive Phase 8 (S1): is `name` a DECODER linear B operand eligible for the
// W4A32 storage policy? Only tensors under the decoder `prefix` (the packed
// q/k/v/o/gate/up/down projections) or the decoder lm_head qualify. Vision /
// connector tensors (e.g. visual.blocks.*.attn.q_proj.weight) NEVER qualify,
// even though they share projection suffixes; they must stay dense FP32.
[[nodiscard]] bool dense_linear_is_quantizable(luisa::string_view name,
                                               luisa::string_view decoder_prefix,
                                               luisa::string_view lm_head_name) noexcept;

// Geometry of ONE dense autoregressive decoder (shared by executor + cache).
// All fields must be concrete (nonzero) before use; validate() enforces the
// structural rules the Phase 3 kernels rely on.
struct DenseDecoderConfig {
    uint32_t vocab_size{0};
    uint32_t hidden_size{0};
    uint32_t num_layers{0};
    uint32_t num_attention_heads{0};
    uint32_t num_kv_heads{0};
    uint32_t head_dim{0};// explicit; must satisfy heads*head_dim == hidden
    uint32_t intermediate_size{0};
    float rms_norm_eps{1e-6f};
    float rope_theta{10000.0f};
    // Explicit RoPE pair layout (see RopePairLayout). Defaults to the
    // synthetic adjacent convention; Qwen2.5-style configs set rotate_half.
    RopePairLayout rope_pair_layout{RopePairLayout::adjacent};
    uint32_t max_position_embeddings{0};// context ceiling from the config
    // Declared generation defaults (from generation_config.json semantics).
    // 0 = none declared. These are data for the decode controller, never a
    // support claim.
    luisa::vector<uint32_t> eos_token_ids;
    uint32_t bos_token_id{0};
    bool has_bos{false};

    [[nodiscard]] uint32_t q_dim() const noexcept { return num_attention_heads * head_dim; }
    [[nodiscard]] uint32_t kv_dim() const noexcept { return num_kv_heads * head_dim; }
    [[nodiscard]] Status validate() const;
};

// Tensor-name contract for the shared dense decoder. `prefix` is either
// "model." or "language_model." (validated); embed/norm/lm-head names are
// derived from it. Weights are stored PACKED [K,N] (layout tag `kn`) for
// every linear projection -- the Phase 1 packer transposes HF [N,K] weights.
struct DenseDecoderTensorNames {
    luisa::string prefix{"model."};

    [[nodiscard]] Status validate() const;
    [[nodiscard]] luisa::string embed_tokens() const;
    [[nodiscard]] luisa::string final_norm() const;
    // LM head; when tied it aliases embed_tokens() and may be absent from the
    // package (the executor reuses the embedding table).
    [[nodiscard]] luisa::string lm_head() const;
    [[nodiscard]] luisa::string layer_prefix(uint32_t layer) const;
    [[nodiscard]] luisa::string input_layernorm(uint32_t layer) const;
    [[nodiscard]] luisa::string post_attention_layernorm(uint32_t layer) const;
    [[nodiscard]] luisa::string q_proj(uint32_t layer) const;
    [[nodiscard]] luisa::string k_proj(uint32_t layer) const;
    [[nodiscard]] luisa::string v_proj(uint32_t layer) const;
    [[nodiscard]] luisa::string q_bias(uint32_t layer) const;
    [[nodiscard]] luisa::string k_bias(uint32_t layer) const;
    [[nodiscard]] luisa::string v_bias(uint32_t layer) const;
    [[nodiscard]] luisa::string o_proj(uint32_t layer) const;
    [[nodiscard]] luisa::string gate_proj(uint32_t layer) const;
    [[nodiscard]] luisa::string up_proj(uint32_t layer) const;
    [[nodiscard]] luisa::string down_proj(uint32_t layer) const;
    // Qwen3 QK-Norm weights (per-head RMSNorm over head_dim). Additive: the
    // Qwen2.5 inventory does NOT include these; only a Qwen3-configured
    // executor resolves them. Names are shared so both contracts agree.
    [[nodiscard]] luisa::string q_norm(uint32_t layer) const;
    [[nodiscard]] luisa::string k_norm(uint32_t layer) const;
};

// One expected tensor: exact name, exact shape, expected layout tag.
struct ExpectedTensor {
    luisa::string name;
    luisa::vector<uint64_t> shape;
    TensorLayout layout{TensorLayout::row_major};
    bool required{true};
};

// Validate a decoder-component package table against the expected inventory
// under an explicit per-linear storage policy (fp32 baseline or packed INT4).
// `decoder_prefix`/`lm_head_name` restrict quantizable linears (S1); every
// other tensor must be dense FP32. Unexpected names are rejected. This is the
// package-load gate the model load paths call BEFORE any weight upload.
[[nodiscard]] Status validate_dense_decoder_tensors_storage(
    luisa::span<const PackageTensor> tensors,
    luisa::span<const ExpectedTensor> expected,
    DenseLinearStorage linear_storage,
    luisa::string_view decoder_prefix,
    luisa::string_view lm_head_name);

// Full expected tensor inventory for `config` under `names`. QKV biases are
// required (Qwen2.5 contract); `tie_word_embeddings` makes lm_head optional.
// The standalone lm_head is a packed linear B operand [hidden, vocab] (K=N
// hidden, N=vocab), layout `kn` -- NOT [vocab, hidden].
[[nodiscard]] luisa::vector<ExpectedTensor> dense_decoder_expected_tensors(
    const DenseDecoderConfig &config, const DenseDecoderTensorNames &names,
    bool tie_word_embeddings);

// Validate a single installed device weight against its exact expected
// tensor: shape, layout tag, AND element count must all match before the
// executor binds a view. This is the per-weight gate the executor uses (the
// package-level validate_dense_decoder_tensors checks the on-disk table; this
// checks the resolved device descriptor). `weight_name` is used in errors.
[[nodiscard]] Status validate_decoder_weight_desc(const TensorDesc &logical,
                                                  uint64_t element_count,
                                                  const ExpectedTensor &expected);

// Validate a package's tensor table against the expected inventory: every
// required tensor present with EXACT shape + layout tag; unexpected decoder
// tensors rejected (vision tensors are allowed only outside the decoder
// prefix contract -- callers pass the decoder component's tensor list).
// Encoding must be dense fp32 for the Phase 3 baseline (packed/quantized
// weights are a Phase 4+ matrix; fail closed here, never reinterpret).
[[nodiscard]] Status validate_dense_decoder_tensors(
    luisa::span<const PackageTensor> tensors,
    luisa::span<const ExpectedTensor> expected);

// Strict Qwen2.5 text-config parsing from the `text_config` object (or a
// flat Qwen2 decoder config). Required keys: vocab_size, hidden_size,
// num_hidden_layers, num_attention_heads, num_key_value_heads,
// intermediate_size, max_position_embeddings. Optional with explicit
// defaults: head_dim (hidden/heads), rms_norm_eps (1e-6), rope_theta (10000),
// eos_token_id (scalar or list), bos_token_id. The resulting config selects
// RopePairLayout::rotate_half (the Qwen2.5 HF text convention).
// FAILS CLOSED on Qwen3-VL markers (any "qwen3" model_type or explicit qk_norm
// key) and on MoE/GDN fields it does not implement. A merely present but
// DISABLED sliding window (use_sliding_window=false, or sliding_window=null)
// is accepted because the ordinary dense path is exact; an ENABLED sliding
// window remains unsupported.
[[nodiscard]] Result<DenseDecoderConfig> parse_qwen2_5_text_config(luisa::string_view json);

}// namespace luisa::ocr
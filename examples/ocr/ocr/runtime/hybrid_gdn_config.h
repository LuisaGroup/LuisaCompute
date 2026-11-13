#pragma once
// OCR engine -- hybrid GDN decoder CONFIG + tensor-name contract (Phase 9).
// HOST-SAFE / DEVICE-SAFE plain data header: pulls NO api.h/model_recipe.h
// (host AttentionMaskKind clash) and NO ops_attention.h / DSL headers, so both
// host model modules and device executor TUs may include it. The executor that
// consumes this data lives in runtime/hybrid_gdn_executor.*; the GDN operator
// kernels live in ops/ops_gdn.*.
//
// HONESTY (plan §1/§6): every dimension here is a DECLARED SYNTHETIC CONTRACT.
// No official checkpoint is pinned; real artifacts fail closed through the
// per-model gates. See MODELS.md.
#include <cstdint>

#include <luisa/core/stl/string.h>
#include <luisa/core/stl/vector.h>

#include "ocr/models/qwen2_5_vl.h"// DenseDecoderConfig / DenseDecoderTensorNames / ExpectedTensor
#include "ocr/package.h"
#include "ocr/runtime/memory_budget.h"
#include "ocr/status.h"

namespace luisa::ocr {

// One decoder layer's kind. The schedule is EXPLICIT data, never inferred.
enum class HybridLayerKind : uint32_t {
    gdn = 0,
    full_attention = 1,
};
[[nodiscard]] luisa::string_view hybrid_layer_kind_name(HybridLayerKind kind) noexcept;

// Geometry of one GDN layer family (shared by ops + cache + executor). The GDN
// math contract (equations/layouts) is declared in ops/ops_gdn.h.
struct GdnLayerConfig {
    uint32_t num_heads{0};
    uint32_t head_dim{0};// key dim == value dim == head_dim
    uint32_t conv_kernel_size{0};// >= 2 (ring holds K-1 taps)
    float l2_eps{1e-6f};         // L2 normalization floor
    float norm_eps{1e-6f};       // gated RMSNorm epsilon
    float softplus_threshold{20.0f};

    [[nodiscard]] uint32_t qk_dim() const noexcept { return num_heads * head_dim; }
    [[nodiscard]] uint32_t conv_channels() const noexcept { return qk_dim(); }
    [[nodiscard]] uint32_t ring_width() const noexcept {
        return conv_kernel_size == 0u ? 0u : conv_kernel_size - 1u;
    }
    [[nodiscard]] Status validate() const;
};

// Hybrid cache geometry (plan §7.2 items 1/3/4/5). Capacities are checked from
// the ACTUAL geometry; paged KV stays unsupported.
struct HybridCacheConfig {
    uint32_t num_full_layers{0};
    uint32_t num_gdn_layers{0};
    uint32_t batch{1};
    uint32_t num_kv_heads{0};
    uint32_t head_dim{0};
    uint32_t max_seq_len{0};
    uint32_t gdn_heads{0};
    uint32_t gdn_head_dim{0};
    uint32_t conv_kernel_size{0};
    uint32_t dtype_bytes{4};

    [[nodiscard]] uint32_t gdn_qk_dim() const noexcept { return gdn_heads * gdn_head_dim; }
    [[nodiscard]] uint32_t conv_ring_width() const noexcept {
        return conv_kernel_size == 0u ? 0u : conv_kernel_size - 1u;
    }
    [[nodiscard]] Status validate() const;
    [[nodiscard]] Result<uint64_t> kv_bytes() const;
    [[nodiscard]] Result<uint64_t> recurrent_bytes() const;
    [[nodiscard]] Result<uint64_t> conv_bytes() const;
    [[nodiscard]] Result<uint64_t> metadata_bytes() const;
    [[nodiscard]] Result<uint64_t> total_bytes() const;
};

// Hybrid executor configuration.
struct HybridGdnExecutorConfig {
    DenseDecoderConfig decoder;// shared vocab/hidden/layers/GQA/MLP/norm/rope
    GdnLayerConfig gdn;        // GDN heads/head_dim/conv kernel/eps
    // EXPLICIT per-layer schedule; size == decoder.num_layers. At least one
    // GDN layer is required (a pure-dense decoder uses DenseVlmExecutor).
    luisa::vector<HybridLayerKind> layer_kinds;
    bool tie_word_embeddings{false};
    bool qkv_bias{true};
    bool qk_norm{false};
    luisa::string component{"decoder"};
    uint32_t prefill_chunk_rows{256};
    MemoryLimits memory;

    [[nodiscard]] uint32_t num_full_layers() const noexcept;
    [[nodiscard]] uint32_t num_gdn_layers() const noexcept;
    [[nodiscard]] Status validate() const;
};

// Tensor-name contract for the hybrid decoder. Full-attention layers reuse the
// dense names; GDN layers add the GDN projection / conv / gate / norm tensors.
// The per-layer input/post norms and the SwiGLU MLP (gate/up/down) are SHARED
// with the dense names for BOTH layer kinds. Linear weights are packed [K,N].
struct HybridGdnTensorNames {
    DenseDecoderTensorNames dense;
    [[nodiscard]] Status validate() const;
    [[nodiscard]] luisa::string layer_prefix(uint32_t layer) const;
    [[nodiscard]] luisa::string gdn_q_proj(uint32_t layer) const;
    [[nodiscard]] luisa::string gdn_k_proj(uint32_t layer) const;
    [[nodiscard]] luisa::string gdn_v_proj(uint32_t layer) const;
    [[nodiscard]] luisa::string gdn_z_proj(uint32_t layer) const;
    [[nodiscard]] luisa::string gdn_a_proj(uint32_t layer) const;
    [[nodiscard]] luisa::string gdn_b_proj(uint32_t layer) const;
    [[nodiscard]] luisa::string gdn_o_proj(uint32_t layer) const;
    [[nodiscard]] luisa::string gdn_conv_q_weight(uint32_t layer) const;
    [[nodiscard]] luisa::string gdn_conv_q_bias(uint32_t layer) const;
    [[nodiscard]] luisa::string gdn_conv_k_weight(uint32_t layer) const;
    [[nodiscard]] luisa::string gdn_conv_k_bias(uint32_t layer) const;
    [[nodiscard]] luisa::string gdn_conv_v_weight(uint32_t layer) const;
    [[nodiscard]] luisa::string gdn_conv_v_bias(uint32_t layer) const;
    [[nodiscard]] luisa::string gdn_a_log(uint32_t layer) const;
    [[nodiscard]] luisa::string gdn_dt_bias(uint32_t layer) const;
    [[nodiscard]] luisa::string gdn_norm_weight(uint32_t layer) const;
};

// Full expected tensor inventory for the hybrid decoder (fp32 baseline).
[[nodiscard]] luisa::vector<ExpectedTensor> hybrid_gdn_expected_tensors(
    const HybridGdnExecutorConfig &config, const HybridGdnTensorNames &names);
[[nodiscard]] Status hybrid_gdn_validate_tensors(luisa::span<const PackageTensor> tensors,
                                                 luisa::span<const ExpectedTensor> expected);

}// namespace luisa::ocr

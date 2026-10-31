#pragma once
// embedding_mm engine -- Gemma4TextModel + PLE + Mean Pool + Projection + MRL Head execution graph.
//
// Corresponds to `model.language_model`, `1_Pooling`, and `2_Dense` in Google EmbeddingGemma-2:
// 1. Word embedding lookup (262144 x 512) + sqrt(512) scaling
// 2. Multimodal Soft Token Scatter (vision & audio contiguous or indexed token positions)
// 3. Global Per-Layer Embedding (PLE) projection (512 -> 24*512) + RMSNorm
// 4. 24 Hybrid Attention Transformer blocks:
//    - 5:1 Local Sliding-Window (512 window, head_dim=256, kv_heads=2, theta=10k)
//      vs. Full Bidirectional Attention (layers 5, 11, 17, 23: head_dim=512, kv_heads=1, theta=1M)
//    - Q/K/V-norm (V uses RMSNorm without weight)
//    - GeGLU MLP (512 -> 2048 -> 512)
//    - Per-Layer Embedding (PLE) gating + layer_scalar residual scaling
// 5. Final RMSNorm (512) + Mask-aware Mean Pooling
// 6. Dense Projection (512 -> 768) + Matryoshka (MRL) Truncation & L2 Normalization

#include <cstdint>

#include <luisa/core/stl/vector.h>
#include <luisa/runtime/buffer.h>
#include <luisa/runtime/command_list.h>
#include <luisa/runtime/device.h>

#include "embedding_mm/embed/executor.h"
#include "embedding_mm/embed/ops/ops_attention.h"
#include "embedding_mm/embed/ops/ops_embed.h"
#include "embedding_mm/embed/ops/ops_gemm.h"
#include "embedding_mm/embed/ops/ops_mlp.h"
#include "embedding_mm/embed/ops/ops_norm.h"
#include "embedding_mm/embed/ops/ops_pool.h"
#include "embedding_mm/embed/ops/ops_rope.h"
#include "embedding_mm/embed/package.h"

namespace luisa::embed_mm {

struct TextLayerWeights {
    compute::Buffer<float> input_norm;
    compute::Buffer<float> q_proj;
    compute::Buffer<float> k_proj;
    compute::Buffer<float> v_proj;
    compute::Buffer<float> q_norm;
    compute::Buffer<float> k_norm;
    compute::Buffer<float> out_proj;
    compute::Buffer<float> post_attn_norm;
    compute::Buffer<float> pre_ffn_norm;
    compute::Buffer<float> gate_proj;
    compute::Buffer<float> up_proj;
    compute::Buffer<float> down_proj;
    compute::Buffer<float> post_ffn_norm;
    compute::Buffer<float> ple_gate;
    compute::Buffer<float> ple_proj;
    compute::Buffer<float> ple_norm;
    float layer_scalar{1.0f};
};

class LanguageModel {
public:
    explicit LanguageModel(compute::Device &device) noexcept : _device(device) {}

    void load_weights(const Package &package,
                      compute::CommandList &upload_cmds,
                      luisa::vector<luisa::vector<float>> &half_storage);

    void compile_shaders();

    void record_encode(
        compute::CommandList &cmd_list,
        compute::BufferView<uint32_t> input_ids,
        compute::BufferView<float> attention_mask,
        compute::BufferView<float> out_embeddings,
        uint32_t B,
        uint32_t S,
        uint32_t target_dim,
        const MultimodalInputs &mm_inputs);

private:
    void ensure_workspace(uint32_t B, uint32_t S);
    void dispatch_rmsnorm(compute::CommandList &cmd_list,
                          const compute::Buffer<float> &x,
                          const compute::Buffer<float> &y,
                          const compute::Buffer<float> &gamma,
                          uint32_t num_rows, uint32_t H, float eps);
    void dispatch_rmsnorm_no_weight(compute::CommandList &cmd_list,
                                    const compute::Buffer<float> &x,
                                    const compute::Buffer<float> &y,
                                    uint32_t num_rows, uint32_t H, float eps);

    compute::Device &_device;

    // Weight buffers
    compute::Buffer<float> _word_embeddings;
    compute::Buffer<float> _ple_proj;
    compute::Buffer<float> _ple_norm;
    compute::Buffer<float> _final_norm;
    compute::Buffer<float> _embedding_proj;
    luisa::vector<TextLayerWeights> _layers;

    template<typename K>
    using ShaderFor = decltype(std::declval<compute::Device &>().compile(std::declval<K>()));

    // Compiled shaders
    ShaderFor<EmbedLookupKernel> _shader_embed_lookup;
    ShaderFor<ScatterSoftTokensKernel> _shader_scatter_soft_tokens;
    ShaderFor<ScatterIndexedTokensKernel> _shader_scatter_indexed_tokens;
    ShaderFor<RMSNormKernel> _shader_rmsnorm;
    ShaderFor<RMSNormNoWeightKernel> _shader_rmsnorm_no_weight;
    ShaderFor<GemmKernel> _shader_gemm;
    ShaderFor<RoPEKernel> _shader_rope;
    ShaderFor<GQAScoresKernel> _shader_gqa_scores;
    ShaderFor<SoftmaxRowKernel> _shader_softmax_row;
    ShaderFor<GQAPvKernel> _shader_gqa_pv;
    ShaderFor<GeGLUKernel> _shader_geglu;
    ShaderFor<ResidualAddKernel> _shader_residual_add;
    ShaderFor<PLEGateKernel> _shader_ple_gate;
    ShaderFor<ScaleAddKernel> _shader_scale_add;
    ShaderFor<MeanPoolKernel> _shader_mean_pool;
    ShaderFor<MRLNormalizeKernel> _shader_mrl_normalize;

    // Workspace activations
    compute::Buffer<float> _h0;
    compute::Buffer<float> _h_norm;
    compute::Buffer<float> _ple_all;
    compute::Buffer<float> _q_buf;
    compute::Buffer<float> _k_buf;
    compute::Buffer<float> _v_buf;
    compute::Buffer<float> _logits;
    compute::Buffer<float> _attn_ctx;
    compute::Buffer<float> _attn_out;
    compute::Buffer<float> _attn_norm;
    compute::Buffer<float> _ffn_norm;
    compute::Buffer<float> _gate_buf;
    compute::Buffer<float> _up_buf;
    compute::Buffer<float> _mlp_act;
    compute::Buffer<float> _mlp_out;
    compute::Buffer<float> _mlp_norm;
    compute::Buffer<float> _ple_gate_buf;
    compute::Buffer<float> _ple_gated;
    compute::Buffer<float> _ple_out;
    compute::Buffer<float> _ple_norm_buf;
    compute::Buffer<float> _pooled;
    compute::Buffer<float> _proj_out;

    size_t _max_tokens{0};
    size_t _max_batch{0};
    size_t _max_s{0};
};

}// namespace luisa::embed_mm

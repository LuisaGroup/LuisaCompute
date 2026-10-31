#pragma once
// embedding_mm engine -- Gemma4AudioModel + embed_audio connector execution graph.
//
// Corresponds to `model.audio_tower` and `model.embed_audio` in Google EmbeddingGemma-2:
// 1. 2-Stage Conv2D Subsampling (3x3 stride 2 + LayerNorm + ReLU) + Linear(1024 -> 1024)
// 2. Relative Sinusoidal Position Table generator [13, 1024]
// 3. 12 Conformer Blocks:
//    - Half-step Macaron FFN 1 (RMSNorm -> Clipped Linear 1024->4096 -> SiLU -> Clipped Linear 4096->1024 -> RMSNorm -> 0.5x Residual)
//    - Chunked Local Self-Attention (12-query chunk, 24-key window, per_dim_scale, relative K bias, softcap 50.0)
//    - LightConv1D (RMSNorm -> Clipped Linear 1024->2048 -> GLU -> Causal Depthwise Conv1D k=5 -> RMSNorm -> SiLU -> Clipped Linear)
//    - Half-step Macaron FFN 2 + Layer Output RMSNorm
// 4. Audio Output Projection: LinearWithBias(1024 -> 1536)
// 5. Multimodal Connector: RMSNormNoWeight(1536) + Linear(1536 -> 512)

#include <cstdint>

#include <luisa/core/stl/vector.h>
#include <luisa/runtime/buffer.h>
#include <luisa/runtime/command_list.h>
#include <luisa/runtime/device.h>

#include "embedding_mm/embed/executor.h"
#include "embedding_mm/embed/ops/ops_attention.h"
#include "embedding_mm/embed/ops/ops_conv.h"
#include "embedding_mm/embed/ops/ops_gemm.h"
#include "embedding_mm/embed/ops/ops_mlp.h"
#include "embedding_mm/embed/ops/ops_norm.h"
#include "embedding_mm/embed/ops/ops_rope.h"
#include "embedding_mm/embed/package.h"

namespace luisa::embed_mm {

struct AudioLayerWeights {
    // FeedForward 1
    compute::Buffer<float> ffn1_pre_norm;
    compute::Buffer<float> ffn1_linear1;
    compute::Buffer<float> ffn1_linear2;
    compute::Buffer<float> ffn1_post_norm;

    // Self Attention
    compute::Buffer<float> norm_pre_attn;
    compute::Buffer<float> q_proj;
    compute::Buffer<float> k_proj;
    compute::Buffer<float> v_proj;
    compute::Buffer<float> per_dim_scale;
    compute::Buffer<float> relative_k_proj;
    compute::Buffer<float> out_proj;
    compute::Buffer<float> norm_post_attn;

    // LightConv1D
    compute::Buffer<float> lconv_pre_norm;
    compute::Buffer<float> lconv_linear_start;
    compute::Buffer<float> lconv_depthwise;
    compute::Buffer<float> lconv_norm;
    compute::Buffer<float> lconv_linear_end;

    // FeedForward 2
    compute::Buffer<float> ffn2_pre_norm;
    compute::Buffer<float> ffn2_linear1;
    compute::Buffer<float> ffn2_linear2;
    compute::Buffer<float> ffn2_post_norm;

    // Norm Out
    compute::Buffer<float> norm_out;

    // Clip bounds for the 10 linears: [10][4]
    float clip_bounds[10][4]{};
};

class AudioTower {
public:
    explicit AudioTower(compute::Device &device) noexcept : _device(device) {}

    void load_weights(const Package &package,
                      compute::CommandList &upload_cmds,
                      luisa::vector<luisa::vector<float>> &half_storage);

    void compile_shaders();

    void record_encode(
        compute::CommandList &cmd_list,
        compute::BufferView<float> input_features,
        compute::BufferView<float> out_soft_tokens,
        uint32_t num_frames,
        uint32_t num_mel_bins,
        uint32_t num_soft_tokens,
        Executor::AudioDebugActivations *debug_acts);

private:
    void ensure_workspace(uint32_t T_in, uint32_t T_out);

    compute::Device &_device;

    // Audio Weight buffers
    compute::Buffer<float> _sub0_conv;
    compute::Buffer<float> _sub0_norm;
    compute::Buffer<float> _sub1_conv;
    compute::Buffer<float> _sub1_norm;
    compute::Buffer<float> _sub_input_proj;
    luisa::vector<AudioLayerWeights> _layers;
    compute::Buffer<float> _pos_embed;
    compute::Buffer<float> _output_proj_weight;
    compute::Buffer<float> _output_proj_bias;
    compute::Buffer<float> _connector_proj;

    template<typename K>
    using ShaderFor = decltype(std::declval<compute::Device &>().compile(std::declval<K>()));

    // Compiled shaders
    ShaderFor<AudioSubsampleConv0Kernel> _shader_subsample_conv0;
    ShaderFor<AudioSubsampleConv1Kernel> _shader_subsample_conv1;
    ShaderFor<GemmKernel> _shader_gemm;
    ShaderFor<GemmBiasKernel> _shader_gemm_bias;
    ShaderFor<RMSNormKernel> _shader_rmsnorm;
    ShaderFor<RMSNormNoWeightKernel> _shader_rmsnorm_no_weight;
    ShaderFor<AudioClampKernel> _shader_clamp;
    ShaderFor<AudioScaleQKKernel> _shader_scale_qk;
    ShaderFor<AudioGenRelPosKernel> _shader_gen_rel_pos;
    ShaderFor<AudioChunkedAttentionKernel> _shader_chunked_attn;
    ShaderFor<AudioGluKernel> _shader_glu;
    ShaderFor<AudioDepthwiseConv1DKernel> _shader_depthwise_conv1d;
    ShaderFor<AudioSiluKernel> _shader_silu;
    ShaderFor<AudioAddScaleKernel> _shader_add_scale;
    ShaderFor<AudioAddKernel> _shader_add;

    // Workspace activations
    compute::Buffer<float> _sub0;
    compute::Buffer<float> _sub_flat;
    compute::Buffer<float> _hidden;
    compute::Buffer<float> _tmp1;
    compute::Buffer<float> _tmp2;
    compute::Buffer<float> _ffn_mid;
    compute::Buffer<float> _q;
    compute::Buffer<float> _k;
    compute::Buffer<float> _v;
    compute::Buffer<float> _rel_k;
    compute::Buffer<float> _attn_out;
    compute::Buffer<float> _glu_in;
    compute::Buffer<float> _glu_out;
    compute::Buffer<float> _tower_out;
    compute::Buffer<float> _conn_in;

    size_t _max_audio_in{0};
    size_t _max_audio_out{0};
};

}// namespace luisa::embed_mm

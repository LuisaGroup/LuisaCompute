#pragma once
// embedding_mm engine -- Gemma4VisionModel + embed_vision connector execution graph.
//
// Corresponds to `model.vision_tower` and `model.embed_vision` in Google EmbeddingGemma-2:
// 1. Pixel rescaling [0, 1] -> [-1, 1] & Patch projection (768 -> 768)
// 2. 2D Factorized Position Embedding table lookup (10240 x 768 per axis)
// 3. 16 Vision Transformer layers (RMSNorm, Q/K/V/V-norm, 2D Axial RoPE, Full Bidirectional Attention, GeGLU MLP)
// 4. 2D 3x3 Spatial Average Pooling + sqrt(768) scaling -> Soft Tokens (e.g. 280 or 1120)
// 5. Multimodal Connector: RMSNormNoWeight(768) + Linear(768 -> 512)

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

struct VisionLayerWeights {
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
};

class VisionTower {
public:
    explicit VisionTower(compute::Device &device) noexcept : _device(device) {}

    void load_weights(const Package &package,
                      compute::CommandList &upload_cmds,
                      luisa::vector<luisa::vector<float>> &half_storage);

    void compile_shaders();

    void record_encode(
        compute::CommandList &cmd_list,
        compute::BufferView<float> pixel_values,
        compute::BufferView<int32_t> position_ids,
        compute::BufferView<float> out_soft_tokens,
        uint32_t B,
        uint32_t num_patches_w,
        uint32_t num_patches_h,
        uint32_t max_patches,
        uint32_t num_soft_tokens,
        Executor::VisionDebugActivations *debug_acts);

private:
    void ensure_workspace(uint32_t max_patches, uint32_t num_soft_tokens);
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

    // Vision Weight buffers
    compute::Buffer<float> _patch_proj;
    compute::Buffer<float> _pos_table;
    compute::Buffer<float> _connector_proj;
    luisa::vector<VisionLayerWeights> _layers;

    template<typename K>
    using ShaderFor = decltype(std::declval<compute::Device &>().compile(std::declval<K>()));

    // Compiled shaders
    ShaderFor<ScalePixelsKernel> _shader_scale_pixels;
    ShaderFor<AddPatchPosEmbedKernel> _shader_add_patch_pos_embed;
    ShaderFor<MaskFromPosKernel> _shader_mask_from_pos;
    ShaderFor<RMSNormKernel> _shader_rmsnorm;
    ShaderFor<RMSNormNoWeightKernel> _shader_rmsnorm_no_weight;
    ShaderFor<GemmKernel> _shader_gemm;
    ShaderFor<AxialRoPEKernel> _shader_axial_rope;
    ShaderFor<GQAScoresKernel> _shader_gqa_scores;
    ShaderFor<SoftmaxRowKernel> _shader_softmax_row;
    ShaderFor<GQAPvKernel> _shader_gqa_pv;
    ShaderFor<GeGLUKernel> _shader_geglu;
    ShaderFor<ResidualAddKernel> _shader_residual_add;
    ShaderFor<VisionPoolKernel> _shader_vision_pool;

    // Workspace activations
    compute::Buffer<float> _scaled_pixels;
    compute::Buffer<float> _h0;
    compute::Buffer<float> _h_norm;
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
    compute::Buffer<float> _mask;
    compute::Buffer<float> _pooled;
    compute::Buffer<float> _pooled_norm;

    size_t _max_patches{0};
    size_t _max_vision_soft{0};
};

}// namespace luisa::embed_mm

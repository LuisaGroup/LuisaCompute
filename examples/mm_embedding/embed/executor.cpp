// mm_embedding engine -- forward pass graph executor implementation.
#include "mm_embedding/embed/executor.h"

#include <cmath>
#include <cstring>

#include <luisa/core/logging.h>
#include <luisa/core/stl/format.h>
#include <luisa/dsl/sugar.h>

#include "mm_embedding/embed/kernels/kernels_attention.h"
#include "mm_embedding/embed/kernels/kernels_gemm.h"
#include "mm_embedding/embed/kernels/kernels_mlp.h"
#include "mm_embedding/embed/kernels/kernels_norm.h"
#include "mm_embedding/embed/kernels/kernels_pool.h"
#include "mm_embedding/embed/kernels/kernels_rope.h"
#include "mm_embedding/embed/kernels/kernels_vision.h"

namespace luisa::mm_embed {

namespace {

using EmbedLookupKernel = compute::Kernel2D<void(
    compute::Buffer<uint32_t> /*input_ids*/,
    compute::Buffer<float> /*embed_table*/,
    compute::Buffer<float> /*out_hidden*/,
    uint32_t /*H*/,
    float /*scale*/)>;

EmbedLookupKernel make_embed_lookup_kernel() noexcept {
    using namespace luisa::compute;

    EmbedLookupKernel kernel = [](BufferUInt input_ids, BufferFloat table, BufferFloat out,
                                  Var<uint> H, Var<float> scale) noexcept {
        set_block_size(16u, 16u, 1u);
        set_name("embed_lookup");

        Var d = dispatch_x();
        Var token_idx = dispatch_y();
        Var total_tokens = dispatch_size().y;

        $if (d < H & token_idx < total_tokens) {
            Var id = input_ids.read(token_idx);
            Var val = table.read(id * H + d) * scale;
            out.write(token_idx * H + d, val);
        };
    };
    return kernel;
}

struct LayerWeights {
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

class ExecutorImpl final : public Executor {
public:
    compute::Device &_device;
    bool _enable_vision{false};

    [[nodiscard]] bool has_vision() const noexcept override { return _enable_vision; }

    // Text Weight buffers
    compute::Buffer<float> word_embeddings;
    compute::Buffer<float> ple_proj;
    compute::Buffer<float> ple_norm;
    compute::Buffer<float> final_norm;
    compute::Buffer<float> embedding_proj;
    luisa::vector<LayerWeights> layers;

    // Vision Weight buffers (Selective Encoder Loading)
    compute::Buffer<float> v_patch_proj;
    compute::Buffer<float> v_pos_table;
    compute::Buffer<float> v_connector_proj;
    luisa::vector<VisionLayerWeights> vision_layers;

    template<typename K>
    using ShaderFor = decltype(std::declval<compute::Device &>().compile(std::declval<K>()));

    // Compiled text shaders
    ShaderFor<EmbedLookupKernel> shader_embed_lookup;
    ShaderFor<RMSNormKernel> shader_rmsnorm;
    ShaderFor<RMSNormNoWeightKernel> shader_rmsnorm_no_weight;
    ShaderFor<GemmKernel> shader_gemm;
    ShaderFor<RoPEKernel> shader_rope;
    ShaderFor<GQAScoresKernel> shader_gqa_scores;
    ShaderFor<SoftmaxRowKernel> shader_softmax_row;
    ShaderFor<GQAPvKernel> shader_gqa_pv;
    ShaderFor<GeGLUKernel> shader_geglu;
    ShaderFor<ResidualAddKernel> shader_residual_add;
    ShaderFor<PLEGateKernel> shader_ple_gate;
    ShaderFor<ScaleAddKernel> shader_scale_add;
    ShaderFor<MeanPoolKernel> shader_mean_pool;
    ShaderFor<MRLNormalizeKernel> shader_mrl_normalize;

    // Compiled vision shaders
    ShaderFor<ScalePixelsKernel> shader_scale_pixels;
    ShaderFor<AddPatchPosEmbedKernel> shader_add_patch_pos_embed;
    ShaderFor<AxialRoPEKernel> shader_axial_rope;
    ShaderFor<VisionPoolKernel> shader_vision_pool;
    ShaderFor<ScatterSoftTokensKernel> shader_scatter_soft_tokens;
    ShaderFor<MaskFromPosKernel> shader_mask_from_pos;

    // Text workspace activations
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

    // Vision workspace activations
    compute::Buffer<float> _v_scaled_pixels;
    compute::Buffer<float> _v_h0;
    compute::Buffer<float> _v_h_norm;
    compute::Buffer<float> _v_q_buf;
    compute::Buffer<float> _v_k_buf;
    compute::Buffer<float> _v_v_buf;
    compute::Buffer<float> _v_logits;
    compute::Buffer<float> _v_attn_ctx;
    compute::Buffer<float> _v_attn_out;
    compute::Buffer<float> _v_attn_norm;
    compute::Buffer<float> _v_ffn_norm;
    compute::Buffer<float> _v_gate_buf;
    compute::Buffer<float> _v_up_buf;
    compute::Buffer<float> _v_mlp_act;
    compute::Buffer<float> _v_mlp_out;
    compute::Buffer<float> _v_mlp_norm;
    compute::Buffer<float> _v_mask;
    compute::Buffer<float> _v_pooled;
    compute::Buffer<float> _v_pooled_norm;

    size_t _max_patches{0};

    explicit ExecutorImpl(compute::Device &device) noexcept : _device(device) {}

    void ensure_workspace(uint32_t B, uint32_t S) {
        size_t total_tokens = static_cast<size_t>(B) * S;
        if (total_tokens <= _max_tokens && B <= _max_batch && S <= _max_s) {
            return;
        }

        _max_tokens = std::max(_max_tokens, total_tokens);
        _max_batch = std::max(_max_batch, static_cast<size_t>(B));
        _max_s = std::max(_max_s, static_cast<size_t>(S));

        constexpr size_t H = 512;
        constexpr size_t I = 2048;
        constexpr size_t max_q_dim = 2048;
        constexpr size_t max_kv_dim = 512;

        _h0 = _device.create_buffer<float>(_max_tokens * H);
        _h_norm = _device.create_buffer<float>(_max_tokens * H);
        _ple_all = _device.create_buffer<float>(_max_tokens * 24 * H);
        _q_buf = _device.create_buffer<float>(_max_tokens * max_q_dim);
        _k_buf = _device.create_buffer<float>(_max_tokens * max_kv_dim);
        _v_buf = _device.create_buffer<float>(_max_tokens * max_kv_dim);
        _logits = _device.create_buffer<float>(_max_batch * 4 * _max_s * _max_s);
        _attn_ctx = _device.create_buffer<float>(_max_tokens * max_q_dim);
        _attn_out = _device.create_buffer<float>(_max_tokens * H);
        _attn_norm = _device.create_buffer<float>(_max_tokens * H);
        _ffn_norm = _device.create_buffer<float>(_max_tokens * H);
        _gate_buf = _device.create_buffer<float>(_max_tokens * I);
        _up_buf = _device.create_buffer<float>(_max_tokens * I);
        _mlp_act = _device.create_buffer<float>(_max_tokens * I);
        _mlp_out = _device.create_buffer<float>(_max_tokens * H);
        _mlp_norm = _device.create_buffer<float>(_max_tokens * H);
        _ple_gate_buf = _device.create_buffer<float>(_max_tokens * H);
        _ple_gated = _device.create_buffer<float>(_max_tokens * H);
        _ple_out = _device.create_buffer<float>(_max_tokens * H);
        _ple_norm_buf = _device.create_buffer<float>(_max_tokens * H);
        _pooled = _device.create_buffer<float>(_max_batch * H);
        _proj_out = _device.create_buffer<float>(_max_batch * 768);
    }

    void ensure_vision_workspace(uint32_t max_patches) {
        if (max_patches <= _max_patches) {
            return;
        }
        _max_patches = max_patches;
        constexpr size_t H = 768;
        constexpr size_t I = 3072;
        constexpr size_t num_heads = 12;

        _v_scaled_pixels = _device.create_buffer<float>(_max_patches * H);
        _v_h0 = _device.create_buffer<float>(_max_patches * H);
        _v_h_norm = _device.create_buffer<float>(_max_patches * H);
        _v_q_buf = _device.create_buffer<float>(_max_patches * H);
        _v_k_buf = _device.create_buffer<float>(_max_patches * H);
        _v_v_buf = _device.create_buffer<float>(_max_patches * H);
        _v_logits = _device.create_buffer<float>(num_heads * _max_patches * _max_patches);
        _v_attn_ctx = _device.create_buffer<float>(_max_patches * H);
        _v_attn_out = _device.create_buffer<float>(_max_patches * H);
        _v_attn_norm = _device.create_buffer<float>(_max_patches * H);
        _v_ffn_norm = _device.create_buffer<float>(_max_patches * H);
        _v_gate_buf = _device.create_buffer<float>(_max_patches * I);
        _v_up_buf = _device.create_buffer<float>(_max_patches * I);
        _v_mlp_act = _device.create_buffer<float>(_max_patches * I);
        _v_mlp_out = _device.create_buffer<float>(_max_patches * H);
        _v_mlp_norm = _device.create_buffer<float>(_max_patches * H);
        _v_mask = _device.create_buffer<float>(_max_patches);
        _v_pooled = _device.create_buffer<float>(280 * H);
        _v_pooled_norm = _device.create_buffer<float>(280 * H);
    }

    void record_vision_encode(
        compute::CommandList &cmd_list,
        compute::BufferView<float> pixel_values,
        compute::BufferView<int32_t> position_ids,
        compute::BufferView<float> out_soft_tokens,
        uint32_t B,
        uint32_t num_patches_w,
        uint32_t num_patches_h,
        uint32_t max_patches,
        uint32_t num_soft_tokens) override {

        if (!_enable_vision) {
            LUISA_ERROR("record_vision_encode: Vision encoder is not loaded (Selective Encoder Loading)");
        }

        ensure_vision_workspace(max_patches);

        constexpr uint32_t H = 768u;
        constexpr uint32_t I = 3072u;
        constexpr uint32_t num_heads = 12u;
        constexpr uint32_t head_dim = 64u;
        constexpr float rope_theta = 100.0f;
        // Pooling scale factor: sqrt(768) / 9.0
        constexpr float scale_pool = 3.079201435678004f;

        // 1. Rescale pixels: [-1.0, 1.0]
        cmd_list << shader_scale_pixels(pixel_values, _v_scaled_pixels, max_patches * H)
                        .dispatch(max_patches * H);

        // 2. Patch embedding GEMM: [max_patches, 768] * [768, 768] -> [max_patches, 768]
        cmd_list << shader_gemm(_v_scaled_pixels, v_patch_proj, _v_h0, max_patches, H, H)
                        .dispatch(gemm_dispatch_size(max_patches, H));

        // 3. Add 2D position embeddings
        cmd_list << shader_add_patch_pos_embed(_v_h0, v_pos_table, position_ids, H, 10240u)
                        .dispatch(H, max_patches);

        // 4. Generate attention mask from position_ids
        cmd_list << shader_mask_from_pos(position_ids, _v_mask, max_patches)
                        .dispatch(max_patches);

        // 5. 16 Transformer layers
        for (uint32_t l = 0u; l < 16u; l++) {
            const auto &layer = vision_layers[l];

            // (a) Input RMSNorm
            cmd_list << shader_rmsnorm(_v_h0, _v_h_norm, layer.input_norm, H, 1e-6f)
                            .dispatch(max_patches * kNormBlock);

            // (b) Q, K, V projections
            cmd_list << shader_gemm(_v_h_norm, layer.q_proj, _v_q_buf, max_patches, H, H)
                            .dispatch(gemm_dispatch_size(max_patches, H));
            cmd_list << shader_gemm(_v_h_norm, layer.k_proj, _v_k_buf, max_patches, H, H)
                            .dispatch(gemm_dispatch_size(max_patches, H));
            cmd_list << shader_gemm(_v_h_norm, layer.v_proj, _v_v_buf, max_patches, H, H)
                            .dispatch(gemm_dispatch_size(max_patches, H));

            // (c) Q, K, V RMSNorm
            cmd_list << shader_rmsnorm(_v_q_buf, _v_q_buf, layer.q_norm, head_dim, 1e-6f)
                            .dispatch(max_patches * num_heads * kNormBlock);
            cmd_list << shader_rmsnorm(_v_k_buf, _v_k_buf, layer.k_norm, head_dim, 1e-6f)
                            .dispatch(max_patches * num_heads * kNormBlock);
            cmd_list << shader_rmsnorm_no_weight(_v_v_buf, _v_v_buf, head_dim, 1e-6f)
                            .dispatch(max_patches * num_heads * kNormBlock);

            // (d) 2D Axial RoPE on Q and K
            cmd_list << shader_axial_rope(_v_q_buf, _v_q_buf, position_ids, num_heads, head_dim, rope_theta)
                            .dispatch(16u, max_patches * num_heads);
            cmd_list << shader_axial_rope(_v_k_buf, _v_k_buf, position_ids, num_heads, head_dim, rope_theta)
                            .dispatch(16u, max_patches * num_heads);

            // (e) Attention Scores + Softmax + PV
            cmd_list << shader_gqa_scores(_v_q_buf, _v_k_buf, _v_mask, _v_logits,
                                          max_patches, num_heads, num_heads, head_dim,
                                          1.0f, 0u, 0u, -1e9f)
                            .dispatch(max_patches, num_heads * max_patches);

            cmd_list << shader_softmax_row(_v_logits, max_patches)
                            .dispatch(kAttentionBlock, num_heads * max_patches);

            cmd_list << shader_gqa_pv(_v_v_buf, _v_logits, _v_attn_ctx,
                                      max_patches, num_heads, num_heads, head_dim)
                            .dispatch(head_dim, num_heads * max_patches);

            // (f) Out projection GEMM
            cmd_list << shader_gemm(_v_attn_ctx, layer.out_proj, _v_attn_out, max_patches, H, H)
                            .dispatch(gemm_dispatch_size(max_patches, H));

            // (g) Post-attn RMSNorm + Residual add
            cmd_list << shader_rmsnorm(_v_attn_out, _v_attn_norm, layer.post_attn_norm, H, 1e-6f)
                            .dispatch(max_patches * kNormBlock);
            cmd_list << shader_residual_add(_v_h0, _v_attn_norm, _v_h0, max_patches * H)
                            .dispatch(max_patches * H);

            // (h) Pre-FFN RMSNorm
            cmd_list << shader_rmsnorm(_v_h0, _v_ffn_norm, layer.pre_ffn_norm, H, 1e-6f)
                            .dispatch(max_patches * kNormBlock);

            // (i) MLP projections
            cmd_list << shader_gemm(_v_ffn_norm, layer.gate_proj, _v_gate_buf, max_patches, I, H)
                            .dispatch(gemm_dispatch_size(max_patches, I));
            cmd_list << shader_gemm(_v_ffn_norm, layer.up_proj, _v_up_buf, max_patches, I, H)
                            .dispatch(gemm_dispatch_size(max_patches, I));

            // (j) GeGLU activation
            cmd_list << shader_geglu(_v_gate_buf, _v_up_buf, _v_mlp_act, max_patches * I)
                            .dispatch(max_patches * I);

            // (k) Down projection GEMM
            cmd_list << shader_gemm(_v_mlp_act, layer.down_proj, _v_mlp_out, max_patches, H, I)
                            .dispatch(gemm_dispatch_size(max_patches, H));

            // (l) Post-FFN RMSNorm + Residual add
            cmd_list << shader_rmsnorm(_v_mlp_out, _v_mlp_norm, layer.post_ffn_norm, H, 1e-6f)
                            .dispatch(max_patches * kNormBlock);
            cmd_list << shader_residual_add(_v_h0, _v_mlp_norm, _v_h0, max_patches * H)
                            .dispatch(max_patches * H);
        }

        // 6. 2D Spatial Pooling (3x3 grid average + sqrt(768)) -> soft tokens
        uint32_t grid_w_soft = num_patches_w / 3u;
        uint32_t grid_h_soft = num_patches_h / 3u;
        uint32_t patch_stride = num_patches_w;
        cmd_list << shader_vision_pool(_v_h0, _v_pooled, grid_w_soft, grid_h_soft, patch_stride, H, scale_pool)
                        .dispatch(H, num_soft_tokens);

        // 7. Connector pre-norm (without scale)
        cmd_list << shader_rmsnorm_no_weight(_v_pooled, _v_pooled_norm, H, 1e-6f)
                        .dispatch(num_soft_tokens * kNormBlock);

        // 8. Connector projection: [num_soft_tokens, 768] * [768, 512] -> [num_soft_tokens, 512]
        cmd_list << shader_gemm(_v_pooled_norm, v_connector_proj, out_soft_tokens, num_soft_tokens, 512u, H)
                        .dispatch(gemm_dispatch_size(num_soft_tokens, 512u));
    }

    void record_encode(
        compute::CommandList &cmd_list,
        compute::BufferView<uint32_t> input_ids,
        compute::BufferView<float> attention_mask,
        compute::BufferView<float> out_embeddings,
        uint32_t B,
        uint32_t S,
        uint32_t target_dim,
        const MultimodalInputs &mm_inputs) override {

        ensure_workspace(B, S);

        constexpr uint32_t H = 512u;
        constexpr uint32_t I = 2048u;
        const float embed_scale = std::sqrt(512.0f);
        const uint32_t total_tokens = B * S;

        // 1. Embedding lookup
        cmd_list << shader_embed_lookup(input_ids, word_embeddings, _h0, H, embed_scale)
                        .dispatch(H, total_tokens);

        // 1.5. Scatter multimodal vision soft tokens if provided
        if (mm_inputs.vision_embeds.size() > 0 && mm_inputs.num_vision_tokens > 0) {
            cmd_list << shader_scatter_soft_tokens(
                _h0, mm_inputs.vision_embeds, mm_inputs.vision_start_idx, mm_inputs.num_vision_tokens, H)
                .dispatch(H, mm_inputs.num_vision_tokens);
        }

        // 2. Global PLE projection & normalization
        cmd_list << shader_gemm(_h0, ple_proj, _ple_all, total_tokens, 24u * H, H)
                        .dispatch(gemm_dispatch_size(total_tokens, 24u * H));
        uint32_t total_ple_rows = total_tokens * 24u;
        constexpr uint32_t kMaxRowsPerDispatch = 32768u;
        for (uint32_t r = 0u; r < total_ple_rows; r += kMaxRowsPerDispatch) {
            uint32_t chunk = std::min(kMaxRowsPerDispatch, total_ple_rows - r);
            auto v = _ple_all.view(r * H, chunk * H);
            cmd_list << shader_rmsnorm(v, v, ple_norm, H, 1e-6f)
                            .dispatch(chunk * kNormBlock);
        }

        // 3. Transformer layers (24 blocks)
        for (uint32_t i = 0u; i < 24u; i++) {
            const auto &layer = layers[i];

            bool is_full = (i == 5u || i == 11u || i == 17u || i == 23u);
            uint32_t head_dim = is_full ? 512u : 256u;
            uint32_t num_q_heads = 4u;
            uint32_t num_kv_heads = is_full ? 1u : 2u;
            float rope_theta = is_full ? 1000000.0f : 10000.0f;
            uint32_t is_sliding = is_full ? 0u : 1u;
            uint32_t sliding_window = 512u;
            float qk_scale = 1.0f;

            uint32_t q_dim = num_q_heads * head_dim;
            uint32_t kv_dim = num_kv_heads * head_dim;

            // (a) input RMSNorm
            cmd_list << shader_rmsnorm(_h0, _h_norm, layer.input_norm, H, 1e-6f)
                            .dispatch(total_tokens * kNormBlock);

            // (b) Q, K, V projections
            cmd_list << shader_gemm(_h_norm, layer.q_proj, _q_buf, total_tokens, q_dim, H)
                            .dispatch(gemm_dispatch_size(total_tokens, q_dim));
            cmd_list << shader_gemm(_h_norm, layer.k_proj, _k_buf, total_tokens, kv_dim, H)
                            .dispatch(gemm_dispatch_size(total_tokens, kv_dim));
            cmd_list << shader_gemm(_h_norm, layer.v_proj, _v_buf, total_tokens, kv_dim, H)
                            .dispatch(gemm_dispatch_size(total_tokens, kv_dim));

            // (c) Q, K, V RMSNorm
            cmd_list << shader_rmsnorm(_q_buf, _q_buf, layer.q_norm, head_dim, 1e-6f)
                            .dispatch(total_tokens * num_q_heads * kNormBlock);
            cmd_list << shader_rmsnorm(_k_buf, _k_buf, layer.k_norm, head_dim, 1e-6f)
                            .dispatch(total_tokens * num_kv_heads * kNormBlock);
            cmd_list << shader_rmsnorm_no_weight(_v_buf, _v_buf, head_dim, 1e-6f)
                            .dispatch(total_tokens * num_kv_heads * kNormBlock);

            // (d) RoPE on Q and K
            uint32_t q_half_pairs = total_tokens * num_q_heads * (head_dim / 2u);
            uint32_t k_half_pairs = total_tokens * num_kv_heads * (head_dim / 2u);
            cmd_list << shader_rope(_q_buf, _q_buf, S, num_q_heads, head_dim, rope_theta)
                            .dispatch(q_half_pairs);
            cmd_list << shader_rope(_k_buf, _k_buf, S, num_kv_heads, head_dim, rope_theta)
                            .dispatch(k_half_pairs);

            // (e) GQA Attention Scores + Softmax + PV
            cmd_list << shader_gqa_scores(_q_buf, _k_buf, attention_mask, _logits,
                                          S, num_q_heads, num_kv_heads, head_dim,
                                          qk_scale, is_sliding, sliding_window, -1e9f)
                            .dispatch(S, B * num_q_heads * S);

            cmd_list << shader_softmax_row(_logits, S)
                            .dispatch(kAttentionBlock, B * num_q_heads * S);

            cmd_list << shader_gqa_pv(_v_buf, _logits, _attn_ctx,
                                      S, num_q_heads, num_kv_heads, head_dim)
                            .dispatch(head_dim, B * num_q_heads * S);

            // (f) Attention Out GEMM
            cmd_list << shader_gemm(_attn_ctx, layer.out_proj, _attn_out, total_tokens, H, q_dim)
                            .dispatch(gemm_dispatch_size(total_tokens, H));

            // (g) Post-attention RMSNorm + Residual Add
            cmd_list << shader_rmsnorm(_attn_out, _attn_norm, layer.post_attn_norm, H, 1e-6f)
                            .dispatch(total_tokens * kNormBlock);
            cmd_list << shader_residual_add(_h0, _attn_norm, _h0, total_tokens * H)
                            .dispatch(total_tokens * H);

            // (h) Pre-FFN RMSNorm
            cmd_list << shader_rmsnorm(_h0, _ffn_norm, layer.pre_ffn_norm, H, 1e-6f)
                            .dispatch(total_tokens * kNormBlock);

            // (i) MLP Gate and Up projections
            cmd_list << shader_gemm(_ffn_norm, layer.gate_proj, _gate_buf, total_tokens, I, H)
                            .dispatch(gemm_dispatch_size(total_tokens, I));
            cmd_list << shader_gemm(_ffn_norm, layer.up_proj, _up_buf, total_tokens, I, H)
                            .dispatch(gemm_dispatch_size(total_tokens, I));

            // (j) GeGLU activation
            cmd_list << shader_geglu(_gate_buf, _up_buf, _mlp_act, total_tokens * I)
                            .dispatch(total_tokens * I);

            // (k) MLP Down projection
            cmd_list << shader_gemm(_mlp_act, layer.down_proj, _mlp_out, total_tokens, H, I)
                            .dispatch(gemm_dispatch_size(total_tokens, H));

            // (l) Post-FFN RMSNorm + Residual Add
            cmd_list << shader_rmsnorm(_mlp_out, _mlp_norm, layer.post_ffn_norm, H, 1e-6f)
                            .dispatch(total_tokens * kNormBlock);
            cmd_list << shader_residual_add(_h0, _mlp_norm, _h0, total_tokens * H)
                            .dispatch(total_tokens * H);

            // (m) PLE Block
            cmd_list << shader_gemm(_h0, layer.ple_gate, _ple_gate_buf, total_tokens, H, H)
                            .dispatch(gemm_dispatch_size(total_tokens, H));
            cmd_list << shader_ple_gate(_ple_gate_buf, _ple_all, _ple_gated, i, total_tokens, H)
                            .dispatch(total_tokens * H);
            cmd_list << shader_gemm(_ple_gated, layer.ple_proj, _ple_out, total_tokens, H, H)
                            .dispatch(gemm_dispatch_size(total_tokens, H));
            cmd_list << shader_rmsnorm(_ple_out, _ple_norm_buf, layer.ple_norm, H, 1e-6f)
                            .dispatch(total_tokens * kNormBlock);

            // (n) Residual Add + Layer Scalar
            cmd_list << shader_scale_add(_h0, _ple_norm_buf, _h0, layer.layer_scalar, total_tokens * H)
                            .dispatch(total_tokens * H);
        }

        // 4. Final RMSNorm
        cmd_list << shader_rmsnorm(_h0, _h_norm, final_norm, H, 1e-6f)
                        .dispatch(total_tokens * kNormBlock);

        // 5. Mean pooling
        cmd_list << shader_mean_pool(_h_norm, attention_mask, _pooled, S, H)
                        .dispatch(H, B);

        // 6. Projection (512 -> 768)
        cmd_list << shader_gemm(_pooled, embedding_proj, _proj_out, B, 768u, H)
                        .dispatch(gemm_dispatch_size(B, 768u));

        // 7. MRL Truncation + L2 Normalization
        cmd_list << shader_mrl_normalize(_proj_out, out_embeddings, 768u, target_dim)
                        .dispatch(B * kPoolBlock);
    }
};

}// namespace

luisa::unique_ptr<Executor> Executor::create(compute::Device &device, const Package &package, bool enable_vision) {
    auto exec = luisa::make_unique<ExecutorImpl>(device);
    auto upload_stream = device.create_stream(compute::StreamTag::COMPUTE);
    compute::CommandList upload_cmds;
    luisa::vector<luisa::vector<float>> half_storage;

    auto upload_tensor = [&](luisa::string_view name) -> compute::Buffer<float> {
        const auto *t = package.find(name);
        if (!t) {
            LUISA_ERROR("Executor: missing required weight tensor '{}'", name);
        }
        auto elem_count = static_cast<size_t>(t->element_count());
        auto buf = device.create_buffer<float>(elem_count);

        if (t->data) {
            upload_cmds << buf.copy_from(t->data);
        } else if (t->half_data) {
            auto &fp32_data = half_storage.emplace_back(elem_count);
            for (size_t i = 0; i < elem_count; i++) {
                uint16_t h = t->half_data[i];
                if (package.precision() == PackagePrecision::bf16) {
                    uint32_t w = static_cast<uint32_t>(h) << 16;
                    std::memcpy(&fp32_data[i], &w, sizeof(float));
                } else {
                    uint32_t sign = (h & 0x8000u) << 16;
                    uint32_t exp = (h & 0x7C00u) >> 10;
                    uint32_t mant = (h & 0x03FFu) << 13;
                    uint32_t f_val = 0;
                    if (exp == 0x1Fu) {
                        f_val = sign | 0x7F800000u | mant;
                    } else if (exp == 0) {
                        if (mant != 0) {
                            exp = 113;
                            while ((mant & (1u << 23)) == 0) {
                                mant <<= 1;
                                exp--;
                            }
                            mant &= 0x007FFFFFu;
                            f_val = sign | (exp << 23) | mant;
                        } else {
                            f_val = sign;
                        }
                    } else {
                        f_val = sign | ((exp + 112) << 23) | mant;
                    }
                    std::memcpy(&fp32_data[i], &f_val, sizeof(float));
                }
            }
            upload_cmds << buf.copy_from(fp32_data.data());
        }
        return buf;
    };

    // Upload global text weights
    exec->word_embeddings = upload_tensor("emb.word");
    exec->ple_proj = upload_tensor("ple.proj");
    exec->ple_norm = upload_tensor("ple.norm");
    exec->final_norm = upload_tensor("final_norm");
    exec->embedding_proj = upload_tensor("embedding_proj");

    // Upload 24 text layer weights
    exec->layers.resize(24);
    for (uint32_t i = 0; i < 24; i++) {
        auto &l = exec->layers[i];
        l.input_norm = upload_tensor(luisa::format("layer{}.input_norm", i));
        l.q_proj = upload_tensor(luisa::format("layer{}.q_proj", i));
        l.k_proj = upload_tensor(luisa::format("layer{}.k_proj", i));
        l.v_proj = upload_tensor(luisa::format("layer{}.v_proj", i));
        l.q_norm = upload_tensor(luisa::format("layer{}.q_norm", i));
        l.k_norm = upload_tensor(luisa::format("layer{}.k_norm", i));
        l.out_proj = upload_tensor(luisa::format("layer{}.out_proj", i));
        l.post_attn_norm = upload_tensor(luisa::format("layer{}.post_attn_norm", i));
        l.pre_ffn_norm = upload_tensor(luisa::format("layer{}.pre_ffn_norm", i));
        l.gate_proj = upload_tensor(luisa::format("layer{}.gate_proj", i));
        l.up_proj = upload_tensor(luisa::format("layer{}.up_proj", i));
        l.down_proj = upload_tensor(luisa::format("layer{}.down_proj", i));
        l.post_ffn_norm = upload_tensor(luisa::format("layer{}.post_ffn_norm", i));
        l.ple_gate = upload_tensor(luisa::format("layer{}.ple_gate", i));
        l.ple_proj = upload_tensor(luisa::format("layer{}.ple_proj", i));
        l.ple_norm = upload_tensor(luisa::format("layer{}.ple_norm", i));

        const auto *t_scalar = package.find(luisa::format("layer{}.layer_scalar", i));
        if (t_scalar) {
            if (t_scalar->data) {
                l.layer_scalar = t_scalar->data[0];
            } else if (t_scalar->half_data) {
                uint16_t h = t_scalar->half_data[0];
                if (package.precision() == PackagePrecision::bf16) {
                    uint32_t w = static_cast<uint32_t>(h) << 16;
                    std::memcpy(&l.layer_scalar, &w, sizeof(float));
                }
            }
        }
    }

    // Selective Encoder Loading: Vision Tower
    if (enable_vision && package.has_vision()) {
        LUISA_INFO("Executor: Selective Encoder Loading -> initializing Vision Tower (~170M params)...");
        exec->v_patch_proj = upload_tensor("vision.patch_embed.proj");
        exec->v_pos_table = upload_tensor("vision.patch_embed.pos_table");
        exec->v_connector_proj = upload_tensor("vision.connector.proj");

        exec->vision_layers.resize(16);
        for (uint32_t i = 0; i < 16; i++) {
            auto &vl = exec->vision_layers[i];
            vl.input_norm = upload_tensor(luisa::format("vision.layer{}.input_norm", i));
            vl.q_proj = upload_tensor(luisa::format("vision.layer{}.q_proj", i));
            vl.k_proj = upload_tensor(luisa::format("vision.layer{}.k_proj", i));
            vl.v_proj = upload_tensor(luisa::format("vision.layer{}.v_proj", i));
            vl.q_norm = upload_tensor(luisa::format("vision.layer{}.q_norm", i));
            vl.k_norm = upload_tensor(luisa::format("vision.layer{}.k_norm", i));
            vl.out_proj = upload_tensor(luisa::format("vision.layer{}.out_proj", i));
            vl.post_attn_norm = upload_tensor(luisa::format("vision.layer{}.post_attn_norm", i));
            vl.pre_ffn_norm = upload_tensor(luisa::format("vision.layer{}.pre_ffn_norm", i));
            vl.gate_proj = upload_tensor(luisa::format("vision.layer{}.gate_proj", i));
            vl.up_proj = upload_tensor(luisa::format("vision.layer{}.up_proj", i));
            vl.down_proj = upload_tensor(luisa::format("vision.layer{}.down_proj", i));
            vl.post_ffn_norm = upload_tensor(luisa::format("vision.layer{}.post_ffn_norm", i));
        }

        exec->_enable_vision = true;
    } else {
        LUISA_INFO("Executor: Selective Encoder Loading -> Vision Tower skipped (text-only mode).");
        exec->_enable_vision = false;
    }

    upload_stream << upload_cmds.commit() << compute::synchronize();

    // Compile core text kernels
    exec->shader_embed_lookup = device.compile(make_embed_lookup_kernel());
    exec->shader_rmsnorm = device.compile(make_rmsnorm_kernel());
    exec->shader_rmsnorm_no_weight = device.compile(make_rmsnorm_no_weight_kernel());
    exec->shader_gemm = device.compile(make_gemm_kernel());
    exec->shader_rope = device.compile(make_rope_kernel());
    exec->shader_gqa_scores = device.compile(make_gqa_scores_kernel());
    exec->shader_softmax_row = device.compile(make_softmax_row_kernel());
    exec->shader_gqa_pv = device.compile(make_gqa_pv_kernel());
    exec->shader_geglu = device.compile(make_geglu_kernel());
    exec->shader_residual_add = device.compile(make_residual_add_kernel());
    exec->shader_ple_gate = device.compile(make_ple_gate_kernel());
    exec->shader_scale_add = device.compile(make_scale_add_kernel());
    exec->shader_mean_pool = device.compile(make_mean_pool_kernel());
    exec->shader_mrl_normalize = device.compile(make_mrl_normalize_kernel());
    exec->shader_scatter_soft_tokens = device.compile(make_scatter_soft_tokens_kernel());

    // Compile vision kernels if enabled
    if (exec->_enable_vision) {
        exec->shader_scale_pixels = device.compile(make_scale_pixels_kernel());
        exec->shader_add_patch_pos_embed = device.compile(make_add_patch_pos_embed_kernel());
        exec->shader_axial_rope = device.compile(make_axial_rope_kernel());
        exec->shader_vision_pool = device.compile(make_vision_pool_kernel());
        exec->shader_mask_from_pos = device.compile(make_mask_from_pos_kernel());
    }

    LUISA_INFO("Executor: successfully initialized and compiled all kernels (vision={})",
               exec->_enable_vision ? "enabled" : "disabled");

    return exec;
}

}// namespace luisa::mm_embed

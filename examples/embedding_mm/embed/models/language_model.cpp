// embedding_mm engine -- Gemma4TextModel + PLE + Mean Pool + Projection + MRL Head implementation.
#include "embedding_mm/embed/models/language_model.h"
#include "embedding_mm/embed/models/embedding_gemma2.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <luisa/core/logging.h>
#include <luisa/core/stl/format.h>

namespace luisa::embed_mm {

void LanguageModel::load_weights(const Package &package,
                                 compute::CommandList &upload_cmds,
                                 luisa::vector<luisa::vector<float>> &half_storage) {
    _word_embeddings = upload_tensor_from_package(_device, package, "emb.word", upload_cmds, half_storage);
    _ple_proj = upload_tensor_from_package(_device, package, "ple.proj", upload_cmds, half_storage);
    _ple_norm = upload_tensor_from_package(_device, package, "ple.norm", upload_cmds, half_storage);
    _final_norm = upload_tensor_from_package(_device, package, "final_norm", upload_cmds, half_storage);
    _embedding_proj = upload_tensor_from_package(_device, package, "embedding_proj", upload_cmds, half_storage);

    _layers.resize(24);
    for (uint32_t i = 0; i < 24; i++) {
        auto &l = _layers[i];
        l.input_norm = upload_tensor_from_package(_device, package, luisa::format("layer{}.input_norm", i), upload_cmds, half_storage);
        l.q_proj = upload_tensor_from_package(_device, package, luisa::format("layer{}.q_proj", i), upload_cmds, half_storage);
        l.k_proj = upload_tensor_from_package(_device, package, luisa::format("layer{}.k_proj", i), upload_cmds, half_storage);
        l.v_proj = upload_tensor_from_package(_device, package, luisa::format("layer{}.v_proj", i), upload_cmds, half_storage);
        l.q_norm = upload_tensor_from_package(_device, package, luisa::format("layer{}.q_norm", i), upload_cmds, half_storage);
        l.k_norm = upload_tensor_from_package(_device, package, luisa::format("layer{}.k_norm", i), upload_cmds, half_storage);
        l.out_proj = upload_tensor_from_package(_device, package, luisa::format("layer{}.out_proj", i), upload_cmds, half_storage);
        l.post_attn_norm = upload_tensor_from_package(_device, package, luisa::format("layer{}.post_attn_norm", i), upload_cmds, half_storage);
        l.pre_ffn_norm = upload_tensor_from_package(_device, package, luisa::format("layer{}.pre_ffn_norm", i), upload_cmds, half_storage);
        l.gate_proj = upload_tensor_from_package(_device, package, luisa::format("layer{}.gate_proj", i), upload_cmds, half_storage);
        l.up_proj = upload_tensor_from_package(_device, package, luisa::format("layer{}.up_proj", i), upload_cmds, half_storage);
        l.down_proj = upload_tensor_from_package(_device, package, luisa::format("layer{}.down_proj", i), upload_cmds, half_storage);
        l.post_ffn_norm = upload_tensor_from_package(_device, package, luisa::format("layer{}.post_ffn_norm", i), upload_cmds, half_storage);
        l.ple_gate = upload_tensor_from_package(_device, package, luisa::format("layer{}.ple_gate", i), upload_cmds, half_storage);
        l.ple_proj = upload_tensor_from_package(_device, package, luisa::format("layer{}.ple_proj", i), upload_cmds, half_storage);
        l.ple_norm = upload_tensor_from_package(_device, package, luisa::format("layer{}.ple_norm", i), upload_cmds, half_storage);

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
}

void LanguageModel::compile_shaders() {
    _shader_embed_lookup = _device.compile(make_embed_lookup_kernel());
    _shader_scatter_soft_tokens = _device.compile(make_scatter_soft_tokens_kernel());
    _shader_scatter_indexed_tokens = _device.compile(make_scatter_indexed_tokens_kernel());
    _shader_rmsnorm = _device.compile(make_rmsnorm_kernel());
    _shader_rmsnorm_no_weight = _device.compile(make_rmsnorm_no_weight_kernel());
    _shader_gemm = _device.compile(make_gemm_kernel());
    _shader_rope = _device.compile(make_rope_kernel());
    _shader_gqa_scores = _device.compile(make_gqa_scores_kernel());
    _shader_softmax_row = _device.compile(make_softmax_row_kernel());
    _shader_gqa_pv = _device.compile(make_gqa_pv_kernel());
    _shader_geglu = _device.compile(make_geglu_kernel());
    _shader_residual_add = _device.compile(make_residual_add_kernel());
    _shader_ple_gate = _device.compile(make_ple_gate_kernel());
    _shader_scale_add = _device.compile(make_scale_add_kernel());
    _shader_mean_pool = _device.compile(make_mean_pool_kernel());
    _shader_mrl_normalize = _device.compile(make_mrl_normalize_kernel());
}

void LanguageModel::ensure_workspace(uint32_t B, uint32_t S) {
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

void LanguageModel::dispatch_rmsnorm(compute::CommandList &cmd_list,
                                     const compute::Buffer<float> &x,
                                     const compute::Buffer<float> &y,
                                     const compute::Buffer<float> &gamma,
                                     uint32_t num_rows, uint32_t H, float eps) {
    constexpr uint32_t kMaxRows = 32768u;
    for (uint32_t r = 0u; r < num_rows; r += kMaxRows) {
        uint32_t chunk = std::min(kMaxRows, num_rows - r);
        cmd_list << _shader_rmsnorm(x.view(r * H, chunk * H),
                                    y.view(r * H, chunk * H),
                                    gamma, H, eps)
                        .dispatch(chunk * kNormBlock);
    }
}

void LanguageModel::dispatch_rmsnorm_no_weight(compute::CommandList &cmd_list,
                                               const compute::Buffer<float> &x,
                                               const compute::Buffer<float> &y,
                                               uint32_t num_rows, uint32_t H, float eps) {
    constexpr uint32_t kMaxRows = 32768u;
    for (uint32_t r = 0u; r < num_rows; r += kMaxRows) {
        uint32_t chunk = std::min(kMaxRows, num_rows - r);
        cmd_list << _shader_rmsnorm_no_weight(x.view(r * H, chunk * H),
                                              y.view(r * H, chunk * H),
                                              H, eps)
                        .dispatch(chunk * kNormBlock);
    }
}

void LanguageModel::record_encode(
    compute::CommandList &cmd_list,
    compute::BufferView<uint32_t> input_ids,
    compute::BufferView<float> attention_mask,
    compute::BufferView<float> out_embeddings,
    uint32_t B,
    uint32_t S,
    uint32_t target_dim,
    const MultimodalInputs &mm_inputs) {

    ensure_workspace(B, S);

    constexpr uint32_t H = 512u;
    constexpr uint32_t I = 2048u;
    const float embed_scale = std::sqrt(512.0f);
    const uint32_t total_tokens = B * S;

    // 1. Embedding lookup
    cmd_list << _shader_embed_lookup(input_ids, _word_embeddings, _h0, H, embed_scale)
                    .dispatch(H, total_tokens);

    // 1.5. Scatter multimodal vision soft tokens if provided
    if (mm_inputs.vision_embeds.size() > 0 && mm_inputs.num_vision_tokens > 0) {
        if (mm_inputs.vision_token_indices.size() > 0) {
            cmd_list << _shader_scatter_indexed_tokens(
                            _h0, mm_inputs.vision_embeds, mm_inputs.vision_token_indices, mm_inputs.num_vision_tokens, H)
                            .dispatch(H, mm_inputs.num_vision_tokens);
        } else {
            cmd_list << _shader_scatter_soft_tokens(
                            _h0, mm_inputs.vision_embeds, mm_inputs.vision_start_idx, mm_inputs.num_vision_tokens, H)
                            .dispatch(H, mm_inputs.num_vision_tokens);
        }
    }

    // 1.6. Scatter multimodal audio soft tokens if provided
    if (mm_inputs.audio_embeds.size() > 0 && mm_inputs.num_audio_tokens > 0) {
        if (mm_inputs.audio_token_indices.size() > 0) {
            cmd_list << _shader_scatter_indexed_tokens(
                            _h0, mm_inputs.audio_embeds, mm_inputs.audio_token_indices, mm_inputs.num_audio_tokens, H)
                            .dispatch(H, mm_inputs.num_audio_tokens);
        } else {
            cmd_list << _shader_scatter_soft_tokens(
                            _h0, mm_inputs.audio_embeds, mm_inputs.audio_start_idx, mm_inputs.num_audio_tokens, H)
                            .dispatch(H, mm_inputs.num_audio_tokens);
        }
    }

    // 2. Global PLE projection & normalization
    cmd_list << _shader_gemm(_h0, _ple_proj, _ple_all, total_tokens, 24u * H, H)
                    .dispatch(gemm_dispatch_size(total_tokens, 24u * H));
    dispatch_rmsnorm(cmd_list, _ple_all, _ple_all, _ple_norm, total_tokens * 24u, H, 1e-6f);

    // 3. Transformer layers (24 blocks)
    for (uint32_t i = 0u; i < 24u; i++) {
        const auto &layer = _layers[i];

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
        cmd_list << _shader_rmsnorm(_h0, _h_norm, layer.input_norm, H, 1e-6f)
                        .dispatch(total_tokens * kNormBlock);

        // (b) Q, K, V projections
        cmd_list << _shader_gemm(_h_norm, layer.q_proj, _q_buf, total_tokens, q_dim, H)
                        .dispatch(gemm_dispatch_size(total_tokens, q_dim));
        cmd_list << _shader_gemm(_h_norm, layer.k_proj, _k_buf, total_tokens, kv_dim, H)
                        .dispatch(gemm_dispatch_size(total_tokens, kv_dim));
        cmd_list << _shader_gemm(_h_norm, layer.v_proj, _v_buf, total_tokens, kv_dim, H)
                        .dispatch(gemm_dispatch_size(total_tokens, kv_dim));

        // (c) Q, K, V RMSNorm
        dispatch_rmsnorm(cmd_list, _q_buf, _q_buf, layer.q_norm, total_tokens * num_q_heads, head_dim, 1e-6f);
        dispatch_rmsnorm(cmd_list, _k_buf, _k_buf, layer.k_norm, total_tokens * num_kv_heads, head_dim, 1e-6f);
        dispatch_rmsnorm_no_weight(cmd_list, _v_buf, _v_buf, total_tokens * num_kv_heads, head_dim, 1e-6f);

        // (d) RoPE on Q and K
        uint32_t q_half_pairs = total_tokens * num_q_heads * (head_dim / 2u);
        uint32_t k_half_pairs = total_tokens * num_kv_heads * (head_dim / 2u);
        cmd_list << _shader_rope(_q_buf, _q_buf, S, num_q_heads, head_dim, rope_theta)
                        .dispatch(q_half_pairs);
        cmd_list << _shader_rope(_k_buf, _k_buf, S, num_kv_heads, head_dim, rope_theta)
                        .dispatch(k_half_pairs);

        // (e) GQA Attention Scores + Softmax + PV
        cmd_list << _shader_gqa_scores(_q_buf, _k_buf, attention_mask, _logits,
                                       S, num_q_heads, num_kv_heads, head_dim,
                                       qk_scale, is_sliding, sliding_window, -1e9f, 0u)
                        .dispatch(S, B * num_q_heads * S);

        cmd_list << _shader_softmax_row(_logits, S)
                        .dispatch(kAttentionBlock, B * num_q_heads * S);

        cmd_list << _shader_gqa_pv(_v_buf, _logits, _attn_ctx,
                                   S, num_q_heads, num_kv_heads, head_dim, 0u)
                        .dispatch(head_dim, B * num_q_heads * S);

        // (f) Attention Out GEMM
        cmd_list << _shader_gemm(_attn_ctx, layer.out_proj, _attn_out, total_tokens, H, q_dim)
                        .dispatch(gemm_dispatch_size(total_tokens, H));

        // (g) Post-attention RMSNorm + Residual Add
        cmd_list << _shader_rmsnorm(_attn_out, _attn_norm, layer.post_attn_norm, H, 1e-6f)
                        .dispatch(total_tokens * kNormBlock);
        cmd_list << _shader_residual_add(_h0, _attn_norm, _h0, total_tokens * H)
                        .dispatch(std::min(total_tokens * H, 32768u * kMLPBlock));

        // (h) Pre-FFN RMSNorm
        cmd_list << _shader_rmsnorm(_h0, _ffn_norm, layer.pre_ffn_norm, H, 1e-6f)
                        .dispatch(total_tokens * kNormBlock);

        // (i) MLP Gate and Up projections
        cmd_list << _shader_gemm(_ffn_norm, layer.gate_proj, _gate_buf, total_tokens, I, H)
                        .dispatch(gemm_dispatch_size(total_tokens, I));
        cmd_list << _shader_gemm(_ffn_norm, layer.up_proj, _up_buf, total_tokens, I, H)
                        .dispatch(gemm_dispatch_size(total_tokens, I));

        // (j) GeGLU activation
        cmd_list << _shader_geglu(_gate_buf, _up_buf, _mlp_act, total_tokens * I)
                        .dispatch(std::min(total_tokens * I, 32768u * kMLPBlock));

        // (k) MLP Down projection
        cmd_list << _shader_gemm(_mlp_act, layer.down_proj, _mlp_out, total_tokens, H, I)
                        .dispatch(gemm_dispatch_size(total_tokens, H));

        // (l) Post-FFN RMSNorm + Residual Add
        cmd_list << _shader_rmsnorm(_mlp_out, _mlp_norm, layer.post_ffn_norm, H, 1e-6f)
                        .dispatch(total_tokens * kNormBlock);
        cmd_list << _shader_residual_add(_h0, _mlp_norm, _h0, total_tokens * H)
                        .dispatch(std::min(total_tokens * H, 32768u * kMLPBlock));

        // (m) PLE Block
        cmd_list << _shader_gemm(_h0, layer.ple_gate, _ple_gate_buf, total_tokens, H, H)
                        .dispatch(gemm_dispatch_size(total_tokens, H));
        cmd_list << _shader_ple_gate(_ple_gate_buf, _ple_all, _ple_gated, i, total_tokens, H)
                        .dispatch(std::min(total_tokens * H, 32768u * kMLPBlock));
        cmd_list << _shader_gemm(_ple_gated, layer.ple_proj, _ple_out, total_tokens, H, H)
                        .dispatch(gemm_dispatch_size(total_tokens, H));
        cmd_list << _shader_rmsnorm(_ple_out, _ple_norm_buf, layer.ple_norm, H, 1e-6f)
                        .dispatch(total_tokens * kNormBlock);

        // (n) Residual Add + Layer Scalar
        cmd_list << _shader_scale_add(_h0, _ple_norm_buf, _h0, layer.layer_scalar, total_tokens * H)
                        .dispatch(std::min(total_tokens * H, 32768u * kMLPBlock));
    }

    // 4. Final RMSNorm
    cmd_list << _shader_rmsnorm(_h0, _h_norm, _final_norm, H, 1e-6f)
                    .dispatch(total_tokens * kNormBlock);

    // 5. Mean pooling
    cmd_list << _shader_mean_pool(_h_norm, attention_mask, _pooled, S, H)
                    .dispatch(H, B);

    // 6. Projection (512 -> 768)
    cmd_list << _shader_gemm(_pooled, _embedding_proj, _proj_out, B, 768u, H)
                    .dispatch(gemm_dispatch_size(B, 768u));

    // 7. MRL Truncation + L2 Normalization
    cmd_list << _shader_mrl_normalize(_proj_out, out_embeddings, 768u, target_dim)
                    .dispatch(B * kPoolBlock);
}

}// namespace luisa::embed_mm

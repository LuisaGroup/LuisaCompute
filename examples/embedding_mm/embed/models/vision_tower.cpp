// embedding_mm engine -- Gemma4VisionModel + embed_vision connector implementation.
#include "embedding_mm/embed/models/vision_tower.h"
#include "embedding_mm/embed/models/embedding_gemma2.h"

#include <algorithm>
#include <luisa/core/logging.h>
#include <luisa/core/stl/format.h>

namespace luisa::embed_mm {

void VisionTower::load_weights(const Package &package,
                               compute::CommandList &upload_cmds,
                               luisa::vector<luisa::vector<float>> &half_storage) {
    _patch_proj = upload_tensor_from_package(_device, package, "vision.patch_embed.proj", upload_cmds, half_storage);
    _pos_table = upload_tensor_from_package(_device, package, "vision.patch_embed.pos_table", upload_cmds, half_storage);
    _connector_proj = upload_tensor_from_package(_device, package, "vision.connector.proj", upload_cmds, half_storage);

    _layers.resize(16);
    for (uint32_t i = 0; i < 16; i++) {
        auto &vl = _layers[i];
        vl.input_norm = upload_tensor_from_package(_device, package, luisa::format("vision.layer{}.input_norm", i), upload_cmds, half_storage);
        vl.q_proj = upload_tensor_from_package(_device, package, luisa::format("vision.layer{}.q_proj", i), upload_cmds, half_storage);
        vl.k_proj = upload_tensor_from_package(_device, package, luisa::format("vision.layer{}.k_proj", i), upload_cmds, half_storage);
        vl.v_proj = upload_tensor_from_package(_device, package, luisa::format("vision.layer{}.v_proj", i), upload_cmds, half_storage);
        vl.q_norm = upload_tensor_from_package(_device, package, luisa::format("vision.layer{}.q_norm", i), upload_cmds, half_storage);
        vl.k_norm = upload_tensor_from_package(_device, package, luisa::format("vision.layer{}.k_norm", i), upload_cmds, half_storage);
        vl.out_proj = upload_tensor_from_package(_device, package, luisa::format("vision.layer{}.out_proj", i), upload_cmds, half_storage);
        vl.post_attn_norm = upload_tensor_from_package(_device, package, luisa::format("vision.layer{}.post_attn_norm", i), upload_cmds, half_storage);
        vl.pre_ffn_norm = upload_tensor_from_package(_device, package, luisa::format("vision.layer{}.pre_ffn_norm", i), upload_cmds, half_storage);
        vl.gate_proj = upload_tensor_from_package(_device, package, luisa::format("vision.layer{}.gate_proj", i), upload_cmds, half_storage);
        vl.up_proj = upload_tensor_from_package(_device, package, luisa::format("vision.layer{}.up_proj", i), upload_cmds, half_storage);
        vl.down_proj = upload_tensor_from_package(_device, package, luisa::format("vision.layer{}.down_proj", i), upload_cmds, half_storage);
        vl.post_ffn_norm = upload_tensor_from_package(_device, package, luisa::format("vision.layer{}.post_ffn_norm", i), upload_cmds, half_storage);
    }
}

void VisionTower::compile_shaders() {
    _shader_scale_pixels = _device.compile(make_scale_pixels_kernel());
    _shader_add_patch_pos_embed = _device.compile(make_add_patch_pos_embed_kernel());
    _shader_mask_from_pos = _device.compile(make_mask_from_pos_kernel());
    _shader_rmsnorm = _device.compile(make_rmsnorm_kernel());
    _shader_rmsnorm_no_weight = _device.compile(make_rmsnorm_no_weight_kernel());
    _shader_gemm = _device.compile(make_gemm_kernel());
    _shader_axial_rope = _device.compile(make_axial_rope_kernel());
    _shader_gqa_scores = _device.compile(make_gqa_scores_kernel());
    _shader_softmax_row = _device.compile(make_softmax_row_kernel());
    _shader_gqa_pv = _device.compile(make_gqa_pv_kernel());
    _shader_geglu = _device.compile(make_geglu_kernel());
    _shader_residual_add = _device.compile(make_residual_add_kernel());
    _shader_vision_pool = _device.compile(make_vision_pool_kernel());
}

void VisionTower::ensure_workspace(uint32_t max_patches, uint32_t num_soft_tokens) {
    size_t needed_soft = std::max<size_t>({280u, num_soft_tokens, (max_patches + 8u) / 9u});
    if (max_patches <= _max_patches && needed_soft <= _max_vision_soft) {
        return;
    }
    _max_patches = std::max<size_t>(_max_patches, max_patches);
    _max_vision_soft = std::max<size_t>(_max_vision_soft, needed_soft);
    constexpr size_t H = 768;
    constexpr size_t I = 3072;
    constexpr size_t num_heads = 12;

    constexpr size_t kMaxVisionAttnRows = 30240u;
    size_t attn_rows = std::min<size_t>(num_heads * _max_patches, kMaxVisionAttnRows);

    _scaled_pixels = _device.create_buffer<float>(_max_patches * H);
    _h0 = _device.create_buffer<float>(_max_patches * H);
    _h_norm = _device.create_buffer<float>(_max_patches * H);
    _q_buf = _device.create_buffer<float>(_max_patches * H);
    _k_buf = _device.create_buffer<float>(_max_patches * H);
    _v_buf = _device.create_buffer<float>(_max_patches * H);
    _logits = _device.create_buffer<float>(attn_rows * _max_patches);
    _attn_ctx = _device.create_buffer<float>(_max_patches * H);
    _attn_out = _device.create_buffer<float>(_max_patches * H);
    _attn_norm = _device.create_buffer<float>(_max_patches * H);
    _ffn_norm = _device.create_buffer<float>(_max_patches * H);
    _gate_buf = _device.create_buffer<float>(_max_patches * I);
    _up_buf = _device.create_buffer<float>(_max_patches * I);
    _mlp_act = _device.create_buffer<float>(_max_patches * I);
    _mlp_out = _device.create_buffer<float>(_max_patches * H);
    _mlp_norm = _device.create_buffer<float>(_max_patches * H);
    _mask = _device.create_buffer<float>(_max_patches);
    _pooled = _device.create_buffer<float>(_max_vision_soft * H);
    _pooled_norm = _device.create_buffer<float>(_max_vision_soft * H);
}

void VisionTower::dispatch_rmsnorm(compute::CommandList &cmd_list,
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

void VisionTower::dispatch_rmsnorm_no_weight(compute::CommandList &cmd_list,
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

void VisionTower::record_encode(
    compute::CommandList &cmd_list,
    compute::BufferView<float> pixel_values,
    compute::BufferView<int32_t> position_ids,
    compute::BufferView<float> out_soft_tokens,
    uint32_t /*B*/,
    uint32_t num_patches_w,
    uint32_t num_patches_h,
    uint32_t max_patches,
    uint32_t num_soft_tokens,
    Executor::VisionDebugActivations *debug_acts) {

    ensure_workspace(max_patches, num_soft_tokens);

    constexpr uint32_t H = 768u;
    constexpr uint32_t I = 3072u;
    constexpr uint32_t num_heads = 12u;
    constexpr uint32_t head_dim = 64u;
    constexpr float rope_theta = 100.0f;
    constexpr float scale_pool = 3.079201435678004f; // sqrt(768) / 9.0

    // 1. Rescale pixels: [-1.0, 1.0]
    cmd_list << _shader_scale_pixels(pixel_values, _scaled_pixels, max_patches * H)
                    .dispatch(max_patches * H);

    // 2. Patch embedding GEMM: [max_patches, 768] * [768, 768] -> [max_patches, 768]
    cmd_list << _shader_gemm(_scaled_pixels, _patch_proj, _h0, max_patches, H, H)
                    .dispatch(gemm_dispatch_size(max_patches, H));

    // 3. Add 2D position embeddings
    cmd_list << _shader_add_patch_pos_embed(_h0, _pos_table, position_ids, H, 10240u)
                    .dispatch(H, max_patches);

    if (debug_acts && debug_acts->patch_embed) {
        cmd_list << debug_acts->patch_embed.copy_from(_h0.view(0, max_patches * H));
    }

    // 4. Generate attention mask from position_ids
    cmd_list << _shader_mask_from_pos(position_ids, _mask, max_patches)
                    .dispatch(max_patches);

    // 5. 16 Vision Transformer layers
    for (uint32_t l = 0u; l < 16u; l++) {
        const auto &layer = _layers[l];

        // (a) Input RMSNorm
        cmd_list << _shader_rmsnorm(_h0, _h_norm, layer.input_norm, H, 1e-6f)
                        .dispatch(max_patches * kNormBlock);

        // (b) Q, K, V projections
        cmd_list << _shader_gemm(_h_norm, layer.q_proj, _q_buf, max_patches, H, H)
                        .dispatch(gemm_dispatch_size(max_patches, H));
        cmd_list << _shader_gemm(_h_norm, layer.k_proj, _k_buf, max_patches, H, H)
                        .dispatch(gemm_dispatch_size(max_patches, H));
        cmd_list << _shader_gemm(_h_norm, layer.v_proj, _v_buf, max_patches, H, H)
                        .dispatch(gemm_dispatch_size(max_patches, H));

        // (c) Q, K, V RMSNorm
        dispatch_rmsnorm(cmd_list, _q_buf, _q_buf, layer.q_norm, max_patches * num_heads, head_dim, 1e-6f);
        dispatch_rmsnorm(cmd_list, _k_buf, _k_buf, layer.k_norm, max_patches * num_heads, head_dim, 1e-6f);
        dispatch_rmsnorm_no_weight(cmd_list, _v_buf, _v_buf, max_patches * num_heads, head_dim, 1e-6f);

        // (d) 2D Axial RoPE on Q and K
        cmd_list << _shader_axial_rope(_q_buf, _q_buf, position_ids, num_heads, head_dim, rope_theta)
                        .dispatch(16u, max_patches * num_heads);
        cmd_list << _shader_axial_rope(_k_buf, _k_buf, position_ids, num_heads, head_dim, rope_theta)
                        .dispatch(16u, max_patches * num_heads);

        // (e) Attention Scores + Softmax + PV (chunked across query-head rows to stay within 65535 grid & 32-bit buffer limits)
        constexpr uint32_t kMaxVisionAttnRows = 30240u;
        uint32_t total_attn_rows = num_heads * max_patches;
        for (uint32_t row_off = 0u; row_off < total_attn_rows; row_off += kMaxVisionAttnRows) {
            uint32_t chunk_rows = std::min(kMaxVisionAttnRows, total_attn_rows - row_off);
            cmd_list << _shader_gqa_scores(_q_buf, _k_buf, _mask, _logits,
                                           max_patches, num_heads, num_heads, head_dim,
                                           1.0f, 0u, 0u, -1e9f, row_off)
                            .dispatch(max_patches, chunk_rows);

            cmd_list << _shader_softmax_row(_logits, max_patches)
                            .dispatch(kAttentionBlock, chunk_rows);

            cmd_list << _shader_gqa_pv(_v_buf, _logits, _attn_ctx,
                                       max_patches, num_heads, num_heads, head_dim, row_off)
                            .dispatch(head_dim, chunk_rows);
        }

        // (f) Out projection GEMM
        cmd_list << _shader_gemm(_attn_ctx, layer.out_proj, _attn_out, max_patches, H, H)
                        .dispatch(gemm_dispatch_size(max_patches, H));

        // (g) Post-attn RMSNorm + Residual add
        cmd_list << _shader_rmsnorm(_attn_out, _attn_norm, layer.post_attn_norm, H, 1e-6f)
                        .dispatch(max_patches * kNormBlock);
        cmd_list << _shader_residual_add(_h0, _attn_norm, _h0, max_patches * H)
                        .dispatch(std::min(max_patches * H, 32768u * kMLPBlock));

        // (h) Pre-FFN RMSNorm
        cmd_list << _shader_rmsnorm(_h0, _ffn_norm, layer.pre_ffn_norm, H, 1e-6f)
                        .dispatch(max_patches * kNormBlock);

        // (i) MLP projections
        cmd_list << _shader_gemm(_ffn_norm, layer.gate_proj, _gate_buf, max_patches, I, H)
                        .dispatch(gemm_dispatch_size(max_patches, I));
        cmd_list << _shader_gemm(_ffn_norm, layer.up_proj, _up_buf, max_patches, I, H)
                        .dispatch(gemm_dispatch_size(max_patches, I));

        // (j) GeGLU activation
        cmd_list << _shader_geglu(_gate_buf, _up_buf, _mlp_act, max_patches * I)
                        .dispatch(std::min(max_patches * I, 32768u * kMLPBlock));

        // (k) Down projection GEMM
        cmd_list << _shader_gemm(_mlp_act, layer.down_proj, _mlp_out, max_patches, H, I)
                        .dispatch(gemm_dispatch_size(max_patches, H));

        // (l) Post-FFN RMSNorm + Residual add
        cmd_list << _shader_rmsnorm(_mlp_out, _mlp_norm, layer.post_ffn_norm, H, 1e-6f)
                        .dispatch(max_patches * kNormBlock);
        cmd_list << _shader_residual_add(_h0, _mlp_norm, _h0, max_patches * H)
                        .dispatch(std::min(max_patches * H, 32768u * kMLPBlock));

        if (l == 0u && debug_acts && debug_acts->layer0_out) {
            cmd_list << debug_acts->layer0_out.copy_from(_h0.view(0, max_patches * H));
        }
    }

    // 6. 2D Spatial Pooling (3x3 grid average + sqrt(768)) -> soft tokens
    uint32_t grid_w_soft = num_patches_w / 3u;
    uint32_t grid_h_soft = num_patches_h / 3u;
    uint32_t patch_stride = num_patches_w;
    cmd_list << _shader_vision_pool(_h0, _pooled, grid_w_soft, grid_h_soft, patch_stride, H, scale_pool)
                    .dispatch(H, num_soft_tokens);

    if (debug_acts && debug_acts->pooled_out) {
        cmd_list << debug_acts->pooled_out.copy_from(_pooled.view(0, num_soft_tokens * H));
    }

    // 7. Connector pre-norm (without scale)
    cmd_list << _shader_rmsnorm_no_weight(_pooled, _pooled_norm, H, 1e-6f)
                    .dispatch(num_soft_tokens * kNormBlock);

    // 8. Connector projection: [num_soft_tokens, 768] * [768, 512] -> [num_soft_tokens, 512]
    cmd_list << _shader_gemm(_pooled_norm, _connector_proj, out_soft_tokens, num_soft_tokens, 512u, H)
                    .dispatch(gemm_dispatch_size(num_soft_tokens, 512u));
}

}// namespace luisa::embed_mm

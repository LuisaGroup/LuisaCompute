// embedding_mm engine -- Gemma4AudioModel + embed_audio connector implementation.
#include "embedding_mm/embed/models/audio_tower.h"
#include "embedding_mm/embed/models/embedding_gemma2.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <luisa/core/logging.h>
#include <luisa/core/stl/format.h>

namespace luisa::embed_mm {

void AudioTower::load_weights(const Package &package,
                              compute::CommandList &upload_cmds,
                              luisa::vector<luisa::vector<float>> &half_storage) {
    _sub0_conv = upload_tensor_from_package(_device, package, "audio.subsample.layer0.conv", upload_cmds, half_storage);
    _sub0_norm = upload_tensor_from_package(_device, package, "audio.subsample.layer0.norm", upload_cmds, half_storage);
    _sub1_conv = upload_tensor_from_package(_device, package, "audio.subsample.layer1.conv", upload_cmds, half_storage);
    _sub1_norm = upload_tensor_from_package(_device, package, "audio.subsample.layer1.norm", upload_cmds, half_storage);
    _sub_input_proj = upload_tensor_from_package(_device, package, "audio.subsample.input_proj", upload_cmds, half_storage);

    _pos_embed = _device.create_buffer<float>(13 * 1024);
    _output_proj_weight = upload_tensor_from_package(_device, package, "audio.output_proj.weight", upload_cmds, half_storage);
    _output_proj_bias = upload_tensor_from_package(_device, package, "audio.output_proj.bias", upload_cmds, half_storage);
    _connector_proj = upload_tensor_from_package(_device, package, "audio.connector.proj", upload_cmds, half_storage);

    const auto *t_clips = package.find("audio.clip_bounds");
    const float *clip_src = nullptr;
    if (t_clips && t_clips->data) {
        clip_src = t_clips->data;
    }

    _layers.resize(12);
    for (uint32_t i = 0; i < 12; i++) {
        auto &al = _layers[i];
        al.ffn1_pre_norm = upload_tensor_from_package(_device, package, luisa::format("audio.layer{}.ffn1_pre_norm", i), upload_cmds, half_storage);
        al.ffn1_linear1 = upload_tensor_from_package(_device, package, luisa::format("audio.layer{}.ffn1_linear1", i), upload_cmds, half_storage);
        al.ffn1_linear2 = upload_tensor_from_package(_device, package, luisa::format("audio.layer{}.ffn1_linear2", i), upload_cmds, half_storage);
        al.ffn1_post_norm = upload_tensor_from_package(_device, package, luisa::format("audio.layer{}.ffn1_post_norm", i), upload_cmds, half_storage);

        al.norm_pre_attn = upload_tensor_from_package(_device, package, luisa::format("audio.layer{}.norm_pre_attn", i), upload_cmds, half_storage);
        al.q_proj = upload_tensor_from_package(_device, package, luisa::format("audio.layer{}.q_proj", i), upload_cmds, half_storage);
        al.k_proj = upload_tensor_from_package(_device, package, luisa::format("audio.layer{}.k_proj", i), upload_cmds, half_storage);
        al.v_proj = upload_tensor_from_package(_device, package, luisa::format("audio.layer{}.v_proj", i), upload_cmds, half_storage);
        al.per_dim_scale = upload_tensor_from_package(_device, package, luisa::format("audio.layer{}.per_dim_scale", i), upload_cmds, half_storage);
        al.relative_k_proj = upload_tensor_from_package(_device, package, luisa::format("audio.layer{}.relative_k_proj", i), upload_cmds, half_storage);
        al.out_proj = upload_tensor_from_package(_device, package, luisa::format("audio.layer{}.out_proj", i), upload_cmds, half_storage);
        al.norm_post_attn = upload_tensor_from_package(_device, package, luisa::format("audio.layer{}.norm_post_attn", i), upload_cmds, half_storage);

        al.lconv_pre_norm = upload_tensor_from_package(_device, package, luisa::format("audio.layer{}.lconv_pre_norm", i), upload_cmds, half_storage);
        al.lconv_linear_start = upload_tensor_from_package(_device, package, luisa::format("audio.layer{}.lconv_linear_start", i), upload_cmds, half_storage);
        al.lconv_depthwise = upload_tensor_from_package(_device, package, luisa::format("audio.layer{}.lconv_depthwise", i), upload_cmds, half_storage);
        al.lconv_norm = upload_tensor_from_package(_device, package, luisa::format("audio.layer{}.lconv_norm", i), upload_cmds, half_storage);
        al.lconv_linear_end = upload_tensor_from_package(_device, package, luisa::format("audio.layer{}.lconv_linear_end", i), upload_cmds, half_storage);

        al.ffn2_pre_norm = upload_tensor_from_package(_device, package, luisa::format("audio.layer{}.ffn2_pre_norm", i), upload_cmds, half_storage);
        al.ffn2_linear1 = upload_tensor_from_package(_device, package, luisa::format("audio.layer{}.ffn2_linear1", i), upload_cmds, half_storage);
        al.ffn2_linear2 = upload_tensor_from_package(_device, package, luisa::format("audio.layer{}.ffn2_linear2", i), upload_cmds, half_storage);
        al.ffn2_post_norm = upload_tensor_from_package(_device, package, luisa::format("audio.layer{}.ffn2_post_norm", i), upload_cmds, half_storage);

        al.norm_out = upload_tensor_from_package(_device, package, luisa::format("audio.layer{}.norm_out", i), upload_cmds, half_storage);

        if (clip_src) {
            std::memcpy(al.clip_bounds, clip_src + i * 40, sizeof(al.clip_bounds));
        } else {
            for (int j = 0; j < 10; j++) {
                al.clip_bounds[j][0] = -1e9f;
                al.clip_bounds[j][1] = 1e9f;
                al.clip_bounds[j][2] = -1e9f;
                al.clip_bounds[j][3] = 1e9f;
            }
        }
    }
}

void AudioTower::compile_shaders() {
    _shader_subsample_conv0 = _device.compile(make_audio_subsample_conv0_kernel());
    _shader_subsample_conv1 = _device.compile(make_audio_subsample_conv1_kernel());
    _shader_gemm = _device.compile(make_gemm_kernel());
    _shader_gemm_bias = _device.compile(make_gemm_bias_kernel());
    _shader_rmsnorm = _device.compile(make_rmsnorm_kernel());
    _shader_rmsnorm_no_weight = _device.compile(make_rmsnorm_no_weight_kernel());
    _shader_clamp = _device.compile(make_audio_clamp_kernel());
    _shader_scale_qk = _device.compile(make_audio_scale_qk_kernel());
    _shader_gen_rel_pos = _device.compile(make_audio_gen_rel_pos_kernel());
    _shader_chunked_attn = _device.compile(make_audio_chunked_attention_kernel());
    _shader_glu = _device.compile(make_audio_glu_kernel());
    _shader_depthwise_conv1d = _device.compile(make_audio_depthwise_conv1d_kernel());
    _shader_silu = _device.compile(make_audio_silu_kernel());
    _shader_add_scale = _device.compile(make_audio_add_scale_kernel());
    _shader_add = _device.compile(make_audio_add_kernel());
}

void AudioTower::ensure_workspace(uint32_t T_in, uint32_t T_out) {
    if (T_in <= _max_audio_in && T_out <= _max_audio_out) {
        return;
    }
    _max_audio_in = std::max(_max_audio_in, static_cast<size_t>(T_in));
    _max_audio_out = std::max(_max_audio_out, static_cast<size_t>(T_out));

    size_t H0 = (_max_audio_in + 1) / 2;
    size_t W0 = 64;
    _sub0 = _device.create_buffer<float>(128 * H0 * W0);
    _sub_flat = _device.create_buffer<float>(_max_audio_out * 1024);
    _hidden = _device.create_buffer<float>(_max_audio_out * 1024);
    _tmp1 = _device.create_buffer<float>(_max_audio_out * 1024);
    _tmp2 = _device.create_buffer<float>(_max_audio_out * 1024);
    _ffn_mid = _device.create_buffer<float>(_max_audio_out * 4096);
    _q = _device.create_buffer<float>(_max_audio_out * 1024);
    _k = _device.create_buffer<float>(_max_audio_out * 1024);
    _v = _device.create_buffer<float>(_max_audio_out * 1024);
    _rel_k = _device.create_buffer<float>(13 * 1024);
    _attn_out = _device.create_buffer<float>(_max_audio_out * 1024);
    _glu_in = _device.create_buffer<float>(_max_audio_out * 2048);
    _glu_out = _device.create_buffer<float>(_max_audio_out * 1024);
    _tower_out = _device.create_buffer<float>(_max_audio_out * 1536);
    _conn_in = _device.create_buffer<float>(_max_audio_out * 1536);
}

void AudioTower::record_encode(
    compute::CommandList &cmd_list,
    compute::BufferView<float> input_features,
    compute::BufferView<float> out_soft_tokens,
    uint32_t num_frames,
    uint32_t num_mel_bins,
    uint32_t num_soft_tokens,
    Executor::AudioDebugActivations *debug_acts) {

    ensure_workspace(num_frames, num_soft_tokens);

    constexpr uint32_t H = 1024u;
    uint32_t H0 = (num_frames + 1u) / 2u;
    uint32_t W0 = num_mel_bins / 2u;
    uint32_t H1 = num_soft_tokens;
    uint32_t W1 = W0 / 2u;

    // 1. Subsample Conv0: [num_frames, 128] -> [128, H0, W0]
    cmd_list << _shader_subsample_conv0(
                    input_features, _sub0_conv, _sub0_norm, _sub0,
                    num_frames, num_mel_bins, H0, W0, 1e-6f)
                    .dispatch(128u, W0, H0);

    // 2. Subsample Conv1: [128, H0, W0] -> flat [H1, 1024]
    cmd_list << _shader_subsample_conv1(
                    _sub0, _sub1_conv, _sub1_norm, _sub_flat,
                    H0, W0, H1, W1, 1e-6f)
                    .dispatch(32u, W1, H1);

    // 3. Subsample Input Proj GEMM: [H1, 1024] @ [1024, 1024] -> [H1, 1024]
    cmd_list << _shader_gemm(_sub_flat, _sub_input_proj, _hidden, H1, H, H)
                    .dispatch(gemm_dispatch_size(H1, H));
    if (debug_acts && debug_acts->subsample_proj) {
        cmd_list << debug_acts->subsample_proj.copy_from(_hidden.view(0, H1 * H));
    }

    // 4. Generate relative position table [13, 1024]
    cmd_list << _shader_gen_rel_pos(_pos_embed, 13u, H)
                    .dispatch(512u, 13u);

    // 5. 12 Conformer layers
    for (uint32_t l = 0u; l < 12u; l++) {
        const auto &layer = _layers[l];

        // --- FeedForward 1 ---
        cmd_list << _shader_rmsnorm(_hidden, _tmp1, layer.ffn1_pre_norm, H, 1e-6f)
                        .dispatch(H1 * kNormBlock);
        cmd_list << _shader_clamp(_tmp1, _tmp1, layer.clip_bounds[0][0], layer.clip_bounds[0][1], H1 * H)
                        .dispatch(H1 * H);
        cmd_list << _shader_gemm(_tmp1, layer.ffn1_linear1, _ffn_mid, H1, 4096u, H)
                        .dispatch(gemm_dispatch_size(H1, 4096u));
        cmd_list << _shader_clamp(_ffn_mid, _ffn_mid, layer.clip_bounds[0][2], layer.clip_bounds[0][3], H1 * 4096u)
                        .dispatch(H1 * 4096u);
        cmd_list << _shader_silu(_ffn_mid, _ffn_mid, H1 * 4096u)
                        .dispatch(H1 * 4096u);
        cmd_list << _shader_clamp(_ffn_mid, _ffn_mid, layer.clip_bounds[1][0], layer.clip_bounds[1][1], H1 * 4096u)
                        .dispatch(H1 * 4096u);
        cmd_list << _shader_gemm(_ffn_mid, layer.ffn1_linear2, _tmp1, H1, H, 4096u)
                        .dispatch(gemm_dispatch_size(H1, H));
        cmd_list << _shader_clamp(_tmp1, _tmp1, layer.clip_bounds[1][2], layer.clip_bounds[1][3], H1 * H)
                        .dispatch(H1 * H);
        cmd_list << _shader_rmsnorm(_tmp1, _tmp2, layer.ffn1_post_norm, H, 1e-6f)
                        .dispatch(H1 * kNormBlock);
        cmd_list << _shader_add_scale(_tmp2, _hidden, _hidden, 0.5f, H1 * H)
                        .dispatch(H1 * H);

        // --- Self Attention ---
        cmd_list << _shader_rmsnorm(_hidden, _tmp1, layer.norm_pre_attn, H, 1e-6f)
                        .dispatch(H1 * kNormBlock);
        cmd_list << _shader_clamp(_tmp1, _tmp1, layer.clip_bounds[2][0], layer.clip_bounds[2][1], H1 * H)
                        .dispatch(H1 * H);
        cmd_list << _shader_gemm(_tmp1, layer.q_proj, _q, H1, H, H)
                        .dispatch(gemm_dispatch_size(H1, H));
        cmd_list << _shader_gemm(_tmp1, layer.k_proj, _k, H1, H, H)
                        .dispatch(gemm_dispatch_size(H1, H));
        cmd_list << _shader_gemm(_tmp1, layer.v_proj, _v, H1, H, H)
                        .dispatch(gemm_dispatch_size(H1, H));
        cmd_list << _shader_clamp(_q, _q, layer.clip_bounds[2][2], layer.clip_bounds[2][3], H1 * H)
                        .dispatch(H1 * H);
        cmd_list << _shader_clamp(_k, _k, layer.clip_bounds[3][2], layer.clip_bounds[3][3], H1 * H)
                        .dispatch(H1 * H);
        cmd_list << _shader_clamp(_v, _v, layer.clip_bounds[4][2], layer.clip_bounds[4][3], H1 * H)
                        .dispatch(H1 * H);

        float q_scale = (std::pow(128.0f, -0.5f)) / std::log(2.0f);
        float k_scale = std::log(1.0f + std::exp(1.0f)) / std::log(2.0f);
        cmd_list << _shader_scale_qk(_q, _k, layer.per_dim_scale, H1, 8u, 128u, q_scale, k_scale)
                        .dispatch(H, H1);
        cmd_list << _shader_gemm(_pos_embed, layer.relative_k_proj, _rel_k, 13u, H, H)
                        .dispatch(gemm_dispatch_size(13u, H));
        cmd_list << _shader_chunked_attn(_q, _k, _v, _rel_k, _attn_out, H1, 8u, 128u, 12u, 50.0f)
                        .dispatch(8u, H1);

        cmd_list << _shader_clamp(_attn_out, _attn_out, layer.clip_bounds[5][0], layer.clip_bounds[5][1], H1 * H)
                        .dispatch(H1 * H);
        cmd_list << _shader_gemm(_attn_out, layer.out_proj, _tmp1, H1, H, H)
                        .dispatch(gemm_dispatch_size(H1, H));
        cmd_list << _shader_clamp(_tmp1, _tmp1, layer.clip_bounds[5][2], layer.clip_bounds[5][3], H1 * H)
                        .dispatch(H1 * H);
        cmd_list << _shader_rmsnorm(_tmp1, _tmp2, layer.norm_post_attn, H, 1e-6f)
                        .dispatch(H1 * kNormBlock);
        cmd_list << _shader_add(_tmp2, _hidden, _hidden, H1 * H)
                        .dispatch(H1 * H);

        // --- LightConv1D ---
        cmd_list << _shader_rmsnorm(_hidden, _tmp1, layer.lconv_pre_norm, H, 1e-6f)
                        .dispatch(H1 * kNormBlock);
        cmd_list << _shader_clamp(_tmp1, _tmp1, layer.clip_bounds[6][0], layer.clip_bounds[6][1], H1 * H)
                        .dispatch(H1 * H);
        cmd_list << _shader_gemm(_tmp1, layer.lconv_linear_start, _glu_in, H1, 2048u, H)
                        .dispatch(gemm_dispatch_size(H1, 2048u));
        cmd_list << _shader_clamp(_glu_in, _glu_in, layer.clip_bounds[6][2], layer.clip_bounds[6][3], H1 * 2048u)
                        .dispatch(H1 * 2048u);
        cmd_list << _shader_glu(_glu_in, _glu_out, H1, H)
                        .dispatch(H, H1);
        cmd_list << _shader_depthwise_conv1d(_glu_out, layer.lconv_depthwise, _tmp1, H1, H)
                        .dispatch(H, H1);
        cmd_list << _shader_rmsnorm(_tmp1, _tmp2, layer.lconv_norm, H, 1e-6f)
                        .dispatch(H1 * kNormBlock);
        cmd_list << _shader_silu(_tmp2, _tmp1, H1 * H)
                        .dispatch(H1 * H);
        cmd_list << _shader_clamp(_tmp1, _tmp1, layer.clip_bounds[7][0], layer.clip_bounds[7][1], H1 * H)
                        .dispatch(H1 * H);
        cmd_list << _shader_gemm(_tmp1, layer.lconv_linear_end, _tmp2, H1, H, H)
                        .dispatch(gemm_dispatch_size(H1, H));
        cmd_list << _shader_clamp(_tmp2, _tmp2, layer.clip_bounds[7][2], layer.clip_bounds[7][3], H1 * H)
                        .dispatch(H1 * H);
        cmd_list << _shader_add(_tmp2, _hidden, _hidden, H1 * H)
                        .dispatch(H1 * H);

        // --- FeedForward 2 ---
        cmd_list << _shader_rmsnorm(_hidden, _tmp1, layer.ffn2_pre_norm, H, 1e-6f)
                        .dispatch(H1 * kNormBlock);
        cmd_list << _shader_clamp(_tmp1, _tmp1, layer.clip_bounds[8][0], layer.clip_bounds[8][1], H1 * H)
                        .dispatch(H1 * H);
        cmd_list << _shader_gemm(_tmp1, layer.ffn2_linear1, _ffn_mid, H1, 4096u, H)
                        .dispatch(gemm_dispatch_size(H1, 4096u));
        cmd_list << _shader_clamp(_ffn_mid, _ffn_mid, layer.clip_bounds[8][2], layer.clip_bounds[8][3], H1 * 4096u)
                        .dispatch(H1 * 4096u);
        cmd_list << _shader_silu(_ffn_mid, _ffn_mid, H1 * 4096u)
                        .dispatch(H1 * 4096u);
        cmd_list << _shader_clamp(_ffn_mid, _ffn_mid, layer.clip_bounds[9][0], layer.clip_bounds[9][1], H1 * 4096u)
                        .dispatch(H1 * 4096u);
        cmd_list << _shader_gemm(_ffn_mid, layer.ffn2_linear2, _tmp1, H1, H, 4096u)
                        .dispatch(gemm_dispatch_size(H1, H));
        cmd_list << _shader_clamp(_tmp1, _tmp1, layer.clip_bounds[9][2], layer.clip_bounds[9][3], H1 * H)
                        .dispatch(H1 * H);
        cmd_list << _shader_rmsnorm(_tmp1, _tmp2, layer.ffn2_post_norm, H, 1e-6f)
                        .dispatch(H1 * kNormBlock);
        cmd_list << _shader_add_scale(_tmp2, _hidden, _hidden, 0.5f, H1 * H)
                        .dispatch(H1 * H);

        // --- Norm Out ---
        cmd_list << _shader_rmsnorm(_hidden, _hidden, layer.norm_out, H, 1e-6f)
                        .dispatch(H1 * kNormBlock);
        if (l == 0u && debug_acts && debug_acts->layer0_out) {
            cmd_list << debug_acts->layer0_out.copy_from(_hidden.view(0, H1 * H));
        }
        if (l == 11u && debug_acts && debug_acts->layer11_out) {
            cmd_list << debug_acts->layer11_out.copy_from(_hidden.view(0, H1 * H));
        }
    }

    // 6. Output Projection: GEMM with Bias [H1, 1024] @ [1024, 1536] + bias -> [H1, 1536]
    cmd_list << _shader_gemm_bias(_hidden, _output_proj_weight, _output_proj_bias, _tower_out, H1, 1536u, H)
                    .dispatch(gemm_dispatch_size(H1, 1536u));
    if (debug_acts && debug_acts->audio_tower_out) {
        cmd_list << debug_acts->audio_tower_out.copy_from(_tower_out.view(0, H1 * 1536u));
    }

    // 7. Connector: RMSNorm without weight [H1, 1536]
    cmd_list << _shader_rmsnorm_no_weight(_tower_out, _conn_in, 1536u, 1e-6f)
                    .dispatch(H1 * kNormBlock);

    // 8. Connector Projection: GEMM [H1, 1536] @ [1536, 512] -> out_soft_tokens [H1, 512]
    cmd_list << _shader_gemm(_conn_in, _connector_proj, out_soft_tokens, H1, 512u, 1536u)
                    .dispatch(gemm_dispatch_size(H1, 512u));
}

}// namespace luisa::embed_mm

// mm_embedding engine -- Audio Tower GPU kernel implementations.
#include "mm_embedding/embed/kernels/kernels_audio.h"
#include "mm_embedding/embed/kernels/kernels_gemm.h"

#include <cmath>
#include <luisa/dsl/sugar.h>

namespace luisa::mm_embed {

// #1: Subsample Conv2D Layer 0
// Conv2D (1 -> 128, 3x3, stride 2, pad 1) + LayerNorm(128, eps) + ReLU
// Block size: (128, 1, 1). Dispatch size: (128, W_out, H_out) = (128, 64, 154)
AudioSubsampleConv0Kernel make_audio_subsample_conv0_kernel() noexcept {
    using namespace luisa::compute;

    AudioSubsampleConv0Kernel kernel = [](BufferFloat in_features,
                                          BufferFloat weight,
                                          BufferFloat norm_weight,
                                          BufferFloat out_h0,
                                          Var<uint> H_in,
                                          Var<uint> W_in,
                                          Var<uint> H_out,
                                          Var<uint> W_out,
                                          Var<float> eps) noexcept {
        set_block_size(128u, 1u, 1u);
        set_name("audio_subsample_conv0");

        Shared<float> red{128u};

        Var tx = thread_x();
        Var w = dispatch_y();
        Var h = dispatch_z();

        // Each thread tx computes one output channel tx for pixel (w, h)
        Var v = def(0.0f);
        $if (w < W_out & h < H_out) {
            Var in_h_base = h.cast<int>() * 2 - 1;
            Var in_w_base = w.cast<int>() * 2 - 1;

            $for (kh, 3u) {
                $for (kw, 3u) {
                    Var ih = in_h_base + kh.cast<int>();
                    Var iw = in_w_base + kw.cast<int>();
                    $if (ih >= 0 & ih < H_in.cast<int>() & iw >= 0 & iw < W_in.cast<int>()) {
                        Var in_idx = ih.cast<uint>() * W_in + iw.cast<uint>();
                        Var w_idx = (tx * 3u + kh) * 3u + kw;
                        v += in_features.read(in_idx) * weight.read(w_idx);
                    };
                };
            };
        };

        // Shared memory reduction for LayerNorm across 128 channels
        auto block_sum = [&](Var<float> partial) noexcept {
            red[tx] = partial;
            sync_block();
            for (auto stride = 64u; stride > 0u; stride >>= 1u) {
                $if (tx < stride) {
                    red[tx] += red[tx + stride];
                };
                sync_block();
            };
            Var total = red[0u];
            sync_block();
            return total;
        };

        Var mean = block_sum(v) * (1.0f / 128.0f);
        Var diff = v - mean;
        Var var = block_sum(diff * diff) * (1.0f / 128.0f);
        Var inv_std = rsqrt(var + eps);

        $if (w < W_out & h < H_out) {
            Var normed = diff * inv_std * norm_weight.read(tx);
            Var val = max(0.0f, normed);
            out_h0.write(tx * (H_out * W_out) + h * W_out + w, val);
        };
    };
    return kernel;
}

// #2: Subsample Conv2D Layer 1
// Conv2D (128 -> 32, 3x3, stride 2, pad 1) + LayerNorm(32, eps) + ReLU
// Block size: (32, 1, 1). Dispatch size: (32, W_out, H_out) = (32, 32, 77)
// Outputs flat [77, 1024] at index h * 1024 + w * 32 + tx
AudioSubsampleConv1Kernel make_audio_subsample_conv1_kernel() noexcept {
    using namespace luisa::compute;

    AudioSubsampleConv1Kernel kernel = [](BufferFloat in_h0,
                                          BufferFloat weight,
                                          BufferFloat norm_weight,
                                          BufferFloat out_flat,
                                          Var<uint> H_in,
                                          Var<uint> W_in,
                                          Var<uint> H_out,
                                          Var<uint> W_out,
                                          Var<float> eps) noexcept {
        set_block_size(32u, 1u, 1u);
        set_name("audio_subsample_conv1");

        Shared<float> red{32u};

        Var tx = thread_x();
        Var w = dispatch_y();
        Var h = dispatch_z();

        Var v = def(0.0f);
        $if (w < W_out & h < H_out) {
            Var in_h_base = h.cast<int>() * 2 - 1;
            Var in_w_base = w.cast<int>() * 2 - 1;

            $for (in_c, 128u) {
                Var c_offset = in_c * (H_in * W_in);
                Var w_c_offset = (tx * 128u + in_c) * 9u;
                $for (kh, 3u) {
                    $for (kw, 3u) {
                        Var ih = in_h_base + kh.cast<int>();
                        Var iw = in_w_base + kw.cast<int>();
                        $if (ih >= 0 & ih < H_in.cast<int>() & iw >= 0 & iw < W_in.cast<int>()) {
                            Var in_idx = c_offset + ih.cast<uint>() * W_in + iw.cast<uint>();
                            Var w_idx = w_c_offset + kh * 3u + kw;
                            v += in_h0.read(in_idx) * weight.read(w_idx);
                        };
                    };
                };
            };
        };

        // Shared memory reduction across 32 channels
        auto block_sum = [&](Var<float> partial) noexcept {
            red[tx] = partial;
            sync_block();
            for (auto stride = 16u; stride > 0u; stride >>= 1u) {
                $if (tx < stride) {
                    red[tx] += red[tx + stride];
                };
                sync_block();
            };
            Var total = red[0u];
            sync_block();
            return total;
        };

        Var mean = block_sum(v) * (1.0f / 32.0f);
        Var diff = v - mean;
        Var var = block_sum(diff * diff) * (1.0f / 32.0f);
        Var inv_std = rsqrt(var + eps);

        $if (w < W_out & h < H_out) {
            Var normed = diff * inv_std * norm_weight.read(tx);
            Var val = max(0.0f, normed);
            out_flat.write(h * 1024u + w * 32u + tx, val);
        };
    };
    return kernel;
}

// #3: Audio Clamping Kernel
AudioClampKernel make_audio_clamp_kernel() noexcept {
    using namespace luisa::compute;

    AudioClampKernel kernel = [](BufferFloat x,
                                 BufferFloat y,
                                 Var<float> min_v,
                                 Var<float> max_v,
                                 Var<uint> total_elements) noexcept {
        set_block_size(256u, 1u, 1u);
        set_name("audio_clamp");

        Var idx = dispatch_x();
        $if (idx < total_elements) {
            Var val = x.read(idx);
            y.write(idx, clamp(val, min_v, max_v));
        };
    };
    return kernel;
}

// #4: GEMM with Bias: C = A @ B + bias
GemmBiasKernel make_gemm_bias_kernel() noexcept {
    using namespace luisa::compute;

    GemmBiasKernel kernel = [](BufferFloat A,
                               BufferFloat B,
                               BufferFloat bias,
                               BufferFloat C,
                               Var<uint> M,
                               Var<uint> N,
                               Var<uint> K) noexcept {
        set_block_size(kGemmTileSize, kGemmTileSize, 1u);
        set_name("gemm_bias");

        Shared<float> s_a{kGemmTileSize * kGemmTileSize};
        Shared<float> s_b{kGemmTileSize * kGemmTileSize};

        Var col = dispatch_x();
        Var row = dispatch_y();
        Var tx = thread_x();
        Var ty = thread_y();

        Var acc = def(0.0f);
        Var k_slab = def(0u);

        $while (k_slab < K) {
            Var a_k = k_slab + tx;
            Var a_val = def(0.0f);
            $if (row < M & a_k < K) {
                a_val = A.read(row * K + a_k);
            };
            s_a[ty * kGemmTileSize + tx] = a_val;

            Var b_k = k_slab + ty;
            Var b_val = def(0.0f);
            $if (col < N & b_k < K) {
                b_val = B.read(b_k * N + col);
            };
            s_b[ty * kGemmTileSize + tx] = b_val;

            sync_block();

            Var k = def(0u);
            $while (k < kGemmTileSize) {
                acc += s_a[ty * kGemmTileSize + k] * s_b[k * kGemmTileSize + tx];
                k += 1u;
            };

            sync_block();
            k_slab += kGemmTileSize;
        };

        $if (row < M & col < N) {
            C.write(row * N + col, acc + bias.read(col));
        };
    };
    return kernel;
}

// #5: Audio Scale Q & K
// q *= q_scale * softplus(per_dim_scale)
// k *= k_scale
AudioScaleQKKernel make_audio_scale_qk_kernel() noexcept {
    using namespace luisa::compute;

    AudioScaleQKKernel kernel = [](BufferFloat q,
                                   BufferFloat k,
                                   BufferFloat per_dim_scale,
                                   Var<uint> T,
                                   Var<uint> num_heads,
                                   Var<uint> head_dim,
                                   Var<float> q_scale,
                                   Var<float> k_scale) noexcept {
        set_block_size(128u, 1u, 1u);
        set_name("audio_scale_qk");

        Var ch = dispatch_x();
        Var t = dispatch_y();

        Var H = num_heads * head_dim;
        $if (ch < H & t < T) {
            Var d = ch % head_dim;
            Var pds = per_dim_scale.read(d);
            Var sp = def(0.0f);
            $if (pds > 20.0f) {
                sp = pds;
            }
            $else {
                sp = log(1.0f + exp(pds));
            };

            Var q_idx = t * H + ch;
            q.write(q_idx, q.read(q_idx) * (q_scale * sp));
            k.write(q_idx, k.read(q_idx) * k_scale);
        };
    };
    return kernel;
}

// #6: Relative Positional Encoding Generator
// pos_embed [13, 1024]: position_ids = [12, 11, ..., 0]
// inv_timescales[i] = exp(-i * ln(10000) / 511)
// pos_embed[p, i] = sin(pos_id * inv_ts), pos_embed[p, 512 + i] = cos(pos_id * inv_ts)
AudioGenRelPosKernel make_audio_gen_rel_pos_kernel() noexcept {
    using namespace luisa::compute;

    AudioGenRelPosKernel kernel = [](BufferFloat out_pos_embed,
                                     Var<uint> pos_len,
                                     Var<uint> hidden_size) noexcept {
        set_block_size(64u, 1u, 1u);
        set_name("audio_gen_rel_pos");

        Var d = dispatch_x(); // 0..511
        Var p = dispatch_y(); // 0..12

        Var half_dim = hidden_size / 2u;
        $if (d < half_dim & p < pos_len) {
            Var pos_id = (pos_len - 1u - p).cast<float>();
            Var log_inc = log(10000.0f) / (half_dim - 1u).cast<float>();
            Var inv_ts = exp(-d.cast<float>() * log_inc);
            Var scaled = pos_id * inv_ts;

            out_pos_embed.write(p * hidden_size + d, sin(scaled));
            out_pos_embed.write(p * hidden_size + half_dim + d, cos(scaled));
        };
    };
    return kernel;
}

// #7: Conformer Chunked Attention
// 12-token chunks, 24-token context, relative positional bias, softcap 50.0
// Dispatch 2D: (num_heads, T) = (8, T)
AudioChunkedAttentionKernel make_audio_chunked_attention_kernel() noexcept {
    using namespace luisa::compute;

    AudioChunkedAttentionKernel kernel = [](BufferFloat q,
                                            BufferFloat k,
                                            BufferFloat v,
                                            BufferFloat rel_k,
                                            BufferFloat out_attn,
                                            Var<uint> T,
                                            Var<uint> num_heads,
                                            Var<uint> head_dim,
                                            Var<uint> chunk_size,
                                            Var<float> softcap) noexcept {
        set_block_size(8u, 16u, 1u);
        set_name("audio_chunked_attention");

        Var h = dispatch_x();
        Var t_q = dispatch_y();

        $if (h < num_heads & t_q < T) {
            Var H = num_heads * head_dim;
            Var b = t_q / chunk_size;
            Var i = t_q % chunk_size;

            Local<float> logits{24u};
            Local<uint> valids{24u};

            Var max_logit = def(-1e9f);

            // Pass 1: compute attention logits for the 24 window positions
            $for (j, 24u) {
                Var t_k = b.cast<int>() * chunk_size.cast<int>() - 12 + j.cast<int>();
                Var is_valid = (t_k >= 0) & (t_k < T.cast<int>()) & (j <= 12u + i) & (j > i);
                valids[j] = is_valid.cast<uint>();

                Var logit = def(-1e9f);
                $if (is_valid) {
                    // Q @ K dot product
                    Var ac = def(0.0f);
                    $for (d, head_dim) {
                        Var q_val = q.read(t_q * H + h * head_dim + d);
                        Var k_val = k.read(t_k.cast<uint>() * H + h * head_dim + d);
                        ac += q_val * k_val;
                    };

                    // Relative positional bias: Q @ rel_k[j - i]
                    Var p = j - i;
                    Var bd = def(0.0f);
                    $if (p < 13u) {
                        $for (d, head_dim) {
                            Var q_val = q.read(t_q * H + h * head_dim + d);
                            Var rk_val = rel_k.read(p * H + h * head_dim + d);
                            bd += q_val * rk_val;
                        };
                    };

                    // Softcap tanh: 50.0 * tanh((ac + bd) / 50.0)
                    logit = tanh((ac + bd) * (1.0f / softcap)) * softcap;
                    $if (logit > max_logit) {
                        max_logit = logit;
                    };
                };
                logits[j] = logit;
            };

            // Pass 2: compute softmax sum
            Var sum_exp = def(0.0f);
            $for (j, 24u) {
                $if (valids[j] != 0u) {
                    sum_exp += exp(logits[j] - max_logit);
                };
            };
            Var inv_sum_exp = 1.0f / max(sum_exp, 1e-12f);

            // Pass 3: accumulate weighted values into output
            $for (d, head_dim) {
                Var acc = def(0.0f);
                $for (j, 24u) {
                    $if (valids[j] != 0u) {
                        Var t_k = (b.cast<int>() * chunk_size.cast<int>() - 12 + j.cast<int>()).cast<uint>();
                        Var weight = exp(logits[j] - max_logit) * inv_sum_exp;
                        Var v_val = v.read(t_k * H + h * head_dim + d);
                        acc += weight * v_val;
                    };
                };
                out_attn.write(t_q * H + h * head_dim + d, acc);
            };
        };
    };
    return kernel;
}

// #8: LightConv1D GLU: y = x1 * sigmoid(x2)
AudioGluKernel make_audio_glu_kernel() noexcept {
    using namespace luisa::compute;

    AudioGluKernel kernel = [](BufferFloat in_x,
                               BufferFloat out_y,
                               Var<uint> T,
                               Var<uint> H) noexcept {
        set_block_size(128u, 1u, 1u);
        set_name("audio_glu");

        Var c = dispatch_x();
        Var t = dispatch_y();

        $if (c < H & t < T) {
            Var in_idx1 = t * (H * 2u) + c;
            Var in_idx2 = t * (H * 2u) + H + c;
            Var x1 = in_x.read(in_idx1);
            Var x2 = in_x.read(in_idx2);
            Var sig = 1.0f / (1.0f + exp(-x2));
            out_y.write(t * H + c, x1 * sig);
        };
    };
    return kernel;
}

// #9: LightConv1D Causal Depthwise Conv1D (kernel 5, left pad 4)
AudioDepthwiseConv1DKernel make_audio_depthwise_conv1d_kernel() noexcept {
    using namespace luisa::compute;

    AudioDepthwiseConv1DKernel kernel = [](BufferFloat in_x,
                                           BufferFloat weight,
                                           BufferFloat out_y,
                                           Var<uint> T,
                                           Var<uint> channels) noexcept {
        set_block_size(128u, 1u, 1u);
        set_name("audio_depthwise_conv1d");

        Var c = dispatch_x();
        Var t = dispatch_y();

        $if (c < channels & t < T) {
            Var acc = def(0.0f);
            $for (k, 5u) {
                Var src_t = t.cast<int>() - 4 + k.cast<int>();
                $if (src_t >= 0) {
                    Var in_val = in_x.read(src_t.cast<uint>() * channels + c);
                    Var w_val = weight.read(c * 5u + k);
                    acc += in_val * w_val;
                };
            };
            out_y.write(t * channels + c, acc);
        };
    };
    return kernel;
}

// #10: SiLU activation: y = x * sigmoid(x)
AudioSiluKernel make_audio_silu_kernel() noexcept {
    using namespace luisa::compute;

    AudioSiluKernel kernel = [](BufferFloat x,
                                BufferFloat y,
                                Var<uint> total_elements) noexcept {
        set_block_size(256u, 1u, 1u);
        set_name("audio_silu");

        Var idx = dispatch_x();
        $if (idx < total_elements) {
            Var val = x.read(idx);
            Var sig = 1.0f / (1.0f + exp(-val));
            y.write(idx, val * sig);
        };
    };
    return kernel;
}

// #11: Add & Scale Residual: y = x * scale + residual
AudioAddScaleKernel make_audio_add_scale_kernel() noexcept {
    using namespace luisa::compute;

    AudioAddScaleKernel kernel = [](BufferFloat x,
                                    BufferFloat residual,
                                    BufferFloat y,
                                    Var<float> scale,
                                    Var<uint> total_elements) noexcept {
        set_block_size(256u, 1u, 1u);
        set_name("audio_add_scale");

        Var idx = dispatch_x();
        $if (idx < total_elements) {
            y.write(idx, x.read(idx) * scale + residual.read(idx));
        };
    };
    return kernel;
}

// #12: Add Residual: y = x + residual
AudioAddKernel make_audio_add_kernel() noexcept {
    using namespace luisa::compute;

    AudioAddKernel kernel = [](BufferFloat x,
                               BufferFloat residual,
                               BufferFloat y,
                               Var<uint> total_elements) noexcept {
        set_block_size(256u, 1u, 1u);
        set_name("audio_add");

        Var idx = dispatch_x();
        $if (idx < total_elements) {
            y.write(idx, x.read(idx) + residual.read(idx));
        };
    };
    return kernel;
}

// #13: Scatter audio soft tokens into language model sequence buffer
AudioScatterTokensKernel make_audio_scatter_tokens_kernel() noexcept {
    using namespace luisa::compute;

    AudioScatterTokensKernel kernel = [](BufferFloat h0,
                                         BufferFloat soft_tokens,
                                         Var<uint> start_token_idx,
                                         Var<uint> num_soft_tokens,
                                         Var<uint> H) noexcept {
        set_block_size(128u, 1u, 1u);
        set_name("audio_scatter_tokens");

        Var c = dispatch_x();
        Var t = dispatch_y();

        $if (c < H & t < num_soft_tokens) {
            Var dst_idx = (start_token_idx + t) * H + c;
            Var src_idx = t * H + c;
            h0.write(dst_idx, soft_tokens.read(src_idx));
        };
    };
    return kernel;
}

}// namespace luisa::mm_embed

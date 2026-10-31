// embedding_mm engine -- Attention operator implementations.
#include "embedding_mm/embed/ops/ops_attention.h"

#include <luisa/dsl/sugar.h>

namespace luisa::embed_mm {

GQAScoresKernel make_gqa_scores_kernel() noexcept {
    using namespace luisa::compute;

    GQAScoresKernel kernel = [](BufferFloat q, BufferFloat k, BufferFloat mask, BufferFloat logits,
                                Var<uint> S, Var<uint> num_q_heads, Var<uint> num_kv_heads,
                                Var<uint> head_dim, Var<float> scale,
                                Var<uint> is_sliding, Var<uint> sliding_window, Var<float> neg_large,
                                Var<uint> row_offset) noexcept {
        set_block_size(16u, 16u, 1u);
        set_name("gqa_scores");

        Var ki = dispatch_x();
        Var y_local = dispatch_y();
        Var total_y = dispatch_size().y;

        $if (ki < S & y_local < total_y) {
            Var y = y_local + row_offset;
            Var qi = y % S;
            Var head_batch = y / S;
            Var h_q = head_batch % num_q_heads;
            Var b = head_batch / num_q_heads;

            Var group_size = num_q_heads / num_kv_heads;
            Var h_kv = h_q / group_size;

            Var q_base = (b * S + qi) * (num_q_heads * head_dim) + h_q * head_dim;
            Var k_base = (b * S + ki) * (num_kv_heads * head_dim) + h_kv * head_dim;

            Var dot = def(0.0f);
            Var d = def(0u);
            $while (d < head_dim) {
                dot += q.read(q_base + d) * k.read(k_base + d);
                d += 1u;
            };

            Var score = dot * scale;

            Var is_masked = def(mask.read(b * S + ki) == 0.0f);
            $if (is_sliding != 0u) {
                Var diff = def(0u);
                $if (qi > ki) {
                    diff = qi - ki;
                }
                $else {
                    diff = ki - qi;
                };
                $if (diff > sliding_window) {
                    is_masked = true;
                };
            };
            $if (is_masked) {
                score = neg_large;
            };

            Var logits_idx = y_local * S + ki;
            logits.write(logits_idx, score);
        };
    };
    return kernel;
}

SoftmaxRowKernel make_softmax_row_kernel() noexcept {
    using namespace luisa::compute;

    SoftmaxRowKernel kernel = [](BufferFloat logits, Var<uint> S) noexcept {
        set_block_size(kAttentionBlock, 1u, 1u);
        set_name("softmax_row");

        Shared<float> red{kAttentionBlock};

        Var tx = thread_x();
        Var bs = block_size().x;
        Var row = dispatch_id().y * (dispatch_size().x / bs) + dispatch_id().x / bs;
        Var base = row * S;

        auto block_max = [&](Var<float> partial) noexcept {
            red[tx] = partial;
            sync_block();
            for (auto stride = kAttentionBlock / 2u; stride > 0u; stride >>= 1u) {
                $if (tx < stride) {
                    red[tx] = max(red[tx], red[tx + stride]);
                };
                sync_block();
            };
            Var total = red[0u];
            sync_block();
            return total;
        };

        auto block_sum = [&](Var<float> partial) noexcept {
            red[tx] = partial;
            sync_block();
            for (auto stride = kAttentionBlock / 2u; stride > 0u; stride >>= 1u) {
                $if (tx < stride) {
                    red[tx] += red[tx + stride];
                };
                sync_block();
            };
            Var total = red[0u];
            sync_block();
            return total;
        };

        Var m = def(-1e30f);
        Var i = tx;
        $while (i < S) {
            m = max(m, logits.read(base + i));
            i += bs;
        };
        Var row_max = block_max(m);

        Var s = def(0.0f);
        i = tx;
        $while (i < S) {
            Var val = exp(logits.read(base + i) - row_max);
            logits.write(base + i, val);
            s += val;
            i += bs;
        };
        Var row_sum = block_sum(s);
        Var inv_sum = 1.0f / max(row_sum, 1e-12f);

        i = tx;
        $while (i < S) {
            logits.write(base + i, logits.read(base + i) * inv_sum);
            i += bs;
        };
    };
    return kernel;
}

GQAPvKernel make_gqa_pv_kernel() noexcept {
    using namespace luisa::compute;

    GQAPvKernel kernel = [](BufferFloat v, BufferFloat logits, BufferFloat out,
                            Var<uint> S, Var<uint> num_q_heads, Var<uint> num_kv_heads,
                            Var<uint> head_dim, Var<uint> row_offset) noexcept {
        set_block_size(16u, 16u, 1u);
        set_name("gqa_pv");

        Var d = dispatch_x();
        Var y_local = dispatch_y();
        Var total_y = dispatch_size().y;

        $if (d < head_dim & y_local < total_y) {
            Var y = y_local + row_offset;
            Var qi = y % S;
            Var head_batch = y / S;
            Var h_q = head_batch % num_q_heads;
            Var b = head_batch / num_q_heads;

            Var group_size = num_q_heads / num_kv_heads;
            Var h_kv = h_q / group_size;

            Var sum = def(0.0f);
            Var ki = def(0u);
            $while (ki < S) {
                Var w = logits.read(y_local * S + ki);
                Var v_base = (b * S + ki) * (num_kv_heads * head_dim) + h_kv * head_dim;
                sum += w * v.read(v_base + d);
                ki += 1u;
            };

            Var out_offset = (b * S + qi) * (num_q_heads * head_dim) + h_q * head_dim + d;
            out.write(out_offset, sum);
        };
    };
    return kernel;
}

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

            $for (j, 24u) {
                Var t_k = b.cast<int>() * chunk_size.cast<int>() - 12 + j.cast<int>();
                Var is_valid = (t_k >= 0) & (t_k < T.cast<int>()) & (j <= 12u + i) & (j > i);
                valids[j] = is_valid.cast<uint>();

                Var logit = def(-1e9f);
                $if (is_valid) {
                    Var ac = def(0.0f);
                    $for (d, head_dim) {
                        Var q_val = q.read(t_q * H + h * head_dim + d);
                        Var k_val = k.read(t_k.cast<uint>() * H + h * head_dim + d);
                        ac += q_val * k_val;
                    };

                    Var p = j - i;
                    Var bd = def(0.0f);
                    $if (p < 13u) {
                        $for (d, head_dim) {
                            Var q_val = q.read(t_q * H + h * head_dim + d);
                            Var rk_val = rel_k.read(p * H + h * head_dim + d);
                            bd += q_val * rk_val;
                        };
                    };

                    logit = tanh((ac + bd) * (1.0f / softcap)) * softcap;
                    $if (logit > max_logit) {
                        max_logit = logit;
                    };
                };
                logits[j] = logit;
            };

            Var sum_exp = def(0.0f);
            $for (j, 24u) {
                $if (valids[j] != 0u) {
                    sum_exp += exp(logits[j] - max_logit);
                };
            };
            Var inv_sum_exp = 1.0f / max(sum_exp, 1e-12f);

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

}// namespace luisa::embed_mm

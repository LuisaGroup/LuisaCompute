// embedding_mm engine -- Positional Encoding operator implementations.
#include "embedding_mm/embed/ops/ops_rope.h"

#include <luisa/dsl/sugar.h>

namespace luisa::embed_mm {

RoPEKernel make_rope_kernel() noexcept {
    using namespace luisa::compute;

    RoPEKernel kernel = [](BufferFloat input, BufferFloat output,
                           Var<uint> S, Var<uint> num_heads,
                           Var<uint> head_dim, Var<float> rope_theta) noexcept {
        set_block_size(kRoPEBlock, 1u, 1u);
        set_name("rope");

        Var half_dim = head_dim / 2u;
        Var idx = dispatch_x();
        Var total_pairs = dispatch_size().x;

        $if (idx < total_pairs) {
            Var c = idx % half_dim;
            Var rem = idx / half_dim;
            Var h = rem % num_heads;
            Var token_idx = rem / num_heads;
            Var pos = token_idx % S;

            Var base_offset = (token_idx * num_heads + h) * head_dim;
            Var idx1 = base_offset + c;
            Var idx2 = base_offset + c + half_dim;

            Var freq_exp = (-2.0f * c.cast<float>()) / head_dim.cast<float>();
            Var theta = pos.cast<float>() * pow(rope_theta, freq_exp);

            Var cos_theta = cos(theta);
            Var sin_theta = sin(theta);

            Var v1 = input.read(idx1);
            Var v2 = input.read(idx2);

            Var rot1 = v1 * cos_theta - v2 * sin_theta;
            Var rot2 = v2 * cos_theta + v1 * sin_theta;

            output.write(idx1, rot1);
            output.write(idx2, rot2);
        };
    };
    return kernel;
}

AxialRoPEKernel make_axial_rope_kernel() noexcept {
    using namespace luisa::compute;

    AxialRoPEKernel kernel = [](BufferFloat qk_in, BufferFloat qk_out, BufferInt pos_ids,
                                Var<uint> num_heads, Var<uint> head_dim, Var<float> rope_theta) noexcept {
        set_block_size(16u, 16u, 1u);
        set_name("axial_rope");

        Var c = dispatch_x();
        Var y = dispatch_y();
        Var total_y = dispatch_size().y;

        $if (c < 16u & y < total_y) {
            Var h = y % num_heads;
            Var p = y / num_heads;

            Var px = pos_ids.read(p * 2u + 0u);
            Var py = pos_ids.read(p * 2u + 1u);

            Var exp_val = (2.0f * c.cast<float>()) / 32.0f;
            Var inv_freq = 1.0f / pow(rope_theta, exp_val);

            Var head_base = p * (num_heads * head_dim) + h * head_dim;

            $if (px >= 0 & py >= 0) {
                Var theta_x = px.cast<float>() * inv_freq;
                Var cos_x = cos(theta_x);
                Var sin_x = sin(theta_x);

                Var theta_y = py.cast<float>() * inv_freq;
                Var cos_y = cos(theta_y);
                Var sin_y = sin(theta_y);

                Var v0 = qk_in.read(head_base + c);
                Var v1 = qk_in.read(head_base + c + 16u);
                qk_out.write(head_base + c, v0 * cos_x - v1 * sin_x);
                qk_out.write(head_base + c + 16u, v1 * cos_x + v0 * sin_x);

                Var w0 = qk_in.read(head_base + 32u + c);
                Var w1 = qk_in.read(head_base + 48u + c);
                qk_out.write(head_base + 32u + c, w0 * cos_y - w1 * sin_y);
                qk_out.write(head_base + 48u + c, w1 * cos_y + w0 * sin_y);
            }
            $else {
                qk_out.write(head_base + c, 0.0f);
                qk_out.write(head_base + c + 16u, 0.0f);
                qk_out.write(head_base + 32u + c, 0.0f);
                qk_out.write(head_base + 48u + c, 0.0f);
            };
        };
    };
    return kernel;
}

AudioGenRelPosKernel make_audio_gen_rel_pos_kernel() noexcept {
    using namespace luisa::compute;

    AudioGenRelPosKernel kernel = [](BufferFloat out_pos_embed,
                                     Var<uint> pos_len,
                                     Var<uint> hidden_size) noexcept {
        set_block_size(64u, 1u, 1u);
        set_name("audio_gen_rel_pos");

        Var d = dispatch_x();
        Var p = dispatch_y();

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

}// namespace luisa::embed_mm

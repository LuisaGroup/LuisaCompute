// mm_embedding engine -- Rotary Position Embedding (RoPE) implementation.
#include "mm_embedding/embed/kernels/kernels_rope.h"

#include <luisa/dsl/sugar.h>

namespace luisa::mm_embed {

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

            // theta = pos * rope_theta^(-2*c / head_dim)
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

}// namespace luisa::mm_embed

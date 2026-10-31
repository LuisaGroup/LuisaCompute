// mm_embedding engine -- Vision Tower kernel implementations.
#include "mm_embedding/embed/kernels/kernels_vision.h"

#include <luisa/dsl/sugar.h>

namespace luisa::mm_embed {

ScalePixelsKernel make_scale_pixels_kernel() noexcept {
    using namespace luisa::compute;

    ScalePixelsKernel kernel = [](BufferFloat in, BufferFloat out, Var<uint> total) noexcept {
        set_block_size(256u, 1u, 1u);
        set_name("scale_pixels");

        Var i = dispatch_x();
        $if (i < total) {
            Var v = in.read(i);
            out.write(i, 2.0f * (v - 0.5f));
        };
    };
    return kernel;
}

AddPatchPosEmbedKernel make_add_patch_pos_embed_kernel() noexcept {
    using namespace luisa::compute;

    AddPatchPosEmbedKernel kernel = [](BufferFloat hidden, BufferFloat table,
                                       BufferInt pos_ids, Var<uint> H, Var<uint> pos_table_size) noexcept {
        set_block_size(16u, 16u, 1u);
        set_name("add_patch_pos_embed");

        Var d = dispatch_x();
        Var p = dispatch_y();
        Var total_patches = dispatch_size().y;

        $if (d < H & p < total_patches) {
            Var px = pos_ids.read(p * 2u + 0u);
            Var py = pos_ids.read(p * 2u + 1u);
            Var idx = p * H + d;

            $if (px >= 0 & py >= 0) {
                Var ux = px.cast<uint>();
                Var uy = py.cast<uint>();
                Var x_emb = table.read(0u * pos_table_size * H + ux * H + d);
                Var y_emb = table.read(1u * pos_table_size * H + uy * H + d);
                hidden.write(idx, hidden.read(idx) + x_emb + y_emb);
            }
            $else {
                hidden.write(idx, 0.0f);
            };
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

        Var c = dispatch_x(); // 0..15 (handles H pair [c, c+16] and W pair [32+c, 48+c])
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

                // Dim 0..31 (H)
                Var v0 = qk_in.read(head_base + c);
                Var v1 = qk_in.read(head_base + c + 16u);
                qk_out.write(head_base + c, v0 * cos_x - v1 * sin_x);
                qk_out.write(head_base + c + 16u, v1 * cos_x + v0 * sin_x);

                // Dim 32..63 (W)
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

VisionPoolKernel make_vision_pool_kernel() noexcept {
    using namespace luisa::compute;

    VisionPoolKernel kernel = [](BufferFloat in_hidden, BufferFloat out_pooled,
                                 Var<uint> grid_w_soft, Var<uint> grid_h_soft,
                                 Var<uint> patch_stride_w, Var<uint> H,
                                 Var<float> scale_factor) noexcept {
        set_block_size(16u, 16u, 1u);
        set_name("vision_pool");

        Var d = dispatch_x();
        Var s = dispatch_y();
        Var total_soft = dispatch_size().y;

        $if (d < H & s < total_soft) {
            Var x_prime = s % grid_w_soft;
            Var y_prime = s / grid_w_soft;

            Var sum = def(0.0f);
            for (uint32_t dy = 0u; dy < 3u; dy++) {
                for (uint32_t dx = 0u; dx < 3u; dx++) {
                    Var patch_x = x_prime * 3u + dx;
                    Var patch_y = y_prime * 3u + dy;
                    Var patch_idx = patch_y * patch_stride_w + patch_x;
                    sum += in_hidden.read(patch_idx * H + d);
                }
            }
            out_pooled.write(s * H + d, sum * scale_factor);
        };
    };
    return kernel;
}

ScatterSoftTokensKernel make_scatter_soft_tokens_kernel() noexcept {
    using namespace luisa::compute;

    ScatterSoftTokensKernel kernel = [](BufferFloat h0, BufferFloat soft_tokens,
                                        Var<uint> start_token_idx, Var<uint> num_soft_tokens,
                                        Var<uint> H) noexcept {
        set_block_size(16u, 16u, 1u);
        set_name("scatter_soft_tokens");

        Var d = dispatch_x();
        Var i = dispatch_y();

        $if (d < H & i < num_soft_tokens) {
            Var val = soft_tokens.read(i * H + d);
            h0.write((start_token_idx + i) * H + d, val);
        };
    };
    return kernel;
}

ScatterIndexedTokensKernel make_scatter_indexed_tokens_kernel() noexcept {
    using namespace luisa::compute;

    ScatterIndexedTokensKernel kernel = [](BufferFloat h0, BufferFloat soft_tokens,
                                           BufferUInt token_indices, Var<uint> num_soft_tokens,
                                           Var<uint> H) noexcept {
        set_block_size(16u, 16u, 1u);
        set_name("scatter_indexed_tokens");

        Var d = dispatch_x();
        Var i = dispatch_y();

        $if (d < H & i < num_soft_tokens) {
            Var token_pos = token_indices.read(i);
            Var val = soft_tokens.read(i * H + d);
            h0.write(token_pos * H + d, val);
        };
    };
    return kernel;
}

MaskFromPosKernel make_mask_from_pos_kernel() noexcept {
    using namespace luisa::compute;

    MaskFromPosKernel kernel = [](BufferInt pos_ids, BufferFloat out_mask, Var<uint> total) noexcept {
        set_block_size(256u, 1u, 1u);
        set_name("mask_from_pos");

        Var i = dispatch_x();
        $if (i < total) {
            Var px = pos_ids.read(i * 2u + 0u);
            out_mask.write(i, ite(px >= 0, 1.0f, 0.0f));
        };
    };
    return kernel;
}

}// namespace luisa::mm_embed

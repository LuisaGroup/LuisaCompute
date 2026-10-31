// embedding_mm engine -- Token Embedding, Patch Position Embedding, and Multimodal Scatter implementations.
#include "embedding_mm/embed/ops/ops_embed.h"

#include <luisa/dsl/sugar.h>

namespace luisa::embed_mm {

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

}// namespace luisa::embed_mm

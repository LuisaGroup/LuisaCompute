// embedding_mm engine -- Mask-aware Mean Pooling, MRL truncation, and L2 normalization implementation.
#include "embedding_mm/embed/kernels/kernels_pool.h"

#include <luisa/dsl/sugar.h>

namespace luisa::embed_mm {

MeanPoolKernel make_mean_pool_kernel() noexcept {
    using namespace luisa::compute;

    MeanPoolKernel kernel = [](BufferFloat hidden, BufferFloat mask, BufferFloat pooled,
                               Var<uint> S, Var<uint> H) noexcept {
        set_block_size(16u, 16u, 1u);
        set_name("mean_pool");

        Var d = dispatch_x();
        Var b = dispatch_y();
        Var total_b = dispatch_size().y;

        $if (d < H & b < total_b) {
            Var sum = def(0.0f);
            Var count = def(0.0f);
            Var s = def(0u);
            $while (s < S) {
                Var m = mask.read(b * S + s);
                Var val = hidden.read((b * S + s) * H + d);
                sum += m * val;
                count += m;
                s += 1u;
            };
            pooled.write(b * H + d, sum / max(count, 1.0f));
        };
    };
    return kernel;
}

MRLNormalizeKernel make_mrl_normalize_kernel() noexcept {
    using namespace luisa::compute;

    MRLNormalizeKernel kernel = [](BufferFloat in_proj, BufferFloat out_embed,
                                   Var<uint> D_proj, Var<uint> target_dim) noexcept {
        set_block_size(kPoolBlock, 1u, 1u);
        set_name("mrl_normalize");

        Shared<float> red{kPoolBlock};

        Var tx = thread_x();
        Var bs = block_size().x;
        Var b = dispatch_id().y * (dispatch_size().x / bs) + dispatch_id().x / bs;
        Var in_base = b * D_proj;
        Var out_base = b * target_dim;

        auto block_sum = [&](Var<float> partial) noexcept {
            red[tx] = partial;
            sync_block();
            for (auto stride = kPoolBlock / 2u; stride > 0u; stride >>= 1u) {
                $if (tx < stride) {
                    red[tx] += red[tx + stride];
                };
                sync_block();
            };
            Var total = red[0u];
            sync_block();
            return total;
        };

        Var s = def(0.0f);
        Var d = tx;
        $while (d < target_dim) {
            Var val = in_proj.read(in_base + d);
            s += val * val;
            d += bs;
        };
        Var sum_sq = block_sum(s);
        Var inv_norm = 1.0f / max(sqrt(sum_sq), 1e-12f);

        d = tx;
        $while (d < target_dim) {
            Var val = in_proj.read(in_base + d);
            out_embed.write(out_base + d, val * inv_norm);
            d += bs;
        };
    };
    return kernel;
}

}// namespace luisa::embed_mm

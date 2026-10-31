// embedding_mm engine -- Normalization operator implementations.
#include "embedding_mm/embed/ops/ops_norm.h"

#include <luisa/dsl/sugar.h>

namespace luisa::embed_mm {

RMSNormKernel make_rmsnorm_kernel() noexcept {
    using namespace luisa::compute;

    RMSNormKernel kernel = [](BufferFloat x, BufferFloat y, BufferFloat gamma,
                              Var<uint> H, Var<float> eps) noexcept {
        set_block_size(kNormBlock, 1u, 1u);
        set_name("rmsnorm");

        Shared<float> red{kNormBlock};

        Var tx = thread_x();
        Var bs = block_size().x;
        Var row = dispatch_id().y * (dispatch_size().x / bs) + dispatch_id().x / bs;
        Var base = row * H;
        Var Hf = H.cast<float>();

        auto block_sum = [&](Var<float> partial) noexcept {
            red[tx] = partial;
            sync_block();
            for (auto stride = kNormBlock / 2u; stride > 0u; stride >>= 1u) {
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
        Var h = tx;
        $while (h < H) {
            Var v = x.read(base + h);
            s += v * v;
            h += bs;
        };
        Var mean_sq = block_sum(s) / Hf;
        Var inv_rms = 1.0f / sqrt(mean_sq + eps);

        h = tx;
        $while (h < H) {
            Var v = x.read(base + h);
            Var g = gamma.read(h);
            y.write(base + h, v * inv_rms * g);
            h += bs;
        };
    };
    return kernel;
}

RMSNormNoWeightKernel make_rmsnorm_no_weight_kernel() noexcept {
    using namespace luisa::compute;

    RMSNormNoWeightKernel kernel = [](BufferFloat x, BufferFloat y,
                                      Var<uint> H, Var<float> eps) noexcept {
        set_block_size(kNormBlock, 1u, 1u);
        set_name("rmsnorm_no_weight");

        Shared<float> red{kNormBlock};

        Var tx = thread_x();
        Var bs = block_size().x;
        Var row = dispatch_id().y * (dispatch_size().x / bs) + dispatch_id().x / bs;
        Var base = row * H;
        Var Hf = H.cast<float>();

        auto block_sum = [&](Var<float> partial) noexcept {
            red[tx] = partial;
            sync_block();
            for (auto stride = kNormBlock / 2u; stride > 0u; stride >>= 1u) {
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
        Var h = tx;
        $while (h < H) {
            Var v = x.read(base + h);
            s += v * v;
            h += bs;
        };
        Var mean_sq = block_sum(s) / Hf;
        Var inv_rms = 1.0f / sqrt(mean_sq + eps);

        h = tx;
        $while (h < H) {
            Var v = x.read(base + h);
            y.write(base + h, v * inv_rms);
            h += bs;
        };
    };
    return kernel;
}

L2NormalizeKernel make_l2_normalize_kernel() noexcept {
    using namespace luisa::compute;

    L2NormalizeKernel kernel = [](BufferFloat x, BufferFloat y, Var<uint> D) noexcept {
        set_block_size(kNormBlock, 1u, 1u);
        set_name("l2_normalize");

        Shared<float> red{kNormBlock};

        Var tx = thread_x();
        Var bs = block_size().x;
        Var row = dispatch_id().y * (dispatch_size().x / bs) + dispatch_id().x / bs;
        Var base = row * D;

        auto block_sum = [&](Var<float> partial) noexcept {
            red[tx] = partial;
            sync_block();
            for (auto stride = kNormBlock / 2u; stride > 0u; stride >>= 1u) {
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
        $while (d < D) {
            Var v = x.read(base + d);
            s += v * v;
            d += bs;
        };
        Var sum_sq = block_sum(s);
        Var inv_norm = 1.0f / max(sqrt(sum_sq), 1e-12f);

        d = tx;
        $while (d < D) {
            Var v = x.read(base + d);
            y.write(base + d, v * inv_norm);
            d += bs;
        };
    };
    return kernel;
}

}// namespace luisa::embed_mm

// embedding_mm engine -- GEMM operator implementations.
#include "embedding_mm/embed/ops/ops_gemm.h"

#include <luisa/dsl/sugar.h>

namespace luisa::embed_mm {

GemmKernel make_gemm_kernel() noexcept {
    using namespace luisa::compute;

    GemmKernel kernel = [](BufferFloat A, BufferFloat B, BufferFloat C,
                           Var<uint> M, Var<uint> N, Var<uint> K) noexcept {
        set_block_size(kGemmTileSize, kGemmTileSize, 1u);
        set_name("gemm");

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
            C.write(row * N + col, acc);
        };
    };
    return kernel;
}

GemmBiasKernel make_gemm_bias_kernel() noexcept {
    using namespace luisa::compute;

    GemmBiasKernel kernel = [](BufferFloat A, BufferFloat B, BufferFloat bias, BufferFloat C,
                               Var<uint> M, Var<uint> N, Var<uint> K) noexcept {
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

}// namespace luisa::embed_mm

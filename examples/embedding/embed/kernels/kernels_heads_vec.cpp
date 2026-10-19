// bge-m3 embedding engine -- retrieval-head vec4 kernel implementations
// (tensor_optimize.md section 4.2 sparse head + section 4.3 ColBERT Float4
// accumulator + FMA; see kernels_heads_vec.h for the ONNX-op mapping and the
// host contract: H % 4 == 0 for sparse_head_vec4, H % 4 == 0 && N % 4 == 0 for
// colbert_proj_vec4; ragged sizes stay on the legacy scalar kernels).
//
// Same conventions as kernels_embed_vec.cpp / kernels_heads.cpp: device-free
// Kernel1D factories, explicit block 256, dispatch() = TOTAL THREADS, runtime
// dims, shared-memory halving-tree block reduction (CPU-fallback-backend safe;
// same idiom and summation order as kernels_norm.cpp), row index valid for a
// host-flattened 2D grid (DX 65535-group cap). The lane-internal horizontal sum
// is PINNED to ((a.x+a.y)+(a.z+a.w)) -- never dot() -- so the only deviation
// from the legacy bit pattern is the in-thread partial grouping; the legacy
// kernels remain the bit-level regression oracle (acceptance uses the existing
// numerical tolerance).
#include "embedding/embed/kernels/kernels_heads_vec.h"

#include <luisa/dsl/sugar.h>

namespace luisa::embed {

SparseHeadVec4Kernel make_sparse_head_vec4_kernel() noexcept {
    using namespace luisa::compute;

    // sparse_linear (kernels_heads.h sparse_head, ONNX MatMul[1024x1] + Add +
    // Relu) at float4 width: out[t] = relu(dot(X[t, :], W_s) + b_s), one scalar
    // per token row. A block per token; the 1024 dot becomes a lane-wise float4
    // partial accumulated over the strided vec4 loop, horizontally summed once
    // with the pinned order (no dot()), then folded through the shared-memory
    // halving tree. Bias Add, the strict ReLU predicate (v > 0) and the single
    // thread-0 scalar store mirror the legacy kernel exactly.
    SparseHeadVec4Kernel kernel = [](BufferFloat4 X4, BufferFloat4 W_s4, Var<float> b_s,
                                     BufferFloat out, Var<uint> H) noexcept {
        set_block_size(kHeadsVecBlock, 1u, 1u);
        set_name("sparse_head_vec4");
        Shared<float> red{kHeadsVecBlock};

        Var tx = thread_x();
        Var bs = block_size().x;
        // Flattened-block row index (1D dispatch -> y == 0).
        Var row = dispatch_id().y * (dispatch_size().x / bs) + dispatch_id().x / bs;
        // Hidden size in float4 lanes (H % 4 == 0 per the host contract); the
        // row base scales to lanes.
        Var H4 = H / 4u;
        Var base4 = row * H4;

        // Block reduction: shared-memory halving tree over the 256 per-thread
        // partials (same idiom and summation order as kernels_norm.cpp; the CPU
        // fallback backend implements only SYNCHRONIZE_BLOCK among the
        // thread-group ops, so warp intrinsics abort there -- the tree lowers to
        // plain shared memory + barriers and is bit-stable on every backend).
        // Each call ends with a barrier so `red` is not overwritten before all
        // threads have read the result.
        auto block_sum = [&](Var<float> partial) noexcept {
            red[tx] = partial;
            sync_block();
            // Plain C++ loop: compile-time-constant strides, fixed unroll.
            for (auto stride = kHeadsVecBlock / 2u; stride > 0u; stride >>= 1u) {
                $if (tx < stride) {
                    red[tx] += red[tx + stride];
                };
                sync_block();
            };
            Var total = red[0u];
            sync_block();// protect red against reuse by the next block_sum
            return total;
        };

        // Pinned lane-wise (vec4_io) summation order: (a.x+a.y)+(a.z+a.w), NOT
        // dot(); the block tree dominates the cost anyway.
        auto hsum = [](Float4 a) noexcept {
            return (a.x + a.y) + (a.z + a.w);
        };

        // Lane-wise float4 dot partial over the strided vec4 loop.
        Float4 p4 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        Var h4 = tx;
        $while (h4 < H4) {
            p4 += X4.read(base4 + h4) * W_s4.read(h4);
            h4 += bs;
        };
        Var partial = hsum(p4);

        Var v = block_sum(partial) + b_s;// dot + bias Add
        Var y = def(0.0f); // Relu epilogue (strict >, exact legacy predicate)
        $if (v > 0.0f) {
            y = v;
        };
        $if (tx == 0u) {
            out.write(row, y);
        };
    };
    return kernel;
}

ColbertProjVec4Kernel make_colbert_proj_vec4_kernel() noexcept {
    using namespace luisa::compute;

    // colbert_linear (kernels_heads.h colbert_proj, ONNX MatMul[1024x1024] +
    // Add; the caller L2-normalizes per token and drops special tokens
    // host-side) at float4 width, per tensor_optimize.md section 4.3: each
    // thread owns FOUR adjacent N outputs as one float4; the inner loop
    // broadcasts ONE float of x against one float4 of weight with FMA; bias,
    // output and the pre-normalization sum of squares are float4.
    ColbertProjVec4Kernel kernel = [](BufferFloat4 X4, BufferFloat4 W_c4, BufferFloat4 b_c4,
                                      BufferFloat4 out4, Var<uint> H, Var<uint> N,
                                      Var<uint> normalize) noexcept {
        set_block_size(kHeadsVecBlock, 1u, 1u);
        set_name("colbert_proj_vec4");
        Shared<float4> xs4{kHeadsVecMaxCachedRow4};// cached X row (2048 floats)
        Shared<float> red{kHeadsVecBlock};

        Var tx = thread_x();
        Var bs = block_size().x;
        Var row = dispatch_id().y * (dispatch_size().x / bs) + dispatch_id().x / bs;
        // Lane counts (H % 4 == 0 && N % 4 == 0 per the host contract) and the
        // row base in float4 units.
        Var H4 = H / 4u;
        Var N4 = N / 4u;
        Var base4 = row * H4;
        Var obase4 = row * N4;

        // Block reduction: shared-memory halving tree over the 256 per-thread
        // partials (same idiom and summation order as kernels_norm.cpp; the CPU
        // fallback backend implements only SYNCHRONIZE_BLOCK among the
        // thread-group ops, so warp intrinsics abort there -- the tree lowers to
        // plain shared memory + barriers and is bit-stable on every backend).
        // Each call ends with a barrier so `red` is not overwritten before all
        // threads have read the result.
        auto block_sum = [&](Var<float> partial) noexcept {
            red[tx] = partial;
            sync_block();
            // Plain C++ loop: compile-time-constant strides, fixed unroll.
            for (auto stride = kHeadsVecBlock / 2u; stride > 0u; stride >>= 1u) {
                $if (tx < stride) {
                    red[tx] += red[tx + stride];
                };
                sync_block();
            };
            Var total = red[0u];
            sync_block();// protect red against reuse by the next block_sum
            return total;
        };

        // Pinned lane-wise (vec4_io) summation order: (a.x+a.y)+(a.z+a.w).
        auto hsum = [](Float4 a) noexcept {
            return (a.x + a.y) + (a.z + a.w);
        };

        // Cooperative X-row load into shared memory when it fits (the block
        // reduction below needs the barrier either way; the sync_block is placed
        // OUTSIDE the divergent $if so every thread reaches it -- the barrier is
        // block-uniform). For H4 > kHeadsVecMaxCachedRow4 the row is re-read
        // from global inside the dot loop below (mirror of the legacy dual path;
        // bge-m3 never takes that path).
        $if (H4 <= kHeadsVecMaxCachedRow4) {
            Var h = tx;
            $while (h < H4) {
                xs4[h] = X4.read(base4 + h);
                h += bs;
            };
        };
        sync_block();

        // Each thread owns vec4-columns c = tx, tx + bs, ... over N4 = N/4
        // columns -- a [1, bs, N4] float4 tile of the GEMM; the per-column
        // weight read W_c4.read(h * N4 + c) coalesces along c (row-major [H, N],
        // pre-transposed per plan section 3.3).
        Float4 sum_sq4 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        Var c = tx;
        $while (c < N4) {
            Float4 acc4 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            // Outer loop over the H4 float4 slots of the X row; inner C++
            // unroll over the 4 floats of that slot broadcasts ONE x component
            // against one weight float4 with FMA -- each weight float4 is read
            // exactly once.
            Var h4 = def(0u);
            $while (h4 < H4) {
                $if (H4 <= kHeadsVecMaxCachedRow4) {
                    Float4 x4 = xs4[h4];
                    acc4 = fma(make_float4(x4.x), W_c4.read((h4 * 4u + 0u) * N4 + c), acc4);
                    acc4 = fma(make_float4(x4.y), W_c4.read((h4 * 4u + 1u) * N4 + c), acc4);
                    acc4 = fma(make_float4(x4.z), W_c4.read((h4 * 4u + 2u) * N4 + c), acc4);
                    acc4 = fma(make_float4(x4.w), W_c4.read((h4 * 4u + 3u) * N4 + c), acc4);
                }
                $else {
                    // Re-read the X slot from global (no shared cache).
                    Float4 x4 = X4.read(base4 + h4);
                    acc4 = fma(make_float4(x4.x), W_c4.read((h4 * 4u + 0u) * N4 + c), acc4);
                    acc4 = fma(make_float4(x4.y), W_c4.read((h4 * 4u + 1u) * N4 + c), acc4);
                    acc4 = fma(make_float4(x4.z), W_c4.read((h4 * 4u + 2u) * N4 + c), acc4);
                    acc4 = fma(make_float4(x4.w), W_c4.read((h4 * 4u + 3u) * N4 + c), acc4);
                };
                h4 += 1u;
            };
            Float4 y4 = acc4 + b_c4.read(c);// bias Add
            out4.write(obase4 + c, y4);
            $if (normalize != 0u) {
                sum_sq4 += y4 * y4;
            };
            c += bs;
        };

        // Fused per-row L2 normalization (plan section 4.3 item 3; mirrors the
        // legacy exactly): lane-wise sum of squares -> pinned horizontal order
        // -> block_sum -> sqrt -> explicit 1e-12 clamp -> a second strided divide
        // pass over the SAME columns this thread wrote. Each thread re-reads
        // only elements it wrote (same-thread global RAW is coherent), so only
        // the reduction needs barriers (inside block_sum).
        $if (normalize != 0u) {
            Var tot = block_sum(hsum(sum_sq4));
            Var d = sqrt(tot);
            $if (d < 1e-12f) {
                d = 1e-12f;
            };
            Var c2 = tx;
            $while (c2 < N4) {
                out4.write(obase4 + c2, out4.read(obase4 + c2) / make_float4(d));
                c2 += bs;
            };
        };
    };
    return kernel;
}

}// namespace luisa::embed

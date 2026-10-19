// bge-m3 embedding engine -- embedding-block vec4 kernel implementation
// (tensor_optimize.md section 4.2: embed lookup vec4 gather + LN).
//
// float4-width variant of embed_lookup (kernels_embed.cpp, plan section 4
// kernel #1). The ONNX-op mapping is identical, only the access width changes
// (H % 4 == 0 and 16B-aligned float4 buffers guaranteed by the caller -- see the
// header contract; ragged H stays on the legacy scalar kernel):
//   Gather (word_emb by token id)  -> word_emb4.read(ids[row]*H4 + h4)
//   Gather (pos_emb by 0..S-1)     -> pos_emb4.read((row % S)*H4 + h4) (absolute
//                                     position embeddings; positions are never
//                                     graph inputs -- plan section 0)
//   Add                            -> x4 = word4 + pos4
//   LayerNormalization (decomposed in ONNX as ReduceMean/Sub/Pow/ReduceMean/
//   Div/Sqrt/Mul/Add, eps=1e-5)    -> two-pass mean/var + affine, below.
// Launch: 1D over B*S token rows, block 256 (same as legacy); dispatch() count is
// TOTAL THREADS = rows * 256. One workgroup per token row; the LN mean/var use a
// shared-memory halving-tree block reduction (fallback-backend safe; same idiom
// as kernels_norm.cpp / kernels_embed.cpp).
// from the flattened (possibly 2D) grid so a host that must dodge the DX
// 65535-group cap can dispatch (cols*256, more_rows/cols) without changing the
// kernel (lc_optimize trap #2).
//
// NOTE for Phase B: dimensions (H, S, eps) are runtime kernel args; the kernel
// is compiled once and reused across shapes. bge-m3: H=1024 (H4=256, fits the
// shared row cache), vocab 250002, max positions 8194. The lane-internal
// horizontal sum is PINNED to ((a.x+a.y)+(a.z+a.w)) -- never dot() -- so the only
// deviation from the legacy bit pattern is the in-thread partial grouping; the
// legacy kernel remains the bit-level regression oracle.
#include "embedding/embed/kernels/kernels_embed_vec.h"

#include <luisa/dsl/sugar.h>

namespace luisa::embed {

EmbedLookupVec4Kernel make_embed_lookup_vec4_kernel() noexcept {
    using namespace luisa::compute;

    EmbedLookupVec4Kernel kernel = [](BufferFloat4 word_emb4, BufferUInt ids,
                                      BufferFloat4 pos_emb4, BufferFloat4 gamma4,
                                      BufferFloat4 beta4, BufferFloat4 out4,
                                      Var<uint> S, Var<uint> H, Var<float> eps,
                                      Var<uint> pos_offset) noexcept {
        set_block_size(kEmbedVecBlock, 1u, 1u);
        set_name("embed_lookup_vec4");

        Shared<float4> xs4{kEmbedVecMaxCachedRow4};// cached x-row (float4) when H4 <= cap
        Shared<float> red{kEmbedVecBlock};         // per-thread partials for the tree reduce

        Var tx = thread_x();
        Var bs = block_size().x;
        // Linear block index, valid for 1D dispatch and for a host-flattened
        // 2D grid (see file header). One workgroup per token row.
        Var row = dispatch_id().y * (dispatch_size().x / bs) + dispatch_id().x / bs;
        // Hidden size in float4 lanes (H % 4 == 0 per the host contract). The
        // LN mean/var divide by the FLOAT count H, not the lane count H4.
        Var H4 = H / 4u;
        Var Hf = H.cast<float>();
        // absolute position + offset: bge-m3's ONNX builds positions as
        // cumsum(mask)*mask + 1 (first real token = position row 2, see the
        // packer's header spec), XLM-R-family graphs use +2 uniformly, and
        // Phase-A-style tests pass 0. Padded rows shift uniformly and are
        // masked/dropped downstream, so a uniform offset is safe for them.
        Var pos = row % S + pos_offset;
        // Row bases in float4 units (same products as legacy, scaled to lanes):
        Var wbase4 = ids.read(row) * H4;// Gather #1 row base
        Var pbase4 = pos * H4;          // Gather #2 row base
        Var obase4 = row * H4;          // output row base

        // Block reduction: shared-memory halving tree over the 256 per-thread
        // partials (same idiom and summation order as kernels_embed.cpp /
        // kernels_norm.cpp). A warp-collective reduction (warp_active_sum) would
        // be cheaper on dx/vk, but the CPU fallback backend implements only
        // SYNCHRONIZE_BLOCK among the thread-group ops
        // (src/backends/fallback/fallback_codegen.cpp::_translate_thread_group_inst),
        // so warp intrinsics abort there -- the tree below lowers to plain
        // shared memory + barriers and behaves bit-identically (fixed summation
        // order) on every backend, including the deterministic parity oracle.
        //
        // Every thread calls it the same number of times (block-uniform control
        // flow only), and each call ends with a barrier so `red` is not
        // overwritten before all threads have read the result.
        auto block_sum = [&](Var<float> partial) noexcept {
            red[tx] = partial;
            sync_block();
            // Plain C++ loop: compile-time-constant strides, fixed unroll.
            for (auto stride = kEmbedVecBlock / 2u; stride > 0u; stride >>= 1u) {
                $if (tx < stride) {
                    red[tx] += red[tx + stride];
                };
                sync_block();
            };
            Var total = red[0u];
            sync_block();// protect red against reuse by the next block_sum
            return total;
        };

        // Pinned horizontal (lane-wise) summation order: (a.x+a.y)+(a.z+a.w).
        // NOT dot() -- this is the vec4_io variant; grouping is fixed and
        // block-uniform so the tree reduction above stays deterministic.
        auto hsum = [](Float4 a) noexcept {
            return (a.x + a.y) + (a.z + a.w);
        };

        // Pass 1: x = word_emb[id] + pos_emb[pos] (Add); thread-local row sum.
        Var sum_x = def(0.0f);
        $if (H4 <= kEmbedVecMaxCachedRow4) {
            Var h4 = tx;
            $while (h4 < H4) {
                Float4 x4 = word_emb4.read(wbase4 + h4) + pos_emb4.read(pbase4 + h4);
                xs4[h4] = x4;
                sum_x += hsum(x4);
                h4 += bs;
            };
        }
        $else {
            // Fallback for H4 > kEmbedVecMaxCachedRow4: no shared cache; passes
            // 2/3 re-gather x. Slower but correct (bge-m3 never takes this path).
            Var h4 = tx;
            $while (h4 < H4) {
                sum_x += hsum(word_emb4.read(wbase4 + h4) + pos_emb4.read(pbase4 + h4));
                h4 += bs;
            };
        };
        Var mean = block_sum(sum_x) / Hf;// ReduceMean (over H floats)

        // Pass 2: variance = ReduceMean((x - mean)^2) (Sub, Pow, ReduceMean).
        // (x4 - mean)*(x4 - mean) accumulated lane-wise, pinned horizontal order.
        Var sum_sq = def(0.0f);
        $if (H4 <= kEmbedVecMaxCachedRow4) {
            Var h4 = tx;
            $while (h4 < H4) {
                Float4 d4 = xs4[h4] - make_float4(mean);
                sum_sq += hsum(d4 * d4);
                h4 += bs;
            };
        }
        $else {
            Var h4 = tx;
            $while (h4 < H4) {
                Float4 d4 = word_emb4.read(wbase4 + h4) + pos_emb4.read(pbase4 + h4) - make_float4(mean);
                sum_sq += hsum(d4 * d4);
                h4 += bs;
            };
        };
        Var var = block_sum(sum_sq) / Hf;// biased variance (no Bessel)

        // Pass 3: y = ((x - mean) * (1 / sqrt(var + eps)) * gamma) + beta
        // (Div, Sqrt, Mul, Add). The CPU test reference uses this exact
        // expression order; keep it identical to kernels_embed.cpp.
        Var inv = 1.0f / sqrt(var + eps);
        $if (H4 <= kEmbedVecMaxCachedRow4) {
            Var h4 = tx;
            $while (h4 < H4) {
                Float4 y4 = ((xs4[h4] - make_float4(mean)) * make_float4(inv)) * gamma4.read(h4) + beta4.read(h4);
                out4.write(obase4 + h4, y4);
                h4 += bs;
            };
        }
        $else {
            Var h4 = tx;
            $while (h4 < H4) {
                Float4 x4 = word_emb4.read(wbase4 + h4) + pos_emb4.read(pbase4 + h4);
                Float4 y4 = ((x4 - make_float4(mean)) * make_float4(inv)) * gamma4.read(h4) + beta4.read(h4);
                out4.write(obase4 + h4, y4);
                h4 += bs;
            };
        };
    };
    return kernel;
}

}// namespace luisa::embed

// bge-m3 embedding engine -- embedding-block kernel implementation
// (plan §4 kernel #1, embed_lookup).
//
// ONNX-op mapping (plan §4 table, row #1):
//   Gather (word_emb by token id)  -> word_emb.read(ids[row]*H + h)
//   Gather (pos_emb by 0..S-1)     -> pos_emb.read((row % S)*H + h)   (absolute
//                                     position embeddings; positions are never
//                                     graph inputs -- plan §0)
//   Add                            -> x = word + pos
//   LayerNormalization (decomposed in ONNX as ReduceMean/Sub/Pow/ReduceMean/
//   Div/Sqrt/Mul/Add, eps=1e-5)    -> two-pass mean/var + affine, below.
// Launch: 1D over B*S token rows, block 256 (plan §4 #1); dispatch() count is
// TOTAL THREADS = rows * 256. One workgroup per token row; the LN mean/var use
// a shared-memory halving-tree block reduction (fallback-backend safe; same
// idiom as kernels_norm.cpp).
// from the flattened (possibly 2D) grid so a host that must dodge the DX
// 65535-group cap can dispatch (cols*256, more_rows/cols) without changing the
// kernel (lc_optimize trap #2).
//
// NOTE for Phase B: dimensions (H, S, eps) are runtime kernel args; the kernel
// is compiled once and reused across shapes. bge-m3: H=1024 (fits the shared
// row cache), vocab 250002, max positions 8194.
#include "embedding/embed/kernels/kernels_embed.h"

#include <luisa/dsl/sugar.h>

namespace luisa::embed {

namespace {
// Maximum hidden size cached in shared memory (4 B * (kEmbedBlock + 2048)
// <= 32 KiB shared floor). Larger H takes the re-gather fallback below.
constexpr uint kMaxCachedRow = 2048u;
}// namespace

EmbedLookupKernel make_embed_lookup_kernel() noexcept {
    using namespace luisa::compute;

    EmbedLookupKernel kernel = [](BufferFloat word_emb, BufferFloat pos_emb, BufferUInt ids,
                         BufferFloat gamma, BufferFloat beta, BufferFloat out,
                                                    Var<uint> S, Var<uint> H, Var<float> eps,
                           Var<uint> pos_offset) noexcept {
        set_block_size(kEmbedBlock, 1u, 1u);
        set_name("embed_lookup");

        Shared<float> xs{kMaxCachedRow}; // cached x-row when H <= kMaxCachedRow
        Shared<float> red{kEmbedBlock}; // per-thread partials for the tree reduce

        Var tx = thread_x();
        Var bs = block_size().x;
        // Linear block index, valid for 1D dispatch and for a host-flattened
        // 2D grid (see file header). One workgroup per token row.
        Var row = dispatch_id().y * (dispatch_size().x / bs) + dispatch_id().x / bs;
        // absolute position + offset: bge-m3's ONNX builds positions as
        // cumsum(mask)*mask + 1 (first real token = position row 2, see the
        // packer's header spec), XLM-R-family graphs use +2 uniformly, and
        // Phase-A-style tests pass 0. Padded rows shift uniformly and are
        // masked/dropped downstream, so a uniform offset is safe for them.
        Var pos = row % S + pos_offset;
        Var wbase = ids.read(row) * H;      // Gather #1 row base
        Var pbase = pos * H;                // Gather #2 row base
        Var obase = row * H;                // output row base
        Var Hf = H.cast<float>();

        // Block reduction: shared-memory halving tree over the 256 per-thread
        // partials (same idiom and summation order as kernels_norm.cpp). A
        // warp-collective reduction (warp_active_sum) would be cheaper on
        // dx/vk, but the CPU fallback backend implements only SYNCHRONIZE_BLOCK
        // among the thread-group ops
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
            for (auto stride = kEmbedBlock / 2u; stride > 0u; stride >>= 1u) {
                $if (tx < stride) {
                    red[tx] += red[tx + stride];
                };
                sync_block();
            };
            Var total = red[0u];
            sync_block(); // protect red against reuse by the next block_sum
            return total;
        };

        // Pass 1: x = word_emb[id] + pos_emb[pos] (Add); thread-local row sum.
        Var sum_x = def(0.0f);
        $if (H <= kMaxCachedRow) {
            Var h = tx;
            $while (h < H) {
                Var x = word_emb.read(wbase + h) + pos_emb.read(pbase + h);
                xs[h] = x;
                sum_x += x;
                h += bs;
            };
        } $else {
            // Fallback for H > kMaxCachedRow: no shared cache; passes 2/3
            // re-gather x. Slower but correct (bge-m3 never takes this path).
            Var h = tx;
            $while (h < H) {
                sum_x += word_emb.read(wbase + h) + pos_emb.read(pbase + h);
                h += bs;
            };
        };
        Var mean = block_sum(sum_x) / Hf; // ReduceMean

        // Pass 2: variance = ReduceMean((x - mean)^2) (Sub, Pow, ReduceMean).
        Var sum_sq = def(0.0f);
        $if (H <= kMaxCachedRow) {
            Var h = tx;
            $while (h < H) {
                Var d = xs[h] - mean;
                sum_sq += d * d;
                h += bs;
            };
        } $else {
            Var h = tx;
            $while (h < H) {
                Var d = word_emb.read(wbase + h) + pos_emb.read(pbase + h) - mean;
                sum_sq += d * d;
                h += bs;
            };
        };
        Var var = block_sum(sum_sq) / Hf;

        // Pass 3: y = ((x - mean) * (1 / sqrt(var + eps)) * gamma) + beta
        // (Div, Sqrt, Mul, Add). The CPU test reference uses this exact
        // expression order.
        Var inv = 1.0f / sqrt(var + eps);
        $if (H <= kMaxCachedRow) {
            Var h = tx;
            $while (h < H) {
                Var y = ((xs[h] - mean) * inv) * gamma.read(h) + beta.read(h);
                out.write(obase + h, y);
                h += bs;
            };
        } $else {
            Var h = tx;
            $while (h < H) {
                Var x = word_emb.read(wbase + h) + pos_emb.read(pbase + h);
                Var y = ((x - mean) * inv) * gamma.read(h) + beta.read(h);
                out.write(obase + h, y);
                h += bs;
            };
        };
    };
    return kernel;
}

}// namespace luisa::embed

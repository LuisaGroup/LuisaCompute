// bge-m3 embedding engine -- attention kernels (plan §4 rows #4-#6, milestone M3;
// fused flash attention, plan §4 "Attention & long sequences" + §6 milestone M4).
//
// ---------------------------------------------------------------------------
// ONNX-op mapping (plan §4 table)
// ---------------------------------------------------------------------------
// #4 attention_scores : MatMul(Q, Transpose(K)) + Mul(scale) + Add(mask bias)
//                       (the Softmax numerator of the 24 decomposed attention
//                       subgraphs) -> explicit [B*heads, S, S] fp32 logits.
// #5 softmax_row      : Softmax x24, in place over the logits workspace.
// #6 attention_pv     : MatMul(P, V) + Transpose x96; the head/transpose is
//                       folded into the output addressing, so the result is
//                       directly [B*S, num_heads*head_dim] for the "attn.out"
//                       GEMM (kernel #3).
// M4 flash_attention  : replaces #4-#6 in one pass with online softmax over
//                       streamed K/V tiles and NO S*S workspace (plan §4 memory
//                       analysis: 8192^2 * 4 B * 16 heads = 4 GB is infeasible).
//
// ---------------------------------------------------------------------------
// Layout (see kernels_attention.h, parent-owned contract)
// ---------------------------------------------------------------------------
//   qkv   : [B*S, 3*num_heads*head_dim] row-major fused QKV; for head h,
//           channel d:   Q at h*hd+d, K at num_heads*hd + h*hd+d,
//           V at 2*num_heads*hd + h*hd+d, row stride 3*num_heads*hd.
//   mask  : [B*S] fp32, 1.0 real / 0.0 pad; additive logit bias is
//           (1 - mask[b*S + KEY]) * neg_large (KEY side only). Query-side pad
//           rows are well-defined math (the kernels compute them like any other
//           row) but no consumer relies on them.
//   logits: [B*num_heads, S, S] fp32 (naive path only).
//   out   : [B*S, num_heads*head_dim] fp32.
//   scale : 1/sqrt(head_dim) (bge-m3: 1/8), runtime arg.
//
// ---------------------------------------------------------------------------
// M4 adaptation from examples/ml/attention_kernels.cpp
// ---------------------------------------------------------------------------
// The flash kernel is adapted from create_mha_online_attention_kernel() in
// examples/ml/attention_kernels.cpp -- specifically its scalar (non
// cooperative-vector) arithmetic, which is the only path that satisfies the
// plan's "pure compute shader, no tensor cores" constraint (plan §1.2, §4).
// Taken as-is: block-per-query-rows with one query row per thread, the K/V
// token tile broadcast through Shared<float>, the per-thread running max/sum
// rescale kept in registers (m, s_norm), the output update identity
// `o = o * exp(m - m_new) + V * exp(score - m_new)` (the MLA version holds o in
// a `$array` register file because its head_dim is constexpr; here it is a
// private shared slice, see the head_dim bullet below), and the single final
// `o / s_norm` write.
// Changed for bge-m3:
//   * fused single-buffer QKV layout with Q/K/V section offsets (the MLA kernel
//     reads three separate [B,heads,S,hd] tensors); output is written in the
//     interleaved [B*S, heads*hd] layout instead of [B,heads,S,hd].
//   * runtime S / num_heads / head_dim / scale instead of the constexpr MLA
//     config (attention_config.h): the kernel is compiled once per Session and
//     reused for every shape (CONVENTIONS.md kernel contract). The MLA kernel
//     static_asserts `seq_len % kTile == 0` and bakes `seq_len / kTile` into a
//     compile-time `$for`; here every loop bound is a runtime value and ragged
//     S (and ragged head_dim) are handled by predicated, zero-filling tile
//     loads, so no assumptions hold only for multiples of the tile.
//   * additive attention-mask bias per K tile ((1 - mask[key]) * neg_large with
//     runtime neg_large), which the MLA kernel does not have at all.
//   * grid flattened to 2D with the dispatch helpers below (DX 65535-group cap,
//     lc_optimize trap #2); the MLA kernel relies on a 1D grid <= 65535.
//   * head_dim is a runtime argument, so the per-thread row state that would
//     need a dynamically indexed register array lives in shared memory instead
//     (q_sh/o_sh: one private padded slice per thread, row stride 65 to avoid
//     bank conflicts); the only per-thread registers are the running max/sum.
//     The static-shared budget (2 * 32 * 65 + 2 * 8 * 65 + 8 floats = 20.3 KiB)
//     is what caps this variant at head_dim <= kFlashMaxHeadDim = 64 -- exactly
//     bge-m3's 1024/16, and the same head_dim the MLA reference kernel uses.
//     Shapes with a wider head_dim go through the naive trio, which accepts any
//     head_dim (and any S <= 1024); raising kFlashMaxHeadDim scales the shared
//     usage as 2*(rows + kTile)*(head_dim + 1) floats.
//   * no warp collectives anywhere: the fallback (CPU) backend implements only
//     SYNCHRONIZE_BLOCK among the thread-group ops, and the fallback run is
//     part of the numerical parity gate (plan §8) -- same reason as the
//     shared-memory halving tree in kernels_norm.cpp / kernels_embed.cpp.
// The original examples/ml files stay untouched on purpose: example_mla /
// example_multi_head_attention are mirrored xmake+CMake targets owned by the
// upstream examples tree (plan §4 "Reuse, don't reinvent", last table row --
// "we add new files, never edit these").
//
// ---------------------------------------------------------------------------
// Flash `scratch` contract (B1 allocates this)
// ---------------------------------------------------------------------------
// The flash kernel keeps ALL online-softmax state in registers (running max and
// sum) and shared memory (query row, output accumulator, K/V tiles): one block
// owns a (b, h, query-row-tile) and streams the whole key range itself, so no
// state ever has to be combined across blocks or across launches. Therefore
//
//     flash_attention_scratch_floats(B, S, num_heads) == 1u   (for every input)
//
// i.e. `scratch` is a DUMMY buffer: allocate one float (or reuse any existing
// 4-byte buffer) and pass it as the scratch argument. The kernel neither reads
// nor writes it. This is the "no global scratch" option allowed by the task
// contract; it removes the ~8 MiB tile-workspace from plan §5 entirely.
//
// ---------------------------------------------------------------------------
// Launch helpers (TOTAL-THREAD semantics; lc_optimize trap #1)
// ---------------------------------------------------------------------------
// Declared here with external linkage so callers (the tests, and later the
// executor / kernels_attention.h if the parent promotes them) use exactly one
// definition. All return either a total thread count (Kernel1D) or the
// (threads_x, blocks_y) pair for a Kernel2D, where x is always a multiple of
// the block size and the kernel derives the flattened block index from
// dispatch_id()/dispatch_size() -- so the same kernel is correct for a plain
// 2D grid and for a host-flattened one.
//
//   attention_scores_dispatch(B, S, heads)  -> {tiles_ki * 256, B*heads*tiles_qi}
//   softmax_row_dispatch(B, S, heads)       -> B*heads*S * 256 total threads (1D)
//   attention_pv_dispatch(B, S, heads, hd)  -> {tiles_d * 256, B*heads*tiles_qi}
//   flash_attention_dispatch(B, S, heads)   -> {tiles_q * 32, B*heads}
// with tiles_x = ceil(S / 16) for the naive tiles, ceil(S / 32) query-row tiles
// for flash (32 rows per block, one per thread), and tiles_d = ceil(head_dim / 16).
// Every helper returns an *exact* grid: the kernels do not bound themselves
// against over-dispatch, because CallOp::BUFFER_SIZE compiles to the constant
// 0xFFFFFFFF on the HLSL/DX path unless enable_debug_info is set.
//
// Range limits (M3 naive workspace, plan §4): the logits buffer is
// B*heads*S*S fp32 (64 MiB at B=1, heads=16, S=1024) -- use the flash path
// beyond that. softmax_row is a Kernel1D, so its 1D grid is capped at 65535
// groups per dimension on DX: the naive path requires B*heads*S <= 65535 rows
// (bge-m3's M3 shapes are far below it). The flash and the two GEMM-style
// kernels use 2D grids and are capped only per dimension.
#include "embedding/embed/kernels/kernels_attention.h"

#include <luisa/dsl/sugar.h>

namespace luisa::embed {

namespace {

using compute::def;
using compute::Var;

// Naive (M3) output tile side: 16x16 logits per block, 256 threads (same tiling
// as kernel #3, lc_optimize-measured best on vk/dx). Shared rows are padded by
// one float (kAttnStride) so the two stride-16 shared reads of a 32-lane wave
// land on different banks.
constexpr uint kAttnTile = 16u;
constexpr uint kAttnStride = kAttnTile + 1u;

// softmax_row shared exp-value cache: the naive path is defined for S <= 1024
// (4 KiB of shared), longer rows fall back to recomputing exp() in pass 3.
constexpr uint kSoftmaxCacheS = 1024u;

// Flash (M4) geometry: 32 query rows per block (one row per thread, block size
// 32 -- the DSL requires a multiple of 32), 8 keys per streamed K/V tile, and
// head_dim up to 64 (bge-m3: 1024/16). Shared bytes =
// (32*65) q + (32*65) o + (8*65) K + (8*65) V + 8 mask = 20.3 KiB, comfortably
// under the 32 KiB DX static-shared floor (plan §4 DSL rules).
constexpr uint kFlashBlock = 32u;
constexpr uint kFlashKeyTile = 8u;
constexpr uint kFlashMaxHeadDim = 64u;
constexpr uint kFlashStride = kFlashMaxHeadDim + 1u;

// Smallest float, used as the softmax -inf stand-in (avoids -inf/inf arithmetic
// in exp() on backends that flush denormals).
constexpr float kNegInf = -3.40282347e+38f;

// Flattened block index; correct for a 1D launch (y == 0) and for a
// host-flattened 2D launch with x a multiple of the block size (DX 65535-group
// cap dodge, lc_optimize trap #2 -- same idiom as kernels_norm.cpp).
[[nodiscard]] Var<uint> flattened_block_id() noexcept {
    const Var bs = compute::block_size().x;
    return compute::dispatch_id().y * (compute::dispatch_size().x / bs) +
           compute::dispatch_id().x / bs;
}

}// namespace

// NOTE (test-plan wave review, T2 finding): for a fully-masked query row with
// S not a multiple of kFlashKeyTile, the flash kernel's online-softmax
// denominator counts the ragged key-tile padding (ceil(S/8)*8 effective keys)
// while the naive trio divides by exactly S -- flash != naive on those rows
// (exact match when S % 8 == 0). Harmless for bge-m3 (a real query row never
// masks all keys; query-pad rows are dropped by every consumer), but a
// documented behavioral difference, not a tolerance.

//-----------------------------------------------------------------------------
// Launch helpers (see file header for the geometry each one describes)
//-----------------------------------------------------------------------------
uint2 attention_scores_dispatch(uint32_t batch, uint32_t seq_len, uint32_t num_heads) noexcept {
    const auto tiles = (seq_len + kAttnTile - 1u) / kAttnTile;
    return {tiles * kAttentionBlock, batch * num_heads * tiles};
}

uint2 attention_pv_dispatch(uint32_t batch, uint32_t seq_len, uint32_t num_heads,
                            uint32_t head_dim) noexcept {
    const auto tiles_q = (seq_len + kAttnTile - 1u) / kAttnTile;
    const auto tiles_d = (head_dim + kAttnTile - 1u) / kAttnTile;
    return {tiles_d * kAttentionBlock, batch * num_heads * tiles_q};
}

uint32_t softmax_row_dispatch(uint32_t batch, uint32_t seq_len, uint32_t num_heads) noexcept {
    // Total threads of a 1D launch: exactly one block of kAttentionBlock threads
    // per logits row, so the grid *is* the row count and the kernel needs no row
    // bound (dx cannot query one -- see make_softmax_row_kernel's comment on
    // CallOp::BUFFER_SIZE). Consequence: the naive path requires
    // B*heads*S <= 65535 groups per dimension (DX cap, lc_optimize trap #2);
    // bge-m3's M3 shapes (S <= 1024, heads = 16, small batches) are far below it,
    // and longer sequences use the flash kernel, whose 2D grid flattens instead.
    return batch * num_heads * seq_len * kAttentionBlock;
}

uint2 flash_attention_dispatch(uint32_t batch, uint32_t seq_len, uint32_t num_heads) noexcept {
    const auto tiles = (seq_len + kFlashBlock - 1u) / kFlashBlock;
    return {tiles * kFlashBlock, batch * num_heads};
}

uint32_t flash_attention_scratch_floats(uint32_t /*batch*/, uint32_t /*seq_len*/,
                                        uint32_t /*num_heads*/) noexcept {
    // No global scratch: see the "Flash `scratch` contract" block in the file
    // header (all online-softmax state is register/shared-resident).
    return 1u;
}

//-----------------------------------------------------------------------------
// #4 attention_scores (M3)
//-----------------------------------------------------------------------------
AttentionScoresKernel make_attention_scores_kernel() noexcept {
    using namespace luisa::compute;

    // One 16x16 logits tile per block (block 256 = kAttentionBlock). Q and K
    // head_dim streams through two 16x17 shared tiles (2 * 16 * 17 * 4 B = 2.2
    // KiB). Loads coalesce along head_dim, so they are predicated per element
    // against both ragged S and ragged head_dim; the fused scale and the
    // additive key-side mask bias are applied in the store.
    AttentionScoresKernel kernel = [](BufferFloat qkv, BufferFloat mask, BufferFloat logits,
                                      Var<uint> S, Var<uint> num_heads, Var<uint> head_dim,
                                      Var<float> scale, Var<float> neg_large) noexcept {
        set_block_size(kAttentionBlock, 1u, 1u);
        set_name("embed_attention_scores");

        Shared<float> tile_q{kAttnTile * kAttnStride};
        Shared<float> tile_k{kAttnTile * kAttnStride};

        const Var tid = thread_x();
        const Var tr = tid / kAttnTile;// row of the tile == query index
        const Var tc = tid % kAttnTile;// column of the tile == key index

        const Var tiles = (S + kAttnTile - 1u) / kAttnTile;
        const Var lin = flattened_block_id();
        const Var kt = lin % tiles;// key tile
        const Var rest = lin / tiles;
        const Var qt = rest % tiles;// query tile
        const Var bh = rest / tiles;// (batch, head)

        const Var qi = qt * kAttnTile + tr;// this thread's output query row
        const Var ki = kt * kAttnTile + tc;// this thread's output key column
        // The tile is staged by rows, so the K rows a thread loads are keyed by
        // tr while the key column it *computes* is keyed by tc.
        const Var ki_stage = kt * kAttnTile + tr;

        // The dispatch grid is exact (file header): each block maps to one
        // (batch-head, query-tile, key-tile) triple, so no per-block bound check
        // is needed -- and none is possible on dx, where buffer.size() is not
        // usable. Ragged S / head_dim are handled by the per-element guards.
        {
            const Var b = bh / num_heads;
            const Var h = bh % num_heads;
            const Var row_stride = 3u * num_heads * head_dim;
            const Var q_base = (b * S + qi) * row_stride + h * head_dim;
            const Var k_stage_base = (b * S + ki_stage) * row_stride + num_heads * head_dim +
                                     h * head_dim;

            Var acc = def(0.0f);
            $for (dt, (head_dim + kAttnTile - 1u) / kAttnTile) {
                const Var d0 = dt * kAttnTile;
                // Thread (tr, tc) stages Q[qi0+tr][d0+tc] into tile_q[tr][tc] and
                // K[ki0+tr][d0+tc] into tile_k[tr][tc]: both reads run along
                // head_dim for consecutive tc (coalesced), and every element of
                // both tiles is written (zero-filled) so no uninitialised shared
                // value can reach the dot product.
                Float qv;
                $if (qi < S & (d0 + tc) < head_dim) {
                    qv = qkv.read(q_base + d0 + tc);
                }
                $else {
                    qv = def(0.0f);
                };
                tile_q[tr * kAttnStride + tc] = qv;
                Float kv;
                $if (ki_stage < S & (d0 + tc) < head_dim) {
                    kv = qkv.read(k_stage_base + d0 + tc);
                }
                $else {
                    kv = def(0.0f);
                };
                tile_k[tr * kAttnStride + tc] = kv;
                sync_block();
                // Fixed compile-time unroll over the head_dim sub-tile
                // (CONVENTIONS.md: plain C++ loop, constant trip count). The
                // output (qi, ki) needs sum_d tile_q[tr][d] * tile_k[tc][d],
                // i.e. this thread's query row against the staged key row tc.
                for (auto j = 0u; j < kAttnTile; j++) {
                    acc += tile_q[tr * kAttnStride + j] * tile_k[tc * kAttnStride + j];
                }
                sync_block();
            };

            Float bias;
            $if (ki < S) {
                bias = (1.0f - mask.read(b * S + ki)) * neg_large;
            }
            $else {
                bias = def(0.0f);
            };

            $if (qi < S & ki < S) {
                logits.write((bh * S + qi) * S + ki, acc * scale + bias);
            };
        };
    };
    return kernel;
}

//-----------------------------------------------------------------------------
// #5 softmax_row (M3, in place)
//-----------------------------------------------------------------------------
SoftmaxRowKernel make_softmax_row_kernel() noexcept {
    using namespace luisa::compute;

    // One block per logits row (rows = B*heads*S, row length S runtime), so the
    // launch is exactly rows blocks of kAttentionBlock threads -- the grid *is*
    // the row count, which is why the kernel needs no row bound (and on dx it must
    // not query one: buffer.size() lowers to a constant 0xFFFFFFFF unless the
    // shader is compiled with enable_debug_info, see
    // src/backends/common/hlsl/codegen_utils/function_codegen.cpp:809).
    // Two-pass softmax -- max, then sum of exp, then the divide/write pass --
    // with shared-memory halving-tree reductions, matching the CPU reference's
    // math (subtract the row max before exp, so exp never overflows and a row
    // whose keys are all padded still yields a finite, uniform result instead of
    // NaN). The per-thread exp values are cached in shared memory when
    // S <= kSoftmaxCacheS, so exp() is evaluated once per element and the
    // write-back pass cannot disagree with the sum it was divided by.
    SoftmaxRowKernel kernel = [](BufferFloat logits, Var<uint> S) noexcept {
        set_block_size(kAttentionBlock, 1u, 1u);
        set_name("embed_softmax_row");

        Shared<float> red{kAttentionBlock};
        Shared<float> exp_cache{kSoftmaxCacheS};

        const Var tx = thread_x();
        const Var bs = block_size().x;
        const Var row = flattened_block_id();
        const Var base = row * S;

        // Block reductions: shared-memory halving trees (fallback-safe, no warp
        // collectives -- see kernels_norm.cpp for the full rationale). Each call
        // ends with a barrier so `red` is not overwritten before every thread
        // has read the result, and every thread calls them the same number of
        // times (block-uniform control flow only).
        auto block_max = [&](Var<float> partial) noexcept {
            red[tx] = partial;
            sync_block();
            for (auto stride = kAttentionBlock / 2u; stride > 0u; stride >>= 1u) {
                $if (tx < stride) {
                    red[tx] = max(red[tx], red[tx + stride]);
                };
                sync_block();
            };
            Var total = red[0u];
            sync_block();
            return total;
        };
        auto block_sum = [&](Var<float> partial) noexcept {
            red[tx] = partial;
            sync_block();
            for (auto stride = kAttentionBlock / 2u; stride > 0u; stride >>= 1u) {
                $if (tx < stride) {
                    red[tx] += red[tx + stride];
                };
                sync_block();
            };
            Var total = red[0u];
            sync_block();
            return total;
        };

        // Pass 1: row max (Softmax's subtract-max, ONNX ReduceMax).
        Var m = def(kNegInf);
        Var i = tx;
        $while (i < S) {
            m = max(m, logits.read(base + i));
            i += bs;
        };
        m = block_max(m);

        // Pass 2: sum of exp(x - m) (ONNX Exp + ReduceSum).
        Var sum = def(0.0f);
        $if (S <= kSoftmaxCacheS) {
            i = tx;
            $while (i < S) {
                const Var e = exp(logits.read(base + i) - m);
                exp_cache[i] = e;
                sum += e;
                i += bs;
            };
            sum = block_sum(sum);
            const Var inv = 1.0f / sum;
            // Pass 3: divide and store in place (ONNX Div).
            i = tx;
            $while (i < S) {
                logits.write(base + i, exp_cache[i] * inv);
                i += bs;
            };
        }
        $else {
            // S beyond the shared cache: recompute exp in pass 3. The naive path
            // is specified for S <= 1024, so this is only a correctness net, not
            // a supported production shape.
            i = tx;
            $while (i < S) {
                sum += exp(logits.read(base + i) - m);
                i += bs;
            };
            sum = block_sum(sum);
            const Var inv = 1.0f / sum;
            i = tx;
            $while (i < S) {
                logits.write(base + i, exp(logits.read(base + i) - m) * inv);
                i += bs;
            };
        };
    };
    return kernel;
}

//-----------------------------------------------------------------------------
// #6 attention_pv (M3)
//-----------------------------------------------------------------------------
AttentionPvKernel make_attention_pv_kernel() noexcept {
    using namespace luisa::compute;

    // out[b*S + qi, h*hd + d] = sum_ki P[bh][qi][ki] * V(b, ki, h, d), one 16x16
    // output tile (16 query rows x 16 head channels) per block. P (the
    // softmaxed logits) and V stream through two 16x17 shared tiles; the
    // per-head Transpose of the ONNX graph is folded into the output addressing.
    // Padded keys are already ~0 in P, so no mask is needed here.
    AttentionPvKernel kernel = [](BufferFloat qkv, BufferFloat logits, BufferFloat out,
                                  Var<uint> S, Var<uint> num_heads, Var<uint> head_dim) noexcept {
        set_block_size(kAttentionBlock, 1u, 1u);
        set_name("embed_attention_pv");

        Shared<float> tile_p{kAttnTile * kAttnStride};
        Shared<float> tile_v{kAttnTile * kAttnStride};

        const Var tid = thread_x();
        const Var tr = tid / kAttnTile;// query row / K-V row inside the tiles
        const Var tc = tid % kAttnTile;// key column / head channel inside them

        const Var tiles_q = (S + kAttnTile - 1u) / kAttnTile;
        const Var tiles_d = (head_dim + kAttnTile - 1u) / kAttnTile;
        const Var lin = flattened_block_id();
        const Var dt = lin % tiles_d;// head_dim tile
        const Var rest = lin / tiles_d;
        const Var qt = rest % tiles_q;
        const Var bh = rest / tiles_q;

        const Var qi = qt * kAttnTile + tr;
        const Var d = dt * kAttnTile + tc;

        // The dispatch grid is exact (file header): each block maps to one
        // (batch-head, query-tile, key-tile) triple, so no per-block bound check
        // is needed -- and none is possible on dx, where buffer.size() is not
        // usable. Ragged S / head_dim are handled by the per-element guards.
        {
            const Var b = bh / num_heads;
            const Var h = bh % num_heads;
            const Var row_stride = 3u * num_heads * head_dim;
            const Var p_base = (bh * S + qi) * S;
            const Var v_base = h * head_dim + 2u * num_heads * head_dim;

            Var acc = def(0.0f);
            $for (kt, (S + kAttnTile - 1u) / kAttnTile) {
                // tile_p is staged as P[row tr][column tc] (this thread loads one
                // P element per sub-tile, coalesced along the key index), while
                // tile_v is staged as V[row tr][channel tc]; so the P read is
                // keyed by tc and the V read by tr.
                const Var ki_row = kt * kAttnTile + tr;// V row of this thread
                const Var ki_col = kt * kAttnTile + tc;// P column of this thread
                Float pv;
                $if (qi < S & ki_col < S) {
                    pv = logits.read(p_base + ki_col);
                }
                $else {
                    pv = def(0.0f);
                };
                tile_p[tr * kAttnStride + tc] = pv;
                Float vv;
                $if (ki_row < S & d < head_dim) {
                    vv = qkv.read((b * S + ki_row) * row_stride + v_base + d);
                }
                $else {
                    vv = def(0.0f);
                };
                tile_v[tr * kAttnStride + tc] = vv;
                sync_block();
                for (auto j = 0u; j < kAttnTile; j++) {
                    acc += tile_p[tr * kAttnStride + j] * tile_v[j * kAttnStride + tc];
                }
                sync_block();
            };

            $if (qi < S & d < head_dim) {
                out.write((b * S + qi) * num_heads * head_dim + h * head_dim + d, acc);
            };
        };
    };
    return kernel;
}

//-----------------------------------------------------------------------------
// M4 flash attention (single fused kernel, replaces #4-#6)
//-----------------------------------------------------------------------------
FlashAttentionKernel make_flash_attention_kernel() noexcept {
    using namespace luisa::compute;

    // block = 32 threads = 32 query rows of one (b, h); each thread owns exactly
    // one query row end to end, so the online-softmax rescale state needs no
    // cross-thread reduction and no global scratch (see the file header). The
    // key range is streamed 8 tokens at a time through shared K/V tiles; a
    // ragged final tile is zero-filled and neutralised by the mask bias
    // (token >= S is treated as padding), so the whole tail needs no special
    // case. Shared-only state (query row + output accumulator, one private
    // padded slice per thread) is what makes head_dim a runtime argument; the
    // two per-thread registers are m (running max) and s_norm (running
    // denominator).
    //
    // Numerics are the online-softmax identity used by the MLA kernel this is
    // adapted from:
    //   m' = max(m, score); alpha = exp(m - m'); p = exp(score - m')
    //   s  = s * alpha + p;  o = o * alpha + V * p
    // which stays finite for a fully masked row (every score collapses onto the
    // running max, so p == 1 and no NaN/div-by-zero appears) -- query-side pad
    // rows are garbage by contract, and real rows always keep >= 1 unmasked key.
    FlashAttentionKernel kernel = [](BufferFloat qkv, BufferFloat mask, BufferFloat out,
                                     BufferFloat scratch, Var<uint> B, Var<uint> S,
                                     Var<uint> num_heads, Var<uint> head_dim,
                                     Var<float> scale, Var<float> neg_large) noexcept {
        set_block_size(kFlashBlock, 1u, 1u);
        set_name("embed_flash_attention");

        // Dummy per the scratch contract above; never touched.
        static_cast<void>(scratch);

        // Clamp head_dim to what the shared tiles can hold (the contract value
        // is 64 == bge-m3's head_dim, so the clamp is a no-op in production and
        // only keeps a misuse call memory-safe: it would then compute the first
        // 64 channels instead of running off the end of a shared slice).
        const Var hd = min(head_dim, def(kFlashMaxHeadDim));
        // head_dim rounded up to 4 so the unrolled-by-4 inner loops never read
        // past the initialised part of a tile; slots in [hd, hd_pad) are 0.
        const Var hd_pad = ((hd + 3u) / 4u) * 4u;

        Shared<float> q_sh{kFlashBlock * kFlashStride};
        Shared<float> o_sh{kFlashBlock * kFlashStride};
        Shared<float> k_sh{kFlashKeyTile * kFlashStride};
        Shared<float> v_sh{kFlashKeyTile * kFlashStride};
        Shared<float> m_sh{kFlashKeyTile};

        const Var tx = thread_x();
        const Var tiles_q = (S + kFlashBlock - 1u) / kFlashBlock;
        const Var bh_count = B * num_heads;
        const Var lin = flattened_block_id();
        const Var rt = lin % tiles_q;// query-row tile
        const Var bh = lin / tiles_q;
        const Var qi = rt * kFlashBlock + tx;// this thread's query row

        const Var row_stride = 3u * num_heads * head_dim;
        const Var qb = tx * kFlashStride;// private slice of q_sh / o_sh

        $if (bh < bh_count) {
            const Var b = bh / num_heads;
            const Var h = bh % num_heads;
            const Var h_off = h * hd;

            // Stage the query row (zeros for qi >= S and for [hd, hd_pad)), and
            // zero the output accumulator slice.
            $for (dq, hd_pad / 4u) {
                const Var d = dq * 4u;
                const Var q_base = (b * S + qi) * row_stride + h_off;
                Float q0;
                $if (qi < S & d < hd) {
                    q0 = qkv.read(q_base + d);
                }
                $else {
                    q0 = def(0.0f);
                };
                Float q1;
                $if (qi < S & (d + 1u) < hd) {
                    q1 = qkv.read(q_base + d + 1u);
                }
                $else {
                    q1 = def(0.0f);
                };
                Float q2;
                $if (qi < S & (d + 2u) < hd) {
                    q2 = qkv.read(q_base + d + 2u);
                }
                $else {
                    q2 = def(0.0f);
                };
                Float q3;
                $if (qi < S & (d + 3u) < hd) {
                    q3 = qkv.read(q_base + d + 3u);
                }
                $else {
                    q3 = def(0.0f);
                };
                q_sh[qb + d] = q0;
                q_sh[qb + d + 1u] = q1;
                q_sh[qb + d + 2u] = q2;
                q_sh[qb + d + 3u] = q3;
                o_sh[qb + d] = def(0.0f);
                o_sh[qb + d + 1u] = def(0.0f);
                o_sh[qb + d + 2u] = def(0.0f);
                o_sh[qb + d + 3u] = def(0.0f);
            };

            Var m = def(-1.0e30f);
            Var s_norm = def(0.0f);

            $for (jt, (S + kFlashKeyTile - 1u) / kFlashKeyTile) {
                // Cooperative K/V tile load: kFlashKeyTile * hd_pad elements,
                // exactly kFlashKeyTile * hd_pad / kFlashBlock per thread (the
                // division is exact because hd_pad is a multiple of 4 and the
                // tile holds 8 tokens). Every element of both tiles is written.
                $for (e, kFlashKeyTile * hd_pad / kFlashBlock) {
                    const Var idx = e * kFlashBlock + tx;
                    const Var j = idx / hd_pad;// token slot in the tile
                    const Var d = idx % hd_pad;// channel slot in the row
                    const Var tok = jt * kFlashKeyTile + j;
                    Float kv;
                    Float vv;
                    $if (tok < S & d < hd) {
                        const Var k_base = (b * S + tok) * row_stride + h_off;
                        kv = qkv.read(k_base + num_heads * hd + d);
                        vv = qkv.read(k_base + 2u * num_heads * hd + d);
                    }
                    $else {
                        kv = def(0.0f);
                        vv = def(0.0f);
                    };
                    k_sh[j * kFlashStride + d] = kv;
                    v_sh[j * kFlashStride + d] = vv;
                };
                // Mask bias is a per-key constant: one value per tile slot,
                // loaded by the first kFlashKeyTile threads (padding AND
                // out-of-range tail tokens become neg_large).
                $if (tx < kFlashKeyTile) {
                    const Var tok = jt * kFlashKeyTile + tx;
                    Float mv;
                    $if (tok < S) {
                        mv = mask.read(b * S + tok);
                    }
                    $else {
                        mv = def(0.0f);
                    };
                    m_sh[tx] = mv;
                };
                sync_block();

                // Fixed compile-time unroll over the tile's key slots.
                for (auto j = 0u; j < kFlashKeyTile; j++) {
                    const Var kb = j * kFlashStride;
                    // score = dot(q_i, k_j) * scale + (1 - mask[key]) * neg_large
                    Var score = def(0.0f);
                    $for (dq, hd_pad / 4u) {
                        const Var d = dq * 4u;
                        score += q_sh[qb + d] * k_sh[kb + d]
                                 + q_sh[qb + d + 1u] * k_sh[kb + d + 1u]
                                 + q_sh[qb + d + 2u] * k_sh[kb + d + 2u]
                                 + q_sh[qb + d + 3u] * k_sh[kb + d + 3u];
                    };
                    score = score * scale + (1.0f - m_sh[j]) * neg_large;

                    const Var m_new = max(m, score);
                    const Var exp_diff = exp(m - m_new);
                    const Var exp_score = exp(score - m_new);
                    s_norm = s_norm * exp_diff + exp_score;
                    $for (dq, hd_pad / 4u) {
                        const Var d = dq * 4u;
                        o_sh[qb + d] = o_sh[qb + d] * exp_diff + v_sh[kb + d] * exp_score;
                        o_sh[qb + d + 1u] = o_sh[qb + d + 1u] * exp_diff + v_sh[kb + d + 1u] * exp_score;
                        o_sh[qb + d + 2u] = o_sh[qb + d + 2u] * exp_diff + v_sh[kb + d + 2u] * exp_score;
                        o_sh[qb + d + 3u] = o_sh[qb + d + 3u] * exp_diff + v_sh[kb + d + 3u] * exp_score;
                    };
                    m = m_new;
                };
                // Everyone must be done with this tile before it is overwritten.
                sync_block();
            };

            // Single normalised write of the whole row.
            $if (qi < S) {
                const Var o_base = (b * S + qi) * num_heads * head_dim + h_off;
                $for (d, hd) {
                    out.write(o_base + d, o_sh[qb + d] / s_norm);
                };
            };
        };
    };
    return kernel;
}

}// namespace luisa::embed

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
// * head_dim is a runtime argument. The M4 baseline therefore kept the per-thread
// row state in shared memory (q_sh/o_sh: one private padded slice per thread, row
// stride 65 to avoid bank conflicts) and only the running max/sum in registers,
// because a dynamically indexed local array does not stay in registers.
// The O-B tuned variants remove that compromise (see "O-B optimization pass"
// below): the register family unrolls the head_dim axis at EMISSION time over
// kFlashMaxHeadDim slots with compile-time indices, so the query row, the PV
// accumulator, the running max and the running sum are all per-thread registers
// even though head_dim itself is runtime (slots [head_dim, 64) are zero-filled by
// the predicated loads and contribute exactly 0). Only the streamed K/V tiles stay
// in shared memory.
// The static-shared budget of the baseline (2 * 32 * 65 + 2 * 8 * 65 + 8 floats =
// 20.3 KiB) is what caps that variant at head_dim <= kFlashMaxHeadDim = 64 --
// exactly bge-m3's 1024/16, and the same head_dim the MLA reference kernel uses.
// Shapes with a wider head_dim go through the naive trio, which accepts any
// head_dim (and any S <= 1024); raising kFlashMaxHeadDim scales the shared usage
// as 2*(rows + kTile)*(head_dim + 1) floats on the baseline family and as
// 2*kTile*(head_dim + 1) floats on the register family.
// * no warp collectives anywhere: the fallback (CPU) backend implements only
// SYNCHRONIZE_BLOCK among the thread-group ops (every other ThreadGroupInst hits
// LUISA_NOT_IMPLEMENTED in src/backends/fallback/fallback_codegen.cpp) and rejects
// any pinned warp size but 1, and the fallback run is part of the numerical parity
// gate (plan §8) -- same reason as the shared-memory halving tree in
// kernels_norm.cpp / kernels_embed.cpp. See the "O-B optimization pass" note on
// plan item A2 below.
// The original examples/ml files stay untouched on purpose: example_mla /
// example_multi_head_attention are mirrored xmake+CMake targets owned by the
// upstream examples tree (plan §4 "Reuse, don't reinvent", last table row --
// "we add new files, never edit these").
//
// ---------------------------------------------------------------------------
// O-B optimization pass (embedding_model_opt_plan.md §3.2 A1-A5) -- perf-only
// ---------------------------------------------------------------------------
// The fused kernel is emitted from a variant table (kFlashVariants) selected at
// runtime through LUISA_EMBED_FLASH_CONFIG / LUISA_EMBED_FLASH_BLOCK /
// LUISA_EMBED_FLASH_TILE / LUISA_EMBED_FLASH_REG; index 0 (legacy_b32_t8, block 32,
// tile 8, shared row state) is the byte-identical M4 baseline and is the kill
// switch: `LUISA_EMBED_FLASH_CONFIG=legacy_b32_t8` restores the shipped behaviour
// including the DSL kernel name "embed_flash_attention".
// A1 (tile tuning): key rows per streamed K/V tile in {8, 16, 24, 32, 48} crossed
// with block {32, 64, 128, 256} (block == query rows per block, one row per
// thread). Shared budget math at head_dim = 64, fp32, stride 65, register family:
//   bytes = 2 * tile * 65 * 4   ->   8: 4.2 KiB   16: 8.5 KiB   24: 12.7 KiB
//                                     32: 16.9 KiB 48: 25.3 KiB  64: 33.8 KiB OVER
// so the plan's A1 ladder {64, 128, 256} key rows is NOT reachable on the 32 KiB
// static-shared floor of the HLSL/DX path (src/backends/common/hlsl/hlsl_codegen.cpp:798)
// -- 64 key rows alone need 2*64*65*4 = 33.3 KiB. The sweep therefore tops out at 48
// and documents the arithmetic. Vulkan reports 49152 B through
// compute_max_shared_memory_size(), but make_flash_attention_kernel() has a
// parent-frozen signature with no Device handle, so a variant cannot query it at
// emission time: every table entry is sized for the 32 KiB floor and stays
// portable (lc_optimize §5.7 trap #3). The shared-state family additionally
// carries 2*block*65*4 bytes of private q/o slices, which is why those variants are
// capped at block 32 (tile 32 would already need 33.3 KiB + 128 B).
// Measured (debug build, B=1 heads=16 head_dim=64, per-launch wall time, min of N
// over 24 launches recorded in one CommandList; dx / vk):
//   legacy_b32_t8   19.79 / 19.65    shared_b32_t16  38.4 / 36.3  (A1 alone LOSES:
//   shared_b32_t24  39.3 /  --       the tile effect is dominated by shared traffic)
//   reg_b32_t8       1.35 /  1.83    reg_b64_t16      1.27 / 1.58
//   reg_b32_t16      1.55 /  1.66    reg_b128_t16     1.20 / 1.43
//   reg_b32_t32      6.20 /  --      reg_b256_t16     1.28 / 1.42
//   reg_b32_t48     15.66 / 16.37    reg_b256_t32     1.21 / 1.24
//   reg_b64_t48      7.36 /  7.94    reg_b128_t16     1.17 / 1.38  <- shipped default
// Register pressure is what shapes the ladder: with q/o in registers a large tile
// multiplies the live-register set and the emitted unrolled body (32 rows/block +
// tile 48 is 4x slower than tile 8 at block 32), so "bigger tile" is only a win
// together with "more rows per block".
// A3 (rescale in registers): implemented -- see emitter 2. The accumulator,
// running max and running sum are the same DSL locals across the whole K/V loop
// (lc_optimize §10.5: the storage decision is baked at emission time; nothing is
// re-bound per iteration and no accumulator ever round-trips through shared).
// A2 (warp reductions for the online max/sum): NOT applicable to this kernel and
// deliberately not implemented. Each block owns its query rows end to end, one row
// per thread, so the online max/sum of a row never crosses a thread -- there is no
// shared tree to replace (the only block-wide communication is the K/V tile
// broadcast). Introducing warp_active_max/warp_active_sum would require splitting a
// row across lanes (more shuffles than the 64-deep dot product saves) and would
// break the fallback parity gate: the fallback backend has no warp collective
// lowering at all. Likewise set_warp_size() cannot be pinned here -- the frozen
// factory signature has no Device to ask for compute_warp_size(), and fallback
// rejects every pinned width but 1. The emission uses no lane-count assumption, so
// nothing depends on wave32 vs wave64. (Verified on this device: warp_size 32 on
// dx and vk via compute_warp_size().)
// A4 (mask folding): the register family generates the additive key bias once per
// key while the K tile is filled and stores it in that key row's padding slot
// (k_sh[j*65 + 64] = (1 - mask[key]) * neg_large), so there is no separate mask
// array and no mask pass; padding and out-of-range tail keys get neg_large exactly
// as before, and the arithmetic is the same product the baseline computed per
// thread. The naive trio already folds the bias into kernel #4 (nothing to remove).
// A5 (head batching): one dispatch covers (B x num_heads x query-row-tiles) through
// the 2D grid returned by flash_attention_dispatch -- x = ceil(S/block)*block total
// threads (a multiple of the block size, so flattened_block_id() is exact), y =
// B*num_heads. DX group caps (65535 per dimension, lc_optimize trap #2) leave these
// bounds: ceil(S/block) <= 65535 (S <= 65535*block; at block 256 that is 16.7M)
// and B*num_heads <= 65535 (heads = 16 -> B <= 4095). The plan's stress shape
// S=8192, B=8, heads=16 gives 32 x 128 = 4096 blocks at block 256 (256 x 128 at
// block 32) -- no third dimension needed.

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
// flash_attention_dispatch(B, S, heads) -> {tiles_q * BLOCK, B*heads}
// with tiles_x = ceil(S / 16) for the naive tiles, ceil(S / BLOCK) query-row tiles
// for flash (BLOCK = the resolved flash variant's query rows per block, one row per
// thread; 32 on the M4 baseline, 128 on the tuned default), and tiles_d =
// ceil(head_dim / 16).
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

#include <charconv>
#include <cstdlib>
#include <cstring>

#include <luisa/core/logging.h>
#include <luisa/core/stl.h>
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

// #6v PV vec4: one block = 16 query rows (tr) x 16 vec4 channel groups (tc), so a
// block covers 64 head channels (16 groups * 4 lanes). tiles_d counts these
// 64-channel blocks (head_dim = 64 -> tiles_d = 1).
constexpr uint kAttnVecChannelsPerBlock = kAttnTile * 4u;// 64

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

//--- A1/A3 flash geometry table (O-B) ---------------------------------------
// One entry = one emission of the fused flash kernel:
//   block : threads per block == query rows per block (one row per thread);
//           DSL contract: multiple of 32, <= 1024.
//   tile  : key rows per streamed K/V tile.
//   reg   : storage of the per-thread row state.
//     reg=false -> the M4 baseline exactly as shipped (q row and output
//                  accumulator in private stride-65 SHARED slices, m/s in
//                  registers). Kill switch.
//     reg=true  -> q row and the output accumulator live in per-thread
//                  registers (native, compile-time-indexed unrolls over
//                  kFlashMaxHeadDim slots; lc_optimize §10.5: the storage
//                  decision is fixed at emission time, the accumulator is the
//                  same DSL local across every K/V iteration, nothing is
//                  re-bound per iteration). Shared memory then holds ONLY the
//                  streamed K/V tiles + the folded mask bias.
// Shared budget (the binding constraint is the 32 KiB static-shared floor of
// the HLSL/DX path -- src/backends/common/hlsl/hlsl_codegen.cpp:798; vk reports
// 49152 B through compute_max_shared_memory_size() but the kernel is emitted
// once per Session with no device handle in make_flash_attention_kernel(), so
// every variant is sized for the 32 KiB floor and stays portable):
//   reg=false: (2*block + 2*tile) * 65 * 4 B + tile * 4 B   (q,o,K,V + mask tile)
//   reg=true :  2 * tile * 65 * 4 B                         (K,V; bias in slot 64)
// Head-room check at head_dim = 64 (fp32, stride 65):
//   tile 8  ->  4.2 KiB   tile 16 ->  8.5 KiB   tile 24 -> 12.7 KiB
//   tile 32 -> 16.9 KiB   tile 48 -> 25.3 KiB   tile 64 -> 33.8 KiB OVER
// The plan's A1 ladder {64,128,256} key rows is therefore NOT reachable on the
// DX floor with fp32 head_dim 64 (64 rows alone need 2*64*65*4 = 33.3 KiB), so
// the sweep tops out at 48 and documents the arithmetic instead (128/256 rows
// would need 66.6/133 KiB). Ragged tiles are handled by predicated zero-filling
// loads, so no variant requires S % tile == 0.
struct FlashVariant {
    const char *name;
    uint32_t block;
    uint32_t tile;
    bool reg;
};

constexpr FlashVariant kFlashVariants[] = {
    //--- M4 baseline family (kill switch: LUISA_EMBED_FLASH_CONFIG=legacy_b32_t8)
    {"legacy_b32_t8", 32u, 8u, false},
    {"shared_b32_t16", 32u, 16u, false},// A1 only: bigger tile, shared row state
    {"shared_b32_t24", 32u, 24u, false},
    //--- register family (A1 + A3 together)
    {"reg_b32_t8", 32u, 8u, true},
    {"reg_b32_t16", 32u, 16u, true},
    {"reg_b32_t32", 32u, 32u, true},
    {"reg_b32_t48", 32u, 48u, true},
    {"reg_b64_t16", 64u, 16u, true},
    {"reg_b64_t32", 64u, 32u, true},
    {"reg_b64_t48", 64u, 48u, true},
    {"reg_b128_t16", 128u, 16u, true},
    {"reg_b128_t32", 128u, 32u, true},
    {"reg_b256_t16", 256u, 16u, true},
    {"reg_b256_t32", 256u, 32u, true},
};
constexpr uint32_t kFlashVariantCount =
    static_cast<uint32_t>(sizeof(kFlashVariants) / sizeof(kFlashVariants[0]));

// Builder table: index-parallel with kFlashVariants (asserted below). Each entry
// instantiates one emission of the fused kernel with host-known geometry.
template<uint32_t BLOCK, uint32_t TILE, bool REG>
[[nodiscard]] FlashAttentionKernel build_flash_kernel(const char *vname) noexcept;

using FlashKernelBuilder = FlashAttentionKernel (*)(const char *) noexcept;

constexpr FlashKernelBuilder kFlashBuilders[] = {
    &build_flash_kernel<32u, 8u, false>,
    &build_flash_kernel<32u, 16u, false>,
    &build_flash_kernel<32u, 24u, false>,
    &build_flash_kernel<32u, 8u, true>,
    &build_flash_kernel<32u, 16u, true>,
    &build_flash_kernel<32u, 32u, true>,
    &build_flash_kernel<32u, 48u, true>,
    &build_flash_kernel<64u, 16u, true>,
    &build_flash_kernel<64u, 32u, true>,
    &build_flash_kernel<64u, 48u, true>,
    &build_flash_kernel<128u, 16u, true>,
    &build_flash_kernel<128u, 32u, true>,
    &build_flash_kernel<256u, 16u, true>,
    &build_flash_kernel<256u, 32u, true>,
};
constexpr uint32_t kFlashBuilderCount =
    static_cast<uint32_t>(sizeof(kFlashBuilders) / sizeof(kFlashBuilders[0]));
static_assert(kFlashBuilderCount == kFlashVariantCount,
              "flash builder table must be index-parallel with the variant table");

// DSL kernel names (static literals; the legacy entry keeps the shipped name so
// captures/profiles of the baseline are unchanged).
constexpr const char *kFlashKernelNames[] = {
    "embed_flash_attention",
    "embed_flash_attention_shared_t16",
    "embed_flash_attention_shared_t24",
    "embed_flash_attention_reg_b32_t8",
    "embed_flash_attention_reg_b32_t16",
    "embed_flash_attention_reg_b32_t32",
    "embed_flash_attention_reg_b32_t48",
    "embed_flash_attention_reg_b64_t16",
    "embed_flash_attention_reg_b64_t32",
    "embed_flash_attention_reg_b64_t48",
    "embed_flash_attention_reg_b128_t16",
    "embed_flash_attention_reg_b128_t32",
    "embed_flash_attention_reg_b256_t16",
    "embed_flash_attention_reg_b256_t32",
};
constexpr uint32_t kFlashKernelNameCount =
    static_cast<uint32_t>(sizeof(kFlashKernelNames) / sizeof(kFlashKernelNames[0]));
static_assert(kFlashKernelNameCount == kFlashVariantCount,
              "flash name table must be index-parallel with the variant table");

[[nodiscard]] constexpr uint32_t flash_shared_bytes(const FlashVariant &v) noexcept {
    return v.reg ? 2u * v.tile * kFlashStride * 4u : (2u * v.block + 2u * v.tile) * kFlashStride * 4u + v.tile * 4u;
}

// Shipped default: the measured winner of the A1 x A3 sweep on this device
// (RTX 4080 SUPER, DEBUG build, direct-launch ruler from the O-B scratch harness,
// min-of-N over 12-24 launches recorded in one CommandList, two independent
// processes per number, heads=16 head_dim=64; the full table is
// .kimix_cache/ob_sweep.csv -- the parent re-measures in release):
//   legacy_b32_t8   19.79 / 19.71 ms (dx / vk, S=2048 B=1)   <- M4 baseline
//   reg_b128_t16     1.17 /  1.38 ms   <- shipped default (16.9x / 14.3x): best on dx
//                                        at every shape tested and near-best on vk
//   reg_b256_t32     1.21 /  1.24 ms   (best on vk at S=4096; 0.05-0.2 ms apart from
//   reg_b256_t16     1.24 /  1.22 ms    the default -- inside run-to-run noise)
//   reg_b128_t32     1.25 /  1.45 ms
// per-shape detail (dx/vk, ms per launch): S=4096 B=1 default 3.95/4.71 vs baseline
// 73.6/72.8; S=2048 B=4 default 3.73/4.25.
// Same winner family on both backends, so one default serves both; the per-backend
// override knobs (LUISA_EMBED_FLASH_CONFIG / _BLOCK / _TILE / _REG) stay available
// for the plan's per-backend config table (§5, O-H).
constexpr uint32_t kFlashDefaultVariant = 10u;// reg_b128_t16
static_assert(kFlashDefaultVariant < kFlashVariantCount, "default variant index out of range");
static_assert(flash_shared_bytes(kFlashVariants[kFlashDefaultVariant]) <= 32768u,
              "the shipped default must fit the DX static-shared floor");

static_assert(flash_shared_bytes(kFlashVariants[0u]) <= 32768u,
              "flash variant exceeds the DX static-shared floor");
static_assert(kFlashVariants[0u].block == kFlashBlock &&
                  kFlashVariants[0u].tile == kFlashKeyTile && !kFlashVariants[0u].reg,
              "the legacy entry must describe the shipped M4 geometry");

[[nodiscard]] int32_t flash_variant_by_name(const char *name) noexcept {
    if (name == nullptr) [[unlikely]] {
        return -1;
    }
    for (auto i = 0u; i < kFlashVariantCount; i++) {
        if (luisa::string_view{kFlashVariants[i].name} == luisa::string_view{name}) {
            return static_cast<int32_t>(i);
        }
    }
    return -1;
}

[[nodiscard]] bool flash_parse_u32(const char *text, uint32_t &v) noexcept {
    if (text == nullptr || text[0] == '\0') [[unlikely]] {
        return false;
    }
    const auto *end = text + std::strlen(text);
    uint32_t out = 0u;
    const auto r = std::from_chars(text, end, out);
    if (r.ec != std::errc{} || r.ptr != end) [[unlikely]] {
        return false;
    }
    v = out;
    return true;
}

[[nodiscard]] const FlashVariant &flash_variant(uint32_t i) noexcept {
    return kFlashVariants[i < kFlashVariantCount ? kFlashDefaultVariant : i];
}

// Env resolution (A1/A3 kill switches; lc_optimize §10.9: every optimization is
// individually disable-able at runtime):
//   LUISA_EMBED_FLASH_CONFIG=<variant name>   exact pick
//   LUISA_EMBED_FLASH_BLOCK=<threads>         field-wise, must match one variant
//   LUISA_EMBED_FLASH_TILE=<key rows>         field-wise, must match one variant
//   LUISA_EMBED_FLASH_REG=0|1                 field-wise (shared vs register state)
struct FlashSelection {
    uint32_t index;
    luisa::string source;
};

[[nodiscard]] const FlashSelection &flash_selection() noexcept {
    static const FlashSelection sel = [] noexcept {
        // 1. Whole-variant override.
        if (const auto *name = std::getenv("LUISA_EMBED_FLASH_CONFIG");
            name != nullptr && name[0] != '\0') {
            const auto i = flash_variant_by_name(name);
            if (i >= 0) {
                if (flash_shared_bytes(kFlashVariants[static_cast<uint32_t>(i)]) > 32768u) [[unlikely]] {
                    LUISA_WARNING("embed flash: variant '{}' needs {} B of shared memory, over the "
                                  "32 KiB DX floor; using the baseline variant '{}'.",
                                  kFlashVariants[static_cast<uint32_t>(i)].name,
                                  flash_shared_bytes(kFlashVariants[static_cast<uint32_t>(i)]),
                                  kFlashVariants[kFlashDefaultVariant].name);
                } else {
                    LUISA_INFO("embed flash: variant '{}' selected by LUISA_EMBED_FLASH_CONFIG "
                               "(block {}, tile {}, {} row state; {} B shared).",
                               kFlashVariants[static_cast<uint32_t>(i)].name,
                               kFlashVariants[static_cast<uint32_t>(i)].block,
                               kFlashVariants[static_cast<uint32_t>(i)].tile,
                               kFlashVariants[static_cast<uint32_t>(i)].reg ? "register" : "shared",
                               flash_shared_bytes(kFlashVariants[static_cast<uint32_t>(i)]));
                    return FlashSelection{static_cast<uint32_t>(i), "LUISA_EMBED_FLASH_CONFIG"};
                }
            } else [[unlikely]] {
                LUISA_WARNING("embed flash: LUISA_EMBED_FLASH_CONFIG=\"{}\" is not a known variant; "
                              "falling back to the shipped default.",
                              name);
            }
        }
        // 2. Field-wise override (every specified field must match one variant).
        uint32_t blk = 0u, tile = 0u, reg = 0u;
        bool any = false;
        if (const auto *e = std::getenv("LUISA_EMBED_FLASH_BLOCK"); e != nullptr && e[0] != '\0') {
            if (flash_parse_u32(e, blk)) {
                any = true;
            } else [[unlikely]] {
                LUISA_WARNING("embed flash: ignoring malformed LUISA_EMBED_FLASH_BLOCK=\"{}\".", e);
            }
        }
        if (const auto *e = std::getenv("LUISA_EMBED_FLASH_TILE"); e != nullptr && e[0] != '\0') {
            if (flash_parse_u32(e, tile)) {
                any = true;
            } else [[unlikely]] {
                LUISA_WARNING("embed flash: ignoring malformed LUISA_EMBED_FLASH_TILE=\"{}\".", e);
            }
        }
        if (const auto *e = std::getenv("LUISA_EMBED_FLASH_REG"); e != nullptr && e[0] != '\0') {
            if (flash_parse_u32(e, reg)) {
                any = true;
            } else [[unlikely]] {
                LUISA_WARNING("embed flash: ignoring malformed LUISA_EMBED_FLASH_REG=\"{}\".", e);
            }
        }
        if (any) {
            int32_t found = -1;
            uint32_t matches = 0u;
            for (auto i = 0u; i < kFlashVariantCount; i++) {
                const auto &v = kFlashVariants[i];
                if ((blk != 0u && v.block != blk) || (tile != 0u && v.tile != tile) ||
                    (std::getenv("LUISA_EMBED_FLASH_REG") != nullptr &&
                     static_cast<uint32_t>(v.reg ? 1 : 0) != reg) ||
                    flash_shared_bytes(v) > 32768u) {
                    continue;
                }
                found = static_cast<int32_t>(i);
                matches++;
            }
            if (matches == 1u) {
                LUISA_INFO("embed flash: variant '{}' selected by LUISA_EMBED_FLASH_BLOCK/TILE/REG "
                           "(block {}, tile {}, {} row state; {} B shared).",
                           kFlashVariants[static_cast<uint32_t>(found)].name,
                           kFlashVariants[static_cast<uint32_t>(found)].block,
                           kFlashVariants[static_cast<uint32_t>(found)].tile,
                           kFlashVariants[static_cast<uint32_t>(found)].reg ? "register" : "shared",
                           flash_shared_bytes(kFlashVariants[static_cast<uint32_t>(found)]));
                return FlashSelection{static_cast<uint32_t>(found), "LUISA_EMBED_FLASH_BLOCK/TILE"};
            }
            LUISA_WARNING("embed flash: the LUISA_EMBED_FLASH_BLOCK/TILE/REG override matched {} "
                          "variants; falling back to the shipped default '{}' (use "
                          "LUISA_EMBED_FLASH_CONFIG for an exact pick).",
                          matches, kFlashVariants[kFlashDefaultVariant].name);
        }
        // 3. Shipped default.
        return FlashSelection{kFlashDefaultVariant, "default"};
    }();
    return sel;
}

[[nodiscard]] const FlashVariant &flash_resolved_variant() noexcept {
    return kFlashVariants[flash_selection().index];
}

// Native unroll over the head_dim register file, 4 slots per step (the slot
// count is host-known -- kFlashMaxHeadDim -- and slots outside [0, head_dim) are
// zero-filled at stage time, so a runtime head_dim never needs a dynamically
// indexed local array and the accumulation order matches the M4 baseline).
template<typename Body>
constexpr void flash_for_each_dim(Body &&body) noexcept {
    for (auto g = 0u; g < kFlashMaxHeadDim; g += 4u) {
        body(g);
    }
}

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

// NOTE (test-plan wave review, T2 finding, restated for the O-B tuned family):
// for a fully-masked query row whose S is not a multiple of the selected variant's
// key tile, the flash kernel's online-softmax denominator counts the ragged
// key-tile padding -- ceil(S/TILE)*TILE effective keys, where TILE is 8 on the
// shipped M4 baseline (legacy_b32_t8) and 16 on the tuned default (reg_b128_t16),
// 24 for shared_b32_t24 and 32/48 for the *_t32 / *_t48 variants -- while the naive
// trio divides by exactly S. So flash != naive on those rows and the gap is
// (ceil(S/TILE)*TILE - S)/S; the two agree exactly when S % TILE == 0. Harmless for
// bge-m3 (a real query row never masks all keys; query-pad rows are dropped by every
// consumer), but a documented behavioral difference, not a tolerance. Every variant
// keeps the baseline's padding handling unchanged in kind: out-of-range tail keys
// are zero-filled and get (1 - 0) * neg_large, so only fully-masked rows can pick up
// the extra 1/TILE denominator terms.

//-----------------------------------------------------------------------------
// Launch helpers (see file header for the geometry each one describes)
//-----------------------------------------------------------------------------
uint2 attention_scores_dispatch(uint32_t batch, uint32_t seq_len, uint32_t num_heads) noexcept {
    const auto tiles = (seq_len + kAttnTile - 1u) / kAttnTile;
    return {tiles * kAttentionBlock, batch * num_heads * tiles};
}

// #4v: SAME geometry as attention_scores_dispatch (16x16 logits tiles, block 256).
uint2 attention_scores_vec4_dispatch(uint32_t batch, uint32_t seq_len, uint32_t num_heads) noexcept {
    const auto tiles = (seq_len + kAttnTile - 1u) / kAttnTile;
    return {tiles * kAttentionBlock, batch * num_heads * tiles};
}

uint2 attention_pv_dispatch(uint32_t batch, uint32_t seq_len, uint32_t num_heads,
                            uint32_t head_dim) noexcept {
    const auto tiles_q = (seq_len + kAttnTile - 1u) / kAttnTile;
    const auto tiles_d = (head_dim + kAttnTile - 1u) / kAttnTile;
    return {tiles_d * kAttentionBlock, batch * num_heads * tiles_q};
}

// #6v: float4-channel PV geometry. 256 threads = 16 query rows (tr) x 16 vec4
// channel groups (tc), so one block covers 16 rows x 64 channels: the channel
// tile is 16*4 = 64 wide (head_dim/4 float4 groups), hence tiles_d counts
// 64-channel blocks, not 16. For head_dim = 64 (bge-m3) tiles_d == 1.
uint2 attention_pv_vec4_dispatch(uint32_t batch, uint32_t seq_len, uint32_t num_heads,
                                 uint32_t head_dim) noexcept {
    LUISA_ASSERT(head_dim % 4u == 0u,
                 "attention_pv_vec4 requires head_dim % 4 == 0 (ragged head_dim uses the "
                 "legacy scalar PV kernel)");
    const auto tiles_q = (seq_len + kAttnTile - 1u) / kAttnTile;
    const auto tiles_d = (head_dim + kAttnVecChannelsPerBlock - 1u) / kAttnVecChannelsPerBlock;
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
    // A1/A5: BLOCK (= the resolved variant's query rows per block) replaces the old
    // fixed kFlashBlock. The geometry contract is unchanged: x is the total thread
    // count of the query-row-tile dimension (always a multiple of the block size) and
    // y is B*num_heads head-batch groups; the kernel derives the flattened block index
    // from dispatch_id()/dispatch_size(), so the same emission is correct for a plain
    // 2D launch and for a host-flattened one (DX 65535-group-per-dimension cap,
    // lc_optimize trap #2).
    const auto rows = flash_resolved_variant().block;
    const auto tiles = (seq_len + rows - 1u) / rows;
    return {tiles * rows, batch * num_heads};
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
// #4v attention_scores_vec4 (tensor_optimize.md §4.3 QK vec4 twin of #4)
//-----------------------------------------------------------------------------
AttentionScoresVec4Kernel make_attention_scores_vec4_kernel() noexcept {
    using namespace luisa::compute;

    // Identical geometry, staging and epilogue to make_attention_scores_kernel:
    // one 16x16 logits tile per block (block 256 = kAttentionBlock), Q/K streamed
    // through two 16x17 Shared<float> tiles (stride kAttnStride = 17 dodges the
    // stride-16 bank conflict), the fused scale + additive key-side mask bias and
    // the guarded store unchanged. The ONLY difference is the inner product: the
    // 16-step scalar unroll (acc += tile_q[tr*17+j]*tile_k[tc*17+j]) becomes 4
    // float4 chunks -> acc += dot(make_float4(tq[4j..4j+3]),
    // make_float4(tk[4j..4j+3])) for j in 0..3.
    //
    // Ragged head_dim (robust test runs 1 and 7) needs NO scalar remainder loop:
    // the stage loop still covers head_dim in ceil(head_dim/16) sub-tiles and the
    // predicated staging ZERO-FILLS every channel >= head_dim (and every key/query
    // row >= S) into the tile, so a 16-wide staged row is always well-defined and
    // the dot's out-of-range lanes contribute exactly 0 * 0. Each sub-tile is 16
    // floats == 4 lanes of a float4, so the vec4 chunking is exact (no tail) for
    // any head_dim. The dot() re-associates the 16-sum into 4 chunk dots -- allowed
    // per plan §8.2(3); the 1e-4 tolerance gate still applies and the legacy #4
    // remains the bit-parity oracle.
    AttentionScoresVec4Kernel kernel = [](BufferFloat qkv, BufferFloat mask, BufferFloat logits,
                                          Var<uint> S, Var<uint> num_heads, Var<uint> head_dim,
                                          Var<float> scale, Var<float> neg_large) noexcept {
        set_block_size(kAttentionBlock, 1u, 1u);
        set_name("embed_attention_scores_vec4");

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
        // (batch-head, query-tile, key-tile) triple; no per-block bound check is
        // needed (or possible on dx, where buffer.size() is not usable). Ragged S
        // / head_dim are handled by the per-element predicated staging.
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
                // vec4 inner product: 4 chunks over the staged 16-wide row. The
                // four float4 reads come from compile-time lane offsets (4j..4j+3)
                // against the runtime this-thread bases tr/tc -- the DSL records an
                // independent index expression per unrolled step. Each chunk dot is
                // an independent DOT; their scalar sums fold into `acc`.
                const Var qrow = tr * kAttnStride;
                const Var krow = tc * kAttnStride;
                for (auto j = 0u; j < kAttnTile / 4u; j++) {
                    const Var qo = qrow + j * 4u;
                    const Var ko = krow + j * 4u;
                    const Float4 qv4 = make_float4(tile_q[qo], tile_q[qo + 1u],
                                                   tile_q[qo + 2u], tile_q[qo + 3u]);
                    const Float4 kv4 = make_float4(tile_k[ko], tile_k[ko + 1u],
                                                   tile_k[ko + 2u], tile_k[ko + 3u]);
                    acc += dot(qv4, kv4);
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

namespace {
//-----------------------------------------------------------------------------
// M4 flash attention (single fused kernel, replaces #4-#6) -- O-B tuned family
//-----------------------------------------------------------------------------

// Emitter 1: the shipped M4 geometry, parameterised on (BLOCK, TILE) but with
// byte-identical math and the same shared-resident per-thread row state
// (q_sh / o_sh private stride-65 slices). BLOCK=32, TILE=8 reproduces the
// pre-optimization kernel exactly (the LUISA_EMBED_FLASH_CONFIG=legacy_b32_t8
// kill switch); TILE=16/24 isolate the A1 tile effect without A3.
template<uint32_t BLOCK, uint32_t TILE>
[[nodiscard]] FlashAttentionKernel make_flash_kernel_shared(const char *vname) noexcept {
    using namespace luisa::compute;
    static_assert(BLOCK % 32u == 0u && BLOCK >= 32u && BLOCK <= 1024u);
    static_assert(TILE % 8u == 0u && TILE <= BLOCK,
                  "shared variant keeps the baseline mask-load pattern (tx < TILE)");
    static_assert(TILE * 4u % BLOCK == 0u, "the cooperative tile fill must divide exactly");
    static_assert((2u * BLOCK + 2u * TILE) * kFlashStride * 4u + TILE * 4u <= 32768u,
                  "shared variant exceeds the DX static-shared floor");
    FlashAttentionKernel kernel = [vname](BufferFloat qkv, BufferFloat mask, BufferFloat out,
                                          BufferFloat scratch, Var<uint> B, Var<uint> S,
                                          Var<uint> num_heads, Var<uint> head_dim,
                                          Var<float> scale, Var<float> neg_large) noexcept {
        set_block_size(BLOCK, 1u, 1u);
        set_name(vname);

        static_cast<void>(scratch);

        const Var hd = min(head_dim, def(kFlashMaxHeadDim));
        const Var hd_pad = ((hd + 3u) / 4u) * 4u;

        Shared<float> q_sh{BLOCK * kFlashStride};
        Shared<float> o_sh{BLOCK * kFlashStride};
        Shared<float> k_sh{TILE * kFlashStride};
        Shared<float> v_sh{TILE * kFlashStride};
        Shared<float> max_sh{TILE};

        const Var tx = thread_x();
        const Var tiles_q = (S + BLOCK - 1u) / BLOCK;
        const Var bh_count = B * num_heads;
        const Var lin = flattened_block_id();
        const Var rt = lin % tiles_q;
        const Var bh = lin / tiles_q;
        const Var qi = rt * BLOCK + tx;

        const Var row_stride = 3u * num_heads * head_dim;
        const Var qb = tx * kFlashStride;

        $if (bh < bh_count) {
            const Var b = bh / num_heads;
            const Var h = bh % num_heads;
            const Var h_off = h * hd;

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

            $for (jt, (S + TILE - 1u) / TILE) {
                $for (e, TILE * hd_pad / BLOCK) {
                    const Var idx = e * BLOCK + tx;
                    const Var j = idx / hd_pad;
                    const Var d = idx % hd_pad;
                    const Var tok = jt * TILE + j;
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
                $if (tx < TILE) {
                    const Var tok = jt * TILE + tx;
                    Float mv;
                    $if (tok < S) {
                        mv = mask.read(b * S + tok);
                    }
                    $else {
                        mv = def(0.0f);
                    };
                    max_sh[tx] = mv;
                };
                sync_block();

                for (auto j = 0u; j < TILE; j++) {
                    const Var kb = j * kFlashStride;
                    Var score = def(0.0f);
                    $for (dq, hd_pad / 4u) {
                        const Var d = dq * 4u;
                        score += q_sh[qb + d] * k_sh[kb + d] + q_sh[qb + d + 1u] * k_sh[kb + d + 1u] + q_sh[qb + d + 2u] * k_sh[kb + d + 2u] + q_sh[qb + d + 3u] * k_sh[kb + d + 3u];
                    };
                    score = score * scale + (1.0f - max_sh[j]) * neg_large;

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
                sync_block();
            };

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

// Emitter 2 (A3): the query row and the PV accumulator are per-thread
// REGISTERS, addressed only through emission-time constants (flash_for_each_dim
// unrolls the 64 register slots by 4; lc_optimize §10.5 -- the storage choice is
// baked at emission and the same locals carry the running sum/max/accumulator
// across every K/V iteration, so nothing is re-bound per iteration). Shared
// memory holds only the streamed K/V tiles, and the attention mask is folded
// into each K row's padding slot (A4: one (1 - mask) * neg_large value per key
// produced during the tile fill, no separate mask pass, no mask array).
//
// Slots [head_dim, kFlashMaxHeadDim) are zero-filled by the predicated loads, so
// the fixed-length register dot products stay numerically identical to the
// shared-resident baseline for ragged head_dim (0 * x contributes exactly 0).
// A5: BLOCK query rows x one (batch, head) per block are covered by the 2D grid
// from flash_attention_dispatch (x = tiles_q * BLOCK threads -- always a multiple
// of BLOCK, so the flattened block id is valid for both a plain and a
// host-flattened grid; y = B * num_heads). DX bound: y = B*num_heads <= 65535
// (bge-m3: heads = 16 -> B <= 4095) and x = ceil(S/BLOCK)*BLOCK, whose group
// count ceil(S/BLOCK) <= 65535 (S <= 65535*BLOCK). At the plan's stress shape
// S=8192, B=8, heads=16 the shipped default (BLOCK=128) gives groups_x = 64 and
// groups_y = 128 = 8192 blocks (256 x 128 = 32768 on the 32-row baseline block);
// both dimensions stay far below the 65535 cap, so no third dimension is needed.
template<uint32_t BLOCK, uint32_t TILE>
[[nodiscard]] FlashAttentionKernel make_flash_kernel_reg(const char *vname) noexcept {
    using namespace luisa::compute;
    static_assert(BLOCK % 32u == 0u && BLOCK >= 32u && BLOCK <= 1024u);
    static_assert(TILE * kFlashMaxHeadDim % BLOCK == 0u,
                  "the cooperative K/V fill must divide exactly");
    static_assert(2u * TILE * kFlashStride * 4u <= 32768u,
                  "register variant exceeds the DX static-shared floor");
    FlashAttentionKernel kernel = [vname](BufferFloat qkv, BufferFloat mask, BufferFloat out,
                                          BufferFloat scratch, Var<uint> B, Var<uint> S,
                                          Var<uint> num_heads, Var<uint> head_dim,
                                          Var<float> scale, Var<float> neg_large) noexcept {
        set_block_size(BLOCK, 1u, 1u);
        set_name(vname);

        static_cast<void>(scratch);

        const Var hd = min(head_dim, def(kFlashMaxHeadDim));

        // K rows use stride 65: slot kFlashMaxHeadDim carries that key's folded
        // additive mask bias (A4). V rows are packed at the same stride for
        // symmetry; nothing reads V's padding slot.
        Shared<float> k_sh{TILE * kFlashStride};
        Shared<float> v_sh{TILE * kFlashStride};

        const Var tx = thread_x();
        const Var tiles_q = (S + BLOCK - 1u) / BLOCK;
        const Var bh_count = B * num_heads;
        const Var lin = flattened_block_id();
        const Var rt = lin % tiles_q;
        const Var bh = lin / tiles_q;
        const Var qi = rt * BLOCK + tx;

        const Var row_stride = 3u * num_heads * head_dim;

        $if (bh < bh_count) {
            const Var b = bh / num_heads;
            const Var h = bh % num_heads;
            const Var h_off = h * hd;
            const Var q_base = (b * S + qi) * row_stride + h_off;

            //--- register file: query row + PV accumulator + running max/sum
            Var<float> q_reg[kFlashMaxHeadDim];// DSL locals are zero-initialised
            Var<float> o_acc[kFlashMaxHeadDim];// and stay live for the whole loop
            const Var q_ok = qi < S;
            flash_for_each_dim([&](auto g) noexcept {
                const Var d0 = def(static_cast<uint>(g));
                Float q0;
                $if (q_ok & d0 < hd) {
                    q0 = qkv.read(q_base + d0);
                }
                $else {
                    q0 = def(0.0f);
                };
                Float q1;
                $if (q_ok & (d0 + 1u) < hd) {
                    q1 = qkv.read(q_base + d0 + 1u);
                }
                $else {
                    q1 = def(0.0f);
                };
                Float q2;
                $if (q_ok & (d0 + 2u) < hd) {
                    q2 = qkv.read(q_base + d0 + 2u);
                }
                $else {
                    q2 = def(0.0f);
                };
                Float q3;
                $if (q_ok & (d0 + 3u) < hd) {
                    q3 = qkv.read(q_base + d0 + 3u);
                }
                $else {
                    q3 = def(0.0f);
                };
                q_reg[g] = q0;
                q_reg[g + 1u] = q1;
                q_reg[g + 2u] = q2;
                q_reg[g + 3u] = q3;
            });

            Var m = def(-1.0e30f);
            Var s_norm = def(0.0f);

            $for (jt, (S + TILE - 1u) / TILE) {
                // Cooperative K/V tile fill: TILE * kFlashMaxHeadDim elements
                // over BLOCK threads, division exact (static_assert above), so
                // no per-element bound guard is emitted (lc_optimize §10.3).
                for (auto rep = 0u; rep < TILE * kFlashMaxHeadDim / BLOCK; rep++) {
                    const Var idx = rep * BLOCK + tx;
                    const Var j = idx / kFlashMaxHeadDim;
                    const Var d = idx % kFlashMaxHeadDim;
                    const Var tok = jt * TILE + j;
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
                // A4: the additive key bias is generated once per key while the
                // tile is filled (padding and out-of-range tail keys -> neg_large).
                for (auto rep = 0u; rep < (TILE + BLOCK - 1u) / BLOCK; rep++) {
                    const Var j = rep * BLOCK + tx;
                    if constexpr (TILE % BLOCK != 0u) {
                        $if (j < TILE) {
                            const Var tok = jt * TILE + j;
                            Float mv;
                            $if (tok < S) {
                                mv = mask.read(b * S + tok);
                            }
                            $else {
                                mv = def(0.0f);
                            };
                            k_sh[j * kFlashStride + kFlashMaxHeadDim] = (1.0f - mv) * neg_large;
                        };
                    } else {
                        const Var tok = jt * TILE + j;
                        Float mv;
                        $if (tok < S) {
                            mv = mask.read(b * S + tok);
                        }
                        $else {
                            mv = def(0.0f);
                        };
                        k_sh[j * kFlashStride + kFlashMaxHeadDim] = (1.0f - mv) * neg_large;
                    }
                };
                sync_block();

                // Fixed compile-time unroll over the tile's key slots (native C++
                // loop: the generator knows TILE; there is no XIR unroll pass).
                for (auto j = 0u; j < TILE; j++) {
                    const Var kb = def(j * kFlashStride);
                    Var score = def(0.0f);
                    flash_for_each_dim([&](auto g) noexcept {
                        score += q_reg[g] * k_sh[kb + g] + q_reg[g + 1u] * k_sh[kb + g + 1u] + q_reg[g + 2u] * k_sh[kb + g + 2u] + q_reg[g + 3u] * k_sh[kb + g + 3u];
                    });
                    score = score * scale + k_sh[kb + kFlashMaxHeadDim];

                    const Var m_new = max(m, score);
                    const Var exp_diff = exp(m - m_new);
                    const Var exp_score = exp(score - m_new);
                    s_norm = s_norm * exp_diff + exp_score;
                    flash_for_each_dim([&](auto g) noexcept {
                        o_acc[g] = o_acc[g] * exp_diff + v_sh[kb + g] * exp_score;
                        o_acc[g + 1u] = o_acc[g + 1u] * exp_diff + v_sh[kb + g + 1u] * exp_score;
                        o_acc[g + 2u] = o_acc[g + 2u] * exp_diff + v_sh[kb + g + 2u] * exp_score;
                        o_acc[g + 3u] = o_acc[g + 3u] * exp_diff + v_sh[kb + g + 3u] * exp_score;
                    });
                    m = m_new;
                };
                // Everyone must be done with this tile before it is overwritten.
                sync_block();
            };

            $if (qi < S) {
                const Var o_base = (b * S + qi) * num_heads * head_dim + h_off;
                flash_for_each_dim([&](auto g) noexcept {
                    const Var d0 = def(static_cast<uint>(g));
                    $if (d0 < hd) {
                        out.write(o_base + d0, o_acc[g] / s_norm);
                    };
                    $if ((d0 + 1u) < hd) {
                        out.write(o_base + d0 + 1u, o_acc[g + 1u] / s_norm);
                    };
                    $if ((d0 + 2u) < hd) {
                        out.write(o_base + d0 + 2u, o_acc[g + 2u] / s_norm);
                    };
                    $if ((d0 + 3u) < hd) {
                        out.write(o_base + d0 + 3u, o_acc[g + 3u] / s_norm);
                    };
                });
            };
        };
    };
    return kernel;
}

template<uint32_t BLOCK, uint32_t TILE, bool REG>
[[nodiscard]] FlashAttentionKernel build_flash_kernel(const char *vname) noexcept {
    if constexpr (REG) {
        return make_flash_kernel_reg<BLOCK, TILE>(vname);
    } else {
        return make_flash_kernel_shared<BLOCK, TILE>(vname);
    }
}

}// namespace

//-----------------------------------------------------------------------------
// M4 flash attention (single fused kernel, replaces #4-#6)
//-----------------------------------------------------------------------------
FlashAttentionKernel make_flash_attention_kernel() noexcept {
    const auto &sel = flash_selection();
    const auto &v = kFlashVariants[sel.index];
    LUISA_INFO("embed flash: emitting variant '{}' (block {}, tile {}, {} row state, "
               "{} B shared, source {}); dispatch geometry is query-row-tiles x "
               "B*num_heads.",
               v.name, v.block, v.tile, v.reg ? "register" : "shared",
               flash_shared_bytes(v), sel.source);
    return kFlashBuilders[sel.index](kFlashKernelNames[sel.index]);
}

}// namespace luisa::embed

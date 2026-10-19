// bge-m3 embedding engine -- GEMM with fused epilogues (plan §4 kernel #3) and
// the standalone residual-add kernel (plan §4 kernel #7).
//
// gemm: C[M,N] = epilogue(A[M,K] @ B[K,N]). A is row-major [M,K], B is row-major
// [K,N] (the packer pre-transposes every weight so reads coalesce along N --
// plan §3.3), C is row-major [M,N]. It replaces the ONNX `MatMul` x192 plus the
// bias `Add` and the elementwise `Erf` / `Relu` chains that follow it in the
// bge-m3 encoder (QKV projection, attention output projection, FFN x2 per layer,
// and the retrieval heads -- plan §0/§4 table row #3).
//
// Epilogues are compile-time: the caller passes a GemmEpilogue and the factory
// returns the matching pre-built Kernel2D. The selection is a host-side switch
// over template instantiations; the unused epilogue math is erased by
// `if constexpr` while the DSL kernel is being recorded.
//
// Launch geometry -- TUNABLE (optimization plan §3.1 G1..G4, G8). One
// `GemmTileConfig` describes the block's work shape:
//   tile_m x tile_n : output tile per thread block
//   tile_k          : K slab streamed through shared memory (register family)
//   reg_m x reg_n   : output sub-tile held in registers by one thread
//   block           : threads per block = (tile_m/reg_m) * (tile_n/reg_n)
//   naive           : true = legacy one-output-per-thread 2D-thread-grid kernel
//                     (reg_m = reg_n = 1, tile_k = 16) -- the kill-switch shape
//   swap_grid       : grid.x carries M-tiles instead of N-tiles (L2 probe, G2)
// `make_gemm_kernel()` builds the kernel for the runtime-resolved config and
// `gemm_dispatch_size()` returns the matching geometry, so the two can never
// disagree (they read the same resolved config). The executor and the parity
// tests only ever call these two entry points, exactly as before.
//
// Shared-memory budget: the register family stages A[tile_m][tile_k] and
// B[tile_k][tile_n] as float arrays with one-float row padding (+1 keeps the
// tile rows in different banks for the warp read pattern below), i.e.
// (tile_m * (tile_k + 1) + tile_k * (tile_n + 1)) * 4 B per block. Every variant
// in the table stays <= 21.1 KiB (rb_256x64_b256_r88), well under the 32 KiB the
// HLSL (DX) path asserts (lc_optimize §4.7.6 / §5.1); the shipped default
// rb_64x64_b128_r84 (64x64 tile, 16-wide K slab) is 8.3 KiB -- the plan's G3
// example of 2 * (64*16*4 B) = 8 KiB plus padding, which leaves >=3 blocks per
// CU resident on the DX 32 KiB budget.
//
// VECTOR register family (`rbv_*`, tensor_optimize.md §4): a comparator variant
// where the thread's N sub-tile is RN = 4 ADJACENT columns held as ONE float4
// accumulator and both shared tiles are float4 arrays (16 B LDS/STS, a +1-slot
// row pad). The A operand is broadcast from a scalar with make_float4() (no
// as<float4> reinterpretation anywhere); the C store stays SCALAR because an
// [M,N] fp32 row is only 4 B-aligned per column, so a vector store has no 16 B
// alignment guarantee on dx/vk. reg_m = RM rows, reg_n is fixed at 4. The same
// block = (tile_m/reg_m)*(tile_n/reg_n) derivation and the same ragged-edge
// guard-elision discipline as the scalar register family apply (see the .cpp).
//
// Grid bound: dispatch x = (tiles along N) * block TOTAL threads, dispatch y =
// tiles along M (or the swapped pair when swap_grid is set). The engine
// envelope is 1 job <= 16383 padded rows (examples/embedding/embed/
// scheduler.cpp kMaxJobRows) and N <= 4096 (ffn.up intermediate), and the
// smallest tile side in the table is 8, so
//   tiles_m <= ceil(16383 / 8)  = 2048   (<= 65535, DX group cap, lc_optimize trap #2)
//   tiles_n <= ceil(4096 / 8)   = 512    (ditto)
// for every variant -- no 2D flattening is needed inside gemm itself.
//
// Ragged M/N/K are handled by predicated (zero-filling) tile loads plus a
// guarded store, with a block-uniform fast path that skips the predicates for
// fully in-range tiles (plan §3.1 G8 guard elision), so arbitrary sizes are
// correct, not just multiples of the tile.
//
// Device-free Kernel2D factory per CONVENTIONS.md: the caller (Phase B executor
// / tests) compiles the result with device.compile(...) and caches it as a
// Shader. All dimensions are runtime kernel args, so one compilation serves
// every shape.
//
// Kill switch / tuning: the config is resolved once per process from the
// environment (see kernels_gemm.cpp):
//   LUISA_EMBED_GEMM_CONFIG=<variant name>   exact variant, wins over the rest
//   LUISA_EMBED_GEMM_TILE=<M>x<N>            output tile per block
//   LUISA_EMBED_GEMM_BLOCK=<threads>         threads per block
//   LUISA_EMBED_GEMM_KSLAB=<k>               K slab width
//   LUISA_EMBED_GEMM_REGTILE=<m>x<n>         per-thread register sub-tile
// An unset (or unmatched) override selects the shipped default, and an invalid
// one logs a LUISA_WARNING and falls back to it -- the engine never fails to
// start because of a tuning typo. Setting TILE=16x16 BLOCK=256 (or
// CONFIG=naive_16x16) reproduces the pre-optimization kernel bit for bit.
//
// WEIGHT FORMAT (optimization plan §3.1 G7 / §4 Q1-Q2, O-G). Each config
// carries a GemmWeightFormat describing how the B operand (the packed [K][N]
// weight matrix) is read; the variant table, the env resolution and the fp32
// default behavior are UNCHANGED by this dimension -- make_gemm_kernel(ep) /
// make_gemm_kernel_config(ep, cfg) keep selecting the fp32 format (cfg.format,
// which is Fp32 for every entry), and only the new (ep, wf) / (ep, cfg, wf)
// overloads build half-precision load paths. Packed-B buffer contract when
// wf != Fp32:
//   - B is a Buffer<float> holding ceil(K*N/2) words; each 32-bit word packs
//     two consecutive little-endian 16-bit elements of the logical [K][N]
//     matrix in emission order e = k*N + n: element 2j in bits [15:0],
//     element 2j+1 in bits [31:16]. This is exactly the on-disk v2 blob
//     layout (package.h), so the upload is a raw byte copy -- no host packing.
//   - bf16 -> fp32 conversion is the exact bit extension (<<16 of the low
//     half, &0xFFFF0000 of the high half); fp16 -> fp32 is the exact integer
//     upcast (all half values, subnormals included, are representable in
//     fp32). A (activations), C and aux (bias/residual) stay plain fp32.
//   - FMA accumulation stays fp32 in every variant: the conversion is
//     storage-only (G7), NO half-precision arithmetic anywhere.
//   - The conversion happens at the B-tile fill (once per staged element);
//     register blocking (G4) amortizes it exactly like the fp32 loads.
#pragma once
#include <luisa/core/stl/string.h>
#include <luisa/dsl/syntax.h>

namespace luisa::embed {

// Legacy naive tile: 16x16 output tile, one output per thread, 256 threads.
// Still in the table as the kill-switch variant (see the header comment).
inline constexpr uint kGemmTile = 16u;
inline constexpr uint kGemmBlock = kGemmTile * kGemmTile;
static_assert(kGemmBlock == 256u, "naive gemm block must be 256 threads");
// Thread block width of residual_add (plan §4 #7: 1D, block 256).
inline constexpr uint kResidualAddBlock = 256u;

// Compile-time epilogue flags (plan §4 #3: NONE / BIAS / BIAS_GELU(erf) /
// BIAS_RELU / RESIDUAL_ADD).
enum class GemmEpilogue : uint32_t {
    None,       ///< C = A @ B
    Bias,       ///< C = A @ B + bias[N]
    BiasGeluErf,///< C = gelu_erf(A @ B + bias[N]) -- the ONNX `Erf` GELU variant
    BiasRelu,   ///< C = relu(A @ B + bias[N])
    ResidualAdd,///< C = A @ B + R[M,N] (the aux slot carries the residual row)
};

// B-operand storage format (plan §3.1 G7, §4 Q1/Q2). The numeric values match
// embed::PackagePrecision (package.h) so the executor can map one to the
// other without a lookup table.
enum class GemmWeightFormat : uint32_t {
    Fp32 = 0u,///< B is the logical [K][N] fp32 matrix (the original path)
    Bf16 = 1u,///< B packs two bf16 per fp32 word (see the header contract)
    Fp16 = 2u,///< B packs two fp16 per fp32 word
};

// "fp32" / "bf16" / "fp16".
[[nodiscard]] const char *gemm_weight_format_name(GemmWeightFormat wf) noexcept;

// Log/debug name of an epilogue flag (also used as the compiled kernel name).
[[nodiscard]] const char *gemm_epilogue_name(GemmEpilogue ep) noexcept;

// C[M,N] = epilogue(A[M,K] @ B[K,N]).
// Buffers: A [M*K], B [K*N] (row-major, coalesced along N), C [M*N],
//          aux = bias[N] for Bias/BiasGeluErf/BiasRelu, residual R[M*N] for
//          ResidualAdd, unused for None (bind a dummy 1-element buffer view).
// Scalars: M, N, K (runtime; arbitrary ragged values allowed).
using GemmKernel = compute::Kernel2D<void(compute::Buffer<float> /*A*/,
                                          compute::Buffer<float> /*B*/,
                                          compute::Buffer<float> /*C*/,
                                          compute::Buffer<float> /*aux*/,
                                          uint32_t /*M*/,
                                          uint32_t /*N*/,
                                          uint32_t /*K*/)>;

// One entry of the tuning table (see the header comment for the fields).
// `block` is derived, not chosen freely: block == (tile_m/reg_m)*(tile_n/reg_n).
// `format` is the precision dimension (plan §3.1 G7): every shipped entry is
// Fp32 (unchanged table/count), but any entry can be BUILT in any format via
// the (ep, cfg, wf) overload -- the builder templates the B-tile load on the
// format, so no per-format table entries exist and the sweep harness (which
// feeds fp32 weights) keeps seeing exactly the fp32 variants.
struct GemmTileConfig {
    const char *name;                                ///< variant key, e.g. "rb_64x64_b256_r4x4_k16"
    uint32_t tile_m;                                 ///< output rows per block
    uint32_t tile_n;                                 ///< output columns per block
    uint32_t tile_k;                                 ///< K slab (0 for the naive family: it is fixed at 16)
    uint32_t reg_m;                                  ///< output rows per thread (1 == naive)
    uint32_t reg_n;                                  ///< output columns per thread (1 == naive)
    uint32_t block;                                  ///< threads per block
    bool naive;                                      ///< legacy one-output-per-thread kernel
    bool swap_grid;                                  ///< grid.x carries M-tiles (L2 locality probe)
    GemmWeightFormat format = GemmWeightFormat::Fp32;///< B-operand element format
};

// Compiled-in variant table (host side; the sweep harness iterates it).
[[nodiscard]] uint32_t gemm_tile_variant_count() noexcept;
[[nodiscard]] const GemmTileConfig *gemm_tile_variant(uint32_t i) noexcept;
// -1 when the name is unknown.
[[nodiscard]] int32_t gemm_tile_index_by_name(const char *name) noexcept;

// The config actually in use: shipped default, or the variant picked by the
// LUISA_EMBED_GEMM_* environment overrides. Resolved once, lazily.
[[nodiscard]] const GemmTileConfig &gemm_resolved_config() noexcept;
// Human-readable one-liner of the resolved config ("name (source=default)").
[[nodiscard]] luisa::string gemm_resolved_config_source() noexcept;

// Kernel for an explicit config (used by the sweep harness; `make_gemm_kernel`
// is just this with the resolved config). The 2-arg forms build the config's
// own format -- for every shipped entry that is Fp32, i.e. the pre-G7
// behavior, bit for bit. The 3-arg form overrides the B-operand format (plan
// §3.1 G7) and is what the executor uses for a half-precision package.
[[nodiscard]] GemmKernel make_gemm_kernel_config(GemmEpilogue ep,
                                                 const GemmTileConfig &cfg) noexcept;
[[nodiscard]] GemmKernel make_gemm_kernel_config(GemmEpilogue ep,
                                                 const GemmTileConfig &cfg,
                                                 GemmWeightFormat wf) noexcept;

// Compile-time epilogue dispatch on the resolved config: returns the Kernel2D
// for the requested flag (host-side switch over template instantiations; see
// kernels_gemm.cpp). The (ep) form keeps the historical fp32 load path;
// (ep, wf) builds the resolved geometry with a packed-B load path.
[[nodiscard]] GemmKernel make_gemm_kernel(GemmEpilogue ep) noexcept;
[[nodiscard]] GemmKernel make_gemm_kernel(GemmEpilogue ep, GemmWeightFormat wf) noexcept;

// Launch geometry of `cfg` for a [M,N] gemm output: the x dimension carries the
// N-tiles as `block`-thread blocks (TOTAL threads, lc_optimize trap #1), the y
// dimension carries the M-tiles as 1-thread-tall blocks. Both grid dimensions
// stay far below the DX 65535-group cap for every bge-m3 shape (see the header
// comment for the bound arithmetic; lc_optimize trap #2).
[[nodiscard]] uint2 gemm_dispatch_size_config(const GemmTileConfig &cfg,
                                              uint32_t m, uint32_t n) noexcept;
// Same, for the runtime-resolved config (the executor's entry point).
[[nodiscard]] uint2 gemm_dispatch_size(uint32_t m, uint32_t n) noexcept;

// residual_add (plan §4 #7): out[i] = a[i] + b[i] over `count` elements -- used
// for the residual `Add` edges the GEMM epilogue does not already fuse
// (ONNX `Add` x341 subset, plan §4 table row #7).
// Launch: block kResidualAddBlock, dispatch(count) = TOTAL THREADS
// (lc_optimize trap #1); the tail thread is guarded. The kernel implements the
// flattened 2D-grid idiom (see kernels_embed.cpp), so the host MAY dodge the
// DX 65535-group cap with dispatch(x = bs*cols, y = blocks/cols) when
// count/kResidualAddBlock would exceed it.
using ResidualAddKernel = compute::Kernel1D<void(compute::Buffer<float> /*a*/,
                                                 compute::Buffer<float> /*b*/,
                                                 compute::Buffer<float> /*out*/,
                                                 uint32_t /*count*/)>;

[[nodiscard]] ResidualAddKernel make_residual_add_kernel() noexcept;

}// namespace luisa::embed

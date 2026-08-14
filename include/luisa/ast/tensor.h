#pragma once
/*
Storage rules:
  R1  C++ host / compile-time value    -> store the C++ variable itself,
                                          e.g. `int32_t rank = 2;`
                                          (a non-type template parameter /
                                          host compile-time constant; no AST
                                          node is allocated for it).

  R2  Host-side constant, fixed at the -> store `const LiteralExpr *`
      compiling stage                  (include/luisa/ast/expression.h):
                                          LiteralExpr(const Type *type,
                                                      detail::LiteralValue v)
                                          - Value = detail::LiteralValue, a
                                            variant over basic_types (bool,
                                            float, int, uint, short, ushort,
                                            slong, ulong, half, double and
                                            their vector/matrix forms)
                                          - Tag::LITERAL, accessor value()
                                          - string literals are NOT in the
                                            variant: use `const StringIDExpr *`
                                            (Tag::STRING_ID, accessor data()).

  R3  Runtime variable truly in kernel  -> store `const RefExpr *`
                                          (include/luisa/ast/expression.h):
                                          RefExpr(Variable v)
                                          - holds a Variable { type, uid,
                                            tag } (include/luisa/ast/
                                            variable.h); Variable::Tag:
                                            LOCAL, SHARED, REFERENCE, BUFFER,
                                            TEXTURE, BINDLESS_ARRAY, ACCEL,
                                            THREAD_ID, BLOCK_ID, DISPATCH_ID,
                                            DISPATCH_SIZE, KERNEL_ID,
                                            WARP_LANE_COUNT, WARP_LANE_ID, ...
                                          - Tag::REF, accessor variable().

Shorthand used below:
  i32     = int32_t                       (R1: host-side value)
  ten     = TensorExpr *                  (tensor operand, section 4)
  lit     = const LiteralExpr *           (R2: host-side constant fixed at
                                            the compiling stage)
  ref     = const RefExpr *               (R3: runtime kernel variable)
  sid     = const StringIDExpr *          (R2 string: static const string)

Reference implementation: include/luisa/dsl/tensor.h (tracing stub).
Namespace: luisa::compute::tile (language = luisa::compute::tile::language).
Dot-syntax handle: constexpr auto T = luisa::compute::tile::language::dsl;

=============================================================================
0. Shared storage legend
=============================================================================
kind  stored as                  used when
----  -------------------------  -------------------------------------------
i32   int32_t                    member is a host-side value fixed on the
                                 host during the compiling stage (rank,
                                 extents, scope, grid, stages...)
ten   TensorExpr *               member is a tensor operand (section 4)
lit const LiteralExpr * member is a constant value embedded in the
                                 kernel IR (0.0f, 1e-12f, block 0)
sid   const StringIDExpr *       member is a constant string in the kernel
                                 (print message / format tag)
ref   const RefExpr *            member is a real kernel variable (tensor,
                                 block id, loop index, tile base)

=============================================================================
1. Scalar / dtype nodes (host-side type tags, R1)
=============================================================================
Node    Member  Kind  C++ member                        Note
------  ------  ----  --------------------------------  ----------------------
half    -       -     (type tag only)                   f16; scalar value
float32 -       -     (type tag only)                   f32; scalar value
int32   -       -     (type tag only)                   i32; scalar value
  -> in the DSL a dtype is used as a template argument (`Tensor<f16, 2>`);
     in the AST the dtype is stored as an ordinary R1 member of TensorExpr
     (e.g. TensorElementType), NOT as a template parameter (F1); its
     *values* (`f32(0.0f)`, `f32(N)`) are R2 `lit`.

=============================================================================
2. Memory scope (R1)
=============================================================================
Node         Member  Kind  C++ member      Note
-----------  ------  ----  --------------  ------------------------------
Scope        -       -     (enum)          Global=0, Shared=1, Fragment=2
scope_name   s       i32   0|1|2           kept as host-side value

=============================================================================
3. Shape / slice nodes
=============================================================================
Node                  Member     Kind  C++ member                Note
--------------------  ---------  ----  -------------------------  ---------------
Shape<R>              dims       i32   std::array<int32_t, R>     R1: extents are
                                                                    host constexpr
shape(Ints...)        (builder)  -     -> Shape<sizeof...(Ints)>  R1
Slice                 begin      i32   int32_t                    R1
                      end        i32   int32_t                    exclusive; -1
                                                                  = to end
                      is_all     i32   int32_t                    R1 (0|1)
all()                 (builder)  -     Slice{0, -1, 1}            R1
range(b,e)            (builder)  -     Slice{b, e, 0}             R1
*/
#include <luisa/core/basic_types.h>
#include <luisa/core/stl/string.h>
#include <luisa/core/stl/vector.h>
#include <luisa/ast/expression.h>
#include <luisa/ast/type.h>
#include <luisa/ast/variable.h>

#include <cstdint>
#include <utility>

namespace luisa::compute {

// =============================================================================
// Tensor AST nodes — concrete implementation of the design above.
//
// Every tensor operation (Gemm, Clear, Copy, ...) is a Statement-like node
// (`TensorStmt`), never an Expression (F2).  `TensorExpr` is the independent,
// NON-TEMPLATE tensor operand that carries the layout (F1).
//
// Ownership:
//   - a TensorStmt OWNS its `output` and `inputs` TensorExpr operands;
//   - LiteralExpr members (R2) are BORROWED and never deleted by the
//     statement; all Expression nodes and TensorStmt nodes are managed by
//     include/luisa/ast/tile_function_builder.h;
//   - RefExpr members (`handle`, runtime scalars, kernel binders) are borrowed
//     (created by a FunctionBuilder) and are never deleted by the statement.
//
// serialization contract:
//   `size_t serialize(luisa::vector<char>&)` appends a compact binary encoding
//   of the STATICALLY MEANINGFUL members and returns the number of bytes
//   appended.  `bool deserialize(char const*&, char const*)` reads the same
//   encoding back (advancing the cursor) and returns false on any failure.
//   Pointers are non-serializable (e.g. `RefExpr*`), so they are skipped;
//   the *values* carried by LiteralExpr (host-side constants fixed at the
//   compiling stage) ARE serialized because they do not change between runs.
// =============================================================================

/// Memory scope of a tensor (R1, stored as a host-side value).
enum struct TensorScope : uint32_t {
    Global = 0, // kernel argument / result buffer
    Shared = 1, // per-block on-chip memory
    Fragment = 2// per-thread registers
};

/// Element dtype tag of a tensor (host-side, R1).  TensorExpr stores the
/// element dtype directly as this tag; only scalar element types are
/// supported.  The first three values (F16 / F32 / I32) are stable and must
/// not be renumbered (they are serialized as raw u32 tags); I8 / FP8 / I4 /
/// FP4 are the quantized element types (TileLang `int8` / `fp8` / `int4` /
/// `fp4`).  `FP8` maps to the fp8 e4m3 encoding (the common default; e5m2
/// can be added on a later R1 tag) and `FP4` to the 4-bit e2m1 encoding.
enum struct TensorElementType : uint32_t {
    F16 = 0,// half
    F32 = 1,// float
    I32 = 2,// int
    I8 = 3, // int8 (signed 8-bit)
    FP8 = 4,// fp8 (e4m3)
    I4 = 5, // int4 (signed 4-bit, sub-byte)
    FP4 = 6 // fp4 (e2m1, sub-byte)
};
/// Maps 1:1 to the TensorStmt sub-classes implemented in this header.
enum struct TileOpKind : uint32_t {
    ALLOC,     // T.empty / T.alloc_shared / T.alloc_fragment
    CLEAR,     // T.clear
    COPY,      // T.copy
    GEMM,      // T.gemm
    REDUCE_SUM,// T.reduce_sum
    PRINT,     // T.print
    STORE,     // tile-store: lhs = rhs / lhs *= rhs
    BINARY,    // whole-tile elementwise binary op (T.Parallel lowering)
    MAX,       // T.max(a, b)
    RSQRT,     // T.rsqrt(a)
    CEILDIV,   // T.ceildiv(a, b)
    KERNEL_1D, // T.Kernel(gx, threads)
    KERNEL_2D, // T.Kernel(gx, gy, threads)
    PIPELINED, // T.Pipelined(count, stages)

    // --- gap-analysis additions: TileLang builtins ------------------------
    // [tile-level data movement]
    FILL,             // T.fill(buf, value)
    TRANSPOSE,        // T.transpose(src, dst)
    IM2COL,           // T.im2col(img, col, ...)
    ASYNC_COPY,       // T.async_copy(src, dst, ...)
    COPY_CLUSTER,     // T.copy_cluster(src, dst, ...)
    TMA_COPY,         // T.tma_copy(src, dst, ...)
    TMA_GATHER4,      // T.tma_gather4(src, dst, ...)
    TMA_SCATTER4,     // T.tma_scatter4(src, dst, ...)
    RESHAPE,          // T.reshape(src, shape)
    VIEW,             // T.view(src, shape, dtype)
    // [reduce family]
    REDUCE,           // T.reduce(buf, out, op, dim, clear)
    FINALIZE_REDUCER, // T.finalize_reducer(reducer, batch)
    WARP_REDUCE,      // T.warp_reduce_sum/max/min/bitand/bitor
    // [scan family]
    CUMSUM,           // T.cumsum(src, dst, dim, reverse)
    CUMMAX,           // T.cummax(src, dst, dim, reverse)
    // [gemm variants / knobs]
    WGGMA_GEMM,             // T.wgmma_gemm
    TCGEN05_GEMM,           // T.tcgen05_gemm
    TCGEN05_GEMM_BLOCKSCALED,// T.tcgen05_gemm_blockscaled
    GEMM_SP,                // T.gemm_sp
    WGGMA_GEMM_SP,          // T.wgmma_gemm_sp
    TCGEN05_GEMM_SP,        // T.tcgen05_gemm_sp
    // [atomic / scalar helpers]
    ATOMIC,           // T.atomic_add/max/min/addx2/addx4/load/or/store
    CLAMP,            // T.clamp(dst, lo, hi)
    DP4A,             // T.dp4a(A, B, C)
    LOOP_BREAK,       // T.loop_break()
    ANY_OF,           // T.any_of(buf)
    ALL_OF,           // T.all_of(buf)
    // [sync / warp / barrier builtins]
    SYNC,             // T.sync_threads / sync_warp / sync_grid / sync_global
    BARRIER,          // T.barrier_arrive / barrier_wait / named_barrier_arrive
    MBARRIER,         // T.mbarrier_arrive / expect_tx / arrive_expect_tx / wait_parity
    WARP_VOTE,        // T.activemask / ballot / ballot_sync / any_sync / all_sync / match_*_sync
    SHUFFLE,          // T.shfl_sync / shfl_xor / shfl_up / shfl_down / shuffle_elect
    SYNC_THREADS_VOTE,// T.syncthreads_count / syncthreads_and / syncthreads_or
    // [math intrinsics]
    FAST_RCP,         // T.fast_rcp
    IEEE_MATH,        // T.ieee_add/sub/mul/fmaf/frcp/fsqrt/frsqrt/fdiv
    PACKED_MATH,      // T.add2/sub2/mul2/fma2/max2/min2/abs2
    FAST_MATH,        // T.__exp / __exp10 / __log / __log2 / __log10 / __sin / __cos / __tan
    // [allocation / loop / annotation machinery]
    ALLOC_SPECIAL,    // T.alloc_var/local/global/barrier/reducer/tmem/descriptor/cluster_barrier
    LOOP_ANNOTATION,  // T.Parallel / Persistent / serial / unroll / vectorized
    ANNOTATE,         // T.use_swizzle / annotate_layout / annotate_safe_value /
                      //   annotate_restrict_buffers / annotate_l2_hit_ratio / annotate_min_blocks_per_sm
    DYNAMIC,          // T.dynamic(name, dtype)
    SYMBOLIC,         // T.symbolic(name, dtype) (deprecated alias of dynamic)
    INLINE,           // T.inline(func) host-side marker
    META_CLASS,       // T.meta_class(cls) host-side marker
    ACCESS_PTR,       // T.access_ptr(base, access_type, ...)
    INDEX_TO_COORDINATES// T.index_to_coordinates(index, shape)
};

// =============================================================================
// R1 discriminators for the gap-analysis builtins (host-side values, serialized
// as u32/i32 tags; see the class comments for the TileLang `T.*` mapping).
// =============================================================================

/// Warp partition policy of T.gemm / T.wgmma_gemm / T.tcgen05_gemm (TileLang
/// GemmWarpPolicy: Square=1x1, FullRow=all warps along M, FullCol=along N).
enum struct GemmWarpPolicy : uint32_t {
    Square = 0,
    FullRow = 1,
    FullCol = 2
};

/// T.reduce reduce_type discriminator (TileLang ReduceKind).
enum struct TileReduceOp : uint32_t {
    SUM = 0, MAX = 1, MIN = 2, ABS_SUM = 3, ABS_MAX = 4,
    BIT_AND = 5, BIT_OR = 6, BIT_XOR = 7
};

/// T.warp_reduce_* discriminator.
enum struct TileWarpReduceOp : uint32_t {
    SUM = 0, MAX = 1, MIN = 2, BIT_AND = 3, BIT_OR = 4
};

/// T.atomic_* discriminator.
enum struct TileAtomicOp : uint32_t {
    ADD = 0, MAX = 1, MIN = 2, ADDX2 = 3, ADDX4 = 4, LOAD = 5, OR = 6, STORE = 7
};

/// Atomic memory-order ids, mirroring TileLang `_MEMORY_ORDER_ID_MAP`.
enum struct TileMemoryOrder : uint32_t {
    RELAXED = 0, CONSUME = 1, ACQUIRE = 2, RELEASE = 3, ACQ_REL = 4, SEQ_CST = 5
};

/// T.sync_threads / sync_warp / sync_grid / sync_global discriminator.
enum struct TileSyncOp : uint32_t {
    THREADS = 0, WARP = 1, GRID = 2, GLOBAL = 3
};

/// T.barrier_arrive / barrier_wait / named_barrier_arrive discriminator.
enum struct TileBarrierOp : uint32_t {
    ARRIVE = 0, WAIT = 1, NAMED_ARRIVE = 2
};

/// T.mbarrier_arrive / arrive_expect_tx / expect_tx / wait_parity discriminator.
enum struct TileMBarrierOp : uint32_t {
    ARRIVE = 0, ARRIVE_EXPECT_TX = 1, EXPECT_TX = 2, WAIT_PARITY = 3
};

/// T.activemask / ballot / ballot_sync / any_sync / all_sync / match_*_sync.
enum struct TileWarpVoteOp : uint32_t {
    ACTIVEMASK = 0, BALLOT = 1, BALLOT_SYNC = 2, ANY_SYNC = 3,
    ALL_SYNC = 4, MATCH_ANY_SYNC = 5, MATCH_ALL_SYNC = 6
};

/// T.shfl_sync / shfl_xor / shfl_up / shfl_down / shuffle_elect.
enum struct TileShuffleOp : uint32_t {
    SYNC = 0, XOR = 1, UP = 2, DOWN = 3, ELECT = 4
};

/// T.syncthreads_count / syncthreads_and / syncthreads_or.
enum struct TileSyncThreadsVoteOp : uint32_t {
    COUNT = 0, AND = 1, OR = 2
};

/// T.ieee_* discriminator; rounding modes are R1 ids 0=rn, 1=rz, 2=ru, 3=rd.
enum struct TileIeeeOp : uint32_t {
    ADD = 0, SUB = 1, MUL = 2, FMAF = 3, FRCP = 4, FSQRT = 5, FRSQRT = 6, FDIV = 7
};

/// packed x2 math discriminator (T.add2 / sub2 / mul2 / fma2 / max2 / min2 / abs2).
enum struct TilePackedOp : uint32_t {
    ADD2 = 0, SUB2 = 1, MUL2 = 2, FMA2 = 3, MAX2 = 4, MIN2 = 5, ABS2 = 6
};

/// fast-math intrinsic discriminator (T.__exp / __exp10 / __log / __log2 /
/// __log10 / __sin / __cos / __tan).
enum struct TileFastMathOp : uint32_t {
    EXP = 0, EXP10 = 1, LOG = 2, LOG2 = 3, LOG10 = 4, SIN = 5, COS = 6, TAN = 7
};

/// T.alloc_var / alloc_local / alloc_global / alloc_barrier / alloc_reducer /
/// alloc_tmem / alloc_descriptor / alloc_cluster_barrier discriminator.
enum struct TileAllocKind : uint32_t {
    VAR = 0, LOCAL = 1, GLOBAL = 2, BARRIER = 3, REDUCER = 4,
    TMEM = 5, DESCRIPTOR = 6, CLUSTER_BARRIER = 7
};

/// loop-annotation discriminator (T.Parallel / Persistent / serial / unroll /
/// vectorized).
enum struct TileLoopAnnotKind : uint32_t {
    PARALLEL = 0, PERSISTENT = 1, SERIAL = 2, UNROLL = 3, VECTORIZED = 4
};

/// buffer/kernel annotation discriminator (T.use_swizzle / annotate_layout /
/// annotate_safe_value / annotate_restrict_buffers / annotate_l2_hit_ratio /
/// annotate_min_blocks_per_sm).
enum struct TileAnnotKind : uint32_t {
    USE_SWIZZLE = 0, LAYOUT = 1, SAFE_VALUE = 2, RESTRICT_BUFFERS = 3,
    L2_HIT_RATIO = 4, MIN_BLOCKS_PER_SM = 5
};

// =============================================================================
// TileLang builtin coverage — every builtin below is implemented as a
// TensorStmt sub-class / TileOpKind entry (the comment of each class names the
// TileLang `T.*` builtin it maps to).  Reference: D:/tilelang/tilelang/language/
// (common.py, copy_op.py, fill_op.py, reduce_op.py, scan_op.py, gemm_op.py,
// experimental/gemm_sp_op.py, atomic.py, customize.py, builtin.py,
// math_intrinsics.py, fastmath.py, logical.py, annotations.py, allocate.py,
// symbolics.py, utils.py).
//
// [tile-level data movement]         FillStmt, TransposeStmt, Im2ColStmt,
//                                    AsyncCopyStmt, CopyClusterStmt, TmaCopyStmt,
//                                    TmaGather4Stmt, TmaScatter4Stmt,
//                                    ReshapeStmt, ViewStmt
// [reduce family]                    ReduceStmt (TileReduceOp), FinalizeReducerStmt,
//                                    WarpReduceStmt (TileWarpReduceOp)
// [scan family]                      CumSumStmt, CumMaxStmt
// [gemm variants / knobs]            GemmStmt knobs (GemmWarpPolicy / clear_accum /
//                                    k_pack / mbar), WgmmaGemmStmt, Tcgen05GemmStmt,
//                                    Tcgen05GemmBlockscaledStmt, GemmSpStmt,
//                                    WgmmaGemmSpStmt, Tcgen05GemmSpStmt
// [atomic / scalar helpers]          AtomicStmt (TileAtomicOp), ClampStmt,
//                                    Dp4aStmt, LoopBreakStmt, AnyOfStmt, AllOfStmt
// [sync / warp / barrier builtins]   SyncStmt, BarrierStmt, MBarrierStmt,
//                                    WarpVoteStmt, ShuffleStmt, SyncThreadsVoteStmt
// [math intrinsics]                  FastRcpStmt, IeeeMathStmt, PackedMathStmt,
//                                    FastMathStmt
// [allocation / loop / annotation]   AllocSpecialStmt, LoopAnnotationStmt,
//                                    AnnotateStmt, DynamicStmt, SymbolicStmt,
//                                    InlineStmt, MetaClassStmt, AccessPtrStmt,
//                                    IndexToCoordinatesStmt
//
// Remaining gap — [dtype]: TensorElementType now models F16/F32/I32 plus the
// quantized I8/FP8/I4/FP4 tags (R1).  TileLang additionally carries f64 / bf16 /
// u8 / i16 / u16 / u32 / i64 / u64 / bool / fp8-e5m2 / fp6 — none of which are
// expressible in TensorExpr yet (extending TensorElementType is an R1 enum
// change, not a TensorStmt).  The regular-kernel lowering can only lower the
// tags that have a core element Type (F16/F32/I32/I8/FP8); I4/FP4 are sub-byte
// dtypes with no core element Type today, so the lowering rejects them.
// =============================================================================

[[nodiscard]] LUISA_AST_API const char *scope_name(TensorScope scope) noexcept;
[[nodiscard]] LUISA_AST_API const char *tensor_element_type_name(TensorElementType e) noexcept;

// ---------------------------------------------------------------------------
// 4. TensorExpr — the independent, non-template tensor node (F1)
// ---------------------------------------------------------------------------
class LUISA_AST_API TensorExpr {

private:
    int32_t _rank{}; // R1: host-side rank
    TensorElementType _dtype{TensorElementType::F32};// R1: element type (scalar only)
    TensorScope _scope{TensorScope::Global}; // R1: Global / Shared / Fragment
    // R1: layout extents; dims are non-negative 32-bit host values
    luisa::fixed_vector<int32_t, 4> _dims;   // R1: M, N, K ... (host)
    luisa::fixed_vector<int32_t, 4> _offset; // R1: tile anchor (host); runtime anchor is R3 `ref`
    luisa::fixed_vector<int32_t, 4> _extent; // R1: tile size BM, BN (host)
    const RefExpr *_handle{nullptr};// R3: kernel-side variable (BUFFER/SHARED/LOCAL)
    // R1: host-side display/identity name (e.g. "A" / "T.alloc_shared#3").
    // Used by the tile lowering (tile_to_kernel) to resolve a view clone back
    // to its AllocStmt storage when several tensors share one layout.  The
    // name is host metadata: it is copied by the (compiler-generated) copy
    // constructor but intentionally NOT serialized (see serialize()).
    luisa::string _name;

public:
    TensorExpr() noexcept = default;

    /// Construct a tensor with the given layout.  When `extent` is empty it
    /// defaults to the whole-tensor extent (`dims`); when `offset` is empty it
    /// defaults to zeros.  `handle` is borrowed and not owned; `name` is
    /// host-side identity metadata used by the tile lowering.
    TensorExpr(int32_t rank,
               TensorElementType dtype,
               TensorScope scope,
               luisa::fixed_vector<int32_t, 4> &&dims,
               luisa::fixed_vector<int32_t, 4> &&offset = {},
               luisa::fixed_vector<int32_t, 4> &&extent = {},
               const RefExpr *handle = nullptr,
               luisa::string_view name = {}) noexcept;

    [[nodiscard]] auto rank() const noexcept { return _rank; }
    [[nodiscard]] auto dtype() const noexcept { return _dtype; }
    [[nodiscard]] auto scope() const noexcept { return _scope; }
    [[nodiscard]] auto dims() const noexcept { return luisa::span<const int32_t>{_dims.data(), _dims.size()}; }
    [[nodiscard]] auto offset() const noexcept { return luisa::span<const int32_t>{_offset.data(), _offset.size()}; }
    [[nodiscard]] auto extent() const noexcept { return luisa::span<const int32_t>{_extent.data(), _extent.size()}; }
    [[nodiscard]] auto handle() const noexcept { return _handle; }
    /// Host-side identity/display name (not serialized).
    [[nodiscard]] auto name() const noexcept { return luisa::string_view{_name}; }

    /// Human readable description, e.g. "A(16,16)@(0,0)".
    [[nodiscard]] luisa::string describe() const;

    /// Append the statically meaningful layout (rank, dtype, scope, dims,
    /// offset, extent) to `output_buffer`; `handle` is a pointer and is not
    /// serialized.  Returns the number of bytes appended.
    [[nodiscard]] size_t serialize(luisa::vector<char> &output_buffer);

    /// Read the layout back, advancing `input_ptr`.  The handle is left null
    /// (pointers are non-serializable).  Returns false on malformed input.
    bool deserialize(char const *&input_ptr, char const *end_ptr);
};

// ---------------------------------------------------------------------------
// 5. Tensor op nodes — STATEMENTS only, no Expression (F2)
// ---------------------------------------------------------------------------
class LUISA_AST_API TensorStmt {

private:
    TileOpKind _op{TileOpKind::ALLOC};                            // meta: op discriminator
    TensorExpr *_output{nullptr};                                 // result tensor (owned; may be null)
    luisa::vector<TensorExpr *> _inputs;                          // input argument tensors (owned)
    luisa::vector<std::pair<luisa::string, int64_t>> _annotations;// host-side meta

protected:
    void _clear_owned() noexcept;

public:
    explicit TensorStmt(TileOpKind op) noexcept;
    TensorStmt(TileOpKind op,
               TensorExpr *output,
               luisa::vector<TensorExpr *> inputs) noexcept;
    virtual ~TensorStmt();

    [[nodiscard]] auto op() const noexcept { return _op; }
    /// Result tensor; null when the operation has no return value.
    [[nodiscard]] auto output() const noexcept { return _output; }
    /// Input argument tensors (READ operands; the output may also be read+write).
    [[nodiscard]] auto inputs() const noexcept {
        return luisa::span<const TensorExpr *const>{_inputs.data(), _inputs.size()};
    }
    /// Host-side annotations (coalesced_width, stages, ...).
    [[nodiscard]] auto annotations() const noexcept {
        return luisa::span<const std::pair<luisa::string, int64_t>>{_annotations.data(), _annotations.size()};
    }
    void set_annotation(luisa::string key, int64_t value) noexcept;
    [[nodiscard]] const int64_t *annotation(luisa::string_view key) const noexcept;

    /// Append op + output + inputs + annotations.  Returns bytes appended.
    [[nodiscard]] virtual size_t serialize(luisa::vector<char> &output_buffer);
    /// Read the common members back.  Returns false on malformed input or a
    /// TileOpKind mismatch.
    virtual bool deserialize(char const *&input_ptr, char const *end_ptr);
};

// --- Gemm: T.gemm(a, b, c) --------------------------------------------------
// c (accum, READ+WRITE) is the result tensor; a, b are the READ operands.
// In addition to the trans flags, T.gemm carries the TileLang knobs
// policy (GemmWarpPolicy), clear_accum, k_pack and the Blackwell mbarrier
// (mbar, an optional third input tensor).
class LUISA_AST_API GemmStmt final : public TensorStmt {
    int32_t _trans_a{};// R1: 0|1
    int32_t _trans_b{};// R1: 0|1
    GemmWarpPolicy _policy{GemmWarpPolicy::Square};// R1: warp partition
    int32_t _clear_accum{};                        // R1: 0|1
    int32_t _k_pack{1};                            // R1: packed matrix cores

public:
    GemmStmt() noexcept : TensorStmt{TileOpKind::GEMM} {}
    GemmStmt(TensorExpr *a, TensorExpr *b, TensorExpr *c,
             int32_t trans_a = 0, int32_t trans_b = 0) noexcept
        : TensorStmt{TileOpKind::GEMM, c, {a, b}}, _trans_a{trans_a}, _trans_b{trans_b} {}
    /// Knobs form: policy / clear_accum / k_pack / mbar (mbar is an optional
    /// Blackwell mbarrier tensor stored as a third input, READ+WRITE).
    GemmStmt(TensorExpr *a, TensorExpr *b, TensorExpr *c,
             GemmWarpPolicy policy, int32_t clear_accum, int32_t k_pack = 1,
             TensorExpr *mbar = nullptr) noexcept
        : TensorStmt{TileOpKind::GEMM, c,
                     mbar != nullptr ? luisa::vector<TensorExpr *>{a, b, mbar}
                                     : luisa::vector<TensorExpr *>{a, b}},
          _policy{policy}, _clear_accum{clear_accum}, _k_pack{k_pack} {}
    [[nodiscard]] auto a() const noexcept { return inputs().size() > 0 ? inputs()[0] : nullptr; }
    [[nodiscard]] auto b() const noexcept { return inputs().size() > 1 ? inputs()[1] : nullptr; }
    [[nodiscard]] auto c() const noexcept { return output(); }
    [[nodiscard]] auto mbar() const noexcept { return inputs().size() > 2 ? inputs()[2] : nullptr; }
    [[nodiscard]] auto trans_a() const noexcept { return _trans_a; }
    [[nodiscard]] auto trans_b() const noexcept { return _trans_b; }
    [[nodiscard]] auto policy() const noexcept { return _policy; }
    [[nodiscard]] auto clear_accum() const noexcept { return _clear_accum; }
    [[nodiscard]] auto k_pack() const noexcept { return _k_pack; }
    [[nodiscard]] size_t serialize(luisa::vector<char> &output_buffer) override;
    bool deserialize(char const *&input_ptr, char const *end_ptr) override;
};

// --- Clear: T.clear(t) ------------------------------------------------------
class LUISA_AST_API ClearStmt final : public TensorStmt {
public:
    ClearStmt() noexcept : TensorStmt{TileOpKind::CLEAR} {}
    explicit ClearStmt(TensorExpr *t) noexcept
        : TensorStmt{TileOpKind::CLEAR, t, {}} {}
    [[nodiscard]] auto t() const noexcept { return output(); }
    [[nodiscard]] size_t serialize(luisa::vector<char> &output_buffer) override;
    bool deserialize(char const *&input_ptr, char const *end_ptr) override;
};

// --- Copy: T.copy(src, dst) -------------------------------------------------
class LUISA_AST_API CopyStmt final : public TensorStmt {
public:
    CopyStmt() noexcept : TensorStmt{TileOpKind::COPY} {}
    CopyStmt(TensorExpr *src, TensorExpr *dst) noexcept
        : TensorStmt{TileOpKind::COPY, dst, {src}} {}
    [[nodiscard]] auto src() const noexcept { return inputs().size() > 0 ? inputs()[0] : nullptr; }
    [[nodiscard]] auto dst() const noexcept { return output(); }
    [[nodiscard]] size_t serialize(luisa::vector<char> &output_buffer) override;
    bool deserialize(char const *&input_ptr, char const *end_ptr) override;
};
// --- ReduceSum: T.reduce_sum(x, y, dim) -------------------------------------
class LUISA_AST_API ReduceSumStmt final : public TensorStmt {
    uint32_t _dim{};// R1: reduction dimension

public:
    ReduceSumStmt() noexcept : TensorStmt{TileOpKind::REDUCE_SUM} {}
    ReduceSumStmt(TensorExpr *x, TensorExpr *y, uint32_t dim) noexcept
        : TensorStmt{TileOpKind::REDUCE_SUM, y, {x}}, _dim{dim} {}
    [[nodiscard]] auto x() const noexcept { return inputs().size() > 0 ? inputs()[0] : nullptr; }
    [[nodiscard]] auto y() const noexcept { return output(); }
    [[nodiscard]] auto dim() const noexcept { return _dim; }
    [[nodiscard]] size_t serialize(luisa::vector<char> &output_buffer) override;
    bool deserialize(char const *&input_ptr, char const *end_ptr) override;
};

// --- TilePrint: T.print(t, "msg") --------------------------------------------
class LUISA_AST_API TilePrintStmt final : public TensorStmt {
    luisa::string _msg;// R1: print message (host-side string)

public:
    TilePrintStmt() noexcept : TensorStmt{TileOpKind::PRINT} {}
    TilePrintStmt(TensorExpr *t, luisa::string msg) noexcept
        : TensorStmt{TileOpKind::PRINT, nullptr, {t}}, _msg{std::move(msg)} {}
    [[nodiscard]] auto t() const noexcept { return inputs().size() > 0 ? inputs()[0] : nullptr; }
    [[nodiscard]] auto msg() const noexcept { return luisa::string_view{_msg}; }
    [[nodiscard]] size_t serialize(luisa::vector<char> &output_buffer) override;
    bool deserialize(char const *&input_ptr, char const *end_ptr) override;
};

// --- Alloc: T.empty / T.alloc_shared / T.alloc_fragment ---------------------
// The statement owns the created TensorExpr (the output); its layout
// (dims / dtype / scope) is the alloc information.  The R3 kernel-side
// variable is kept in the TensorExpr::handle.
class LUISA_AST_API AllocStmt final : public TensorStmt {
public:
    AllocStmt() noexcept : TensorStmt{TileOpKind::ALLOC} {}
    AllocStmt(luisa::fixed_vector<int32_t, 4> dims, TensorElementType dtype, TensorScope scope,
              const RefExpr *handle = nullptr, luisa::string_view name = {}) noexcept;
    [[nodiscard]] auto tensor() const noexcept { return output(); }
    [[nodiscard]] auto rank() const noexcept { return output() == nullptr ? 0 : output()->rank(); }
    [[nodiscard]] auto dims() const noexcept {
        return output() == nullptr ? luisa::span<const int32_t>{} : output()->dims();
    }
    [[nodiscard]] auto dtype() const noexcept {
        return output() == nullptr ? TensorElementType::F32 : output()->dtype();
    }
    [[nodiscard]] auto scope() const noexcept {
        return output() == nullptr ? TensorScope::Global : output()->scope();
    }
    [[nodiscard]] auto handle() const noexcept { return output() == nullptr ? nullptr : output()->handle(); }
    [[nodiscard]] size_t serialize(luisa::vector<char> &output_buffer) override;
    bool deserialize(char const *&input_ptr, char const *end_ptr) override;
};

// --- TileStore: C_local[BM, BN] = expr ; A_local[...] *= ... ----------------
// rhs is a tensor (stored in base inputs), or a scalar literal (R2), or a
// runtime scalar RefExpr (R3, not serializable).
class LUISA_AST_API TileStoreStmt final : public TensorStmt {
    int32_t _op{};                           // R1: 0 = `=`, 1 = `*=` row-broadcast
    const LiteralExpr *_rhs_literal{nullptr};// R2 (borrowed, managed by TileFunctionBuilder)
    const RefExpr *_rhs_ref{nullptr};        // R3 (borrowed, non-serializable)

public:
    TileStoreStmt() noexcept : TensorStmt{TileOpKind::STORE} {}
    TileStoreStmt(int32_t op, TensorExpr *lhs, TensorExpr *rhs_tensor = nullptr,
                  const LiteralExpr *rhs_literal = nullptr,
                  const RefExpr *rhs_ref = nullptr) noexcept;
    [[nodiscard]] auto op() const noexcept { return _op; }
    [[nodiscard]] auto lhs() const noexcept { return output(); }
    [[nodiscard]] auto rhs_tensor() const noexcept {
        return inputs().size() > 0 ? inputs()[0] : nullptr;
    }
    [[nodiscard]] auto rhs_literal() const noexcept { return _rhs_literal; }
    [[nodiscard]] auto rhs_ref() const noexcept { return _rhs_ref; }
    [[nodiscard]] size_t serialize(luisa::vector<char> &output_buffer) override;
    bool deserialize(char const *&input_ptr, char const *end_ptr) override;
};

// --- TileBinary: whole-tile elementwise A+B, A*B, A/2.0f ---------------------
// lhs/rhs are READ operands; the result is a temporary tile (no output).
class LUISA_AST_API TileBinaryStmt final : public TensorStmt {
    BinaryOp _op{BinaryOp::ADD};             // R1: BinaryOp as int32_t
    const LiteralExpr *_rhs_literal{nullptr};// R2 (borrowed, managed by TileFunctionBuilder)
    const RefExpr *_rhs_ref{nullptr};        // R3 (borrowed, non-serializable)

public:
    TileBinaryStmt() noexcept : TensorStmt{TileOpKind::BINARY} {}
    TileBinaryStmt(BinaryOp op, TensorExpr *lhs, TensorExpr *rhs_tensor = nullptr,
                   const LiteralExpr *rhs_literal = nullptr,
                   const RefExpr *rhs_ref = nullptr) noexcept;
    [[nodiscard]] auto op() const noexcept { return _op; }
    [[nodiscard]] auto lhs() const noexcept { return inputs().size() > 0 ? inputs()[0] : nullptr; }
    [[nodiscard]] auto rhs_tensor() const noexcept {
        return inputs().size() > 1 ? inputs()[1] : nullptr;
    }
    [[nodiscard]] auto rhs_literal() const noexcept { return _rhs_literal; }
    [[nodiscard]] auto rhs_ref() const noexcept { return _rhs_ref; }
    [[nodiscard]] size_t serialize(luisa::vector<char> &output_buffer) override;
    bool deserialize(char const *&input_ptr, char const *end_ptr) override;
};

// --- Max: T.max(a, b) -------------------------------------------------------
class LUISA_AST_API MaxStmt final : public TensorStmt {
    const LiteralExpr *_b{nullptr};// R2 (borrowed, managed by TileFunctionBuilder): e.g. LiteralExpr(Type::of<float>(), 1e-12f)

public:
    MaxStmt() noexcept : TensorStmt{TileOpKind::MAX} {}
    MaxStmt(TensorExpr *a, const LiteralExpr *b) noexcept
        : TensorStmt{TileOpKind::MAX, nullptr, {a}}, _b{b} {}
    [[nodiscard]] auto a() const noexcept { return inputs().size() > 0 ? inputs()[0] : nullptr; }
    [[nodiscard]] auto b() const noexcept { return _b; }
    [[nodiscard]] size_t serialize(luisa::vector<char> &output_buffer) override;
    bool deserialize(char const *&input_ptr, char const *end_ptr) override;
};

// --- Rsqrt: T.rsqrt(a) ------------------------------------------------------
class LUISA_AST_API RsqrtStmt final : public TensorStmt {
public:
    RsqrtStmt() noexcept : TensorStmt{TileOpKind::RSQRT} {}
    explicit RsqrtStmt(TensorExpr *a) noexcept
        : TensorStmt{TileOpKind::RSQRT, nullptr, {a}} {}
    [[nodiscard]] auto a() const noexcept { return inputs().size() > 0 ? inputs()[0] : nullptr; }
    [[nodiscard]] size_t serialize(luisa::vector<char> &output_buffer) override;
    bool deserialize(char const *&input_ptr, char const *end_ptr) override;
};

// --- CeilDiv: T.ceildiv(a, b) (host-side helper, no tensors) ----------------
class LUISA_AST_API CeilDivStmt final : public TensorStmt {
    int32_t _a{};// R1
    int32_t _b{};// R1; result (a + b - 1) / b

public:
    CeilDivStmt() noexcept : TensorStmt{TileOpKind::CEILDIV} {}
    CeilDivStmt(int32_t a, int32_t b) noexcept
        : TensorStmt{TileOpKind::CEILDIV}, _a{a}, _b{b} {}
    [[nodiscard]] auto a() const noexcept { return _a; }
    [[nodiscard]] auto b() const noexcept { return _b; }
    [[nodiscard]] auto result() const noexcept { return (_a + _b - 1) / _b; }
    [[nodiscard]] size_t serialize(luisa::vector<char> &output_buffer) override;
    bool deserialize(char const *&input_ptr, char const *end_ptr) override;
};

// --- Fill: T.fill(buf, value) ------------------------------------------------
// Fill a buffer / region with a scalar: an R2 constant or an R3 runtime scalar.
class LUISA_AST_API FillStmt final : public TensorStmt {
    const LiteralExpr *_value_literal{nullptr};// R2 (borrowed, managed by TileFunctionBuilder)
    const RefExpr *_value_ref{nullptr};        // R3 (borrowed, non-serializable)

public:
    FillStmt() noexcept : TensorStmt{TileOpKind::FILL} {}
    FillStmt(TensorExpr *buf, const LiteralExpr *value) noexcept
        : TensorStmt{TileOpKind::FILL, buf, {}}, _value_literal{value} {}
    FillStmt(TensorExpr *buf, const RefExpr *value) noexcept
        : TensorStmt{TileOpKind::FILL, buf, {}}, _value_ref{value} {}
    [[nodiscard]] auto buf() const noexcept { return output(); }
    [[nodiscard]] auto value_literal() const noexcept { return _value_literal; }
    [[nodiscard]] auto value_ref() const noexcept { return _value_ref; }
    [[nodiscard]] size_t serialize(luisa::vector<char> &output_buffer) override;
    bool deserialize(char const *&input_ptr, char const *end_ptr) override;
};

// --- Transpose: T.transpose(src, dst) ---------------------------------------
// Shared-memory 2D transpose: dst[j, i] = src[i, j].  dst is the result tensor.
class LUISA_AST_API TransposeStmt final : public TensorStmt {
public:
    TransposeStmt() noexcept : TensorStmt{TileOpKind::TRANSPOSE} {}
    TransposeStmt(TensorExpr *src, TensorExpr *dst) noexcept
        : TensorStmt{TileOpKind::TRANSPOSE, dst, {src}} {}
    [[nodiscard]] auto src() const noexcept { return inputs().size() > 0 ? inputs()[0] : nullptr; }
    [[nodiscard]] auto dst() const noexcept { return output(); }
    [[nodiscard]] size_t serialize(luisa::vector<char> &output_buffer) override;
    bool deserialize(char const *&input_ptr, char const *end_ptr) override;
};

// --- Im2Col: T.im2col(img, col, ...) ----------------------------------------
// Convolution im2col; nhw_step / c_step are R3 runtime step variables.
class LUISA_AST_API Im2ColStmt final : public TensorStmt {
    const RefExpr *_nhw_step{nullptr};// R3 (borrowed, non-serializable)
    const RefExpr *_c_step{nullptr};  // R3 (borrowed, non-serializable)
    int32_t _kernel{};                // R1
    int32_t _stride{1};               // R1
    int32_t _dilation{1};             // R1
    int32_t _pad{};                   // R1
    int32_t _eviction_policy{};       // R1: 0=evict_normal, 1=evict_first, 2=evict_last

public:
    Im2ColStmt() noexcept : TensorStmt{TileOpKind::IM2COL} {}
    Im2ColStmt(TensorExpr *img, TensorExpr *col,
               const RefExpr *nhw_step = nullptr, const RefExpr *c_step = nullptr,
               int32_t kernel = 0, int32_t stride = 1, int32_t dilation = 1,
               int32_t pad = 0, int32_t eviction_policy = 0) noexcept
        : TensorStmt{TileOpKind::IM2COL, col, {img}},
          _nhw_step{nhw_step}, _c_step{c_step}, _kernel{kernel}, _stride{stride},
          _dilation{dilation}, _pad{pad}, _eviction_policy{eviction_policy} {}
    [[nodiscard]] auto img() const noexcept { return inputs().size() > 0 ? inputs()[0] : nullptr; }
    [[nodiscard]] auto col() const noexcept { return output(); }
    [[nodiscard]] auto nhw_step() const noexcept { return _nhw_step; }
    [[nodiscard]] auto c_step() const noexcept { return _c_step; }
    [[nodiscard]] auto kernel() const noexcept { return _kernel; }
    [[nodiscard]] auto stride() const noexcept { return _stride; }
    [[nodiscard]] auto dilation() const noexcept { return _dilation; }
    [[nodiscard]] auto pad() const noexcept { return _pad; }
    [[nodiscard]] auto eviction_policy() const noexcept { return _eviction_policy; }
    [[nodiscard]] size_t serialize(luisa::vector<char> &output_buffer) override;
    bool deserialize(char const *&input_ptr, char const *end_ptr) override;
};

// --- AsyncCopy: T.async_copy(src, dst, ...) ---------------------------------
// cp.async copy (no implicit wait; synchronization is explicit).
class LUISA_AST_API AsyncCopyStmt final : public TensorStmt {
    int32_t _coalesced_width{};// R1: 0 = unset

public:
    AsyncCopyStmt() noexcept : TensorStmt{TileOpKind::ASYNC_COPY} {}
    AsyncCopyStmt(TensorExpr *src, TensorExpr *dst, int32_t coalesced_width = 0) noexcept
        : TensorStmt{TileOpKind::ASYNC_COPY, dst, {src}}, _coalesced_width{coalesced_width} {}
    [[nodiscard]] auto src() const noexcept { return inputs().size() > 0 ? inputs()[0] : nullptr; }
    [[nodiscard]] auto dst() const noexcept { return output(); }
    [[nodiscard]] auto coalesced_width() const noexcept { return _coalesced_width; }
    [[nodiscard]] size_t serialize(luisa::vector<char> &output_buffer) override;
    bool deserialize(char const *&input_ptr, char const *end_ptr) override;
};

// --- CopyCluster: T.copy_cluster(src, dst, ...) -----------------------------
// TMA multicast (cluster_mask) / SM-to-SM copy (dst_block).
class LUISA_AST_API CopyClusterStmt final : public TensorStmt {
    int32_t _dst_block{-1};     // R1: destination CTA rank (-1 = unset)
    int32_t _cluster_mask{0};   // R1: TMA multicast mask (0 = unset)
    int32_t _coalesced_width{0};// R1: SIMT fallback vectorization (0 = unset)

public:
    CopyClusterStmt() noexcept : TensorStmt{TileOpKind::COPY_CLUSTER} {}
    CopyClusterStmt(TensorExpr *src, TensorExpr *dst, int32_t dst_block = -1,
                    int32_t cluster_mask = 0, int32_t coalesced_width = 0) noexcept
        : TensorStmt{TileOpKind::COPY_CLUSTER, dst, {src}},
          _dst_block{dst_block}, _cluster_mask{cluster_mask}, _coalesced_width{coalesced_width} {}
    [[nodiscard]] auto src() const noexcept { return inputs().size() > 0 ? inputs()[0] : nullptr; }
    [[nodiscard]] auto dst() const noexcept { return output(); }
    [[nodiscard]] auto dst_block() const noexcept { return _dst_block; }
    [[nodiscard]] auto cluster_mask() const noexcept { return _cluster_mask; }
    [[nodiscard]] auto coalesced_width() const noexcept { return _coalesced_width; }
    [[nodiscard]] size_t serialize(luisa::vector<char> &output_buffer) override;
    bool deserialize(char const *&input_ptr, char const *end_ptr) override;
};

// --- TmaCopy: T.tma_copy(src, dst, ...) -------------------------------------
// User-managed TMA copy; the optional mbarrier tensor (inputs[1]) is required
// for global->shared loads (expect_tx + tma_load, no wait).
class LUISA_AST_API TmaCopyStmt final : public TensorStmt {
    int32_t _leader_scope_threads{0};// R1: 0 = default
    int32_t _eviction_policy{0};     // R1: 0=evict_normal, 1=evict_first, 2=evict_last

public:
    TmaCopyStmt() noexcept : TensorStmt{TileOpKind::TMA_COPY} {}
    TmaCopyStmt(TensorExpr *src, TensorExpr *dst, TensorExpr *barrier = nullptr,
                int32_t leader_scope_threads = 0, int32_t eviction_policy = 0) noexcept
        : TensorStmt{TileOpKind::TMA_COPY, dst,
                     barrier != nullptr ? luisa::vector<TensorExpr *>{src, barrier}
                                        : luisa::vector<TensorExpr *>{src}},
          _leader_scope_threads{leader_scope_threads}, _eviction_policy{eviction_policy} {}
    [[nodiscard]] auto src() const noexcept { return inputs().size() > 0 ? inputs()[0] : nullptr; }
    [[nodiscard]] auto dst() const noexcept { return output(); }
    [[nodiscard]] auto barrier() const noexcept { return inputs().size() > 1 ? inputs()[1] : nullptr; }
    [[nodiscard]] auto leader_scope_threads() const noexcept { return _leader_scope_threads; }
    [[nodiscard]] auto eviction_policy() const noexcept { return _eviction_policy; }
    [[nodiscard]] size_t serialize(luisa::vector<char> &output_buffer) override;
    bool deserialize(char const *&input_ptr, char const *end_ptr) override;
};

// --- TmaGather4 / TmaScatter4: Blackwell tile::gather4 / scatter4 -----------
// 2D global <-> shared (4 x K_box) with exactly 4 runtime row indices.
class LUISA_AST_API TmaGather4Stmt final : public TensorStmt {
    const RefExpr *_col{nullptr};                 // R3 (borrowed, non-serializable)
    luisa::fixed_vector<int32_t, 4> _rows;        // R1: exactly 4 row indices
    int32_t _eviction_policy{};                   // R1

public:
    TmaGather4Stmt() noexcept : TensorStmt{TileOpKind::TMA_GATHER4} {}
    TmaGather4Stmt(TensorExpr *src, TensorExpr *dst, const RefExpr *col,
                   luisa::fixed_vector<int32_t, 4> rows, TensorExpr *barrier,
                   int32_t eviction_policy = 0) noexcept
        : TensorStmt{TileOpKind::TMA_GATHER4, dst, {src, barrier}},
          _col{col}, _rows{std::move(rows)}, _eviction_policy{eviction_policy} {}
    [[nodiscard]] auto src() const noexcept { return inputs().size() > 0 ? inputs()[0] : nullptr; }
    [[nodiscard]] auto dst() const noexcept { return output(); }
    [[nodiscard]] auto col() const noexcept { return _col; }
    [[nodiscard]] auto rows() const noexcept { return luisa::span<const int32_t>{_rows.data(), _rows.size()}; }
    [[nodiscard]] auto barrier() const noexcept { return inputs().size() > 1 ? inputs()[1] : nullptr; }
    [[nodiscard]] auto eviction_policy() const noexcept { return _eviction_policy; }
    [[nodiscard]] size_t serialize(luisa::vector<char> &output_buffer) override;
    bool deserialize(char const *&input_ptr, char const *end_ptr) override;
};

class LUISA_AST_API TmaScatter4Stmt final : public TensorStmt {
    const RefExpr *_col{nullptr};                 // R3 (borrowed, non-serializable)
    luisa::fixed_vector<int32_t, 4> _rows;        // R1: exactly 4 row indices
    int32_t _eviction_policy{};                   // R1

public:
    TmaScatter4Stmt() noexcept : TensorStmt{TileOpKind::TMA_SCATTER4} {}
    TmaScatter4Stmt(TensorExpr *src, TensorExpr *dst, const RefExpr *col,
                    luisa::fixed_vector<int32_t, 4> rows,
                    int32_t eviction_policy = 0) noexcept
        : TensorStmt{TileOpKind::TMA_SCATTER4, dst, {src}},
          _col{col}, _rows{std::move(rows)}, _eviction_policy{eviction_policy} {}
    [[nodiscard]] auto src() const noexcept { return inputs().size() > 0 ? inputs()[0] : nullptr; }
    [[nodiscard]] auto dst() const noexcept { return output(); }
    [[nodiscard]] auto col() const noexcept { return _col; }
    [[nodiscard]] auto rows() const noexcept { return luisa::span<const int32_t>{_rows.data(), _rows.size()}; }
    [[nodiscard]] auto eviction_policy() const noexcept { return _eviction_policy; }
    [[nodiscard]] size_t serialize(luisa::vector<char> &output_buffer) override;
    bool deserialize(char const *&input_ptr, char const *end_ptr) override;
};

// --- Reshape / View: T.reshape(src, shape) / T.view(src, shape, dtype) -----
// The statement builds and owns the derived view tensor (same handle as src).
class LUISA_AST_API ReshapeStmt final : public TensorStmt {
public:
    ReshapeStmt() noexcept : TensorStmt{TileOpKind::RESHAPE} {}
    ReshapeStmt(TensorExpr *src, luisa::fixed_vector<int32_t, 4> dims) noexcept
        : TensorStmt{TileOpKind::RESHAPE,
                     new TensorExpr{static_cast<int32_t>(dims.size()), src->dtype(), src->scope(),
                                    std::move(dims), {}, {}, src->handle()},
                     {src}} {}
    [[nodiscard]] auto src() const noexcept { return inputs().size() > 0 ? inputs()[0] : nullptr; }
    [[nodiscard]] auto dst() const noexcept { return output(); }
    [[nodiscard]] size_t serialize(luisa::vector<char> &output_buffer) override;
    bool deserialize(char const *&input_ptr, char const *end_ptr) override;
};

class LUISA_AST_API ViewStmt final : public TensorStmt {
public:
    ViewStmt() noexcept : TensorStmt{TileOpKind::VIEW} {}
    /// dtype view of `src`; when `dims` is empty the source dims are reused.
    ViewStmt(TensorExpr *src, TensorElementType dtype,
             luisa::fixed_vector<int32_t, 4> dims = {}) noexcept
        : TensorStmt{TileOpKind::VIEW,
                     new TensorExpr{dims.empty() ? src->rank() : static_cast<int32_t>(dims.size()),
                                    dtype, src->scope(),
                                    dims.empty() ? luisa::fixed_vector<int32_t, 4>{src->dims().begin(), src->dims().end()}
                                                 : std::move(dims),
                                    {}, {}, src->handle()},
                     {src}} {}
    [[nodiscard]] auto src() const noexcept { return inputs().size() > 0 ? inputs()[0] : nullptr; }
    [[nodiscard]] auto dst() const noexcept { return output(); }
    [[nodiscard]] size_t serialize(luisa::vector<char> &output_buffer) override;
    bool deserialize(char const *&input_ptr, char const *end_ptr) override;
};

// --- Reduce: T.reduce(buf, out, op, dim, clear, ...) ------------------------
// Generic reduction covering T.reduce and the T.reduce_max / reduce_min /
// reduce_abssum / reduce_absmax / reduce_bitand / reduce_bitor / reduce_bitxor
// shorthands (they differ only in TileReduceOp).
class LUISA_AST_API ReduceStmt final : public TensorStmt {
    TileReduceOp _op{TileReduceOp::SUM};// R1
    uint32_t _dim{};                    // R1: reduction dimension
    int32_t _clear{1};                  // R1: 0|1
    int32_t _batch{1};                  // R1: batched AllReduce width
    int32_t _nan_propagate{0};          // R1: 0|1 (f16/bf16 max/min only)

public:
    ReduceStmt() noexcept : TensorStmt{TileOpKind::REDUCE} {}
    ReduceStmt(TileReduceOp op, TensorExpr *buf, TensorExpr *out, uint32_t dim,
               int32_t clear = 1, int32_t batch = 1, int32_t nan_propagate = 0) noexcept
        : TensorStmt{TileOpKind::REDUCE, out, {buf}},
          _op{op}, _dim{dim}, _clear{clear}, _batch{batch}, _nan_propagate{nan_propagate} {}
    [[nodiscard]] auto buf() const noexcept { return inputs().size() > 0 ? inputs()[0] : nullptr; }
    [[nodiscard]] auto out() const noexcept { return output(); }
    [[nodiscard]] auto op() const noexcept { return _op; }
    [[nodiscard]] auto dim() const noexcept { return _dim; }
    [[nodiscard]] auto clear() const noexcept { return _clear; }
    [[nodiscard]] auto batch() const noexcept { return _batch; }
    [[nodiscard]] auto nan_propagate() const noexcept { return _nan_propagate; }
    [[nodiscard]] size_t serialize(luisa::vector<char> &output_buffer) override;
    bool deserialize(char const *&input_ptr, char const *end_ptr) override;
};

// --- FinalizeReducer: T.finalize_reducer(reducer, batch) --------------------
// Finalize an AllReduce reducer; the reducer tensor is READ+WRITE (output).
class LUISA_AST_API FinalizeReducerStmt final : public TensorStmt {
    int32_t _batch{1};// R1

public:
    FinalizeReducerStmt() noexcept : TensorStmt{TileOpKind::FINALIZE_REDUCER} {}
    explicit FinalizeReducerStmt(TensorExpr *reducer, int32_t batch = 1) noexcept
        : TensorStmt{TileOpKind::FINALIZE_REDUCER, reducer, {}}, _batch{batch} {}
    [[nodiscard]] auto reducer() const noexcept { return output(); }
    [[nodiscard]] auto batch() const noexcept { return _batch; }
    [[nodiscard]] size_t serialize(luisa::vector<char> &output_buffer) override;
    bool deserialize(char const *&input_ptr, char const *end_ptr) override;
};

// --- WarpReduce: T.warp_reduce_sum/max/min/bitand/bitor(value) --------------
// Register-level warp reduction; `value` is the READ operand (a fragment
// tensor); the reduced scalar is a caller-owned temporary (Rsqrt pattern).
class LUISA_AST_API WarpReduceStmt final : public TensorStmt {
    TileWarpReduceOp _op{TileWarpReduceOp::SUM};// R1

public:
    WarpReduceStmt() noexcept : TensorStmt{TileOpKind::WARP_REDUCE} {}
    WarpReduceStmt(TileWarpReduceOp op, TensorExpr *value) noexcept
        : TensorStmt{TileOpKind::WARP_REDUCE, nullptr, {value}}, _op{op} {}
    [[nodiscard]] auto op() const noexcept { return _op; }
    [[nodiscard]] auto value() const noexcept { return inputs().size() > 0 ? inputs()[0] : nullptr; }
    [[nodiscard]] size_t serialize(luisa::vector<char> &output_buffer) override;
    bool deserialize(char const *&input_ptr, char const *end_ptr) override;
};

// --- CumSum / CumMax: T.cumsum / T.cummax(src, dst, dim, reverse) -----------
class LUISA_AST_API CumSumStmt final : public TensorStmt {
    uint32_t _dim{};    // R1
    int32_t _reverse{}; // R1: 0|1

public:
    CumSumStmt() noexcept : TensorStmt{TileOpKind::CUMSUM} {}
    CumSumStmt(TensorExpr *src, TensorExpr *dst, uint32_t dim, int32_t reverse = 0) noexcept
        : TensorStmt{TileOpKind::CUMSUM, dst, {src}}, _dim{dim}, _reverse{reverse} {}
    [[nodiscard]] auto src() const noexcept { return inputs().size() > 0 ? inputs()[0] : nullptr; }
    [[nodiscard]] auto dst() const noexcept { return output(); }
    [[nodiscard]] auto dim() const noexcept { return _dim; }
    [[nodiscard]] auto reverse() const noexcept { return _reverse; }
    [[nodiscard]] size_t serialize(luisa::vector<char> &output_buffer) override;
    bool deserialize(char const *&input_ptr, char const *end_ptr) override;
};

class LUISA_AST_API CumMaxStmt final : public TensorStmt {
    uint32_t _dim{};    // R1
    int32_t _reverse{}; // R1: 0|1

public:
    CumMaxStmt() noexcept : TensorStmt{TileOpKind::CUMMAX} {}
    CumMaxStmt(TensorExpr *src, TensorExpr *dst, uint32_t dim, int32_t reverse = 0) noexcept
        : TensorStmt{TileOpKind::CUMMAX, dst, {src}}, _dim{dim}, _reverse{reverse} {}
    [[nodiscard]] auto src() const noexcept { return inputs().size() > 0 ? inputs()[0] : nullptr; }
    [[nodiscard]] auto dst() const noexcept { return output(); }
    [[nodiscard]] auto dim() const noexcept { return _dim; }
    [[nodiscard]] auto reverse() const noexcept { return _reverse; }
    [[nodiscard]] size_t serialize(luisa::vector<char> &output_buffer) override;
    bool deserialize(char const *&input_ptr, char const *end_ptr) override;
};

// --- WgmmaGemm: T.wgmma_gemm(a, b, c, ...) ----------------------------------
// Explicit Hopper WGMMA GEMM (no implicit wait).
class LUISA_AST_API WgmmaGemmStmt final : public TensorStmt {
    int32_t _trans_a{};                              // R1: 0|1
    int32_t _trans_b{};                              // R1: 0|1
    GemmWarpPolicy _policy{GemmWarpPolicy::Square};  // R1
    int32_t _clear_accum{};                          // R1: 0|1

public:
    WgmmaGemmStmt() noexcept : TensorStmt{TileOpKind::WGGMA_GEMM} {}
    WgmmaGemmStmt(TensorExpr *a, TensorExpr *b, TensorExpr *c,
                  int32_t trans_a = 0, int32_t trans_b = 0,
                  GemmWarpPolicy policy = GemmWarpPolicy::Square,
                  int32_t clear_accum = 0) noexcept
        : TensorStmt{TileOpKind::WGGMA_GEMM, c, {a, b}},
          _trans_a{trans_a}, _trans_b{trans_b}, _policy{policy}, _clear_accum{clear_accum} {}
    [[nodiscard]] auto a() const noexcept { return inputs().size() > 0 ? inputs()[0] : nullptr; }
    [[nodiscard]] auto b() const noexcept { return inputs().size() > 1 ? inputs()[1] : nullptr; }
    [[nodiscard]] auto c() const noexcept { return output(); }
    [[nodiscard]] auto trans_a() const noexcept { return _trans_a; }
    [[nodiscard]] auto trans_b() const noexcept { return _trans_b; }
    [[nodiscard]] auto policy() const noexcept { return _policy; }
    [[nodiscard]] auto clear_accum() const noexcept { return _clear_accum; }
    [[nodiscard]] size_t serialize(luisa::vector<char> &output_buffer) override;
    bool deserialize(char const *&input_ptr, char const *end_ptr) override;
};

// --- Tcgen05Gemm: T.tcgen05_gemm(a, b, c, ..., mbar) ------------------------
// Explicit Blackwell TCGEN05 GEMM (no implicit wait); mbar is an optional
// mbarrier tensor (inputs[2]).
class LUISA_AST_API Tcgen05GemmStmt final : public TensorStmt {
    int32_t _trans_a{};                             // R1: 0|1
    int32_t _trans_b{};                             // R1: 0|1
    GemmWarpPolicy _policy{GemmWarpPolicy::Square}; // R1
    int32_t _clear_accum{};                         // R1: 0|1

public:
    Tcgen05GemmStmt() noexcept : TensorStmt{TileOpKind::TCGEN05_GEMM} {}
    Tcgen05GemmStmt(TensorExpr *a, TensorExpr *b, TensorExpr *c,
                    int32_t trans_a = 0, int32_t trans_b = 0,
                    GemmWarpPolicy policy = GemmWarpPolicy::Square,
                    int32_t clear_accum = 0, TensorExpr *mbar = nullptr) noexcept
        : TensorStmt{TileOpKind::TCGEN05_GEMM, c,
                     mbar != nullptr ? luisa::vector<TensorExpr *>{a, b, mbar}
                                     : luisa::vector<TensorExpr *>{a, b}},
          _trans_a{trans_a}, _trans_b{trans_b}, _policy{policy}, _clear_accum{clear_accum} {}
    [[nodiscard]] auto a() const noexcept { return inputs().size() > 0 ? inputs()[0] : nullptr; }
    [[nodiscard]] auto b() const noexcept { return inputs().size() > 1 ? inputs()[1] : nullptr; }
    [[nodiscard]] auto c() const noexcept { return output(); }
    [[nodiscard]] auto mbar() const noexcept { return inputs().size() > 2 ? inputs()[2] : nullptr; }
    [[nodiscard]] auto trans_a() const noexcept { return _trans_a; }
    [[nodiscard]] auto trans_b() const noexcept { return _trans_b; }
    [[nodiscard]] auto policy() const noexcept { return _policy; }
    [[nodiscard]] auto clear_accum() const noexcept { return _clear_accum; }
    [[nodiscard]] size_t serialize(luisa::vector<char> &output_buffer) override;
    bool deserialize(char const *&input_ptr, char const *end_ptr) override;
};

// --- Tcgen05GemmBlockscaled: T.tcgen05_gemm_blockscaled ---------------------
// FP8/FP6/FP4 block-scaled GEMM; A/B are shared, C is tmem, SFA/SFB are the
// E8M0 scale-factor tensors in tmem.
class LUISA_AST_API Tcgen05GemmBlockscaledStmt final : public TensorStmt {
    int32_t _trans_a{};        // R1: 0|1
    int32_t _trans_b{};        // R1: 0|1
    int32_t _clear_accum{};    // R1: 0|1
    int32_t _k_start{};        // R1: logical K-axis start offset
    int32_t _sf_a_granularity_k{1};// R1: K elements per A scale factor
    int32_t _sf_b_granularity_k{1};// R1: K elements per B scale factor

public:
    Tcgen05GemmBlockscaledStmt() noexcept : TensorStmt{TileOpKind::TCGEN05_GEMM_BLOCKSCALED} {}
    Tcgen05GemmBlockscaledStmt(TensorExpr *a, TensorExpr *b, TensorExpr *c,
                               TensorExpr *sfa, TensorExpr *sfb,
                               int32_t trans_a = 0, int32_t trans_b = 0,
                               int32_t clear_accum = 0, int32_t k_start = 0,
                               int32_t sf_a_granularity_k = 1, int32_t sf_b_granularity_k = 1) noexcept
        : TensorStmt{TileOpKind::TCGEN05_GEMM_BLOCKSCALED, c, {a, b, sfa, sfb}},
          _trans_a{trans_a}, _trans_b{trans_b}, _clear_accum{clear_accum},
          _k_start{k_start}, _sf_a_granularity_k{sf_a_granularity_k},
          _sf_b_granularity_k{sf_b_granularity_k} {}
    [[nodiscard]] auto a() const noexcept { return inputs().size() > 0 ? inputs()[0] : nullptr; }
    [[nodiscard]] auto b() const noexcept { return inputs().size() > 1 ? inputs()[1] : nullptr; }
    [[nodiscard]] auto c() const noexcept { return output(); }
    [[nodiscard]] auto sfa() const noexcept { return inputs().size() > 2 ? inputs()[2] : nullptr; }
    [[nodiscard]] auto sfb() const noexcept { return inputs().size() > 3 ? inputs()[3] : nullptr; }
    [[nodiscard]] auto trans_a() const noexcept { return _trans_a; }
    [[nodiscard]] auto trans_b() const noexcept { return _trans_b; }
    [[nodiscard]] auto clear_accum() const noexcept { return _clear_accum; }
    [[nodiscard]] auto k_start() const noexcept { return _k_start; }
    [[nodiscard]] auto sf_a_granularity_k() const noexcept { return _sf_a_granularity_k; }
    [[nodiscard]] auto sf_b_granularity_k() const noexcept { return _sf_b_granularity_k; }
    [[nodiscard]] size_t serialize(luisa::vector<char> &output_buffer) override;
    bool deserialize(char const *&input_ptr, char const *end_ptr) override;
};

// --- Sparse GEMM: T.gemm_sp / T.wgmma_gemm_sp / T.tcgen05_gemm_sp -----------
// A_sparse is the compressed non-zero operand, E its sparsity metadata, B the
// dense operand and C the accumulator (READ+WRITE).
class LUISA_AST_API GemmSpStmt final : public TensorStmt {
    int32_t _trans_a{};                             // R1: 0|1
    int32_t _trans_e{};                             // R1: 0|1
    int32_t _trans_b{};                             // R1: 0|1
    GemmWarpPolicy _policy{GemmWarpPolicy::Square}; // R1
    int32_t _clear_accum{};                         // R1: 0|1

public:
    GemmSpStmt() noexcept : TensorStmt{TileOpKind::GEMM_SP} {}
    GemmSpStmt(TensorExpr *a_sparse, TensorExpr *e, TensorExpr *b, TensorExpr *c,
               int32_t trans_a = 0, int32_t trans_e = 0, int32_t trans_b = 0,
               GemmWarpPolicy policy = GemmWarpPolicy::Square,
               int32_t clear_accum = 0) noexcept
        : TensorStmt{TileOpKind::GEMM_SP, c, {a_sparse, e, b}},
          _trans_a{trans_a}, _trans_e{trans_e}, _trans_b{trans_b},
          _policy{policy}, _clear_accum{clear_accum} {}
    [[nodiscard]] auto a() const noexcept { return inputs().size() > 0 ? inputs()[0] : nullptr; }
    [[nodiscard]] auto e() const noexcept { return inputs().size() > 1 ? inputs()[1] : nullptr; }
    [[nodiscard]] auto b() const noexcept { return inputs().size() > 2 ? inputs()[2] : nullptr; }
    [[nodiscard]] auto c() const noexcept { return output(); }
    [[nodiscard]] auto trans_a() const noexcept { return _trans_a; }
    [[nodiscard]] auto trans_e() const noexcept { return _trans_e; }
    [[nodiscard]] auto trans_b() const noexcept { return _trans_b; }
    [[nodiscard]] auto policy() const noexcept { return _policy; }
    [[nodiscard]] auto clear_accum() const noexcept { return _clear_accum; }
    [[nodiscard]] size_t serialize(luisa::vector<char> &output_buffer) override;
    bool deserialize(char const *&input_ptr, char const *end_ptr) override;
};

class LUISA_AST_API WgmmaGemmSpStmt final : public TensorStmt {
    int32_t _trans_a{};                             // R1: 0|1
    int32_t _trans_e{};                             // R1: 0|1
    int32_t _trans_b{};                             // R1: 0|1
    GemmWarpPolicy _policy{GemmWarpPolicy::Square}; // R1
    int32_t _clear_accum{};                         // R1: 0|1

public:
    WgmmaGemmSpStmt() noexcept : TensorStmt{TileOpKind::WGGMA_GEMM_SP} {}
    WgmmaGemmSpStmt(TensorExpr *a_sparse, TensorExpr *e, TensorExpr *b, TensorExpr *c,
                    int32_t trans_a = 0, int32_t trans_e = 0, int32_t trans_b = 0,
                    GemmWarpPolicy policy = GemmWarpPolicy::Square,
                    int32_t clear_accum = 0) noexcept
        : TensorStmt{TileOpKind::WGGMA_GEMM_SP, c, {a_sparse, e, b}},
          _trans_a{trans_a}, _trans_e{trans_e}, _trans_b{trans_b},
          _policy{policy}, _clear_accum{clear_accum} {}
    [[nodiscard]] auto a() const noexcept { return inputs().size() > 0 ? inputs()[0] : nullptr; }
    [[nodiscard]] auto e() const noexcept { return inputs().size() > 1 ? inputs()[1] : nullptr; }
    [[nodiscard]] auto b() const noexcept { return inputs().size() > 2 ? inputs()[2] : nullptr; }
    [[nodiscard]] auto c() const noexcept { return output(); }
    [[nodiscard]] auto trans_a() const noexcept { return _trans_a; }
    [[nodiscard]] auto trans_e() const noexcept { return _trans_e; }
    [[nodiscard]] auto trans_b() const noexcept { return _trans_b; }
    [[nodiscard]] auto policy() const noexcept { return _policy; }
    [[nodiscard]] auto clear_accum() const noexcept { return _clear_accum; }
    [[nodiscard]] size_t serialize(luisa::vector<char> &output_buffer) override;
    bool deserialize(char const *&input_ptr, char const *end_ptr) override;
};

class LUISA_AST_API Tcgen05GemmSpStmt final : public TensorStmt {
    int32_t _trans_a{};                             // R1: 0|1
    int32_t _trans_e{};                             // R1: 0|1
    int32_t _trans_b{};                             // R1: 0|1
    GemmWarpPolicy _policy{GemmWarpPolicy::Square}; // R1
    int32_t _clear_accum{};                         // R1: 0|1

public:
    Tcgen05GemmSpStmt() noexcept : TensorStmt{TileOpKind::TCGEN05_GEMM_SP} {}
    Tcgen05GemmSpStmt(TensorExpr *a_sparse, TensorExpr *e, TensorExpr *b, TensorExpr *c,
                      int32_t trans_a = 0, int32_t trans_e = 0, int32_t trans_b = 0,
                      GemmWarpPolicy policy = GemmWarpPolicy::Square,
                      int32_t clear_accum = 0) noexcept
        : TensorStmt{TileOpKind::TCGEN05_GEMM_SP, c, {a_sparse, e, b}},
          _trans_a{trans_a}, _trans_e{trans_e}, _trans_b{trans_b},
          _policy{policy}, _clear_accum{clear_accum} {}
    [[nodiscard]] auto a() const noexcept { return inputs().size() > 0 ? inputs()[0] : nullptr; }
    [[nodiscard]] auto e() const noexcept { return inputs().size() > 1 ? inputs()[1] : nullptr; }
    [[nodiscard]] auto b() const noexcept { return inputs().size() > 2 ? inputs()[2] : nullptr; }
    [[nodiscard]] auto c() const noexcept { return output(); }
    [[nodiscard]] auto trans_a() const noexcept { return _trans_a; }
    [[nodiscard]] auto trans_e() const noexcept { return _trans_e; }
    [[nodiscard]] auto trans_b() const noexcept { return _trans_b; }
    [[nodiscard]] auto policy() const noexcept { return _policy; }
    [[nodiscard]] auto clear_accum() const noexcept { return _clear_accum; }
    [[nodiscard]] size_t serialize(luisa::vector<char> &output_buffer) override;
    bool deserialize(char const *&input_ptr, char const *end_ptr) override;
};

// --- Atomic: T.atomic_add/max/min/addx2/addx4/load/or/store -----------------
// dst (READ+WRITE) is the result tensor; the value is an optional tensor
// (inputs[0]) or an R2 literal / R3 runtime scalar.  memory_order / return_prev /
// use_tma are the TileLang knobs.
class LUISA_AST_API AtomicStmt final : public TensorStmt {
    TileAtomicOp _op{TileAtomicOp::ADD};              // R1
    TileMemoryOrder _memory_order{TileMemoryOrder::RELAXED};// R1
    int32_t _return_prev{};                           // R1: 0|1
    int32_t _use_tma{};                               // R1: 0|1
    const LiteralExpr *_value_literal{nullptr};       // R2 (borrowed)
    const RefExpr *_value_ref{nullptr};               // R3 (borrowed, non-serializable)

public:
    AtomicStmt() noexcept : TensorStmt{TileOpKind::ATOMIC} {}
    AtomicStmt(TileAtomicOp op, TensorExpr *dst, TensorExpr *value_tensor = nullptr,
               const LiteralExpr *value_literal = nullptr,
               const RefExpr *value_ref = nullptr,
               TileMemoryOrder memory_order = TileMemoryOrder::RELAXED,
               int32_t return_prev = 0, int32_t use_tma = 0) noexcept
        : TensorStmt{TileOpKind::ATOMIC, dst,
                     value_tensor != nullptr ? luisa::vector<TensorExpr *>{value_tensor}
                                             : luisa::vector<TensorExpr *>{}},
          _op{op}, _memory_order{memory_order}, _return_prev{return_prev},
          _use_tma{use_tma}, _value_literal{value_literal}, _value_ref{value_ref} {}
    [[nodiscard]] auto dst() const noexcept { return output(); }
    [[nodiscard]] auto value_tensor() const noexcept {
        return inputs().size() > 0 ? inputs()[0] : nullptr;
    }
    [[nodiscard]] auto value_literal() const noexcept { return _value_literal; }
    [[nodiscard]] auto value_ref() const noexcept { return _value_ref; }
    [[nodiscard]] auto op() const noexcept { return _op; }
    [[nodiscard]] auto memory_order() const noexcept { return _memory_order; }
    [[nodiscard]] auto return_prev() const noexcept { return _return_prev; }
    [[nodiscard]] auto use_tma() const noexcept { return _use_tma; }
    [[nodiscard]] size_t serialize(luisa::vector<char> &output_buffer) override;
    bool deserialize(char const *&input_ptr, char const *end_ptr) override;
};

// --- Clamp: T.clamp(dst, lo, hi) --------------------------------------------
// Elementwise clamp dst into [lo, hi]; lo/hi are R2 literals or R3 scalars.
class LUISA_AST_API ClampStmt final : public TensorStmt {
    const LiteralExpr *_lo_literal{nullptr};// R2 (borrowed)
    const RefExpr *_lo_ref{nullptr};        // R3 (borrowed, non-serializable)
    const LiteralExpr *_hi_literal{nullptr};// R2 (borrowed)
    const RefExpr *_hi_ref{nullptr};        // R3 (borrowed, non-serializable)

public:
    ClampStmt() noexcept : TensorStmt{TileOpKind::CLAMP} {}
    ClampStmt(TensorExpr *dst, const LiteralExpr *lo, const LiteralExpr *hi) noexcept
        : TensorStmt{TileOpKind::CLAMP, dst, {}},
          _lo_literal{lo}, _hi_literal{hi} {}
    ClampStmt(TensorExpr *dst, const RefExpr *lo, const RefExpr *hi) noexcept
        : TensorStmt{TileOpKind::CLAMP, dst, {}},
          _lo_ref{lo}, _hi_ref{hi} {}
    ClampStmt(TensorExpr *dst, const LiteralExpr *lo, const RefExpr *hi) noexcept
        : TensorStmt{TileOpKind::CLAMP, dst, {}},
          _lo_literal{lo}, _hi_ref{hi} {}
    ClampStmt(TensorExpr *dst, const RefExpr *lo, const LiteralExpr *hi) noexcept
        : TensorStmt{TileOpKind::CLAMP, dst, {}},
          _lo_ref{lo}, _hi_literal{hi} {}
    [[nodiscard]] auto dst() const noexcept { return output(); }
    [[nodiscard]] auto lo_literal() const noexcept { return _lo_literal; }
    [[nodiscard]] auto lo_ref() const noexcept { return _lo_ref; }
    [[nodiscard]] auto hi_literal() const noexcept { return _hi_literal; }
    [[nodiscard]] auto hi_ref() const noexcept { return _hi_ref; }
    [[nodiscard]] size_t serialize(luisa::vector<char> &output_buffer) override;
    bool deserialize(char const *&input_ptr, char const *end_ptr) override;
};

// --- Dp4a: T.dp4a(A, B, C) --------------------------------------------------
// Four-element signed int8 dot product accumulated into int32 C (READ+WRITE).
class LUISA_AST_API Dp4aStmt final : public TensorStmt {
public:
    Dp4aStmt() noexcept : TensorStmt{TileOpKind::DP4A} {}
    Dp4aStmt(TensorExpr *a, TensorExpr *b, TensorExpr *c) noexcept
        : TensorStmt{TileOpKind::DP4A, c, {a, b}} {}
    [[nodiscard]] auto a() const noexcept { return inputs().size() > 0 ? inputs()[0] : nullptr; }
    [[nodiscard]] auto b() const noexcept { return inputs().size() > 1 ? inputs()[1] : nullptr; }
    [[nodiscard]] auto c() const noexcept { return output(); }
    [[nodiscard]] size_t serialize(luisa::vector<char> &output_buffer) override;
    bool deserialize(char const *&input_ptr, char const *end_ptr) override;
};

// --- LoopBreak: T.loop_break() ----------------------------------------------
class LUISA_AST_API LoopBreakStmt final : public TensorStmt {
public:
    LoopBreakStmt() noexcept : TensorStmt{TileOpKind::LOOP_BREAK} {}
    [[nodiscard]] size_t serialize(luisa::vector<char> &output_buffer) override;
    bool deserialize(char const *&input_ptr, char const *end_ptr) override;
};

// --- AnyOf / AllOf: T.any_of(buf) / T.all_of(buf) ---------------------------
// Logical tile reduction to a scalar boolean; `buf` is the READ operand and the
// boolean result is a caller-owned temporary (Rsqrt pattern).
class LUISA_AST_API AnyOfStmt final : public TensorStmt {
public:
    AnyOfStmt() noexcept : TensorStmt{TileOpKind::ANY_OF} {}
    explicit AnyOfStmt(TensorExpr *buf) noexcept
        : TensorStmt{TileOpKind::ANY_OF, nullptr, {buf}} {}
    [[nodiscard]] auto buf() const noexcept { return inputs().size() > 0 ? inputs()[0] : nullptr; }
    [[nodiscard]] size_t serialize(luisa::vector<char> &output_buffer) override;
    bool deserialize(char const *&input_ptr, char const *end_ptr) override;
};

class LUISA_AST_API AllOfStmt final : public TensorStmt {
public:
    AllOfStmt() noexcept : TensorStmt{TileOpKind::ALL_OF} {}
    explicit AllOfStmt(TensorExpr *buf) noexcept
        : TensorStmt{TileOpKind::ALL_OF, nullptr, {buf}} {}
    [[nodiscard]] auto buf() const noexcept { return inputs().size() > 0 ? inputs()[0] : nullptr; }
    [[nodiscard]] size_t serialize(luisa::vector<char> &output_buffer) override;
    bool deserialize(char const *&input_ptr, char const *end_ptr) override;
};

// --- Sync: T.sync_threads / sync_warp / sync_grid / sync_global -------------
class LUISA_AST_API SyncStmt final : public TensorStmt {
    TileSyncOp _op{TileSyncOp::THREADS};// R1
    int32_t _mask{0};                   // R1: sync_warp lane mask (0 = full warp)
    int32_t _barrier_id{-1};            // R1: sync_threads named barrier id (-1 = none)
    int32_t _arrive_count{0};           // R1: sync_threads arrive count (0 = unset)

public:
    SyncStmt() noexcept : TensorStmt{TileOpKind::SYNC} {}
    explicit SyncStmt(TileSyncOp op, int32_t mask = 0, int32_t barrier_id = -1,
             int32_t arrive_count = 0) noexcept
        : TensorStmt{TileOpKind::SYNC}, _op{op}, _mask{mask},
          _barrier_id{barrier_id}, _arrive_count{arrive_count} {}
    [[nodiscard]] auto op() const noexcept { return _op; }
    [[nodiscard]] auto mask() const noexcept { return _mask; }
    [[nodiscard]] auto barrier_id() const noexcept { return _barrier_id; }
    [[nodiscard]] auto arrive_count() const noexcept { return _arrive_count; }
    [[nodiscard]] size_t serialize(luisa::vector<char> &output_buffer) override;
    bool deserialize(char const *&input_ptr, char const *end_ptr) override;
};

// --- Barrier: T.barrier_arrive / barrier_wait / named_barrier_arrive --------
// arrive/wait operate on an mbarrier tensor (inputs[0]); named_barrier_arrive
// carries barrier_id + thread_count and has no tensor operand.
class LUISA_AST_API BarrierStmt final : public TensorStmt {
    TileBarrierOp _op{TileBarrierOp::ARRIVE};// R1
    int32_t _parity{0};                      // R1: barrier_wait parity
    int32_t _barrier_id{0};                  // R1: named_barrier_arrive index
    int32_t _thread_count{0};                // R1: named_barrier_arrive participants

public:
    BarrierStmt() noexcept : TensorStmt{TileOpKind::BARRIER} {}
    explicit BarrierStmt(TileBarrierOp op, TensorExpr *mbarrier = nullptr,
                int32_t parity = 0, int32_t barrier_id = 0,
                int32_t thread_count = 0) noexcept
        : TensorStmt{TileOpKind::BARRIER, nullptr,
                     mbarrier != nullptr ? luisa::vector<TensorExpr *>{mbarrier}
                                         : luisa::vector<TensorExpr *>{}},
          _op{op}, _parity{parity}, _barrier_id{barrier_id}, _thread_count{thread_count} {}
    [[nodiscard]] auto op() const noexcept { return _op; }
    [[nodiscard]] auto mbarrier() const noexcept { return inputs().size() > 0 ? inputs()[0] : nullptr; }
    [[nodiscard]] auto parity() const noexcept { return _parity; }
    [[nodiscard]] auto barrier_id() const noexcept { return _barrier_id; }
    [[nodiscard]] auto thread_count() const noexcept { return _thread_count; }
    [[nodiscard]] size_t serialize(luisa::vector<char> &output_buffer) override;
    bool deserialize(char const *&input_ptr, char const *end_ptr) override;
};

// --- MBarrier: T.mbarrier_arrive / arrive_expect_tx / expect_tx / wait_parity
class LUISA_AST_API MBarrierStmt final : public TensorStmt {
    TileMBarrierOp _op{TileMBarrierOp::ARRIVE};// R1
    int32_t _tx{0};                            // R1: expected transaction bytes
    int32_t _parity{0};                        // R1: wait parity
    int32_t _cta_id{-1};                       // R1: peer CTA rank (-1 = current CTA)

public:
    MBarrierStmt() noexcept : TensorStmt{TileOpKind::MBARRIER} {}
    MBarrierStmt(TileMBarrierOp op, TensorExpr *mbarrier,
                 int32_t tx = 0, int32_t parity = 0, int32_t cta_id = -1) noexcept
        : TensorStmt{TileOpKind::MBARRIER, nullptr, {mbarrier}},
          _op{op}, _tx{tx}, _parity{parity}, _cta_id{cta_id} {}
    [[nodiscard]] auto op() const noexcept { return _op; }
    [[nodiscard]] auto mbarrier() const noexcept { return inputs().size() > 0 ? inputs()[0] : nullptr; }
    [[nodiscard]] auto tx() const noexcept { return _tx; }
    [[nodiscard]] auto parity() const noexcept { return _parity; }
    [[nodiscard]] auto cta_id() const noexcept { return _cta_id; }
    [[nodiscard]] size_t serialize(luisa::vector<char> &output_buffer) override;
    bool deserialize(char const *&input_ptr, char const *end_ptr) override;
};

// --- WarpVote: T.activemask / ballot / ballot_sync / any_sync / all_sync /
// match_any_sync / match_all_sync ---------------------------------------------
// mask (R3) is the warp lane mask of the *_sync forms; pred/value is an R2
// literal or R3 runtime scalar.  The mask/ballot result is a caller-owned temp.
class LUISA_AST_API WarpVoteStmt final : public TensorStmt {
    TileWarpVoteOp _op{TileWarpVoteOp::ACTIVEMASK};// R1
    const RefExpr *_mask{nullptr};                 // R3 (borrowed, non-serializable)
    const LiteralExpr *_pred_literal{nullptr};     // R2 (borrowed)
    const RefExpr *_pred_ref{nullptr};             // R3 (borrowed, non-serializable)

public:
    WarpVoteStmt() noexcept : TensorStmt{TileOpKind::WARP_VOTE} {}
    explicit WarpVoteStmt(TileWarpVoteOp op, const RefExpr *mask = nullptr,
                 const LiteralExpr *pred_literal = nullptr,
                 const RefExpr *pred_ref = nullptr) noexcept
        : TensorStmt{TileOpKind::WARP_VOTE, nullptr, {}},
          _op{op}, _mask{mask}, _pred_literal{pred_literal}, _pred_ref{pred_ref} {}
    [[nodiscard]] auto op() const noexcept { return _op; }
    [[nodiscard]] auto mask() const noexcept { return _mask; }
    [[nodiscard]] auto pred_literal() const noexcept { return _pred_literal; }
    [[nodiscard]] auto pred_ref() const noexcept { return _pred_ref; }
    [[nodiscard]] size_t serialize(luisa::vector<char> &output_buffer) override;
    bool deserialize(char const *&input_ptr, char const *end_ptr) override;
};

// --- Shuffle: T.shfl_sync / shfl_xor / shfl_up / shfl_down / shuffle_elect ---
class LUISA_AST_API ShuffleStmt final : public TensorStmt {
    TileShuffleOp _op{TileShuffleOp::SYNC};// R1
    int32_t _width{32};                    // R1: subgroup width
    int32_t _delta{0};                     // R1: xor/up/down lane delta
    int32_t _thread_extent{0};             // R1: shuffle_elect group size (0 = whole block)
    const RefExpr *_mask{nullptr};         // R3 (borrowed, non-serializable)
    const RefExpr *_src_lane{nullptr};     // R3 (borrowed, non-serializable): shfl_sync source
    const LiteralExpr *_value_literal{nullptr};// R2 (borrowed)
    const RefExpr *_value_ref{nullptr};    // R3 (borrowed, non-serializable)

public:
    ShuffleStmt() noexcept : TensorStmt{TileOpKind::SHUFFLE} {}
    explicit ShuffleStmt(TileShuffleOp op, const LiteralExpr *value_literal = nullptr,
                const RefExpr *value_ref = nullptr, const RefExpr *mask = nullptr,
                const RefExpr *src_lane = nullptr, int32_t width = 32,
                int32_t delta = 0, int32_t thread_extent = 0) noexcept
        : TensorStmt{TileOpKind::SHUFFLE, nullptr, {}},
          _op{op}, _width{width}, _delta{delta}, _thread_extent{thread_extent},
          _mask{mask}, _src_lane{src_lane},
          _value_literal{value_literal}, _value_ref{value_ref} {}
    [[nodiscard]] auto op() const noexcept { return _op; }
    [[nodiscard]] auto width() const noexcept { return _width; }
    [[nodiscard]] auto delta() const noexcept { return _delta; }
    [[nodiscard]] auto thread_extent() const noexcept { return _thread_extent; }
    [[nodiscard]] auto mask() const noexcept { return _mask; }
    [[nodiscard]] auto src_lane() const noexcept { return _src_lane; }
    [[nodiscard]] auto value_literal() const noexcept { return _value_literal; }
    [[nodiscard]] auto value_ref() const noexcept { return _value_ref; }
    [[nodiscard]] size_t serialize(luisa::vector<char> &output_buffer) override;
    bool deserialize(char const *&input_ptr, char const *end_ptr) override;
};

// --- SyncThreadsVote: T.syncthreads_count / syncthreads_and / syncthreads_or -
class LUISA_AST_API SyncThreadsVoteStmt final : public TensorStmt {
    TileSyncThreadsVoteOp _op{TileSyncThreadsVoteOp::COUNT};// R1
    const LiteralExpr *_pred_literal{nullptr};// R2 (borrowed)
    const RefExpr *_pred_ref{nullptr};        // R3 (borrowed, non-serializable)

public:
    SyncThreadsVoteStmt() noexcept : TensorStmt{TileOpKind::SYNC_THREADS_VOTE} {}
    explicit SyncThreadsVoteStmt(TileSyncThreadsVoteOp op, const LiteralExpr *pred_literal = nullptr,
                        const RefExpr *pred_ref = nullptr) noexcept
        : TensorStmt{TileOpKind::SYNC_THREADS_VOTE, nullptr, {}},
          _op{op}, _pred_literal{pred_literal}, _pred_ref{pred_ref} {}
    [[nodiscard]] auto op() const noexcept { return _op; }
    [[nodiscard]] auto pred_literal() const noexcept { return _pred_literal; }
    [[nodiscard]] auto pred_ref() const noexcept { return _pred_ref; }
    [[nodiscard]] size_t serialize(luisa::vector<char> &output_buffer) override;
    bool deserialize(char const *&input_ptr, char const *end_ptr) override;
};

// --- FastRcp: T.fast_rcp(a) -------------------------------------------------
class LUISA_AST_API FastRcpStmt final : public TensorStmt {
public:
    FastRcpStmt() noexcept : TensorStmt{TileOpKind::FAST_RCP} {}
    explicit FastRcpStmt(TensorExpr *a) noexcept
        : TensorStmt{TileOpKind::FAST_RCP, nullptr, {a}} {}
    [[nodiscard]] auto a() const noexcept { return inputs().size() > 0 ? inputs()[0] : nullptr; }
    [[nodiscard]] size_t serialize(luisa::vector<char> &output_buffer) override;
    bool deserialize(char const *&input_ptr, char const *end_ptr) override;
};

// --- IeeeMath: T.ieee_add/sub/mul/fmaf/frcp/fsqrt/frsqrt/fdiv ---------------
// Unary (frcp/fsqrt/frsqrt), binary (add/sub/mul/fdiv) or ternary (fmaf) with
// an R1 rounding-mode id (0=rn, 1=rz, 2=ru, 3=rd).
class LUISA_AST_API IeeeMathStmt final : public TensorStmt {
    TileIeeeOp _op{TileIeeeOp::ADD};// R1
    int32_t _rounding_mode{0};      // R1

public:
    IeeeMathStmt() noexcept : TensorStmt{TileOpKind::IEEE_MATH} {}
    IeeeMathStmt(TileIeeeOp op, TensorExpr *a, TensorExpr *b = nullptr,
                 TensorExpr *c = nullptr, int32_t rounding_mode = 0) noexcept
        : TensorStmt{TileOpKind::IEEE_MATH, nullptr,
                     c != nullptr ? luisa::vector<TensorExpr *>{a, b, c}
                     : b != nullptr ? luisa::vector<TensorExpr *>{a, b}
                                    : luisa::vector<TensorExpr *>{a}},
          _op{op}, _rounding_mode{rounding_mode} {}
    [[nodiscard]] auto op() const noexcept { return _op; }
    [[nodiscard]] auto rounding_mode() const noexcept { return _rounding_mode; }
    [[nodiscard]] auto a() const noexcept { return inputs().size() > 0 ? inputs()[0] : nullptr; }
    [[nodiscard]] auto b() const noexcept { return inputs().size() > 1 ? inputs()[1] : nullptr; }
    [[nodiscard]] auto c() const noexcept { return inputs().size() > 2 ? inputs()[2] : nullptr; }
    [[nodiscard]] size_t serialize(luisa::vector<char> &output_buffer) override;
    bool deserialize(char const *&input_ptr, char const *end_ptr) override;
};

// --- PackedMath: T.add2/sub2/mul2/fma2/max2/min2/abs2 -----------------------
class LUISA_AST_API PackedMathStmt final : public TensorStmt {
    TilePackedOp _op{TilePackedOp::ADD2};// R1

public:
    PackedMathStmt() noexcept : TensorStmt{TileOpKind::PACKED_MATH} {}
    PackedMathStmt(TilePackedOp op, TensorExpr *a, TensorExpr *b = nullptr,
                   TensorExpr *c = nullptr) noexcept
        : TensorStmt{TileOpKind::PACKED_MATH, nullptr,
                     c != nullptr ? luisa::vector<TensorExpr *>{a, b, c}
                     : b != nullptr ? luisa::vector<TensorExpr *>{a, b}
                                    : luisa::vector<TensorExpr *>{a}},
          _op{op} {}
    [[nodiscard]] auto op() const noexcept { return _op; }
    [[nodiscard]] auto a() const noexcept { return inputs().size() > 0 ? inputs()[0] : nullptr; }
    [[nodiscard]] auto b() const noexcept { return inputs().size() > 1 ? inputs()[1] : nullptr; }
    [[nodiscard]] auto c() const noexcept { return inputs().size() > 2 ? inputs()[2] : nullptr; }
    [[nodiscard]] size_t serialize(luisa::vector<char> &output_buffer) override;
    bool deserialize(char const *&input_ptr, char const *end_ptr) override;
};

// --- FastMath: T.__exp / __exp10 / __log / __log2 / __log10 / __sin / __cos /
// __tan (all unary) ------------------------------------------------------------
class LUISA_AST_API FastMathStmt final : public TensorStmt {
    TileFastMathOp _op{TileFastMathOp::EXP};// R1

public:
    FastMathStmt() noexcept : TensorStmt{TileOpKind::FAST_MATH} {}
    FastMathStmt(TileFastMathOp op, TensorExpr *a) noexcept
        : TensorStmt{TileOpKind::FAST_MATH, nullptr, {a}}, _op{op} {}
    [[nodiscard]] auto op() const noexcept { return _op; }
    [[nodiscard]] auto a() const noexcept { return inputs().size() > 0 ? inputs()[0] : nullptr; }
    [[nodiscard]] size_t serialize(luisa::vector<char> &output_buffer) override;
    bool deserialize(char const *&input_ptr, char const *end_ptr) override;
};

// --- AllocSpecial: T.alloc_var/local/global/barrier/reducer/tmem/descriptor/
// cluster_barrier --------------------------------------------------------------
// The allocated object is the output TensorExpr (its dtype/scope encode the
// object kind); count / reducer_op / desc_kind / init carry the per-kind knobs.
class LUISA_AST_API AllocSpecialStmt final : public TensorStmt {
    TileAllocKind _kind{TileAllocKind::VAR};// R1
    int32_t _count{0};                      // R1: barrier/cluster_barrier arrive count
    int32_t _reducer_op{0};                 // R1: reducer 0=sum, 1=max, 2=min
    int32_t _desc_kind{0};                  // R1: descriptor 0=wgmma, 1=tcgen05_smem, 2=tcgen05_instr
    const LiteralExpr *_init{nullptr};      // R2 (borrowed): alloc_var initializer

public:
    AllocSpecialStmt() noexcept : TensorStmt{TileOpKind::ALLOC_SPECIAL} {}
    AllocSpecialStmt(TileAllocKind kind, TensorExpr *tensor, int32_t count = 0,
                     int32_t reducer_op = 0, int32_t desc_kind = 0,
                     const LiteralExpr *init = nullptr) noexcept
        : TensorStmt{TileOpKind::ALLOC_SPECIAL, tensor, {}},
          _kind{kind}, _count{count}, _reducer_op{reducer_op},
          _desc_kind{desc_kind}, _init{init} {}
    [[nodiscard]] auto kind() const noexcept { return _kind; }
    [[nodiscard]] auto tensor() const noexcept { return output(); }
    [[nodiscard]] auto count() const noexcept { return _count; }
    [[nodiscard]] auto reducer_op() const noexcept { return _reducer_op; }
    [[nodiscard]] auto desc_kind() const noexcept { return _desc_kind; }
    [[nodiscard]] auto init() const noexcept { return _init; }
    [[nodiscard]] size_t serialize(luisa::vector<char> &output_buffer) override;
    bool deserialize(char const *&input_ptr, char const *end_ptr) override;
};

// --- LoopAnnotation: T.Parallel / Persistent / serial / unroll / vectorized --
class LUISA_AST_API LoopAnnotationStmt final : public TensorStmt {
    TileLoopAnnotKind _kind{TileLoopAnnotKind::PARALLEL};// R1
    int32_t _extent{0};          // R1: loop extent / unroll count / vector width
    int32_t _coalesced_width{0}; // R1: 0 = unset

public:
    LoopAnnotationStmt() noexcept : TensorStmt{TileOpKind::LOOP_ANNOTATION} {}
    explicit LoopAnnotationStmt(TileLoopAnnotKind kind, int32_t extent = 0,
                       int32_t coalesced_width = 0) noexcept
        : TensorStmt{TileOpKind::LOOP_ANNOTATION},
          _kind{kind}, _extent{extent}, _coalesced_width{coalesced_width} {}
    [[nodiscard]] auto kind() const noexcept { return _kind; }
    [[nodiscard]] auto extent() const noexcept { return _extent; }
    [[nodiscard]] auto coalesced_width() const noexcept { return _coalesced_width; }
    [[nodiscard]] size_t serialize(luisa::vector<char> &output_buffer) override;
    bool deserialize(char const *&input_ptr, char const *end_ptr) override;
};

// --- Annotate: T.use_swizzle / annotate_layout / annotate_safe_value /
// annotate_restrict_buffers / annotate_l2_hit_ratio / annotate_min_blocks_per_sm
// The annotated buffer is the output tensor (may be null for kernel-level
// annotations such as min_blocks_per_sm).
class LUISA_AST_API AnnotateStmt final : public TensorStmt {
    TileAnnotKind _kind{TileAnnotKind::USE_SWIZZLE};// R1
    int32_t _panel_size{0};  // R1: use_swizzle panel size
    int32_t _order{0};       // R1: use_swizzle 0=row, 1=column, 2=mlx
    int32_t _enable{1};      // R1: use_swizzle 0|1
    int32_t _value{0};       // R1: min_blocks_per_sm
    float _hit_ratio{0.0f};  // R1: l2_hit_ratio (serialized as raw float bits)
    const LiteralExpr *_safe_value{nullptr};// R2 (borrowed): annotate_safe_value

public:
    AnnotateStmt() noexcept : TensorStmt{TileOpKind::ANNOTATE} {}
    explicit AnnotateStmt(TileAnnotKind kind, TensorExpr *tensor = nullptr,
                 int32_t panel_size = 0, int32_t order = 0, int32_t enable = 1,
                 int32_t value = 0, float hit_ratio = 0.0f,
                 const LiteralExpr *safe_value = nullptr) noexcept
        : TensorStmt{TileOpKind::ANNOTATE, tensor, {}},
          _kind{kind}, _panel_size{panel_size}, _order{order}, _enable{enable},
          _value{value}, _hit_ratio{hit_ratio}, _safe_value{safe_value} {}
    [[nodiscard]] auto kind() const noexcept { return _kind; }
    [[nodiscard]] auto tensor() const noexcept { return output(); }
    [[nodiscard]] auto panel_size() const noexcept { return _panel_size; }
    [[nodiscard]] auto order() const noexcept { return _order; }
    [[nodiscard]] auto enable() const noexcept { return _enable; }
    [[nodiscard]] auto value() const noexcept { return _value; }
    [[nodiscard]] auto hit_ratio() const noexcept { return _hit_ratio; }
    [[nodiscard]] auto safe_value() const noexcept { return _safe_value; }
    [[nodiscard]] size_t serialize(luisa::vector<char> &output_buffer) override;
    bool deserialize(char const *&input_ptr, char const *end_ptr) override;
};

// --- Dynamic / Symbolic: T.dynamic(name, dtype) / T.symbolic(...) -----------
// Host-side dynamic-shape markers (R1 name + R1 dtype tag).
class LUISA_AST_API DynamicStmt final : public TensorStmt {
    luisa::string _name;                       // R1
    TensorElementType _dtype{TensorElementType::I32};// R1

public:
    DynamicStmt() noexcept : TensorStmt{TileOpKind::DYNAMIC} {}
    explicit DynamicStmt(luisa::string name, TensorElementType dtype = TensorElementType::I32) noexcept
        : TensorStmt{TileOpKind::DYNAMIC}, _name{std::move(name)}, _dtype{dtype} {}
    [[nodiscard]] auto name() const noexcept { return luisa::string_view{_name}; }
    [[nodiscard]] auto dtype() const noexcept { return _dtype; }
    [[nodiscard]] size_t serialize(luisa::vector<char> &output_buffer) override;
    bool deserialize(char const *&input_ptr, char const *end_ptr) override;
};

class LUISA_AST_API SymbolicStmt final : public TensorStmt {
    luisa::string _name;                       // R1
    TensorElementType _dtype{TensorElementType::I32};// R1

public:
    SymbolicStmt() noexcept : TensorStmt{TileOpKind::SYMBOLIC} {}
    explicit SymbolicStmt(luisa::string name, TensorElementType dtype = TensorElementType::I32) noexcept
        : TensorStmt{TileOpKind::SYMBOLIC}, _name{std::move(name)}, _dtype{dtype} {}
    [[nodiscard]] auto name() const noexcept { return luisa::string_view{_name}; }
    [[nodiscard]] auto dtype() const noexcept { return _dtype; }
    [[nodiscard]] size_t serialize(luisa::vector<char> &output_buffer) override;
    bool deserialize(char const *&input_ptr, char const *end_ptr) override;
};

// --- Inline / MetaClass: T.inline(func) / T.meta_class(cls) -----------------
// Host-side Python machinery markers; only the marker text is stored (R1).
class LUISA_AST_API InlineStmt final : public TensorStmt {
    luisa::string _message;// R1

public:
    InlineStmt() noexcept : TensorStmt{TileOpKind::INLINE} {}
    explicit InlineStmt(luisa::string message) noexcept
        : TensorStmt{TileOpKind::INLINE}, _message{std::move(message)} {}
    [[nodiscard]] auto message() const noexcept { return luisa::string_view{_message}; }
    [[nodiscard]] size_t serialize(luisa::vector<char> &output_buffer) override;
    bool deserialize(char const *&input_ptr, char const *end_ptr) override;
};

class LUISA_AST_API MetaClassStmt final : public TensorStmt {
    luisa::string _message;// R1

public:
    MetaClassStmt() noexcept : TensorStmt{TileOpKind::META_CLASS} {}
    explicit MetaClassStmt(luisa::string message) noexcept
        : TensorStmt{TileOpKind::META_CLASS}, _message{std::move(message)} {}
    [[nodiscard]] auto message() const noexcept { return luisa::string_view{_message}; }
    [[nodiscard]] size_t serialize(luisa::vector<char> &output_buffer) override;
    bool deserialize(char const *&input_ptr, char const *end_ptr) override;
};

// --- AccessPtr: T.access_ptr(base, access_type, ...) -------------------------
// Device pointer of `base`; access_type is 1=r, 2=w, 3=rw (R1).
class LUISA_AST_API AccessPtrStmt final : public TensorStmt {
    int32_t _access_type{1};      // R1: 1=r, 2=w, 3=rw
    int32_t _offset{0};           // R1: element offset
    int32_t _extent{-1};          // R1: extent (-1 = unset)
    int32_t _ignore_last_ndim{0}; // R1

public:
    AccessPtrStmt() noexcept : TensorStmt{TileOpKind::ACCESS_PTR} {}
    explicit AccessPtrStmt(TensorExpr *base, int32_t access_type = 1, int32_t offset = 0,
                  int32_t extent = -1, int32_t ignore_last_ndim = 0) noexcept
        : TensorStmt{TileOpKind::ACCESS_PTR, nullptr, {base}},
          _access_type{access_type}, _offset{offset}, _extent{extent},
          _ignore_last_ndim{ignore_last_ndim} {}
    [[nodiscard]] auto base() const noexcept { return inputs().size() > 0 ? inputs()[0] : nullptr; }
    [[nodiscard]] auto access_type() const noexcept { return _access_type; }
    [[nodiscard]] auto offset() const noexcept { return _offset; }
    [[nodiscard]] auto extent() const noexcept { return _extent; }
    [[nodiscard]] auto ignore_last_ndim() const noexcept { return _ignore_last_ndim; }
    [[nodiscard]] size_t serialize(luisa::vector<char> &output_buffer) override;
    bool deserialize(char const *&input_ptr, char const *end_ptr) override;
};

// --- IndexToCoordinates: T.index_to_coordinates(index, shape) ----------------
// Flatten a linear index into per-axis coordinates; index is an R2 literal or
// R3 runtime scalar, shape is the R1 extent list.
class LUISA_AST_API IndexToCoordinatesStmt final : public TensorStmt {
    const RefExpr *_index_ref{nullptr};            // R3 (borrowed, non-serializable)
    const LiteralExpr *_index_literal{nullptr};    // R2 (borrowed)
    luisa::fixed_vector<int32_t, 4> _shape;        // R1: per-axis extents

public:
    IndexToCoordinatesStmt() noexcept : TensorStmt{TileOpKind::INDEX_TO_COORDINATES} {}
    IndexToCoordinatesStmt(const RefExpr *index, luisa::fixed_vector<int32_t, 4> shape) noexcept
        : TensorStmt{TileOpKind::INDEX_TO_COORDINATES, nullptr, {}},
          _index_ref{index}, _shape{std::move(shape)} {}
    IndexToCoordinatesStmt(const LiteralExpr *index, luisa::fixed_vector<int32_t, 4> shape) noexcept
        : TensorStmt{TileOpKind::INDEX_TO_COORDINATES, nullptr, {}},
          _index_literal{index}, _shape{std::move(shape)} {}
    [[nodiscard]] auto index_ref() const noexcept { return _index_ref; }
    [[nodiscard]] auto index_literal() const noexcept { return _index_literal; }
    [[nodiscard]] auto shape() const noexcept { return luisa::span<const int32_t>{_shape.data(), _shape.size()}; }
    [[nodiscard]] size_t serialize(luisa::vector<char> &output_buffer) override;
    bool deserialize(char const *&input_ptr, char const *end_ptr) override;
};

// ---------------------------------------------------------------------------
// 6. Control-flow / index binder statements
// ---------------------------------------------------------------------------

/// T.Kernel(gx, threads) — grid loop is host-side; bx is the yielded builtin
/// Variable{Tag::BLOCK_ID, uid=0} (R3, not serializable).
class LUISA_AST_API Kernel1DStmt final : public TensorStmt {
    int32_t _gx{};              // R1
    int32_t _threads{};         // R1
    const RefExpr *_bx{nullptr};// R3 (borrowed)

public:
    Kernel1DStmt() noexcept : TensorStmt{TileOpKind::KERNEL_1D} {}
    Kernel1DStmt(int32_t gx, int32_t threads, const RefExpr *bx = nullptr) noexcept
        : TensorStmt{TileOpKind::KERNEL_1D}, _gx{gx}, _threads{threads}, _bx{bx} {}
    [[nodiscard]] auto gx() const noexcept { return _gx; }
    [[nodiscard]] auto threads() const noexcept { return _threads; }
    [[nodiscard]] auto bx() const noexcept { return _bx; }
    [[nodiscard]] size_t serialize(luisa::vector<char> &output_buffer) override;
    bool deserialize(char const *&input_ptr, char const *end_ptr) override;
};

/// T.Kernel(gx, gy, threads).
class LUISA_AST_API Kernel2DStmt final : public TensorStmt {
    int32_t _gx{};              // R1
    int32_t _gy{};              // R1
    int32_t _threads{};         // R1
    const RefExpr *_bx{nullptr};// R3 (borrowed)
    const RefExpr *_by{nullptr};// R3 (borrowed)

public:
    Kernel2DStmt() noexcept : TensorStmt{TileOpKind::KERNEL_2D} {}
    Kernel2DStmt(int32_t gx, int32_t gy, int32_t threads,
                 const RefExpr *bx = nullptr, const RefExpr *by = nullptr) noexcept
        : TensorStmt{TileOpKind::KERNEL_2D}, _gx{gx}, _gy{gy}, _threads{threads},
          _bx{bx}, _by{by} {}
    [[nodiscard]] auto gx() const noexcept { return _gx; }
    [[nodiscard]] auto gy() const noexcept { return _gy; }
    [[nodiscard]] auto threads() const noexcept { return _threads; }
    [[nodiscard]] auto bx() const noexcept { return _bx; }
    [[nodiscard]] auto by() const noexcept { return _by; }
    [[nodiscard]] size_t serialize(luisa::vector<char> &output_buffer) override;
    bool deserialize(char const *&input_ptr, char const *end_ptr) override;
};

/// T.Pipelined(count, stages) — software pipeline metadata (R1); k is the
/// yielded runtime loop variable (R3, not serializable).
class LUISA_AST_API PipelinedStmt final : public TensorStmt {
    int32_t _count{};          // R1
    int32_t _stages{};         // R1
    const RefExpr *_k{nullptr};// R3 (borrowed)

public:
    PipelinedStmt() noexcept : TensorStmt{TileOpKind::PIPELINED} {}
    PipelinedStmt(int32_t count, int32_t stages, const RefExpr *k = nullptr) noexcept
        : TensorStmt{TileOpKind::PIPELINED}, _count{count}, _stages{stages}, _k{k} {}
    [[nodiscard]] auto count() const noexcept { return _count; }
    [[nodiscard]] auto stages() const noexcept { return _stages; }
    [[nodiscard]] auto k() const noexcept { return _k; }
    [[nodiscard]] size_t serialize(luisa::vector<char> &output_buffer) override;
    bool deserialize(char const *&input_ptr, char const *end_ptr) override;
};

}// namespace luisa::compute

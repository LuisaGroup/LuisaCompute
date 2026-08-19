/*
tile_to_kernel.cpp — Tile IR (TensorStmt) → regular Luisa GPU kernel: lowering plan
=============================================================================
This file is the *plan* for the future lowering pass that consumes the tile
IR produced by <luisa/ast/tensor.h> (a flat list of TensorStmt nodes traced
into luisa::compute::detail::TileFunctionBuilder by the tile DSL of
<luisa/dsl/tensor.h>) and emits a REGULAR Luisa compute kernel
(Kernel1D/2D/3D + set_block_size + .dispatch(...)) that implements the same
tile program on every backend.

Sources studied for this plan:
  - D:/tilelang (the reference TileLang project) and D:/tilelang/simd_to_simt.md
    (how TileLang lowers tile / SIMD-style programs to a SIMT thread group)
  - include/luisa/ast/tensor.h (every TileOpKind + TensorStmt sub-class)
  - include/luisa/ast/tile_function_builder.h (the tile IR container)
  - include/luisa/dsl/tensor.h (the tile-DSL tracing stub we lower from)
  - .agents/skills/lc_dsl/SKILL.md + include/luisa/dsl/*.h (the Luisa DSL
    primitives the plan emits: block size, thread ids, shared memory,
    atomics, warp intrinsics, async_copy + mbarrier, cooperative vectors)
  - D:/tilelang/src/transform/{lower_tile_op,loop_partition,lower_thread_allreduce,
    thread_storage_sync,inject_pipeline,lower_shared_barrier}.cc,
    D:/tilelang/src/op/{parallel,reduce,atomic_add,copy,fill,scan,gemm}.cc,
    D:/tilelang/src/tl_templates/cuda/{reduce,atomic,scan,copy,barrier}.h

---------------------------------------------------------------------------
0. What "regular GPU kernel" means in Luisa DSL
---------------------------------------------------------------------------
A normal Luisa kernel is a C++ lambda traced by FunctionBuilder into a real
AST and compiled by a backend (cuda/dx/vk/metal/cpu):

  Kernel2D k = [](BufferVar<float> A, BufferVar<float> B,
                  BufferVar<float> C, UInt N) noexcept {
      set_block_size(128u, 1u, 1u);      // blockDim (compile-time)
      set_warp_size(32u);                // warp/wave size
      UInt bx = block_id().x;            // blockIdx
      UInt tx = thread_id().x;           // threadIdx.x
      // ... body using Shared<T>, atomics, warp intrinsics, $if/$for ...
  };
  auto sh = device.compile(k);
  stream << sh(A, B, C, N).dispatch(gx) << synchronize();  // gridDim (runtime)

The lowering below therefore never emits a second kernel and never relies on
a TileLang-style layout inference: it performs the SIMD→SIMT step explicitly
(partition each tile loop over thread_id, guard with predicates, insert
barriers/atomics where TileLang's passes would inject them).

---------------------------------------------------------------------------
1. Shared lowering skeleton (applies to every op)
---------------------------------------------------------------------------
1.1 Launch configuration.
  KERNEL_1D (gx, threads)  -> Kernel1D, set_block_size(threads), dispatch(gx)
  KERNEL_2D (gx, gy, thrs) -> Kernel2D, set_block_size(threads), dispatch(gx, gy)
  PIPELINED  (count, stages) -> no launch change; drives the async pipeline
      structure of the copies inside its body (see 2.14).
  gridDim is data-dependent (ceildiv extents may be runtime): pass them as
  UInt kernel args and call sh(...).dispatch(gx, gy) with the same values.

1.2 Memory mapping (TensorScope / TensorExpr layout → DSL objects).
  Global   -> Buffer<T> kernel argument (device Buffer bound by the caller);
              linear index = dot(offset + tile_coord, row_stride)
              (row-major strides derived from TensorExpr::dims()).
  Shared   -> Shared<T> s{product(extent)} declared at kernel top
              (Luisa manages per-block allocation; no dyn-smem byte math).
  Fragment -> per-thread register tile: a local Var<T> / fixed C++ array of
              Vars (unrolled); TileLang's "fragment layout" lane/replicate
              mapping is reproduced by indexing with warp_lane_id().
  Reshape/View reuse the same handle (same Buffer/Shared storage, different
  dims/dtype), so they are compile-time metadata only.

1.3 Tile → thread partition (the core SIMD→SIMT step).
  Every whole-tile parallel loop (COPY/FILL/CLEAR/BINARY/STORE/TRANSPOSE/
  REDUCE/... body, and T.Parallel / T.Pipelined iterations) is emitted as the
  explicit TileLang PartitionLoop pattern:

    UInt total = product(extent);            // logical tile element count
    UInt iters = ceildiv(total, block_size().x);
    $for (i, 0u, iters) {
        UInt idx = i * block_size().x + thread_id().x;   // linear lane
        $if (idx < total) {
            // decompose idx -> per-axis coords (row-major),
            // emit the elementwise tile body guarded by the predicate
        };
    };

  For 2D tiles prefer the 2D partition:
    $for (r, thread_id().y, rows, block_size().y)
      $for (c, thread_id().x, cols, block_size().x) { body(r, c) };
  Vectorized (coalesced) variants iterate chunks of
  block_size().x * vector_width and use float4/half2 stores (see
  TileLang vectorize_loop + SelectMinPaddingVectorSize in op/parallel.cc).

1.4 Barrier discipline (mirrors TileLang ThreadSync / LoopUnswitching).
  - sync_block() = __syncthreads: emitted after shared stores before any
    cross-thread read of that shared tile, and at pipeline stage boundaries.
  - NEVER place sync_block()/mbarrier waits inside a thread_id()-divergent
    branch ($if guarded by thread_id()/lane predicates).
  - Cross-thread reduction inside one block = shared workspace + sync_block
    (tl::AllReduce pattern); cross-block = hardware atomics (no mutex).

---------------------------------------------------------------------------
2. Per-op lowering plan (one entry per TileOpKind)
---------------------------------------------------------------------------

--- 2.1 ALLOC (T.empty / T.alloc_shared / T.alloc_fragment) ----------------
  TileLang: allocation of a global/shared/fragment buffer; layout inferred
  later. Luisa: emit nothing at the statement level — allocation becomes the
  declaration in the skeleton:
    Global:  the TensorExpr handle is a kernel Buffer<T> argument;
    Shared:  Shared<T> s{extent} at kernel top;
    Fragment: per-thread register array (see 1.2).
  The output TensorExpr's dims/dtype/scope are the declaration parameters.

--- 2.2 CLEAR (T.clear(t)) -------------------------------------------------
  TileLang: memset / parallel store loop with predicate. Luisa:
    $for over partition of t.extent: t.write(idx, zero<T>())
  (T.zero literal of the tensor dtype; fragment scope = local var loop.)

--- 2.3 COPY (T.copy(src, dst)) --------------------------------------------
  TileLang copy.cc: chooses global→shared (cp.async path, or vectorized
  load/store), shared→fragment (ldmatrix / plain), shared→shared, etc.
  Luisa:
    global↔shared: vectorized (float4/half2) ld/st loops over the partition
      (1.3); on CUDA optionally async_copy for global→shared (2.14).
    shared→fragment / fragment→shared: plain element loops; keep the
      fragment side indexed by warp_lane_id() where the layout demands.
    shared→shared: temp-free element copy with NO barrier needed inside a
      single pass (each thread copies distinct elements).
    global→global: element/vectorized copy.
  Bounds: when the source/dest tile is a slice (offset/extent < dims), add
  the global base offset and guard with $if like TileLang
  LegalizeSafeMemoryAccess.

--- 2.4 GEMM (T.gemm(a, b, c)) and the GEMM family -------------------------
  TileLang: MMA/wgmma/tcgen05 tensor-core path when available, else SIMT
  fallback. Luisa has no tensor-core DSL except Vulkan cooperative vectors,
  so the REGULAR-kernel implementation is a software shared-memory GEMM
  (the test_warp.cpp / test_softmax.cpp pattern):
    1. copy A/B tiles (or slices) into Shared (2.3);
    2. sync_block();
    3. per-thread micro-tile: each thread owns an (tm, tn) fragment of C
       (row/col partition over the C tile, e.g. 4x4 per thread with 128
       threads → 32x16 C tile), loop over K in chunks:
           acc += A_sh[r][k] * B_sh[k][c];   // unrolled inner K
    4. optionally accumulate in Float for F16 inputs (AccType rule from
       tl_templates/cuda/reduce.h);
    5. T.clear(c) (clear_accum) is emitted as CLEAR first (2.2);
    6. copy fragment C back to global C tile (2.3).
  Knobs:
    trans_a/trans_b -> swap indexing in step 3;
    GemmWarpPolicy   -> partition the C tile across warps: Square = both dims
      split, FullRow = all warps along M, FullCol = along N (mirrors
      TileLang GemmWarpPolicy);
    k_pack           -> unroll factor for the inner K loop;
    mbar (Blackwell mbarrier input) -> ignore in the SIMT fallback (the
      wait is implied by sync_block) or keep for the async path (2.14).
  WGGMA_GEMM / TCGEN05_GEMM / TCGEN05_GEMM_BLOCKSCALED: same SIMT fallback
  (hardware WGMMA/TCGEN05 require new DSL builtins; see section 4 gap list).
  GEMM_SP / WGGMA_GEMM_SP / TCGEN05_GEMM_SP: TileLang sparse GEMM works on
  the compressed A_sparse + metadata E (2:4 sparsity). Regular-kernel plan:
    - dense SIMT fallback: decompress A_sparse via E (or treat as dense and
      let the caller pre-decompress); the metadata E layout determines the
      K-groups of 4 with 2 non-zeros;
    - otherwise identical to GEMM with A replaced by the decompressed tile.

--- 2.5 REDUCE_SUM / REDUCE (T.reduce_sum / T.reduce family) ---------------
  TileLang ReduceLowerer: thread-local reduction, then tl::AllReduce
  (offset>=32 → shared + barrier, offset<32 → shfl_xor), then guarded
  write-back. Luisa (one kernel):
    1. per-thread partial: each thread reduces its owned elements of the
       reduce dim (partitioned loop);
    2. warp level: warp_active_sum/max/min/bit_and/bit_or/bit_xor over the
       partial (this is TileLang's shfl_xor butterfly, expressed as the
       built-in warp reduce); lane 0 accumulates into a Shared<T> slot;
    3. block level: sync_block(), then reduce the num_warps Shared values
       (warp 0 tree or single-thread loop), or continue the XOR-butterfly
       with shared workspace + sync_block for >32-thread reduce groups;
    4. write the reduced scalar to out (or accumulate with atomics when the
       reduce is cross-block / the output is global, TileReduceOp drives the
       op: SUM/MAX/MIN/ABS_SUM/ABS_MAX/BIT_AND/BIT_OR/BIT_XOR; ABS_* reduce
       abs(v) first);
    dim -> which axis is partitioned; clear=0 -> accumulate into out
    (read-modify-write) instead of overwrite; batch -> run several
    independent channels sharing one barrier/workspace
    (tl::AllReduce::run_batch, workspace_stride = block size).
  REDUCE_SUM is REDUCE with op=SUM, dim given, clear=1.

--- 2.6 FINALIZE_REDUCER (T.finalize_reducer(reducer, batch)) --------------
  TileLang finalize_reducer: after T.reduce with clear=0 accumulated into a
  per-block reducer workspace, finalize converts the per-thread partials
  into the final value. Luisa: emit the block-level tail of 2.5 — the
  Shared workspace cross-thread reduction + write-back, honoring batch
  (batch independent channels in the same sync_block()).

--- 2.7 WARP_REDUCE (T.warp_reduce_sum/max/min/bitand/bitor) ---------------
  TileLang: warp_reduce in tl_templates/cuda/reduce.h (__reduce_*_sync or
  shfl_xor chain). Luisa: single expression
    warp_active_sum/max/min/bit_and/bit_or(value)
  (value is a fragment tensor: reduce over the fragment lanes, i.e. lanes of
  warp_lane_id()); result lands on every lane; caller keeps lane 0 when a
  single scalar is wanted.

--- 2.8 CUMSUM / CUMMAX (T.cumsum / T.cummax(src, dst, dim, reverse)) ------
  TileLang scan.cc + tl_templates/cuda/scan.h: Hillis-Steele / Blelloch scan
  with warp shfl + shared stages. Luisa:
    1. thread-local scan over the elements owned by each thread;
    2. warp scan: warp_prefix_sum(value) (exclusive) or warp_prefix_product;
       CUMMAX has no built-in warp scan → emulate with
       warp_active_max + shfl-style butterfly using warp_read_lane(value,
       lane ^ offset) (the same chain TileLang emits for max);
    3. per-warp totals to Shared, sync_block(), block scan of warp totals,
       sync_block(), then add the block-exclusive offset back to each
       element;
    4. reverse=1 → scan over the reversed index space (or scan and flip).
  dim selects the scanned axis of dst (a tile may be 1D/2D).

--- 2.9 PRINT (T.print(t, "msg")) ------------------------------------------
  TileLang: printf of the tile with __syncthreads + guard. Luisa: no tile
  print builtin; emit a per-element debug write guarded to a small set of
  elements (e.g. $if (thread_id().x < min(extent, 8u)) { printf-like via
  debug/print CallOp if available }); the message string is carried by
  msg() and becomes the format tag. Lowest priority; leave a hook for a
  future print builtin.

--- 2.10 STORE / BINARY / MAX / RSQRT (whole-tile elementwise ops) ---------
  TileLang lowers these through T.Parallel-style loops (op/parallel.cc:
  PartitionLoop + vectorize + predicate). Luisa: partitioned element loop
  (1.3):
    STORE op=0:  dst = rhs  (rhs_tensor / rhs_literal / rhs_ref);
    STORE op=1:  dst *= rhs (row-broadcast scale: rhs indexed by row only);
    BINARY:      temp = lhs OP rhs (OP = BinaryOp: +,-,*,/,min,max,...),
                 result is a fragment temp consumed by a following STORE;
    MAX:         temp = max(a, b_literal)  (clamp-to-min, e.g. 1e-12f);
    RSQRT:       temp = rsqrt(a)  → Luisa 1.f / sqrt(a) or fast rsqrt if
                 the backend exposes one (CUDA __frsqrt_rn via math call);
    vectorization: when extent/block allows, emit float4/half2 chunks.
  CEILDIV (host helper, no tensors): constant-fold (a + b - 1) / b on the
  host while lowering; emits no kernel code.

--- 2.11 ATOMIC (T.atomic_add/max/min/addx2/addx4/load/or/store) -----------
  TileLang atomic_add.cc/atomic_reduce.h → atomicAdd/atomicMax/atomicMin/
  atomicCAS loop. Luisa:
    add/max/min/or  → buf.atomic(i).fetch_add/fetch_max/fetch_min/
                      fetch_or(value)  (dst = output tensor, READ+WRITE);
    addx2/addx4     → packed 2/4-wide add: implement as element-wise
                      fetch_add over the vector components (or as<uint2>
                      CAS loop when a single 64-bit atomic is desired);
    load            → volatile_read / atomic load emulation;
    store           → volatile_write (or CAS loop for return-prev);
    return_prev=1   → capture the fetch_* return value into the temp;
    memory_order    → relaxed default; acq_rel/seq_cst map to volatile +
                      fence (Luisa exposes fence via builtin calls where
                      available);
    use_tma         → Blackwell TMA atomics: no DSL support → fall back to
                      the plain atomic path (gap list).
  The value may be a tensor (inputs[0]), a literal (value_literal) or a
  runtime scalar (value_ref).

--- 2.12 CLAMP / DP4A / LOOP_BREAK / ANY_OF / ALL_OF -----------------------
  CLAMP: partitioned loop dst = clamp(dst, lo, hi) (lo/hi literal or ref).
  DP4A: 4-element signed int8 dot into int32 C: per-element partition over
    the 4-wide groups: c += a[k]*b[k] (unrolled 4), or a single
    int8→int32 dot via Luisa vector ops if the backend has a dot intrinsic.
  LOOP_BREAK: break_ inside the innermost lowered $for/$loop (TileLang
    loop_break → break).
  ANY_OF / ALL_OF: tile → scalar boolean. Partition the tile across threads,
    reduce with warp_active_any/all + Shared/sync_block for the block part,
    lane 0 writes the boolean temp:
      any_of: local = any(elem != 0) then block-any;
      all_of: local = all(elem != 0) then block-all.

--- 2.13 SYNC / BARRIER / MBARRIER / WARP_VOTE / SHUFFLE / SYNC_THREADS_VOTE
  SYNC:
    THREADS → sync_block(); named variant (barrier_id/arrive_count) → not
      exposed in Luisa; approximate with sync_block() (gap).
    WARP    → no-op (warp ops are implicitly synchronized) or
      warp_active_all(true) as a cheap barrier.
    GRID/GLOBAL → cross-block sync is NOT a barrier in Luisa; only
      meaningful for grid-sync kernels → document as unsupported (gap).
  BARRIER (mbarrier arrive/wait/named arrive): mbarrier_init(bar, count)
    once (thread 0), mbarrier_arrive_expect_tx / mbarrier_try_wait_parity
    for arrive/wait; named_barrier_arrive (barrier_id + thread_count) has
    no Luisa equivalent → sync_block() approximation (gap).
  MBARRIER (arrive/arrive_expect_tx/expect_tx/wait_parity):
    mbarrier_arrive_expect_tx(bar, tx) / mbarrier_try_wait_parity(bar,
    phase); tx/parity/cta_id drive the arguments; cta_id (peer CTA) is
    cluster-only → unsupported in regular kernels (gap).
  WARP_VOTE (activemask/ballot/any/all/match):
    activemask   → warp_active_bit_mask(true);
    ballot       → warp_active_bit_mask(pred);
    any_sync     → warp_active_any(pred);
    all_sync     → warp_active_all(pred);
    match_any/all → warp_active_all_equal(value) (all_equal) / first-match
      via warp_read_first_active_lane for match_any approximation;
    mask (R3)    → Luisa warp intrinsics operate on the full warp; a
      sub-warp mask needs a manual shuffle loop (gap).
  SHUFFLE (shfl_sync/xor/up/down/elect):
    shfl_sync    → warp_read_lane(value, src_lane);
    shfl_xor     → warp_read_lane(value, lane ^ delta);
    shfl_up/down → warp_read_lane(value, lane -/+ delta) guarded by
      $if (lane >= delta) for up;
    shuffle_elect→ lane 0 of the group; emulate with
      warp_read_first_active_lane / warp_first_active_lane.
  SYNC_THREADS_VOTE (count/and/or): block-wide vote of pred:
    count → sync_block(); Shared<uint> counter; thread 0 sums? — better:
      per-warp warp_active_count_bits(pred), warp 0 writes, then
      sync_block() + tree sum over the Shared warp-counts;
    and/or → sync_block() + Shared vote: every thread writes pred, thread 0
      reduces (or all threads reduce via shared + butterfly).

--- 2.14 ASYNC_COPY / COPY_CLUSTER / TMA_COPY / TMA_GATHER4 / TMA_SCATTER4 /
       IM2COL / TRANSPOSE / FILL -------------------------------------------
  ASYNC_COPY: global→shared cp.async: async_copy(scope=shared, dst=Shared
    element lvalue, src=Buffer.device_address()+byte_off, elem_bytes=4/8/16,
    num, stride, event) + pipeline_commit()/pipeline_wait_prior(stages)
    around the pipelined loop (2.16); the fragment/vectorized fallback is
    plain loads. coalesced_width drives the vector chunk width.
  COPY_CLUSTER (TMA multicast / SM-to-SM): cluster_mask/dst_block are
    cluster features; regular-kernel fallback = plain COPY (2.3) or
    async_copy when dst is shared (gap: no cluster API in Luisa).
  TMA_COPY: TMA global↔shared with optional mbarrier; regular fallback =
    async_copy + mbarrier (leader_scope_threads = the elected thread count;
    eviction_policy ignored in SIMT). Requires CUDA/DX support of
    async_copy (check backend coverage; else plain vectorized copy).
  TMA_GATHER4 / TMA_SCATTER4: 2D gather/scatter of 4 rows; SIMT fallback =
    partitioned element loop over the 4 (row, col) pairs, col from the R3
    runtime index, barrier from inputs[1]; rows are the R1 4-row list.
  IM2COL: convolution → column tile: partitioned loop over output col tile
    elements computing src offsets from kernel/stride/dilation/pad and the
    runtime nhw_step/c_step; bounds-guarded like TileLang
    LegalizeSafeMemoryAccess.
  TRANSPOSE: shared-memory transpose (dst[j,i] = src[i,j]): partitioned
    load of the src tile into Shared (or register tile), sync_block(),
    partitioned transposed store; optionally pad the shared row to avoid
    bank conflicts (the classic stride trick).
  FILL: partitioned store of a constant (literal) or runtime scalar (ref)
    into the whole buf tile (2.2 with a non-zero value).

--- 2.15 RESHAPE / VIEW ----------------------------------------------------
  Purely metadata: the output TensorExpr shares the handle and the same
  storage with a new dims/dtype. Lowering emits nothing (the consuming ops
  use the output's dims/offset); a VIEW changing dtype may require
  as<T>/byte reinterpretation at access time (element size change → the
  linear index math must switch to byte offsets).

--- 2.16 PIPELINED / LOOP_ANNOTATION (T.Parallel / Persistent / unroll /
       vectorized) ----------------------------------------------------------
  PIPELINED (count, stages): the traced IR holds ONE representative body
  (copy/copy/gemm...). Lowering expands it into a software-pipelined loop:
    for (int k = 0; k < count; ++k) {   // multi-buffered Shared
        pipeline_wait_prior(stages - 1); // wait until stage slot free
        async_copy(stage[k % stages]);   // prefetch next global tile
        pipeline_commit();
        compute on stage[(k - 1) % stages]; // use the previously copied tile
    }
    drain: pipeline_wait_prior(0) + last compute;
  with per-stage Shared<T> buffers (num_stages copies of each shared tile)
  and mbarrier_init/arrive_expect_tx/try_wait_parity where async_copy does
  not carry an implicit completion event. Fallback when async_copy is
  unavailable: plain copy + sync_block at each stage (correct, no overlap).
  LOOP_ANNOTATION:
    PARALLEL    → the 1.3 partition loop;
    PERSISTENT  → persistent kernel: outer $for over
      (dispatch_id().x, dispatch_size().x, dispatch_size().x) so one block
      processes many tiles (grid-stride loop);
    SERIAL      → plain $for with no thread partition (all threads execute
      the same serial body);
    UNROLL      → emit a C++ compile-time loop (constant extent) so the
      FunctionBuilder/AST unrolls (dynamic_range with const bounds);
    VECTORIZED  → chunk the loop by coalesced_width (1.3) and use vector
      loads/stores.

--- 2.17 ALLOC_SPECIAL / DYNAMIC / SYMBOLIC / INLINE / META_CLASS /
       ACCESS_PTR / INDEX_TO_COORDINATES / ANNOTATE ------------------------
  ALLOC_SPECIAL (alloc_var/local/global/barrier/reducer/tmem/descriptor/
    cluster_barrier):
      VAR          → Var<T> local (init literal);
      LOCAL        → Local<T> / local array;
      GLOBAL       → no alloc (caller-owned Buffer arg);
      BARRIER      → Shared<ulong> mbarrier slot (mbarrier_init);
      REDUCER      → Shared<T> reducer workspace + op tag
        (reducer_op: 0=sum,1=max,2=min) consumed by REDUCE/FINALIZE_REDUCER;
      TMEM/DESCRIPTOR/CLUSTER_BARRIER → tensor-core/cluster features with no
        Luisa equivalent → fallback to Shared/plain buffers or reject
        (gap list).
  DYNAMIC / SYMBOLIC (host-side shape markers): no kernel code; recorded as
    metadata so extents can be runtime UInt args.
  INLINE / META_CLASS (host-side markers): ignored by the device lowering.
  ACCESS_PTR (device pointer of base): Buffer.device_address() +
    element_offset (access_type r/w/rw becomes the later access mode);
    the produced pointer is passed to TMA-ish ops or raw copy sources.
  INDEX_TO_COORDINATES: linear index (literal/ref) → per-axis coords:
    emitted inline as div/mod chains over shape (row-major);
  ANNOTATE (use_swizzle/layout/safe_value/restrict_buffers/l2_hit_ratio/
    min_blocks_per_sm): backend hints; in a regular kernel most are no-ops
    (safe_value → default init of shared/fragment, min_blocks_per_sm →
    launch-bound hint if the backend exposes it; others recorded in the
    Shader compile options).

--- 2.18 FAST_RCP / IEEE_MATH / PACKED_MATH / FAST_MATH (math intrinsics) ---
  FAST_RCP (T.fast_rcp(a)): fast approximate reciprocal → 1.f / a (the
    backend's fast division path) or rsqrt-style approximation when the
    target exposes a fast rcp intrinsic; partitioned elementwise loop (1.3).
  IEEE_MATH (T.ieee_add/sub/mul/fmaf/frcp/fsqrt/frsqrt/fdiv, rounding mode
    0=rn,1=rz,2=ru,3=rd): IEEE-exact ops. Luisa:
      add/sub/mul → operator +,-,*;
      fmaf        → fma(a, b, c);
      frcp        → 1.f / a;
      fsqrt       → sqrt(a);
      frsqrt      → rsqrt(a);
      fdiv        → a / b;
    rounding modes rz/ru/rd are NOT exposed by the Luisa DSL → default to
    round-to-nearest (rn) and document the loss (gap list).
  PACKED_MATH (T.add2/sub2/mul2/fma2/max2/min2/abs2): packed 2-wide math on
    half2/float2 pairs (CUDA __hadd2/__hfma2 etc.). Luisa: use the native
    vector types — make_float2/make_half2 with component-wise +,-,*,fma,
    max/min/abs operators, or bit-cast the packed u32 pair with as<float2>/
    as<half2> when the value arrives as a raw uint (TileLang stores packed
    pairs as uint); the vector operators compile to the packed HW ops where
    the backend supports them.
  FAST_MATH (T.__exp/__exp10/__log/__log2/__log10/__sin/__cos/__tan): fast
    approximate intrinsics (CUDA __expf/__logf/__sinf/...). Luisa exposes
    the precise exp/exp2/exp10/log/log2/log10/sin/cos/tan; the fast variants
    are backend-dependent → emit the standard functions first (correct,
    possibly slower) and map to the fast intrinsic only when a backend
    CallOp exists (gap/optimization note).

--- 2.19 Math expression lowering (CallOp catalog → Luisa DSL) ---------------
  The tile IR is a flat op list, but every tile-op body is ultimately a
  scalar/vector/matrix *expression* over F16/F32/I32 values (TileBinaryStmt
  ops, MaxStmt, RsqrtStmt, ClampStmt, Dp4aStmt, GEMM accumulation,
  ReduceStmt/WarpReduceStmt element ops, IEEE/PACKED/FAST math, Fill/STORE
  values).  Lowering materializes those expressions with the Luisa DSL math
  functions below — one CallOp → one DSL call, emitted verbatim into the
  FunctionBuilder AST.  No lowering pass re-implements math: the backend JIT
  owns correctness; this section only documents which DSL call to emit for
  each op and the precision/type contract the emitted expression must meet.

  Direct 1:1 mapping (op.h CallOp / BinaryOp / UnaryOp → DSL builtin in
  <luisa/dsl/builtin.h> / <luisa/dsl/syntax.h>):
    Unary       PLUS/MINUS/NOT/BIT_NOT  -> +x / -x / !x / ~x
    Binary arith ADD/SUB/MUL/DIV/MOD    -> a+b / a-b / a*b / a/b / a%b
                BIT_AND/BIT_OR/BIT_XOR  -> a&b / a|b / a^b
                SHL/SHR                 -> a<<b / a>>b
                AND/OR                  -> a&&b / a||b
    Relational  LESS/GREATER/LE/GE/EQ/NE -> a<b / a>b / a<=b / a>=b /
                                           a==b / a!=b (bool result; feed
                                           into ite/select to get 0/1)
    Common      ABS/MIN/MAX/CLAMP       -> abs/min/max/clamp
                SATURATE/LERP/SMOOTHSTEP -> saturate/lerp/smoothstep
                STEP                    -> step(edge, x); note op.h STEP is
                                           (x,y): (x>=y)?1:0 → DSL step(y,x)
    Trig/hyp    ACOS/ACOSH/ASIN/ASINH   -> acos/acosh/asin/asinh
                ATAN/ATAN2/ATANH        -> atan/atan2/atanh
                COS/COSH/SIN/SINH/TAN/TANH -> cos/cosh/sin/sinh/tan/tanh
    Exp/log     EXP/EXP2/EXP10          -> exp/exp2/exp10
                LOG/LOG2/LOG10/POW      -> log/log2/log10/pow
                SQRT/RSQRT              -> sqrt/rsqrt
    Rounding    CEIL/FLOOR/FRACT/TRUNC/ROUND -> ceil/floor/fract/trunc/round
    Vector      CROSS/DOT               -> cross/dot
                LENGTH/LENGTH_SQUARED   -> length/length_squared
                NORMALIZE               -> normalize
                FACEFORWARD/REFLECT     -> faceforward/reflect
    Matrix      DETERMINANT/TRANSPOSE/INVERSE -> determinant/transpose/inverse
                OUTER_PRODUCT           -> outer_product
                MATRIX_COMPONENT_WISE_MULTIPLICATION -> component-wise *
    Per-value   REDUCE_SUM/PRODUCT/MIN/MAX -> reduce_sum/reduce_prod/
    reduce                                 reduce_min/reduce_max (register/
                                           vector reduce — NOT the tile-level
                                           REDUCE of 2.5; used when a
                                           fragment/vector value must
                                           collapse to one scalar)
    Integer     CLZ/CTZ/POPCOUNT/REVERSE -> clz/ctz/popcount/reverse
    Classify    ISINF/ISNAN             -> isinf/isnan (bool; combine with ite)
    Fused       FMA/COPYSIGN            -> fma/copysign
    Selection   SELECT                  -> select(x, y, c) / ite(c, x, y)

  Tile-op → expression lowering table (the actual emission sites):
    TileBinaryStmt(op) -> a OP b, op∈BinaryOp above (per-thread elementwise
                          per 1.3; scalar rhs from rhs_literal/rhs_ref).
    MaxStmt(a, b)      -> max(a, b_literal)   (clamp-to-min, e.g. 1e-12f)
    RsqrtStmt(a)       -> rsqrt(a)
    ClampStmt          -> clamp(dst, lo, hi)  (lo/hi literal or ref)
    Dp4aStmt           -> signed 4-lane int8 dot: unrolled c += a[k]*b[k]
                          (int32 accum; same shape as a dot-style reduction
                          if a backend offers one)
    GEMM accum (2.4)   -> fma(a_sh[r][k], b_sh[k][c], acc) per inner K;
                          optional F16→F32 accumulator promotion
    REDUCE element ops (2.5): SUM='+' / MAX=max / MIN=min /
                          ABS_SUM=abs(x)+ / ABS_MAX=abs(x) then max /
                          BIT_AND/BIT_OR/BIT_XOR = &,|,^
    WARP_REDUCE (2.7)  -> warp_active_sum/max/min/bit_and/bit_or(value)
    CUMSUM/CUMMAX (2.8)-> warp_prefix_sum/product (exclusive) or the
                          shfl-butterfly max emulation; totals via + / max
    IEEE_MATH (2.18)   -> fmaf→fma, frcp→1.f/a, fsqrt→sqrt, frsqrt→rsqrt,
                          fdiv→a/b, add/sub/mul→operator
    PACKED_MATH (2.18) -> native vector ops or as<float2>/as<half2> on the
                          packed u32 pair (component +,-,*,fma,max,min,abs)
    FAST_MATH (2.18)   -> standard exp/exp2/exp10/log/log2/log10/sin/cos/tan
                          first; backend fast intrinsic only if a CallOp
                          exists
    STORE op=1 (2.10)  -> row-broadcast scale: dst *= rhs[row] (scalar ×
                          vector; == component-wise MUL)
    FILL (2.14)        -> store literal/ref constant (zero<T> for CLEAR 2.2)

  Type/vector coverage (must hold for every emitted function):
    - scalars: F16/F32/I32 (+ I64/U64/U32/other ints inside structs)
    - vectors: 2/3/4 components for float/half/int/double; matrices NxN
      (2x2/3x3/4x4, half/float/double) for determinant/transpose/inverse/
      outer_product/component-wise-mul
    - math on `half` and `double` uses the same DSL names (type-generic
      overloads); f64 must match <cmath> at 1e-9 (test tolerance)
    - bool results (relational / isinf / isnan / ALL / ANY) flow into
      ite/select; ALL/ANY (vector→bool) map to the all()/any() DSL helpers

  Precision contract:
    - elementwise ops must match C++ <cmath> within approx_eq eps=1e-3
      (relative+absolute); doubles within 1e-9
    - exp10 may be emitted as pow(10.f, x) on backends without a native
      exp10 (tolerance 5e-2)
    - when a backend lacks an intrinsic (e.g. no native rsqrt), the emitted
      fallback must stay inside the tolerance above and be documented here

---------------------------------------------------------------------------
3. Implementation order (milestones)
---------------------------------------------------------------------------
  M1  Skeleton: walk TileFunctionBuilder::body()->statements(), emit the
      launch (KERNEL_1D/2D + set_block_size/set_warp_size), the partition
      loop helper and the memory mapping (Global/Shared/Fragment).
  M2  Elementwise ops: ALLOC, CLEAR, COPY, STORE, BINARY, MAX, RSQRT,
      CEILDIV, FILL, TRANSPOSE, CLAMP, FAST_RCP, IEEE_MATH, PACKED_MATH,
      FAST_MATH, PRINT(stub), LOOP_BREAK, DYNAMIC, SYMBOLIC,
      INDEX_TO_COORDINATES, RESHAPE/VIEW, ANNOTATE (no-ops).
      Plus the full 2.19 math-expression lowering (every BINARY/STORE/CLAMP/
      ... body is one of the DSL calls in the catalog).
      Gate for M1+M2: every emitted tile-op body compiles and runs correctly
      on each enabled backend.
  M3  Reduce/scan: REDUCE_SUM/REDUCE, FINALIZE_REDUCER, WARP_REDUCE,
      ANY_OF/ALL_OF, CUMSUM/CUMMAX via warp intrinsics + Shared + sync_block.
  M4  Atomics + votes: ATOMIC (fetch_*), DP4A, SYNC(threads), WARP_VOTE,
      SHUFFLE, SYNC_THREADS_VOTE, LOOP_ANNOTATION(PARALLEL/SERIAL/UNROLL).
  M5  GEMM family SIMT fallback (2.4) incl. trans/policy/clear_accum/SP.
  M6  Async/pipeline: ASYNC_COPY, TMA_COPY fallback, PIPELINED,
      BARRIER/MBARRIER, IM2COL, ALLOC_SPECIAL(VAR/LOCAL/BARRIER/REDUCER),
      ACCESS_PTR.
  M7  Persistent/vectorized annotations, PERSISTENT grid-stride loops,
      coalesced_width vectorization; backend-specific fast paths.

---------------------------------------------------------------------------
4. Gaps & risks (documented, not silently mis-lowered)
---------------------------------------------------------------------------
  - WGMMA / TCGEN05 / blockscaled / sparse MMA / TMEM / descriptors need
    new Luisa DSL builtins (cooperative vectors are Vulkan-only today); the
    regular-kernel fallback is a portable SIMT shared-memory GEMM.
  - TMA gather4/scatter4/copy_cluster/mbarrier-CTA, sync_grid/sync_global,
    named barriers, sub-warp masks, addx2/addx4 packed atomics, use_tma
    atomics have no (or partial) Luisa equivalents → fallback or reject.
  - mbarrier_* / async_copy backend coverage must be checked per target
    (CUDA/DX/VK); otherwise fall back to sync_block + plain copies.
  - Barrier correctness is manual: the lowering must never put a barrier in
    a thread_id()-divergent branch (TileLang LoopUnswitching rule).
  - Dtype support: F16/F32/I32/I8 lower to the matching core element Type
    (half/float/int/byte); FP8 lowers to the fp8 e4m3 element Type (literals
    are carried as raw zero bytes); the 4-bit I4/FP4 dtypes have no core
    element Type yet and are rejected by the lowering.  f64/bf16/u8/i16/i64
    need an R1 enum extension before they can lower.
  - Math intrinsics are backend-dependent: the emitted expression must match
    <cmath> within the 2.19 precision contract (1e-3, doubles 1e-9); exact
    bit-identical results across backends are NOT guaranteed and must not be
    asserted (compare with an epsilon tolerance instead).
  - IEEE rounding modes rz/ru/rd and the fast-intrinsic variants have no
    (or partial) DSL exposure — see 2.18/2.19 gap notes.

=============================================================================
*/

// ============================================================================
// Implementation — SIMD -> SIMT lowering of the tile IR (design: plan above)
// ============================================================================
//
// `tile_to_kernel` consumes a traced tile function (a TileFunctionBuilder, a
// flat list of TensorStmt nodes) and emits a REGULAR Luisa kernel: a
// FunctionBuilder of Tag::KERNEL with one Buffer<T> argument per Global
// tensor (in AllocStmt order), shared arrays and per-thread local arrays for
// Shared / Fragment tensors, set_block_size from T.Kernel, and a dispatch
// grid returned to the caller.
//
// Layout model (SIMD->SIMT):
//   * Global  tiles  -> partitioned element loops; each element is produced
//                       by exactly one thread.
//   * Shared  tiles  -> partitioned element loops + sync_block() after every
//                       statement that touches shared memory (never inside a
//                       thread-divergent branch).
//   * Fragment tiles -> replicated per-thread register arrays: every thread
//                       holds the whole tile and every fragment op runs a
//                       full (non-partitioned) element loop.  This is the
//                       "replicate" fragment layout of TileLang — simple and
//                       correct; a lane-mapped layout (warp_lane_id
//                       partitioning) is future work (plan 1.2).
//   * Value temporaries (BINARY / MAX / RSQRT outputs) are NOT materialized:
//     the lowering records an *expression evaluator* and inlines it at the
//     consuming statement (STORE / COPY / ...), so no register staging and no
//     cross-thread ordering issue arises.
//
// Global base-offset reconstruction: the tracing stub iterates ONE
// representative block (bx=by=0) and one representative pipeline step (ko=0),
// so the recorded tile offsets are all zero.  The lowering reconstructs the
// per-block / per-pipeline base from the launch grid and the tile extents
// (TileLang-style layout inference, plan 1.3):
//   * 2D kernel: axis 0 base += block_id().y * E0,
//                axis 1 base += block_id().x * E1
//   * 1D kernel: axis 0 base += block_id().x * E0 (axis 1 is assumed whole)
//   * inside a T.Pipelined body the pipeline variable advances the axis with
//     the smallest tile extent (the K axis of a GEMM tile) by ko * E;
//   * the row stride of a global tensor is reconstructed as gx*E1 (2D), E1
//     (1D) or pipeline_count*E1 when axis 1 is the pipeline axis.
//
// View identity: statement operands are *clones* of the AllocStmt output
// TensorExpr (TensorStmt owns its operands, so two statements never share a
// TensorExpr pointer).  Clones are resolved back to their allocation by the
// host-side tensor name (added to TensorExpr / AllocStmt by the tile DSL);
// name-less IR falls back to a first-layout-match heuristic.

#include <luisa/ast/tile_to_kernel.h>

#include <luisa/core/logging.h>
#include <luisa/core/stl/format.h>
#include <luisa/core/stl/unordered_map.h>
#include <luisa/core/stl/vector.h>
#include <luisa/ast/expression.h>
#include <luisa/ast/op.h>
#include <luisa/ast/type.h>

#include <array>
#include <functional>
#include <limits>
#include <utility>

namespace luisa::compute {
namespace {

using detail::FunctionBuilder;
using detail::TileFunctionBuilder;

// ---------------------------------------------------------------------------
// dtype / layout helpers (R1 tags of <luisa/ast/tensor.h>)
// ---------------------------------------------------------------------------

const Type *tensor_element_type(TensorElementType e) noexcept {
    switch (e) {
        case TensorElementType::F16: return Type::of<half>();
        case TensorElementType::F32: return Type::of<float>();
        case TensorElementType::I32: return Type::of<int>();
        case TensorElementType::I8: return Type::of<byte>();
        case TensorElementType::FP8: return Type::from("float8e4m3");
        // I4 / FP4 are 4-bit sub-byte dtypes with no core element Type yet:
        // they reach the error below instead of being silently mis-lowered.
    }
    LUISA_ERROR_WITH_LOCATION("Unsupported tensor element type {}.",
                              static_cast<uint32_t>(e));
}

// Tile extent along axis i: extent[i] when positive, else the whole-tensor dim.
int32_t axis_extent(const TensorExpr *t, uint32_t i) noexcept {
    auto ext = t->extent();
    if (i < ext.size() && ext[i] > 0) { return ext[i]; }
    auto dims = t->dims();
    return i < dims.size() ? dims[i] : 1;
}

uint32_t tile_element_count(const TensorExpr *t) noexcept {
    uint32_t n = 1u;
    for (auto i = 0u; i < t->rank(); ++i) {
        n *= static_cast<uint32_t>(axis_extent(t, i));
    }
    return n;
}

// A whole-tile op needs a fully known extent.  The traced IR records the
// extent of the "local" operand (shared / fragment / explicit slice) but
// global views carry extent {0,0}, so the op extent is taken from whichever
// operand of the op has a positive extent on every axis.
//
// NOTE: this inspects the RAW extent list only — a global view whose tile
// extent is {0,0} is NOT "known" even when the underlying tensor dims are
// nonzero (the op extent must come from the other, local operand).
bool extent_known(const TensorExpr *t) noexcept {
    auto ext = t->extent();
    if (t->rank() == 0u || ext.size() != t->rank()) { return false; }
    for (auto e : ext) {
        if (e <= 0) { return false; }
    }
    return true;
}

const TensorExpr *op_extent_of(const TensorExpr *a, const TensorExpr *b) noexcept {
    if (extent_known(a)) { return a; }
    if (extent_known(b)) { return b; }
    LUISA_ERROR_WITH_LOCATION(
        "Tile op has no fully-known extent on either operand "
        "(the traced tile IR records tile extents only on shared/fragment "
        "tensors or explicit slices; global views are extent-less).");
}


// Layout signature used to resolve a view clone back to its AllocStmt storage
// on the (fallback) path where tensors carry no name.
struct Layout {
    TensorScope scope{TensorScope::Global};
    TensorElementType dtype{TensorElementType::F32};
    luisa::fixed_vector<int32_t, 4> dims;
    bool operator==(const Layout &o) const noexcept {
        return scope == o.scope && dtype == o.dtype && dims == o.dims;
    }
};

// ---------------------------------------------------------------------------
// The lowering engine
// ---------------------------------------------------------------------------
class TileLowerer {

public:
    TileCompileResult lower(const luisa::shared_ptr<const TileFunctionBuilder> &tile_fn,
                            const TileToKernelConfig &config) {
        _use_cooperative = config.use_cooperative;
        auto meta = tile_fn->compile_meta_data();// block size + dispatch grid
        _threads = meta.block_size[0];
        _gx = meta.dispatch_size[0];
        _gy = meta.dispatch_size[1];
        _kernel2d = _gy > 1u;
        auto builder = luisa::make_shared<FunctionBuilder>(Function::Tag::KERNEL);
        {
            FunctionBuilder::FunctionStackGuard guard{builder.get()};
            builder->with(builder->body(), [&] {
                _fb = builder.get();
                _tile = tile_fn.get();
                builder->set_block_size(uint3{meta.block_size[0], meta.block_size[1], meta.block_size[2]});
                _emit_all(tile_fn->body()->statements());
            });
        }
        // T.Kernel(gx, gy, threads) launches gx*gy BLOCKS of `threads`
        // threads; the Luisa `.dispatch(...)` argument is the TOTAL number of
        // threads (grid = ceildiv(dispatch, block_size)), so the returned
        // dispatch size is (gx * threads, gy, 1) — the caller dispatches with
        // .dispatch(result.dispatch_size.x, result.dispatch_size.y).
        auto dispatch = _kernel2d ?
                            uint3{meta.dispatch_size[0] * _threads, meta.dispatch_size[1], 1u} :
                            uint3{meta.dispatch_size[0] * _threads, 1u, 1u};
        return {builder, dispatch};
    }

private:
    // per-axis coordinates as raw expression pointers (Expr is
    // non-assignable, so the coord array holds the underlying
    // AST pointers and is built/assigned elementwise)
    using Coord = std::array<const Expression *, 4>;

    struct Storage {
        TensorScope scope{TensorScope::Global};
        TensorElementType dtype{TensorElementType::F32};
        const RefExpr *buffer = nullptr;  // Global: Buffer<T> kernel argument
        const RefExpr *shared = nullptr;  // Shared: Type::array(elem, n)
        const RefExpr *fragment = nullptr;// Fragment: Type::array(elem, n) local
        uint32_t array_size = 0u;
    };

    struct TempValue {
        TensorElementType dtype{TensorElementType::F32};
        std::function<const Expression *(const Coord &)> eval;
    };

    FunctionBuilder *_fb = nullptr;
    const TileFunctionBuilder *_tile = nullptr;
    // lowering configuration
    bool _use_cooperative = false;
    // launch metadata
    uint32_t _threads = 1u;
    uint32_t _gx = 1u;
    uint32_t _gy = 1u;
    bool _kernel2d = false;
    // pipelined-loop context
    const Expression *_pipeline_var = nullptr;
    uint32_t _pipeline_count = 0u;
    uint32_t _pipeline_axis = 0u;
    // the effective op extent of the statement being emitted (global views
    // carry no extent of their own; the op extent comes from the other
    // operand / the loop target)
    const TensorExpr *_current_extent = nullptr;
    // storage
    luisa::unordered_map<const TensorExpr *, Storage> _storage_by_ptr;
    luisa::unordered_map<luisa::string, Storage> _storage_by_name;
    luisa::vector<std::pair<Layout, Storage>> _storage_by_layout;// name-less fallback
    // value temporaries (BINARY / MAX / RSQRT outputs), keyed by their
    // TensorExpr pointer (the same pointer flows into the consuming statement).
    // The association temp-pointer -> producer statement is recorded by the
    // TileFunctionBuilder (temp_output()), so no guessing is needed.
    luisa::unordered_map<const TensorExpr *, TempValue> _temps;
      // shared staging tiles backing block-partitioned fragment producers
      luisa::unordered_map<const TensorExpr *, const RefExpr *> _fragment_staging;

      // Fragments with at least this many elements are backed by a block-shared
      // array instead of a per-thread local array.  A per-thread local array of
      // that size would spill to local/global memory on the GPU (the dominant
      // dispatch cost for the tensor example kernels), so we stage them in
      // shared memory and process them with partition loops (each element is
      // computed once across the block instead of once per thread).  The
      // existing "sync after every shared-accessing statement" discipline
      // (see _accesses_shared + the trailing _sync_block() in _emit) provides
      // the required barriers between the shared-backed ops.
      static constexpr uint32_t kFragmentSharedThreshold = 512u;

      // True when a fragment tensor is backed by a block-shared array instead of
      // a per-thread local array (see _emit_alloc / kFragmentSharedThreshold).
      [[nodiscard]] bool _is_fragment_shared_backed(const TensorExpr *t) noexcept {
          if (t == nullptr || t->scope() != TensorScope::Fragment) { return false; }
          auto *st = _try_storage(t);
          return st != nullptr && st->shared != nullptr;
      }

    // ---- expression helpers -------------------------------------------------

    [[nodiscard]] const Expression *_literal_u(uint32_t v) const noexcept {
        return _fb->literal(Type::of<uint>(), v);
    }

    [[nodiscard]] Coord _zero_coord() const noexcept {
        return Coord{_literal_u(0u), _literal_u(0u), _literal_u(0u), _literal_u(0u)};
    }

    [[nodiscard]] const Expression *_bin(BinaryOp op, const Expression *l, const Expression *r) const noexcept {
        return _fb->binary(l->type(), op, l, r);
    }

    // single-component vector access: swizzle size 1, code = component index
    // (x = 0, y = 1, ...); matches the DSL thread_id().x / block_id().y sugar.
    [[nodiscard]] const Expression *_vec_comp(const Expression *v, uint index) const noexcept {
        return _fb->swizzle(Type::of<uint>(), v, 1u, index);
    }

    void _sync_block() const noexcept {
        _fb->call(CallOp::SYNCHRONIZE_BLOCK, {});
    }

    // ---- warp/wave helpers (lc_optimize: warp collectives) -------------------

    [[nodiscard]] const Expression *_tid_x() const noexcept {
        return _vec_comp(_fb->thread_id(), 0u);
    }

    [[nodiscard]] const Expression *_lane_count() const noexcept {
        return _fb->warp_lane_count();
    }

    [[nodiscard]] const Expression *_lane() const noexcept {
        return _fb->warp_lane_id();
    }

    [[nodiscard]] const Expression *_warp_id() const noexcept {
        return _fb->binary(Type::of<uint>(), BinaryOp::DIV, _tid_x(), _lane_count());
    }

    [[nodiscard]] const Expression *_num_warps() const noexcept {
        return _fb->binary(Type::of<uint>(), BinaryOp::DIV, _literal_u(_threads), _lane_count());
    }

    [[nodiscard]] const Expression *_ceildiv_expr(const Expression *a, const Expression *b) const noexcept {
        auto t = _fb->binary(Type::of<uint>(), BinaryOp::ADD, a, b);
        t = _fb->binary(Type::of<uint>(), BinaryOp::SUB, t, _literal_u(1u));
        return _fb->binary(Type::of<uint>(), BinaryOp::DIV, t, b);
    }

    // all-lane warp reduction matching a TileReduceOp (lc_optimize 2.2/2.5:
    // XOR butterfly via WARP_READ_LANE; every lane ends with the total).
    // ABS_* must be pre-folded per element by the caller.
    [[nodiscard]] const Expression *_warp_reduce(TileReduceOp op, const Type *elem_t,
                                                 const Expression *v) {
        auto lane = _lane();
        auto lanes = _lane_count();
        auto result = _fb->local(elem_t);
        _fb->assign(result, v);
        for (uint32_t d = 1u; d <= 64u; d <<= 1u) {
            // warp-uniform guard: skip the steps above the actual warp size
            auto d_active = _fb->binary(Type::of<bool>(), BinaryOp::LESS,
                                        _literal_u(d), lanes);
            _if(d_active, [&] {
                auto peer = _fb->binary(Type::of<uint>(), BinaryOp::BIT_XOR,
                                        lane, _literal_u(d));
                auto other = _fb->call(elem_t, CallOp::WARP_READ_LANE, {result, peer});
                _fb->assign(result, _reduce_combine(op, elem_t, result, other));
            });
        }
        return result;
    }

    // ---- fragment staging ------------------------------------------------------
    // Fragment tiles are replicated per-thread.  Block-partitioned producers
    // (GEMM / REDUCE / SCAN / global->fragment COPY) publish their owned
    // elements through a small shared staging tile, then every thread
    // refreshes its whole replica from it (one staging tile per fragment).
    [[nodiscard]] const RefExpr *_staging_for(const TensorExpr *t, const Type *elem_t) {
        if (auto it = _fragment_staging.find(t); it != _fragment_staging.end()) {
            return it->second;
        }
        auto n = tile_element_count(t);
        if (n == 0u) [[unlikely]] {
            LUISA_ERROR_WITH_LOCATION("Fragment staging with zero elements: {}", t->describe());
        }
        auto s = _fb->shared(Type::array(elem_t, n));
        _fragment_staging.emplace(t, s);
        return s;
    }

    // refresh every thread's fragment replica from the staging tile
    void _replicate_from_staging(const TensorExpr *t, const Type *elem_t,
                                 const RefExpr *staging) {
        _sync_block();
        _full_loop(t, [&](const Coord &c) {
            auto idx = _local_index(t, c);
            _write_to(t, c, _fb->access(elem_t, staging, idx));
        });
    }

    // emit `if (cond) { body() }` (the AST-level equivalent of the DSL $if)
    template<typename Body>
    void _if(const Expression *cond, Body &&body) {
        auto *stmt = _fb->if_(cond);
        _fb->push_scope(stmt->true_branch());
        body();
        _fb->pop_scope(stmt->true_branch());
    }

    // emit `for (var = begin; var < end; var += step) { body(var) }`
    // (the AST-level equivalent of the DSL dynamic_range); `var` is a fresh
    // local created at the current scope, exactly like the DSL loop variable.
    template<typename Body>
    void _for_range(const Expression *begin, const Expression *end,
                    const Expression *step, Body &&body) {
        auto *var = _fb->local(Type::of<uint>());
        _fb->assign(var, begin);
        auto *cond = _fb->binary(Type::of<bool>(), BinaryOp::LESS, var, end);
        auto *stmt = _fb->for_(var, cond, step);
        _fb->push_scope(stmt->body());
        body(var);
        _fb->pop_scope(stmt->body());
    }

    [[nodiscard]] const Expression *_zero_of(TensorElementType e) const noexcept {
        switch (e) {
            case TensorElementType::F16: return _fb->literal(Type::of<half>(), half{0.f});
            case TensorElementType::F32: return _fb->literal(Type::of<float>(), 0.f);
            case TensorElementType::I32: return _fb->literal(Type::of<int>(), 0);
            case TensorElementType::I8: return _fb->literal(Type::of<byte>(), byte{0});
            case TensorElementType::FP8:
                // The all-zero bit pattern is a valid fp8 zero in both the e4m3
                // and e5m2 encodings; the LiteralValue variant has no fp8
                // alternative yet, so the zero is carried as a byte and cast to
                // the fp8 element type by the caller (_write_to / _maybe_cast).
                return _fb->literal(Type::of<byte>(), byte{0});
            // I4 / FP4 have no core element Type: the caller fails via
            // tensor_element_type() before a zero can be materialized.
        }
        LUISA_ERROR_WITH_LOCATION("Unsupported tensor element type.");
    }

    [[nodiscard]] const Expression *_recreate_literal(const LiteralExpr *lit) const noexcept {
        return _fb->literal(lit->type(), lit->value());
    }

    // identity element of a TileReduceOp for the given dtype
    [[nodiscard]] const Expression *_reduce_identity(TileReduceOp op, TensorElementType e) const {
        switch (op) {
            case TileReduceOp::SUM:
            case TileReduceOp::ABS_SUM:
            case TileReduceOp::BIT_OR:
            case TileReduceOp::BIT_XOR:
                return _zero_of(e);
            case TileReduceOp::MAX:
            case TileReduceOp::ABS_MAX:
                switch (e) {
                    case TensorElementType::F16: return _fb->literal(Type::of<half>(), half{-65504.f});
                    case TensorElementType::F32: return _fb->literal(Type::of<float>(), std::numeric_limits<float>::lowest());
                    case TensorElementType::I32: return _fb->literal(Type::of<int>(), std::numeric_limits<int>::min());
                    default: break;
                }
                break;
            case TileReduceOp::MIN:
                switch (e) {
                    case TensorElementType::F16: return _fb->literal(Type::of<half>(), half{65504.f});
                    case TensorElementType::F32: return _fb->literal(Type::of<float>(), std::numeric_limits<float>::max());
                    case TensorElementType::I32: return _fb->literal(Type::of<int>(), std::numeric_limits<int>::max());
                    default: break;
                }
                break;
            case TileReduceOp::BIT_AND:
                if (e == TensorElementType::I32) { return _fb->literal(Type::of<int>(), -1); }
                break;
        }
        LUISA_ERROR_WITH_LOCATION(
            "tile_to_kernel: reduce op {} has no identity for dtype {}.",
            static_cast<uint32_t>(op), tensor_element_type_name(e));
    }

    // combine step of a TileReduceOp: acc <- acc `op` v
    [[nodiscard]] const Expression *_reduce_combine(TileReduceOp op, const Type *elem_t,
                                                    const Expression *acc, const Expression *v) const {
        switch (op) {
            case TileReduceOp::SUM:
            case TileReduceOp::ABS_SUM:
                return _fb->binary(elem_t, BinaryOp::ADD, acc, v);
            case TileReduceOp::MAX:
            case TileReduceOp::ABS_MAX:
                return _fb->call(elem_t, CallOp::MAX, {acc, v});
            case TileReduceOp::MIN:
                return _fb->call(elem_t, CallOp::MIN, {acc, v});
            case TileReduceOp::BIT_AND:
                return _fb->binary(elem_t, BinaryOp::BIT_AND, acc, v);
            case TileReduceOp::BIT_OR:
                return _fb->binary(elem_t, BinaryOp::BIT_OR, acc, v);
            case TileReduceOp::BIT_XOR:
                return _fb->binary(elem_t, BinaryOp::BIT_XOR, acc, v);
        }
        LUISA_ERROR_WITH_LOCATION("tile_to_kernel: invalid tile reduce op.");
    }

    [[nodiscard]] const Expression *_maybe_cast(const Expression *v, const Type *t) const noexcept {
        return v->type() == t ? v : _fb->cast(t, CastOp::STATIC, v);
    }

    // ---- storage resolution --------------------------------------------------

    // resolve a statement operand to its storage, or nullptr when it is a
    // value temporary (BINARY / MAX / RSQRT output) instead
    [[nodiscard]] Storage *_try_storage(const TensorExpr *t) {
        if (auto it = _storage_by_ptr.find(t); it != _storage_by_ptr.end()) {
            return &it->second;
        }
        auto name = t->name();
        if (!name.empty()) {
            if (auto it = _storage_by_name.find(luisa::string{name}); it != _storage_by_name.end()) {
                return &it->second;
            }
        }
        Layout layout{t->scope(), t->dtype(),
                      luisa::fixed_vector<int32_t, 4>{t->dims().begin(), t->dims().end()}};
        for (auto &kv : _storage_by_layout) {
            if (kv.first == layout) { return &kv.second; }
        }
        return nullptr;
    }

    [[nodiscard]] Storage &_storage_for(const TensorExpr *t) {
        if (auto *st = _try_storage(t)) { return *st; }
        LUISA_ERROR_WITH_LOCATION(
            "Tile operand is not an allocated tensor and does not match any "
            "AllocStmt storage (scope={}, dtype={}, tensor={}).",
            scope_name(t->scope()), tensor_element_type_name(t->dtype()),
            t->describe());
    }

    [[nodiscard]] bool _is_temp(const TensorExpr *t) const noexcept {
        return _temps.contains(t);
    }

    // ---- index math ----------------------------------------------------------

    // row-major linear index inside a shared/fragment tile
    [[nodiscard]] const Expression *_local_index(const TensorExpr *t, const Coord &c) const {
        auto idx = _literal_u(0u);
        uint32_t stride = 1u;
        for (int32_t i = static_cast<int32_t>(t->rank()) - 1; i >= 0; --i) {
            auto term = _fb->binary(Type::of<uint>(), BinaryOp::MUL,
                                    c[i], _literal_u(stride));
            idx = _fb->binary(Type::of<uint>(), BinaryOp::ADD, idx, term);
            stride *= static_cast<uint32_t>(axis_extent(t, i));
        }
        return idx;
    }

    [[nodiscard]] uint32_t _min_extent_axis(const TensorExpr *t) const noexcept {
        uint32_t best_axis = 0u;
        uint32_t best = ~0u;
        for (auto a = 0u; a < t->rank(); ++a) {
            auto e = static_cast<uint32_t>(axis_extent(t, a));
            if (e < best) { best = e; best_axis = a; }
        }
        return best_axis;
    }

    // reconstructed global buffer index (plan: base-offset reconstruction)
    [[nodiscard]] const Expression *_global_index(const TensorExpr *t, const Coord &c) const {
        auto rank = t->rank();
        auto ext = _current_extent != nullptr ? _current_extent : t;
        auto E = [&](uint32_t i) { return static_cast<uint32_t>(axis_extent(ext, i)); };
        // per-axis runtime base
        auto base_expr = [&](uint32_t i) -> const Expression * {
            auto b = _literal_u(0u);
            auto off = t->offset();
            auto host_off = i < off.size() && off[i] > 0 ? static_cast<uint32_t>(off[i]) : 0u;
            if (_pipeline_var != nullptr && i == _pipeline_axis) {
                auto t1 = _fb->binary(Type::of<uint>(), BinaryOp::MUL,
                                      _pipeline_var, _literal_u(E(i)));
                b = _fb->binary(Type::of<uint>(), BinaryOp::ADD, b, t1);
            } else if (_kernel2d) {
                auto bid = i == 0u ? _vec_comp(_fb->block_id(), 1u)
                                       : _vec_comp(_fb->block_id(), 0u);
                auto t1 = _fb->binary(Type::of<uint>(), BinaryOp::MUL,
                                      bid, _literal_u(E(i)));
                b = _fb->binary(Type::of<uint>(), BinaryOp::ADD, b, t1);
            } else if (i == 0u) {
                auto t1 = _fb->binary(Type::of<uint>(), BinaryOp::MUL,
                                      _vec_comp(_fb->block_id(), 0u), _literal_u(E(i)));
                b = _fb->binary(Type::of<uint>(), BinaryOp::ADD, b, t1);
            }
            if (host_off != 0u) {
                b = _fb->binary(Type::of<uint>(), BinaryOp::ADD, b, _literal_u(host_off));
            }
            return b;
        };
        // per-axis full length (host), used as row stride
        auto full_len = [&](uint32_t i) -> uint32_t {
            if (_pipeline_var != nullptr && i == _pipeline_axis) {
                return _pipeline_count * E(i);
            }
            if (_kernel2d) { return i == 0u ? _gy * E(i) : _gx * E(i); }
            return i == 0u ? _gx * E(i) : E(i);
        };
        auto row_stride = [&](uint32_t i) -> uint32_t {
            uint32_t s = 1u;
            for (uint32_t j = i + 1u; j < rank; ++j) { s *= full_len(j); }
            return s;
        };
        auto idx = _literal_u(0u);
        for (uint32_t i = 0u; i < rank; ++i) {
            auto base_i = base_expr(i);
            auto sum = _fb->binary(Type::of<uint>(), BinaryOp::ADD, base_i, c[i]);
            auto term = _fb->binary(Type::of<uint>(), BinaryOp::MUL,
                                    sum, _literal_u(row_stride(i)));
            idx = _fb->binary(Type::of<uint>(), BinaryOp::ADD, idx, term);
        }
        return idx;
    }

    // ---- value access --------------------------------------------------------

    [[nodiscard]] const Expression *_value_at(const TensorExpr *t, const Coord &c) {
        if (_is_temp(t)) { return _temps[t].eval(c); }
        if (auto *st = _try_storage(t)) {
            auto elem_t = tensor_element_type(st->dtype);
            switch (st->scope) {
                case TensorScope::Global: {
                    auto idx = _global_index(t, c);
                    return _fb->call(elem_t, CallOp::BUFFER_READ, {st->buffer, idx});
                }
                case TensorScope::Shared: {
                    auto idx = _local_index(t, c);
                    return _fb->access(elem_t, st->shared, idx);
                }
                case TensorScope::Fragment: {
                    // large fragments are backed by a block-shared array
                    if (st->shared != nullptr) {
                        auto idx = _local_index(t, c);
                        return _fb->access(elem_t, st->shared, idx);
                    }
                    auto idx = _local_index(t, c);
                    return _fb->access(elem_t, st->fragment, idx);
                }
            }
            LUISA_ERROR_WITH_LOCATION("Invalid tensor scope.");
        }
        LUISA_ERROR_WITH_LOCATION(
            "Tile lowering: a statement references a value temporary that was "
            "not recorded by its producing statement ({}).",
            t->describe());
    }

    void _write_to(const TensorExpr *t, const Coord &c, const Expression *value) {
        auto &st = _storage_for(t);
        auto elem_t = tensor_element_type(st.dtype);
        value = _maybe_cast(value, elem_t);
        switch (st.scope) {
            case TensorScope::Global: {
                auto idx = _global_index(t, c);
                _fb->call(CallOp::BUFFER_WRITE, {st.buffer, idx, value});
                break;
            }
            case TensorScope::Shared: {
                auto idx = _local_index(t, c);
                _fb->assign(_fb->access(elem_t, st.shared, idx), value);
                break;
            }
            case TensorScope::Fragment: {
                // large fragments are backed by a block-shared array
                if (st.shared != nullptr) {
                    auto idx = _local_index(t, c);
                    _fb->assign(_fb->access(elem_t, st.shared, idx), value);
                    break;
                }
                auto idx = _local_index(t, c);
                _fb->assign(_fb->access(elem_t, st.fragment, idx), value);
                break;
            }
        }
    }

    [[nodiscard]] Coord _decompose(const TensorExpr *t, const Expression *idx) const {
        auto c = _zero_coord();
        uint32_t stride = 1u;
        for (int32_t i = static_cast<int32_t>(t->rank()) - 1; i >= 0; --i) {
            auto div = _fb->binary(Type::of<uint>(), BinaryOp::DIV,
                                   idx, _literal_u(stride));
            auto rem = _fb->binary(Type::of<uint>(), BinaryOp::MOD,
                                   div, _literal_u(static_cast<uint32_t>(axis_extent(t, i))));
            c[i] = rem;
            stride *= static_cast<uint32_t>(axis_extent(t, i));
        }
        return c;
    }

    // ---- loop helpers ---------------------------------------------------------

    // each element exactly once across the block (Global / Shared targets)
    template<typename Body>
    void _partition_loop(const TensorExpr *t, Body &&body) {
        auto total = tile_element_count(t);
        auto iters = (total + _threads - 1u) / _threads;
        auto tid = _vec_comp(_fb->thread_id(), 0u);
        _for_range(_literal_u(0u), _literal_u(iters), _literal_u(1u),
                   [&](const Expression *i) {
                       auto t1 = _fb->binary(Type::of<uint>(), BinaryOp::MUL,
                                             i, _literal_u(_threads));
                       auto idx = _fb->binary(Type::of<uint>(), BinaryOp::ADD, t1, tid);
                       auto cond = _fb->binary(Type::of<bool>(), BinaryOp::LESS,
                                               idx, _literal_u(total));
                       _if(cond, [&] { body(_decompose(t, idx)); });
                   });
    }

    // every thread processes the whole tile (replicated Fragment layout)
    template<typename Body>
    void _full_loop(const TensorExpr *t, Body &&body) {
        auto total = tile_element_count(t);
        _for_range(_literal_u(0u), _literal_u(total), _literal_u(1u),
                   [&](const Expression *i) { body(_decompose(t, i)); });
    }

    // ---------------------------------------------------------------------------
    // statement emission
    // ---------------------------------------------------------------------------

    void _emit_all(luisa::span<const TensorStmt *const> stmts) {
        for (auto i = 0u; i < stmts.size();) {
            auto *stmt = stmts[i];
            if (stmt->op() == TileOpKind::PIPELINED) {
                auto *p = static_cast<const PipelinedStmt *>(stmt);
                // flat IR: the pipelined body is the run of statements that
                // touches at least one shared tensor (copy/copy/gemm pattern);
                // the run ends at the first statement with no shared operand.
                auto end = i + 1u;
                while (end < stmts.size()) {
                    auto *candidate = stmts[end];
                    if (candidate->op() == TileOpKind::PIPELINED ||
                        candidate->op() == TileOpKind::KERNEL_1D ||
                        candidate->op() == TileOpKind::KERNEL_2D) {
                        break;
                    }
                    if (!_accesses_shared(candidate)) { break; }
                    ++end;
                }
                _emit_pipelined(p, luisa::span<const TensorStmt *const>{stmts.data() + i + 1u, end - i - 1u});
                i = end;
            } else {
                _emit(stmt);
                ++i;
            }
        }
    }

    void _emit_pipelined(const PipelinedStmt *p,
                         luisa::span<const TensorStmt *const> body) {
        auto count = static_cast<uint32_t>(p->count());
        if (count == 0u) { return; }
        _pipeline_count = count;
        _for_range(_literal_u(0u), _literal_u(count), _literal_u(1u),
                   [&](const Expression *ko) {
                       _pipeline_var = ko;
                       for (auto *s : body) { _emit(s); }
                   });
        _pipeline_var = nullptr;
        _pipeline_count = 0u;
    }

  [[nodiscard]] bool _accesses_shared(const TensorStmt *s) {
      // A statement touches shared memory when an operand is a Shared tensor or
      // a large fragment backed by a block-shared array (the latter is created
      // by _emit_alloc when a fragment is large enough to spill; see
      // kFragmentSharedThreshold).  The trailing _sync_block() in _emit then
      // provides the barrier discipline for both kinds of shared storage.
      auto shared_scope = [this](const TensorExpr *t) {
          if (t == nullptr) { return false; }
          if (t->scope() == TensorScope::Shared) { return true; }
          return _is_fragment_shared_backed(t);
      };
        switch (s->op()) {
            case TileOpKind::ALLOC: {
                auto *a = static_cast<const AllocStmt *>(s);
                return shared_scope(a->tensor());
            }
            case TileOpKind::CLEAR: {
                auto *c = static_cast<const ClearStmt *>(s);
                return shared_scope(c->t());
            }
            case TileOpKind::COPY: {
                auto *c = static_cast<const CopyStmt *>(s);
                return shared_scope(c->src()) || shared_scope(c->dst());
            }
            case TileOpKind::GEMM: {
                auto *g = static_cast<const GemmStmt *>(s);
                return shared_scope(g->a()) || shared_scope(g->b()) || shared_scope(g->c());
            }
            case TileOpKind::REDUCE_SUM: {
                auto *r = static_cast<const ReduceSumStmt *>(s);
                return shared_scope(r->x()) || shared_scope(r->y());
            }
            case TileOpKind::STORE: {
                auto *st = static_cast<const TileStoreStmt *>(s);
                return shared_scope(st->lhs()) || shared_scope(st->rhs_tensor());
            }
            case TileOpKind::BINARY: {
                auto *b = static_cast<const TileBinaryStmt *>(s);
                return shared_scope(b->lhs()) || shared_scope(b->rhs_tensor());
            }
            case TileOpKind::MAX: {
                auto *m = static_cast<const MaxStmt *>(s);
                return shared_scope(m->a());
            }
            case TileOpKind::RSQRT: {
                auto *r = static_cast<const RsqrtStmt *>(s);
                return shared_scope(r->a());
            }
            case TileOpKind::PRINT: {
                auto *p = static_cast<const TilePrintStmt *>(s);
                return shared_scope(p->t());
            }
            case TileOpKind::TRANSPOSE: {
                auto *t = static_cast<const TransposeStmt *>(s);
                return shared_scope(t->src()) || shared_scope(t->dst());
            }
            case TileOpKind::CLAMP: {
                auto *c = static_cast<const ClampStmt *>(s);
                return shared_scope(c->dst());
            }
            case TileOpKind::ATOMIC: {
                auto *a = static_cast<const AtomicStmt *>(s);
                return shared_scope(a->dst()) || shared_scope(a->value_tensor());
            }
            case TileOpKind::FILL: {
                auto *f = static_cast<const FillStmt *>(s);
                return shared_scope(f->buf());
            }
            case TileOpKind::REDUCE: {
                auto *r = static_cast<const ReduceStmt *>(s);
                return shared_scope(r->buf()) || shared_scope(r->out());
            }
            case TileOpKind::CUMSUM: {
                auto *c = static_cast<const CumSumStmt *>(s);
                return shared_scope(c->src()) || shared_scope(c->dst());
            }
            case TileOpKind::CUMMAX: {
                auto *c = static_cast<const CumMaxStmt *>(s);
                return shared_scope(c->src()) || shared_scope(c->dst());
            }
            case TileOpKind::ANY_OF: {
                auto *a = static_cast<const AnyOfStmt *>(s);
                return shared_scope(a->buf());
            }
            case TileOpKind::ALL_OF: {
                auto *a = static_cast<const AllOfStmt *>(s);
                return shared_scope(a->buf());
            }
            case TileOpKind::SHUFFLE: {
                auto *sh = static_cast<const ShuffleStmt *>(s);
                return shared_scope(sh->value_tensor());
            }
            case TileOpKind::MIN: {
                auto *m = static_cast<const MinStmt *>(s);
                return shared_scope(m->a());
            }
            case TileOpKind::ABS: {
                auto *a = static_cast<const AbsStmt *>(s);
                return shared_scope(a->a());
            }
            case TileOpKind::FAST_MATH: {
                auto *f = static_cast<const FastMathStmt *>(s);
                return shared_scope(f->a());
            }
            case TileOpKind::IEEE_MATH: {
                auto *ie = static_cast<const IeeeMathStmt *>(s);
                if (shared_scope(ie->a())) { return true; }
                if (ie->b() != nullptr && shared_scope(ie->b())) { return true; }
                if (ie->c() != nullptr && shared_scope(ie->c())) { return true; }
                return false;
            }
            default: return false;
        }
    }

    void _emit(const TensorStmt *stmt) {
        switch (stmt->op()) {
            case TileOpKind::ALLOC: _emit_alloc(static_cast<const AllocStmt *>(stmt)); break;
            case TileOpKind::CLEAR: _emit_clear(static_cast<const ClearStmt *>(stmt)); break;
            case TileOpKind::COPY: _emit_copy(static_cast<const CopyStmt *>(stmt)); break;
            case TileOpKind::STORE: _emit_store(static_cast<const TileStoreStmt *>(stmt)); break;
            case TileOpKind::BINARY: _emit_binary(static_cast<const TileBinaryStmt *>(stmt)); break;
            case TileOpKind::MAX: _emit_max(static_cast<const MaxStmt *>(stmt)); break;
            case TileOpKind::RSQRT: _emit_rsqrt(static_cast<const RsqrtStmt *>(stmt)); break;
            case TileOpKind::REDUCE_SUM: _emit_reduce_sum(static_cast<const ReduceSumStmt *>(stmt)); break;
            case TileOpKind::GEMM: _emit_gemm(static_cast<const GemmStmt *>(stmt)); break;
            case TileOpKind::PRINT: _emit_print(static_cast<const TilePrintStmt *>(stmt)); break;
            case TileOpKind::FILL: _emit_fill(static_cast<const FillStmt *>(stmt)); break;
            case TileOpKind::TRANSPOSE: _emit_transpose(static_cast<const TransposeStmt *>(stmt)); break;
            case TileOpKind::CLAMP: _emit_clamp(static_cast<const ClampStmt *>(stmt)); break;
            case TileOpKind::ATOMIC: _emit_atomic(static_cast<const AtomicStmt *>(stmt)); break;
            case TileOpKind::SYNC: _emit_sync(static_cast<const SyncStmt *>(stmt)); break;
            case TileOpKind::WARP_REDUCE: _emit_warp_reduce(static_cast<const WarpReduceStmt *>(stmt)); break;
            case TileOpKind::LOOP_BREAK: _fb->break_(); break;
            case TileOpKind::REDUCE: _emit_reduce(static_cast<const ReduceStmt *>(stmt)); break;
            case TileOpKind::CUMSUM: {
                auto *s = static_cast<const CumSumStmt *>(stmt);
                _emit_scan(s->src(), s->dst(), s->dim(), s->reverse(), false);
                break;
            }
            case TileOpKind::CUMMAX: {
                auto *s = static_cast<const CumMaxStmt *>(stmt);
                _emit_scan(s->src(), s->dst(), s->dim(), s->reverse(), true);
                break;
            }
            case TileOpKind::ANY_OF: _emit_any_all(static_cast<const AnyOfStmt *>(stmt)->buf(), false); break;
            case TileOpKind::ALL_OF: _emit_any_all(static_cast<const AllOfStmt *>(stmt)->buf(), true); break;
            case TileOpKind::SHUFFLE: _emit_shuffle(static_cast<const ShuffleStmt *>(stmt)); break;
            case TileOpKind::MIN: _emit_min(static_cast<const MinStmt *>(stmt)); break;
            case TileOpKind::ABS: _emit_abs(static_cast<const AbsStmt *>(stmt)); break;
            case TileOpKind::FAST_MATH: _emit_fast_math(static_cast<const FastMathStmt *>(stmt)); break;
            case TileOpKind::IEEE_MATH: _emit_ieee_math(static_cast<const IeeeMathStmt *>(stmt)); break;
            // host-side / metadata statements: no kernel code
            case TileOpKind::CEILDIV:
            case TileOpKind::KERNEL_1D:
            case TileOpKind::KERNEL_2D:
            case TileOpKind::PIPELINED:
            case TileOpKind::RESHAPE:
            case TileOpKind::VIEW:
            case TileOpKind::LOOP_ANNOTATION:
            case TileOpKind::ANNOTATE:
            case TileOpKind::DYNAMIC:
            case TileOpKind::SYMBOLIC:
            case TileOpKind::INLINE:
            case TileOpKind::META_CLASS:
                break;
            default:
                LUISA_ERROR_WITH_LOCATION(
                    "tile_to_kernel: TileOpKind {} is not yet supported by the "
                    "regular-kernel lowering (see the gap list in the plan).",
                    static_cast<uint32_t>(stmt->op()));
        }
        // barrier discipline: sync after every statement that touches shared
        // memory (never inside a thread-divergent branch — all our shared
        // accesses live inside $for/$if bodies, the sync is at top level)
        if (_accesses_shared(stmt)) { _sync_block(); }
    }

    void _emit_alloc(const AllocStmt *s) {
        auto *t = s->tensor();
        Storage st;
        st.scope = t->scope();
        st.dtype = t->dtype();
        auto elem_t = tensor_element_type(t->dtype());
        switch (t->scope()) {
            case TensorScope::Global:
                // one Buffer<T> kernel argument per Global tensor (AllocStmt order)
                switch (t->dtype()) {
                    case TensorElementType::F16: st.buffer = _fb->buffer(Type::of<Buffer<half>>()); break;
                    case TensorElementType::F32: st.buffer = _fb->buffer(Type::of<Buffer<float>>()); break;
                    case TensorElementType::I32: st.buffer = _fb->buffer(Type::of<Buffer<int>>()); break;
                    case TensorElementType::I8: st.buffer = _fb->buffer(Type::of<Buffer<byte>>()); break;
                    case TensorElementType::FP8: st.buffer = _fb->buffer(Type::from("buffer<float8e4m3>")); break;
                    default:
                        // I4 / FP4 are 4-bit sub-byte dtypes with no core
                        // element Type: reject instead of mis-allocating.
                        LUISA_ERROR_WITH_LOCATION(
                            "tile_to_kernel: tensor element type {} is not "
                            "lowerable to a kernel buffer (no core element "
                            "Type for 4-bit dtypes).",
                            tensor_element_type_name(t->dtype()));
                }
                break;
            case TensorScope::Shared: {
                auto n = tile_element_count(t);
                if (n == 0u) [[unlikely]] {
                    LUISA_ERROR_WITH_LOCATION("Shared tile allocation with zero elements: {}", t->describe());
                }
                st.shared = _fb->shared(Type::array(elem_t, n));
                st.array_size = n;
                break;
            }
          case TensorScope::Fragment: {
              auto n = tile_element_count(t);
              if (n == 0u) [[unlikely]] {
                  LUISA_ERROR_WITH_LOCATION("Fragment tile allocation with zero elements: {}", t->describe());
              }
              if (n >= kFragmentSharedThreshold) {
                  // Large fragment: back it with a block-shared array instead of a
                  // per-thread local array.  Ops on it use partition loops (one
                  // compute per element across the block) and the shared barrier
                  // discipline; see _is_fragment_shared_backed.
                  st.shared = _fb->shared(Type::array(elem_t, n));
              } else {
                  st.fragment = _fb->local(Type::array(elem_t, n));
              }
              st.array_size = n;
              break;
          }
        }
        _storage_by_ptr[t] = st;
        auto name = t->name();
        if (!name.empty()) { _storage_by_name[luisa::string{name}] = st; }
        _storage_by_layout.emplace_back(
            std::pair<Layout, Storage>{
                Layout{t->scope(), t->dtype(),
                       luisa::fixed_vector<int32_t, 4>{t->dims().begin(), t->dims().end()}},
                st});
    }

    void _emit_clear(const ClearStmt *s) {
        auto *t = s->t();
        auto saved = _current_extent;
        _current_extent = t;
        auto zero = _zero_of(t->dtype());
      if (t->scope() == TensorScope::Fragment && !_is_fragment_shared_backed(t)) {
          _full_loop(t, [&](const Coord &c) { _write_to(t, c, zero); });
      } else {
          _partition_loop(t, [&](const Coord &c) { _write_to(t, c, zero); });
      }
      _current_extent = saved;
  }

  void _emit_copy(const CopyStmt *s) {
        auto *src = s->src();
        auto *dst = s->dst();
        auto *ext = op_extent_of(dst, src);
        auto saved = _current_extent;
        auto saved_axis = _pipeline_axis;
        _current_extent = ext;
        if (_pipeline_var != nullptr) {
            // per-copy pipeline axis: the axis with the smallest tile extent
            // (the K axis of a GEMM-style pipelined copy)
            _pipeline_axis = _min_extent_axis(ext);
        }
        auto body = [&](const Coord &c) {
            _write_to(dst, c, _value_at(src, c));
        };
          if (dst->scope() == TensorScope::Fragment && !_is_fragment_shared_backed(dst)) {
              if (src->scope() == TensorScope::Global) {
                  // coalesced global->fragment staging (lc_optimize 4): the block
                  // cooperatively streams the tile through shared memory instead
                  // of every thread redundantly re-reading the whole tile
                  auto elem_t = tensor_element_type(dst->dtype());
                  auto staging = _staging_for(dst, elem_t);
                  _sync_block();// staging write-after-read hazard
                  _partition_loop(ext, [&](const Coord &c) {
                      _fb->assign(_fb->access(elem_t, staging, _local_index(dst, c)),
                                  _maybe_cast(_value_at(src, c), elem_t));
                  });
                  _replicate_from_staging(dst, elem_t, staging);
              } else {
                  _full_loop(ext, body);
              }
          } else {
              _partition_loop(ext, body);
          }
        _pipeline_axis = saved_axis;
        _current_extent = saved;
    }

    void _emit_store(const TileStoreStmt *s) {
        auto *lhs = s->lhs();
        auto *ext = op_extent_of(lhs, s->rhs_tensor());
        auto saved = _current_extent;
        _current_extent = ext;
        auto body = [&](const Coord &c) {
            const Expression *rhs = nullptr;
            if (s->rhs_tensor() != nullptr) {
                rhs = _value_at(s->rhs_tensor(), c);
            } else if (s->rhs_literal() != nullptr) {
                rhs = _recreate_literal(s->rhs_literal());
            } else if (s->rhs_ref() != nullptr) {
                LUISA_ERROR_WITH_LOCATION(
                    "tile_to_kernel: runtime scalar (R3 RefExpr) tile-store "
                    "operands are not supported (the traced DSL records "
                    "host-side literals only).");
            } else {
                LUISA_ERROR_WITH_LOCATION("tile_to_kernel: tile-store without a right-hand side.");
            }
            if (s->op() == 1) {// lhs *= rhs (row-broadcast scale)
                auto lhs_t = tensor_element_type(lhs->dtype());
                rhs = _maybe_cast(rhs, lhs_t);
                rhs = _bin(BinaryOp::MUL, _value_at(lhs, c), rhs);
            }
            _write_to(lhs, c, rhs);
        };
          if (lhs->scope() == TensorScope::Fragment && !_is_fragment_shared_backed(lhs)) {
              _full_loop(ext, body);
          } else {
              _partition_loop(ext, body);
          }
        _current_extent = saved;
    }

    void _emit_binary(const TileBinaryStmt *s) {
        auto *lhs = s->lhs();
        auto op = s->op();
        auto elem_t = tensor_element_type(lhs->dtype());
        auto temp = _tile->temp_output(s);
        _temps[temp] = TempValue{
            lhs->dtype(),
            [this, s, op, elem_t](const Coord &c) -> const Expression * {
                auto l = _value_at(s->lhs(), c);
                const Expression *r = nullptr;
                if (s->rhs_tensor() != nullptr) {
                    r = _value_at(s->rhs_tensor(), c);
                } else if (s->rhs_literal() != nullptr) {
                    r = _recreate_literal(s->rhs_literal());
                } else if (s->rhs_ref() != nullptr) {
                    LUISA_ERROR_WITH_LOCATION(
                        "tile_to_kernel: runtime scalar (R3 RefExpr) binary "
                        "operands are not supported.");
                }
                r = _maybe_cast(r, elem_t);
                switch (op) {
                    case BinaryOp::ADD: return _bin(BinaryOp::ADD, l, r);
                    case BinaryOp::SUB: return _bin(BinaryOp::SUB, l, r);
                    case BinaryOp::MUL: return _bin(BinaryOp::MUL, l, r);
                    case BinaryOp::DIV: return _bin(BinaryOp::DIV, l, r);
                    case BinaryOp::MOD: return _bin(BinaryOp::MOD, l, r);
                    case BinaryOp::BIT_AND: return _bin(BinaryOp::BIT_AND, l, r);
                    case BinaryOp::BIT_OR: return _bin(BinaryOp::BIT_OR, l, r);
                    case BinaryOp::BIT_XOR: return _bin(BinaryOp::BIT_XOR, l, r);
                    default:
                        LUISA_ERROR_WITH_LOCATION(
                            "tile_to_kernel: unsupported tile binary op {}.",
                            static_cast<uint32_t>(op));
                }
                return nullptr;// unreachable
            }};
    }

    void _emit_max(const MaxStmt *s) {
        auto *a = s->a();
        auto elem_t = tensor_element_type(a->dtype());
        _temps[_tile->temp_output(s)] = TempValue{
            a->dtype(),
            [this, s, a, elem_t](const Coord &c) -> const Expression * {
                auto av = _value_at(a, c);
                auto bv = _maybe_cast(_recreate_literal(s->b()), elem_t);
                return _fb->call(elem_t, CallOp::MAX, {av, bv});
            }};
    }

    void _emit_rsqrt(const RsqrtStmt *s) {
        auto *a = s->a();
        auto elem_t = tensor_element_type(a->dtype());
        _temps[_tile->temp_output(s)] = TempValue{
            a->dtype(),
            [this, s, a, elem_t](const Coord &c) -> const Expression * {
                auto av = _value_at(a, c);
                return _fb->call(elem_t, CallOp::RSQRT, {av});
            }};
    }

    // Software erf. There is no CallOp::ERF and no portable backend support
    // for an "erf" external function, so build the Abramowitz & Stegun 7.1.26
    // approximation from ops every backend lowers:
    //
    //   erf(x) = sign(x) * (1 - exp(-x^2) * P(t)),  t = 1 / (1 + p*|x|)
    //   P(t) = a1*t + a2*t^2 + a3*t^3 + a4*t^4 + a5*t^5
    //
    // (p = 0.3275911, a1..a5 below; max absolute error < 1.5e-7).  The math
    // is evaluated in f32 and cast back to the requested element type so the
    // approximation is not degraded by half-precision arithmetic.
    [[nodiscard]] const Expression *_erf(const Expression *x, const Type *result_t) const {
        auto f32 = Type::of<float>();
        auto xf = _maybe_cast(x, f32);
        auto zero = _fb->literal(f32, 0.f);
        auto one = _fb->literal(f32, 1.f);
        // t = 1 / (1 + p*|x|)
        auto absx = _fb->call(f32, CallOp::ABS, {xf});
        auto p = _fb->literal(f32, 0.3275911f);
        auto denom = _fb->binary(f32, BinaryOp::ADD, one,
                                 _fb->binary(f32, BinaryOp::MUL, p, absx));
        auto t = _fb->binary(f32, BinaryOp::DIV, one, denom);
        // Horner form of a1*t + a2*t^2 + a3*t^3 + a4*t^4 + a5*t^5
        auto a5 = _fb->literal(f32, 1.061405429f);
        auto a4 = _fb->literal(f32, -1.453152027f);
        auto a3 = _fb->literal(f32, 1.421413741f);
        auto a2 = _fb->literal(f32, -0.284496736f);
        auto a1 = _fb->literal(f32, 0.254829592f);
        const Expression *poly = a5;
        poly = _fb->binary(f32, BinaryOp::ADD,
                           _fb->binary(f32, BinaryOp::MUL, poly, t), a4);
        poly = _fb->binary(f32, BinaryOp::ADD,
                           _fb->binary(f32, BinaryOp::MUL, poly, t), a3);
        poly = _fb->binary(f32, BinaryOp::ADD,
                           _fb->binary(f32, BinaryOp::MUL, poly, t), a2);
        poly = _fb->binary(f32, BinaryOp::ADD,
                           _fb->binary(f32, BinaryOp::MUL, poly, t), a1);
        poly = _fb->binary(f32, BinaryOp::MUL, poly, t);
        // exp(-x^2)
        auto x2 = _fb->binary(f32, BinaryOp::MUL, absx, absx);
        auto neg_x2 = _fb->unary(f32, UnaryOp::MINUS, x2);
        auto e = _fb->call(f32, CallOp::EXP, {neg_x2});
        auto erf_abs = _fb->binary(f32, BinaryOp::SUB, one,
                                   _fb->binary(f32, BinaryOp::MUL, poly, e));
        // sign(x) * erf(|x|): (x < 0) ? -erf_abs : erf_abs
        auto is_neg = _fb->binary(Type::of<bool>(), BinaryOp::LESS, xf, zero);
        auto neg_erf_abs = _fb->unary(f32, UnaryOp::MINUS, erf_abs);
        auto result = _fb->call(f32, CallOp::SELECT,
                                {erf_abs, neg_erf_abs, is_neg});
        return _maybe_cast(result, result_t);
    }

    void _emit_fast_math(const FastMathStmt *s) {
        auto *a = s->a();
        auto elem_t = tensor_element_type(a->dtype());
        _temps[_tile->temp_output(s)] = TempValue{
            a->dtype(),
            [this, s, a, elem_t](const Coord &c) -> const Expression * {
                auto av = _value_at(a, c);
                switch (s->op()) {
                    case TileFastMathOp::EXP: return _fb->call(elem_t, CallOp::EXP, {av});
                    case TileFastMathOp::EXP10: return _fb->call(elem_t, CallOp::EXP10, {av});
                    case TileFastMathOp::LOG: return _fb->call(elem_t, CallOp::LOG, {av});
                    case TileFastMathOp::LOG2: return _fb->call(elem_t, CallOp::LOG2, {av});
                    case TileFastMathOp::LOG10: return _fb->call(elem_t, CallOp::LOG10, {av});
                    case TileFastMathOp::SIN: return _fb->call(elem_t, CallOp::SIN, {av});
                    case TileFastMathOp::COS: return _fb->call(elem_t, CallOp::COS, {av});
                    case TileFastMathOp::TAN: return _fb->call(elem_t, CallOp::TAN, {av});
                    case TileFastMathOp::TANH: return _fb->call(elem_t, CallOp::TANH, {av});
                    case TileFastMathOp::ERF: return _erf(av, elem_t);
                    default:
                        LUISA_ERROR_WITH_LOCATION(
                            "tile_to_kernel: unsupported fast math op {}.",
                            static_cast<uint32_t>(s->op()));
                }
            }};
    }

    void _emit_ieee_math(const IeeeMathStmt *s) {
        auto *a = s->a();
        auto *b = s->b();
        auto elem_t = tensor_element_type(a->dtype());
        // For CAST, the result type is the cast target dtype.
        auto result_dtype = (s->op() == TileIeeeOp::CAST) ? s->cast_dtype() : a->dtype();
        auto result_elem_t = tensor_element_type(result_dtype);
        _temps[_tile->temp_output(s)] = TempValue{
            result_dtype,
            [this, s, a, b, elem_t, result_elem_t](const Coord &c) -> const Expression * {
                auto av = _value_at(a, c);
                switch (s->op()) {
                    case TileIeeeOp::SQRT:
                    case TileIeeeOp::FSQRT:
                        return _fb->call(elem_t, CallOp::SQRT, {av});
                    case TileIeeeOp::POW: {
                        LUISA_ASSERT(b != nullptr,
                                     "tile_to_kernel: ieee POW requires a second "
                                     "input tensor (b).");
                        auto bv = _value_at(b, c);
                        return _fb->call(elem_t, CallOp::POW, {av, bv});
                    }
                    case TileIeeeOp::CEIL:
                        return _fb->call(elem_t, CallOp::CEIL, {av});
                    case TileIeeeOp::FLOOR:
                        return _fb->call(elem_t, CallOp::FLOOR, {av});
                    case TileIeeeOp::ROUND:
                        return _fb->call(elem_t, CallOp::ROUND, {av});
                    case TileIeeeOp::ISINF:
                    case TileIeeeOp::ISNAN: {
                        // ISINF/ISNAN produce a *boolean* predicate in the core
                        // IR (the XIR verifier rejects a float-typed result),
                        // so emit the call typed bool and cast back to the
                        // fragment's element type; downstream copies then cast
                        // to the destination dtype (e.g. int32) via _maybe_cast.
                        auto pred = _fb->call(
                            Type::of<bool>(),
                            s->op() == TileIeeeOp::ISINF ? CallOp::ISINF : CallOp::ISNAN,
                            {av});
                        return _fb->cast(elem_t, CastOp::STATIC, pred);
                    }
                    case TileIeeeOp::CAST:
                        // Use a CastExpr to convert the value to the target type.
                        return _fb->cast(result_elem_t, CastOp::STATIC, av);
                    default:
                        LUISA_ERROR_WITH_LOCATION(
                            "tile_to_kernel: unsupported ieee math op {}.",
                            static_cast<uint32_t>(s->op()));
                }
            }};
    }

    // warp-collective tile reduction (lc_optimize 3.2): output elements are
    // assigned to whole warps; lanes stride the reduce axis with an
    // identity-padded guard, a built-in warp all-reduce combines the partials
    // (no shared memory, no barrier), and lane 0 writes the result.
    // Fragment outputs (replicated layout) are published through a shared
    // staging tile and then re-replicated into every thread's local copy.
    void _emit_tile_reduce(const TensorExpr *x, const TensorExpr *y,
                           uint32_t dim, TileReduceOp op) {
        auto elem_t = tensor_element_type(x->dtype());
        auto saved = _current_extent;
        _current_extent = x;
        auto reduce_len = static_cast<uint32_t>(axis_extent(x, dim));
        // output space = x without the reduce axis
        uint32_t out_count = 1u;
        for (auto i = 0u; i < x->rank(); ++i) {
            if (i != dim) { out_count *= static_cast<uint32_t>(axis_extent(x, i)); }
        }
        auto lanes = _lane_count();
        auto lane = _lane();
        auto warp = _warp_id();
        auto nw = _num_warps();
        auto k_iters = _ceildiv_expr(_literal_u(reduce_len), lanes);
        auto o_iters = _ceildiv_expr(_literal_u(out_count), nw);
        const bool frag_out = y->scope() == TensorScope::Fragment;
        auto out_t = tensor_element_type(y->dtype());
        const RefExpr *staging = frag_out ? _staging_for(y, out_t) : nullptr;
        if (frag_out) { _sync_block(); }// staging write-after-read hazard
        _for_range(_literal_u(0u), o_iters, _literal_u(1u),
                   [&](const Expression *oi) {
            auto t1 = _fb->binary(Type::of<uint>(), BinaryOp::MUL, oi, nw);
            auto o = _fb->binary(Type::of<uint>(), BinaryOp::ADD, t1, warp);
            auto o_valid = _fb->binary(Type::of<bool>(), BinaryOp::LESS,
                                       o, _literal_u(out_count));
            _if(o_valid, [&] {
                // decompose o over x's shape minus the reduce axis
                Coord xc = _zero_coord();
                Coord yc = _zero_coord();
                auto rem = o;
                for (int32_t i = static_cast<int32_t>(x->rank()) - 1; i >= 0; --i) {
                    auto ui = static_cast<uint32_t>(i);
                    if (ui == dim) { continue; }
                    auto e = _literal_u(static_cast<uint32_t>(axis_extent(x, ui)));
                    auto ci = _fb->binary(Type::of<uint>(), BinaryOp::MOD, rem, e);
                    rem = _fb->binary(Type::of<uint>(), BinaryOp::DIV, rem, e);
                    xc[ui] = ci;
                    yc[ui < dim ? ui : ui - 1u] = ci;
                }
                // per-lane partial over the strided reduce axis
                auto acc = _fb->local(elem_t);
                _fb->assign(acc, _reduce_identity(op, x->dtype()));
                _for_range(_literal_u(0u), k_iters, _literal_u(1u),
                           [&](const Expression *ki) {
                    auto t2 = _fb->binary(Type::of<uint>(), BinaryOp::MUL, ki, lanes);
                    auto k = _fb->binary(Type::of<uint>(), BinaryOp::ADD, t2, lane);
                    auto v = _fb->local(elem_t);
                    _fb->assign(v, _reduce_identity(op, x->dtype()));
                    auto k_valid = _fb->binary(Type::of<bool>(), BinaryOp::LESS,
                                               k, _literal_u(reduce_len));
                    _if(k_valid, [&] {
                        xc[dim] = k;
                        auto xv = _maybe_cast(_value_at(x, xc), elem_t);
                        if (op == TileReduceOp::ABS_SUM || op == TileReduceOp::ABS_MAX) {
                            xv = _fb->call(elem_t, CallOp::ABS, {xv});
                        }
                        _fb->assign(v, xv);
                    });
                    _fb->assign(acc, _reduce_combine(op, elem_t, acc, v));
                });
                auto total = _warp_reduce(op, elem_t, acc);
                auto is_lane0 = _fb->binary(Type::of<bool>(), BinaryOp::EQUAL,
                                            lane, _literal_u(0u));
                _if(is_lane0, [&] {
                    if (frag_out) {
                        _fb->assign(_fb->access(out_t, staging, _local_index(y, yc)),
                                    _maybe_cast(total, out_t));
                    } else {
                        _write_to(y, yc, total);
                    }
                });
            });
        });
        if (frag_out) { _replicate_from_staging(y, out_t, staging); }
        _current_extent = saved;
    }

    void _emit_reduce_sum(const ReduceSumStmt *s) {
        _emit_tile_reduce(s->x(), s->y(), s->dim(), TileReduceOp::SUM);
    }

    void _emit_gemm(const GemmStmt *s) {
        if (_use_cooperative) {
            _emit_gemm_cooperative(s);
            return;
        }
        auto *a = s->a();
        auto *b = s->b();
        auto *c = s->c();
        auto wide_t = Type::of<float>();// f16 inputs accumulate in f32
        auto bk = static_cast<uint32_t>(axis_extent(a, 1u));// K extent
        auto saved = _current_extent;
        _current_extent = c;
        // SIMT partition (lc_optimize): each thread owns a strided slice of the
        // C tile and accumulates it in a register; the K-loop reads the
        // (usually shared) A/B tiles.  Previously every thread redundantly
        // computed the WHOLE C tile (_full_loop), i.e. threads x more work.
        auto compute_acc = [&](const Coord &cc) -> const Expression * {
            auto r = cc[0];
            auto n = cc[1];
            auto acc = _fb->local(wide_t);
            if (s->clear_accum() != 0) {
                _fb->assign(acc, _fb->literal(wide_t, 0.f));
            } else {
                _fb->assign(acc, _maybe_cast(_value_at(c, cc), wide_t));
            }
            _for_range(_literal_u(0u), _literal_u(bk), _literal_u(1u),
                       [&](const Expression *k) {
                auto ac = _zero_coord();
                auto bc = _zero_coord();
                if (s->trans_a() != 0) {
                    ac[0] = k;
                    ac[1] = r;
                } else {
                    ac[0] = r;
                    ac[1] = k;
                }
                if (s->trans_b() != 0) {
                    bc[0] = n;
                    bc[1] = k;
                } else {
                    bc[0] = k;
                    bc[1] = n;
                }
                auto av = _maybe_cast(_value_at(a, ac), wide_t);
                auto bv = _maybe_cast(_value_at(b, bc), wide_t);
                _fb->assign(acc, _fb->call(wide_t, CallOp::FMA, {av, bv, acc}));
            });
            return acc;
        };
      if (c->scope() == TensorScope::Fragment && !_is_fragment_shared_backed(c)) {
          // fragment C is replicated: publish the partitioned results through
          // a shared staging tile, then refresh every thread's replica
          auto out_t = tensor_element_type(c->dtype());
          auto staging = _staging_for(c, out_t);
          _sync_block();// staging write-after-read hazard vs. previous use
          _partition_loop(c, [&](const Coord &cc) {
              auto acc = compute_acc(cc);
              _fb->assign(_fb->access(out_t, staging, _local_index(c, cc)),
                          _maybe_cast(acc, out_t));
          });
          _replicate_from_staging(c, out_t, staging);
      } else {
          // global C, or a large shared-backed fragment C: each element is
          // written exactly once across the block (the A/B tiles were staged
          // in shared memory and published by the copies' trailing barriers;
          // this statement's own writes are published by _accesses_shared's
          // trailing _sync_block() in _emit).
          _partition_loop(c, [&](const Coord &cc) {
              _write_to(c, cc, compute_acc(cc));
          });
      }
        _current_extent = saved;
    }

    // Cooperative-vector GEMM (TileToKernelConfig::use_cooperative): the
    // whole-tile matrix multiply is computed with cooperative vectors instead
    // of the per-thread FMA loop above.  For every row r of the MxK A tile
    // the K-loop becomes
    //     acc[0:N] += splat(A[r][k]) * B[k][0:N]
    // where the accumulator and the two operand rows are cooperative vectors
    // (Type::cooperative_vector), the A scalar is broadcast with
    // COOPERATIVE_VECTOR_SPLAT and the multiply-accumulate is the elementwise
    // FMA expansion over cooperative-vector components — exactly what the DSL
    // of <luisa/dsl/coop_vector.h> + cooperative_vector_splat / _fma
    // (<luisa/dsl/resource.h>) emits.  Loads/stores between the cooperative
    // vectors and the shared/fragment tiles go through per-component access
    // (COOPERATIVE_VECTOR_WORKGROUP_LOAD/_STORE cannot bind shared arrays on
    // current backends, and COOPERATIVE_VECTOR_LOAD/_STORE require a byte
    // buffer, which a shared/fragment tile is not).
    void _emit_gemm_cooperative(const GemmStmt *s) {
        auto *a = s->a();
        auto *b = s->b();
        auto *c = s->c();
        // ---- constraint checks: fail loudly instead of mis-lowering --------
        if (a->rank() != 2u || b->rank() != 2u || c->rank() != 2u) [[unlikely]] {
            LUISA_ERROR_WITH_LOCATION(
                "tile_to_kernel: cooperative GEMM requires rank-2 tiles "
                "(got ranks {}/{}/{}).",
                a->rank(), b->rank(), c->rank());
        }
        auto in_e = a->dtype();
        auto out_e = c->dtype();
        auto is_coop_dtype = [](TensorElementType e) noexcept {
            return e == TensorElementType::F16 || e == TensorElementType::F32;
        };
        if (!is_coop_dtype(in_e) || b->dtype() != in_e || !is_coop_dtype(out_e)) [[unlikely]] {
            LUISA_ERROR_WITH_LOCATION(
                "tile_to_kernel: cooperative GEMM requires F16/F32 tiles with "
                "matching input dtypes (got a={}, b={}, c={}).",
                tensor_element_type_name(in_e), tensor_element_type_name(b->dtype()),
                tensor_element_type_name(out_e));
        }
        if (s->trans_a() != 0 || s->trans_b() != 0) [[unlikely]] {
            LUISA_ERROR_WITH_LOCATION(
                "tile_to_kernel: cooperative GEMM requires non-transposed "
                "row-major operand tiles (trans_a={}, trans_b={}).",
                s->trans_a(), s->trans_b());
        }
        auto &sa = _storage_for(a);
        auto &sb = _storage_for(b);
        auto &sc = _storage_for(c);
        static_cast<void>(sa);
        if (sb.scope == TensorScope::Global ||
            sc.scope == TensorScope::Global) [[unlikely]] {
            LUISA_ERROR_WITH_LOCATION(
                "tile_to_kernel: cooperative GEMM requires shared/fragment "
                "operand tiles (cooperative vectors stage through local/shared "
                "storage, not global buffers).");
        }
        auto M = static_cast<uint32_t>(axis_extent(c, 0u));
        auto N = static_cast<uint32_t>(axis_extent(c, 1u));
        auto K = static_cast<uint32_t>(axis_extent(a, 1u));
        if (static_cast<uint32_t>(axis_extent(a, 0u)) != M ||
            static_cast<uint32_t>(axis_extent(b, 0u)) != K ||
            static_cast<uint32_t>(axis_extent(b, 1u)) != N) [[unlikely]] {
            LUISA_ERROR_WITH_LOCATION(
                "tile_to_kernel: cooperative GEMM tile shape mismatch "
                "(a={}x{}, b={}x{}, c={}x{}).",
                axis_extent(a, 0u), K, K, axis_extent(b, 1u), M, N);
        }
        auto saved = _current_extent;
        _current_extent = c;
        auto wide_t = Type::of<float>();// accumulate in f32 (f16 inputs too)
        auto in_t = tensor_element_type(in_e);
        auto out_t = tensor_element_type(out_e);
        auto acc_vec_t = Type::cooperative_vector(wide_t, N);
        auto in_vec_t = Type::cooperative_vector(in_t, N);
        // component access / assignment of a cooperative vector (the AST-level
        // form of the DSL CoopVector::operator[])
        auto vec_at = [&](const Expression *v, const Type *elem, uint32_t i) {
            return _fb->access(elem, v, _literal_u(i));
        };
        // uniform (non-partitioned) row loop: every thread of the block
        // executes the same cooperative ops, block-wide
        _for_range(_literal_u(0u), _literal_u(M), _literal_u(1u),
                   [&](const Expression *r) {
            auto acc = _fb->local(acc_vec_t);
            if (s->clear_accum() != 0) {
                _fb->assign(acc, _fb->call(acc_vec_t, CallOp::COOPERATIVE_VECTOR_SPLAT,
                                           {_fb->literal(wide_t, 0.f)}));
            } else {
                // load the existing C row into the accumulator
                for (auto i = 0u; i < N; ++i) {
                    Coord cc = _zero_coord();
                    cc[0] = r;
                    cc[1] = _literal_u(i);
                    _fb->assign(vec_at(acc, wide_t, i),
                                _maybe_cast(_value_at(c, cc), wide_t));
                }
            }
            _for_range(_literal_u(0u), _literal_u(K), _literal_u(1u),
                       [&](const Expression *k) {
                // broadcast A[r][k] into a cooperative vector
                Coord ac = _zero_coord();
                ac[0] = r;
                ac[1] = k;
                auto av = _maybe_cast(_value_at(a, ac), wide_t);
                auto a_vec = _fb->local(acc_vec_t);
                _fb->assign(a_vec, _fb->call(acc_vec_t, CallOp::COOPERATIVE_VECTOR_SPLAT, {av}));
                // stage B[k][0:N] into a cooperative vector
                auto b_vec = _fb->local(in_vec_t);
                for (auto i = 0u; i < N; ++i) {
                    Coord bc = _zero_coord();
                    bc[0] = k;
                    bc[1] = _literal_u(i);
                    _fb->assign(vec_at(b_vec, in_t, i), _value_at(b, bc));
                }
                // acc += a_vec * b_vec (elementwise FMA expansion, as the DSL
                // cooperative_vector_fma emits)
                for (auto i = 0u; i < N; ++i) {
                    auto ai = vec_at(a_vec, wide_t, i);
                    auto bi = _maybe_cast(vec_at(b_vec, in_t, i), wide_t);
                    auto ci = vec_at(acc, wide_t, i);
                    _fb->assign(ci, _fb->call(wide_t, CallOp::FMA, {ai, bi, ci}));
                }
            });
            // store the accumulated row back into the C tile
            for (auto i = 0u; i < N; ++i) {
                Coord cc = _zero_coord();
                cc[0] = r;
                cc[1] = _literal_u(i);
                _write_to(c, cc, _maybe_cast(vec_at(acc, wide_t, i), out_t));
            }
        });
        _current_extent = saved;
    }

    void _emit_print(const TilePrintStmt *s) {
        auto *t = s->t();
        auto saved = _current_extent;
        _current_extent = t;
        auto c0 = _zero_coord();
        auto tid_x = _vec_comp(_fb->thread_id(), 0u);
        auto cond = _fb->binary(Type::of<bool>(), BinaryOp::EQUAL, tid_x, _literal_u(0u));
        _if(cond, [&] {
            auto v = _value_at(t, c0);
            auto fmt = luisa::format("[tile] {} tile[0] = {{}}", luisa::string{s->msg()});
            _fb->print_(fmt, luisa::span<const Expression *const>{&v, 1u});
        });
        _current_extent = saved;
    }

    void _emit_fill(const FillStmt *s) {
        auto *buf = s->buf();
        auto saved = _current_extent;
        _current_extent = buf;
        const Expression *value = nullptr;
        if (s->value_literal() != nullptr) {
            value = _recreate_literal(s->value_literal());
        } else if (s->value_ref() != nullptr) {
            LUISA_ERROR_WITH_LOCATION("tile_to_kernel: R3 RefExpr fill values are not supported.");
        }
      if (buf->scope() == TensorScope::Fragment && !_is_fragment_shared_backed(buf)) {
          _full_loop(buf, [&](const Coord &c) { _write_to(buf, c, value); });
      } else {
          _partition_loop(buf, [&](const Coord &c) { _write_to(buf, c, value); });
        }
        _current_extent = saved;
    }

    void _emit_transpose(const TransposeStmt *s) {
        auto *src = s->src();
        auto *dst = s->dst();
        auto *ext = op_extent_of(dst, src);
        auto saved = _current_extent;
        _current_extent = ext;
        auto body = [&](const Coord &cd) {
            auto cs = _zero_coord();
            cs[0] = cd[1];
            cs[1] = cd[0];
            _write_to(dst, cd, _value_at(src, cs));
        };
        if (dst->scope() == TensorScope::Fragment && !_is_fragment_shared_backed(dst)) {
            _full_loop(ext, body);
        } else {
            _partition_loop(ext, body);
        }
        _current_extent = saved;
    }

    void _emit_clamp(const ClampStmt *s) {
        auto *dst = s->dst();
        auto elem_t = tensor_element_type(dst->dtype());
        auto saved = _current_extent;
        _current_extent = dst;
        auto body = [&](const Coord &c) {
            auto v = _value_at(dst, c);
            const Expression *lo = nullptr;
            const Expression *hi = nullptr;
            if (s->lo_literal() != nullptr) { lo = _maybe_cast(_recreate_literal(s->lo_literal()), elem_t); }
            if (s->hi_literal() != nullptr) { hi = _maybe_cast(_recreate_literal(s->hi_literal()), elem_t); }
            auto clamped = v;
            if (lo != nullptr) { clamped = _fb->call(elem_t, CallOp::MAX, {clamped, lo}); }
            if (hi != nullptr) { clamped = _fb->call(elem_t, CallOp::MIN, {clamped, hi}); }
            _write_to(dst, c, clamped);
        };
        if (dst->scope() == TensorScope::Fragment && !_is_fragment_shared_backed(dst)) {
            _full_loop(dst, body);
        } else {
            _partition_loop(dst, body);
        }
        _current_extent = saved;
    }

    void _emit_atomic(const AtomicStmt *s) {
        auto *dst = s->dst();
        auto &st = _storage_for(dst);
        auto elem_t = tensor_element_type(st.dtype);
        auto saved = _current_extent;
        _current_extent = dst;
        const Expression *value = nullptr;
        if (s->value_tensor() != nullptr) {
            value = _value_at(s->value_tensor(), _zero_coord());
        } else if (s->value_literal() != nullptr) {
            value = _recreate_literal(s->value_literal());
        } else if (s->value_ref() != nullptr) {
            LUISA_ERROR_WITH_LOCATION("tile_to_kernel: R3 RefExpr atomic values are not supported.");
        }
        // atomic_load has no value operand; only cast when one exists.
        value = value != nullptr ? _maybe_cast(value, elem_t) : nullptr;
        auto body = [&](const Coord &c) {
            auto idx = _global_index(dst, c);
            // execute the atomic; the returned old value (if any) is captured
            // into a throw-away local so the call is type-correct and alive
            auto tmp = _fb->local(elem_t);
            switch (s->op()) {
                case TileAtomicOp::ADD:
                    _fb->assign(tmp, _fb->call(elem_t, CallOp::ATOMIC_FETCH_ADD, {st.buffer, idx, value}));
                    break;
                case TileAtomicOp::MAX:
                    _fb->assign(tmp, _fb->call(elem_t, CallOp::ATOMIC_FETCH_MAX, {st.buffer, idx, value}));
                    break;
                case TileAtomicOp::MIN:
                    _fb->assign(tmp, _fb->call(elem_t, CallOp::ATOMIC_FETCH_MIN, {st.buffer, idx, value}));
                    break;
                case TileAtomicOp::OR:
                    _fb->assign(tmp, _fb->call(elem_t, CallOp::ATOMIC_FETCH_OR, {st.buffer, idx, value}));
                    break;
                case TileAtomicOp::LOAD:
                    _fb->assign(tmp, _fb->call(elem_t, CallOp::BUFFER_VOLATILE_READ, {st.buffer, idx}));
                    break;
                case TileAtomicOp::STORE:
                    _fb->call(CallOp::BUFFER_VOLATILE_WRITE, {st.buffer, idx, value});
                    break;
                default:
                    LUISA_ERROR_WITH_LOCATION(
                        "tile_to_kernel: unsupported tile atomic op {}.",
                        static_cast<uint32_t>(s->op()));
            }
        };
        _partition_loop(dst, body);
        _current_extent = saved;
    }

    void _emit_sync(const SyncStmt *s) {
        switch (s->op()) {
            case TileSyncOp::THREADS: _sync_block(); break;
            case TileSyncOp::WARP: break;// warp ops are implicitly synchronized
            default:
                LUISA_ERROR_WITH_LOCATION(
                    "tile_to_kernel: sync op {} (grid/global) has no regular-kernel "
                    "equivalent (see the plan's gap list).",
                    static_cast<uint32_t>(s->op()));
        }
    }

    // generic reduce family: T.reduce_max / reduce_min / reduce_abssum /
    // reduce_absmax / reduce_bitand / reduce_bitor / reduce_bitxor
    void _emit_reduce(const ReduceStmt *s) {
        _emit_tile_reduce(s->buf(), s->out(), s->dim(), s->op());
    }

    // inclusive prefix scan along `dim`: T.cumsum / T.cummax (src, dst, dim, reverse)
    // Warp-collective scan (lc_optimize 3.4): each warp owns whole scan lines;
    // lanes load one element per chunk, WARP_PREFIX_SUM (or a WARP_READ_LANE
    // butterfly for max) produces the in-chunk inclusive scan and a running
    // carry stitches the chunks together.  Replaces the previous O(n^2)
    // per-element re-accumulation.
    void _emit_scan(const TensorExpr *src, const TensorExpr *dst,
                    uint32_t dim, int32_t reverse, bool is_max) {
        auto elem_t = tensor_element_type(src->dtype());
        auto saved = _current_extent;
        _current_extent = src;
        auto scan_len = static_cast<uint32_t>(axis_extent(src, dim));
        uint32_t line_count = 1u;
        for (auto i = 0u; i < src->rank(); ++i) {
            if (i != dim) { line_count *= static_cast<uint32_t>(axis_extent(src, i)); }
        }
        auto lanes = _lane_count();
        auto lane = _lane();
        auto warp = _warp_id();
        auto nw = _num_warps();
        auto chunks = _ceildiv_expr(_literal_u(scan_len), lanes);
        auto line_iters = _ceildiv_expr(_literal_u(line_count), nw);
        const bool frag_out = dst->scope() == TensorScope::Fragment;
        auto out_t = tensor_element_type(dst->dtype());
        const RefExpr *staging = frag_out ? _staging_for(dst, out_t) : nullptr;
        if (frag_out) { _sync_block(); }// staging write-after-read hazard
        auto identity = [&] {
            return _reduce_identity(is_max ? TileReduceOp::MAX : TileReduceOp::SUM,
                                    src->dtype());
        };
        _for_range(_literal_u(0u), line_iters, _literal_u(1u),
                   [&](const Expression *li) {
            auto t1 = _fb->binary(Type::of<uint>(), BinaryOp::MUL, li, nw);
            auto line = _fb->binary(Type::of<uint>(), BinaryOp::ADD, t1, warp);
            auto line_valid = _fb->binary(Type::of<bool>(), BinaryOp::LESS,
                                          line, _literal_u(line_count));
            _if(line_valid, [&] {
                // decompose the line index over src's shape minus the scan axis
                Coord cc = _zero_coord();
                auto rem = line;
                for (int32_t i = static_cast<int32_t>(src->rank()) - 1; i >= 0; --i) {
                    auto ui = static_cast<uint32_t>(i);
                    if (ui == dim) { continue; }
                    auto e = _literal_u(static_cast<uint32_t>(axis_extent(src, ui)));
                    auto ci = _fb->binary(Type::of<uint>(), BinaryOp::MOD, rem, e);
                    rem = _fb->binary(Type::of<uint>(), BinaryOp::DIV, rem, e);
                    cc[ui] = ci;
                }
                auto carry = _fb->local(elem_t);
                _fb->assign(carry, identity());
                _for_range(_literal_u(0u), chunks, _literal_u(1u),
                           [&](const Expression *ch) {
                    auto t2 = _fb->binary(Type::of<uint>(), BinaryOp::MUL, ch, lanes);
                    auto off = _fb->binary(Type::of<uint>(), BinaryOp::ADD, t2, lane);
                    // element position along the scan axis (from the scan side)
                    const Expression *pos = off;
                    if (reverse != 0) {
                        pos = _fb->binary(Type::of<uint>(), BinaryOp::SUB,
                                          _literal_u(scan_len - 1u), off);
                    }
                    auto valid = _fb->binary(Type::of<bool>(), BinaryOp::LESS,
                                             off, _literal_u(scan_len));
                    auto v = _fb->local(elem_t);
                    _fb->assign(v, identity());
                    _if(valid, [&] {
                        cc[dim] = pos;
                        _fb->assign(v, _maybe_cast(_value_at(src, cc), elem_t));
                    });
                    // in-chunk inclusive scan across the warp: butterfly
                    // inclusive scan via WARP_READ_LANE (lc_optimize 2.2; the
                    // lane read is unconditional/clamped so it is never
                    // divergent — the built-in WARP_PREFIX_SUM miscompiles in
                    // this nested control flow on some backends)
                    auto incl = _fb->local(elem_t);
                    _fb->assign(incl, v);
                    for (uint32_t d = 1u; d <= 64u; d <<= 1u) {
                        auto d_active = _fb->binary(Type::of<bool>(), BinaryOp::LESS,
                                                    _literal_u(d), lanes);
                        _if(d_active, [&] {
                            auto clamped = _fb->call(Type::of<uint>(), CallOp::MIN,
                                                     {lane, _literal_u(d)});
                            auto peer = _fb->binary(Type::of<uint>(), BinaryOp::SUB,
                                                    lane, clamped);
                            // the wave read must stay UNCONDITIONAL (a
                            // divergent wave intrinsic is UB): stage it in a
                            // local and guard only the combine step
                            auto other = _fb->local(elem_t);
                            _fb->assign(other, _fb->call(elem_t, CallOp::WARP_READ_LANE,
                                                         {incl, peer}));
                            auto has_prev = _fb->binary(Type::of<bool>(),
                                                        BinaryOp::GREATER_EQUAL,
                                                        lane, _literal_u(d));
                            _if(has_prev, [&] {
                                auto combined = is_max
                                                    ? static_cast<const Expression *>(_fb->call(elem_t, CallOp::MAX, {incl, other}))
                                                    : static_cast<const Expression *>(_fb->binary(elem_t, BinaryOp::ADD, incl, other));
                                _fb->assign(incl, combined);
                            });
                        });
                    }
                    // chunk total = the last lane's inclusive value
                    auto last = _fb->binary(Type::of<uint>(), BinaryOp::SUB,
                                            lanes, _literal_u(1u));
                    auto total = _fb->call(elem_t, CallOp::WARP_READ_LANE, {incl, last});
                    const Expression *res = is_max
                                                ? static_cast<const Expression *>(_fb->call(elem_t, CallOp::MAX, {carry, incl}))
                                                : static_cast<const Expression *>(_fb->binary(elem_t, BinaryOp::ADD, carry, incl));
                    _if(valid, [&] {
                        cc[dim] = pos;
                        if (frag_out) {
                            _fb->assign(_fb->access(out_t, staging, _local_index(dst, cc)),
                                        _maybe_cast(res, out_t));
                        } else {
                            _write_to(dst, cc, res);
                        }
                    });
                    const Expression *new_carry = is_max
                                                      ? static_cast<const Expression *>(_fb->call(elem_t, CallOp::MAX, {carry, total}))
                                                      : static_cast<const Expression *>(_fb->binary(elem_t, BinaryOp::ADD, carry, total));
                    _fb->assign(carry, new_carry);
                });
            });
        });
        if (frag_out) { _replicate_from_staging(dst, out_t, staging); }
        _current_extent = saved;
    }

    // logical tile reduction: T.any_of / T.all_of(buf); the scalar result has
    // no consumer in the tile IR, so it is folded into a throw-away local
    // (the same pattern as WARP_REDUCE).
    void _emit_any_all(const TensorExpr *buf, bool is_all) {
        auto elem_t = tensor_element_type(buf->dtype());
        auto saved = _current_extent;
        _current_extent = buf;
        auto acc = _fb->local(Type::of<bool>());
        _fb->assign(acc, _fb->literal(Type::of<bool>(), is_all));
        _full_loop(buf, [&](const Coord &c) {
            auto v = _value_at(buf, c);
            auto truth = _fb->binary(Type::of<bool>(), BinaryOp::NOT_EQUAL,
                                     v, _maybe_cast(_zero_of(buf->dtype()), elem_t));
            _fb->assign(acc, _fb->binary(Type::of<bool>(),
                                         is_all ? BinaryOp::AND : BinaryOp::OR, acc, truth));
        });
        // block-level vote keeps the folded value alive on every lane
        auto voted = _fb->call(Type::of<bool>(),
                               is_all ? CallOp::WARP_ACTIVE_ALL : CallOp::WARP_ACTIVE_ANY, {acc});
        auto tmp = _fb->local(Type::of<bool>());
        _fb->assign(tmp, voted);
        _current_extent = saved;
    }

    // warp shuffle of a fragment scalar: T.shfl_xor / shfl_up / shfl_down
    // (emulated with WARP_READ_LANE at the computed peer lane).
    void _emit_shuffle(const ShuffleStmt *s) {
        auto *v = s->value_tensor();
        if (v == nullptr) [[unlikely]] {
            LUISA_ERROR_WITH_LOCATION("tile_to_kernel: shuffle requires a fragment-tile value.");
        }
        auto elem_t = tensor_element_type(v->dtype());
        auto saved = _current_extent;
        _current_extent = v;
        auto val = _value_at(v, _zero_coord());
        auto lane = _fb->warp_lane_id();
        auto delta = _literal_u(static_cast<uint32_t>(s->delta()));
        const Expression *peer = nullptr;
        switch (s->op()) {
            case TileShuffleOp::XOR:
                peer = _fb->binary(Type::of<uint>(), BinaryOp::BIT_XOR, lane, delta);
                break;
            case TileShuffleOp::UP:
                peer = _fb->binary(Type::of<uint>(), BinaryOp::SUB, lane, delta);
                break;
            case TileShuffleOp::DOWN:
                peer = _fb->binary(Type::of<uint>(), BinaryOp::ADD, lane, delta);
                break;
            default:
                LUISA_ERROR_WITH_LOCATION(
                    "tile_to_kernel: shuffle op {} is not supported by the "
                    "regular-kernel lowering.",
                    static_cast<uint32_t>(s->op()));
        }
        auto tmp = _fb->local(elem_t);
        _fb->assign(tmp, _fb->call(elem_t, CallOp::WARP_READ_LANE, {val, peer}));
        _current_extent = saved;
    }

    void _emit_min(const MinStmt *s) {
        auto *a = s->a();
        auto elem_t = tensor_element_type(a->dtype());
        _temps[_tile->temp_output(s)] = TempValue{
            a->dtype(),
            [this, s, a, elem_t](const Coord &c) -> const Expression * {
                auto av = _value_at(a, c);
                auto bv = _maybe_cast(_recreate_literal(s->b()), elem_t);
                return _fb->call(elem_t, CallOp::MIN, {av, bv});
            }};
    }

    void _emit_abs(const AbsStmt *s) {
        auto *a = s->a();
        auto elem_t = tensor_element_type(a->dtype());
        _temps[_tile->temp_output(s)] = TempValue{
            a->dtype(),
            [this, a, elem_t](const Coord &c) -> const Expression * {
                return _fb->call(elem_t, CallOp::ABS, {_value_at(a, c)});
            }};
    }

    void _emit_warp_reduce(const WarpReduceStmt *s) {
        // register-level warp reduction; the IR has no consumer, so the value
        // is computed into a throw-away local to keep the call alive
        auto *v = s->value();
        auto elem_t = tensor_element_type(v->dtype());
        auto saved = _current_extent;
        _current_extent = v;
        auto val = _value_at(v, _zero_coord());
        auto tmp = _fb->local(elem_t);
        switch (s->op()) {
            case TileWarpReduceOp::SUM:
                _fb->assign(tmp, _fb->call(elem_t, CallOp::WARP_ACTIVE_SUM, {val}));
                break;
            case TileWarpReduceOp::MAX:
                _fb->assign(tmp, _fb->call(elem_t, CallOp::WARP_ACTIVE_MAX, {val}));
                break;
            case TileWarpReduceOp::MIN:
                _fb->assign(tmp, _fb->call(elem_t, CallOp::WARP_ACTIVE_MIN, {val}));
                break;
            case TileWarpReduceOp::BIT_AND:
                _fb->assign(tmp, _fb->call(elem_t, CallOp::WARP_ACTIVE_BIT_AND, {val}));
                break;
            case TileWarpReduceOp::BIT_OR:
                _fb->assign(tmp, _fb->call(elem_t, CallOp::WARP_ACTIVE_BIT_OR, {val}));
                break;
            default:
                LUISA_ERROR_WITH_LOCATION("tile_to_kernel: invalid tile warp-reduce op.");
        }
        _current_extent = saved;
    }
};

}// namespace

TileCompileResult tile_to_kernel(
    luisa::shared_ptr<const detail::TileFunctionBuilder> const &tile_function,
    TileToKernelConfig const &config) {
    if (tile_function == nullptr) [[unlikely]] {
        LUISA_ERROR_WITH_LOCATION("tile_to_kernel: null tile function.");
    }
    return TileLowerer{}.lower(tile_function, config);
}

}// namespace luisa::compute

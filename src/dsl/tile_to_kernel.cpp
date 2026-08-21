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

  For 2D tiles use the explicit 2D partition helper `_partition_loop_2d`
  (the block is 1D in x only, so the linear thread id is decomposed into a
  (r0, c0) start ONCE per thread and both axes are strided):
    UInt tid = thread_id().x;
    UInt tw  = min(threads, cols);            // threads along the fast axis
    UInt th  = ceildiv(threads, tw);          // threads along the slow axis
    UInt r0  = tid / tw;  UInt c0 = tid % tw; // hoisted, not per element
    $for (r, r0, rows, th)
      $for (c, c0, cols, tw) { body(r, c) }; // coalesced along c
  This removes the per-element div/mod decomposition (`_decompose`) for
  rank-2 tiles.  The 1D helper skips its `$if (idx < total)` guard when
  total % threads == 0 (compile-time known extents).  GEMM additionally
  partitions the C tile into TM x TN register micro-tiles (2.4).
  Vectorized (coalesced) variants iterate chunks of
  block_size().x * vector_width and use float4/half2 stores (see
  TileLang vectorize_loop + SelectMinPaddingVectorSize in op/parallel.cc);
  vectorization remains a documented future refinement (not emitted yet).

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
  so the REGULAR-kernel implementation is a software shared-memory GEMM with
  REGISTER TILING (the test_warp.cpp / test_softmax.cpp pattern, optimized):
    1. copy A/B tiles (or slices) into Shared (2.3);
    2. sync_block();
    3. per-thread TM x TN register micro-tile of C (e.g. 4x4 per thread):
       the C tile is partitioned into a grid of micro-tiles; each thread
       owns one (TM x TN) micro-tile per iteration and accumulates it in
       registers, so per k-step it loads TM A values + TN B values and
       issues TM*TN FMAs (2 LDS loads per FMA becomes (TM+TN)/(TM*TN));
    4. K loop with k_pack unrolling:
           $for (kk, 0, ceildiv(K, k_pack)) {
               $for (u, 0, k_pack) {           // host-unrolled
                   k = kk*k_pack + u; $if (k < K) {
                       acc[i][j] = fma(A_sh[r+i][k], B_sh[k][c+j], acc[i][j]);
                   };
               };
           };
    5. optionally accumulate in Float for F16 inputs (AccType rule from
       tl_templates/cuda/reduce.h);
    6. T.clear(c) (clear_accum) is emitted as CLEAR first (2.2);
    7. copy the register tile back to C (staging for replicated fragments).
  Thread → micro-tile mapping honors GemmWarpPolicy (Square = both dims
  split, FullRow = rows of C split across threads with the full N strip,
  FullCol = columns of C split across threads with the full M strip).
  Knobs:
    trans_a/trans_b -> swap indexing in step 4;
    GemmWarpPolicy   -> the micro-tile mapping above;
    k_pack           -> unroll factor for the inner K loop (step 4);
    mbar (Blackwell mbarrier input) -> ignore in the SIMT fallback (the
      wait is implied by sync_block) or keep for the async path (2.14).
  Warp-level GEMM path (implemented; lc_optimize §2.1/§2.6):
    When the per-thread mapping would leave lanes idle — the micro-tile
    grid is smaller than the block (MT*NT < threads), threads >= 32, the
    GEMM is not batched/cooperative, and K is large enough to amortize
    the reduction (K >= 256) — switch to a WARP-K-SPLIT:
      - each WARP owns one micro-tile at a time (warp-level 2D strided
        partition of the MT x NT grid with warp_id/num_warps);
      - lane l accumulates the TM x TN partial over its strided K-slice
        k = kk*lanes + l (tail-guarded when K % lanes != 0);
     *   // after the K loop ONE scalar warp_active_sum per micro-tile element
     *   // gives every lane the finished tile (no barrier, no shared memory for
     *   // the reduction). A packed vector warp_active_sum was prototyped but
     *   // hits a CUDA XIR codegen issue (make_vector prints as lc_make_ulong4),
     *   // so the portable per-component reduce is used (like _emit_warp_reduce).
      - write-back: non-fragment C -> lane 0 writes the TM x TN tile
        (each element once); fragment C with a single-warp block -> every
        lane writes the tile into its OWN replica (barrier-free, §3.7);
        fragment C with several warps -> lane 0 publishes into the shared
        staging tile + sync_block, as before (cross-warp exchange must
        stay in shared memory, §3.7 rule 5).
    k_pack is ignored in the warp path (the lane-strided K loop already
    gives each lane a strided sequence; documented — the old path keeps
    honoring k_pack).
  WGGMA_GEMM / TCGEN05_GEMM / TCGEN05_GEMM_BLOCKSCALED: same SIMT fallback
  (hardware WGMMA/TCGEN05 require new DSL builtins; see section 4 gap list).
  GEMM_SP / WGGMA_GEMM_SP / TCGEN05_GEMM_SP: TileLang sparse GEMM works on
  the compressed A_sparse + metadata E (2:4 sparsity). Regular-kernel plan:
    - dense SIMT fallback: decompress A_sparse via E (or treat as dense and
      let the caller pre-decompress); the metadata E layout determines the
      K-groups of 4 with 2 non-zeros;
    - otherwise identical to GEMM with A replaced by the decompressed tile.
  Deferred (documented, not implemented in this pass): true multi-buffered
  async pipelining of the K loop (2.16), lane-mapped fragment layouts (1.2),
  tensor-core/WGMMA paths (section 4 gap list).

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
  Optimization (implemented): when the OUTPUT space is tiny (fewer output
  elements than warps, e.g. a full-tile 1D reduction where out_count == 1),
  the default warp-per-output partition would leave most warps idle.  Use
  the two-level BLOCK reduction instead:
    - every warp reduces its own strided slice of the reduce axis with a
      warp collective (no barrier);
    - lane 0 of each warp writes its partial to a Shared<T> slot [warp];
    - sync_block();
    - warp 0 (or one warp) reduces the num_warps partials and writes out.
  This keeps the whole block busy for full-tile / few-output reductions.

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
  Optimization (implemented): when there are FEWER scan lines than warps
  (line_count < nw, e.g. a 1D full-tensor scan with threads > 32), the
  default warp-per-line partition would leave most warps idle.  Use the
  TWO-PASS BLOCK scan instead (plan step 3):
    - pass 1: each warp scans its contiguous segment of the scan axis
      (butterfly + intra-segment carry) and publishes the segment total;
    - sync_block();
    - pass 2: warp 0 scans the segment totals -> per-segment exclusive
      prefix;
    - sync_block();
    - pass 3: each warp recomputes its segment scan, combines every
      element with its exclusive prefix, and writes the output.
  Keeps the whole block busy for full-tile / few-line scans.  Segments
  are defined in the scan-axis (off) order, so reverse is honored by the
  existing per-element position mapping inside each segment.

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
  (1.3); rank-2 tiles use `_partition_loop_2d` to avoid per-element div/mod:
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
  ANY_OF / ALL_OF: tile → scalar boolean. For Global/Shared tiles the
  lowering partitions the tile across threads (one element per thread), each
  thread keeps a register `acc` (OR for any_of / AND for all_of), then the
  two-level block reduction reduces the per-thread partials (warp collective
  → Shared → block), so global buffers are read once per element instead of
  once per thread:
      any_of: local = any(elem != 0) then block-any;
      all_of: local = all(elem != 0) then block-all.
  Replicated Fragment tiles keep the per-thread `_full_loop` (each thread
  already owns the whole tile), followed by the warp vote.

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

  Optimization pass notes (implemented in this file):
  - GEMM uses TM x TN register micro-tiles + k_pack unroll + GemmWarpPolicy
    mapping (2.4); when the micro-tile grid is smaller than the block
    (MT*NT < threads), threads >= 32 and K >= 256 it switches to a
    warp-K-split GEMM: every lane of a warp accumulates a K-slice of a
    micro-tile and a scalar warp_active_sum all-reduce finishes the tile
    (no barrier; fragment single-warp blocks write replicas directly).
  - rank-2 elementwise loops use the div/mod-free `_partition_loop_2d`;
    REDUCE uses a block-wide two-level reduction for few-output
    reductions; ANY_OF/ALL_OF partition Global/Shared tiles; TRANSPOSE
    stages through Shared for Global operands; CUMSUM/CUMMAX use a
    two-pass block scan when there are fewer scan lines than warps.
  - Deferred (correct today, not optimized): true async multi-buffered
    PIPELINED (2.16), vectorized float4/half2 chunks (1.3), lane-mapped
    fragment layouts (1.2), tensor-core/WGMMA/TCGEN05 paths, packed
    addx2/addx4 atomics, per-thread atomic aggregation.

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
//                       by exactly one thread.  Rank-2 tiles use
//                       `_partition_loop_2d` (strided r/c loops, no div/mod).
//   * Shared  tiles  -> partitioned element loops + sync_block() after every
//                       statement that touches shared memory (never inside a
//                       thread-divergent branch).
//   * Fragment tiles -> replicated per-thread register arrays: every thread
//                       holds the whole tile and every fragment op runs a
//                       full (non-partitioned) element loop.  This is the
//                       "replicate" fragment layout of TileLang — simple and
//                       correct; a lane-mapped layout (warp_lane_id
//                       partitioning) is future work (plan 1.2).
  // * GEMM (2.4) partitions the C tile into TM x TN register micro-tiles
  // * instead of single elements, so each thread reuses A/B loads across a
  // * TM x TN FMA block; when the micro-tile grid is smaller than the
  // * block it additionally switches to a warp-K-split (every lane of a
  // * warp accumulates a K-slice and scalar warp_active_sum all-reduces
  // * finish each micro-tile — see 2.4).
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
//     (1D) or pipeline_count*E1 when axis 1 is the pipeline axis;
//   * with dynamic batching the z dispatch axis carries the runtime batch
//     count: each z-thread of a block owns one batch item, the per-block z
//     size is `_batch_block_z` (1 when disabled), and every Global access adds
//     `batch_index * volume(t)` (clamped to batch 0 for idle tz threads of
//     the tail z-block).  Warp math is flat over tid_x + tid_z * _threads so
//     each warp stays inside one batch slice (enforced by the % 64 guard for
//     warp-collective kernels).
//
// View identity: statement operands are *clones* of the AllocStmt output
// TensorExpr (TensorStmt owns its operands, so two statements never share a
// TensorExpr pointer).  Clones are resolved back to their allocation by the
// host-side tensor name (added to TensorExpr / AllocStmt by the tile DSL);
// name-less IR falls back to a first-layout-match heuristic.

#include <luisa/dsl/tile_to_kernel.h>

#include <luisa/core/logging.h>
#include <luisa/core/stl/format.h>
#include <luisa/core/stl/unordered_map.h>
#include <luisa/core/stl/vector.h>
#include <luisa/ast/expression.h>
#include <luisa/ast/op.h>
#include <luisa/ast/type.h>
#include <luisa/dsl/syntax.h>// DSL writing: Expr/Var sugar, if_/dynamic_range, builtins

#include <algorithm>
#include <array>
#include <functional>
#include <limits>
#include <utility>

namespace luisa::compute {
namespace {

using detail::FunctionBuilder;
using detail::TileFunctionBuilder;

// ---------------------------------------------------------------------------
// DSL type dispatch
// ---------------------------------------------------------------------------
// The tile IR carries the element dtype as a runtime tag (TensorElementType),
// while the Luisa DSL expressions (Expr<T>/Var<T>) are typed at compile time.
// `with_elem_type` instantiates the passed template lambda for the concrete
// C++ element type so the element-level code can be written in the DSL sugar
// (Expr<T> operators, max/min/abs/rsqrt, warp_* intrinsics, cast, ...) instead
// of raw FunctionBuilder::binary/call calls.  FP8 has no C++ scalar type in
// the core (it is created via Type::from("float8e4m3")), so FP8 stays on the
// dtype-erased raw path (see _value_at/_write_to).
template<typename F>
[[nodiscard]] static decltype(auto) with_elem_type(TensorElementType e, F &&f) {
    switch (e) {
        case TensorElementType::F16: return std::forward<F>(f).template operator()<half>();
        case TensorElementType::F32: return std::forward<F>(f).template operator()<float>();
        case TensorElementType::I32: return std::forward<F>(f).template operator()<int>();
        case TensorElementType::I8: return std::forward<F>(f).template operator()<byte>();
        case TensorElementType::FP8:
            LUISA_ERROR_WITH_LOCATION(
                "tile_to_kernel: fp8 has no C++ scalar type; use the raw "
                "dtype-erased access path instead of with_elem_type.");
    }
    LUISA_ERROR_WITH_LOCATION("Unsupported tensor element type {}.",
                              static_cast<uint32_t>(e));
}

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
        _tile = tile_fn.get();
        // Dynamic-batching config contract (the failure branches are the
        // strongly-skewed "almost never" side, so mark them [[unlikely]]).
        if (config.min_batching_size < 1u || config.max_batching_size < 1u) [[unlikely]] {
            LUISA_ERROR_WITH_LOCATION(
                "tile_to_kernel: min/max batching size must be >= 1 "
                "(got min={}, max={}).",
                config.min_batching_size, config.max_batching_size);
        }
        if (config.min_batching_size > config.max_batching_size) [[unlikely]] {
            LUISA_ERROR_WITH_LOCATION(
                "tile_to_kernel: min_batching_size ({}) must be <= "
                "max_batching_size ({}).",
                config.min_batching_size, config.max_batching_size);
        }
        auto meta = tile_fn->compile_meta_data();// block size + dispatch grid
        _threads = meta.block_size[0];
        _gx = meta.dispatch_size[0];
        _gy = meta.dispatch_size[1];
        _kernel2d = _gy > 1u;
        _batching = config.min_batching_size != 1u || config.max_batching_size != 1u;
        _batch_block_z = _select_batch_block_z(config);
        _block_threads = _threads * _batch_block_z;
        // Block-size strategy constraint: total threads/group <= 1024.  The B_z
        // heuristic caps this already (B_z <= max(1, 1024 / _threads)), so the
        // assert is a safety net for degenerate T.Kernel thread counts.
        if (_batching && _block_threads > 1024u) [[unlikely]] {
            LUISA_ERROR_WITH_LOCATION(
                "tile_to_kernel: batched block size {} (threads) * {} (B_z) = {} "
                "exceeds the 1024 threads/group limit.",
                _threads, _batch_block_z, _block_threads);
        }
        // Warp/batch alignment guard (decision 7): warps are formed over the
        // flat linear thread index (tid_x + tid_z * _threads), so a warp
        // straddles two batch slices iff _threads is not a multiple of the
        // lane count.  Batched kernels that use warp collectives therefore
        // require _threads % 64 == 0 (safe for both 32- and 64-lane
        // backends); the LUISA_ASSERT failure path is already [[unlikely]].
        if (_batching && _kernel_uses_warp_collectives(tile_fn.get())) {
            LUISA_ASSERT(_threads % 64u == 0u,
                         "batched tile kernels with warp collectives require "
                         "T.Kernel threads to be a multiple of 64 (got {}), so "
                         "warps never straddle batch slices",
                         _threads);
        }
        auto builder = luisa::make_shared<FunctionBuilder>(Function::Tag::KERNEL);
        {
            FunctionBuilder::FunctionStackGuard guard{builder.get()};
            builder->with(builder->body(), [&] {
                _fb = builder.get();
                // x/y stay the tile's thread count; z is the per-block batch
                // size (1 when batching is disabled).
                builder->set_block_size(uint3{meta.block_size[0], meta.block_size[1], _batch_block_z});
                if (_batching) { _emit_batch_prologue(); }
                _prescan_gemm_fragments(tile_fn->body()->statements());
                _emit_all(tile_fn->body()->statements());
            });
        }
        // T.Kernel(gx, gy, threads) launches gx*gy BLOCKS of `threads`
        // threads; the Luisa `.dispatch(...)` argument is the TOTAL number of
        // threads (grid = ceildiv(dispatch, block_size)), so the returned
        // dispatch size is (gx * threads, gy).  z is reserved for the runtime
        // batch count when batching is enabled and never appears here.
        auto dispatch = uint2{meta.dispatch_size[0] * _threads, meta.dispatch_size[1]};
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
    // dynamic batching (see the lowering plan in this file): when enabled
    // each z-thread of a block owns one batch item, so the block computes
    // `_batch_block_z` items at once and the z dispatch axis carries the
    // runtime batch count.
    bool _batching = false;// min/max batching size != (1,1)
    uint32_t _batch_block_z = 1u;// z component of set_block_size
    uint32_t _block_threads = 1u;// _threads * _batch_block_z (flat warp math)
    const Expression *_batch_index = nullptr;// block_id().z * B_z + thread_id().z
    const Expression *_batch_valid = nullptr;// _batch_index < dispatch_size().z
    // pipelined-loop context
    const Expression *_pipeline_var = nullptr;
    uint32_t _pipeline_count = 0u;
    uint32_t _pipeline_axis = 0u;
    // per-copy pipeline axis for GEMM-style pipelined copies (A: MxK, B: KxN
    // share the K extent); set by _emit_pipelined, consumed by _emit_copy.  The
    // fallback _min_extent_axis heuristic breaks when block_K > block_M, so the
    // K-extent pair inference is the primary path for pipelined GEMM copies.
    luisa::unordered_map<const CopyStmt *, uint32_t> _pipeline_copy_axes;
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

      // GEMM-accumulator fragments forced to a block-shared backing by
      // _prescan_gemm_fragments (matched by name first, layout as fallback).
      luisa::unordered_set<luisa::string> _forced_shared_names;
      luisa::vector<Layout> _forced_shared_layouts;

      // Lazy fragment values (lc_optimize for the replicated-fragment layout):
      // a whole-tile STORE into a small per-thread Fragment is recorded as an
      // expression evaluator keyed by the tensor NAME (operands are clones, so
      // the pointer-keyed _temps map cannot resolve them) instead of being
      // materialized into the per-thread local array with a _full_loop (which
      // would replicate product(extent) local-array writes on EVERY thread).
      // The consumer (typically a partitioned fragment->global COPY) then
      // evaluates the expression only for the elements it actually owns.
      luisa::unordered_map<luisa::string, TempValue> _temps_by_name;
      // Re-entrancy guard while evaluating a lazy fragment value: a store like
      // `x = rsqrt(x / N + eps)` reads the PREVIOUS value of x, which lives in
      // the materialized storage; names in this set skip the lazy lookup so the
      // read falls through to _try_storage (no infinite recursion, old value).
      luisa::vector<luisa::string> _lazy_evaluating;

      // Drop the lazy value of a fragment (if any) because a statement
      // materializes real storage for it (CLEAR/FILL/CLAMP/TRANSPOSE/COPY into
      // a fragment, a materialized STORE, or a staging replicate).
      void _invalidate_lazy(const TensorExpr *t) {
          if (t != nullptr && t->scope() == TensorScope::Fragment && !t->name().empty()) {
              _temps_by_name.erase(luisa::string{t->name()});
          }
      }

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
        // DSL literal: Expr<uint>{v} traces FunctionBuilder::current()->literal.
        return Expr<uint>{v}.expression();
    }

    [[nodiscard]] Coord _zero_coord() const noexcept {
        return Coord{_literal_u(0u), _literal_u(0u), _literal_u(0u), _literal_u(0u)};
    }

    void _sync_block() const noexcept {
        sync_block();
    }

    // ---- warp/wave helpers (lc_optimize: warp collectives) -------------------

    [[nodiscard]] const Expression *_tid_x() const noexcept {
        // Wrap the raw builtin ref so the emitted AST keeps a direct
        // thread_id() swizzle (the test suite asserts builtin refs, and it
        // avoids the local-alias Var that the dsl::thread_id() helper creates).
        return Expr<uint3>{_fb->thread_id()}.x.expression();
    }

    [[nodiscard]] const Expression *_lane_count() const noexcept {
        return Expr<uint>{_fb->warp_lane_count()}.expression();
    }

    [[nodiscard]] const Expression *_lane() const noexcept {
        return Expr<uint>{_fb->warp_lane_id()}.expression();
    }

    // Flat warp math (decision 5): with batching the block is
    // (blockDim.x = _threads, blockDim.z = _batch_block_z) and warps form over
    // the flat linear thread index tid_x + tid_z * _threads, so each warp lies
    // entirely inside one batch slice (enforced by the % 64 alignment guard).
    // Disabled batching keeps the legacy _threads-only path bit-identical.
    //
    // DSL form (see the pseudo-code in the header plan):
    //   UInt linear = thread_id().x;
    //   if (batching) linear += thread_id().z * _threads;
    //   return linear / warp_lane_count();
    [[nodiscard]] const Expression *_warp_id() const noexcept {
        // Var<uint> accumulator: Expr<uint> is not assignable, and the
        // initializer keeps the direct thread_id() swizzles in the AST.
        auto linear = Var<uint>{Expr<uint3>{_fb->thread_id()}.x};
        if (_batching) {
            linear = linear + Expr<uint3>{_fb->thread_id()}.z * _threads;
        }
        return (linear / Expr<uint>{_lane_count()}).expression();
    }

    // Warps per batch slice: reduce/scan partition each batch item's output
    // space across the slice's own warps (_threads / lanes), NOT across the
    // flat block-wide warp set.  A flat partition (block_threads / lanes)
    // would give every batch slice only 1/B_z of the output rows and leave the
    // rest at the reduce/scan identity — a correctness bug verified on device.
    // With batching disabled _threads == _block_threads, so this is the legacy
    // `_threads / lanes` count.
    [[nodiscard]] const Expression *_num_warps() const noexcept {
        return (Expr<uint>{_threads} / Expr<uint>{_lane_count()}).expression();
    }

    // Warp id local to the batch slice: the flat _warp_id ranges over the
    // whole block; the slice-local id is _warp_id % warps_per_slice (valid
    // because the % 64 alignment guard keeps each slice's warps contiguous and
    // _threads % lane_count == 0).  When batching is disabled _warp_id is
    // already < _threads/lanes, so return it unchanged (zero-overhead legacy
    // path, bit-identical kernel).
    [[nodiscard]] const Expression *_slice_warp() const noexcept {
        if (!_batching) [[likely]] { return _warp_id(); }
        return (Expr<uint>{_warp_id()} % Expr<uint>{_num_warps()}).expression();
    }

    [[nodiscard]] const Expression *_ceildiv_expr(const Expression *a, const Expression *b) const noexcept {
        // DSL: (a + b - 1u) / b
        auto ae = Expr<uint>{a};
        auto be = Expr<uint>{b};
        return ((ae + be - 1u) / be).expression();
    }

    // ---- dynamic batching helpers -------------------------------------------

    // Full tensor volume = the batch stride for Global tensors: every batch
    // item is stored contiguously in the same Buffer<T>, so batch item `b`
    // starts at element offset `b * volume(t)` (full tensor size, NOT the
    // tile extent).  The traced IR records dims only for tensors created with
    // an explicit shape (T.empty); function-INPUT tensors carry {0,0} dims,
    // so the full tensor size is reconstructed from the launch grid and the
    // tile extents — exactly the full_len/row-stride math of _global_index
    // (grid x tile extent per axis).  Using product(t->dims()) would return 0
    // for input tensors and silently disable the batch offset.
    [[nodiscard]] uint32_t _tensor_volume(const TensorExpr *t) const {
        auto ext = _current_extent != nullptr ? _current_extent : t;
        auto E = [&](uint32_t i) { return static_cast<uint32_t>(axis_extent(ext, i)); };
        auto full_len = [&](uint32_t i) -> uint32_t {
            if (_pipeline_var != nullptr && i == _pipeline_axis) {
                return _pipeline_count * E(i);
            }
            if (_kernel2d) { return i == 0u ? _gy * E(i) : _gx * E(i); }
            return i == 0u ? _gx * E(i) : E(i);
        };
        uint32_t v = 1u;
        for (auto i = 0u; i < static_cast<uint32_t>(t->rank()); ++i) { v *= full_len(i); }
        return v;
    }

    // True when the tile body allocates any block-shared storage: a Shared
    // tensor, or a Fragment large enough to be backed by a block-shared array
    // (kFragmentSharedThreshold).  Such kernels are compute/LDS-bound and use
    // the higher block-size target; pure-fragment / elementwise kernels are
    // memory/IO-bound and use the lower target (see _select_batch_block_z).
    [[nodiscard]] static bool _kernel_uses_shared(const TileFunctionBuilder *tile_fn) {
        for (auto *stmt : tile_fn->body()->statements()) {
            if (stmt->op() == TileOpKind::ALLOC) {
                auto *a = static_cast<const AllocStmt *>(stmt);
                auto *t = a->tensor();
                if (t->scope() == TensorScope::Shared) { return true; }
                if (t->scope() == TensorScope::Fragment &&
                    tile_element_count(t) >= kFragmentSharedThreshold) {
                    return true;
                }
            }
        }
        return false;
    }

    // True when the tile body contains warp-collective ops.  Warp collectives
    // communicate only within one warp; when batching, a warp must never
    // straddle two batch slices (see the 2.10 guard in lower()), otherwise
    // reduce/scan/warp-reduce would silently mix different batches.  ANY_OF /
    // ALL_OF also lower to WARP_ACTIVE_ANY/ALL and share the same hazard.
    [[nodiscard]] static bool _kernel_uses_warp_collectives(const TileFunctionBuilder *tile_fn) {
        for (auto *stmt : tile_fn->body()->statements()) {
            switch (stmt->op()) {
                case TileOpKind::WARP_REDUCE:
                case TileOpKind::REDUCE:
                case TileOpKind::REDUCE_SUM:
                case TileOpKind::CUMSUM:
                case TileOpKind::CUMMAX:
                case TileOpKind::WARP_VOTE:
                case TileOpKind::SHUFFLE:
                case TileOpKind::SYNC_THREADS_VOTE:
                case TileOpKind::ANY_OF:
                case TileOpKind::ALL_OF:
                    return true;
                default: break;
            }
        }
        return false;
    }

    // Select the z-axis block size (batch items per block) from the config
    // and a workload pre-scan (block-size strategy in the header plan):
    //   target = 256 (compute/LDS-bound: any shared alloc) else 128;
    //   B_z    = clamp(ceil(target / _threads), 1,
    //                  min(min_batching_size, 64, max(1, 1024 / _threads))).
    // Disabled batching (min == max == 1) takes the legacy B_z = 1 fast path.
    // NOTE (shared-memory budget): with batching, per-block LDS grows by B_z
    // (one tz slice per batch item); B_z <= min_batching_size bounds it, but
    // for large shared tiles keep `B_z * (largest shared alloc bytes) <= 32 KiB`
    // or occupancy drops (lc_optimize 4.1/4.7) — profiling is the arbiter.
    [[nodiscard]] uint32_t _select_batch_block_z(TileToKernelConfig const &config) const {
        if (!_batching) [[likely]] { return 1u; }
        auto target = _kernel_uses_shared(_tile) ? 256u : 128u;
        auto by_threads = (target + _threads - 1u) / _threads;
        auto cap = std::min(config.min_batching_size, 64u);
        cap = std::min(cap, std::max(1u, 1024u / _threads));
        return std::max(1u, std::min(by_threads, cap));
    }

    // Pure-expression batch prologue: batch_index and batch_valid (no
    // statements are emitted).  Called once per kernel, reused by every
    // access site.
    /*
     * _emit_batch_prologue pseudo-code (luisa-dsl):
     *
     *   UInt batch_index = block_id().z * B_z + thread_id().z;
     *   Bool batch_valid = batch_index < dispatch_size().z;
     *
     * These two expressions are cached and reused by every global access and
     * every guarded write so idle z-threads of the tail batch-block never
     * touch live data.
     */
    void _emit_batch_prologue() {
        // DSL form (pseudo-code above):
        //   UInt batch_index = block_id().z * B_z + thread_id().z;
        //   Bool batch_valid = batch_index < dispatch_size().z;
        // The builtin refs are wrapped directly (Expr<uint3>{_fb->...}) so the
        // AST keeps direct block_id()/thread_id()/dispatch_size() swizzles.
        auto block_z = Expr<uint3>{_fb->block_id()}.z;
        auto thread_z = Expr<uint3>{_fb->thread_id()}.z;
        auto batch_index = block_z * _batch_block_z + thread_z;
        auto batch_valid = batch_index < Expr<uint3>{_fb->dispatch_size()}.z;
        _batch_index = batch_index.expression();
        _batch_valid = batch_valid.expression();
    }

    // combine step of a TileReduceOp for a compile-time element type T
    // (DSL form of _reduce_combine; used by the typed warp/scan helpers).
    template<typename T>
    [[nodiscard]] static const Expression *_combine_expr(TileReduceOp op,
                                                        const Expression *acc,
                                                        const Expression *v) {
        auto a = Expr<T>{acc};
        auto b = Expr<T>{v};
        switch (op) {
            case TileReduceOp::SUM:
            case TileReduceOp::ABS_SUM:
                return (a + b).expression();
            case TileReduceOp::MAX:
            case TileReduceOp::ABS_MAX:
                return max(a, b).expression();
            case TileReduceOp::MIN:
                return min(a, b).expression();
            case TileReduceOp::BIT_AND:
                if constexpr (std::is_integral_v<T>) { return (a & b).expression(); }
                break;
            case TileReduceOp::BIT_OR:
                if constexpr (std::is_integral_v<T>) { return (a | b).expression(); }
                break;
            case TileReduceOp::BIT_XOR:
                if constexpr (std::is_integral_v<T>) { return (a ^ b).expression(); }
                break;
        }
        LUISA_ERROR_WITH_LOCATION("tile_to_kernel: invalid tile reduce op.");
    }

    // all-lane warp reduction matching a TileReduceOp (lc_optimize 2.2/2.5:
    // XOR butterfly via warp_read_lane; every lane ends with the total).
    // ABS_* must be pre-folded per element by the caller.
    //
    // DSL form:
    //   UInt lane = warp_lane_id(); UInt lanes = warp_lane_count();
    //   T result = v;
    //   for (d = 1, 2, 4, ...) {
    //       $if (d < lanes) {
    //           UInt peer = lane ^ d;
    //           T other = warp_read_lane(result, peer);
    //           result = combine(op, result, other);
    //       };
    //   };
    template<typename T>
    [[nodiscard]] const Expression *_warp_reduce_typed(TileReduceOp op,
                                                       const Expression *v) {
        auto lane = Expr<uint>{_lane()};
        auto lanes = Expr<uint>{_lane_count()};
        auto result = Var<T>{Expr<T>{v}};
        for (uint32_t d = 1u; d <= 64u; d <<= 1u) {
            // warp-uniform guard: skip the steps above the actual warp size
            if_(Expr<uint>{d} < lanes, [&] {
                auto peer = lane ^ Expr<uint>{d};
                auto other = warp_read_lane(result, peer);
                result = Expr<T>{_combine_expr<T>(op, result.expression(),
                                                  other.expression())};
            });
        }
        return result.expression();
    }

    // combine step of a TileReduceOp: acc <- acc `op` v (runtime-dtype entry)
    [[nodiscard]] const Expression *_reduce_combine(TileReduceOp op, TensorElementType e,
                                                    const Expression *acc, const Expression *v) const {
        return with_elem_type(e, [&]<typename T>() -> const Expression * {
            return _combine_expr<T>(op, acc, v);
        });
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
        // With batching the staging tile is block-shared and holds one slice
        // per batch item (B_z * n); per-thread fragments are addressed through
        // _staging_index so every tz thread writes/reads its own slice.
        auto alloc_n = _batching ? n * _batch_block_z : n;
        auto s = _fb->shared(Type::array(elem_t, alloc_n));
        _fragment_staging.emplace(t, s);
        return s;
    }

    // refresh every thread's fragment replica from the staging tile
    void _replicate_from_staging(const TensorExpr *t, TensorElementType e,
                                 const RefExpr *staging) {
        _invalidate_lazy(t);
        _sync_block();
        _full_loop(t, [&](const Coord &c) {
            auto idx = _staging_index(t, c);
            auto value = with_elem_type(e, [&]<typename T>() -> const Expression * {
                // DSL access: staging[idx] (array element read).  A named
                // Var<std::array<T,1>> wrapper is required: the rvalue
                // Expr<std::array<T,1>>{...}[idx] form is rejected for half /
                // byte (array element must be >= 4-byte aligned) and its
                // operator[] returns a temporary whose assignment is deleted.
                Var<std::array<T, 1>> arr{staging};
                return arr[Expr<uint>{idx}].expression();
            });
            _write_to(t, c, value);
        });
    }

    // emit `if (cond) { body() }` (the DSL $if sugar: if_(Expr<bool>, body))
    template<typename Body>
    void _if(const Expression *cond, Body &&body) {
        if_(Expr<bool>{cond}, std::forward<Body>(body));
    }

    // emit `for (var = begin; var < end; var += step) { body(var) }`
    // (the DSL $for sugar: for (auto i : dynamic_range(begin, end, step)));
    // `var` is the DSL loop variable and body receives its raw expression.
    template<typename Body>
    void _for_range(const Expression *begin, const Expression *end,
                    const Expression *step, Body &&body) {
        for (auto i : dynamic_range(Expr<uint>{begin},
                                    Expr<uint>{end},
                                    Expr<uint>{step})) {
            body(i.expression());
        }
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

    // row-major linear index inside a shared/fragment tile.  With batching,
    // block-shared storage (Shared tensors and shared-backed fragments) holds
    // one slice per batch item: index += tid_z * n.  Per-thread fragments are
    // replicated per thread (each thread already owns its own batch item's
    // tile), so they get no slice.
    [[nodiscard]] const Expression *_local_index(const TensorExpr *t, const Coord &c) {
        // DSL form: row-major linear index inside a shared/fragment tile
        // (Var<uint> is the assignable DSL accumulator; Expr<uint> cannot be
        // reassigned).
        auto idx = Var<uint>{Expr<uint>{0u}};
        uint32_t stride = 1u;
        for (int32_t i = static_cast<int32_t>(t->rank()) - 1; i >= 0; --i) {
            idx = idx + Expr<uint>{c[i]} * stride;
            stride *= static_cast<uint32_t>(axis_extent(t, i));
        }
        if (_batching) {
            if (auto *st = _try_storage(t)) {
                auto shared_backed = st->scope == TensorScope::Shared ||
                                     (st->scope == TensorScope::Fragment && st->shared != nullptr);
                if (shared_backed) {
                    idx = idx + Expr<uint3>{_fb->thread_id()}.z * tile_element_count(t);
                }
            }
        }
        return idx.expression();
    }

    // Staging-tile index for a fragment producer.  The staging tile is
    // block-shared and holds one slice per batch item (B_z * n elements).
    // Per-thread fragments do NOT get a batch slice from _local_index, so the
    // tz offset is added here; shared-backed fragments already include it via
    // _local_index (2.5), so they must not be offset twice.
    [[nodiscard]] const Expression *_staging_index(const TensorExpr *t, const Coord &c) {
        auto idx = Var<uint>{Expr<uint>{_local_index(t, c)}};
        if (_batching && !_is_fragment_shared_backed(t)) {
            idx = idx + Expr<uint3>{_fb->thread_id()}.z * tile_element_count(t);
        }
        return idx.expression();
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

    // reconstructed global buffer index (plan: base-offset reconstruction).
    // DSL form: each per-axis base is built with block_id()/pipeline arithmetic
    // on Expr<uint>, then summed with row strides exactly like the raw version.
    [[nodiscard]] const Expression *_global_index(const TensorExpr *t, const Coord &c) const {
        auto rank = t->rank();
        auto ext = _current_extent != nullptr ? _current_extent : t;
        auto E = [&](uint32_t i) { return static_cast<uint32_t>(axis_extent(ext, i)); };
        // per-axis runtime base
        auto base_expr = [&](uint32_t i) -> const Expression * {
            auto b = Var<uint>{Expr<uint>{0u}};
            auto off = t->offset();
            auto host_off = i < off.size() && off[i] > 0 ? static_cast<uint32_t>(off[i]) : 0u;
            if (_pipeline_var != nullptr && i == _pipeline_axis) {
                b = b + Expr<uint>{_pipeline_var} * E(i);
            } else if (_kernel2d) {
                auto bid = i == 0u ? Expr<uint3>{_fb->block_id()}.y
                                   : Expr<uint3>{_fb->block_id()}.x;
                b = b + bid * E(i);
            } else if (i == 0u) {
                b = b + Expr<uint3>{_fb->block_id()}.x * E(i);
            }
            if (host_off != 0u) {
                b = b + Expr<uint>{host_off};
            }
            return b.expression();
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
        auto idx = Var<uint>{Expr<uint>{0u}};
        for (uint32_t i = 0u; i < rank; ++i) {
            auto base_i = Expr<uint>{base_expr(i)};
            auto sum = base_i + Expr<uint>{c[i]};
            auto term = sum * row_stride(i);
            idx = idx + term;
        }
        if (_batching) {
            // Clamped batch offset: invalid threads (the tail z-block) read
            // batch 0's in-bounds data; every downstream global WRITE is
            // guarded by `_batch_valid`, so those reads are discarded and no
            // out-of-bounds access can happen.
            // DSL: select(0u, batch_index, batch_valid) = batch_valid ? batch_index : 0u
            auto safe = select(Expr<uint>{0u}, Expr<uint>{_batch_index},
                               Expr<bool>{_batch_valid});
            auto volume = _tensor_volume(t);
            if (volume != 0u) {
                idx = idx + safe * volume;
            }
        }
        return idx.expression();
    }

    // ---- value access --------------------------------------------------------

    [[nodiscard]] const Expression *_value_at(const TensorExpr *t, const Coord &c) {
        if (_is_temp(t)) { return _temps[t].eval(c); }
        // Lazy fragment value (whole-tile STORE into a replicated fragment,
        // recorded by _emit_store): evaluate the stored expression instead of
        // reading a materialized local array.  Skipped while that very value
        // is being evaluated (self-referential stores read the OLD value from
        // storage via the fall-through below).
        if (auto name = t->name(); !name.empty()) {
            luisa::string key{name};
            auto guarded = std::find(_lazy_evaluating.begin(), _lazy_evaluating.end(), key) != _lazy_evaluating.end();
            if (!guarded) {
                if (auto it = _temps_by_name.find(key); it != _temps_by_name.end()) {
                    _lazy_evaluating.emplace_back(key);
                    auto *v = it->second.eval(c);
                    _lazy_evaluating.pop_back();
                    return v;
                }
            }
        }
        if (auto *st = _try_storage(t)) {
            // fp8 has no C++ scalar type: keep the dtype-erased raw access
            // (byte storage + later cast to the fp8 element type).
            if (st->dtype == TensorElementType::FP8) {
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
                        auto idx = _local_index(t, c);
                        return _fb->access(elem_t, st->shared != nullptr ? st->shared : st->fragment, idx);
                    }
                }
                LUISA_ERROR_WITH_LOCATION("Invalid tensor scope.");
            }
            // DSL access: Buffer<T>{ref}.read(idx) / shared_array[idx] /
            // fragment_array[idx].  Use a named Var<std::array<T,1>> wrapper
            // for array reads: the rvalue Expr<std::array<T,1>>{...}[idx] form
            // is rejected for half/byte (array element >= 4-byte alignment).
            return with_elem_type(st->dtype, [&]<typename T>() -> const Expression * {
                switch (st->scope) {
                    case TensorScope::Global: {
                        auto idx = Expr<uint>{_global_index(t, c)};
                        return Expr<Buffer<T>>{st->buffer}.read(idx).expression();
                    }
                    case TensorScope::Shared: {
                        auto idx = Expr<uint>{_local_index(t, c)};
                        Var<std::array<T, 1>> arr{st->shared};
                        return arr[idx].expression();
                    }
                    case TensorScope::Fragment: {
                        // large fragments are backed by a block-shared array
                        auto idx = Expr<uint>{_local_index(t, c)};
                        auto ref = st->shared != nullptr ? st->shared : st->fragment;
                        Var<std::array<T, 1>> arr{ref};
                        return arr[idx].expression();
                    }
                }
                LUISA_ERROR_WITH_LOCATION("Invalid tensor scope.");
                return nullptr;
            });
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
        if (st.dtype == TensorElementType::FP8) {
            // dtype-erased raw path (fp8 has no C++ scalar type)
            switch (st.scope) {
                case TensorScope::Global: {
                    auto idx = _global_index(t, c);
                    if (_batching) {
                        _if(_batch_valid, [&] { _fb->call(CallOp::BUFFER_WRITE, {st.buffer, idx, value}); });
                    } else {
                        _fb->call(CallOp::BUFFER_WRITE, {st.buffer, idx, value});
                    }
                    break;
                }
                case TensorScope::Shared: {
                    auto idx = _local_index(t, c);
                    _fb->assign(_fb->access(elem_t, st.shared, idx), value);
                    break;
                }
                case TensorScope::Fragment: {
                    auto idx = _local_index(t, c);
                    _fb->assign(_fb->access(elem_t, st.shared != nullptr ? st.shared : st.fragment, idx), value);
                    break;
                }
            }
            return;
        }
        with_elem_type(st.dtype, [&]<typename T>() {
            auto v = Expr<T>{value};
            switch (st.scope) {
                case TensorScope::Global: {
                    auto idx = Expr<uint>{_global_index(t, c)};
                    // Guard every global write with the batch-validity predicate
                    // (decision 4): idle tz threads in the tail z-block must not
                    // write; the guard sits inside the existing element guards and
                    // contains no barrier, so it is divergence-safe.  Global reads
                    // are intentionally unguarded (clamped index, values discarded).
                    auto write = [&] { Expr<Buffer<T>>{st.buffer}.write(idx, v); };
                    if (_batching) { _if(_batch_valid, write); } else { write(); }
                    break;
                }
                case TensorScope::Shared: {
                    auto idx = Expr<uint>{_local_index(t, c)};
                    Var<std::array<T, 1>> arr{st.shared};
                    arr[idx] = v;
                    break;
                }
                case TensorScope::Fragment: {
                    // large fragments are backed by a block-shared array
                    auto idx = Expr<uint>{_local_index(t, c)};
                    auto ref = st.shared != nullptr ? st.shared : st.fragment;
                    Var<std::array<T, 1>> arr{ref};
                    arr[idx] = v;
                    break;
                }
            }
        });
    }

    [[nodiscard]] Coord _decompose(const TensorExpr *t, const Expression *idx) const {
        // DSL form: row-major linear index -> per-axis coords with % and /
        // (Var<uint> is the assignable DSL accumulator).  This is the standard
        // decomposition: c[i] = rem % extent_i; rem = rem / extent_i, walking
        // the axes from the fastest (last) to the slowest (first).
        auto c = _zero_coord();
        auto rem = Var<uint>{Expr<uint>{idx}};
        for (int32_t i = static_cast<int32_t>(t->rank()) - 1; i >= 0; --i) {
            auto e = static_cast<uint32_t>(axis_extent(t, i));
            auto coord = rem % e;
            c[i] = coord.expression();
            rem = rem / e;
        }
        return c;
    }

    // ---- loop helpers ---------------------------------------------------------
    // each element exactly once across the block (Global / Shared targets)
    // 2D partition over a host-known (rows, cols) grid using the 1D block:
    // decompose the linear thread id once into (r0, c0), then stride both axes.
    // (plan 1.3: removes the per-element div/mod for rank-2 tiles.)
    template<typename Body>
    void _partition_2d(uint32_t rows, uint32_t cols, Body &&body) {
        // DSL form: decompose the linear thread id once, then stride both axes.
        //   UInt tid = thread_id().x;
        //   UInt r0 = tid / tw; UInt c0 = tid % tw;
        //   for (auto r : dynamic_range(r0, rows, th)) {
        //       if (r < rows) {
        //           for (auto c : dynamic_range(c0, cols, tw)) { body(r, c); }
        //       }
        //   }
        auto tid = Expr<uint3>{_fb->thread_id()}.x;
        auto tw = std::min(_threads, cols);// threads along the fast axis
        auto th = (_threads + tw - 1u) / tw;// threads along the slow axis
        auto r0 = tid / tw;
        auto c0 = tid % tw;
        _for_range(r0.expression(), _literal_u(rows), _literal_u(th),
                   [&](const Expression *r) {
            auto emit_cols = [&] {
                _for_range(c0.expression(), _literal_u(cols), _literal_u(tw),
                           [&](const Expression *c) { body(r, c); });
            };
            if (rows % th != 0u) [[unlikely]] {// some threads start at r0 >= rows
                _if((Expr<uint>{r} < rows).expression(), emit_cols);
            } else {
                emit_cols();
            }
        });
    }

    // rank-2 partition loop: adapts the Coord-based body used everywhere else.
    template<typename Body>
    void _partition_loop_2d(const TensorExpr *t, Body &&body) {
        LUISA_ASSERT(t->rank() == 2u, "tile_to_kernel: _partition_loop_2d requires a rank-2 tile.");
        auto rows = static_cast<uint32_t>(axis_extent(t, 0u));
        auto cols = static_cast<uint32_t>(axis_extent(t, 1u));
        _partition_2d(rows, cols, [&](const Expression *r, const Expression *c) {
            Coord cc = _zero_coord();
            cc[0] = r;
            cc[1] = c;
            body(cc);
        });
    }

    /*
     * _partition_loop(t, body) pseudo-code (luisa-dsl):
     *
     * UInt total = product(extent of t); // logical tile element count
     * UInt iters = ceildiv(total, block_size().x);
     * $for (i, 0u, iters) {
     * UInt idx = i * block_size().x + thread_id().x;  // linear lane
     * $if (idx < total) { body(decompose(idx)); }; // row-major coords
     * };
     *
     * Optimization: when total % block_size().x == 0 the guard is always
     * true and is omitted (compile-time known extents).  Rank-2 tiles use
     * _partition_loop_2d instead (no per-element div/mod).
     */
    template<typename Body>
    void _partition_loop(const TensorExpr *t, Body &&body) {
        // DSL form (pseudo-code above):
        //   UInt total = product(extent of t);
        //   UInt iters = ceildiv(total, block_size().x);
        //   UInt tid = thread_id().x;
        //   for (auto i : dynamic_range(0u, iters)) {
        //       UInt idx = i * _threads + tid;
        //       if (idx < total) { body(decompose(idx)); }
        //   }
        auto total = tile_element_count(t);
        auto iters = (total + _threads - 1u) / _threads;
        auto tid = Expr<uint3>{_fb->thread_id()}.x;
        auto emit_body = [&](const Expression *idx) { body(_decompose(t, idx)); };
        if (total % _threads != 0u) [[unlikely]] {
            _for_range(_literal_u(0u), _literal_u(iters), _literal_u(1u),
                       [&](const Expression *i) {
                           auto idx = (Expr<uint>{i} * _threads + tid).expression();
                           auto cond = (Expr<uint>{idx} < total).expression();
                           _if(cond, [&] { emit_body(idx); });
                       });
        } else {
            _for_range(_literal_u(0u), _literal_u(iters), _literal_u(1u),
                       [&](const Expression *i) {
                           auto idx = (Expr<uint>{i} * _threads + tid).expression();
                           emit_body(idx);
                       });
        }
    }

    // every thread processes the whole tile (replicated Fragment layout)
    /*
     * _full_loop(t, body) pseudo-code (luisa-dsl):
     *
     *   UInt total = product(extent of t);
     *   $for (i, 0u, total) { body(decompose(i)); };
     *
     * Used only for replicated per-thread Fragment tiles; a lane-mapped
     * fragment layout (future work, plan 1.2) would remove most call sites.
     */
    template<typename Body>
    void _full_loop(const TensorExpr *t, Body &&body) {
        auto total = tile_element_count(t);
        _for_range(_literal_u(0u), _literal_u(total), _literal_u(1u),
                   [&](const Expression *i) { body(_decompose(t, i)); });
    }

    // ---------------------------------------------------------------------------
    // statement emission
    // ---------------------------------------------------------------------------

    // Prescan: GEMM statements whose C operand is a small per-thread
    // (replicated) fragment AND which do NOT qualify for the warp-K-split
    // path get their fragment forced to a block-shared backing (the
    // kFragmentSharedThreshold mechanism of _emit_alloc).  Rationale: such a
    // fragment is typically accumulated across pipeline iterations, and the
    // replicated-fragment lowering forces every GEMM to publish its
    // partitioned result through a shared staging tile and then refresh
    // EVERY thread's replica (_replicate_from_staging: product(extent)
    // shared loads + local-array writes per thread per iteration) — the
    // dominant cost of pipelined small-tile GEMMs.  With a shared backing
    // the gemm partition loop writes each element exactly once and reads it
    // back directly; no staging tile, no replica refresh.  Warp-path
    // fragments keep the per-thread replica: the single-warp write-back is
    // deliberately barrier-free (each lane writes its own replica).
    void _prescan_gemm_fragments(luisa::span<const TensorStmt *const> stmts) {
        for (auto *stmt : stmts) {
            if (stmt->op() != TileOpKind::GEMM) { continue; }
            auto *g = static_cast<const GemmStmt *>(stmt);
            auto *a = g->a();
            auto *c = g->c();
            if (a == nullptr || c == nullptr ||
                c->scope() != TensorScope::Fragment ||
                a->rank() != 2u || c->rank() != 2u ||
                !extent_known(a) || !extent_known(c)) [[unlikely]] {
                continue;
            }
            // mirror the warp-K-split gate of _emit_gemm exactly
            auto M = static_cast<uint32_t>(axis_extent(c, 0u));
            auto N = static_cast<uint32_t>(axis_extent(c, 1u));
            auto K = static_cast<uint32_t>(axis_extent(a, 1u));
            auto TM = (M % 4u == 0u) ? 4u : ((M % 2u == 0u) ? 2u : 1u);
            auto TN = (N % 4u == 0u) ? 4u : ((N % 2u == 0u) ? 2u : 1u);
            auto use_warp = !_use_cooperative && !_batching &&
                            _threads >= 32u &&
                            (M / TM) * (N / TN) < _threads &&
                            K >= 256u;
            if (use_warp) { continue; }
            if (auto name = c->name(); !name.empty()) {
                _forced_shared_names.emplace(luisa::string{name});
            } else {
                _forced_shared_layouts.emplace_back(
                    Layout{TensorScope::Fragment, c->dtype(),
                           luisa::fixed_vector<int32_t, 4>{c->dims().begin(), c->dims().end()}});
            }
        }
    }

    [[nodiscard]] bool _is_forced_shared_fragment(const TensorExpr *t) const {
        if (t == nullptr || t->scope() != TensorScope::Fragment) { return false; }
        if (auto name = t->name(); !name.empty()) {
            return _forced_shared_names.contains(luisa::string{name});
        }
        Layout key{t->scope(), t->dtype(),
                   luisa::fixed_vector<int32_t, 4>{t->dims().begin(), t->dims().end()}};
        for (auto &l : _forced_shared_layouts) {
            if (l == key) { return true; }
        }
        return false;
    }

    /*
     * _emit_all(stmts) pseudo-code (host-side statement walk; each emitted
     * device body is luisa-dsl, see _partition_loop and the _emit_* helpers):
     *
     *   i = 0
     *   while i < len(stmts):
     *       if stmts[i] is PIPELINED:
     *           // gather contiguous shared-memory statements into one pipeline body
     *           end = i + 1
     *           while end < len(stmts) and stmts[end] is not PIPELINED/KERNEL
     *                 and stmts[end] touches shared memory:
     *               end += 1
     *           _emit_pipelined(stmts[i], stmts[i+1:end])
     *           i = end
     *       else:
     *           _emit(stmts[i])
     *           i += 1
     */
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
                    if (!_accesses_shared(candidate) || _writes_global(candidate)) { break; }
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

    /*
     * _emit_pipelined(p, body) pseudo-code:
     *
     *   // host-side: emit a device $for over the pipeline steps, then run the
     *   // body statements under the pipeline context
     *   count = p->count()
     *   $for (ko, 0u, count) {
     *       _pipeline_var = ko            // drives per-axis base offset
     *       for each stmt in body:
     *           _emit(stmt)
     *   };
     *   _pipeline_var = null
     */
    void _emit_pipelined(const PipelinedStmt *p,
                         luisa::span<const TensorStmt *const> body) {
        auto count = static_cast<uint32_t>(p->count());
        if (count == 0u) { return; }
        _pipeline_count = count;
        // GEMM-style pipelined copies (A: MxK, B: KxN) share the K extent across
        // their two rank-2 shared-tile destinations.  Record each copy's pipeline
        // axis from that shared extent — robust to block_K being larger OR
        // smaller than block_M/block_N, unlike the _min_extent_axis heuristic
        // (which breaks when block_K > block_M, e.g. the 4096^3 f32 GEMM
        // benchmark with block_M = block_N = 16, block_K = 128).
        _pipeline_copy_axes.clear();
        luisa::vector<const CopyStmt *> copies;
        for (auto *s : body) {
            if (s->op() == TileOpKind::COPY) {
                copies.emplace_back(static_cast<const CopyStmt *>(s));
            }
        }
        for (auto i = 0u; i < copies.size() && _pipeline_copy_axes.empty(); ++i) {
            auto *di = copies[i]->dst();
            if (di == nullptr || di->rank() != 2u || di->scope() != TensorScope::Shared) { continue; }
            auto ai0 = static_cast<uint32_t>(axis_extent(di, 0u));
            auto ai1 = static_cast<uint32_t>(axis_extent(di, 1u));
            for (auto j = i + 1u; j < copies.size(); ++j) {
                auto *dj = copies[j]->dst();
                if (dj == nullptr || dj->rank() != 2u || dj->scope() != TensorScope::Shared) { continue; }
                auto bj0 = static_cast<uint32_t>(axis_extent(dj, 0u));
                auto bj1 = static_cast<uint32_t>(axis_extent(dj, 1u));
                if (ai1 == bj0) { _pipeline_copy_axes[copies[i]] = 1u; _pipeline_copy_axes[copies[j]] = 0u; break; }
                if (ai0 == bj1) { _pipeline_copy_axes[copies[i]] = 0u; _pipeline_copy_axes[copies[j]] = 1u; break; }
            }
        }
        _for_range(_literal_u(0u), _literal_u(count), _literal_u(1u),
                   [&](const Expression *ko) {
                       _pipeline_var = ko;
                       for (auto *s : body) { _emit(s); }
                   });
        _pipeline_copy_axes.clear();
        _pipeline_var = nullptr;
        _pipeline_count = 0u;
    }

    // True when the statement WRITES to a Global tensor (the destination of
    // a copy / store / fill / transpose / clamp / atomic).  The flat IR bakes
    // block-derived global offsets as host constants, so a global-writing
    // statement can never be part of a pipelined loop body: the pipeline-axis
    // reconstruction in _global_index would replace the block offset with
    // pipeline arithmetic and mis-place every write.  _emit_all therefore
    // stops the pipeline body run before such a statement (e.g. the final
    // fragment->global tile store after a pipelined GEMM whose C fragment is
    // shared-backed — it "touches shared" but belongs AFTER the loop).
    [[nodiscard]] bool _writes_global(const TensorStmt *s) const noexcept {
        auto global = [](const TensorExpr *t) noexcept {
            return t != nullptr && t->scope() == TensorScope::Global;
        };
        switch (s->op()) {
            case TileOpKind::COPY: return global(static_cast<const CopyStmt *>(s)->dst());
            case TileOpKind::STORE: return global(static_cast<const TileStoreStmt *>(s)->lhs());
            case TileOpKind::FILL: return global(static_cast<const FillStmt *>(s)->buf());
            case TileOpKind::TRANSPOSE: return global(static_cast<const TransposeStmt *>(s)->dst());
            case TileOpKind::CLAMP: return global(static_cast<const ClampStmt *>(s)->dst());
            case TileOpKind::ATOMIC: return global(static_cast<const AtomicStmt *>(s)->dst());
            default: return false;
        }
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

    /*
     * _emit(stmt) pseudo-code (host-side dispatch; each _emit_* below emits
     * luisa-dsl device code):
     *
     *   switch stmt->op():
     *       ALLOC       -> _emit_alloc(stmt)
     *       CLEAR       -> _emit_clear(stmt)
     *       COPY        -> _emit_copy(stmt)
     *       STORE       -> _emit_store(stmt)
     *       BINARY      -> _emit_binary(stmt)
     *       MAX/MIN/ABS -> _emit_max/_min/_abs(stmt)
     *       RSQRT       -> _emit_rsqrt(stmt)
     *       REDUCE_SUM  -> _emit_reduce_sum(stmt)
     *       REDUCE      -> _emit_reduce(stmt)
     *       GEMM        -> _emit_gemm(stmt)
     *       PRINT       -> _emit_print(stmt)
     *       FILL        -> _emit_fill(stmt)
     *       TRANSPOSE   -> _emit_transpose(stmt)
     *       CLAMP       -> _emit_clamp(stmt)
     *       ATOMIC      -> _emit_atomic(stmt)
     *       SYNC        -> _emit_sync(stmt)
     *       WARP_REDUCE -> _emit_warp_reduce(stmt)
     *       CUMSUM      -> _emit_scan(src,dst,dim,reverse,is_max=false)
     *       CUMMAX      -> _emit_scan(src,dst,dim,reverse,is_max=true)
     *       ANY_OF      -> _emit_any_all(buf, is_all=false)
     *       ALL_OF      -> _emit_any_all(buf, is_all=true)
     *       SHUFFLE     -> _emit_shuffle(stmt)
     *       FAST_MATH   -> _emit_fast_math(stmt)
     *       IEEE_MATH   -> _emit_ieee_math(stmt)
     *       metadata ops -> no-op
     *   // barrier discipline: sync after every statement that touches shared
     *   // memory (never inside a thread-divergent branch)
     *   if stmt touches shared memory:
     *       sync_block()
     */
    void _emit(const TensorStmt *stmt) {
        _emit_core(stmt);
        // barrier discipline: sync after every statement that touches shared
        // memory (never inside a thread-divergent branch — all our shared
        // accesses live inside $for/$if bodies, the sync is at top level)
        if (_accesses_shared(stmt)) { _sync_block(); }
    }

    // The statement dispatch of _emit WITHOUT the conservative trailing
    // _sync_block: _emit_pipelined calls this directly and places the body
    // barriers itself via hazard tracking (see _emit_pipelined).
    void _emit_core(const TensorStmt *stmt) {
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
    }

    /*
     * _emit_alloc(s) pseudo-code:
     *
     *   t = s->tensor()
     *   switch t->scope():
     *     case Global:
     *         // one Buffer<T> kernel argument per Global tensor
     *         st.buffer = Buffer<half/float/int/byte/fp8>()   // kernel arg
     *     case Shared:
     *         n = product(extent)
     *         alloc_n = batching ? n * B_z : n
     *         st.shared = Shared<T> s{alloc_n}                // block-shared
     *     case Fragment:
     *         n = product(extent)
     *         alloc_n = batching ? n * B_z : n
     *         if n >= kFragmentSharedThreshold:
     *             // large fragment: back with block-shared array
     *             st.shared = Shared<T> s{alloc_n}            // B_z slices
     *         else:
     *             // small fragment: per-thread replicated local array
     *             st.fragment = Local<T> v[n]                 // per-thread
     *   record storage by pointer / name / layout
     */
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
                // With batching the shared array holds one slice per batch item
                // (B_z * n); _local_index adds the tid_z slice on access.
                auto alloc_n = _batching ? n * _batch_block_z : n;
                st.shared = _fb->shared(Type::array(elem_t, alloc_n));
                st.array_size = alloc_n;
                break;
            }
          case TensorScope::Fragment: {
              auto n = tile_element_count(t);
              if (n == 0u) [[unlikely]] {
                  LUISA_ERROR_WITH_LOCATION("Fragment tile allocation with zero elements: {}", t->describe());
              }
              if (n >= kFragmentSharedThreshold || _is_forced_shared_fragment(t)) {
                  // Large fragment (or a GEMM accumulator forced by
                  // _prescan_gemm_fragments): back it with a block-shared
                  // array instead of a per-thread local array.  Ops on it
                  // use partition loops (one compute per element across the
                  // block) and the shared barrier discipline; see
                  // _is_fragment_shared_backed.  With batching
                  // the array is B_z * n (one slice per batch item).
                  auto alloc_n = _batching ? n * _batch_block_z : n;
                  st.shared = _fb->shared(Type::array(elem_t, alloc_n));
                  st.array_size = alloc_n;
              } else {
                  st.fragment = _fb->local(Type::array(elem_t, n));
                  st.array_size = n;
              }
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

    /*
     * _emit_clear(s) pseudo-code (luisa-dsl):
     *
     *   t = s->t()
     *   zero = literal 0 of t's dtype
     *   if t is Fragment and not shared-backed:   // replicated per-thread tile
     *       _full_loop(t, c => _write_to(t, c, zero))       // $for (i, 0u, total)
     *   else:                                     // Global / Shared / shared-backed
     *       rank==2 ? _partition_loop_2d(t, body) : _partition_loop(t, body)
     */
    void _emit_clear(const ClearStmt *s) {
        auto *t = s->t();
        auto saved = _current_extent;
        _current_extent = t;
        auto zero = _zero_of(t->dtype());
      if (t->scope() == TensorScope::Fragment && !_is_fragment_shared_backed(t)) {
          _full_loop(t, [&](const Coord &c) { _write_to(t, c, zero); });
          _invalidate_lazy(t);
      } else if (t->rank() == 2u) {
          _partition_loop_2d(t, [&](const Coord &c) { _write_to(t, c, zero); });
      } else {
          _partition_loop(t, [&](const Coord &c) { _write_to(t, c, zero); });
      }
      _current_extent = saved;
  }

    /*
     * _emit_copy(s) pseudo-code (luisa-dsl):
     *
     *   ext = operand with fully known extent (dst preferred, then src)
     *   if inside a pipeline:
     *       _pipeline_axis = axis with smallest extent of ext
     *   body(c) = dst[c] = src[c]     // _write_to/_value_at resolve the storage
     *   if dst is Fragment and not shared-backed:
     *       if src is Global:
     *           // coalesced global -> fragment through shared staging
     *           staging = _staging_for(dst)                 // Shared<T> staging{n}
     *           sync_block()
     *           _partition_loop(ext, c => staging[_staging_index(dst, c)] = src[c])
     *           _replicate_from_staging(dst, staging)       // every thread refills dst
     *       else:
     *           _full_loop(ext, body)
     *   else:
     *       rank==2 ? _partition_loop_2d(ext, body) : _partition_loop(ext, body)
     */
    void _emit_copy(const CopyStmt *s) {
        auto *src = s->src();
        auto *dst = s->dst();
        auto *ext = op_extent_of(dst, src);
        auto saved = _current_extent;
        auto saved_axis = _pipeline_axis;
        _current_extent = ext;
        if (_pipeline_var != nullptr) {
            // per-copy pipeline axis: GEMM-style copies use the K axis recorded
            // by _emit_pipelined (robust for block_K > block_M); the fallback is
            // the smallest-extent heuristic (the K axis of a GEMM-style copy
            // when block_K < block_M).
            if (auto it = _pipeline_copy_axes.find(s); it != _pipeline_copy_axes.end()) {
                _pipeline_axis = it->second;
            } else {
                _pipeline_axis = _min_extent_axis(ext);
            }
        }
        auto body = [&](const Coord &c) {
            _write_to(dst, c, _value_at(src, c));
        };
        if (auto pit = _pipeline_copy_axes.find(s);
            pit != _pipeline_copy_axes.end() &&
            src->scope() == TensorScope::Global &&
            dst->scope() == TensorScope::Shared &&
            ext->rank() == 2u && extent_known(ext)) [[likely]] {
            auto rows = static_cast<uint32_t>(axis_extent(ext, 0u));
            auto cols = static_cast<uint32_t>(axis_extent(ext, 1u));
            auto chunk = cols % 8u == 0u ? 8u : (cols % 4u == 0u ? 4u : 1u);
            if (chunk > 1u) {
                // GEMM-feeding global->shared tile copy (lc_optimize: memory-
                // level parallelism).  The generic partition loop alternates
                // a dependent global load -> shared store per element, so
                // every element pays the full memory latency; this dominates
                // the pipelined GEMM cost (e.g. the 8x256 f32 tiles of
                // bench_gemm_4096: 64 dependent loads per thread per copy).
                // Copy 4 consecutive fast-axis elements per chunk instead:
                // the 4 independent global loads issue back-to-back (their
                // latencies overlap in the memory pipeline) before the 4
                // shared stores.  The fast axis is contiguous in both the
                // global row and the shared tile (row-major, stride 1).
                auto chunks_per_row = cols / chunk;
                auto nchunks = rows * chunks_per_row;
                auto tid = Expr<uint3>{_fb->thread_id()}.x;
                auto emit_chunk = [&](const Expression *cid) {
                    auto r = (Expr<uint>{cid} / chunks_per_row).expression();
                    auto cb = (Expr<uint>{cid} % chunks_per_row) * chunk;
                    // `chunk` independent global loads materialized in locals
                    std::array<const Expression *, 8> v{};
                    for (uint32_t u = 0u; u < chunk; ++u) {
                        Coord cc = _zero_coord();
                        cc[0] = r;
                        cc[1] = (Expr<uint>{cb} + u).expression();
                        v[u] = with_elem_type(src->dtype(), [&]<typename T>() -> const Expression * {
                            return Var<T>{Expr<T>{_value_at(src, cc)}}.expression();
                        });
                    }
                    for (uint32_t u = 0u; u < chunk; ++u) {
                        Coord cc = _zero_coord();
                        cc[0] = r;
                        cc[1] = (Expr<uint>{cb} + u).expression();
                        _write_to(dst, cc, v[u]);
                    }
                };
                if (nchunks % _threads != 0u) {
                    _for_range(tid.expression(), _literal_u(nchunks), _literal_u(_threads),
                               [&](const Expression *cid) {
                                   _if((Expr<uint>{cid} < nchunks).expression(), [&] { emit_chunk(cid); });
                               });
                } else {
                    _for_range(tid.expression(), _literal_u(nchunks), _literal_u(_threads),
                               [&](const Expression *cid) { emit_chunk(cid); });
                }
                _pipeline_axis = saved_axis;
                _current_extent = saved;
                return;
            }
        }
          if (dst->scope() == TensorScope::Fragment && !_is_fragment_shared_backed(dst)) [[unlikely]] {
              _invalidate_lazy(dst);
              if (src->scope() == TensorScope::Global) {
                  // coalesced global->fragment staging (lc_optimize 4): the block
                  // cooperatively streams the tile through shared memory instead
                  // of every thread redundantly re-reading the whole tile
                  auto elem_t = tensor_element_type(dst->dtype());
                  auto staging = _staging_for(dst, elem_t);
                  _sync_block();// staging write-after-read hazard
                  _partition_loop(ext, [&](const Coord &c) {
                      // DSL: staging[_staging_index(dst, c)] = cast(src[c], elem_t)
                      auto idx = Expr<uint>{_staging_index(dst, c)};
                      auto val = _maybe_cast(_value_at(src, c), elem_t);
                      with_elem_type(dst->dtype(), [&]<typename T>() {
                          Var<std::array<T, 1>> arr{staging};
                          arr[idx] = Expr<T>{val};
                      });
                  });
                  _replicate_from_staging(dst, dst->dtype(), staging);
              } else {
                  _full_loop(ext, body);
              }
          } else if (ext->rank() == 2u) {
              _partition_loop_2d(ext, body);
          } else {
              _partition_loop(ext, body);
          }
        _pipeline_axis = saved_axis;
        _current_extent = saved;
    }

    /*
     * _emit_store(s) pseudo-code (luisa-dsl):
     *
     *   lhs = s->lhs()
     *   ext = operand with known extent (lhs preferred, then rhs_tensor)
     *   body(c):
     *       rhs = rhs_tensor[c] / rhs_literal / (error on rhs_ref)
     *       if s->op() == 1:          // row-broadcast scale
     *           rhs = lhs[c] * cast(rhs, elem_t)
     *       lhs[c] = rhs
     *   if lhs is Fragment and not shared-backed:
     *       _full_loop(ext, body)
     *   else:
     *       rank==2 ? _partition_loop_2d(ext, body) : _partition_loop(ext, body)
     */
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
                // DSL: lhs[c] * cast(rhs, elem_t)
                rhs = with_elem_type(lhs->dtype(), [&]<typename T>() -> const Expression * {
                    return (Expr<T>{_value_at(lhs, c)} * Expr<T>{rhs}).expression();
                });
            }
            _write_to(lhs, c, rhs);
        };
        // Lazy fragment store (see _temps_by_name): a whole-tile, non-
        // read-modify-write store into a small replicated fragment emits NO
        // device code; the value expression is inlined at the consumer, so a
        // fragment->global copy computes only its own partitioned elements
        // instead of every thread materializing the whole tile into a
        // per-thread local array (the bench_add pathology: 256 threads x
        // 256 elements of redundant local stores + spilling).
        auto lazy_ok = lhs->scope() == TensorScope::Fragment &&
                       !_is_fragment_shared_backed(lhs) &&
                       s->op() == 0 && !lhs->name().empty() &&
                       s->rhs_ref() == nullptr;
        if (lazy_ok) [[likely]] {
            _temps_by_name[luisa::string{lhs->name()}] = TempValue{
                lhs->dtype(),
                [this, s, lhs_elem_t = tensor_element_type(lhs->dtype())](const Coord &c) -> const Expression * {
                    const Expression *rhs = nullptr;
                    if (s->rhs_tensor() != nullptr) {
                        rhs = _value_at(s->rhs_tensor(), c);
                    } else {
                        rhs = _recreate_literal(s->rhs_literal());
                    }
                    return _maybe_cast(rhs, lhs_elem_t);
                }};
            _current_extent = saved;
            return;
        }
        if (lhs->scope() == TensorScope::Fragment && !_is_fragment_shared_backed(lhs)) {
            _full_loop(ext, body);
            _invalidate_lazy(lhs);
        } else if (ext->rank() == 2u) {
            _partition_loop_2d(ext, body);
        } else {
            _partition_loop(ext, body);
        }
        _current_extent = saved;
    }

    /*
     * _emit_binary(s) pseudo-code (luisa-dsl):
     *
     *   elem_t = dtype of lhs
     *   temp = TileFunctionBuilder::temp_output(s)
     *   _temps[temp] = lambda(c):      // expression inlined at the consumer
     *       l = lhs[c]
     *       r = rhs_tensor[c] / rhs_literal / (error on rhs_ref)
     *       r = cast(r, elem_t)
     *       switch op:
     *           ADD      -> l + r
     *           SUB      -> l - r
     *           MUL      -> l * r
     *           DIV      -> l / r
     *           MOD      -> l % r
     *           BIT_AND  -> l & r
     *           BIT_OR   -> l | r
     *           BIT_XOR  -> l ^ r
     *           default  -> error
     */
    void _emit_binary(const TileBinaryStmt *s) {
        auto *lhs = s->lhs();
        auto op = s->op();
        auto dtype = lhs->dtype();
        auto elem_t = tensor_element_type(dtype);
        auto temp = _tile->temp_output(s);
        _temps[temp] = TempValue{
            dtype,
            [this, s, op, dtype, elem_t](const Coord &c) -> const Expression * {
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
                // DSL: l OP r on the concrete element type
                return with_elem_type(dtype, [&]<typename T>() -> const Expression * {
                    auto le = Expr<T>{l};
                    auto re = Expr<T>{r};
                    switch (op) {
                        case BinaryOp::ADD: return (le + re).expression();
                        case BinaryOp::SUB: return (le - re).expression();
                        case BinaryOp::MUL: return (le * re).expression();
                        case BinaryOp::DIV: return (le / re).expression();
                        case BinaryOp::MOD:
                            if constexpr (std::is_integral_v<T>) { return (le % re).expression(); }
                            break;
                        case BinaryOp::BIT_AND:
                            if constexpr (std::is_integral_v<T>) { return (le & re).expression(); }
                            break;
                        case BinaryOp::BIT_OR:
                            if constexpr (std::is_integral_v<T>) { return (le | re).expression(); }
                            break;
                        case BinaryOp::BIT_XOR:
                            if constexpr (std::is_integral_v<T>) { return (le ^ re).expression(); }
                            break;
                        default:
                            LUISA_ERROR_WITH_LOCATION(
                                "tile_to_kernel: unsupported tile binary op {}.",
                                static_cast<uint32_t>(op));
                    }
                    return nullptr;// unreachable
                });
            }};
    }

    void _emit_max(const MaxStmt *s) {
        auto *a = s->a();
        auto dtype = a->dtype();
        auto elem_t = tensor_element_type(dtype);
        _temps[_tile->temp_output(s)] = TempValue{
            dtype,
            [this, s, a, dtype, elem_t](const Coord &c) -> const Expression * {
                auto av = _value_at(a, c);
                auto bv = _maybe_cast(_recreate_literal(s->b()), elem_t);
                // DSL: max(a[c], cast(b_literal, elem_t))
                return with_elem_type(dtype, [&]<typename T>() -> const Expression * {
                    return max(Expr<T>{av}, Expr<T>{bv}).expression();
                });
            }};
    }

    void _emit_rsqrt(const RsqrtStmt *s) {
        auto *a = s->a();
        auto dtype = a->dtype();
        auto elem_t = tensor_element_type(dtype);
        _temps[_tile->temp_output(s)] = TempValue{
            dtype,
            [this, s, a, dtype, elem_t](const Coord &c) -> const Expression * {
                auto av = _value_at(a, c);
                // DSL: rsqrt(a[c]) — floating element types only; integral
                // tiles fall back to the dtype-erased call (never exercised
                // by the tile IR, which only rsqrt's F16/F32 tiles).
                return with_elem_type(dtype, [&]<typename T>() -> const Expression * {
                    if constexpr (std::is_floating_point_v<T>) {
                        return rsqrt(Expr<T>{av}).expression();
                    } else {
                        return _fb->call(elem_t, CallOp::RSQRT, {av});
                    }
                });
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
        // DSL form of the same Horner approximation (all f32 Expr ops).
        auto xf = Expr<float>{_maybe_cast(x, Type::of<float>())};
        auto zero = Expr<float>{0.f};
        auto one = Expr<float>{1.f};
        // t = 1 / (1 + p*|x|)
        auto absx = abs(xf);
        auto p = Expr<float>{0.3275911f};
        auto t = one / (one + p * absx);
        // Horner form of a1*t + a2*t^2 + a3*t^3 + a4*t^4 + a5*t^5
        auto a5 = Expr<float>{1.061405429f};
        auto a4 = Expr<float>{-1.453152027f};
        auto a3 = Expr<float>{1.421413741f};
        auto a2 = Expr<float>{-0.284496736f};
        auto a1 = Expr<float>{0.254829592f};
        auto poly = ((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t;
        // exp(-x^2)
        auto x2 = absx * absx;
        auto e = exp(-x2);
        auto erf_abs = one - poly * e;
        // sign(x) * erf(|x|): (x < 0) ? -erf_abs : erf_abs
        auto result = select(erf_abs, -erf_abs, xf < zero);
        return _maybe_cast(result.expression(), result_t);
    }

    void _emit_fast_math(const FastMathStmt *s) {
        auto *a = s->a();
        auto dtype = a->dtype();
        auto elem_t = tensor_element_type(dtype);
        _temps[_tile->temp_output(s)] = TempValue{
            dtype,
            [this, s, a, dtype, elem_t](const Coord &c) -> const Expression * {
                auto av = _value_at(a, c);
                // DSL: exp/exp10/log/log2/log10/sin/cos/tan/tanh on Expr<T>
                return with_elem_type(dtype, [&]<typename T>() -> const Expression * {
                    auto x = Expr<T>{av};
                    if constexpr (std::is_floating_point_v<T>) {
                        switch (s->op()) {
                            case TileFastMathOp::EXP: return exp(x).expression();
                            case TileFastMathOp::EXP10: return exp10(x).expression();
                            case TileFastMathOp::LOG: return log(x).expression();
                            case TileFastMathOp::LOG2: return log2(x).expression();
                            case TileFastMathOp::LOG10: return log10(x).expression();
                            case TileFastMathOp::SIN: return sin(x).expression();
                            case TileFastMathOp::COS: return cos(x).expression();
                            case TileFastMathOp::TAN: return tan(x).expression();
                            case TileFastMathOp::TANH: return tanh(x).expression();
                            case TileFastMathOp::ERF: return _erf(av, elem_t);
                            default:
                                LUISA_ERROR_WITH_LOCATION(
                                    "tile_to_kernel: unsupported fast math op {}.",
                                    static_cast<uint32_t>(s->op()));
                        }
                    } else {
                        // integral/byte tiles: dtype-erased fallback
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
                    }
                    return nullptr;
                });
            }};
    }

    void _emit_ieee_math(const IeeeMathStmt *s) {
        auto *a = s->a();
        auto *b = s->b();
        auto dtype = a->dtype();
        auto elem_t = tensor_element_type(dtype);
        // For CAST, the result type is the cast target dtype.
        auto result_dtype = (s->op() == TileIeeeOp::CAST) ? s->cast_dtype() : a->dtype();
        auto result_elem_t = tensor_element_type(result_dtype);
        _temps[_tile->temp_output(s)] = TempValue{
            result_dtype,
            [this, s, a, b, dtype, elem_t, result_dtype, result_elem_t](const Coord &c) -> const Expression * {
                auto av = _value_at(a, c);
                // CAST changes the value type: the source and result dtypes
                // differ, so keep the dtype-erased cast expression.
                if (s->op() == TileIeeeOp::CAST) {
                    return _fb->cast(result_elem_t, CastOp::STATIC, av);
                }
                // DSL form of the remaining ops on the result element type
                // (for ISINF/ISNAN the result dtype equals the source dtype).
                return with_elem_type(result_dtype, [&]<typename T>() -> const Expression * {
                    auto x = Expr<T>{av};
                    if constexpr (std::is_floating_point_v<T>) {
                        switch (s->op()) {
                            case TileIeeeOp::SQRT:
                            case TileIeeeOp::FSQRT:
                                return sqrt(x).expression();
                            case TileIeeeOp::POW: {
                                LUISA_ASSERT(b != nullptr,
                                             "tile_to_kernel: ieee POW requires a second "
                                             "input tensor (b).");
                                auto bv = Expr<T>{_value_at(b, c)};
                                return pow(x, bv).expression();
                            }
                            case TileIeeeOp::CEIL:
                                return ceil(x).expression();
                            case TileIeeeOp::FLOOR:
                                return floor(x).expression();
                            case TileIeeeOp::ROUND:
                                return round(x).expression();
                            case TileIeeeOp::ISINF:
                            case TileIeeeOp::ISNAN: {
                                // ISINF/ISNAN produce a *boolean* predicate in the
                                // core IR; cast it back to the element type.
                                auto pred = s->op() == TileIeeeOp::ISINF ? isinf(x) : isnan(x);
                                return cast<T>(pred).expression();
                            }
                            default:
                                LUISA_ERROR_WITH_LOCATION(
                                    "tile_to_kernel: unsupported ieee math op {}.",
                                    static_cast<uint32_t>(s->op()));
                        }
                    } else {
                        // integral/byte tiles: dtype-erased fallback
                        auto raw = [&](CallOp call) { return _fb->call(elem_t, call, {av}); };
                        switch (s->op()) {
                            case TileIeeeOp::SQRT:
                            case TileIeeeOp::FSQRT: return raw(CallOp::SQRT);
                            case TileIeeeOp::POW: {
                                LUISA_ASSERT(b != nullptr,
                                             "tile_to_kernel: ieee POW requires a second "
                                             "input tensor (b).");
                                return _fb->call(elem_t, CallOp::POW, {av, _value_at(b, c)});
                            }
                            case TileIeeeOp::CEIL: return raw(CallOp::CEIL);
                            case TileIeeeOp::FLOOR: return raw(CallOp::FLOOR);
                            case TileIeeeOp::ROUND: return raw(CallOp::ROUND);
                            case TileIeeeOp::ISINF:
                            case TileIeeeOp::ISNAN:
                                return _fb->cast(elem_t, CastOp::STATIC,
                                                 _fb->call(Type::of<bool>(),
                                                           s->op() == TileIeeeOp::ISINF ? CallOp::ISINF : CallOp::ISNAN,
                                                           {av}));
                            default:
                                LUISA_ERROR_WITH_LOCATION(
                                    "tile_to_kernel: unsupported ieee math op {}.",
                                    static_cast<uint32_t>(s->op()));
                        }
                    }
                    return nullptr;
                });
            }};
    }

    // warp-collective tile reduction (lc_optimize 3.2): output elements are
    // assigned to whole warps; lanes stride the reduce axis with an
    // identity-padded guard, a built-in warp all-reduce combines the partials
    // (no shared memory, no barrier), and lane 0 writes the result.
    // Fragment outputs (replicated layout) are published through a shared
    // staging tile and then re-replicated into every thread's local copy.
    /*
     * _emit_tile_reduce(x, y, dim, op) pseudo-code (luisa-dsl):
     *
     * reduce_len = extent of x along dim
     * out_count  = product of x's extents except dim
     * UInt lane = warp_lane_id(); UInt lanes = warp_lane_count()
     * UInt warp = slice-local warp id; UInt nw = warps per slice
     * UInt k_iters = ceildiv(reduce_len, lanes)
     * UInt o_iters = ceildiv(out_count, nw)
     * if y is Fragment:
     * staging = _staging_for(y) // Shared<T> staging{n}
     * sync_block()
     *
     * if out_count < nw:            // FEW OUTPUTS -> BLOCK-WIDE reduction
     *   // every warp reduces a strided slice of the reduce axis, then the
     *   // per-warp partials are combined through Shared (lc_optimize 4.5):
     *   workspace = Shared<T> workspace{nw}   // one slot per warp
     *   $for (ki, 0u, k_iters) {              // k = ki*lanes + lane (same as below)
     *       v = identity(op); $if (k < reduce_len) { xc[dim]=k; v = _value_at(x, xc); };
     *       acc = combine(op, acc, v);
     *   };
     *   total = _warp_reduce(op, acc);
     *   $if (lane == 0) { workspace[warp] = total; };
     *   sync_block();
     *   $if (warp == 0) {                     // warp 0 reduces the partials
     *       block = identity(op);
     *       $for (w, lane, nw, lanes) { block = combine(op, block, workspace[w]); };
     *       block = _warp_reduce(op, block);
     *       $if (lane == 0) { write y[yc] = block; };  // single output
     *   };
     * else:                          // NORMAL warp-per-output partition
     *   $for (oi, 0u, o_iters) {
     *   UInt o = oi * nw + warp;
     *   $if (o < out_count) {
     *   // decompose o -> coords for all axes except dim
     *   acc = identity(op);
     *   $for (ki, 0u, k_iters) {
     *   UInt k = ki * lanes + lane;
     *   v = identity(op);
     *   $if (k < reduce_len) {
     *   xc[dim] = k;
     *   v = _value_at(x, xc);
     *   $if (op is ABS_SUM/ABS_MAX) { v = abs(v); };
     *   };
     *   acc = combine(op, acc, v);
     *   };
     *   total = _warp_reduce(op, acc); // XOR butterfly over lanes
     *   $if (lane == 0) {
     *   $if (y is Fragment) { staging[_staging_index(y, yc)] = cast(total, out_t); };
     *   $else { _write_to(y, yc, total); };
     *   };
     *   };
     *   };
     * }
     * if y is Fragment:
     * _replicate_from_staging(y, staging)
     */
    void _emit_tile_reduce(const TensorExpr *x, const TensorExpr *y,
                           uint32_t dim, TileReduceOp op) {
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
        // Slice-local warp partition: each batch item's output space is
        // covered by its own slice's warps (see _num_warps / _slice_warp).
        auto warp = _slice_warp();
        auto nw = _num_warps();
        auto k_iters = _ceildiv_expr(_literal_u(reduce_len), lanes);
        auto o_iters = _ceildiv_expr(_literal_u(out_count), nw);
        const bool frag_out = y->scope() == TensorScope::Fragment;
        auto out_t = tensor_element_type(y->dtype());
        const RefExpr *staging = frag_out ? _staging_for(y, out_t) : nullptr;
        if (frag_out) { _sync_block(); }// staging write-after-read hazard
        // The element arithmetic is dtype-generic (runtime tag), so the whole
        // device body is written in the DSL sugar inside with_elem_type:
        //   Var<T> acc = identity; $for (...) { ...; acc = combine(op, acc, v); }
        //   total = warp_reduce_typed<T>(op, acc); ...
        with_elem_type(x->dtype(), [&]<typename T>() {
            auto identity_v = [&]() -> const Expression * {
                return _reduce_identity(op, x->dtype());
            };
            // Block-wide two-level reduction (plan 2.5 / lc_optimize 4.5): when
            // the output space is smaller than the warp count, the default
            // warp-per-output partition would leave most warps idle.  Instead
            // every warp reduces its own strided slice of the reduce axis, lane 0
            // publishes the per-warp partial into a shared workspace, and warp 0
            // combines the partials and writes the output.
            const uint32_t nw_est = _threads / 32u;// host, upper bound for warps
            if (out_count < nw_est && !_batching) {
                Shared<T> workspace{nw_est};
                auto block_k_iters = (reduce_len + _threads - 1u) / _threads;
                for (uint32_t o = 0u; o < out_count; ++o) {// out_count is small
                    // decompose host-constant o -> coords for all axes except dim
                    Coord xc = _zero_coord();
                    Coord yc = _zero_coord();
                    auto rem = o;
                    for (int32_t i = static_cast<int32_t>(x->rank()) - 1; i >= 0; --i) {
                        auto ui = static_cast<uint32_t>(i);
                        if (ui == dim) { continue; }
                        auto e = static_cast<uint32_t>(axis_extent(x, ui));
                        auto ci = rem % e;
                        rem /= e;
                        xc[ui] = _literal_u(ci);
                        yc[ui < dim ? ui : ui - 1u] = _literal_u(ci);
                    }
                    // per-thread partial over the whole slice's strided reduce axis
                    auto acc = Var<T>{Expr<T>{identity_v()}};
                    // Guard elision: _threads is host-known, so when the
                    // reduce extent divides evenly every strided k is valid.
                    const bool k_tail_free = reduce_len % _threads == 0u;
                    _for_range(_literal_u(0u), _literal_u(block_k_iters), _literal_u(1u),
                               [&](const Expression *ki) {
                        auto k = (Expr<uint>{ki} * _threads + Expr<uint>{_tid_x()}).expression();
                        auto load = [&] {
                            xc[dim] = k;
                            auto xv = _maybe_cast(_value_at(x, xc), Type::of<T>());
                            if (op == TileReduceOp::ABS_SUM || op == TileReduceOp::ABS_MAX) {
                                xv = abs(Expr<T>{xv}).expression();
                            }
                            return xv;
                        };
                        if (k_tail_free) [[likely]] {
                            acc = Expr<T>{_combine_expr<T>(op, acc.expression(), load())};
                        } else {
                            auto v = Var<T>{Expr<T>{identity_v()}};
                            auto k_valid = (Expr<uint>{k} < reduce_len).expression();
                            _if(k_valid, [&] { v = Expr<T>{load()}; });
                            acc = Expr<T>{_combine_expr<T>(op, acc.expression(), v.expression())};
                        }
                    });
                    // warp-level combine, lane 0 publishes to the workspace
                    auto total = _warp_reduce_typed<T>(op, acc.expression());
                    auto is_lane0 = (Expr<uint>{lane} == 0u).expression();
                    _if(is_lane0, [&] {
                        workspace[Expr<uint>{warp}] = Expr<T>{total};
                    });
                    _sync_block();// publish workspace before warp 0 reads it
                    // warp 0 reduces the per-warp partials and writes the output
                    auto is_warp0 = (Expr<uint>{warp} == 0u).expression();
                    _if(is_warp0, [&] {
                        auto val = Var<T>{Expr<T>{identity_v()}};
                        auto lane_valid = (Expr<uint>{lane} < Expr<uint>{nw}).expression();
                        _if(lane_valid, [&] {
                            val = Expr<T>{workspace[Expr<uint>{lane}].expression()};
                        });
                        auto block = _warp_reduce_typed<T>(op, val.expression());
                        auto lane0 = (Expr<uint>{lane} == 0u).expression();
                        _if(lane0, [&] {
                            if (frag_out) {
                        auto sidx = Expr<uint>{_staging_index(y, yc)};
                                auto bcast = _maybe_cast(block, out_t);
                                with_elem_type(y->dtype(), [&]<typename U>() {
                                    Var<std::array<U, 1>> arr{staging};
                                    arr[sidx] = Expr<U>{bcast};
                                });
                            } else {
                                _write_to(y, yc, block);
                            }
                        });
                    });
                    _sync_block();// avoid WAR on workspace before the next o iteration
                }
            } else {
                // ---- normal warp-per-output partition (existing path) ----
                _for_range(_literal_u(0u), o_iters, _literal_u(1u),
                           [&](const Expression *oi) {
                    auto o = (Expr<uint>{oi} * Expr<uint>{nw} + Expr<uint>{warp}).expression();
                    auto o_valid = (Expr<uint>{o} < out_count).expression();
                    _if(o_valid, [&] {
                        // decompose o over x's shape minus the reduce axis
                        Coord xc = _zero_coord();
                        Coord yc = _zero_coord();
                        auto rem = Var<uint>{Expr<uint>{o}};
                        for (int32_t i = static_cast<int32_t>(x->rank()) - 1; i >= 0; --i) {
                            auto ui = static_cast<uint32_t>(i);
                            if (ui == dim) { continue; }
                            auto e = Expr<uint>{static_cast<uint32_t>(axis_extent(x, ui))};
                            auto ci = (rem % e).expression();
                            rem = rem / e;
                            xc[ui] = ci;
                            yc[ui < dim ? ui : ui - 1u] = ci;
                        }
                        // per-lane partial over the strided reduce axis.
                        // Full/tail split (lc_optimize: guard elision): the
                        // bounds guard is only needed in the last chunk, so the
                        // hot full chunks load/combine unconditionally (no
                        // identity-init, no predicated branch per element); the
                        // tail chunk keeps the identity-padded guard.  When the
                        // reduce extent is a host-known multiple of every
                        // possible power-of-two warp size (<= 128), the tail is
                        // provably empty and is not emitted at all.
                        auto acc = Var<T>{Expr<T>{identity_v()}};
                        auto full_k = (Expr<uint>{reduce_len} / Expr<uint>{lanes}).expression();
                        const bool tail_free = reduce_len % 128u == 0u;
                        _for_range(_literal_u(0u), full_k, _literal_u(1u),
                                   [&](const Expression *ki) {
                            auto k = (Expr<uint>{ki} * Expr<uint>{lanes} + Expr<uint>{lane}).expression();
                            xc[dim] = k;
                            auto xv = _maybe_cast(_value_at(x, xc), Type::of<T>());
                            if (op == TileReduceOp::ABS_SUM || op == TileReduceOp::ABS_MAX) {
                                xv = abs(Expr<T>{xv}).expression();
                            }
                            acc = Expr<T>{_combine_expr<T>(op, acc.expression(), xv)};
                        });
                        if (!tail_free) [[likely]] {
                            _if((Expr<uint>{full_k} < Expr<uint>{k_iters}).expression(), [&] {
                                auto k = (Expr<uint>{full_k} * Expr<uint>{lanes} + Expr<uint>{lane}).expression();
                                auto v = Var<T>{Expr<T>{identity_v()}};
                                auto k_valid = (Expr<uint>{k} < reduce_len).expression();
                                _if(k_valid, [&] {
                                    xc[dim] = k;
                                    auto xv = _maybe_cast(_value_at(x, xc), Type::of<T>());
                                    if (op == TileReduceOp::ABS_SUM || op == TileReduceOp::ABS_MAX) {
                                        xv = abs(Expr<T>{xv}).expression();
                                    }
                                    v = Expr<T>{xv};
                                });
                                acc = Expr<T>{_combine_expr<T>(op, acc.expression(), v.expression())};
                            });
                        }
                        auto total = _warp_reduce_typed<T>(op, acc.expression());
                        auto is_lane0 = (Expr<uint>{lane} == 0u).expression();
                        _if(is_lane0, [&] {
                            if (frag_out) {
                                auto sidx = Expr<uint>{_staging_index(y, yc)};
                                auto tcast = _maybe_cast(total, out_t);
                                with_elem_type(y->dtype(), [&]<typename U>() {
                                    Var<std::array<U, 1>> arr{staging};
                                    arr[sidx] = Expr<U>{tcast};
                                });
                            } else {
                                _write_to(y, yc, total);
                            }
                        });
                    });
                });
            }
        });
        if (frag_out) { _replicate_from_staging(y, y->dtype(), staging); }
        _current_extent = saved;
    }

    /*
     * _emit_reduce_sum(s) pseudo-code:
     *
     *   _emit_tile_reduce(s->x(), s->y(), s->dim(), TileReduceOp::SUM)
     */
    void _emit_reduce_sum(const ReduceSumStmt *s) {
        _emit_tile_reduce(s->x(), s->y(), s->dim(), TileReduceOp::SUM);
    }

    /*
     * _emit_gemm(s) pseudo-code (luisa-dsl, SIMT fallback):
     *   register-tiled per-thread micro-tiles, or warp-K-split when the
     *   per-thread mapping would leave lanes idle (lc_optimize 2.1/2.6).
     *
     *   if use_cooperative:
     *       return _emit_gemm_cooperative(s)
     *   a, b, c = operands
     *   K = extent of a along axis 1
     *   _current_extent = c
     *   TM/TN = largest of {4,2,1} dividing M/N; MT = M/TM; NT = N/TN
     *
     *   // ---- warp-K-split path (lc_optimize 2.1/2.6) --------------------
     *   // Active when !cooperative && !batching && threads >= 32 &&
     * // MT*NT < threads (idle-lane case) && K >= 256 (reduction amortized).
     *   if use_warp_gemm:
     *       lane = warp_lane_id(); lanes = warp_lane_count()
     *       wid = _warp_id(); nw = _num_warps()
     *       // warp-level 2D strided partition of the MT x NT micro-tile grid
     *       tw = min(nw, NT); th = ceildiv(nw, tw)
     *       r0 = wid / tw; c0 = wid % tw
     *       $for (r, r0, MT, th) {
     *         $for (c, c0, NT, tw) {
     *           Float acc[TM][TN] = 0.f;                // per-lane K-slice partial
     *           $for (kk, 0u, ceildiv(K, lanes)) {      // lane-strided K
     *               k = kk * lanes + lane;
     *               $if (k < K) {                       // only when K % lanes != 0
     *                   // resolve ac, bc according to trans_a / trans_b
     *                   a_row[i] = cast(a[ac_i][k], float);   // TM loads
     *                   b_col[j] = cast(b[k][bc_j], float);   // TN loads
     *                   acc[i][j] = fma(a_row[i], b_col[j], acc[i][j]);
     *               };
     *           };
     *           for i in 0..TM: {
     *               for j in 0..TN: {                   // scalar all-reduce
     *                   acc[i][j] = warp_active_sum(acc[i][j])  // every lane gets the tile
     *               }
     *           }
     *           if !clear_accum: acc[i][j] += cast(c[r+i][c+j], float)
     *           // write-back
     *           if frag && nw == 1:                     // single-warp block
     *               _write_to(c, (r+i, c+j), acc[i][j]) // each lane's OWN replica
     *           else:
     *               $if (lane == 0u) {
     *                   if frag: staging[_staging_index(c, (r+i,c+j))] = cast(acc[i][j], c_dtype)
     *                   else:   _write_to(c, (r+i, c+j), acc[i][j])
     *               };
     *         };
     *       };
     *       if frag && nw > 1: { sync_block(); _replicate_from_staging(c, staging); }
     *       return
     *
     *   // ---- per-thread TM x TN register micro-tile (lc_optimize: GEMM) --
     *   // C is partitioned into a grid of micro-tiles (MT = ceil(M/TM),
     *   // NT = ceil(N/TN)); each thread owns one micro-tile per iteration.
     *   // Thread -> micro-tile mapping honors GemmWarpPolicy:
     *   //   Square   -> 2D strided partition over (MT, NT)
     *   //   FullRow  -> rows split across threads, each thread walks NT cols
     *   //   FullCol  -> cols split across threads, each thread walks MT rows
     *   // TM/TN default to 4x4; 1x1 fallback for tiny tiles (M<2 || N<2).
     *   compute_acc(r, c):               // r,c = micro-tile top-left coord
     *       Float acc[TM][TN] = clear_accum ? 0.f : cast(c[r+i][c+j], float);
     *       $for (kk, 0u, ceildiv(K, k_pack)) {          // k_pack unroll
     *           $for (u, 0u, k_pack) {                    // host-unrolled
     *               k = kk * k_pack + u;
     *               $if (k < K) {
     *                   // resolve ac, bc according to trans_a / trans_b
     *                   a_row[i] = cast(a[ac_i][k], float);  // TM loads
     *                   b_col[j] = cast(b[k][bc_j], float);  // TN loads
     *                   acc[i][j] = fma(a_row[i], b_col[j], acc[i][j]); // TM*TN FMA
     *               };
     *           };
     *       };
     *       return acc;
     *   emit_micro_tile(r, c):           // write back the TM x TN tile
     *       if c is Fragment and not shared-backed:
     *           staging[_staging_index(c, (r+i, c+j))] = cast(acc[i][j], c_dtype)
     *       else:
     *           _write_to(c, (r+i, c+j), acc[i][j])
     *   if c is Fragment and not shared-backed:
     *       staging = _staging_for(c)
     *       sync_block()                                  // staging write-after-read
     *       partition micro-tiles; for each: compute_acc + write staging
     *       _replicate_from_staging(c, staging)
     *   else:
     *       partition micro-tiles; for each: compute_acc + _write_to(c, ...)
     */
    void _emit_gemm(const GemmStmt *s) {
        if (_use_cooperative) {
            _emit_gemm_cooperative(s);
            return;
        }
        auto *a = s->a();
        auto *b = s->b();
        auto *c = s->c();
        auto wide_t = Type::of<float>();// f16 inputs accumulate in f32
        auto out_t = tensor_element_type(c->dtype());
        auto saved = _current_extent;
        _current_extent = c;
        // ---- per-thread TM x TN register micro-tile (plan 2.4 / lc_optimize) ---
        // C is partitioned into a grid of MT x NT micro-tiles of TM x TN
        // elements; each thread owns one micro-tile per iteration, so each
        // k-step issues only TM + TN A/B loads for TM*TN FMAs.
        auto M = static_cast<uint32_t>(axis_extent(c, 0u));
        auto N = static_cast<uint32_t>(axis_extent(c, 1u));
        auto K = static_cast<uint32_t>(axis_extent(a, 1u));
        // largest of {4, 2, 1} that divides M/N (1x1 fallback for primes)
        auto TM = (M % 4u == 0u) ? 4u : ((M % 2u == 0u) ? 2u : 1u);
        auto TN = (N % 4u == 0u) ? 4u : ((N % 2u == 0u) ? 2u : 1u);
        auto MT = M / TM;// micro-tile grid (exact division)
        auto NT = N / TN;
        // clamp k_pack into [1, K]; a degenerate K == 0 tile keeps k_pack = 1
        // so k_iters = 0 (the K loop is skipped, exactly like the old lowering)
        auto k_pack = static_cast<uint32_t>(std::min(std::max(s->k_pack(), 1),
                                                     static_cast<int32_t>(std::max(K, 1u))));
        if (s->k_pack() <= 1 && K % 8u == 0u && K > k_pack) {
            // the tile DSL defaulted k_pack to 1 (a fully dynamic K loop);
            // unroll 8-deep to cut the loop/index overhead in the inner loop
            k_pack = 8u;
        }
        auto k_iters = (K + k_pack - 1u) / k_pack;
        const bool frag = c->scope() == TensorScope::Fragment && !_is_fragment_shared_backed(c);
        const RefExpr *staging = nullptr;
        // ---- warp-K-split path (lc_optimize 2.1/2.6) ------------------------
        // Active when !cooperative && !batching && threads >= 32 &&
        // MT*NT < threads (idle-lane case) && K >= 256 (reduction amortized).
        // Measured on the 4096^3 f32 CUDA benchmark: ~5% faster than the
        // per-thread path at block_K = 256 (K = 256 >= the gate) and a clear
        // regression below it, hence the K >= 256 threshold.
        // Each warp walks a warp-strided slice of the MT x NT micro-tile grid;
        // every lane accumulates a K-slice of one micro-tile and the per-row
        // partials are finished by a vector WARP_ACTIVE_SUM all-reduce (no
        // shared memory, no barrier inside the warp path).
        auto host_nw = _threads / 32u;// warp size is pinned to 32
        auto use_warp = !_use_cooperative && !_batching &&
                        _threads >= 32u &&
                        (MT * NT) < _threads &&
                        K >= 256u;
        if (!use_warp) {
            // Occupancy (lc_optimize: thread-group tuning): when the
            // micro-tile grid is smaller than the block, the remaining
            // threads idle for the whole K loop.  Shrink TM/TN (keeping
            // exact division) until the grid covers the block — trading
            // FMA:shared-load ratio for parallelism, a large win whenever
            // MT*NT < threads (e.g. the 16x16 C tile / 256-thread
            // bench_gemm: 4x4 micro-tiles keep only 16 of 256 threads
            // busy; 1x1 micro-tiles engage all 256).
            while ((TM > 1u || TN > 1u) && MT * NT < _threads) {
                if (TM >= TN && TM > 1u) { TM >>= 1u; } else { TN >>= 1u; }
                MT = M / TM;
                NT = N / TN;
            }
        }
        if (use_warp) {
            auto lanes = _lane_count();// runtime expr
            auto lane = _lane();       // warp_lane_id()
            auto wid = _warp_id();     // runtime expr; batching disabled here
            // ---- fused single-warp path (host_nw == 1) ---------------------
            // The whole MT x NT micro-tile grid belongs to the single warp,
            // so ALL micro-tiles accumulate in ONE lane-strided K loop and
            // share their A-row / B-column shared loads: M + N loads per
            // k-step feed M*N FMAs (vs. (TM+TN) loads per TM*TN FMAs when
            // each tile loops separately).  Register-capped at 64 f32
            // accumulators; larger grids use the per-tile loop below.
            if (host_nw == 1u && M * N <= 64u) [[likely]] {
                auto k_iters_w = (K + 32u - 1u) / 32u;
                std::array<Var<float>, 64> acc{};
                // one k-step: M A loads + N B loads feed all M*N FMAs (the A
                // row / B col values are shared across the micro-tile grid)
                auto emit_k = [&](const Expression *k) {
                    std::array<const Expression *, 64> a_vals{};
                    for (uint32_t rr = 0u; rr < M; ++rr) {
                        Coord ac = _zero_coord();
                        if (s->trans_a() != 0) {
                            ac[0] = k;
                            ac[1] = _literal_u(rr);
                        } else {
                            ac[0] = _literal_u(rr);
                            ac[1] = k;
                        }
                        a_vals[rr] = Var<float>{Expr<float>{_maybe_cast(_value_at(a, ac), wide_t)}}.expression();
                    }
                    std::array<const Expression *, 64> b_vals{};
                    for (uint32_t cc = 0u; cc < N; ++cc) {
                        Coord bc = _zero_coord();
                        if (s->trans_b() != 0) {
                            bc[0] = _literal_u(cc);
                            bc[1] = k;
                        } else {
                            bc[0] = k;
                            bc[1] = _literal_u(cc);
                        }
                        b_vals[cc] = Var<float>{Expr<float>{_maybe_cast(_value_at(b, bc), wide_t)}}.expression();
                    }
                    for (uint32_t rr = 0u; rr < M; ++rr) {
                        for (uint32_t cc = 0u; cc < N; ++cc) {
                            acc[rr * N + cc] = fma(Expr<float>{a_vals[rr]},
                                                   Expr<float>{b_vals[cc]},
                                                   Expr<float>{acc[rr * N + cc].expression()});
                        }
                    }
                };
                if (K % 32u == 0u && k_iters_w <= 16u) {
                    // host-unrolled K loop (lanes pinned to 32): removes the
                    // dynamic $for overhead and folds kk*lanes into literals
                    for (uint32_t kk = 0u; kk < k_iters_w; ++kk) {
                        emit_k((Expr<uint>{lane} + kk * 32u).expression());
                    }
                } else {
                    _for_range(_literal_u(0u), _literal_u(k_iters_w), _literal_u(1u),
                               [&](const Expression *kk) {
                        // DSL: k = kk * lanes + lane
                        auto k = (Expr<uint>{kk} * Expr<uint>{lanes} + Expr<uint>{lane}).expression();
                        if (K % 32u != 0u) {// k may run past K on the last kk
                            _if((Expr<uint>{k} < K).expression(), [&] { emit_k(k); });
                        } else {
                            emit_k(k);
                        }
                    });
                }
                // one scalar all-reduce per C element: every lane receives the
                // finished tile (see the per-tile path for why this is not a
                // packed vector reduce)
                for (uint32_t rr = 0u; rr < M; ++rr) {
                    for (uint32_t cc = 0u; cc < N; ++cc) {
                        acc[rr * N + cc] = warp_active_sum(Expr<float>{acc[rr * N + cc].expression()});
                    }
                }
                // clear_accum == 0: add the existing C element (uniform; the
                // per-lane K-slice partial is already all-reduced)
                if (s->clear_accum() == 0) {
                    for (uint32_t rr = 0u; rr < M; ++rr) {
                        for (uint32_t cc = 0u; cc < N; ++cc) {
                            Coord cd = _zero_coord();
                            cd[0] = _literal_u(rr);
                            cd[1] = _literal_u(cc);
                            acc[rr * N + cc] = Expr<float>{acc[rr * N + cc].expression()} +
                                               Expr<float>{_maybe_cast(_value_at(c, cd), wide_t)};
                        }
                    }
                }
                // write-back: fragment C (single-warp block) -> every lane
                // writes its OWN replica, barrier-free; otherwise lane 0
                // writes each element exactly once
                auto write_tile = [&] {
                    for (uint32_t rr = 0u; rr < M; ++rr) {
                        for (uint32_t cc = 0u; cc < N; ++cc) {
                            Coord cd = _zero_coord();
                            cd[0] = _literal_u(rr);
                            cd[1] = _literal_u(cc);
                            _write_to(c, cd, _maybe_cast(acc[rr * N + cc].expression(), out_t));
                        }
                    }
                };
                if (frag) {
                    write_tile();
                    // the direct replica writes above replace any lazy
                    // fragment value recorded earlier (the documented
                    // single-warp invalidation gap)
                    _invalidate_lazy(c);
                } else {
                    _if((Expr<uint>{lane} == 0u).expression(), write_tile);
                }
                _current_extent = saved;
                return;
            }
            // warp-level 2D strided partition of the MT x NT micro-tile grid
            auto tw = std::min(host_nw, NT);
            auto th = (host_nw + tw - 1u) / tw;
            // DSL: r0 = wid / tw; c0 = wid % tw
            auto r0 = (Expr<uint>{wid} / Expr<uint>{tw}).expression();
            auto c0 = (Expr<uint>{wid} % Expr<uint>{tw}).expression();
            // per-micro-tile, per-lane: K-slice partials + vector all-reduce
            auto warp_micro_tile = [&](const Expression *rt, const Expression *ct) {
                // DSL: r = rt * TM; ct0 = ct * TN
                auto r = (Expr<uint>{rt} * TM).expression();
                auto ct0 = (Expr<uint>{ct} * TN).expression();
                // TM*TN f32 accumulator locals; always start from 0.f because
                // the per-lane partial covers only a K-slice (clear_accum==0
                // C-addition happens after the all-reduce below).  Var<float>
                // default-init emits the same local + 0.f initializer as the
                // old _fb->local/_fb->assign pair.
                std::array<Var<float>, 16> acc{};
                // lane-strided K loop (lanes pinned to 32); the tail guard only
                // wraps the loads/FMAs, so every lane still reaches the reduce
                auto k_iters_w = (K + 32u - 1u) / 32u;
                _for_range(_literal_u(0u), _literal_u(k_iters_w), _literal_u(1u),
                           [&](const Expression *kk) {
                    // DSL: k = kk * lanes + lane
                    auto k = (Expr<uint>{kk} * Expr<uint>{lanes} + Expr<uint>{lane}).expression();
                    auto emit_k = [&] {
                        // TM A loads + TN B loads, then TM*TN FMAs (same
                        // trans_a/trans_b indexing as the per-thread path)
                        std::array<const Expression *, 16> a_row{};
                        for (uint32_t i = 0u; i < TM; ++i) {
                            Coord ac = _zero_coord();
                            auto ri = (Expr<uint>{r} + i).expression();
                            if (s->trans_a() != 0) {
                                ac[0] = k;
                                ac[1] = ri;
                            } else {
                                ac[0] = ri;
                                ac[1] = k;
                            }
                            // DSL local: Var<float> initialized from the cast A value
                            a_row[i] = Var<float>{Expr<float>{_maybe_cast(_value_at(a, ac), wide_t)}}.expression();
                        }
                        std::array<const Expression *, 16> b_col{};
                        for (uint32_t j = 0u; j < TN; ++j) {
                            Coord bc = _zero_coord();
                            auto cj = (Expr<uint>{ct0} + j).expression();
                            if (s->trans_b() != 0) {
                                bc[0] = cj;
                                bc[1] = k;
                            } else {
                                bc[0] = k;
                                bc[1] = cj;
                            }
                            // DSL local: Var<float> initialized from the cast B value
                            b_col[j] = Var<float>{Expr<float>{_maybe_cast(_value_at(b, bc), wide_t)}}.expression();
                        }
                        for (uint32_t i = 0u; i < TM; ++i) {
                            for (uint32_t j = 0u; j < TN; ++j) {
                                // DSL: acc[i][j] = fma(a_row[i], b_col[j], acc[i][j])
                                acc[i * TN + j] = fma(Expr<float>{a_row[i]},
                                                      Expr<float>{b_col[j]},
                                                      Expr<float>{acc[i * TN + j].expression()});
                            }
                        }
                    };
                    if (K % 32u != 0u) {// k may run past K on the last kk
                        _if((Expr<uint>{k} < K).expression(), emit_k);
                    } else {
                        emit_k();
                    }
                });
                // scalar all-reduce per micro-tile element: every lane receives
                // the full TM x TN tile (uniform; no sync_block inside a $if).
                // A packed vector WARP_ACTIVE_SUM would cut the reduction cost
                // ~4x, but the CUDA XIR codegen currently prints the float4
                // make_vector AGGREGATE with the wrong element type (lc_make_uint4
                // -> NVRTC mismatch), so the portable per-component reduce is used
                // (like _emit_warp_reduce).  The warp path still wins when the
                // per-lane K-slice is long enough to amortize the reductions
                // (K >= ~256 at 8x8 blocks / 32 threads).
                for (uint32_t i = 0u; i < TM; ++i) {
                    for (uint32_t j = 0u; j < TN; ++j) {
                        // DSL: acc[i][j] = warp_active_sum(acc[i][j])
                        acc[i * TN + j] = warp_active_sum(Expr<float>{acc[i * TN + j].expression()});
                    }
                }
                // clear_accum == 0: every lane adds the existing C element to
                // its (now full) tile; a uniform add, no divergence issue
                if (s->clear_accum() == 0) {
                    for (uint32_t i = 0u; i < TM; ++i) {
                        for (uint32_t j = 0u; j < TN; ++j) {
                            Coord cc = _zero_coord();
                            cc[0] = (Expr<uint>{r} + i).expression();
                            cc[1] = (Expr<uint>{ct0} + j).expression();
                            // DSL: acc[i][j] += cast(c[r+i][c+j], float)  (the
                            // per-lane K-slice partial is already all-reduced)
                            acc[i * TN + j] = Expr<float>{acc[i * TN + j].expression()} +
                                              Expr<float>{_maybe_cast(_value_at(c, cc), wide_t)};
                        }
                    }
                }
                // write-back
                if (frag && host_nw == 1u) {
                    // single-warp block (fragment C not shared-backed): every
                    // lane writes the tile into its OWN replica; no staging, no
                    // barrier in the warp path
                    for (uint32_t i = 0u; i < TM; ++i) {
                        for (uint32_t j = 0u; j < TN; ++j) {
                            Coord cc = _zero_coord();
                            cc[0] = (Expr<uint>{r} + i).expression();
                            cc[1] = (Expr<uint>{ct0} + j).expression();
                            _write_to(c, cc, _maybe_cast(acc[i * TN + j].expression(), out_t));
                        }
                    }
                } else {
                    _if((Expr<uint>{lane} == 0u).expression(), [&] {
                        for (uint32_t i = 0u; i < TM; ++i) {
                            for (uint32_t j = 0u; j < TN; ++j) {
                                Coord cc = _zero_coord();
                                cc[0] = (Expr<uint>{r} + i).expression();
                                cc[1] = (Expr<uint>{ct0} + j).expression();
                                if (frag) {
                                    auto sidx = Expr<uint>{_staging_index(c, cc)};
                                    auto cval = _maybe_cast(acc[i * TN + j].expression(), out_t);
                                    // DSL staging write (output dtype may be half):
                                    //   Var<std::array<U,1>> s{staging}; s[idx] = cast(value, U)
                                    with_elem_type(c->dtype(), [&]<typename U>() {
                                        Var<std::array<U, 1>> s{staging};
                                        s[sidx] = Expr<U>{cval};
                                    });
                                } else {
                                    _write_to(c, cc, _maybe_cast(acc[i * TN + j].expression(), out_t));
                                }
                            }
                        }
                    });
                }
            };
            // staging/barrier setup for the warp path (outside the partition
            // loops): fragment C with a multi-warp block publishes through the
            // shared staging tile and refreshes every replica afterwards
            if (frag && host_nw > 1u) {
                staging = _staging_for(c, out_t);
                _sync_block();// staging write-after-read hazard vs. previous use
            }
            // warp-level 2D partition of the MT x NT micro-tile grid
            _for_range(r0, _literal_u(MT), _literal_u(th), [&](const Expression *r) {
                auto emit_cols = [&] {
                    _for_range(c0, _literal_u(NT), _literal_u(tw), [&](const Expression *c) {
                        warp_micro_tile(r, c);
                    });
                };
                if (MT % th != 0u) {// some warps start at r0 >= MT
                    _if((Expr<uint>{r} < MT).expression(), emit_cols);
                } else {
                    emit_cols();
                }
            });
            if (frag && host_nw > 1u) {
                _replicate_from_staging(c, c->dtype(), staging);
            } else if (frag) {
                // single-warp replica writes above replace any lazy fragment
                // value recorded earlier (the documented invalidation gap)
                _invalidate_lazy(c);
            }
            _current_extent = saved;
            return;
        }
        // compute one TM x TN micro-tile with top-left (r, c) = rt*TM, ct*TN
        // (r/c are runtime expressions; TM/TN/MT/NT are host constants).
        auto micro_tile = [&](const Expression *rt, const Expression *ct) {
            // DSL: r = rt * TM; c0 = ct * TN
            auto r = (Expr<uint>{rt} * TM).expression();
            auto c0 = (Expr<uint>{ct} * TN).expression();
            // TM*TN f32 accumulator locals (max 4x4 = 16; TM/TN are runtime
            // host values, so use the fixed upper bound and index i*TN+j).
            // Var<float> default-init emits the same local + 0.f initializer
            // as the old _fb->local/_fb->assign(0.f) for the clear_accum==1
            // case; for clear_accum==0 the locals are overwritten below before
            // any read.
            std::array<Var<float>, 16> acc{};
            if (s->clear_accum() == 0) {
                for (uint32_t i = 0u; i < TM; ++i) {
                    for (uint32_t j = 0u; j < TN; ++j) {
                        Coord cc = _zero_coord();
                        cc[0] = (Expr<uint>{r} + i).expression();
                        cc[1] = (Expr<uint>{c0} + j).expression();
                        // DSL: acc[i][j] = cast(c[r+i][c+j], float)
                        acc[i * TN + j] = Expr<float>{_maybe_cast(_value_at(c, cc), wide_t)};
                    }
                }
            }
            // K loop with k_pack host unrolling; tail guarded when K % k_pack != 0
            _for_range(_literal_u(0u), _literal_u(k_iters), _literal_u(1u),
                       [&](const Expression *kk) {
                for (uint32_t u = 0u; u < k_pack; ++u) {
                    // DSL: k = kk * k_pack + u
                    auto k = (Expr<uint>{kk} * k_pack + u).expression();
                    auto emit_k = [&] {
                        // TM A loads + TN B loads, then TM*TN FMAs
                        std::array<const Expression *, 16> a_row{};
                        for (uint32_t i = 0u; i < TM; ++i) {
                            Coord ac = _zero_coord();
                            auto ri = (Expr<uint>{r} + i).expression();
                            if (s->trans_a() != 0) {
                                ac[0] = k;
                                ac[1] = ri;
                            } else {
                                ac[0] = ri;
                                ac[1] = k;
                            }
                            // DSL local: Var<float> initialized from the cast A value
                            a_row[i] = Var<float>{Expr<float>{_maybe_cast(_value_at(a, ac), wide_t)}}.expression();
                        }
                        std::array<const Expression *, 16> b_col{};
                        for (uint32_t j = 0u; j < TN; ++j) {
                            Coord bc = _zero_coord();
                            auto cj = (Expr<uint>{c0} + j).expression();
                            if (s->trans_b() != 0) {
                                bc[0] = cj;
                                bc[1] = k;
                            } else {
                                bc[0] = k;
                                bc[1] = cj;
                            }
                            // DSL local: Var<float> initialized from the cast B value
                            b_col[j] = Var<float>{Expr<float>{_maybe_cast(_value_at(b, bc), wide_t)}}.expression();
                        }
                        for (uint32_t i = 0u; i < TM; ++i) {
                            for (uint32_t j = 0u; j < TN; ++j) {
                                // DSL: acc[i][j] = fma(a_row[i], b_col[j], acc[i][j])
                                acc[i * TN + j] = fma(Expr<float>{a_row[i]},
                                                      Expr<float>{b_col[j]},
                                                      Expr<float>{acc[i * TN + j].expression()});
                            }
                        }
                    };
                    if (K % k_pack != 0u) {// k may run past K on the last kk
                        _if((Expr<uint>{k} < K).expression(), emit_k);
                    } else {
                        emit_k();
                    }
                }
            });
            // write-back the micro-tile
            for (uint32_t i = 0u; i < TM; ++i) {
                for (uint32_t j = 0u; j < TN; ++j) {
                    Coord cc = _zero_coord();
                    cc[0] = (Expr<uint>{r} + i).expression();
                    cc[1] = (Expr<uint>{c0} + j).expression();
                    if (frag) {
                        auto sidx = Expr<uint>{_staging_index(c, cc)};
                        auto cval = _maybe_cast(acc[i * TN + j].expression(), out_t);
                        // DSL staging write (output dtype may be half):
                        //   Var<std::array<U,1>> s{staging}; s[idx] = cast(value, U)
                        with_elem_type(c->dtype(), [&]<typename U>() {
                            Var<std::array<U, 1>> s{staging};
                            s[sidx] = Expr<U>{cval};
                        });
                    } else {
                        _write_to(c, cc, acc[i * TN + j].expression());
                    }
                }
            }
        };
        // Thread -> micro-tile mapping honors GemmWarpPolicy (plan 2.4):
        //   Square  -> 2D strided partition over (MT, NT)
        //   FullRow -> rows split across threads, each thread walks NT cols
        //   FullCol -> cols split across threads, each thread walks MT rows
        auto gemm_partition = [&](auto &&body) {
            switch (s->policy()) {
                case GemmWarpPolicy::FullRow: {
                    _for_range(_tid_x(), _literal_u(MT), _literal_u(_threads),
                               [&](const Expression *rt) {
                        for (uint32_t ct = 0u; ct < NT; ++ct) {
                            body(rt, _literal_u(ct));
                        }
                    });
                    break;
                }
                case GemmWarpPolicy::FullCol: {
                    _for_range(_tid_x(), _literal_u(NT), _literal_u(_threads),
                               [&](const Expression *ct) {
                        for (uint32_t rt = 0u; rt < MT; ++rt) {
                            body(_literal_u(rt), ct);
                        }
                    });
                    break;
                }
                default: {// Square
                    _partition_2d(MT, NT, body);
                    break;
                }
            }
        };
        if (frag) {
            // fragment C is replicated: publish the partitioned results through
            // a shared staging tile, then refresh every thread's replica
            staging = _staging_for(c, out_t);
            _sync_block();// staging write-after-read hazard vs. previous use
            gemm_partition([&](const Expression *rt, const Expression *ct) {
                micro_tile(rt, ct);
            });
            _replicate_from_staging(c, c->dtype(), staging);// ISOLATE
        } else {
            // global C, or a large shared-backed fragment C: each element is
            // written exactly once across the block (the A/B tiles were staged
            // in shared memory and published by the copies' trailing barriers;
            // this statement's own writes are published by _accesses_shared's
            // trailing _sync_block() in _emit).
            gemm_partition([&](const Expression *rt, const Expression *ct) {
                micro_tile(rt, ct);
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
    /*
     * _emit_gemm_cooperative(s) pseudo-code (luisa-dsl):
     *
     *   assert rank-2, F16/F32, non-transposed, shared/fragment operands only
     *   M, N, K = C rows, C cols, A inner dim
     *   $for (r, 0u, M) {                  // uniform row loop (block-wide)
     *       CoopVector<float> acc{N};      // cooperative_vector<float, N>
     *       if not clear_accum:
     *           for i = 0 .. N-1:          // host-unrolled load of C row
     *               acc[i] = cast(c[r][i], float)
     *       $for (k, 0u, K) {
     *           Float a_scalar = cast(a[r][k], float);
     *           a_vec = cooperative_vector_splat<float>(a_scalar, N);
     *           for i = 0 .. N-1:          // host-unrolled B row -> b_vec
     *               b_vec[i] = b[k][i]
     *           for i = 0 .. N-1:          // host-unrolled FMA expansion
     *               acc[i] = fma(a_vec[i], cast(b_vec[i], float), acc[i])
     *       };
     *       for i = 0 .. N-1:              // host-unrolled store of C row
     *           _write_to(c, (r,i), cast(acc[i], c_dtype))
     *   };
     */
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
        // NOTE: the cooperative-vector body below deliberately stays on the
        // dtype-erased raw FunctionBuilder path (_fb->local/assign/call +
        // _fb->access for the per-component loads/stores).  The DSL wrapper
        // (CoopVector<float> / CoopVector<T>) requires the element type at
        // compile time, while the cooperative GEMM supports F16/F32 inputs at
        // runtime, and the mixed float-accumulator/half-operand staging is not
        // worth the with_elem_type plumbing for a path that is NOT exercised by
        // the default tests.  The raw calls are correct (Type::cooperative_vector
        // + COOPERATIVE_VECTOR_SPLAT + per-component FMA expansion).  Only the
        // index math and control flow use the DSL helpers (_for_range /
        // _literal_u / _zero_coord).
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

    /*
     * _emit_print(s) pseudo-code (luisa-dsl):
     *
     *   Bool cond = (thread_id().x == 0);
     *   if batching:
     *       cond = cond && batch_valid;
     *   $if (cond) {
     *       v = _value_at(t, origin);
     *       print_("[tile] msg tile[0] = {}", v);
     *   };
     */
    void _emit_print(const TilePrintStmt *s) {
        auto *t = s->t();
        auto saved = _current_extent;
        _current_extent = t;
        auto c0 = _zero_coord();
        // DSL: Bool cond = (thread_id().x == 0);
        auto cond = (Expr<uint3>{_fb->thread_id()}.x == 0u).expression();
        if (_batching) {
            // Skip printing from idle tz threads of the tail z-block.
            // (Scalar bools use the bitwise & -- the DSL has no &&/|| for
            // scalars; AND and BIT_AND coincide on {0,1}.)
            cond = (Expr<bool>{cond} & Expr<bool>{_batch_valid}).expression();
        }
        _if(cond, [&] {
            auto v = _value_at(t, c0);
            auto fmt = luisa::format("[tile] {} tile[0] = {{}}", luisa::string{s->msg()});
            // no DSL equivalent for print_: keep the raw call
            _fb->print_(fmt, luisa::span<const Expression *const>{&v, 1u});
        });
        _current_extent = saved;
    }

    /*
     * _emit_fill(s) pseudo-code (luisa-dsl):
     *
     *   buf = s->buf()
     *   value = value_literal (error on R3 ref)
     *   if buf is Fragment and not shared-backed:
     *       _full_loop(buf, c => buf[c] = value)
     *   else:
     *       _partition_loop(buf, c => buf[c] = value)
     */
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
          _invalidate_lazy(buf);
      } else if (buf->rank() == 2u) {
          _partition_loop_2d(buf, [&](const Coord &c) { _write_to(buf, c, value); });
      } else {
          _partition_loop(buf, [&](const Coord &c) { _write_to(buf, c, value); });
      }
        _current_extent = saved;
    }

    /*
     * _emit_transpose(s) pseudo-code (luisa-dsl):
     *
     *   ext = operand with known extent
     *   body(cd):
     *       cs = (cd[1], cd[0])   // swap row/col
     *       dst[cd] = src[cs]
     *   if dst is Fragment and not shared-backed:
     *       _full_loop(ext, body)
     *   else if src or dst is Global and tile is not tiny:
     *       // shared-staged tiled transpose: coalesced read -> shared ->
     *       // coalesced write, instead of strided global read/write
     *       staging = Shared<T> staging{product(extent)}
     *       _partition_loop(ext, c => staging[idx(c)] = src[c])  // coalesced read
     *       sync_block()
     *       _partition_loop(ext, c => dst[transpose(c)] = staging[idx(transpose(c))])
     *   else:
     *       rank==2 ? _partition_loop_2d(ext, body) : _partition_loop(ext, body)
     */
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
            _invalidate_lazy(dst);
        } else if ((src->scope() == TensorScope::Global || dst->scope() == TensorScope::Global) &&
                   tile_element_count(ext) >= 64u &&
                   _pipeline_var == nullptr && !_batching) {
            // Staged tiled transpose (plan 2.14): coalesced read of the src
            // tile into a shared staging tile, sync, then coalesced write of
            // the transposed tile to dst.  This replaces the strided global
            // read/write pattern for non-tiny Global operands.  The internal
            // sync covers the staging hazard; cross-statement hazards are
            // covered by _emit's trailing barrier logic via the operand scopes.
            // DSL: Shared<T> staging{n} (element type is the runtime tag, so
            // the whole staged block is dtype-generic inside with_elem_type).
            with_elem_type(ext->dtype(), [&]<typename T>() {
                auto n = tile_element_count(ext);
                Shared<T> staging{n};
                // pass 1: coalesced read of src into staging
                auto pass1 = [&](const Coord &c) {
                    auto idx = Expr<uint>{_local_index(ext, c)};
                    staging[idx] = Expr<T>{_maybe_cast(_value_at(src, c), Type::of<T>())};
                };
                if (ext->rank() == 2u) { _partition_loop_2d(ext, pass1); } else { _partition_loop(ext, pass1); }
                _sync_block();
                // pass 2: coalesced write of dst from the transposed staging index
                auto pass2 = [&](const Coord &cd) {
                    Coord cs = _zero_coord();
                    cs[0] = cd[1];
                    cs[1] = cd[0];
                    _write_to(dst, cd, staging[Expr<uint>{_local_index(ext, cs)}].expression());
                };
                if (ext->rank() == 2u) { _partition_loop_2d(ext, pass2); } else { _partition_loop(ext, pass2); }
            });
        } else if (ext->rank() == 2u) {
            _partition_loop_2d(ext, body);
        } else {
            _partition_loop(ext, body);
        }
        _current_extent = saved;
    }

    /*
     * _emit_clamp(s) pseudo-code (luisa-dsl):
     *
     *   dst = s->dst()
     *   body(c):
     *       v = dst[c]
     *       if lo_literal exists: v = max(v, cast(lo, elem_t))
     *       if hi_literal exists: v = min(v, cast(hi, elem_t))
     *       dst[c] = v
     *   if dst is Fragment and not shared-backed:
     *       _full_loop(dst, body)
     *   else:
     *       rank==2 ? _partition_loop_2d(dst, body) : _partition_loop(dst, body)
     */
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
            // DSL: max/min on the concrete element type
            if (lo != nullptr) {
                clamped = with_elem_type(dst->dtype(), [&]<typename T>() -> const Expression * {
                    return max(Expr<T>{clamped}, Expr<T>{lo}).expression();
                });
            }
            if (hi != nullptr) {
                clamped = with_elem_type(dst->dtype(), [&]<typename T>() -> const Expression * {
                    return min(Expr<T>{clamped}, Expr<T>{hi}).expression();
                });
            }
            _write_to(dst, c, clamped);
        };
        if (dst->scope() == TensorScope::Fragment && !_is_fragment_shared_backed(dst)) {
            _full_loop(dst, body);
            _invalidate_lazy(dst);
        } else if (dst->rank() == 2u) {
            _partition_loop_2d(dst, body);
        } else {
            _partition_loop(dst, body);
        }
        _current_extent = saved;
    }

    /*
     * _emit_atomic(s) pseudo-code (luisa-dsl):
     *
     *   dst = s->dst()
     *   value = value_tensor[origin] / value_literal (error on R3 ref)
     *   value = cast(value, elem_t) if present
     *   rank==2 ? _partition_loop_2d(dst, body) : _partition_loop(dst, body):
     *       UInt idx = global_index(dst, c)
     *       if batching: guard body with _batch_valid
     *       switch op:
     *           ADD  -> tmp = buf.atomic(idx).fetch_add(value)
     *           MAX  -> tmp = buf.atomic(idx).fetch_max(value)
     *           MIN  -> tmp = buf.atomic(idx).fetch_min(value)
     *           OR   -> tmp = buf.atomic(idx).fetch_or(value)
     *           LOAD -> tmp = buf.volatile_read(idx)
     *           STORE-> buf.volatile_write(idx, value)
     *           default -> error
     *   (per-thread aggregation / packed addx2/addx4 remain documented gaps.)
     */
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
            auto emit_body = [&] {
                auto idx = _global_index(dst, c);
                // NOTE: the DSL atomic surface (Expr<Buffer<T>>::atomic(i).
                // fetch_add/max/min/or) is only defined for int/uint/slong/
                // ulong/float, while the tile atomics also support F16/I8 with
                // a runtime dtype tag (and fp8 goes through the raw byte path
                // in _value_at).  The dtype-erased raw call path below is kept
                // intact because it is correct for every supported element
                // type; wrapping it in with_elem_type would fail to compile
                // for half/byte (no AtomicRef specialization) and for fp8.
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
            // All atomics target Global storage; guard them with the batch
            // validity predicate so idle tz threads never touch batch 0.
            if (_batching) { _if(_batch_valid, emit_body); } else { emit_body(); }
        };
        if (dst->rank() == 2u) {
            _partition_loop_2d(dst, body);
        } else {
            _partition_loop(dst, body);
        }
        _current_extent = saved;
    }

    /*
     * _emit_sync(s) pseudo-code:
     *
     *   switch s->op():
     *       THREADS -> sync_block()
     *       WARP    -> no-op
     *       default -> error (grid/global sync unsupported)
     */
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
    /*
     * _emit_reduce(s) pseudo-code:
     *
     *   _emit_tile_reduce(s->buf(), s->out(), s->dim(), s->op())
     */
    void _emit_reduce(const ReduceStmt *s) {
        _emit_tile_reduce(s->buf(), s->out(), s->dim(), s->op());
    }

    // inclusive prefix scan along `dim`: T.cumsum / T.cummax (src, dst, dim, reverse)
    // Warp-collective scan (lc_optimize 3.4): each warp owns whole scan lines;
    // lanes load one element per chunk, WARP_PREFIX_SUM (or a WARP_READ_LANE
    // butterfly for max) produces the in-chunk inclusive scan and a running
    // carry stitches the chunks together.  Replaces the previous O(n^2)
    // per-element re-accumulation.
    /*
     * _emit_scan(src, dst, dim, reverse, is_max) pseudo-code (luisa-dsl):
     *
     *   scan_len = extent of src along dim
     *   line_count = product of src's extents except dim
     *   UInt lane = warp_lane_id(); UInt lanes = warp_lane_count()
     *   UInt warp = slice-local warp id; UInt nw = warps per slice
     *   UInt chunks = ceildiv(scan_len, lanes)
     *   UInt line_iters = ceildiv(line_count, nw)
     *   if dst is Fragment:
     *       staging = _staging_for(dst); sync_block()
     *   $for (li, 0u, line_iters) {
     *       UInt line = li * nw + warp;
     *       $if (line < line_count) {
     *           // decompose line -> coords for all axes except dim
     *           carry = identity(op);
     *           $for (ch, 0u, chunks) {
     *               UInt off = ch * lanes + lane;
     *               UInt pos = reverse ? (scan_len - 1u - off) : off;
     *               v = identity(op);
     *               $if (off < scan_len) {
     *                   cc[dim] = pos;
     *                   v = cast(src[cc], elem_t);
     *               };
     *               // inclusive scan within warp via warp_read_lane butterfly
     *               incl = v;
     *               for d = 1,2,4,... while d < lanes:
     *                   UInt peer = lane - min(lane, d);
     *                   other = warp_read_lane(incl, peer);
     *                   $if (lane >= d) { incl = combine(op, incl, other); };
     *               total = warp_read_lane(incl, lanes - 1u);
     *               res = combine(op, carry, incl);
     *               $if (off < scan_len) {
     *                   $if (dst is Fragment) { staging[_staging_index(dst, cc)] = cast(res, out_t); };
     *                   $else { _write_to(dst, cc, res); };
     *               };
     *               carry = combine(op, carry, total);
     *           };
     *       };
     *   };
     *   if dst is Fragment:
     *       _replicate_from_staging(dst, staging)
     *
     *   // ---- two-pass block scan (plan 2.8 / lc_optimize 4.5) -----------
     *   // Active when !batching && line_count < nw_est && scan_len > lanes:
     *   // fewer scan lines than warps -> the warp-per-line partition leaves
     *   // most warps idle; split each line's scan axis across ALL warps.
     *   if use_block_scan:
     *       seg_count = min(nw_est, scan_len)
     *       seg_len = ceildiv(scan_len, seg_count)   // segment max length
     *       totals_s = Shared<T>{seg_count}; prefix_s = Shared<T>{seg_count}
     *       scan_segment(line, seg_start, seg_len_w, base):  // helper
     *           // same butterfly inclusive scan as above, but over the
     *           // segment's off range [seg_start, seg_start + seg_len_w)
     *           carry = base
     *           $for (ch, 0u, ceildiv(seg_len_w, lanes)) {
     *               off = seg_start + ch * lanes + lane;
     *               pos = reverse ? (scan_len - 1u - off) : off;
     *               ... butterfly incl scan (v loaded from src at pos) ...
     *               total = warp_read_lane(incl, lanes - 1u);
     *               res = combine(carry, incl);
     *               $if (off < seg_start + seg_len_w) {
     *                   // write dst/staging at pos (fragment -> staging)
     *               };
     *               carry = combine(carry, total);
     *           }
     *           return carry   // segment total
     *       $for (line, 0u, line_count) {          // few lines, sequential
     *           // pass 1: each warp scans its segment, publishes the total
     *           seg_w = min(seg_len, scan_len - warp * seg_len)
     *           total = scan_segment(line, warp*seg_len, seg_w, identity)
     *           $if (lane == 0u) { totals_s[warp] = total; }
     *           sync_block()
     *           // pass 2: warp 0 inclusive-scans the seg_count totals
     *           $if (warp == 0u) {
     *               v = lane < seg_count ? totals_s[lane] : identity
     *               incl = butterfly_scan(v)          // inclusive over totals
     *               prev = lane == 0u ? identity : warp_read_lane(incl, lane - 1u)
     *               $if (lane < seg_count) { prefix_s[lane] = prev; }
     *           }
     *           sync_block()
     *           // pass 3: each warp recomputes its segment scan + prefix
     *           seg_w = min(seg_len, scan_len - warp * seg_len)
     *           scan_segment(line, warp*seg_len, seg_w, prefix_s[warp])
     *       }
     *       if dst is Fragment:
     *           _replicate_from_staging(dst, staging)
     *       return
     */
    void _emit_scan(const TensorExpr *src, const TensorExpr *dst,
                    uint32_t dim, int32_t reverse, bool is_max) {
        auto saved = _current_extent;
        _current_extent = src;
        auto scan_len = static_cast<uint32_t>(axis_extent(src, dim));
        uint32_t line_count = 1u;
        for (auto i = 0u; i < src->rank(); ++i) {
            if (i != dim) { line_count *= static_cast<uint32_t>(axis_extent(src, i)); }
        }
        auto lanes = _lane_count();
        auto lane = _lane();
        // Slice-local warp partition: each batch item's scan lines are covered
        // by its own slice's warps (see _num_warps / _slice_warp).
        auto warp = _slice_warp();
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
        const auto op = is_max ? TileReduceOp::MAX : TileReduceOp::SUM;
        // ---- two-pass block scan (plan 2.8 / lc_optimize 4.5) -----------------
        // Fewer scan lines than warps -> the warp-per-line partition leaves most
        // warps idle; split each line's scan axis across ALL warps instead.
        const uint32_t nw_est = _threads / 32u;// host, pinned warp size 32
        const bool use_block_scan = !_batching && line_count < nw_est && scan_len > 32u;
        if (use_block_scan) {
            const uint32_t seg_count = std::min<uint32_t>(nw_est, scan_len);
            const uint32_t seg_len = (scan_len + seg_count - 1u) / seg_count;
            const uint32_t seg_chunks = (seg_len + 32u - 1u) / 32u;// lanes pinned to 32
            // The element arithmetic is dtype-generic (runtime tag), so the
            // device body is written in the DSL sugar inside with_elem_type.
            with_elem_type(src->dtype(), [&]<typename T>() {
                // DSL shared workspaces (element type T)
                Shared<T> totals_s{seg_count};
                Shared<T> prefix_s{seg_count};
                for (uint32_t line = 0u; line < line_count; ++line) {
                    // decompose host-constant line -> coords for all axes except dim
                    Coord cc = _zero_coord();
                    auto rem = line;
                    for (int32_t i = static_cast<int32_t>(src->rank()) - 1; i >= 0; --i) {
                        auto ui = static_cast<uint32_t>(i);
                        if (ui == dim) { continue; }
                        auto e = static_cast<uint32_t>(axis_extent(src, ui));
                        auto ci = rem % e;
                        rem /= e;
                        cc[ui] = _literal_u(ci);
                    }
                    // per-line carry: pass 1 leaves the segment total here for the
                    // lane-0 publish; pass 3 accumulates its scan with the prefix
                    auto carry = Var<T>{Expr<T>{identity()}};
                    // ONE local lambda for both passes: butterfly inclusive scan of
                    // the segment's off range [seg_start, seg_start + seg_w) with an
                    // initial `base` carry; writes output only when emit_writes.
                    auto scan_segment = [&](const Coord &cc0,
                                            const Expression *seg_start,
                                            uint32_t seg_w,
                                            const Expression *base,
                                            bool emit_writes) -> void {
                        // Segment bounds: the scan must stay inside
                        // [seg_start, seg_start + seg_w) (clamped to scan_len).
                        // Guarding with off < scan_len alone would let a ragged
                        // tail chunk (seg_w % lanes != 0) read — and in pass 3
                        // WRITE — elements owned by the NEXT segment, racing the
                        // neighbour warp's correct values and contaminating this
                        // segment's published total (lc_optimize: correctness of
                        // partitioned scans).  With evenly-dividing extents the
                        // extra min() folds away and the guards are always true.
                        auto seg_end = (min(Expr<uint>{scan_len},
                                            Expr<uint>{seg_start} + Expr<uint>{seg_w})).expression();
                        // Host-provable guard elision: when seg_len is a
                        // multiple of the (pinned 32) warp size and divides
                        // scan_len evenly, every chunk of every segment is in
                        // bounds, so the identity-padded guard is dead code.
                        const bool seg_guard_free = (seg_len % 32u == 0u) && (scan_len % seg_len == 0u);
                        Coord scc = cc0;
                        carry = Expr<T>{base};
                        _for_range(_literal_u(0u), _literal_u(seg_chunks), _literal_u(1u),
                                   [&](const Expression *ch) {
                            // DSL: off = seg_start + ch * lanes + lane
                            auto off = (Expr<uint>{seg_start} +
                                        Expr<uint>{ch} * Expr<uint>{lanes} +
                                        Expr<uint>{lane}).expression();
                            // element position along the scan axis (from the scan side)
                            const Expression *pos = off;
                            if (reverse != 0) {
                                pos = (Expr<uint>{scan_len - 1u} - Expr<uint>{off}).expression();
                            }
                            auto valid = (Expr<uint>{off} < Expr<uint>{seg_end}).expression();
                            auto v = Var<T>{Expr<T>{identity()}};
                            if (seg_guard_free) [[likely]] {
                                scc[dim] = pos;
                                v = Expr<T>{_maybe_cast(_value_at(src, scc), Type::of<T>())};
                            } else {
                                _if(valid, [&] {
                                    scc[dim] = pos;
                                    v = Expr<T>{_maybe_cast(_value_at(src, scc), Type::of<T>())};
                                });
                            }
                            // in-chunk inclusive scan across the warp: butterfly
                            // inclusive scan via WARP_READ_LANE (lc_optimize 2.2;
                            // the lane read is unconditional/clamped so it is never
                            // divergent)
                            auto incl = Var<T>{Expr<T>{v.expression()}};
                            for (uint32_t d = 1u; d <= 64u; d <<= 1u) {
                                auto d_active = (Expr<uint>{d} < Expr<uint>{lanes}).expression();
                                _if(d_active, [&] {
                                    auto clamped = min(Expr<uint>{lane}, Expr<uint>{d});
                                    auto peer = (Expr<uint>{lane} - clamped).expression();
                                    // stage the unconditional wave read in a local
                                    auto other = Var<T>{warp_read_lane(incl, Expr<uint>{peer})};
                                    auto has_prev = (Expr<uint>{lane} >= Expr<uint>{d}).expression();
                                    _if(has_prev, [&] {
                                        incl = Expr<T>{_combine_expr<T>(op, incl.expression(),
                                                                        other.expression())};
                                    });
                                });
                            }
                            // chunk total = the last lane's inclusive value
                            auto last = (Expr<uint>{lanes} - 1u).expression();
                            auto total = warp_read_lane(incl, Expr<uint>{last}).expression();
                            const Expression *res = _combine_expr<T>(op, carry.expression(),
                                                                     incl.expression());
                            if (emit_writes) {
                                if (seg_guard_free) [[likely]] {
                                    scc[dim] = pos;
                                    if (frag_out) {
                                        auto sidx = Expr<uint>{_staging_index(dst, scc)};
                                        auto res_cast = _maybe_cast(res, out_t);
                                        // DSL staging write (dst dtype may differ):
                                        //   Var<std::array<U,1>> s{staging}; s[idx] = cast(res, U)
                                        with_elem_type(dst->dtype(), [&]<typename U>() {
                                            Var<std::array<U, 1>> s{staging};
                                            s[sidx] = Expr<U>{res_cast};
                                        });
                                    } else {
                                        _write_to(dst, scc, res);
                                    }
                                } else {
                                    _if(valid, [&] {
                                        scc[dim] = pos;
                                        if (frag_out) {
                                            auto sidx = Expr<uint>{_staging_index(dst, scc)};
                                            auto res_cast = _maybe_cast(res, out_t);
                                            // DSL staging write (dst dtype may differ):
                                            //   Var<std::array<U,1>> s{staging}; s[idx] = cast(res, U)
                                            with_elem_type(dst->dtype(), [&]<typename U>() {
                                                Var<std::array<U, 1>> s{staging};
                                                s[sidx] = Expr<U>{res_cast};
                                            });
                                        } else {
                                            _write_to(dst, scc, res);
                                        }
                                    });
                                }
                            }
                            carry = Expr<T>{_combine_expr<T>(op, carry.expression(), total)};
                        });
                    };
                    auto seg_valid = (Expr<uint>{warp} < seg_count).expression();
                    _if(seg_valid, [&] {
                        // pass 1: scan segment, publish the segment total
                        auto seg_start = (Expr<uint>{warp} * seg_len).expression();
                        carry = Expr<T>{identity()};
                        scan_segment(cc, seg_start, seg_len, identity(), false);
                        _if((Expr<uint>{lane} == 0u).expression(), [&] {
                            totals_s[Expr<uint>{warp}] = Expr<T>{carry.expression()};
                        });
                    });
                    _sync_block();// publish totals before warp 0 scans them
                    // pass 2: warp 0 inclusive-scans the seg_count totals -> per-
                    // segment exclusive prefixes
                    _if((Expr<uint>{warp} == 0u).expression(), [&] {
                        auto v = Var<T>{Expr<T>{identity()}};
                        auto lane_valid = (Expr<uint>{lane} < seg_count).expression();
                        _if(lane_valid, [&] {
                            v = Expr<T>{totals_s[Expr<uint>{lane}].expression()};
                        });
                        auto incl = Var<T>{Expr<T>{v.expression()}};
                        // butterfly inclusive scan of totals (same loop as above)
                        for (uint32_t d = 1u; d <= 64u; d <<= 1u) {
                            auto d_active = (Expr<uint>{d} < Expr<uint>{lanes}).expression();
                            _if(d_active, [&] {
                                auto clamped = min(Expr<uint>{lane}, Expr<uint>{d});
                                auto peer = (Expr<uint>{lane} - clamped).expression();
                                auto other = Var<T>{warp_read_lane(incl, Expr<uint>{peer})};
                                auto has_prev = (Expr<uint>{lane} >= Expr<uint>{d}).expression();
                                _if(has_prev, [&] {
                                    incl = Expr<T>{_combine_expr<T>(op, incl.expression(),
                                                                    other.expression())};
                                });
                            });
                        }
                        // exclusive prefix = predecessor lane's inclusive total
                        auto pred = (Expr<uint>{lane} -
                                     min(Expr<uint>{lane}, Expr<uint>{1u})).expression();
                        auto prev = Var<T>{warp_read_lane(incl, Expr<uint>{pred})};
                        _if((Expr<uint>{lane} == 0u).expression(), [&] {
                            prev = Expr<T>{identity()};
                        });
                        _if(lane_valid, [&] {
                            prefix_s[Expr<uint>{lane}] = Expr<T>{prev.expression()};
                        });
                    });
                    _sync_block();// publish prefixes before pass 3 reads them
                    _if(seg_valid, [&] {
                        // pass 3: recompute the segment scan with the exclusive prefix
                        auto seg_start = (Expr<uint>{warp} * seg_len).expression();
                        auto base = Var<T>{Expr<T>{prefix_s[Expr<uint>{warp}].expression()}};
                        scan_segment(cc, seg_start, seg_len, base.expression(), true);
                    });
                }
            });
            if (frag_out) { _replicate_from_staging(dst, dst->dtype(), staging); }
            _current_extent = saved;
            return;
        }
        // ---- normal warp-per-line scan (existing path) ----
        with_elem_type(src->dtype(), [&]<typename T>() {
            _for_range(_literal_u(0u), line_iters, _literal_u(1u),
                       [&](const Expression *li) {
                // DSL: line = li * nw + warp
                auto line = (Expr<uint>{li} * Expr<uint>{nw} + Expr<uint>{warp}).expression();
                auto line_valid = (Expr<uint>{line} < line_count).expression();
                _if(line_valid, [&] {
                    // decompose the line index over src's shape minus the scan axis
                    Coord cc = _zero_coord();
                    auto rem = Var<uint>{Expr<uint>{line}};
                    for (int32_t i = static_cast<int32_t>(src->rank()) - 1; i >= 0; --i) {
                        auto ui = static_cast<uint32_t>(i);
                        if (ui == dim) { continue; }
                        auto e = Expr<uint>{static_cast<uint32_t>(axis_extent(src, ui))};
                        auto ci = (rem % e).expression();
                        rem = rem / e;
                        cc[ui] = ci;
                    }
                    auto carry = Var<T>{Expr<T>{identity()}};
                    // Full/tail chunk split (lc_optimize: guard elision): the
                    // off < scan_len guard is only false in the last chunk, so
                    // full chunks load/scan/store unconditionally; the tail
                    // chunk keeps the identity-padded guards.  When scan_len is
                    // a host-known multiple of every possible power-of-two warp
                    // size (<= 128), the tail is provably empty and not emitted.
                    auto full_ch = (Expr<uint>{scan_len} / Expr<uint>{lanes}).expression();
                    const bool tail_free = scan_len % 128u == 0u;
                    auto chunk_body = [&](const Expression *ch, bool guarded) {
                        // DSL: off = ch * lanes + lane
                        auto off = (Expr<uint>{ch} * Expr<uint>{lanes} + Expr<uint>{lane}).expression();
                        // element position along the scan axis (from the scan side)
                        const Expression *pos = off;
                        if (reverse != 0) {
                            pos = (Expr<uint>{scan_len - 1u} - Expr<uint>{off}).expression();
                        }
                        auto valid = (Expr<uint>{off} < scan_len).expression();
                        auto v = Var<T>{Expr<T>{identity()}};
                        if (guarded) {
                            _if(valid, [&] {
                                cc[dim] = pos;
                                v = Expr<T>{_maybe_cast(_value_at(src, cc), Type::of<T>())};
                            });
                        } else {
                            cc[dim] = pos;
                            v = Expr<T>{_maybe_cast(_value_at(src, cc), Type::of<T>())};
                        }
                        // in-chunk inclusive scan across the warp: butterfly
                        // inclusive scan via WARP_READ_LANE (lc_optimize 2.2; the
                        // lane read is unconditional/clamped so it is never
                        // divergent — the built-in WARP_PREFIX_SUM miscompiles in
                        // this nested control flow on some backends)
                        auto incl = Var<T>{Expr<T>{v.expression()}};
                        for (uint32_t d = 1u; d <= 64u; d <<= 1u) {
                            auto d_active = (Expr<uint>{d} < Expr<uint>{lanes}).expression();
                            _if(d_active, [&] {
                                auto clamped = min(Expr<uint>{lane}, Expr<uint>{d});
                                auto peer = (Expr<uint>{lane} - clamped).expression();
                                // the wave read must stay UNCONDITIONAL (a
                                // divergent wave intrinsic is UB): stage it in a
                                // local and guard only the combine step
                                auto other = Var<T>{warp_read_lane(incl, Expr<uint>{peer})};
                                auto has_prev = (Expr<uint>{lane} >= Expr<uint>{d}).expression();
                                _if(has_prev, [&] {
                                    incl = Expr<T>{_combine_expr<T>(op, incl.expression(),
                                                                    other.expression())};
                                });
                            });
                        }
                        // chunk total = the last lane's inclusive value
                        auto last = (Expr<uint>{lanes} - 1u).expression();
                        auto total = warp_read_lane(incl, Expr<uint>{last}).expression();
                        const Expression *res = _combine_expr<T>(op, carry.expression(),
                                                                 incl.expression());
                        auto emit_write = [&] {
                            cc[dim] = pos;
                            if (frag_out) {
                                auto sidx = Expr<uint>{_staging_index(dst, cc)};
                                auto res_cast = _maybe_cast(res, out_t);
                                with_elem_type(dst->dtype(), [&]<typename U>() {
                                    Var<std::array<U, 1>> s{staging};
                                    s[sidx] = Expr<U>{res_cast};
                                });
                            } else {
                                _write_to(dst, cc, res);
                            }
                        };
                        if (guarded) {
                            _if(valid, emit_write);
                        } else {
                            emit_write();
                        }
                        carry = Expr<T>{_combine_expr<T>(op, carry.expression(), total)};
                    };
                    _for_range(_literal_u(0u), full_ch, _literal_u(1u),
                               [&](const Expression *ch) { chunk_body(ch, false); });
                    if (!tail_free) [[likely]] {
                        _if((Expr<uint>{full_ch} < Expr<uint>{chunks}).expression(), [&] {
                            chunk_body(full_ch, true);
                        });
                    }
                });
            });
        });
        if (frag_out) { _replicate_from_staging(dst, dst->dtype(), staging); }
        _current_extent = saved;
    }

    // logical tile reduction: T.any_of / T.all_of(buf); the scalar result has
    // no consumer in the tile IR, so it is folded into a throw-away local
    // (the same pattern as WARP_REDUCE).
    /*
     * _emit_any_all(buf, is_all) pseudo-code (luisa-dsl):
     *
     *   if buf is Fragment (replicated):        // each thread owns the whole tile
     *       Bool acc = is_all ? true : false;
     *       _full_loop(buf, c):
     *           Bool truth = (buf[c] != 0);
     *           acc = is_all ? (acc && truth) : (acc || truth);
     *       Bool voted = is_all ? warp_active_all(acc) : warp_active_any(acc);
     *   else:                                   // Global / Shared: partition once
     *       Bool acc = is_all ? true : false;
     *       _partition_loop(buf, c):            // one element per thread
     *           Bool truth = (buf[c] != 0);
     *           acc = is_all ? (acc && truth) : (acc || truth);
     *       // two-level block reduction (warp -> Shared -> block)
     *       warp_acc = is_all ? warp_active_all(acc) : warp_active_any(acc);
     *       $if (lane == 0) { workspace[warp] = warp_acc; };  sync_block();
     *       $if (warp == 0) { block = identity; for w: combine; voted = warp vote; };
     *   tmp = voted;   // keep the call alive
     */
    void _emit_any_all(const TensorExpr *buf, bool is_all) {
        auto elem_t = tensor_element_type(buf->dtype());
        auto saved = _current_extent;
        _current_extent = buf;
        // DSL: Bool acc = is_all ? true : false;
        auto acc = Var<bool>{Expr<bool>{is_all}};
        auto truth_at = [&](const Coord &c) {
            auto v = _value_at(buf, c);
            const Expression *truth = nullptr;
            if (buf->dtype() == TensorElementType::FP8) {
                // fp8 has no C++ scalar type: keep the dtype-erased raw compare
                truth = _fb->binary(Type::of<bool>(), BinaryOp::NOT_EQUAL,
                                    v, _maybe_cast(_zero_of(buf->dtype()), elem_t));
            } else {
                // DSL: truth = (buf[c] != 0)
                truth = with_elem_type(buf->dtype(), [&]<typename T>() -> const Expression * {
                    auto zero = _maybe_cast(_zero_of(buf->dtype()), Type::of<T>());
                    return (Expr<T>{v} != Expr<T>{zero}).expression();
                });
            }
            // DSL: acc = is_all ? (acc & truth) : (acc | truth)
            // (scalar bools use the bitwise ops; AND/OR coincide on {0,1})
            auto a = Expr<bool>{acc.expression()};
            acc = is_all ? (a & Expr<bool>{truth}) : (a | Expr<bool>{truth});
        };
        if (buf->scope() == TensorScope::Fragment || _batching) {
            // Replicated Fragment tiles keep the per-thread _full_loop (each
            // thread already owns the whole tile), and batched kernels keep
            // this path for correctness (plan 2.12).
            _full_loop(buf, truth_at);
            // block-level vote keeps the folded value alive on every lane
            auto vote = [&] {
                auto a = Expr<bool>{acc.expression()};
                return is_all ? warp_active_all(a) : warp_active_any(a);
            };
            auto tmp = Var<bool>{vote()};
        } else {
            // Global / Shared tiles: partition the tile across threads (one
            // element per thread), then combine the per-thread partials with a
            // two-level block reduction (warp collective -> Shared -> block),
            // so the buffer is read once per element (plan 2.12).
            if (buf->rank() == 2u) { _partition_loop_2d(buf, truth_at); } else { _partition_loop(buf, truth_at); }
            auto vote = [&] {
                auto a = Expr<bool>{acc.expression()};
                return is_all ? warp_active_all(a) : warp_active_any(a);
            };
            auto warp_acc = vote();
            auto nw_est = std::max(1u, _threads / 32u);// host, upper bound for warps
            // DSL: Shared<bool> workspace{nw_est}
            Shared<bool> workspace{nw_est};
            auto lane = _lane();
            auto warp = _slice_warp();
            _if((Expr<uint>{lane} == 0u).expression(), [&] {
                workspace[Expr<uint>{warp}] = Expr<bool>{warp_acc.expression()};
            });
            _sync_block();// publish workspace before warp 0 reads it
            _if((Expr<uint>{warp} == 0u).expression(), [&] {
                auto val = Var<bool>{Expr<bool>{is_all}};
                auto lane_valid = (Expr<uint>{lane} < Expr<uint>{_num_warps()}).expression();
                _if(lane_valid, [&] {
                    val = Expr<bool>{workspace[Expr<uint>{lane}].expression()};
                });
                auto a = Expr<bool>{val.expression()};
                auto voted = is_all ? warp_active_all(a) : warp_active_any(a);
                auto tmp = Var<bool>{voted};
            });
        }
        _current_extent = saved;
    }

    // warp shuffle of a fragment scalar: T.shfl_xor / shfl_up / shfl_down
    // (emulated with WARP_READ_LANE at the computed peer lane).
    /*
     * _emit_shuffle(s) pseudo-code (luisa-dsl):
     *
     *   v = value_tensor[origin]
     *   UInt lane = warp_lane_id()
     *   UInt delta = s->delta()
     *   switch op:
     *       XOR  -> peer = lane ^ delta
     *       UP   -> peer = lane - delta
     *       DOWN -> peer = lane + delta
     *   tmp = warp_read_lane(v, peer);
     */
    void _emit_shuffle(const ShuffleStmt *s) {
        auto *v = s->value_tensor();
        if (v == nullptr) [[unlikely]] {
            LUISA_ERROR_WITH_LOCATION("tile_to_kernel: shuffle requires a fragment-tile value.");
        }
        auto elem_t = tensor_element_type(v->dtype());
        auto saved = _current_extent;
        _current_extent = v;
        auto val = _value_at(v, _zero_coord());
        // DSL: UInt lane = warp_lane_id(); UInt delta = s->delta()
        auto lane = warp_lane_id();
        auto delta = Expr<uint>{static_cast<uint32_t>(s->delta())};
        const Expression *peer = nullptr;
        switch (s->op()) {
            case TileShuffleOp::XOR:
                peer = (lane ^ delta).expression();
                break;
            case TileShuffleOp::UP:
                peer = (lane - delta).expression();
                break;
            case TileShuffleOp::DOWN:
                peer = (lane + delta).expression();
                break;
            default:
                LUISA_ERROR_WITH_LOCATION(
                    "tile_to_kernel: shuffle op {} is not supported by the "
                    "regular-kernel lowering.",
                    static_cast<uint32_t>(s->op()));
        }
        if (v->dtype() == TensorElementType::FP8) {
            // fp8 has no C++ scalar type: keep the dtype-erased raw path
            auto tmp = _fb->local(elem_t);
            _fb->assign(tmp, _fb->call(elem_t, CallOp::WARP_READ_LANE, {val, peer}));
        } else {
            // DSL: tmp = warp_read_lane(v, peer) (kept alive in a local)
            with_elem_type(v->dtype(), [&]<typename T>() {
                auto tmp = Var<T>{warp_read_lane(Expr<T>{val}, Expr<uint>{peer})};
            });
        }
        _current_extent = saved;
    }

    /*
     * _emit_min(s) pseudo-code (luisa-dsl):
     *
     *   temp = temp_output(s)
     *   _temps[temp] = lambda(c):
     *       return min(a[c], cast(b_literal, elem_t))
     */
    void _emit_min(const MinStmt *s) {
        auto *a = s->a();
        auto elem_t = tensor_element_type(a->dtype());
        _temps[_tile->temp_output(s)] = TempValue{
            a->dtype(),
            [this, s, a, elem_t](const Coord &c) -> const Expression * {
                auto av = _value_at(a, c);
                auto bv = _maybe_cast(_recreate_literal(s->b()), elem_t);
                // DSL: min(a[c], cast(b_literal, elem_t))
                return with_elem_type(a->dtype(), [&]<typename T>() -> const Expression * {
                    return min(Expr<T>{av}, Expr<T>{bv}).expression();
                });
            }};
    }

    /*
     * _emit_abs(s) pseudo-code (luisa-dsl):
     *
     *   temp = temp_output(s)
     *   _temps[temp] = lambda(c):
     *       return abs(a[c])
     */
    void _emit_abs(const AbsStmt *s) {
        auto *a = s->a();
        _temps[_tile->temp_output(s)] = TempValue{
            a->dtype(),
            [this, a](const Coord &c) -> const Expression * {
                // DSL: abs(a[c]) on the concrete element type
                return with_elem_type(a->dtype(), [&]<typename T>() -> const Expression * {
                    return abs(Expr<T>{_value_at(a, c)}).expression();
                });
            }};
    }

    /*
     * _emit_warp_reduce(s) pseudo-code (luisa-dsl):
     *
     *   v = value[origin]
     *   tmp = switch op:
     *           SUM     -> warp_active_sum(v)
     *           MAX     -> warp_active_max(v)
     *           MIN     -> warp_active_min(v)
     *           BIT_AND -> warp_active_bit_and(v)
     *           BIT_OR  -> warp_active_bit_or(v)
     */
    void _emit_warp_reduce(const WarpReduceStmt *s) {
        // register-level warp reduction; the IR has no consumer, so the value
        // is computed into a throw-away local to keep the call alive
        auto *v = s->value();
        auto elem_t = tensor_element_type(v->dtype());
        auto saved = _current_extent;
        _current_extent = v;
        auto val = _value_at(v, _zero_coord());
        if (v->dtype() == TensorElementType::FP8) {
            // fp8 has no C++ scalar type: keep the dtype-erased raw path
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
            return;
        }
        // DSL: tmp = warp_active_sum/max/min/bit_and/bit_or(val) (typed T);
        // the bitwise collectives are integral-only, so they are guarded with
        // if constexpr to keep the with_elem_type instantiations (half/float)
        // compiling.
        with_elem_type(v->dtype(), [&]<typename T>() {
            auto e = Expr<T>{val};
            Var<T> tmp;
            switch (s->op()) {
                case TileWarpReduceOp::SUM:
                    tmp = warp_active_sum(e);
                    break;
                case TileWarpReduceOp::MAX:
                    tmp = warp_active_max(e);
                    break;
                case TileWarpReduceOp::MIN:
                    tmp = warp_active_min(e);
                    break;
                case TileWarpReduceOp::BIT_AND:
                    if constexpr (std::is_integral_v<T>) {
                        tmp = warp_active_bit_and(e);
                    } else {
                        LUISA_ERROR_WITH_LOCATION(
                            "tile_to_kernel: WARP_ACTIVE_BIT_AND requires an integral element type.");
                    }
                    break;
                case TileWarpReduceOp::BIT_OR:
                    if constexpr (std::is_integral_v<T>) {
                        tmp = warp_active_bit_or(e);
                    } else {
                        LUISA_ERROR_WITH_LOCATION(
                            "tile_to_kernel: WARP_ACTIVE_BIT_OR requires an integral element type.");
                    }
                    break;
                default:
                    LUISA_ERROR_WITH_LOCATION("tile_to_kernel: invalid tile warp-reduce op.");
            }
        });
        _current_extent = saved;
    }
};

}// namespace

/*
 * tile_to_kernel(tile_function, config) pseudo-code:
 *
 *   if tile_function is null: error
 *   return TileLowerer{}.lower(tile_function, config)
 */
TileCompileResult tile_to_kernel(
    luisa::shared_ptr<const detail::TileFunctionBuilder> const &tile_function,
    TileToKernelConfig const &config) {
    if (tile_function == nullptr) [[unlikely]] {
        LUISA_ERROR_WITH_LOCATION("tile_to_kernel: null tile function.");
    }
    return TileLowerer{}.lower(tile_function, config);
}

}// namespace luisa::compute

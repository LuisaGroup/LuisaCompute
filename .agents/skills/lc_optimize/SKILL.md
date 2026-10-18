---
name: lc_optimize
description: Optimize LuisaCompute DSL kernels using warp/wave primitives, shared-memory aggregation, block-level collectives, thread-group (block) size / occupancy tuning, async copy / software pipelining, roofline and profiler-stall diagnosis, and C++ branch-prediction hints (`[[likely]]`/`[[unlikely]]`). Includes cross-platform hardware experience for NVIDIA (Ampere/Hopper/Blackwell, incl. TMA, wgmma, tcgen05/TMEM, clusters, PDL), AMD (GCN/RDNA/CDNA wave32/wave64, VGPR/LDS budgets, MFMA), Apple Metal, Intel Xe, and Vulkan/DX subgroup portability. Use when kernels bottleneck on atomics, reductions, inter-thread communication, memory bandwidth/coalescing, launch overhead, or poor occupancy, or when porting/tuning kernels across GPU vendors.
---

# LuisaCompute DSL Kernel Optimization Guide

> **Zero-initialization note:** All temporary local variables in DSL kernels — scalars, vectors, matrices, structs, and arrays (excluding shared arrays `Shared<T>`) — are created with a zero value automatically. There is no need to manually set them to zero before use. This applies to variables declared with `Var<T>`, `auto`, or type-inferred syntax inside a kernel or callable body.

## 1. Available Warp/Wave Primitives

LuisaCompute exposes the following warp-level (subgroup) intrinsics via `luisa/dsl/builtin.h`. All operate on *active lanes within the current warp*. This list is complete: `builtin.h` defines exactly 19 collective/communication entry points, one per `CallOp::WARP_*` kind, plus `warp_lane_count()`, `warp_lane_id()` and `set_warp_size()` (which are `FunctionBuilder` builtins, not call ops). There is **no** `wave_*` alias, no `warp_write_lane`, and no `warp_reduce_*` in the DSL.

### 1.1 Query / Metadata

| DSL Call | Returns | Description |
|---|---|---|
| `warp_lane_count()` | `UInt` | Total lanes in the warp (e.g. 32 or 64). |
| `warp_lane_id()` | `UInt` | Current lane index `[0, warp_lane_count())`. |
| `warp_is_first_active_lane()` | `Bool` | True if this lane is the first active lane in the warp. Lowers to `WaveIsFirstLane` (HLSL) / `OpGroupNonUniformElect` (SPIR-V) / `__ffs(__activemask())-1 == laneid` (CUDA). |
| `warp_first_active_lane()` | `UInt` | Lane index of the first active lane. **Not implemented in the HLSL codegen path** (`src/backends/common/hlsl/codegen_utils/function_codegen.cpp:1985` → `LUISA_NOT_IMPLEMENTED()`); prefer `warp_is_first_active_lane()`, or `ctz(warp_active_bit_mask(true).x)`. |
| `device.compute_warp_size()` | `uint` | Host-side query for the backend/device's native warp size (see section 6). |

### 1.2 Active-Lane Reductions (All-Reduce)

Each lane receives the same reduced value.

| DSL Call | Signature | Semantics |
|---|---|---|
| `warp_active_sum(v)` | `T -> T` | Sum across active lanes. |
| `warp_active_product(v)` | `T -> T` | Product across active lanes. |
| `warp_active_min(v)` | `T -> T` | Minimum across active lanes. |
| `warp_active_max(v)` | `T -> T` | Maximum across active lanes. |
| `warp_active_all(v)` | `Bool -> Bool` | Logical AND across active lanes. |
| `warp_active_any(v)` | `Bool -> Bool` | Logical OR across active lanes. |
| `warp_active_all_equal(v)` | `T -> Bool` (or `Vec<Bool,N>`) | True if all active lanes have the same value. |
| `warp_active_bit_and(v)` | `Int -> Int` | Bitwise AND across active lanes. |
| `warp_active_bit_or(v)` | `Int -> Int` | Bitwise OR across active lanes. |
| `warp_active_bit_xor(v)` | `Int -> Int` | Bitwise XOR across active lanes. |
| `warp_active_count_bits(v)` | `Bool -> UInt` | Population count of true predicates. |
| `warp_active_bit_mask(v)` | `Bool -> UInt4` | Ballot mask of true predicates — `spv::OpGroupNonUniformBallot` (uvec4) / DX `WaveActiveBallot` / CUDA `__ballot_sync` packed into `uint4(ballot,0,0,0)`, so `.y/.z/.w` are only meaningful for subgroup widths above 32. |

`warp_active_sum/product/min/max` take a scalar or vector operand and *reject* booleans (`is_scalar_expr_v || is_vector_expr_v`, `&& !is_boolean_or_vector_expr_v`); `warp_active_bit_and/or/xor` require integral or vector-of-integral (`is_integral_or_vector_expr_v`); `warp_active_all_equal` takes any scalar or vector and returns one `Bool` per component.

"Active" means lanes that actually executed the call — an irregular (sparse) subset is legal, and `src/tests/unit/runtime/test_warp_sparse_collectives.cpp` pins the expected results for the non-contiguous mask `{0, 1, 6}` across reductions, exclusive scans, vector `all_equal` and 16-bit/matrix shuffles.

### 1.3 Prefix (Exclusive Scan)

Each lane receives the exclusive prefix of all *preceding active* lanes. The first active lane receives the identity (0 for sum, 1 for product, 0u for count_bits — see `lc_warp_prefix_sum_impl`/`lc_warp_prefix_product_impl`, `src/backends/cuda/cuda_builtin/cuda_device_resource.h:3130-3137`). Exclusive (not inclusive) is asserted by `test_warp_prefix_scan.cpp:36-44`: with only even lanes active, lane `2n` must observe `n` preceding contributions.

| DSL Call | Signature | Description |
|---|---|---|
| `warp_prefix_sum(v)` | `T -> T` | Exclusive prefix sum (T: arithmetic). |
| `warp_prefix_product(v)` | `T -> T` | Exclusive prefix product (T: arithmetic). |
| `warp_prefix_count_bits(v)` | `Bool -> UInt` | Exclusive prefix popcount of true predicates. |

### 1.4 Lane Communication (Shuffle)

| DSL Call | Signature | Description |
|---|---|---|
| `warp_read_lane(v, lane_idx)` | `(T, UInt) -> T` | Read `v` from lane `lane_idx`. T can be scalar, vector, or matrix. |
| `warp_read_first_active_lane(v)` | `T -> T` | Read `v` from the first active lane. |

### 1.5 Configuration

| DSL Call | Description |
|---|---|
| `set_warp_size(uint8_t)` | Asserted by `luisa_compute_validate_warp_size` (`src/dsl/builtin.cpp:10`): must be `1`, `2`, or a power of two in `[4, 128]` — i.e. one of 1, 2, 4, 8, 16, 32, 64, 128. Call *inside* kernel lambda before compilation; it sets `Function::allowed_warp_size()`, an **exact** width requirement the backend can refuse (CUDA and both Metal backends `LUISA_ERROR` on anything but 32, the CPU fallback on anything but 1, the SIMD backend on anything but its configured width, and Vulkan errors if subgroup-size control is unavailable or out of `[minSubgroupSize, maxSubgroupSize]`). |
| `sync_block()` | Full block barrier. All threads in the block must reach it (`CallOp::SYNCHRONIZE_BLOCK`). |

**Critical rule:** Warp operations only communicate within the *same warp*. No `sync_block()` is needed for warp collectives — they are guaranteed to complete within the warp without barriers.

---

## 2. Usage Patterns (Project Analysis)

### 2.1 Warp-Level Matrix Multiplication

Pattern: each warp computes one output tile via `warp_active_sum` reduction over the K dimension.

```cpp
auto warp_size = device.compute_warp_size();
Kernel2D mat_mul = [&](BufferFloat lhs, BufferFloat rhs, BufferFloat result, UInt lhs_row_size) {
    set_block_size(128, 1, 1);
    set_warp_size(warp_size);

    UInt lhs_y = dispatch_id().x / warp_size;
    UInt rhs_x = dispatch_id().y;
    UInt warp_lane = warp_lane_id();

    UInt tile_count = (lhs_row_size + warp_size - 1) / warp_size;
    Float acc = 0.f;

    for (auto t : dynamic_range(tile_count)) {
        UInt lhs_x = t * warp_size + warp_lane;
        Float local_v;
        $if (lhs_x < lhs_row_size) {
            local_v = lhs.read(lhs_y * lhs_row_size + lhs_x)
                    * rhs.read(rhs_x * lhs_row_size + lhs_x);
        } $else {
            local_v = 0.f;
        };
        acc += warp_active_sum(local_v);   // all-reduce sum within warp
    }

    $if (warp_lane == 0) {
        // Row-major [M x N] result: N == dispatch_size().x / warp_size
        // (test_warp.cpp's rhs_matrix_size.x).
        result.write((dispatch_size().x / warp_size) * lhs_y + rhs_x, acc);
    };
};
```

**Key insight:** Warp-active reductions eliminate the need for shared memory entirely. All lanes in a warp already execute in lockstep, so `warp_active_sum` is one `WaveActiveSum` intrinsic on DX and one `OpGroupNonUniform{I,F}Add` on SPIR-V; on CUDA it is `__reduce_add_sync` for the integer types but a 5-step `__shfl_xor_sync` butterfly for float/half (`src/backends/cuda/cuda_builtin/cuda_device_resource.h:2772-2780`, `:2981-2997`) — still cheaper than shared memory plus a barrier.

### 2.2 Butterfly Reduction via `warp_read_lane`

For finding the maximum across a logical group smaller than the warp, use pairwise `warp_read_lane` with XOR lane masks (butterfly / tree-reduction pattern; `lane` here is `warp_lane_id()`):

```cpp
// 8-lane max reduction within a group of 8
Float m = input;
m = max(m, warp_read_lane(m, lane ^ 4u));  // distance 4
m = max(m, warp_read_lane(m, lane ^ 2u));  // distance 2
m = max(m, warp_read_lane(m, lane ^ 1u));  // distance 1
// Now lane 0..7 all have the max of lanes 0..7 (modulo diverged lanes)
```

This is used for the softmax normalization constant. Each logical group of 8 lanes computes its own max independently, *without* a barrier.

### 2.3 Grouped Prefix + Inter-Group Read

When a warp contains multiple independent logical groups, compute the inclusive prefix sum per group, then use `warp_read_lane` to fetch the last element of the previous group:

```cpp
constexpr uint kWarpSize = 32u;
constexpr uint kGroupLanes = 8u;
constexpr uint kGroupsPerWarp = kWarpSize / kGroupLanes;  // 4

auto lane = warp_lane_id();
auto group_id = lane / kGroupLanes;      // which group (0..3)
auto group_lane = lane % kGroupLanes;    // position within group (0..7)

auto prefix = warp_prefix_sum(value);    // exclusive prefix across whole warp
auto inclusive = prefix + value;

// Last lane of this group
auto last_lane = group_id * kGroupLanes + (kGroupLanes - 1u);
auto incl_last = warp_read_lane(inclusive, last_lane);

// Last lane of previous group (or 0 for group 0)
auto prev_last = ite(group_id == 0u, 0u, last_lane - kGroupLanes);
auto prev_incl = warp_read_lane(inclusive, prev_last);

auto group_sum = incl_last - ite(group_id == 0u, make_float2(0.f), prev_incl);
```

This avoids separate `warp_prefix_sum` calls per group and instead uses a single warp-wide prefix plus lane reads to extract group boundaries. Pattern from `src/tests/unit/runtime/test_mha_warp_reduction.cpp:38-62` (grouped `float2` scan) and `:110-127` (grouped scalar softmax); note the pinned `set_warp_size(kWarpSize)` and `set_block_size(256u, 1u, 1u)` in that kernel.

### 2.4 Warp-Polling Decoupled Look-Back

For inter-block scan, tiles publish their status and other tiles poll via warp collectives:

```cpp
// Poll across warp: any lane sees INVALID? (ScanTileStateViewer::WaitForValid,
// test_decoupled_look_back.cpp:223-229 — `delay` is a caller-supplied host
// lambda, not a DSL builtin; `volatile_read` is BufferView's)
$while (warp_active_any(status == SCAN_TILE_INVALID)) {
    delay();
    status = tile_status.volatile_read(predecessor_idx);
};

// All lanes agree predecessor is inclusive? (:320-336)
$while (warp_active_all(predecessor_status != SCAN_TILE_INCLUSIVE)) {
    predecessor_idx -= compute::Int(32);
    // poll next window...
    exclusive_prefix = scan_op(windows_aggregate, exclusive_prefix);
};
```

Also uses `warp_active_bit_mask` for segmented reductions within warps (`test_decoupled_look_back.cpp:82-95`; `get_lane_mask_ge` is a `static Callable` there, `LOGIC_WARP_SIZE` a template parameter):

```cpp
UInt warp_flags = warp_active_bit_mask(flag == 1u).x;
warp_flags >>= 1;  // for HEAD_SEGMENT mode
warp_flags &= get_lane_mask_ge();  // mask of lanes with id >= mine
warp_flags |= 1u << (UInt(LOGIC_WARP_SIZE) - 1u);  // sentinel
UInt last_lane = ctz(warp_flags);  // first set bit = end of my segment
```

### 2.5 Shuffle-Down for Warp Reduction

A software implementation of warp reduce using `warp_read_lane` with increasing offsets:

```cpp
// WarpReduceShfl::ReduceStep in test_decoupled_look_back.cpp: a shuffle-down
// butterfly. warp_read_lane takes exactly (value, src_lane) — the out-of-range
// fix-up is a separate guarded assignment (see ShuffleDown, same file:24-34).
Var<T> result = input;
UInt offset = 1u;
$while (offset < warp_lane_count()) {
    UInt src_lane = lane_id + offset;
    Var<T> temp = warp_read_lane(result, src_lane);
    $if (src_lane > valid_item) { temp = result; };  // ShuffleDown fix-up
    $if (lane_id + offset <= valid_item) {
        result = reduce_op(result, temp);
    };
    offset <<= 1;
};
```

This is a fallback pattern; prefer `warp_active_sum` / `warp_active_min` / `warp_active_max` when the operation matches the built-in.

### 2.6 Quantized Matmul with Warp

Warp-level GEMM where each warp computes one output tile. Threads cooperatively load quantized weights via `warp_read_lane` to assemble dequantized values, then accumulate with `warp_active_sum` (pattern from `src/tests/unit/runtime/test_fp8_quantization.cpp:225-256`; the same shape appears in `test_fp4_quantization.cpp`):

```cpp
auto warp_lane = warp_lane_id();
UInt tile_count = (K + warp_size - 1) / warp_size;
for (auto t : dynamic_range(tile_count)) {
    UInt tile_begin = t * warp_size;
    UInt tile_size = min(warp_size, K - tile_begin);
    UInt k = tile_begin + warp_lane;

    // Each lane loads one packed 4-byte word, then shares it via warp_read_lane
    UInt rel_byte = k - tile_begin;
    UInt word = warp_read_lane(warp_word, rel_byte / 4u);
    // ... dequantize (word >> (byte_offset % 4u) * 8u) & 0xff ... and multiply ...

    acc += warp_active_sum(local_v);
}
```

---

## 3. Optimization Transformations

### 3.1 Shared-Memory Atomic → Warp Collective

**Before:** Block-level atomic on shared memory.
```cpp
Shared<int> shared{1u};
shared[0u] = 0;
sync_block();
shared.atomic(0u).fetch_add(1);  // contended within block
sync_block();
$if (thread_x() == 0u) {
    global_counter.atomic(0u).fetch_add(shared.read(0u));
};
```

**After:** Use `warp_active_sum` per warp, then one lane writes.
```cpp
// Each warp computes its own partial sum
Int warp_partial = warp_active_sum(1);   // a raw literal is a valid Expr argument
// First lane of each warp writes that warp's slot in the block
$if (warp_is_first_active_lane()) {
    shared.atomic(thread_x() / warp_lane_count()).fetch_add(warp_partial);
};
sync_block();
// One thread (e.g. thread 0) combines across warps
$if (thread_x() == 0u) {
    Int block_total = 0;
    for (auto w : dynamic_range(num_warps_per_block)) {
        block_total += shared.read(w);
    };
    global_counter.atomic(0u).fetch_add(block_total);
};
```

### 3.2 Shared-Memory Reduction → Warp Reduction

**Before:** Entire block reduces into shared memory.
```cpp
Shared<float> smem{block_size};
smem[tid] = value;
sync_block();
for (uint stride = block_size / 2; stride > 0; stride >>= 1) {
    $if (tid < stride) {
        smem[tid] += smem[tid + stride];
    };
    sync_block();
};
```

**After:** Warp-level reduction + cross-warp shared reduction.
```cpp
// Phase 1: warp-level reduction (no shared memory, no barrier)
Float warp_sum = warp_active_sum(value);
// One slot per warp, declared in the kernel body (not inside a divergent $if)
UInt warp_in_block = thread_x() / warp_lane_count();
Shared<float> warp_results{num_warps_per_block};
$if (warp_is_first_active_lane()) {
    warp_results[warp_in_block] = warp_sum;
};
sync_block();
// Only num_warps_per_block threads participate in phase 2
$if (thread_x() < num_warps_per_block) {
    Float v = warp_results[thread_x()];
    Float warp_partial = warp_active_sum(v);  // second warp reduce
    $if (warp_is_first_active_lane()) {
        result = warp_partial;
    };
};
```

### 3.3 Pairwise Max/Min → Built-in Warp Max/Min

**Before:** Butterfly pattern with `warp_read_lane`.
```cpp
Float m = value;
m = max(m, warp_read_lane(m, lane ^ 4u));
m = max(m, warp_read_lane(m, lane ^ 2u));
m = max(m, warp_read_lane(m, lane ^ 1u));
```

**After:** Single `warp_active_max` when reduction spans the *whole* warp.
```cpp
Float m = warp_active_max(value);
```

**When to keep butterfly:** Only when reducing over a logical group *smaller* than the warp (e.g. 8-lane groups inside a 32-lane warp). In that case `warp_active_max` would reduce over all 32 lanes, which is incorrect.

### 3.4 Sequential Lane Reads → `warp_prefix_sum`

**Before:** Manually accumulating values from lower lanes via a loop of `warp_read_lane` (a device loop, so `dynamic_range`, and a host `for` over `warp_lane_id()` is invalid because the bound is a device value).
```cpp
Float prefix = 0.f;
for (auto i : dynamic_range(warp_lane_id())) {
    prefix += warp_read_lane(value, i);
};
```

**After:** Single `warp_prefix_sum` call.
```cpp
Float prefix = warp_prefix_sum(value);
```

### 3.5 Conditional Participation

When only a subset of lanes should participate in a warp collective, wrap the call in a conditional. Lanes that *don't* execute the call are excluded from the reduction/scan:

```cpp
$if (thread_x() % 2u == 0u) {
    // Only even-index threads contribute
    auto result = warp_prefix_sum(make_half4(.5_h));
    device_log("{} -> {}", dispatch_x(), result);
};
// Odd threads don't participate; they receive no result.
```

This pattern is commonly used for partial-warp scans where only a subset of lanes needs results.

### 3.6 Ballot + Count Bits for Control Flow

Use `warp_active_bit_mask` to build a lane mask, then `ctz` / `popcount` to locate lanes or count participants:

```cpp
UInt4 mask = warp_active_bit_mask(condition);
UInt flag_mask = mask.x;  // first 32 lanes
UInt first_true = ctz(flag_mask);
UInt num_true = popcount(flag_mask);
```

### 3.7 Shared-Memory Cross-Lane Communication → Warp Intrinsics (Generic Recipe)

`Shared<T>` used *only* to exchange values between lanes of one warp can be replaced by warp intrinsics: no barrier, no shared capacity, and usually a single hardware instruction. This is valid whenever cooperation is fully intra-warp — either the block is exactly one warp, or each warp processes an independent work item and uses `warp_lane_id()` as its lane index.

**Step 1 — Audit every `Shared<T>` access.** Classify each use:
- Slot indexed by lane id (`shared[lane]`, `shared[lane + k]`) → shuffle / broadcast.
- Ballot, reduction, or scan over lanes → warp collective.
- Any access indexed by something other than the lane (arbitrary thread id, work-item id, dynamic offsets), or any value that another warp must see → keep shared (section 4).

Only when **every** access is lane-local is the refactor sound.

**Step 2 — Map each shared-memory idiom to a warp intrinsic.**

| Shared-memory idiom | Warp replacement |
|---|---|
| Ballot: `shared[tid] = pred;` tree-OR with barriers; read `shared[0]` | `warp_active_bit_mask(pred).x` |
| Broadcast: `$if (tid == src) { shared[0] = v; };` barrier; read `shared[0]` | `warp_read_lane(v, src)` (or `warp_read_first_active_lane(v)`) |
| Shuffle: `shared[src] = v;` barrier; read `shared[tid]` | `warp_read_lane(v, src)` |
| Exclusive scan: Hillis–Steele `shared` loop + barriers | `warp_prefix_sum(v)` |
| All-reduce: shared tree + barriers | `warp_active_sum` / `warp_active_min` / `warp_active_max` / ... |
| Neighbor gather: lane `tid` reads `shared[tid + k]` | `warp_read_lane(v, tid + k)` |
| Vote then count: `shared` ballot + `popcount` | `warp_active_bit_mask(pred).x` + `popcount`, or `warp_active_count_bits(pred)` |

**Step 3 — Apply the correctness rules.**

1. **All lanes must participate in every shuffle.** `warp_read_lane` reads the *active* source lane; if a source lane skipped the call inside a divergent `$if`, the read is undefined. Compute shuffled values unconditionally into locals, then guard only the consumer `$if`.
2. **Clamp out-of-range lane indices.** `warp_read_lane(v, min(tid + k, warp_lane_count() - 1u))` keeps boundary lanes in range; their unused results are discarded, so clamping is safe.
3. **Delete now-redundant `sync_block()` calls.** They existed to publish shared writes across lanes; warp intrinsics are complete for the calling lane immediately. Leaving them adds a block-wide stall.
4. **Use `warp_lane_id()` as the lane index** once the block has more than one warp; `thread_x()` then only computes the warp index (section 3.8).
5. **Never mix warp and block scope.** Values that must cross warps, or non-lane indexing, stay in shared memory (section 4).

### 3.8 Block-Size Scaling: One Warp per Independent Work Item

Kernels often start with `set_block_size(32, 1, 1)` and one warp per item because the item's cooperative step needs exactly 32 lanes. Once the cross-lane logic is warp-only (section 3.7), enlarge the block to 64/128 threads — several warps per block, each still owning one item — which usually improves occupancy and reduces per-block overhead:

```cpp
constexpr uint kBlockThreads = 128u;
constexpr uint kWarpThreads = 32u;
constexpr uint kWarpsPerBlock = kBlockThreads / kWarpThreads;
set_block_size(kBlockThreads, 1u, 1u);
set_warp_size(static_cast<uint8_t>(kWarpThreads)); // pin so the mapping is exact

UInt lane = warp_lane_id();                         // 0..warp_lane_count()-1
UInt warp_in_block = thread_x() / warp_lane_count();// which warp inside this block
UInt item_idx = block_id().x * kWarpsPerBlock + warp_in_block;
$if (item_idx < num_items) { /* ... work on item_idx with lanes `lane` ... */ };
```

- **Pin the warp size.** If the algorithm assumes 32 lanes per item, call `set_warp_size(32)` (host: `device.compute_warp_size()`). Without pinning, a backend may choose a wider wave/subgroup (e.g. 64), which shrinks `warp_in_block` and silently skips items.
- **Keep the tail guard.** `$if (item_idx < num_items)` makes idle warps in the last block harmless, so the host dispatch needs no change: `dispatch(num_items * warp_size)` still covers every item.
- **When each thread owns an item** (no cross-lane cooperation), the same block-size increase is simpler: keep `dispatch_id().x` indexing and just raise `set_block_size`; the total thread count is unchanged.
- **Verify on multiple backends.** Warp intrinsics lower to different hardware ops (`Wave*` intrinsics on DX, `OpGroupNonUniform*` subgroup ops on Vulkan/SPIR-V, `__shfl*`/`__reduce_*_sync` on CUDA). Re-run the same correctness cases — including tail/partial-item sizes that exercise idle warps and boundary lanes — on at least two backends after the change.

---

## 4. Shared Array (Workgroup Memory) Optimization

Warp collectives (section 3) only communicate *within one warp*. When cooperation must span the **whole thread block** (multiple warps), or you need persistent per-block scratch, arbitrary cross-thread indexing, or block-local privatization of a global atomic, use a **shared array** (`Shared<T>` / `$shared<T>`). Shared memory is on-chip and orders of magnitude faster than global memory, so staging data there once and reusing it, or aggregating locally before touching global memory, is a core optimization. (For the reverse direction — replacing *lane-local* shared memory with warp intrinsics — see section 3.7.)

### 4.1 API (`include/luisa/dsl/shared.h`, `include/luisa/dsl/atomic.h`; `$shared` macro = `include/luisa/dsl/sugar.h:497`)

| DSL | Description |
|---|---|
| `Shared<T> s{n}` / `$shared<T> s{n}` | Allocate `n` elements of `T` in workgroup memory (`explicit Shared(size_t n)`, shared.h:42). Must be constructed **inside** the kernel/callable body (uses `FunctionBuilder::current()->shared()`). |
| `s[i]` | Returns `Var<T>&` — a temporary reference for read or write; `i` must be an integral expr (`requires is_integral_expr_v<U>`, shared.h:57-58). |
| `s.read(i)` / `s.write(i, v)` | Explicit read / write helpers (`read` returns `s[i]`, `write` does `s[i] = v`; shared.h:68, 74). |
| `s.atomic(i)` | Returns `detail::AtomicRef<T>` for that slot (shared.h:17, via the `detail::SharedAsAtomic<T>` base). It is **not** a no-op: it must be followed by a scalar op — `.fetch_add(v)`, `.fetch_sub`, `.fetch_min/max`, `.fetch_and/or/xor`, `.exchange`, `.compare_exchange(e, v)` (atomic.h:63-122). `fetch_and/or/xor` exist only on the int/uint/slong/ulong specialization, not on the `float` one (atomic.h:128-175). |
| vector / array / matrix / tuple elements | `AtomicRef<Vector<T,N>>` exposes no aggregate op — reach the scalar lanes first: `s.atomic(i).x.fetch_add(v)` or `s.atomic(i)[k]` (atomic.h:204-249); same for `std::array` (`operator[]`), `Matrix<float,N>` and `std::tuple` (`get<i>()`). |
| custom-struct elements | Unavailable: `detail::SharedAsAtomic<T>` has an empty `requires is_custom_struct_v<T>` specialization (shared.h:26-28), so `s.atomic(i)` does not exist. |
| `s.size()` | Element count (`size_t`, host-side constant; shared.h:53). |
| `new Shared<T>{n}` | Heap-allocate so helper classes can *own* shared scratch. `Shared<T>` deletes copy-construction and **both** assignment operators and only defaults the move constructor (shared.h:47-50), so it cannot be stored by value in a container. The shared variable itself is registered on the enclosing `FunctionBuilder`, which is why the pointer form works. See `WarpReduce` in `test_decoupled_look_back.cpp:122`. |

Always `set_block_size(...)` and size the array to the block (`Shared<T> s{block_size}`). Use `sync_block()` to make writes visible across warps.

### 4.2 When to prefer shared memory over warp collectives

| Situation | Use |
|---|---|
| Reduction/scan fits in a single warp | Warp collective (section 3) — no barrier, single instruction. |
| Reduction spans a whole block (block_size > warp_size) | Two-level: warp collective → shared → block (section 4.5), or full shared-memory tree reduction (4.4). |
| Many threads append to one global counter/queue | Block-local privatization in shared, then **one** global atomic per block (4.3). |
| Global data reused by many threads in a block | Stage global → shared once, `sync_block()`, then reuse (4.6). |
| Arbitrary cross-thread indexing (not just lane shuffles) | Shared array indexed by `thread_id()`. |

### 4.3 Block-Local Atomic Privatization → One Global Atomic

The biggest shared-memory win: replace *up to block_size* contended **global** atomics with per-thread **shared** atomics plus a **single** global atomic per block. Pattern from `test_atomic_queue.cpp` (`push_if`) and `test_shared_memory.cpp` (`AtomicQueue::push`):

```cpp
// Append `value` to a global queue when `pred` holds, minimizing global contention.
Shared<uint> index{1};
$if (thread_x() == 0u) { index.write(0u, 0u); };   // init block counter
sync_block();

auto local_index = def(0u);
$if (pred) { local_index = index.atomic(0).fetch_add(1u); };  // cheap SHARED atomic
sync_block();

$if (thread_x() == 0u) {                            // ONE global atomic for the whole block
    auto local_count   = index.read(0u);
    auto global_offset = _counter->atomic(0u).fetch_add(local_count);
    index.write(0u, global_offset);                 // reuse slot to broadcast the base
};
sync_block();

$if (pred) {                                        // scatter to reserved, contiguous range
    auto global_index = index.read(0u) + local_index;
    _buffer->write(global_index, value);
};
```

**Insight:** Global-atomic traffic drops from O(active threads) to O(1) per block. Contention moves from device-wide global memory to fast on-chip shared memory. This is the standard stream-compaction / queue-append optimization.

### 4.4 Block-Wide Tree Reduction in Shared Memory

When the reduction spans the whole block, stage each thread's value in shared memory and reduce pairwise with a halving loop. Pattern from `test_softmax.cpp` (block sum for softmax) and `test_complex_kernel.cpp`:

```cpp
set_block_size(block_size, 1, 1);          // power of two
Shared<float> shared_arr(block_size);       // one slot per thread
auto tid = thread_id().x;
shared_arr[tid] = value;                     // stage per-thread value

UInt half = block_size / 2u;
sync_block();
$while (half > 0u) {
    $if (tid < half) {                       // compute into a register FIRST
        value = shared_arr[tid * 2] + shared_arr[tid * 2 + 1];
    };
    sync_block();                            // barrier between read and write-back
    $if (tid < half) {
        shared_arr[tid] = value;             // write reduced value back
    };
    half /= 2u;
    sync_block();                            // barrier before next iteration's reads
};
$if (tid == 0u) { output.write(block_id().x, shared_arr[0]); };  // thread 0 emits block result
```

**Why two `sync_block()` per step:** reducing into a local `value` register and only writing back after a barrier avoids the read-after-write / write-after-read hazard where one thread overwrites a slot another thread is still reading. Prefer this whole-block form only when `block_size > warp_size`; inside a single warp, `warp_active_sum` (section 3.2) is faster and barrier-free.

### 4.5 Two-Level Reduction: Warp Collective → Shared → Block

Combine both tools: reduce within each warp with a warp collective (no barrier), write one partial per warp to a *small* shared array, then reduce those partials. This minimizes both shared traffic and barriers vs. a full block tree reduction (see also sections 3.1/3.2):

```cpp
UInt warp_id = thread_x() / warp_size;            // which warp in the block (warp_size = device.compute_warp_size())
Float warp_sum = warp_active_sum(value);          // phase 1: intra-warp, no barrier
Shared<float> warp_results{num_warps_per_block};
$if (warp_is_first_active_lane()) {
    warp_results[warp_id] = warp_sum;             // one write per warp
};
sync_block();
$if (thread_x() < num_warps_per_block) {          // phase 2: reduce the few partials
    Float block_sum = warp_active_sum(warp_results[thread_x()]);
    $if (warp_is_first_active_lane()) { /* thread 0 has the block total */ };
};
```

### 4.6 Shared as Staging / Scratch for Reuse & Exchange

Load global data into shared once, then reuse it many times or exchange it between threads, avoiding repeated global reads. Patterns from `test_shared_mem.cpp`, `test_async_copy.cpp`, and hierarchical mip reduction in `test_mipmap.cpp`:

```cpp
set_block_size(N, 1u, 1u);
Shared<uint> s_src{N};
auto tid = thread_x();
s_src[tid] = src_buf.read(dispatch_x());     // global -> shared, once
sync_block();                                 // publish to the whole block
// ... now reuse s_src[...] / read neighbors' values without touching global memory ...
```

`test_async_copy.cpp` fills a shared staging buffer with `async_copy(scope, dst_lvalue, src_addr, elem_bytes, num, stride, event)` — **every** thread issues its own copy (thread 0 alone would leave the rest unfilled), then `pipeline_commit()`, `pipeline_wait_prior(0u)` and `sync_block()` before consumers read (`src/tests/unit/dsl/test_async_copy.cpp:53-57`; the builtin and its `OpGroupAsyncCopy`/`cp.async` semantics are documented at `include/luisa/dsl/builtin.h:2053-2063`). `test_mipmap.cpp` writes 2×2 block averages into `Shared<float3>` and reduces level-by-level with a `sync_block()` between levels (`src/tests/unit/runtime/test_mipmap.cpp:106`).

### 4.7 Correctness & Performance Rules

1. **Construct inside the kernel body.** `Shared<T>` needs `FunctionBuilder::current()` (shared.h:43); declaring it outside a kernel/callable is invalid. Declare it in the function body, not inside a divergent `$if` you later read from outside that `$if` (C++ scope).
2. **Barrier discipline.** A `sync_block()` is required (a) after initializing/filling shared before other threads read, and (b) between the read and write-back phases of each reduction step. Unlike warp collectives, **shared memory is NOT self-synchronizing across warps**.
3. **Size to the block.** Match the array length to `set_block_size(...)`; use a power-of-two block for the halving tree reduction, and pad out-of-range lanes with the reduction identity (e.g. `0.f` for sum) — see the `$if (index < size) {...} $else {...}` guards in `test_softmax.cpp:46-51`.
4. **Register-then-write.** In tree reductions, compute into a `Var`/register and write back only after a barrier to avoid RAW/WAR hazards.
5. **Move-construct only.** `Shared<T>` deletes copy construction and both assignment operators; store `Shared<T> *` (via `new`) when a helper class must hold shared scratch.
6. **Stay inside the 32 KiB HLSL floor.** The HLSL codegen path asserts total shared size `<= 32768` bytes per group (`src/backends/common/hlsl/hlsl_codegen.cpp:798`); the Vulkan path allows more (`Device::compute_max_shared_memory_size()`).
7. **Prefer warp collectives when they suffice.** Shared memory costs a barrier and on-chip capacity; only reach for it when cooperation exceeds one warp or needs privatization/staging/arbitrary indexing.

---

## 5. Thread-Group (Block) Size Selection and Occupancy

Choosing `set_block_size(x, y, z)` is one of the most impactful tuning decisions for a kernel. The right size depends on whether the kernel is **memory/IO-bound** or **compute-bound**, because group size directly controls **occupancy** — how many thread groups (blocks) can co-reside on one compute unit (CU/SM) — which in turn determines how well the GPU hides latency by switching between waves/warps.

### 5.1 Hardware Constraints

A thread group cannot be split across CUs: every wave/warp of a group must fit into **one CU's resources simultaneously** before the group can begin executing. The limits LuisaCompute itself enforces (`luisa_compute_validate_block_size`, `src/dsl/builtin.cpp:19-27`) are:

| Limit | Value | Enforced by |
|---|---|---|
| Each block dimension | `[1, 1024]` | `LUISA_ASSERT(all(size >= 1u && size <= 1024u), ...)` |
| Threads per group (x·y·z) | `<= 1024` **and** a multiple of 32 | `LUISA_ASSERT(thread_count <= 1024u && thread_count % 32u == 0u, ...)` |
| Shared memory per group | `<= 32768` B on the HLSL (DX) path; backend-queried on Vulkan | `src/backends/common/hlsl/hlsl_codegen.cpp:798`; `Device::compute_max_shared_memory_size()` (per-backend, `src/backends/vk/device.cpp:626`) |

Hardware wave/warp widths are not a repo constant — see section 6 for the per-backend query. Because all waves of a group must fit simultaneously, a larger group can reduce the number of resident groups per CU, lowering occupancy and the GPU's ability to hide memory latency.

### 5.2 Memory/IO-Bound Kernels

> **Scope note:** §5.2–§5.5 are vendor architecture guidance (GCN/RDNA wave counts, VGPR budgets, profiler names), not facts asserted by this repo — the only repo-enforced limits are the ones tabulated in §5.1, and the only repo-measured numbers are in §5.7. Treat the rest as priors to confirm with a sweep.

**Characteristic:** the kernel spends most of its time waiting for global memory loads/stores; arithmetic intensity (FLOPs per byte moved) is low.

| Goal | Approach |
|---|---|
| Maximize latency hiding | Use **smaller groups (64–256 threads)** so multiple groups fit on one CU |
| Maximize occupancy | Reduce register pressure and LDS usage per group |
| Exploit cache hierarchy | Use shared memory (section 4) to coalesce/redundant loads, but keep it small |

**Why smaller groups help:** a memory-bound CU stalls waiting for global memory; hiding that latency requires **more concurrent waves** from multiple groups. A 1024-thread group with moderate register usage can consume so many VGPRs that only one group fits per CU, leaving SIMD units idle when its waves stall.

Practical defaults:
- **64 threads** — one wave on AMD, two warps on NVIDIA; minimal footprint; barriers can often be eliminated entirely.
- **128 or 256 threads** — good balance when some LDS is needed; AMD recommends 256 as the default when LDS is not heavily used.

LDS caveat: if shared memory is used to cut global traffic (e.g. tiling), keep the per-group LDS small enough that **at least 2 groups fit per CU**; otherwise LDS becomes the new bottleneck.

### 5.3 Compute-Bound Kernels

**Characteristic:** dominated by ALU operations (complex math, loops, branching); arithmetic intensity is high.

| Goal | Approach |
|---|---|
| Maximize ALU throughput | Use **larger groups (512–1024 threads)** to saturate the CU's SIMD units |
| Reduce redundant computation | Store reusable intermediates in LDS or registers |
| Balance resources per block | Trade off register pressure vs. shared memory |

**Why larger groups help:** compute-bound kernels want the maximum number of active threads doing math per CU. A 1024-thread group gives more waves (16 on AMD GCN) filling SIMD slots — but only if the register file is not exhausted. On AMD GCN a CU has 65,536 VGPRs; at 40 VGPRs/thread, 1024 threads need 40,960, leaving headroom for a second group but pushing the limit. **Register spilling** (compiler moves variables to global memory) collapses performance due to memory latency.

Practical guidance:
- **LDS-heavy reuse** (stencils, convolution tiles): 512 threads is often the sweet spot — large enough to amortize LDS setup, small enough to fit multiple groups per CU.
- **Pure register-heavy compute** with no LDS: prefer 256 threads to avoid occupancy cliffs; profile with Nsight Compute or Radeon GPU Profiler.

### 5.4 Summary Table

| Workload Type | Recommended Group Size | Key Tuning Knob |
|---|---|---|
| **Memory/IO-bound** (bandwidth/latency limited) | 64–256 threads | Occupancy; minimize registers & LDS |
| **Compute-bound** (ALU heavy, high arithmetic intensity) | 256–1024 threads | Saturate SIMD; balance registers vs. LDS |
| **LDS-heavy tiling** (image filters, stencils) | 128–512 threads | Fit ≥2 groups per CU; watch the LDS limit |

### 5.5 Profiling Is Non-Negotiable

The right size is found empirically:
- **NVIDIA:** Nsight Compute — check "Speed of Light" throughput and Roofline charts. If memory utilization > 60% while SM utilization < 60%, the kernel is memory-bound → reduce group size.
- **AMD:** Radeon GPU Profiler — inspect wave occupancy and LDS pressure.
- **Rule of thumb:** if the kernel uses no LDS at all, default to **256 threads** for broad hardware compatibility.

### 5.6 Relationship to the Rest of This Guide

- Group size interacts with warp collectives: when cooperation is per-warp, enlarging the block packs more warps per CU (`item_idx = block_id().x * warps_per_block + thread_x() / warp_lane_count()`, section 3.8). Both `set_block_size(...)` and `set_warp_size(...)` are called inside the kernel lambda.
- The LDS budget (section 4) caps how large a shared tile can be while keeping ≥2 groups resident.

### 5.7 Empirical Benchmark Results (Measured, Not Theoretical)

Measured with the block-size sweep harness `src/tests/unit/runtime/test_block_size_bench.cpp` (8 workload archetypes × block sizes 32–1024, constant total work, correctness-checked before timing; run it against a release build). Device: NVIDIA GeForce RTX 5060 (warp 32, ~448 GB/s DRAM), Windows, LuisaCompute `vk` and `dx` backends. Every number below reproduced twice within 2.6% by an independent re-run; the harness writes the raw rows to `benchmark_results/<backend>_block_size_bench.csv` (CSV rows are appended and flushed incrementally so a case that hard-fails at dispatch cannot lose earlier cases; the register-heavy case runs last because CUDA fails such launches fatally instead of spilling — see §24). The same harness re-run on the `cuda` backend (RTX 4080 SUPER) reproduced every lesson below; deltas are tabulated in §24.

Per-case winners (block size → throughput, VK / DX):

| Case | Best size | Result | Worst | Lesson |
|---|---|---|---|---|
| elementwise float4 stream (256 MB traffic) | 64–1024 (plateau) | ~404 GB/s both backends | 32 on VK only: 247 GB/s (−39%); DX unaffected | On VK a 1-warp block underutilizes SM scheduling; 2+ warps per block saturate DRAM. Throughput identical from 64 to 1024 — for pure streaming, block size barely matters once ≥ 2 warps. |
| block reduction (Shared<float> tree + barriers) | 64–128 | 159–166 GB/s | 1024: −46…51% | Barrier stragglers grow with warps/group. 1024-thread blocks halve throughput even though occupancy math suggests they should fit. |
| tiled GEMM (shared staging) | 16×16 = 256 | 1572 GFLOPS VK / 1197 DX | 128×1, 512×1 column shapes: 4–5× slower on VK, ~1.6–2× on DX | Square tile shapes beat flat column shapes at equal thread count — LDS bank/layout and coalescing dominate, not raw size. |
| warp-intrinsic reduction (warp_active_sum per item) | 64 (2 warps/block) | 297 GB/s VK / 332 DX | 32 (1 warp/block): 2.1× slower | Warp-scope kernels still want ≥ 2 warps per block: 1-warp blocks pay per-block overhead and limit scheduling slots. Match the *warp* to the item, not the *block*. |
| divergent irregular loop (varied trip counts) | 64–512 (flat) | 0.86 ms VK / 0.34 ms DX | 32 on DX: 1.77× slower | Small blocks are fine, but 1-warp blocks are again the outlier on DX. Note the 2.6× VK-vs-DX absolute gap on identical code — always tune per backend. |
| register-heavy (32+ live float4 accumulators) | 64–128 | baseline | 1024: 19× slower — but it still *launches* | High register pressure + huge blocks does not fail to launch; it spills and crawls. Treat "runs" ≠ "fits": check the slowdown, not just launch success. |
| histogram: shared privatization vs global atomics | global wins ≤ 256; shared wins 512–1024 | 1.3–1.7× at 512+ | shared is 8× SLOWER at 64 | Privatization pays only when the zero-fill + drain cost is amortized over enough threads. Below ~256 threads/block it is a pessimization. |
| image 2D write pattern | 16×16 or 256×1 | ~420 GB/s | 1×64, 1×256 column shapes: ~3.6× worse | Shape ≫ size. Never use 1-wide thread groups for 2D data. 8×8 was mid-pack (~290 GB/s), below 64×1 — row-major-linear beats square at small sizes on this part. |

Cross-cutting measured findings:

1. "Optimal" sizes cluster at 64–256. Not a single measured case was fastest at 512 or 1024; 1024 was worst or near-worst in every case except histogram-at-scale. This strongly confirms §5.2–5.3: start at 64/128/256 and only grow when the algorithm's tile demands it.
2. Non-multiple-of-bundle sizes: the DSL hard-asserts `block_size % 32 == 0` (`src/dsl/builtin.cpp:25`), so a true 100-thread group is impossible. Emulated as a 128-thread block with 28 idle lanes (`"100e"`, i.e. `case_elementwise_partial<128, 100>`), it *beat* the full 128 block on the streaming case (514 vs 404 GB/s) — the partial warp leaves scheduling headroom. Do not chase this: it also wastes lanes on compute-bound work, and idle-lane groups are not portable. Just never deliberately pick non-multiples on hardware that allows them; here the API forces you into the right choice.
3. VK vs DX on identical DSL kernels: streaming cases within ~4%, warp_reduce 12% apart, divergent 2.6× apart, GEMM ±30%. There is no universal winner — re-run the sweep on both backends when shipping cross-backend.
4. Measured bandwidths: effective (L2-inclusive) numbers can slightly exceed DRAM peak on repeated back-to-back dispatches; compare sizes within one case, not against theoretical peaks.

LuisaCompute-specific traps hit while building the harness (verified in source — read the harness before writing your own sweep):

1. `dispatch(n)` takes a *total thread count*, not a block count — the runtime computes grid = ceil(n / block_size). Dispatching n/block_size silently under-dispatches by BS× and every case "passes" with ~0 ms.
2. D3D12 caps dispatch *group* dimensions at `D3D12_CS_DISPATCH_MAX_THREAD_GROUPS_PER_DIMENSION` (65535); a 1D grid over >65535 groups fails on DX with `"Dispatch size X out of range"` (`src/backends/dx/DXRuntime/CommandBuffer.cpp:82`). Split into a 2D grid (round dim0 to a multiple of the block size) and flatten with `dispatch_id().y * dispatch_size().x + dispatch_id().x`.
3. The DX/HLSL path asserts shared memory ≤ 32 KiB per group (`src/backends/common/hlsl/hlsl_codegen.cpp:798`); the VK path reports a larger `VkPhysicalDeviceLimits::maxComputeSharedMemorySize` via `compute_max_shared_memory_size()` (`src/backends/vk/device.cpp:626`). Size shared tiles for the 32 KiB floor for portability.
4. `warp_active_sum(v)` of a warp-uniform v returns 32×v — chained collectives compound this (32^k). Scale back by 1/lane_count or reduce non-uniform values.

## 6. Hardware Mapping

Every backend implements `DeviceInterface::compute_warp_size()` (`include/luisa/runtime/rhi/device_interface.h:116`), exposed as `Device::compute_warp_size()` (`include/luisa/runtime/device.h:128`). Each one's source of truth:

| Backend | Terminology | How `compute_warp_size()` is obtained | Anchored value / behaviour |
|---|---|---|---|
| CUDA | warp | `return 32u;` | `src/backends/cuda/cuda_device.h:199`; `create_shader` hard-rejects any other pinned size ("CUDA backend only support warp size 32.", `src/backends/cuda/cuda_device.cpp:949`) |
| HIP | wavefront | `hipDeviceGetAttribute(hipDeviceAttributeWarpSize)` | `src/backends/hip/hip_device.cpp:1015` (queried, not assumed) |
| Vulkan | subgroup | `VkPhysicalDeviceSubgroupProperties.subgroupSize` | `src/backends/vk/device.cpp:617`; a pinned size becomes `requiredSubgroupSize` and errors if subgroup-size control is unavailable (`src/backends/vk/compute_shader.cpp:69-85`) |
| DirectX | wave | D3D12 `WaveLaneCountMax` (`feature_check.wave_lane_count_max()`) | `src/backends/dx/DXApi/LCDevice.cpp:935` → `src/backends/dx/DXRuntime/Device.cpp:442`, `src/backends/dx/Resource/FeatureCheck.cpp:44` — note this is the **maximum** wave width the device supports, not the width a given dispatch uses; pin with `set_warp_size`, which emits `[WaveSize(n)]` and raises the target to shader model 6.6 (`src/backends/common/hlsl/codegen_utils/entry_points.cpp:819`, `src/backends/dx/DXApi/LCDevice.cpp:417`) |
| Metal / Metal4 | SIMD-group | `[MTLDevice threadExecutionWidth]` | `src/backends/metal/metal_device.cpp:210`, `src/backends/metal4/metal_device.cpp:373`; both reject a pinned size other than 32 (`metal_device.cpp:391`, `metal4/metal_device.cpp:568`) |
| SIMD (CPU) | configured lane width | device-created `_warp_width` | `src/backends/simd/runtime/simd_device.cpp:181`; a mismatching `set_warp_size` errors (`src/backends/simd/runtime/simd_shader.cpp:297`) |
| Fallback (CPU) | — | `return 1;` | `src/backends/fallback/fallback_device.cpp:142`; warp collectives degenerate to one lane |

Never hardcode a width. Query `device.compute_warp_size()` on the host and `warp_lane_count()` on the device (it lowers to the *hardware* lane count: `WaveGetLaneCount()` in HLSL, `spv::BuiltIn::SubgroupSize` in SPIR-V, CUDA's `warpSize` — `src/backends/common/hlsl/codegen_utils/variable.cpp:72`, `src/backends/common/spirv/spirv_codegen/emit.cpp:647`, `src/backends/cuda/cuda_builtin/cuda_device_resource.h:2691`). `set_warp_size(n)` is a *request* the backend can reject, not a guarantee.

---

## 7. Rules of Thumb

1. **Prefer warp collectives over shared memory.** `warp_active_sum`, `warp_active_max`, `warp_prefix_sum` lower to native group operations: one DX `WaveActive*` / `WavePrefix*` intrinsic each (`src/backends/common/hlsl/codegen_utils/function_codegen.cpp:1924-1975`), one SPIR-V `OpGroupNonUniform*` (`…FAdd`/`…IAdd` at `src/backends/common/spirv/spirv_codegen/instruction.cpp:5010`, `ExclusiveScan` at `:4869`), but on CUDA they are built in the builtin header — a 5-step `__shfl_xor_sync` butterfly for the reductions (`src/backends/cuda/cuda_builtin/cuda_device_resource.h:2772-2780`) and a `__shfl_sync` + 5-step `__shfl_up_sync` chain for the prefixes (`:3117-3129`), except integer min/max/sum which use `__reduce_*_sync` on SM 8.0+ (`:2981-2997`). Either way: no barrier needed.

2. **Set warp size explicitly** when using warp collectives: `set_warp_size(device.compute_warp_size())` inside the kernel lambda (see section 6 for which sizes a backend will accept).

3. **Don't mix warp and block assumptions.** `warp_active_sum` only reduces within the current warp. If you have multiple warps per block, use a two-level reduction (warp → shared → block).

4. **`sync_block()` is NOT needed between warp collectives** within the same warp. Warp ops are guaranteed complete for the calling lane immediately.

5. **Divergence matters.** Lanes that don't execute the warp collective call are excluded. Use this for conditional participation (section 3.5).

6. **`warp_prefix_sum` is exclusive** (not inclusive). Lane 0 always gets 0 (for sum) or 1 (for product).

7. **Vector types work.** The reductions and prefixes (`warp_active_sum/product/min/max`, `warp_prefix_sum/product`, `warp_active_all_equal`, `warp_read_lane`, `warp_read_first_active_lane`) accept `float2`, `float3`, `float4`, `int2`, … and apply component-wise (CUDA expands them per component via the `LC_WARP_ACTIVE_REDUCE_VECTOR2/3/4` and `LC_WARP_PREFIX_REDUCE_VECTOR2/3/4` macros in `src/backends/cuda/cuda_builtin/cuda_device_resource.h:3059-3103,3145`), and the lane index of `warp_read_lane` must be an integral expr. `warp_read_lane` / `warp_read_first_active_lane` also accept **matrix** operands. The vote/ballot ops (`warp_active_all/any/count_bits/prefix_count_bits/bit_mask`) take a **scalar** `Expr<bool>` only.

8. **Logic warp size.** You can logically group lanes (e.g. 4 groups of 8 within a 32-lane warp) using `lane % kGroupLanes` and `lane / kGroupLanes` arithmetic. Use `warp_read_lane` to communicate across groups.

9. **Use shared memory for block-wide cooperation** (section 4). When cooperation exceeds one warp, or you need privatization/staging/arbitrary cross-thread indexing, `Shared<T>` beats warp collectives. Always `set_block_size` and size the array to the block.

10. **Privatize global atomics into shared memory** (section 4.3). Aggregate per-thread contributions with cheap shared atomics, then issue **one** global atomic per block. This is the key stream-compaction / queue-append win.

11. **Shared memory needs `sync_block()`; warp collectives do not.** Barrier after filling shared and between the read/write-back phases of a tree reduction (section 4.4). Reduce into a register first, then write back after the barrier to avoid RAW/WAR hazards.

12. **Replace lane-local shared memory with warp intrinsics (section 3.7).** Ballot → `warp_active_bit_mask(p).x`, broadcast/shuffle → `warp_read_lane`, scan → `warp_prefix_sum`. Only valid when every shared access is intra-warp and lane-indexed; cross-warp or arbitrary indexing must stay in `Shared<T>`.

13. **All lanes must execute a shuffle; clamp the lane index.** Put `warp_read_lane` outside the divergent `$if` that consumes it, and clamp with `min(lane + k, warp_lane_count() - 1u)` at the boundary.

14. **Scale block size by packing one warp per item (section 3.8).** After warp-only refactors, enlarge `set_block_size` to 64/128 and map `item_idx = block_id().x * warps_per_block + thread_x() / warp_lane_count()`; pin the warp size so the mapping is exact.

---

## 8. Host-Side Command Batching with CommandList

GPU kernel optimization (sections 1–7) focuses on what happens *inside* a single kernel dispatch. Equally important is how you *submit* work from the host: every `stream << command` call adds driver overhead. When a per-frame or per-iteration hot loop issues many small stream submissions (upload, dispatch A, dispatch B, download, ...), the accumulated driver cost can become a bottleneck, especially on D3D12 and Vulkan where command submission is not free.

### 8.1 The Pattern

`CommandList` lets you batch multiple commands into a single submission. Commands are recorded into a `CommandList` object, then committed to the stream in one shot:

```cpp
auto cmdlist = CommandList::create();   // static CommandList create(size_t reserved_cmds, size_t reserved_cbs)
cmdlist << buffer.view().copy_from(luisa::span{upload_data})
        << kernel_a(dst.view(), src.view(), scale).dispatch(n)
        << kernel_b(dst.view(), src.view(), scale).dispatch(n)
        << buffer.view().copy_to(luisa::span{download_data});
stream << cmdlist.commit() << synchronize();   // Commit operator<<(CommandList::Commit&&)
```

Ordering is preserved for everything that *matters*: `commit()` keeps record order, and the Vulkan/DX backends then run a command-reorder pass that groups consecutive commands whose resource accesses do not alias into one barrier-free layer, so independent commands may overlap instead of being serialized by a barrier between every pair (`include/luisa/backends/ext/command_reorder_ext.h:7-14`, `src/backends/common/command_reorder_visitor.h:314-334`). Alias hazards (WAW/RAW/WAR) on overlapping ranges force separate layers, so a producer→consumer chain still runs in the recorded order. Do not assume anything about the *relative timing of unrelated* commands, and do not assume strict FIFO either — see 8.9 for the switch.

### 8.2 When to Use

- **Hot loops**: Rendering loops, training iterations, or per-frame update loops that submit several commands each iteration.
- **Dependent pipeline stages**: Commands with producer-consumer relationships (e.g., kernel A writes a buffer, kernel B reads it) that can be submitted together because GPU ordering guarantees correct sequencing.
- **Upload → compute → download chains**: Batched uploads followed by multiple kernels then final downloads, all in one commit.

### 8.3 When NOT to Use

- **Interactive latency-sensitive paths**: If a command produces results that need immediate host feedback (e.g., debug readbacks), avoid delaying it behind unrelated work.
- **Cross-stream synchronization**: Commands in different streams (COMPUTE vs GRAPHICS) must use events for ordering; a single `CommandList` cannot span multiple streams.
- **Very long command sequences**: Extremely large command lists may starve the GPU if they take too long to record; split into chunks if recording itself becomes a bottleneck.

### 8.4 General Recipe

1. **Identify the hot loop** — look for repeated `stream <<` statements inside a loop or per-frame function.
2. **Group dependent commands** — all commands that form an in-order GPU pipeline (upload → kernel A → kernel B → download) belong in the same `CommandList`.
3. **Create and fill** — call `CommandList::create()` once at the start of the group, then append commands with `<<` (`operator<<(luisa::unique_ptr<Command>&&)`) or `append()`; merge a sub-list with `add_range()` / `operator<<(CommandList&&)`.
4. **Commit once** — `stream << cmdlist.commit()` submits the batch; follow with a single `synchronize()` if host-readback is needed.
5. **Verify correctness** — ensure the sequence of operations inside the CommandList matches the dependency order (each command sees the effect of earlier ones).

### 8.5 Performance Impact

Batching N separate `stream << cmd` submissions into one `CommandList` reduces:
- **Driver submission overhead**: Each stream submission incurs a kernel transition / command-queue flush cost. With CommandList, that cost is paid once per batch.
- **Host-device synchronization points**: A single `commit() + synchronize()` replaces N pairs of `stream << ... << synchronize()`.

In practice, replacing 5+ stream submissions per iteration with a single `CommandList::create()` → `commit()` can yield measurable wall-clock speedups in offline rendering or training-data export loops, where the CPU-side submission overhead is a meaningful fraction of the iteration time. The repo has no benchmark for that ratio; the closest measured harness is `src/tests/benchmark/benchmark_command_reorder_host.cpp`, which times **pure host submission cost** of a batch (`usage: benchmark_command_reorder_host <backend:dx|vk> [mode] [dispatches] [batches] [threads] [rounds] [verbose]`) and shows how that cost scales with the number of hazard layers, not with the number of commands — so a batch that merges into one layer is cheap to submit, while a same-range WAW batch pays a barrier per command.

### 8.6 Comparison with Other Optimizations

| Optimization | Scope | Impact |
|---|---|---|
| Warp collectives (section 3) | GPU kernel — replaces shared memory and atomics | Reduces latency/contention within a warp |
| Shared memory privatization (section 4.3) | GPU kernel — aggregates block atomics | Reduces global atomic contention from O(block_size) to O(1) per block |
| **CommandList batching** (this section) | **Host submission — batches stream commands** | **Reduces driver overhead from O(N) to O(1) per iteration** |

CommandList batching is orthogonal to kernel-level optimizations. Apply both: optimize the kernel with warp/shared-memory techniques, then batch the host submissions for maximum throughput.

### 8.7 Key Rules

1. **One CommandList, one commit, one sync.** Create a single `CommandList` for a group of dependent commands, commit it once, and synchronize once rather than submitting each command separately.
2. **Record order is the dependency order.** No explicit barriers between kernel dispatches and buffer copies inside one CommandList; hazards are what the reorder pass uses to insert layer boundaries. When commands touch disjoint resources they may overlap — that is the point of the pass, not a bug.
3. **Don't reuse a committed CommandList.** `commit()` moves the list into the returned `Commit` (`src/runtime/command_list.cpp:92`), leaving the source empty; build a fresh one for the next batch. Destructing a non-empty, uncommitted list asserts (`src/runtime/command_list.cpp:9`).
4. **Prefer CommandList over chaining on `stream <<`.** Batched submission is more efficient than long chains of `stream << a << b << c << synchronize()` because it reduces internal queue flushes.
5. **Combine with kernel optimization.** Host-side batching and kernel-level warp/shared-memory optimization are complementary — use both.

### 8.8 Async Callbacks — Replacing `synchronize()` with Non-Blocking Completion

`CommandList` provides two callback hooks that decouple host work from GPU execution:

```cpp
// Runs AFTER all GPU commands in this CommandList finish.
// Signature is add_callback(luisa::move_only_function<void()> &&) — a nullary
// move-only callable, not a variadic one.
cmdlist.add_callback([/* by-value captures */]() noexcept {
    // GPU work is done — safe to read back buffers, write files, etc.
});

// Runs when the recorded CommandList is DESTROYED (~CommandList), i.e. right
// after the submission consumed it — still on the host, still before completion.
cmdlist.add_dtor_callback([/* by-value captures */]() noexcept {
    // Host-side cleanup of resources no longer needed by the recorded commands.
});
```

Both take `luisa::move_only_function<void()> &&` (`include/luisa/runtime/command_list.h:47,49`).

#### 8.8.1 Understanding the Two Callbacks

| Callback | When it fires | GPU status | Typical use |
|---|---|---|---|
| `add_dtor_callback` | From `~CommandList` (`src/runtime/command_list.cpp:8-15`) — after `commit()` has moved the list into the `Commit` and the backend has consumed it | Commands already recorded/submitted; **completion not implied** | Release temporary host buffers, close files, or free staging memory that was only needed to construct the commands (see `fill_buffer`'s use in `include/luisa/runtime/builtin_kernel.h:59`). |
| `add_callback` | After all GPU commands in the list have completed (backend completion/worker thread — e.g. `Stream::_thd` on Vulkan, `CommandQueue::_execute_thread` on DX) | **Done** | Read back downloaded buffers, write output files, signal host work queues, or launch dependent host tasks. |

**Critical difference:** `add_dtor_callback` is a destruction hook, not a completion callback — it gives no ordering guarantee against GPU execution. Only `add_callback` guarantees GPU work is finished. Note also that dtor callbacks travel with a *move-constructed* `CommandList` (`src/runtime/command_list.cpp:103`) but are **not** transferred by `add_range()` / `operator<<(CommandList&&)` — that path moves only commands, callbacks and presents and then `clear()`s the source, dropping its dtor callbacks (`src/runtime/command_list.cpp:44-58,27`).

#### 8.8.2 Avoiding `synchronize()` Stalls

Without callbacks, host→GPU data exchange typically looks like:

```cpp
// Blocking pattern: host stalls until GPU finishes.
stream << cmdlist.commit() << synchronize();  // host waits here
process_results(host_buffer);                  // then processes
```

`add_callback` lets you flip this into a non-blocking, continuation-passing style:

```cpp
// Non-blocking pattern: callback processes results when GPU is done.
cmdlist.add_callback([host_buffer = std::move(host_buf)]() noexcept {
    process_results(host_buffer);  // runs after GPU finishes
});
stream << cmdlist.commit();  // host returns immediately, no stall
// Host can begin preparing the next frame/iteration NOW...
```

This is most impactful when:
- The host has independent work (e.g., preparing the next config, loading assets, updating UI).
- The GPU work is long enough that blocking would waste host cycles.
- You process results per-iteration and can pipeline iterations (iteration N's callback runs while iteration N+1's GPU work is already in flight).

#### 8.8.3 The Pipelined Iteration Pattern

The classic pattern for hiding latency: overlap GPU execution of iteration N+1 with host processing of iteration N's results.

```cpp
// Host-side buffers must outlive the GPU work.
// Use double-buffering or shared ownership (e.g., shared_ptr).
for (int i = 0; i < num_iterations; i++) {
    auto host_buf = std::make_shared<luisa::vector<float>>(size);
    auto cmdlist = CommandList::create();

    // Record GPU commands (upload, dispatch, download into *host_buf)
    cmdlist << src.copy_from(luisa::span{src_data})
            << kernel(dst.view(), src.view(), scale).dispatch(n)
            << dst.view().copy_to(luisa::span{*host_buf});

    // Install the completion callback — captures host_buf by shared_ptr
    cmdlist.add_callback([host_buf, i]() noexcept {
        // GPU done: safely read host_buf, write to disk, etc.
        save_result(*host_buf, i);
    });

    // Submit without blocking — host returns immediately
    stream << cmdlist.commit();
    // Host can prepare iteration i+1's work right away...
}
// Final sync: wait for the very last iteration to finish.
stream << synchronize();
```

With this pattern:
- **No `synchronize()` per iteration** — only one final sync at the very end.
- **CPU and GPU overlap** — iteration N's result processing runs concurrently with iteration N+1's GPU execution.
- **Throughput improves** by the cost of one `synchronize()` stall × (N−1) iterations.

#### 8.8.4 Capturing Resources for Callbacks

Lambdas passed to `add_callback` / `add_dtor_callback` must own their captured resources because the callback outlives the `CommandList` object. Use:

```cpp
// ✅ Shared ownership (recommended for buffers)
auto data = std::make_shared<luisa::vector<float>>(size);
cmdlist.add_callback([data]() noexcept { /* safe */ });

// ✅ Move semantics for unique resources
auto owned = std::make_unique<luisa::vector<float>>(size);
cmdlist.add_callback([data = std::move(owned)]() noexcept { /* safe */ });

// ❌ Capturing raw pointers or references to stack/local variables is UAF.
float *raw = ...;
cmdlist.add_callback([raw]() noexcept { /* DANGER: raw may be dangling */ });
```

`add_dtor_callback` has the same ownership rules — it runs from `~CommandList`, which is after the submitting expression has finished, so stack variables captured by reference would be invalid.

#### 8.8.5 When to Use Which

| Situation | Use |
|---|---|
| Read back GPU results and save/process them | `add_callback` |
| Free host staging buffers after upload | `add_dtor_callback` |
| Close files or decrement refcounts after submission | `add_dtor_callback` |
| Signal a host work queue that GPU output is ready | `add_callback` |
| Launch the next iteration's host prep work | Just place after `commit()` on the host (no callback needed) |

#### 8.8.6 Key Rules

1. **`add_callback` fires after GPU completion** — it is the non-blocking replacement for `synchronize()`.
2. **`add_dtor_callback` is a destruction hook** — it carries no GPU-ordering guarantee at all; use it only for host-side cleanup, never for reading back GPU results.
3. **Always capture by value** (shared_ptr, unique_ptr, or copy). Raw pointers and references to stack variables are dangling by the time the callback runs.
4. **One final `synchronize()` is still needed** at the end of a pipeline to ensure the last iteration's callbacks have fired before the program exits.
5. **Callbacks execute on an internal worker thread** — they should not throw, block on the GPU, or perform GPU API calls on the same stream.

### 8.9 Command Reordering (the contract behind 8.1)

The Vulkan and DirectX backends run a reorder pass over every batch. It is on by default and queryable/toggleable at runtime:

```cpp
#include <luisa/backends/ext/command_reorder_ext.h>
if (auto *ro = device.extension<CommandReorderExt>()) {
    ro->set_command_reorder_enabled(false);  // strict submission order (A/B baseline)
}
// process-wide kill switch, cannot be re-enabled from code:
//   LUISA_DISABLE_COMMAND_REORDER=1
```

Seeded from `VulkanDeviceConfigExt::enable_command_reorder()` / `DirectXDeviceConfigExt::EnableCommandReorder()` (`include/luisa/backends/ext/vk_config_ext.h:104`, `include/luisa/backends/ext/dx_config_ext.h:67`), sampled when a backend starts a batch (a change applies to the next submission only) — `include/luisa/backends/ext/command_reorder_ext.h:16-27`, `src/backends/common/command_reorder_switch.h`. Related harnesses/tests: `src/tests/benchmark/benchmark_command_reorder.cpp`, `src/tests/benchmark/benchmark_command_reorder_host.cpp`, `src/tests/unit/ext/test_command_reorder_ranges.cpp`, `src/tests/unit/ext/test_command_reorder_bindless.cpp`.

---

## 9. C++ Branch-Prediction Hints: `[[likely]]` / `[[unlikely]]`

The engine compiles with C++20 (`lc_cxx_standard`, default `cxx20`), so the `[[likely]]` / `[[unlikely]]` attributes are available in every translation unit. They are **hints, never semantic changes**: they bias branch layout, inlining, and code generation toward the annotated direction. Apply them to native C++ `if`/`else` branches whose runtime direction is strongly skewed — hot serialization/deserialization, validation, decode, and dispatch paths. This is a generic rule applied per branch; do not copy annotations from one file to another without re-checking that branch's own frequency.

### 9.1 Syntax (codebase convention)

Place the attribute **after the condition (or the `else` keyword), before the branch body**:

```cpp
if (cond) [[unlikely]] { return false; }        // exceptional / error path
if (ptr != nullptr) [[likely]] { use(ptr); }    // common fast path
if (a) { ... } else [[likely]] { ... }          // also valid on the else side
```

### 9.2 Generic decision rule

Classify each branch by *how often it runs at runtime*, then hint accordingly:

- `[[unlikely]]` — the branch that almost never runs:
  - error / validation failures (malformed input, out-of-range enum or index, size mismatch, not-found);
  - early-return guards (`return false` / `return nullptr` / `return error`);
  - boundary / sentinel cases that only trigger at the edge of a loop or data range.
- `[[likely]]` — the branch that almost always runs:
  - the common path of a hot `if`/`else` (e.g. data is present: `ptr != nullptr`, presence flags set);
  - default setup that usually applies (e.g. filling in default views/offsets);
  - the non-empty case in guarded bulk operations (e.g. `if (n != 0u)` around `memcpy`).

### 9.3 When NOT to hint

1. **Ambiguous frequency** — both sides run often (general lookups, formatting separators, balanced `if`/`else`). A wrong hint misleads the optimizer; leave the branch unannotated.
2. **Cold code** — debug/describe/formatting helpers where the hint cannot affect a measurable hot path.
3. **DSL branches** — `$if` / `$else` inside kernels are LuisaCompute expression-building macros (see `include/luisa/dsl/sugar.h`), not native C++ statements; these attributes do not apply to them. Optimize kernel control flow with the warp/shared-memory techniques in sections 1–4 instead.
4. **Balanced branches** — never annotate when the split is close to 50/50; the hint is a promise about frequency, not a preference.

### 9.4 Workflow

1. Identify frequently-executed functions (serialize/deserialize, validators, decode, hot dispatch loops).
2. For each `if`/`else`, ask: *“which side runs almost always / almost never?”*
3. Annotate only branches with a clear answer; leave the rest untouched.
4. Verify syntax with the project checker: `python scripts/check_cpp_syntax.py <file>` (C++20 is the default; these attributes compile on all supported toolchains).

---

## 10. Codegen / Kernel-Lowering-Time Optimizations

Sections 1–9 optimize hand-written kernels. When you instead write or optimize **kernel-generating code** (a lowering pass, a DSL emitter, or host code that builds kernels from an IR), a different class of opportunities appears: the generator runs on the host and knows things the device compiler cannot (compile-time extents, divisibility, launch shape). Exploit that knowledge; the device binary should contain only work that is genuinely runtime-dependent.

Before adding a transformation of your own, check whether an existing XIR pass already covers it (`src/xir/passes/`, headers in `include/luisa/xir/passes/`):

| Recipe below | Real XIR pass / analysis | Test coverage |
|---|---|---|
| 10.2 defer stores | `dead_store_elimination`, `local_store_forward`, `defer_local_aggregate_load` | `src/tests/unit/xir/test_xir_pass_defer_local_aggregate_load.cpp` |
| 10.3 guard/branch hoisting | `loop_unswitch` (invariant-condition cloning; `include/luisa/xir/passes/loop_unswitch.h`) | `src/tests/unit/xir/test_xir_passes.cpp` (also `src/tests/unit/simd/test_llvm_schedule_codegen.cpp`) |
| 10.4 batched loads/stores | `slp_vectorization`, `fuse_consecutive_buffer_reads` | `src/tests/unit/runtime/test_dsl_slp_vectorization.cpp`, `test_dsl_fuse_buffer_reads.cpp` |
| 10.6 address strength reduction | `indvar_simplify` ("Strength reduction is plain-CFG-only", `include/luisa/xir/passes/indvar_simplify.h`) | `src/tests/unit/runtime/test_dsl_indvar_strength_reduce.cpp`, `src/tests/unit/xir/test_xir_pass_indvar_simplify.cpp` |
| 10.7 / hoisting & const-folding | `licm`, `early_cse`, `sccp`, `const_fold` | `src/tests/unit/xir/test_xir_pass_licm.cpp`, `test_xir_pass_early_cse.cpp`, `test_xir_pass_sccp.cpp` |

Caveat from `src/xir/passes/pass_pipeline.cpp:280-282`: the structured loop transforms (`loop_fusion`, `loop_rotation`, `loop_vectorization`) are deliberately **excluded from the default pipelines** — treat them as opt-in (`test_xir_pass_loop_rotation.cpp`, `test_xir_pass_loop_fusion.cpp`, `test_xir_pass_loop_vectorization.cpp` exercise them directly), not as something that will clean up after the emitter. There is no loop-unroll pass and no `lower_switch` pass in this repo; if you need unrolling the generator must do it (10.6), and a `SwitchInst` reaches raw-CFG `IndexedBranchInst` only through `destructure_cfg`.

### 10.1 Audit for replicated work first

The most pathological slowdowns come from **each thread redundantly doing the whole tile's work** instead of a partitioned slice:

- Any per-thread data structure (register array, local array) that is *materialized in full by every thread* but only *consumed per-element* multiplies cost by the block size. Either partition the producer loop across threads, or back the structure with a single block-shared array so each element is computed once.
- A useful smell test: if a loop's trip count is the *tile size* but the block has *many threads*, ask who consumes each element. If consumers only read their own slice, the producer must be partitioned too.
- Eliminating replicated work is routinely worth more than every arithmetic micro-optimization combined (order-of-magnitude, not percent).

### 10.2 Defer stores: lazy expression evaluation

When the IR describes "tile B = f(tile A)" and B is only read elementwise later, the generator can **record the expression and re-evaluate it at each read site** instead of materializing B. Rules:

- Only safe when the value is a pure function of its inputs (no intervening mutation) and the read pattern matches the producer's indexing.
- Keep a **re-entrancy guard**: self-referential statements (`x = f(x)`) must read the *old* materialized value, not recurse into the lazy one.
- **Invalidate** the lazy entry on every later mutation of the same tile (clear, fill, copy-into, accumulate). Miss one invalidation site and you get silent stale data — enumerate every writer before enabling laziness.

### 10.3 Full/tail loop splitting and guard elision

Partitioned loops usually carry a bounds guard (`if (idx < total)`) that is false in at most the last iteration. The generator should:

- Emit **unguarded full chunks** plus **one guarded tail chunk**; the tail is a uniform branch, not a per-element predicate.
- When extents and block/warp size are host-known, prove `total % stride == 0` at generation time and **omit the tail entirely** — same for per-lane guards in warp-strided loops (`k < K` vanishes when `K % lanes == 0`).
- Apply the same elision to identity-initialization of accumulators: if the guard is gone, the "else: identity" path is gone too.
- Never elide based on the *runtime* warp size unless it was pinned with `set_warp_size`; eliding against a fixed host constant is only sound because warp sizes are powers of two within the DSL's contract (`luisa_compute_validate_warp_size` accepts 1, 2, 4, 8, 16, 32, 64, 128 only — `src/dsl/builtin.cpp:10-16`).

### 10.4 Memory-level parallelism in generated copy loops

A naive generated copy loop alternates dependent `global load → shared store` per element, so each element pays full memory latency and the loop is latency-bound. Restructure into **batched chunks**: load K elements into locals (K independent loads in flight), then store all K. Choose K = 4–8; partition the chunk grid across threads (guard when `chunk_count % threads != 0`). This is the single biggest fix for copy-dominated kernels and applies to any generated gather/scatter loop, not just copies.

### 10.5 Accumulation into per-thread state across a device loop

When a device `$for` loop body accumulates into per-thread registers, remember the body is **emitted once on the host**: any storage decision (local array vs shared backing vs lazy value) is baked at emission time and applies to every iteration. Consequences:

- Tricks that change *where* a value lives between iterations (e.g. publish to shared, read back next iteration) cannot be expressed by rebinding host-side handles — they silently bind to the emission-time storage and lose accumulation.
- If per-thread register replicas force an expensive re-synchronization per loop iteration (broadcast + refill), consider **promoting the accumulator to shared-backed storage** so the loop body reads/writes each element exactly once. Weigh the added shared-memory budget against the eliminated per-iteration round trip.

### 10.6 Strength-reduce generated address math

Per-element index computation (div/mod decompositions, stride multiplies) is invisible in the source but dominates generated inner loops:

- Decompose multi-dimensional coordinates **once per thread** (from the linear thread id), then stride both axes — never per element.
- Hoist loop-invariant base addresses (block offsets, row bases) out of the `$for`; a codegen bug where an outer-loop induction variable leaks into a hoisted offset silently scatters all writes.
- Unroll inner loops by a small host-known factor (`k_pack` = 4–8) when the trip count divides evenly; unrolled iterations need no guards. There is **no loop-unroll XIR pass** in `src/xir/passes` (unrolling happens only inside `autodiff.cpp` for fixed-trip loops), so this must be done at emission time — and there is no `lower_switch` pass either; `SwitchInst` is lowered to raw-CFG `IndexedBranchInst` by `destructure_cfg`.

### 10.7 Barriers in generated code

- Keep the conservative rule: `sync_block()` after shared stores before any cross-thread read, and **never inside a `thread_id`-divergent branch**.
- Elision is only safe with a precise hazard analysis (which statements write shared, which read it, who publishes what); historically such elision yields ~nothing in single-warp blocks where barriers are nearly free — prefer correctness.
- In single-warp blocks, warp collectives replace shared staging entirely and remove the barrier question (section 3.7).

### 10.8 Host emission hygiene

- `[[likely]]`/`[[unlikely]]` on the *host-side* emission branches (section 9) is free and appropriate: common shapes (fragment dest, rank-2 tile) are strongly skewed.
- Small per-dispatch kernels (< 0.2 ms) are dominated by fixed launch overheads and measurement noise (±50% run-to-run is normal); evaluate them with min-of-N runs and distrust single-run deltas. Large kernels are stable within ~5%.
- A structural fix (replication removal, MLP batching, guard elision) beats micro-tuning of tile sizes; once the structure is right, tile-size parameters usually stop mattering.

### 10.9 Verification discipline for lowering changes

- Always gate on an **end-to-end correctness check against a host reference**, per kernel, after every change — lowering bugs produce plausible-but-wrong numbers far more often than crashes.
- Exercise **ragged extents** (sizes not divisible by the block/warp/segment size): guard-elision and segment-bound bugs hide behind evenly-dividing test sizes. Ragged segment boundaries are a classic source of cross-thread races (one thread writes into a neighbor's region).
- Revert-fast: keep each transformation independently toggleable during development; a change that regresses or breaks one kernel shape gets reverted, not patched over.

---

# PART II — Cross-Platform GPU Kernel Optimization Experience

Sections 1–10 are anchored in the LuisaCompute repository. Sections 11 onward collect general, vendor-documented and measured GPU-kernel optimization experience across NVIDIA (Ampere/Hopper/Blackwell), AMD (GCN/RDNA/CDNA), Apple Metal, Intel Xe, and the Vulkan/DX12 abstraction layers. Claims that come from external sources carry an inline link; treat per-device numbers as priors to re-measure with the block-size sweep harness in §5.7.

## 11. Roofline First: Diagnose Before Optimizing

### 11.1 The Model

The roofline model bounds attainable throughput by `P = min(peak_FLOPS, AI × peak_bandwidth)`, where arithmetic intensity **AI = FLOPs / bytes moved from the memory level you care about** (always state the level: HBM, L2, L1) ([NERSC](https://docs.nersc.gov/tools/performance/roofline/), [ai-infrastructure.net](https://ai-infrastructure.net/roofline-arithmetic-intensity/)). The ridge point (machine balance) `I_ridge = peak_FLOPS / peak_bandwidth` separates memory-bound from compute-bound kernels. Compute it from datasheet numbers before touching code:

| GPU | Dense BF16/FP16 peak | HBM bandwidth | Ridge (FLOP/byte) |
|---|---|---|---|
| A100 80GB | 312 TFLOPS | 2.04 TB/s | ≈ 153 |
| H100 SXM | 989 TFLOPS | 3.35 TB/s | ≈ 295 |
| H200 | 989 TFLOPS | 4.8 TB/s | ≈ 206 |
| B200 | 2,250 TFLOPS | 8.0 TB/s | ≈ 281 |
| MI300X | 1,307 TFLOPS | 5.3 TB/s | ≈ 247 |
| RTX 4080 SUPER (FP32) | 52.2 TFLOPS | 736 GB/s | ≈ 71 |

The RTX 4080 SUPER row is locally verified (§24): a scalar saxpy (AI = 1/6 FLOP/byte ≪ 71) measured 636–643 GB/s ≈ 87% of the bandwidth roof while implying only ~106 GFLOPS ≈ 0.2% of the FP32 roof — exactly the memory-bound side the ridge predicts. Note consumer cards have far lower tensor peaks than the datacenter parts above, so their *tensor* ridge is much higher; the FP32 row is the honest comparison for non-tensor kernels.

([AI Infra Interviews](https://aiinfrainterviews.com/q/arithmetic-intensity-and-the-roofline); H100/B200 numbers also in [NextPCB](https://www.nextpcb.com/blog/nvidia-blackwell-architecture-b200-gb200-pcb-design) and the [Blackwell microbenchmarking paper](https://arxiv.org/html/2512.02189v2)). Note the ridge **creeps up across generations** — FLOPS grow faster than bandwidth — so a kernel balanced on A100 is memory-bound on H100, and more kernels land memory-bound every generation ([ai-infrastructure.net](https://ai-infrastructure.net/roofline-arithmetic-intensity/)).

### 11.2 The Two Levers

When a kernel is memory-bound (AI < ridge), only two things help:

1. **Move fewer bytes for the same FLOPs** — kernel fusion (read/write the activation once instead of per-op), lower-precision storage (FP8/BF16 weights halve bytes *and* double the tensor-core peak), avoiding materialized intermediates (the FlashAttention trick, §17).
2. **Reuse each byte more** — shared-memory/register tiling in GEMM (a tile loaded once serves `TILE` multiply-adds, raising GEMM AI to ≈ `TILE/4` FLOP/B ([riteshyadav.xyz](https://www.riteshyadav.xyz/notes/device-software/tiled-matrix-multiplication))), batching (a GEMV at batch 64 reads the same weights as batch 1 for 64× the FLOPs), and cache residency.

Optimizing compute on a memory-bound kernel (or chasing bandwidth on a compute-bound one) is wasted effort: "a team three weeks into optimizing a kernel's arithmetic, whose intensity was 2 FLOP/byte against a ridge of 295, was never going to move" ([AI Infra Interviews](https://aiinfrainterviews.com/q/arithmetic-intensity-and-the-roofline)).

### 11.3 When the Roofline Does Not Apply

Roofline assumes steady-state throughput. It does **not** describe:

- **Latency-bound kernels** — high occupancy, but neither memory nor compute near peak: warps stall on dependency chains. Fix with ILP (multiple accumulators, unrolling), not more occupancy (§12, §15).
- **Launch/schedule-bound workloads** — many tiny kernels where CPU dispatch overhead exceeds GPU execution (§20).
- A kernel can sit at 100% occupancy and still be memory-bound — occupancy hides latency, it does not raise AI ([ai-infrastructure.net](https://ai-infrastructure.net/roofline-arithmetic-intensity/)).

Nsight Compute's "Memory Throughput %" and "Compute (SM) Throughput %" per kernel is the roofline restated one kernel at a time; the hierarchical roofline (L1/L2/HBM levels) tells you which cache level your working set actually lives in ([NERSC ISC2019 tutorial](https://www.nersc.gov/assets/Uploads/Tutorial-ISC2019-Intro.pdf)).

## 12. Profiler-Guided Diagnosis: The Warp-Stall Taxonomy

The dominant warp-stall reason routes you to the fix. Cross-vendor mapping:

| Stall (Nsight name) | Meaning | Fix | AMD analog |
|---|---|---|---|
| **Long Scoreboard** | Waiting on a global/local/texture load (L1 hit ~33 cy, L2 ~200 cy, HBM ~290–566 cy) ([Software Frontier](https://softwarefrontier.substack.com/p/mastering-cuda-and-high-performance-ebc)) | More occupancy, prefetch earlier, async copy, restructure to independent loads, improve L2 hit rate | `SQ_WAIT_INST_LDS` / VMEM waits in rocprof |
| **Short Scoreboard** | Waiting on shared-memory load or SFU/MUFU result (~16–23 cy) ([Nsight guide via zinlab](https://zinlab.github.io/tutorials/nsight-compute-mastery.html)) | Fix bank conflicts (pad/swizzle), restructure math, hide with independent work | LDS waits; check bank conflicts |
| **MIO / LG Throttle** | Memory instruction input FIFO full — too many requests in flight ([Software Frontier](https://softwarefrontier.substack.com/p/mastering-cuda-and-high-performance-ebc)) | Vectorize loads (`float4`), coalesce, reduce memory-instruction count, cut atomics | VALU/LDS queue pressure |
| **Barrier** | Warp waiting at `__syncthreads()` for slower sibling warps | Balance per-warp work, fewer barriers, smaller blocks, warp-scope sync instead | Same; check wave imbalance |
| **Math Pipe Throttle** | Execution pipe saturated | **This is the good stall** — you are compute-bound; only tensor/matrix cores or lower precision go further | VALU utilization >90% (RGP) |
| **Not Selected** | Eligible but scheduler picked another warp | Healthy sign of surplus warps; high values mean you could *lower* occupancy | n/a |
| **Drain** | Warp exiting | Kernel too short → launch-bound; fuse kernels (§20) | Same |

([zinlab](https://zinlab.github.io/tutorials/nsight-compute-mastery.html), [ai-infrastructure.net ILP](https://ai-infrastructure.net/instruction-level-parallelism-gpu/), [quant67](https://quant67.com/post/gpu-kernel/08-nsight-profiling/08-nsight-profiling.html)).

Interpretation rule ([ai-infrastructure.net](https://ai-infrastructure.net/instruction-level-parallelism-gpu)): `Long Scoreboard` high **and** `Not Selected` low **and** DRAM % low ⇒ latency-bound (add ILP/occupancy). `Not Selected` high ⇒ issue-bound/compute-bound (already fast). Throttle reasons high with DRAM near peak ⇒ bandwidth-bound (raise arithmetic intensity).

Practical workflow:
1. **Nsight Systems first** (`nsys profile --trace=cuda,nvtx ./app`) for the timeline — gaps between kernels = host overhead; then **Nsight Compute** (`ncu --set full`) on the hot kernel ([praveentn.live](https://www.praveentn.live/learn/concepts/profiling-gpu-workloads)). Compile with `-lineinfo` so stalls attribute to source lines ([ai-infrastructure.net](https://ai-infrastructure.net/instruction-level-parallelism-gpu/)).
2. Check `ptxas -v` output: **nonzero `lmem` (local-memory spill) is a silent killer** — fix via `__launch_bounds__` or restructuring before anything else ([anylearn.cc](https://anylearn.cc/lessons/cuda-profiling-and-optimization)).
3. Quick sanity thresholds ([kunwar.page](https://www.kunwar.page/chapter/039-cuda-programming-for-ml-engineers)): achieved occupancy < 30% → cut registers/SMEM; memory throughput < 50% of peak → fix coalescing/locality; long-scoreboard > 50% → latency-bound.
4. On AMD: `amdclang++ -Rpass-analysis=kernel-resource-usage` prints SGPR/VGPR/spills/LDS/occupancy per kernel at compile time; confirm against hardware with `rocprofv3 --kernel-trace` (the `LDS_Block_Size` column matches the compiler exactly) ([ROCm blogs](https://rocm.blogs.amd.com/software-tools-optimization/rdna-fp16-gemm/README.html)). Use `llvm-objdump --disassemble-symbols` on the `.co` to inspect `.vgpr_spill_count` ([ROCm blogs](https://rocm.blogs.amd.com/software-tools-optimization/register-pressure/README.html), [FlyDSL issue](https://github.com/sunway513/FlyDSL/issues/8)).
5. Optimization order of operations ([anylearn.cc](https://anylearn.cc/lessons/cuda-profiling-and-optimization)): correctness → coalescing → kill spills → shared-memory reuse → block-size tuning → divergence → stream overlap → bank conflicts/ILP/tensor cores. "Most kernels live or die on steps 2–3."

## 13. Occupancy, Wave Quantization, and Grid Sizing

### 13.1 Occupancy Is a `min()` Over Three Resources

`occupancy = resident_warps / max_warps_per_SM`, limited by the tightest of warp slots, register file, and shared memory ([sankethbk](https://sankethbk.github.io/blog/posts/gpu/2026-09-01-2-cuda-occupancy/)). Target "**enough** occupancy to cover the latency your kernel sees," not 100% — high occupancy with high ILP is often interchangeable (Volkov: ILP and TLP are substitute latency-hiding mechanisms ([ai-infrastructure.net](https://ai-infrastructure.net/instruction-level-parallelism-gpu/))). Compute-bound kernels can run fast at low occupancy; memory-bound kernels want ≥ 50% achieved occupancy ([anylearn.cc](https://anylearn.cc/lessons/cuda-profiling-and-optimization)).

Register math example: 64 regs/thread on a 65,536-register SM caps residency at 1,024 threads = 50% occupancy ([anylearn.cc](https://anylearn.cc/lessons/cuda-profiling-and-optimization)). On Blackwell/Hopper the constraint is per-SMSP (each of 4 SMSPs holds 16,384 registers) — violating it in one SMSP limits the whole SM ([Software Frontier](https://softwarefrontier.substack.com/p/mastering-cuda-and-high-performance-ebc)).

### 13.2 Theoretical vs Achieved: Quantization, Tails, Imbalance

Achieved occupancy is usually below theoretical, for three structural reasons ([aicassindra](https://aicassindra.com/blogs/gpu/gpu_occupancy.html)):

- **Wave quantization**: launching 40 blocks onto a GPU that runs 32 at a time leaves a second wave 8 blocks wide with most SMs idle. Size the grid to a multiple of the concurrent-block count, or use grid-stride loops / persistent kernels so the tail disappears.
- **Tail effects**: a block's resources are not released until its last warp retires; the end of a kernel drains toward empty. Keep per-block runtime small relative to total kernel time.
- **Load imbalance**: divergent per-block work strands some SMs. Use work-stealing (atomic ticket counter), tile schedulers, or persistent CTAs that pull tiles until exhausted.

Persistent kernels (grid = #SMs × blocks/SM, inner loop over tiles) also let you add a **grouped/swizzled tile order** for L2 locality — without it the L2 hit rate drops ([tessl.io fla skill](https://tessl.io/registry/skills/github/fla-org/flash-linear-attention/fla-triton-to-gluon)).

### 13.3 Empirical Autotuning Discipline

A six-tier optimization playbook measured in agent-driven kernel search ([arXiv 2603.21331](https://arxiv.org/html/2603.21331v1)) gives typical payoff ranges: block-size/tile sweeps **10–50%**, memory-access fixes (coalescing, prefetch, L2 swizzle, SMEM padding) **10–30%**, compute-level (TF32 accumulation, epilogue fusion, invariant hoisting) **5–15%**, advanced (split-K, persistent kernels, warp specialization) **5–20%**, architecture-specific (TMA on Hopper, cp.async on Ampere) **5–15%**. Practical rules:

- Sweep powers of two **and** rectangular tiles; change one thing at a time; re-autotune after every structural change (in one published walkthrough, the best pre-pipelining config lost >100 TFLOPS once pipelining was added ([tessl.io](https://tessl.io/registry/skills/github/fla-org/flash-linear-attention/fla-triton-to-gluon))).
- For small kernels (< 0.2 ms) use min-of-N runs — run-to-run noise is ±50% (§10.8).
- Block-size results do not transfer across architectures; a sweep on one GPU must be redone on another ([SCITEPRESS 113148](https://www.scitepress.org/PublishedPapers/2022/113148/113148.pdf)), which matches the repo's measured §5.7 findings.

## 14. Memory-Access Optimization: Coalescing, Vectorization, Caching

### 14.1 Coalescing Is Still Optimization #1

Consecutive threads → consecutive addresses. Coalesced access reaches 70–90% of peak bandwidth on AMD hardware; random access gets 5–15% ([HIP hardware docs](https://rocm.docs.amd.com/projects/HIP/en/latest/understand/hardware_implementation.html)). Prefer **SoA over AoS** layouts ([shivam2003.com](https://shivam2003.com/posts/gpu-1-2-memory)). On AMD, aim for **256-byte coalesced blocks per wave** when writing images — an 8×8 thread group writing an 8×8 pixel block; a Morton-like 2×2 swizzle matched standard texture layout and gained ~8% in AMD's mip-chain case study ([AMD GPUOpen RDNA Performance Guide](https://gpuopen.com/learn/rdna-performance-guide/), [Interplay of Light](https://interplayoflight.wordpress.com/2020/05/23/optimizing-for-the-rdna-architecture-presentation-notes/)). Cross-vendor confirmation of §5.7's "shape ≫ size" lesson: never use 1-wide thread groups for 2D data.

### 14.2 Vectorized Loads (`LDG.128` / `float4` / `int4`)

Loading `float4` (16 B/lane) makes a warp move 512 B per instruction instead of 128 B, cutting instruction issue count 4× and typically adding **15–40% throughput on memory-bound kernels** ([ai-infra-odyssey](https://github.com/lilinji/ai-infra-odyssey/blob/main/03_CUDA%E4%B8%8E%E7%AE%97%E5%AD%90%E6%80%A7%E8%83%BD%E4%BC%98%E5%8C%96/22_CUDA%E7%BC%96%E7%A8%8B%E6%A8%A1%E5%9E%8B%E4%B8%8E%E5%86%85%E5%AD%98%E5%B1%82%E6%AC%A1_Thread_Block_Coalescing.md), [arpanpathak.github.io](https://arpanpathak.github.io/gpu-parallel-book/chapters/part3_optimisation/ch07_memory_optimisation_deep_dive.html)). Rules:

- The base pointer must be **16-byte aligned** (`cudaMalloc` guarantees ≥16 B; offsets by odd element counts do not). Handle misaligned prologue/epilogue with a scalar fallback loop ([aispe.notlate.cn](https://aispe.notlate.cn/chapters/ch11/)).
- In LuisaCompute DSL terms: prefer `BufferFloat4`/`read` of vector types over four scalar reads when the layout permits; the SLP vectorization pass (`slp_vectorization`, §10) automates part of this.
- Applies equally to stores (`STG.128`) and to AMD (`float4` → `flat_load_dwordx4`).
- **Measured caveat (RTX 4080 SUPER, cuda backend, §24):** on a *DRAM-saturated* streaming saxpy, scalar coalesced `float` already reached 636–643 GB/s (~87% of the 736 GB/s peak) and `float4` was *not faster* (611–619 GB/s). The 15–40% figure materializes when the kernel is **issue/latency-bound** (MIO throttle from too many memory instructions) — not when DRAM is the wall. Check which side your kernel is on (§12) before rewriting layouts.

### 14.3 Bank Conflicts: Padding and Swizzling

Shared memory (SMEM/LDS) is banked: 32 banks of 32 bits on NVIDIA and AMD RDNA/GCN ([AMD GPUOpen](https://gpuopen.com/learn/rdna-performance-guide/)). A 32-way conflict can push one LDS access from 1 cycle to as many as 64 cycles on RDNA — slower than going to VRAM ([AMD RDNA2 ISA via yyy3d.online](https://yyy3d.online/blog/gpu-architecture-ep03-shader-core/)). Fixes:

- **Padding**: `TILE[N][N+1]` shifts each row off the previous row's bank pattern; a matrix-transpose benchmark measured 1.10 ms → 0.92 ms (~20%) from either padding or XOR swizzle on an RTX 3090 ([Spheron](https://www.spheron.network/blog/shared-memory-bank-conflicts-why-your-quantized-kernel-is-sl/)). AMD explicitly recommends SoA layout or padding to cut access strides ([GPUOpen](https://gpuopen.com/learn/rdna-performance-guide/)).
- **XOR swizzle**: `col ^ (row % k)` permutes banks without wasting SMEM — required when SMEM budget is tight (padding inflates footprint and can cost occupancy, §13).
- RDNA4 (gfx1201) widened LDS to **64 banks × 4 B**, so conflict patterns differ between wave32 and wave64 dispatches — a padding tuned for wave32 is narrower than wave64 needs ([zolotukhin.ai](https://zolotukhin.ai/blog/2026-05-11-the-wave32-commit-that-closes-rdna4-long-context-flash-attention-gap/)).
- Watch tensor-core staging layouts: `ldmatrix`/wgmma/tcgen05 operand layouts interact with bank/swizzle modes; CUTLASS's swizzle atoms exist precisely for this.

### 14.4 L2 Cache Control (CUDA)

Devices of compute capability ≥ 8.0 expose an **L2 persisting-access window**: set aside part of L2 (`cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, ...)`) and mark a global-memory region as `cudaAccessPropertyPersisting` on a stream or graph node so hot data (e.g. attention KV heads, embedding tables) survives across kernels ([CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/l2-cache-control.html)). Tune `hitRatio` below 1.0 to avoid thrashing when the window exceeds the set-aside area ([CUDA Best Practices 11.5](https://docs.nvidia.com/cuda/archive/11.5.0/pdf/CUDA_C_Best_Practices_Guide.pdf)). **Reset with `cudaAccessPropertyNormal` / `cudaCtxResetPersistingL2Cache()`** when done — persistence survives past its intended use and shrinks L2 for later kernels. Useful immediately after producer kernels in the CommandList pipelines of §8.

## 15. Async Data Movement and Software Pipelining

### 15.1 Why Pipelining

A sequential `load → sync → compute → sync` K-loop leaves the memory system idle during compute and the compute units idle during loads. **Double-buffered software pipelining** — prefetch tile `k+1` into a second SMEM buffer while computing tile `k` — is worth **1.2–1.5×** on staged kernels once residency and swizzling are in place, and "every serious GEMM implementation does this" ([ocannl issue #487](https://github.com/ahrefs/ocannl/issues/487)). **Measured caveat (RTX 4080 SUPER, cuda backend, §24):** on a staged shared-memory reduction that is *DRAM-bound* (512 MiB working set), a 2-stage `cp.async` double buffer gave **no gain** over a synchronous copy (657 vs 659 GB/s, ~90% of peak both) — thousands of resident warps already hide the latency (§13.1). When the same kernel was *L2-resident* (64 MiB), `cp.async` was **29% slower** than the synchronous path (1599 vs 2251 GB/s). The 1.2–1.5× prior applies to loops whose loads are genuinely exposed (low occupancy, dependent chains, compute-heavy stages); re-measure before relying on it. CUTLASS stage-count guidance: Ampere 3–4 stages, Hopper 4–7 (async TMA), Blackwell dynamic per problem ([CUTLASS optimization notes](https://mintlify.com/NVIDIA/cutlass/performance/optimization)).

### 15.2 Choosing the Copy Mechanism

| Mechanism | Introduced | What it does | Register cost | Notes |
|---|---|---|---|---|
| LDG into registers, STS after barrier | pre-Ampere fallback | prefetch via registers | high (staging regs) | still a win vs no pipelining |
| `cp.async` / `cuda::memcpy_async` (`__pipeline_memcpy_async`) | Ampere (sm_80) | global→SMEM **without** staging through registers | zero for the copy | L1 bypass optional; 16 B/thread copies can bypass L1 ([CUDA Best Practices](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html)) |
| TMA (`cp.async.bulk.tensor`) | Hopper (sm_90) | descriptor-driven multi-dim tile copy by **one thread**; completion via mbarrier | ~none (hardware does addressing) | also multicasts one tile into up to 16 SMs' SMEM within a cluster ([The Index](https://theindex26.com/tensor-memory-accelerator-tma-nvidia-hopper/)) |

The measured decision rule on GA104: compute/load ratio < 5:1 → prefer `cp.async` (+35% on IGEMM); 5–20:1 → implement both and benchmark; > 20:1 → warp interleaving suffices ([pipeline-gpu-kernel skill](https://www.skillsdirectory.com/skills/pjt222-pipeline-gpu-kernel-agent-almanac)). LuisaCompute exposes `async_copy` + `pipeline_commit` + `pipeline_wait_prior` (§4.6); remember **every thread must issue its own copy** and you still need `sync_block()` before consumers read.

Structure the loop as **prologue / main loop / epilogue**: prefetch stage 0 before entering the loop; in iteration `k` acquire the pipeline, issue the copy into buffer `(k+1) % S`, commit, then wait ≤ S−1 outstanding groups and compute on buffer `k % S`; drain after the loop ([jared-hpc.com](https://jared-hpc.com/ai-hardware-engineer-roadmap/Phase%205%20-%20Advanced%20Topics%20and%20Specialization/1.%20GPU%20Infrastructure/Nvidia%20GPU/HPC%20Setup/CUDA-Advanced-Optimization/05-Warp-Specialization/), [mlc.ai](https://mlc.ai/modern-gpu-programming-for-mlsys/chapter_gemm_async/index.html)).

### 15.3 Budget Check Before You Double-Buffer

Doubling (or staging) SMEM can push you over an occupancy cliff: e.g. GA104 has 100 KB SMEM/SM and a 50 KB/block cliff above which only one block resides ([pipeline-gpu-kernel skill](https://www.skillsdirectory.com/skills/pjt222-pipeline-gpu-kernel-agent-almanac)). Always recompute occupancy *after* adding stages — halving tile depth `BK` is the usual way back under the cliff. On RDNA the same trade measured: halving LDS per block from 63 KB to 31.5 KB doubled occupancy from 2 to 4 waves/SIMD in AMD's FP16 GEMM walkthrough ([ROCm blogs](https://rocm.blogs.amd.com/software-tools-optimization/rdna-fp16-gemm/README.html)).

## 16. Hopper (sm_90) Experience: wgmma, TMA, Warp Specialization

### 16.1 The Async Stack

Hopper made both data movement **and** compute asynchronous: TMA copies tiles GMEM↔SMEM under one thread's control with mbarrier completion, and `wgmma.mma_async` (warpgroup-wide, 128 threads, up to m64n256k16) reads operands **directly from SMEM** and accumulates into registers, returning immediately ([FlashAttention-3 paper](https://arxiv.org/html/2407.08608v2), [ai-infrastructure.net](https://ai-infrastructure.net/tensor-core-programming/)). wgmma latency scales linearly with tile width (32 cy at m64n64k16 → 128 cy at m64n256k16) — a temporal pipeline ([Delaware microbenchmarks](https://arxiv.org/html/2512.02189v3)).

### 16.2 Warp Specialization + `setmaxnreg`

With TMA doing addressing in hardware, a producer warp needs almost no registers. Hopper's `setmaxnreg` lets warpgroups **redistribute the SM register pool at runtime**: producer warpgroup drops to 24–56 regs/thread, consumers rise to 240–256, with the constraint `Σ(regs×128)` across warpgroups ≤ 65,536/SM (FA3's configs land on 64,512/65,536 — the entire file, to the register) ([FlashAttention-3 paper](https://arxiv.org/html/2407.08608v2), [thesatyajit.com](https://ai.thesatyajit.com/articles/flash-attention-3)). Without reallocation, combined demand of 3 warpgroups forces 1 CTA/SM; Colfax measured an FP32-accumulate kernel collapsing ~500 → ~21 TFLOPS without it ([pinghaoyang.com.cn](https://pinghaoyang.com.cn/aigc/posts/gpu-gemm-part-3/)). Rules: every warp in the warpgroup executes the same `setmaxnreg`, value is a multiple of 8 in [24, 256], and explicit warpgroup sync is required between successive adjustments ([arXiv 2604.17834](https://arxiv.org/pdf/2604.17834)).

Warp specialization is **spatial** division (different warps do different jobs) as opposed to multi-stage pipelining's **temporal** overlap; it pays only when producer and consumer genuinely differ — when all threads must cooperatively gather (indirect loads TMA cannot express), specialization loses its benefit and a single-warpgroup design with thread-0 TMA + cooperative loads wins ([arXiv 2604.17834](https://arxiv.org/pdf/2604.17834)).

### 16.3 Hiding Non-Matmul Work Under GEMMs

The H100's FP16 matmul peak (989 TFLOPS) dwarfs its special-function throughput (3.9 TFLOPS for exp), so softmax becomes the bottleneck unless overlapped ([rudrite.com FA3 notes](https://research.rudrite.com/flash-attention-3)). FA3's two schedules: **pingpong** (consumer warpgroup A's softmax overlaps warpgroup B's GEMMs, coordinated by named barriers) and **intra-warpgroup 2-stage** (softmax of iteration j overlaps wgmma of j+1) ([l1-ca0.github.io](https://l1-ca0.github.io/posts/flash-attention-3-on-hopper/)). Result: FA2's ~35% utilization on H100 (it was designed for Ampere's synchronous model) rose to 75–85%, 740 TFLOPS FP16 / ~1.2–1.3 PFLOPS FP8 ([pinghaoyang.com.cn](https://pinghaoyang.com.cn/aigc/posts/gpu-gemm-part-3/), [NVIDIA CUDA Tile blog](https://developer.nvidia.com/blog/tuning-flash-attention-for-peak-performance-in-nvidia-cuda-tile/)).

### 16.4 Thread-Block Clusters and Distributed Shared Memory

Hopper clusters let up to 8 CTAs (16 non-portably on B200, opting in via `cudaFuncAttributeNonPortableClusterSizeAllowed`) share **distributed shared memory** and TMA multicast ([NVIDIA Blackwell Tuning Guide](https://docs.nvidia.com/cuda/blackwell-tuning-guide/index.html)). The killer use is **multicast**: one TMA fetch feeds up to 16 SMs' SMEM simultaneously, slashing global traffic for shared operands (MoE weights) ([The Index](https://theindex26.com/tensor-memory-accelerator-tma-nvidia-hopper/)). Caveat: larger clusters reduce max active blocks per GPU (occupancy), so measure net effect.

### 16.5 Programmatic Dependent Launch (PDL)

PDL lets a dependent kernel start its prologue (e.g. weight prefetch via `griddepcontrol.launch_dependent_grids`) while the producer kernel is still finishing, hiding launch latency on Hopper and Blackwell. CUTLASS example 63 exposes `overlap_ratio` (when to trigger, default 0.5) and `prefetch_ratio` (how many K-tiles to prefetch); TensorRT-LLM's expanded PDL use contributed to up to 2.8× throughput gains on Blackwell MoE inference ([CSDN CUTLASS PDL guide](https://blog.csdn.net/gitblog_01183/article/details/165467821), [NVIDIA Developer Blog](https://developer.nvidia.com/blog/delivering-massive-performance-leaps-for-mixture-of-experts-inference-on-nvidia-blackwell/)).

## 17. Blackwell (sm_100 / sm_120) Experience

### 17.1 The tcgen05 Revolution — and the Compatibility Cliff

Blackwell's 5th-gen tensor cores abandon warp-synchronous MMA: **`tcgen05.mma` is issued by a single elected thread**; operands come from SMEM (or TMEM for A), and the **accumulator lives in TMEM — a dedicated 256 KB/SM on-chip memory (128 lanes × 512 columns × 32-bit), not the register file** ([CUTLASS tcgen05 guide](https://docs.nvidia.com/cutlass/4.7.0/media/docs/pythonDSL/guides/mma/tcgen05_programming.html), [The Software Frontier](https://www.thesoftwarefrontier.com/p/how-blackwells-tensor-memory-actually)). The forcing function: the largest single-CTA atom (m128n256k16) produces 32,768 FP32 accumulators = 256 registers/thread against a 255 ceiling — literally unrepresentable in the old model ([The Software Frontier](https://www.thesoftwarefrontier.com/p/how-blackwells-tensor-memory-actually)).

**Critical porting fact:** `wgmma` is not deprecated on Blackwell — it is **removed from every Blackwell target** (`sm_100a`, `sm_103a`, `sm_120a`); Hopper warpgroup-MMA kernels fail at assembly time, not at runtime, and NVIDIA estimates staying on wgmma leaves 30–40% of peak on the table for MX-FP4 workloads ([The Software Frontier](https://www.thesoftwarefrontier.com/p/how-blackwells-tensor-memory-actually), [pinghaoyang.com.cn](https://pinghaoyang.com.cn/aigc/posts/gpu-gemm-part-3/)). Consumer Blackwell (RTX 50, sm_120) keeps **extended `mma.sync`** (warp-level, register operands) and has **no TMEM** — datacenter and consumer Blackwell need different GEMM paths ([backend.ai](https://www.backend.ai/blog/2026-02-is-dgx-spark-actually-a-blackwell)).

### 17.2 Measured Microarchitecture Facts

From the University of Delaware microbenchmark suite ([arXiv 2512.02189](https://arxiv.org/html/2512.02189v3)):

- `tcgen05.mma` single-instruction latency is **11.0–11.4 cycles, nearly flat across tile sizes** (m64n64k16 → m256n256k16) — a spatial array, vs Hopper's linearly-scaling temporal pipeline; 2.9–11.6× lower single-instruction latency.
- Removing the 4-warp wgmma synchronization cut scheduler stalls **18–23%** in memory-bound kernels.
- Chained GEMMs keeping intermediates in TMEM avoid ~12 TB/s of data movement vs Hopper's global round-trip; TMEM staging cut board power 15% at equal throughput for 2048² GEMM — but for small problems fitting in L1, forcing TMEM costs 3–5% overhead. **TMEM for multi-stage large-tile pipelines; classic SMEM path for single-shot small ops.**
- B200: 148 SMs, dual-die with NV-HBI, 4 L2 partitions (2× Hopper), 192 GB HBM3e at 8 TB/s; SMEM/L1 still 256 KB combined with 228 KB max shared per SM (227 KB addressable per block after the 1 KB reservation; static allocations still capped at 48 KB) ([NVIDIA Blackwell Tuning Guide](https://docs.nvidia.com/cuda/blackwell-tuning-guide/index.html)).
- tcgen05 throughput is **2–4× Hopper wgmma** depending on dtype (TF32→NVFP4) ([Joe Fioti / CUTLASS](https://www.linkedin.com/posts/joe-fioti-24a986189_blackwells-tensor-core-introduces-a-new-activity-7495334431197392896-xz0c)); two adjacent CTAs can form a **CTA pair** executing one MMA across a wider tile.

### 17.3 Programming-Model Consequences

The kernel becomes a **dataflow choreography**: TMA stages tiles; a producer warp manages descriptors/mbarriers; one elected thread issues `tcgen05.mma`; accumulators drain TMEM→registers via `tcgen05.ld` for the epilogue; TMEM must be explicitly allocated/deallocated (`tcgen05.alloc/dealloc`) and is another scarce resource to budget (512 columns per SM — the largest accumulator takes half) ([The Software Frontier](https://www.thesoftwarefrontier.com/p/how-blackwells-tensor-memory-actually), [tessl.io](https://tessl.io/registry/skills/github/fla-org/flash-linear-attention/fla-triton-to-gluon)). Latency hiding shifts from *occupancy* (many CTAs per SM) to *intra-CTA pipelining* (multiple tcgen05 stages in flight, one mbarrier per stage) — the classic "more warps" advice matters less on Blackwell GEMMs than correct pipeline depth. Also: `use_acc=false`-style zero-init tricks and `tcgen05.cp` (SMEM→TMEM without register round-trip) remove common epilogue costs ([tessl.io](https://tessl.io/registry/skills/github/fla-org/flash-linear-attention/fla-triton-to-gluon)).

On the host side, Blackwell GPUs are so fast that **CPU dispatch becomes the bottleneck**: vLLM measured ~10% end-to-end gain from async scheduling alone on H200/B200/GB200 ([vLLM blog](https://vllm.ai/blog/2026-02-01-gpt-oss-optimizations)) — see §20.

## 18. AMD Platform Experience (GCN / RDNA / CDNA)

### 18.1 Wave32 vs Wave64 — the Single Most Expensive Line in a Port

NVIDIA warp = 32; AMD wavefront = 64 on GCN/CDNA, 32 native on RDNA ([HIP docs](https://rocm.docs.amd.com/projects/HIP/en/latest/understand/hardware_implementation.html)). Code must never assume either — use `warpSize`/`__hip_warp_size` ([labhub.hopto.org](https://labhub.hopto.org/blog/gpu-cuda/2026-08-02-amd-vs-nvidia-gpu-stack?lang=en), [dev.to](https://dev.to/javadinteger/advanced-gpu-optimization-cuda-hip-from-zero-to-hero-1dle)). Classic silent bug: a shuffle reduction loop `for (off = 16; off > 0; off >>= 1)` assumes 32 lanes; on a 64-lane wavefront **it compiles, doesn't crash, and halves the reduction** — start at `warpSize/2` ([labhub.hopto.org](https://labhub.hopto.org/blog/gpu-cuda/2026-08-02-amd-vs-nvidia-gpu-stack?lang=en)). The ballot return type likewise widens (64-bit mask on wave64). In LuisaCompute this is exactly why `warp_lane_count()`/`warp_active_bit_mask().x` (with `.y` for >32 lanes) exist — never slice `.x` alone on wave64 backends.

Mode selection guidance:

- **CDNA (MI-series)**: optimize for 64-thread wave granularity, leverage matrix cores, consider AGPRs for accumulators ([HIP hardware docs](https://rocm.docs.amd.com/projects/HIP/en/latest/understand/hardware_implementation.html)).
- **RDNA**: design for wave32; wave64 is emulated as two sequential wave32 passes on gfx11 and **loses VOPD dual-issue**; SGPR read/write ordering across the two passes can surprise; cross-half permutes need LDS staging ([rdna3-zen3-unified](https://github.com/thejeangenie18/rdna3-zen3-unified)). Divergence-heavy shaders favor wave32 (half the wasted lanes); uniform compute-dense shaders can favor wave64 (dual-issue constraints always satisfied in wave64 on RDNA3) ([Chips and Cheese](https://chipsandcheese.com/p/starfield-on-the-rx-6900-xt-rx-7600-and-rtx-2060-mobile), [velog.io](https://velog.io/@lunacat16/%EA%B7%B8%EB%9E%98%ED%94%BD-%ED%94%84%EB%A1%9C%EA%B7%B8%EB%9E%98%EB%B0%8D-Fundamentals-1)).
- **RDNA4 (gfx1201) went wave32-only**, added dynamic VGPR allocation (wave32-only feature), split `vmcnt` into per-class counters, and doubled LDS banks to 64 — together these closed a long-context FlashAttention gap: a wave32 toggle plus row-splitting and Q-register caching delivered the measured win ([zolotukhin.ai](https://zolotukhin.ai/blog/2026-05-11-the-wave32-commit-that-closes-rdna4-long-context-flash-attention-gap/)).
- **CDNA5 (MI455X) also abandons wave64 for wave32**, with up to 1,024 VGPRs/thread addressable and 64 resident waves/WGP ([pcgameshardware.de](https://www.pcgameshardware.de/Grafikkarten-Grafikkarte-97980/Specials/AMD-Instinct-MI455X-CDNA-5-432-GiB-HBM4-Technik-Check-1549242/)) — the industry is converging on 32-lane groups; LuisaCompute's `set_warp_size(32)`-pinned kernels age well.
- Workgroup size: **make it a multiple of 64** to be optimal across all AMD generations even when wave32 is native ([AMD GPUOpen](https://gpuopen.com/learn/rdna-performance-guide/)).

### 18.2 Register/LDS Budgets and Occupancy on AMD

Occupancy = min over VGPR, LDS, SGPR limits, capped at hardware max waves/SIMD ([FlyDSL tuning guide](https://rocm.github.io/FlyDSL/kernel_tuning_guide.html)). Key numbers:

| Arch | Wave | VGPR model | LDS/CU (WGP) | Notes |
|---|---|---|---|---|
| gfx942 (CDNA3, MI300X) | 64 | 512 combined (256 arch + 256 accum AGPR) / SIMD | 64 KB | MFMA v3; AGPRs hold MFMA accumulators but are not reachable from MLIR — inline ASM only |
| gfx950 (CDNA4) | 64 | same combined 512 | 160 KB | MFMA-scale / transpose loads; 2.5× LDS headroom |
| gfx1250 (RDNA) | 32 | wave32 register file | 320 KB | per-wave accounting differs — don't reuse the CDNA 512 rule |

([FlyDSL tuning guide](https://rocm.github.io/FlyDSL/kernel_tuning_guide.html), [FlyDSL issue #8](https://github.com/sunway513/FlyDSL/issues/8)). On RDNA4, ≤96 VGPRs/thread = max occupancy (16 waves/SIMD); 97–128 → 12; 129–192 → 8; 193–256 → 6; dynamic allocation rounds to 16- or 32-register blocks ([zolotukhin.ai reference](https://zolotukhin.ai/zinc/docs/amd-gpu-reference/)). A real occupancy-cliff example: reducing VGPRs 102 → 96 raised occupancy 4 → 5 waves/SIMD on CDNA2 ([ROCm register-pressure blog](https://rocm.blogs.amd.com/software-tools-optimization/register-pressure/README.html)).

Reducing register pressure on AMD (measured techniques, [ROCm register-pressure blog](https://rocm.blogs.amd.com/software-tools-optimization/register-pressure/README.html)):
1. Always set `__launch_bounds__` — without it the compiler assumes 1024 threads/block.
2. Move variable definitions close to first use (shorten live ranges).
3. Avoid stack arrays (`double array[10]` lives in scratch), avoid passing big objects as kernel args, replace `pow(x,2.0)` with `x*x` (all device functions are inlined; each inflates pressure).
4. Control `#pragma unroll` — Clang unrolls more literally than nvcc; unrolling raises pressure.
5. Compile with `-Rpass-analysis=kernel-resource-usage` and check `VGPRs Spill`/`ScratchSize` > 0.

### 18.3 MFMA / Matrix-Core Experience

MI300X (gfx942): 304 CUs, 64-thread wavefronts, 192 GB HBM3 at 5.3 TB/s peak (measured ~4.3 TB/s, 81% utilization), 4 matrix-core engines per CU, 64 KB LDS per CU, 32 MB L2 ([mfma-cdna-amd](https://github.com/bogdannadev/mfma-cdna-amd/blob/master/README.md), [emergentmind](https://www.emergentmind.com/topics/amd-mi300x-gpu)). Practical MFMA lessons ([ROCmKernelWiki](https://github.com/jhinpan/ROCmKernelWiki/blob/main/wiki/patterns/mfma-underutilized.md)): pipelining + double buffering removes load-induced bubbles; pre-shuffling operand layouts removes LDS-read bubbles; together they move a "compute-bound but slow" kernel from a small fraction of peak toward datasheet TFLOPS. Verify with rocprof counters — MFMA utilization should rise and `SQ_WAIT_INST_LDS` should fall. Beware **over-tiling to feed the matrix core**: the VGPR/AGPR pressure pattern is the failure mode when accumulator tiles get too large.

### 18.4 CUDA→HIP Porting Checklist

HIPIFY translates 80–95% of straightforward CUDA automatically; the rest is where the time goes ([clusterbid.com](https://clusterbid.com/blog/nvidia-cuopt-vs-amd-hip-optimization-comparison)):

- **Warp-size assumptions** — never auto-translated (HIPIFY can't know if a literal 32 means warp size). Ginkgo's workflow: factor all arch-specific parameters (warp size, tile constants) into config variables *before* porting ([arXiv 2006.14290](https://arxiv.org/html/2006.14290v1)).
- **`__launch_bounds__` semantics differ**: CUDA's `MIN_BLOCKS_PER_MULTIPROCESSOR` becomes HIP's `MIN_WARPS_PER_EXECUTION_UNIT`; convert with `(blocks × threads) / (warpSize × 2)` in CU mode or `× 4` in WGP mode ([HIP porting guide](https://rocm.docs.amd.com/projects/HIP/en/docs-7.0.1/how-to/hip_porting_guide.html)).
- **Inline PTX has no counterpart** — rewrite as HIP intrinsics or AMDGCN asm; this is "the single most reliable time sink" ([labhub.hopto.org](https://labhub.hopto.org/blog/gpu-cuda/2026-08-02-amd-vs-nvidia-gpu-stack?lang=en)).
- CUDA-only libraries (cuDNN/CUTLASS/cuBLASLt) need ROCm equivalents or rewrites; NVIDIA-tuned memory patterns can be suboptimal on AMD's cache hierarchy — **performance tuning after porting is a separate phase** (2–8 weeks per significant kernel) ([uplatz](https://uplatz.com/blog/the-architecture-of-acceleration-a-comprehensive-analysis-of-gpu-driven-computing/), [clusterbid.com](https://clusterbid.com/blog/nvidia-cuopt-vs-amd-hip-optimization-comparison)).
- Experiment with `-mwavefrontsize64` / `-mno-wavefrontsize64` and `-mcumode` per kernel and re-profile ([beefed.ai](https://beefed.ai/en/port-cuda-to-hip-amd-performance)).
- Recommended workflow: hipify once, verify functional correctness **on the NVIDIA GPU** via HIP-on-CUDA first, then move to ROCm — this isolates remaining bugs to the ROCm stack ([uplatz](https://uplatz.com/blog/the-architecture-of-acceleration-a-comprehensive-analysis-of-gpu-driven-computing/)).
- AMD quirks worth knowing: routing Vulkan compute dispatches through the graphics queue dodged an AMD driver memory-fence stall worth a large fraction of a 56% regression in llama.cpp ([zolotukhin.ai](https://zolotukhin.ai/blog/2026-05-11-the-wave32-commit-that-closes-rdna4-long-context-flash-attention-gap/)) — when a backend is inexplicably slow, suspect the queue/driver path, not just the kernel.

## 19. Other Platforms: Apple Metal, Intel Xe, Vulkan/DX12 Subgroups

### 19.1 Apple Silicon / Metal

- Apple GPUs run **SIMD-groups of 32** (`threadExecutionWidth`); both LuisaCompute Metal backends reject any other pinned size (§6). Threadgroup memory is **32 KiB** by default (64 KiB on newer families) — note this is *smaller* than the 48 KB static-SMEM floor on NVIDIA; size portable kernels to the lowest target.
- Apple Silicon is **unified memory**: prefer `.shared` buffers for CPU↔GPU exchange (zero-copy), `.private` for GPU-only intermediates ([Medium/macoclock](https://medium.com/macoclock/unlocking-the-gpu-a-deep-dive-into-metal-kernels-on-apple-silicon-adc0859413bd)).
- Two-tier locality model: registers + `simd_shuffle` (1–2 cy) first, threadgroup memory (32 KiB, ~2–4 cy) as an exchange tier between SIMD-groups ([arXiv 2603.27569](https://www.alphaxiv.org/abs/2603.27569)). Cross-vendor study found **intra-wave shuffle is essential on NVIDIA but optional on Apple** — Apple's scheduler tolerates barrier-based reductions with ~98% efficiency ([arXiv 2603.28793](https://arxiv.org/html/2603.28793v1)) — so warp-collective refactors (§3) are still correct on Metal, just less decisive.
- Query capabilities at runtime (`supportsFamily:`) and select kernels accordingly; never hardcode a family. Metal 4 adds MTLTensor and quantized types ([macoclock](https://medium.com/macoclock/unlocking-the-gpu-a-deep-dive-into-metal-kernels-on-apple-silicon-adc0859413bd)).
- Hierarchical atomics (SIMD → threadgroup → global) instead of flat global atomics ([macoclock](https://medium.com/macoclock/unlocking-the-gpu-a-deep-dive-into-metal-kernels-on-apple-silicon-adc0859413bd)) — matches §4.3.

### 19.2 Intel Xe (Arc / Data Center Max)

Intel's oneAPI GPU Optimization Guide covers the same pillars with Xe terminology: sub-groups and SIMD vectorization, registerization and avoiding spills (including a small-vs-large register mode tradeoff), shared local memory, `restrict` pointer aliasing, work-group size selection, prefetch, and reduction patterns ([Intel oneAPI GPU Optimization Guide 2024.2](https://cdrdv2-public.intel.com/824351/oneapi_optimization-guide-gpu_2024.2-771772-824351.pdf)). XMX matrix extensions are programmed via the SYCL joint-matrix extension. Notable Intel-specific advice: remove per-lane conditional checks where possible, prefer subgroup collectives, and watch register-spill behavior when porting high-pressure kernels to Max GPUs — the guide has a dedicated chapter for it.

### 19.3 Vulkan / DX12 Subgroup Portability

- Vulkan subgroup sizes range **8–64** (128 possible on some mobile GPUs, e.g. Adreno/Imagination); on many GPUs the size is dynamic per dispatch and unknowable without `VK_EXT_subgroup_size_control`, which also exposes min/max sizes and a required-size pipeline flag ([compute-shader-101](https://github.com/googlefonts/compute-shader-101/blob/main/docs/glossary.md), [Khronos registry](https://registry.khronos.org/VulkanSC/specs/1.0-extensions/man/html/VK_EXT_subgroup_size_control.html), [Phoronix](https://www.phoronix.com/news/Vulkan-1.1.116-Released)). LuisaCompute's Vulkan backend errors if you pin a size outside `[minSubgroupSize, maxSubgroupSize]` (§1.5) — query before pinning.
- Not all GPUs support all subgroup ops; capability bits must be checked ([compute-shader-101](https://github.com/googlefonts/compute-shader-101/blob/main/docs/glossary.md)).
- DX12 SM6.x wave ops map onto vendor paths: on RDNA, quad ops (`QuadReadAcrossX`) lower to fast DPP8/DPP16 instructions, shuffles across ≤ 8 lanes are very fast, ≤ 16 still OK, and **shuffles across more than 32 lanes force slow compiler fallbacks** (LDS round-trips) — keep logical groups ≤ 8–16 lanes on AMD ([Interplay of Light](https://interplayoflight.wordpress.com/2020/05/23/optimizing-for-the-rdna-architecture-presentation-notes/)). Vulkan subgroup ops fix the wavefront size to 64 on some AMD paths unless an extension is available — the same shader can silently change subgroup width between APIs ([Interplay of Light](https://interplayoflight.wordpress.com/2020/05/23/optimizing-for-the-rdna-architecture-presentation-notes/)).
- Group active threads together in batches of 32 so RDNA can skip wholly-inactive wave32s (~7% measured in the mip-chain study) ([Interplay of Light](https://interplayoflight.wordpress.com/2020/05/23/optimizing-for-the-rdna-architecture-presentation-notes/)).

## 20. Host-Side Throughput: Launch Overhead, CUDA Graphs, Fusion

### 20.1 Quantifying Launch Overhead

A single CUDA kernel launch costs roughly **2–15 µs** of CPU/driver time depending on version and warmth — negligible for millisecond kernels, dominant for microsecond kernels: a 7B LLM forward pass at batch 1 launches ~3,000–5,000 kernels, so launch overhead alone can exceed the compute budget ([gozymtrace.com](https://gozymtrace.com/blog/zymtrace-03-cuda-kernel-launch-latency), [llms.blog](https://www.llms.blog/posts/cuda-graph-capture-in-production-llm-serving-dynamic-batch-bucketing-static-memory-pooling-and-kernel-launch-elimination)). The LuisaCompute CommandList batching of §8 is the portable mitigation; vendor-specific ones:

- **CUDA Graphs**: capture once, replay with one `cudaGraphLaunch`; measured 1.56–2.1× on a 48-kernel transformer batch, 32% wall-clock cut in libtorch, and 30–50% per-launch overhead reduction in production ([martinuke0.github.io](https://martinuke0.github.io/posts/2026-09-14-deep-dive-into-cuda-graphs-for-efficient-kernel-launch-overhead-reduction/), [zenn.dev](https://zenn.dev/kosuke_sugimoto/articles/b36f2628daa2be?locale=en)). **Measured locally (RTX 4080 SUPER, 2000 tiny 256-thread dispatches, cuda backend, §24):** per-dispatch enqueue cost 5.0–6.5 µs; packing the same dispatches into one `CommandList` barely helped (~5%) because the CUDA backend already enqueues asynchronously — but `CudaGraphExt` replay of the captured graph ran at **0.8 µs/dispatch (~7–8× faster than per-dispatch enqueue)**. On this backend the graph, not the command list, is what removes the launch tax. Constraints: static shapes/pointers (update with `cudaGraphExecKernelNodeSetParams` or copy into mapped pointers), heavy one-time `cudaGraphInstantiate` (cache exec graphs), and capture forbids host-dependent control flow ([mortalapps.com](https://mortalapps.com/infrastructure/cuda-triton-compiler-systems/cuda-graphs-launch-overhead/)). Multi-step scheduling pushes further: encapsulate K decode iterations (K∈[4,16]) in one graph so the CPU scheduler runs once per K tokens ([llms.blog](https://www.llms.blog/posts/cuda-graph-capture-in-production-llm-serving-dynamic-batch-bucketing-static-memory-pooling-and-kernel-launch-elimination)).
- **PDL** (§16.5) overlaps dependent kernels' prologue/epilogue.
- **Kernel fusion** removes both launches and intermediate HBM round-trips — every unfused op writes to global memory and the next reads it back ([aigrants.in](https://aigrants.in/topics/how-to-optimize-low-level-neural-kernels)); FlashAttention is the canonical fusion win (§17 case study below).
- If `Drain` stalls dominate a kernel's profile, the kernel is too short — fuse it ([zinlab](https://zinlab.github.io/tutorials/nsight-compute-mastery.html)).

### 20.2 Case Study: FlashAttention as the Model Cross-Platform Kernel

The FA lineage is a compressed course in everything above ([llms.blog](https://www.llms.blog/posts/flashattention-io-aware-tiling-online-softmax-mathematics-memory-hierarchy-dynamics-and-kernel-evolution), [aicassindra.com](https://aicassindra.com/blogs/llm/llm_flash_attention.html)):

| Version | Key techniques | Utilization | Binding constraint |
|---|---|---|---|
| Naive attention | materializes N×N in HBM (Θ(N²) IO) | 15–25% peak | HBM bandwidth |
| FA-1 | tiling + online softmax, Θ(N²d²/M) IO | 25–40% on A100 | SMEM sync, non-matmul overhead |
| FA-2 | parallelize over query blocks, fewer non-matmul FLOPs, warp-level partitioning | 50–73% on A100 | register pressure |
| FA-3 | TMA + wgmma + warp specialization + pingpong, FP8 block scaling | 75–85% on H100 | instruction issue |

Generic lessons that transfer to LuisaCompute kernels: (1) **never materialize quadratic intermediates** — recompute in fast memory (backward pass recomputation trades free FLOPs for HBM traffic); (2) **defer normalization divisions to the end** (lazy softmax); (3) rescale accumulators with running statistics (`exp(m_old − m_new)` correction) so tiles stream; (4) small working sets defeat fast hardware: FA's advantage shrinks when head_dim > 256 because IO complexity flips ([shaiyy.cn](https://www.shaiyy.cn/3062.html)).

## 21. Atomic Contention: The Full Hierarchy

Atomics under contention serialize at ~1/(2×memory latency) per conflicting update ([PMPP vault](https://github.com/HuaTsai/pmpp-vault-sources/blob/main/09-Parallel-Histogram/02-Histogram-Optimizations-Privatization-Coarsening-Aggregation.md)). The escalation ladder, cheapest first:

1. **Warp aggregation** — reduce within the warp (`__shfl`/`warp_active_sum`), one atomic per warp: 32× fewer atomics, 5–20× faster ([chipfoundryservices.com](https://www.chipfoundryservices.com/topics/gpu-atomic-operations)). This is §3.1.
2. **Block privatization in shared memory** — per-block private counters/histograms with cheap SMEM atomics, then one global commit per block: 10–100× fewer global atomics ([yencal.github.io](https://yencal.github.io/gpu-histogram-optimization/) measured 0.1% → 68.5% of peak bandwidth from privatization + `uint4` vectorized loads). This is §4.3.
3. **Thread coarsening** — fewer blocks, each thread processes multiple elements (interleaved partitioning for coalescing); cuts redundant commit phases ([PMPP vault](https://github.com/HuaTsai/pmpp-vault-sources/blob/main/09-Parallel-Histogram/02-Histogram-Optimizations-Privatization-Coarsening-Aggregation.md)).
4. **Aggregation** — accumulate consecutive same-bin updates in a register, flush on bin change; wins on skewed data, costs registers/divergence ([PMPP vault](https://github.com/HuaTsai/pmpp-vault-sources/blob/main/09-Parallel-Histogram/02-Histogram-Optimizations-Privatization-Coarsening-Aggregation.md)).
5. **Atomic-free restructure** — sort-then-reduce or scan-based compaction beats atomics entirely for some patterns ([chipfoundryservices.com](https://www.chipfoundryservices.com/topics/gpu-atomic-operations)).

Caveats that match §5.7's measured histogram result (privatization only wins ≥ 512 threads/block, 8× *slower* at 64): privatization pays only when contention actually exists — a cross-vendor study found per-warp privatized histograms no faster than a single shared histogram when contention was low ([arXiv 2603.28793](https://arxiv.org/html/2603.28793v1)). Modern GPUs execute global atomics in L2, not DRAM — fast at low contention ([mtuann.github.io](https://mtuann.github.io/blog/cuda/pmpp/chapter-09/)). Watch Nsight's atomic-replay metric: > 10 replays = severe contention. Use narrowest memory scope (`atomicAdd_block` > device > system); FP16 atomics don't exist natively — use CAS on bits or integer atomics ([chipfoundryservices.com](https://www.chipfoundryservices.com/topics/gpu-atomic-operations)).

## 22. Cross-Platform Portability Checklist

The bugs that survive porting are the ones that **compile and pass small tests**. Checklist, ordered by pain observed in the wild:

1. **Warp/wave/subgroup width**: 32 (NVIDIA, RDNA, Metal, Intel typical) vs 64 (GCN/CDNA) vs dynamic (Vulkan). Never hardcode 32/64 in shuffle loops, ballot masks (32-bit vs 64-bit return!), or `threadIdx.x / 32` warp indexing. Query (`warpSize`, `warp_lane_count()`, `WaveGetLaneCount()`, `gl_SubgroupSize`, `threadExecutionWidth`) or pin and verify the backend accepted it (§1.5, §6).
2. **Shuffle semantics**: masks on `__shfl_sync`, out-of-range lane behavior, and whether inactive lanes' values are defined differ per vendor — clamp indices and require full participation (§7.13). AMD shuffles beyond 32 lanes degrade to slow paths (§19.3).
3. **Lockstep assumptions**: post-Volta NVIDIA has **independent thread scheduling** — lanes no longer share a program counter, so implicit "same warp ⇒ synchronized" reasoning (e.g. reading a neighbor's register write without a shuffle/sync) is UB. Always use explicit `_sync` intrinsics/`__syncwarp()`; this is also why LuisaCompute warp collectives lower to `_sync`-suffixed forms.
4. **Shared-memory capacity floors differ**: 48 KB static SMEM on NVIDIA, 32 KiB floor in LuisaCompute's HLSL path (§5.1), 32 KiB default on Metal, 64 KB LDS on CDNA3. Size portable tiles to the minimum or query per-device limits.
5. **Block-size optima don't transfer** (§13.3, §5.7): re-sweep per backend; VK vs DX alone differed 2.6× on identical DSL kernels.
6. **Driver/queue paths**: identical kernels can stall on one queue and not another (§18.4 AMD graphics-queue case).
7. **Feature asymmetry**: tensor/matrix core instructions (mma/wgmma/tcgen05 vs MFMA/WMMA vs XMX vs simdgroup_matrix) have no portable form — guard vendor paths behind macros or accept the library route (cuBLAS/rocBLAS/CUTLASS/CK) ([dev.to](https://dev.to/javadinteger/advanced-gpu-optimization-cuda-hip-from-zero-to-hero-1dle)).
8. **Numerics**: double-precision ratios, SFU accuracy, and FP16 conversion behaviors differ; re-run correctness on each backend with ragged sizes (§10.9).
9. **Dispatch limits**: D3D12's 65,535 groups/dimension (§5.7 trap 2) — flatten to 2D grids portably.
10. **Verify on ≥ 2 backends with tail/partial sizes** (§3.8 rule) — idle-warp and boundary-lane paths are where portability bugs hide.

## 23. Cross-Platform Rules of Thumb (Addendum to §7)

15. **Roofline before code.** Compute AI and the ridge point for the target GPU; pick the lever (fewer bytes vs more reuse vs more compute throughput) that matches the side of the ridge you're on (§11).
16. **Read the dominant stall, then fix that one thing.** Long scoreboard → latency/occupancy; short scoreboard → SMEM/bank conflicts; MIO throttle → too many memory instructions (vectorize); barrier → imbalance; math-pipe throttle → you're done (§12).
17. **Spills before everything.** `ptxas -v` `lmem` or AMD `ScratchSize`/`vgpr_spill_count` nonzero → fix that first (§12, §18.2).
18. **Vectorize memory access** (16 B/lane, aligned) before exotic restructuring — 15–40% on *issue-bound* memory kernels; measured zero gain when the kernel is already DRAM-saturated (scalar coalesced hit ~87% of peak, §14.2, §24).
19. **Pipeline staged loops** — double-buffer with `cp.async`/TMA/`async_copy`; re-check the SMEM/LDS occupancy cliff after adding stages (§15.3).
20. **Match the abstraction to the silicon generation**: warp-synchronous MMA (≤ Ampere) → wgmma+TMA warp specialization (Hopper) → tcgen05/TMEM single-thread-issue dataflow (Blackwell DC) — and remember consumer Blackwell has no TMEM and no wgmma (§17.1).
21. **Redistribute registers when warps have different jobs** (`setmaxnreg`, Hopper+); a TMA producer warp needs ~24 (§16.2).
22. **Wave32 is the industry's converging granularity** (RDNA native, RDNA4/CDNA5 exclusive) — algorithms pinned to 32 lanes port best (§18.1).
23. **Privatize atomics hierarchically** (warp → block → global), but only when contention is measured, not assumed (§21).
24. **Batch host submissions** (CommandList §8) and capture graphs when shapes are static; on the newest GPUs the CPU is often the bottleneck (§20.1). Measured on the cuda backend: CommandList alone ≈5%, CUDA-graph replay ≈7–8× on tiny kernels (§24) — prefer the graph when shapes are static.
25. **Occupancy is a means, not a goal.** "Runs" ≠ "fits": 1024-thread blocks with heavy registers spill and crawl 19× slower while launching fine (§5.7, measured on vk/dx); on CUDA the same over-subscribed launch **hard-fails at dispatch** with `CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES` (observed at 512 and 1024 threads with 32+ live `float4` accumulators, §24) — either way, check the slowdown *and* the launch, not just launch success. Latency is hidden by ILP as much as by warps (§13.1).
26. **Re-measure per backend, per generation, per tile shape.** Every measured table in this document (§5.7, §24, FA3 configs, RDNA GEMM steps) was the result of a sweep; none of the constants transfer unchanged. Warm the GPU first: on a cold RTX 4080 SUPER the first sweep run read up to 70% slower on compute-bound cases than warm back-to-back runs (§24).

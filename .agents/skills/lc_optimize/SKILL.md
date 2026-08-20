---
name: lc_optimize
description: Optimize LuisaCompute DSL kernels using warp/wave primitives, shared-memory aggregation, block-level collectives, thread-group (block) size / occupancy tuning, and C++ branch-prediction hints (`[[likely]]`/`[[unlikely]]`). Use when kernels bottleneck on atomics, reductions, inter-thread communication, or poor occupancy.
---

# LuisaCompute DSL Kernel Optimization Guide

> **Zero-initialization note:** All temporary local variables in DSL kernels — scalars, vectors, matrices, structs, and arrays (excluding shared arrays `Shared<T>`) — are created with a zero value automatically. There is no need to manually set them to zero before use. This applies to variables declared with `Var<T>`, `auto`, or type-inferred syntax inside a kernel or callable body.

## 1. Available Warp/Wave Primitives

LuisaCompute exposes the following warp-level (subgroup) intrinsics via `luisa/dsl/builtin.h`. All operate on *active lanes within the current warp*.

### 1.1 Query / Metadata

| DSL Call | Returns | Description |
|---|---|---|
| `warp_lane_count()` | `UInt` | Total lanes in the warp (e.g. 32 or 64). |
| `warp_lane_id()` | `UInt` | Current lane index `[0, warp_lane_count())`. |
| `warp_is_first_active_lane()` | `Bool` | True if this lane is the first active lane in the warp. |
| `warp_first_active_lane()` | `UInt` | Lane index of the first active lane. |
| `device.compute_warp_size()` | `uint` | Host-side query for backend's native warp size. |

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
| `warp_active_bit_mask(v)` | `Bool -> UInt4` | 128-bit ballot mask of true predicates. |

All accept scalar or vector types (except bit ops which require integral types).

### 1.3 Prefix (Exclusive Scan)

Each lane receives the exclusive prefix of all *preceding* active lanes. Lane 0 receives identity (0 for sum, 1 for product, 0u for count_bits).

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
| `set_warp_size(uint8_t)` | Must be a power-of-two in [8,128]. Call *inside* kernel lambda before compilation. |
| `sync_block()` | Full block barrier. All threads in the block must reach it. |

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
        result.write(rhs_x * /*M*/ ...  + lhs_y, acc);
    };
};
```

**Key insight:** Warp-active reductions eliminate the need for shared memory entirely. All lanes in a warp already execute in lockstep, so `warp_active_sum` is a single hardware instruction on most GPUs.

### 2.2 Butterfly Reduction via `warp_read_lane`

For finding the maximum across a logical group smaller than the warp, use pairwise `warp_read_lane` with XOR lane masks (butterfly / tree-reduction pattern):

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

This avoids separate `warp_prefix_sum` calls per group and instead uses a single warp-wide prefix plus lane reads to extract group boundaries.

### 2.4 Warp-Polling Decoupled Look-Back

For inter-block scan, tiles publish their status and other tiles poll via warp collectives:

```cpp
// Poll across warp: any lane sees INVALID?
$while (warp_active_any(status == SCAN_TILE_INVALID)) {
    delay();
    status = tile_status.volatile_read(predecessor_idx);
};

// All lanes agree predecessor is inclusive?
$while (warp_active_all(predecessor_status != SCAN_TILE_INCLUSIVE)) {
    predecessor_idx -= 32;
    // poll next window...
    exclusive = scan_op(window_aggregate, exclusive);
};
```

Also uses `warp_active_bit_mask` for segmented reductions within warps:

```cpp
UInt warp_flags = warp_active_bit_mask(flag == 1u).x;
warp_flags >>= 1;  // for HEAD_SEGMENT mode
warp_flags &= get_lane_mask_ge();  // mask of lanes with id >= mine
warp_flags |= 1u << (LOGIC_WARP_SIZE - 1u);  // sentinel
UInt last_lane = ctz(warp_flags);  // first set bit = end of my segment
```

### 2.5 Shuffle-Down for Warp Reduction

A software implementation of warp reduce using `warp_read_lane` with increasing offsets:

```cpp
Var<T> result = input;
UInt offset = 1u;
$while (offset < warp_lane_count()) {
    Var<T> temp = warp_read_lane(result, lane_id + offset, valid_item);
    $if (lane_id + offset <= valid_item) {
        result = reduce_op(result, temp);
    };
    offset <<= 1;
};
```

This is a fallback pattern; prefer `warp_active_sum` / `warp_active_min` / `warp_active_max` when the operation matches the built-in.

### 2.6 Quantized Matmul with Warp

Warp-level GEMM where each warp computes one output tile. Threads cooperatively load quantized weights via `warp_read_lane` to assemble dequantized values, then accumulate with `warp_active_sum`:

```cpp
auto warp_lane = warp_lane_id();
UInt tile_count = (K + warp_size - 1) / warp_size;
for (auto t : dynamic_range(tile_count)) {
    UInt tile_begin = t * warp_size;
    UInt tile_size = min(warp_size, K - tile_begin);

    // Each lane loads one quantized word, then shares via warp_read_lane
    UInt rel_byte = warp_lane * kElementByteSize;
    UInt word = warp_read_lane(warp_word, rel_byte / 4u);
    // ... dequantize and multiply ...

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
Int warp_partial = warp_active_sum(1);
// First lane of each warp writes to shared
$if (warp_is_first_active_lane()) {
    shared.atomic(warp_lane_id() / warp_size).fetch_add(warp_partial);
};
sync_block();
// One thread (e.g. thread 0) combines across warps
$if (thread_x() == 0u) {
    Int block_total = 0;
    for (auto w : range(num_warps_per_block)) {
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

// Phase 2: cross-warp reduction in shared memory (much smaller)
$if (warp_is_first_active_lane()) {
    Shared<float> warp_results{num_warps_per_block};
    warp_results[warp_id()] = warp_sum;
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

**Before:** Manually accumulating values from lower lanes via a loop of `warp_read_lane`.
```cpp
Float prefix = 0.f;
for (uint i = 0; i < warp_lane_id(); i++) {
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
- **Verify on multiple backends.** Warp intrinsics lower to different hardware ops (WaveIntrinsics on DX, subgroup ops on Vulkan/SPIR-V, `__shfl*` on CUDA). Re-run the same correctness cases — including tail/partial-item sizes that exercise idle warps and boundary lanes — on at least two backends after the change.

---

## 4. Shared Array (Workgroup Memory) Optimization

Warp collectives (section 3) only communicate *within one warp*. When cooperation must span the **whole thread block** (multiple warps), or you need persistent per-block scratch, arbitrary cross-thread indexing, or block-local privatization of a global atomic, use a **shared array** (`Shared<T>` / `$shared<T>`). Shared memory is on-chip and orders of magnitude faster than global memory, so staging data there once and reusing it, or aggregating locally before touching global memory, is a core optimization. (For the reverse direction — replacing *lane-local* shared memory with warp intrinsics — see section 3.7.)

### 4.1 API (`include/luisa/dsl/shared.h`, `include/luisa/dsl/sugar.h:105`)

| DSL | Description |
|---|---|
| `Shared<T> s{n}` / `$shared<T> s{n}` | Allocate `n` elements of `T` in workgroup memory. Must be constructed **inside** the kernel/callable body (uses `FunctionBuilder::current()`). |
| `s[i]` | Reference access (read or write); `i` must be an integral expr. |
| `s.read(i)` / `s.write(i, v)` | Explicit read / write helpers (alias for `s[i]`). |
| `s.atomic(i).fetch_add(v)` / `.compare_exchange(e, v)` / ... | Atomic ops on a shared slot. Available for scalar/vector element types (disabled for custom structs). |
| `s.size()` | Element count. |
| `new Shared<T>{n}` | Heap-allocate so helper classes can *own* shared scratch (`Shared<T>` is move-only, non-copyable). Lifetime is tied to the enclosing kernel's function builder. See `WarpReduce` in `test_decoupled_look_back.cpp`. |

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

`test_async_copy.cpp` fills a shared staging buffer with `async_copy(...)` (thread 0 issues the copy, then `sync_block()` before consumers read). `test_mipmap.cpp` writes 2×2 block averages into `Shared<float3>` and reduces level-by-level with a `sync_block()` between levels.

### 4.7 Correctness & Performance Rules

1. **Construct inside the kernel body.** `Shared<T>` needs `FunctionBuilder::current()`; declaring it outside a kernel/callable is invalid.
2. **Barrier discipline.** A `sync_block()` is required (a) after initializing/filling shared before other threads read, and (b) between the read and write-back phases of each reduction step. Unlike warp collectives, **shared memory is NOT self-synchronizing across warps**.
3. **Size to the block.** Match the array length to `set_block_size(...)`; use a power-of-two block for the halving tree reduction, and pad out-of-range lanes with the reduction identity (e.g. `0.f` for sum) — see the `$if (id < size) {...} $else { value = 0.f; }` guards in `test_softmax.cpp`.
4. **Register-then-write.** In tree reductions, compute into a `Var`/register and write back only after a barrier to avoid RAW/WAR hazards.
5. **Move-only ownership.** `Shared<T>` cannot be copied; store `Shared<T> *` (via `new`) when a helper class must hold shared scratch.
6. **Prefer warp collectives when they suffice.** Shared memory costs a barrier and on-chip capacity; only reach for it when cooperation exceeds one warp or needs privatization/staging/arbitrary indexing.

---

## 5. Thread-Group (Block) Size Selection and Occupancy

Choosing `set_block_size(x, y, z)` is one of the most impactful tuning decisions for a kernel. The right size depends on whether the kernel is **memory/IO-bound** or **compute-bound**, because group size directly controls **occupancy** — how many thread groups (blocks) can co-reside on one compute unit (CU/SM) — which in turn determines how well the GPU hides latency by switching between waves/warps.

### 5.1 Hardware Constraints

A thread group cannot be split across CUs: every wave/warp of a group must fit into **one CU's resources simultaneously** before the group can begin executing. The binding limits (DX11+ / modern GPUs):

| Limit | Value |
|---|---|
| Max threads per group | 1024 (X×Y×Z) |
| Max shared memory (LDS) per group | 32 KiB (DX11-era floor; backend/device-dependent on Vulkan/DX12 — verify against the target API) |
| Hardware wave/warp size | 64 (AMD), 32 (NVIDIA), 8 (Intel) |

Never hardcode the wave width: query `device.compute_warp_size()` on the host and `warp_lane_count()` on the device. Because all waves of a group must fit simultaneously, a larger group can reduce the number of resident groups per CU, lowering occupancy and the GPU's ability to hide memory latency.

### 5.2 Memory/IO-Bound Kernels

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

---

## 6. Hardware Mapping

| GPU Backend | Warp/Lane Terminology | Native Width |
|---|---|---|
| CUDA | Warp (32 lanes) | 32 |
| HIP | Wavefront (32/64 lanes) | 32 or 64 |
| Vulkan | Subgroup | Varies (usually 32 or 64) |
| DirectX | Wave | 32 or 64 |
| Metal | SIMD group | 32 |

Always query `device.compute_warp_size()` on the host and `warp_lane_count()` on the device rather than hardcoding 32.

---

## 7. Rules of Thumb

1. **Prefer warp collectives over shared memory.** `warp_active_sum`, `warp_active_max`, `warp_prefix_sum` compile to single hardware instructions (e.g. `__shfl_xor_sync` on CUDA, `OpGroupNonUniformFAdd` on SPIR-V). No barrier needed.

2. **Set warp size explicitly** when using warp collectives: `set_warp_size(device.compute_warp_size())` inside the kernel lambda.

3. **Don't mix warp and block assumptions.** `warp_active_sum` only reduces within the current warp. If you have multiple warps per block, use a two-level reduction (warp → shared → block).

4. **`sync_block()` is NOT needed between warp collectives** within the same warp. Warp ops are guaranteed complete for the calling lane immediately.

5. **Divergence matters.** Lanes that don't execute the warp collective call are excluded. Use this for conditional participation (section 3.5).

6. **`warp_prefix_sum` is exclusive** (not inclusive). Lane 0 always gets 0 (for sum) or 1 (for product).

7. **Vector types work.** All warp collectives accept `float2`, `float3`, `float4`, `int2`, etc. The operation applies component-wise.

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
CommandList cmdlist = CommandList::create();
cmdlist << upload_command
        << dispatch_a
        << dispatch_b
        << download_command;
stream << cmdlist.commit() << synchronize();
```

All commands execute in FIFO order on the GPU, exactly as if they were submitted individually — but with a single driver round-trip instead of many.

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
3. **Create and fill** — call `CommandList::create()` once at the start of the group, then append commands with `<<`.
4. **Commit once** — `stream << cmdlist.commit()` submits the batch; follow with a single `synchronize()` if host-readback is needed.
5. **Verify correctness** — ensure the sequence of operations inside the CommandList matches the dependency order (commands execute in FIFO order on the GPU).

### 8.5 Performance Impact

Batching N separate `stream << cmd` submissions into one `CommandList` reduces:
- **Driver submission overhead**: Each stream submission incurs a kernel transition / command-queue flush cost. With CommandList, that cost is paid once per batch.
- **Host-device synchronization points**: A single `commit() + synchronize()` replaces N pairs of `stream << ... << synchronize()`.

In practice, replacing 5+ stream submissions per iteration with a single `CommandList::create()` → `commit()` can yield measurable wall-clock speedups in offline rendering or training-data export loops, where the CPU-side submission overhead is a meaningful fraction of the iteration time.

### 8.6 Comparison with Other Optimizations

| Optimization | Scope | Impact |
|---|---|---|
| Warp collectives (section 3) | GPU kernel — replaces shared memory and atomics | Reduces latency/contention within a warp |
| Shared memory privatization (section 4.3) | GPU kernel — aggregates block atomics | Reduces global atomic contention from O(block_size) to O(1) per block |
| **CommandList batching** (this section) | **Host submission — batches stream commands** | **Reduces driver overhead from O(N) to O(1) per iteration** |

CommandList batching is orthogonal to kernel-level optimizations. Apply both: optimize the kernel with warp/shared-memory techniques, then batch the host submissions for maximum throughput.

### 8.7 Key Rules

1. **One CommandList, one commit, one sync.** Create a single `CommandList` for a group of dependent commands, commit it once, and synchronize once rather than submitting each command separately.
2. **FIFO ordering preserved.** Commands execute in record order — no need for explicit barriers between kernel dispatches and buffer copies inside the same CommandList (GPU pipeline dependencies are handled automatically).
3. **Don't reuse a committed CommandList.** After `commit()` the list is consumed; create a fresh one for the next batch.
4. **Prefer CommandList over chaining on `stream <<`.** Batched submission is more efficient than long chains of `stream << a << b << c << synchronize()` because it reduces internal queue flushes.
5. **Combine with kernel optimization.** Host-side batching and kernel-level warp/shared-memory optimization are complementary — use both.

### 8.8 Async Callbacks — Replacing `synchronize()` with Non-Blocking Completion

`CommandList` provides two callback hooks that decouple host work from GPU execution:

```cpp
// Runs AFTER all GPU commands in this CommandList finish.
cmdlist.add_callback([](auto &&... captured) noexcept {
    // GPU work is done — safe to read back buffers, write files, etc.
});

// Runs when the CommandList is committed/destructed, BEFORE GPU starts.
cmdlist.add_dtor_callback([](auto &&... captured) noexcept {
    // Host-side cleanup of resources no longer needed.
});
```

#### 8.8.1 Understanding the Two Callbacks

| Callback | When it fires | GPU status | Typical use |
|---|---|---|---|
| `add_dtor_callback` | At `commit()` (or when `Commit` object is destroyed) | **Not started yet** | Release temporary host buffers, close files, or free staging memory that was only needed to construct the commands. |
| `add_callback` | After all GPU commands in the list have completed | **Done** | Read back downloaded buffers, write output files, signal host work queues, or launch dependent host tasks. |

**Critical difference:** `add_dtor_callback` runs *before* GPU execution begins — it is not a completion callback. Only `add_callback` guarantees GPU work is finished.

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
    auto host_buf = std::make_shared<std::vector<float>>(size);
    CommandList cmdlist = CommandList::create();

    // Record GPU commands (upload, dispatch, download into host_buf)
    cmdlist << upload << kernel.dispatch(...) << download(host_buf->data());

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
auto data = std::make_shared<std::vector<float>>(size);
cmdlist.add_callback([data]() noexcept { /* safe */ });

// ✅ Move semantics for unique resources
auto data = std::make_unique<std::vector<float>>(size);
cmdlist.add_callback([data = std::move(data)]() noexcept { /* safe */ });

// ❌ Capturing raw pointers or references to stack/local variables is UAF.
float *raw = ...;
cmdlist.add_callback([raw]() noexcept { /* DANGER: raw may be dangling */ });
```

`add_dtor_callback` has the same ownership rules, despite running earlier — it still fires after `commit()` returns, so stack variables captured by reference would be invalid.

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
2. **`add_dtor_callback` fires before GPU starts** — use only for host-side cleanup, never for reading back GPU results.
3. **Always capture by value** (shared_ptr, unique_ptr, or copy). Raw pointers and references to stack variables are dangling by the time the callback runs.
4. **One final `synchronize()` is still needed** at the end of a pipeline to ensure the last iteration's callbacks have fired before the program exits.
5. **Callbacks execute on an internal worker thread** — they should not throw, block on the GPU, or perform GPU API calls on the same stream.

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

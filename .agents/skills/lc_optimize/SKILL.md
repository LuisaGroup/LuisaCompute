---
name: lc_optimize
description: Optimize LuisaCompute DSL kernels using warp/wave primitives, shared-memory aggregation, and block-level collectives. Use when kernels bottleneck on atomics, reductions, or inter-thread communication.
---

# LuisaCompute DSL Kernel Optimization Guide

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

---

## 4. Hardware Mapping

| GPU Backend | Warp/Lane Terminology | Native Width |
|---|---|---|
| CUDA | Warp (32 lanes) | 32 |
| HIP | Wavefront (32/64 lanes) | 32 or 64 |
| Vulkan | Subgroup | Varies (usually 32 or 64) |
| DirectX | Wave | 32 or 64 |
| Metal | SIMD group | 32 |

Always query `device.compute_warp_size()` on the host and `warp_lane_count()` on the device rather than hardcoding 32.

---

## 5. Rules of Thumb

1. **Prefer warp collectives over shared memory.** `warp_active_sum`, `warp_active_max`, `warp_prefix_sum` compile to single hardware instructions (e.g. `__shfl_xor_sync` on CUDA, `OpGroupNonUniformFAdd` on SPIR-V). No barrier needed.

2. **Set warp size explicitly** when using warp collectives: `set_warp_size(device.compute_warp_size())` inside the kernel lambda.

3. **Don't mix warp and block assumptions.** `warp_active_sum` only reduces within the current warp. If you have multiple warps per block, use a two-level reduction (warp → shared → block).

4. **`sync_block()` is NOT needed between warp collectives** within the same warp. Warp ops are guaranteed complete for the calling lane immediately.

5. **Divergence matters.** Lanes that don't execute the warp collective call are excluded. Use this for conditional participation (section 3.5).

6. **`warp_prefix_sum` is exclusive** (not inclusive). Lane 0 always gets 0 (for sum) or 1 (for product).

7. **Vector types work.** All warp collectives accept `float2`, `float3`, `float4`, `int2`, etc. The operation applies component-wise.

8. **Logic warp size.** You can logically group lanes (e.g. 4 groups of 8 within a 32-lane warp) using `lane % kGroupLanes` and `lane / kGroupLanes` arithmetic. Use `warp_read_lane` to communicate across groups.

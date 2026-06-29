---
name: lc_optimize
description: Optimize fixed-index atomic operations in LuisaCompute DSL kernels by replacing contended global atomics with shared-memory aggregation. Use when kernels bottleneck on global buffer.atomic(idx).fetch_add or similar.
---

# Replace Contended Global Atomics with Shared Memory

When many threads atomically operate on the same fixed global index, the memory location becomes a hotspot. Reduce contention by aggregating per-block in shared memory first, then issuing one global atomic per block.

## Pattern

**Before:** every thread hits the same global address.

```cpp
Kernel1D before = [](BufferInt counter) noexcept {
    set_block_size(256u, 1u, 1u);
    counter.atomic(0u).fetch_add(1);
};
```

**After:** aggregate inside the block, then one thread writes the partial sum globally.

```cpp
Kernel1D after = [](BufferInt counter) noexcept {
    set_block_size(256u, 1u, 1u);
    Shared<int> shared{1u};
    shared[0u] = 0;
    sync_block();

    // Per-block atomic accumulation (low contention)
    shared.atomic(0u).fetch_add(1);
    sync_block();

    // Single global atomic per block (from thread 0)
    $if (thread_x() == 0u) {
        counter.atomic(0u).fetch_add(shared.read(0u));
    };
};
```

## Why This Works

- Global atomic operations on a single address serialize across all blocks.
- `Shared<T>` lives in fast per-block memory; contention is limited to threads inside one block.
- Emitting one global atomic per block reduces global pressure by the block size factor.

## Generalization

For operations other than `fetch_add`:

1. Initialize one or more `Shared<T>` slots.
2. Have each thread update the shared slot with the local atomic or plain operation.
3. `sync_block()` to ensure visibility.
4. Let thread 0 (or one lane per warp) apply the reduced/shared result to the global buffer.

Use `warp_active_sum` / `warp_active_all` / `warp_read_first_active_lane` when the reduction can be expressed as a warp or block collective instead of a shared atomic.

## Barriers Are Required

Always `sync_block()` before thread 0 reads values that other threads wrote to `Shared<T>`. Without the barrier the read may see stale data.

## Applicability

- Works for any fixed-index atomic: `fetch_add`, `fetch_sub`, `fetch_max`, `fetch_min`, `compare_exchange`, etc.
- Most beneficial when the dispatch size is much larger than the block size.
- For fully general reductions over a buffer, consider scan or warp primitives instead of a single shared slot.

# Shared-capacity auxiliary pipelines

## Root cause and contract

The original auxiliary-work protocol has one count per work object and requires
one dispatch to consume everything. It cannot represent independently scheduled
stages of a side path. Registering each stage as a separate work object is not a
valid workaround when those stages share an allocation: admission sees only the
entry queue, while items parked at later stages still own slots.

Minimal capacity counterexample: capacity C = 8, entry count q0 = 0, later-stage
count q1 = 7, and a main producer of n = 2 invocations with emission bound b = 1.
Entry-only admission accepts 2 <= 8 - 0, but actual occupancy can become 9 > 8.
Treating all seven items as a single consumable queue avoids overflow only by
losing independently scheduled stages and their queue cardinalities.

The repaired protocol groups stages under one capacity owner:

    L = sum(q[i]) <= C
    admit a main producer iff uint64(n) * b <= C - L

A stage transition moves one item from i to j, including j = i, or terminates
it. It never clones an item. Therefore L is unchanged by a transition and
decreases by one at termination. The full producer bound is admitted before
dispatch; its actual publications increase L by at most n * b. This proves the
invariant inductively without assumptions about stage arrival distributions.

Every stage competes separately against main continuations by observed count.
Capacity-blocked producers instead request a non-empty drain stage from their
capacity owner; the default uses the first non-empty stage. This separates
capacity accounting, normal greedy priority, and capacity-release priority.
The scheduler validates the observed total against the sum of stage counts.
All host observations follow same-stream device readbacks. No stage dispatch
may secretly create additional items.

`prepare_for_producer` permits storage-only compaction before an admitted main
dispatch. It does not change counts or item semantics. Payload storage and
allocation remain client-owned; this change adds no shader IR operation and
does not change scalar/vector initialization or inlining policy.

Existing single-stage clients retain their original default methods, dispatch
semantics, names, and statistics. Multi-stage clients use one statistics entry
per stage, not their total live occupancy as an executed invocation count.

## Permanent regression

`src/tests/unit/coro/test_coro_wavefront_auxiliary_pipeline.cpp` uses 67 logical
main paths and an eight-slot auxiliary pool, with sparse and dense publications,
a bypass edge, a self-loop, and terminal release. It checks all per-item visits,
payload integrity, independently materialized token counts, capacity, forced
drain priority, main continuation progress with live auxiliary items, stage
statistics, and reset/reuse. Both main-frame layouts and incremental/non-
incremental accounting are exercised. The test contains no renderer or CPU
rendering reference.

The baseline compatibility dispatch lets the test finish against the old
scheduler, then reports aggregate dispatch and absent per-stage statistics as
assertion failures instead of hanging or faulting the GPU.

## Validation

Evidence directory: `/var/tmp/psycles-independent-shadow-cGjf52`.

- Red HIP run: 32 expected failures, all in the four scheduler-protocol
  assertions across eight executions; payload and visit assertions passed.
- Green pipeline: 2,272 assertions on each of HIP, fallback, and Vulkan.
  Vulkan ran with all three native-XIR / no-DXC guards enabled.
- Existing complete wavefront suite: 30 tests on HIP and fallback.
- Existing all-schedulers suite: 20 tests on HIP and fallback.
- Full original Psycles module built with `--parallel 32`.
- Original-module HIP CTest: 156/156 passed in 214.36 seconds.
- The isolated candidate based on origin/next fa044f146 compiled with its
  own headers and passed the new test on HIP and fallback; it also passed HIP
  with `LUISA_CORO_WAVEFRONT_VERIFY_QUEUES=1`.

Candidate validation linked the active workspace's Luisa shared libraries;
it was not a clean rebuild of every backend or of the unrelated pending XIR
changes. The candidate contains only these protocol/header changes, the new
test, its CMake registration, and this note. The active dirty compiler work
and the Psycles gitlink are deliberately outside this commit.

These are scheduling/correctness results, not a renderer speedup claim.

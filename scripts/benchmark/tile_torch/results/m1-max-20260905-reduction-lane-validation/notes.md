# Consecutive-worker reduction layout checkpoint

September 5, 2026; Apple M1 Max; FP32; TIRx Metal subgroup-reduction family.

## Implementation and semantics

This checkpoint adds a generic ownership-layout dimension, not an RMSNorm
special case or a new Tile DSL entity. For W collaborating workers and V
consecutive elements per worker:

```text
logical element i = (chunk * W + worker) * V + element
private slot     = chunk * V + element
0 <= worker < W; 0 <= element < V; valid iff i < N

Example, W=4 and V=2:
             worker 0   worker 1   worker 2   worker 3
chunk 0       0, 1       2, 3       4, 5       6, 7
chunk 1       8, 9      10,11      12,13      14,15
```

The inverse is unique: `element=i%V`, `worker=(i/V)%W`,
`chunk=i/(W*V)`. Producer stores, reducer loads and output consumers use the
same map. Existing same-logical-element ownership, purity, noalias and
non-escape checks remain prerequisites; the new layout does not grant
permission for cross-worker local reads. Private storage is bounded by
`floor(N/(W*V))*V + min(N%(W*V), V)` slots per materialized Tile. The bound
includes the final worker-local prefix and is enforced before profitability
scoring. Backend cost policies receive V and the recomputed resource features.

V is 1, 2, 4 or 8 and defaults to 1. It changes worker ownership and FP32
addition order under the existing explicit reduction-tree permission. U,
the separate stripe-unroll factor, retains each selected worker's recurrence
order. The generated loops separate complete packs from the single guarded
partial pack. This enables contiguous accesses but is not a vector-ISA claim.
No default cost coefficient or automatic V heuristic has been changed.

The source comparison used the installed PyTorch commit, not a guessed latest
implementation: [RMSNorm Metal kernel](https://github.com/pytorch/pytorch/blob/08187d9e0fba026dc8217405802ab5381dc88d90/aten/src/ATen/native/mps/kernels/RMSNorm.metal)
and [host dispatch](https://github.com/pytorch/pytorch/blob/08187d9e0fba026dc8217405802ab5381dc88d90/aten/src/ATen/native/mps/operations/RMSNorm.mm).
It uses four consecutive input elements per worker and also rereads input in
the output phase. Input rereading alone is therefore not evidence of our
structural gap. Its host launch sizing also differs from this bridge's
current bounded subgroup candidate family.

## Measurement contract

The new explicit `--tuning-metric gpu-control` chooses candidates using
no-counter GPU command-buffer throughput. Default JIT selection is still
host-wall throughput. Missing GPU controls reject a candidate; compute-pass
probes never substitute for them. Model regret uses the selected objective.
Every winner is recaptured and remeasured, and a separate frozen-parameter,
position-balanced replay is required for a performance claim.

GPU command-buffer intervals include GPU work and gaps inside the buffer;
they are not individual-kernel timestamps. E2E batch and synchronized
single-call latency remain separate uninstrumented phases. The earlier
[observer audit](../m1-max-20260905-device-timing-counter-control/notes.md)
explains why instrumented pass times are retained only as diagnostics.

## Verification checkpoint

- Full selected CMake tree build succeeded before tests.
- Complete Tile CTest: **31/33 passed**. The two failures are the existing
  cooperative/memory source assertions requiring `mem_flags(3)` while the
  user's pre-existing, unowned `cooperative.cpp` edit emits `2`. Neither the
  edit nor the assertions were modified. All numerical checks, the new lane
  mapping tests and CPU/Metal execution tests passed. See [full log](tests.log).
- The new mapping test executes 24 softmax configurations: V=2/4/8 across
  N=1/31/33/127/128/129/257/4103, five rows, U=3, packed short rows and
  eight-subgroup wide rows. It covers shared exp storage, max/add collectives,
  private-slot accounting, policy metadata and complete output validation.
  Invalid widths, absent numerical permission, resource overflow and an exact
  reduction request on an element-only kernel fail closed.
- **82/82 Python contracts passed**, including lane realization/replay,
  Cartesian search budgeting, explicit GPU-vs-host objective choice, missing
  control rejection and fresh winner measurement.
- Project clangd checks passed for the four changed translation units;
  the shared planner header was checked through those consumers.

Performance results are not inferred from these correctness checks. The
reference, finite search and independent replay are recorded separately.

# Consecutive-worker reduction layout checkpoint

September 5, 2026; Apple M1 Max; FP32; TIRx Metal subgroup-reduction family.

## Result: three local wins; wide throughput and dispatch remain separate problems

**Assessment: share with caveats.** A four-round, position-balanced,
same-binary replay validates 64 complete native/Torch outputs. Compared with
the current V=1 automatic mapper, the frozen joint layout/worker choices
improve the first three RMSNorm shapes in every paired GPU-throughput round.
The largest shape is flat/noisy; it is not a demonstrated win. This is a
four-shape, one-device result, not a universal reduction policy or a claim
that all lowering paths now beat Torch/MPS.

| RMSNorm shape | Selected workers / V | Reference GPU µs/op | Candidate GPU µs/op | Paired speedup median [range] | Candidate-run Torch GPU µs/op |
|---|---:|---:|---:|---:|---:|
| 1×127 | 32 / 4 | 3.421 | 2.829 | 1.208× [1.114, 1.231] | 4.396 |
| 17×257 | 256 / 4 | 4.748 | 3.204 | 1.469× [1.404, 1.530] | 3.471 |
| 64×4096 | 128 / 4 | 10.542 | 9.216 | 1.156× [1.104, 1.228] | 9.172 |
| 1024×4096 | 256 / 2 | 68.942 | 68.148 | 1.015× [0.976, 1.031] | 68.891 |

GPU values are no-counter command-buffer intervals divided by that phase's
own batch size, not individual-kernel timestamps. Table values are medians
of per-round p50s; gains are medians of paired round ratios. Ranges are
observed min–max, not confidence intervals. Candidate/native versus Torch
paired GPU ratios are respectively 0.645, 0.923, 1.005 and 0.992. The two
wider shapes are approximately at Torch throughput, not consistent wins.

End-to-end batch throughput gains are 1.158×, 1.493×, 1.169× and 0.992×.
Synchronized single-call latency still tells a different story:

| RMSNorm shape | Candidate GPU single µs | Candidate E2E single µs | Torch GPU single µs | Torch E2E single µs |
|---|---:|---:|---:|---:|
| 1×127 | 4.896 | 210.021 | 7.771 | 223.500 |
| 17×257 | 5.708 | 217.730 | 7.146 | 223.188 |
| 64×4096 | 12.417 | 270.500 | 12.354 | 250.416 |
| 1024×4096 | 77.625 | 348.812 | 97.229 | 333.729 |

These phases are sampled separately; subtracting their medians would not
identify CPU cost. Warm measurements exclude compilation/transfers. Native
outputs are preallocated; the measured eager Torch RMSNorm returns an output
allocation on each invocation. No `torch.compile`, MPSGraph substitution,
reduced precision or fast-math override is used.

The reference already contains the new full/tail splitting at V=1. Thus this
replay compares ownership/worker choices **within** the new compiler, not
the entire compiler change against the previous commit. For 17×257 the map
also changes from eight rows packed into each of three threadgroups to one
row per group (17 groups); the gain cannot be attributed solely to V=4.
For 64×4096 the cooperating width changes from eight to four subgroups.

## Evidence and what remains

- [Reference](../m1-max-20260905-reduction-lane-reference/results.json):
  four shapes, V=1 automatic mapping, eight complete outputs valid.
- [Finite GPU-objective search](../m1-max-20260905-reduction-lane-search/results.json):
  12 candidates per shape, 48/48 numerically valid trials plus four fresh
  winner measurements: 104 complete outputs valid. W is automatic/32/128/256;
  V is 1/2/4; U=1. Automatic choices can duplicate exact realizations. No
  tuning score is presented as the replay result.
- [Independent frozen replay](../m1-max-20260905-reduction-lane-replay/results.md):
  four rounds, nine samples, 30 ms host batch target, 200 ms warmup, balanced
  A/B and native/Torch positions; 64 complete outputs valid. Recorded binary,
  timing helper and adjacent library hashes remained unchanged.
- [Generic V=4 operator coverage](../m1-max-20260905-reduction-lane-operators/results.json):
  sum, softmax, RMSNorm, LayerNorm, residual LayerNorm and cross-entropy on
  these four shapes: 48/48 complete outputs valid. This five-sample,
  10 ms / 100 ms smoke retains GPU/E2E data but is not an independent
  cross-operator speed ranking or evidence that V=4 is optimal for them.
- [Independent audit script](verify.py) recomputes the headline statistics
  from raw control samples and checks all 224 recorded validated outputs,
  source SHA256 values, metric scopes and frozen replay layouts.

All four directories use the same measured executable SHA256
`3cc9b62e7f8129e6ab23dea1cb9c96749d12248c12112409014eb56a4344279e`.
The implementation was committed as `a88431c5c`; reference/search metadata
records the preceding HEAD with its then-uncommitted implementation. Every
binary includes the pre-existing local `mem_flags(2)` edit; these are not
clean-checkout build results. That edit remains unowned and unsubmitted.

The default V=1 and cost coefficients remain unchanged. The scalar-round
prior does not yet price the loss of active groups from short-row packing
or the issue/memory behavior of V. Within this finite search its model regret
is 15.3%, 38.7%, 15.4% and 0.5% respectively; these are diagnostic search
ratios, not held-out cost-model accuracy. Next work is to expose active-group
and live-lane features to backend policies, add shape-derived collaborating
widths, and validate those predictions across operators. Wider RMSNorm needs
memory/issue investigation; single-call Runtime overhead remains distinct.

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

Reproduce from the repository root (use new empty output directories):

```sh
uv run --no-project --python 3.13 --with torch --with numpy python \
  scripts/benchmark/tile_torch/run.py \
  --native cmake-build-tirx/bin/benchmark_tile_tirx --output NEW_SEARCH \
  --backends metal --operations rmsnorm \
  --row-shapes 1x127,17x257,64x4096,1024x4096 --metal-subgroup-reductions \
  --tune-group-threads 0,32,128,256 --tune-reduction-lane-elements 1,2,4 \
  --tuning-metric gpu-control --max-tuning-candidates 12 \
  --samples 7 --sample-ms 20 --warmup-ms 150 --capture-sources \
  --metal-device-timing cmake-build-tirx/bin/libluisa-benchmark-metal-timing.dylib

uv run --no-project --python 3.13 --with torch --with numpy python \
  scripts/benchmark/tile_torch/repeat.py \
  --reference scripts/benchmark/tile_torch/results/m1-max-20260905-reduction-lane-reference/results.json \
  --candidate scripts/benchmark/tile_torch/results/m1-max-20260905-reduction-lane-search/results.json \
  --native cmake-build-tirx/bin/benchmark_tile_tirx --output NEW_REPLAY \
  --operations rmsnorm --rounds 4 --samples 9 --sample-ms 30 --warmup-ms 200 \
  --compiler-artifact cmake-build-tirx/bin/libluisa-tile-bridge-tirx.dylib \
  --metal-device-timing cmake-build-tirx/bin/libluisa-benchmark-metal-timing.dylib \
  --capture-sources

python3 scripts/benchmark/tile_torch/results/m1-max-20260905-reduction-lane-validation/verify.py
```

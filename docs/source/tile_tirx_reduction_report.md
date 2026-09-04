# TileIR to TIRx Metal reductions: design and evidence

Status: implemented, opt-in, and exercised on Apple M1 Max as of September 5,
2026. This document specifies the current FP32 Metal subset, its proof
obligations, mapping algebra, finite solver, generated resource plan and
performance evidence. It deliberately separates measured facts from the
general Tile-language design.

## Outcome at a glance

The old TIRx Metal path mapped one logical row program to one scalar worker.
A width-4096 RMSNorm therefore performed a serial 4096-element recurrence in
one thread; softmax could additionally allocate one private `float[4096]` per
thread. Launch-width tuning could not repair that execution structure.

The new lowering proves the reduction program, then maps one logical program
to one or more 32-lane SIMD groups. It packs independent short programs into a
threadgroup, cooperates across up to eight SIMD groups for wide programs, and
compacts eligible compiler-owned Tiles to worker-private stripes. The source
C++ kernel and logical TileIR remain unchanged.

On the saved 12-case Apple M1 Max cohort, all complete FP64 checks pass and
Tile/TIRx is faster than eager PyTorch MPS in every row. Tile/Torch ranges from
0.124× to 0.902× in synchronized device-resident host-wall throughput. Sum and
softmax use preallocated output on both sides; PyTorch's functional RMSNorm
allocates its returned output inside timing, so that comparison is explicitly
qualified below. A separate four-round balanced RMSNorm A/B measures a
21.19×--49.87× speedup over the old TIRx realization while using the same
current binary for both variants; that native-to-native result is unaffected
by the PyTorch output policy.

These results close one identified structural gap. They do **not** establish
all-operator, all-shape, low-precision, cross-device or pure-kernel parity.

```{figure} ../_static/tile/tirx-subgroup-reduction.svg
:alt: A logical TileIR reduction passes fail-closed proofs and a bounded cost solver before becoming either packed independent SIMD groups or several cooperating SIMD groups, with memory resources planned separately.
:width: 100%

The lowering first proves semantics and ownership, then searches legal
execution maps. Private and shared storage are consequences of that map, not
execution levels in the source language.
```

## 1. The frontend stays execution-first

The optimization needs no warp-specific DSL object and leaks no
`mma_team`-style hardware role into an ordinary reduction. This is the complete
source for a row softmax:

```cpp
auto definition = tile_kernel(
    "softmax",
    [](TensorView<const float, 2> input,
       TensorView<float, 2> output) {
        auto rows = axis("rows", input.extent<0>());
        auto one = axis("one", 1);
        auto columns = axis("columns", input.extent<1>());

        for (auto &nest : parallel(shape(rows))) {
            auto origin = coord(nest.index(), 0);
            auto value = input[origin, shape(one, columns)];
            auto shifted =
                exp(value - reduce(value, columns, maximum));
            output(origin, shape(one, columns))
                .store(shifted / reduce(shifted, columns, add));
        }
    });
```

The syntax preserves the language decisions made elsewhere:

- `parallel(shape(rows))` defines independent logical row programs. It does
  not choose threads, warps, SIMD groups or memory.
- `nest.index()` is a logical execution coordinate, not a memory access.
- `input[origin, shape(...)]` loads a Tile; `output(...).store(...)` makes the
  memory effect explicit.
- `reduce` is a semantic algebraic region/library operation. It can become a
  serial fold, tree or target collective only under its numerical contract.
- `exp`, subtraction and division remain Tile-level elementwise operations.

The compile option `metal_subgroup_reductions` is explicit. For FP32 addition,
enabling it is also the numerical permission to replace the reference left
fold with a tree order. A user who requires the exact serial recurrence keeps
the option disabled or spells an ordered `serial` computation. The current
surface does not yet expose richer accuracy/deterministic-tree policies.

```{figure} ../_static/tile/execution-to-memory.svg
:alt: Logical execution coordinates and memory coordinates are related by explicit maps rather than by treating a memory tile as a hardware execution level.
:width: 88%

The same logical row program can be remapped without changing its Tensor
views. Conversely, several memory resources can be accessed by one execution
level without reversing the hierarchy.
```

## 2. Where the optimization lives

The path is intentionally split by responsibility:

```text
C++ Tile DSL
  -> mutable, typed Candidate TileIR
  -> shared structural TileIR-to-TIRx export
  -> target-independent TIRx simplification / semantic annotations
  -> Metal subgroup reduction proof + finite planner
  -> standard scheduled TIRx (thread bindings, buffers, barriers, intrinsics)
  -> TVMx Metal code generation
  -> backend kernel/binary
  -> Runtime shader handle
  -> stream launch
```

Candidate TileIR remains the thin, transformable semantic layer. The Metal
mapper is a bridge-local target realization; it does not force subgroup
details into TileIR or manufacture a general Machine TileIR before one is
needed. `GroupPlan` is an inspectable decision record containing threads,
programs, subgroup factor, storage, synchronization and cost facts.

The implementation uses TVMx's C++ TIRx API directly. It emits no Python and
introduces no MLIR dependency. Reduction recognition is structural and typed.
The strings `simd_sum`, `simd_max` and `simd_min` appear only at the final Metal
intrinsic ABI boundary; operation selection is not performed by comparing
arbitrary operation names.

The same scheduled TIRx can be compiled by the standalone TVM route or
extracted as a typed `DeviceArtifact` for the ordinary Luisa Metal Runtime.
Backend compilation and launch ownership remain outside the bridge.

## 3. Semantic domain and fail-closed admission

Let a root logical parallel region contain `P > 0` independent programs. Its
body may contain reduction regions `r in R` and independent Tile-element
domains `d in D`. A reduction has a static positive extent `N_r`, FP32 state,
identity `z_r`, associative operation `op_r` and pure contribution expression
`f_r(e)`. An element domain has a static count `I_d`.

The current mapper admits only the following bounded subset:

| Requirement | Checked fact |
|---|---|
| Target and binding | Metal target, 32-lane subgroup, outermost automatic or explicit subgroup root |
| Kernel ABI | noalias arguments; read-only forwarding is re-proved before use |
| Reduction type | FP32 add, maximum or minimum |
| Canonical state | one local scalar carry, exact identity, one update, no extra observation |
| Contribution | pure typed expression; no carry/temporary escape or unknown call effect |
| Control | static unit-step serial source loop; no conditional containing the reduction |
| Execution nesting | no nested logical `parallel` inside the mapped program |
| Global effects | stores occur only inside a distributed independent-element domain |
| Manual resources | only compatible private constraints; unsupported placement rejects the path |
| Dynamic control | no while, break, continue, return, assert or opaque Tile primitive |

The structural exporter attaches a reducer-kind marker only after matching the
Candidate TileIR update. The Metal pass does not trust the marker by itself: it
rechecks the exact initializer, allocation, loop, update, operands, effects and
types in current TIRx. Every marked reduction must match exactly. An explicit
subgroup binding that cannot be realized is an error; automatic binding may
fall back to the reference path.

The option permits a floating-point tree order. The saved tests use finite
inputs and compare against an FP64 oracle within recorded tolerances. Complete
NaN payload, signed-zero and deterministic bitwise policies for max/min/add
remain future typed numerical contracts, not properties inferred from the
current performance tests.

## 4. Execution mapping algebra

For a candidate using `S` cooperating SIMD groups per logical program, define

```text
L = 32                         lanes per SIMD group
W = L * S                      workers per logical program
s = floor(thread / L)          SIMD-group coordinate
l = thread mod L               lane coordinate
w = L * s + l                  worker coordinate within a program
```

The planner considers `S in {1, 2, 4, 8}`, restricted by the target's maximum
thread count and shared-memory capacity. This is a finite realization set,
not a claim that four values form a complete universal schedule space.

### 4.1 One SIMD group per program: spatial packing

When `S = 1` wins under automatic planning, a threadgroup can contain `Q`
independent logical programs, where

```text
Q = min(P, floor(max_threads / 32))
blocks = ceil_div(P, Q)
threadgroup_threads = 32 * Q

p = block * Q + subgroup
e = 32 * chunk + lane
```

The final partial group is guarded by `p < P`. Each subgroup executes one
program and no shared partial is needed. Packing amortizes threadgroup setup
without inventing an order or dependence between the source `parallel`
instances.

For example, `P=17, N=257` chooses one subgroup per program but reports a
256-thread group: eight independent row programs are packed into its eight
subgroups. `reduction_subgroups_per_program=1` distinguishes this from one row
using all 256 workers.

### 4.2 Several SIMD groups per program: cooperative striping

When `S > 1`, one threadgroup owns one logical program:

```text
blocks = P
threadgroup_threads = W
p = block
e = W * chunk + w
```

Each worker folds its strided elements into a private scalar. A reduction then
uses this uniform protocol:

```text
worker-local serial fold
  -> simd_{sum|max|min} in every subgroup
  -> lane 0 writes shared partial[r, s]
  -> one threadgroup barrier
  -> lanes l < S read partial[r, l], other lanes use the identity
  -> uniform simd_{sum|max|min}
  -> every worker receives the program result
```

The first collective must execute uniformly; placing it under `lane == 0`
would be invalid. Only the shared write is lane-zero conditional. The second
collective also executes uniformly, so subsequent replicated element domains
can consume the result without a hidden cross-subgroup race.

The shared footprint is exactly

```text
shared_bytes(S) = |R| * S * sizeof(float),  when S > 1
shared_bytes(1) = 0.
```

The emitted barrier count is recorded as a static site count, not a dynamic
execution count. Cooperating subgroups report
`independent_subgroups=false`; the existing barrier-elision proof cannot
misclassify them as independent programs.

### 4.3 Coverage and ancestry

For every distributed domain of count `N`, the mapper enumerates

```text
chunk in [0, ceil_div(N, W))
e = W * chunk + w
execute iff e < N.
```

`e` is unflattened into the domain's named logical axes using their original
mixed-radix order. The root program coordinate `p` and every ancestor
coordinate are substituted independently of memory indexing. Therefore the
mapping covers each `(p,e)` exactly once and preserves the ancestor execution
prefix used by Tensor origins.

This is the concrete instance of the more general relationship

```text
ancestor execution coordinates + local element coordinate
    --layout correspondence--> logical Tensor coordinate
    --Tensor layout-----------> physical address.
```

The execution map changes the producer of the local element coordinate; it
does not reinterpret the Tensor layout.

## 5. Logical Tile materialization to physical worker storage

Softmax exposes the key execution/memory separation. The logical expression

```cpp
auto shifted = exp(value - reduce(value, columns, maximum));
output(...).store(shifted / reduce(shifted, columns, add));
```

has two consumers, so the shared structural lowering materializes `shifted`
once. Before execution mapping that logical object has `N` elements. Naively
allocating it after distributing rows but before distributing elements creates
`N` private values per worker.

The subgroup mapper may replace logical `T[N]` with worker-private
`T_worker[ceil_div(N,W)]`, but only after proving all of the following:

1. the object is a compiler-owned local FP32 materialization with one
   allocation and one defining store;
2. its expression is pure and does not read or expose the object recursively;
3. it has at least one load and no opaque pointer/variable escape;
4. for every store and load in its enclosing loop domain,
   `flatten(indices) == owner_coordinate` is provable; and
5. the owner coordinate is the same distributed logical element `e` used by
   that domain or reduction.

After mapping, the physical slot is `chunk = floor(e / W)`. A fixed,
permuted, cross-element or escaping access fails the equality proof and keeps
the original storage; it is never silently redirected to another worker's
slot.

For width 4096 with 256 workers, each worker owns only 16 FP32 values. The
unit test inspects generated Metal and requires `_worker_stripe[16]` while
rejecting the old per-thread `[4096]` form. This is a resource-planning result,
not a new source-level `Memory` declaration. Explicit manual `Memory` remains
for deliberate expert placement and still requires explicit `.store()`.

```{figure} ../_static/tile/reduction-model.svg
:alt: A semantic reduction domain is factored into execution participants and local slots while its reducer contract and result remain target independent.
:width: 86%

Reduction factorization changes participants and local slots. The semantic
contribution set, grouping map, identity and merge contract remain intact.
```

## 6. Cost model and finite solver

Legality is decided before profitability. Let `D` be the independent element
domains, `R` the reductions, `I_d` and `N_r` their counts, `S` the candidate
subgroups per program, `W=32S`, and `Q` the packed-program count (`Q>1` only
for automatic `S=1`). The bootstrap v1 score is

```text
rounds(S) = sum[d in D] ceil_div(I_d, W)
          + sum[r in R] ceil_div(N_r, W)

score(S) = alpha * rounds(S)
         + beta  * |R| * S
         + gamma / Q

alpha = 1, beta = 2, gamma = 16.
```

Each independent domain is rounded separately. Combining their total before
`ceil_div` would underprice separate softmax passes at a tail width.

The three terms represent worker stripe rounds, SIMD collective work and
amortized program/threadgroup setup. They are dimensionless priors, not
measured instructions, occupancy or nanoseconds. Ties retain the smaller `S`.

Before scoring, a candidate is rejected if it violates any of:

```text
S <= min(8, floor(target_max_threads / 32))
threadgroup_threads <= target_max_threads
shared_bytes(S) <= target_shared_memory
enumerated_widths <= max_thread_candidates
finite, nonnegative coefficients
supported exact thread constraint, when present.
```

There are at most four automatic candidates, so exhaustive enumeration is
clearer and more exact than integer programming or simulated annealing here.
Those methods become relevant only when later planners jointly choose tile
factorizations, layouts, atoms, storage versions and pipeline schedules over a
much larger discrete space.

`threads_per_group=0` means automatic choice. A nonzero value is an exact JIT
constraint; for this realization it fixes the number of cooperating workers
per program and must be a legal multiple of 32. It is not silently clamped and
does not mean “pack this many unrelated workers however convenient.”

### 6.1 Plans selected in the saved run

| Case | Threads/group | SIMD groups/program | Shared bytes | Private stripe/worker | Reductions | Model score |
|---|---:|---:|---:|---:|---:|---:|
| sum 1×127 | 32 | 1 | 0 | 0 | 1 | 23 |
| sum 17×257 | 256 | 1 | 0 | 0 | 1 | 14 |
| sum 128×1024 | 128 | 4 | 16 | 0 | 1 | 33 |
| sum 64×4096 | 256 | 8 | 32 | 0 | 1 | 49 |
| softmax 1×127 | 64 | 2 | 16 | 2 | 2 | 32 |
| softmax 17×257 | 256 | 1 | 0 | 9 | 2 | 42 |
| softmax 128×1024 | 128 | 4 | 32 | 8 | 2 | 64 |
| softmax 64×4096 | 256 | 8 | 64 | 16 | 2 | 112 |
| RMSNorm 1×127 | 64 | 2 | 8 | 0 | 1 | 24 |
| RMSNorm 17×257 | 256 | 1 | 0 | 0 | 1 | 22 |
| RMSNorm 128×1024 | 128 | 4 | 16 | 0 | 1 | 40 |
| RMSNorm 64×4096 | 256 | 8 | 32 | 0 | 1 | 64 |

`threads/group` is the whole threadgroup width. For the 17-row `S=1` plans,
256 threads mean eight independently packed row programs, not eight groups
cooperating on one row.

## 7. Staged/JIT tuning is the outer authority

The analytic solver chooses among physical maps for one captured shape. It
does not require a single universal capture. A staged outer tuner can simply
compile several concrete parameterizations:

```bash
uv run --no-project --python 3.13 --with torch --with numpy python \
  scripts/benchmark/tile_torch/run.py \
  --native cmake-build-tirx/bin/benchmark_tile_tirx \
  --output NEW_EMPTY_DIRECTORY \
  --backends metal --operations rmsnorm \
  --metal-subgroup-reductions \
  --tune-group-threads '32,64,128,256' \
  --samples 9 --sample-ms 60 --warmup-ms 100
```

For every width, the runner recaptures/JIT-compiles the ordinary host
configuration, validates the complete output, and retains failed candidates.
It selects by native throughput only among valid trials, then recaptures,
recompiles, validates and measures the winner again. The published row is that
fresh measurement, not the optimistic search minimum. Candidate order rotates
across shapes and a finite compile budget rejects oversized products.

This cleanly separates three questions:

| Layer | Role | Authority |
|---|---|---|
| Semantic proof | Is a tree/distribution legal? | compiler; never timing |
| Analytic cost model | Which legal maps should be shortlisted/defaulted? | replaceable target prior |
| Staged/JIT measurement | Which concrete valid candidate is fastest here? | correctness-checked measurement |

Measured data may calibrate `alpha`, `beta`, `gamma` or a richer device
profile. It must never make an illegal candidate legal.

## 8. Correctness and structural verification

The regression suite checks both generated structure and actual hardware
results:

- option/target/noalias contracts and absence of silent subgroup intrinsics by
  default;
- row sums at widths 127, 257, 1024 and 4096, including expected
  one/one/four/eight-group plans;
- softmax at `3×4096`, requiring two reductions, 256 threads, eight groups,
  64 shared bytes, a 16-scalar private stripe, `simd_max`, `simd_sum`, and every
  output element against FP64;
- minimum and maximum together at `7×1024`;
- uniform first collectives and guarded tails in generated Metal;
- CPU fallback behavior and unchanged reference paths; and
- benchmark/replay metadata, policy preservation, cooperating-subgroup facts
  and staged/JIT winner revalidation.

After the final per-domain cost correction:

```text
test_tile_tirx_planner:              5,890 assertions / 7 tests passed
test_tile_tirx_execution cpu:       33,064 assertions / 17 tests passed
Metal subgroup contract:                 4 assertions passed
Metal subgroup sum:                     60 assertions passed
Metal subgroup softmax:             12,300 assertions passed
Metal subgroup extrema:                 20 assertions passed
Python benchmark contract suite:        67 tests passed
```

The wider repository contains an unrelated local edit to Metal memory flags;
the four targeted subgroup hardware tests avoid attributing that user's change
to this feature. The complete Tile cohort was also run with the submitted
source value restored temporarily, then the user's local edit was restored.

## 9. Performance evidence

### 9.1 Current implementation versus eager PyTorch

The complete report is
{download}`Metal subgroup reductions <../../scripts/benchmark/tile_torch/results/m1-max-20260905-metal-subgroup-reductions/notes.md>`;
raw samples are in its adjacent `results.json`.

| Operator / rows×width | Tile µs | Torch µs | Tile/Torch |
|---|---:|---:|---:|
| sum 1×127 | 3.268 | 7.211 | 0.453× |
| sum 17×257 | 3.106 | 4.340 | 0.716× |
| sum 128×1024 | 3.387 | 5.604 | 0.604× |
| sum 64×4096 | 4.721 | 16.119 | 0.293× |
| softmax 1×127 | 3.578 | 26.111 | 0.137× |
| softmax 17×257 | 3.305 | 26.594 | 0.124× |
| softmax 128×1024 | 5.385 | 30.376 | 0.177× |
| softmax 64×4096 | 8.881 | 31.029 | 0.286× |
| RMSNorm 1×127 | 3.904 | 7.155 | 0.546× |
| RMSNorm 17×257 | 5.335 | 6.154 | 0.867× |
| RMSNorm 128×1024 | 6.673 | 8.707 | 0.766× |
| RMSNorm 64×4096 | 11.177 | 12.392 | 0.902× |

These are p50 warm synchronized host-wall times across 11 samples with
100 ms calibrated sample windows and 100 ms warmup. Inputs remain
device-resident and native outputs are preallocated. Torch sum/softmax use
preallocated `out=` storage; the public functional RMSNorm has no `out=`
overload, so its returned-output allocation remains inside the Torch warm
timing and is recorded per row. Capture, compilation, transfers and cold calls
are separately recorded. PyTorch is eager and no `torch.compile` path is
claimed.

### 9.2 Balanced causal A/B against the old lowering

The independent
{download}`RMSNorm replay <../../scripts/benchmark/tile_torch/results/m1-max-20260905-metal-subgroup-rmsnorm-replay/notes.md>`
uses one current executable for both variants and changes only the explicit
subgroup policy. It rotates case and implementation order over four rounds and
freshly captures/JIT-compiles every row.

| Rows×width | Old reference µs | New subgroup µs | Paired speedup median [range] |
|---|---:|---:|---:|
| 1×127 | 103.180 | 3.792 | 27.216× [24.924, 28.020] |
| 17×257 | 268.202 | 5.366 | 49.871× [49.180, 54.574] |
| 128×1024 | 144.082 | 6.805 | 21.192× [20.989, 21.207] |
| 64×4096 | 524.444 | 11.160 | 47.096× [46.344, 50.864] |

All 32 reference/candidate outputs pass. Ranges are observed paired-round
minima/maxima, not confidence intervals. The result demonstrates a structural
execution-mapping gain; it does not prove the chosen map globally optimal.

## 10. What this closes, and what remains

This work closes the specific defect “logical reduction hierarchy is exported
but mechanically scalarized on Metal” for the admitted FP32 row-program
subset. It also demonstrates the intended architecture:

- execution structure is primary;
- execution distribution is a target-chosen map, not a source memory level;
- resource layout follows a proved ownership correspondence;
- a thin mutable semantic IR can feed TVMx without becoming a serialization
  format;
- the target bridge can add specific analyses/passes incrementally; and
- finite analytic planning and staged/JIT measurement compose naturally.

The next honest milestones are:

1. add typed reduction policy for deterministic tree shape, accuracy, NaN and
   signed-zero behavior;
2. extend the atom catalog to FP16/BF16 and pair/tuple reducers such as
   Welford, argmax and online attention state;
3. share target-independent reduction/ownership facts between the TIRx and XIR
   bridges rather than re-deriving them from target IR;
4. calibrate the shortlist prior on held-out shapes and at least one other
   Apple GPU, while retaining exact JIT overrides;
5. measure LayerNorm, cross-entropy, fused activation/reduction, decode and
   prefill attention, Top-K/sort and representative end-to-end LLM blocks;
6. add equivalent CUDA and CPU realization families without pretending their
   binding, memory or collective costs are Metal's; and
7. introduce a general Machine TileIR only when multiple backends need the
   same scheduled atom/resource representation and its invariants can be
   stated more cleanly than bridge-local plans.

Until those milestones are measured, the correct claim is narrow but useful:
the TIRx route now has a proof-driven, cost-ranked, high-performance Metal
reduction realization, and the previously measured RMSNorm structural gap is
closed on the recorded M1 Max cohort.

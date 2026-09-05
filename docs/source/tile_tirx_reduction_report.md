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
threadgroup, cooperates across up to 32 SIMD groups for wide programs, and
compacts eligible compiler-owned Tiles to worker-private stripes. The source
C++ kernel and logical TileIR remain unchanged.

Across the saved 24-case Apple M1 Max row-program cohort, all complete FP64 checks pass and
Tile/TIRx is faster than eager PyTorch MPS in every row. Tile/Torch ranges from
0.032× to 0.902× in synchronized device-resident host-wall throughput. Sum and
softmax use preallocated output on both sides; PyTorch's functional RMSNorm,
LayerNorm and cross-entropy allocate returned outputs inside timing, so those
external comparisons are explicitly qualified below. Separate four-round,
same-binary native A/B replays measure 21.19×--49.87× for RMSNorm and
14.04×--75.54× for the LayerNorm/cross-entropy extension. Those causal
native-to-native results are unaffected by PyTorch's output policy.

The final four cases in that cohort are fused residual LayerNorm. They expose
and repair a second structural gap: cloning a cheap shared Tile expression into each
consumer duplicated device reads even after the execution hierarchy was
mapped correctly. The structural exporter now preserves all multi-consumer
pure Tile SSA by default, and the target mapper can compact it to bounded
worker-private stripes. A staged/JIT candidate can still choose recomputation.

These results close two identified structural gaps. They do **not** establish
all-operator, all-shape, low-precision, cross-device or pure-kernel parity.
The later target-width experiment (Section 9.6)
adds separately measured GPU and E2E evidence, including search winners that
regress in independent replay. The fixed-width input-reuse experiment
(Section 9.7) then isolates a resource-planning improvement: caching audited
immutable inputs improves the three large normalization cases in every
paired round, while smaller mixed results and identical-source controls
remain visible. Input caching stays opt-in.

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

Cross-entropy needs no loss-specific primitive. It composes the same semantic
reductions with a guarded indirect Tile operation:

```cpp
auto definition = tile_kernel(
    "cross_entropy",
    [](TensorView<const float, 2> logits,
       TensorView<const int64_t, 1> labels,
       TensorView<float, 1> losses) {
        auto rows = axis("rows", logits.extent<0>());
        auto one = axis("one", 1);
        auto columns = axis("columns", logits.extent<1>());

        for (auto &nest : parallel(shape(rows))) {
            auto origin = coord(nest.index(), 0);
            auto row = logits[origin, shape(one, columns)];
            auto label = labels[coord(nest.index()), shape(one)];
            auto peak = reduce(row, columns, maximum);
            auto total = reduce(exp(row - peak), columns, add);
            auto selected = gather(row, label, columns);
            losses(coord(nest.index()), shape(one))
                .store(log(total) + peak - selected);
        }
    });
```

`gather` remains a general Tile-level library operation. The lowering derives
its guarded Tensor access from the input view; neither the source hierarchy nor
the primitive names a subgroup, thread or memory level.

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
| Distributed private storage | every nonscalar local buffer with distributed stores has compact row-major accesses proved equal to the current worker owner |
| Guarded input views | every delayed source is immutable; each lazy consumer path proves its temporary and source indices in bounds |
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

The planner considers every integer `S` from 1 through
`min(32, floor(target_max_threads/32))`, restricted by shared/private capacity
and the explicit search budget. Non-power-of-two widths are legal. The
32-subgroup algorithmic bound comes from the second collective: one lane
reads each subgroup's partial. It is not a universal schedule space.

### 4.1 One SIMD group per program: spatial packing

When `S = 1` wins under automatic planning, a threadgroup can contain `Q`
independent logical programs, where

```text
1 <= Q <= min(P, 8, floor(max_threads / 32))
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

For every distributed domain of count `N`, the default V=1 mapper enumerates

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

The optional `PlannerOptions::reduction_lane_elements` generalizes this to
blocked-cyclic ownership without changing program ancestry:

```text
e = (W * chunk + w) * V + v,  0 <= v < V,  V in {1,2,4,8}
private_slot = chunk * V + v
inverse: v=e%V, w=(e/V)%W, chunk=e/(W*V)
```

The inverse establishes unique coverage, while the existing ownership audit
requires producers and consumers to refer to the same logical element before
either is remapped. The maximum private allocation becomes
`floor(N/(W*V))*V + min(N%(W*V), V)`. Only the partial final pack carries a
bounds predicate; no invalid load/store is speculatively evaluated. V changes
the worker-local FP32 recurrence and requires the existing reduction-tree
permission, unlike ordered stripe unrolling. It is not a vector-instruction
promise. The
{download}`layout and GPU/E2E evidence <../../scripts/benchmark/tile_torch/results/m1-max-20260905-reduction-lane-validation/notes.md>`
records the generic implementation, resource checks and frozen replay.

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

has two consumers, so the shared structural lowering preserves `shifted` as
one logical definition. By default the same rule applies to every pure
multi-consumer Tile, independent of whether the opcode is `exp`, `add` or
`sub`. Before execution mapping that logical object has `N` elements. Naively
allocating it after distributing rows but before distributing elements creates
`N` private values per worker.

At V=1 the subgroup mapper may replace logical `T[N]` with worker-private
`T_worker[ceil_div(N,W)]` (the generalized V-dependent bound is above), but
only after proving all of the following:

1. the object is a compiler-owned local FP32 materialization with one
   allocation and one defining store;
2. its expression is pure and does not read or expose the object recursively;
3. it has at least one load and no opaque pointer/variable escape;
4. for every store and load in its enclosing loop domain,
   `flatten(indices) == owner_coordinate` is provable; and
5. the owner coordinate is the same distributed logical element `e` used by
   that domain or reduction.

After mapping, the physical slot is `chunk = floor(e / W)`. A fixed,
permuted, cross-element or escaping access with an unknown owner proof cannot
use this substitution. The independent distributed-local audit then either
proves the uncompacted private allocation safe or declines the complete
subgroup map; it is never silently redirected to another worker's slot.

For width 4096 with exactly 256 workers, each worker owns only 16 FP32 values.
Auto-layout tests independently enumerate the current objective and check
the private stripe required by the chosen width, while rejecting the old
per-thread `[4096]` form. This is a resource-planning result,
not a new source-level `Memory` declaration. Explicit manual `Memory` remains
for deliberate expert placement and still requires explicit `.store()`.

### 5.1 Shared SSA is not a physical allocation decision

Fused residual LayerNorm makes the distinction observable:

```cpp
auto combined = X[origin, shape(one, columns)] +
                residual[origin, shape(one, columns)];
auto mean = reduce(combined, columns, add) / width;
auto centered = combined - mean;
auto variance = reduce(centered * centered, columns, add) / width;
Y(origin, shape(one, columns))
    .store(centered / sqrt(variance + 1e-5f));
```

`combined` and `centered` each have several consumers. Cloning their producer
expressions makes generated Metal load every `X` and residual element four
times. Preserving the two SSA definitions does **not** require two source-level
`Memory` declarations. The Metal mapper proves element ownership and realizes
them as two worker stripes, so each input element is loaded once.

Structural `lower(function)` defaults to `SharedTileMaterialization::PRESERVE`
because erasing sharing is irreversible and deprives later analyses of a legal
choice. `EXPENSIVE_ONLY` is retained as an explicit lowering/JIT candidate: it
preserves shared transcendental Tiles but recomputes cheap arithmetic. Neither
mode changes TileIR semantics. On the measured M1 Max, Metal selects
`PRESERVE`; LLVM CPU selects `EXPENSIVE_ONLY`. This is precisely why the
decision belongs to target planning rather than the C++ DSL.

Worker stripes also have an explicit software-state bound:

```text
stripe_scalars(S) = sum[t in materialized Tiles]
                    ceil_div(elements(t), 32*S)
stripe_scalars(S) <= max_reduction_striped_scalars_per_worker = 64.
```

The value is a compiler-created storage budget, not a claim about allocated
hardware registers. A backend may scalarize or spill it. A candidate above the
bound is rejected before code generation instead of silently creating an
unbounded private array. At residual LayerNorm width 4096, 32- and 64-thread
maps would require 256 and 128 scalars per worker and are rejected; 128/256
threads require 64/32 and remain legal.

```{figure} ../_static/tile/reduction-model.svg
:alt: A semantic reduction domain is factored into execution participants and local slots while its reducer contract and result remain target independent.
:width: 86%

Reduction factorization changes participants and local slots. The semantic
contribution set, grouping map, identity and merge contract remain intact.
```

### 5.2 Guarded immutable views and dynamic gather

Cross-entropy revealed another ownership case. Before mapping, a logical row
snapshot can legally be consumed both as `V[e]` by reductions and as
`V[label]` by a guarded gather. Mechanically distributing its initialization
while retaining a private `V[N]` per worker is invalid:

```text
worker w initializes only private_w.V[e(w)]
worker 0 later reads private_0.V[label]
```

Unless `label == e(0)`, that read does not observe the logical snapshot. The
first prototype did exactly this and failed six of seven cross-entropy rows;
the reductions themselves were correct.

The view analysis is now path-sensitive for pure lazy `if_then_else` calls.
For a target-buffer load under path predicate `G`, every index bound must be
proved from the loop domain, contained syntactically in `G`, or follow from
`G`. Under the independent noalias/effect proof, substitution preserves the
lazy source guard and produces:

```text
selected = ite(0 <= label && label < N,
               global_logits[row, label], 0)
```

The logical snapshot allocation disappears; reductions and gather read the
same immutable Tensor directly. Conditions, address expressions and fill
values remain lazy. An unguarded memory-dependent index is still unknown, as
are predicates outside the supported pure Boolean fragment.

This optimization is not the correctness firewall. A separate whole-program
audit examines every nonscalar local buffer that has a distributed store. It
requires one allocation, compact storage and
`flatten(access_indices) == owner_coordinate` for every observed load and
distributed store. The current proof intentionally rejects otherwise safe but
unrecognized permutations. Automatic execution then falls back to the
reference map; an explicit subgroup request reports that it is unrealizable.

```{figure} ../_static/tile/guarded-view-ownership.svg
:alt: A guarded dynamic gather cannot read a logically distributed Tile from one worker's private full array. Path-sensitive immutable view forwarding removes that array, while a separate ownership audit rejects unsupported maps.
:width: 100%

Forwarding is a profitable realization; the distributed-local audit is the
independent fail-closed correctness boundary.
```

## 6. Cost model and finite solver

Legality is decided before profitability. Let `D` be the independent element
domains, `R` the reductions, `I_d` and `N_r` their counts, `S` the candidate
subgroups per program, `W=32S`, and `Q` the separately searched packed-program
count (`Q>1` only for `S=1`). At V=1 the bootstrap score is

```text
rounds(S) = sum[d in D] ceil_div(I_d, W)
          + sum[r in R] ceil_div(N_r, W)

score(S, Q) = alpha * rounds(S)
         + beta  * |R| * S
         + gamma / Q

alpha = 1, beta = 2, gamma = 16.
```

For non-default consecutive-worker width V, each `ceil_div(N,W)` becomes
`floor(N/(W*V))*V + min(N%(W*V),V)`, the maximum live scalar work per worker.
The resource bound uses the same layout-dependent expression, and backend
policies receive V explicitly. This accounts for tail ownership but does not
model vector issue, coalescing or reduced active-group concurrency.

Policies also receive physical group count `ceil_div(programs,Q)`, total
useful scalar elements summed over these separate domains, and useful lane
work `elements/(rounds*W)`. These immutable facts do not assert measured
occupancy. Device thread capacity is now queried by the benchmark adapter,
instead of inheriting TVM's 256-thread target default. The algorithm's
32-partial collective bound remains independent of that device limit.

Each independent domain is rounded separately. Combining their total before
`ceil_div` would underprice separate softmax passes at a tail width.

The three terms represent worker stripe rounds, SIMD collective work and
amortized program/threadgroup setup. They are dimensionless priors, not
measured instructions, occupancy or nanoseconds. Ties retain the smaller `S`.

The v1 score currently treats every scalar round alike. It does not yet count
global loads duplicated by recomputation, expression depth, or the different
service costs of local stripe access. The residual LayerNorm policy search
therefore exposes up to 43.66% model regret on this finite candidate set. This
is recorded as a model defect; correctness-checked staged/JIT measurement is
the selection authority until those features are calibrated.

Before scoring, a candidate is rejected if it violates any of:

```text
S <= min(32, floor(target_max_threads / 32))
threadgroup_threads <= target_max_threads
shared_bytes(S) <= target_shared_memory
stripe_scalars(S) <= max_reduction_striped_scalars_per_worker
enumerated_widths <= max_thread_candidates
finite, nonnegative coefficients
supported exact thread constraint, when present.
```

There are at most 39 automatic `(S,Q)` candidates (eight `Q` choices for
`S=1`, 31 wider single-program choices), so exhaustive enumeration is
clearer and more exact than integer programming or simulated annealing here.
Those methods become relevant only when later planners jointly choose tile
factorizations, layouts, atoms, storage versions and pipeline schedules over a
much larger discrete space.

`threads_per_group=0` means automatic choice. A nonzero value is an exact JIT
constraint and must be a legal multiple of 32. Without an exact packing
request it fixes cooperating workers per program, retaining the original
interface. `reduction_programs_per_group>1` instead explicitly chooses packed
one-subgroup programs; any simultaneous thread request must equal `32*Q`.
Constraints are never silently clamped or reinterpreted.

The bridge now exposes a backend-owned cost-policy interface and bounded
ordered stripe unrolling. See [the implemented policy and search contract](tile_execution_planner.md).
The JIT harness can search `--tune-reduction-packing`,
`--tune-reduction-unroll` and `--tune-reduction-lane-elements` alongside exact
thread widths. The original model remains an uncalibrated prior; it does not
price the unrolling choice or V's hardware issue behavior. Host-wall
throughput remains the default tuning objective; explicit
`--tuning-metric gpu-control` instead requires the no-counter Metal
command-buffer metric. Both objectives remeasure the winner and retain GPU
and E2E scopes separately in frozen replay reports.

### 6.1 Plans selected in the original 20-case run

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
| LayerNorm 1×127 | 64 | 2 | 16 | 0 | 2 | 30 |
| LayerNorm 17×257 | 256 | 1 | 0 | 0 | 2 | 33 |
| LayerNorm 128×1024 | 128 | 4 | 32 | 0 | 2 | 56 |
| LayerNorm 64×4096 | 256 | 8 | 64 | 0 | 2 | 96 |
| cross-entropy 1×127 | 32 | 1 | 0 | 0 | 2 | 31 |
| cross-entropy 17×257 | 256 | 1 | 0 | 0 | 2 | 27 |
| cross-entropy 128×1024 | 128 | 4 | 32 | 0 | 2 | 51 |
| cross-entropy 64×4096 | 256 | 8 | 64 | 0 | 2 | 83 |

`threads/group` is the whole threadgroup width. For the 17-row `S=1` plans,
256 threads mean eight independently packed row programs, not eight groups
cooperating on one row.

LayerNorm's independent element count is the row width because its affine
output is distributed. Cross-entropy has only three scalar independent
elements after immutable logits/label views are forwarded; its two width-sized
loops are the reductions already counted in `R`.

The table is retained as the exact historical plan report. The current
default structural exporter additionally preserves LayerNorm's shared
`centered` Tile, so a newly compiled width-4096 LayerNorm reports a bounded
worker stripe rather than zero. Its exact size follows the selected width,
not this historical 256-thread entry. Saved artifacts are never rewritten to
pretend they were produced by the newer policy.

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

Materialization is another ordinary staged dimension:

```bash
uv run --no-project --python 3.13 --with torch --with numpy python \
  scripts/benchmark/tile_torch/run.py \
  --native cmake-build-tirx/bin/benchmark_tile_tirx \
  --output NEW_EMPTY_DIRECTORY \
  --backends metal --operations residual_layernorm \
  --metal-subgroup-reductions \
  --tune-shared-tile-materializations preserve,expensive-only
```

The runner includes this choice in the finite Cartesian product with tile,
pipeline and execution-width candidates, enforces the global JIT budget,
retains failed candidates, and recompiles/revalidates the measured winner.
There is no capture-once super-kernel.

## 8. Correctness and structural verification

The regression suite checks both generated structure and actual hardware
results:

- option/target/noalias contracts and absence of silent subgroup intrinsics by
  default;
- row sums at widths 127, 257, 1024 and 4096, including expected
  one/one/four/eight-group plans;
- softmax at `3×4096`, requiring two reductions, an independently enumerated
  minimum-cost width, the corresponding exact private stripe, `simd_max`,
  `simd_sum`, and every output element against FP64;
- LayerNorm at `3×4096`, requiring two reductions and checking all 12,288
  affine outputs against an independent FP64 mean/variance formula; the
  test independently checks the selected compact stripe for its shared cheap
  arithmetic Tile;
- 14 additional V=1/4 softmax layouts, automatic and exact widths
  96/160/224/288/512/1024 on a capable device, ragged domains, both collectives,
  complete outputs, every legal candidate reaching the backend policy and
  rejection of insufficient search budgets;
- canonical multi-consumer arithmetic under both `PRESERVE` and
  `EXPENSIVE_ONLY`, proving that the first retains the SSA boundary and the
  second is a real recomputation candidate;
- cross-entropy at `7×4096`, including two collectives, a guarded dynamic
  label gather, absence of the private `[4096]` input snapshot, and every loss
  against a stable FP64 log-sum-exp formula;
- an explicitly materialized derived-logits negative case, which must decline
  subgroup mapping because its gather crosses private worker ownership;
- a CPU structural pair showing that an unguarded memory-dependent consumer
  index retains its snapshot while the same index under a complete lazy bounds
  guard is safely forwarded;
- minimum and maximum together at `7×1024`;
- uniform first collectives and guarded tails in generated Metal;
- CPU fallback behavior and unchanged reference paths; and
- benchmark/replay metadata, policy preservation, cooperating-subgroup facts
  and staged/JIT winner revalidation, including materialization policy.

At the earlier shared-Tile checkpoint (historical counts retained):

```text
complete CTest /^test_tile_/:            32 / 32 tests passed
guarded CPU view proof:               1,572 assertions passed
Metal subgroup LayerNorm:            12,297 assertions passed
Metal subgroup cross-entropy:            20 assertions passed
focused TIRx execution, CPU:          33,071 assertions passed
focused TIRx execution, Metal:        38,363 assertions passed
focused TIRx planner:                  5,891 assertions passed
Python benchmark contract discovery:    69 / 69 tests passed
```

The wider repository contains an unrelated local edit to Metal memory flags;
the targeted subgroup hardware tests avoid attributing that user's change to
this feature. The complete Tile cohort was also run with the submitted
source value restored temporarily, then the user's local edit was restored.
Commands and the full warning boundary are in the
{download}`shared-Tile validation note <../../scripts/benchmark/tile_torch/results/m1-max-20260905-shared-tile-validation/notes.md>`.

The target-width checkpoint instead runs the full 33-test
Tile suite without modifying that local edit: 31/33 pass, with only the two
known cooperative/memory source-assertion conflicts; their numerical checks
pass. All 83 Python benchmark contract tests and the 14 new ragged Metal
layouts pass. The
{download}`target-width validation log and boundary <../../scripts/benchmark/tile_torch/results/m1-max-20260905-reduction-width-validation/notes.md>`
retain the exact build/test provenance; the older 32/32 counts above are not
relabeled as a new clean-source run.

The subsequent input-reuse checkpoint adds 22 Metal numeric configurations
and passes all 84 Python contracts. It rebuilds the full selected tree and
again runs all 33 Tile tests without touching the local memory-flag edit:
31/33 pass with the same two source-assertion failures. Section 9.7 and its
linked validation note retain this latest run separately.

## 9. Performance evidence

### 9.1 Base reductions versus eager PyTorch

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

### 9.2 LayerNorm and cross-entropy versus eager PyTorch

The independent eight-case extension is
{download}`LayerNorm/cross-entropy <../../scripts/benchmark/tile_torch/results/m1-max-20260905-metal-subgroup-row-extensions/notes.md>`;
its adjacent JSON retains every sample, plan, error, setup phase and generated
Metal source.

| Operator / rows×width | Tile µs | Torch µs | Tile/Torch |
|---|---:|---:|---:|
| LayerNorm 1×127 | 4.500 | 8.400 | 0.536× |
| LayerNorm 17×257 | 5.714 | 8.821 | 0.648× |
| LayerNorm 128×1024 | 7.542 | 13.726 | 0.549× |
| LayerNorm 64×4096 | 12.413 | 24.313 | 0.511× |
| cross-entropy 1×127 | 4.513 | 107.246 | 0.042× |
| cross-entropy 17×257 | 3.449 | 107.695 | 0.032× |
| cross-entropy 128×1024 | 4.290 | 110.171 | 0.039× |
| cross-entropy 64×4096 | 5.838 | 112.263 | 0.052× |

These use the same synchronized host-wall protocol, now with 11 samples and
100 ms windows. PyTorch's functional LayerNorm and cross-entropy calls return
new output tensors, so their allocation is inside timing. Cross-entropy also
includes the general eager operator's dispatch and semantic machinery. The
table is therefore a real API-level comparison, not evidence that the Tile
kernel is 19--31× faster than an isolated MPS kernel. The native A/B below is
the causal lowering comparison.

### 9.3 RMSNorm causal A/B against the old lowering

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

### 9.4 LayerNorm and cross-entropy causal A/B

The
{download}`balanced extension replay <../../scripts/benchmark/tile_torch/results/m1-max-20260905-metal-subgroup-row-extensions-replay/notes.md>`
uses the same executable for both policies, counterbalances order over four
rounds, recaptures/JIT-compiles every variant and checks every output.

| Operator / rows×width | Old reference µs | New subgroup µs | Paired speedup median [range] |
|---|---:|---:|---:|
| LayerNorm 1×127 | 131.413 | 4.577 | 28.675× [27.944, 29.063] |
| LayerNorm 17×257 | 337.366 | 5.693 | 58.942× [57.105, 64.220] |
| LayerNorm 128×1024 | 280.352 | 7.517 | 37.333× [36.900, 37.661] |
| LayerNorm 64×4096 | 928.945 | 12.306 | 75.536× [74.338, 82.088] |
| cross-entropy 1×127 | 62.412 | 4.446 | 14.042× [13.737, 14.854] |
| cross-entropy 17×257 | 191.603 | 3.228 | 59.357× [53.681, 61.339] |
| cross-entropy 128×1024 | 74.350 | 4.370 | 17.015× [16.097, 17.463] |
| cross-entropy 64×4096 | 355.493 | 5.774 | 60.879× [59.618, 63.291] |

All 64 native variant measurements pass, and all fingerprinted artifacts are
unchanged across the replay. This attributes the gain to the execution/view/
resource realization family rather than PyTorch output allocation or a
different binary. The ranges are observed paired-round extrema, not confidence
intervals.

### 9.5 Fused residual LayerNorm and materialization choice

The current
{download}`Metal materialization search <../../scripts/benchmark/tile_torch/results/m1-max-20260905-residual-layernorm-materialization-search/notes.md>`
JIT-compiles both shared-Tile policies for every shape. All measured winners
use `PRESERVE`:

| Rows×width | Tile µs | Eager Torch MPS µs | Tile/Torch | Stripe scalars/worker |
|---|---:|---:|---:|---:|
| 1×127 | 3.426 | 10.671 | 0.321× | 4 |
| 17×257 | 3.655 | 11.705 | 0.312× | 6 |
| 128×1024 | 6.321 | 18.592 | 0.340× | 8 |
| 64×4096 | 8.324 | 27.046 | 0.308× | 32 |

PyTorch evaluates eager `layer_norm(X + residual)` and allocates its returned
output, so this is an API-level comparison. The clean policy attribution is
the separate
{download}`four-round A/B replay <../../scripts/benchmark/tile_torch/results/m1-max-20260905-residual-layernorm-materialization-replay/notes.md>`:

| Rows×width | Expensive-only µs | Preserve µs | Paired preserve speedup [range] |
|---|---:|---:|---:|
| 1×127 | 3.692 | 3.506 | 1.057× [1.027, 1.137] |
| 17×257 | 3.648 | 3.632 | 1.008× [0.957, 1.039] |
| 128×1024 | 8.244 | 6.084 | 1.354× [1.313, 1.366] |
| 64×4096 | 13.548 | 9.591 | 1.421× [1.392, 1.471] |

All 32 native A/B measurements pass and use unchanged fingerprinted artifacts.
The independent
{download}`bounded thread search <../../scripts/benchmark/tile_torch/results/m1-max-20260905-residual-layernorm-bounded-thread-search/notes.md>`
also records the 64-scalar legality bound and rejected wide-shape candidates.

The
{download}`CPU materialization search <../../scripts/benchmark/tile_torch/results/m1-max-20260905-cpu-residual-layernorm-materialization-search/notes.md>`
selects the other legal policy, `EXPENSIVE_ONLY`, with native/Torch ratios
0.109×, 0.225×, 0.382× and 0.643×. That run includes native LLVM codegen,
proved input views, automatic element packing and the explicit 64 KiB local
stack budget. It demonstrates target-dependent resource planning, not a
universal preference for either materialization policy.

### 9.6 Target-aware widths: GPU and dispatch acceptance

The latest
{download}`width evidence and independent audit <../../scripts/benchmark/tile_torch/results/m1-max-20260905-reduction-width-validation/notes.md>`
cover 15 FP32 sum/softmax/RMSNorm cases: 17×257, 64×4096, 1024×4096,
7×1537 and 128×8192. A six-width GPU-objective search compares
{32,96,128,256,512,1024}; its reference is the best valid member of the
restricted {32,128,256} subfamily. Both frozen variants retain V=4, P=1 and
U=1. This is neither a full-width optimum nor an old/new default-policy test.

All 240 replayed native/Torch outputs pass in four order-balanced rounds,
with unchanged fingerprinted executable, bridge and TVM compiler/runtime
libraries. GPU values below are **no-counter command-buffer execution
intervals**, not isolated kernel timestamps. Host batches and single-call
dispatch latency are separate uninstrumented phases. Times are medians of
per-round p50s; gains are median paired reference/candidate ratios.

At 1024×4096, all columns below are GPU measurements. Ref/new are the
frozen native reference/candidate; Torch is measured with the candidate.

| Op | Ref µs | New µs | Gain [min, max] | Torch µs |
|---|---:|---:|---:|---:|
| sum | 24.837 | 23.625 | 1.051× [1.031, 1.080] | 26.511 |
| softmax | 67.543 | 59.174 | 1.141× [1.111, 1.157] | 121.056 |
| RMSNorm | 70.910 | 64.210 | 1.101× [1.092, 1.132] | 68.802 |

All three wider-row gains are positive in every pair. Their separate
batched E2E throughput gains are 1.045×, 1.156× and 1.092×. RMSNorm's paired
native/Torch GPU time ratio is 0.931, but single-call GPU medians are
93.417/92.458 µs and E2E medians 316.479/318.855 µs: approximately parity,
not a general dispatch-latency win. Torch's functional RMSNorm retains its
returned-output allocation; sum and softmax use preallocated output on both
sides. The reference-to-candidate comparison uses the same native API and
allocation policy.

The full cohort also contains necessary counterexamples. W=1024 sum at
128×8192 and W=96 softmax at 17×257 were search winners but regress in all
four independent GPU pairs, costing 6.79% and 25.88% more time. Five
identical-plan controls quantify observed variability, including a
0.848--1.131 apparent gain range for the shortest sum control. These are not
confidence intervals or correction factors. The four-round finite cohort
does not justify a universal winner, V default, or fitted cost coefficients.
It does justify retaining an incumbent in independent measured acceptance.

At N=4096/W=1024/V=4 the generated MSL has one straight-line four-element
pack; at W=128 it has eight chunk iterations. Softmax's private stripe also
shrinks from 32 to four scalars. These are real code-shape/resource features,
not evidence of physical register counts or the exact performance cause.
The remaining policy needs memory/issue/collective service and whole-device
subgroup demand, followed by held-out ranking validation. A more elaborate
solver cannot compensate for missing features or noisy timing labels.

### 9.7 Budgeted immutable-input reuse

The optional `PlannerOptions::cache_reduction_inputs` restores a scheduling
choice that unconditional input forwarding erased: a proved immutable input
Tile used in distinct element/reduction domains may remain materialized.
The existing reduction ownership audit then maps it to worker-private
stripes and charges the same cumulative scalar budget as computed Tiles.
This does not add a DSL entity, change logical execution coordinates, or ask the user
to place input memory manually. The default remains reload/forward.

The view analysis still requires noalias, immutable source/address/guard/fill,
complete initialization, dominance, bounds, and non-escape. It counts distinct
consumer domains, so `x*x` in one recurrence alone does not cache `x`.
Preserved copies carry compiler provenance; manual memory is not relabeled.
The mapping must also prove that every later access belongs to the same
worker. Cross-worker dynamic gathers and over-budget requests fail closed;
enabling this option is an exact request for the reduction mapping family,
not permission to silently fall back to a replicated Tile.

The benchmark switch `--cache-reduction-inputs` and frozen replay preserve
this choice explicitly. The
{download}`complete input-cache evidence <../../scripts/benchmark/tile_torch/results/m1-max-20260905-input-cache-validation/notes.md>`
contains a fixed W=512, V=4, U=1, P=1 experiment for sum, softmax, RMSNorm,
LayerNorm and residual LayerNorm at five shapes. Four-round counterbalanced
replay validates all 400 outputs; the two unscreened pilots add 100 outputs.
All realized source hashes and complete plans match their frozen pilots.

At 1024×4096 the cache candidate reduces GPU execution time for all three
affected normalizations in every paired round. GPU means no-counter
command-buffer time, not an isolated kernel timestamp. Times are medians of
per-round p50s; gain is the paired ratio median, with observed min–max:

| Op | Reload µs | Cache µs | Gain [min, max] | Torch µs |
|---|---:|---:|---:|---:|
| softmax | 74.198 | 53.949 | 1.378× [1.373, 1.395] | 121.715 |
| RMSNorm | 70.668 | 55.863 | 1.265× [1.246, 1.345] | 69.108 |
| LayerNorm | 79.150 | 64.704 | 1.221× [1.213, 1.251] | 206.598 |

E2E-throughput gains are 1.381×, 1.279× and 1.229× in the same order.
These compare cache/reload at the **same fixed mapping**, not the default
planner, earlier tuned widths or an exhaustive hardware optimum. Torch uses
the recorded eager/output-allocation policy. All 15 changed-source cases
have positive median GPU gains, but RMSNorm 17×257/64×4096 and LayerNorm
17×257 include an individual pair at or below parity. All ten unchanged
sum/residual LayerNorm cases are retained as identical-source controls;
their timing variation is not credited as an optimization.

The unchanged analytic score exposes the next problem concretely: caching
raises the 4096-column RMSNorm score from 64 to 72 because it adds a private
copy traversal, while the measured GPU time falls. It does not distinguish
global from private access service or price eliminated cross-phase input
reads. Cache/reload candidates must not be pruned by that uncalibrated score.
Resource-sensitive features and held-out ranking validation remain necessary;
defaults and cost coefficients are unchanged.

The implementation checkpoint passes 22 new Metal numeric configurations
(including packed-program tails, V=1/V=4, three-way unrolling, non-power-of-two
cooperation and zero-padded input). Tests also reject unsupported targets,
missing noalias, cross-worker gathers and over-budget caches. The full Tile
suite remains 31/33: only the two pre-existing source assertions against the
untouched local `mem_flags(2)` edit fail. Benchmark Python tests pass 84/84.

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
4. add global/local traffic, expression-depth and live-state features to the
   shortlist prior, then calibrate it on held-out shapes and at least one other
   Apple GPU while retaining exact JIT overrides;
5. measure cross-entropy backward, decode and prefill attention, Top-K/sort
   and representative end-to-end LLM blocks;
6. add equivalent CUDA and CPU realization families without pretending their
   binding, memory or collective costs are Metal's; and
7. introduce a general Machine TileIR only when multiple backends need the
   same scheduled atom/resource representation and its invariants can be
   stated more cleanly than bridge-local plans.

Until those milestones are measured, the correct claim is narrow but useful:
the TIRx route now has a proof-driven, cost-ranked, high-performance Metal
reduction realization, and the previously measured RMSNorm, LayerNorm and
forward cross-entropy gaps plus the fused residual LayerNorm shared-SSA defect
are closed on the recorded M1 Max cohort.

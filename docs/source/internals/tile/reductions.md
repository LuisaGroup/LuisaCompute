# TIRx Metal reduction lowering

The bridge proves a semantic FP32 row reduction before assigning it to
one or more SIMD groups and deriving worker-private storage. This is a
target realization of execution-first TileIR, not a warp-specific DSL.

This page owns the mapping, proof, resource and intrinsic contracts.
Performance measurements and regressions are maintained separately in
[Metal reduction measurements](../../performance/tile/reductions.md).

```{contents} On this page
:local:
:depth: 2
```

```{figure} ../../../_static/tile/tirx-subgroup-reduction.svg
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

```{figure} ../../../_static/tile/execution-to-memory.svg
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

### Warp and SIMD-group intrinsics in the generated code

**The Metal path uses `simd_sum`, `simd_max` and `simd_min`.** They are present
in emitted MSL, not merely listed as future planner capabilities. Each worker
first accumulates its own ordered stripe; the intrinsic then combines the
32 lanes of its SIMD group. For a single-group program, the result is already
available to every lane and no threadgroup scratch or barrier is needed.

With `S > 1` cooperating SIMD groups, each group's lane zero writes one
partial to shared memory. One threadgroup barrier makes those partials visible.
Every participating SIMD group then performs a second collective over the
`S` partials, padding unused lanes with the reducer's identity. For sum, the
generated structure is equivalent to:

```cpp
float partial = simd_sum(local_sum);
if (lane == 0) { shared_partials[program_base + simd_group] = partial; }
threadgroup_barrier(mem_flags::mem_threadgroup);
float total = simd_sum(lane < S ? shared_partials[program_base + lane] : 0.0f);
```

`program_base` is zero for a one-program group. With explicit cooperating
program packing it is `packed_program * S`, isolating each program's partials
while all physical threads participate in the same barrier.

The second collective is replicated across participating groups so subsequent
distributed consumers have the result without an additional broadcast barrier.
This implementation uses the native collective rather than an explicit
shuffle-down loop. It does not currently need ballot, match or scan intrinsics;
using more intrinsics is not itself an optimization objective. Generated MSL
does not by itself prove a particular final machine-instruction sequence.

The scope is the admitted **Metal FP32 add/max/min** family. The explicit
`metal_subgroup_reductions` option also permits floating-point reassociation;
disabled or unproved automatic cases retain the reference path, and an
unrealizable explicit subgroup binding is rejected. This is not a claim that
the CPU, CUDA or arbitrary reducer path has the same collective optimization.

The implementation is `ReductionProgramMapper::_reduction` in
{download}`reduction.cpp <../../../../src/tile/bridge/tirx/reduction.cpp>`.
The {download}`execution tests <../../../../src/tests/unit/tile/bridge/test_tirx_execution.cpp>`
check the generated intrinsic names, one-/multi-group barrier/storage counts,
numerical outputs and disabled/fallback behavior.

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
q = floor(thread / W)          program coordinate within a group
w = thread mod W               worker coordinate within a program
s = floor(w / L)               SIMD-group coordinate within a program
l = w mod L                    lane coordinate
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

p = block * Q + q
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

When `S > 1`, automatic packing retains one logical program per threadgroup:

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

### 4.3 Explicit packing of cooperating programs

An explicit `reduction_programs_per_group=Q>1` can also pack programs that
each use `S>1` SIMD groups. Collaboration within one program and packing
independent programs are separate factors; neither is a memory hierarchy.

```text
T = 32 * S * Q                 total physical threads per group
W = 32 * S                     workers cooperating on one program
q = floor(thread / W)          packed program coordinate
w = thread mod W               worker coordinate within that program
s = floor(w / 32), l = w % 32   local subgroup and lane coordinates
p = block * Q + q              logical program coordinate before loop min
partial[r, q*S + s]            shared slot written by local subgroup s
blocks = ceil_div(P, Q)         P is the logical program count

one physical threadgroup, Q=2, S=2
  program q=0                   program q=1
    subgroup 0 -> partial[0]      subgroup 0 -> partial[2]
    subgroup 1 -> partial[1]      subgroup 1 -> partial[3]
  ------------------ one uniform group barrier ------------------
    read partial[0:2]             read partial[2:4]
    both groups get result 0      both groups get result 1
```

Shared storage grows to `|R| * Q * S * sizeof(float)`. Worker-private stripes
are still derived from **W**, not T; packing does not give a worker twice the
private capacity or another program's elements. An exact thread request fixes
T and must be divisible by `32*Q`. Without one, the bounded solver searches
fitting cooperating widths for that explicit Q. Thread, shared-memory and
private-state checks are applied before cost ranking.

For a partial final group, a whole-program `if (p<P)` would put the group
barrier under divergent control. Instead, inactive programs replay the last
valid row (`min(p,P-1)`, plus the original loop minimum) and suppress only
external stores. Their input addresses remain valid even for guarded,
data-dependent gathers; they still perform reads and arithmetic. Active
programs retain unique element coverage and store ownership.

This requires an additional conservative proof: reduction-containing enclosing
loops must have constant minimum, unit extent and unit serial step. Repeated
or dynamic enclosing loops are rejected, including fully packed cases: scratch
reuse across iterations needs a read-before-next-write fence proof. A packed
tail also rejects any external buffer observed both read and written, under
the existing noalias and pure-effect admission checks. Replay must not race
with a valid program's writes. Unknown proofs decline this realization.

The automatic `Q=0` option keeps the established candidate family; it does
not begin choosing cooperating packed groups. The
[fixed packing replay](../../performance/tile/reductions.md#cooperating-program-packing)
finds substantial regressions as well as two narrow gains, so the extension
remains explicit/JIT-only, without a new default or fitted coefficient.

### 4.4 Coverage and ancestry

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
{download}`layout and GPU/E2E evidence <../../../../scripts/benchmark/tile_torch/results/m1-max-20260905-reduction-lane-validation/notes.md>`
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

```{figure} ../../../_static/tile/reduction-model.svg
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

```{figure} ../../../_static/tile/guarded-view-ownership.svg
:alt: A guarded dynamic gather cannot read a logically distributed Tile from one worker's private full array. Path-sensitive immutable view forwarding removes that array, while a separate ownership audit rejects unsupported maps.
:width: 100%

Forwarding is a profitable realization; the distributed-local audit is the
independent fail-closed correctness boundary.
```

## 6. Cost model and finite solver

Legality is decided before profitability. Let `D` be the independent element
domains, `R` the reductions, `I_d` and `N_r` their counts, `S` the candidate
subgroups per program, `W=32S`, and `Q` the separately searched packed-program
count (automatic `Q>1` only for `S=1`; explicit Q can also use cooperating
groups). At V=1 the bootstrap score is

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
shared_bytes(S, Q) <= target_shared_memory
stripe_scalars(S) <= max_reduction_striped_scalars_per_worker
enumerated_widths <= max_thread_candidates
finite, nonnegative coefficients
supported exact thread constraint, when present.
```

There are at most 39 unconstrained automatic `(S,Q)` candidates (eight `Q` choices for
`S=1`, 31 wider single-program choices), so exhaustive enumeration is
clearer and more exact than integer programming or simulated annealing here.
Those methods become relevant only when later planners jointly choose tile
factorizations, layouts, atoms, storage versions and pipeline schedules over a
much larger discrete space.

`threads_per_group=0` means automatic choice. A nonzero value is an exact JIT
constraint and must be a legal multiple of 32. Without an exact packing
request it fixes cooperating workers per program, retaining the original
interface. `reduction_programs_per_group>1` instead explicitly fixes Q;
any simultaneous thread request must equal `32*S*Q` for a supported S.
Constraints are never silently clamped or reinterpreted.

The optional service policy prices packed-tail replay reads for all
`ceil_div(P,Q)*Q` cooperating program slots, but global writes only for the
P active programs. Payload facts per program remain useful work, not padded
launch totals. The analytic prior has no such service term and currently
rewards packing's setup amortization without its occupancy/synchronization
costs; the fixed packing measurements expose that limitation.

The bridge now exposes a backend-owned cost-policy interface and bounded
ordered stripe unrolling. See [the implemented policy and search contract](planner.md).
The JIT harness can search `--tune-reduction-packing`,
`--tune-reduction-unroll` and `--tune-reduction-lane-elements` alongside exact
thread widths. The original model remains an uncalibrated prior; it does not
price the unrolling choice or V's hardware issue behavior. Host-wall
throughput remains the default tuning objective; explicit
`--tuning-metric gpu-control` instead requires the no-counter Metal
command-buffer metric. Both objectives remeasure the winner and retain GPU
and E2E scopes separately in frozen replay reports.

### 6.1 Plans selected in the original 20-case run

The [historical launch-plan table](../../performance/tile/reductions.md#historical-launch-plans)
now lives with the measurements that produced it. It is not the current
automatic schedule contract: later shared-SSA preservation, target-width and
resource policies change the candidate facts. This link retains the original
section anchor without duplicating experiment data in the lowering reference.

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
- 36 cooperating-packing configurations for sum, softmax, no-affine LayerNorm
  and paired min/max, including non-power-of-two widths, cached/ragged inputs,
  a 1024-thread group, automatic width selection and sentinel guard rows;
- six raw-IR packing admission cases: direct/unit-wrapper success and repeated,
  row-varying, read/write-tail and over-limit rejection, with a typed check that
  the accepted group fence is not nested under a conditional;
- CPU fallback behavior and unchanged reference paths; and
- benchmark/replay metadata, policy preservation, cooperating-subgroup facts
  and staged/JIT winner revalidation, including materialization policy.

Executed counts, historical checkpoints and the two known local
source-assertion failures belong to the
[validation record](../../performance/tile/validation.md#metal-reduction-validation-checkpoints).
The [cooperating-packing measurements](../../performance/tile/reductions.md#cooperating-program-packing)
link the latest full CTest log, independent audit and retained negative results.
The contracts above describe required coverage, not a claim that every current
worktree or target passes all tests.

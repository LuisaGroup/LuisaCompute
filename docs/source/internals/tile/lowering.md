# TIRx export and target realization

This page owns the shared structural export and target-specific realization contracts. [Runtime integration](runtime.md) owns shader creation, argument ABI and dispatch; [planning](planner.md) owns legal candidate selection. Proposed scheduled forms must not be confused with the implemented bridge-local plans.

```{contents} On this page
:local:
:depth: 2
```

## Compiler bridges and native backends

### Boundary

TileIR is the source of truth. TVM is an optional lowering bridge used to
bootstrap scheduling, legalization, and code generation. TVM types never
appear in the C++ DSL API, TileIR core headers, cache ABI, or runtime ABI.

Interoperability code lives below `luisa/tile/bridge/`: `bridge/tirx` owns the
TVM dependency and `bridge/xir` owns TileIR-to-Luisa-XIR translation. The first
XIR realization targets independent CPU workers and connects to the existing
SIMD backend; it is separate from both native MPP and TIRx.
The TIRx bridge constructs `tvm::tirx::PrimFunc`, layouts, expressions, and
statements directly through TVM's public C++ API. TVMScript text is useful for
debug printing and differential tests, but source generation or Python parsing
is not a compiler boundary.

Native hardware lowering is not another bridge. `DeviceInterface` exposes an
optional `create_tile_kernel` factory: TileIR and compile options enter the
active backend, which owns the compiler-route choice and returns a normal
`ShaderCreationInfo` plus launch/argument metadata. The default implementation
reports unsupported. TVM types do not cross this ABI. A backend can use native
lowering or an installed bridge to obtain its executable; a bridge is not a
new hardware device. Validation wraps this factory like ordinary shader
creation, and launches use existing shader handles and Stream commands.

Metal MPP and the optional TIRx device-artifact route are implemented through
this factory, described in [Independent Tile Runtime](runtime.md).
`tile::Lowering::NATIVE` and `tile::Lowering::TIRX` select explicitly. TIRx
retains its independent CPU/Metal TVM-runtime regression and comparison path;
the new device artifact preserves a typed PrimFunc, argument binding map and
static launch geometry, with unchanged generated Metal source. Both factory
routes return ordinary Runtime shaders. Neither silently falls back to TVM,
MPP or MPS when its selected realization rejects a kernel. Intra-kernel
Tile/SIMT DSL mixing is deliberately out of scope.

### Layout bridge

TVM TIRx models a layout as named physical-axis coordinates with shard,
replica, and offset components; see the official
[TIRx layout model](https://tvm.apache.org/docs/tirx/layout.html). The TileIR
embedding preserves that exact set-valued direction. For logical space `X`,
replica space `R`, and named physical space `P`:

~~~text
F = X x R
left(x, r)  = x
right(x, r) = D(x) + replica(r) + O

TIRx Layout X -> Set(P)
  = LayoutCorr<X, P; F>(left, right)
~~~

TIRx logical axes become `X`, its replica iters become `R`, named hardware or
memory axes become `P`, its shard body becomes `D`, and its offset becomes
`Translate`. This is an exact embedding, including replication.

A Triton-style distribution arrives in the other direction as a map
`PhysicalSlot -> Logical`. TileIR views it as a correspondence and swaps the
two legs when an exporter needs logical-to-physical placement. No unique
inverse is assumed. Export to a TIRx `TileLayout` is structural when the
reoriented correspondence factors as shard plus replica plus offset. Registered
swizzles export as `ComposeLayout` when supported. Other layouts lower to
explicit TIRx index computation or a legal materialization; they are never
silently approximated.

### Execution bridge

The open logical hierarchy remains in TileIR until target binding. Current
[TIRx execution scopes](https://tvm.apache.org/docs/tirx/api/execution.html)
name a fixed set of GPU-like scopes, so they cannot be the canonical model for
an open hierarchy.

After `ExecBinding` is concrete, the exporter maps target axes to TIRx scopes
or TIRx loop/thread-binding constructs. Serial and vector axes remain explicit.

There is one shared structural exporter, not a complete lowerer per backend.
The changing component is the execution schedule: a CPU plan may map a logical
parallel prefix to a task loop and SIMD suffix, while a GPU plan may map the
same prefix through an affine split to grid, threadgroup, subgroup, and worker
coordinates. Target-specific resource selection and intrinsic dispatch happen
after this binding. Schematically:

~~~text
Candidate TileIR (logical execution tree)
                 |
          target/autotuned ExecBinding
                 v
Scheduled TileIR (physical scopes + index maps)
                 |
       shared structural TIRx exporter
                 v
       target-specific TVM code generation
~~~

A target realization may collapse a proved region into one opaque atom without
mechanically preserving every loop as a machine loop. It must preserve the
region's observable values/effects, layout correspondence, alias contract and
selected numerical policy. Current examples are a complete compact FP32 GEMM
realized as one CBLAS call and shared exp/add/max/min regions realized through
vForce/vDSP. Their versioned contracts come from typed TileIR dataflow and are
revalidated against the transformed TIRx body. This is atom selection behind
the execution model, not a GEMM/softmax frontend primitive or a memory-derived
hierarchy.

In the complete planner, execution binding and atom selection are joint
choices: a matrix atom may constrain participant/layout maps, while a whole
CPU provider atom may consume an entire logical region and make an internal
worker mapping irrelevant. The current CPU provider choices are explicit
compile policies rather than an automatic cost-model decision; the portable
reference path remains available for comparison and unsupported explicit
requests fail closed.

The reference schedule leaves logical `parallel` as marked serial TIRx during
structural export, **including its optional execution-scope constraint**. The
target mapper resolves each constraint before consuming its annotation. The
outermost unbound region maps to LLVM `kParallel` or to a Metal/CUDA-style
`blockIdx.x * threads + threadIdx.x` grid with a tail predicate. Empty logical
domains become no-ops, not invalid zero-sized GPU launches.

The currently implemented reference-mapper subset is listed below. The optional
[Metal row-reduction realization](reductions.md) additionally admits proved
subgroup/cooperating-group maps; it does not extend every reference operation
to those scopes.

| Target | Constraint | Realization |
|---|---|---|
| LLVM | outer `exec::Scope::WORKER` | host parallel loop |
| LLVM | `exec::Scope::VECTOR` root or worker suffix | vector loop with lane-private temporaries |
| Metal | outer `exec::Scope::WORKER` | one logical instance per GPU thread |
| Metal | outer `exec::Scope::GROUP` | one logical instance per threadgroup; independent elements or child workers cooperate |

Outside a cooperative group, unbound nested parallel regions remain serial.
Inside a group, the first child parallel or independent Tile-element domain
uses the worker coordinate; deeper unbound regions remain serial per worker.
Explicit nested worker/vector rebindings need a coordinate factorization that
this reference planner does not yet implement, so they are rejected even
through unbound/serial intermediate scopes. Device, subgroup, unknown scope
names, CPU group bindings, and other unavailable bindings also fail closed.
Disabling vectorization cannot silently override an explicit vector
constraint. The default remains the reference worker mapping; these choices
are not a public `CPU_THREADS`/`GPU_GRID` compile option.

Vector binding includes a separate resource transformation. A compiler-local
temporary declared inside a vector instance has one independent copy per
logical lane; a temporary in an ancestor scope does not. Before TIRx
vectorization the bridge expands compact local storage by the address map:

~~~text
Address(local_coord, lane) = flatten(local_coord) * vector_width + lane
~~~

The allocation is hoisted immediately outside the vector loop, and all of its
loads/stores retain the lane coordinate. This is necessary because the current
upstream TIRx vectorizer does not privatize `AllocBuffer`; merely changing the
loop kind can incorrectly broadcast the last lane's value to every lane.
Parent Tiles, lane-local multidimensional Tiles, and simultaneous loop-carried
updates have separate numerical regressions, alongside a generated LLVM
vector-instruction check. Explicitly placed/laid-out resources are not
rewritten by this compact-local-storage transformation.

Opt-in CPU automatic vectorization consumes the independent-element domain
contract, without matching a particular operator or re-proving independence.
For a supported rectangular domain it factors the innermost element axis as
`i = min + pack * width + lane`, emits power-of-two vector packs of 4--16
logical lanes plus a scalar tail,
and preserves the entire per-element body. In particular, an MMA's K loop
remains serial inside each output instance; no horizontal reduction or
floating-point reassociation is introduced. Native TIRx `ConvertSSA` renews
definitions copied into the full/tail bodies, and `VectorizeLoop`/LLVM perform
SIMD legalization. A logical pack can become several native vectors, exposing
independent accumulator chains without changing temporal order. There is no
new per-MMA scratch allocation or DSL entity.
Bodies containing allocations, while loops, or nested nonserial execution
keep their reference mapping until the corresponding vectorization/storage
support exists. `CompileOptions::auto_vectorize` defaults to false: measurements
show both wins and substantial regressions, so this is a tunable candidate,
not the default realization. Enabling it requires `vectorize = true`.
Explicit vector execution bindings remain independent of this opt-in; disabling
`vectorize` still rejects those explicit bindings. Tests cover
scalar tails, transposes, ordered cancellation, and generated vector products.

Metal group binding similarly includes a resource transformation, not just a
loop tag. The structural exporter marks **independent element domains**;
contraction axes remain sequential in the reference realization. Temporal
iterations retain their dependences even when a safe pipeline plan overlaps
independent producer/consumer work.
It retains each domain as a rectangular serial loop nest with a rank marker.
Flattening is a cooperative-binding decision, not a shared-export decision;
CPU worker/vector paths keep the individual axes available to their optimizer.
The group mapper partitions the flattened independent domain as
`element = chunk * worker_count + worker`, with a tail predicate. Compact
compiler temporaries allocated in the group context become shared memory;
those allocated inside a distributed worker region stay private. Multiple
resources can therefore share the same execution level, and descendants can
read ancestor Tiles without recreating a private copy:

~~~text
parallel(..., GROUP)              one threadgroup per logical group
  |
  +-- load A -> temporary X -----+   group resource instance
  +-- map X  -> temporary Y -----+   another resource, same exec level
  |                             |
  +-- parallel(..., WORKER) -----+-> read X and Y; private local state
  |       each worker handles its own elements
  |
  +-- fence shared + device memory (all workers, including inactive tail)
  +-- next Tile operation        may consume a neighbor's result
~~~

The first implementation synchronizes conservatively between distributed
operations. A scalar group effect executes once and publishes its result.
The fence orders both shared resources and global views **within that group**;
it is not device-wide synchronization between groups. Safe pipeline cuts may
use the [two-window software-prefetch plan](#implemented-native-software-prefetch-path); other pipelines stay
ordered. Neither plan implies hardware-asynchronous copy. Reference MMA
distributes output elements and retains each element's serial contraction;
parallel reduction trees remain separate planner work.

#### Guarded native Metal matrix realization

An optional native atom selector consumes the same MMA operation; no extra
frontend hierarchy or memory object is required. `CompileOptions` has an
explicit `cooperative_matrix` capability contract (default off). For Metal,
the caller must target a device supporting native FP32 SIMD-group matrices
and specify `thread_warp_size=32`; merely naming `metal` does not prove that
capability. This keeps cross-compilation independent of the host GPU.
The initial atom follows [Apple's SIMD-group matrix model](https://developer.apple.com/videos/play/tech-talks/10858/)
introduced with Apple GPU family 7; the target contract must be established
before selecting it.

~~~text
TileIR mma(A, B, C), with typed arithmetic policy
  |
  +-- reference TIRx contraction + MMA provenance marker
  |
  +-- capability + numerical policy + actual body/operand-map proof
       |
       +-- proven: complete 32-worker SIMD groups, 8x8 FP32 matrix atoms
       |     native TIRx matrix load / multiply[/accumulate] / store
       |
       +-- otherwise: the existing checked contraction loops
~~~

The first selector handles rank-two FP32 operands/results, constant or
independent Tile accumulators, positive strided row-major/transposed shared
projections, and M/N/K extents divisible by eight. Global ragged edges are
already handled by bounded Tile loads/stores; two-window pipeline slots remain
uniform outer address coordinates. The matcher rechecks the actual typed
load/multiply/add/store body, not just its marker, so a transform cannot leave
stale metadata that changes semantics. Mixed conversions, opaque placed
layouts, explicit worker-local execution, insufficient participants, and an
ordered MMA policy keep the reference path.

Native TIRx `metal.simdgroup` allocations and registered matrix operations go
through TVMx's C++ compiler. Each matrix instruction runs on a complete SIMD
group; job-tail predicates are uniform within that group, and the following
group-wide fence remains outside those predicates. The selector never treats
an individual worker's Tile as a cooperative fragment. Matrix selection is a
legality decision, not a profitability claim or a change to default mapping.
`test_tile_tirx_matrix` checks generated instructions and full numerical output
on CPU and physical Metal, including transposes, ragged shapes, pipeline
versions, numerical/capability fallbacks, stale markers, and zero contraction.

The Metal matrix path now has a bounded-family execution planner: it searches
complete-subgroup thread counts and exact two-dimensional subgroup/fragment
factorizations. A subgroup can retain several output fragments and reuse A/B
fragments between them. A proved closed accumulator recurrence can also stay
in native fragments across temporal iterations. Other observers of the carried
state prevent that promotion. The plan is a side-table result, not another
frontend hierarchy or a serialized program.

The cost model ranks legal plans by relative issue work and fragment pressure;
its initial coefficients are priors, not measured nanoseconds or hardware
occupancy. Pareto dynamic programming preserves shared-memory/cost tradeoffs
between multiple matrix operations. The exact solver is exact only for this
implemented finite family and this model, not for all GPU programs. The
[execution planner design](planner.md) specifies the constraints,
layout correspondence, objective, solver, calibration, and extension boundary.

Shared-memory budgeting remains the conservative sum of compact group
allocations minus the eliminated result buffers of proved resident recurrences,
checked against target capacity. General lifetime-based reuse and selective
materialization remain planner work. Unsupported placed layouts,
opaque buffer escapes, noncanonical distributed loop steps, and captures of
host-local materialized storage are rejected. Unsupported descendant scopes
are still diagnosed inside empty groups, which otherwise lower to no-ops.
Regressions cover generated shared declarations and full fences, multiple
resources, cross-worker and cross-phase communication, ragged shapes,
softmax/GEMM numerics, resource exhaustion, and illegal scope mappings.

### Implemented native software-prefetch path

The current capture implementation retains cursor cuts as mutable `STAGE`
boundary operations in its single pipeline body. Native export consumes those
boundaries into labeled TIRx statement segments; it does not discard them or
interpret a source ordinal as a hardware cycle. Ordered child regions remain
the canonical TileIR representation planned above, not a claim that this
capture representation has already been migrated.

The CPU and Metal bridge currently chooses a safe cut into two scheduling
phases. Several source segments can belong to one phase. For example, the
`load / score / update` source stages of attention can become an early load
phase and a late score/update phase without moving the recurrence forward.

~~~text
Source iterations                 Scheduled execution (two-phase example)

i=0: load(0) -> compute(0)         prologue:  load(0)
i=1: load(1) -> compute(1)         steady:    load(1) -> compute(0)
i=2: load(2) -> compute(2)                    load(2) -> compute(1)
                                  epilogue:               compute(2)

cross-phase temporary: slot(i) = i % 2
recurrence acc:         updated only by compute, in original iteration order
~~~

This is software prefetching, **not** a promise of an asynchronous DMA engine
or warp-specialized execution. The bridge invokes TVMx's native C++
`InjectSoftwarePipeline` pass through a temporary opaque-SBlock adapter, then
restores ordinary TIRx `AllocBuffer`/`For`/`IfThenElse` statements before
target execution mapping. The public bridge boundary remains native TIRx;
there is no Python source, parser round-trip, or MLIR dependency.

Legality is intentionally conservative:

- Only iteration-local, statically sized, unplaced storage is versioned.
  Allocation/resource annotations are never lost to a buffer-rebuilding pass.
- Cross-phase access to an outer resource with any write prevents that cut.
  This includes loop-carried Scalars/Tiles after materialization and explicit
  Memory. Carries confined to the consumer phase preserve their original order.
- External buffers may alias unless `CompileOptions::noalias` explicitly
  promises otherwise. Read-only aliasing does not prevent prefetching, but
  cross-phase read/write or write/write alias hazards do.
- Buffer live ranges include writes as well as reads, so a late dead write
  cannot overwrite a newer iteration's live slot. Yield snapshots still update
  multiple carried values simultaneously.
- Extra versions must fit a conservative shared-memory capacity bound before
  Metal group mapping. If versioning would overflow it, the legal ordered
  implementation is retained.

`PipelinePolicy::stages` is the current C++ spelling of the scheduling-window
bound (`0` lets the planner choose; `1` disables iteration overlap), not a
source-stage count. This initial planner uses at most two in-flight iterations.
It requires unit `initiation_interval`; other positive intervals retain ordered
reference execution until a target latency/issue model is available. Zero
intervals and invalid IR policy payloads are rejected. Pipelines without cuts,
opaque effects, unsupported placement, or unproved dependencies likewise keep
their ordered semantics. General modulo schedules, inferred cuts, and hardware
async events are not implemented by this path.

`test_tile_tirx_pipeline` checks zero/one/short loops, ragged shapes, multiaxis
iteration order, multiple carries, aliasing, local/outer Memory, late writes,
and capacity fallback on CPU and physical Metal. It also checks preserved
native segments and actual doubled Metal storage, so numerically correct but
still-serial lowering cannot satisfy the positive pipeline case.

### Pipeline and memory bridge

Scheduled TileIR exports:

| TileIR | TVM destination |
|---|---|
| memory declaration | `alloc_buffer` or target memory object |
| logical/physical correspondence | TIRx `TileLayout` when factorizable |
| view/address map | buffer layout or explicit TIRx index map |
| execution binding | TIRx scope IDs or TIRx thread/loop bindings |
| pipeline schedule | software-pipeline annotations or explicit staged loops |
| async copy/token | target-supported async operation and dependence |
| semantic tile op | TIRx tile op when available, otherwise decomposed scalar/vector TIRx |

The initial bridge is one-way. Round-trip import is not required and must not
weaken TileIR invariants.

The bridge's native compiler driver mirrors the TIRx pass pipeline in C++ and
dispatches `target.build.<kind>` through TVM's C++ registry. It partitions host
and device `PrimFunc`s by their bound target, finalizes each partition, and
imports generated device modules into the host runtime module. A scalar
`PrimFunc` compile-and-execute test is the minimum ABI smoke test; Python is not
loaded even for pass orchestration.

The native implementation lowers static-JIT-specialized view arguments,
Scalar and Tile constants, typed elementwise operations with named-dimension
broadcasting, bounded subtile loads/stores, pure Tile extraction/map, semantic
MMA, and `parallel`/`serial`/`pipeline`/`reduce` with inferred Scalar or Tile
carried state. Loads preserve snapshots across later writes; structured yields
snapshot every carried value before updating any carry storage.

Pure elementwise expressions can fuse into their consumer. The initial
reference schedule materializes loaded Tiles, map results, and MMA results
into compiler-owned storage. Worker mapping keeps this private; explicit
Metal group mapping shares group-level allocations as described above. MMA
retains a checked contraction fallback, with the opt-in Metal matrix
[matrix realization](#guarded-native-metal-matrix-realization). Explicit Memory resources preserve
MemoryState ordering and old-value snapshots; provable strided/composed
layouts map to native address expressions, while unsupported placements fail
closed. Safe pipelines use software prefetching, not hardware-asynchronous
transfers. General distribution planning, storage reuse, and additional atom
families remain work in progress. The compiler distinguishes ordinary
TIRx statements (`STANDARD`) from programs containing native TIRx
`TilePrimitive` calls (`TILE`), because only the latter require `LowerTIRx`.
Both paths run `LowerTIRxOpaque` before host/device splitting so thread-binding
loops become device regions. Buffer `noalias` is an explicit caller contract
and defaults off until TileIR carries enough alias metadata to prove it.

### Bootstrap lowering path

~~~text
C++ capture
  -> Candidate TileIR
  -> value-to-SSA and structural verification
  -> layout/distribution inference
  -> execution binding and guarded variant selection
  -> pipeline/resource planning
  -> Scheduled TileIR
  -> native C++ TVM TIRx
  -> existing TVM target lowering
~~~

Later native passes can replace any segment:

~~~text
Scheduled TileIR
  -> target atom legalization
  -> Machine TileIR
  -> Luisa XIR / LLVM / native backend IR
~~~

The bridge is therefore scaffolding, not an architectural dependency trap.

## Target catalog

A target plugin provides data, not frontend syntax:

- target execution axes and legal bindings;
- the target-scope containment poset and parent projections;
- legal region anchors, execution frontiers, and convergence rules;
- resource kinds, instance topology, capacity, bank geometry, visibility, and
  coherence;
- copy, MMA, vector, reduction, and synchronization atoms;
- atom operand/result layouts and type constraints;
- engines, events, barriers, and pipeline capabilities;
- cost hooks and legality predicates.

The same logical hierarchy can target GPU blocks and warps, CPU thread teams
and vectors, an accelerator core hierarchy, or a simulator. The verifier
rejects a schedule whose resource-instance or visibility maps are illegal.

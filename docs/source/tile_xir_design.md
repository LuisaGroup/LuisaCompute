# TileIR → XIR: execution planning and SIMD realization

Status: first executable CPU realization, September 5, 2026. The finite
root-mapping solver below is implemented. General Tile distribution, packed
matrix atoms, software pipelining and measured cost calibration are not.

This document complements the [language/layout design](tile_programming_design.md),
[target-independent planner formulation](tile_execution_planner.md) and
[Runtime integration](tile_native_runtime.md). It does not redefine the DSL for
the CPU backend.

## 1. Preserve the execution-first program

The frontend declares **which independent programs and ordered computations
exist**, not a fixed CPU vector width or GPU block arrangement. For example:

```cpp
auto definition = tile_kernel("gemm", [=](TensorView<const float, 2> A,
                                         TensorView<const float, 2> B,
                                         TensorView<float, 2> C) {
    auto gm = axis("gm", ceil_div(M, BM));
    auto gn = axis("gn", ceil_div(N, BN));
    auto m = axis("m", BM), n = axis("n", BN), k = axis("k", BK);
    for (auto &nest : parallel(shape(gm, gn))) {
        auto m0 = nest.index(gm) * BM, n0 = nest.index(gn) * BN;
        auto acc = zeros<float>(shape(m, n));
        for (auto &step : nest.pipeline(shape(ceil_div(K, BK)),
                                        {.stages = 2, .initiation_interval = 1})) {
            step.stage("load");
            auto k0 = step.index() * BK;
            auto a = A.tile(coord(m0, k0), shape(m, k)).load();
            auto b = B.tile(coord(k0, n0), shape(k, n)).load();
            step.stage("compute");
            acc = mma(a, b, acc);
        }
        C(coord(m0, n0), shape(m, n)).store(acc);
    }
});
```

The loop variables are Nests, loads produce Tile SSA, assignment captures
loop-carried dataflow, and stores are explicit effects. No CPU-only `lane`,
`mma_team`, memory-owner annotations or builder-qualified math leaks into the
program. Several independent resources may be used by the same Nest.

## 2. Module and ownership boundaries

```{figure} ../_static/tile/xir-planning-pipeline.svg
:alt: One TileIR program feeds the planned XIR/SIMD path and separate Metal-native and TIRx routes.
:width: 100%

Planner, bridge, backend and Runtime are separate owners. No Python source or AST reconstruction is inserted between TileIR and XIR.
```

| Component | Owns | Does not own |
|---|---|---|
| TileIR | Typed semantic operations, nested regions, mutable use-def structure | LLVM, TVM, device queues |
| XIR planner | Finite candidate enumeration, target facts, cost decomposition | Runtime allocation or JIT |
| XIR lowerer | An owned XIR Module and typed argument/dispatch metadata | AST reconstruction or serialization |
| SIMD backend | Schedule/LLVM compilation, native shader and CPU dispatch | A second Tile language |
| Runtime adapter | Shader lifetime, argument/range checking, normal dispatch commands | Hardware scheduling policy |

Public headers live in `include/luisa/tile/bridge/xir/`; implementations live
in `src/tile/bridge/xir/`. The bridge links TileIR and XIR, not TVM, LLVM,
SIMD or Runtime. Native Metal lowering remains in the Metal backend. A future
backend may consume the same XIR result without relocating this bridge.

The input Module is borrowed and unchanged. The lowerer returns an owning
Module, not a dangling function pointer. Passes may rewrite its basic blocks,
PHIs, use lists and instructions; this is deliberately not a wire format.
The SIMD adapter runs the existing shared SSA optimization factory, then
CFG simplification and reachable-block verification. It does not rerun AST
destructuring or inlining on already plain SSA. Diagnostic LLVM capture is
independent of assembly capture, so normal compilation does not perform a
second machine-code compilation merely to retain source identity.

## 3. Formal mapping: ancestry plus local access

Let the root independent domain be a finite box
`D = [0,d0) × ... × [0,dr−1)`. A candidate permutation `π` is ordered from
outermost to innermost execution axis. Define:

```text
jπ(c) = Σt c[π(t)] × Πu>t d[π(u)]
c = unflattenπ(j),  0 ≤ j < P,  P = Πi di

block   = floor(j / B)
packet  = floor((j mod B) / W)
lane    = j mod W                 (B is divisible by W)
```

`B` is the logical workers per Runtime block; `W` is the backend's packet
width. Incomplete final packets are masked by the existing SIMD ABI. The
worker pool assigns blocks dynamically; a block is not pinned to a specific
OS thread. The permutation is a bijection of the **existing** parallel
instances. It does not require proving the independence already promised by
`parallel`, and it does not create new independent instances.

For a local Tile coordinate `e`, lexical descendant coordinates `s` and a
view origin `o`, the logical access is:

```text
ancestor coordinates c, descendant coordinates s, local Tile coordinate e
                        │
                        ▼
            logical buffer coordinate v = o(c, s) + e
                        │
                        ▼
      compact row-major element address = Σi v[i] × stride[i]
                        │
                        ▼
             bound BufferView byte offset + sizeof(T) × address
```

The execution map changes `j ↔ c`, not the buffer strides. Each buffer may
have a different origin, logical shape and access relation at the same scope.
This is the concrete subset of the general typed composition
`AddressMap ∘ ViewMap ∘ LocalAccess ∘ AncestorProjection`.

The current exporter materializes compact static buffer indexing only. The
language's richer layout algebra and proof system are not all realizable by
this exporter yet. Representability in TileIR must not be confused with
backend support. Unknown layouts/bindings fail closed.

Before creating bounds diamonds, the exporter derives integer intervals from
actual Nest coordinates and supported signed-i64 expressions. Checked
add/subtract/multiply and positive constant division/modulo may prove an
axis access in bounds. Negative offsets, unknown expressions and any possible
signed overflow keep the guard. This proof is separate from the cost model's
floating-point address-slope estimate. It neither asserts noalias nor moves a
load across an effect. LLVM simplification alone is too late to prevent
unnecessary per-element branches from inflating the earlier SIMD Schedule.

## 4. The implemented solver

### Candidate space and hard constraints

The first solver searches the Cartesian product of:

- All permutations of root parallel axes, unless an exact order is supplied.
- Block worker counts `{32, 64, 128, 256, 512, 1024}`, unless fixed explicitly.

The target packet width is an existing Device property, not a compiler guess.
Block counts must satisfy XIR's block-size contract and be divisible by that
width. Root domains must be static, nonempty, uint32-addressable and independent;
there must be one root parallel with no escaping state. Supported descendant
regions retain their existing local order. Unsupported operation, explicit
binding or manual resource requirements are rejected.

The solver enumerates the entire declared finite space and returns its exact
minimum **under the specified cost function**. It is not a globally optimal
hardware scheduler. The default budget is 1024 candidates; exceeding it is an
error asking for tighter constraints, not silent partial search. Ties are
deterministic. Input IR verification and the lowerer's own legality checks
remain authoritative; a low score never makes unsupported code legal.

### Cost units and formula

`ExecutionCostModel` is an **uncalibrated relative-work prior**, not nanoseconds,
hardware instruction counts or measured cache behavior. Default weights:
arithmetic 1, broadcast load 1, contiguous memory 2, gathered lane 2, block
dispatch 128. All coefficients must be finite and nonnegative.

For each candidate, the estimator counts static Tile work, local-loop
repetition, ordered MMA multiply/add work, and Tile-extract selection work.
It estimates a buffer's flat address slope relative to the innermost root
axis, using operand identity and supported constant/linear expressions.
Slope zero on a load has a broadcast prior; absolute slope one has a
contiguous prior; other or unknown addressing pays the gather prior times W.
An innermost extent not divisible by W conservatively doubles memory work.
These classifications are **not passed to codegen as proven facts**.

Let `a,m` be estimated arithmetic/memory work per packet, `H` available CPU
workers, `P` root programs, `Q=ceil(P/W)`, `L=ceil(P/B)`,
`h=min(H,L)`, `waves=ceil(L/h)`, and `d` the dispatch weight:

```text
arithmetic = a × Q / h
memory     = m × Q / h
dispatch   = d × waves
imbalance  = max(0, waves × ceil(min(P,B)/W) − Q/h) × (a+m)
score      = arithmetic + memory + dispatch + imbalance
```

All four terms are retained in the plan and reported in shader realization
metadata, along with the selected order and candidate count. This homogeneous
wave model intentionally does not pretend to model the M1's heterogeneous
cores, cache sharing, variable mask density, spills or actual thread timing.

### Reproducible fixed-plan controls

```cpp
bridge::xir::PlannerOptions options;
options.block_size = 64;
options.root_axis_order = {0, 1}; // outer-to-inner; exact, not a hint
auto shader = tile::compile(device, kernel, {.xir = &options});
```

`CompileOptions::threads_per_group` and the XIR block constraint must agree
when both are supplied. Configuration is borrowed only during synchronous
compilation. Metal rejects XIR options rather than silently ignoring them.

Searching a physical plan is distinct from tuning the C++ specialization:
changing BM/BN/BK or the semantic pipeline shape simply recaptures the lambda.
An outer JIT tuner can search these variants, using the same correctness and
measurement gates. There is no capture-once restriction.

## 5. Lowering invariants and supported behavior

| Tile semantics | XIR realization |
|---|---|
| Tile value | One scalar SSA value per local element, packed across independent workers later |
| Named dimensions | Identity-based projection/broadcast; names are diagnostics |
| Load snapshot | Load at the source operation before subsequent effects |
| Bounds/fill | Per-axis guards; actual load executes only in the valid branch |
| Store | Explicit guarded buffer effect, including BufferView offsets |
| Loop-carried assignment | Header PHIs; zero-trip initial state and simultaneous edge updates |
| Pipeline/stage | Ordered CPU loop and source-order phase cuts; no claimed physical overlap |
| MMA | Ordered multiply/add expansion with initial accumulator and dimension contraction |
| `ite(c,t,f)` | Correctly reordered to XIR's `SELECT(f,t,c)` |

The checked expansion budget defaults to 262144 values. Supported scalar
types are bool, i32/u32, i64/u64, f32/f64; fp16/bf16 are not yet supported by
this lowerer. Cooperative group/subgroup bindings, explicit manual Memory,
arbitrary resource/address mappings and multi-launch programs remain outside
the implemented subset. SIMD vectorization of workers is not tensor-core or
matrix-extension lowering.

## 6. Why not split every Tile into more workers?

Changing the enumeration of independent root programs is safe. Introducing
new worker boundaries *inside* one program needs additional justification.

```text
One original worker:
  x = input.tile(...).load();   // snapshot of all elements
  output.tile(...).store(x);   // input/output may overlap

Naive split:
  worker 0 loads and stores its element
  worker 1 may load after worker 0 overwrites its source
```

The second program can violate the first program's semantics even if output
coordinates are distinct. Const input views are not noalias promises.
Reductions, dynamic extraction and shared loop-carried state introduce further
dependencies. Thus a general distribution candidate must carry a dependence
and alias proof, a collective realization, or a checked invocation contract
with a safe fallback. Shape alone is insufficient.

## 7. Extension plan: richer plans, not more DSL entities

```{figure} ../_static/tile/xir-mapping-roadmap.svg
:alt: The implemented root-order and packing search precedes future Tile partitioning, collective atoms, resource planning and physical pipelining.
:width: 100%

Dashed boxes are planned extensions. Every new search family needs a supported emitter and its own correctness obligations.
```

New candidate families enter only when their emitters and proof obligations
exist. A small space uses exhaustive search. A larger factorized space may use
dynamic programming or branch-and-bound; ILP/CP-SAT is useful for discrete
resource and dependence constraints; annealing/beam search may propose
profitable candidates. Approximate solvers must report budget, explored space
and absence of a global guarantee. None is implemented by merely adding the
algorithm's name to a cost model.

Calibration should measure independent mechanisms and retain uncertainty:
packet memory coherence, masked work, dispatch batching, arithmetic mix,
working sets, compile size and spills. Evaluate top-choice regret, top-K
coverage and JIT cost on held-out shapes **and held-out operator families**.
Do not fit a GEMM-only model and label it an LLM model. Cache keys must include
IR specialization, plan schema, compiler/device identity, numerical policy and
cost-model revision; structural transforms invalidate affected plans.

A future Machine TileIR should expose typed realized atoms, execution maps,
resource instances, layouts, lifetimes and synchronization so passes can
inspect and rewrite them. It should not duplicate frontend math or become a
serialized backend instruction list. The current `ExecutionPlan` and XIR
Module are concrete, smaller stepping stones, not a claim that this full
intermediate representation already exists.

## 8. Validation entry points

- `test_tile_xir`: typed ABI, output verification, repeat lowering, bounds on
  expansion, unsupported bindings, permutation legality, exact minimum and
  fixed-plan/budget failure cases.
- `test_tile_xir_runtime`: ragged/transposed GEMM, nonzero initial values,
  changed non-dyadic inputs, reductions/softmax, offset views, guards, shader
  moves, zero-trip loops and read/write snapshot recurrences.
- `test_tile_xir_llm`: normalization, activations, RoPE, masked softmax and
  online prefill/decode/GQA; same capture through XIR and native-target TIRx,
  each checked independently against an FP64 oracle.
- `test_simd_phi_parallel_copy`: pure PHI cycles, uniform/varying loops,
  packet widths 1/2/4/8/16 and every active-lane count, independent of TileIR.
- `benchmark_tile_xir`: isolated warm host-wall timing, full output export,
  realized plan and LLVM identity, with planned/canonical/reversed controls.

These are validation mechanisms, not by themselves performance results. See
the [status and evidence report](tile_status_report.md) for actual runs,
limitations and links to raw evidence.

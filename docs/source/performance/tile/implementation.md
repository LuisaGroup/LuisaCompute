# Tile implementation coverage

## What is implemented, and what remains design

| Area | Implemented and exercised | Important remaining boundary |
|---|---|---|
| C++ surface | Signature parameters; range-for Nests; direct carried assignment; explicit stores; Tile-level operations | Not arbitrary C++ capture or intra-kernel SIMT/Tile mixing |
| Execution | `parallel`, `serial`, `pipeline`, `reduce`; scope constraints | The backend must realize the requested binding; unsupported bindings are errors |
| Data/layout | Typed layout representation and proof mechanisms; Tensor as storage plus layout/view | Not every represented layout has an emitter on every bridge |
| TileIR | Mutable typed SSA, regions and intrusive ownership/use structure | General Machine TileIR and its pass suite are not implemented |
| TIRx | Native C++ export preserving pure multi-consumer SSA; target-selectable recomputation; CPU/Metal realizations; typed MPP modes and optional proved K/M/N-tail views/bounded output; Metal FP32 subgroup reductions; bounded target-specific cost/solvers; opt-in rectangular group-program traversal | MPP bounds require optional capabilities and canonical proved guards; arbitrary masks retain fallback storage; traversal is explicit and Metal-group-only; materialization/reuse lacks traffic/spill calibration; broader atoms/operators remain necessary |
| Native Metal | Typed FP32 MMA/view-forwarding subset; ordinary Runtime shader and launch | Not general epilogues, K pipelines, manual Memory, all dtypes or arbitrary operators |
| XIR/SIMD | Direct verified XIR; local Tile expansion; loop PHIs; ordinary CPU Runtime | No matrix-extension atom, packed GEMM microkernel or general Tile distribution |
| CPU planner / realizations | Root-axis permutations × legal worker-block widths; bounded storage/SIMD/launch choices; proved CBLAS and Accelerate atoms | Provider selection is explicit; no fitted break-even model, whole-program optimum, general Tile partitioning or physical pipeline solver |
| Autotuning | Recapture/JIT variants, Cartesian execution/resource/materialization candidates, exact Metal reduction-width sweeps and frozen-plan benchmarking | Broader search requires legal emitters and measured ranking; one capture is not mandatory |
| Composed Metal programs | Automatic cooperative admission for supported MMA-containing programs, singleton-axis projection and pipeline capacity reservation | No general reduction redistribution, sibling-scope fusion or calibrated whole-region resource/time solver |
| Execution calculus | Documented contracts and finite reference tests | Proposed refinement rules and relative-completeness scope, not machine-checked production correctness or a novelty result |

The existing CuTe-derived mixed-radix/composition design is not a claim of a
complete decision procedure over arbitrary programs. The language design
distinguishes representational closure, proof fragments, finite fallback and
unknown results. Likewise, XIR's current compact-buffer realization is a
subset of the layout representation, not an alternative, less general DSL.

## Generality and attribution

The goal is reusable optimization over IR structure, not a table of operator
names or favorable shapes. Three different claims require separate evidence:

- **Semantic applicability:** a transformation matches proved access,
  dependence, ownership and numerical contracts, using independence already
  supplied by valid parallel primitives rather than re-proving it. Pointwise grid fusion and
  shared-SSA scalarization can serve different expression graphs, including
  several independently written output domains; canonical
  add/max/min reduction mapping serves several row programs. Matrix input
  forwarding serves the admitted affine zero-padded MMA family, not every
  operator containing a reduction.
  The optional [closed matrix epilogue](../../internals/tile/matrix.md#scalar-epilogues-use-the-same-element-owner)
  similarly matches an ordinary scalar DAG and same-element accesses. It
  rejects manual memory, neighbor reads, extra consumers and unproved inputs;
  it does not introduce activation-specific production primitives.
- **Realization coverage:** a legal execution/resource combination has a
  target emitter. The M/N-tail extension expands this space without changing
  planner coefficients or the subgroup distribution. Existing staged/JIT
  selection can then use newly legal K blocks. That is not evidence of better
  analytic ranking, nor does it improve native MPP or XIR automatically.
- **Performance generalization:** a frozen policy must be tested on disjoint
  shapes and composed operators, with unchanged incumbents, complete output
  checks, both GPU/E2E objectives and reported regressions. Current small-row
  held-out failures and large-GEMM gaps show that this claim is still open.

The latest matrix epilogue replay makes this distinction concrete: the same
rule releases storage for both ReLU and GELU, yet has sizeable regressions
on large GEMMs. It is opt-in through `PlannerOptions::fuse_matrix_epilogues`,
with scalar math retained in the cost proxy. Additional memory operands
(bias/residual), free variables, arbitrary layouts/dtypes and other backend
emitters remain outside this first contract. Staged/JIT can compare candidate
realizations; a legality proof alone must not silently select one as faster.

Execution partitioning, worker ownership, materialization/reuse and atom
selection should remain independently represented choices with coupled
legality/resource checks. Shared analysis and search consume backend-owned
capabilities and costs; target emitters implement the selected contracts.
This does not require one identical schedule or cost profile across CPU and
GPU, and does not add operator-specific concepts to the public DSL. The next
acceptance work must exercise these choices across pointwise chains,
reductions/normalizations and matrix-based compositions, rather than only
selecting another GEMM winner. Attention, convolution/filter and sort/Top-K
PoCs are correctness coverage, not a broad optimized-performance claim.

## Next work and acceptance criteria

1. **Generalize the CPU atom catalog:** add layout/stride/transpose and fused
   epilogue contracts without turning whole operators into DSL primitives.
   Select reference, library and native microkernel atoms with an explicit
   break-even model; preserve the current opt-in policies as controls.
2. **Close the direct XIR/reference gap:** choose Tile/vector axes, reduction
   trees, register blocking and cache/packing only with dependence, alias and
   numerical proofs. Provider parity must not hide the missing general SIMD
   and matrix realization family.
3. **Calibrated cost and search:** use MPP v2, the CPU launch threshold and the
   exposed residual-LayerNorm regret as bootstrap evidence. Add duplicated
   global/local traffic, expression depth, live-state and measured spill
   features, then evaluate on disjoint shapes/operators; report held-out
   regret, top-K coverage and uncertainty.
4. **Production LLM coverage:** add hidden widths/context lengths, mask corner
   cases, dtypes and realistic prefill/decode sizes. Benchmark fused and
   unfused XIR/TIRx/Torch paths with identical inputs and explicit math policy.
5. **Generalize Metal MPP planning:** retain MPS, handwritten MPP, original
   TIRx, staged TIRx-MPP and native-MPP controls. Extend the legal realization
   family and test v2's rectangle/K/thread features on held-out GEMMs and
   production LLM operators. Do not turn the winning 128×32 schedule into a
   shape table; this cohort's GEMM parity is not universal library parity.
6. **Machine TileIR when needed:** promote realized maps, atoms, resource
   lifetimes and protocols into mutable typed records when multiple passes
   need them. Keep the public DSL minimal and avoid a new serialization layer.

Open questions are therefore concrete: which dependency-safe distribution
space pays off first, what calibrated features predict held-out performance,
and which physical realization explains the remaining library gap? The
current evidence supports pursuing those questions, not declaring completion.

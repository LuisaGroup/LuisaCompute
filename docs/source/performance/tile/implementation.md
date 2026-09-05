# Tile implementation coverage

## What is implemented, and what remains design

| Area | Implemented and exercised | Important remaining boundary |
|---|---|---|
| C++ surface | Signature parameters; range-for Nests; direct carried assignment; explicit stores; Tile-level operations | Not arbitrary C++ capture or intra-kernel SIMT/Tile mixing |
| Execution | `parallel`, `serial`, `pipeline`, `reduce`; scope constraints | The backend must realize the requested binding; unsupported bindings are errors |
| Data/layout | Typed layout representation and proof mechanisms; Tensor as storage plus layout/view | Not every represented layout has an emitter on every bridge |
| TileIR | Mutable typed SSA, regions and intrusive ownership/use structure | General Machine TileIR and its pass suite are not implemented |
| TIRx | Native C++ export preserving pure multi-consumer SSA; target-selectable recomputation; CPU/Metal realizations; typed MPP v2 modes and optional proved K-tail views; Metal FP32 subgroup reductions; bounded target-specific cost/solvers | MPP view forwarding still requires full M/N; materialization lacks traffic/spill calibration; broader atoms/operators remain necessary |
| Native Metal | Typed FP32 MMA/view-forwarding subset; ordinary Runtime shader and launch | Not general epilogues, K pipelines, manual Memory, all dtypes or arbitrary operators |
| XIR/SIMD | Direct verified XIR; local Tile expansion; loop PHIs; ordinary CPU Runtime | No matrix-extension atom, packed GEMM microkernel or general Tile distribution |
| CPU planner / realizations | Root-axis permutations × legal worker-block widths; bounded storage/SIMD/launch choices; proved CBLAS and Accelerate atoms | Provider selection is explicit; no fitted break-even model, whole-program optimum, general Tile partitioning or physical pipeline solver |
| Autotuning | Recapture/JIT variants, Cartesian execution/resource/materialization candidates, exact Metal reduction-width sweeps and frozen-plan benchmarking | Broader search requires legal emitters and measured ranking; one capture is not mandatory |

The existing CuTe-derived mixed-radix/composition design is not a claim of a
complete decision procedure over arbitrary programs. The language design
distinguishes representational closure, proof fragments, finite fallback and
unknown results. Likewise, XIR's current compact-buffer realization is a
subset of the layout representation, not an alternative, less general DSL.

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

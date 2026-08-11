# XIR Pass Correctness Audit

This file is the canonical checklist for the semantic audit of the XIR pass
library. A checked item means that the implementation, all production call
sites, and the named regression tests have been reviewed against the contract
below. Merely compiling or passing an existing happy-path test is not
sufficient.

## Global proof obligations

Every transform must establish the following obligations, or reject the input
without mutation:

- **Type preservation:** every replacement has the same XIR type and value
  category required by all of its uses.
- **Use-def preservation:** no linked instruction refers to an unlinked value;
  replacement and deletion order keeps use lists valid at every mutation step.
- **SSA preservation:** every definition dominates each non-Phi use; every Phi
  has exactly the incoming CFG edges required by its block, with edge-specific
  values dominating the corresponding predecessor.
- **CFG preservation:** executable successors, structural child roles, merge
  roles, and continue roles are not conflated. Structured `IfInst`,
  `SwitchInst`, `LoopInst`, and `SimpleLoopInst` remain structured unless an
  explicitly requested destructuring boundary is crossed. Raw multi-way CFG
  uses `IndexedBranchInst`.
- **Memory preservation:** volatile operations, atomics, barriers, calls, and
  potentially aliasing accesses retain their observable order. Pointer or
  resource `Value *` inequality is not, by itself, a no-alias proof.
- **Arithmetic preservation:** rewrites account for integer width, wrapping,
  signedness, division edge cases, NaNs, infinities, signed zero, and
  reassociation/fast-math requirements.
- **Failure atomicity:** all rejection checks happen before the first mutation,
  or the pass uses a transaction/rollback mechanism.
- **Termination:** fixed-point pipelines have a monotone measure or a hard
  iteration bound; an individual pass does not report a change when it made
  none.
- **Metadata/debug preservation:** semantic metadata is cloned or deliberately
  reconstructed; dropping non-semantic debug metadata is documented.
- **Verifier closure:** a successful transform produces verifier-valid XIR.
  Rejection leaves byte-for-byte-equivalent XIR modulo explicitly documented
  analysis caches.

## Required test pattern for every discovered defect

- [x] Add a minimal regression that fails before the fix.
- [x] Add at least one generalized sibling case (inverted branch, alternate
      type/width, nested construct, alias, zero/one-trip, or multi-exit as
      applicable).
- [x] Assert both semantic/result behavior and verifier validity.
- [x] Assert rejection is mutation-free when the pass cannot prove safety.
- [x] Exercise composition with the adjacent production pipeline stages.

## Phase A: inventory and production wiring

- [x] Public headers match implementation files and CMake/XMake registration.
- [x] Every pass has a direct unit-test target or a documented analysis-only
      consumer test.
- [x] Every production call site is recorded as structured-CFG, plain-CFG, or
      either-CFG.
- [x] Deleted generic `lower_switch` and `loop_unroll` passes have no remaining
      build, include, test, or pipeline reference.
- [x] SPIRV-Tools loop unrolling remains unchanged and is not conflated with the
      deleted generic XIR pass.
- [x] Default pipelines contain only transforms whose contracts are proved for
      their invocation domain.

## Phase B: shared analyses and proof dependencies

- [x] `helpers` — memory effects, side effects, structured-control detection,
      cloning utilities, and pointer-base tracing.
- [x] `natural_loop` — reachable back-edge discovery, loop membership,
      preheader/latch/exit-edge classification, canonical continuation
      predicate, induction recurrence, and overflow-safe trip count.
- [x] `aggregate_field_bitmask`
- [x] `alias_analysis`
- [x] `call_graph`
- [x] `convergence_region`
- [x] `dom_tree`
- [x] `lex_scope_analysis`
- [x] `pointer_usage`
- [x] `post_dom_tree`
- [x] `scalar_evolution`
- [x] `uniformity_analysis`

## Phase C: structured and plain CFG transforms

- [x] `destructure_cfg`
- [x] `restructure_cfg`
- [x] `simplify_cfg`
- [x] `early_return_elimination`
- [x] `if_conversion`
- [x] `lower_break_continue`
- [x] `lower_ray_query_loop`
- [x] `lower_ray_query_loop_to_loop`
- [x] `loop_rotation`
- [x] `loop_fusion`
- [x] `loop_vectorization`

Corner-case matrix for every applicable CFG transform:

- [x] Empty, one-block, and unreachable regions.
- [x] Constant and dynamic conditions.
- [x] True/false arm inversion.
- [x] Nested `If`/`Switch`/`Loop` ownership.
- [x] Duplicate switch destinations and default/case aliasing.
- [x] Break, continue, return, unreachable, and mixed exits.
- [x] Multiple latches, multiple exit edges to one block, and multiple exit
      blocks.
- [x] Header/latch/exit Phis, including self-referential Phis.
- [x] Irreducible SCCs and generated dispatch regions.
- [x] Round trip: structured -> destructured -> transformed -> restructured.

## Phase D: SSA, scalar, and arithmetic transforms

- [x] `algebraic_simplify`
- [x] `const_fold`
- [x] `cvp`
- [x] `div_rem_pairs`
- [x] `early_cse`
- [x] `fix_self_referential`
- [x] `gvn`
- [x] `indvar_simplify`
- [x] `mem2reg`
- [x] `phi_cleanup`
- [x] `reassociate`
- [x] `reg2mem`
- [x] `scalarizer`
- [x] `sccp`
- [x] `simplify_libcalls`
- [x] `sroa`
- [x] `trace_gep`
- [x] `transpose_gep`

Arithmetic corner-case matrix:

- [x] Signed and unsigned 8/16/32/64-bit integers.
- [x] Minimum/maximum values and overflow/wrapping boundaries.
- [x] Shift counts at zero, width minus one, width, and above width.
- [x] Integer division/remainder by zero and signed minimum divided by `-1`.
- [x] Float NaN, infinity, signed zero, and subnormal behavior.
- [x] Strict versus fast-math/reassociation mode.
- [x] Scalar, vector, matrix, and aggregate type boundaries.

## Phase E: memory and vector transforms

- [x] `dce`
- [x] `dead_store_elimination`
- [x] `fuse_consecutive_buffer_reads`
- [x] `licm`
- [x] `local_load_elimination`
- [x] `local_store_forward`
- [x] `promote_ref_arg`
- [x] `slp_vectorization`

Memory corner-case matrix:

- [x] Same pointer/resource value.
- [x] Distinct arguments/views that may alias the same allocation.
- [x] Nested GEPs and overlapping byte ranges.
- [x] Read-after-write, write-after-read, and write-after-write dependencies.
- [x] Volatile reads/writes, atomics, barriers, and opaque calls.
- [x] Local versus shared versus global memory.
- [x] Unaligned, boundary, and mixed-width byte-buffer accesses.
- [x] Intervening side effects and control-flow boundaries.

## Phase F: interprocedural and callable transforms

- [x] `dead_arg_elim`
- [x] `dead_field_elimination`
- [x] `inline`
- [x] `outline`
- [x] `unused_callable_removal`

Interprocedural corner-case matrix:

- [x] Recursive and mutually recursive call graphs.
- [x] Multiple call sites with aliased reference/resource arguments.
- [x] Nested callables and captured resources.
- [x] Declaration-only/external functions.
- [x] Return values, reference arguments, and metadata/signature constraints.

## Phase G: autodiff and coroutine transforms

- [x] `autodiff`
- [x] `coro_cfg_distill`
- [x] `coro_materialize`
- [x] `coro_reg2mem`
- [x] `coro_split`

Autodiff/coroutine corner-case matrix:

- [x] Nested structured control flow and raw `IndexedBranchInst`.
- [x] Suspend/resume inside branch, switch case, loop body, and continue path.
- [x] Values live across suspend, including Phis and resources.
- [x] Multiple suspend points, empty states, and unreachable states.
- [x] Frame layout, dead fields, and state-machine restructuring round trip.
- [x] Autodiff loops at zero/one/bounded maximum trip counts and rejection above
      the local expansion bound.

## Phase H: pipeline framework and composition

- [x] `pass_pipeline` accounting, fixed-point convergence, and change reports.
- [x] Basic optimization pipeline.
- [x] Post-inline cleanup pipeline.
- [x] SSA optimization pipeline.
- [x] Post-restructure cleanup pipeline.
- [x] Vulkan native XIR -> SPIR-V structured optimization pipeline.
- [x] Fallback XIR -> LLVM pipeline.
- [x] Coroutine/state-machine pipelines.

## Audit conclusions and proof sketches

The audit uses a fail-closed rule: an optimization is accepted only when its
local proof establishes every precondition below. Unsupported input is not an
invitation to approximate the transformation.

| Area | Acceptance argument | Rejection / transaction boundary | Representative regressions |
| --- | --- | --- | --- |
| Structured versus raw CFG | `SwitchInst` owns a merge role; its selector and case/default operands are executable edges. `IndexedBranchInst` has the same executable multi-way edges and deliberately has no merge role. Destructuring changes only that role distinction. | Plain-CFG-only passes scan every owned block before mutation. `restructure_cfg` runs on a shadow module and commits only verifier-valid output. | `destructure_switch_preserves_multiway_edges_as_indexed_branch`, `restructure_rebuilds_switch_from_indexed_branch`, `restructure_roundtrips_loop_switch_nested_break_continue` |
| Post-merge selection re-entry | For a selection `(H, M)`, an edge `(P, E)` crosses back into its interior exactly when `H` and `M` dominate `P`, `H` dominates `E`, and `M` does not dominate `E`. The possible `H` values are exactly the dominator-tree ancestors of `E`; walking them deepest-first therefore selects the same deepest owner as an all-block scan. Loop-boundary membership is materialized once for each immutable CFG version. Exit-dispatch arms are searched through finite, side-effect-free forwarding chains, so a fallback proxy cannot hide the crossing. The `E`-owned region is cloned with `H`, `M`, and sibling entries as its frontier, and `P` is retargeted to the `M`-dominated copy. | Forwarding walks and clone discovery are cycle-guarded. Node splitting is applied one boundary edge at a time and the enclosing fixed point retains its hard bound; after every mutation, dominance and the exact loop-boundary relation are rebuilt before the next query. The shadow-module transaction rejects rather than commits any graph that still has a post-merge re-entry. | `restructure_nested_selection_exit_to_shared_continuation_converges`, `restructure_splits_dispatch_reentry_through_fallback_proxy` (including 128 reachable sibling selections outside the owner chain), `restructure_does_not_reenter_selection_after_its_merge` |
| Dominance, loops, and SSA | Dominance is computed over executable successors. Natural loops are reachable dominated back-edge closures; counted-loop consumers additionally require one preheader, one latch, one header-owned exit edge, a matching Phi recurrence, and a no-wrap trip count. Reducibility recursively decomposes each SCC through its unique header, so a multi-entry inner cycle cannot hide inside a natural outer loop. The optional lowering converges entry edges through one selector dispatcher without cloning the region body. | Multiple latches/exits, malformed Phis, and exhausted restructuring bounds are rejected before commit. `restructure_cfg` rejects irreducible regions atomically; `lower_irreducible_cfg` requires raw successor edges and preflights the complete module before mutation. | `natural_loop_discovers_multiple_latches_but_rejects_canonical_bounds`, `natural_loop_rejects_multiple_exit_edges_to_one_block`, `restructure_irreducible_scc_rejected_atomically`, `lower_irreducible_cfg_builds_one_entry_dispatch_without_cloning_body`, `lower_irreducible_cfg_finds_nested_region_inside_single_entry_outer_scc` |
| SSA storage metadata | `NameMD` is non-semantic storage identity. `mem2reg` may promote an alloca whose only annotation is one consistent name and transfers that name to every inserted Phi. A `reg2mem` spill reload with that same single name may recover the original Phi name. | Any other alloca/load/store metadata, or conflicting spill names, blocks promotion. If no replacement Phi exists, a storage-only name is deliberately dropped with the removed storage. | `mem2reg_promotes_named_alloca_and_names_inserted_phi`, `reg2mem_mem2reg_roundtrip_recovers_phi_name`, annotated load/store retention tests |
| Scalar and arithmetic rewrites | Replacements preserve the complete scalar/vector/matrix type. Integer evaluation uses declared widths and unsigned bit arithmetic where C++ signed overflow would be undefined. Strict floating-point rewrites preserve NaN, infinity, subnormal, and signed-zero behavior. | Target-dependent transcendental results and rewrites requiring reassociation remain in IR unless fast math explicitly authorizes them. Division by zero, `INT_MIN / -1`, and out-of-range shifts are not folded. | `constfold_signed_overflow_wraps_without_ub`, `constfold_shift_count_uses_its_declared_integer_width`, `algsimpl_float_mul_zero_keeps_nan_inf_semantics`, `constfold_rint_is_host_rounding_mode_independent` |
| Memory transforms | Elimination/forwarding requires an exact local object or a proven non-alias relation, exact byte ranges and alignment, and no intervening read/write/call/atomic/barrier hazard. | Unknown pointer/resource provenance is `may-alias`; opaque effects end the candidate region. An instruction whose metadata has no unique replacement owner is retained. | alias-analysis mixed-width/nested-GEP tests, byte-buffer footprint/overflow tests, `dse_retains_annotated_overwritten_store_*`, SLP side-effect barrier tests |
| Interprocedural transforms | Inlining validates argument categories, return shape, recursion, structured/plain CFG domain, Phi edge repair, and metadata ownership before cloning. Dead argument/callable transforms preserve constrained or externally visible signatures. | Selected-call-site inlining has a whole-plan preflight. Unmappable call/return/block metadata, declarations, recursive callables, and unsupported structured callees are retained. | `inline_selected_call_sites_preflight_is_atomic`, `inline_single_block_with_block_metadata_is_rejected_without_mutation`, `inline_multiblock_split_retargets_phis_in_all_successors`, unused-callable SCC/declaration tests |
| Autodiff and coroutine state machines | Scope cloning is definition-before-use, branch-role retargeting changes only executable edges, frame fields are a bijection between names, indices, and types, and live-across-suspend values are explicitly materialized. | Module worklists and distilled CFG metadata are completely validated before split/materialize mutation. Dynamic autodiff expansion has an explicit finite bound and overflow exit. | autodiff nested-loop/scope-slot tests; coroutine invalid-metadata atomicity, live aggregate, multi-suspend, frame remap, and restructure round-trip tests |
| Pipeline framework | Leaf changes are reported once; one-shot sequences are not mistaken for fixed points; fixed-point groups terminate after an unchanged round or report their hard bound as failure. | A zero budget or a still-changing final round is explicitly non-converged. Default pipelines exclude quarantined loop transforms and place structured/plain CFG adapters at named boundaries. | `fixed_point_converges`, `fixed_point_respects_max_iterations`, `fixed_point_zero_budget_is_reported_not_silently_converged`, `one_shot_sequence_can_change_without_false_nonconvergence` |
| Optional transform configuration | The native SPIR-V scalarizer is disabled by default, enabled by `ShaderOption::enable_scalarizer`, and explicitly overridden by `LUISA_XIR_ENABLE_SCALARIZER` when that variable exists (`1` enables; every other value disables). Both inputs are part of the Vulkan shader-cache identity. | Autodiff's required scalarization is a separate semantic phase and is not gated. A cache namespace bump prevents binaries produced under the former always-on behavior from being reused. | `vk_user_compute_scalarizer_option_and_environment_precedence`, width-preserving bitcast integration test, Vulkan shader-binary cache contract tests |
| Native SPIR-V planning | The structural closure follows every executable case/default edge, while the control-flow plan separately records selection merges, loop headers/continues, physical trampolines, and Phi predecessor remaps. | Raw `IndexedBranchInst` is rejected at the final structured dialect/codegen boundary. Ambiguous physical merge rotation and illegal loop entries/backedges fail planning. | `spirv_structural_closure_follows_raw_indexed_branch_edges`, `spirv_loop_multiple_breaks_nested_if_and_continue_validates`, `spirv_loop_switch_nested_exits_preserve_phi`, physical-boundary rejection tests |

### Deliberate soundness boundaries

- `loop_rotation`, `loop_fusion`, `loop_vectorization`, and `if_conversion`
  remain opt-in plain-CFG transforms. They are not present in default
  structured pipelines.
- `outline` is an explicit unsupported, mutation-free operation; it no longer
  silently reports success for an `OutlineInst`.
- Unreachable recursive callable SCCs are retained until SCC-wide deletion is
  implemented. Removing less code is preferable to releasing live
  function-use edges.
- Autodiff's local bounded-loop expansion is not the deleted generic XIR
  `loop_unroll` pass. SPIRV-Tools and LLVM backend unrolling are separate
  downstream optimizers and remain untouched.
- The removed DSL loop rotation/fusion/vectorization executables did not
  actually enable those opt-in XIR passes and therefore could only test the
  untransformed kernels. Their verifier-backed pass tests are the authoritative
  coverage. The ad-hoc `test_debug` and empty `test_minimal` executables were
  likewise not correctness tests.

### Defects fixed during this audit

- Structured `SwitchInst` now has an explicit merge; raw multi-way CFG uses
  `IndexedBranchInst`. Destructure/restructure, cloning, interchange, text,
  AST, verifier, coroutine, CUDA/fallback guards, and SPIR-V planning agree on
  that split.
- Nested loop `break`/`continue` and nested selection exits no longer create a
  physical SPIR-V edge that crosses a selection merge. Phi predecessors are
  remapped to the physical trampoline/merge selected by the plan. Redundant
  exit-state dispatches are collapsed only when pass-local provenance proves
  they were generated by restructuring and both forwarding arms have the same
  terminal target; equivalent-looking user selections are outside the rewrite
  domain. Multi-target exit dispatches now follow their final unconditional
  fallback proxy and split the exact dominance boundary when it re-enters a
  selection after that selection's new merge.
- Post-merge re-entry discovery no longer scans every owned block and
  re-traverses every loop region for each forwarding edge. Candidate owners
  are the exact dominator-ancestor relation of the edge destination, ordered
  deepest-first, and loop-boundary membership is materialized once per
  immutable CFG version. The pass report exposes relation-build, edge-query,
  and owner-query counts so scale regressions can distinguish structural
  nesting from unrelated graph width.
- The SPIR-V structural closure no longer omits raw indexed-branch targets, and
  final codegen fails closed if an un-restructured indexed branch survives.
- Vulkan bindless-property validation now compares the unbounded
  `array_size`; it no longer accidentally treats the enum value itself as the
  boolean condition.
- SCEV loop ownership is deterministic inner-to-outer, and natural-loop trip
  counts are rejected when the induction recurrence can wrap.
- Dead-store elimination retains annotated overwritten stores when no unique
  metadata replacement exists. Single-block inlining similarly rejects
  callee-block metadata that cannot be assigned a semantically equivalent
  owner.
- `mem2reg` no longer treats an AST local's debug `NameMD` as semantic storage
  metadata. It promotes named locals, transfers a consistent name to inserted
  Phis, recovers the same name after `reg2mem`, and still rejects every other
  annotated or conflicting-memory case.
- Autodiff cloning preserves definition order, evaluates the final prepare
  condition in SSA order, resets nested-scope slots, and retargets executable
  successors without rewriting declarative merge/body/update roles.
- Integer `POW_INT`, shifts, division/remainder, signed overflow, `ROUND`
  versus `RINT`, and host floating-point environment behavior now share an
  explicit cross-backend contract with regression coverage.
- The native SPIR-V scalarizer now obeys its public shader option and an
  explicit environment override instead of running unconditionally. Both
  values participate in the Vulkan cache key, and the cache namespace was
  bumped for the changed default.
- The bundled DXC invocation no longer passes
  `-fspv-use-unknown-image-format`, which that compiler rejects; compute,
  raster, and ray-tracing compilation use the same supported flag set.
- XIR-to-AST integration coverage now states the plain-CFG contract directly:
  a side-effect-free destructured diamond is converted to `select`, while
  non-convertible constructs are reconstructed and verifier-checked.

## Final validation gate

- [x] `git diff --check`
- [x] Full configured build, after the final source edit.
- [x] Every dedicated XIR pass test.
- [x] `unit_xir` label (including coroutine/XIR tests).
- [x] Full `unit` label.
- [x] `test_complex_kernel` on every available non-window backend.
- [x] Headless integration tests supported by available devices.
- [x] Hardware/window/material-dependent tests explicitly recorded as blocked,
      never reported as passed.
- [x] Staged diff contains only intended files.
- [x] Commit created only after all applicable gates pass.

Measured validation evidence on 2026-07-25:

- `cmake --build build-cmake-ninja-xir-llvm -j6`: complete configured build
  succeeded (935 compile/link steps in the final dependency rebuild).
- `ctest --test-dir build-cmake-ninja-xir-llvm -L unit
  --output-on-failure -j6`: 114/114 tests passed.
- `ctest --test-dir build-cmake-ninja-xir-llvm -L unit_xir
  --output-on-failure -j6`: 48/48 tests passed (35 base XIR tests plus 13
  coroutine/XIR tests).
- `test_xir_pass_restructure_cfg`: 43 tests, 940 assertions passed.
- `test_vk_spirv_codegen_path vk`: 85 tests, 2026 assertions passed,
  including native SPIR-V validation and scalarizer option/environment/cache
  precedence.
- `test_complex_kernel` passed 920 assertions on each available executable
  backend: Vulkan, HIP, and fallback. CUDA hardware was unavailable
  (`nvidia-smi` absent), so no CUDA execution result is claimed.
- VK/fallback buffer-read fusion, induction-strength-reduction, SLP, and XIR
  optimization integration executables passed. The nested-callable path tracer
  passed on VK with explicit `--offline --spp 1`; its default remains the
  interactive window path and was not launched in the headless test session.
Incremental validation evidence on 2026-07-31:

- The minimal three-target exit-dispatch regression fails with one illegal
  construct when the re-entry search is restricted to immediate dispatch
  targets, and succeeds when the same search follows the final unconditional
  fallback proxy.
- `test_xir_pass_restructure_cfg`: 57 tests and 1,072 assertions passed.
- `ctest -L unit_xir --output-on-failure -j32`: 48/48 tests passed after a
  complete 32-thread configured build.
- A cold Psycles Vulkan path-tracing kernel converged in four post-restructure
  iterations (`155 -> 317 -> 370 -> 371 -> 371` blocks), passed SPIR-V
  validation, and compiled successfully on RADV GFX1201. The restructure phase
  took 420 ms of 526 ms XIR legalization; complete shader JIT took 0.805 s.

Incremental validation evidence on 2026-08-01:

- A production Psycles volume-path cache miss exposed a width-scaling defect
  in post-merge selection re-entry discovery. Across the same six native
  XIR-to-SPIR-V module passes, `split_exit_dispatch_selection_reentries`
  consumed `30,399.418 ms` before the fix and `168.987 ms` after it.
- The complete `restructure_cfg_pass_run_on_module` time fell from
  `36,533.716 ms` to `6,269.162 ms`; whole-process wall time fell from
  `40.49 s` to `9.86 s`. Both measurements used
  `LUISA_XIR_TRACE_PASSES=1`, `LUISA_SPIRV_OPT_LEVEL=2`, an isolated empty
  cache, the same generated kernels, and the same RADV device.
- `test_xir_pass_restructure_cfg`: 57 tests and 1,076 assertions passed. Its
  fallback-proxy fixture now includes 128 reachable sibling selections that
  cannot dominate the re-entered block and pins owner queries to the
  destination's dominator-ancestor chain.
- A complete configured 32-thread build succeeded in `112.58 s`;
  `ctest -L unit_xir -j32` passed `48/48` tests in `0.46 s`.
- The wider `unit` label passed `115/116`; the deterministic unrelated
  `test_eastl_allocation` failure is confined to eight `fixed_vector`
  assumptions against the currently pinned
  `EASTL@d9d9a86560f5fe23d1eb559b20ae89e9e3676f5f`. It is recorded as a
  separate baseline defect and is not reported as passing validation for this
  XIR change.
- `restructure_cfg` now models mutation ownership explicitly. Its default
  `TRANSACTIONAL` mode retains shadow validation plus identity-preserving replay
  and leaves the input unchanged on every failure.
  `IN_PLACE_DISCARDABLE` retains the complete input/output verifier boundaries
  but invokes the mutating definition transform once; it is valid only for an
  exclusively owned module that the caller discards on failure. Native SPIR-V
  legalization satisfies that ownership contract and aborts code generation
  on any failed pass.
- On the same 15-definition, 6.83 MB Lone Monk path kernel,
  `restructure-cfg` fell from `34,336.31 ms` to `17,684.16 ms`, XIR
  legalization from `59,494.996 ms` to `42,961.471 ms`, and complete native
  AST-to-SPIR-V compilation from `75,476.640 ms` to `58,758.916 ms`.
  `definition_transform_invocation` reported 15 rather than 30 while
  `boundary_verifier` remained 2. The resulting SPIR-V artifact is
  byte-identical to the transactional baseline
  (`SHA-256`
  `10840657fcb78b5e5ba5e759ddf8987dc29de717af1e606976f5c4412f872cc7`).
- The same first cold compile immediately persisted both the 6.83 MB SPIR-V
  artifact and its 6.85 MB Vulkan pipeline cache. A second process with Mesa's
  disk shader cache explicitly disabled created the large compute pipeline in
  `1.702 ms` and completed shader JIT in `1.396 s`; before first-compile PSO
  persistence, the corresponding path took `41,287.313 ms` and `42.710 s`.
- The complete 32-thread build and `ctest -L unit_xir -j32` passed 48/48.
  `test_xir_pass_restructure_cfg` passed 57 tests/1,076 assertions,
  `test_xir_pass_mutation_safety` passed 26 tests/172 assertions,
  `test_vk_shader_cache vk` passed 1 test/8 assertions, and
  `test_vk_spirv_codegen_path vk` passed 86 tests/2,029 assertions.

Incremental validation evidence on 2026-08-04:

- A production Random Walk subsurface kernel reduced the failure to a raw
  three-block indexed branch whose five case labels shared one return block
  while the default label selected an unreachable block. Duplicate-label
  normalization left one direct case entry and four forwarding proxies.
- Single-exit canonicalization rerouted the four proxy edges through a fresh
  merge but left the equivalent direct header edge untouched. The shared
  return consequently became both a pre-merge arm entry and a post-merge
  continuation, which is exactly one post-merge selection re-entry.
- The fix treats the collected selection-exit edges as a graph cut and closes
  it over canonical-target equivalence classes. A declared arm is added only
  when its forwarding path has no existing cut edge; this moves the missing
  zero-length header-to-exit path without collapsing the distinct switch case
  proxies. The rule is independent of case count, value, return type, or
  shader provenance.
- `restructure_rebuilds_terminal_indexed_branch_with_aliased_cases` reproduces
  the reduced graph, failed before the fix, and now checks structured success,
  absence of post-merge re-entry, unique arm entries, target reachability, and
  second-pass idempotence.
- `test_xir_pass_restructure_cfg` passed 58 tests and 1,096 assertions. A
  complete 32-thread Psycles build also passed.
- With `PSYCLES_DISABLE_SHADER_CACHE=1`, the original RADV GFX1201 kernel
  completed the entire cold XIR-to-SPIR-V path: optimization reduced 243,328
  words to 229,290 words, shader JIT completed in 4.779 s, and the Vulkan
  backend rendered the 64x64 Random Walk probe and wrote its multilayer EXR.
- `LUISA_XIR_TRACE_PASSES=1` now reports stable module-definition ordinals and,
  for a small failed definition, a bounded function dump identifying the
  offending construct. This is trace-only diagnostics; default verification
  remains exactly once at pass input and once at pass output, with intermediate
  checks still controlled solely by `LUISA_XIR_VERIFY_INTERMEDIATE=1`.

## Incremental coroutine scalar/lifetime audit (2026-08-12)

The coroutine pre-distill pipeline now reuses SCCP and GVN over the actual
coroutine execution relation. An ordinary CFG traversal stops at
`CoroSuspendInst`, so applying either pass only to the entry component silently
ignored every resume component. `CoroSemanticGraph` augments ordinary
successors with each uniquely token-matched `suspend -> resume` edge and is
accepted only when that relation is valid. Invalid coroutine metadata falls
back to the ordinary pass domain; it is never guessed or partially augmented.

The audited contracts are:

- SCCP starts only the coroutine entry block as executable. A resume component
  becomes executable exactly when its matching suspend edge does. Its existing
  three-point value lattice is unchanged, and rewriting is restricted to the
  semantic graph's block domain. Consequently a constant in a reachable
  continuation folds, while a continuation reached only from a constant-dead
  suspend arm is neither visited nor rewritten.
- GVN uses augmented dominance to discover continuation-local equivalents, but
  rejects a leader-to-duplicate replacement whenever any semantic path between
  their distinct blocks may cross a suspension. Dominance alone proves value
  availability, not zero frame cost; this additional rejection prevents GVN
  from converting intentional rematerialization into a new live-across-suspend
  value. Same-block leaders are earlier in instruction order and execute on
  every visit. Loads, resource reads, ray-query state, and calls without a
  proved purity contract remain outside value numbering. Hashes only select a
  bucket; exact opcode, type, and operand-value-number comparison decides
  equality.
- Predicate-sensitive allocation lifetime proofs value-number only pure,
  total arithmetic expressions. Hash collisions are resolved by exact term
  comparison. Dynamically re-executed instruction leaves kill every dependent
  predicate fact, so a condition from an earlier loop iteration cannot justify
  a later read. Arguments and constants are stable leaves; special registers
  are rejected because their continuation semantics are not generally stable.
- The unconditional initialization proof is the greatest fixed point of a
  forward Must system, initialized at top with an empty lifetime-start boundary
  and predecessor intersection. The guarded refinement carries a bounded
  disjunction of predicate cubes with one Must fact set per cube. Subsumption
  intersects facts, and both predicate/state caps widen only by forgetting
  predicates and intersecting facts. Widening can therefore reject a legal
  contraction but cannot manufacture definite initialization.
- First-definition delay is admitted only for one full-root store, projections
  and loads as the only other pointer uses, a non-escaping pointer, a store
  dominating the observation common dominator, and an SSA value available at
  the new insertion point. Moving that unique store cannot change which value
  any load observes; unsupported partial stores, special-register values,
  reference uses, or Phi-edge uses are rejected without mutation.

Pass placement was measured rather than selected from one synthetic fixture.
SCCP runs after algebraic simplification and constant folding expose constants,
and before `simplify_cfg` consumes executable-edge information. GVN runs after
aggregate projection, rematerialization, SROA, and their DCE, then another DCE
removes the dead dependency chains before allocation lifetime placement. An
early-GVN A/B replaced 615 instructions but left rematerialization work
unchanged and increased the final distillation domain by 215 atoms
(`31,476` versus `31,261`). The retained late placement replaced 266
instructions in 4.14 ms, rejected 407 cross-suspend candidates, and did not
increase frame liveness.

Validation evidence:

- `test_xir_pass_coro_alloca_scope`: 16 tests / 117 assertions, including
  branch correlation, inverted predicates, dynamic-leaf invalidation, loops,
  exact subaggregate versions, bounded conservative rejection, and unique
  first-definition availability.
- `test_xir_pass_sccp`: 10 tests / 41 assertions, including reachable and dead
  token-matched continuations.
- `test_xir_passes`: 366 tests / 2,257 assertions, including a GVN continuation
  case that rejects the pre-suspend leader while merging the duplicate inside
  the resume scope.
- `test_coro_compile_trigger`: 10 tests / 25 assertions. Its aggregate contract
  now distinguishes stable rematerializable constants (one four-byte frame
  field) from two independently dynamic observed components (two fields).
- `ctest -L unit_xir -j$(nproc)`: 53/53 passed; `ctest -L unit
  -j$(nproc)`: 121/121 passed, including the EASTL allocation contract.

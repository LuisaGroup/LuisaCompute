# XIR CFG pass safety contract

XIR supports two control-flow representations:

- structured terminators (`IfInst`, `SwitchInst`, `LoopInst`,
  `SimpleLoopInst`, `BreakInst`, and `ContinueInst`) retain lexical information
  required by source-oriented code generators;
- plain CFG terminators (`BranchInst` and `ConditionalBranchInst`) are used by
  analyses and transforms that operate on graph edges directly.

An optimization must not silently cross this boundary. Only passes whose public
purpose is CFG lowering may turn structured constructs into plain CFG.

## Structure-preserving optimization

- `dce` computes executable reachability for constant `IfInst`, `SwitchInst`,
  and loop conditions, but keeps their structured terminators. It replaces only
  a constant `ConditionalBranchInst` with `BranchInst`. If a structured merge is
  not executable, the merge marker is cleared rather than used to keep dead code
  alive.
- `simplify_cfg` folds only plain `ConditionalBranchInst`. Empty-block threading
  must preserve structured entry/merge targets and keep PHI incoming blocks in
  sync.
- Analyses must treat merge blocks as metadata unless the analysis explicitly
  asks for structural reachability. Execution reachability alone is sufficient
  for DCE.

## Explicit lowering passes

`lower_switch`, `lower_break_continue`, `lower_ray_query_loop*`, and
`destructure_cfg` are explicit representation changes. Callers opt into their
documented lowering. `destructure_cfg` lowers `IfInst`, `LoopInst`,
`SimpleLoopInst`, `BreakInst`, and `ContinueInst` in every owned block; it does
not lower `SwitchInst`, so a fully plain CFG requires:

1. `lower_switch`;
2. `destructure_cfg`;
3. `simplify_cfg` when graph cleanup is desired.

A structured zero-case `SwitchInst` is rejected atomically by `lower_switch`:
turning it directly into `BranchInst` would erase its lexical merge frame.
Callers must inspect `LowerSwitchInfo::succeeded()` and stop with a clear error.

Every lowering must leave all owned blocks terminated and must update PHI
incoming blocks whenever an edge source changes.

## Plain-CFG-only transforms

`loop_unroll`, `loop_rotation`, `loop_fusion`, `loop_vectorization`, and
`if_conversion` reject structured input before mutation and report a non-zero
`structured_cfg_error_count`. Their current plain-CFG implementations are
conservatively quarantined until natural-loop discovery, SSA repair, multiple
exit handling, and verifier-backed cloning are complete.

`coro_cfg_distill` is a read-only plain-CFG analysis. `coro_split` and
`coro_materialize` are plain-CFG-only transforms: they preflight their complete
module worklist and reject atomically. The coroutine production pipeline runs
`lower_switch` followed by `destructure_cfg` before distillation.

Generic `xir_to_ast_normalize_module` exposes the generated, materialized
continuation callables but deliberately retains the source coroutine definition;
complete source-coroutine ownership and graph construction belong to
`compile_coroutine_pipeline`. Because exposed continuations intentionally have
no ordinary IR call users, generic normalization skips unused-callable removal
after generating them. It does not claim that the whole module is coroutine-free.

## Restructuring

`restructure_cfg` converts reducible plain CFG regions back into structured
control flow. A cyclic SCC with multiple entry blocks is irreducible for the
current implementation and is rejected before any mutation; callers inspect
`RestructureCFGInfo::irreducible_region_count`/`succeeded()`.

## Regression-test requirements

CFG tests must assert graph facts, not just pass counters: terminator kinds and
targets, merge ownership, PHI incoming blocks/values, block termination,
def-before-use order, and unchanged IR on rejected input. Corner cases include
constant untaken arms whose merge is unreachable, duplicate Switch targets,
zero-trip loops, infinite/non-exiting regions, disconnected owned blocks,
multi-entry SCCs, and structured coroutine regions spanning suspension scopes.

# XIR CFG pass safety contract

XIR supports two control-flow representations:

- structured terminators (`IfInst`, `SwitchInst`, `LoopInst`,
  `SimpleLoopInst`, `BreakInst`, and `ContinueInst`) retain lexical information
  required by source-oriented code generators;
- raw CFG terminators (`BranchInst`, `ConditionalBranchInst`, and
  `IndexedBranchInst`) are used by analyses and transforms that operate on
  graph edges directly.

An optimization must not silently cross this boundary. Only passes whose public
purpose is CFG lowering may turn structured constructs into plain CFG.

## Representation invariants

For every well-formed terminator, distinguish two relations:

- `succ(T)`, the executable successor set followed by dominance,
  post-dominance, SCC, reachability, PHI, and edge-rewrite algorithms;
- `role(T)`, the declarative blocks owned by a structured construct, including
  merge, loop body, and loop update/continue roles.

An edge rewrite may modify only `succ(T)`. It must not retarget a block in
`role(T)` merely because both relations currently name the same block.
Structural rewrites update roles explicitly and atomically.

`SwitchInst` and `IndexedBranchInst` share the same selector and executable
case/default edges:

```text
succ(switch) = succ(indexed_branch)
             = {default} union {case[i]}
```

`SwitchInst` additionally owns exactly one non-null merge role;
`IndexedBranchInst` owns none. Case literals are canonical unsigned bit
patterns at the selector width and are unique after canonicalization. Thus
destructure/restructure preserves the executable relation exactly while
removing and then reconstructing only the merge role.

## Structure-preserving optimization

- `dce` computes executable reachability for constant `IfInst`, `SwitchInst`,
  and loop conditions, but keeps their structured terminators. It replaces only
  a constant `ConditionalBranchInst` with `BranchInst`. Non-executable blocks
  still named by a structured role are retained as unreachable structural
  shells rather than allowing DCE to invalidate the construct.
- `simplify_cfg` folds constant raw `ConditionalBranchInst` and
  `IndexedBranchInst` terminators. Empty-block threading must preserve
  structured entry/merge targets and keep PHI incoming blocks in sync.
- Analyses must treat merge blocks as metadata unless the analysis explicitly
  asks for structural reachability. Execution reachability alone is sufficient
  for DCE.

## Explicit lowering passes

`lower_break_continue`, `lower_ray_query_loop*`, and `destructure_cfg` are
explicit representation changes. Callers opt into their documented lowering.
`destructure_cfg` lowers `IfInst`, `SwitchInst`, `LoopInst`,
`SimpleLoopInst`, `BreakInst`, and `ContinueInst` in every owned block.
`SwitchInst` becomes `IndexedBranchInst`, preserving its selector, case
literals, case/default edges, and metadata while deliberately dropping only
the structured merge role. `restructure_cfg` converts the raw multi-way
branch back to `SwitchInst` and reconstructs a valid merge.

Every lowering must leave all owned blocks terminated and must update PHI
incoming blocks whenever an edge source changes.

## Plain-CFG-only transforms

`loop_rotation`, `loop_fusion`, `loop_vectorization`, and `if_conversion`
reject structured input before mutation and report a non-zero
`structured_cfg_error_count`. Their current plain-CFG implementations are
conservatively quarantined until natural-loop discovery, SSA repair, multiple
exit handling, and verifier-backed cloning are complete.

`coro_cfg_distill` is a read-only plain-CFG analysis. `coro_split` and
`coro_materialize` are plain-CFG-only transforms: they preflight their complete
module worklist and reject atomically. The coroutine production pipeline runs
`destructure_cfg` before distillation.

Generic `xir_to_ast_normalize_module` exposes the generated, materialized
continuation callables but deliberately retains the source coroutine definition;
complete source-coroutine ownership and graph construction belong to
`compile_coroutine_pipeline`. Because exposed continuations intentionally have
no ordinary IR call users, generic normalization skips unused-callable removal
after generating them. It does not claim that the whole module is coroutine-free.

## Restructuring

`restructure_cfg` converts reducible raw CFG regions back into structured
control flow, including `IndexedBranchInst` back into `SwitchInst` with a
reconstructed merge. Reducibility is checked by recursively decomposing SCCs
through their unique headers; this detects a multi-entry inner cycle even when
a natural single-entry outer loop hides it from maximal-SCC analysis.
`restructure_cfg` rejects such a region before any mutation, and callers inspect
`RestructureCFGInfo::irreducible_region_count`/`succeeded()`.

Callers that accept arbitrary raw CFG may first run `lower_irreducible_cfg`.
For every multi-entry cyclic region it redirects both internal and external
entry edges through selector stores and one dispatcher. The dispatcher becomes
the unique header without cloning shader-body blocks. The function pass lowers
outer and newly exposed nested regions to a fixed point; the module overload
preflights every definition before mutating any definition.

The reconstruction order is inner-to-outer. Each indexed branch receives a
private merge block. A real common post-dominator is placed after that private
merge; if no real post-dominator exists because every arm terminates or exits
an enclosing construct, the private merge is an unreachable structural block.
After every CFG mutation, dominance/post-dominance facts used by the next
rewrite are recomputed. The postcondition is:

Non-local construct exits are represented temporarily by a finite state
protocol: every rewritten exit stores the identity of its original target,
enters the construct's fresh merge, and a generated dispatch selects that
target. The continue target of a loop is internal to that loop and is never
encoded as a non-local exit. This distinction prevents a local continue and an
enclosing break/continue from being joined into a selection that crosses the
eventual SPIR-V merge boundary.

Generated dispatch headers carry pass-local provenance distinct from their
current terminator role. A later restructuring step may turn such a raw header
into `IfInst`, but cleanup may collapse it only when both arms are
terminator-only forwarding chains with the same terminal target. Replacing
that dispatch by a direct branch preserves the successor relation for every
selector value; its now-write-only selector slice is then removed by
use-def-closed dead-expression deletion. User-authored `IfInst` nodes are never
eligible merely because they happen to have the same shape.

1. no `ConditionalBranchInst` remains except canonical `Loop.prepare`;
2. no `IndexedBranchInst` remains;
3. every structured selection owns a unique merge;
4. every construct exit reaches its own merge or a declared enclosing
   break/continue boundary without crossing a sibling construct;
5. rerunning `restructure_cfg` is graph-size idempotent.

## Regression-test requirements

CFG tests must assert graph facts, not just pass counters: terminator kinds and
targets, merge ownership, PHI incoming blocks/values, block termination,
def-before-use order, and unchanged IR on rejected input. Corner cases include
constant untaken arms whose merge is unreachable, duplicate Switch targets,
zero-trip loops, infinite/non-exiting regions, disconnected owned blocks,
multi-entry SCCs, and structured coroutine regions spanning suspension scopes.

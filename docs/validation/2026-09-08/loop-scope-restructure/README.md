# Loop epochs before selection-merge inference

The native SPIR-V continuation now restructures successfully. The repair is
in the generic XIR pass: recover natural loops before inferring indexed
selection merges, and enforce a finite progress obligation inside
`fixup_construct_exits`. No shader-specific branch, increased iteration
budget, removed executable payload, or backend escape hatch is involved.

## Failure and reduction

The original input is the complete pre-restructure native-backend module,
not the earlier coroutine-splitting module. It contains three functions;
the largest has 1,685 blocks, 57,534 instructions, 513 raw binary branches
and three indexed branches. Replaying this exact module without a GPU
reproduced the failure. Projecting arithmetic/resource operations to three
observable store markers, then reducing branches, produced this 11-block
CFG (`S` lists default followed by cases):

```text
entry   -> outer
outer   S unreachable, enter_inner, outer_payload
outer_payload: store -> outer
enter_inner C inner_guard, return
inner_guard C after_inner, inner_body
inner_body C inner_switch, inner_guard
inner_switch S inner_payload, inner_payload
inner_payload: store -> inner_guard
after_inner: store -> return
return
unreachable
```

Before the fix, an outer Switch merge was inferred inside the inner loop.
Exit repair then treated a genuine inner-loop edge as a nonlocal exit,
inserted a selector funnel, and exposed another apparent loop on the next
canonicalization. Two SimpleLoop headers alternated under the same parent;
each visit added exits instead of discharging them. The original module
grew past 72,000 blocks. The old comment that hierarchy distance always
decreased did not hold: the inferred regions were not a hierarchy.

This is a phase-precondition violation, not floating-point behavior,
texture sampling, a large coroutine frame, or a slow GPU compiler.

## Model and repair obligations

Let `V` be owned blocks, `R` entry-reachable executable blocks, and `E` the
executable edge relation. Luisa's ordinary dominator tree covers `R`, not
all of `V`, and does not turn declared merge/body/update roles into extra
execution edges. Physical SPIR-V constructs impose additional lexical
boundary constraints. These domains must not be interchanged.

1. A selection merge is inferred only after recoverable natural-loop
   epochs are known. A raw indexed arm may revisit its header; reachability
   alone cannot distinguish a selection continuation from a nested loop's
   next iteration. Indexed branches remain intact during loop recovery,
   then become Switches before generic If inference and entry cloning.
2. `try_restructure_loop` recovers **one** loop per successful invocation.
   No remaining binary branch is not a proof that no loop remains:
   unconditional and indexed backedges are possible. Remove that shortcut
   and rebuild analyses after each successful recovery.
3. Exit normalization must discharge, rather than recreate, an obligation
   `(construct header, current parent header)`. The drain records every
   discharged pair. Encountering that same pair again is an invalid,
   crossing/unstable ownership relation, and immediately fails closed.
   Reparenting to another ancestor is a different obligation.

Inside this drain no rewrite creates a structured header: fresh selector
dispatches and stubs are raw CFG. Consequently the header set `H` is fixed
and the discharged relation is a monotonically growing subset of
`H x H`. A rewrite either adds a previously unseen pair, finishes, or
reports invalid structure. This is a finite progress certificate, not a
time limit or a CFG-size cap. The phase-order correction fixes the valid
raw input; the certificate additionally protects partially structured
inputs and future phase regressions.

The existing exit-state protocol preserves an executable edge `A -> T`
as `A -> store(selector(T)) -> new_merge -> dispatch(T)`. Selector IDs are
assigned from stable owned-block order. Loop-continue edges are not loop
exits and are excluded from that funnel. The rest of the payload and
branch conditions remain unchanged. Both callers now propagate an invalid
construct result immediately. Transactional failure discards the shadow
and its new constants; it must not begin replay on the original objects.

## Audit across phase boundaries

The review covered loop recovery and prepare/update normalization,
indexed/binary selection inference, executable-edge retargeting,
construct-entry splitting, loop/selection boundary classification,
selection-exit dependency updates, construct-exit repair, final checks,
and transactional replay. The important cross-phase obligations are:

| Phase | Required invariant / progress evidence |
| --- | --- |
| Preflight and shadow creation | Typed, phi-free input; failed transactions preserve original objects and constants. |
| Natural-loop recovery | Consume an unowned backedge; invalidate dominance before the next recovery or selection query. |
| Selection inference | Use known loop epochs; each batch consumes existing raw conditionals and adds transparent merge subdivisions. |
| Boundary normalization | Preserve executable targets while separating lexical break/continue, prepare, update and merge roles. |
| Entry splitting | Clone only executable owned regions; enclosing boundary entries are subdivided, not payload-cloned. |
| Selection-exit drain | Requery dependencies on the exact graph version; repeated sites need decreasing invalid-exit counts. |
| Construct-exit drain | Inner-to-outer finite header/parent obligations; reject crossing ownership instead of unbounded growth. |
| Final boundary | Check owned residual branches, unique merges, canonical prepare/break/continue forms and the XIR verifier. |

This is not a claim of a machine-checked proof for the entire 10,000-line
pass. Three audit follow-ups remain explicit: allocator-order independence
of all mutating unordered-set scans (transactional replay relies on it),
equivalence of every ordinary-dominance epoch cut to physical SPIR-V
structural dominance, and a shared ranking proof for the full composition
of reentry splitting and canonicalization drains. The outer pipeline's
existing iteration bounds are not substitutes for those proofs.

The reference constraints are SPIR-V's
[structured control flow](https://registry.khronos.org/SPIR-V/specs/unified1/SPIRV.html)
and LLVM's
[SPIRVStructurizer implementation](https://llvm.org/doxygen/SPIRVStructurizer_8cpp_source.html).
LLVM's construct-exit repair walks one construct tree in postorder; Luisa's
repeated graph/tree rebuilding needs its own explicit progress argument.

## Permanent regressions

`test_xir_pass_restructure_cfg_loop_scopes` is registered in CMake and
xmake, with a 30-second CTest timeout as a harness guard, not the pass fix.

- The 11-block raw case runs with one/eight duplicate indexed cases,
  three block-creation orders and both mutation modes. It verifies success,
  retained stores/return, the production output contract and idempotence.
- A 21-block partially structured crossing-epoch input reproduces the
  repeated exit obligation even when natural loops are already declared.
  It must fail with invalid structure, not iteration exhaustion. The
  transactional module's serialized image must be exactly unchanged.

Canonical `Loop.prepare` remains a raw binary guard by design. The test
therefore checks that precise legal form, rather than using the blanket
no-raw-control-flow verifier, which would reject valid production XIR.

The old implementation timed out on both reduced regressions. The fixed
original full module reports success with zero residual/invalid constructs.
The first complete host run passes 152/152 unit targets; 100 repeated
regression runs and 1,000 graph-isomorphic creation-order permutations
also pass. The permanent creation-order expansion is revalidated by the
final all-target build, 153/153 selected unit tests (152 carrying the
literal `unit` label, plus one selected by another `unit*` label), and
another 100 repetitions of the expanded regression.

```sh
cmake --build build-codex-xir --parallel 32
ctest --test-dir build-codex-xir --parallel 32 --output-on-failure -L '^unit'
ctest --test-dir build-codex-xir --repeat until-fail:100 \
  --output-on-failure -R '^test_xir_pass_restructure_cfg_loop_scopes$'
```

## Complete application and evidence

The full unchanged Barbershop application compiled and rendered at
2048x858 / 64 spp after the CFG repair. Its six-stage coroutine remains
93 fields / 416 bytes. Strict native Vulkan also compiled the original
production lamp-routing fixture. Its subsequent *runtime* red case was
independent: both ordinary and coroutine kernels relied on an unspecified
miss-hit distance. Establishing Cycles' miss distance at the Psycles scene
traversal boundary made ordinary, SoA and AoS execution all pass without
another Luisa compiler change. No DXC route was used.

Local evidence root: `/var/tmp/psycles-lamp-routing-sCLxKs`.

- `spirv-input-4.xir`, SHA-256
  `c8f1adaad0b72690a0d00d0da34ef11f8252e7367d63eed1ee2262b85c9b93ae`:
  immutable full native input.
- `replay-full-red.log`, `reduced-small-switch-trace.log`,
  `loop-scope-regression-red.log`, `construct-progress-red-v2.log`:
  original failures. `construct-progress-red.log` was a rejected test
  draft and is **not** the crossing-epoch red proof.
- `replay-full-progress.log`, `construct-progress-all-unit.log`,
  `construct-progress-repeat-100.log`, `order-check.log`: fixed replay
  and host validation.
- `final-child-build.log`, `final-child-unit.log`: final expanded tests.
- `monolithic-control-native-vk.log`, `traversal-miss-native-vk.log`:
  distinct application miss-distance red/green results.

Temporary module/CFG capture instrumentation is removed from production.
These are compilation and correctness gates, not a renderer speedup claim.

# Scope-specific executable coroutine transitions

Status: isolated candidate built; 81/81 host XIR/coroutine tests pass. The full
original renderer module passes the actual production normalization boundary
under both frozen baseline and candidate libraries. Device integration is
pending. No renderer frame reduction or performance result is claimed here.
The predicate-retirement follow-up also passes the expanded host suite and the
same complete original module, without widening. Actual renderer integration
and device gates remain pending.

## Root cause and immutable red evidence

At SDK `98f4667ca678a3a1425ff4467e0d7803a0a0d12e`, distillation walks both
successors of each raw conditional branch separately from the latest
continuation's incoming state. A six-block, Phi-free raw XIR reproducer has
`search=true` at entry, yields A when true, assigns `search=false` after A,
and yields B when false. The original verifier accepts it, but distillation
reports all four edges `entry -> A/B` and `A -> A/B` instead of the two
executable edges `entry -> A` and `A -> B`. There are no structured Loop/If
operands in this counterexample.

The original host-only reduction and DSL controls are archived outside the
worktree at `/var/tmp/psycles-holdout-CXlJR7/coro-transition-predicate-reduction`.
The frozen baseline ran 5 test cases / 46 assertions, with 10 failed precision
assertions. It did not execute a shader or render a scene.

| Artifact | SHA-256 |
| --- | --- |
| Frozen reduction source | `510cf7400e4eeeca8f5a6b1002c6184915539ac5d6a56dc421f311fd5437d6ea` |
| Frozen baseline executable | `cc23dba87744bfd4b4029e2d64090fc440410814a1b139eca3724b7cbdb295f2` |
| Baseline log | `bf078d0b3d88cb18a4d22172893000fe72d910ab8103475228bfd0717b7e137a` |

The source and binary hashes above identify the immutable first reduction;
the permanent regression adds adversarial controls and is a different file.

## Abstract semantics

The forward May state is `(latest resume owner, executable block, possible
Boolean valuations)`. It reuses `CoroBooleanSetManager` and predicate liveness.
Boolean slots and SSA snapshots are different predicates. A direct store
strongly updates its private slot; a load defines a new snapshot; repeated
execution replaces the old dynamic instance of an SSA definition. Unsupported
Boolean producers forget their prior value. Constant, Boolean copy, NOT,
AND/OR/XOR and Boolean equality transfers are exact within this domain.

Only exact scalar local allocations with no address escape are tracked.
GEPs, casts, ordinary reference calls and unsupported uses disqualify a slot
globally. A callee's lack of a MUST-write is never used as a no-MAY-write proof.
Every writable Extension binding invalidates its tracked slot at the actual
suspension, for boundary, queued and resumed lifetimes alike. A prior SSA load
is not invalidated by a later external memory write. An ordinary branch into
a resume changes owner without inventing Extension effects.

Incoming states join by union. A conditional successor is selected only after
the monotone fixed point has seen exactly one possible arm for that owner.
Dead predicates are existentially projected; ROBDD budget exhaustion widens
toward unknown. Both operations can retain extra paths but cannot remove a
concrete execution. Unknown IndexedBranch/call-return selectors retain all
raw successors. Existing matched-call analysis remains conservative and is
not filtered by another owner's certificate.

## Compiler integration boundary

The chosen successor is sealed with each scope and the immutable source CFG.
Scope discovery, cross-scope transition discovery, local May/Must liveness,
and split operand dominance consume this same relation. Split emits a branch
to the sole executable successor and does not resolve the discarded target:
resolving it could synthesize an unwanted token store or fallback return.
This is not a graph-only edge deletion or a renderer-specific stage rule.

A surviving ordinary transfer into a resume remains legal even when its
matching static suspend was proved unreachable. Split validates the incoming
sealed transition instead of requiring a reached suspend for every owner.
Only executable suspensions contribute mandatory designated frame snapshots;
global input validation still checks every source export declaration.

`LUISA_CORO_DUMP_FRAME_LAYOUT=1` reports reachability state count, widening,
selected-arm count, and the full feasible token relation before frame liveness
and coloring. Current-ABI original-module capture, final graph and frame layout
have been checked below; candidate renderer execution remains an integration gate.

## Permanent regression and pending gates

`test_coro_transition_predicates` covers the original raw witness, dynamic
entry/resume controls, real self edges, conditional reference writes, all three
writable Extension lifetimes, preserved SSA snapshots, late diamond joins,
re-executed Boolean definitions, ordinary resume entry, shared suspending
callables, and full DSL mandatory-yield / terminated-background patterns.
Raw cases split and verify the actual module and compare emitted continuation
token stores against the certified transition set. A tampered arm certificate
must be rejected.

Executed in the isolated SDK root with matching rebuilt libraries:

```sh
cmake --build build --parallel 32 --target test_coro_transition_predicates
ctest --test-dir build -R '^test_coro_transition_predicates$' -V
ctest --test-dir build -L 'unit_xir|unit_coro' --parallel 32 --output-on-failure
```

All 79 selected executable targets were built with `--parallel 32` before the
full test command. Focused result: 11 tests / 161 assertions pass. Full result:
79/79 pass. Logs in the archived reduction directory are
`candidate-focused-2.log` and `candidate-host-ctest-2.log`; build logs and first
failed runs are also retained. The first focused follow-up exposed mandatory
collection of a dead export; its new assertion was preserved and the semantic
collection fixed. Four older distill tests used constant true while requiring
the false branch; their inputs now use genuine Boolean arguments, preserving
all prior bypass/export-rejection, greatest-fixed-point, transition-store and
shared-merge assertions.

Next validate the integrated renderer on HIP, fallback, and strict native
XIR-to-SPIR-V Vulkan. Do not load a pre-change
distill observer into this changed Scope ABI. No GPU gate has run for this
candidate yet.

## Complete original module: production-boundary replay

The fresh current-ABI observer links only the frozen production libraries.
Its complete original Barbershop parent module is archived under
`/var/tmp/psycles-holdout-CXlJR7/coro-distill-current-BsxSjd/barbershop-large-egmbFh`.
The module has 1,111,095 instructions and SHA-256
`3e62bd7b664d5deb3bb1afb122b20bf7c23c08d6762fc461eccfbe9de1131568`.
The diagnostic original render terminated 0. A temporary bounded transport
budget increase was required; the capture directory documents that isolated
writer/reader change and the frozen earlier serializer-limit failure.

Both matching-ABI replayers process that same complete input and preserve exact
serialization across read-only distillation. They split all six continuations,
materialize, detach the owned source, run cross-block reg2mem, and verify the
complete module at the production boundary. Both terminate 0. Actual emitted
nonterminal token-store sets equal the certified transitions for every stage.

| Static compiler result | Frozen baseline | Isolated candidate |
| --- | ---: | ---: |
| Final graph boundaries | 17 | 12 |
| Incoming surface boundaries | 6 | 2 (closest and volume) |
| Total frame fields / AoS bytes | 92 / 416 | 92 / 416 |
| Payload slots / logical values | 85 / 93 | 85 / 93 |
| Normalized module instructions | 1,115,843 | 1,114,333 |
| Production verifier errors | 0 | 0 |

The first replay checked raw post-split SSA too early. Frozen baseline has 276
and candidate 156 intermediate dominance diagnostics; both have zero after the
production reg2mem boundary. These intermediate diagnostics are not evidence of
a candidate execution regression and were not patched individually.

Some static field masks shrink despite unchanged allocation: background target
live fields 73 -> 18; forward-light 81 -> 74; closest-to-background source stores
9 -> 2; volume-to-background 47 -> 15. These are compiler metadata counts, not
measured bandwidth or renderer speedups. Candidate analysis still reports ROBDD
widening (2,937 states / 44 selected arms), so generic intra-block predicate
retirement is being reduced independently before any follow-up implementation.

Replay evidence is in `../replay/` relative to the capture directory above:
`barbershop-baseline-production.log` and `barbershop-candidate-production.log`.
Their SHA-256 hashes are respectively
`db6479a26091aeeebd5acc585ffe9da0a90d3f89492344a54c00e6f601e9991d` and
`5a8c2698f14e5dfbbb3e23c7220f9dae55081b1ec31005a5201e7c3c914116c4`.

## Independent follow-up: retiring dead intra-block predicates

An independent raw XIR block snapshots n Boolean arguments through private
slots, then inverts and stores each result into a distinct reference-array
element. Every result is observable. With all early snapshot variables ordered
before all late result variables, retaining their dead equalities admits 2^n
distinct cofactors. A known-true gate after the chain exposes the precision loss
when the existing finite ROBDD budget widens. This is a compiler-domain issue,
not a renderer-specific instruction-count threshold.

`test_coro_transition_predicate_retirement` was built with 32 threads against the
unchanged preceding candidate and failed as expected: three cases / 56 assertions,
five failed precision requirements. n=3/8 preserve the constant gate without
widening; n=24 widens and retains a false edge. The dynamic n=24 control retains
both real edges and fails only the no-widening requirement. A genuinely later-used
snapshot relation also loses precision at n=24. All split, verifier and observable
output-store checks pass. Source, binary, logs and six matching libraries are
frozen under the reduction archive's `pre-retirement-eJ6IeP/`, manifest
`RETIREMENT_RED_SHA256SUMS`; RED log SHA-256 is
`bb8863aa2c85f649d5a019c47c6ba079b7c5d755e7f4cceb51d9666a0cfb4706`.

The follow-up adds sparse deaths to the existing Boolean liveness
helper: `DEAD[i] = (USE[i] union DEF[i]) - LIVE_AFTER[i]`, computed before killing
definitions, including never-used definitions. The forward analysis applies
existential projection after each nonterminator transfer. Terminator conditions
remain until outgoing states are refined. The backward fixed point still uses
every raw semantic successor, including loop and resume edges. Projection
distributes over May union; re-executed definitions still replace old instances.
Aliases, writable Extensions, unsupported producers and shared-call matching
retain their existing conservative handling. The budget and transfer functions
are unchanged. This family has a linear live relation frontier after retirement;
no polynomial claim is made for arbitrary genuinely live Boolean relations.

The unchanged retirement source and executable now pass all 3 cases / 56
assertions against the rebuilt compiler library. Every n=24 chain reports no
widening, known/retained-snapshot gates have one edge, and dynamic gates retain
both edges. Direct private-API controls cover overlapping USE=DEF with a live
result, never-used definitions, and liveness through a raw false successor:
3 cases / 14 assertions pass. The original transition suite remains 11 / 161.
All 81 host XIR/coroutine targets were built with 32 threads and pass CTest.

Logs are `retirement-green-focused.log` and `retirement-green-host-ctest.log` in
the reduction archive; full source/library/test hashes are in
`retirement-green.sha256`. The focused log SHA-256 is
`df7062dfc716c5fd7e0eb7643742413e494739052e5e45da26b072ca296c4817`;
full host log is `f23648de2e28a95723b596c9e8c4449c44676855cf9f98b1747ad065c9646412`.

The unchanged 1,111,095-instruction complete module was then replayed sequentially
with the frozen pre-retirement libraries and the post-retirement candidate.
Both terminate 0, preserve exact source serialization, and verify normalized
output with zero errors. Every stage's actual token stores match its certificate.
Retirement removes widening (true -> false), reduces analysis states 2,937 ->
2,882 and increases selected arms 44 -> 99. The surface scope contains 2,614
instead of 2,669 blocks. Normalized instructions decrease 1,114,333 -> 1,114,278.
Both final graphs have six nodes / 12 boundaries, both frames have 92 fields /
416 AoS bytes, and all printed stage/edge field masks are identical. This extra
retirement step therefore has no demonstrated frame-capacity or field-mask gain.

These correctness replays overlapped another validation job. Their distill
durations (2.85892 s before, 2.00719 s after) are overlapping-load diagnostics,
not controlled timing, full-JIT measurements or renderer performance claims.
Logs in the full capture archive's `replay/` directory are
`barbershop-pre-retirement-overlap.log` (SHA-256
`090d28c1a5355dda7a31e6529ea1cea84f7f0c835d1fa03d5ca5c58622d772ed`)
and `barbershop-post-retirement-overlap.log` (SHA-256
`465f5191f8bd2d7b8bbc5e87379fe3eab17a59db4546c3e988d0f7e646fb49ca`).

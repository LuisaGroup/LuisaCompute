# Generated exit dispatches and loop-epoch convergence

## Failure and reduction

At Luisa `8911828eb`, the original upstream Psycles Lone Monk graph-wavefront
continuation 4 fails restructuring on both Metal and Metal4, before backend
shader compilation. It retains one unstructured conditional, with no
irreducible region, invalid construct or exhausted iteration limit.

The 2027-block input was captured before the pass. A topology abstraction
preserving owned-block order, constant predicates and nonempty payload blocks
reproduces the same failure and the same 110 final raw conditionals (109 are
legal loop-boundary forms). Delta reduction produces an eight-block input:

1. Enter a loop through a forwarding header.
2. An indexed branch has a default payload arm, a terminating arm and a
   forwarding arm which also reaches the payload.
3. The payload performs a store and returns to the loop header.
4. Switch-exit normalization generates a conditional selecting termination
   versus a payload-bearing continue.

The permanent test uses only two cases plus default. Both return and
unreachable variants fail in transactional and discardable in-place modes.
The old focused suite reports four failed assertions in the new test. An
earlier direct continue/break reduction already passed and is retained only
as a negative control, not represented as the failing counterexample.

## Control-flow argument

Let H be a conditional in an enclosing loop L, and let E contain L's
continue/update and exit boundaries. A lexical arm path ends on its first
visit to E or a function terminator. A valid normal merge belongs to the
current epoch, not to a path obtained by following L's backedge again.

The existing `infer_selection_merge` searches this bounded relation. For the
counterexample it correctly returns no merge: one arm terminates, while the
other performs its payload and continues. Global post-dominance can still
choose a terminal reached after a later iteration. The fallback's safety walk
detects that it crosses a continue, but previously just skipped the candidate;
the post-pass fixed point then retained the raw conditional forever.

The correction is restricted to a missing lexical merge and a fallback walk
crossing an enclosing continue, verified by the existing source-relative
boundary ownership relation. Foreign loop-role crossings and rejected
explicit lexical candidates retain their conservative rejection. The normal
synthetic-merge construction replaces the raw conditional with an If and
fresh unreachable merge blocks. It neither adds executable arm edges nor
moves, clones or suppresses the payload. Contracting the new unreachable
structural shell leaves every original executable path, side effect,
continue and termination unchanged, including infinite executions.

No verifier condition is weakened, no iteration budget is raised, and no
Psycles program or scheduler is changed.

## Gates

The permanent fixture checks strict structured verification, unique merges,
canonical break/continue targets, XIR-to-AST translation, retained payload
location, bounded block growth and a second-pass fixed point. The new failing
test reports four failures before the correction and passes afterward. The
original 2027-block topology also restructures with zero irreducible,
unstructured, invalid or iteration-limit results.

| Focused host fixture | Tests | Assertions |
| --- | ---: | ---: |
| `test_xir_pass_restructure_cfg` | 90 | 1803 |
| loop scopes | 2 | 409 |
| owned blocks | 2 | 310 |
| entry boundaries | 1 | 5 |
| construct exits | 1 | 7 |
| selection relations | 1 | 206 |
| Total passing | 97 | 2740 |

Both complete original Lone Monk graph modules also compile and dispatch at
1920x1080 / one sample on Apple M1 Max, macOS 26.6.2, Clang 21.1.8, with
Metal4 LLVM 22.1.8. Both use seven stages, 90 frame fields / 456 B, 131072
workers, selective scheduling, inline shadow work and automatic tail. Both
EXRs have exactly 15 passes / 46 channels and zero nonfinite values. Psycles
renderer sources remain identical to upstream `f690ead6`; main shader cache
is disabled. All temporary continuation-dump instrumentation was removed and
the production coroutine library rebuilt before these gates.

These one-sample runs establish original-module compilation and execution,
not 256-sample performance or Cycles compatibility. Metal4's first graph
session initialization took 244.353 s; a brief host sample during that gate
placed the wait in Apple's compute-pipeline compilation. This sampled canary
is not included in the formal timing campaign. Backend cache and initialization
costs remain separate from render-only time. This is focused validation, not
a full Luisa suite pass.

Local evidence is in the Psycles build directory:
`build-macos/benchmarks/2026-09-09/lone-monk-metal-schedulers`.
The original failed v3 graph matrix, both backend diagnostics, complete
before/after IR, topology abstraction, reducer history and permanent red log
are retained separately from performance observations.
Green logs are `cfg-permanent-green.log` and
`test_xir_pass_restructure_cfg*-green.log`; complete scene commands, logs and
pass inventories are under `graph-cfg-canary/{metal,metal4}`.

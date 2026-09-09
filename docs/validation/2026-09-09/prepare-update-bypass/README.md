# Preserve update payload when normalizing prepare backedges

Validated against `origin/next` `2bdbcf5a488574d22df6930f0392f95f7fba33b8`.
The residual fix is generic XIR CFG normalization, with no renderer-specific
conditions, additional SSA spill repair, or floating-point changes.

## Cause and invariant

For a Loop with prepare P and distinct update U, an executable edge B -> P
skips U's payload. Both boundary-branch normalization and the batch continuation
planner classified prepare as a continue target, then redirected such edges to
U. That changes the executed store sequence when U is not transparent:

```text
original:     define -> U(payload) -> P     skip -> P
incorrect:    define -> U(payload) -> P     skip -> U(payload) -> P
factored:     define -> U(payload) -> C -> P     skip -> C -> P
```

C is a payload-free continuation trampoline. The existing update-region
factoring retains entry into executable U while separating iteration completion.
It must run before either normalization family, after prepare subdivision has
established the actual backedge target.

Do not factor every ordinary payloadful update: preserving a unique latch's
update role prevents unrelated selection-entry repair from cloning its payload.
For an otherwise canonical U ending in Branch(P), factoring is required only
if P has an **executable** predecessor other than U or the loop owner's entry.
Forwarding F -> P remains a bypass; declarative block-role uses do not count.
This condition is reevaluated in every post round, including newly recovered
loops and generated exit dispatches.

## Baselines and permanent regressions

The frozen `6e58928d84604bf0d976f0d6b00f44006b21ceb5` SDK rejected an actual
native-Vulkan zero-BSDF module with 77 non-dominating stores. Checkpoints first
located the violation in `normalize_loop_boundary_conditional_branches`:
dispatch -> prepare was changed to dispatch -> payloadful update.

After fetching, **unmodified latest-next already accepts that complete module**;
its newer ownership fixes take a different structural route. It is not correct
to claim this residual patch was needed to make that original module verify.
However, both minimal counterexamples still fail on unmodified latest-next:

- Six blocks: a load in the non-skip arm feeds a store in U. Output verification
  fails when skip is redirected into U. The new `update_bypass` unit covers both
  mutation modes, three allocation orders, and uint/float3 payloads.
- Seven blocks: an additional exit generates a state dispatch. Earlier reg2mem
  can conceal the dominance violation, but the skip arm now reads stale or
  uninitialized spill state and executes an extra output store. The existing
  scalar **compiler-test** interpreter compares complete effect sequences;
  selectors 0/1/2 must respectively execute U, skip U, or break. Coverage includes
  raw/structured prepare and Branch/Continue entry into U. This is not a CPU
  shader or reference renderer.
- The existing no-cloning control remains and now covers both mutation modes,
  requiring an ordinary unique latch to retain its update role.

The permanent additions were built and run red before changing the pass.
An initial unconditional factoring candidate failed the existing no-cloning
test; it was replaced by the exact executable-bypass condition above, not by
weakening the test.

## Validation and reproduction

The isolated SDK was rebuilt with its own compatible core/AST/XIR libraries and
all 32 threads. Dependency sources were reused only after checking clean status
and exact pinned gitlink identities. Temporary dependency symlinks are not part
of this patch.

```sh
cmake --build build --parallel 32 --target $(
  ctest --test-dir build --show-only=json-v1 -L unit_xir | jq -r '.tests[].name'
)
ctest --test-dir build -L unit_xir --output-on-failure --parallel 32
```

Result: **71/71 host XIR tests passed**, including 72 assertions in the new
six-block unit, the expanded effect-sequence suite, and the two-mode no-cloning
control. Complete original-module replay also passes with zero invalid,
irreducible, unstructured, iteration-limit, or verifier errors.

Retained evidence directory:
`/var/tmp/psycles-holdout-CXlJR7/restructure-capture-Dd85c6`.
Full build/test logs are `candidate-2-full-build.log` and
`candidate-2-full-ctest.log`; original-module output is
`candidate-2-original.result.txt`. Unmodified latest libraries are archived in
`latest-unmodified-libs/`, with a SHA256SUMS file. The original valid interchange
is `restructure-0.input.xir`, not an extracted sibling kernel.

Exact host replay used the independently compiled tool linked to the isolated
latest SDK (library resolution checked with `ldd`):

```sh
/var/tmp/psycles-holdout-CXlJR7/latest-replay/build/latest_replay \
  /var/tmp/psycles-holdout-CXlJR7/restructure-capture-Dd85c6/restructure-0.input.xir \
  /var/tmp/psycles-holdout-CXlJR7/restructure-capture-Dd85c6/NEW_REPLAY_PREFIX
```

SHA-256 proof identities:

| Artifact | SHA-256 |
| --- | --- |
| Frozen original valid interchange | `7a443750f9c8b6377ead4a89236613dcbb884a4928fe9d87f4dc0a9cac7c4001` |
| Unmodified latest `libluisa-xir.so` | `5aee3366c7e29460c70dd2bcce08bfdde84ade1ecb518305ca018f4b2c153224` |
| Validated candidate `libluisa-xir.so` | `04c73103e58c915b668265b90c21415681f4f3d0694de2ac8942eb9c3a6fa621` |
| Candidate `restructure_cfg.cpp` | `16342b23c8bc12f789302920fb8018e355d76dd79f404d7ef4b543699ef0aff4` |

Unmodified latest and the candidate both produce 6,833 debug block lines and
153,909 instruction lines for the complete capture. This shows no aggregate
IR growth on this input; it is not a renderer performance measurement.

Actual renderer/HIP/fallback/strict native Vulkan validation after compatible
integration remains pending. No backend success for this patch is inferred
from host XIR verification.

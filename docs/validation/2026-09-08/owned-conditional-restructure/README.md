# Reconstruct disconnected owned conditional blocks

## Failure domain and proof

Let V be the blocks owned by a FunctionDefinition, R its entry-rooted
structural traversal (including declared construct boundaries), and D its
executable dominator-tree domain. In general D is a subset of R, which is a
subset of V. CFG canonicalization can disconnect an old block without
releasing it. That block remains part of the public XIR module contract.

The verifier and final residual-branch check quantify over V, but the
remaining-divergent conditional index previously covered only R. Thus a raw
conditional in V minus R could never enter the rewrite worklist and would
always fail the final check. More fixed-point iterations cannot repair this
domain mismatch. Indexed-branch reconstruction already covers V.

The reduced example has four blocks and one boolean argument:

```text
entry: return
dead_header: branch condition, dead_true, dead_false
dead_true: return
dead_false: return
```

Both transactional and in-place restructuring reject it before the repair,
reporting one remaining conditional and no invalid or irreducible construct.
No renderer, GPU, floating-point computation or coroutine ABI is required.

## Minimal repair

Index every owned block's structured roles and conditional candidates. Keep
the old traversal order for R, then append V minus R in ownership order, so
live-CFG rewrite ordering is unchanged. The index and its fresh-analysis
oracle now use the same domain as the final contract.

A header outside D has no executable lexical dominance context. As in
indexed-branch reconstruction, it receives a fresh synthetic unreachable
merge instead of adopting a global post-dominator that might belong to live
code. Its condition, payload and executable branch targets are unchanged.
No block is deleted and no new entry-reachable edge is introduced.

Each rewrite still consumes exactly one original raw conditional and adds
only an If plus branch/unreachable blocks. No new raw conditional or loop
role is created, so the existing finite drain and exact dominance/post-
dominance overlay contracts remain applicable. Structural validation is not
weakened, and no application-specific compiler setting is introduced.

## Permanent tests

`test_xir_pass_restructure_cfg_owned_blocks` is registered with CMake and
xmake. It checks both public mutation modes, the four-block red case,
idempotence, retained block/instruction identities and strict output
verification. A second fixture has seventeen disconnected selections whose
nontrivial store arms converge on live entry code. It checks that no live
merge is claimed, no store is dropped and no executable edge is retargeted.
Both fixtures enable the fresh remaining-divergent index/post-dominator
oracle; 310 assertions pass, also with intermediate verification enabled.

```sh
cmake --build build-codex-xir --parallel 32
ctest --test-dir build-codex-xir --parallel 32 --output-on-failure \
  -L '^(unit_xir|unit_coro)$'
env LUISA_XIR_VERIFY_INTERMEDIATE=1 \
  ./build-codex-xir/bin/test_xir_pass_restructure_cfg_owned_blocks
```

The all-target build succeeds and the XIR/coroutine host selection is 70/70.
Do not use an unanchored `unit_coro` label as a host-only selection: it also
matches the separately labeled `unit_coro_runtime` device test.

An exploratory all-unit run was 149/151. The remaining assertions are in
`test_spirv_xir_dialect` (old special-register enum endpoint) and
`test_spirv_target_feature_codegen` (a required optimized instruction
spelling), not the restructuring tests. A separate test-only correction
extends both enum scans through the appended raster depth/builtin tags. The
constant fixture's actual SPIR-V retains all four signed 16-bit literals and
selects between them, so it now checks those exact payloads plus the existing
positive/negative storage-feature requirements instead of insisting on
OpConstantComposite. No code-generation or numerical policy changes.
The subsequent all-unit run passes 151/151 (`luisa-unit-all-v2.log`);
`spirv-constant-red-dump.log` retains the original emitted selection chain.

## Original complete application

The triggering Psycles change restores the original Cycles post-lamp
INTERSECT_CLOSEST stage, including a fresh BVH traversal and the last lamp's
ray-self identity. Its small production-pipeline HIP regression initially
passed, but full Barbershop failed during continuation 5 restructuring:
one residual conditional at owned block 242, with targets 263 and 264.
All three blocks are absent from the final entry-rooted structural dump.

After this generic repair, the unchanged full original application compiles
and renders Barbershop at 2048x858, 64 spp, with the index oracle enabled.
Its six-stage frame remains 93 fields / 416 bytes. Actual closest visits are
444,263,321 versus original Cycles 444,271,208. This is a correctness and
whole-module compilation gate, not a renderer-speedup claim. The same small
production routing regression also passes HIP and fallback with SoA/AoS,
capacity refills and repeated dispatches.

Local evidence is `/var/tmp/psycles-lamp-routing-sCLxKs`:
`psycles.log`, `restructure-trace.log`, `restructure-dump-v2.log`,
`xir-owned-red.log`, `xir-owned-expanded-green.log`,
`xir-coro-all-v2.log`, `luisa-unit-all.log`, `xir-full-build.log`,
`psycles-fixed.log`, and `final-focused-{hip,fallback}.log`.
Large-function dump instrumentation used to diagnose the failure has been
removed. No diagnostic switch or application workaround is part of the fix.

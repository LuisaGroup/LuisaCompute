# Shared switch bodies across representation and CFG lowering

Several labels can now name one owned AST case body, matching the existing
XIR relation `add_case(label, block)`. `$case(a, b, c)` and the case builder's
host integral span overload record that body once. The runtime still selects
exactly one body; no classifier, callable, or state copy is introduced.

## Representation invariant and first counterexample

Let a switch map distinct literals to target blocks. Labels with the same
target are parallel incoming edges to one case construct, not separate copies
of its effects. The previous XIR-to-AST translation copied the body once per
label; AST-to-XIR then created separate blocks. The permanent three-label,
one-buffer-write roundtrip regression failed three assertions before repair.

`SwitchCaseStmt` retains its single-label API and stores additional labels
only for a group. A new statement tag is appended, preserving existing tag
values, single-label hashes and the old binary record. Multi-label records
have an explicit count. JSON groups use `values`; the existing JSON int32
literal format is not silently widened. Binary/XIR paths cover wide labels.
Duplication, serialization, traversal, all AST code generators and both XIR
translations preserve one body. XIR-to-AST grouping uses first-occurrence
order and an O(N) target-to-group map, never hash iteration order. A shared
default/case body is still emitted separately by XIR-to-AST; this change does
not claim to remove that additional duplication.

Single-label byte-idempotence across serialization was not assumed: that
property also failed for the old representation. The controls instead verify
decoded labels/body, recomputed duplicate hash, repeated loading, and equal
hash/binary output for the old scalar builder versus a one-element span at
the same source location.

## Shared targets are not distinct construct entries

The first native Vulkan regression exposed a separate defect. A five-block,
ten-instruction switch with three labels sharing one write body was expanded
by `split_switch_cases`: distinct label proxies entered the shared body, so
neither proxy dominated that payload. Exit repair then misclassified the new
cross-case entries, introduced an artificial cycle and left an unstructured
conditional. The failed module had grown to 21 blocks / 36 instructions.

The [OpSwitch specification](https://registry.khronos.org/SPIR-V/specs/unified1/SPIRV.html#OpSwitch)
requires distinct literal values, not distinct target labels. The
[maximal-reconvergence extension](https://github.khronos.org/SPIRV-Registry/extensions/KHR/SPV_KHR_maximal_reconvergence.html)
also explicitly discusses multiple labels for one case construct. Removing
the unconditional target-uniqueness normalization preserves that relation;
genuine cross-construct entry enforcement remains enabled. There is no
case-specific exception in `fixup_construct_exits` or validator bypass.

An independent test builds this CFG directly with the existing XIR API.
Its 48 shapes cover signed/unsigned 32-/64-bit selectors, raw/structured
input, aliased/separate default, and one/two/three shared labels. Before the
repair, 40 assertions failed; afterwards, all 552 pass. The tests require
valid output, shared target identity, preserved observable writes and no
introduced loop. Two old tests were updated because they explicitly required
the now-removed proxy blocks; their effect, target and verifier checks remain.
The broader outstanding proof obligations in the
[loop-epoch audit](../../2026-09-08/loop-scope-restructure/README.md)
are not claimed complete by this repair.

## Narrow signed physical literals

After the CFG fix, signed/unsigned 64-bit native kernels passed, but signed
16-bit cases failed SPIR-V validation. XIR's canonical `0x8000` / `0xffff`
must be encoded in a physical 32-bit literal word as `0xffff8000` /
`0xffffffff`. The old single-label spelling reproduced the same failure.
The generic correction sign-extends signed 8-/16-bit literals at physical
OpSwitch emission, after canonical-bit validation. Unsigned and 32-/64-bit
encoding, XIR semantics and device arithmetic are unchanged.

## Permanent and whole-application validation

- `test_xir2ast_translators`: shared-body roundtrip red/green.
- `test_switch_case_group`: variadic/span DSL, filtered label sets, duplicate,
  JSON, binary, XIR and single-label compatibility controls.
- `test_xir_restructure_shared_switch`: all 48 CFG shapes / 552 assertions.
- `test_switch_case_group_runtime`: 211 assertions on HIP, fallback and strict
  native Vulkan. Covers separate and grouped signed/unsigned 8-/16-/64-bit
  labels, default, loop break/continue, early return, and StateMachine plus
  Wavefront coroutine suspension. No failing type is excluded.
- Full 32-thread build and 133/133 CTests selected by `-L '^unit'` in
  `build-tests-hip`; the narrower exact `unit` label selects 132/132.

The Vulkan run sets `LUISA_VULKAN_USE_XIR=1`,
`LUISA_VULKAN_REQUIRE_NATIVE_XIR_SPIRV=1` and `LUISA_VULKAN_DISABLE_DXC=1`.
Loader diagnostics show no DXC/DXIL library load. The HIP-only child build
uses its own ABI-matched runtime; fallback/Vulkan use a separate executable
linked against the application's SYSTEM_STL=ON libraries, not mixed plugins.

The full unchanged Barbershop input also completes A/control/B/B/A renders
at 2048x858 / 64 spp with all 46 channels finite. Grouping the application's
original-Cycles BSDF labels reduces its main function from 184,759 to 159,801
bounded machine instructions, without changing the four-function set,
49 call sites, 416-byte coroutine frame or 256 VGPRs. Surface medians are
5.860 / 5.809 seconds: only a small measured change, not efficiency parity.
Restoring only the old application case spelling restores byte-identical
baseline `.text`, with the generic compiler fixes still present.

Local evidence: `/var/tmp/psycles-shared-case-Btjjlq`. Primary records are
`formal-analysis.md`, `roundtrip-red.log`, `cfg-shared-target-analysis.md`,
`cfg-red.log`, `cfg-green.log`, `spirv-literal-analysis.md`,
`runtime-vk-narrow-red.log`, `runtime-hip-final.log`,
`runtime-fallback-final.log`, `runtime-vk-green.log`,
`child-full-build-final.log`, `child-unit-complete.log`,
`child-unit-complete-expanded.log` and `profiles.json`.
The application owns the complete image/timing report. These primitive tests
and structural controls do not implement a CPU shader oracle.

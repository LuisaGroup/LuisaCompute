# Local storage epochs and coroutine frame reconstruction

This integrates the unpublished Local/coroutine fixes onto `3485bd89d`, not
onto the older dirty checkout. Already-published scheduler, HIP ABI, late-inline,
and SPIR-V changes are not reverted or duplicated. Evidence is retained in
`/var/tmp/luisa-publish-coro-bQIjua`.

## Storage contract

Ordinary DSL `Var<T>` values, including scalars, vectors and fixed `Array`,
retain their default-zero semantics. `Local<T>` is uninitialized storage.
Its physical allocation is function-scoped, but every dynamic execution of
its lexical declaration starts a fresh storage epoch. AST records that
boundary as an exact local-reference assignment from `UNDEFINED`.

`UNDEFINED` is not a zero fill and is not evidence of a program definition.
Direct AST code generators omit this exact whole-local assignment; partial
assignments, nonlocal writes and ordinary `ZERO` expressions are unaffected.
Coroutine scope analysis consumes the epoch before removing the seed. Retaining
the old physical bits is a valid refinement of an arbitrary value, not a
promise that those bits may be read as initialized state.

`CoroFrame::create` no longer clears all fields. Entry/resume protocols own the
fields live in that state. `CoroSlotAccess::_read` reconstructs the complete
static leaf partition of a binding and therefore does not need a preceding
aggregate clear. Radix-sort local keys and ranks are explicitly assigned for
every item before any read, so their scratch arrays use `Local` as well.

## Generic scope proof

For payload P, same-sized unsigned tag array T and counter C, the proof tracks
the possible tags of undefined records in `[0,C)`, a separate tail record,
and the pending physical record at C. A read needs a valid index and a tag
constraint disjoint from every applicable unsafe set. Publication, saved
allocation tickets, conditional writes, reset, and nonwrapping rollback are
ordinary XIR state transitions. Rollback removes semantic membership but
does not erase an already-written physical record.

Unsafe sets use May union at CFG joins; definite index/definition facts use
Must intersection. Boolean assignments and edge constraints act on a common
finite valuation domain. Unknown aliases, unrecognized counter mutations,
stale discriminator observations and insufficient range/definition evidence
cannot justify contraction. No renderer name, scene profile, noinline marker,
or application-supplied lifetime assertion enters this proof.

Earlier alloca moves may relocate stores in a companion array/scalar. Each
discriminated candidate therefore snapshots the current instruction order,
not a stale pass-entry order. The existing scalar snapshot query uses a cached
two-state Clean/Dirty dataflow solution. Backward relation liveness uses a
worklist for the same finite monotone equations as the previous full scans.

## Two defects caught during integration

The initial candidate did not pass its negative rollback regression:
`vector<uint64_t>{word_count, 0}` selects the initializer-list constructor.
It creates two elements and treats bits of `word_count` as initialized payload
and tag indices. The intended Must lattice bottom is an N-word zero bitset;
the fix uses `vector(count, value)` construction. The permanent static-index
witness checks capacities on both sides of word boundaries and also requires
that the discriminated domain was actually exercised.

After correcting that error, the positive rollback witness exposed premature
Boolean projection. A branch can be dead as a direct SSA use while still
relating a stored flag to a conditional payload definition. In general,
`exists b (F(b) and G(b))` is not equivalent to projecting b independently
from F and G. This product retains the shared valuation rather than applying
direct-use liveness to each component separately. Dynamic definitions still
kill the previous Boolean value; unsafe reads remain rejected.

`prefix-candidate.log` records the original false acceptance.
`prefix-minimal-green.log` records the positive witness failing after only
the bitset correction. `prefix-product-green.log` records both directions
passing after preserving the relation. Counterfactual and final logs are
separate; no counterfactual implementation is published.

## Callable boundary contract

A recursively read-only reference can be replaced by a call-site snapshot
only when every actual is thread-local and is provably disjoint from every
writable reference actual. Distinct local-allocation roots suffice; a shared
root, unknown pointer, external signature constraint or root callable does
not. Writable arguments remain references. Promotion runs before the ordinary
callable/coroutine pass-domain boundary, preserving rematerializable SSA
captures without requesting inlining or noinline. XIR-to-AST also preserves
function names through this round trip.

The unmodified upstream implementation is built separately under
`build-baseline`. New tests use only test-local compatibility inspection for
the missing AST predicate and diagnostic counter; production baseline code
is unchanged. Observable allocation/frame assertions remain unchanged.
The root-callable regression guards subsequent pointer inspection after a
failed identity assertion so old code reports failures instead of crashing.

## Validation scope

The independent SDK enables HIP, fallback and native XIR-to-SPIR-V Vulkan,
with DXC compatibility disabled at configure time. All builds use 32 threads.
The whole Psycles validation snapshot includes its current tracked changes
and the two required microfacet source files, and uses only this SDK's headers,
generated configuration and rebuilt libraries. It excludes profiling dumps
and the unfinished volume-oracle work.

The full initial CTest run also exposed three unrelated failures: the HLSL
validation executable requires DXC despite this native-only configuration;
`test_spirv_xir_dialect` and `test_spirv_target_feature_codegen` reproduce with
the unmodified baseline AST/XIR/core libraries and unchanged SPIR-V code.
Those failures are not counted as successful validation or silently fixed by
changing shader semantics.

## Final verification

The final prefix regression passes 26 cases / 361 assertions. Its static-index
witness covers capacities 1, 4, 64, 65, 128 and 255, with a separate publication
block so constant forwarding cannot bypass candidate discovery. Reintroducing
only the erroneous bitset constructors produces 14 assertion failures in two
cases (`prefix-final-counterfactual-red.log`); restoring the fix passes
(`prefix-publication-green.log`). No counterfactual code remains in the build.

| Validation | Result |
| --- | --- |
| SDK focused host CTest | 151/151; excludes the three failures above and two render tutorials |
| SDK HIP runtime executables | 7/7 |
| SDK fallback runtime executables | 7/7 |
| SDK strict native Vulkan runtime executables | 6/6 |
| Whole Psycles host CTest | 120/120; source-size and external Blender tests excluded |
| Whole Psycles HIP CTest | 168/168, serial GPU execution |
| Whole Psycles fallback CTest | 170/170 |
| Whole Psycles strict native Vulkan focused CTest | 12/12 |

The seven SDK runtime executables are `test_coro_frame_runtime`,
`test_coro_all_schedulers`, `test_coro_radix_sort`, `test_counted_local_array`,
`test_coro_wavefront_auxiliary_pipeline`,
`test_coro_wavefront_resume_annotation`, and
`test_coro_wavefront_auxiliary_admission`. Vulkan runs the same set except
radix sort. The frame executable includes the new ordinary scalar/vector/Array
default-zero regression (306 assertions in 22 cases on each backend).

The first parallel HIP CTest run was interrupted after 133 reported passes
while 32 test processes remained active without further output. The subsequent
serial run passed all 168. The interrupted run is not counted as a pass, and
its cause has not been established.

On this AMD-only runtime, configure also uses
`LUISA_COMPUTE_ENABLE_VK_CUDA_INTEROP=OFF`: that optional feature is independent
of the disabled CUDA backend and otherwise links the toolkit driver stub,
leaving an unsatisfied `libcuda.so.1` runtime dependency. No source change or
fake driver is needed. Native Vulkan is configured with
`LUISA_COMPUTE_ENABLE_VK_XIR_SPIRV=ON`,
`LUISA_COMPUTE_ENABLE_VK_AST_LLVM_SPIRV=OFF`, and
`LUISA_COMPUTE_VULKAN_ENABLE_DXC_COMPATIBILITY=OFF`. Runtime gates are:

```sh
LUISA_VULKAN_USE_XIR=1
LUISA_VULKAN_REQUIRE_NATIVE_XIR_SPIRV=1
LUISA_VULKAN_DISABLE_DXC=1
LD_DEBUG=libs
```

Both SDK and whole-renderer canaries compile native SPIR-V. Loader logs contain
no DXC/DXIL loading attempts; the rebuilt Vulkan library has no CUDA driver
dependency. The map-range whole-scene canary passes at 16x16 / 4 spp on fallback
and strict native Vulkan, with all 46 output channels finite.

The rebuilt SDK also completes uncached HIP staged-wavefront canaries:

| Scene | Resolution / samples | Render time | Coroutine frame |
| --- | --- | --- | --- |
| Lone Monk | 1440x1080 / 256 spp | 13.7828 s | 55 fields / 220 bytes |
| Monster | 1080x1080 / 256 spp | 14.9817 s | 71 fields / 284 bytes |

Both have 46 finite output channels. These are single canary measurements,
not a paired performance benchmark. Against the recorded Cycles 5.2.1 HIP
reference, Lone Monk Combined relative RMSE is 0.0124035 and DiffInd relative
RMSE remains 0.128819. The remaining renderer-parity gap is not claimed fixed.
Detailed evidence is in `whole-psycles-*.log`, `final-runtime-*.log`,
`final-native-vk-*.log`, `map-native-vk.log`, `monk-hip.log`,
`monster-hip.log`, and `monk-comparison.json` under the evidence directory above.

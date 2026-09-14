# Fallback barrier-coroutine allocation

Final validation: 2026-09-08.

## Contract and failure

The fallback backend lowers a kernel containing block barriers into LLVM
coroutines, one per simulated GPU lane. The launcher can keep every lane's
frame alive simultaneously. A frame's size comes from LLVM `coro.size`, not
from an application profile or an estimate of Luisa's application-level
coroutine frame.

The previous thread-local bump allocator asserted that the sum of aligned
frame allocations for a block fit a fixed 4 MiB buffer. Neither the DSL
contract nor LLVM's coroutine ABI imposes this aggregate bound. For example,
64 lanes retaining 18,432 uints each need at least 4.5 MiB before coroutine
bookkeeping. This valid kernel aborts at `Coroutine buffer overflow`; a
64-lane/256-uint negative control passes.

The original application failure was Psycles' persistent scheduler after
native SVM became the default. Its printed 264-byte application coroutine
frame is unrelated to the larger LLVM barrier frame holding the persistent
kernel's live CPU temporaries. The defect is in the backend allocator, not
permission to shrink SVM storage or to change inlining decisions.

## Minimal repair and invariants

Keep the existing 4 MiB allocation-free buffer as the common-case path, but
remove its role as a correctness limit. Requests that do not fit use cached
overflow chunks with capacity `max(compiler-provided request, 4 MiB)`.
Multiple lane frames share each chunk. This is a reuse/growth policy, not
a frame-size bound, and does not require per-lane/per-dispatch heap allocation.

- Returned spans are aligned to the same `2 * sizeof(intptr_t)` ABI boundary.
  Aligned headers keep overflow payloads aligned as well.
- Frames never overlap or move during a block. A linked list grows without
  relocating prior frames or freeing bookkeeping storage; this preserves
  the codegen helper's `NoFree` and no-alias contracts.
- The existing launch boundary resets allocation cursors only after the
  preceding block has completed. It does not clear frame memory. Individual
  coroutine frees remain no-ops, as before.
- Overflow allocations are reused for subsequent blocks and are released
  when the owning CPU worker exits. Ordinary scalar/vector DSL zero
  initialization and Local lifetime semantics are untouched.
- Integer overflow and allocation failure are checked before exposing a
  payload. There is no larger hard-coded maximum or scene-specific override.

No AST/XIR transformation, HIP/Vulkan code generation, function-inlining
policy or application coroutine stage is changed by this allocator repair.

## Permanent regression

`test_fallback_coro_allocation` executes real fallback shaders with live
Local arrays crossing two block barriers. It checks all lanes' integer sums
and selected values for three blocks, three repeated dispatches with changing
uniforms, 64/128-lane block widths, 36/72 KiB live arrays, and small-array
controls before/after the overflow cases. Expected values are the input
integer expression, independent of scheduling and frame storage.

The reduced original overflow fails before the allocator repair (8.47 s).
The runtime regression passes after the repair (17.80 s), with cold shader
compilation and fast math enabled. A separate 0.04 s host regression invokes
the production arena directly: 4.5 MiB of live frames creates exactly one
overflow chunk, repeated epochs preserve addresses and heap-allocation count,
and individual frames larger than 4 MiB preserve all live data and alignment.
It covers zero-sized/unaligned requests and changes in block width/frame size.
The larger single-frame allocation does not need a huge shader to test its
host-side allocation contract. Build command uses all 32 threads:

```sh
cmake --build build-fallback-sampler --parallel 32 --target test_fallback_coro_allocation test_fallback_coro_arena
ctest --test-dir build-fallback-sampler --output-on-failure -R '^test_fallback_coro_(arena|allocation)$'
```

Local evidence: `/var/tmp/psycles-native-volume-svm-06XnDX`,
`fallback-arena-red.log`, `fallback-arena-reuse-build.log`,
`fallback-arena-reuse.log`. Existing shared-memory/barrier/atomic queue tests
(six assertions) and fallback cold/cache/boolean/minimal-codegen spill tests (11 assertions)
also pass: `fallback-arena-shared.log`, `fallback-arena-cache.log`.

## Original application verification

Psycles `9404ce27` is rebuilt with all 32 threads, retaining its full native
SVM renderer and persistent-scheduler case. The latter no longer aborts on
the arena size. Complete fallback selection is 172/173 in 190.91 s: the
sample-dispatch test subsequently reports a wavefront-versus-megakernel
`light_ng.z` trace discrepancy (`-0.62323` versus `-0.623204`, slot 30,
component 2). It is not a green complete suite. No tolerance change, float
emulation or inlining adjustment is introduced to hide that remaining result.

Focused HIP is 4/4 and strict native-XIR-to-SPIR-V Vulkan is 4/4 after the
allocator repair. Logs use the `fallback-arena-final-integration-` prefix.
The original whole HIP 171/171 run precedes this fallback-only allocator
change; no HIP code is modified. This is an allocation-correctness repair,
not a renderer-performance claim.

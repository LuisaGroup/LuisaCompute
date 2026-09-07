# Materializable auxiliary storage and admission

## Cause and proof

The existing side-pool protocol proves occupancy safety with `q <= C` and
`n * b <= C - q`: a batch of `n` main invocations may emit at most `b` items
each. That contract requires `prepare_for_producer` to materialize *all*
abstract free capacity on demand. It cannot express an append allocator which
deliberately postpones relocation of dead holes. Treating such holes as already
available either forces a different client policy or overflows its append tail.

The new storage-only pre-admission hook preserves every item, stage count, and
capacity. Afterwards the client reports materializable slots `A`, constrained
by `0 <= A <= C - q`. Admission additionally requires `n * b <= A`, using the
existing exact 64-bit product of 32-bit factors. If blocked, the client's usual
non-empty admission stage runs. An empty pool must materialize its full capacity
to retain the existing bounded-producer progress guarantee. An assertion checks
host occupancy, capacity, and the availability bound after preparation.

Both preparation hooks enqueue storage operations on the same stream as the
producer, never semantic stages. They may not change observed populations.
Default preparation is a no-op and default availability is `C - q`, so existing
clients retain their previous behavior. The final `prepare_for_producer` hook
still guarantees admitted storage before the actual continuation resumes.

No allocator threshold, rendering state, device-kernel ID, or application
policy is encoded in the scheduler. Physical queues with pre-resume annotations
continue to use their logical target's producer bound.

## Smallest counterexample

Two main frames publish three times into four auxiliary append slots. Each
consumer releases two items; the client reclaims its append extent only when
empty. After two publications and one drain, `q=2`, `extent=4`, `C=4`: abstract
free capacity is two, but append availability is zero.

The old scheduler forces an unrequested reclaim and drains populations
`[4, 4, 2]`. The new contract first drains `[4, 2]`, reclaims the empty pool,
admits the last publication, and then drains `[2]`. All six publications and
both final main results remain correct in the old witness. It records any
premature preparation and reclaims defensively before an invalid memory access;
it does not depend on corruption or undefined behavior to fail.

`test_coro_wavefront_auxiliary_admission` covers plain/annotated producers,
SoA/AoS, legacy/incremental/fused accounting, and scheduler reuse. The original
header, with the test's future hooks non-overriding, fails 72 of 144 assertions
on HIP; the new protocol passes all 144 on HIP and fallback. This is a scheduler
protocol extension, not a compiler transformation or initialization change.

Evidence directory: `/var/tmp/psycles-wavefront-admission-6YMdzL`.
`analysis.md`, `red.cpp`, `red-build.log`, `red-hip.log`, `green-build.log`,
`green-hip.log`, and `green-fallback.log` retain the reasoning and witness.

The application's original shadow-pool module also fails its newly added
policy regression before integration (`psycles-red-hip.log`), then passes HIP
and fallback after integration. Its independent oracle invokes verbatim
external Cycles host functions; the actual pool regression checks live NEE
payload and shade-owned batch relocation, delayed reclamation, film values,
and all per-stage counts. The generic SDK test has no renderer implementation.

## Whole-module validation

All builds use 32 threads. `full-sdk-build.log` builds all 14 affected coroutine
executables from the clean candidate headers and test sources. All pass HIP
and fallback: 157 cases per backend, 13607/13606 assertions respectively.
`full-build.log` rebuilds the original Psycles application and tests (207
steps). The HIP selection passes 157/157, fallback 159/159, and compiler/adapter
selection 101/101. The latter excludes the already-known
`blender_export_render_settings` failure; it is not a full-repository green claim.

`native-sdk.log` and `native-psycles.log` run the independent witness and actual
shadow pool with `LUISA_VULKAN_USE_XIR=1`,
`LUISA_VULKAN_REQUIRE_NATIVE_XIR_SPIRV=1`, `LUISA_VULKAN_DISABLE_DXC=1`, and
`LD_DEBUG=libs`. Both pass, with 46 and 36 native SPIR-V compilations and no
DXC/DXIL library loads. These tests link the active compiler/backend libraries;
they do not claim a clean rebuild of unrelated pending Local/scope/XIR changes.

The application A/B/B at 1440x1080, 256 samples is neutral: old admission
14.3138 seconds, new admission 14.2950 and 14.3686 seconds. No device-kernel
instruction reduction or end-to-end speedup is claimed for this host protocol.

# Exact queue handoff for pre-resume annotations

## Cause and contract

An ordinary Extension is an independently scheduled state transition. Its
handler owns a selected index queue, but only frame writes survive transition
to the next queue. Gathering that next queue again does not preserve a
permutation applied by the previous handler. This is not a failure of the
existing frame-writeback contract: that contract promises no resume order.

A sorting annotation needs a different, generic scheduling boundary: an
ordered suffix of handlers and its target continuation must consume the same
selected index queue without a gather, refill, or relocation in between.
Each handler returns an exact permutation (same cardinality and membership),
and the next handler or actual continuation receives that view. No scheduler
knowledge of the annotation schema or of a rendering application is necessary.

The suffix remains attached to its static suspend boundary. Different incoming
boundaries may bind the same logical annotation to different colored frame
slots; their binding plans must never be conflated. An independently scheduled
operation cannot follow the pre-resume suffix. Skippable annotations retain
their existing fallback behavior; ordinary semantic Extensions retain their
existing frame-writeback and scheduling behavior.

The selected queue identity and the continuation identity need not coincide.
Queue accounting consumes the former exactly once, while producer capacity,
refill eligibility, tie priority, execution block size, and continuation statistics refer to
the latter. The continuation consumes the already-resident target-live fields;
annotation-only bindings remain resident only as required by boundary liveness.

## Smallest witness

`test_coro_wavefront_resume_annotation` uses 16 keys in reverse order. A typed
binding reader scatters the selected frame indices into their unique key slots.
The handler verifies its sorted queue, while the continuation records physical
`thread_x()` separately from logical `dispatch_x()`. Logical results remain
correct on the old scheduler, but the continuation does not see the sorted
queue. There is no CPU coroutine implementation or renderer in the witness.

The multi-boundary case also distinguishes the documented binding semantics:
`read` snapshots an rvalue at the suspend boundary; `read_write` binds a
writable lvalue. A preceding +5 write therefore changes the continuation's
state but not the sort annotation's snapshot. The original test incorrectly
compared them as equal (observed state/snapshot pairs 6/1, 12/7, 18/13).
Correcting that assertion follows `docs/source/coro_suspend_extensions.md`,
not a compiler workaround or an alteration of the queue order expectation.

The equal-population witness creates two eight-frame continuations. Moving
the lower-priority-number continuation behind a physical annotation queue
must not let the other continuation win a tie. Before projecting tie priority
through the logical continuation owner, all 16 order assertions fail while
frame values and queue counts stay valid. Queue cardinality still dominates
priority; equal logical owners retain deterministic physical queue order.

## Descriptor ownership

Whole-program validation exposed a pre-existing Extension scheduler lifetime
gap: `WavefrontCoroScheduler` borrowed its source `Coroutine` and its boundary,
binding, stage, and frame descriptors. Handlers are prepared after construction
and finalized/executed after the factory's local source has been destroyed.
This is a host use-after-free, independent of node contents, queue policy, or
backend. The fallback scene test observed a corrupted boundary descriptor.

The smallest lifetime regression returns a scheduler compiled from a local
source, then registers and executes a typed permutation handler after source
destruction. The scheduler must own an immutable copy of the descriptors it
publishes to handlers. This copies compilation metadata, not coroutine state
storage; there is still exactly one physical frame allocation per scheduler.

## Validation

Evidence directory: `/var/tmp/psycles-coro-resume-8fNIXm`.

- `red-hip.log`: 64 order failures on the original independently scheduled
  handler, while its own permutation and all logical values remain correct.
- `priority-red-hip.log`: 16 tie-order failures before logical-owner priority.
- `lifetime-red-hip.log`: destroyed-source descriptor failure (including a
  corrupted schema name), before scheduler-owned metadata.
- `clean-sdk-build.log`: all 13 affected scheduler test executables built
  with 32 threads using the isolated candidate's headers and source.
- `clean-hip-*.log` and `clean-fallback-*.log`: all 13 executables pass on
  both backends, 156 test cases per backend. HIP: 13,463 assertions; fallback:
  13,462 (one capability-dependent assertion). Queue verification is enabled.
- The four new resume-annotation tests contribute 4,987 assertions, including
  returned scratch views, several incoming boundaries, a self edge, semantic
  writes plus snapshot reads, ignored annotations, compaction, fixed slots,
  token sorting, all-queue and greedy execution, both incremental count modes,
  scheduler reuse, tie priority, and source destruction.
- The shared-capacity pipeline contributes 6,816 assertions, with and without
  producer annotations, on SoA/AoS and all three accounting modes.
- `clean-native-vulkan.log`: the four new tests pass with all three native
  XIR/SPIR-V guards and loader tracing; 94 SPIR-V compilations, no DXC/DXIL
  library loads.
- Whole original Psycles build, 32 threads: `whole-build.log`. HIP 157/157
  (`whole-hip.log`), fallback 159/159 (`whole-fallback.log`), compiler/adapter
  100/100 (`whole-core.log`, excluding the pre-existing
  `blender_export_render_settings` test).
- The application sorting oracle tests both bucket and 32-lane radix paths
  at capacities 65, 131,073, and 1,048,576. Fallback preserves the existing
  capability-based ignore behavior for the unsupported large radix path.
  `sort-native-vulkan-loader.log` records 26 native SPIR-V compilations and
  no DXC/DXIL library loads.

The isolated SDK candidate is based on published `a6bb5a7e7`, with only this
protocol, regressions, documentation, and stream-submission migration. It links
the existing backend/compiler libraries from the active build: this validates
the clean SDK surface, not a clean rebuild of the unrelated pending XIR work.
The Psycles gitlink is intentionally not advanced from the dirty SDK checkout.

# Coroutine extension examples

These headless examples exercise suspend extensions as executable runtime
contracts. They intentionally keep scheduling policy outside the core
coroutine compiler while using the compiler's exact frame-slot certificates.

The common workflow is:

1. The scheduler creates a local `CoroFrame` and reconstructs the slots listed
   by `Stage::reconstruct_slot_span()`, plus any explicit scheduler or
   application fields.
2. The handler reads and writes extension bindings only through their typed
   `CoroSlotAccess` callables.
3. The scheduler commits `Stage::required_writeback_slot_span()`, plus any
   explicit application outputs, to its chosen backing storage.

`external_stage_common.h` resolves complete normalized extensions without
creating extension-private slots. A side-band route buffer distinguishes
static suspend boundaries that share a schema or continuation token. Route
snapshots prevent a continuation from consuming work produced after the
current scheduling phase began.

## Structured debugger

`external_stage_debug.cpp` captures `(coroutine id, watched value)` records in
a dedicated buffer. The debugger adds `coro_id_x` to its reconstruction and
source-transport policies, reads the watched binding through its access
callable, and performs zero frame writeback. It does not use `device_log` or
device-side printing.

```text
example_coro_external_stage_debug metal
```

## Neural-SDF path tracing

`external_stage_neural_sdf.cpp` runs the supplied compact neural bunny SDF as
separate distance and normal passes between path-tracing continuations. The
network is absent from the continuation kernels: the coroutine exposes only
typed point/input and sample/output bindings. The default run writes
`coro_neural_sdf.png`; `--test` renders a 64 x 64 validation image without
creating an artifact.

```text
example_coro_external_stage_neural_sdf metal
example_coro_external_stage_neural_sdf metal --test
```

## On-demand virtual texture

`on_demand_texture.cpp` uses the source pattern:

```cpp
$while (table.read(page) == 0u) {
    $suspend("texture_miss", request);
};
```

The host owns a 64-page virtual texture and exposes only eight physical cache
slots to the device. Structured request bindings drive page loading and
eviction over eight fault rounds. The regression compares every reconstructed
texel with the host source. The default run writes
`coro_on_demand_texture.png`; `--test` uses a smaller image with the same page
and cache counts.

```text
example_coro_on_demand_texture metal
example_coro_on_demand_texture metal --test
```

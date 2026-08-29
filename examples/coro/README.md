# Coroutine extension examples

These headless examples execute suspend Extensions through
`WavefrontCoroScheduler`'s responsibility chain. The scheduler owns the frame
pool, exact static-stage queues, compaction, routing, and continuation resume;
each registered handler owns only the external operation it understands.

The public shape is:

```cpp
class MyInstance final
    : public WavefrontCoroSchedulerExtensionInstance {
    void dispatch(
        const WavefrontCoroExtensionDispatchContext &) override;
};

class MyHandler final
    : public WavefrontCoroSchedulerExtensionHandler {
    bool can_handle(const WavefrontCoroExtensionStage &) const override;
    luisa::unique_ptr<WavefrontCoroSchedulerExtensionInstance>
    prepare(const WavefrontCoroExtensionPrepareContext &,
            const WavefrontCoroExtensionStage &) override;
};

WavefrontCoroScheduler scheduler{device, coroutine, config};
scheduler.register_extension_handler(handler_a);
scheduler.register_extension_handler(handler_b);
stream << scheduler(args...).dispatch(size);
```

Registration order is semantic: the first matching handler owns each static
stage. One handler may claim any number of suspend Extension stages and
`prepare()` returns a separate executable instance for every one of them. It
runs at registration time, outside dispatch timing. When a queue is selected,
that stage's instance receives the same stream, the scheduler-owned frame
buffer, the exact Extension metadata, and an exact `frame_indices` subview;
work enqueued on the stream completes before those frames advance to the next
stage or continuation. Neither handlers nor instances are templated on, or
receive, the coroutine's complete invocation argument list. External resources
needed by an operation are bound explicitly to the handler or its prepared
instance.

Handlers follow the compiler's partial-frame contract:

1. Reconstruct `Stage::reconstruct_slot_span()` into a local `CoroFrame`.
2. Read and write values only through the stage's typed `CoroSlotAccess`
   bindings.
3. Commit `Stage::required_writeback_slot_span()` to the scheduler frame
   buffer.

These projections reuse the coroutine's interference-colored slots. There is
no Extension-private frame allocation and no duplicate slot model. Unclaimed
semantic Extensions are rejected. Unclaimed read-only annotations follow their
fallback policy and can be spliced out of the executable chain; an annotation
that writes frame state cannot be skipped safely.

## Structured debugger

`external_stage_debug.cpp` registers a read-only observer that captures
`(logical id, watched value)` records in a dedicated buffer. Both values are
explicit typed bindings, so the debugger does not need whole-frame access. It
does not use `device_log` or device-side printing.

```text
example_coro_external_stage_debug metal
```

## Neural-SDF path tracing

`external_stage_neural_sdf.cpp` registers one handler for the supplied compact
neural bunny's distance and normal schemas. The network stays outside the path
continuations: suspend sites expose only typed point inputs and sample/normal
outputs. The scheduler repeatedly selects those queues until every path
terminates; the example no longer contains a copied route/resume loop.

The default run writes `coro_neural_sdf.png`; `--test` renders a 64 x 64
validation image without creating an artifact.

```text
example_coro_external_stage_neural_sdf metal
example_coro_external_stage_neural_sdf metal --test
```

## On-demand virtual texture

`on_demand_texture.cpp` keeps the intended source spelling:

```cpp
$while (table.read(page) == 0u) {
    $suspend("texture_miss", request);
};
```

The registered cache handler reads the selected queue's structured page
bindings, loads or evicts host pages, uploads the page table and physical
cache on the scheduler stream, and returns. Unresolved paths naturally resume
and suspend at the same boundary again. The host exposes 64 virtual pages
through eight physical slots, and the regression verifies eight handler rounds,
64 loads, and every reconstructed texel.

The default run writes `coro_on_demand_texture.png`; `--test` uses a smaller
image with the same page and cache counts.

```text
example_coro_on_demand_texture metal
example_coro_on_demand_texture metal --test
```

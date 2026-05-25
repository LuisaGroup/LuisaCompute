# Coroutine pipeline: design notes

This document tracks the XIR-side coroutine pipeline shipped in
`feat/xir-gpu-coroutines` and the work remaining to reach full feature parity
with the SIGGRAPH Asia 2024 paper "GPU Coroutines for Flexible Splitting and
Scheduling of Rendering Tasks" and the archived Rust IR reference impl
(`LuisaCompute-coroutine/src/rust/luisa_compute_ir/src/transform/materialize_coro.rs`).

## Shipped (verified)

```
DSL ($suspend / coro_bind)
  → AST (SuspendStmt, CoroBindStmt)
  → ast2xir (CoroSuspendInst, CoroRegisterInst)
  → coroutine_analysis (markers, continuations, transitions, frame candidates)
  → choose lowering:
        coroutine_lower             (in-place state machine)
            ↓
        single function with switch dispatch
            ↓
        xir2ast roundtrip OK, runs PT/SDF examples at PSNR=100 dB

        coroutine_split             (cloning, flat coroutines only)
            ↓
        N+1 CallableFunctions: (frame_ref, ...args) → void
        + frame struct (slot 0 = target_token, slots 1.. = alloca-backed state)
            ↓
        coroutine_state_machine_scheduler_emit
            ↓
        kernel: alloca frame, call entry, simple_loop+switch dispatch
        + coroutine::CoroGraph (AST-side wrapper exposing per-continuation
          FunctionBuilder via xir_to_ast_translate)
```

`coroutine_split` rejects loop-containing coroutines (`is_supported=false`)
with a diagnostic pointing here.

## Not shipped — condition replay

PT/SDF use `$for(depth) { ... $suspend("bounce") ... }`. To support this
shape `coroutine_split` needs the condition-replay extension from the paper
appendix (Sec. CF reconstruction).

### Required new infrastructure

1. **CoroGraph CFG analysis** — paper Sec. 4.4.1 / Rust ref `coro_graph.rs`
   (~1600 LOC). Walks the structured CFG once, identifies the scope tree,
   and for each suspend collects:
   - The reachable node set per scope (paper Sec. 4.4.1 rules 1–4).
   - The transition graph: per (from-continuation, suspend-id) pair, the
     next continuation id or `exits` flag.
   - The condition-stack replay items: for each parent control-flow node
     (if/switch/loop) on the path from function entry to a suspend, the
     branch chosen at that node.
2. **Replayability detection** — Rust ref `materialize_coro.rs::replay_value`.
   Per-node decision tree: pure arithmetic / vector / matrix / constant /
   coro-id is replayable; thread/block/dispatch-id and side-effecting calls
   are not. Required so the transform can prefer in-line re-execution over
   frame loads where safe.
3. **Intra-scope use-define + inter-scope liveness** — Rust ref
   `coro_use_def.rs` + `coro_frame.rs` (~1200 LOC combined). Drives the
   frame layout: only values defined in a continuation A and used in a
   continuation B reachable from A across a suspend become frame slots.
4. **First-run flag mechanism** — paper Sec. CF reconstruction. Each parent
   loop on the path to a suspend gets an extra frame slot tracking whether
   the current iteration is the resumption iteration; nodes preceding the
   suspend in that loop are wrapped in `if(!first_run) { ... }`. After the
   resumption iteration, the flag is cleared so the loop runs normally.

### XIR primitives needed

The Rust IR builder has `value_or_load`, `def_or_assign`, `ref_or_local`,
`Pooled<BasicBlock>` cloning that automate the per-node "load from frame /
re-execute / reference original" choice. XIR has the lower-level
`InstructionCloneValueResolver` and `clone_with_metadata`, but no equivalent
of the higher-level helpers. Either add them or hand-roll the choice in the
transform.

### Estimated effort

- ~2000 LOC of new analysis + transform (excluding helpers)
- A synthetic-loop unit test + a runnable PT or SDF demo with PSNR
  comparison vs `coroutine_lower` as the verification gate. PSNR ≥ 30 dB
  required; ≥ 50 dB expected for a deterministic kernel.
- 2–3 dedicated sessions

## Not shipped — schedulers (paper Sec. 4.2 – 4.3)

`coroutine_state_machine_scheduler_emit` (Sec. 4.1) is shipped at the XIR
level. The paper also describes:

- **Persistent threads**: shared-memory frame queue, per-warp work-counter
  argmax, work redistribution across global memory queues. Hand-authored
  reference: `LuisaCompute-coroutine/src/coro/schedulers/persistent_threads.cpp`
  (~220 LOC of DSL kernel construction).
- **Wavefront**: multi-kernel host driver (setup / per-stage / finalize)
  with double-buffered atomic queues. Hand-authored reference at the
  example level: `examples/rendering/{path_tracing,sdf_renderer}_xir_wavefront.cpp`.

Building these as DSL templates on top of `coroutine::CoroGraph` (the same
shape as the old `StateMachineCoroScheduler<Args...>` etc.) is the next
public-API milestone. The CoroGraph bridge already exposes the per-continuation
`FunctionBuilder`s a custom user scheduler would compose against.

## Test coverage on this branch

| Pass / module | File | Tests | Asserts |
|---|---|---:|---:|
| analysis | `test_xir_pass_coroutine.cpp` | 4 | 38 |
| in-place lowering | `test_xir_pass_coroutine_lower.cpp` | 2 | 14 |
| split (flat) | `test_xir_pass_coroutine_split.cpp` | 3 | 22 |
| state-machine emitter | `test_xir_pass_coroutine_state_machine.cpp` | 2 | 9 |
| CoroGraph bridge | `test_coro_graph.cpp` | 2 | 10 |
| **Total** | | **13** | **93** |

All five pass. Plus `xir2ast` roundtrip is exercised inside the split test
on every emitted continuation, so the AST layer's CoroGraph wrapper has a
verified path from XIR.

## Reference

- Paper: SIGGRAPH Asia 2024, "GPU Coroutines for Flexible Splitting and
  Scheduling of Rendering Tasks"
- Old impl: https://github.com/LuisaGroup/LuisaCompute/tree/coroutine
  - `src/rust/luisa_compute_ir/src/transform/materialize_coro.rs` (transform)
  - `src/rust/luisa_compute_ir/src/analysis/coro_graph.rs` (CFG analysis)
  - `src/rust/luisa_compute_ir/src/analysis/coro_frame.rs` (frame layout)
  - `src/rust/luisa_compute_ir/src/analysis/coro_use_def.rs` (liveness)
  - `src/coro/schedulers/{state_machine,persistent_threads,wavefront}.cpp`

# CFG Normalization Plan

Two parallel pipelines for normalizing XIR control flow to the canonical structured form (only `IfInst`, `SwitchInst`, `LoopInst`, `SimpleLoopInst`, single final `ReturnInst`, plus `UnreachableInst`/`RasterDiscardInst`).

## Pipeline A — Per-Instruction Surgery (Low Priority, Fallback/Checker)

Bespoke passes that each handle one specific non-canonical construct via direct IR rewriting. Kept as a sanity-check baseline and fallback path.

- [x] **`lower_break_continue`** — replace every `BreakInst`/`ContinueInst` with `BranchInst` to the enclosing loop's `merge_block`/`update_block`
  - Header: `include/luisa/xir/passes/lower_break_continue.h`
  - Impl: `src/xir/passes/lower_break_continue.cpp`
  - Tests: `src/tests/unit/xir/test_xir_pass_lower_break_continue.cpp` (12/12 passing)
- [x] **`lower_ray_query_loop`** — existing pass: outlines candidate blocks to separate `Callable` functions and replaces the whole loop with a single `RayQueryPipelineInst`. **NOTE:** this is NOT the LoopInst+primitives form Pipeline B wants — it's an alternate lowering for backends that expose a "pipeline" RQ API rather than primitive operations. Pipeline B Pass 1 needs its own separate RQ-lowering routine.
- [ ] **`early_return_elimination`** — IfInst-wrap strategy
  - Stub exists at `src/xir/passes/early_return_elimination.cpp` with the flag-alloca scaffolding; `eliminate_early_return()` body is empty
  - Algorithm: spill return value (if non-void) to local; replace each early `ReturnInst` with `store(flag, false); br(<enclosing merge>)`; wrap all post-merge code along the chain from `body_block` to the final return in nested `IfInst(load(flag))` guards; final return loads from the spilled value
  - [ ] Implement `eliminate_early_return(ReturnInst*, AllocaInst*)`
  - [ ] Implement the post-pass IfInst-wrapping along the merge chain
  - [ ] Handle non-void return value spill
  - [ ] Tests: no-op (single return), two early returns, nested-if early return, early-return-inside-loop, non-void return, external function skip, idempotence

## Pipeline B — Destructure → Simplify → Restructure (High Priority, Generic)

Three-stage classical compiler pipeline. Lower structured control flow to a goto-graph, clean it up, then re-derive structured form. Unlocks future lowering passes that don't want to know about every control-flow construct.

**Naming:** new pass files use the `*_cfg` suffix to keep them visibly distinct from Pipeline A's `lower_*`/`*_elimination` files.

### Invariants

- **Preserved opaquely through all three passes:** `SwitchInst` (recurse into case blocks but never destructure the dispatch itself)
- **Lowered in destructure, stays lowered:** `RayQueryLoopInst` → `LoopInst` + primitive RQ ops
- **Lowered in destructure, re-derived in restructure:** `IfInst`, `LoopInst`, `SimpleLoopInst`, `BreakInst`, `ContinueInst`, early `ReturnInst`
- **Irreducible CFG = hard error** with diagnostic (restructure pass rejects)

### Pass 1 — `destructure_cfg`

- [x] Header: `include/luisa/xir/passes/destructure_cfg.h`
- [x] Impl: `src/xir/passes/destructure_cfg.cpp`
- [x] Pre-step: lower every `RayQueryLoopInst`+`RayQueryDispatchInst` to a `LoopInst` skeleton using `PROCEED`, `IS_TERMINATED`, `IS_TRIANGLE_CANDIDATE`, `IS_PROCEDURAL_CANDIDATE` primitive ops via `RayQueryObjectReadInst` and branches over candidate types. **NEW code** (not reusing existing `lower_ray_query_loop` which targets `RayQueryPipelineInst` instead).
- [x] Lower `IfInst` → `cond_br` to true/false blocks; both branch to the original `merge_block`
- [x] Lower `LoopInst`/`SimpleLoopInst` → unstructured back-edge using `br`/`cond_br`
- [x] Lower `BreakInst` → `br(enclosing_merge_block)`
- [x] Lower `ContinueInst` → `br(enclosing_update_block)`
- [ ] Lower early `ReturnInst` → flag-store + `br(exit_block)`; synthesize single final exit block with single `ReturnInst`; spill return value to local if non-void
- [x] Recurse into `SwitchInst` case blocks (do NOT destructure the switch itself)
- [ ] Output invariant check: only `BranchInst`/`ConditionalBranchInst`/`SwitchInst`/`ReturnInst`(single)/`UnreachableInst`/`RasterDiscardInst` terminators remain
- [x] Tests (12 tests, 46 asserts, all green):
- [x] Lone if → cond_br pattern
- [x] Lone loop → cond_br back-edge
- [x] break/continue → forward branches
- [ ] Early return → exit-block branch
- [ ] Non-void early return → spilled value
- [x] Switch preserved opaquely
- [x] RayQueryLoop → LoopInst + primitives
- [x] Nested constructs (if-in-loop, loop-in-if, switch-in-loop)
- [x] External function skip
- [x] Idempotence check (running again produces no change once destructured)
- [ ] Output invariant check (no IfInst/LoopInst/etc. in terminators except SwitchInst)

### Pass 2 — `simplify_cfg` ✅

- [x] Header: `include/luisa/xir/passes/simplify_cfg.h`
- [x] Impl: `src/xir/passes/simplify_cfg.cpp`
- [ ] Merge straight-line blocks (single predecessor, single successor, both unconditional) — subsumed by jump-threading + unreachable removal
- [x] Fold `cond_br` with constant boolean condition → `br`
- [x] Remove unreachable blocks (not reachable from entry)
- [x] Remove empty blocks containing only an unconditional `br` (jump-threading) — handles switch case redirects too
- [x] Do NOT touch `SwitchInst` case-block boundaries (treat each case block opaquely)
- [x] Fixed-point iteration (repeat until no change)
- [x] Tests (8 tests, 22 asserts, all green):
  - [x] Constant-condition cond_br folding (true + false)
  - [x] Unreachable block removal
  - [x] Empty-block jump-threading
  - [x] Switch default+case retarget through empty blocks (switch preserved)
  - [x] Idempotence
  - [x] No-op on empty function
  - [x] Module entry point

### Pass 3 — `restructure_cfg`

- [ ] Header: `include/luisa/xir/passes/restructure_cfg.h`
- [ ] Impl: `src/xir/passes/restructure_cfg.cpp`
- [ ] Build dominator tree (reuse existing `dom_tree.cpp` if applicable)
- [ ] Build post-dominator tree
- [ ] Detect natural loops via dominator back-edges; reject irreducible CFG with `LUISA_ERROR`
- [ ] Identify single-entry/single-exit regions (hammocks) for `IfInst` formation
- [ ] Materialize `LoopInst`/`SimpleLoopInst` for natural loops (choose `SimpleLoopInst` when no `prepare`/`update` distinction needed)
- [ ] Materialize `IfInst` for hammocks with a conditional branch at the entry
- [ ] Recurse into `SwitchInst` case blocks
- [ ] Output invariant check: only `IfInst`/`SwitchInst`/`LoopInst`/`SimpleLoopInst`/`ReturnInst`(single)/`UnreachableInst`/`RasterDiscardInst` terminators
- [ ] Tests:
  - [ ] Round-trip: destructured if → restructured if
  - [ ] Round-trip: destructured loop → restructured loop
  - [ ] Nested round-trip
  - [ ] Switch case bodies restructured
  - [ ] Irreducible CFG produces diagnostic
  - [ ] Idempotence

### Pipeline B Integration

- [ ] End-to-end round-trip test: structured input → destructure → simplify → restructure → semantic equivalence with original
- [ ] Add labels to CMake test registration so all Pipeline B tests run together (`pipeline_b_cfg`)
- [ ] Document recommended pass ordering in each header

### Backend Validation (end-to-end)

- [x] Fallback backend integration: env-guarded `LUISA_XIR_NORMALIZE_CFG=1` runs destructure+simplify after `lower_ray_query_loop` in `fallback_shader.cpp`. Build green. Runtime validation blocked by pre-existing fallback hang during scene/shader setup (unrelated to CFG passes).
- [x] HIP backend integration: env-guarded `LUISA_XIR_NORMALIZE_CFG=1` runs destructure+simplify after AST→XIR in `hip_device.cpp::create_shader`. Build green.
- [x] HIP A/B codegen validation: `test_runtime hip` and `test_sdf_renderer hip` produce **bit-identical AMDGPU bytes** with and without `LUISA_XIR_NORMALIZE_CFG=1` (19268+8192 bytes for SDF kernels). +1–2 ms pass overhead. No new errors. Codegen-level semantic preservation confirmed for non-RQ kernels.
- [x] Full path-tracing offline run on HIP: `test_path_tracing hip --offline --spp 1` exits 0 after EASTL `fixed_vector` move-ctor/move-assign fix at `src/ext/EASTL/include/EASTL/fixed_vector.h` (root cause: `has_overflowed()` read uninitialized `mpBegin`). Image output matches; no SIGSEGV.
- [x] CMake refactor: `examples/CMakeLists.txt` now provides `MIRROR_AS_TEST` opt-in flag on `luisa_compute_add_example` plus `luisa_example_pair_link` helper. 26 auto-checkable examples (all `path_tracing*`, `sdf_renderer[_ir]`, `photon_mapping`, `blackhole`, `voxel_raytracer`, `procedural`, `shader_toy*`, `shader_visuals_present`, all simulations, `image_processing`, `helloworld`) are built as both `example_<name>` and `test_<name>`. GUI/interop demos opt out.
- [ ] Full path-tracing image diff against `docs/gallery/*.png` oracles via `test_path_tracing` mirror.

## Execution Order

1. Start with **Pipeline B Pass 1 (`destructure_cfg`)** — foundation for the rest
2. Then **Pass 2 (`simplify_cfg`)**
3. Then **Pass 3 (`restructure_cfg`)** — the hard one
4. End-to-end integration test
5. Loop back to finish **Pipeline A `early_return_elimination`** (low priority fallback)

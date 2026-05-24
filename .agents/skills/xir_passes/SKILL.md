---
name: xir_passes
description: Writing XIR transformation passes — APIs, idioms, and pitfalls discovered while building Pipeline B (destructure_cfg, simplify_cfg, restructure_cfg). Load when authoring or modifying any pass under src/xir/passes/.
---

# XIR Passes: Authoring Guide

This skill captures hard-won knowledge from implementing the CFG normalization pipeline. Read before touching anything under `src/xir/passes/` or `include/luisa/xir/passes/`.

## Layout & Registration

- Header: `include/luisa/xir/passes/<name>.h`
- Impl: `src/xir/passes/<name>.cpp`
- Register impl in `src/xir/CMakeLists.txt` (look for the `passes/` block, ~line 80-90; alphabetical).
- Test: `src/tests/unit/xir/test_xir_pass_<name>.cpp`, registered in `src/tests/CMakeLists.txt` (look for `test_xir_pass_*` block).

## Standard Pass Interface

Every pass exposes a `<Name>Info` POD with counters plus two entry points:

```cpp
struct FooPassInfo {
    size_t did_something_count = 0u;
};

[[nodiscard]] LUISA_XIR_API FooPassInfo foo_pass_run_on_function(Function *function) noexcept;
[[nodiscard]] LUISA_XIR_API FooPassInfo foo_pass_run_on_module(Module *module) noexcept;
```

The module entry point iterates `module->functions()` and dispatches by `func->is_definition()`.

## Hook Rules (project priority-3)

- No docstrings.
- No memo/explanatory comments.
- Only `// namespace foo` trailers after closing braces of namespaces are allowed.
- BDD (`// given / when / then`) is also allowed in tests.

## Core APIs

### Module / Function

```cpp
module->function_list()                          // iterable (NOT `functions()`)
func->is_definition()                            // gate before downcast
auto def = static_cast<FunctionDefinition*>(f);  // NEVER cast_or_null — no such template in XIR
def->body_block()                                // entry block (NEVER remove)
def->create_basic_block()                        // orphan block
def->basic_blocks()                              // ManagedIntrusiveList<BasicBlock>
def->traverse_basic_blocks(visitor)              // walks only reachable blocks from body_block
```

### BasicBlock

```cpp
block->instructions()                                  // ManagedIntrusiveList<Instruction>
block->instructions().empty()                          // always false if terminator present
block->instructions().front()                          // first inst
block->terminator()                                    // last inst (may be nullptr if malformed)
block->traverse_instructions(visitor)
block->traverse_predecessors(exclude_self, visit)      // visits via use list
block->traverse_successors(exclude_self, visit)        // visits terminator's target operands
block->remove_self()                                   // returns ManagedPtr<BasicBlock>; detaches from func block list
```

### Constant detection

```cpp
if (auto v = inst->cond(); v->isa<Constant>()) {
    auto c = static_cast<Constant*>(v);
    bool b = c->as<bool>();    // checks size; safe for bool
}
```

### Cast pattern

XIR does **not** have `cast_or_null<>` or LLVM-style `cast<>`. Use:

```cpp
if (v->isa<SomeType>()) {
    auto s = static_cast<SomeType*>(v);
    ...
}
```

For instruction-tag switch: `inst->derived_instruction_tag()` returns `DerivedInstructionTag::*`.

## Terminator Inventory & APIs

| Terminator | Header | Key API |
|---|---|---|
| `BranchInst` (br) | `instructions/branch.h` | `target_block()`, `set_target_block(BasicBlock*)` |
| `ConditionalBranchInst` (cond_br) | `instructions/branch.h` | **Getters**: `condition()`, `true_block()`, `false_block()`. **Setters**: `set_true_target` / `set_false_target` (asymmetric naming — getter says `block`, setter says `target`) |
| `SwitchInst` | `instructions/switch.h` | `value()`, `default_block()`, `case_count()`, `case_value(i)`, `case_block(i)`, `set_case_block(i, bb)`, `set_default_block(bb)`, `add_case(v, bb)` |
| `ReturnInst` | `instructions/return.h` | `value()` |
| `UnreachableInst` | `instructions/unreachable.h` | none |
| `RasterDiscardInst` | `instructions/raster_discard.h` | none |
| `IfInst` (structured) | `instructions/if.h` | `condition()`, `true_block()`, `false_block()`, `merge_block()` |
| `LoopInst` (structured) | `instructions/loop.h` | `prepare_block()`, `body_block()`, `update_block()`, `merge_block()`. **No `condition()` getter** — the loop condition lives as the terminating `cond_br(cond, body_block, merge_block)` of `prepare_block()`. Canonical shape from `ast2xir.cpp:970-1004`: `prepare: cond_br(cond, body, merge)`, `body → br(update)`, `update → br(prepare)`. Setters: `set_prepare_block`, `set_body_block`, `set_update_block`. Creators: `create_prepare_block(overwrite=false)`, `create_body_block(...)`, `create_update_block(...)`. |
| `SimpleLoopInst` (structured) | `instructions/loop.h` | `body_block()`, `merge_block()` |
| `BreakInst` (structured) | `instructions/break.h` | `target_block()` |
| `ContinueInst` (structured) | `instructions/continue.h` | `target_block()` |
| `RayQueryLoopInst` (structured) | `instructions/ray_query.h` | `dispatch_block()`, `merge_block()` |
| `RayQueryDispatchInst` | `instructions/ray_query.h` | `query_object()`, `on_surface_candidate_block()`, `on_procedural_candidate_block()` (parent is `RayQueryLoopInst`) |

After Pipeline B `destructure_cfg`, only the **unstructured** terminators + `SwitchInst` + `ReturnInst` + `UnreachableInst` + `RasterDiscardInst` remain.

## XIRBuilder

```cpp
XIRBuilder b;
b.set_insertion_point(block);            // or block->instructions().front() etc.
b.br(target)                             // BranchInst
b.cond_br(cond, true_target, false_target)
b.if_(cond)                              // returns IfInst*; populate sub-blocks via if->true_block() etc.
b.loop()                                 // LoopInst*; fill prepare/body/update
b.simple_loop()                          // SimpleLoopInst*; fill body
b.ray_query_loop()                       // 0 args; query object only passed to ray_query_dispatch
b.ray_query_dispatch(query_value)        // inside dispatch_block
b.call(type, op, operands)               // typed call (read ops)
b.call(op, operands)                     // void call (write ops, e.g., RQ PROCEED)
b.return_(value)
b.unreachable_()
b.break_(target)                         // structured
b.continue_(target)                      // structured
```

For RQ primitive ops (`include/luisa/xir/op.h` ~line 170-187):

- `RayQueryObjectReadOp::IS_TERMINATED, IS_TRIANGLE_CANDIDATE, IS_PROCEDURAL_CANDIDATE, ...`
- `RayQueryObjectWriteOp::PROCEED, COMMIT_TRIANGLE, COMMIT_PROCEDURAL, TERMINATE`

## Mutation Idiom: Two-Phase Collect-Rewrite

You cannot reliably mutate the instruction list while iterating it. Pattern from `lower_break_continue.cpp`:

```cpp
luisa::vector<IfInst*> to_lower;
def->traverse_basic_blocks([&](BasicBlock *bb) {
    if (auto t = bb->terminator(); t && t->isa<IfInst>()) {
        to_lower.push_back(static_cast<IfInst*>(t));
    }
});

for (auto if_inst : to_lower) {
    auto bb = if_inst->parent_block();
    auto true_b = if_inst->true_block();
    auto false_b = if_inst->false_block();
    auto cond = if_inst->condition();
    if_inst->remove_self();
    XIRBuilder b; b.set_insertion_point(bb);
    b.cond_br(cond, true_b, false_b);
}
```

For passes that grow the worklist (e.g., RayQueryLoop → new LoopInst → re-process), wrap in a **fixed-point loop**:

```cpp
bool changed = true;
while (changed) {
    changed = false;
    luisa::vector<...> worklist;
    def->traverse_basic_blocks(...);
    if (!worklist.empty()) { changed = true; rewrite(); }
}
```

## Constant Folding / Branch Retargeting

To redirect every reference to block `from` in a terminator to point at `to`:

```cpp
auto retarget = [&](Instruction *term, BasicBlock *from, BasicBlock *to) {
    switch (term->derived_instruction_tag()) {
        case DerivedInstructionTag::BRANCH: {
            auto br = static_cast<BranchInst*>(term);
            if (br->target_block() == from) br->set_target_block(to);
            break;
        }
        case DerivedInstructionTag::CONDITIONAL_BRANCH: {
            auto cb = static_cast<ConditionalBranchInst*>(term);
            if (cb->true_target() == from) cb->set_true_target(to);
            if (cb->false_target() == from) cb->set_false_target(to);
            break;
        }
        case DerivedInstructionTag::SWITCH: {
            auto sw = static_cast<SwitchInst*>(term);
            if (sw->default_block() == from) sw->set_default_block(to);
            for (size_t i = 0; i < sw->case_count(); ++i) {
                if (sw->case_block(i) == from) sw->set_case_block(i, to);
            }
            break;
        }
        default: break;
    }
};
```

## Reachability / Dead Block Removal

`def->traverse_basic_blocks(...)` already walks only reachable blocks from `body_block()`. To remove unreachable blocks:

```cpp
luisa::unordered_set<BasicBlock*> reachable;
def->traverse_basic_blocks([&](BasicBlock *bb) { reachable.insert(bb); });
luisa::vector<BasicBlock*> dead;
for (auto bb : def->basic_blocks()) {
    if (!reachable.contains(bb)) dead.push_back(bb);
}
for (auto bb : dead) bb->remove_self();
```

**Always preserve `def->body_block()`** — never remove it even if it looks empty.

## Test Patterns (Boost.UT / `doctest`?)

XIR unit tests live in `src/tests/unit/xir/`. Check existing `test_xir_pass_*.cpp` for framework; they use the project's chosen harness (was `boost::ut` last checked, see `/test` skill).

Key test fixtures:

```cpp
Module m;
auto *k = m.create_kernel();                            // KernelFunction*
auto body = k->create_body_block();                     // entry BB
// or:
auto *c = m.create_callable(Type::of<float>());         // CallableFunction*
auto def = static_cast<FunctionDefinition*>(k);         // both kernel/callable are FunctionDefinitions

XIRBuilder b;
b.set_insertion_point(body);
// build IR ...
b.return_void();

auto info = my_pass_run_on_function(def);
```

**Reachability gotcha**: `traverse_basic_blocks` only visits blocks reachable from `body_block`. If you build orphan blocks for a test, you must wire them up via `br`/`cond_br` from `body_block` or the pass will see nothing. Trick: `m.create_constant_one(Type::of<bool>())` + `cond_br(true_const, target, other)` to force reachability.

## Pipeline B Status (CFG Normalization)

Master plan: `src/xir/passes/CFG_NORMALIZATION_PLAN.md`.

| Pass | Status | File |
|---|---|---|
| Pipeline A `lower_break_continue` | ✅ done (12 tests) | `lower_break_continue.{h,cpp}` |
| Pipeline A `lower_ray_query_loop` | ✅ existing (lowers to `RayQueryPipelineInst` — **NOT** reusable for Pipeline B) | `lower_ray_query_loop.cpp` |
| Pipeline A `lower_ray_query_loop_to_loop` | ✅ done (lowers to structured `LoopInst` + nested `IfInst` dispatch) | `lower_ray_query_loop_to_loop.{h,cpp}` |
| Pipeline A `early_return_elimination` | ⏳ stub (low pri) | `early_return_elimination.cpp` |
| Pipeline B Pass 1 `destructure_cfg` | ✅ done (12 tests, 46 asserts) | `destructure_cfg.{h,cpp}` |
| Pipeline B Pass 2 `simplify_cfg` | ✅ done (8 tests, 22 asserts) | `simplify_cfg.{h,cpp}` |
| Pipeline B Pass 3 `restructure_cfg` | ✅ done | `restructure_cfg.{h,cpp}` |
| Round-trip Pipeline B test | ✅ verified (path_tracing_cutout PSNR>30) | via `test_path_tracing_cutout vk` |

### `destructure_cfg` lowerings (reference)

- `IfInst` → `cond_br(cond, true, false)`; merge_block reachable via inner brs.
- `LoopInst` → `br(prepare)`.
- `SimpleLoopInst` → `br(body)`.
- `BreakInst` / `ContinueInst` → `br(target)`.
- `RayQueryLoopInst` → emit `LoopInst{prepare→body, body: PROCEED + cond_br cascade on IS_TERMINATED→merge / IS_TRIANGLE_CANDIDATE→on_surface / IS_PROCEDURAL_CANDIDATE→on_procedural / else→update, update→prepare}`; rewrite child `br dispatch_block` → `br update_block`; remove orphaned `RayQueryDispatchInst`. New `LoopInst` destructured on next fixed-point iteration.
- `SwitchInst` **preserved as-is**; recursion handled naturally by `traverse_basic_blocks`.

### `simplify_cfg` ops

1. Constant `cond_br` fold → `br`.
2. Empty-block jump-threading (block with only a `br C` terminator; redirect all preds; never remove `body_block`).
3. Unreachable block removal (collect reachable from `body_block`, remove rest).
4. Fixed-point until no change.
5. Counters: `folded_constant_cond_br_count`, `threaded_empty_block_count`, `merged_straight_line_count`, `removed_unreachable_block_count`.

## Pitfalls Catalogue

- ❌ `cast_or_null<T>(v)` — doesn't exist. Use `isa<T>` + `static_cast`.
- ❌ `set_true_block` / `set_false_block` on ConditionalBranchInst — wrong names. Asymmetric: getters are `true_block()` / `false_block()`, setters are `set_true_target` / `set_false_target`.
- ❌ `module->functions()` — wrong. Use `module->function_list()`.
- ❌ `b.ray_query_loop(query)` — wrong; takes 0 args. Pass query to `ray_query_dispatch`.
- ❌ Mutating instructions while iterating — always two-phase collect-rewrite.
- ❌ Removing `body_block()` — never. Even if empty, it must stay.
- ❌ Building orphan test blocks without wiring reachability — `traverse_basic_blocks` will skip them silently.
- ❌ Forgetting fixed-point loop when transformation creates new candidates (RayQueryLoop → new LoopInst).
- ❌ Touching `SwitchInst` case-block contents structurally — Pipeline B preserves switches; only fold/thread within cases.
- ❌ Writing memo comments — project hook will flag and force you to apologize.
- ❌ Calling `LoopInst::condition()` — **does not exist**. The loop condition is the terminating `cond_br(cond, body, merge)` of `prepare_block()`. To read the condition: `static_cast<ConditionalBranchInst*>(loop->prepare_block()->terminator())->condition()`. Likewise no `set_condition`; rewrite the prepare-block terminator instead.
- ❌ Restructuring CFG with live `PhiInst` nodes — splitting/inserting blocks (preheaders, latches, exit stubs) invalidates phi `incoming_blocks`. Run `reg2mem_pass_run_on_module` before `restructure_cfg_pass_run_on_module` so the input is phi-free; assert this as a precondition.
- ❌ Computing post-dominators without a virtual exit — multi-sink CFGs (`ReturnInst`, `UnreachableInst`, `RasterDiscardInst` in different blocks) yield wrong/null ipostdoms for blocks whose successors reach different sinks. Add a synthetic virtual exit that all sinks point to before running the iterative ipostdom algorithm.
- ❌ Running `reg2mem` immediately after `restructure_cfg` without DCE — restructure_cfg may leave orphan blocks not reachable from `body_block()`. These blocks are absent from the dom tree, causing assertion failures in reg2mem. Always run `dce_pass_run_on_module` between `restructure_cfg` and `reg2mem`.
- ❌ Using `OpCopyMemory` on `OpTypeRayQueryKHR` in SPIR-V emission — forbidden since Rev 15. Instead, remap `_value_map[store->variable()] = val` so subsequent loads resolve to the source variable directly.

## Memory Effects & Instruction Purity

Optimization passes (GVN, DCE, SCCP) must respect memory effects. Instructions fall into three categories:

### Pure (safe to value-number, CSE, reorder, DCE if unused)

| Tag | Examples |
|---|---|
| `ARITHMETIC` | all ops — no memory side effects |
| `CAST` | all cast ops |
| `GEP` | pointer arithmetic only, no dereference |
| `RESOURCE_QUERY` | `buffer_size`, `texture_size` — read-only metadata |
| `RAY_QUERY_OBJECT_READ` | `IS_TERMINATED`, `IS_TRIANGLE_CANDIDATE`, etc. |
| `CLOCK` | technically pure but non-deterministic |

### Memory-reading (safe to DCE if unused, NOT safe to reorder past writes or value-number without alias analysis)

| Tag | Examples |
|---|---|
| `LOAD` | local alloca/GEP load |
| `RESOURCE_READ` | `buffer_read`, `texture_read`, `byte_buffer_read` |

### Memory-writing / side-effecting (NEVER DCE, NEVER reorder past other writes/reads to same location)

| Tag | Examples |
|---|---|
| `STORE` | local alloca/GEP store |
| `RESOURCE_WRITE` | `buffer_write`, `texture_write`, `byte_buffer_write` |
| `CALL` (to definitions) | may have arbitrary side effects |
| `ATOMIC` | read-modify-write |
| `PRINT` | observable side effect |
| `ASSERT` / `ASSUME` | control flow / UB |
| `AUTODIFF_INTRINSIC` (non-GRADIENT) | tape manipulation |

### Implications for pass authors

1. **GVN**: only value-number pure instructions + `RESOURCE_QUERY`. `RESOURCE_READ` and `LOAD` require memory dependency analysis (not yet implemented) to prove no intervening write.

2. **DCE**: remove instructions with `use_list().empty()` ONLY if they are pure or memory-reading. Never remove writes, atomics, calls to definitions, prints, or asserts.

3. **SCCP**: only fold `ARITHMETIC` on constant operands. Branch elimination is safe (replaces `cond_br` with `br`) but must call `term->remove_self()` BEFORE `builder.set_insertion_point(block)` — otherwise the builder targets the tail sentinel and asserts.

4. **Code motion**: pure instructions can be hoisted/sunk freely. Reads can be hoisted past other reads but not past writes to the same resource. Writes cannot be reordered with respect to other accesses to the same resource.

5. **`is_safe_to_remove` (used by GVN/DCE cleanup)**: checks `use_list().empty()` + instruction tag whitelist. Current whitelist: `PHI`, `ALLOCA`, `LOAD`, `GEP`, `ARITHMETIC`, `CAST`, `CLOCK`, `RAY_QUERY_OBJECT_READ`, `RESOURCE_QUERY`, `RESOURCE_READ`, `AUTODIFF_INTRINSIC(GRADIENT)`.

### Checking purity in code

```cpp
// No single API exists yet. Use the tag switch:
[[nodiscard]] static bool is_pure(Instruction *inst) noexcept {
    switch (inst->derived_instruction_tag()) {
        case DerivedInstructionTag::ARITHMETIC:
        case DerivedInstructionTag::CAST:
        case DerivedInstructionTag::GEP:
        case DerivedInstructionTag::RESOURCE_QUERY:
        case DerivedInstructionTag::RAY_QUERY_OBJECT_READ:
            return true;
        default:
            return false;
    }
}
```

## Build & Test Commands

```bash
# Build XIR library only (fast iteration)
cmake --build cmake-build-release --target luisa-compute-xir -j

# Build one specific test
cmake --build cmake-build-release --target test_xir_pass_destructure_cfg -j

# Run test
cmake-build-release/bin/test_xir_pass_destructure_cfg

# Or via ctest filter
ctest --test-dir cmake-build-release -R destructure_cfg --output-on-failure
```

Build dir convention: `cmake-build-release`.

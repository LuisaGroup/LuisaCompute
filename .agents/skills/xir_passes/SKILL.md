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
| `LoopInst` (structured) | `instructions/loop.h` | `prepare_block()`, `condition()`, `body_block()`, `update_block()`, `merge_block()` |
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
| Pipeline A `early_return_elimination` | ⏳ stub (low pri) | `early_return_elimination.cpp` |
| Pipeline B Pass 1 `destructure_cfg` | ✅ done (12 tests, 46 asserts) | `destructure_cfg.{h,cpp}` |
| Pipeline B Pass 2 `simplify_cfg` | ✅ done (8 tests, 22 asserts) | `simplify_cfg.{h,cpp}` |
| Pipeline B Pass 3 `restructure_cfg` | ⏳ pending | not yet |
| Round-trip Pipeline B test | ⏳ pending | not yet |

### `destructure_cfg` lowerings (reference)

- `IfInst` → `cond_br(cond, true, false)`; merge_block reachable via inner brs.
- `LoopInst` → `br(prepare)`.
- `SimpleLoopInst` → `br(body)`.
- `BreakInst` / `ContinueInst` → `br(target)`.
- `RayQueryLoopInst` → emit `LoopInst{prepare→body, body: PROCEED + cond_br cascade on IS_TERMINATED→merge / IS_TRIANGLE_CANDIDATE→on_surface / IS_PROCEDURAL_CANDIDATE→on_procedural / else→update, update→prepare}`; rewrite child `br dispatch_block` → `br update_block`; remove orphaned `RayQueryDispatchInst`. New `LoopInst` destructured on next fixed-point iteration.
- `SwitchInst` **preserved as-is**; recursion handled naturally by `traverse_basic_blocks`.

### `simplify_cfg` planned ops

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

---
name: xir-passes
description: Implement, compose, test, and debug LuisaCompute XIR transformation and analysis passes. Use when modifying files under src/xir/passes or include/luisa/xir/passes, changing CFG or SSA invariants, authoring a PassPipeline, or diagnosing pass-order, reachability, PHI, dominance, aliasing, or memory-effect bugs.
---

# XIR Pass Development

## Read the exact contract first

Open the pass's public header, implementation, nearest related pass, and tests
before editing. Treat the header as authoritative: pass entry points are not
uniform. Depending on the pass, an API may be module-only, accept `Function *`
or `FunctionDefinition *`, omit `PassReport`, or return a specialized result.

Use the common layout without assuming a fixed line number or alphabetical
registration:

- public header: `include/luisa/xir/passes/<pass>.h`
- implementation: `src/xir/passes/<pass>.cpp`
- library registration: `src/xir/CMakeLists.txt`
- focused tests: `src/tests/unit/xir/`
- test registration: `src/tests/CMakeLists.txt`

Copy the closest correct pass pattern. Aggregate per-function counters rather
than overwriting the result on each function.

## Use core APIs safely

Resolve a definition before traversing blocks:

~~~cpp
if (auto *definition = function->definition()) {
    definition->traverse_basic_blocks([](xir::BasicBlock *block) noexcept {
        // inspect reachable blocks
    });
}
~~~

Use these exact APIs:

| Need | API |
|---|---|
| all module functions | `module->function_list()` |
| all blocks owned by a function | `function->basic_blocks()` |
| reachable blocks from the body | `definition->traverse_basic_blocks(...)` |
| function entry | `definition->body_block()` |
| return operand | `return_inst->return_value()` |
| conditional targets | `true_block()`, `false_block()` |
| retarget a conditional | `set_true_target(...)`, `set_false_target(...)` |
| branch target | `target_block()`, `set_target_block(...)` |
| detach a node | `remove_self()` |

Guard `BasicBlock::terminator()` and `traverse_successors()` with
`is_terminated()`. `terminator()` asserts on an unterminated block; it does not
return null as a malformed-IR probe.

Do not use LLVM casting helpers such as `cast_or_null`. Check `isa<T>()` and
then use `static_cast<T *>`.

## Distinguish structured and plain CFG

Structured terminators own or reference regions and merge blocks. Plain CFG
uses `BranchInst` and `ConditionalBranchInst` edges. `SwitchInst` also carries
merge information and `contains_structured_control_flow` classifies it as
structured. Inspect `DerivedInstructionTag` for the complete current
instruction inventory rather than copying an exhaustive list into a pass.

Apply the relevant lowering order:

1. Run `lower_ray_query_loop_to_loop` to convert ray-query loops into ordinary
   structured loop/if constructs, and check its rejection result.
2. Run `lower_switch` if the consumer cannot retain switches, and check its
   rejection result before continuing.
3. Run `destructure_cfg` to lower `IfInst`, `LoopInst`, `SimpleLoopInst`,
   `BreakInst`, and `ContinueInst`; it also handles early-return spilling and
   patches unterminated owned blocks.

Do not claim that `destructure_cfg` lowers `RayQueryLoopInst` directly or
removes every specialized terminator. It deliberately preserves switches and
other instruction families outside its public contract.

Use `contains_structured_control_flow` from `src/xir/passes/helpers.h` before a
plain-CFG-only mutation.

## Keep inlining after destructuring when it is multi-block

Follow the inliner's actual contract:

- Inline a single-block callee into a structured caller without splitting CFG.
- Reject a multi-block call while either caller or callee contains structured
  control flow.
- Destructure caller and callee, then run `inline_all` immediately. Do not
  insert cleanup, SSA, or another CFG pass between those two steps. This
  enables eligible multi-block call sites.
- Treat a nonzero `rejected_malformed_call_count` as a hard error in backend
  normalization; a zero inlined-call count by itself is not a failure.
- In ordinary lowering, run `mem2reg` immediately after `inline_all` to remove
  inliner-created argument and return spill slots before SSA optimization. In
  pre-autodiff normalization, allow autodiff scopes in the caller, then use
  cleanup and `reg2mem` before restructuring instead.
- Remember that `destructure_cfg` deliberately preserves switches and other
  specialized structured operations; reference-argument and other unsupported
  uses may also leave allocas after `mem2reg`.
- Preserve opaque custom references. `create_value_argument` rejects custom
  types and `promote_ref_arg` intentionally excludes them; a backend-owned
  handle may have a concrete ABI only after backend lowering.
- Expect recursive callables and preserved structured cases to remain.

When changing this schedule, add a shape test that checks the call disappears,
the inliner does not report a structured skip, eligible generated temporaries
are promoted, every reachable block remains terminated, and successor PHI
incoming blocks still match real predecessors after block splitting.

After basic optimization, the Metal4 AIR consumer conditionally normalizes
autodiff scopes with:

~~~text
lower_ray_query_loop_to_loop (checked)
-> lower_switch (checked)
-> destructure_cfg (checked)
-> inline_all (immediately adjacent; allow_autodiff_scope_in_caller)
-> post-inline cleanup (one fixed-point iteration)
-> simplify_cfg
-> reg2mem
-> restructure_cfg (checked)
-> verify(no PHIs, unique merge blocks)
-> autodiff
-> reg2mem
-> verify(no PHIs, unique merge blocks)
~~~

It then lowers every module with:

~~~text
lower_ray_query_loop_to_loop (checked)
-> lower_switch (checked)
-> destructure_cfg (checked)
-> inline_all (immediately adjacent)
-> mem2reg
-> SSA optimization
-> unused_callable_removal
-> simplify_cfg
-> verify(require_reachable_blocks = true)
~~~

The AIR entry point repeats the reachable-block verification before XIR-to-LLVM
translation. Treat `destructure_cfg -> inline_all` as the common adjacency and
`inline_all -> mem2reg` as the ordinary-phase adjacency. Read
`src/backends/metal4/metal_xir_pipeline.cpp` before changing either schedule;
read `src/xir/passes/pass_pipeline.cpp` for the current contents of factory
pipelines rather than copying their expansion here.

## Mutate lists deliberately

Prefer collect-then-rewrite when a transform removes or replaces nodes during
traversal. Mutate in place only when iterator advancement and node lifetime are
explicitly controlled by an established local pattern.

For a replacement:

1. Capture every operand, parent, and target needed after removal.
2. Detach the old instruction.
3. Set the builder insertion point.
4. Create or append the replacement.
5. Repair use-def edges, PHIs, and metadata.

Do not remove blocks with a naive reachable-set recipe on structured CFG.
Merge blocks and owned regions require structurally aware traversal. On plain
CFG, clean incoming PHI entries and detach contained instructions before
deleting a block. Never remove `body_block()`.

## Preserve SSA and dominance invariants

- Create all destination blocks before cloning branch targets.
- Create PHIs before resolving cyclic incoming values; attach incoming pairs
  after all blocks and values are mapped.
- Recompute dominance and loop analyses after CFG mutation.
- Ensure each PHI incoming block remains a real predecessor.
- Run `mem2reg` only after CFG and alloca placement are valid.
- Use `reg2mem` before a transform that cannot preserve live PHIs, when that
  transform's documented precondition requires it.

Remember that `traverse_basic_blocks` visits blocks reachable from
`body_block()`, while `basic_blocks()` includes disconnected owned blocks.
Choose intentionally.

## Model memory effects conservatively

Use `get_memory_info()` from `src/xir/passes/helpers.h`; do not maintain a
second hard-coded purity whitelist in a new pass.

Check all of the following independently:

- memory scope and read/write effects;
- volatility and synchronization;
- aliasing with intervening accesses;
- dominance and availability;
- speculation safety for the exact opcode.

Purity alone does not make an instruction safe to hoist or speculate. Integer
division, remainder, and shifts are examples that require additional checks.
Treat `CLOCK` as a non-deterministic read, not a pure instruction.

Treat these `ResourceQueryOp` values as stateful ray-query constructors, not
ordinary pure queries:

- `RAY_TRACING_QUERY_ALL`
- `RAY_TRACING_QUERY_ANY`
- `RAY_TRACING_QUERY_ALL_MOTION_BLUR`
- `RAY_TRACING_QUERY_ANY_MOTION_BLUR`

Each creates fresh mutable traversal state. `get_memory_info()` classifies it
as `GLOBAL/READ`, non-volatile: it remains removable when unused, but is not
safe to value-number, common with Early CSE/GVN, speculate, or hoist with LICM.
Keep the classification centralized in `helpers.cpp`, and make optimizers
consult it instead of whitelisting all `ResourceQueryInst` values.

## Build pipelines with accurate change reporting

Use `PassPipeline::add` adapters that return whether the module changed:

~~~cpp
xir::PassPipeline pipeline;
pipeline.add("inline-all", [](xir::Module *module,
                              xir::PassReport &report) {
    auto info = xir::inline_all_pass_run_on_module(module, &report);
    return info.inlined_call_count != 0u;
});
auto stats = pipeline.run(module);
~~~

Use `add_fixed_point` only when every child is safe to repeat and every change
predicate is correct. A false negative terminates the group early; a false
positive wastes iterations or masks non-convergence.

Do not infer implementation maturity from registration. Read the header and
source. Some current loop transforms and outlining APIs are placeholders that
validate or report input without rewriting it.

## Respect backend preflight normal forms

A pass can preserve valid XIR while making a backend reject it. For the Metal4
AIR path, inspect `luisa_compute_metal_codegen_llvm_supported` alongside any
change that affects types, special registers, calls, atomics, or resource-use
shape. Its checks run before LLVM construction for JIT compute, reverse
autodiff, raster JIT, and compile-only raster archive creation. All four paths
fail closed on unsupported XIR; Metal4 has no MSL or legacy-IR shader fallback.
Compute and raster AOT loading consume validated archives and bypass XIR
preflight.

This preflight and its pass schedule live in the independent `metal4` backend;
the original `metal` backend remains the source-MSL compatibility path.

Keep producer and preflight assumptions paired. In particular, texture uses
must reach preflight as direct resource operations after normalization. A
compute AIR module requires exactly one kernel and rejects raster special
registers. A raster AIR module instead requires exactly one
`RasterStageFunction` with the configured vertex or fragment role; object ID
is valid in both roles, while barycentrics, derivatives, and discard require a
fragment role. External declarations may remain only when `native_include`
supplies ABI-compatible LLVM IR/bitcode definitions at link time. Ensure every
operand and result type is checked; constants can otherwise carry an
unsupported type past an instruction-only scan.

Treat raster stage functions as ABI roots:

- `dead_arg_elim` must not remove their arguments, even when a stage body does
  not read a slot.
- `unused_callable_removal` must seed reachability from both kernels and raster
  stages.
- Keep `destructure_cfg -> inline_all` immediately adjacent for each separately
  translated vertex and fragment module, then run the same SSA cleanup and
  reachable-block verification used by compute AIR.
- Preserve `RasterStageFunction::stage()` through interchange text/bitcode and
  debug text; use `raster_vertex` and `raster_fragment` interchange kinds.

When a pass change affects this boundary, add both:

1. a XIR shape test for the required normal form; and
2. a Metal4 runtime test for JIT compute, or the dedicated raster test. The
   AIR-only backend fails closed, and the dedicated raster test exercises AIR
   end to end.

## Test behavior and shape

Use Boost.UT for XIR unit tests. Wire every test block to `body_block()` when it
must be reached by traversal. Test both the information counters and the IR
invariants that matter:

- expected instruction count or absence;
- all reachable blocks terminated;
- branch targets and PHI predecessors valid;
- no stale uses after removal;
- idempotence when promised;
- rejection leaves the module unchanged;
- pass reports match returned counters.

Build and run focused tests with CMake/Ninja:

~~~sh
cmake --build cmake-build-metal4-air --target test_xir_passes -j 8
ctest --test-dir cmake-build-metal4-air -R '^test_xir_passes$' --output-on-failure
ctest --test-dir cmake-build-metal4-air -L unit_xir --output-on-failure
~~~

Use the Metal4 backend for an end-to-end AIR test. The dedicated
`test_metal_xir_air` CTest requires no selection environment. Use
`test_metal_xir_air_raster` for vertex/fragment stage identity, raster-only
operations, and render readback; launch other binaries with backend `metal4`.

---
name: ir-pipeline
description: Navigate and modify LuisaCompute AST-to-XIR lowering, CFG/SSA representation changes, pass composition, and Metal XIR-to-LLVM/AIR normalization. Use when tracing AST-to-XIR calls, changing pass order, debugging structured versus plain CFG, or extending AIR ABI features such as bool/byte layout, PACK, acceleration, raster stages, runtime builtins, stateful ray queries, external linkage, and fail-closed boundaries.
---

# LuisaCompute IR and XIR Pipelines

## Start from the implementation

Treat these files as the source of truth:

- `src/xir/translators/ast2xir.cpp` for AST-to-XIR lowering.
- `include/luisa/xir/` and `src/xir/` for the XIR object model.
- `include/luisa/xir/passes/` and `src/xir/passes/` for pass contracts and implementations.
- `include/luisa/xir/passes/pass_pipeline.h` and
  `src/xir/passes/pass_pipeline.cpp` for pipeline composition and current
  factory expansions.
- `src/xir/translators/xir2text.cpp` for debugging output.
- `src/backends/metal4/metal_xir_pipeline.cpp` for the Metal4-specific XIR
  schedule.
- `src/backends/metal4/llvm_codegen/metal_codegen_llvm.cpp` for the AIR ABI,
  support preflight, and XIR-to-LLVM lowering.
- `src/backends/metal4/llvm_codegen/metal_codegen_llvm_builtin.cpp` and
  `src/backends/metal4/metal_builtin_air.cpp` for LLVM-generated runtime
  support kernels, verification, LLVM 14 downgrade, and five-entry MTLB
  packaging.
- `src/backends/metal4/metal_air_pipeline.cpp` for LLVM O2, verification, and
  LLVM 14 bitcode downgrade, including explicit macOS/iOS AIR targets.
- `src/backends/metal4/metal_metallib.cpp` for target-specific metallib
  container headers and validation.
- `src/backends/metal4/tools/ios_path_tracing_aot.cpp` and
  `src/backends/metal4/tools/ios_path_tracer_app/` for the host-AOT regression
  and the runtime-linked, on-device XIR/LLVM/AIR physical-device probe.
- `src/backends/metal4/metal_raster_ext.cpp`, `metal_raster_shader.cpp`, and
  `metal_command_encoder.cpp` for the Metal raster host ABI, PSO creation, and
  render-command encoding.

The LLVM/AIR implementation is owned by the independent `metal4` backend.
Do not add these stages back to `src/backends/metal`: the original backend is
the source-MSL compatibility path. Both modules share the upgraded
`src/backends/common/metal-cpp` headers, not a shader code-generation pipeline.

Inspect the relevant public header and implementation before changing a pass
order. Do not rely on a copied pass inventory: several registered passes are
intentional placeholders, and factory contents evolve.

## Use the current IR path

The removed Rust/JSON legacy IR tree is not a fallback implementation. Current
compiler work starts at AST and lowers into native C++ XIR before backend
legalization. Do not recreate a parallel legacy route to bypass an XIR or AIR
failure.

Do not use XIR JSON as serialization. `xir2json.cpp` emits debug-only metadata
and flat XIR text, while `json2xir.cpp` has no implementation. Use
`xir_to_text_translate` or `xir_to_flat_text_translate` for inspection.

Treat XIR-to-AST as a constrained lowering, not a lossless module round trip.
Call `xir_to_ast_normalize_module` when required, and account for its limits on
PHIs, external functions, recursion, and top-level function count.

## Understand the XIR CFG forms

XIR represents both structured and plain control flow.

- AST-to-XIR initially emits structured `IfInst`, `LoopInst`,
  `SimpleLoopInst`, `SwitchInst`, and ray-query constructs with owned blocks
  and merge information.
- `BranchInst` and `ConditionalBranchInst` represent plain CFG edges.
- `destructure_cfg` lowers supported structured constructs to branches, spills
  supported early returns, and terminates leaked unterminated owned blocks.
- `restructure_cfg` reconstructs reducible structured regions from suitable
  plain CFG.

Interpret `SimpleLoopInst` as an unconditional loop with a body backedge and
explicit `break` exit. It has no condition and is not a do-while construct.

Do not assume `destructure_cfg` makes every function completely plain. It
lowers `IfInst`, `LoopInst`, `SimpleLoopInst`, `BreakInst`, and
`ContinueInst`, while deliberately preserving `SwitchInst` and specialized
terminators. Run `lower_switch` first only when the consumer requires switch
lowering and handle that pass's rejection result.

Use `contains_structured_control_flow` from `src/xir/passes/helpers.h` when a
pass requires a plain-CFG mutation boundary.

## Compose pipelines explicitly

Use `PassPipeline::add` for one pass and `add_fixed_point` only for a group
that is safe and useful to repeat. Return `true` from a callback exactly when
the pass changed the module.

~~~cpp
xir::PassPipeline pipeline;
pipeline.add("destructure-cfg", [](xir::Module *module,
                                   xir::PassReport &report) {
    auto info = xir::destructure_cfg_pass_run_on_module(module, &report);
    if (!info.succeeded()) {
        LUISA_ERROR_WITH_LOCATION(
            "CFG destructuring failed (errors={}, leaked_blocks={}).",
            info.error_count, info.leaked_block_count);
    }
    return info.destructured_if_count != 0u ||
           info.destructured_loop_count != 0u ||
           info.destructured_simple_loop_count != 0u ||
           info.destructured_break_count != 0u ||
           info.destructured_continue_count != 0u ||
           info.destructured_early_return_count != 0u;
});
auto stats = pipeline.run(module);
stats.log("backend lowering");
~~~

Available factory entry points are:

- `create_basic_optimization_pipeline`
- `create_post_inline_cleanup_pipeline`
- `create_ssa_optimization_pipeline`
- `create_post_restructure_cleanup_pipeline`

Read `pass_pipeline.cpp` for their exact current order rather than duplicating
the expansion in another skill.

## Preserve ordering invariants

Apply these constraints when they match the selected passes:

1. Run `lower_ray_query_loop_to_loop`, then `lower_switch`, before
   `destructure_cfg` when the consumer requires a plain CFG. Check each
   lowering result and stop on rejection; both passes create or normalize
   structured control flow that destructuring must subsequently flatten.
2. Run `destructure_cfg`, then `inline_all` immediately. Do not insert cleanup,
   SSA, or another CFG pass between them: the inliner accepts a single-block
   callee in a structured caller, but rejects multi-block calls when either
   caller or callee still contains structured control flow. When normalizing
   autodiff, pass `InlineOptions{.allow_autodiff_scope_in_caller = true}`.
   Treat `rejected_malformed_call_count` as a hard pipeline error.
3. In the ordinary lowering phase, run `mem2reg` immediately after
   `inline_all`; the inliner can create argument and return-value temporaries
   that promotion should remove before SSA optimization. The pre-autodiff
   phase is intentionally different: perform its cleanup, demote cross-block
   SSA with `reg2mem`, and restructure before running autodiff.
4. Run `create_ssa_optimization_pipeline` only after the ordinary CFG lowering
   and `mem2reg` have established the representation that factory expects.
5. Recompute dominance-dependent analyses after any CFG mutation. Do not
   reuse a dominance tree, frontier, loop analysis, or PHI assumption across
   a structural change.

The Metal4 AIR path in `src/backends/metal4/metal_xir_pipeline.cpp` first runs
basic optimization. If the module contains an autodiff scope, it then uses:

~~~text
lower_ray_query_loop_to_loop (checked)
-> lower_switch (checked)
-> destructure_cfg (checked)
-> inline_all (immediately adjacent; autodiff scopes allowed)
-> post-inline cleanup (one fixed-point iteration)
-> simplify_cfg
-> reg2mem
-> restructure_cfg (checked)
-> verify(no PHIs, unique merge blocks)
-> autodiff
-> reg2mem
-> verify(no PHIs, unique merge blocks)
~~~

Every module then passes through the ordinary AIR lowering phase:

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

The AIR entry point repeats the reachable-block verifier before constructing
LLVM IR. Keep inlining immediately after destructuring in both phases.
Recursive callables and call sites that still contain preserved structured
operations may remain uninlined.

For ray queries, this order is part of the lowering contract rather than a
generic cleanup preference. `lower_ray_query_loop_to_loop` first converts the
callback-shaped query construct into a structured loop containing explicit
query-object reads and writes. `destructure_cfg` then exposes ordinary branch
CFG, and the immediately following `inline_all` keeps every opaque query
lifecycle inside one AIR function. Moving inlining before destructuring or
putting another CFG/SSA transform between those passes can leave a query in an
unsupported structured or cross-call form.

For the AIR-facing pipeline, ABI, and fail-closed contract, read
[AIR_INTRINSICS_AND_TRANSLATION.md](../../../src/backends/metal4/llvm_codegen/AIR_INTRINSICS_AND_TRANSLATION.md).

## Protect the Metal4 AIR boundary

Preserve these ABI rules when changing lowering or adding types:

- Keep separate register and memory forms. SSA vectors use LLVM vectors;
  memory vectors use padded arrays with `align(N, 2)` elements. Structures use
  explicit gaps and Luisa member offsets rather than LLVM's default aggregate
  layout.
- Keep AIR's `i1:8:8` layout. A scalar `bool` occupies one byte, a Luisa
  `bool4` in memory is `[4 x i1]` and occupies four bytes, and `<4 x i8>` also
  occupies four bytes. Never use `<4 x i1>` as the memory form: its compact
  LLVM vector storage is the one-byte representation that does not match
  Luisa. A four-field `{bool, bool, bool, bool}` structure also occupies four
  bytes because each `i1` structure field has an eight-bit allocation unit.
- Initialize every register-to-memory aggregate from zero before inserting
  logical fields. This makes vector lanes, explicit structure gaps, and tail
  padding deterministic for bitwise operations such as `PACK`.
- Keep address spaces fixed: private/generic `0`, device `1`, constant `2`,
  and threadgroup `3`.
- Align every root kernel argument to 16 bytes and round the root block to at
  least 16 bytes. A buffer is `{ptr addrspace(1), i64}`; a texture binding is
  an opaque eight-byte handle in its own 16-byte argument slot.
- Materialize that minimum 16-byte root at runtime even when a compute shader
  has no logical arguments. Direct argument tables and the built-in
  `prepare_indirect_dispatches` kernel must receive a nonzero GPU address;
  Metal API Validation rejects a null `kernel_args` binding even when the
  indirect target does not read buffer zero.
- Treat vertex and fragment functions as first-class `RasterStageFunction`
  entries. Translate an AST raster function only with an explicit
  `AST2XIRConfig::raster_stage`; the AST tag alone does not distinguish vertex
  from fragment. Preserve raster entry arguments in dead-argument elimination
  and seed their callable graphs in unused-callable removal.
- Build one shared raster root block in vertex-extra-argument then
  fragment-extra-argument order. Skip each stage's payload argument, align
  every remaining slot to 16 bytes, keep the minimum block size at 16 bytes,
  bind it at Metal buffer index 0 in both stages, and bind the per-draw object
  ID at index 1. Vertex streams start at index 2.
- Flatten mesh attributes in stream-major order. AIR vertex inputs use
  `air.vertex_input` plus `air.location_index`; truncate or extend the concrete
  `PixelFormat` value into the fixed `AppData` member. Vertex outputs use
  `air.position` followed by matching `user(locnN)` varyings. Fragment inputs
  default to `air.center` plus `air.perspective` for floating varyings and
  `air.flat` for non-floating varyings. Preserve reflected
  `LUISA_RASTER_VARYING_INTERPOLATION(...)` member attributes and map the
  selectable center/centroid/sample plus perspective/no-perspective modes to
  their exact AIR metadata pairs. Reject non-floating perspective modes and
  invalid attributes before code generation; render values use
  `air.render_target`.
- Keep raster barycentrics as `float3` from `FunctionBuilder` through
  `SPR_Barycentrics` and AIR `air.barycentric_coord`. Lower fragment
  derivatives to `air.dfdx.*`/`air.dfdy.*` and discard to
  `air.discard_fragment`. Raster object ID is valid in both vertex and fragment
  stages; primitive ID and barycentrics are fragment-only values.
- Keep `raster_is_front_face()` as an AST bool builtin and
  `SPR_FrontFacing` special register. The physical AIR fragment parameter is
  `i1 noundef` with `air.front_facing`, type name `bool`, and argument name
  `front_facing`; this register ABI does not change byte-sized bool memory
  layout. Pass it through the hidden raster callable state and reject its use
  outside a fragment stage.
- Keep `raster_base_instance()` as a vertex-only AST uint builtin and
  `SPR_BaseInstance` special register. AIR receives it as the third builtin
  vertex parameter, `i32 noundef`, with `air.base_instance` metadata. Carry it
  through the hidden raster callable state and reject fragment or compute use.
  `RasterMesh` stores a draw-time base instance (default zero); forward it to
  indexed and non-indexed MTL4, DX12, and Vulkan draw calls rather than leaving
  the shader-visible value permanently zero.
- Lower `raster_set_z_depth`, `raster_set_z_depth_greater_equal`, and
  `raster_set_z_depth_less_equal` to fragment-only `ThreadGroupOp` values and
  inline them into the fragment entry. With colors, the implementation return
  is `{logical_color, f32 depth}`; a void depth-only fragment uses Apple's
  packed singleton `<{ float }>` return. The external entry flattens colors,
  appends depth without incrementing the color-attachment count, and reflects
  it with `air.depth`, `air.depth_qualifier`, and exactly one of `air.any`,
  `air.greater`, or `air.less`. Allow zero color attachments only for a
  fragment that writes depth. Reject mixed qualifiers and depth operations
  left in compute, vertex, or callable code.
- Treat ordinary stencil as host pipeline/render-pass state, not an AIR
  intrinsic or fragment return. `StencilState` has one public eight-bit
  reference, eight-bit read/write masks, and separate front/back comparison,
  stencil-fail, depth-fail, and pass operations limited to Keep/Zero/Replace.
  Metal4 binds matching `MTL::StencilDescriptor` objects and the dynamic
  reference, excludes that reference from its PSO cache key, and attaches the
  same texture to both depth and stencil render-pass planes. Logical D24S8 uses
  physical D32S8A24 when the device does not support depth24-stencil8; preserve
  the logical `DepthFormat`. Shader-written stencil reference and conservative
  rasterization remain fail-closed.
- Keep direct and indirect entry generation separate. Direct dispatch reads a
  constant-space `uint3`; indirect dispatch reads a device-space `uint4` whose
  W component is the kernel ID.
- Opaque `Type::Tag::CUSTOM` arguments cannot be XIR value arguments. When an
  AST API presents a backend handle by value, AST-to-XIR must create an opaque
  reference and the backend must define its concrete ABI. Do not call
  `Type::size()` or `Type::alignment()` on a custom type in shared code.
- Metal's supported `LC_IndirectDispatchBuffer` binding is
  `{device pointer, uint offset, uint capacity}` (16 bytes). Its allocation has
  a 16-byte count header followed by 32-byte slots containing aligned `uint3`
  block size and `uint4(dispatch_size, kernel_id)`. The AIR reflection consumes
  three physical locations and spells the type `LCIndirectDispatchBuffer`.
- Metal's supported acceleration binding is a 16-byte `LCAccel` containing an
  instanced acceleration-structure handle and a device pointer to 72-byte,
  8-byte-aligned `LCInstance` records. The final eight-byte field is the
  primitive acceleration structure's `MTLResourceID`; reflection consumes two
  physical locations. AIR covers static triangle/curve closest/any traces,
  direct triangle/curve primitive-and-instance motion traces, instance
  transform/user-ID/visibility queries, and transform/visibility/opacity/
  user-ID writes. `ShaderOption::enable_extended_accel_limits` selects the
  oracle-matched `.extended_limits` variant of all four static/curve/motion/
  curve-motion direct-trace suffixes without changing their ABI. It also covers
  local stateful traversal of static triangles, procedural bounding boxes, and
  curves. Direct procedural traces and stateful motion queries fail closed.
  Apple's Metal 4 frontend rejects `extended_limits` on
  `intersection_query`, so stateful-query shaders and raster shaders must also
  reject that option instead of silently emitting a non-extended intrinsic.
- Do not infer Metal4 AS-build support from successful MTL4 queue/compiler
  creation. Address-driven AS builds require Apple9. The runtime checks
  `supportsFamily(MTL::GPUFamilyApple9)`: Apple9+ encodes the MTL4 descriptor
  path, while Apple7/Apple8 synchronously bridge only build/refit/compact to an
  isolated legacy `MTL::CommandQueue`. Shaders, AIR pipelines, argument tables,
  and compute/render dispatch remain MTL4 on both paths.
- Keep standalone `MotionInstance` runtime packing separate from the AIR
  `LCAccel` ABI. Shaders always see the 72-byte `LCInstance` record; a motion
  TLAS is built from 48-byte indirect-motion descriptors and a separate
  transform buffer. MATRIX keyframes are packed as column-major
  `PackedFloat4x3` after `outer * keyframe` composition. Static instances in a
  motion TLAS receive one keyframe. Component/SRT transforms are Apple9-only,
  and their nonzero quaternions must be normalized before native upload;
  one TLAS cannot mix MATRIX and SRT modes, and a non-translation outer SRT
  transform must fail closed rather than change interpolation semantics.
- A `MotionInstanceBuildCommand` captures the built child and keyframes; it
  performs no GPU work by itself. `AccelBuildCommand` snapshots and uploads
  those values. Refit coverage must submit `motion_instance.build()` followed
  by `accel.build()`, and should verify mixed static/motion metadata as well as
  closest/any ray-time traversal.
- Recognize only `LC_RayQueryAll` and `LC_RayQueryAny` as AIR query custom
  types. Each must be a compiler-owned function-entry local allocation lowered
  to one generic address-space-zero `%struct._intersection_query_t*`: allocate
  once, reset once, and deallocate in reverse order on every exit. Query
  objects must not be loaded as ordinary storage, indexed, passed across
  calls, returned, or reflected as kernel resources.
- Treat `RAY_TRACING_QUERY_ALL`, `RAY_TRACING_QUERY_ANY`, and their motion-blur
  variants as constructors of fresh mutable traversal state. Their XIR memory
  effect is a removable global read: Early CSE and GVN must not common them,
  and LICM must not speculate or hoist them out of their control-flow site.
- Preserve the native reset tail exactly. After the visibility mask it is
  `0,0,0,0,0,geometry,basis,0,control_points,false,accept_any`. Geometry is a
  bit mask: triangle `1`, bounding box `2`, curve `4`; therefore supported
  combinations are `1`, `3`, `5`, and `7`. Curve basis/control-point pairs are
  B-spline `0/4`, Catmull--Rom `1/4`, linear `2/2`, Bezier `3/4`, or mixed
  `0xffffffff/0`. `accept_any` distinguishes `LC_RayQueryAny` from
  `LC_RayQueryAll`; stateful motion remains disabled because native
  `intersection_query` has neither motion suffixes nor a ray-time operand.
- Give direct AST `RAY_QUERY_PROCEED` both XIR operations the same query lvalue:
  emit one `RAY_QUERY_OBJECT_PROCEED` write, immediately read
  `RAY_QUERY_OBJECT_IS_TERMINATED`, and return its logical negation. Lower that
  pair to one native `next` call and a cached terminated result; the read must
  not call `next` again.
  Candidate native types are triangle `1`, procedural `2`, and curve `3`;
  triangle and curve map to Luisa surface kind `1`, while procedural maps to
  kind `2`. Curves expose `(parameter, -1)` as barycentrics and use the native
  curve commit. Use committed distance as the returned world ray's current
  `t_max`, guard procedural commit to the inclusive
  `[ray_min_distance, committed_distance]` interval, and lower terminate to
  native abort.
- Keep LLVM 21 query pointers opaque in the IR but attach downgrade metadata:
  allocate has `ret_eltype = %struct._intersection_query_t`, every query
  argument zero has the matching `arg_eltypes` entry, and reset argument five
  has `%struct._instance_acceleration_structure_t`. Native getters are
  `argmemonly ... readonly` with a `nocapture readonly` query argument;
  lifecycle, next, commit, and abort calls are not read-only. Consult the AIR
  document for the exact symbol spellings and attribute sets.
- Lower `PACK`/`UNPACK` through a one-member, at-least-four-byte-aligned Luisa
  structure, then bitcast the complete wrapper to or from an array of uint
  words. Scalar bool/byte/short therefore consume one word, `float3` consumes
  four, and padding must be zero rather than poison.
- Normalize `TYPED_BINDLESS_*` and `TYPED_UNIFORM_BINDLESS_*` AST aliases to
  the existing ordinary XIR bindless `ResourceQueryOp`, `ResourceReadOp`, or
  `ResourceWriteOp`. Preserve the typed result and operands; do not invent a
  parallel backend opcode family.
- Preserve external AST declarations/calls in XIR using values for ordinary
  read-only arguments, references for write/inout and opaque-custom arguments,
  and resource arguments for resources. Metal4 emits exact LLVM declarations;
  `native_include` must supply ABI-compatible LLVM IR/bitcode definitions,
  which are linked before O2 and downgrade. Missing or incompatible symbols
  fail shader creation.
- Lower `PrintInst` to Apple's `air.os_log` ABI through a module-local variadic
  helper. It contributes no root field, physical argument location, staging
  buffer, or callable state. Attach `MTL::LogState` to every MTL4 command
  buffer via `MTL4::CommandBufferOptions`; keep direct/indirect format tables
  deterministic and normalize bool markers in the log callback.
- Keep one immutable `MTL4::CommandBufferOptions` with that log state per
  stream and create a fresh command buffer for each submission. Pool only
  command allocators, reset them from commit feedback after GPU completion,
  and cap the pool. Argument tables and residency sets remain submission-
  owned. Never reset or recycle an allocator while any command encoded from it
  is in flight, and do not pool command buffers without separate lifetime and
  numerical-correctness proof.
- Verify final XIR with reachable-block ownership required both at the end of
  normalization and at the AIR entry point. For each entry, then verify emitted
  LLVM IR, run the default per-module O2 pipeline, verify again, and only then
  write LLVM 14-compatible bitcode with the in-tree downgrade code.

Treat `luisa_compute_metal_codegen_llvm_supported` as part of the backend
contract, not a convenience check. Keep it synchronized with emission and
validate operand/result types, special registers, resource-use normalization,
and operation-specific constraints before LLVM construction. For native ray
queries, preflight must prove the local one-initializer lifecycle, recognized
read/write operation set, matching query type, and non-escaping uses before
LLVM construction.

Metal4 is unconditionally XIR/LLVM/AIR and does not compile or link the MSL AST
code generator. Ordinary JIT compute, reverse-autodiff JIT, raster JIT, and
compile-only raster archive creation all fail closed when XIR normalization,
preflight, LLVM emission, downgrade, or AIR loading fails. Compute and raster
AOT loaders consume their compiled archives directly and do not rerun XIR
preflight; neither has a source-code fallback.

Apply the same rule to runtime support. Acceleration-instance update,
bindless-table update, indirect-command preparation, and swapchain presentation
live in `metal_codegen_llvm_builtin.cpp`; `metal_builtin_air.cpp` verifies,
optimizes, downgrades, and packages them as three kernel plus vertex/fragment
entries. BC6H/BC7 are fixed support metallibs compiled for the selected SDK at
build time. Runtime code loads their bytes and must not construct MSL source,
compile options, or a source library descriptor.

Choose the AIR target explicitly when the artifact is not for the current
runtime device. The default target follows the current macOS or iOS process,
while `MetalAIRTarget` can select iOS and its minimum OS/SDK versions for host
AOT. An iOS artifact must use an
`air64_v28-apple-ios...` triple and SDK metadata matching the selected target.
Its metallib header uses platform byte `0x82` and leaves the file-major high bit
clear; macOS uses platform byte `0x81` and sets that bit. Keep the generator and
validator symmetric, and use Apple's `metallib --app-store-validate` as an
independent check.

Do not reject a current iOS device merely because it has a newer minor/patch
than the linked SDK (for example runtime 26.6 with SDK 26.4). The current-device
path accepts that same-major update and targets the runtime version; it still
rejects a runtime major newer than the SDK. Explicit host-AOT targets retain
the stricter full `SDK >= deployment` check.

Do not describe the iOS path as AOT-only. The device app statically links LLVM
21, the in-tree downgrade, AST, XIR, runtime, DSL, AIR codegen, and the Metal4
backend, then performs AST -> XIR -> LLVM -> LLVM 14-compatible AIR -> MTLB on
the phone before dispatching through the real `DeviceInterface`. LLVM is used
as an IR optimizer/writer and does not generate executable CPU pages, so this
does not cross iOS's arbitrary-CPU-JIT boundary. Keep the older host-AOT probe
as a separate container/runtime baseline; account for app size, startup
compilation cost, private AIR ABI stability, and distribution policy.

Every Metal4 graphics or rendering pass is AIR coverage by construction, but
still keep focused tests that localize the failing feature. Put each new AIR
feature in a small test with dynamic inputs so constant
folding cannot erase the intrinsic. For atomics, check both the returned old
value and the final memory value. For textures, use read-only sources and
write-only destinations when same-dispatch visibility is not the feature under
test.
Use `test_metal_xir_air_accel` as the strict regression for the supported
acceleration ABI, intrinsic, reflection, and result-field mapping. Keep true
vertex/fragment coverage in `test_metal_xir_air_raster`: it must traverse
stage XIR, AIR metadata, metallib loading, render PSOs, command encoding, and
perform color plus D32 shader-depth readback across both JIT and AOT. Keep the
front-facing assertion dynamic: disable culling, invert
`RasterState::front_counter_clockwise`, and require opposite JIT/AOT pixel
values. Also use a nonzero `RasterMesh` base instance, carry it through an
integer flat varying, change only that draw-time value, and require opposite
visibility across indexed/non-indexed JIT/AOT draws. Compile a second void
fragment that writes depth, bind only a DSV, require no `air.render_target`,
and read back the same depth from JIT and archive-loaded AOT draws. For
selectable interpolation, compile every supported qualifier, inspect the exact
AIR metadata token pair, and render nonuniform clip W so perspective-correct
and screen-linear values must diverge; require the archive-loaded AOT image to
equal JIT exactly. Use the
registered `test_metal_xir_air_ray_query` for stateful
triangle, procedural-bounding-box, and static-curve semantics, then run the
cutout renderer to exercise its production `LC_RayQueryAll` path. For
image-correctness claims,
use the deterministic offline cutout seed and compare Metal4 AIR against the
separate same-build legacy `metal` MSL backend.

Use `test_metal4_raster_stencil` for executing host-stencil coverage. It must
exercise logical D24S8 and D32S8A24, nonzero reference, read/write masks,
front/back state, Replace/Zero/Keep across pass, stencil-fail, and depth-fail,
and clear/load/store across consecutive draws. A descriptor-only test is not
sufficient. Its Validation registration must run with a visible Metal device.

Distinguish Metal API Validation from GPU Shader Validation. The full Metal4
integration label should pass with `LUISA_ENABLE_VALIDATION=1` and
`MTL_DEBUG_LAYER=1`. Apple documents that Shader Validation requires pipeline
and buffer inheritance for ICBs; Luisa's GPU-written ICB deliberately carries
per-command root and dispatch-record pointers, so do not enable instrumentation
for that path by changing its ABI. The private nested bindless sampler table is
also numerically checked under ordinary/API-Validation execution because the
current Shader Validation instrumentation changes its sampler modes without a
reported fault. Keep these exclusions explicit instead of loosening output
tolerances or claiming complete GPU-Validation coverage.

## Recognize placeholder passes

Do not infer implementation status from a pass name or CMake registration.
At the time of this skill update:

- XIR JSON export is debug-only and import remains unimplemented.
- `outline` reports unsupported outline instructions without outlining.
- loop rotation, loop unroll, loop fusion, and loop vectorization reject
  unsupported structured inputs and otherwise leave accepted plain CFG
  unchanged.
- SLP vectorization performs real transformations.

Recheck the public header and implementation before relying on any status in
this list.

## Debug a pipeline change

1. Dump the module before and after the changed boundary.
2. Check whether every reachable block is terminated before calling APIs that
   require a terminator.
3. Count structured terminators, calls, allocas, and PHIs at each stage.
4. Verify pass reports and the `bool changed` adapter agree.
5. Add a regression that fails under the old order and asserts the intended
   IR shape, not only runtime output.
6. Run the Metal4 backend end to end; it is AIR-only and fails closed.

For Metal, set `LUISA_DUMP_XIR=1` to emit initial and optimized XIR. Use
`LUISA_DUMP_LLVM_IR=1` for the post-O2 AIR-targeted LLVM modules.

## Validate

Use the configured CMake/Ninja build directory:

~~~sh
cmake --build cmake-build-metal4-air --target test_xir_passes -j 8
ctest --test-dir cmake-build-metal4-air -R '^test_xir_passes$' --output-on-failure
cmake --build cmake-build-metal4-air --target test_metal_xir_air -j 8
ctest --test-dir cmake-build-metal4-air -R '^test_metal_xir_air$' --output-on-failure
cmake --build cmake-build-metal4-air --target test_metal_xir_air_accel test_ast_pack_usage test_ast_external_lowering -j 8
ctest --test-dir cmake-build-metal4-air -R '^(test_metal_xir_air_accel|test_ast_pack_usage|test_ast_external_lowering)$' --output-on-failure
cmake --build cmake-build-metal4-air --target test_metal4_air_extended_accel_limits -j 8
ctest --test-dir cmake-build-metal4-air -R '^test_metal4_air_extended_accel_limits$' --output-on-failure
cmake --build cmake-build-metal4-air --target test_metal_xir_air_ray_query -j 8
ctest --test-dir cmake-build-metal4-air -R '^test_metal_xir_air_ray_query$' --output-on-failure
cmake --build cmake-build-metal4-air --target test_printer test_printer_custom_callback -j 8
cmake-build-metal4-air/bin/test_printer metal4
cmake-build-metal4-air/bin/test_printer_custom_callback metal4
cmake --build cmake-build-metal4-air --target test_indirect -j 8
cmake-build-metal4-air/bin/test_indirect metal4
cmake --build cmake-build-metal4-air --target test_metal4_raster_stencil -j 8
ctest --test-dir cmake-build-metal4-air -R '^test_metal4_raster_stencil(_validation)?$' --output-on-failure
cmake --build cmake-build-metal4-air --target luisa-metal4-ios-path-tracer-aot -j 8
ctest --test-dir cmake-build-metal4-air -R '^test_metal4_ios_air_aot_codegen$' --output-on-failure
cmake --build cmake-build-metal4-air --target test_texture_compress -j 8
cmake-build-metal4-air/bin/test_texture_compress metal4
LUISA_ENABLE_VALIDATION=1 MTL_DEBUG_LAYER=1 \
  ctest --test-dir cmake-build-metal4-air -L integration_metal4 --output-on-failure -j 1
~~~

For physical-device iOS validation, configure the project with the Xcode
generator, `iphoneos`, LLVM 21 iOS static libraries, Metal4 enabled, and the
signing team, then build the
`luisa-metal4-ios-device-air-path-tracer` application target. Verify the final
link contains `libluisa-backend-metal4.a` and the static create/destroy symbols,
then sign, install, and launch it. Retrieve both PNG and JSON. A successful
build or launch alone is insufficient: require metadata identifying the
on-device XIR/LLVM/AIR plus `DeviceInterface` path, successful MTL4 library and
pipeline creation, completed readback, a stable pixel hash, and a visually
nondegenerate render. Keep the host-AOT SHA comparison as a distinct regression
rather than substituting it for the runtime-linked device run.

Use the `unit_xir` CTest label when a change can affect more than one XIR pass.
For another Metal4 runtime test, invoke the binary with the `metal4` backend;
no AIR-selection environment is required. Add `LUISA_DUMP_XIR=1` or
`LUISA_DUMP_LLVM_IR=1` only when the textual boundary is needed for diagnosis.

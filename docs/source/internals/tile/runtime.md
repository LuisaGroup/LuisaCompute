# Independent Tile kernels on Luisa Runtime

The first native path lowers actual typed TileIR to MPP inside the Metal
backend and returns an ordinary shader handle. It does not substitute a
handwritten GEMM by kernel name, call MPS, or pass through TVM. Tile kernels
are independent kernels; there is no intra-kernel SIMT/Tile DSL mixing.

## Boundary and usage

`tile::compile` is an opt-in frontend adapter in `luisa/tile/runtime.h`.
It calls the optional `DeviceInterface::create_tile_kernel` factory. The
backend owns lowering and binary compilation; the returned `tile::Shader`
owns the normal Runtime shader resource and encodes `ShaderDispatchCommand`.
Runtime core only forward-declares the Tile types, and TileIR core still
depends only on Core. Runtime and TileIR have no TVM dependency. When the
optional TIRx bridge is enabled, Metal privately links it as a second compiler
route; without that build option, requesting TIRx fails explicitly.

```{figure} ../../../_static/tile/xir-planning-pipeline.svg
:alt: TileIR reaches the SIMD backend through XIR, while native Metal MPP and TIRx remain independent compiler routes behind the same Runtime factory.
:width: 100%

The bridge owns representation, the backend owns machine compilation, and
Runtime owns dispatch and resource lifetime.
```

```cpp
auto kernel = definition.capture(tensor_shape(M, K),
                                 tensor_shape(K, N),
                                 tensor_shape(M, N));
auto shader = tile::compile(device, kernel);
// Check shader; metadata().error explains unsupported programs/targets.
stream << a.copy_from(host_a.data())
       << b.copy_from(host_b.data())
       << shader(a, b, c).dispatch()
       << c.copy_to(host_c.data())
       << synchronize();
```

Launch geometry comes from the captured execution hierarchy, not from Buffer
sizes. `CompileOptions::threads_per_group == 0` selects a legal default;
nonzero is an exact constraint checked against explicit child domains and
the compiled pipeline. Select the compiler at this boundary, not in DSL math
operations:

```cpp
auto native = tile::compile(device, kernel);
auto tirx = tile::compile(device, kernel, {.lowering = tile::Lowering::TIRX});
// Both own ordinary Runtime shader handles, with fast math off by default.
```

For explicit bridge tuning, include `luisa/tile/bridge/tirx/compiler.h` and
pass a borrowed `bridge::tirx::CompileOptions` through `CompileOptions::tirx`.
Its lifetime need only cover the synchronous compile call. The backend owns
the physical target, noalias/range contract and capability checks. Conflicting
thread constraints are errors, not hints. The existing bridge `compile()` and
TVM-runtime tests/benchmarks remain intact.

## XIR bridge and SIMD CPU realization

The bridge belongs to `include/luisa/tile/bridge/xir` and
`src/tile/bridge/xir`, alongside `tirx`, not inside the SIMD backend.
`bridge::xir::lower()` returns an owned, verified XIR Module and its kernel,
dispatch extent and typed argument ABI. It directly builds mutable XIR
instructions, def-use chains, basic blocks and PHIs. It neither emits a
serialization format nor reconstructs an AST. The bridge links TileIR and
XIR; it does not depend on SIMD, LLVM, TVM or Runtime.

`SIMDDevice::create_tile_kernel()` privately consumes that result, calls the
existing XIR-to-Schedule-to-LLVM/ORC compiler, and creates an ordinary
`SIMDShader`. Buffers, view offsets, worker pools, command lists and resource
lifetimes use the existing Luisa Runtime. `Lowering::NATIVE` selects this
route on a `simd` device. Requesting TIRx on that factory currently fails
explicitly; the standalone TIRx CPU compiler/runtime remains the independent
comparison route. This first integration is CMake-only, like the existing
SIMD backend and XIR targets; it does not add incomplete XMake substitutes.

The backend first calls `bridge::xir::plan()` to choose root axis order and
workers per block. It then passes the exact selected map to the lowerer.
The finite exhaustive solver and its uncalibrated cost breakdown are described
in [XIR execution planning](xir.md). Fixed constraints can be
supplied through `CompileOptions::xir`; conflicting constraints fail explicitly.
This is a bounded first mapping space, not general Tile partitioning.

The resulting mapping remains inspectable:

```text
Root parallel domain D
  logical worker j = packet * W + lane
  execution coordinate = unflatten_D,selected_order(j)
    nested parallel / serial / reduce / pipeline: ordered local loops
      Tile element e: one scalar XIR SSA value per worker
      buffer address = flatten_buffer(origin(execution coordinate) + e)

SIMD Schedule packs W independent workers into native vector operations.
It does not equate a Tile's memory dimensions with hardware lanes.
```

This realizes compact row-major, statically shaped buffer arguments, with
dimension-identity projection/broadcast, tile elementwise operations,
tile maps/extracts, reductions and MMA expanded into ordered multiply/add
operations. Loop-carried Tiles become per-element PHIs, including simultaneous
updates and zero-trip initial values. A load creates its SSA snapshot at the
source operation, before later writes; guarded reads/stores preserve bounds
and fill behavior. Pipeline stage cuts retain source order in a synchronous
CPU execution; a requested pipeline window is not a claim of overlapping
physical execution or of achieving a cycle-level initiation interval.

There must be one nonempty root `parallel`, without escaping state. Explicit
`worker`/automatic bindings are accepted; cooperative/group/subgroup bindings,
manual Memory, unsupported types/operations, arbitrary external layouts and
multi-launch programs are rejected. Static Tile expansion has a checked
budget rather than unbounded compile-time growth. This is a first realization,
not a complete layout planner, a packed GEMM microkernel, or an AMX lowering.

`test_tile_xir` checks verification, ABI, repeat lowering and fail-closed
constraints. `test_tile_xir_runtime` exercises elementwise/broadcast,
reductions, softmax, transposed/ragged GEMM, nonzero accumulators, loop carries,
snapshot effects, buffer-view offsets, guards and shader moves. When TIRx is
enabled, it checks the same captured GEMM TileIR through both routes against
an FP64 oracle. `benchmark_tile_xir` separately records XIR/SIMD timings,
source identity and realization diagnostics; these must not be labeled as
TIRx performance. `metadata.source` is the pre-optimization LLVM IR, not final
assembly; `LUISA_SIMD_DUMP_ASSEMBLY_DIR` captures the native artifact for audit.

`test_tile_xir_llm` additionally covers RMSNorm, LayerNorm, SwiGLU,
GELU+residual, RoPE, masked softmax and online prefill/decode/GQA against an
independent FP64 oracle, through both bridges when TIRx is enabled. The
[evidence report](../../performance/tile/index.md) records actual validation status.
When both compiler stacks load LLVM into one process, configure them against
the same LLVM major version; the tested setup uses LLVM 21.1.8 for both.

## TIRx device artifact and Runtime ABI

`bridge::tirx::compile_device()` shares execution mapping and device passes
with `compile()`, but extracts a `DeviceArtifact` instead of generating an
LLVM packed host wrapper. The artifact retains a transformable `PrimFunc`,
unchanged Metal source, entry symbol, static grid/block extents and a binding
map from device slots to original kernel parameters. These are obtained from
typed TIRx nodes and builtin identities, not source parsing or operation names.

TVMx may sort device arguments and remove unused ones. Flattening also adds
pure buffer declarations. The exporter tracks those aliases back to host
parameter identity. Only one unconditional launch is accepted: host loops,
conditionals, effects, host allocations, multiple launches, scalar/pointer-
offset arguments and dynamic launch resources fail closed. Storage ABI passes
run on the device partition after extraction, leaving the typed host signature
available for analysis.

Metal's direct-buffer shader binding uses those slot indices and the actual
`BufferView` byte offsets. It uses the same command, resource usage/hazard
tracking, queue, synchronization and shader lifetime as native MPP. It does
not rewrite the generated MSL into Luisa's argument-buffer ABI. Initially the
Runtime adapter accepts static nonempty FP32 buffers with signed-int32 element
addressing; the standalone bridge remains more general.

The direct-buffer compiler honors `ShaderOption::enable_fast_math`, checks the
compiled pipeline's resource limits, and uses an ABI/entry/shape/options-keyed
in-memory cache. Its disk/AOT archive format and indirect launches are not yet
implemented. Native MPP continues to use the existing ordinary shader archive.

### CPU provider realizations through TIRx

The standalone C++ TIRx bridge keeps portable LLVM lowering as the default and
adds two explicit target-realization switches:

| Option | Current proved input | Emitted target operation |
|---|---|---|
| `CpuMatrixBackend::CBLAS` | whole compact rank-two FP32 `C=A*B`, static positive extents, three noalias buffers | one registered `tvm.contrib.cblas.matmul` packed call |
| `CpuMathBackend::ACCELERATE` | compact shared FP32 exp map and contiguous rank-one FP32 add/max/min recurrences | synchronous vForce/vDSP calls |

These switches do not add frontend operations or infer an execution hierarchy
from memory. The shared exporter first derives versioned semantic contracts
from typed TileIR structure. The CPU pass then rechecks the actual TIRx body,
buffer ABI, layouts and caller alias contract. External symbol spelling is
used only at the provider ABI. If CBLAS was explicitly requested and the
whole-kernel contract or registered TVM provider is absent, compilation fails.
An unmatched local Accelerate pattern retains its ordinary TIRx body; a build
or target without Accelerate rejects the explicit policy.

~~~text
TileIR capture -> structural TIRx
  |
  +-- whole-GEMM CBLAS policy -> prove ABI -> one packed provider call
  |
  +-- ordinary function
        -> optional Accelerate exp/reduction realization
        -> read-only view + pipeline planning
        -> CPU root task/SIMD mapping
        -> TVM C++ passes -> LLVM JIT module
~~~

Structural export preserves every pure multi-consumer Tile SSA definition by
default. A target planner may later retain it, materialize it or choose the
explicit `EXPENSIVE_ONLY` recomputation candidate. This lets softmax feed both
its reduction and normalization from one exp result instead of duplicating
scalar/vector transcendental work. Only a revalidated shared FP32 exp
expression can become the vForce provider atom; a generic materialization
annotation is insufficient. The vForce call uses a compiler-owned compact
input/output pair; vDSP reductions consume a contiguous last dimension. The
exported wrappers are synchronous and do not retain or alias their array
pointers, so eligible scratch remains compatible with the bounded stack
planner.

`CpuMathBackend::ACCELERATE` is a different explicit numerical policy. vDSP
may reassociate a reduction and vForce has documented denormal/exception
differences from scalar libm. It is never enabled implicitly by `parallel`, a
pipeline stage or a memory placement. The benchmark report records the policy,
static provider call sites, complete-output error and tolerances. See the
[implementation/evidence report](../../performance/tile/index.md) for the balanced A/B.

### Optional MPP realization inside TVM

TIRx retains its existing SIMD-group lowering as the default. An independent,
opt-in `bridge::tirx::CompileOptions::metal_mpp` path uses TVM's own Metal
code generator, not the native emitter above. It requires the versioned
{download}`C++ TVM extension <../../../../src/tile/bridge/tirx/patches/README.md>`;
unpatched TVM
continues to work normally and rejects an explicit MPP request. Neither path
uses Python to generate source or hides an MPS library call.

This realization keeps the verified group plan, shared operand snapshots,
bounds handling, pipeline versions and persistent accumulators. Each matched
subgroup rectangle becomes a typed memory-input MMA; only its destination
cooperative tensor is materialized. Contract v2 has two explicit operations:
overwriting `D = A * B` and accumulating `D = A * B + C`. The bridge selects
the first only for a proved positive-zero, reassociable, single-iteration
direct-output recurrence; it removes both the destination zero-fill and the C
input. Nonzero/negative-zero C, memory C, multiple iterations and observed or
escaped carry state retain the accumulating form. Unsupported rectangles and
incompatible fragment contracts fail explicitly. The planner now uses a
separate `metal_mpp_memory_v2` relative-work basis. It enumerates legal thread
widths and rectangular subgroup factorizations, rejects descriptor, fragment
and shared-capacity violations, and ranks subgroup critical-path work plus
whole-kernel subgroup waves. The default M1-class wave capacity and the other
coefficients are replaceable priors, not queried occupancy or calibrated time.
Correctness-checked recapture/JIT measurement remains the authoritative tuner.

MPP's opaque execution layout is kept separate from memory orientation:
column-major A/B are canonicalized to physical row-major views plus descriptor
transpose flags. Column-major C/D use public cooperative-tensor coordinates
composed with the memory stride. Merely permuting inline-tensor strides was
numerically incorrect in the tested toolchain. All sixteen A/B/C/D orientation
combinations are covered by direct intrinsic tests.

The same TIRx artifact can run through TVM or through the ordinary Metal
backend. A typed cooperative-tensor allocation sets
`DeviceArtifact::requires_metal4`; the backend validates capability, selects
MSL 4, and keys the language version in its direct-buffer compiler cache.
This does not enable the separate Metal4 backend or add a second Runtime.

`CompileOptions::forward_readonly_tile_loads` optionally replaces proved
immutable compiler snapshots with memory inputs before resource planning.
Forwarding iterates to a fixed point so anonymous-axis relabeling does not
leave a second materialized copy. Each round rechecks effects, bounds,
dominance and escapes; explicit manual memory stays materialized.
The optional bounded-K capability also forwards a proved common zero-padded
K suffix using a positive actual memory extent, while M/N stay in bounds.
It retains the same Runtime artifact and falls back to strict snapshot
planning when the atom proof is unavailable; see
[bounded K views](matrix.md#bounded-k-views-avoid-nominal-padding-storage).

Independent subgroup execution is a separate realization proof. When the
whole group has only private MPP state, immutable inputs and one partitioned
terminal store, the planner reports `independent_subgroups=true`. Removing
its compiler-owned group fences additionally requires the default-off
`PlannerOptions::elide_independent_subgroup_barriers` choice. Legal elision
is not guaranteed to improve performance; original TIRx, staged MPP and the
default forwarding path keep their existing fence policy.

### Native SIMD-group reductions inside TIRx

`PlannerOptions::metal_subgroup_reductions` selects a separate Metal
realization for structurally proved FP32 add/max/min row programs. It does not
call MPS, MPP or a provider library. The bridge maps the logical root program
to one, two, four or eight 32-lane SIMD groups, emits ordinary TIRx thread
bindings and `simd_sum`/`simd_max`/`simd_min` Metal intrinsics, and uses a
small shared partial array only when multiple groups cooperate.

This pass runs before the generic root mapper and after shared structural
export. It accepts automatic or explicit subgroup roots; an unrealizable
explicit subgroup request fails, while an automatic root can retain the
reference path. The compile option is also explicit permission for a
floating-point tree order. Target width, noalias arguments, reducer identity,
pure contribution, uniform control, effect placement and memory capacity are
rechecked before mapping.

Repeated compiler-owned Tile values can be compacted from logical full-row
storage to worker-private stripes. The transformation proves every flattened
load/store index equals the current distributed element coordinate; an
escaping, fixed or permuted access is not compacted. A separate audit rejects
the subgroup map if a distributed private Tile would then be read by a
different logical owner. Guarded gathers from immutable Tensor snapshots can
instead be forwarded as lazy direct input reads. Explicit manual Memory is not
inferred from this optimization and keeps its own placement/store contract.

All compact stripes for one logical row program additionally obey
`PlannerOptions::max_reduction_striped_scalars_per_worker` (64 by default).
This is a bounded compiler-state policy, not a physical register claim. An
over-budget execution width is rejected before source generation; another
legal width or the reference realization may still be selected.

The resulting `DeviceArtifact` uses the same direct-buffer ABI and ordinary
Metal Runtime path described above. The standalone TVM runtime is retained as
an independent comparison. Both routes consume scheduled TIRx; no source-level
warp role or new Runtime resource type is introduced. The exact mapping,
finite cost model, staged/JIT controls and measured evidence are documented in
[TIRx Metal reductions](reductions.md).

### CUDA device artifact and PTX ABI

The CUDA backend overrides the same optional `DeviceInterface::create_tile_kernel`
factory with a reference-realization route: `tile::bridge::tirx::lower` ->
`compile_device` -> CUDA C source artifact -> the existing standalone-NVRTC
pipeline -> PTX -> a direct static `cuLaunchKernel`. CMake builds that enable
the optional TIRx bridge define `LUISA_CUDA_TILE_TIRX` on the backend and link
`luisa-compute-tile-bridge-tirx`; xmake builds (which have no TIRx target yet)
return the explicit "without the optional TIRx bridge" diagnostic.

The generated `DeviceArtifact` carries `Format::CUDA_SOURCE` (target kind
`"cuda"`; the NVPTX target kind is accepted as `Format::PTX` and skips NVRTC).
As on Metal, the typed launch is never parsed from source: `entry`, static
`grid`/`block`, and `buffer_arguments` (device slot -> original host parameter)
come from the typed TIRx launch nodes. The CUDA factory validates the same
FP32, nonempty, static, int32-addressable buffer ABI, derives argument usages
from `VIEW_LOAD`/`VIEW_STORE`, and records `dispatch_size = grid * block`.

CUDA launches are direct and statically shaped. A dedicated `CUDAShaderTile`
reorders `ShaderDispatchCommand` arguments into `buffer_arguments` order and
encodes each `BufferView` as a raw `CUdeviceptr` (base + offset), matching the
plain typed-pointer kernel parameters emitted by TVM. There is no trailing
`uint4` launch-size parameter and no `kernel_launcher`/cudadevrt indirect path.
The block must be warp aligned and at most the device `MAX_THREADS_PER_BLOCK`;
the factory also honors an exact `tile::CompileOptions::threads_per_group` by
checking it against the generated artifact block.

  PTX caching mirrors the ordinary CUDA shader path: cache identity is
  `hash(source, entry, block, grid, fast-math, ABI-marker "cuda-tile-direct-buffers-v1")`,
  stored in-memory in `CUDACompiler`'s LRU and on disk as
  `kernel_<hash>.tile.ptx` plus a `// METADATA: ...` sidecar written through the
  same `write_shader_cache` / `write_shader_bytecode` conventions. A stable
  `ShaderOption::name` behaves as a persistent runtime cache key today
  (compile-only offline archives remain gated, matching Metal's direct-buffer
  limitation). Old-driver PTX versions are patched with
  `CUDAShader::_patch_ptx_version`; other module load failures retry once with a
  `compute_60` recompile, mirroring builtin kernels.
  The PTX version transform is a dependency-light pure string/byte edit in
  `src/backends/cuda/cuda_ptx_version.h`
  (`patch_cuda_ptx_version` / `patch_cuda_ptx_version_bytes`), host-unit-tested
  without a CUDA driver. After a *successful* patch the factory persists the
  patched PTX plus its sidecar back to the same user/disk cache path when the
  bytes actually changed, so a cold process loads the compatible image instead
  of re-patching the stale cache entry on every launch.
  CUDA blocks must be warp aligned. The reference mapper therefore rounds a
  partial worker/elementwise domain *up* to the 32-thread warp (padding lanes
  are discarded by the existing `linear < extent` guard) and rounds an unaligned
  per-block cap *down*; a user-supplied `tile::CompileOptions::threads_per_group`
  must itself be a multiple of 32 and is rejected before codegen otherwise. The
  NVPTX (`Format::PTX`) route emits final PTX text that this backend cannot
  recompile for another architecture: when such an artifact fails to load for a
  non-version reason, the error names the driver error, quotes the requested
  `.target` directive when present, and recommends compiling through the
  `"cuda"` (CUDA C source) TIRx target for the device instead.

**Review scope.** CUDA tile kernels never use cooperative-vector /
cooperative-matrix / tensor-core / `mma.sync` / WMMA atoms, and they never use
indirect dispatch. Semantic Tile `MMA`/matmul, `REDUCE`, elementwise lift and
tile map/extract are all realized through the target-neutral reference
expansion (ordered multiply/add contraction, ordinary left folds and per-element
SIMT loops) and compiled to PTX. Explicit cooperative-matrix, MPP, subgroup
reduction, program-order planner and exact planner-thread requests on CUDA are
hard errors with descriptive diagnostics, never silently downgraded options.
  Host and runtime tests cover elementwise, add/max/min reductions, softmax and
  GEMM kernels (including ragged/batched edges, transposed operands and ordered
  `allow_reassociation`), padded `BufferView`-offset guard checks, bounded
  attention/row fixtures, cache round-trip and simulated old-driver retry
  against FP64 oracles, and assert the artifact/PTX contains no Metal
  cooperative scope. The cache and retry suites are TIRX/CUDA-gated; the
  simulated old-driver path is enabled by the test-only
  `LUISA_CUDA_TILE_FORCE_UNSUPPORTED_PTX=1` environment knob.

## First native realization, not a complete Machine TileIR

`src/backends/metal/tile/metal_tile_codegen.cpp` reads Candidate TileIR and
builds backend-local typed MMA realization records retaining the original
load/MMA/store identities. It does **not** mutate the input, manufacture a
Machine-form label, or implement a generic Machine TileIR transformation
framework. General scheduled/machine forms, atom catalogs, solver search and
cross-backend Machine TileIR remain follow-up work. The TIRx bridge's bounded
MPP solver and cost basis are target-local today; the independent native
emitter does not yet consume them.

The implemented subset is deliberately explicit:

- Dense static rank-two FP32 arguments; nonnegative signed-int32 element
  addressing. Inputs may be transposed through their logical dimensions.
- One root `parallel` mapped to groups, optionally with an explicitly bound
  subgroup child. The child's domain is independent subgroup work, not an
  allocation hierarchy. Group coordinates and subgroup coordinates are
  derived from each domain's mixed-radix index map.
- Single-use argument Tile loads feeding a same-scope MMA with zero initial
  accumulator, permitted reassociation, and one explicit output store. The
  complete K dimension is passed to the MPP atom as dynamic K; native K
  pipelines are not yet supported. M/N tiles meet the MPP atom constraints.
- One subgroup or the entire group participates in each atom. MPP owns the
  physical cooperative-fragment layout; the logical Tile shape does not
  claim an explicit lane/register layout for that opaque representation.
- Static M/N slices only for interior tiles; bounded dynamic slices preserve
  zero padding and clipped output stores at edges. K tails are dynamic.
- Compiler fast math defaults off, MPP relaxed precision is off, and ordered
  MMA is rejected. Unsupported dtype, manual Memory, pipeline, worker-level
  computation, nonzero initial accumulators or tile epilogues fail with a
  diagnostic, never with a silent fallback or erased effect.

Directly forwarding readonly views is safe only under the current disjoint
writable-argument contract. The compiler rejects an argument that is both
read and written. Invocation checks buffer element type, minimum footprint
and overlapping bound ranges whenever either argument writes. Disjoint
views of one Buffer are allowed. Buffer arguments also check Device identity;
existing BufferView does not carry a Device pointer, so cross-device views
remain the caller's responsibility, as in the existing Runtime ABI. External
aliases through distinct native resource objects must also obey the contract.

The Metal factory requires a supported Apple GPU and macOS/iOS 26, compiles
MSL 4 through the existing compiler/cache, checks the actual pipeline's
thread/resource limits, and creates an ordinary `MetalShader`. Language
version participates in the cache key without changing existing MSL 3 keys.
No alternate queue, private allocator or separate dispatch runtime is added.
Compile-only/AOT Tile loading and arbitrary indirect launch shapes are not
exposed by this first adapter.

## Regression and performance contracts

`test_tile_native_codegen` validates typed selection, execution mapping,
capacity constraints and fail-closed cases without a GPU. The runtime test
checks full FP64 references, changed inputs, transposes, ragged extents,
nonzero BufferView offsets, guard regions, disjoint subviews of one Buffer,
and shader move/destruction through the normal validation layer. With TIRx
enabled it also exercises the same Runtime factory with parameter permutation,
unused argument elimination, exact original argument usages and rejection of
multi-launch programs.

When TIRx is enabled, supported group cases lower the **same captured TileIR**
through both routes and check both complete outputs. The full-K 64×64 fixture
also preserves a current bridge limitation: materialized Tile storage exceeds
its shared-memory budget, while native readonly-view forwarding succeeds.
Explicit subgroup cases are not flattened to pretend TIRx supports that path.
All existing CPU and Metal TIRx tests remain registered.

CPU-specific regressions also inspect generated LLVM. They require one shared
TIRx `exp` operation after structural export, exact provider symbols/counts,
correct automatic-root launch decisions, and full numerical output for a
ragged three-row exp-plus-reduction kernel. The automatic SIMD suite exercises
proved full packs and unchanged slow tails, including lane-dependent statement
stores. An explicit Accelerate request on Metal must fail with a target
diagnostic.

Benchmark protocols and results have a separate owner. The
[route report](../../performance/tile/results.md) retains CPU provider, native MPP,
TIRx/MPP, MPS and Torch comparisons; the [validation record](../../performance/tile/validation.md)
records executed test checkpoints. The
{download}`benchmark guide <../../../../scripts/benchmark/tile_torch/README.md>`
owns command-line variants, balanced replay orders, artifact hashes and timing
modes. Keep GPU control, instrumented compute-pass and host-wall measurements
separate, and never infer direct XIR performance from a TIRx provider result.

# Optional TVMx Metal MPP extension

`metal-mpp-memory-v2.patch` is an experimental, native C++ TVM codegen extension,
against Apache TVM commit `c7b458e946bc4266915da582457476bdcd9705ae`. Its pinned
tvm-ffi submodule is `12dbf053b3d9ba4ebd9da3123b1aeca79cf74229`. This is not a
Python source bridge and does not call Luisa's native MPP emitter.

Normal TIRx SIMD-group lowering still works with an unpatched TVM installation.
Opting into `bridge::tirx::CompileOptions::metal_mpp` requires the native registry
capability `target.metal.mpp_memory_contract_version() == 2`; an absent or
incompatible capability is a compilation error. CMake does not fetch, patch,
or replace TVM automatically. Keep the patched compiler in a separate build.

The extension initially supports FP32, one complete 32-lane SIMD group per MPP
operation, static 8-multiple M/N/K, positive constant leading strides, and
global/shared A/B views. At least one of local M/N must be a multiple of 16;
an 8×8 subgroup rectangle is rejected. K is static in the IR but represented
by MPP's dynamic-K descriptor, including K=8 and K=24. Only destination
cooperative tensors are materialized.
Their statically indexed collections become separate opaque MSL variables,
because cooperative tensors cannot be C++ array elements. Shape/role/stride and
fragment-index mismatches fail closed. Existing owning A/B cooperative-tensor
operations are not implemented by this patch.

The two additional intrinsics read their memory inputs **at the MMA call**.
`cooperative_tensor_multiply_accumulate_from_memory` implements
`D = A * B + C`, including distinct C/D fragments and nonzero C;
`cooperative_tensor_multiply_from_memory` implements overwriting
`D = A * B` and does not read the previous D. Their descriptors explicitly
select `multiply_accumulate` and `multiply`, respectively, with relaxed
precision disabled. A cooperative-tensor allocation cannot mix the two modes;
inconsistent contracts fail closed.

The Tile bridge emits the overwriting form only after the existing MMA matcher
has established relaxed/reassociable arithmetic and either (a) a standalone
MMA has a literal positive-zero initializer, or (b) a closed direct-output
accumulator recurrence has exactly one iteration and a literal positive-zero
initializer. It therefore removes both the zero-fill loop and the C input from
canonical full-K GEMM. Negative zero, nonzero/literal or memory C, multiple K
iterations, escaped/observed carry state, and unproved direct output retain
multiply-accumulate. The caller still supplies uniform subgroup participation,
valid memory ranges, and synchronization; the bridge derives these from group,
ownership, bounds, and recurrence proofs. Dynamic fragment indices, partial
subgroups, and arbitrary pointer/layout expressions are not supported.

Column-major A/B become physical row-major tensor views with transpose flags
in the MPP descriptor. The validated SDK did not correctly implement the same
operation by simply exchanging inline-tensor strides. Column-major C/D
transfers instead compose the cooperative tensor's public per-lane coordinates
with the memory leading stride. Row-major C/D keep MPP's bulk load/store.
Thus memory orientation does not redefine the execution distribution. Each
materialized fragment currently requires one consistent M/N/K and A/B layout
contract across its MMA uses; incompatible uses are rejected.

The MPP option searches the existing exact rectangular distribution family with
`MatrixCostBasis::METAL_MPP_MEMORY`: MPP-specific relative-work features and
backend-overridable coefficients. `GroupPlan::cost` is not a time prediction
or measured register usage, and the earlier small-shape calibration does not
establish large-shape accuracy. Late worker-private prefetching is disabled for this realization.
Ordered/rejected MMA patterns still retain their semantic reference expansion.
An otherwise matched MMA with a nonrectangular plan is rejected, not silently
passed off as an MPP kernel.

`CompileOptions::forward_readonly_tile_loads` is a separate, default-off option
also used by non-MPP paths. Before pipeline/resource planning, the bridge can forward
compiler-local snapshots to immutable parameter views. It requires the caller's
`noalias` contract, audits the entire function for writes/escapes/unknown effects,
matches a complete independent-element copy, proves its guard and indices with
native TIRx simplification, and checks all consumers are dominated and in bounds.
Proof inputs reject memory-derived indices and call expressions: immutable
contents alone do not establish immutable addresses. Unproved tails, mutable
inputs, aliases and manual memory retain their snapshots.
Manual memory is marked even without a resource constraint. The matrix matcher
accepts only explicitly authorized input identities; `global` alone grants no
alias permission. Writable accumulators still require owned storage.

This makes a full-K memory-input MPP candidate possible without allocating full-K
A/B shared tiles. It does not establish profitability or replace the MPP cost
policy. The patch also fixes a TVM `PointerValueTypeRewrite` bug exposed by
these large global strides: an address modular coefficient must not overflow the
signed 16-bit lane encoding. Unrepresentable vectorization hints conservatively
remain scalar, rather than turning into accidental scalable-vector types.

## Build in isolation

### Optional bounded-K extension

Apply `metal-mpp-bounded-k-v1.patch` **after** the v2 patch. It retains the
original v2 ABI and separately advertises
`target.metal.mpp_bounded_k_contract_version() == 1`. An installation without
this capability keeps the strict, fully-in-bounds forwarding behavior.

The memory multiply and multiply-accumulate calls may append one signed
scalar integer `actual_k` (12 or 14 arguments respectively). Their nominal
static M/N/K and destination layout contracts do not change. The caller
must prove `0 < actual_k <= K`, uniform participation, immutable inputs, and
valid full M×actual_k / actual_k×N memory rectangles at the supplied pointers.
Strides must fit those **actual** rectangles. Dynamic values are caller
preconditions, not runtime assertions; malformed literal extents, scalar
types and known leading strides are rejected before launching.

The code generator builds inline tensors with a dynamic K extent while M/N
remain static. Transposes still belong to the descriptor. No extra buffer,
CPU fallback, padding copy, precision reduction or Python source is involved.
The bridge only omits a common zero×zero suffix: it proves canonical zero
padding, equal actual A/B K lengths, nonnegative origins, positive lengths, unit-stride
logical projections and fully in-bounds M/N under enclosing execution domains.
It commits guarded forwarding only if every reassociable MMA still receives
a verified atom plan; otherwise it retries the unchanged strict path. M/N
tails, extra masks, nonzero fills and mismatched reduction intervals therefore
retain snapshots. Accumulator lifetime, overwrite/accumulate mode, stage
ordering and synchronization continue to use the existing proofs.

This removes the nominal A/B shared allocation from eligible K-tail programs.
The cost model still charges nominal K work conservatively; it is not a new
calibration or a claim that the selected schedule beats MPS.

### Commands

Use a clean checkout at the pinned commit; initialize its `3rdparty/tvm-ffi`
submodule recursively. Set these paths to separate, task-owned directories:

```sh
TVM_SRC=/path/to/tvm
TVM_BUILD=/path/to/tvm-mpp-build
LUISA_SRC=/path/to/LuisaCompute
LUISA_BUILD=/path/to/luisa-mpp-build

git -C "$TVM_SRC" apply --check "$LUISA_SRC/src/tile/bridge/tirx/patches/metal-mpp-memory-v2.patch"
git -C "$TVM_SRC" apply "$LUISA_SRC/src/tile/bridge/tirx/patches/metal-mpp-memory-v2.patch"
# Optional: enable proved zero-padded K suffixes without nominal shared tiles.
git -C "$TVM_SRC" apply --check "$LUISA_SRC/src/tile/bridge/tirx/patches/metal-mpp-bounded-k-v1.patch"
git -C "$TVM_SRC" apply "$LUISA_SRC/src/tile/bridge/tirx/patches/metal-mpp-bounded-k-v1.patch"

cmake -S "$TVM_SRC" -B "$TVM_BUILD" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DUSE_LLVM=/opt/homebrew/opt/llvm@21/bin/llvm-config \
  -DUSE_METAL=ON -DUSE_MLIR=OFF -DUSE_RPC=OFF -DUSE_Z3=OFF -DUSE_GTEST=OFF
cmake --build "$TVM_BUILD" --parallel 8

cmake -S "$LUISA_SRC" -B "$LUISA_BUILD" -G Ninja \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
  -DLUISA_COMPUTE_ENABLE_METAL=ON -DLUISA_COMPUTE_ENABLE_METAL4=OFF \
  -DLUISA_COMPUTE_ENABLE_CUDA=OFF -DLUISA_COMPUTE_ENABLE_VULKAN=OFF \
  -DLUISA_COMPUTE_ENABLE_HIP=OFF -DLUISA_COMPUTE_ENABLE_FALLBACK=OFF \
  -DLUISA_COMPUTE_ENABLE_GUI=OFF -DLUISA_COMPUTE_ENABLE_SIMD=OFF \
  -DLUISA_COMPUTE_ENABLE_UNITY_BUILD=OFF -DLUISA_COMPUTE_DOWNLOAD_OIDN=OFF \
  -DLUISA_COMPUTE_BUILD_TESTS=ON -DLUISA_COMPUTE_ENABLE_TILE_TIRX_BRIDGE=ON \
  -DLUISA_COMPUTE_TVM_INCLUDE_DIR="$TVM_SRC/include" \
  -DLUISA_COMPUTE_TVM_LIBRARY_DIR="$TVM_BUILD/lib" \
  -DLUISA_COMPUTE_TVM_FFI_INCLUDE_DIR="$TVM_SRC/3rdparty/tvm-ffi/include" \
  -DLUISA_COMPUTE_TVM_FFI_LIBRARY_DIR="$TVM_BUILD/lib"
cmake --build "$LUISA_BUILD" --parallel 8
```

This configuration requires the Metal 4 SDK/toolchain and macOS/iOS 26 at
runtime for MPP. It does not enable Luisa's separate Metal4 backend. Both the
existing TVM Runtime and Luisa's ordinary Metal Runtime can launch the resulting
TIRx MPP code. `DeviceArtifact::requires_metal4` comes from typed allocations;
the backend validates OS/device capability and includes language version in
the compiler cache key. No shader-source rewriting is involved.

## Validation and comparison

After a **full build**, run `test_tile_tirx_matrix_metal` and
`test_tile_native_runtime`. The former covers nonzero C, literal fills,
positive-zero one-shot multiply selection and multi-iteration fallback,
transposes, pipeline versions, ragged tiles, multiple K iterations, direct
global views, all sixteen A/B/C/D layout combinations, distinct C/D fragments,
large full-K strides, manual memory with/without placement, mutable inputs and
address indices, address escape, and malformed intrinsic rejection. The
latter covers Luisa buffer offsets, guards, resource ownership, and aliases.
The unpatched build checks that the explicit MPP request is rejected.

`benchmark_tile_tirx` accepts `mpp` in its matrix-mode argument. It records
separate static SIMD-group/MPP call counts and the planner's cost basis. Use
`compare_lowerings.py --tirx-mpp` to add this path beside native MPP, independent
TIRx SIMD-group, handwritten MPP, direct MPS, and Torch. Pass each externally
linked TVM compiler/runtime/FFI library through `--compiler-artifact` so it is
fingerprinted before and after timing. Six paths require twelve rounds for
balanced ordering. Full output validation precedes accepting every timing row.

The v2 balanced replay covers eight square, rectangular, and ragged FP32 GEMMs,
seven independent implementations, fourteen position-balanced rounds, and
784/784 complete FP64-oracle-valid outputs. With the previously frozen plan,
1024³ TIRx MPP views improved from 291.736 us under v1 to 273.996 us under v2,
versus Torch 284.805 us, MPS 272.694 us, and handwritten MPP 266.476 us in the
same v2 replay. This closes the Torch gap and leaves a 0.40% paired MPS gap for
that frozen 64x64 plan. A separately reported 128x32 exploratory plan reached
270.66 us with conservative barriers and about 264.5 us with proved fence
elision, demonstrating that remaining schedule selection belongs in the MPP
cost model/solver rather than a shape-specific lowering rule. See the
[complete v2 table](../../../../../scripts/benchmark/tile_torch/results/m1-max-20260905-mpp-multiply-v2/results.md)
and its raw artifact-fingerprinted evidence. These are synchronized host-wall
batched times, not GPU-event times or a universal performance claim.

`benchmark_tile_tirx` also accepts `mpp-views`. Search this family with
`run.py --matrix-realization mpp-views --cooperative-matrix --execution-scope group
--backends metal --operations gemm` and the existing explicit JIT candidate lists.
Then supply its frozen report to `compare_lowerings.py --tirx-mpp
--tirx-view-plan path/to/results.json`. This adds a seventh path without replacing
either TIRx control; fourteen rounds balance the comparison. Its geometry may
differ, so do not call it a same-schedule ablation.

The [recorded seven-path replay](../../../../../scripts/benchmark/tile_torch/results/m1-max-20260904-tirx-views/notes.md)
keeps all 784 valid complete outputs, rejected search candidates, frozen
schedules and artifact fingerprints, including the remaining MPS gap.

Keep native and TIRx schedules independently recorded. Do not infer a speedup
from compilation success, compare GPU-event time with host-wall time, or call
an analytic MPP cost a measured time.

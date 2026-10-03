# CUDA Workflow-B: TVMx + CUDA build and first-run checklist

Workflow-B is the environment that can actually **compile and run** the CUDA
TIRx-gated paths: the `#ifdef LUISA_CUDA_TILE_TIRX` factory body in
`src/backends/cuda/tile/cuda_tile.cpp`, the bridge CUDA/NVPTX device-artifact
route in `src/tile/bridge/tirx`, and the two TIRX-gated CUDA suites. Ordinary
development machines without the pinned TVMx tree only build the fail-closed
non-TIRX variant; that is expected and is not a signal that the CUDA Tile work is
stubbed.

The TVMx steps below are a build recipe. The independent native Tile IR section
records its own validation checkpoint. Update this page if pinned commits or
commands change.

## 1. TVMx checkout/build (CUDA codegen)

```bash
TVM_SRC=/path/to/tvm-src
TVM_BUILD=/path/to/tvm-build
git -C "$TVM_SRC" checkout c7b458e946bc4266915da582457476bdcd9705ae
git -C "$TVM_SRC/3rdparty/tvm-ffi" fetch
git -C "$TVM_SRC/3rdparty/tvm-ffi" checkout 12dbf053b3d9ba4ebd9da3123b1aeca79cf74229
git -C "$TVM_SRC/3rdparty/tvm-ffi" submodule update --init --recursive

cmake -S "$TVM_SRC" -B "$TVM_BUILD" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DUSE_CUDA=ON \
  # optional: -DUSE_CUDA_ARCHITECTURES=<host CC>
  # Keep Metal OFF; no MPP patches are required for the CUDA reference route.
cmake --build "$TVM_BUILD"
```

Verify both runtime builders exist before touching LuisaCompute:

```bash
# target.build.cuda  -> InspectSource("cuda")  -> DeviceArtifact::CUDA_SOURCE
# target.build.nvptx -> InspectSource("ptx")   -> DeviceArtifact::Format::PTX
```

If the pinned tree lacks either builder, record the missing-builder failure that
`compile_device()` reports; do **not** patch silently.

## 2. LuisaCompute CMake configure

```bash
cmake -S /path/to/luisa -B /path/to/luisa-build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DLUISA_COMPUTE_ENABLE_CUDA=ON \
  -DLUISA_COMPUTE_ENABLE_TILE_TIRX_BRIDGE=ON \
  -DLUISA_COMPUTE_TVM_INCLUDE_DIR="$TVM_SRC/include" \
  -DLUISA_COMPUTE_TVM_LIBRARY_DIR="$TVM_BUILD/lib" \
  -DLUISA_COMPUTE_TVM_FFI_INCLUDE_DIR="$TVM_SRC/3rdparty/tvm-ffi/include" \
  -DLUISA_COMPUTE_TVM_FFI_LIBRARY_DIR="$TVM_BUILD/lib" \
  -DLUISA_COMPUTE_BUILD_TESTS=ON
cmake --build /path/to/luisa-build --target luisa-compute-backend-cuda test_tirx_device_cuda test_tile_cuda_ptx test_cuda_ptx_version
```

Expected wiring:
- `luisa-compute-backend-cuda` links `luisa-compute-tile-bridge-tirx` and is
  compiled with `LUISA_CUDA_TILE_TIRX=1`;
- `test_tile_cuda_ptx` links `luisa::tile-tirx` and defines
  `LUISA_TEST_TILE_CUDA_TIRX=1`;
- `test_tirx_device_cuda` links `luisa::tile-tirx`.

## 3. First-run triage list

The first compile-fix pass on Workflow-B should expect (at least) the risks
listed below. Every failure becomes a source fix in this phase's file set;
never remove or silence a test.

- TVM C++ API drift versus the pinned headers used by
  `src/tile/bridge/tirx/compiler.cpp` and `cuda_tile.cpp`
  (`max_num_threads`/`max_shared_memory_per_block` int64 attrs,
  `PrimFunc::GetAttr` returns, `InspectSource` lifetimes, IRModule function
  counts).
- The warp-aligned mapper change interacting with the CUDA gates.
- `compile_device` for `nvptx`: `InspectSource("ptx")` must be non-empty and
  NUL-safe for `cuModuleLoadData`.
- C++ errors in the `#ifdef LUISA_CUDA_TILE_TIRX` factory body that only
  surface with the define set.
- Host test build issues in `test_tirx_device_cuda.cpp`.

## 4. CTest invocations

```bash
ctest --test-dir /path/to/luisa-build -R 'test_cuda_ptx_version' --output-on-failure   # host, no GPU
ctest --test-dir /path/to/luisa-build -R 'test_tirx_device_cuda' --output-on-failure   # host artifacts
ctest --test-dir /path/to/luisa-build -R 'test_tile_cuda_ptx' --output-on-failure      # device oracle + cache/retry
```

`test_tile_cuda_ptx` passes `cuda` as the backend argument; it needs a healthy
CUDA device. The cache round-trip suite uses an in-memory `BinaryIO`
(`DeviceConfig::binary_io`), and the simulated old-driver retry sets
`LUISA_CUDA_TILE_FORCE_UNSUPPORTED_PTX=1` inside the test process for the first
compile and clears it for the cold-cache compile.

## 5. Environment caveats

- **No CUDA device**: the host suites (`test_cuda_ptx_version`,
  `test_tirx_device_cuda`) still run; `test_tile_cuda_ptx` cannot.
- **Old driver**: the patch-retry test also exercises the real
  unsupported-version path.
- xmake: enable the optional bridge with `lc_tile_tirx_bridge=y` plus the
  `lc_tvm_*` paths (same TVM options as the CMake configure). When it is off,
  xmake builds intentionally run the fail-closed variant; when it is on, the
  tile XIR bridge is always compiled into `lc-tile` and the CUDA backend/test
  define `LUISA_CUDA_TILE_TIRX` / `LUISA_TEST_TILE_CUDA_TIRX` exactly like the
  CMake TIRX bridge wiring.

## 6. Experimental native CUDA Tile IR

On a CUDA device, `Lowering::NATIVE` with the exact environment setting
`LUISA_CUDA_TILE_IR=1` selects a separate route:

```text
Luisa TileIR -> CUDA Tile C++ -> NVRTC Tile IR -> tileiras -> cubin
```

```cpp
auto shader = tile::compile(device, kernel,
    {.lowering = tile::Lowering::NATIVE}, {.enable_fast_math = false});
// Check shader; metadata().error reports unsupported programs or missing tools.
```

CMake builds the `luisa-cuda-tile-compiler` target, producing
`luisa_cuda_tile_compiler[.exe]`, when CUDA 13.4 or newer, `cuda_tile.h`, and
`tileiras` from the selected toolkit are available on Windows or Linux. Keep
the helper beside the runtime binaries. This route does not require TVMx;
`Lowering::TIRX` continues to use its independent PTX route regardless of the
native opt-in. xmake currently builds the native route as explicitly unavailable.

The initial scope is static, contiguous FP32 buffers of rank 1–3, one root
`parallel`, supported elementwise operations, ordered serial/pipeline loops,
and rank-two FP32 MMA. Logical Tile extents must be powers of two; buffer
extents may be ragged and use masked loads/stores. Bool and 32/64-bit integer
intermediate values are supported. FP32 MMA uses ascending-K, elementwise FMA
with ties-to-even rounding and preserved subnormals, without input narrowing.
Unsupported types, operations and explicit execution/layout constraints fail
with a diagnostic.

There is no cache on this experimental route: `enable_cache` is accepted as a
hint, but every compile invokes the tools again. Named archives, compile-only,
`native_include`, fast math, and nonzero `threads_per_group` or `max_registers`
are rejected. The generated CUDA Tile source and realization string remain
available in shader metadata.

With CUDA and tests enabled in the selected CMake build, use these PowerShell
commands after configuration:

```powershell
cmake --build build
$env:LUISA_CUDA_TILE_IR = "1"
./build/bin/test_tile_cuda_ir.exe cuda --require-native
Remove-Item Env:LUISA_CUDA_TILE_IR
./build/bin/test_tile_cuda_ir.exe cuda --expect-disabled
```

For a build without the helper, set the variable to `1` and use
`cuda --expect-unavailable` to check the capability diagnostic; that is not a
positive runtime test. On 2026-10-01, the native runtime suite passed all eight
cases and 53,096 assertions on Windows with CUDA 13.4 and an RTX 4060 Laptop GPU.
The suite covers full FP64 GEMM oracles, transposes and tails, ordered FMA,
BufferView offsets and guards, alias/snapshot behavior, negative origins,
special-value copies, simultaneous loop carries, and rejected options.

## 7. Opt-in integer warp extrema in CUDA TIRx

`LUISA_DIAGNOSTIC_TIRX_INTEGER_EXTREMA=1` changes only the CUDA subgroup
warp MIN/MAX helpers. Unset or exact `0` retains the original helper source;
other values fail compilation. Exact `1` requires CUDA subgroup reduction
planning. This is a private diagnostic switch, not a new Tile primitive,
public compile option, cost policy, or default selection. Metal and NVPTX
paths are unchanged.

The existing mapper requires unordered FP32 extrema initialized with
`+Inf` for MIN and `-Inf` for MAX. Its unchanged, non-FTZ `min.f32`/`max.f32`
local chain suppresses a single NaN, so both warp call sites receive
non-NaN values, including identity padding. For such values, flipping the
sign bit of nonnegative encodings and complementing negative encodings
gives an order-preserving unsigned key. Integer `__reduce_min_sync` or
`__reduce_max_sync` uses the existing full-warp mask, then the inverse
transform restores the exact FP32 bits. This preserves subnormals,
infinities and `-0 < +0`; all-NaN rows retain the original seeded identity.
It is not a general unseeded, NaN-preserving float collective. SUM, local
arithmetic order, shared partials, barriers and elementwise math policy
remain unchanged. SM80 and later use integer redux; lower targets retain
the original shuffle implementation. See the [PTX redux contract](https://docs.nvidia.com/cuda/archive/13.2.0/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-redux-sync).

The host regression `tile_tirx_cuda_subgroup_integer_extrema` checks exact
default/zero source equality, reverses only the two changed helpers to
recover the entire original source, checks launch/argument metadata, and
rejects invalid values or an absent subgroup capability. Existing GPU
special-value and partial-tree tests were also run with the switch enabled
on SM89: 2,143,669 assertions across the two filters passed. After adding
the host group, the full main build and the host suite passed, including
59 assertions in the new group; the project no-throw scan passed. A separate
strict/fast seeded helper probe checked 985,088 output words plus guards
for T32/T128/T1024, including signed zeros, subnormals, infinities,
quiet/signaling NaNs, random raw bits and identity tails. SM75 strict/fast
compile-only probes emitted no redux. Current exact-source-key PTX has two
unpredicated, full-mask MAX redux instructions before lane-zero publication
and after partial/identity reconvergence, respectively; the ten SUM
shuffles and two CTA barriers remain. This is cache inspection, not captured
timed-JIT machine code or SASS.

The 2026-10-04 closed paired cohort used FP16 softmax, fast elementwise math,
T128/P2/U16, graph batches of 100, seven samples targeting 100 ms and 500 ms
warmup. Each fixture kept its mapping fixed while toggling only the extrema
switch. The small case used L1 scalar storage; the large case used the
separate private L8 guarded storage/coordinate experiment, which this
change does not enable.

| Shape | Flag 0 initial, us | Flag 1, us | Flag 0 recheck, us | Fresh Torch, us | Flag 1 / recheck |
|---|---:|---:|---:|---:|---:|
| 32 x 512 | 1.129729 | 1.064617 | 1.124895 | 1.087499 | 0.946414 |
| 1024 x 512 | 3.385967 | 2.839070 | 3.377445 | 4.307133 | 0.840597 |

Lower is better. All six native and two fresh Torch processes passed:
56 samples, 600 actual native graph nodes, and ten saved logical outputs
(2,703,360 elements). Full saved logical outputs were independently checked
against the original FP64 bounds; physical guards/read-only allocation
checks remain runtime reports. All native entries reported 23 registers,
80 static shared bytes and zero local bytes. Control drift was -0.43% and
-0.25%; the candidate ranges did not overlap either control range in this
cohort. This does not establish cross-session stability or a thermal
explanation. Torch consumed the first control's exact fixture with fresh
per-case caches; its XBLOCK1/W4 selection records had null cubin hashes, so
no unique timed cubin identity is claimed.

After a complete build of a tree with the TIRx bridge and CUDA enabled,
these persisted tests can be run independently of the private benchmark:

```powershell
./build-msvc-llvm/bin/test_tirx_device_cuda.exe tile_tirx_cuda_subgroup_integer_extrema
$env:LUISA_DIAGNOSTIC_TIRX_INTEGER_EXTREMA = '1'
./build-msvc-llvm/bin/test_tile_cuda_ptx.exe cuda tile_cuda_ptx_subgroup_special_values
./build-msvc-llvm/bin/test_tile_cuda_ptx.exe cuda tile_cuda_ptx_subgroup_partial_tree_values
Remove-Item Env:LUISA_DIAGNOSTIC_TIRX_INTEGER_EXTREMA
```

The [56 retained samples](../../../../scripts/benchmark/tile_torch/results/2026-10-04-cuda-integer-extrema/samples.csv)
allow the table's medians and ratios to be recomputed without the private
benchmark. They do not reproduce GPU execution, full tensor validation,
source compilation or physical guard checks.

The retained local audit is
`.deps/oct04-tirx-integer-extrema-pairs-summary-v1/checkpoint.json`, SHA256
`143756b7aaa1b052ae32c6f3955d4c69613eb7b583889667aefb170f662ef922`.
It preserves both controls, all samples, source/helper identity, actual graph
bindings, resources and whole-stage telemetry. Execution-time source/DLL
hash checks are recorded by the passed frozen queue; after the source lock
was released, the independent replay checked the saved snapshot and frozen
validation helpers without pretending current mutable binaries were still
the measurement binaries. This local packet is not bundled by this source
change. The experiment remains opt-in.

## 8. Coordinate rematerialization before CUDA subgroup planning

`LUISA_DIAGNOSTIC_TIRX_FORWARD_COORDINATES=1` removes closed, pure integer
or Boolean coordinate Tiles after readonly snapshot selection. It substitutes
their expressions at exact, bounded axis projections and repeats the analysis
for dependent coordinate producers. It introduces no Tile DSL primitive or
execution scope. Unset or exact `0` retains the existing pipeline; other values,
a missing CUDA subgroup capability, or an explicit request with no eligible
coordinate Tile fail compilation. Other targets ignore this CUDA experiment.

The proof requires one complete producer, one allocation, dominated consumers,
and no escape. Memory reads, floating values, calls, opaque coordinates,
partial writes, and arbitrary gathers remain materialized. Input snapshots
are not revisited after the pass. Constant Boolean selection uses only the
consumer element domain, so a logical program bound cannot erase a physical
packed-tail guard. Floating expressions and their lazy branches remain intact.

The existing `test_tirx_device_cuda` host suite covers dependent producers,
true/false and unresolved guards, floating expression preservation, snapshot
ordering, and rejected producer/consumer patterns. The CUDA numerical sweeps
also cover FP16/BF16/FP32 softmax and normalization, odd row counts, and
independently offset input/output pointers. This pass only removes coordinate
state; actual vector loads and performance require separate backend evidence.

## 9. Guarded storage packs and reduction contributions

`LUISA_DIAGNOSTIC_TIRX_VECTOR_PACKS=1` is a private CUDA subgroup experiment
for lane widths 2, 4 or 8. It admits compact FP16/BF16/FP32 accesses with at
most 16 global bytes per lane, after proving stride, bounds and the alignment
of owned local storage. Caller pointers receive runtime alignment guards;
misaligned inputs retain the original scalar path. Thus FP32 width 8 is not
an eligible global pack. Default compilation and the original cost policy
remain unchanged.

Pure reduction contributions can be loaded into an owned FP32 pack before
the original ordered carry updates. Only complete worker chunks use this
form; the entire residual chunk and the scalar fallback retain their guards
and arithmetic order. Coordinate conditions are folded only when proved
under the same domain used for access admission and emission. Data-dependent
or floating conditions are not assumed, and BF16 rounding/NaN handling stays
intact. Each reduction's scratch allocation counts against its candidate's
remaining private-storage budget.

Proofs include retained enclosing loop bounds and use the actual worker
coordinate before mapping, including one-warp programs. This admits singleton
row Tiles and S1 programs packed 1, 2 or 4 per group without assuming inactive
programs are valid. After these fixes, the full MSVC build, 1196 host assertions
and 68 CUDA numerical cases passed. The CUDA cases cover all three storage
types, SUM/MAX, inactive packed rows, misaligned inputs and 32 admitted SUM
geometry/storage configurations. The host checks also retain a true scalar
residual chunk and reject unproved odd-pitch packing. These correctness runs
are not performance measurements.

The experimental planner prepares each candidate's actual body before
scoring and retains the winner without remapping it. The associated memory
facts describe per-instruction 32-byte sector requests for a full active
warp in an admitted phase. They are not DRAM traffic, ISA counts or a complete
kernel cost: scalar tails, predicates, scratch traffic and launch participation
can remain unknown. Unknown costs are never replaced with zero. These partial
facts do not yet choose a new policy; the existing callbacks and tie order are
preserved. Staged plans mark the old payload accounting incomplete.

The first closed SUM cohort used original fixtures, seven graph-event samples
per stage, batches of 100, and fresh per-case Torch caches. All 15 processes,
1200 actual native graph nodes and 18 saved logical outputs passed validation.
Times below are medians in microseconds; lower is better.

| SUM fixture | T/P/U | L1 initial | L4 scalar | L4 vector | Fresh Torch | L1 recheck |
|---|---|---:|---:|---:|---:|---:|
| FP16 129 x 2048, fast | 128/1/64 | 1.351240 | 1.393228 | 1.352020 | 1.296327 | 1.340821 |
| BF16 128 x 8192, strict | 128/2/64 | 2.718521 | 2.731349 | 2.450855 | 2.655414 | 2.588385 |
| FP32 3 x 8192, strict | 256/1/64 | 1.326218 | 1.367922 | 1.351403 | 1.353750 | 1.319911 |

Vector packs do not beat L1 uniformly. BF16 improved in this cohort, but its
control drift was -4.79%; FP16 and FP32 retain L1 as the stronger native
configuration. Whole-stage telemetry included power and thermal event flags,
without assigning a cause to individual samples. No automatic promotion or
cross-session stability is claimed. The [105 retained samples](../../../../scripts/benchmark/tile_torch/results/2026-10-04-cuda-sum-contribution-packs/samples.csv)
include both controls and the negative results. The local independent audit
is `.deps/oct04-tirx-sum-contribution-pairs-summary-v1/checkpoint.json`, SHA256
`d15128c8eb4fb35fb5cf17944013e28a2884d466948f69408999b61778399d6f`;
that full local packet is not bundled with the source.

## 10. Measured SUM geometry choices

A second cohort compared 29 candidates across the same three SUM fixtures.
It varied threads per group (`T`), programs per group (`P`), lane width (`L`)
and scalar/vector storage, retaining U64, BR1 capture, the original math
policy, inputs and FP64 bounds. The original L1 configuration ran before and
after the candidates; Torch used that first control's exact fixture and new
per-case caches. Each process retained seven 100 ms graph-event samples,
500 ms warmup and 100 nodes per graph, with the same four-core CPU affinity.
No model was fitted or production selection policy changed.

The table reports medians in microseconds. "Lowest candidate" is a post-hoc
minimum within this cohort, not a separately validated schedule choice.

| SUM fixture | Original T/P | L1 initial | Lowest candidate T/P/L | Candidate | L1 recheck | Fresh Torch | Candidate / recheck | Control drift |
|---|---|---:|---|---:|---:|---:|---:|---:|
| FP16 129 x 2048, fast | 128/1 | 1.354865 | 32/1/4 vector | 1.349508 | 1.348166 | 1.458718 | 1.000995 | -0.49% |
| BF16 128 x 8192, strict | 128/2 | 2.724686 | 64/1/4 vector | 2.421356 | 2.594069 | 2.664875 | 0.933420 | -4.79% |
| FP32 3 x 8192, strict | 256/1 | 1.329793 | 256/1/4 vector | 1.335910 | 1.311057 | 1.281319 | 1.018956 | -1.41% |

BF16's lowest candidate was 6.66% faster than the final control, with sample
ranges of 2.420887--2.431987 us and 2.592768--2.594816 us respectively. Its
11.13% improvement against the initial control includes a -4.79% control
drift, so the larger percentage must not be presented as a stable gain.
FP16 candidates all lost to the final L1 control; FP32 candidates all lost
to both L1 controls and fresh Torch. Negative schedules remain in the data:
BF16 T64/P2/L1 was 30.45% slower than its final control, and FP32 T128/P4/L1
was 122.57% slower. These results do not justify a uniform geometry rule.

The fresh Torch selection also changed. Its FP16 record moved from
XBLOCK=1, R0_BLOCK=64, two warps in section 9 to XBLOCK=4, R0_BLOCK=512,
four warps here; FP32 moved from R0_BLOCK=4096 to 1024 with 16 warps.
Consequently, beating this cohort's FP16 Torch median does not demonstrate
native improvement. Ratios use only this cohort's denominator. The retained
selection records have no Triton cache hash and do not establish a uniquely
bound timed cubin or a cause for the timing changes.

All 38 processes passed: 35 native and three Torch, with 266 primary samples,
3500 actual native graph nodes and 41 complete saved logical outputs (3636
elements). Independent CPU replay checked every saved output against the
original FP64 reference and bounds, fixture hashes, admission source bytes,
actual graph bindings and resources, and byte-identical first/last control
source. Runtime guard and read-only reports remain distinct from the saved
logical outputs; physical allocation contents are not bundled. All native
resource records reported zero local bytes, which alone does not explain
their performance differences. Whole-stage telemetry includes setup and
warmup and cannot assign a thermal or power cause to individual samples.

The [266 retained samples](../../../../scripts/benchmark/tile_torch/results/2026-10-04-cuda-sum-geometry/samples.csv)
include every candidate, both controls, fresh Torch and sample outliers.
They permit recomputing medians and ratios, not rerunning GPU correctness.
The local independent audit is
`.deps/oct04-tirx-sum-geometry-pairs-summary-v1/checkpoint.json`, SHA256
`9a7e7da723698de43b8722c26da8b2a56d5d5bb2fb24a749c8d33db3cb002abc`.
That full packet is not bundled with the source. Historical execution
receipts bind the saved source, helper closure and before/after snapshots;
current mutable source or DLL files are not substituted for that evidence.

## 11. Normalization geometry and storage comparison

A separate cohort retained the original fast-math LayerNorm and RMSNorm
fixtures, including random inputs, FP64 references and per-element bounds.
It compared 19 candidates with each fixture's strong L1 control before and
after the candidate sweep, plus fresh Torch from the first control's exact
manifest. The protocol remained seven 100 ms samples, 500 ms warmup and
100-node graphs with four-core CPU affinity. U64, BR1 capture, pad64 and the
private-storage budget were unchanged; no model was fitted.

All 19 candidates were slower than the final L1 control and fresh Torch by
median. The lowest observed candidates below are post-hoc minima, not
promoted schedules. Times are microseconds; ratios below one are faster.

| Fixture | Original T/P | L1 initial | Lowest candidate T/P/L | Candidate | L1 recheck | Fresh Torch | Candidate / recheck | Candidate / Torch | Control drift |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|
| LayerNorm BF16 128 x 1024 | 128/2 | 1.863783 | 32/1/1 scalar | 1.848727 | 1.843398 | 1.822875 | 1.002891 | 1.014182 | -1.09% |
| RMSNorm FP16 32 x 4096 | 256/1 | 1.599550 | 256/1/4 vector | 1.602061 | 1.594396 | 1.498414 | 1.004807 | 1.069171 | -0.32% |

L1 used coordinate forwarding disabled, while both L4 modes enabled it.
Therefore L1/L4 comparisons change coordinate handling as well as ownership;
they do not isolate lane width. Within each fixed geometry, L4 scalar versus
L4 vector changes only the vector flag. This narrower comparison improved
RMSNorm T256/P1 from 2.389308 to 1.602061 us (32.95%), but still failed to
beat the stronger L1 control. LayerNorm vectorization was mixed: it made
T128/P2 L4 2.92% slower, while improving T64/P2 L4 by 18.57%; the latter
remained 42.12% slower than the original final control. The complete negative
results remain available rather than selecting only favorable comparisons.

The lowest LayerNorm candidate's seven samples ranged from
1.846304--1.970869 us, versus 1.841140--1.845636 us for its final control.
The lowest RMSNorm candidate ranged from 1.600713--1.602464 us, versus
1.590244--1.598619 us. These are observed ranges, not confidence intervals
or evidence of cross-session stability. The fresh Torch denominators belong
only to this cohort; no earlier Torch timing is reused.

All 25 processes passed: 23 native and two Torch, with 175 primary samples,
2300 actual native graph nodes and 27 complete saved outputs. Independent
CPU replay checked all 3,538,944 output elements against the original FP64
bounds, all fixture/source identities, graph bindings and resources, and
byte-identical initial/recheck L1 source. The saved snapshot and executed
helper closure also matched before and after replay. Runtime physical guard
and read-only reports are retained separately; logical output replay does
not reconstruct those allocations or prove an internal vector branch.
All native resource records reported zero local bytes. Whole-stage telemetry
does not identify the cause of a particular timing sample or schedule loss.

The [175 retained samples](../../../../scripts/benchmark/tile_torch/results/2026-10-04-cuda-norm-geometry/samples.csv)
include all candidates, controls, fresh Torch, coordinate/vector flags and
resource observations. They support recomputing medians and ratios, not
rerunning GPU validation. The local full audit is
`.deps/oct04-tirx-norm-geometry-pairs-summary-v1/checkpoint.json`, SHA256
`32740135f857fc9245d2bdfb6c82a7c67da3710a4419265996d46e82fa2833eb`;
that packet is not bundled with the source. No current mutable source or
DLL receipt replaces the saved execution-time evidence.

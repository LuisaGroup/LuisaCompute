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

## 12. Larger SUM thread groups

The next independent cohort tested T512/P1 and T1024/P1, each with L1
scalar or L4 vector storage, against the original L1 controls from section
10. It retained the same three fixtures, math policies, FP64 bounds, U64,
BR1 capture, pad64 and coordinate forwarding disabled. Initial controls,
11 admitted candidates, fresh Torch and final controls ran with seven
100 ms samples, 500 ms warmup and 100-node graphs on the same CPU affinity.
No earlier Torch timing was used as a denominator.

All 11 timed candidates were slower than their final original controls by
median. The table shows the post-hoc lowest candidate per fixture; times
are microseconds. It does not select or promote a new policy.

| SUM fixture | Original T/P | L1 initial | Lowest candidate T/P/L | Candidate | L1 recheck | Fresh Torch | Candidate / recheck | Control drift |
|---|---|---:|---|---:|---:|---:|---:|---:|
| FP16 129 x 2048, fast | 128/1 | 1.351827 | 512/1/1 scalar | 1.994251 | 1.338158 | 1.403007 | 1.490295 | -1.011% |
| BF16 128 x 8192, strict | 128/2 | 2.735928 | 512/1/4 vector | 2.747457 | 2.587798 | 2.475620 | 1.061697 | -5.414% |
| FP32 3 x 8192, strict | 256/1 | 1.328337 | 512/1/1 scalar | 1.316313 | 1.314930 | 1.355308 | 1.001052 | -1.009% |

FP32 T512/P1/L1 was effectively tied with the final control, not a
demonstrated improvement: it was 0.105% slower, with overlapping sample
ranges of 1.315104--1.316954 us and 1.314522--1.333487 us. Its lower median
than the initial control cannot be separated from the -1.009% control
drift. FP16 large groups lost substantially: T512/P1/L1 was 49.03% slower
than the final control and T1024/P1/L1 was 197.07% slower.

BF16's controls drifted -5.414%. Its lowest candidate remained 6.17% slower
than the final control, and its full seven-sample range was
2.699217--4.920616 us. The 4.92 us spike is retained in the CSV; no trimming
or thermal/power explanation is applied. Whole-stage driver telemetry
includes setup and warmup and cannot identify the cause of that sample.

The separate admission run rejected FP16 T1024/P1/L4 vector before
dispatch because it had no eligible storage-pack phase. That result remains
unsupported, with no numerical pass or timing; no zero timing or performance
ratio is substituted. The formal cohort contained 20 successful processes:
17 native and three Torch, with 140 primary samples, 1700 actual native graph nodes and
23 complete saved logical outputs (1951 elements). Its independent audit
replayed the original full FP64 checks, fixture/source identities, graph
bindings and resources, and initial/recheck source equality. Runtime
physical guard/read-only reports remain separate from saved logical output
checks; no internal vector branch or uniquely bound timed cubin is inferred.

The [140 retained samples](../../../../scripts/benchmark/tile_torch/results/2026-10-04-cuda-sum-large-groups/samples.csv)
include all 11 candidates, both controls, fresh Torch and every outlier.
They permit recomputing medians and ratios, not rerunning the omitted GPU
or tensor validation. The local full audit is
`.deps/oct04-tirx-sum-large-groups-summary-v1/checkpoint.json`, SHA256
`e255a41fc602843d3fb859cc9fa8ab3d7e996cec7973a485bb4b788f417fccb2`;
that packet is not bundled with the source. Saved execution receipts,
rather than current mutable source/DLL files, define the measured cohort.

## 13. Existing materialization policy

A private runtime switch compared the existing `EXPENSIVE_ONLY` lowering
policy with the original policy on the two normalization fixtures from
section 11. `LUISA_DIAGNOSTIC_TIRX_EXPENSIVE_ONLY` unset or `0` preserves
`PRESERVE`; `1` selects the existing `EXPENSIVE_ONLY` policy. Other values
were rejected in preflight. This reused the existing lowering option; it
introduced no DSL primitive or default-policy change. Each fixture retained separate L1 scalar
and L4 vector controls before and after the candidates. Fresh Torch used
only the first L1 control's exact manifest and new per-case caches. All
processes used seven 100 ms samples, 500 ms warmup and 100-node graphs with
the same four-core CPU affinity.

The table records all four control pairs and this cohort's Torch medians,
in microseconds. Each drift compares that control's last and first median.

| Fixture | Control T/P/L | Initial | Recheck | Control drift | Fresh Torch |
|---|---|---:|---:|---:|---:|
| LayerNorm BF16 128 x 1024 | 128/2/1 scalar | 1.857500 | 1.850320 | -0.387% | 1.825069 |
| LayerNorm BF16 128 x 1024 | 128/2/4 vector | 2.348995 | 2.354546 | +0.236% | 1.825069 |
| RMSNorm FP16 32 x 4096 | 256/1/1 scalar | 1.599437 | 1.602866 | +0.214% | 1.821111 |
| RMSNorm FP16 32 x 4096 | 256/1/4 vector | 1.597751 | 1.596501 | -0.078% | 1.821111 |

All six policy candidates are retained below. Ratios use the final controls
and fresh Torch from this cohort; values below one are faster.

| Fixture | Policy candidate T/P/L | Median us | Seven-sample range us | / last L1 | / last L4 | / fresh Torch |
|---|---|---:|---|---:|---:|---:|
| LayerNorm BF16 | 128/2/1 scalar | 1.883140 | 1.873530--1.890343 | 1.017738 | 0.799789 | 1.031818 |
| LayerNorm BF16 | 128/2/4 vector | 2.523756 | 2.518016--2.547874 | 1.363956 | 1.071865 | 1.382827 |
| RMSNorm FP16 | 256/1/1 scalar | 1.706482 | 1.703707--1.708809 | 1.064644 | 1.068888 | 0.937055 |
| RMSNorm FP16 | 256/1/4 vector | 1.583888 | 1.580878--1.586274 | 0.988160 | 0.992099 | 0.869737 |
| RMSNorm FP16 | 128/4/1 scalar | 4.278119 | 4.276278--4.279062 | 2.669043 | 2.679685 | 2.349182 |
| RMSNorm FP16 | 128/4/4 vector | 3.411975 | 3.407180--3.415509 | 2.128671 | 2.137158 | 1.873568 |

The L1 comparisons isolate the lowering-policy request: coordinate
forwarding remains disabled and geometry is unchanged. They regressed
1.77% for LayerNorm and 6.46% for RMSNorm against their final L1 controls.
For L4, the original policy used coordinate forwarding enabled, while the
candidate disabled it. RMSNorm's 0.79% improvement against its final L4
control therefore changes both requests and cannot establish a policy-only
gain. LayerNorm L4 was 7.19% slower than its final L4 control. The RMSNorm
T128/P4 candidates became admissible, but were still 166.90% and 112.87%
slower than the final L1 control. Their original-policy counterparts remain
separate unsupported admissions, without numerical passes or timing ratios.

Fresh Torch RMSNorm was 1.821111 us, compared with 1.498414 us in section 11.
That change is not native progress, and the earlier denominator is not reused.
The saved selected configuration changed from XBLOCK=1, R0_BLOCK=4096 to
XBLOCK=2, R0_BLOCK=2048; both used 16 warps and one stage. Both rounds have
a null Triton cache hash, so these are saved compiler-selection observations,
not a unique binding to the timed cubin or a causal explanation.
The observed candidate ranges and control drifts are preserved without a
cross-session stability claim, a new fit or a promoted default.

All 16 formal processes passed: 14 native and two Torch, yielding 112
primary samples, 1400 actual native graph nodes and 18 complete saved
logical outputs. Independent CPU replay checked all 2,359,296 output
elements against the original FP64 references and per-element bounds,
exact fixture and admitted-source identities, policy markers, graph
bindings, final pointers and resources. All four final control sources
matched their corresponding initial sources byte for byte. The original
fast-math settings, seed, U64, BR1 capture, pad64, storage budget and
integer-extrema-disabled request were retained. The invalid policy value
was separately rejected before the formal cohort.

LLVM 22 and LLVM 23 MSVC full builds passed. The LLVM 23 host suite passed
1196 assertions in 17 groups; LLVM 22 had passed the same host suite before
this factory switch. The 13-line factory integration was covered by 17
admission/preflight requests, retaining unsupported cases separately.

Native resource records reported 39 or 40 registers and zero local bytes;
shared memory was 80 bytes for LayerNorm, 32 bytes for RMSNorm T256/P1 and
zero for RMSNorm T128/P4. These counts do not establish a performance cause.
Whole-stage driver telemetry recorded software power/thermal reason samples,
but includes setup, compilation and warmup and is not synchronized to each
timing sample. Runtime physical guard/read-only reports remain separate from
the independently replayed logical outputs; no internal vector branch,
physical memory traffic or uniquely bound timed cubin is inferred.

The [112 retained samples](../../../../scripts/benchmark/tile_torch/results/2026-10-04-cuda-materialization-policy/samples.csv)
include all candidates, four control pairs, fresh Torch, request flags and
resource observations. They permit recomputing medians and ratios, not
rerunning the omitted GPU or tensor validation. The local full audit is
`.deps/oct04-tirx-expensive-only-pairs-summary-v1/checkpoint.json`, SHA256
`d3e04e0f73c5cc0221dc6cc6ce57ac36b10570829ab9981180ccccaa29991a66`;
that packet is not bundled with the source. Saved snapshots and the
executed helper closure define the measured cohort, rather than current
mutable source or DLL files.

## 14. Terminal row-only collective

The private `LUISA_DIAGNOSTIC_TIRX_ROW_ONLY_COLLECTIVE` switch tested a
terminal scalar-output optimization on the three existing SUM fixtures.
Unset or `0` retains the original source; `1` requests the proved terminal
suffix, and other values are rejected. The first reduction tree and shared
publication barrier remain collective. Only the second reduction tree and
its closed scalar-output suffix run in the first full warp of each program;
the external store still requires the active row's worker zero. A failed
ownership proof retains the original implementation. This adds no DSL
primitive, changes no numerical permission, and is not a promoted default.

Four configurations were fixed before timing: the three original L1 scalar
controls and the existing BF16 T64/P1/L4 vector configuration. Each had its
own flag-0 initial and final control. Fresh Torch used only each fixture's
first L1 manifest and new caches; the additional BF16 vector pair did not
replace that fixture anchor. Original math policies, U64, BR1 capture,
coordinate forwarding disabled, integer extrema enabled and pad64 remained
fixed. The protocol used seven 100 ms samples, 500 ms warmup, 100-node
graphs and four-core affinity `0x15400`.

All four candidate requests applied. Times below are microseconds; ratios
use the geometry-matched controls and fresh Torch from this cohort.

| SUM fixture | T/P/L | Initial | Candidate | Recheck | Fresh Torch | Candidate / first | Candidate / last | Candidate / Torch | Control drift |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| FP16 129 x 2048, fast | 128/1/1 scalar | 1.352522 | 1.347481 | 1.340824 | 1.545497 | 0.996273 | 1.004965 | 0.871876 | -0.865% |
| BF16 128 x 8192, strict | 128/2/1 scalar | 2.745064 | 2.586747 | 2.585420 | 2.455022 | 0.942327 | 1.000513 | 1.053655 | -5.816% |
| FP32 3 x 8192, strict | 256/1/1 scalar | 1.329276 | 1.316123 | 1.307570 | 1.838534 | 0.990105 | 1.006541 | 0.715854 | -1.633% |
| BF16 128 x 8192, strict | 64/1/4 vector | 2.413752 | 2.381131 | 2.400266 | 2.455022 | 0.986485 | 0.992028 | 0.969902 | -0.559% |

None of the original L1 configurations improved against its final control.
BF16 L1's apparent 5.77% gain against the initial control coincided with
-5.816% control drift; it was 0.051% slower than the final control. The
additional BF16 vector pair was 0.797% faster than its final control and
1.352% faster than its initial control. This small gain in one paired cohort
does not establish cross-session stability or justify a new default.

| Configuration | Candidate seven-sample range us | Final control range us |
|---|---|---|
| FP16 128/1/1 | 1.346388--1.348652 | 1.336851--1.342041 |
| BF16 128/2/1 | 2.585348--2.589601 | 2.583616--2.588361 |
| FP32 256/1/1 | 1.308607--1.321630 | 1.304568--1.312355 |
| BF16 64/1/4 vector | 2.379577--2.386506 | 2.397226--2.404737 |

These ranges are descriptive, not confidence intervals. All samples and
negative results remain. Fresh Torch belongs only to this round; a changed
Torch denominator is not evidence of native progress.

The saved current Torch selections were X/R/warps = 8/512/4 for FP16,
2/64/2 for BF16 and 2/2048/16 for FP32, all with one stage. These differ
from the corresponding earlier geometry and large-group rounds. All nine
saved records across those three rounds have a null `triton_cache_hash`;
the configurations are compiler-selection observations, not unique bindings
to the timed cubins or a causal explanation for the timing differences.

All 15 formal processes passed: 12 native and three Torch, with 105 primary
samples, 1200 actual native graph nodes and 18 complete saved logical
outputs. Independent CPU replay checked all 1684 output values against the
original FP64 references and bounds, fixture and admitted-source identities,
actual row-only receipts, graph bindings, final pointers and resources.
All four recheck sources matched their corresponding initial sources byte
for byte. The earlier admission had 25 numerical passes and 2500 graph
nodes, plus one separate invalid-value rejection. LLVM 22 and LLVM 23 MSVC
full builds and the host suite's 10008 assertions in 18 groups passed.
The default CUDA, vector CUDA and Metal source captures and 12 ordered
candidate callbacks were byte-identical to the bound earlier capture;
only exact diagnostic log records were removed for that comparison.

Native local memory remained zero. Registers/shared bytes were 25/16 for
FP16, 39/16 to 38/16 for BF16 L1, 38/32 for FP32 and 40/8 for BF16 vector.
Those observations do not establish the timing cause. Whole-stage driver
telemetry includes compilation and warmup rather than individual timing
samples. Its Torch stage contains an anomalous 590.01 W value, retained as
an untrusted telemetry reading, not evidence that the GPU drew 590.01 W.
No power or thermal cause is assigned. Physical guard/read-only checks
retain runtime reports separately from saved logical-output replay; neither
source text nor graph identity proves an internal branch or timed SASS.

The [105 retained samples](../../../../scripts/benchmark/tile_torch/results/2026-10-04-cuda-terminal-row/samples.csv)
include every candidate, matched control, fresh Torch result, request flag
and resource observation. They permit recomputing medians and ratios, not
rerunning omitted tensor or GPU validation. The local full audit is
`.deps/oct04-tirx-row-only-pairs-summary-v1/checkpoint.json`, SHA256
`391059cc28559f41f9f54472fb2a8774f353bf5110e97fd6bf4ef8c139b635c9`;
that packet is not bundled with the source. Saved snapshots and the executed
helper closure define the measured cohort; current mutable source or DLL
files are not substituted for those execution-time receipts.

## 15. Explicit fast DIV/SQRT reassociation

`LowerOptions::allow_fp32_div_sqrt_reassociation` defaults to false. The
private CUDA switch `LUISA_DIAGNOSTIC_TIRX_FAST_DIV_SQRT=1` grants the
existing fast-math strategy only when fast math is already enabled:
direct FP32 `x / sqrt(y)` may become `x * rsqrt(y)`. Unset or `0` keeps the
former lowering; other values and `1` with strict math are rejected before
compilation. Other backends retain the default permission. Matching does
not cross a cast, load or arithmetic node. Original SQRT consumers,
operand-axis projections, load snapshots, guards and reduction contracts
remain intact. This adds no DSL primitive and is not enabled by default.

This fixed cohort used LayerNorm 128 x 1024 BF16 at T128/P2 and RMSNorm
32 x 4096 FP16 at T256/P1. Both retained fast math, PRESERVE, L1/U64,
vector and coordinate forwarding disabled, integer extrema disabled,
cachefalse and pad64. Each fixture ran flag0, flag1, fresh Torch, then
flag0 recheck. Torch used the initial manifest and fresh caches. The
protocol remained seven 100 ms samples, 500 ms warmup, 100-node graphs
and four-core affinity `0x15400`.

| Fixture | Initial us | Candidate us | Recheck us | Fresh Torch us | Candidate / first | Candidate / last | Candidate / Torch | Control drift |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| LayerNorm BF16 128 x 1024 | 1.861933 | 1.851314 | 1.840030 | 1.754377 | 0.994297 | 1.006133 | 1.055255 | -1.1763% |
| RMSNorm FP16 32 x 4096 | 1.596306 | 1.595887 | 1.592182 | 1.701622 | 0.999737 | 1.002327 | 0.937862 | -0.2584% |

Neither candidate improved against its final control: LayerNorm was
0.613% slower and RMSNorm 0.233% slower. LayerNorm's candidate range was
1.848914--1.858240 us versus 1.839696--1.840551 us for its final control;
RMSNorm's ranges were 1.588053--1.603703 and 1.583211--1.593435 us and
overlapped. Both candidate ranges overlapped their initial controls.
These are descriptive ranges, not confidence intervals. This round
provides no robust performance gain or basis for a default change;
RMSNorm's fresh Torch comparison does not establish a native improvement.

All eight processes passed: six native and two Torch, with 56 primary
samples, 600 actual native graph nodes and ten complete saved logical
outputs. Independent CPU replay checked all 1,310,720 values against the
unchanged original FP64 references and bounds. Fixture, source, actual
function/grid/block/pointer and resource checks also passed. Candidate
sources matched the admitted sources and contained `rsqrtf`; both
flag0 rechecks matched their initial sources byte for byte. Registers,
shared bytes and local bytes remained 40/80/0 for LayerNorm and 39/32/0
for RMSNorm. Source and graph identity are not executed-instruction traces.

LLVM 22 and LLVM 23 MSVC full builds and the host suite's 10202 assertions
in 19 groups passed. The bounded admission retained 14 numerical passes
and two separate prelaunch rejections. Default CUDA, vector CUDA and Metal
sources and 12 ordered callbacks matched the earlier bound capture exactly
after removing only the known complete diagnostic log records.

The separate 1046-input direct-expression special-value probe retained
81 output-bit differences and no classification differences between the
two fast expressions. Its checked normal/special domains passed, but
subnormal inputs and output underflow/overflow were observations without
an independent classification gate. It does not prove strict equivalence,
Tile snapshot safety or performance. The full norm oracle was not loosened.

Whole-stage telemetry includes compilation and warmup, not individual
timing samples. No clock, power or thermal cause is assigned. Physical
guard/read-only checks retain runtime reports separately from saved
logical-output replay. All negatives and control drift remain in the
[56 retained samples](../../../../scripts/benchmark/tile_torch/results/2026-10-04-cuda-fast-div-sqrt/samples.csv),
which support recomputing medians and ratios, not omitted GPU/tensor
validation. The local audit checkpoint is
`.deps/oct04-tirx-fast-div-sqrt-pairs-summary-v1/checkpoint.json`, SHA256
`89ba88dd800954ad3b44dfe1f2f3b37742633f976a6332888ad150835d1dcbff`.
Saved snapshots and executed helper receipts define the timed cohort;
later mutable sources and DLLs are not substituted for them.

## 16. BF16 conversion and vector-phase eligibility

This is the retained first experiment with globally eager BF16 conversion,
not the final implementation. This compiler package made LayerNorm's
gamma/beta loads and output stores eligible for the existing L2/L4 vector
phases. The round-to-BF16 expression used a pure UInt32 Select, TVM generated the
condition SSA value with its own type, and the phase audit admitted bounded
pure scalar bitcasts. Original rounding, NaN quieting, guards and numerical
bounds remained the contract; no DSL primitive was added. L8 retained a
scalar BF16 epilogue while its independent input phase remained eligible.
This round compares the complete old/new bridge and TVM packages, not the
isolated effect of any one change.

The fixed fixture was LayerNorm 128 x 1024 BF16, fast math, BR1,
T128/P2/U64, cachefalse and pad64. PRESERVE, FAST_DIV_SQRT=0,
terminal-row=0 and integer-extrema=0 stayed fixed. L1 used scalar accesses
and coordinates0; L2/L4 used vector1 and coordinates1. The order was old
L1/L2/L4, new L1/L2/L4, one fresh Torch run, then three matching old
rechecks. Torch used only the first old L1 manifest and fresh caches.
Seven 100 ms samples, 500 ms warmup, graph100 and four-core affinity
`0x15400` were retained.

| Lane configuration | Old initial us | New us | Old recheck us | New / first | New / last | New / fresh Torch | Old control drift |
|---|---:|---:|---:|---:|---:|---:|---:|
| L1 scalar | 1.846940 | 1.880563 | 1.848300 | 1.018204 | 1.017455 | 1.072418 | +0.0736% |
| L2 vector | 1.935488 | 1.820869 | 1.926592 | 0.940780 | 0.945125 | 1.038377 | -0.4597% |
| L4 vector | 2.355609 | 1.807418 | 2.357904 | 0.767283 | 0.766536 | 1.030706 | +0.0974% |

Fresh Torch's median was 1.753573 us. Although new L2/L4 were 5.488% and
23.346% faster than their own old final controls, those old vector paths
were weaker than old L1. Against the strong old L1 final control, new L2
improved by only 1.484% and new L4 by 2.212%. New L1 regressed by 1.746%.
Every new configuration remained slower than this round's Torch. These
negatives rule out presenting the 23.346% figure as a general native gain
or a reason to promote the globally changed scalar lowering by default.
A later implementation restricted to the vector copy, with the original
lazy scalar lowering restored, requires separate validation and a new
cohort; it must not replace these results.

| Lane | Old initial seven-sample range us | New range us | Old recheck range us |
|---|---|---|---|
| L1 | 1.846021--1.871508 | 1.873307--1.892070 | 1.848000--1.848680 |
| L2 | 1.924393--1.955391 | 1.803890--1.836702 | 1.924436--1.927462 |
| L4 | 2.351012--2.370956 | 1.795491--1.823961 | 2.348891--2.359056 |

Torch's range was 1.749882--1.754695 us. These are descriptive ranges,
not confidence intervals or evidence of cross-session stability. Its saved
selection was X2/R1024, eight warps and one stage, with a null
`triton_cache_hash`; that compiler observation does not uniquely identify
the timed cubin. No historical Torch denominator is substituted.

All ten processes passed: nine native and one Torch, with 70 primary
samples, 900 actual native graph nodes and eleven complete saved logical
outputs. Independent CPU replay checked all 1,441,792 values against the
unchanged original FP64 references and per-element bounds. Physical
guard/read-only checks remain runtime reports, separately from replay of
saved logical outputs. Each native graph retained the actual function,
grid, block and all four final argument pointers. Registers/shared/local
bytes were 40/80/0 for every native process; these are Driver resource
observations, not occupancy or a performance explanation.

The new L2/L4 sources contain guarded `ushort2`/`ushort4` gamma and beta
loads and output stores, with scalar fallbacks. Their joint epilogue
alignment predicate covers the final gamma, beta and output pointers;
the input vector phase has its own guard. This is generated-source
evidence, not proof of the executed branch, machine instruction width or
memory traffic. All sources matched their admitted anchors, including
the freshly captured old L2 source, and each old recheck matched its
same-lane initial source byte for byte.

Both packages used the same private executable and fixed backend/common
dependencies. Observations before and after each native run verified the
unique loaded bridge and TVM compiler paths in the selected package,
their module bases within that process, and the observed common modules.
The archived package files and common files matched the saved execution
receipts during audit. This does not hash loaded memory or substitute
current production sources for historical build evidence.

LLVM 22/23 MSVC full builds and 12804 host assertions in 22 groups passed;
the CUDA runtime suite passed 2184696 assertions in 12 groups. The bounded
admission passed all 17 cases, with 1700 actual graph nodes and 1,323,776
saved output values independently checked. A separate generated-source
BF16 probe passed 129360 exact output-bit comparisons across scalar,
two-lane and four-lane strict/fast routes, including special raw FP32
patterns and allocation guard/read-only checks. That conversion probe
does not replace the complete LayerNorm oracle. The three existing
default CUDA/vector-CUDA/Metal captures and twelve ordered callbacks
retained exact parity after removing only known diagnostic log records;
this is bounded capture parity, not a claim that globally changed BF16
scalar lowering preserves every source.

Whole-stage telemetry includes setup, compilation and warmup and is not
aligned to individual timing samples. Software power-cap and thermal
slowdown reasons were reported active during parts of the run; no thermal
or power cause is assigned. The Torch-stage 588.21 W reading is retained
as untrusted anomalous telemetry, not evidence that the GPU consumed that
power. Hardware thermal/power-brake, application-clock and board-limit
reasons were reported inactive.

The [70 retained samples](../../../../scripts/benchmark/tile_torch/results/2026-10-04-cuda-bfloat-vector/samples.csv)
include every native and Torch result, matched-control ratios and native
resource observations. They support recomputing medians and ratios, not
rerunning omitted tensor or GPU validation. The local audit checkpoint is
`.deps/oct04-tirx-bfloat-select-pairs-summary-v1/checkpoint.json`, SHA256
`3523bc66f943dfc7e63cd2c37bb58ec24eaee53fcb462b4a1881b3b13d48c1af`.
Saved execution snapshots and the executed helper closure define this
cohort; later mutable sources and DLLs are not substituted for them.

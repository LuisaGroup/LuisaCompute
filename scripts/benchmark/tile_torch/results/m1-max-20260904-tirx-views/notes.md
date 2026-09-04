# Read-only TIRx views: large-GEMM gap reduced, not closed

The new opt-in TIRx MPP realization forwards **proved immutable snapshots**
before resource planning. This admits full-K global-input candidates without
allocating full-K A/B shared tiles. Original TIRx SIMD-group, staged TIRx MPP,
native MPP, handwritten MPP, MPS and Torch all remain independent controls.
No production default or size-based dispatch rule changed.

The seven-path, eight-shape, fourteen-round replay validated **784/784 complete
outputs**. For 512³, the new path measured **43.523 µs**, versus Torch's
**48.794 µs**. For 1024³ it measured **291.736 µs**, versus original TIRx
**320.162 µs**, staged TIRx MPP **348.847 µs**, Torch **291.133 µs**, and MPS
**278.687 µs**. Its paired median time ratio to Torch is **1.001860** and it
was slower in 12/14 rounds: this is near Torch, not an established win. The
paired MPS gap is still **4.8%**. Native MPP itself remains about **5.8%**
behind MPS in this replay.

All these times are synchronized, device-resident **host-wall batched times**,
including each runtime's dispatch/encoding/submission, not GPU kernel durations.
See the [complete seven-path table](results.md) and [raw samples](results.json).

## Semantic boundary

`CompileOptions::forward_readonly_tile_loads` is default-off and currently
requires `metal_mpp`. The pass runs before pipeline scheduling and cooperative
resource planning, on typed native TIRx. It does not modify generated source or
call Luisa's native MPP emitter.

The proof requires the caller's noalias contract, an immutable compact FP32
input parameter, complete initialization of a nonescaping compiler-local
snapshot, and dominated, bounded consumers. The complete function is audited
for writes, aliases, address escapes and unknown effects. Bounds use native
TIRx simplification; memory-dependent coordinates and call expressions are not
accepted as proof inputs. This matters even when the source buffer itself is
read-only: changing a memory-derived index must not change a previous snapshot.
Only proved input **identities**, not a `global` scope string, authorize matrix
memory operands. Writable accumulators still require owned storage.

Manual memory is marked even without an explicit placement constraint and is
not removed. Unproved padding, aliases, mutable inputs and address escapes keep
the snapshot path. Stage cuts remain intact; both one- and two-version pipeline
tests exercise the forwarding path. This is deliberately a conservative subset,
not a general dependence solver or a universal zero-copy transformation.

The isolated TVM patch also fixes `PointerValueTypeRewrite`: an address modular
coefficient of 32768 or greater could overflow DLPack's signed 16-bit lane
encoding, producing an accidental scalable-vector type. The debugger located
the first exception in `BufferVarInfo::get_preferred_dtype`; this was not a
shared-memory-capacity error. Unrepresentable hints now stay scalar. The full-K
transpose regressions exercise the large-stride case.

## Search, freeze, replay

Two explicit Staged/JIT searches considered **360 candidates: 152 valid and
208 rejected**, all retained. The [32/64/128-thread search](../m1-max-20260904-tirx-views-search/results.json)
contains 264 candidates; the [256-thread/copy-batch search](../m1-max-20260904-tirx-views-256-search/results.json)
contains 96. Each search freshly recaptured, compiled, validated and timed its
selected candidate. The [frozen manifest](../m1-max-20260904-tirx-views-plan.json)
then selected between those post-selection results. It contains configurations
and provenance, **not comparative performance evidence**.

The following schedules were frozen before the independent replay. All use
group execution and pipeline window 1. `copy batch` is the requested policy;
when forwarding eliminates copies it need not describe emitted work.

| M×N×K | BM×BN×BK | Threads | Copy batch |
|---|---|---:|---:|
| 32×32×32 | 32×32×32 | 128 | 1 |
| 128×128×128 | 32×32×128 | 256 | 8 |
| 512×512×512 | 32×64×32 | 128 | 1 |
| 1024×1024×1024 | 64×64×1024 | 128 | 1 |
| 256×1024×128 | 64×64×32 | 256 | 8 |
| 1024×128×256 | 32×32×32 | 64 | 1 |
| 127×193×61 | 32×32×32 | 256 | 8 |
| 513×257×129 | 32×64×32 | 256 | 8 |

The original and staged-MPP TIRx paths keep the same
[joint-search schedule](../m1-max-20260904-joint-search/results.json).
Native/handwritten MPP keep their matching, separately selected
[MPP configuration](../m1-max-20260904-mpp-search/results.json).
Only configurations are reused, never historical timing scores. The view
realization is independently tuned, **not a same-geometry ablation**.

## Paired comparison

Each cell is the median of fourteen **within-round** time ratios, with the
view realization in the numerator. Below one means less time. These are
descriptive paired medians, not confidence intervals or general hardware claims.

| M×N×K | / Original TIRx | / Staged TIRx MPP | / Torch | / MPS |
|---|---:|---:|---:|---:|
| 32×32×32 | 0.651253 | 0.668777 | 0.116612 | 0.291279 |
| 128×128×128 | 0.796879 | 0.891164 | 0.192361 | 0.329398 |
| 512×512×512 | 0.813974 | 0.881930 | 0.893298 | 0.814945 |
| 1024×1024×1024 | 0.911123 | 0.836454 | 1.001860 | 1.047919 |
| 256×1024×128 | 0.860499 | 0.905773 | 0.559522 | 0.799542 |
| 1024×128×256 | 0.876353 | 0.928201 | 0.598567 | 0.627036 |
| 127×193×61 | 0.730974 | 0.744543 | 0.228673 | 0.346166 |
| 513×257×129 | 0.934018 | 1.004271 | 0.608465 | 0.557356 |

All seven shapes other than 1024³ were faster than Torch in every round.
For 513×257×129, staged MPP and forwarding MPP have **identical generated
source in all fourteen rounds**; their 0.4% difference is not evidence of a
forwarding benefit. Its padded accesses remain staged. The smaller ragged
case also retains padded copies; its changed geometry must not be credited
to removing them.

## What changed structurally

The new [1024³ source](sources/4b94b798e9cec6d4ee340fdc8627f9a46995da66a723f3bdfcbd4cb8345c1806.metal)
uses a 64×64 outer tile, 128 threads, a 2×2 subgroup grid and a 32×32 MPP
output per subgroup. A/B are global input views, the accumulator is a
persistent cooperative tensor, and the result is stored directly to global
memory. There are **zero shared-memory bytes** and one static full-K MPP
call site; the single-iteration outer K loop simplifies away.

The [staged control](sources/dd9c9558b7034ab150dd6ce5e6d808dd28ed93ece06bccbe71bc4f036af5c29e.metal)
keeps 256 threads, 16 KiB of shared storage and 32 outer K steps. The new source
still has two conservative threadgroup barriers. Removing either needs an
effect/participation and successor-use proof; no barrier was stripped for this
experiment. The relative contribution of geometry, storage, K granularity,
compilation and internal MPP scheduling is not isolated by this comparison.

The [512³ winner](sources/169923c00aeb4e7b6966c7087547a7f83a165baf88767900d5ef0128a9e7b640.metal)
still uses BK=32. Full K is therefore not a universal policy. Planner fields
such as `matrix_issues`, `shared_fragment_transfers` and
`fragment_scalars_per_lane` remain explicitly tagged
`cost_basis=simdgroup_reference_geometry`: they are not measured MPP
instruction counts, transfers or registers. The next cost-model step must
distinguish realization family, storage choice, K granularity, participation,
recurrence and edge policy, then calibrate on held-out shapes. No calibrated
MPP solver is claimed here, and CPU performance was not optimized in this run.

## Validation and provenance

- Apple M1 Max, macOS 26.6.2; Torch 2.14.0, NumPy 2.5.2. Exact versions,
  commands and removed environment overrides are recorded in JSON metadata.
- Eight shapes, seven paths, fourteen rounds balancing path positions and
  pairwise precedence; shape order rotates. Seven samples per path, 30 ms
  target sample duration and 200 ms warmup. No failed/slow row was discarded.
- Every output element was checked against an FP64 oracle with
  `atol=rtol=1e-4`: 784 valid outputs, 184,014,208 checked elements, maximum
  observed absolute error zero for these deterministic dyadic inputs.
  Separate matrix regressions use non-dyadic data and changed inputs.
- JIT, allocation and transfers are excluded from warm timing. Native and
  handwritten MPP request fast math off and relaxed precision off. TVM's
  Metal runtime hardcodes fast math on; its original path uses MSL 3, MPP
  uses MSL 4. These compiler-policy differences are not hidden.
- No build, test or profiler ran during timing. All **22 binaries/libraries**
  matched their before/after hashes. All **23 content-addressed Metal sources**
  were independently rehashed. Post-benchmark test-only rebuilds do not change
  the recorded benchmark/compiler/runtime artifacts.
- The patched TVM and both Luisa configurations completed full builds. The
  25-test native/TIRx/system cohort passed **23/25 in each build**. The only
  failures are the existing `mem_flags(3)` expectations in
  `test_tile_tirx_cooperative_metal` and `test_tile_tirx_memory_metal`; the
  pre-existing worktree change emits `mem_flags(2)`. Neither that change nor
  those assertions was overwritten. This is not a completely green suite or
  proof that the weaker fence is generally sufficient.
- Expanded matrix regressions cover pipeline windows 1/2, full-K large
  strides, A/B transposes and literal/loaded C, noalias rejection, padded
  bounds, mutable inputs, manual memory with/without placement, address
  escape and mutable address indices. The ordinary CPU and Metal matrix tests
  also run against the unpatched TVM library; MPP capability requests remain
  explicit errors there. The benchmark-driver unit suite passes **43/43**.

TVM base: `c7b458e946bc4266915da582457476bdcd9705ae`; tvm-ffi:
`12dbf053b3d9ba4ebd9da3123b1aeca79cf74229`. The
[native C++ extension patch](../../../../../src/tile/bridge/tirx/patches/README.md)
has SHA-256 `9c8a32500442d1b156f1020f2658242b6ff00f61669d0a4ae50c906e002a59bb`.
It builds separately; the ordinary installed TVM libraries were not replaced.

Reproduce after the full build and correctness tests, using a new output
directory and the frozen plans above:

```sh
uv run --no-project --python 3.13 --with torch --with numpy python \
  scripts/benchmark/tile_torch/compare_lowerings.py \
  --native /tmp/luisa-tvm-mpp.VaKmzx/luisa-build/bin/benchmark_tile_native \
  --tirx /tmp/luisa-tvm-mpp.VaKmzx/luisa-build/bin/benchmark_tile_tirx \
  --mpp /tmp/luisa-tvm-mpp.VaKmzx/luisa-build/bin/benchmark_tile_mpp \
  --mps /tmp/luisa-tvm-mpp.VaKmzx/luisa-build/bin/benchmark_tile_system \
  --mpp-plan scripts/benchmark/tile_torch/results/m1-max-20260904-mpp-search/results.json \
  --tirx-plan scripts/benchmark/tile_torch/results/m1-max-20260904-joint-search/results.json \
  --tirx-mpp \
  --tirx-view-plan scripts/benchmark/tile_torch/results/m1-max-20260904-tirx-views-plan.json \
  --compiler-artifact /tmp/luisa-tvm-mpp.VaKmzx/build/lib/libtvm_compiler.dylib \
  --compiler-artifact /tmp/luisa-tvm-mpp.VaKmzx/build/lib/libtvm_runtime.dylib \
  --compiler-artifact /tmp/luisa-tvm-mpp.VaKmzx/build/lib/libtvm_runtime_metal.dylib \
  --compiler-artifact /tmp/luisa-tvm-mpp.VaKmzx/build/lib/libtvm_ffi.dylib \
  --rounds 14 --samples 7 --sample-ms 30 --warmup-ms 200 \
  --output /tmp/tile-seven-way-replay
```

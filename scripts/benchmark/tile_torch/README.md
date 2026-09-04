# Native TileIR / TVMx vs PyTorch

This is an opt-in **correctness-checked, multi-shape** benchmark, not a CTest
performance threshold. It compares FP32 GEMM, add, row sum, softmax and RMSNorm
on CPU and actual Metal / PyTorch MPS. GEMM includes small/large squares,
tall/wide matrices, and non-multiple tail sizes; reductions vary both row count
and width.

Latest M1 Max evidence: [Metal subgroup sum/softmax/RMSNorm cohort](results/m1-max-20260905-metal-subgroup-reductions/notes.md),
[balanced same-binary RMSNorm lowering A/B](results/m1-max-20260905-metal-subgroup-rmsnorm-replay/notes.md),
[MPP cost-model v1→v2 calibration-cohort study](results/m1-max-20260905-mpp-cost-v2-search/notes.md),
[frozen seven-path MPP-v2 replay](results/m1-max-20260905-mpp-cost-v2-replay/notes.md),
[proved CPU CBLAS plan](results/m1-max-20260905-cpu-cblas-v2-plan/notes.md),
[six-order CPU CBLAS/Torch/direct-BLAS replay](results/m1-max-20260905-cpu-cblas-v2-replay/notes.md),
[CPU reference add/sum/softmax plan](results/m1-max-20260905-cpu-reference-ops-v2/notes.md),
[CPU Accelerate add/sum/softmax plan](results/m1-max-20260905-cpu-accelerate-ops-v2/notes.md),
[six-round CPU array-math policy A/B](results/m1-max-20260905-cpu-accelerate-ops-replay/notes.md),
[CPU provider/code/docs validation](results/m1-max-20260905-cpu-provider-validation/notes.md),
[ragged CPU guard repair A/B](results/m1-max-20260905-cpu-guards-replay/notes.md),
[fresh guarded-view CPU/Torch/BLAS comparison](results/m1-max-20260905-cpu-guards-system/notes.md),
[seven-path Metal maintenance smoke check](results/m1-max-20260905-guards-lowerings-smoke/notes.md),
[other-operator guard smoke check](results/m1-max-20260905-cpu-guards-ops/notes.md),
[retained scalarization failure](results/m1-max-20260904-cpu-guard-plan/notes.md),
[CPU input-view A/B](results/m1-max-20260904-cpu-views-replay/notes.md),
[fresh input-view CPU/Torch/BLAS comparison](results/m1-max-20260904-cpu-views-system/notes.md),
[CPU Cartesian packing A/B](results/m1-max-20260904-cpu-cartesian-replay/notes.md),
[fresh Cartesian CPU/Torch/BLAS comparison](results/m1-max-20260904-cpu-cartesian-system/notes.md),
[seven-way explicit subgroup-sync candidate](results/m1-max-20260904-subgroup-sync-lowerings/notes.md),
[fixed-geometry subgroup-fence A/B replay](results/m1-max-20260904-subgroup-sync-replay/notes.md),
[seven-way replay with proved read-only TIRx views](results/m1-max-20260904-tirx-views/notes.md),
[independent TIRx MPP codegen and its retained regression](results/m1-max-20260904-tirx-mpp/notes.md),
[seven-way native/TIRx/MPP/MPS/Torch and Runtime controls](results/m1-max-20260904-runtime-controls/notes.md),
[rejected fragment-load batching, with saved patch and six-round replay](results/m1-max-20260904-fragment-batch/notes.md),
[five-path native/TIRx/MPP/MPS/Torch replay](results/m1-max-20260904-joint-lowerings/notes.md),
[larger tiles and controlled cooperative-copy batching](results/m1-max-20260903-copy-plan.md),
[six-round direct Accelerate/BLAS and MPS comparison](results/m1-max-20260903-system-baselines.md),
[execution planner, cost-model errors, and controlled comparisons](results/m1-max-20260903-planner.md),
[proved direct accumulator output and controlled measurements](results/m1-max-20260903-direct-store.md),
[dependence-aware synchronization and controlled measurements](results/m1-max-20260903-barrier-plan.md),
[matrix/copy layout sensitivity missing from the model](results/m1-max-20260903-layout-tie.md),
[rejected structural experiments and joint execution search](results/m1-max-20260904-tirx-structure.md),
[Staged/JIT and four-round comparisons](results/m1-max-20260903-jit.md),
and [actual PyTorch dispatch / Xcode profiling](results/m1-max-20260903-profile.md).
The reports include unsuccessful tuning choices and remaining library gaps.

## Direct XIR/SIMD planner comparison

`compare_xir.py` is the independent CPU pilot for the direct
TileIR→XIR→SIMD Runtime path. It does not call or relabel the TIRx binary. It
balances automatic planning, a fixed `{root order [0,1], 64 workers/block}`
control and eager Torch over all six orders, retains full outputs and LLVM,
and rechecks an FP64 oracle before accepting a timing row:

```bash
uv run --no-project --python 3.13 --with numpy --with torch \
  python scripts/benchmark/tile_torch/compare_xir.py \
  --native BUILD/bin/benchmark_tile_xir \
  --compiler-artifact /path/to/the/actual/libLLVM.dylib \
  --output NEW_EMPTY_DIRECTORY \
  --rounds 6 --samples 5 --sample-ms 20 --warmup-ms 100 --threads 8
```

The C++ binary also accepts `planned`, `canonical`, or `reversed`, followed
optionally by an exact block-worker count. Planner policy, realized order,
cost decomposition, source identity, cold/JIT phases and both throughput and
single-call latency samples are emitted as JSON. The report hashes the binary,
all adjacent Luisa dynamic libraries and explicit compiler artifacts before
and after timing. Do not run builds, tests or profilers concurrently.

The first saved pilot is
[m1-max-20260905-xir-simd](results/m1-max-20260905-xir-simd/notes.md).
It is intentionally a negative-result report: a narrow mapping search does
not replace packed/register-blocked CPU matrix realization.

## Direct BLAS and MPS GEMM baselines

On macOS the build also creates `benchmark_tile_system`, an independent
Objective-C++ executable linked only to system frameworks, not TileIR or TVM.
It uses CPU Accelerate `cblas_sgemm` (classic LP64 API) and GPU
`MPSMatrixMultiplication` (`MPSKernelOptionsNone`, no reduced-precision option).
These are **comparison baselines, not replacement lowering paths**.
PyTorch's default MPS execution is listed separately; it is not assumed to use
the same internal kernel as the direct MPS matrix API.

Both library baselines use compact row-major FP32 `C=A*B`, alpha=1, beta=0,
no transpose, and the same deterministic values as the Tile/PyTorch driver.
MPS inputs and output use private device buffers, uploaded once before warm
timing; no padded strides or input prepacking are introduced. Every repeated
call writes the same preallocated output. One timed MPS batch encodes all
calls into one command buffer, commits and waits once. Single-call latency
uses a separate command buffer per call. API/encoding/submission costs remain
in warm timings; library-internal scratch management is not artificially
subtracted. Timing is host wall time, **not pure GPU time**.

After a complete build, validate both native paths explicitly:

```sh
ctest --test-dir cmake-build-tirx -R '^test_tile_system_' --output-on-failure
```

The CPU self-test is a host unit test; the Metal self-test is an explicit
hardware integration test. Both check every output against FP64 on four
shapes, including tails and repeated beta=0 calls starting with NaN output.
Add `--system-baseline cmake-build-tirx/bin/benchmark_tile_system` to `run.py`
to include the appropriate system baseline for each GEMM case; other
operations retain the existing two-implementation comparison.

For published comparisons, freeze valid CPU/Metal schedules in ordinary
`run.py` reports, then replay all three implementations in six balanced orders:

```sh
uv run --no-project --python 3.13 --with torch --with numpy python \
  scripts/benchmark/tile_torch/compare_system.py \
  --native cmake-build-tirx/bin/benchmark_tile_tirx \
  --system-baseline cmake-build-tirx/bin/benchmark_tile_system \
  --plan /path/to/cpu/results.json --plan /path/to/metal/results.json \
  --output /tmp/tile-system-comparison --rounds 6 \
  --samples 9 --sample-ms 40 --warmup-ms 200 --threads 8
```

The old reports supply **parameters only**, never timing scores. Each shape
gets all six permutations of Tile/PyTorch/system order; case order rotates.
All outputs use the same full FP64 check. Failures remain failures, not dropped
rounds, and binaries are hash-checked before/after the run. Raw timings,
individually synchronized latency, API/storage/stride metadata and correctness
errors are retained. Thread settings are requests, not measured library worker
counts. Compare device-specific baselines within each backend, not CPU BLAS
against Metal as if they used the same resources.

## Tile/PyTorch driver

For the LLVM path, `--cpu-stack-bytes 8192` enables the bounded
compiler-temporary stack realization. Zero (the default) retains workspace
allocation. The budget is cumulative with alignment padding, has a 65536-byte
ceiling, and does not override explicit Memory or permit pointer escapes.
It changes storage realization, not the tile geometry or arithmetic policy.
`repeat.py --candidate-cpu-stack-bytes 8192` changes only the candidate, so the
same frozen schedule and a pre-change binary can serve as the reference.
`compare_system.py --cpu-stack-bytes 8192` overrides CPU cases only and still
compares with eager PyTorch and direct Accelerate BLAS in six balanced orders.

`--auto-vectorize --cpu-vector-lanes 64` enables Cartesian CPU register packs.
The lane budget accepts 16/32/64/128; 16 preserves the single-row realization.
Larger packs keep separate contiguous row vectors inside a common serial
recurrence, exposing multi-row operand reuse without reassociating K. The
budget is not a hardware vector width. Unsupported regions keep the old
single-row packing, and CPU stack planning is independent.
`repeat.py --cpu-stack-bytes 8192 --candidate-cpu-vector-lanes 64` holds storage
fixed for both variants while changing only the candidate's pack budget.
`compare_system.py --cpu-vector-lanes 64` changes CPU cases only. Replay keeps
the exact reported policy; missing legacy lane metadata means 16, never an
inferred target-specific default.

`--cpu-input-views` independently enables proved immutable input expressions
on LLVM. Padded reads retain their original lazy guard and fill; explicit
Memory, mutable or aliased inputs, pointer escapes, and unproved consumer
indices keep snapshots. This is not MPP, and does not change its strict view
policy or the default reference lowering.
Use `repeat.py --cpu-stack-bytes 8192 --cpu-vector-lanes 64
--candidate-cpu-input-views` to hold both storage and register packing fixed
while changing only input forwarding. `compare_system.py --cpu-input-views`
applies to CPU cases only, preserving fresh Torch/BLAS comparisons. Keep
regressions: direct global reads can lose the benefits of compact packing.

`--cpu-matrix-backend cblas` is restricted to CPU GEMM with automatic root
binding. The bridge accepts it only for a proved whole compact rank-two FP32
`C=A*B` contract with noalias buffers and a registered
`tvm.contrib.cblas.matmul` provider. The native JSON must report
`cpu_matrix_backend="cblas"` and exactly one semantic external matrix call;
the runner rejects a missing or unexpected call. This is an actual Tile
lowering candidate, unlike `--system-baseline`, which is a separate executable
used only as a control. Keep both in reports when measuring provider overhead:

```sh
uv run --no-project --python 3.13 --with torch --with numpy python \
  scripts/benchmark/tile_torch/run.py \
  --native cmake-build-tirx/bin/benchmark_tile_tirx \
  --system-baseline cmake-build-tirx/bin/benchmark_tile_system \
  --output NEW_EMPTY_DIRECTORY --backends cpu --operations gemm \
  --cpu-model native --cpu-matrix-backend cblas \
  --samples 9 --sample-ms 40 --warmup-ms 200 --threads 8 --capture-sources
```

`--cpu-math-backend accelerate` is restricted to CPU but can be combined with
add/sum/softmax. It leaves add unchanged, realizes structurally proved
contiguous add/max/min reductions with vDSP, and realizes only a versioned
compiler-owned shared FP32 exp map with vForce. The static call-site diagnostic
is zero for add and nonzero for eligible sum/softmax. It can be larger than the
number of provider kinds when a small serial root is unrolled, so it is not a
dynamic call counter.

This policy permits provider reduction order and vForce denormal/floating-
exception behavior; reference is the default. Compare the two policies with
separate initial reports and `repeat.py`, which preserves each report's
`cpu_math_backend` field:

```sh
uv run --no-project --python 3.13 --with torch --with numpy python \
  scripts/benchmark/tile_torch/repeat.py \
  --reference /path/to/reference/results.json \
  --candidate /path/to/accelerate/results.json \
  --native cmake-build-tirx/bin/benchmark_tile_tirx \
  --output NEW_EMPTY_DIRECTORY --operations add,sum,softmax \
  --rounds 6 --samples 7 --sample-ms 40 --warmup-ms 200 \
  --threads 8 --capture-sources
```

Automatic CPU roots also report `cpu_parallel_task_threshold` (currently 64).
Below it, cheap automatic roots stay serial; explicit worker roots and bodies
with transcendental/opaque calls retain parallel mapping. This is a scheduling
prior, not a change to the source `parallel` semantics.

`--capture-sources` archives both LLVM IR (`.ll`) and Metal (`.metal`) by SHA256.
The repeat/system runners fingerprint both executables and their adjacent
shared libraries; use repeatable `--compiler-artifact PATH` arguments to cover
external TVM compiler/runtime libraries. Every replay checks the reported CPU
budget exactly; older reports without the field mean the old zero-budget path.

The [M1 Max CPU storage A/B](results/m1-max-20260904-cpu-stack-replay/notes.md)
and [six-order Torch/BLAS comparison](results/m1-max-20260904-cpu-stack-system/notes.md)
record the first measured realization. Its median gains over the old lowering
do not establish library parity; 512³ regresses in two A/B pairs and the policy
remains default-off. Raw LLVM, complete output checks, and artifact fingerprints
are retained alongside the reports.

First configure TVMx support as described in the Tile design document, and
complete the full build and correctness tests. The driver never builds or
changes the build configuration:

```sh
cmake --build cmake-build-tirx -j 8
ctest --test-dir cmake-build-tirx -L unit_tile --output-on-failure
uv run --no-project --python 3.13 --with torch --with numpy python \
  scripts/benchmark/tile_torch/run.py \
  --native cmake-build-tirx/bin/benchmark_tile_tirx \
  --output /tmp/tile-torch-results \
  --threads 8
```

The output directory must not exist. Use `--quick` for a smoke run only;
published comparisons should include the full default matrix. `--gemm-block
8,8,16` changes ordinary host configuration and therefore captures/JITs another
native variant. Record that setting, and do not silently select the best of a
larger tuning search against an untuned baseline.

An explicit search can recapture/JIT ordinary host configurations per shape:

```sh
uv run --no-project --python 3.13 --with torch --with numpy python \
  scripts/benchmark/tile_torch/run.py \
  --native cmake-build-tirx/bin/benchmark_tile_tirx \
  --output /tmp/tile-metal-jit-search \
  --backends metal --operations gemm --execution-scope group \
  --cooperative-matrix \
  --tune-gemm-blocks '8,8,16;16,32,32;32,32,32' \
  --tune-pipeline-windows '1,2'
```

Metal group searches can also specify `--tune-group-threads '128,256'` and
`--tune-copy-batches '1,4,8'`. These form a joint product with the block/window
lists; they do not alter the frontend kernel semantics. Zero in the thread
list invokes the planner's automatic choice, not a zero-thread launch.
Unspecified dimensions retain their ordinary command-line setting. Duplicate
configurations are removed; a product exceeding `--max-tuning-candidates`
(default 256 per shape) is rejected, never silently truncated. Include the
incumbent schedule in the lists when assessing an optimization.

No tuning happens by default. Every trial
uses the full FP64 correctness check; rejected candidates remain in JSON and
cannot win. Candidate order rotates across shapes. After selection, the driver
recaptures/JITs and measures the entire winning block/window/thread/copy
configuration again: the published table is that
fresh result, not the search minimum. Failed revalidation is never replaced by
an earlier favorable trial. Search cost (including validation/framework timing)
is separate from warm execution cost. This does not autotune PyTorch or
establish a globally optimal configuration.

On CPU, `--auto-vectorize` opts in to experimental independent-element SIMD
packing. It is off by default: the current heuristic has both wins and
substantial regressions. Packing uses semantic independence and preserves
inner serial/reduction order; it does not infer a new reduction permission or
narrow input precision. Explicit vector execution scopes still work without
this option. `--no-vectorize` disables TIRx vectorization altogether; it does
not disable LLVM's own optimizations. The two flags are mutually exclusive.
Reports record the executable hash and adjacent Tile/bridge shared-library
hashes separately, because rebuilding a dynamic bridge need not change the
executable itself. These hashes are not a complete runtime dependency trace.

Use `--backends metal --execution-scope group` for cooperative threadgroups,
or `--execution-scope worker` for one logical instance per worker. Both use the
same kernel source and block configuration; the requested mapping is recorded
and checked in every native result. `auto` retains the reference worker
mapping. CPU does not yet support `group` and rejects it explicitly. Run each
variant separately into a new output directory; do not change the block shape
when attributing a difference to execution mapping alone.

`--group-threads 128` supplies an exact Metal group worker-count constraint,
separate from `--threads 8` (the host/PyTorch CPU thread setting). Zero, the
default, lets the compiler planner choose. This supports controlled cost-model
ranking tests with the same TileIR and tile shape. The native report records
both the requested constraint and the realized `execution_plans`; the driver
checks their agreement. Repeated comparisons preserve the constraint. An
unsupported count fails rather than being silently clamped. The prior
reference worker launch width of 256 is not a hardware limit on group plans.

`--copy-batch 4` opts into up to four independent reads/computed values before
their shared-memory stores. The default, one, preserves the reference sequence.
This needs Metal group execution and is independent of the matrix capability.
Only supported compiler-owned shared destinations are transformed; external
writes and opaque effects retain their original lowering. Full worker chunks
are batched, while bounded accesses and the remainder remain guarded. There
is no async-copy, vector-alignment, or barrier-elision assumption. Native JSON
records `max_copy_batch` and `batched_copy_operations` per group; replay checks
and preserves the policy. The matrix score does not yet rank copy-batch sizes.

Use `--pipeline-window 1` and `--pipeline-window 2` for a same-binary GEMM
schedule comparison. Each ordinary host configuration captures/JITs a fresh
candidate: window 1 keeps ordered execution, while window 2 permits the safe
software-prefetch planner. The requested window is recorded in native output
and verified by the driver. Keep execution scope, tile shape, inputs, and
timing settings fixed, and run the two modes sequentially. This does not imply
hardware-asynchronous copies; the flag has no effect on non-pipelined operators.

Use `--backends metal --execution-scope group --cooperative-matrix` to opt in
to native FP32 SIMD-group matrices on a compatible device (Apple GPU family
7+, including this M1 Max). The flag is a device-capability assertion, not
automatic GPU detection; the helper supplies `thread_warp_size=32`. It is off
by default. Eligible M/N/K tile extents must be multiples of eight. Other
scopes, types, and shapes keep the reference loops; global ragged edges are
handled by the existing bounded Tile accesses. MMA allows reassociation at
the declared FP32 types, without TF32 or other input-precision reduction.

Compare runs with and without this flag using the same binary, block shape,
pipeline window, execution scope, and timing settings. Native output records
both the requested capability and the actual number of static
`simdgroup_multiply_accumulate` call sites in generated Metal source. The
driver checks that eligible cases select those instructions and fallback
cases do not. A call-site count is not a dynamic instruction counter.

### Metal SIMD-group reductions

`--metal-subgroup-reductions` opts automatic Metal sum, softmax and RMSNorm
into the proved FP32 add/max/min collective realization. It requires the
ordinary `simdgroup` TIRx path, automatic root execution, no cooperative
matrix option and only those operations. The option is also explicit
permission to replace the reference FP32 left fold with a tree order; it is
never inferred merely because the target supports SIMD collectives.

The planner enumerates one, two, four and eight SIMD groups per logical row.
One-group programs may be packed independently into a wider threadgroup;
multi-group programs use uniform SIMD collectives, small shared partial arrays
and a group barrier. Eligible reused compiler-owned Tiles are compacted to
worker-private stripes only after an affine access/ownership proof. Native JSON
records groups per program, whole-group threads, shared bytes, private stripe
size, reduction counts, barrier sites and the separately versioned
`metal_subgroup_reduction_v1` score.

Run the automatic planner with:

```sh
uv run --no-project --python 3.13 --with torch --with numpy python \
  scripts/benchmark/tile_torch/run.py \
  --native cmake-build-tirx/bin/benchmark_tile_tirx \
  --output NEW_EMPTY_DIRECTORY --backends metal \
  --operations sum,softmax,rmsnorm --metal-subgroup-reductions \
  --samples 11 --sample-ms 100 --warmup-ms 100 --capture-sources
```

`--group-threads 128` is an exact cooperating-worker constraint for this
realization. A measured staged/JIT sweep can instead use
`--tune-group-threads '32,64,128,256'`. Each width is separately captured,
compiled and fully validated; invalid trials remain in JSON. The measured
winner is then captured/JIT-compiled and measured again, so the reported row
is not a search minimum. GEMM block/pipeline and copy-batch tuning are rejected
for this reduction mode.

The saved [12-case report](results/m1-max-20260905-metal-subgroup-reductions/notes.md)
and [balanced RMSNorm A/B](results/m1-max-20260905-metal-subgroup-rmsnorm-replay/notes.md)
document exact plans, commands, hashes, complete errors and interpretation
limits. They are an M1 Max FP32 cohort, not all-device or production-LLM parity.

Measurement contract:

- Identical deterministic contiguous FP32 inputs; full outputs checked against
  a CPU FP64 reference with the same tolerances for both implementations.
- CPU thread environment is set before importing either framework. MPS or
  native Metal unavailability is an error, never a CPU fallback.
- Inputs and native outputs are allocated before warm measurements. PyTorch
  uses eager `out=` operations under inference mode where the operator exposes
  one. `torch.nn.functional.rms_norm` has no `out=` overload, so its returned
  output allocation remains inside warm timing and is recorded as
  `output_policy=framework_return_value`; other current operations report
  `preallocated_out`.
- Capture, native compilation, allocation/upload, first invocation, and
  download are reported separately. First-call timings are not a claim of an
  empty OS/driver cache, and PyTorch's process is reused across cases.
- At least 150 ms warmup, calibrated ~20 ms batches, 9 samples, p50/p90, plus
  individually synchronized latency samples. Native/PyTorch order alternates.
- Warm measurements use a host clock around dispatch plus synchronization.
  They exclude transfers but include C++/Python binding and launch overhead;
  they are **not pure GPU-event kernel timings**. Do not run alongside builds.
- The default native schedule is a reference realization. Explicit Metal
  `group` partitions independent elements across workers and shares
  group-owned compiler temporaries. The opt-in matrix selector replaces only
  proven compatible MMA bodies; hardware-asynchronous transfers and parallel
  reduction trees remain separate work. A correctness pass must not be
  described as competitive performance.

`results.json` contains raw samples, numerical errors, setup phases, compiler
and hardware information, thread settings, the binary hash, and source
revision. `results.md` is the readable comparison. Failed cases are retained
and cause a nonzero exit code; no speed ratio is published for an invalid case.

## Five-path Tile lowering comparison

`benchmark_tile_native` is the actual TileIR→Metal-backend→MPP route, launched
through Luisa Runtime. It is distinct from the handwritten `benchmark_tile_mpp`
probe. `benchmark_tile_tirx` remains the native C++ TVM bridge comparison;
neither benchmark calls MPS as a fallback. Build with both Metal and the TIRx
bridge enabled, complete the **full build**, and run correctness tests before
timing. Do not build or run other GPU work during measurement.

```sh
uv run --no-project --python 3.13 --with numpy --with torch python \
  scripts/benchmark/tile_torch/compare_lowerings.py \
  --native cmake-build-tirx/bin/benchmark_tile_native \
  --tirx cmake-build-tirx/bin/benchmark_tile_tirx \
  --mpp cmake-build-tirx/bin/benchmark_tile_mpp \
  --mps cmake-build-tirx/bin/benchmark_tile_system \
  --mpp-plan scripts/benchmark/tile_torch/results/m1-max-20260904-mpp-search/results.json \
  --tirx-plan scripts/benchmark/tile_torch/results/m1-max-20260904-joint-search/results.json \
  --rounds 10 --samples 7 --sample-ms 30 --warmup-ms 200 \
  --output /tmp/tile-five-path-replay
```

The plans supply configurations only; all binaries recapture/recompile and
measure afresh. Without plan arguments, explicit fixed defaults are recorded,
not described as autotuned winners. The default eight shapes cover small and
large squares, rectangles and tails; `--shape MxNxK` is repeatable. Full FP64
validation uses identical deterministic inputs and preallocated outputs.
Ten rounds balance positions and pairwise precedence. Fewer rounds are allowed
only as clearly labeled unbalanced smoke/exploration runs. All five paths run
even when another fails, and failures cause a nonzero exit code.

Tables compare **host-wall** batched throughput for all five paths, including
their different Runtime/API overheads; they do not claim pure GPU speedups.
MPS/handwritten-MPP GPU intervals remain in raw JSON only. Native/hand-MPP
ratios use matched descriptors/cohorts but include different host runtimes.
The current native subset is FP32, dynamic K, inline tensors and cooperative
output; an incompatible MPP plan is rejected rather than silently changed.

FP32 here specifies tensor types, not identical compiler math policies or
bitwise reproducibility. Native/handwritten MPP explicitly disable fast math
and relaxed precision; MPS uses `MPSKernelOptionsNone`, and Torch uses its
recorded default MPS route. This driver does not independently override or
validate the external TVMx Metal runtime's MSL fast-math flag. The deterministic
benchmark inputs are dyadic and cannot alone certify multiplication precision;
the separate kernel tests use non-dyadic inputs, transposes, tails and changed
buffers. Keep those tests and policy checks alongside performance validation.

### Same-source TIRx Runtime controls

The Metal backend now accepts `tile::Lowering::TIRX` through the same Tile
factory and ordinary Runtime shader/stream path as native MPP. The standalone
`benchmark_tile_tirx` accepts a final `tvm` (default), `luisa`, or `luisa-fast`
argument. The latter two compile unchanged TIRx-generated Metal source with
fast math explicitly off/on. They share the same `capture()` function and
planner options as the TVM runtime benchmark. No LLVM host wrapper is JITed
for the Luisa path. Only a single static launch with FP32 buffer arguments is
currently supported; unsupported host work is rejected rather than dropped.

Add `--tirx-runtime-controls` to `compare_lowerings.py` to measure all seven
paths. Its default becomes **14 rounds**, balancing positions and pairwise
precedence across seven implementations. Every TIRx path dumps its generated
source outside timing; the runner stores content-addressed sources and requires
identical SHA-256 plus matching threadgroup widths before accepting a Runtime
comparison. Whole-output FP64 validation still applies independently to every
path. Compiler language/resource options and submission APIs may differ, so
an equal source hash does not imply identical binaries or isolate launch cost.

```sh
uv run --no-project --python 3.13 --with torch --with numpy python \
  scripts/benchmark/tile_torch/compare_lowerings.py \
  --native cmake-build-tirx/bin/benchmark_tile_native \
  --tirx cmake-build-tirx/bin/benchmark_tile_tirx \
  --mpp cmake-build-tirx/bin/benchmark_tile_mpp \
  --mps cmake-build-tirx/bin/benchmark_tile_system \
  --mpp-plan scripts/benchmark/tile_torch/results/m1-max-20260904-mpp-search/results.json \
  --tirx-plan scripts/benchmark/tile_torch/results/m1-max-20260904-joint-search/results.json \
  --tirx-runtime-controls --rounds 14 \
  --output /tmp/tile-runtime-comparison
```

### Independent TIRx MPP codegen

`--tirx-mpp` adds a sixth path through TVM's own Metal MPP code generator,
beside native MPP, default TIRx SIMD-group, handwritten MPP, MPS and Torch.
It does not call the native emitter. Build the optional
[pinned C++ TVM extension](../../../src/tile/bridge/tirx/patches/README.md)
in isolation; normal TIRx builds need no patch. Complete the full build and
correctness checks before comparing. The standalone matrix-mode argument
`mpp` explicitly selects this realization; an unavailable extension is an
error, never a fallback.

Use the command above with the isolated binaries, `--tirx-mpp --rounds 12`,
and repeat `--compiler-artifact /path/to/library` for `libtvm_compiler.dylib`,
`libtvm_runtime.dylib`, `libtvm_runtime_metal.dylib` and `libtvm_ffi.dylib`.
Those externally linked libraries are hashed before and after the run.
The two TIRx paths use the same frozen geometry. The additional path must
report actual generated MPP call sites, zero SIMD-group MMA call sites, and
`cost_basis=metal_mpp_memory_v2`. That separately versioned relative-work score
uses MPP memory/shape features, subgroup critical-path work and an outer
program-wave prior. It is not an internal instruction count, measured register
or occupancy value, or calibrated time prediction. Do not merge its timings
into the SIMD-group cost model. With both optional comparisons enabled, eight
paths require sixteen rounds instead.

The [2026-09-04 six-path replay](results/m1-max-20260904-tirx-mpp/notes.md)
records 576 valid complete outputs, library/source fingerprints, the retained
1024³ regression versus original TIRx, and the remaining native/MPS gap.

### Read-only snapshot forwarding before resource planning

The optional TIRx MPP extension also supports a separately measured read-only
view-forwarding family. Use `run.py --matrix-realization mpp-views` together with
`--cooperative-matrix --execution-scope group --backends metal --operations gemm`
and explicit tuning lists. Every candidate is captured/JIT-compiled again and
validated; rejected capacities or padded full-K candidates remain in the report.
Forwarding requires proved immutable, noalias, bounded input snapshots. Manual
memory remains explicit. The `metal_mpp_memory_v2` basis remains an analytic
prior, not a calibrated time model; Staged/JIT measurements choose the frozen
schedule.

For the independent comparison, add `--tirx-mpp --tirx-view-plan SEARCH/results.json`
to `compare_lowerings.py`, with the ordinary `--tirx-plan` and external
`--compiler-artifact` fingerprints. This retains original TIRx, staged TIRx MPP,
native MPP, handwritten MPP, MPS and Torch and adds forwarding MPP as a seventh
path. Fourteen rounds balance order. Both requested forwarding policy and actual
MPP calls are checked. The view schedule is recorded separately; a geometry
change is not presented as a same-geometry lowering-only speedup.

The [seven-path replay](results/m1-max-20260904-tirx-views/notes.md) records
784 valid complete outputs. On this M1 Max, 512³ measured 43.523 µs versus
Torch's 48.794 µs; 1024³ measured 291.736 µs versus Torch's 291.133 µs and
MPS's 278.687 µs. These are synchronized host-wall batched times, not GPU
kernel durations. The large-GEMM library gap is reduced, not closed.

The newer [MPP cost-model v2 search](results/m1-max-20260905-mpp-cost-v2-search/notes.md)
separates subgroup critical-path work from whole-device waves. On the same
finite calibration cohort it reduces mean/median/max model regret from
74.18/43.05/239.58% to 8.82/2.59/34.37%; this is explicitly not a held-out
result. Its [independent 14-round replay](results/m1-max-20260905-mpp-cost-v2-replay/notes.md)
validates all 784 outputs and beats Torch and MPS on all eight tested FP32
GEMMs. At 1024³, TIRx MPP views measure 270.675 µs versus MPS at 272.572 µs
and Torch at 284.654 µs; the paired ratios are 0.9938× and 0.9513×. This does
not establish other-device, low-precision or non-GEMM parity.

Subgroup-fence elision is a separate, default-off tuning choice, not an
automatic consequence of forwarding. `run.py --elide-independent-subgroup-barriers`
selects it for `mpp-views`; `repeat.py --candidate-subgroup-fences elide` and
`compare_lowerings.py --tirx-view-subgroup-fences elide` select it explicitly
for their candidate/view path. Frozen reports preserve the policy. The native
benchmark records both the requested policy and each group's independently
proved `independent_subgroups` fact. Unsafe groups retain their fences even
when elision is requested. The first four-round A/B found a **512³ regression**,
so fewer barriers must not be assumed profitable or enabled by a shape table.

For compiler implementation A/B tests, `repeat.py --capture-sources` requires
both binaries to dump generated Metal, archives content-addressed sources and
records each command. The runner fingerprints both executables, every adjacent
runtime library and any explicit `--compiler-artifact` before and after timing;
changing an artifact fails the run. No JIT/compiler boundary uses Python source
generation: these Python programs only orchestrate benchmarks and oracles.

## Handwritten MPP versus MPS

`benchmark_tile_mpp` is a standalone hand-written Metal Performance Primitives
GEMM probe, not a TileIR lowering and not an MPS library fallback. It needs a
macOS 26 SDK/runtime and supported Apple GPU. Its default `tensor_inline`
representation is an ordinary buffer plus extents/strides; it uses the classic
tracked command queue, like `benchmark_tile_system`. An optional tensor-handle
mode uses Metal 4 with explicit dispatch barriers and commit-feedback checks.

The configuration is
`tile_m,tile_n,operation_simdgroups,cooperative_output,static_k,inline_tensors[,group_simdgroups,cohort_rows]`.
One operation can use a single subgroup or the whole group. Independent
single-subgroup operations can form a spatial cohort; the group need not have
the same size as the operation. Static M/N slices are used only inside the
matrix. Dynamic slices retain bounds for ragged output and K tails. Static K
requires a multiple of 16 for this FP32 family.

After a full build, screen a small family, then freeze and replay it:

```sh
uv run --no-project --python 3.13 --with numpy python \
  scripts/benchmark/tile_torch/compare_mpp.py \
  --mpp cmake-build-tirx/bin/benchmark_tile_mpp \
  --mps cmake-build-tirx/bin/benchmark_tile_system \
  --config 64,64,4,1,0,1 \
  --config 32,32,1,1,0,1,4,4 \
  --config 16,16,1,1,0,1 \
  --output /tmp/mpp-search

uv run --no-project --python 3.13 --with numpy python \
  scripts/benchmark/tile_torch/compare_mpp.py \
  --mpp cmake-build-tirx/bin/benchmark_tile_mpp \
  --mps cmake-build-tirx/bin/benchmark_tile_system \
  --plan /tmp/mpp-search/results.json \
  --rounds 6 --samples 9 --sample-ms 40 --warmup-ms 200 \
  --output /tmp/mpp-replay
```

This runner permits only FP32, no fast math or relaxed precision, and checks
every full output against the same FP64 oracle as the Tile comparisons. Search
selects by GPU batch time; its minimum is not a validated speedup. Replay
executes all six orders of MPS/default-MPP/selected-MPP for every shape, rotates
shape order, verifies binary stability and records all failures. Keep binaries
unchanged and do not build or run other GPU work during either campaign.

Both probes also report `gpu_throughput_us` and `gpu_latency_us`, in addition
to host wall times. These are Metal command-buffer GPU intervals; they include
dispatch/barrier work and are not individual arithmetic-instruction timings.
The existing TileIR/PyTorch timers remain host-wall timers.

The [M1 Max experiment](results/m1-max-20260904-mpp.md) records the candidate
family, numerical checks, frozen replay and remaining gap. These samples can
calibrate MPP-specific ranking only after a matching emitter exists; they must
not be pasted into the native 8x8 atom model as interchangeable issue costs.

## Repeat and profile

Freeze schedules from two reports and repeat them without selecting new winners:

```sh
uv run --no-project --python 3.13 --with torch --with numpy python \
  scripts/benchmark/tile_torch/repeat.py \
  --native cmake-build-tirx/bin/benchmark_tile_tirx \
  --reference /path/to/reference/results.json \
  --candidate /path/to/selected/results.json \
  --output /tmp/tile-schedule-repeats --rounds 4
```

Each case keeps the recorded block, pipeline, mapping, and matrix/vectorization
policies. The even number of rounds balances schedule order and framework order
separately, rotates shape order, and validates every fresh capture/JIT result.
Reports missing the explicit `auto_vectorize` policy cannot be replayed because
the old implicit behavior is ambiguous. `--candidate-vector-mode auto-vectorize`
can isolate packing against the same report passed as both inputs. Use
`--operations gemm,add,sum,softmax` when both reports contain those cases.
The paired min–max ratios are observed ranges, **not confidence intervals**.
Use `--candidate-native` for a second, separately prebuilt executable when
comparing implementation changes at frozen schedules. The report fingerprints
both executables and their adjacent Tile libraries; verify their dynamic-loader
paths before relying on a copied baseline. A build must never run during replay.

`profile_torch.py` provides a long, checked, preallocated eager GEMM workload
for an external profiler. `--backend cpu` uses the actual installed CPU build;
`--backend metal --signposts` enables MPS signposts without per-dispatch waiting.
`--mps-path metal` explicitly selects PyTorch's alternative Metal implementation;
it is not the default-MPS baseline. `--capture-dir /tmp/new-capture-directory`
captures one warmed invocation for launch/resource inspection. PyTorch chooses
the filename inside that new directory; the script verifies it exists and
records the actual path. Profiler timings are not
mixed into uninstrumented benchmark results. Record the exact PyTorch version
and build commit before studying its dispatch heuristics.

For the native executable, set `LUISA_TILE_BENCH_DUMP_SOURCE` to a **new file**
to inspect generated LLVM IR (CPU) or MSL (Metal). Dumping is outside all timed
phases, and never substitutes an LLVM host wrapper for Metal device code. The
normal positional arguments are unchanged. External profilers can launch the
same executable with longer `sample-ms` / `warmup-ms` arguments. Complete the
full build first, and do not profile while compiling or running other tests.
Filter exported trace rows to the target process: system GPU tracks can include
other applications even for a process-targeted recording. Do not publish those
unfiltered traces with benchmark reports. Encoder intervals may contain many
dispatches; they are not automatically per-kernel GPU durations.

## Native matrix measurements

The [M1 Max matrix comparison, 2026-09-03](results/m1-max-20260903-matrix.md)
records 24 validated pairs at `37e1337fd`: eight Metal cases with/without
matrix selection at identical 8×8×16 tiles and a two-window pipeline, plus
eight fresh CPU cases. Metal 1024³ improved from 4.618 to 3.244 ms (1.42×),
but remained 9.41× slower than PyTorch. CPU measured 12.425 versus 1.103 ms.
Raw generated-instruction counts distinguish actual matrix selection from
permission to select it. No autotuning was performed and the default is unchanged.

## Pipeline-window measurements

The [M1 Max pipeline comparison, 2026-09-03](results/m1-max-20260903-pipeline.md)
records all 48 GEMM pairs at `aeeee8eaf`: the same binary and 8×8×16 block
shape, two pipeline windows, and CPU worker / Metal worker / Metal group.
All numerical checks passed. CPU 1024³ improved from 37.870 to 12.577 ms
(3.01×), but Metal group moved from 4.334 to 4.439 ms and remained about
12.8× slower than PyTorch. The report includes every shape, raw samples,
compilation costs, and measurement limits; legality alone does not establish
pipeline profitability.

## Cooperative execution measurements

The [M1 Max execution comparison, 2026-09-03](results/m1-max-20260903-execution.md)
records all final worker/group cases at `2f8f2a044`, plus the earlier
experimental results and a CPU GEMM repeat. At identical 8×8×16 tile shape,
1024³ Metal GEMM improved from 14.725 ms worker to 4.275 ms group (3.44×), but
remained about 12× slower than PyTorch. The CPU loop-flattening regression was
substantially reduced by retaining axes; the largest CPU case still measured
48.226/38.508 ms in two runs, above the historical 31.571 ms observation.
This remaining performance issue is not considered closed. All numerical
checks passed, and no default mapping was changed based on these timings.

## Recorded reference baseline

The [M1 Max baseline, 2026-09-03](results/m1-max-20260903-reference/results.md)
contains all 40 CPU/Metal cases, with [raw samples and metadata](results/m1-max-20260903-reference/results.json).
The implementation is commit `8591c599d`; all 40 pairs passed the shared
numerical checks. This is one machine/run and one untuned native configuration,
not a hardware limit or a best-of-tuning result.

For FP32 1024³ GEMM, native/PyTorch warm batched times were **31.571/1.192 ms
on CPU** and **14.876/0.353 ms on Metal**: native was about 26.5× and 42.1×
slower, respectively. These measurements include host dispatch and predate the
explicit cooperative-group mapping. They are retained as a historical
reference, not a measurement of the current cooperative implementation.

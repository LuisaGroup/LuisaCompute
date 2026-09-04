# Direct Accelerate/BLAS and MPS baselines on M1 Max

## Outcome

The new direct-library baselines confirm a substantial CPU realization gap
and a smaller, still material large-square Metal gap. They also show why an
advantage over PyTorch is not automatically an equally large advantage over
the platform's native matrix API.

Six rounds × eight GEMM shapes × two backends produced **96 valid comparison
rows / 288 completely checked outputs**, all with maximum absolute error zero
against the shared FP64 oracle on this deterministic input set. There were no
failed or discarded rounds. Binary/library hashes were unchanged across the
run. This does not claim exact arithmetic for arbitrary FP32 inputs.

Times below are medians of per-round p50 synchronized host-wall batch times,
in microseconds. They include API/encoding/submission overhead, not compilation,
allocation or transfer. Ratios are medians of paired round ratios, **not**
ratios of the displayed medians; ranges are min–max, not confidence intervals.

| Backend / M×N×K | Tile µs | PyTorch µs | Direct BLAS/MPS µs | Tile / direct library [range] |
|---|---:|---:|---:|---:|
| CPU / 32×32×32 | 5.429 | 0.891 | 0.312 | 17.577× [11.769, 21.993] |
| CPU / 512×512×512 | 1474.382 | 142.577 | 137.491 | 10.694× [10.070, 11.853] |
| CPU / 1024×1024×1024 | 10208.031 | 1028.191 | 1112.837 | 9.494× [8.104, 10.146] |
| Metal / 32×32×32 | 5.352 | 28.603 | 9.756 | 0.555× [0.502, 0.578] |
| Metal / 128×128×128 | 12.190 | 28.774 | 14.314 | 0.839× [0.803, 0.888] |
| Metal / 512×512×512 | 53.771 | 47.645 | 50.643 | 1.061× [1.042, 1.088] |
| Metal / 1024×1024×1024 | 318.295 | 286.703 | 281.503 | 1.134× [1.005, 1.172] |
| Metal / 127×193×61 | 10.922 | 28.971 | 16.139 | 0.678× [0.638, 0.725] |

All sixteen backend/shape combinations, not just these rows, appear in the
[complete six-round table](m1-max-20260903-system-repeat/results.md) and
[raw results](m1-max-20260903-system-repeat/results.json).

For the tested fixed Metal schedule, six of eight shape medians are faster
than direct MPS. The wide 256×1024×128 case has a mixed round range
(Tile/MPS 0.897–1.018), so this is not six uniformly proven wins. Both 512³ and
1024³ remain slower in every measured round. CPU is slower in all eight
shapes, with large scheduling/timing variation retained in the ranges.

## What was compared

Hardware: Apple M1 Max, macOS 26.6.2. PyTorch 2.14.0, commit
`08187d9e0fba026dc8217405802ab5381dc88d90`, built with Accelerate. The host thread
requests `TVM_NUM_THREADS`, `OMP_NUM_THREADS`, and `VECLIB_MAXIMUM_THREADS` were
all 8; this is not a measurement of actual library worker counts.

- Tile CPU: fixed 4×16×32 tile, worker scope, pipeline window 2, experimental
  independent-element SIMD packing explicitly enabled. This is one replayed
  schedule, not a claim to have exhausted all CPU realizations.
- Tile Metal: fixed 64×64×32 tile, group scope, 256 workers, pipeline window 1,
  native FP32 SIMD-group matrix capability, copy batch 4. Existing proved
  accumulator residency/direct output and barrier coalescing remain enabled.
- Direct CPU: [Accelerate `cblas_sgemm`](https://developer.apple.com/documentation/accelerate/blas-library),
  classic LP64 API, row-major, no transpose, alpha=1, beta=0.
- Direct Metal: [`MPSMatrixMultiplication`](https://developer.apple.com/documentation/metalperformanceshaders/mpsmatrixmultiplication),
  FP32 matrices, `MPSKernelOptionsNone`, compact row bytes, private device
  buffers, one command buffer per timed batch. Inputs/output are allocated and
  uploaded before warmup; there is no timed upload, stride padding or prepack.
- PyTorch: eager `torch.mm(..., out=...)`, preallocated output, inference mode,
  CPU or default MPS backend. Optional MPS fast-math, prefer-Metal and CPU
  fallback environment overrides were cleared before importing PyTorch.

The direct-library executable links Foundation, Accelerate, Metal, MPS, and
system C++/Objective-C runtimes only (`otool -L` checked). It does **not** link
TileIR or TVM and is not a shortcut inside the native candidate's lowering.

All three implementations receive identical deterministic compact FP32 inputs
and pass the same complete FP64-reference comparison (atol=rtol=1e-4).
Warmup is at least 200 ms; nine samples use calibrated approximately 40 ms
batches. Nine individually synchronized latency samples and all setup/cold
phases are also retained, separately from warm throughput.

Each shape sees all six permutations of Tile/PyTorch/system order. Case order
rotates; CPU and Metal cases are interleaved. No build, test, profiler or
second benchmark ran alongside the timed measurements. Source schedules were
frozen before these rounds; their earlier timings were not used as scores.

## Implications for the planner and cost model

On CPU, PyTorch and direct BLAS have broadly comparable large-GEMM times,
whereas Tile remains about an order of magnitude behind. This is evidence
against blaming the gap primarily on Python dispatch. It does not identify
an AMX implementation or attribute the whole gap to any one pass.

For small Metal GEMM, PyTorch is substantially slower than the direct MPS
matrix API. The three paths may choose different internal kernels and have
different submission costs; subtracting their times is **not** a direct
measurement of Python overhead. Large-square Metal still needs a stronger
realization, even though several other shapes already beat both baselines.

Cost-model training must use our actual legal candidates and controlled
microbenchmarks. External-library times are performance targets, not proven
lower bounds or reachable members of our solver's search space. The next
structural work remains joint pipeline/resource planning on Metal and better
CPU microkernels, memory lifetimes and task granularity. More annealing or
integer-programming iterations cannot select a realization we cannot emit.

## Reproduce and validate

```sh
cmake --build cmake-build-tirx -j8
ctest --test-dir cmake-build-tirx -L '^unit$' --output-on-failure -j1
ctest --test-dir cmake-build-tirx -R '^test_tile_system_' --output-on-failure -j1
uv run --no-project --python 3.13 python -m unittest discover \
  -s scripts/benchmark/tile_torch -p test_run.py
uv run --no-project --python 3.13 --with torch --with numpy python \
  scripts/benchmark/tile_torch/compare_system.py \
  --native cmake-build-tirx/bin/benchmark_tile_tirx \
  --system-baseline cmake-build-tirx/bin/benchmark_tile_system \
  --plan scripts/benchmark/tile_torch/results/m1-max-20260903-system-cpu-pilot/results.json \
  --plan scripts/benchmark/tile_torch/results/m1-max-20260903-barrier-plan-64x64x32/results.json \
  --output /tmp/luisa-tile-system-repeat --rounds 6 \
  --samples 9 --sample-ms 40 --warmup-ms 200 --threads 8
```

Use a new output directory. Both plans contain all eight shapes; they record
parameters only. The CPU plan came from a fresh valid fixed-schedule pilot,
not a minimum selected from an expanded search.

Measured source base: `e32339be801d50aa39a64fd76cb3ffde37d0aa71`, with the
new benchmark/runner changes in the working tree. Production lowering code
matches that base. Measured artifact SHA-256:

- Tile benchmark: `589f8e1bf909e62bdce96bebb00abed4e484b418276a101684c5dc3cafdd49b8`
- System benchmark: `00b4fd1df00861676a5ba8fdb5d08e60d7b9ec14ab4c3ec21268bb1ffb25fe4b`
- TIRx bridge: `4da2a54ce640d79ca4c494bbf753b9277e248b7b323688b0c0424cc2f4bec211`
- Tile library: `2fca1d5b1db5249e66d9582f7ed9e1819863f2741009b00ed4c46b8de3b4de66`

Validation: complete build; final 138/138 unit-label tests (58.06 seconds);
direct CPU and Metal self-tests; per-file native syntax check; all 27 Python
benchmark-contract tests. Metal system validation is explicitly classified as
hardware integration, not a new default host-unit requirement. The
bridge-disabled complete build and all eight relevant Tile/system tests also
passed, confirming no TVM dependency in the new baseline.

## Separate, rejected short-loop experiment

Before adding these baselines, an experimental `ForKind::kUnrolled` on the
rectangular MMA K-atom loop when `K/8 <= 8` passed 27 Tile tests and all eight
GEMM pilot outputs. It did not justify a default change: the single pilot
reported 382.680 µs for 1024³ and 55.268 µs for 512³. This was **not** a paired
six-round experiment and is not evidence for an exact causal slowdown.
The source change was fully reverted before the BLAS/MPS comparison. Raw
[pilot results](m1-max-20260903-unroll-64x64x32/results.json) are retained;
no bounded-unroll policy was added to the planner.

# Native TileIR / TVMx vs PyTorch

This is an opt-in **correctness-checked, multi-shape** benchmark, not a CTest
performance threshold. It compares FP32 GEMM, add, row sum, and softmax on CPU
and actual Metal / PyTorch MPS. GEMM includes small/large squares, tall/wide
matrices, and non-multiple tail sizes; reductions vary both row count and width.

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

Use `--backends metal --execution-scope group` for cooperative threadgroups,
or `--execution-scope worker` for one logical instance per worker. Both use the
same kernel source and block configuration; the requested mapping is recorded
and checked in every native result. `auto` retains the reference worker
mapping. CPU does not yet support `group` and rejects it explicitly. Run each
variant separately into a new output directory; do not change the block shape
when attributing a difference to execution mapping alone.

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

Measurement contract:

- Identical deterministic contiguous FP32 inputs; full outputs checked against
  a CPU FP64 reference with the same tolerances for both implementations.
- CPU thread environment is set before importing either framework. MPS or
  native Metal unavailability is an error, never a CPU fallback.
- Inputs and outputs are allocated before warm measurements; PyTorch uses
  eager `out=` operations under inference mode, with no per-call allocation.
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

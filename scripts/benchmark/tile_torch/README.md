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

The candidate set is the product of those explicit block/window lists, with
duplicates removed. With just one tuning flag, the other setting remains at
`--gemm-block` / `--pipeline-window`. No tuning happens by default. Every trial
uses the full FP64 correctness check; rejected candidates remain in JSON and
cannot win. Candidate order rotates across shapes. After selection, the driver
recaptures/JITs and measures the winner again: the published table is that
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

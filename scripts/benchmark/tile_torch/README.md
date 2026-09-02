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
- The current native schedule is a reference realization. Semantic `mma`
  lowers to loops; tensor-core atom selection, cooperative distribution,
  asynchronous pipelining, and tuned reductions remain separate work. A
  correctness pass must not be described as competitive performance.

`results.json` contains raw samples, numerical errors, setup phases, compiler
and hardware information, thread settings, the binary hash, and source
revision. `results.md` is the readable comparison. Failed cases are retained
and cause a nonzero exit code; no speed ratio is published for an invalid case.

## Recorded reference baseline

The [M1 Max baseline, 2026-09-03](results/m1-max-20260903-reference/results.md)
contains all 40 CPU/Metal cases, with [raw samples and metadata](results/m1-max-20260903-reference/results.json).
The implementation is commit `8591c599d`; all 40 pairs passed the shared
numerical checks. This is one machine/run and one untuned native configuration,
not a hardware limit or a best-of-tuning result.

For FP32 1024³ GEMM, native/PyTorch warm batched times were **31.571/1.192 ms
on CPU** and **14.876/0.353 ms on Metal**: native was about 26.5× and 42.1×
slower, respectively. These measurements include host dispatch. The reference
lowering is not yet competitive; cooperative value distribution, matrix-atom
selection, and asynchronous pipeline scheduling remain implementation work.

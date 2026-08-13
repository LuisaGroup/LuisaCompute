# Standalone SIMD/ISPC comparisons

This directory contains opt-in performance controls for the SIMD CPU backend.
It is deliberately outside the project CMake graph and CTest suite. The driver
requires explicit paths to an existing LuisaCompute build and an ISPC compiler,
creates every ISPC object and host launcher in a temporary directory, validates
the output of every process, and then removes the temporary build by default.

The comparison algorithms are independently implemented in Luisa DSL and ISPC.
Mandelbrot and the masked-stream/AoS-to-SoA cases are based on the corresponding
ISPC v1.31.0 examples and micro-benchmarks. GEMM matches the repository's
row-major 256x256 DSL control. The analytic path tracer is a small asset-free
compiler comparison; it does not replace the repository's real Embree
path-tracing measurements.

ISPC v1.31.0 is BSD-3-Clause licensed. No ISPC source, generated object, compiler
path, CPU target, or benchmark executable is part of the production build.
The driver never reads an environment variable or CMake cache entry for ISPC:
`--ispc` is mandatory. It does not invoke CMake or write the supplied build.

The default comparison is precise: Luisa shaders explicitly disable fast math,
and ISPC uses its default math library with FMA contraction disabled. The
Mandelbrot, masked-stream, AoS-to-SoA, and GEMM outputs must be bit-exact across
all selected implementations. The analytic path tracer uses an explicit
absolute-plus-relative floating-point tolerance.

Example:

```bash
python3 scripts/benchmark/simd_ispc/run.py \
  --build-dir build-simd-fast \
  --ispc /path/to/ispc \
  --cpu znver5 \
  --workers 16 \
  --cpus 0-15 \
  --workloads mandelbrot,gemm,path_trace \
  --process-rounds 7
```

For stable measurements, pass an explicit CPU set and use several alternating
process rounds. The driver runs output validation in separate processes before
timing, then rotates and reverses implementation order between rounds. The
result JSON records the compiler versions, exact commands, host affinity, every
raw sample, output validation, paired ratios, and 95% confidence intervals.

Run the driver tests without an ISPC installation:

```bash
python3 scripts/benchmark/simd_ispc/test_run.py
```

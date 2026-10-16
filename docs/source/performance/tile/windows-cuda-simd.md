# Windows CUDA and SIMD Tile checkpoint

The initial October 1, 2026 matrix runs actual `tile::compile` kernels
through CUDA TIRx/PTX, experimental native CUDA Tile IR with a fixed
**16×16×8** tile, and XIR/SIMD. All **22 records passed**: 18 Tile
configurations and four library reference processes. Separate
[larger-tile](#follow-up-three-additional-native-tile-ir-schedules) and
[K-block](#follow-up-k-block-schedules) experiments bring the lowest
observed native 512³ time to **460.268 µs with 32×32×1**. The original
TIRx observation remains lower at 397.882 µs, from a different measurement
window. All three windows are retained below.
These bounded experiments do not establish a general backend or library
performance advantage.

## Platform and measurement

The machine has an Intel Core Ultra 7 155H (16 cores, 22 logical processors)
and an NVIDIA GeForce RTX 4060 Laptop GPU (SM 8.9), running Windows 11 Pro
for Workstations 10.0.26200. The Release build uses MSVC 14.51.36231,
LLVM 22.1.8, CUDA 13.4, CMake/Ninja, and the native C++ TIRx bridge;
`TVM_COMPILE_FORCE_FALLBACK` is unset. The native Tile IR route is enabled
only for its own processes with `LUISA_CUDA_TILE_IR=1` and `Lowering::NATIVE`.
See its [supported contract](../../internals/tile/cuda-workflow-b.md#6-experimental-native-cuda-tile-ir).

Each process starts suspended, receives affinity `0x15400` (logical CPUs
10, 12, 14, 16, four distinct physical P cores), and then resumes. SIMD uses
four CPU workers and width eight; OpenBLAS uses four threads. No clock or
power-plan settings are changed. Each configuration has 500 ms warmup,
seven calibrated batches targeting 100 ms each, and seven separately
recorded single-dispatch latency samples. Compilation, allocation, uploads,
readback, and numerical validation are outside the warm timing interval.

The tables report the **median synchronized host-wall time per GEMM**.
Luisa measures C++ command construction, submission, execution, and final
synchronization. Library references measure Python/NumPy or Python/ctypes
submission and synchronization; cuBLAS also records two events per batch.
These are different submission paths. Neither a library/Tile ratio nor a
CUDA event span containing host submission gaps establishes isolated GPU
kernel speed. This is one serial tuning pass, without repeated-process
ABBA confidence estimates.

## Actual Luisa strict FP32 results

All kernels compute `C = A * B` with FP32 storage and accumulation,
`enable_fast_math=false`, and no FP16/BF16/TF32 input conversion. The default
Tile MMA policy permits reassociation; this comparison does not require all
routes to use identical contraction order. Native Tile IR currently emits
ascending-K FP32 FMA with round-to-nearest-even and preserved subnormals.

TIRx and SIMD each test four Tile schedules: `(1,8,16)`, `(4,8,16)`,
`(8,8,16)`, and `(16,16,16)`. Their rows select the lowest measured median.
Native Tile IR tests only `(16,16,8)`, so its row is a fixed candidate rather
than a best-of-four result. The candidate counts refer to these external
Tile schedules, not the SIMD backend's internal mapping search.

| GEMM | Actual Luisa route | Selected tile M×N×K | Passed candidates | Median host-wall µs | Seven-batch range µs |
|---|---|---|---:|---:|---:|
| 128³ | CUDA TIRx → CUDA C → NVRTC PTX | 1×8×16 | 4/4 | 40.631 | 40.484–40.658 |
| 128³ | XIR → SIMD → LLVM | 8×8×16 | 4/4 | 89.429 | 76.579–107.599 |
| 128³ | CUDA Tile C++ → NVRTC Tile IR → cubin | 16×16×8 | 1/1 | 17.925 | 17.720–18.255 |
| 512³ | CUDA TIRx → CUDA C → NVRTC PTX | 8×8×16 | 4/4 | 397.882 | 397.617–408.360 |
| 512³ | XIR → SIMD → LLVM | 16×16×16 | 4/4 | 3057.770 | 2827.640–3260.810 |
| 512³ | CUDA Tile C++ → NVRTC Tile IR → cubin | 16×16×8 | 1/1 | 748.928 | 746.966–772.041 |

Native Tile IR takes 0.441× the selected TIRx host time at 128³ and 1.882×
at 512³. Different schedules and the small candidate set prevent treating
this as an intrinsic compiler-route ranking.

Every Tile configuration checks all 16,384 or 262,144 outputs against an
FP64 oracle, with zero nonfinite outputs and **maximum absolute error 0**.
Inputs are fixed integer multiples of 1/64; these dimensions make their
products and partial sums exactly representable in FP32. The runner also
requires error ≤ 1e-6, independently of the executable's looser legacy exit
tolerance. This timing fixture alone does not test general FP32 rounding.
Separate native runtime tests cover full-mantissa random inputs,
cancellation, ordered FMA, masks, aliases, views, and special bit patterns.

## Library brackets and drift

Each dimension is bracketed by fresh reference processes. cuBLAS uses
FP32 storage, `CUBLAS_PEDANTIC_MATH`, `CUBLAS_COMPUTE_32F_PEDANTIC`,
`CUBLAS_GEMM_DEFAULT`, and `NVIDIA_TF32_OVERRIDE=0`. The CPU reference is
NumPy 2.3.5 with OpenBLAS 0.3.30, its Haswell kernel, and four threads.
All reference outputs also have zero maximum error on the fixed fixture.

| GEMM | Reference | Before host-wall µs | After host-wall µs | After/before change |
|---|---|---:|---:|---:|
| 128³ | cuBLAS strict FP32 | 11.6266 | 11.6248 | −0.016% |
| 512³ | cuBLAS strict FP32 | 49.3398 | 50.2860 | +1.918% |
| 128³ | NumPy/OpenBLAS FP32 | 66.576 | 115.025 | +72.772% |
| 512³ | NumPy/OpenBLAS FP32 | 935.668 | 1236.253 | +32.125% |

Both CUDA routes take more host time than both cuBLAS brackets at these
shapes. SIMD at 512³ also takes more time than both OpenBLAS brackets.
The 128³ SIMD result lies between its two reference values: averaging that
large drift would create an unsupported small apparent win. These library
comparisons retain the different host-call overheads described above.

## Reproducing the selected kernels

Build with the CMake Tile/TIRx/CUDA/SIMD configuration and the CUDA 13.4
native helper described in the [CUDA workflow](../../internals/tile/cuda-workflow-b.md).
From the repository root, the selected rows use these CLI settings; required
CUDA and TVMx runtime DLL directories must be on `PATH`:

```powershell
cmake --build build-msvc-llvm
$bench = Join-Path $PWD 'build-msvc-llvm/bin/benchmark_tile_xir_gpu.exe'
$env:LUISA_SIMD_WORKER_COUNT = '4'
$env:LUISA_SIMD_WARP_WIDTH = '8'
$env:NVIDIA_TF32_OVERRIDE = '0'
Remove-Item Env:LUISA_CUDA_TILE_IR -ErrorAction SilentlyContinue
Remove-Item Env:TVM_COMPILE_FORCE_FALLBACK -ErrorAction SilentlyContinue
& $bench cuda 128 128 128 1 8 16 7 100 500 0 tirx
& $bench cuda 512 512 512 8 8 16 7 100 500 0 tirx
& $bench simd 128 128 128 8 8 16 7 100 500 0 native
& $bench simd 512 512 512 16 16 16 7 100 500 0 native
$env:LUISA_CUDA_TILE_IR = '1'
& $bench cuda 128 128 128 16 16 8 7 100 500 0 native
& $bench cuda 512 512 512 16 16 8 7 100 500 0 native
Remove-Item Env:LUISA_CUDA_TILE_IR
```

These commands reproduce kernels and sample settings. Matching the measured
environment also requires the suspended-process affinity setup, serial
execution, clean benchmark environment, all four TIRx/SIMD schedules, and
before/after library brackets. The local evidence packet retains those
steps in `.deps/oct01-native-runtime-performance.py` and all 22 records in
`.deps/native-runtime-performance/results.json` (14:51:50–14:53:12 UTC+8,
October 1). These ignored local artifacts are not distributed with the
repository. The packet records exact commands, source/binary SHA-256 hashes,
raw samples, realization strings, and the prior native correctness report;
it rejects changed runtime inputs.

## Separate handwritten CUDA Tile API probe

An earlier same-machine API experiment passed all ten records, including
six handwritten kernel/shape combinations. It bypasses Luisa's emitter and
runtime. Its FP32 FMA kernel uses a 32×32 output tile with K step one;
FP16/BF16 MMA uses K step 16, converts inputs inside the kernel, and
accumulates in FP32. All timings below include Python/ctypes submission.

| GEMM | Handwritten FP32 FMA µs | Handwritten FP16 MMA µs | Handwritten BF16 MMA µs | Strict cuBLAS before→after µs |
|---|---:|---:|---:|---:|
| 128³ | 29.710 | 11.171 | 10.769 | 11.093→11.591 |
| 512³ | 423.015 | 185.524 | 186.067 | 49.245→49.401 |

The 12 random/cancellation checks cover 1,671,168 outputs and pass their
respective FP64 oracles. Narrow modes are checked against **quantized**
operands. For 512³ random inputs, maximum error against original FP32
operands is 3.37e-5 for FP32 FMA, 8.97e-3 for FP16, and 7.41e-2 for BF16.
Consequently, lower MMA timings are not strict-FP32 speedups and must not
be presented as Luisa native-runtime results. cuBLAS bracket drift is
+4.49%/+0.32% at 128³/512³; this earlier packet's OpenBLAS drift is
−36.85%/−19.98%, also unsuitable for stable CPU superiority claims.
The separate local packet is `.deps/oct01-native-tile-performance-run2`.

## Follow-up: three additional native Tile IR schedules

A separate run at 15:00:33–15:01:10 UTC+8 on October 1 tests three new
native schedules per shape. All **10 records passed**: six actual Luisa
native Tile IR configurations and four library reference processes. It
keeps the same strict FP32 fixture, seven 100 ms target samples, 500 ms
warmup, and four-P-core affinity. Every Tile output is finite and has zero
maximum error against FP64. The original 22-record matrix above is retained
without replacing its schedules or pooling measurements across runs.

| GEMM | Native tile M×N×K | Median host-wall µs | Seven-batch range µs |
|---|---|---:|---:|
| 128³ | 16×32×16 | 64.732 | 64.700–65.209 |
| 128³ | 32×32×16 | 59.550 | 59.480–59.587 |
| 128³ | 64×64×16 | 40.576 | 40.424–40.651 |
| 512³ | 16×32×16 | 3413.070 | 3411.550–3413.600 |
| 512³ | 32×32×16 | 2302.080 | 2301.360–2302.700 |
| 512³ | 64×64×16 | 663.668 | 663.019–664.471 |

The best of these **3/3** candidates is 64×64×16 at both shapes. The older
16×16×8 result is 17.925 µs at 128³ and 748.928 µs at 512³. Matching
executable/backend/helper hashes establish the same implementation, but
these are different measurement windows: the historical comparison is not
a controlled speedup estimate. The new candidates do not improve on the
old small-shape observation. The new 512³ observation also exceeds the
earlier TIRx result of 397.882 µs, subject to the same different-window
limitation, and still leaves a large gap to its own cuBLAS brackets.

| GEMM | Reference | Before host-wall µs | After host-wall µs | After/before change |
|---|---|---:|---:|---:|
| 128³ | cuBLAS strict FP32 | 10.7096 | 11.6958 | +9.208% |
| 512³ | cuBLAS strict FP32 | 49.0995 | 49.4378 | +0.689% |
| 128³ | NumPy/OpenBLAS FP32 | 105.579 | 48.815 | −53.765% |
| 512³ | NumPy/OpenBLAS FP32 | 901.043 | 1000.375 | +11.024% |

All library output checks pass with zero maximum error. The same host-call
and baseline-drift qualifications apply; no isolated GPU-kernel ratio is
inferred. The local packet `.deps/native-runtime-tuning/results.json`
retains all six candidates, their before/after references, and a hashed
snapshot of the earlier native results. Replay these kernels with the
previous CLI, replacing the three tile arguments with each row's schedule
and keeping `LUISA_CUDA_TILE_IR=1`, variant `0`, and lowering `native`.

The emitter's FP32 path loads operand tiles, then loops over K, extracts
one column and row, and performs one broadcast FP32 FMA per contraction
element. It does not select `ct::mma` for FP32, and the pipeline emits an
ordinary ordered loop. Increasing the K tile therefore does not reduce
the total K FMA steps or introduce hardware MMA. All three new candidates
use K=16 and also change M/N versus the old candidate; isolating the effect
of K requires a separate same-M/N experiment.

## Follow-up: K-block schedules

The 15:04:05–15:04:46 UTC+8 run on October 1 adds four native schedules
per shape, including a same-M/N comparison of K=1 and K=4. All **12 records
passed**: eight actual Luisa native configurations and four library
reference processes. It uses the same binary guards, strict FP32 fixture,
four-P-core affinity, 500 ms warmup, and seven batches targeting 100 ms.
All eight configurations validate every output as finite with zero
maximum FP64 error. Earlier tables remain separate historical observations.

| GEMM | Native tile M×N×K | Median host-wall µs | Seven-batch range µs |
|---|---|---:|---:|
| 128³ | 16×16×1 | 19.224 | 19.165–19.274 |
| 128³ | 32×32×1 | 28.827 | 28.290–29.394 |
| 128³ | 64×64×1 | 54.839 | 54.723–54.883 |
| 128³ | 32×32×4 | 20.402 | 20.244–20.421 |
| 512³ | 16×16×1 | 736.184 | 720.517–740.267 |
| 512³ | 32×32×1 | 460.268 | 452.061–460.705 |
| 512³ | 64×64×1 | 490.794 | 489.803–491.238 |
| 512³ | 32×32×4 | 670.401 | 669.785–670.818 |

Within this run, the best of **4/4** candidates is 16×16×1 at 128³ and
32×32×1 at 512³. Holding M/N at 32×32, K=1 takes 460.268 µs versus K=4's
670.401 µs at 512³; at 128³ the ordering reverses (28.827 versus 20.402 µs).
K blocking matters for these shapes, but smaller K is not universally
better. This serial observation does not identify the machine-level cause
or establish a statistically controlled improvement over earlier windows.

| GEMM | Reference | Before host-wall µs | After host-wall µs | After/before change |
|---|---|---:|---:|---:|
| 128³ | cuBLAS strict FP32 | 10.6199 | 11.7722 | +10.851% |
| 512³ | cuBLAS strict FP32 | 49.7551 | 49.8300 | +0.151% |
| 128³ | NumPy/OpenBLAS FP32 | 103.414 | 113.712 | +9.958% |
| 512³ | NumPy/OpenBLAS FP32 | 1141.153 | 1137.083 | −0.357% |

The libraries also pass all output checks with zero error. Even the best
native candidate remains above both cuBLAS brackets. Different C++ and
Python/FFI call overheads still prevent interpreting these host-wall
numbers as isolated GPU-kernel ratios. The evidence packet is
`.deps/native-runtime-k-tuning/results.json`; replay uses the same native
CLI and each row's tile arguments. No earlier measurements are pooled
into its four-candidate selection.

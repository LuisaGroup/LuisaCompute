# M1 Max: inspect the actual PyTorch paths

2026-09-03, Apple M1 Max, macOS 26.6.2, Xcode Instruments 16.0
(17F113), PyTorch 2.14.0 at
`08187d9e0fba026dc8217405802ab5381dc88d90`. The workload is contiguous FP32
`1024 × 1024 × 1024` eager `torch.mm`, with resident inputs, preallocated
output, warmup, and full FP64-reference validation. CPU thread settings are 8.

## Dispatch is not a single PyTorch tile-size table

The installed CPU wheel reports `BLAS_INFO=accelerate`. Its eligible FP32 GEMM
path calls BLAS `sgemm_`; the Time Profiler confirms execution inside Apple's
`libBLAS.dylib`, rather than a Python loop. After the first 3 seconds of the
trace, 8,295 ms of 8,547 ms sampled running **self weight** is in libBLAS
(97.05%). These are summed samples across threads, not elapsed latency.
Private BLAS symbols were unresolved, so this does **not** establish a
particular AMX microkernel. See the pinned
[PyTorch CPU dispatch source](https://github.com/pytorch/pytorch/blob/08187d9e0fba026dc8217405802ab5381dc88d90/aten/src/ATen/native/CPUBlas.cpp).

For the eight GEMM shapes in this benchmark, the default MPS path uses
MPSGraph matrix multiplication. Rank-one GEMV and the alternative Metal-MM
path have separate dispatch conditions. The alternative `do_metal_mm` uses
`TILE_DIM=16`, but that is **not** the selected default path here. The actual
MPSGraph implementation and its internal blocking are inside Apple's
framework. `PYTORCH_MPS_PREFER_METAL`, fast-math, and MPS fallback overrides
were unset. See the pinned
[PyTorch MPS dispatch source](https://github.com/pytorch/pytorch/blob/08187d9e0fba026dc8217405802ab5381dc88d90/aten/src/ATen/native/mps/operations/LinearAlgebra.mm).

## What the GPU capture actually shows

A warmed single-invocation PyTorch Metal capture was replayed and profiled in
Xcode. Its command list contains one compute encoder and one dispatch:

| Property | Captured default MPS GEMM |
|---|---|
| Shader | `NDArrayMatrixMultiplyNNA14` |
| Threadgroups | `32 × 16 × 1` = 512 |
| Threads per threadgroup | `128 × 1 × 1` = 4 SIMD groups |
| Dynamic threadgroup memory | 4,096 + 8,192 bytes = 12 KiB |
| Input/output device buffers | Three buffers, 4 MiB each |

Dividing the output area by 512 gives 2,048 output elements per group. A
`32 × 64` block is therefore a **hypothesis**, not recovered shader source:
neither the M/N orientation nor BK is uniquely determined by the launch and
memory sizes. In particular, 12 KiB alone cannot distinguish a particular BK
from a buffered smaller BK.

Selected counters from this **single replay at Xcode's Medium performance
state**:

| Counter | Value |
|---|---:|
| GPU replay time | 499.81 µs |
| ALU utilization | 76.83% |
| F32 utilization | 74.05% |
| Kernel occupancy | 18.39% |
| Threadgroup memory bytes read | 384.00 MiB |
| Device memory bytes read | 59.52 MiB |
| LLC miss rate | 32.44% |

Occupancy and ALU utilization are different metrics. This kernel can keep
its ALUs busy without high occupancy; copying its thread count alone is not
an optimization strategy. Register counts were not available in the viewed
pipeline-statistics panel. The replay time is **not** substituted for warm
uninstrumented timings, and these MPS-only counters do not quantify a native
kernel's bottleneck.

## Native evidence and the resulting change

The pre-change native `16 × 32 × 32`, window-1 GEMM uses 2,048 groups of
256 threads. Its generated MSL allocates 12 KiB of threadgroup arrays:
2 KiB carry, 2 KiB A, 4 KiB B, 2 KiB MMA result, and 2 KiB yield snapshot.
Each K chunk loads A/B, performs SIMD-group MMA, and copies the materialized
result through the extra snapshot before updating the carry. The source has
five threadgroup barriers per K chunk, plus initialization and final-store
barriers: 162 executions per group for 32 K chunks, before Metal compiler
optimization. This is a source-level count, not a hardware stall counter.

`2e0172700` removes the snapshot when the yielded expression reads no
compiler-owned carry allocation. All dependent expressions are still
snapshotted before **any** carry is overwritten. Mixed updates and swaps
retain simultaneous SSA semantics; no external-buffer noalias assumption or
FP reassociation was added. This is a general yield lowering improvement,
not a GEMM-specific DSL primitive.

The post-change generated MSL confirms 10 KiB of arrays and four barriers per
K chunk (130 per group including initialization/final-store barriers), with
the same matrix atom and launch. Separately dumped CPU and Metal `1024³`
outputs both matched the FP64 oracle exactly. Source-dump smoke timings are
not included in the performance tables.

The change is measured separately with old/new executables at identical
frozen schedules in [the four-round A/B report](m1-max-20260903-repeat-metal-yield/results.md).
Larger blocks are a separate search experiment, not part of that causal A/B.
Remaining design work includes accumulator storage reuse/register residency,
dependence-aware barrier placement, and target-aware execution/resource
planning. Merely matching MPS's launch dimensions does not provide those.

The native CPU profile also confirms parallel worker execution. Its JIT
frames are largely unresolved; after the first 2 seconds, main-thread
`switch_pri` self samples account for about 12% of total running sample
weight. Only about 1.6% of that weight is on efficiency cores in this trace.
Neither observation explains all run-to-run variation. The inspected TVMx
`SetThreadAffinity` implementation is Linux/Android-only; `TVM_BIND_THREADS`
does not pin Apple performance cores in this build.

The generated CPU LLVM IR retains runtime workspace allocation/free calls
inside the logical output-tile loop. Snapshot removal deletes one allocation,
but does not establish a complete worker-local scratch plan. This is a
concrete follow-up for CPU lowering, not a measured attribution of the entire
gap to Accelerate. The four-round CPU binary A/B has mixed signs for every
shape and establishes no reliable speedup.

## Reproduction and evidence boundaries

Use [profile_torch.py](../profile_torch.py) under Time Profiler or Metal
System Trace; `--signposts` adds MPS signposts. For a single warmed launch,
use `--backend metal --capture-dir /tmp/new-capture-directory`. The helper
passes a basename because PyTorch prefixes a counter and appends `.gputrace`,
then records the actual generated path. The corrected path handling was
verified with a checked `32³` capture. The native benchmark's
`LUISA_TILE_BENCH_DUMP_SOURCE=/tmp/new-source-file` records generated LLVM IR
or device MSL outside timing.

Raw traces and captures remain locally under
`/tmp/luisa-tile-profile-20260903-yDw53g`. Metal System Trace can include other
processes: only the benchmark/python process was selected for this analysis,
and unfiltered system traces are deliberately not committed. The first
PyTorch CPU launch failed because a transient uv interpreter path was reused
after its environment exited; the successful replacement launched xctrace
from inside the live uv environment. Empty CLI GPU-counter exports were not
treated as measurements; the counters above came from Xcode capture replay.
GPU replay was stopped before subsequent uninstrumented benchmarks.

Pre-change generated MSL SHA-256:
`8bdaea66d92cbb5efd563db63623d52bf2bae05dac423fb05a2bd7de55975ada`.
Pre-change CPU LLVM IR SHA-256:
`ea9ced628f46cc633981e4e78c121ce14a2b14e3fadec1eaab888c24e165fcb1`.
Post-change generated MSL SHA-256:
`bd9d5b9ca4fd9231ef2f484b8a828cdaa00da4741f82fb72ada6eb9ad6bb38d3`.
Post-change CPU LLVM IR SHA-256:
`74a9e6130103dd95d2f16a79456b10efa69d543ada3a7769c207c27e8e377e3c`.

TVMx source inspected: `c7b458e946bc4266915da582457476bdcd9705ae`.

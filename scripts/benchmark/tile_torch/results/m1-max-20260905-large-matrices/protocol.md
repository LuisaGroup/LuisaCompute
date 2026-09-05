# Larger-matrix coverage protocol

Recorded before the new cohort's timings. This is a **scale-generalization
test of existing realizations**, not a new coefficient fit, parameter search,
low-precision claim, or an extrapolation from the old 1024-cubed results.

## Fixed workloads and policies

Metal, Apple M1 Max, FP32, compact row-major, no transpose/prepacking. GEMM
uses alpha=1, beta=0, preallocated output and these M×N×K shapes in order:

```text
2048x2048x2048
4096x4096x4096
8192x8192x8192
256x11008x4096
4096x4096x11008
2049x4097x1025
```

The squares expose scaling beyond the previous 1024-cubed ceiling. The two
rectangles exercise projection-like aspect ratios/long K; they are individual
matrix operations, not an end-to-end model benchmark. The final case checks
large ragged M/N/K boundaries. Keep every shape, failure and regression.

Seven GEMM paths remain separate: native Tile→MPP, ordinary Tile→TIRx Metal
SIMD-group matrix lowering, handwritten MPP, direct MPSMatrixMultiplication,
eager Torch MPS, TIRx→MPP without input forwarding, and TIRx→MPP with proved
input views. Native and handwritten MPP keep the prior 1024-cubed control's
`32,32,1,1,0,1,4,4` configuration. Ordinary and non-forwarding TIRx keep
32×32×32, 128 threads, ordered pipeline, copy batch 1. Forwarding TIRx transfers
the previously measured 1024-cubed winner unchanged: 128×32×1024, 128 threads,
ordered pipeline, copy batch 1. This is **not** automatic optimality at the new
shapes. No timing at a new shape selects its schedule.

The view path first runs a six-case parameter/correctness pilot with three
host samples, 10 ms windows, 100 ms warmup and source capture. Freeze that
report's exact schedule for a **14-round** seven-path comparison: every
implementation occupies every position twice, with balanced pair precedence
and rotating case order. Use five samples, 20 ms windows and 100 ms warmup;
300-second subprocess timeout. Pilot timings are not acceptance results.

Separately test softmax, RMSNorm and affine LayerNorm at these rows×widths:

```text
37x8191,1024x8192,1024x16384,4096x8192,8192x4096,16384x4096
```

Retain the automatic analytic execution family, V=4, ordered U=1, input views,
opt-in immutable input caching, preserved shared Tile SSA and the existing
64-scalar private bound. No forced S=2/P=4 extrapolation to wide rows. Run
`run.py` once in the listed order and once in its reverse, with five samples,
30 ms windows and 100 ms warmup. For each operator/shape this reverses both
case order and native/Torch precedence. Two rounds establish coverage and
initial scale trends, not a broad stable-performance acceptance claim.

## Measurement and correctness

Continue using the existing independently checked benchmark binaries;
fingerprint native executables, adjacent Luisa libraries, the Metal timing
helper and externally linked TVM/FFI artifacts before and after timing.
Unrelated local C++ changes are not rebuilt into this experiment. Report that
source/binary boundary explicitly; this run adds no new CTest claim.

Check **every output element**, not a checksum or sampled subset, against the
existing FP64 oracle and unchanged operator tolerances. The deterministic
input family is unchanged, which does not prove arbitrary-distribution
accuracy. Allocation, transfers, compilation and the CPU oracle remain
outside warm timing. Run one GPU workload at a time, no concurrent builds,
tests or profilers. Do not tune from the first failures or stop a comparison
because one path loses.

Measure batched and individually synchronized host-wall E2E separately from
no-counter GPU command-buffer throughput/single-call intervals. Shared
compute-pass instrumentation remains a diagnostic; it is not the reported
GPU control. Handwritten MPP retains its direct command-buffer timestamp
implementation. These GPU intervals include command-buffer work/gaps, not an
isolated instruction/kernel timestamp. Never subtract independent medians or
mix host and GPU ratios. Native/handwritten MPP disable fast/relaxed math;
TVM Metal's current fast-math behavior is unchanged. Torch GEMM/softmax use
preallocated `out=`; functional norms return newly allocated output in timing.

The machine has 64 GiB unified memory. Largest resident GEMM A+B+C payload is
768 MiB; largest reduction input/output pair is 512 MiB. FP64 oracle and host
temporaries add several GiB outside timing. Retain memory-pressure failures
as failures; do not silently shrink dimensions or switch dtype/device.

## Reporting

Publish scale results in the existing `docs/source/performance/tile/` section,
not the language guide or a parallel website. Keep raw JSON, generated source
identities, round ordering, complete-output validation records, all four
timing metrics, and observed ranges. The broader MPS/Torch performance goal
remains open unless the expanded evidence actually supports it.

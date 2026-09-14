# Rejected TIRx fragment-load batching experiment

The large-GEMM gap remains open. Loading two or four consecutive K-atom
steps into distinct A/B fragments before issuing their MMA operations does
not improve 1024³. The experimental implementation, option, and tests have
been **reverted**; the delta is retained in [experiment.patch](experiment.patch).
Neither the native MPP path nor the existing independent TIRx path is replaced
by a library call. The [seven-way comparison](../m1-max-20260904-runtime-controls/notes.md)
remains the cross-implementation baseline; this run compares TIRx schedules
only and does not remeasure MPS, MPP, or Torch.

## Frozen, same-binary experiment

Eight existing joint-search schedules are frozen. The only additional tuning
input is a fragment-load batching ceiling of 1, 2, or 4. The emitter uses the
largest divisor of its K-atom count under that ceiling and retains ascending
K accumulation order. The planner accounts for the extra live fragment state
and independently checks resource bounds. No asynchronous copy is claimed.

All **144/144 complete outputs** pass the same FP64 oracle (`atol=rtol=1e-4`).
Three rotations and their reversals balance positions and pairwise precedence for the
three variants. Every trial uses fresh capture/JIT, 200 ms warmup, and seven
30 ms samples. No builds, tests, or profiler ran during measurements. All
samples, including slow ones, remain in [results.json](results.json).

The table is synchronized **host-wall batched throughput**, in µs/call, not
GPU time. Each time is the median of six per-round medians. Ratios are medians
of paired round times, so they need not equal a ratio of table medians.

| M×N×K | Batch 1 | Batch 2 | Batch 4 | Paired batch 2 / 1 | Paired batch 4 / 1 |
|---|---:|---:|---:|---:|---:|
| 32³ | 4.813 | 5.040 | 4.929 | 1.050 | 1.016 |
| 128³ | 6.584 | 6.511 | 6.474 | 0.990 | 0.946 |
| 512³ | 53.372 | 52.938 | 54.226 | 0.992 | 1.018 |
| 1024³ | 319.196 | 333.904 | 376.091 | 1.046 | 1.178 |
| 256×1024×128 | 19.060 | 19.850 | 19.160 | 1.028 | 0.987 |
| 1024×128×256 | 20.117 | 20.037 | 19.528 | 1.005 | 0.975 |
| 127×193×61 | 8.742 | 8.491 | 8.245 | 0.973 | 0.951 |
| 513×257×129 | 22.432 | 22.600 | 22.190 | 1.006 | 0.993 |

1024³ regresses in every round: about **4.6%** with batch two and **17.8%**
with batch four by paired ratios. Some smaller cases have nominal wins, but
this experiment is not an independent replay of selected shape-specific
winners. No default or new public planner option is promoted from it.

For 1024³ all variants retain BM=BN=64, BK=32, 256 threads, a 2×4 subgroup
grid, and 16 KiB shared storage. Dynamic logical counts remain 8192 MMA atom
issues and 6144 shared fragment transfers per group. Estimated live fragment
scalars per lane grow from **28 → 40 → 64**; these are representation counts,
not measured physical registers or proof of spilling/occupancy. The observed
regression does not identify a hardware bottleneck by itself.

## Provenance and verification

- All 24 content-addressed Metal sources are retained in `sources/`, and their
  hashes were verified. All eight batch-one source hashes exactly match the
  previous seven-way Runtime-controls campaign.
- The benchmark binary and its recorded adjacent shared-library hashes stayed
  unchanged throughout timing. This is not a hermetic bundle of external TVMx,
  system frameworks, or the dirty working tree.
- All variants run through the independent TVM runtime, whose local Metal
  compiler requests fast math on. Input precision is FP32, with no intentional
  reduced-precision option. The dyadic benchmark values cannot alone certify
  multiply precision; the separate matrix tests use non-dyadic sine inputs.
- Full selected CMake builds preceded tests and timing. CPU/Metal matrix and
  planner CTests passed with Luisa/Metal validation before measurement. The
  experimental test includes 42 shape/batch/window combinations, changed
  inputs, all input transpose combinations, and ragged M/N/K.
- The first experimental test incorrectly expected exactly one MMA site for
  every configuration. BK=8/window=2 fits the existing eager pipeline version
  budget and has steady-state and drain sites even at batch one. The corrected
  test checks both sites and their fragment accounting; no numerical tolerance
  was relaxed. The test delta is preserved in the patch along with the emitter.

The run used Luisa HEAD `b8c3c54f81f2a4ad947e295f1f75e57605bf8833` plus the
existing Runtime/TIRx working changes and the saved experimental delta. The
patch is relative to that working state, **not** a complete patch from HEAD.
The exact executed [runner](probe.py) is retained and its hash matches the raw
report; its workspace path is machine-local. Reproduction requires applying
the experimental patch and completing a full build. Do not pass its removed
benchmark argument to the restored production executable.

The inspected TVMx source checkout is
`c7b458e946bc4266915da582457476bdcd9705ae`. Its `cooperative_tensor_*` builtins
are declared and registered, but the inspected Metal codegen implements only
the `simdgroup_*` matrix family. Those declarations do not establish a working
TIRx→MPP route. Extending that realization family is a separate implementation
task; relabeling native MPP as a TIRx comparison would not satisfy it.

All six experimental source files were restored byte-for-byte to their
pre-experiment snapshots. The saved patch passes `git apply --check` against
that restored working state. A subsequent full selected CMake build succeeded;
**23/23 selected CTests** passed with Luisa/Metal validation (native Runtime,
CPU/Metal TIRx, and CPU BLAS/Metal MPS baselines), and **41/41** benchmark-script
unit tests passed. `git diff --check` also passed.

The two existing Metal cooperative/memory source-assertion conflicts with the
unowned `mem_flags(3)` → `mem_flags(2)` hunk remain excluded from this selected
run. Neither that hunk nor those assertions was modified; this is not a claim
that the entire CTest suite is green.

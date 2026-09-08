# CPU TIRx versus Torch and Accelerate: remaining gap

With the new 8192-byte compiler-stack policy, CPU TIRx is still slower than
Torch and direct Accelerate BLAS on every measured GEMM shape. At 1024³ the
median times are **11.492 ms / 1.062 ms / 1.019 ms** respectively. Paired
slowdown medians are **10.841× versus Torch** and **11.267× versus BLAS**.
The allocation improvement does not close the performance goal.

All eight shapes use the same frozen 4×16×32, worker, window-two, auto-vectorized
schedule as the [controlled before/after comparison](../m1-max-20260904-cpu-stack-replay/notes.md).
Only the candidate is measured here, with six rounds providing all six orders
of TIRx/Torch/BLAS for each shape. There is no parameter search or best-round
selection. Each run uses seven 30 ms samples, 200 ms warmup, and eight requested
CPU threads; actual library worker counts and processor clocks are not measured.
Warm timings include dispatch and kernel-internal resource management, not
capture/JIT or input/output allocation and transfers.

- **144/144 complete outputs pass**, covering 33,798,528 elements. Maximum
  absolute error is zero on these dyadic inputs against the FP64 oracle.
- All 20 executable/library artifact paths remain unchanged and were
  independently rehashed after the run. All 48 raw LLVM archives pass their
  filename SHA256 checks; none contains a workspace allocation call. Raw
  TBAA identity names differ between JITs and are deliberately preserved.
- Inputs/outputs are compact row-major FP32, alpha=1, beta=0, with no transpose,
  prepacking, mixed precision, or library substitution in the TIRx path. BLAS is
  the explicitly labeled classic LP64 `cblas_sgemm` baseline. The exact Torch
  build/configuration and environment are recorded in the raw report.
- No build, test, profiler, shader validation, or other agent-run benchmark
  overlaps the timing session. This does not eliminate normal OS or user
  workload variation; earlier historical absolute times are not a controlled
  comparison with this session.

The generated LLVM still has separate K loops per output row and explicit
operand staging. These are concrete next investigation points, not a proof
that fixing them alone reaches library throughput or that a particular
undocumented CPU matrix unit explains the whole gap. Multi-row register
microtiles, K-resident accumulation, and task/cache granularity need their own
legal realization families and measured cost models.

Full eight-shape tables are in [results.md](results.md); all samples,
correctness results, commands, policies, artifact hashes, and implementation
orders are in [results.json](results.json).

The independently maintained Metal paths and their MPS/MPP/Torch comparisons
remain available; see the prior
[seven-path Metal run](../m1-max-20260904-subgroup-sync-lowerings/notes.md).
This report is a CPU experiment and does not claim new GPU performance numbers.

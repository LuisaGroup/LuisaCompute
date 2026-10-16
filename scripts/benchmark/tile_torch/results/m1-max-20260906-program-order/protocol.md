# Native TIRx program traversal × K partition

The new traversal is a bounded bijection over the last two axes of independent
Metal group programs; earlier axes remain separate batches. It does not
move effects within a program, change allocation/ownership, or assume an
execution order between groups. The default remains row-major. All traversal
requests are explicit JIT constraints, not an automatically calibrated model.

After a full build, run layout/planner tests, CPU/Metal execution, matrix,
basic/neural/algorithm PoCs, pipeline, memory, cooperative and native runtime
regressions. Validate the generated traversal independently by enumerating
actual rectangle partitions, with partial rectangles and overflow rejection.

Predeclared exploratory performance cohort: FP32 compact row-major GEMM on
512³, 4096³, 8192³, 4096×4096×11008, 2049×4097×1025 and 257×769×113.
Fix output block 128×64, 256 threads, the existing 4×2 subgroup map of 32×32
outputs, pipeline window one, copy batch one, input views and retained fences.
Cross K blocks {512,4096} with program rectangles {1×1,2×4,4×8,8×16}.
These correspond to row-major and 256²/512²/1024² local output regions.

Two rounds rotate then reverse configuration and shape order. Five samples,
20 ms requested sample windows and 100 ms warmup per timing phase. Eager Torch
MPS and direct MPS matrix multiplication are fresh controls in every call;
the standard driver rotates their ordering independently. Record no-counter
GPU command-buffer batch/single intervals and separate host E2E batch/single
wall times. These are not isolated instruction/kernel timings. TVM's existing
fast-math setting is unchanged; no low-precision or strict cross-library math
equivalence claim is made.

Every output is preallocated and validated in full against the same FP64
dyadic oracle, atol=rtol=1e-4. Save all raw samples, MSL, exact plans, failures,
and before/after compiler/runtime/driver hashes. No slow or failed case is
removed, no block is substituted on rejection. Desktop activity is not
controlled and no other task-owned benchmark, build or GPU profiler runs
concurrently. A screen minimum is not an accepted result: any chosen plan
needs separate fresh-JIT, frozen, balanced replay. Do not fit the analytic
cost coefficients to these two-round labels or change the default traversal.

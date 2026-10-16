# MPP execution-scope diagnostic

Record the candidate family before timing. This experiment screens a missing
realization family, not new default coefficients or a performance acceptance.

Use the existing handwritten strict-FP32 MPP binary and direct MPS control.
Compact row-major C=A×B, alpha=1, beta=0, no transpose, prepacking, relaxed
precision or fast math. Test 1024³, 4096³, 8192³, 256×11008×4096,
4096×4096×11008, and 2049×4097×1025. Keep all failures and slow candidates.

Fixed configurations (M,N,operation-SG,cooperative-output,static-K,
inline-tensor,group-SG,cohort-M):

```text
32,32,1,1,0,1,4,4
128,32,4,1,0,1,4,1
64,64,4,1,0,1,4,1
32,128,4,1,0,1,4,1
64,32,2,1,0,1,2,1
```

The first two cover the same 128×32 group output domain with four subgroups:
four independent 32×32 operations versus one collective 128×32 operation.
This isolates a candidate scope choice without claiming that the MPP compiler
uses any particular cache or staging strategy. The other rectangles check
whether that comparison transfers across atom aspect and participation width.

Use `compare_mpp.py` search mode, five samples, 20 ms target windows, 100 ms
warmup, and a 300-second per-process timeout. Finish the full selected build
first. Run one GPU process at a time, without builds/tests/profilers during
timing. Validate every output against the unchanged FP64 oracle
(atol=rtol=1e-4). The deterministic dyadic inputs do not establish arbitrary
input-distribution accuracy.

Report host-wall batch/single-call separately from no-counter GPU
command-buffer batch/single-call intervals. GPU intervals are not isolated
instruction timestamps. Allocation, compilation, transfer and validation are
untimed. The script fingerprints both native executables before/after the
run. A search minimum is only a reason to implement/test the realization;
any acceptance needs a separately frozen, counterbalanced replay.

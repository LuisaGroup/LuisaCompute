# Independent program partition checkpoint - 2026-10-03

This checkpoint covers the strict SUM/MAX independent-program partition in commit `0154c82ec`, measured on an RTX 4060 Laptop GPU (SM89, 24 SMs), CUDA 13.4, MSVC and LLVM 22. The native kernel retains FP32 arithmetic and the original FP32/FP16/BF16 storage contract. Candidate source appends a partition entry while preserving the complete default source prefix; the shared host selector uses the actual bound pointer spans to select the candidate or the original entry **and grid**. Benchmark entry/grid receipts predict that selector; separate correctness tests observed actual graph template CUfunctions/grids.

## Failure provenance and measurement protocol

The original 16-case default cohort remains **failed**: one Torch compile process exited with Windows `C000070A` (`STATUS_THREADPOOL_HANDLE_EXCEPTION`). All 16 original native runs passed. A separate rescue queue ran rows1, rows2 and default recheck, followed by one independent native+Torch retry of the failed case. The new summary has status `completed_validated_native`; it does not change the original queue/cohort status. There are 64 main native executions plus one native retry, 15 initial Torch executions plus one later successful Torch retry, and **567 retained raw samples**. The missing initial Torch timing stays missing.

Each process used four P cores (mask `0x15400`), four host threads, seven adaptive graph samples targeting 100 ms each, 500 ms warmup and 100 operations per graph. All samples, replay counts, actual spans, calibration and warmup receipts, cold timings and full saved-output validation receipts are retained in `validation.json`. Native fixed outputs and Torch functional allocation are different API contracts; these are graph operation medians, not isolated kernel measurements. Initial Torch is `torch.compile` with the same exported inputs/oracle; candidate phases did not rerun Torch. Default recheck drift ranged from -0.537% to +0.303%.

## Observed medians

Times are microseconds per operation. B is the original rows per program; candidates are 1 or 2 rows per program. The table includes the regressions at 1024x512.

| Operation / shape / storage / B | Default | Rows1 | Rows2 | Recheck | Initial Torch | LOGO selection |
|---|---:|---:|---:|---:|---:|---|
| reduce_sum 17x1024 fp16 B4 | 2.1989 | 0.9235 | 1.3005 | 2.1985 | 0.9790 | rows1 |
| reduce_max 17x1024 fp16 B4 | 2.1641 | 0.9298 | 1.4173 | 2.1639 | 0.9132 | rows1 |
| reduce_sum 65x2048 fp16 B8 | 6.3262 | 1.1212 | 2.4183 | 6.3253 | missing | rows1 |
| reduce_max 65x2048 fp16 B8 | 6.3705 | 1.1252 | 2.4180 | 6.3740 | 1.2673 | rows1 |
| reduce_sum 128x2048 bf16 B8 | 1.6550 | 1.3524 | 1.4814 | 1.6461 | 2.1370 | rows1 |
| reduce_max 128x2048 bf16 B8 | 1.6631 | 1.3524 | 1.4863 | 1.6642 | 1.3326 | rows1 |
| reduce_sum 65x512 fp32 B8 | 1.3280 | 1.0062 | 1.0506 | 1.3320 | 1.0007 | rows1 |
| reduce_max 65x512 fp32 B8 | 1.3000 | 0.9978 | 1.0414 | 1.3005 | 1.0280 | rows1 |
| reduce_sum 1024x512 fp16 B4 | 2.7475 | 3.1183 | 3.0472 | 2.7525 | 2.5812 | default |
| reduce_max 1024x512 fp16 B4 | 2.7265 | 2.9568 | 2.7930 | 2.7198 | 2.5984 | default |
| reduce_sum 37x257 fp32 B4 | 0.9954 | 0.9343 | 0.9495 | 0.9964 | 0.9200 | rows1 |
| reduce_max 37x257 fp32 B4 | 0.9766 | 0.9581 | 0.9398 | 0.9768 | 0.9180 | rows1 |
| reduce_sum 3x8191 bf16 B4 | 4.8974 | 1.3986 | 2.4232 | 4.9037 | 1.9676 | rows1 |
| reduce_max 3x8191 bf16 B4 | 5.5860 | 1.3840 | 2.4215 | 5.5842 | 1.3850 | rows1 |
| reduce_sum 256x128 bf16 B8 | 1.2780 | 1.2373 | 1.2701 | 1.2792 | 0.9310 | rows1 |
| reduce_max 256x128 bf16 B8 | 1.3363 | 1.2012 | 1.1803 | 1.3389 | 0.9347 | rows1 |

The separate retry of SUM 65x2048 FP16 measured Torch **1.2831 us**; its new native median differed by +0.016% from the original native median. This later reference is stored separately and is not substituted into the initial Torch column.

The observations do not show a universal Torch win. For example, selected BF16 256x128 remains about 1.20 to 1.24 us versus initial Torch about 0.93 us, and the retained-default 1024x512 cases remain slower than Torch. The 1.89% MAX 37x257 and 3.18% SUM 256x128 improvements are small changes, not robust gain claims.

## Fixed three-parameter diagnostic model

For valid rows R, padded contribution width W and rows per program b, let P=ceil(R/b). The proposal was fixed before these measurements:

`q = a0 + aP*P + aD*ceil(P/24)*b*W`

`a0=0.9028899747245084`, `aP=0`, `aD=0.00011327031090119965`.

These are nonnegative least-squares coefficients. Logical batches and collective volume are not measured occupancy, physical registers or memory traffic. No peak live-state estimate is invented. The model has no operation-name, benchmark-ID, dtype or algebra term; those unmodeled effects remain visible in the residuals.

Training uses 48 observations (16 cases x default/rows1/rows2), with equal total weight per `[R,N,B,W]` geometry. All algebra/storage variants and schedules of each geometry stay together in leave-one-geometry-out validation: **8 groups**, not 16 independent shapes. Recheck samples diagnose drift and have no fitting weight. Feature scaling is computed only from each training fold. The solver enumerates all eight nonnegative active subsets and retains feasible objectives and rejected subsets.

The decision considers only original, rows1 and rows2; rows4 has not been calibrated. Strictly more than 5% lower predicted score selects a candidate. Ties involving the original retain it; equally improving rows1/rows2 candidates prefer rows1. Both full-fit and leave-one-geometry-out decisions choose rows1 for 14 cases and retain the two 1024x512 defaults. The geometry-equal-weight observed policy/default geometric mean is **0.56988747**, with no observed selected regressions. This is cross-validation on the calibration inventory, **not independent held-out GPU validation**.

The model is an uncalibrated ranking proxy. Its maximum leave-one-geometry-out relative timing residual is **+82.06%** (BF16 MAX 128x2048 rows1), and maximum absolute residual is **3.898 us**. Every residual, fold model, decision and alternative candidate is retained. Good choices here do not establish accurate runtime prediction or a confidence guarantee. The optional automatic backend policy and its independent validation are outside this frozen calibration result.

## Files and reproduction

- `validation.json`: all 64+1 native and 15+1 Torch observations, unchanged numerical contract and saved-output audit, original failed process, source/fixture hashes, entry/grid receipts and cold timings.
- `samples.csv`: all 567 samples, with original case/cohort status and distinct retry labels.
- `model.json`: full-fit and fold coefficients, training-only scales/weights, active-subset objectives, residuals and decisions.
- `proof-receipts.json`: compact relevant source/runtime/build receipts and the hash of the original complete frozen inventory. No binaries, tensor outputs or installation sources are bundled.
- `provenance.json`: original summary/model/script hashes and profile ID. Those hashes refer to original local bytes; the redacted public files have independent hashes in `receipts.json`.

From this directory, with Python and NumPy 2.3.5:

```powershell
python -m pip install -r requirements.txt
python reproduce.py
python -m unittest discover -s code -p test_calibrate.py
```

Reproduction verifies public hashes, sample medians and retained proof consistency, then refits the model and every leave-one-group-out fold. The local export reproduced the entire numerical projection exactly. The verifier permits only 1e-12 floating-point variation for other NumPy/BLAS builds; discrete decisions remain exact. The public profile ID binds redacted profile bytes and is distinct from the original provenance profile ID. Rechecking the original full output oracle requires the original tensor artifacts or a new benchmark run; this package verifies their recorded audit receipts and does not pretend to contain those tensors.

All public text uses LF, enforced by the local `.gitattributes`, so declared hashes remain stable across Windows Git checkout. Absolute user/toolkit paths and the GPU UUID are redacted. The original failed record, negative schedule results and independent retry are retained.

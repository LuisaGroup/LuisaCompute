# CUDA TIRx subgroup tuning: three closed experiments

These are three independent native-only experiments, each with six geometries and control/candidate/control-recheck cohorts. The packet retains **54 native case executions, 378 primary graph samples and 5,400 observed graph nodes**. No Torch timing denominator, earlier performance sample, U64 experiment or failed startup is substituted into these results.

Unroll V2 compares U1/U8/U1 with lane8. Its BF16 tail RMS case improves about 30%, while most other differences are small. Lane V1 compares lane8/lane1/lane8 at U8: five cases improve, but BF16 tail RMS regresses about 32% and reports 304 local bytes instead of zero. Lane V1 used the experimental **narrow second-level tree**. The later independent A/B/A comparison holds lane8/U8 fixed and changes captured bridge implementations: A uses the original full-warp tree; B uses the narrower tree. B regresses large softmax by 12.7% versus initial A and 15.7% versus A recheck, despite improving some smaller cases. **The narrow-tree candidate was withdrawn.** Lane V1 is evidence for further testing, not validation of lane1 on the restored full tree or justification for automatic deployment.

All runs use fixed fast elementwise math, preserved FP32 reduction merges, T128/P2, cache=false, seven samples targeting 100 ms, at least 500 ms graph warmup, graph batch100 and the recorded four-logical-CPU affinity. Existing Tile IR and lowering options express these candidates; no DSL or execution-nest primitive was added. Changing lane ownership may change an authorized unordered SUM's association. Unrolling preserves each lane's recurrence order. Neither fact equates arithmetic implementations with Torch.

Run `python reproduce.py` in this directory. The standalone standard-library replay checks every bundled file hash, all samples and span normalization, medians/ratios/control drift, exact source/control identity, emitted schedule markers, fixture/oracle receipts, all graph nodes and their real function/binding/grid/block relationships, and compiled resources. For A/B it also checks captured bridge identities against actual process module observations and the two archived source versions. All losses remain in the tables.

Input/output/reference/bound tensors and DLL binaries are **not bundled**. The original closed summaries retain full saved-output FP64-bound checks, runtime guards and readonly checks with their file hashes. Replay validates those retained records; it does not redo omitted tensor arithmetic, execute GPU work or authenticate omitted DLL bytes. Driver registers/shared/local/max-thread attributes are compiled facts, not measured occupancy, spill traffic, or causal explanations. A small time difference is not a robust gain claim.

`index.json` distinguishes original raw SHA/byte receipts from portable public copies. Sources and CSV are byte-exact, protected by `.gitattributes`; complete JSON records have path prefixes replaced by `${REPO}`, `${USER_TEMP}` or `${USER}` and compact formatting. These tokens are provenance labels, not locations the replay attempts to access. Original build/runtime library paths and source inventories remain recorded, but rebuilding requires the compatible full repository/dependencies. A/B source inventories were assembled after DLL capture from archived source and unchanged common files, not represented as contemporaneous compiler traces. Its separately retained failed V1 startup and recovery smokes are outside these formal denominators.

Results are never pooled across the three experiments. Their controls, runtime source snapshots and bridge provenance differ. U64 was unfinished when this packet was prepared and is deliberately absent.


## unroll-v2

Control / candidate / recheck: `unroll1` / `unroll8` / `unroll1_recheck`. Ratios below one favor the candidate.

| Case | Control µs | Candidate µs | Recheck µs | Candidate/control | Candidate/recheck | Control drift |
|---|---:|---:|---:|---:|---:|---:|
| regroup-rmsnorm-128x1024-bf16 | 3.907927 | 3.921505 | 3.919938 | 1.003475 | 1.000400 | 1.003074 |
| regroup-rmsnorm-256x1536-fp16 | 8.236458 | 8.236288 | 8.235179 | 0.999979 | 1.000135 | 0.999845 |
| regroup-rmsnorm-7x2051-bf16 | 3.533802 | 2.459659 | 3.531187 | 0.696038 | 0.696553 | 0.999260 |
| regroup-softmax-1024x512-fp16 | 6.924122 | 6.890799 | 6.895787 | 0.995187 | 0.999277 | 0.995908 |
| regroup-softmax-32x512-fp16 | 1.578540 | 1.575700 | 1.577695 | 0.998201 | 0.998735 | 0.999465 |
| regroup-softmax-9x769-fp16 | 1.949035 | 1.931525 | 1.950069 | 0.991016 | 0.990491 | 1.000531 |

## lane-v1

Control / candidate / recheck: `lane8` / `lane1` / `lane8_recheck`. Ratios below one favor the candidate.

| Case | Control µs | Candidate µs | Recheck µs | Candidate/control | Candidate/recheck | Control drift |
|---|---:|---:|---:|---:|---:|---:|
| regroup-rmsnorm-128x1024-bf16 | 3.897016 | 1.455849 | 3.920096 | 0.373580 | 0.371381 | 1.005922 |
| regroup-rmsnorm-256x1536-fp16 | 8.403627 | 2.396753 | 8.394612 | 0.285205 | 0.285511 | 0.998927 |
| regroup-rmsnorm-7x2051-bf16 | 2.437932 | 3.209894 | 2.434205 | 1.316646 | 1.318662 | 0.998471 |
| regroup-softmax-1024x512-fp16 | 7.620135 | 3.410708 | 7.615385 | 0.447591 | 0.447871 | 0.999377 |
| regroup-softmax-32x512-fp16 | 1.483847 | 1.036472 | 1.487234 | 0.698503 | 0.696912 | 1.002283 |
| regroup-softmax-9x769-fp16 | 1.839438 | 1.103964 | 1.844408 | 0.600164 | 0.598546 | 1.002702 |

## partial-tree-ab-v1

Control / candidate / recheck: `A_initial` / `B` / `A_recheck`. Ratios below one favor the candidate.

| Case | Control µs | Candidate µs | Recheck µs | Candidate/control | Candidate/recheck | Control drift |
|---|---:|---:|---:|---:|---:|---:|
| regroup-rmsnorm-128x1024-bf16 | 3.906966 | 3.914677 | 3.939934 | 1.001974 | 0.993590 | 1.008438 |
| regroup-rmsnorm-256x1536-fp16 | 8.234922 | 8.414129 | 8.282181 | 1.021762 | 1.015932 | 1.005739 |
| regroup-rmsnorm-7x2051-bf16 | 2.465623 | 2.439261 | 2.472696 | 0.989308 | 0.986478 | 1.002869 |
| regroup-softmax-1024x512-fp16 | 7.106485 | 8.012231 | 6.926178 | 1.127453 | 1.156804 | 0.974628 |
| regroup-softmax-32x512-fp16 | 1.583158 | 1.495576 | 1.588290 | 0.944679 | 0.941627 | 1.003241 |
| regroup-softmax-9x769-fp16 | 1.933915 | 1.851215 | 1.939405 | 0.957237 | 0.954527 | 1.002839 |

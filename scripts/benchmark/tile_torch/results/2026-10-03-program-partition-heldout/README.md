# Program partition: independent heldout validation (2026-10-03)

This checkpoint tests the already frozen three-parameter partition policy on twelve heldout SUM/MAX cases (six geometries). It does not refit or filter the policy using these results. The training and leave-geometry-out record remains in the sibling `2026-10-03-program-partition` checkpoint.

The unchanged fit is `63e0677c8554b46b707aa9f1fca3f29f223c653b1ec35fd57b5534b79595b6c2`. It selected a smaller program row extent for 10 cases and retained the original for 2. Geometry-weighted policy/default ratios are 0.345372 initially and 0.345115 on repeat. These are local measurements, not a calibrated guarantee.

Two original Torch processes exited with Windows `C000070A` during compilation. The initial default cohort remains failed. All 48 native executions were independently revalidated. The first rescue later failed before GPU execution because its generated one-case JSON omitted `schema: 1`; that failure and its log receipt remain preserved. A separate corrected continuation reran native plus Torch for those two cases, producing 50 native and 12 successful Torch executions in total. The two later Torch values never replace the missing initial denominators.

| Case | Default us | Policy us | Repeat us | Recheck us | Repeat / recheck | Initial Torch us | Later retry Torch us | Decision |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| heldout-sum-31x1024-fp16-br8 | 3.5307 | 0.9653 | 0.9702 | 3.5189 | 0.2757 | 0.9522 | missing | selected, rows 1 |
| heldout-max-31x1024-fp16-br8 | 3.2880 | 0.9663 | 0.9708 | 3.2859 | 0.2954 | 0.9599 | missing | selected, rows 1 |
| heldout-sum-97x4096-bf16-br4 | 3.9028 | 1.5596 | 1.5570 | 3.9031 | 0.3989 | missing | 1.5611 | selected, rows 1 |
| heldout-max-97x4096-bf16-br4 | 3.9318 | 1.5643 | 1.5733 | 3.9394 | 0.3994 | 1.8557 | missing | selected, rows 1 |
| heldout-sum-511x256-fp32-br4 | 1.4595 | 1.4673 | 1.4624 | 1.4733 | 0.9926 | 1.4193 | missing | retained, rows 4 |
| heldout-max-511x256-fp32-br4 | 1.5154 | 1.5169 | 1.5000 | 1.5177 | 0.9883 | 1.4252 | missing | retained, rows 4 |
| heldout-sum-11x3073-fp16-br4 | 4.9471 | 1.0616 | 1.0616 | 4.9535 | 0.2143 | 1.0854 | missing | selected, rows 1 |
| heldout-max-11x3073-fp16-br4 | 5.0111 | 1.0578 | 1.0545 | 5.0125 | 0.2104 | 1.3884 | missing | selected, rows 1 |
| heldout-sum-257x128-bf16-br8 | 1.1320 | 1.2389 | 1.2373 | 1.1315 | 1.0935 | missing | 1.0159 | selected, rows 1 |
| heldout-max-257x128-bf16-br8 | 1.1360 | 1.2016 | 1.2008 | 1.1325 | 1.0603 | 1.1392 | missing | selected, rows 1 |
| heldout-sum-9x8192-fp32-br8 | 14.9013 | 1.3616 | 1.3619 | 14.9263 | 0.0912 | 1.8433 | missing | selected, rows 1 |
| heldout-max-9x8192-fp32-br8 | 28.5711 | 1.3362 | 1.3318 | 28.6144 | 0.0465 | 1.8345 | missing | selected, rows 1 |

Selected cases exceeding the matched default recheck by more than 5% in both policy passes: `heldout-sum-257x128-bf16-br8`, `heldout-max-257x128-bf16-br8`. These negative heldout observations are retained. The policy is not retroactively changed.

Every case uses the original strict FP32-compute arithmetic and quantized input/storage oracle, including full saved-output checks, input read-only checks and output guards. The public projection preserves all 434 raw graph samples, medians, adaptive replay counts and spans, warmup, cold phases, model scores/decisions, source hashes, fixture hashes, and launch-selection receipts.

Protocol: seven samples, 100 ms target per sample, 500 ms graph warmup, 100 logical operations per graph, fixed four P-core affinity (`0x15400`), four host threads. Native-only policy/recheck/repeat phases have no independent Torch timing. Native buffer reuse and functional Torch output allocation differ; timing is complete-operation graph throughput, not an isolated instruction count. Do not pool this dataset with other binaries or independent Driver probes.

The original emitted source prefix is byte-identical to the matched default; repeat decisions and full source hashes match. Host launch receipts use actual BufferView addresses and static byte spans to predict entry/grid selection. They are not a GPU trace and compiler worker hints are not measured physical thread counts. Correctness tests and source evidence do not prove every workload benefits.

## Reproduce the public calculations

From this directory with Python 3.10 or later (standard library only):

```text
python reproduce.py
python -m unittest test_reproduce.py
```

The first command verifies every public file SHA256, all raw-sample medians and ratios, the unchanged policy formula and decisions, repeat/source/fixture relationships, original failure provenance, and execution counts. It does not recompile or run GPU work, refit the model, or rerun tensor oracles whose large binary tensors are intentionally not bundled. `validation.json` records the completed local full-oracle revalidation; `proof-receipts.json` binds its frozen source/build/binary snapshot.

`provenance.json` references original pre-redaction file bytes. `receipts.json` hashes the actual public redacted files. The opaque fit ID is unchanged provenance, not the SHA of this new public package. `${REPO}`, `${CUDA_TOOLKIT}`, `${USER_HOME}` and UUID placeholders are path labels only. `.gitattributes` fixes all published text to LF so checkout bytes match the receipts.

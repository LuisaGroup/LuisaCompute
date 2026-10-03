# V5 streamed SUM: alignment-isolation checkpoint

All 56 native processes passed their unchanged strict oracle and full guard/read-only checks. All 16 aligned streamed candidates are slower than their same-chunk plain controls: **1.5356–2.4097×**. Structured aligned loads did not improve the V4 streamed formulation in this cohort. This is an independent standalone prototype checkpoint; it does not change production defaults or install a selection model.

The original whole-row kernel responds differently: FP16 sometimes improves, while BF16 R3×16384 regresses by 81.53%. Lower reported register counts do not by themselves predict a faster kernel. No negative result has been removed.

## Complete median table

Graph event microseconds per complete operation, lower is better. P = plain load, A = aligned structured load. The seven stages ran in the column order below as separate processes. All seven samples, graph replay counts/spans, warmup, stream/host measurements and cold/setup costs are retained in the JSON.

| Case | P original | A original | P C1024 | A C1024 | P C2048 | A C2048 | P recheck |
|---|---:|---:|---:|---:|---:|---:|---:|
| r3-n8192-fp16 | 1.1739 | 1.1357 | 1.1511 | 1.7677 | 1.1521 | 2.2740 | 1.1838 |
| r3-n8192-bf16 | 1.1527 | 1.1535 | 1.1506 | 1.7715 | 1.1560 | 2.2859 | 1.1552 |
| r128-n8192-fp16 | 2.6390 | 2.4069 | 2.5409 | 4.4149 | 2.5392 | 5.0431 | 2.5570 |
| r128-n8192-bf16 | 2.5323 | 2.4276 | 2.5047 | 4.4169 | 2.5202 | 5.0628 | 2.5082 |
| r3-n16384-fp16 | 1.4707 | 1.3841 | 1.4023 | 2.7126 | 1.5474 | 3.7288 | 1.4694 |
| r3-n16384-bf16 | 1.4032 | 2.5472 | 1.5403 | 2.7047 | 2.1820 | 3.7647 | 1.4026 |
| r128-n16384-fp16 | 4.2479 | 3.8654 | 4.1846 | 7.9304 | 4.2279 | 8.7538 | 4.2561 |
| r128-n16384-bf16 | 4.1439 | 4.3680 | 4.1952 | 7.8513 | 4.2325 | 8.8176 | 4.2250 |

## Matched layout changes and drift

Ratios below compare A/P within the same IR formulation. The final column is the plain-original recheck divided by its initial measurement. These are distinct questions from changing the IR and the layout together.

| Case | A/P original | A/P C1024 | A/P C2048 | Plain control drift |
|---|---:|---:|---:|---:|
| r3-n8192-fp16 | 0.9675x | 1.5356x | 1.9738x | +0.84% |
| r3-n8192-bf16 | 1.0007x | 1.5396x | 1.9773x | +0.22% |
| r128-n8192-fp16 | 0.9120x | 1.7375x | 1.9861x | -3.11% |
| r128-n8192-bf16 | 0.9587x | 1.7634x | 2.0089x | -0.95% |
| r3-n16384-fp16 | 0.9411x | 1.9344x | 2.4097x | -0.09% |
| r3-n16384-bf16 | 1.8153x | 1.7560x | 1.7253x | -0.04% |
| r128-n16384-fp16 | 0.9099x | 1.8951x | 2.0705x | +0.19% |
| r128-n16384-bf16 | 1.0541x | 1.8715x | 2.0833x | +1.96% |

- Original FP16 R128×8192 improves 8.80% versus its initial plain control, but 5.87% versus recheck; that control drifts −3.11%. Original FP16 R128×16384 improves 9.01%/9.18% versus the two plain controls. Original FP16 R3×16384 improves 5.89%/5.80%. These observations are not a universal alignment benefit.
- Original BF16 R3×16384 rises from 1.4032 to 2.5472 us (+81.53%; +81.61% versus recheck). Every aligned streamed version is also slower than its aligned original: the candidate/original ratios span 1.0618–2.6940×.
- Raw outliers remain visible: plain C2048 FP16 R128×8192 has a 5.8064 us sample and a 2.5392 us median. No trimming or minimum-only reporting was used. The aligned streamed regressions are present across their seven samples; sample-range overlap is not treated as a significance test.
- Whole-stage telemetry includes compilation, allocation, uploads and warmup. This run does not identify a thermal, clock, cache, allocation or compiler-layout cause. Fixed stage order is not randomized/interleaved causal evidence.

## Actual entry and loaded resources

Every timed graph records all 100 nodes. The closed validator checked their actual CUfunction against the loaded entry, complete grid/block/shared descriptors, all four final pointer values and residues, and consistent resource-function identity inside each process. Aligned stages selected `luisa_tile_aligned16`; plain stages selected `luisa_tile_main`. Grid is `[R,1,1]`, the Tile launch API block is `[1,1,1]`, and dynamic shared argument is zero. These are stronger than a source-only prediction, but the Tile API block and `max_threads` attribute are not the physical worker count or occupancy. Live non-graph dispatches have numerical/guard checks, not a device trace.

All selected entries report local_bytes=0. The aligned streamed C1024 entries report 40 registers and **8192 B static shared**, while C2048 reports 74 registers and **16384 B static shared**. Their plain counterparts report 16 B static shared. This is a compiled-resource difference associated with the slower cases; it is not by itself a causal proof.

Resource table: REG / static shared bytes / max_threads for the actual selected entry. R3 and R128 have the same resource tuple for each listed width/type/formulation.

| Columns / storage | P original | A original | P C1024 | A C1024 | P C2048 | A C2048 |
|---|---|---|---|---|---|---|
| 8192 / fp16 | 72/16/896 | 46/16/1024 | 72/16/896 | 40/8192/1024 | 72/16/896 | 74/16384/768 |
| 8192 / bf16 | 71/16/896 | 47/16/1024 | 72/16/896 | 40/8192/1024 | 72/16/896 | 74/16384/768 |
| 16384 / fp16 | 86/16/640 | 78/16/768 | 136/16/384 | 40/8192/1024 | 94/16/640 | 74/16384/768 |
| 16384 / bf16 | 137/16/384 | 46/16/1024 | 86/16/640 | 40/8192/1024 | 45/16/1024 | 74/16384/768 |

The BF16 R3×16384 original is a direct counterexample to register-count-only selection: 137→46 registers, unchanged 16 B shared, yet +81.53% runtime. Unknown resource values remain unknown in the evidence. No GPU occupancy API or new SASS/resource compilation was run for this analysis.

## Correctness, source and protocol

Eight fixtures: R={3,128}, N={8192,16384}, storage={FP16,BF16}; strict FP32 computation, final narrow rounding, unchanged FP64 per-element bounds, seed 20261003. R3 uses cancellation inputs and R128 uses random inputs. Each transformed kernel carries a `[1,C]` FP32 tile through N/C SERIAL iterations and performs one post-loop reduction. The verified logical facts remain collective=1 and contribution=C; they do not quantify physical registers or memory traffic.

Seven graph-v2 samples target 100 ms with adaptive replay count, 500 ms warmup, 100 complete operations/graph, four CPU threads and affinity 0x15400. All outputs were saved and independently checked against the original oracle by the closed validator; runtime readonly/guard checks remain separately identified. Source proof retains each complete plain entry byte-for-byte as the aligned module prefix and keeps plain original/recheck identical. Compile wall includes transformation and diagnostic writes; it is not isolated compiler overhead.

There is **no Torch measurement or Torch denominator in this V5 cohort**. The independent Torch appendix describes retained generated code/configuration/resources from another closed run. It neither supplies ratios for this table nor establishes the cause of its unusually slow timing states. Previously published V3/V4 evidence remains unchanged.

## Package verification

Run `python reproduce.py` in this directory. The standard-library verifier checks public file SHA/size and LF bytes; included source/manifest hashes; all 392 primary samples and medians; graph denominators/calibration; 56 actual graph/resource records; same-fixture/oracle receipts; original prefixes; control and same-layout ratios. It does not execute CUDA or recompute the omitted large tensor outputs. Recorded full-output/readonly/guard validation remains traceable to original raw receipts.

Public content-addressed files have their own SHA after LF normalization and path-label redaction. Original raw SHA/size are separate provenance, never claimed to match modified bytes. `${REPO}`/`${EXTERNAL}` labels in receipts are not downloadable package paths. Included files are named by actual public content SHA. Dynamic handles in graph CSVs are process-local opaque identities, not comparable across processes. Runtime binaries and large tensors are omitted.

The separate `torch-static.json` and `README-torch-static.md` omit timing denominators and include retained configuration, source/IR/PTX/resource artifacts. Their scope is static inspection only; no benchmark was rerun to produce this appendix.

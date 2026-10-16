# Streamed SUM Tile IR prototypes

This checkpoint keeps V3 and V4 as independent standalone prototypes. Production defaults were not changed by these measurements. Inclusion is not an adoption decision.

V3 reduces every chunk and carries a scalar. V4 carries a chunk-wide FP32 Tile and reduces once after the serial loop. Both use the actual workload fixture, unchanged FP64 oracle and final-storage rounding allowance. Logical Tile storage facts are not physical register counts.

All 12 cases and both candidate chunk sizes are retained, including regressions. Each version measured fresh default -> chunk1024 -> chunk2048 -> default recheck. Fixed stage order is not a randomized or interleaved causal experiment.

Seven graph-v2 samples use a 100 ms target, 500 ms warmup, batch 100 and four host cores. Raw spans, replay counts, warmup, cold costs and all samples remain in `evidence.json`. Compile wall includes the prototype transform and small receipt writes.

Fresh Torch was run after all V4 native stages using those exact exported inputs/oracles and max-autotune/fullgraph. It has a fresh denominator for each case. These ratios do not reuse historical Torch measurements or claim drift-controlled interleaving.

## V3 full case table

| Case | Default us | C1024 us | C2048 us | Recheck us | C1024 / recheck | C2048 / recheck |
|---|---:|---:|---:|---:|---:|---:|
| sum-r3-n8192-fp32 | 1.367106 | 2.119005 | 1.731633 | 1.361032 | 1.5569 | 1.2723 |
| sum-r3-n8192-fp16 | 1.177639 | 2.046023 | 1.585176 | 1.178676 | 1.7359 | 1.3449 |
| sum-r3-n8192-bf16 | 1.155434 | 1.997894 | 1.496565 | 1.155669 | 1.7288 | 1.2950 |
| sum-r128-n8192-fp32 | 4.000743 | 4.537500 | 3.916491 | 3.804626 | 1.1926 | 1.0294 |
| sum-r128-n8192-fp16 | 2.712643 | 4.026605 | 3.135174 | 2.560156 | 1.5728 | 1.2246 |
| sum-r128-n8192-bf16 | 2.666961 | 4.029461 | 3.137021 | 2.517351 | 1.6007 | 1.2462 |
| sum-r3-n16384-fp32 | 1.849544 | 3.524230 | 2.717563 | 1.845335 | 1.9098 | 1.4727 |
| sum-r3-n16384-fp16 | 1.465810 | 3.374457 | 2.424811 | 1.476556 | 2.2854 | 1.6422 |
| sum-r3-n16384-bf16 | 1.408808 | 3.329512 | 2.268258 | 1.405471 | 2.3690 | 1.6139 |
| sum-r128-n16384-fp32 | 7.061115 | 8.179762 | 7.023280 | 6.726698 | 1.2160 | 1.0441 |
| sum-r128-n16384-fp16 | 4.823310 | 7.592501 | 5.691677 | 4.283089 | 1.7727 | 1.3289 |
| sum-r128-n16384-bf16 | 4.978149 | 7.527338 | 5.744155 | 4.193718 | 1.7949 | 1.3697 |

## V4 full case table

| Case | Default us | C1024 us | C2048 us | Recheck us | C1024 / recheck | C2048 / recheck |
|---|---:|---:|---:|---:|---:|---:|
| sum-r3-n8192-fp32 | 1.361428 | 1.302564 | 1.303363 | 1.364619 | 0.9545 | 0.9551 |
| sum-r3-n8192-fp16 | 1.173364 | 1.150595 | 1.152577 | 1.178284 | 0.9765 | 0.9782 |
| sum-r3-n8192-bf16 | 1.151062 | 1.150012 | 1.154396 | 1.161512 | 0.9901 | 0.9939 |
| sum-r128-n8192-fp32 | 3.801914 | 3.733701 | 3.737016 | 3.797943 | 0.9831 | 0.9840 |
| sum-r128-n8192-fp16 | 2.555583 | 2.537022 | 2.545896 | 2.568873 | 0.9876 | 0.9911 |
| sum-r128-n8192-bf16 | 2.505878 | 2.507784 | 2.516284 | 2.510239 | 0.9990 | 1.0024 |
| sum-r3-n16384-fp32 | 1.842841 | 1.747742 | 1.738899 | 1.845828 | 0.9469 | 0.9421 |
| sum-r3-n16384-fp16 | 1.460307 | 1.406751 | 1.547511 | 1.460897 | 0.9629 | 1.0593 |
| sum-r3-n16384-bf16 | 1.393542 | 1.540164 | 2.176811 | 1.404851 | 1.0963 | 1.5495 |
| sum-r128-n16384-fp32 | 6.639533 | 6.636339 | 6.621252 | 6.671009 | 0.9948 | 0.9925 |
| sum-r128-n16384-fp16 | 4.210521 | 4.266240 | 4.211566 | 4.211777 | 1.0129 | 0.9999 |
| sum-r128-n16384-bf16 | 4.160483 | 4.201712 | 4.275267 | 4.227763 | 0.9938 | 1.0112 |

## Fresh Torch matched comparison

| Case | Torch us | V4 default / Torch | V4 C1024 / Torch | V4 C2048 / Torch | V4 recheck / Torch |
|---|---:|---:|---:|---:|---:|
| sum-r3-n8192-fp32 | 1.279019 | 1.0644 | 1.0184 | 1.0190 | 1.0669 |
| sum-r3-n8192-fp16 | 1.092114 | 1.0744 | 1.0535 | 1.0554 | 1.0789 |
| sum-r3-n8192-bf16 | 1.391303 | 0.8273 | 0.8266 | 0.8297 | 0.8348 |
| sum-r128-n8192-fp32 | 4.715439 | 0.8063 | 0.7918 | 0.7925 | 0.8054 |
| sum-r128-n8192-fp16 | 2.626342 | 0.9731 | 0.9660 | 0.9694 | 0.9781 |
| sum-r128-n8192-bf16 | 2.471026 | 1.0141 | 1.0149 | 1.0183 | 1.0159 |
| sum-r3-n16384-fp32 | 2.604773 | 0.7075 | 0.6710 | 0.6676 | 0.7086 |
| sum-r3-n16384-fp16 | 1.892255 | 0.7717 | 0.7434 | 0.8178 | 0.7720 |
| sum-r3-n16384-bf16 | 2.229948 | 0.6249 | 0.6907 | 0.9762 | 0.6300 |
| sum-r128-n16384-fp32 | 33.741980 | 0.1968 | 0.1967 | 0.1962 | 0.1977 |
| sum-r128-n16384-fp16 | 14.672834 | 0.2870 | 0.2908 | 0.2870 | 0.2870 |
| sum-r128-n16384-bf16 | 4.711253 | 0.8831 | 0.8918 | 0.9075 | 0.8974 |

## Independent verification

Run `python reproduce.py` from this directory; only Python standard library is required. It checks every public file hash, actual included source/manifest bytes, all medians, graph normalization, matching fixtures, per-version ratios, fresh Torch ratios and V3/V4 logical facts. It does not rerun CUDA or reconstruct omitted tensor outputs.

Source and manifest files are included with content-addressed public names. CRLF is normalized to LF before their public SHA is computed. Original raw SHA/size remain separate provenance; when normalization changed bytes, the original raw file is not claimed to be present or reconstructible. `.gitattributes` pins LF for all public text.

`${REPO}` and `${EXTERNAL}` labels in receipts are provenance labels, not executable public file paths. Included `sources/`, `manifests/`, `torch-code/` and `code/` paths are real package files. Adapter/IR code snapshots document measured code; they are not a turnkey GPU build detached from the corresponding repository/runtime.

Full readonly/guard checks were recorded by the runtime; saved full outputs were independently rechecked by the closed local validator. This compact package keeps their receipts and summaries, not the large tensor binaries. Correctness/build evidence is separately retained in `correctness.json`, without inventing unavailable logs. Whole-stage telemetry includes compilation and setup, so no clock/thermal cause is inferred.

No negative case is dropped and no automatic production selection or model refit is performed by this exporter.

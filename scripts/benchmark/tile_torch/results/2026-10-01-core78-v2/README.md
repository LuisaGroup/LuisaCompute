# CUDA Tile checkpoint: core78, 2026-10-01

The native CUDA Tile path passes all 78 recorded configurations, but does **not** yet match `torch.compile` across this suite. Large GEMM, 4096-element sort, scan, and some normalization cases remain slower. Repeated-extrema topk, selected GEMV, and attention with an appropriate query tile have lower graph medians on this machine.

This is the historical implementation at `a3aaac11ce4deaabe9b7b47d0d6200f1323881fb`, before the later chunked-sort/BMM/work-budget/rsqrt bundle. Quick12 ran before that commit was created; quick12 and remainder66 have identical complete captured source/binary receipt sets. The original build marker names `a961e30d...` and is preserved. Git HEAD is not used as a substitute for content identity.

The 78 records cover 16 operation names, 36 operation/shape combinations, and 51 operation/shape/dtype combinations. Schedules, algorithms and math-policy repeats are additional configurations, not independent workloads. There are 61 default-math and 17 explicit-fast configurations. The default/fast split is distinct from the numerical acceptance contract:

| Math policy | Original exported FP64 per-element bound | Predeclared narrow-attention envelope |
| --- | ---: | ---: |
| Default | 55 | 6 |
| Explicit fast | 12 | 5 |

All 156 native/Torch route records passed and have no exporter evidence issues. Both sides consumed the same typed input bytes and exported oracle; all output elements were checked. The 11 common-envelope cases retain secondary strict-oracle failures and `backend_precision_contract_verified=false`: accepting automatic SDPA against that envelope does not prove identical internal arithmetic. Native ranking is stable with index tie-breaking; the primary Torch reference uses standard `torch.topk`/unstable `torch.sort`, accepting valid tie permutations.

Selected graph medians are below, in microseconds per complete logical operation. `Torch/native > 1` means the native median is lower. Unless marked otherwise, rows use default math and the original strict oracle. Attention dimensions shown omit the shared B1/Hq4/Hkv1 prefix. These are individual observations, not a pooled win rate or statistical speedup claim.

| Configuration | Native us | Torch us | Torch/native |
| --- | ---: | ---: | ---: |
| FP32 GEMM 512x512x512; tile 32x32x32 | 96.578 | 52.524 | 0.544 |
| FP32 GEMM 512x512x512; tile 32x32x128 | 490.460 | 52.268 | 0.107 |
| BF16 GEMM 512x512x512; tile 64x64x32 | 28.851 | 16.534 | 0.573 |
| FP32 GEMV 1024x1x4096; K tile 1024 | 20.053 | 70.217 | 3.502 |
| FP32 prefill Q128/K256/D64; Q tile 16 | 32.763 | 40.049 | 1.222 |
| FP32 decode Q1/K1024/D128; Q tile 1 | 35.706 | 179.715 | 5.033 |
| FP16 decode Q1/K1024/D128; Q tile 1 [common] | 38.268 | 43.276 | 1.131 |
| FP16 decode same dimensions; Q tile 16 [common] | 82.194 | 43.172 | 0.525 |
| FP32 topk 128x1024/K7; repeated extrema | 5.904 | 73.843 | 12.508 |
| FP32 topk same dimensions; full sort prefix | 47.813 | 73.825 | 1.544 |
| FP16 sort 1x4096; full bitonic sort | 75.554 | 13.420 | 0.178 |
| FP16 scan 128x1024; unordered tree | 2.127 | 1.278 | 0.601 |
| FP32 RMSNorm 1x8192; default math | 3.117 | 1.882 | 0.604 |
| BF16 RMSNorm 1x8192; explicit fast math | 1.712 | 1.518 | 0.887 |

The Q-tile-1 and Q-tile-16 decode rows deliberately execute different schedules for one logical query; the latter uses fifteen padded query lanes. Full-sort-prefix and repeated-extrema are distinct topk algorithms. No numbers from the earlier single-replay timing protocol, ordered-scan implementation, frozen-source compiler probes, or abandoned memory-layout candidate are pooled here.

Measurement used an RTX 4060 Laptop GPU (8 GiB, sm_89), Windows 11, driver 617.14, MSVC, LLVM 22.1.8 and system CUDA 13.4. Torch was 2.14.1+cu130 with triton-windows 3.8.0.post29, Python 3.12.14 and NumPy 2.3.5. These timings are LLVM22-only. The two cohorts ran serially, using four selected P cores (local affinity mask `0x15400`), four host threads, `fullgraph=True`, Inductor `max-autotune`, and one compile thread.

Graph protocol `adaptive_replay_span_v2` primes events before 500 ms graph warmup. Each graph contains 100 complete operations. Up to four calibration attempts choose a measured replay count R for an approximately 100 ms sample (cap 65536 replays and 10 million logical operations). Seven samples reuse that fixed R and divide both event and synchronized host spans by `100*R`. All samples, raw spans, R, calibration, warmup and cold times are in [checkpoint.json](checkpoint.json); none were trimmed. An approximately 100 ms target is not a minimum-duration guarantee.

Native output is preallocated; Torch returns functional outputs and reuses its capture memory pool. No extra output copy is added to Torch. Event spans can include host submission starvation, so these are graph throughput observations, not isolated instruction latency. Compilation boundaries differ (native compile vs Torch compile plus first call); no compile-speed ratio is reported. Each pair ran once in serial order, not ABBA, and clocks were not locked. One-second telemetry across whole cohorts (including compilation and transitions) ranged from 1845-2610 MHz in quick12 and 210-2610 MHz in remainder66 among readings at least 70% busy; this cannot assign a clock to every timed sample or establish causality for small differences.

## Reproduction and receipts

Use the recorded implementation and package versions, configure MSVC/Ninja with CUDA/native Tile enabled, and complete `cmake --build <build> --parallel 8`. Supply a genuine successful full-build marker as required by `cuda_matrix.py`; never fabricate it. In an environment containing the configured CUDA/runtime DLLs, run from the repository root with a fresh output directory:

```powershell
$env:TORCHINDUCTOR_COMPILE_THREADS = '1'
$torchPython = (Get-Command python).Source # activate the matching Torch environment first
& $torchPython scripts/benchmark/tile_torch/cuda_matrix.py --cases scripts/benchmark/tile_torch/cuda_llm_primitives.json --build-dir build-msvc-llvm --build-marker build-msvc-llvm/logs/full-build-success.json --torch-python $torchPython --output benchmark-results/core78-reproduction --routes native --affinity-mask 0x15400 --threads 4 --samples 7 --sample-ms 100 --warmup-ms 500 --graph-batch 100 --torch-mode max-autotune --ranking-contract standard --native-timeout 180 --torch-timeout 180
```

Choose an appropriate affinity mask on another CPU; the recorded mask is machine-specific. A single 78-case run reproduces the inventory/protocol but not the historical quick12/remainder66 chronology. Data, generated-source, compiler evidence, raw log, runtime/source, and matrix receipts remain in the JSON. Local raw artifacts are referenced for audit; their paths are labels, not a promise that ignored artifacts are distributed. No absolute local paths, PATH environment or GPU UUID is published. The checkpoint uses LF line endings for Git; only its final newline differs from the original local CRLF export. All embedded evidence hashes remain unchanged.

| Artifact | SHA-256 |
| --- | --- |
| This checkpoint.json (Git LF text) | `e0e703cabf8a142dba537522732756b9369ff5538fbeca5f36c9535ea514c564` |
| Original local CRLF export (same JSON data) | `1db630de9081b9ab855f7cab82404816adc87928e6299963ed7673d479ce3cbe` |
| Public cuda_llm_primitives.json inventory | `bae0a476e0ad22ad9e14c1cb9eb803a8f3cf6b88ea42d5ad325f5a972680428d` |
| Local .deps/oct01-tile-core-quick12-v2/results.json | `bd94f26d2ece2237c1bc42bb93b3a8e384a194d57110a2fd24a8bb6d567981c3` |
| Local .deps/oct01-tile-core-remainder66-v2/results.json | `f24e63933795347d3405182d4a249531808908fb47cd5e527b53c2c1c18da75b` |

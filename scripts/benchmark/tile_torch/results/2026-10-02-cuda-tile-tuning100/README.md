# CUDA Tile tuning: 100 configurations

Measured implementation: `bca9b9c9de7bc20fc2e5601ba325644fb783f788`, MSVC/LLVM 22, RTX 4060 Laptop GPU. All 100 configurations passed their complete numerical/input/guard checks. These are 27 operation/shape and 42 operation/shape/dtype combinations, including repeated schedules and 18 aligned-entry controls; they are not 100 different operators.

The [82 default-alignment records](standard/README.md) and [18 aligned16 records](aligned16/README.md) retain all seven graph samples, calibration, actual replay counts, source/binary receipts and cold times. Their configuration remains separate. Timing is CUDA graph event span per complete logical operation, in microseconds. Torch uses `torch.compile(mode="max-autotune", fullgraph=True)`, with the recorded precision policy. Sorting comparisons below require stable index ties. Native preallocated outputs and Torch functional graph-pool allocation differ. Small percentage differences are not robust wins, and no overall win rate is calculated.

| Workload | Original native | Best tested native | Matched Torch | Observation |
| --- | ---: | ---: | ---: | --- |
| SwiGLU 1x8192 BF16 | 3.342 | 0.865 (BD256) | 0.820 | 74% lower native time; about 6% remaining gap |
| RoPE 1x8192 BF16 | 1.548 | 0.816 (BD256) | 0.859 | Near parity; smaller tail tiles did not improve |
| Stable sort 16x1024 FP32 | 12.743 | 11.442 (packed) | 14.991 | Exact values and stable indices retained |
| Stable sort 128x1024 FP32 | 47.548 | 37.517 (packed) | 23.937 | Significant remaining gap |
| Stable sort 4x8192 FP32 | 331.412 | 60.571 (chunk512) | 50.495 | 5.47x native improvement; still 20% slower |
| GEMM 1024 cubed FP16 | 206.862 | 168.038 (aligned16) | 97.259 | Alignment helps; original entry retained |
| GEMM 1024 cubed BF16 | 207.575 | 167.762 (aligned16) | 95.587 | Alignment helps; substantial gap remains |
| GEMV 4096x4096 FP32 | 269.356 | 269.356 | 269.043 | Three schedules were essentially tied |
| Causal MHA B2/H4/Q1/K65/D32 FP32 | -- | 3.863 | 14.573 | Strict contract; small decode configuration |
| Causal MHA B2/H4/Q17/K65/D32 FP32 | -- | 6.957 | 16.594 | Strict contract; rectangular prefill |

Aligned16 remains opt-in: narrow scan128x8192 became roughly 3–6% slower, despite GEMM improving about 19%. Ineligible FP32/tail controls stayed close. The recorded selected entry is the shared host selector's prediction from actual final-pointer residues, not a device execution trace.

Embedding D64 did not benefit from serial BR4/8; BR8 was about 45% slower. At D4096/T512, BR1/BD1024 was the best native schedule tested (FP32 11.679, FP16 8.496, BF16 8.453 us). Repeated identical Torch embedding kernels varied materially across visits, so isolated apparent wins are not treated as reliable advantages. Smaller M tiles also regressed both small-M GEMM examples. Full samples and each corresponding Torch result remain visible.

## Independent compiler experiments

[diagnostics.json](diagnostics.json) preserves the complete seven alternating samples, correctness checks, resource counts and cleanup outcomes for the standalone paired Driver experiments. This protocol is separate from the production graph-v2/Torch comparison above.

| 1024-cubed GEMM | Aligned Tile control | Candidate | Outcome |
| --- | ---: | ---: | --- |
| FP16, worker8 hint | 162.572 | 186.151 | Correct, 14.5% slower; do not adopt |
| BF16, worker8 hint | 162.197 | 184.185 | Correct, 13.6% slower; do not adopt |
| FP16, CuTe two-stage | 162.212 | 95.728 | Correct, 41.0% lower time |
| BF16, CuTe two-stage | 162.318 | 95.031 | Correct, 41.5% lower time |

The tested [CuTe source](cute_gemm_sm89.cu) uses pinned CUTLASS v4.5.2 (`db1c288993354c88e551c40c19a8fb93a774a241`), real 128-thread blocks, cp.async/LDSM, FP32 accumulation and one final narrow conversion. Both full 1,048,576-element outputs passed the original FP64 bounds, readonly/guard checks and direct/graph launches. Resources were 96 registers, 16 KiB shared and zero local bytes. The source is only a fixed-shape, aligned, disjoint-storage diagnostic, not a production backend route or a direct Torch result. Original BSD license is retained.

A subsequent system NVRTC compile succeeded after replacing the host type-traits include/assert with `cuda::std`; it needs `--device-as-default-execution-space`. Removing that option failed compilation. NVRTC output has not yet been run, so its correctness/performance is unverified. Production CuTe integration remains a draft.

## Reproduction and next work

The main inventories are `cuda_pointwise_schedules.json`, `cuda_packed_sort.json`, `cuda_alignment.json`, `cuda_embedding_schedules.json` and `cuda_attention_mha.json` in the benchmark directory. The additional measured inventories are copied here as [sort3.json](sort3.json) and [matrix9.json](matrix9.json).

Use the existing `cuda_matrix.py` after a full CMake build, with `--routes native --samples 7 --sample-ms 100 --warmup-ms 500 --graph-batch 100 --threads 4 --affinity-mask 0x15400`. Sorting requires `--ranking-contract stable`; other cohorts used `standard`. Only the aligned18-on cohort adds `--native-aligned16`. The checkpoint retains exact executable, build-marker and environment identities.

See [CONTINUE.md](CONTINUE.md) for the shutdown handoff and explicitly uncompiled candidates. No new candidate is made a default by this results commit.

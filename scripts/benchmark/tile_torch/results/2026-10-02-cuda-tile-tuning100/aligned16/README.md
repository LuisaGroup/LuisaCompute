# CUDA Tile graph-v2 cohort checkpoint

This report records individual configurations, not independent workload counts or a pooled win rate. Source/binary receipt sets must match across cohorts; it does not combine historical core78 binaries. Ranking and numerical acceptance contracts remain separate group keys. Aligned16 off/on cohorts cannot be merged.

Records: 18; operation/shape combinations: 9; operation/shape/dtype combinations: 18.

All seven graph samples, full event/host spans, measured R, calibration, warmup, cold times, compiler evidence, failures and pipeline stage receipts are in [checkpoint.json](checkpoint.json). No samples are trimmed. Event and host spans divide by graph batch times R, never stage count. Native pipeline stages may overlap across complete calls when hazards permit. Native preallocated storage and Torch functional capture-pool allocation differ. Compilation boundaries differ, so no compiler-speed ratio is calculated. Serial visits do not establish frequency-controlled causation.

## alignment18-on

Matrix status: `passed`. Torch ranking contract: `standard`. Torch mode: `max-autotune`. Aligned16 opt-in: `True`. Original matrix SHA-256: `95b94ddbd494388dcb20508ca74d24d2b84d3024e7d505ca222ce59e9d0257e1`.

| Case | Dtype/math | Tile | Realized algorithm | Alignment (expected) | Numerical contract | Compare status | Native us | Torch us | Torch/native |
| --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: |
| aligned16-gemm-1024x1024x1024-fp16 | fp16/default | 64x64x32 | tile_mma_typed_inputs_fp32_accumulator_reassociation | aligned (expected) | strict_exported_fp64_per_element_bound | valid_strict_oracle | 168.038 | 97.259 | 0.579 |
| aligned16-scan-128x8192-fp16 | fp16/default | 1x8192x1 | inclusive_sum_unordered_tree | aligned (expected) | strict_exported_fp64_per_element_bound | valid_strict_oracle | 12.707 | 5.475 | 0.431 |
| aligned16-reduce_sum-128x8192-fp16 | fp16/default | 1x8192x1 | whole_row_tile_fp32_compute | aligned (expected) | strict_exported_fp64_per_element_bound | valid_strict_oracle | 2.391 | 2.461 | 1.030 |
| aligned16-rmsnorm-1x8192-fp16 | fp16/default | 1x8192x1 | whole_row_tile_fp32_compute | aligned (expected) | strict_exported_fp64_per_element_bound | valid_strict_oracle | 2.710 | 1.463 | 0.540 |
| aligned16-layernorm-1x8192-fp16 | fp16/default | 1x8192x1 | whole_row_tile_fp32_compute | aligned (expected) | strict_exported_fp64_per_element_bound | valid_strict_oracle | 2.993 | 1.880 | 0.628 |
| aligned16-bmm-2x64x64x64-fp16 | fp16/default | 32x32x32 | tile_bmm_singleton_batch_typed_inputs_fp32_accumulator_reassociation | aligned (expected) | strict_exported_fp64_per_element_bound | valid_strict_oracle | 1.303 | 1.521 | 1.167 |
| aligned16-gemm-127x257x65-fp16 | fp16/default | 32x32x32 | tile_mma_typed_inputs_fp32_accumulator_reassociation | requested; ineligible | strict_exported_fp64_per_element_bound | valid_strict_oracle | 5.376 | 4.589 | 0.854 |
| aligned16-gemm-1024x1024x1024-bf16 | bf16/default | 64x64x32 | tile_mma_typed_inputs_fp32_accumulator_reassociation | aligned (expected) | strict_exported_fp64_per_element_bound | valid_strict_oracle | 167.762 | 95.587 | 0.570 |
| aligned16-scan-128x8192-bf16 | bf16/default | 1x8192x1 | inclusive_sum_unordered_tree | aligned (expected) | strict_exported_fp64_per_element_bound | valid_strict_oracle | 13.106 | 6.275 | 0.479 |
| aligned16-reduce_sum-128x8192-bf16 | bf16/default | 1x8192x1 | whole_row_tile_fp32_compute | aligned (expected) | strict_exported_fp64_per_element_bound | valid_strict_oracle | 2.404 | 2.471 | 1.028 |
| aligned16-rmsnorm-1x8192-bf16 | bf16/default | 1x8192x1 | whole_row_tile_fp32_compute | aligned (expected) | strict_exported_fp64_per_element_bound | valid_strict_oracle | 2.727 | 1.501 | 0.550 |
| aligned16-layernorm-1x8192-bf16 | bf16/default | 1x8192x1 | whole_row_tile_fp32_compute | aligned (expected) | strict_exported_fp64_per_element_bound | valid_strict_oracle | 3.026 | 1.955 | 0.646 |
| aligned16-bmm-2x64x64x64-bf16 | bf16/default | 32x32x32 | tile_bmm_singleton_batch_typed_inputs_fp32_accumulator_reassociation | aligned (expected) | strict_exported_fp64_per_element_bound | valid_strict_oracle | 1.300 | 1.608 | 1.237 |
| aligned16-gemm-127x257x65-bf16 | bf16/default | 32x32x32 | tile_mma_typed_inputs_fp32_accumulator_reassociation | requested; ineligible | strict_exported_fp64_per_element_bound | valid_strict_oracle | 5.383 | 4.601 | 0.855 |
| aligned16-control-scan-128x8192-fp32 | fp32/default | 1x8192x1 | inclusive_sum_unordered_tree | requested; ineligible | strict_exported_fp64_per_element_bound | valid_strict_oracle | 15.225 | 13.897 | 0.913 |
| aligned16-control-rmsnorm-17x65-fp16 | fp16/default | 1x128x1 | whole_row_tile_fp32_compute | requested; ineligible | strict_exported_fp64_per_element_bound | valid_strict_oracle | 1.056 | 0.908 | 0.860 |
| aligned16-softmax-128x1024-fp16 | fp16/default | 1x1024x1 | whole_row_tile_fp32_compute | aligned (expected) | strict_exported_fp64_per_element_bound | valid_strict_oracle | 1.815 | 1.855 | 1.022 |
| aligned16-softmax-128x1024-bf16 | bf16/default | 1x1024x1 | whole_row_tile_fp32_compute | aligned (expected) | strict_exported_fp64_per_element_bound | valid_strict_oracle | 1.825 | 1.561 | 0.855 |

Common-envelope SDPA acceptance, where present, is not proof of identical internal arithmetic; secondary strict validation remains in the JSON. A `failed_route`, `unsupported_route`, or `invalid_evidence` record is not a valid performance comparison. Alignment is the predicted shared host selector result from actual final-pointer residues, not a device execution trace; requested-but-ineligible cases stay explicit. Original raw packet/log hashes identify local artifacts even when path labels are normalized. PATH and GPU UUID are omitted.

Checkpoint bytes use UTF-8 with LF newlines. SHA-256: `8433f265eae1c0139de21a6cf9149c89b32f771804a66faeab7c582ee8057fd1`.

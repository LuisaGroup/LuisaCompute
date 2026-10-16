# CUDA Tile graph-v2 cohort checkpoint

This report records individual configurations, not independent workload counts or a pooled win rate. Source/binary receipt sets must match across cohorts; it does not combine historical core78 binaries. Ranking and numerical acceptance contracts remain separate group keys.

Records: 48; operation/shape combinations: 13; operation/shape/dtype combinations: 33.

All seven graph samples, full event/host spans, measured R, calibration, warmup, cold times, compiler evidence, failures and pipeline stage receipts are in [checkpoint.json](checkpoint.json). No samples are trimmed. Event and host spans divide by graph batch times R, never stage count. Native pipeline stages may overlap across complete calls when hazards permit. Native preallocated storage and Torch functional capture-pool allocation differ. Compilation boundaries differ, so no compiler-speed ratio is calculated. Serial visits do not establish frequency-controlled causation.

## fastnorm12

Matrix status: `passed`. Torch ranking contract: `standard`. Torch mode: `max-autotune`. Original matrix SHA-256: `89a17f444e078c1d630082e3509063a46c3a69abb8adaf00dc98f85cad57225b`.

| Case | Dtype/math | Tile | Realized algorithm | Numerical contract | Compare status | Native us | Torch us | Torch/native |
| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: |
| v7-rmsnorm-1x8192-fp32-strict | fp32/default | 1x8192x1 | whole_row_tile_fp32_compute | strict_exported_fp64_per_element_bound | valid_strict_oracle | 3.111 | 1.837 | 0.591 |
| v7-rmsnorm-1x8192-fp32-fast | fp32/fast | 1x8192x1 | whole_row_tile_fp32_compute | strict_exported_fp64_per_element_bound | valid_strict_oracle | 2.146 | 1.885 | 0.878 |
| v7-rmsnorm-1x8192-fp16-strict | fp16/default | 1x8192x1 | whole_row_tile_fp32_compute | strict_exported_fp64_per_element_bound | valid_strict_oracle | 2.891 | 1.468 | 0.508 |
| v7-rmsnorm-1x8192-fp16-fast | fp16/fast | 1x8192x1 | whole_row_tile_fp32_compute | strict_exported_fp64_per_element_bound | valid_strict_oracle | 1.763 | 1.458 | 0.827 |
| v7-rmsnorm-1x8192-bf16-strict | bf16/default | 1x8192x1 | whole_row_tile_fp32_compute | strict_exported_fp64_per_element_bound | valid_strict_oracle | 2.802 | 1.498 | 0.534 |
| v7-rmsnorm-1x8192-bf16-fast | bf16/fast | 1x8192x1 | whole_row_tile_fp32_compute | strict_exported_fp64_per_element_bound | valid_strict_oracle | 1.697 | 1.448 | 0.853 |
| v7-layernorm-128x1024-fp32-strict | fp32/default | 1x1024x1 | whole_row_tile_fp32_compute | strict_exported_fp64_per_element_bound | valid_strict_oracle | 2.305 | 1.882 | 0.816 |
| v7-layernorm-128x1024-fp32-fast | fp32/fast | 1x1024x1 | whole_row_tile_fp32_compute | strict_exported_fp64_per_element_bound | valid_strict_oracle | 2.074 | 1.884 | 0.909 |
| v7-layernorm-128x1024-fp16-strict | fp16/default | 1x1024x1 | whole_row_tile_fp32_compute | strict_exported_fp64_per_element_bound | valid_strict_oracle | 2.164 | 1.839 | 0.850 |
| v7-layernorm-128x1024-fp16-fast | fp16/fast | 1x1024x1 | whole_row_tile_fp32_compute | strict_exported_fp64_per_element_bound | valid_strict_oracle | 1.936 | 1.474 | 0.761 |
| v7-layernorm-128x1024-bf16-strict | bf16/default | 1x1024x1 | whole_row_tile_fp32_compute | strict_exported_fp64_per_element_bound | valid_strict_oracle | 2.190 | 1.840 | 0.840 |
| v7-layernorm-128x1024-bf16-fast | bf16/fast | 1x1024x1 | whole_row_tile_fp32_compute | strict_exported_fp64_per_element_bound | valid_strict_oracle | 1.919 | 1.840 | 0.959 |

## chunk9

Matrix status: `passed`. Torch ranking contract: `stable`. Torch mode: `max-autotune`. Original matrix SHA-256: `aada9d68762ad421bc30f2c6094418eedd116c3119919642636512c906d4a7b4`.

| Case | Dtype/math | Tile | Realized algorithm | Numerical contract | Compare status | Native us | Torch us | Torch/native |
| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: |
| chunked-v2-low-row-4x4096-fp32-full_sort_prefix | fp32/default | 1x4096x1 | padded_bitonic_full_sort | strict_exported_fp64_per_element_bound | valid_strict_oracle | 74.301 | 18.658 | 0.251 |
| chunked-v2-low-row-4x4096-fp32-chunked_bitonic_c256 | fp32/default | 1x4096x1 | stable_chunked_bitonic_whole_tile_merge | strict_exported_fp64_per_element_bound | valid_strict_oracle | 21.609 | 18.660 | 0.864 |
| chunked-v2-low-row-4x4096-fp32-chunked_bitonic_c512 | fp32/default | 1x4096x1 | stable_chunked_bitonic_whole_tile_merge | strict_exported_fp64_per_element_bound | valid_strict_oracle | 23.464 | 18.669 | 0.796 |
| chunked-v2-many-row-128x1024-fp32-full_sort_prefix | fp32/default | 1x1024x1 | padded_bitonic_full_sort | strict_exported_fp64_per_element_bound | valid_strict_oracle | 47.535 | 23.934 | 0.504 |
| chunked-v2-many-row-128x1024-fp32-chunked_bitonic_c256 | fp32/default | 1x1024x1 | stable_chunked_bitonic_whole_tile_merge | strict_exported_fp64_per_element_bound | valid_strict_oracle | 96.286 | 23.931 | 0.249 |
| chunked-v2-many-row-128x1024-fp32-chunked_bitonic_c512 | fp32/default | 1x1024x1 | stable_chunked_bitonic_whole_tile_merge | strict_exported_fp64_per_element_bound | valid_strict_oracle | 47.719 | 23.929 | 0.501 |
| chunked-v2-ragged-3x1537-fp32-full_sort_prefix | fp32/default | 1x2048x1 | padded_bitonic_full_sort | strict_exported_fp64_per_element_bound | valid_strict_oracle | 32.034 | 15.146 | 0.473 |
| chunked-v2-ragged-3x1537-fp32-chunked_bitonic_c256 | fp32/default | 1x2048x1 | stable_chunked_bitonic_whole_tile_merge | strict_exported_fp64_per_element_bound | valid_strict_oracle | 13.967 | 15.143 | 1.084 |
| chunked-v2-ragged-3x1537-fp32-chunked_bitonic_c512 | fp32/default | 1x2048x1 | stable_chunked_bitonic_whole_tile_merge | strict_exported_fp64_per_element_bound | valid_strict_oracle | 10.663 | 15.142 | 1.420 |

## bmm12

Matrix status: `passed`. Torch ranking contract: `standard`. Torch mode: `max-autotune`. Original matrix SHA-256: `7cf8c05b6737233b3b2c6c252d4b3044254364a07d14becf99fa2ca9c9aabdf1`.

| Case | Dtype/math | Tile | Realized algorithm | Numerical contract | Compare status | Native us | Torch us | Torch/native |
| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: |
| bmm-small-32x16x16x32-fp32 | fp32/default | 16x16x16 | tile_bmm_singleton_batch_typed_inputs_fp32_accumulator_reassociation | strict_exported_fp64_per_element_bound | valid_strict_oracle | 1.395 | 1.842 | 1.320 |
| bmm-medium-8x128x128x128-fp32 | fp32/default | 32x32x32 | tile_bmm_singleton_batch_typed_inputs_fp32_accumulator_reassociation | strict_exported_fp64_per_element_bound | valid_strict_oracle | 16.098 | 13.165 | 0.818 |
| bmm-tail-3x31x37x19-fp32 | fp32/default | 16x16x8 | tile_bmm_singleton_batch_typed_inputs_fp32_accumulator_reassociation | strict_exported_fp64_per_element_bound | valid_strict_oracle | 1.187 | 4.845 | 4.083 |
| bmm-cancellation-3x31x37x19-fp32 | fp32/default | 16x16x8 | tile_bmm_singleton_batch_typed_inputs_fp32_accumulator_reassociation | strict_exported_fp64_per_element_bound | valid_strict_oracle | 1.184 | 4.846 | 4.094 |
| bmm-small-32x16x16x32-fp16 | fp16/default | 16x16x16 | tile_bmm_singleton_batch_typed_inputs_fp32_accumulator_reassociation | strict_exported_fp64_per_element_bound | valid_strict_oracle | 1.318 | 1.391 | 1.056 |
| bmm-medium-8x128x128x128-fp16 | fp16/default | 32x32x32 | tile_bmm_singleton_batch_typed_inputs_fp32_accumulator_reassociation | strict_exported_fp64_per_element_bound | valid_strict_oracle | 5.912 | 3.940 | 0.666 |
| bmm-tail-3x31x37x19-fp16 | fp16/default | 16x16x8 | tile_bmm_singleton_batch_typed_inputs_fp32_accumulator_reassociation | strict_exported_fp64_per_element_bound | valid_strict_oracle | 1.166 | 1.408 | 1.208 |
| bmm-cancellation-3x31x37x19-fp16 | fp16/default | 16x16x8 | tile_bmm_singleton_batch_typed_inputs_fp32_accumulator_reassociation | strict_exported_fp64_per_element_bound | valid_strict_oracle | 1.162 | 1.407 | 1.211 |
| bmm-small-32x16x16x32-bf16 | bf16/default | 16x16x16 | tile_bmm_singleton_batch_typed_inputs_fp32_accumulator_reassociation | strict_exported_fp64_per_element_bound | valid_strict_oracle | 1.327 | 1.454 | 1.096 |
| bmm-medium-8x128x128x128-bf16 | bf16/default | 32x32x32 | tile_bmm_singleton_batch_typed_inputs_fp32_accumulator_reassociation | strict_exported_fp64_per_element_bound | valid_strict_oracle | 5.841 | 4.833 | 0.827 |
| bmm-tail-3x31x37x19-bf16 | bf16/default | 16x16x8 | tile_bmm_singleton_batch_typed_inputs_fp32_accumulator_reassociation | strict_exported_fp64_per_element_bound | valid_strict_oracle | 1.167 | 1.448 | 1.240 |
| bmm-cancellation-3x31x37x19-bf16 | bf16/default | 16x16x8 | tile_bmm_singleton_batch_typed_inputs_fp32_accumulator_reassociation | strict_exported_fp64_per_element_bound | valid_strict_oracle | 1.167 | 1.446 | 1.239 |

## large15

Matrix status: `passed`. Torch ranking contract: `standard`. Torch mode: `max-autotune`. Original matrix SHA-256: `fd99e1c1d21c3b5c4bcd89207d44082e780976835d30b7489257f2b00e7dbb1e`.

| Case | Dtype/math | Tile | Realized algorithm | Numerical contract | Compare status | Native us | Torch us | Torch/native |
| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: |
| large-gemm-1024x1024x1024-fp32-tile32x32x32 | fp32/default | 32x32x32 | tile_mma_typed_inputs_fp32_accumulator_reassociation | strict_exported_fp64_per_element_bound | valid_strict_oracle | 762.629 | 362.865 | 0.476 |
| large-gemm-1024x1024x1024-fp16-tile64x64x32 | fp16/default | 64x64x32 | tile_mma_typed_inputs_fp32_accumulator_reassociation | strict_exported_fp64_per_element_bound | valid_strict_oracle | 203.434 | 96.923 | 0.476 |
| large-gemm-1024x1024x1024-bf16-tile64x64x32 | bf16/default | 64x64x32 | tile_mma_typed_inputs_fp32_accumulator_reassociation | strict_exported_fp64_per_element_bound | valid_strict_oracle | 204.214 | 94.835 | 0.464 |
| large-gemm-2048x2048x2048-fp32-tile32x32x32 | fp32/default | 32x32x32 | tile_mma_typed_inputs_fp32_accumulator_reassociation | strict_exported_fp64_per_element_bound | valid_strict_oracle | 5843.743 | 2706.391 | 0.463 |
| large-gemm-2048x2048x2048-fp16-tile64x64x32 | fp16/default | 64x64x32 | tile_mma_typed_inputs_fp32_accumulator_reassociation | strict_exported_fp64_per_element_bound | valid_strict_oracle | 1624.863 | 673.193 | 0.414 |
| large-gemm-2048x2048x2048-bf16-tile64x64x32 | bf16/default | 64x64x32 | tile_mma_typed_inputs_fp32_accumulator_reassociation | strict_exported_fp64_per_element_bound | valid_strict_oracle | 1599.724 | 659.558 | 0.412 |
| large-gemm-128x4096x4096-fp32-tile32x32x32 | fp32/default | 32x32x32 | tile_mma_typed_inputs_fp32_accumulator_reassociation | strict_exported_fp64_per_element_bound | valid_strict_oracle | 1791.078 | 771.338 | 0.431 |
| large-gemm-128x4096x4096-fp16-tile64x64x32 | fp16/default | 64x64x32 | tile_mma_typed_inputs_fp32_accumulator_reassociation | strict_exported_fp64_per_element_bound | valid_strict_oracle | 480.891 | 223.455 | 0.465 |
| large-gemm-128x4096x4096-bf16-tile64x64x32 | bf16/default | 64x64x32 | tile_mma_typed_inputs_fp32_accumulator_reassociation | strict_exported_fp64_per_element_bound | valid_strict_oracle | 467.285 | 213.531 | 0.457 |
| large-bmm-2x1024x1024x1024-fp32-tile32x32x32 | fp32/default | 32x32x32 | tile_bmm_singleton_batch_typed_inputs_fp32_accumulator_reassociation | strict_exported_fp64_per_element_bound | valid_strict_oracle | 1631.007 | 640.532 | 0.393 |
| large-bmm-2x1024x1024x1024-fp16-tile64x64x32 | fp16/default | 64x64x32 | tile_bmm_singleton_batch_typed_inputs_fp32_accumulator_reassociation | strict_exported_fp64_per_element_bound | valid_strict_oracle | 421.540 | 181.632 | 0.431 |
| large-bmm-2x1024x1024x1024-bf16-tile64x64x32 | bf16/default | 64x64x32 | tile_bmm_singleton_batch_typed_inputs_fp32_accumulator_reassociation | strict_exported_fp64_per_element_bound | valid_strict_oracle | 415.918 | 181.267 | 0.436 |
| large-gemv-2048x1x8192-fp32-tile1x1x1024 | fp32/default | 1x1x1024 | tile_gemv_product_tree_sum | strict_exported_fp64_per_element_bound | valid_strict_oracle | 269.366 | 268.890 | 0.998 |
| large-gemv-2048x1x8192-fp16-tile1x1x1024 | fp16/default | 1x1x1024 | tile_gemv_product_tree_sum | strict_exported_fp64_per_element_bound | valid_strict_oracle | 75.949 | 135.799 | 1.788 |
| large-gemv-2048x1x8192-bf16-tile1x1x1024 | bf16/default | 1x1x1024 | tile_gemv_product_tree_sum | strict_exported_fp64_per_element_bound | valid_strict_oracle | 75.702 | 135.670 | 1.792 |

Common-envelope SDPA acceptance, where present, is not proof of identical internal arithmetic; secondary strict validation remains in the JSON. A `failed_route`, `unsupported_route`, or `invalid_evidence` record is not a valid performance comparison. Original raw packet/log hashes identify local artifacts even when path labels are normalized. PATH and GPU UUID are omitted.

Checkpoint bytes use UTF-8 with LF newlines. SHA-256: `ec7dd8e3f59090a96cca3229ccc6c9a046161733d42acb0f00df6c114d8d1462`.

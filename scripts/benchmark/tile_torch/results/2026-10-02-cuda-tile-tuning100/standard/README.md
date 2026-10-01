# CUDA Tile graph-v2 cohort checkpoint

This report records individual configurations, not independent workload counts or a pooled win rate. Source/binary receipt sets must match across cohorts; it does not combine historical core78 binaries. Ranking and numerical acceptance contracts remain separate group keys. Aligned16 off/on cohorts cannot be merged.

Records: 82; operation/shape combinations: 27; operation/shape/dtype combinations: 42.

All seven graph samples, full event/host spans, measured R, calibration, warmup, cold times, compiler evidence, failures and pipeline stage receipts are in [checkpoint.json](checkpoint.json). No samples are trimmed. Event and host spans divide by graph batch times R, never stage count. Native pipeline stages may overlap across complete calls when hazards permit. Native preallocated storage and Torch functional capture-pool allocation differ. Compilation boundaries differ, so no compiler-speed ratio is calculated. Serial visits do not establish frequency-controlled causation.

## pointwise12

Matrix status: `passed`. Torch ranking contract: `standard`. Torch mode: `max-autotune`. Aligned16 opt-in: `False`. Original matrix SHA-256: `08f2e29754725c376711c88d468f7bca8e25b5984396f58516b61c26d81aaad1`.

| Case | Dtype/math | Tile | Realized algorithm | Alignment (expected) | Numerical contract | Compare status | Native us | Torch us | Torch/native |
| --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: |
| feature-swiglu-1x8192-bf16-bd8192 | bf16/default | 1x8192x1 | whole_row_tile_fp32_compute | off | strict_exported_fp64_per_element_bound | valid_strict_oracle | 3.342 | 0.820 | 0.245 |
| feature-swiglu-1x8192-bf16-bd256 | bf16/default | 1x256x1 | feature_tiled_pointwise_fp32_compute | off | strict_exported_fp64_per_element_bound | valid_strict_oracle | 0.865 | 0.820 | 0.948 |
| feature-swiglu-1x8192-bf16-bd1024 | bf16/default | 1x1024x1 | feature_tiled_pointwise_fp32_compute | off | strict_exported_fp64_per_element_bound | valid_strict_oracle | 1.048 | 0.822 | 0.784 |
| feature-rope-1x8192-bf16-bd4096 | bf16/default | 1x4096x1 | whole_row_tile_fp32_compute | off | strict_exported_fp64_per_element_bound | valid_strict_oracle | 1.548 | 0.858 | 0.554 |
| feature-rope-1x8192-bf16-bd256 | bf16/default | 1x256x1 | feature_tiled_pointwise_fp32_compute | off | strict_exported_fp64_per_element_bound | valid_strict_oracle | 0.816 | 0.859 | 1.052 |
| feature-rope-1x8192-bf16-bd1024 | bf16/default | 1x1024x1 | feature_tiled_pointwise_fp32_compute | off | strict_exported_fp64_per_element_bound | valid_strict_oracle | 0.968 | 0.859 | 0.887 |
| feature-swiglu-3x65-fp32-bd128 | fp32/default | 1x128x1 | whole_row_tile_fp32_compute | off | strict_exported_fp64_per_element_bound | valid_strict_oracle | 0.880 | 0.785 | 0.891 |
| feature-swiglu-3x65-fp32-bd32 | fp32/default | 1x32x1 | feature_tiled_pointwise_fp32_compute | off | strict_exported_fp64_per_element_bound | valid_strict_oracle | 0.898 | 0.784 | 0.873 |
| feature-gelu_residual-3x65-fp16-bd128 | fp16/default | 1x128x1 | whole_row_tile_fp32_compute | off | strict_exported_fp64_per_element_bound | valid_strict_oracle | 0.779 | 0.853 | 1.095 |
| feature-gelu_residual-3x65-fp16-bd32 | fp16/default | 1x32x1 | feature_tiled_pointwise_fp32_compute | off | strict_exported_fp64_per_element_bound | valid_strict_oracle | 0.782 | 0.786 | 1.005 |
| feature-rope-3x130-bf16-bd128 | bf16/default | 1x128x1 | whole_row_tile_fp32_compute | off | strict_exported_fp64_per_element_bound | valid_strict_oracle | 0.789 | 0.903 | 1.146 |
| feature-rope-3x130-bf16-bd32 | bf16/default | 1x32x1 | feature_tiled_pointwise_fp32_compute | off | strict_exported_fp64_per_element_bound | valid_strict_oracle | 0.789 | 0.903 | 1.145 |

## packed14

Matrix status: `passed`. Torch ranking contract: `stable`. Torch mode: `max-autotune`. Aligned16 opt-in: `False`. Original matrix SHA-256: `4d4c832e108a5470656ef7554d4181077b4b735c28d4e49cd48754fbf2607ee3`.

| Case | Dtype/math | Tile | Realized algorithm | Alignment (expected) | Numerical contract | Compare status | Native us | Torch us | Torch/native |
| --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: |
| packed-opt-in-sort-16x1024x1024-fp32-full_sort_prefix | fp32/default | 1x1024x1 | padded_bitonic_full_sort | off | strict_exported_fp64_per_element_bound | valid_strict_oracle | 12.743 | 14.993 | 1.177 |
| packed-opt-in-sort-16x1024x1024-fp32-packed_fp32 | fp32/default | 1x1024x1 | stable_packed_fp32_full_sort_prefix | off | strict_exported_fp64_per_element_bound | valid_strict_oracle | 11.442 | 14.991 | 1.310 |
| packed-opt-in-sort-128x1024x1024-fp32-full_sort_prefix | fp32/default | 1x1024x1 | padded_bitonic_full_sort | off | strict_exported_fp64_per_element_bound | valid_strict_oracle | 47.548 | 23.943 | 0.504 |
| packed-opt-in-sort-128x1024x1024-fp32-packed_fp32 | fp32/default | 1x1024x1 | stable_packed_fp32_full_sort_prefix | off | strict_exported_fp64_per_element_bound | valid_strict_oracle | 37.517 | 23.937 | 0.638 |
| packed-opt-in-sort-4x4096x4096-fp32-full_sort_prefix | fp32/default | 1x4096x1 | padded_bitonic_full_sort | off | strict_exported_fp64_per_element_bound | valid_strict_oracle | 74.560 | 18.731 | 0.251 |
| packed-opt-in-sort-4x4096x4096-fp32-packed_fp32 | fp32/default | 1x4096x1 | stable_packed_fp32_full_sort_prefix | off | strict_exported_fp64_per_element_bound | valid_strict_oracle | 70.174 | 18.737 | 0.267 |
| packed-opt-in-sort-4x1537x1537-fp32-full_sort_prefix | fp32/default | 1x2048x1 | padded_bitonic_full_sort | off | strict_exported_fp64_per_element_bound | valid_strict_oracle | 31.808 | 15.235 | 0.479 |
| packed-opt-in-sort-4x1537x1537-fp32-packed_fp32 | fp32/default | 1x2048x1 | stable_packed_fp32_full_sort_prefix | off | strict_exported_fp64_per_element_bound | valid_strict_oracle | 27.419 | 15.244 | 0.556 |
| packed-opt-in-sort-16x1024x1024-fp16-full_sort_prefix | fp16/default | 1x1024x1 | padded_bitonic_full_sort | off | strict_exported_fp64_per_element_bound | valid_strict_oracle | 12.717 | 11.375 | 0.894 |
| packed-opt-in-sort-16x1024x1024-fp16-packed_fp32 | fp16/default | 1x1024x1 | stable_packed_fp32_full_sort_prefix | off | strict_exported_fp64_per_element_bound | valid_strict_oracle | 11.434 | 11.378 | 0.995 |
| packed-opt-in-sort-16x1024x1024-bf16-full_sort_prefix | bf16/default | 1x1024x1 | padded_bitonic_full_sort | off | strict_exported_fp64_per_element_bound | valid_strict_oracle | 11.952 | 11.377 | 0.952 |
| packed-opt-in-sort-16x1024x1024-bf16-packed_fp32 | bf16/default | 1x1024x1 | stable_packed_fp32_full_sort_prefix | off | strict_exported_fp64_per_element_bound | valid_strict_oracle | 11.435 | 11.380 | 0.995 |
| packed-opt-in-topk-4x1024x7-fp32-full_sort_prefix | fp32/default | 1x1024x1 | padded_bitonic_full_sort_prefix | off | strict_exported_fp64_per_element_bound | valid_strict_oracle | 12.696 | 14.762 | 1.163 |
| packed-opt-in-topk-4x1024x7-fp32-packed_fp32 | fp32/default | 1x1024x1 | stable_packed_fp32_full_sort_prefix | off | strict_exported_fp64_per_element_bound | valid_strict_oracle | 11.341 | 14.768 | 1.302 |

## sort3

Matrix status: `passed`. Torch ranking contract: `stable`. Torch mode: `max-autotune`. Aligned16 opt-in: `False`. Original matrix SHA-256: `84c344ee2ca3791baaf0bdc042d0a1509178b4b8b28e3087af5b625646dd14f6`.

| Case | Dtype/math | Tile | Realized algorithm | Alignment (expected) | Numerical contract | Compare status | Native us | Torch us | Torch/native |
| --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: |
| schedule-sort-4x8192-fp32-full_sort_prefix | fp32/default | 1x8192x1 | padded_bitonic_full_sort | off | strict_exported_fp64_per_element_bound | valid_strict_oracle | 331.412 | 50.647 | 0.153 |
| schedule-sort-4x8192-fp32-chunked_bitonic_c256 | fp32/default | 1x8192x1 | stable_chunked_bitonic_whole_tile_merge | off | strict_exported_fp64_per_element_bound | valid_strict_oracle | 83.472 | 50.164 | 0.601 |
| schedule-sort-4x8192-fp32-chunked_bitonic_c512 | fp32/default | 1x8192x1 | stable_chunked_bitonic_whole_tile_merge | off | strict_exported_fp64_per_element_bound | valid_strict_oracle | 60.571 | 50.495 | 0.834 |

## alignment18-off

Matrix status: `passed`. Torch ranking contract: `standard`. Torch mode: `max-autotune`. Aligned16 opt-in: `False`. Original matrix SHA-256: `c790941c9feaebbcc410327133a4a83ba806d24d651676732c5940f032c6e552`.

| Case | Dtype/math | Tile | Realized algorithm | Alignment (expected) | Numerical contract | Compare status | Native us | Torch us | Torch/native |
| --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: |
| aligned16-gemm-1024x1024x1024-fp16 | fp16/default | 64x64x32 | tile_mma_typed_inputs_fp32_accumulator_reassociation | off | strict_exported_fp64_per_element_bound | valid_strict_oracle | 206.862 | 97.039 | 0.469 |
| aligned16-scan-128x8192-fp16 | fp16/default | 1x8192x1 | inclusive_sum_unordered_tree | off | strict_exported_fp64_per_element_bound | valid_strict_oracle | 12.353 | 5.577 | 0.451 |
| aligned16-reduce_sum-128x8192-fp16 | fp16/default | 1x8192x1 | whole_row_tile_fp32_compute | off | strict_exported_fp64_per_element_bound | valid_strict_oracle | 2.542 | 2.470 | 0.972 |
| aligned16-rmsnorm-1x8192-fp16 | fp16/default | 1x8192x1 | whole_row_tile_fp32_compute | off | strict_exported_fp64_per_element_bound | valid_strict_oracle | 2.914 | 1.465 | 0.503 |
| aligned16-layernorm-1x8192-fp16 | fp16/default | 1x8192x1 | whole_row_tile_fp32_compute | off | strict_exported_fp64_per_element_bound | valid_strict_oracle | 3.225 | 1.881 | 0.583 |
| aligned16-bmm-2x64x64x64-fp16 | fp16/default | 32x32x32 | tile_bmm_singleton_batch_typed_inputs_fp32_accumulator_reassociation | off | strict_exported_fp64_per_element_bound | valid_strict_oracle | 1.381 | 1.521 | 1.101 |
| aligned16-gemm-127x257x65-fp16 | fp16/default | 32x32x32 | tile_mma_typed_inputs_fp32_accumulator_reassociation | off | strict_exported_fp64_per_element_bound | valid_strict_oracle | 5.377 | 4.588 | 0.853 |
| aligned16-gemm-1024x1024x1024-bf16 | bf16/default | 64x64x32 | tile_mma_typed_inputs_fp32_accumulator_reassociation | off | strict_exported_fp64_per_element_bound | valid_strict_oracle | 207.575 | 95.729 | 0.461 |
| aligned16-scan-128x8192-bf16 | bf16/default | 1x8192x1 | inclusive_sum_unordered_tree | off | strict_exported_fp64_per_element_bound | valid_strict_oracle | 12.402 | 5.640 | 0.455 |
| aligned16-reduce_sum-128x8192-bf16 | bf16/default | 1x8192x1 | whole_row_tile_fp32_compute | off | strict_exported_fp64_per_element_bound | valid_strict_oracle | 2.537 | 2.475 | 0.975 |
| aligned16-rmsnorm-1x8192-bf16 | bf16/default | 1x8192x1 | whole_row_tile_fp32_compute | off | strict_exported_fp64_per_element_bound | valid_strict_oracle | 2.811 | 1.505 | 0.535 |
| aligned16-layernorm-1x8192-bf16 | bf16/default | 1x8192x1 | whole_row_tile_fp32_compute | off | strict_exported_fp64_per_element_bound | valid_strict_oracle | 3.301 | 1.958 | 0.593 |
| aligned16-bmm-2x64x64x64-bf16 | bf16/default | 32x32x32 | tile_bmm_singleton_batch_typed_inputs_fp32_accumulator_reassociation | off | strict_exported_fp64_per_element_bound | valid_strict_oracle | 1.379 | 1.607 | 1.165 |
| aligned16-gemm-127x257x65-bf16 | bf16/default | 32x32x32 | tile_mma_typed_inputs_fp32_accumulator_reassociation | off | strict_exported_fp64_per_element_bound | valid_strict_oracle | 5.368 | 4.600 | 0.857 |
| aligned16-control-scan-128x8192-fp32 | fp32/default | 1x8192x1 | inclusive_sum_unordered_tree | off | strict_exported_fp64_per_element_bound | valid_strict_oracle | 15.212 | 12.998 | 0.854 |
| aligned16-control-rmsnorm-17x65-fp16 | fp16/default | 1x128x1 | whole_row_tile_fp32_compute | off | strict_exported_fp64_per_element_bound | valid_strict_oracle | 1.056 | 0.908 | 0.860 |
| aligned16-softmax-128x1024-fp16 | fp16/default | 1x1024x1 | whole_row_tile_fp32_compute | off | strict_exported_fp64_per_element_bound | valid_strict_oracle | 1.934 | 1.852 | 0.957 |
| aligned16-softmax-128x1024-bf16 | bf16/default | 1x1024x1 | whole_row_tile_fp32_compute | off | strict_exported_fp64_per_element_bound | valid_strict_oracle | 1.889 | 1.665 | 0.882 |

## embedding24

Matrix status: `passed`. Torch ranking contract: `standard`. Torch mode: `max-autotune`. Aligned16 opt-in: `False`. Original matrix SHA-256: `b20aae84a015099f21da5f6c3c21b33ebe46dc1db5aeedc4f873759e364793d1`.

| Case | Dtype/math | Tile | Realized algorithm | Alignment (expected) | Numerical contract | Compare status | Native us | Torch us | Torch/native |
| --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: |
| serial-embedding-v4096-d64-t512-fp32-br1-bd64 | fp32/default | 1x64x1 | uniform_int64_row_gather | off | strict_exported_fp64_per_element_bound | valid_strict_oracle | 1.416 | 1.093 | 0.772 |
| serial-embedding-v4096-d64-t512-fp32-br4-bd64 | fp32/default | 4x64x1 | serial_grouped_uniform_int64_row_gather | off | strict_exported_fp64_per_element_bound | valid_strict_oracle | 1.442 | 1.030 | 0.714 |
| serial-embedding-v4096-d64-t512-fp32-br8-bd64 | fp32/default | 8x64x1 | serial_grouped_uniform_int64_row_gather | off | strict_exported_fp64_per_element_bound | valid_strict_oracle | 2.051 | 1.052 | 0.513 |
| serial-embedding-v4096-d64-t512-fp16-br1-bd64 | fp16/default | 1x64x1 | uniform_int64_row_gather | off | strict_exported_fp64_per_element_bound | valid_strict_oracle | 1.413 | 0.961 | 0.680 |
| serial-embedding-v4096-d64-t512-fp16-br4-bd64 | fp16/default | 4x64x1 | serial_grouped_uniform_int64_row_gather | off | strict_exported_fp64_per_element_bound | valid_strict_oracle | 1.435 | 0.961 | 0.670 |
| serial-embedding-v4096-d64-t512-fp16-br8-bd64 | fp16/default | 8x64x1 | serial_grouped_uniform_int64_row_gather | off | strict_exported_fp64_per_element_bound | valid_strict_oracle | 2.062 | 0.993 | 0.482 |
| serial-embedding-v4096-d64-t512-bf16-br1-bd64 | bf16/default | 1x64x1 | uniform_int64_row_gather | off | strict_exported_fp64_per_element_bound | valid_strict_oracle | 1.413 | 1.070 | 0.758 |
| serial-embedding-v4096-d64-t512-bf16-br4-bd64 | bf16/default | 4x64x1 | serial_grouped_uniform_int64_row_gather | off | strict_exported_fp64_per_element_bound | valid_strict_oracle | 1.435 | 0.963 | 0.671 |
| serial-embedding-v4096-d64-t512-bf16-br8-bd64 | bf16/default | 8x64x1 | serial_grouped_uniform_int64_row_gather | off | strict_exported_fp64_per_element_bound | valid_strict_oracle | 2.064 | 0.963 | 0.467 |
| serial-embedding-v4096-d4096-t512-fp32-br1-bd256 | fp32/default | 1x256x1 | uniform_int64_row_gather | off | strict_exported_fp64_per_element_bound | valid_strict_oracle | 13.257 | 11.368 | 0.858 |
| serial-embedding-v4096-d4096-t512-fp32-br4-bd256 | fp32/default | 4x256x1 | serial_grouped_uniform_int64_row_gather | off | strict_exported_fp64_per_element_bound | valid_strict_oracle | 12.551 | 13.816 | 1.101 |
| serial-embedding-v4096-d4096-t512-fp32-br8-bd256 | fp32/default | 8x256x1 | serial_grouped_uniform_int64_row_gather | off | strict_exported_fp64_per_element_bound | valid_strict_oracle | 12.687 | 11.400 | 0.899 |
| serial-embedding-v4096-d4096-t512-fp16-br1-bd256 | fp16/default | 1x256x1 | uniform_int64_row_gather | off | strict_exported_fp64_per_element_bound | valid_strict_oracle | 11.822 | 5.925 | 0.501 |
| serial-embedding-v4096-d4096-t512-fp16-br4-bd256 | fp16/default | 4x256x1 | serial_grouped_uniform_int64_row_gather | off | strict_exported_fp64_per_element_bound | valid_strict_oracle | 11.228 | 5.745 | 0.512 |
| serial-embedding-v4096-d4096-t512-fp16-br8-bd256 | fp16/default | 8x256x1 | serial_grouped_uniform_int64_row_gather | off | strict_exported_fp64_per_element_bound | valid_strict_oracle | 11.375 | 7.266 | 0.639 |
| serial-embedding-v4096-d4096-t512-bf16-br1-bd256 | bf16/default | 1x256x1 | uniform_int64_row_gather | off | strict_exported_fp64_per_element_bound | valid_strict_oracle | 11.856 | 5.887 | 0.497 |
| serial-embedding-v4096-d4096-t512-bf16-br4-bd256 | bf16/default | 4x256x1 | serial_grouped_uniform_int64_row_gather | off | strict_exported_fp64_per_element_bound | valid_strict_oracle | 11.236 | 5.644 | 0.502 |
| serial-embedding-v4096-d4096-t512-bf16-br8-bd256 | bf16/default | 8x256x1 | serial_grouped_uniform_int64_row_gather | off | strict_exported_fp64_per_element_bound | valid_strict_oracle | 11.376 | 5.669 | 0.498 |
| wide-feature-embedding-v4096-d4096-t512-fp32-br1-bd1024 | fp32/default | 1x1024x1 | uniform_int64_row_gather | off | strict_exported_fp64_per_element_bound | valid_strict_oracle | 11.679 | 11.367 | 0.973 |
| wide-feature-embedding-v4096-d4096-t512-fp32-br1-bd4096 | fp32/default | 1x4096x1 | uniform_int64_row_gather | off | strict_exported_fp64_per_element_bound | valid_strict_oracle | 12.459 | 14.494 | 1.163 |
| wide-feature-embedding-v4096-d4096-t512-fp16-br1-bd1024 | fp16/default | 1x1024x1 | uniform_int64_row_gather | off | strict_exported_fp64_per_element_bound | valid_strict_oracle | 8.496 | 5.912 | 0.696 |
| wide-feature-embedding-v4096-d4096-t512-fp16-br1-bd4096 | fp16/default | 1x4096x1 | uniform_int64_row_gather | off | strict_exported_fp64_per_element_bound | valid_strict_oracle | 9.625 | 7.318 | 0.760 |
| wide-feature-embedding-v4096-d4096-t512-bf16-br1-bd1024 | bf16/default | 1x1024x1 | uniform_int64_row_gather | off | strict_exported_fp64_per_element_bound | valid_strict_oracle | 8.453 | 7.305 | 0.864 |
| wide-feature-embedding-v4096-d4096-t512-bf16-br1-bd4096 | bf16/default | 1x4096x1 | uniform_int64_row_gather | off | strict_exported_fp64_per_element_bound | valid_strict_oracle | 9.613 | 5.635 | 0.586 |

## matrix9

Matrix status: `passed`. Torch ranking contract: `standard`. Torch mode: `max-autotune`. Aligned16 opt-in: `False`. Original matrix SHA-256: `9c9e1b5fcc21a531c5aad3f12c9ea48d3687b861b26e0e4e8e25da29b0c69985`.

| Case | Dtype/math | Tile | Realized algorithm | Alignment (expected) | Numerical contract | Compare status | Native us | Torch us | Torch/native |
| --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: |
| schedule-gemm-32x1024x4096-fp32-32x32x32 | fp32/default | 32x32x32 | tile_mma_typed_inputs_fp32_accumulator_reassociation | off | strict_exported_fp64_per_element_bound | valid_strict_oracle | 133.208 | 52.728 | 0.396 |
| schedule-gemm-32x1024x4096-fp32-16x32x32 | fp32/default | 16x32x32 | tile_mma_typed_inputs_fp32_accumulator_reassociation | off | strict_exported_fp64_per_element_bound | valid_strict_oracle | 156.312 | 53.292 | 0.341 |
| schedule-gemm-32x1024x4096-fp32-32x16x32 | fp32/default | 32x16x32 | tile_mma_typed_inputs_fp32_accumulator_reassociation | off | strict_exported_fp64_per_element_bound | valid_strict_oracle | 146.745 | 52.274 | 0.356 |
| schedule-gemm-4x4096x4096-bf16-16x64x32 | bf16/default | 16x64x32 | tile_mma_typed_inputs_fp32_accumulator_reassociation | off | strict_exported_fp64_per_element_bound | valid_strict_oracle | 100.615 | 85.289 | 0.848 |
| schedule-gemm-4x4096x4096-bf16-4x64x32 | bf16/default | 4x64x32 | tile_mma_typed_inputs_fp32_accumulator_reassociation | off | strict_exported_fp64_per_element_bound | valid_strict_oracle | 121.520 | 85.670 | 0.705 |
| schedule-gemm-4x4096x4096-bf16-4x32x32 | bf16/default | 4x32x32 | tile_mma_typed_inputs_fp32_accumulator_reassociation | off | strict_exported_fp64_per_element_bound | valid_strict_oracle | 121.227 | 84.716 | 0.699 |
| schedule-gemv-4096x4096-fp32-1x1x1024 | fp32/default | 1x1x1024 | tile_gemv_product_tree_sum | off | strict_exported_fp64_per_element_bound | valid_strict_oracle | 269.356 | 269.043 | 0.999 |
| schedule-gemv-4096x4096-fp32-4x1x1024 | fp32/default | 4x1x1024 | tile_gemv_product_tree_sum | off | strict_exported_fp64_per_element_bound | valid_strict_oracle | 269.412 | 269.519 | 1.000 |
| schedule-gemv-4096x4096-fp32-1x1x512 | fp32/default | 1x1x512 | tile_gemv_product_tree_sum | off | strict_exported_fp64_per_element_bound | valid_strict_oracle | 270.134 | 269.568 | 0.998 |

## mha2

Matrix status: `passed`. Torch ranking contract: `standard`. Torch mode: `max-autotune`. Aligned16 opt-in: `False`. Original matrix SHA-256: `93f79b218758f5f45e014dbd12b7383f7da9ed406604e7351554555bcf7e4949`.

| Case | Dtype/math | Tile | Realized algorithm | Alignment (expected) | Numerical contract | Compare status | Native us | Torch us | Torch/native |
| --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: |
| coverage-attention-mha-b2-h4-q1-k65-d32-fp32 | fp32/default | 1x32x1 | causal_online_softmax_gqa | off | strict_exported_fp64_per_element_bound | valid_strict_oracle | 3.863 | 14.573 | 3.772 |
| coverage-attention-mha-b2-h4-q17-k65-d32-fp32 | fp32/default | 16x32x1 | causal_online_softmax_gqa | off | strict_exported_fp64_per_element_bound | valid_strict_oracle | 6.957 | 16.594 | 2.385 |

Common-envelope SDPA acceptance, where present, is not proof of identical internal arithmetic; secondary strict validation remains in the JSON. A `failed_route`, `unsupported_route`, or `invalid_evidence` record is not a valid performance comparison. Alignment is the predicted shared host selector result from actual final-pointer residues, not a device execution trace; requested-but-ineligible cases stay explicit. Original raw packet/log hashes identify local artifacts even when path labels are normalized. PATH and GPU UUID are omitted.

Checkpoint bytes use UTF-8 with LF newlines. SHA-256: `2c4d71ec4ef18f3b601427d9033c77aad7457257215416d895017dfa33dd928e`.

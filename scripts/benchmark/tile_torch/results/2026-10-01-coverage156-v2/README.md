# CUDA Tile graph-v2 cohort checkpoint

This report records individual configurations, not independent workload counts or a pooled win rate. Source/binary receipt sets must match across cohorts; it does not combine historical core78 binaries. Ranking and numerical acceptance contracts remain separate group keys.

Records: 156; operation/shape combinations: 51; operation/shape/dtype combinations: 94.

All seven graph samples, full event/host spans, measured R, calibration, warmup, cold times, compiler evidence, failures and pipeline stage receipts are in [checkpoint.json](checkpoint.json). No samples are trimmed. Event and host spans divide by graph batch times R, never stage count. Native pipeline stages may overlap across complete calls when hazards permit. Native preallocated storage and Torch functional capture-pool allocation differ. Compilation boundaries differ, so no compiler-speed ratio is calculated. Serial visits do not establish frequency-controlled causation.

## smoke6

Matrix status: `passed`. Torch ranking contract: `standard`. Torch mode: `max-autotune`. Original matrix SHA-256: `bbfc67bad712ddb0a152fb5393dce56542d69b25f2a05a9ae0bbead47e295f85`.

| Case | Dtype/math | Tile | Realized algorithm | Numerical contract | Compare status | Native us | Torch us | Torch/native |
| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: |
| argmax-17x65-fp32 | fp32/default | 1x128x1 | stable_first_index_argmax | strict_exported_fp64_per_element_bound | valid_strict_oracle | 1.103 | 0.963 | 0.873 |
| argmax-17x65-fp16 | fp16/default | 1x128x1 | stable_first_index_argmax | strict_exported_fp64_per_element_bound | valid_strict_oracle | 1.125 | 0.973 | 0.865 |
| argmax-17x65-bf16 | bf16/default | 1x128x1 | stable_first_index_argmax | strict_exported_fp64_per_element_bound | valid_strict_oracle | 1.126 | 0.973 | 0.865 |
| embedding-v17-d65-t37-bd32-adversarial-fp32 | fp32/default | 1x32x1 | uniform_int64_row_gather | strict_exported_fp64_per_element_bound | valid_strict_oracle | 0.926 | 0.888 | 0.959 |
| embedding-v17-d65-t37-bd32-adversarial-fp16 | fp16/default | 1x32x1 | uniform_int64_row_gather | strict_exported_fp64_per_element_bound | valid_strict_oracle | 0.927 | 0.897 | 0.967 |
| embedding-v17-d65-t37-bd32-adversarial-bf16 | bf16/default | 1x32x1 | uniform_int64_row_gather | strict_exported_fp64_per_element_bound | valid_strict_oracle | 0.927 | 0.956 | 1.031 |

## selection30

Matrix status: `passed`. Torch ranking contract: `standard`. Torch mode: `max-autotune`. Original matrix SHA-256: `18501e4d27a9d5f239578b72bc8e23d0233004631e33944e62c6a2083df7374b`.

| Case | Dtype/math | Tile | Realized algorithm | Numerical contract | Compare status | Native us | Torch us | Torch/native |
| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: |
| argmax-1x1-fp32 | fp32/default | 1x1x1 | stable_first_index_argmax | strict_exported_fp64_per_element_bound | valid_strict_oracle | 0.736 | 1.359 | 1.848 |
| argmax-1x1-fp16 | fp16/default | 1x1x1 | stable_first_index_argmax | strict_exported_fp64_per_element_bound | valid_strict_oracle | 0.747 | 1.360 | 1.820 |
| argmax-1x1-bf16 | bf16/default | 1x1x1 | stable_first_index_argmax | strict_exported_fp64_per_element_bound | valid_strict_oracle | 0.748 | 1.362 | 1.821 |
| argmax-4x33-fp32 | fp32/default | 1x64x1 | stable_first_index_argmax | strict_exported_fp64_per_element_bound | valid_strict_oracle | 1.050 | 0.881 | 0.840 |
| argmax-4x33-fp16 | fp16/default | 1x64x1 | stable_first_index_argmax | strict_exported_fp64_per_element_bound | valid_strict_oracle | 1.064 | 0.927 | 0.872 |
| argmax-4x33-bf16 | bf16/default | 1x64x1 | stable_first_index_argmax | strict_exported_fp64_per_element_bound | valid_strict_oracle | 1.064 | 0.848 | 0.797 |
| argmax-128x1024-fp32 | fp32/default | 1x1024x1 | stable_first_index_argmax | strict_exported_fp64_per_element_bound | valid_strict_oracle | 1.832 | 2.012 | 1.098 |
| argmax-128x1024-fp16 | fp16/default | 1x1024x1 | stable_first_index_argmax | strict_exported_fp64_per_element_bound | valid_strict_oracle | 1.736 | 1.609 | 0.927 |
| argmax-128x1024-bf16 | bf16/default | 1x1024x1 | stable_first_index_argmax | strict_exported_fp64_per_element_bound | valid_strict_oracle | 1.736 | 2.014 | 1.160 |
| argmax-1x4096-fp32 | fp32/default | 1x4096x1 | stable_first_index_argmax | strict_exported_fp64_per_element_bound | valid_strict_oracle | 1.450 | 1.410 | 0.972 |
| argmax-1x4096-fp16 | fp16/default | 1x4096x1 | stable_first_index_argmax | strict_exported_fp64_per_element_bound | valid_strict_oracle | 1.372 | 1.696 | 1.237 |
| argmax-1x4096-bf16 | bf16/default | 1x4096x1 | stable_first_index_argmax | strict_exported_fp64_per_element_bound | valid_strict_oracle | 1.374 | 1.324 | 0.964 |
| argmax-4x8192-fp32 | fp32/default | 1x8192x1 | stable_first_index_argmax | strict_exported_fp64_per_element_bound | valid_strict_oracle | 1.821 | 1.797 | 0.987 |
| argmax-4x8192-fp16 | fp16/default | 1x8192x1 | stable_first_index_argmax | strict_exported_fp64_per_element_bound | valid_strict_oracle | 1.652 | 1.736 | 1.051 |
| argmax-4x8192-bf16 | bf16/default | 1x8192x1 | stable_first_index_argmax | strict_exported_fp64_per_element_bound | valid_strict_oracle | 1.642 | 3.137 | 1.911 |
| embedding-v4096-d64-t1-bd64-adversarial-fp32 | fp32/default | 1x64x1 | uniform_int64_row_gather | strict_exported_fp64_per_element_bound | valid_strict_oracle | 0.829 | 0.887 | 1.070 |
| embedding-v4096-d64-t1-bd64-adversarial-fp16 | fp16/default | 1x64x1 | uniform_int64_row_gather | strict_exported_fp64_per_element_bound | valid_strict_oracle | 0.826 | 0.873 | 1.057 |
| embedding-v4096-d64-t1-bd64-adversarial-bf16 | bf16/default | 1x64x1 | uniform_int64_row_gather | strict_exported_fp64_per_element_bound | valid_strict_oracle | 0.826 | 0.873 | 1.058 |
| embedding-v4096-d1024-t32-bd256-adversarial-fp32 | fp32/default | 1x256x1 | uniform_int64_row_gather | strict_exported_fp64_per_element_bound | valid_strict_oracle | 0.998 | 1.033 | 1.035 |
| embedding-v4096-d1024-t32-bd256-adversarial-fp16 | fp16/default | 1x256x1 | uniform_int64_row_gather | strict_exported_fp64_per_element_bound | valid_strict_oracle | 0.964 | 1.006 | 1.043 |
| embedding-v4096-d1024-t32-bd256-adversarial-bf16 | bf16/default | 1x256x1 | uniform_int64_row_gather | strict_exported_fp64_per_element_bound | valid_strict_oracle | 0.964 | 0.971 | 1.007 |
| embedding-v4096-d4096-t512-bd256-adversarial-fp32 | fp32/default | 1x256x1 | uniform_int64_row_gather | strict_exported_fp64_per_element_bound | valid_strict_oracle | 13.267 | 13.793 | 1.040 |
| embedding-v4096-d4096-t512-bd256-adversarial-fp16 | fp16/default | 1x256x1 | uniform_int64_row_gather | strict_exported_fp64_per_element_bound | valid_strict_oracle | 11.820 | 5.737 | 0.485 |
| embedding-v4096-d4096-t512-bd256-adversarial-bf16 | bf16/default | 1x256x1 | uniform_int64_row_gather | strict_exported_fp64_per_element_bound | valid_strict_oracle | 11.887 | 7.285 | 0.613 |
| embedding-v4096-d64-t512-bd64-random-fp32 | fp32/default | 1x64x1 | uniform_int64_row_gather | strict_exported_fp64_per_element_bound | valid_strict_oracle | 1.417 | 1.033 | 0.729 |
| embedding-v4096-d64-t512-bd64-random-fp16 | fp16/default | 1x64x1 | uniform_int64_row_gather | strict_exported_fp64_per_element_bound | valid_strict_oracle | 1.413 | 0.995 | 0.704 |
| embedding-v4096-d64-t512-bd64-random-bf16 | bf16/default | 1x64x1 | uniform_int64_row_gather | strict_exported_fp64_per_element_bound | valid_strict_oracle | 1.413 | 0.962 | 0.681 |
| embedding-v4096-d4096-t32-bd256-random-fp32 | fp32/default | 1x256x1 | uniform_int64_row_gather | strict_exported_fp64_per_element_bound | valid_strict_oracle | 1.650 | 1.662 | 1.008 |
| embedding-v4096-d4096-t32-bd256-random-fp16 | fp16/default | 1x256x1 | uniform_int64_row_gather | strict_exported_fp64_per_element_bound | valid_strict_oracle | 1.448 | 1.465 | 1.012 |
| embedding-v4096-d4096-t32-bd256-random-bf16 | bf16/default | 1x256x1 | uniform_int64_row_gather | strict_exported_fp64_per_element_bound | valid_strict_oracle | 1.442 | 1.212 | 0.840 |

## row54

Matrix status: `passed`. Torch ranking contract: `standard`. Torch mode: `max-autotune`. Original matrix SHA-256: `9100d8c159d53db3007c750cea88b4111a35b76a4f7a4e9fcec84942baa25fe7`.

| Case | Dtype/math | Tile | Realized algorithm | Numerical contract | Compare status | Native us | Torch us | Torch/native |
| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: |
| rowblock-scan-128x65-fp32-br1 | fp32/default | 1x128x1 | inclusive_sum_unordered_tree | strict_exported_fp64_per_element_bound | valid_strict_oracle | 0.971 | 1.002 | 1.032 |
| rowblock-scan-128x65-fp32-br4 | fp32/default | 4x128x1 | inclusive_sum_unordered_tree | strict_exported_fp64_per_element_bound | valid_strict_oracle | 1.018 | 0.980 | 0.963 |
| rowblock-scan-128x65-fp32-br8 | fp32/default | 8x128x1 | inclusive_sum_unordered_tree | strict_exported_fp64_per_element_bound | valid_strict_oracle | 1.041 | 0.968 | 0.930 |
| rowblock-scan-128x65-fp16-br1 | fp16/default | 1x128x1 | inclusive_sum_unordered_tree | strict_exported_fp64_per_element_bound | valid_strict_oracle | 0.978 | 1.013 | 1.035 |
| rowblock-scan-128x65-fp16-br4 | fp16/default | 4x128x1 | inclusive_sum_unordered_tree | strict_exported_fp64_per_element_bound | valid_strict_oracle | 1.034 | 1.012 | 0.979 |
| rowblock-scan-128x65-fp16-br8 | fp16/default | 8x128x1 | inclusive_sum_unordered_tree | strict_exported_fp64_per_element_bound | valid_strict_oracle | 1.064 | 1.015 | 0.954 |
| rowblock-scan-128x65-bf16-br1 | bf16/default | 1x128x1 | inclusive_sum_unordered_tree | strict_exported_fp64_per_element_bound | valid_strict_oracle | 0.981 | 1.011 | 1.031 |
| rowblock-scan-128x65-bf16-br4 | bf16/default | 4x128x1 | inclusive_sum_unordered_tree | strict_exported_fp64_per_element_bound | valid_strict_oracle | 1.054 | 1.037 | 0.984 |
| rowblock-scan-128x65-bf16-br8 | bf16/default | 8x128x1 | inclusive_sum_unordered_tree | strict_exported_fp64_per_element_bound | valid_strict_oracle | 1.051 | 1.037 | 0.987 |
| rowblock-reduce_sum-128x65-fp32-br1 | fp32/default | 1x128x1 | whole_row_tile_fp32_compute | strict_exported_fp64_per_element_bound | valid_strict_oracle | 1.022 | 0.979 | 0.958 |
| rowblock-reduce_sum-128x65-fp32-br4 | fp32/default | 4x128x1 | whole_row_tile_fp32_compute | strict_exported_fp64_per_element_bound | valid_strict_oracle | 0.947 | 0.978 | 1.033 |
| rowblock-reduce_sum-128x65-fp32-br8 | fp32/default | 8x128x1 | whole_row_tile_fp32_compute | strict_exported_fp64_per_element_bound | valid_strict_oracle | 0.959 | 1.003 | 1.046 |
| rowblock-reduce_sum-128x65-fp16-br1 | fp16/default | 1x128x1 | whole_row_tile_fp32_compute | strict_exported_fp64_per_element_bound | valid_strict_oracle | 1.028 | 0.984 | 0.957 |
| rowblock-reduce_sum-128x65-fp16-br4 | fp16/default | 4x128x1 | whole_row_tile_fp32_compute | strict_exported_fp64_per_element_bound | valid_strict_oracle | 0.942 | 1.005 | 1.066 |
| rowblock-reduce_sum-128x65-fp16-br8 | fp16/default | 8x128x1 | whole_row_tile_fp32_compute | strict_exported_fp64_per_element_bound | valid_strict_oracle | 0.967 | 1.005 | 1.040 |
| rowblock-reduce_sum-128x65-bf16-br1 | bf16/default | 1x128x1 | whole_row_tile_fp32_compute | strict_exported_fp64_per_element_bound | valid_strict_oracle | 1.030 | 1.093 | 1.060 |
| rowblock-reduce_sum-128x65-bf16-br4 | bf16/default | 4x128x1 | whole_row_tile_fp32_compute | strict_exported_fp64_per_element_bound | valid_strict_oracle | 0.938 | 1.007 | 1.073 |
| rowblock-reduce_sum-128x65-bf16-br8 | bf16/default | 8x128x1 | whole_row_tile_fp32_compute | strict_exported_fp64_per_element_bound | valid_strict_oracle | 0.967 | 0.992 | 1.025 |
| rowblock-reduce_max-128x65-fp32-br1 | fp32/default | 1x128x1 | whole_row_tile_fp32_compute | strict_exported_fp64_per_element_bound | valid_strict_oracle | 1.017 | 0.978 | 0.962 |
| rowblock-reduce_max-128x65-fp32-br4 | fp32/default | 4x128x1 | whole_row_tile_fp32_compute | strict_exported_fp64_per_element_bound | valid_strict_oracle | 0.937 | 1.011 | 1.079 |
| rowblock-reduce_max-128x65-fp32-br8 | fp32/default | 8x128x1 | whole_row_tile_fp32_compute | strict_exported_fp64_per_element_bound | valid_strict_oracle | 0.954 | 0.975 | 1.022 |
| rowblock-reduce_max-128x65-fp16-br1 | fp16/default | 1x128x1 | whole_row_tile_fp32_compute | strict_exported_fp64_per_element_bound | valid_strict_oracle | 1.025 | 1.009 | 0.984 |
| rowblock-reduce_max-128x65-fp16-br4 | fp16/default | 4x128x1 | whole_row_tile_fp32_compute | strict_exported_fp64_per_element_bound | valid_strict_oracle | 0.937 | 1.012 | 1.079 |
| rowblock-reduce_max-128x65-fp16-br8 | fp16/default | 8x128x1 | whole_row_tile_fp32_compute | strict_exported_fp64_per_element_bound | valid_strict_oracle | 0.955 | 0.980 | 1.027 |
| rowblock-reduce_max-128x65-bf16-br1 | bf16/default | 1x128x1 | whole_row_tile_fp32_compute | strict_exported_fp64_per_element_bound | valid_strict_oracle | 1.026 | 0.995 | 0.970 |
| rowblock-reduce_max-128x65-bf16-br4 | bf16/default | 4x128x1 | whole_row_tile_fp32_compute | strict_exported_fp64_per_element_bound | valid_strict_oracle | 0.937 | 1.009 | 1.077 |
| rowblock-reduce_max-128x65-bf16-br8 | bf16/default | 8x128x1 | whole_row_tile_fp32_compute | strict_exported_fp64_per_element_bound | valid_strict_oracle | 0.953 | 1.008 | 1.057 |
| rowblock-scan-17x1024-fp32-br1 | fp32/default | 1x1024x1 | inclusive_sum_unordered_tree | strict_exported_fp64_per_element_bound | valid_strict_oracle | 1.048 | 0.978 | 0.933 |
| rowblock-scan-17x1024-fp32-br4 | fp32/default | 4x1024x1 | inclusive_sum_unordered_tree | strict_exported_fp64_per_element_bound | valid_strict_oracle | 2.484 | 0.964 | 0.388 |
| rowblock-scan-17x1024-fp32-br8 | fp32/default | 8x1024x1 | inclusive_sum_unordered_tree | strict_exported_fp64_per_element_bound | valid_strict_oracle | 4.649 | 1.086 | 0.234 |
| rowblock-scan-17x1024-fp16-br1 | fp16/default | 1x1024x1 | inclusive_sum_unordered_tree | strict_exported_fp64_per_element_bound | valid_strict_oracle | 1.020 | 0.932 | 0.913 |
| rowblock-scan-17x1024-fp16-br4 | fp16/default | 4x1024x1 | inclusive_sum_unordered_tree | strict_exported_fp64_per_element_bound | valid_strict_oracle | 2.716 | 0.932 | 0.343 |
| rowblock-scan-17x1024-fp16-br8 | fp16/default | 8x1024x1 | inclusive_sum_unordered_tree | strict_exported_fp64_per_element_bound | valid_strict_oracle | 6.156 | 0.932 | 0.151 |
| rowblock-scan-17x1024-bf16-br1 | bf16/default | 1x1024x1 | inclusive_sum_unordered_tree | strict_exported_fp64_per_element_bound | valid_strict_oracle | 1.021 | 0.933 | 0.914 |
| rowblock-scan-17x1024-bf16-br4 | bf16/default | 4x1024x1 | inclusive_sum_unordered_tree | strict_exported_fp64_per_element_bound | valid_strict_oracle | 2.566 | 0.933 | 0.364 |
| rowblock-scan-17x1024-bf16-br8 | bf16/default | 8x1024x1 | inclusive_sum_unordered_tree | strict_exported_fp64_per_element_bound | valid_strict_oracle | 4.812 | 0.932 | 0.194 |
| rowblock-reduce_sum-17x1024-fp32-br1 | fp32/default | 1x1024x1 | whole_row_tile_fp32_compute | strict_exported_fp64_per_element_bound | valid_strict_oracle | 0.950 | 0.954 | 1.004 |
| rowblock-reduce_sum-17x1024-fp32-br4 | fp32/default | 4x1024x1 | whole_row_tile_fp32_compute | strict_exported_fp64_per_element_bound | valid_strict_oracle | 1.281 | 0.955 | 0.746 |
| rowblock-reduce_sum-17x1024-fp32-br8 | fp32/default | 8x1024x1 | whole_row_tile_fp32_compute | strict_exported_fp64_per_element_bound | valid_strict_oracle | 1.758 | 0.955 | 0.543 |
| rowblock-reduce_sum-17x1024-fp16-br1 | fp16/default | 1x1024x1 | whole_row_tile_fp32_compute | strict_exported_fp64_per_element_bound | valid_strict_oracle | 0.930 | 0.907 | 0.975 |
| rowblock-reduce_sum-17x1024-fp16-br4 | fp16/default | 4x1024x1 | whole_row_tile_fp32_compute | strict_exported_fp64_per_element_bound | valid_strict_oracle | 2.213 | 0.914 | 0.413 |
| rowblock-reduce_sum-17x1024-fp16-br8 | fp16/default | 8x1024x1 | whole_row_tile_fp32_compute | strict_exported_fp64_per_element_bound | valid_strict_oracle | 3.543 | 0.915 | 0.258 |
| rowblock-reduce_sum-17x1024-bf16-br1 | bf16/default | 1x1024x1 | whole_row_tile_fp32_compute | strict_exported_fp64_per_element_bound | valid_strict_oracle | 0.931 | 0.917 | 0.985 |
| rowblock-reduce_sum-17x1024-bf16-br4 | bf16/default | 4x1024x1 | whole_row_tile_fp32_compute | strict_exported_fp64_per_element_bound | valid_strict_oracle | 1.292 | 0.918 | 0.710 |
| rowblock-reduce_sum-17x1024-bf16-br8 | bf16/default | 8x1024x1 | whole_row_tile_fp32_compute | strict_exported_fp64_per_element_bound | valid_strict_oracle | 1.770 | 0.987 | 0.558 |
| rowblock-reduce_max-17x1024-fp32-br1 | fp32/default | 1x1024x1 | whole_row_tile_fp32_compute | strict_exported_fp64_per_element_bound | valid_strict_oracle | 0.954 | 0.976 | 1.023 |
| rowblock-reduce_max-17x1024-fp32-br4 | fp32/default | 4x1024x1 | whole_row_tile_fp32_compute | strict_exported_fp64_per_element_bound | valid_strict_oracle | 1.247 | 1.022 | 0.820 |
| rowblock-reduce_max-17x1024-fp32-br8 | fp32/default | 8x1024x1 | whole_row_tile_fp32_compute | strict_exported_fp64_per_element_bound | valid_strict_oracle | 1.743 | 0.960 | 0.551 |
| rowblock-reduce_max-17x1024-fp16-br1 | fp16/default | 1x1024x1 | whole_row_tile_fp32_compute | strict_exported_fp64_per_element_bound | valid_strict_oracle | 0.934 | 0.922 | 0.987 |
| rowblock-reduce_max-17x1024-fp16-br4 | fp16/default | 4x1024x1 | whole_row_tile_fp32_compute | strict_exported_fp64_per_element_bound | valid_strict_oracle | 2.179 | 0.921 | 0.423 |
| rowblock-reduce_max-17x1024-fp16-br8 | fp16/default | 8x1024x1 | whole_row_tile_fp32_compute | strict_exported_fp64_per_element_bound | valid_strict_oracle | 3.309 | 0.920 | 0.278 |
| rowblock-reduce_max-17x1024-bf16-br1 | bf16/default | 1x1024x1 | whole_row_tile_fp32_compute | strict_exported_fp64_per_element_bound | valid_strict_oracle | 0.937 | 0.916 | 0.977 |
| rowblock-reduce_max-17x1024-bf16-br4 | bf16/default | 4x1024x1 | whole_row_tile_fp32_compute | strict_exported_fp64_per_element_bound | valid_strict_oracle | 1.283 | 0.915 | 0.713 |
| rowblock-reduce_max-17x1024-bf16-br8 | bf16/default | 8x1024x1 | whole_row_tile_fp32_compute | strict_exported_fp64_per_element_bound | valid_strict_oracle | 1.756 | 0.920 | 0.524 |

## gemm8

Matrix status: `passed`. Torch ranking contract: `standard`. Torch mode: `max-autotune`. Original matrix SHA-256: `c81d613aea36e74f3cf9fe9b4a519c38e196b67297c64a32fa4420bce0c4db9a`.

| Case | Dtype/math | Tile | Realized algorithm | Numerical contract | Compare status | Native us | Torch us | Torch/native |
| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: |
| narrow1024-fp16-tile64x64x32 | fp16/default | 64x64x32 | tile_mma_typed_inputs_fp32_accumulator_reassociation | strict_exported_fp64_per_element_bound | valid_strict_oracle | 198.351 | 95.104 | 0.479 |
| narrow1024-fp16-tile128x64x32 | fp16/default | 128x64x32 | tile_mma_typed_inputs_fp32_accumulator_reassociation | strict_exported_fp64_per_element_bound | valid_strict_oracle | 271.782 | 96.047 | 0.353 |
| narrow1024-fp16-tile64x128x32 | fp16/default | 64x128x32 | tile_mma_typed_inputs_fp32_accumulator_reassociation | strict_exported_fp64_per_element_bound | valid_strict_oracle | 257.600 | 95.795 | 0.372 |
| narrow1024-fp16-tile64x64x64 | fp16/default | 64x64x64 | tile_mma_typed_inputs_fp32_accumulator_reassociation | strict_exported_fp64_per_element_bound | valid_strict_oracle | 229.429 | 96.478 | 0.421 |
| narrow1024-bf16-tile64x64x32 | bf16/default | 64x64x32 | tile_mma_typed_inputs_fp32_accumulator_reassociation | strict_exported_fp64_per_element_bound | valid_strict_oracle | 201.351 | 94.459 | 0.469 |
| narrow1024-bf16-tile128x64x32 | bf16/default | 128x64x32 | tile_mma_typed_inputs_fp32_accumulator_reassociation | strict_exported_fp64_per_element_bound | valid_strict_oracle | 275.540 | 94.794 | 0.344 |
| narrow1024-bf16-tile64x128x32 | bf16/default | 64x128x32 | tile_mma_typed_inputs_fp32_accumulator_reassociation | strict_exported_fp64_per_element_bound | valid_strict_oracle | 254.346 | 94.754 | 0.373 |
| narrow1024-bf16-tile64x64x64 | bf16/default | 64x64x64 | tile_mma_typed_inputs_fp32_accumulator_reassociation | strict_exported_fp64_per_element_bound | valid_strict_oracle | 221.497 | 95.463 | 0.431 |

## long27

Matrix status: `failed`. Torch ranking contract: `standard`. Torch mode: `max-autotune`. Original matrix SHA-256: `9cc10be5488894a9d0ff5b7f02f489e61b6693ff94241a74343f35ae6c103006`.

| Case | Dtype/math | Tile | Realized algorithm | Numerical contract | Compare status | Native us | Torch us | Torch/native |
| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: |
| rowblock-scan-128x8192-fp32-br1 | fp32/default | 1x8192x1 | inclusive_sum_unordered_tree | strict_exported_fp64_per_element_bound | valid_strict_oracle | 14.378 | 13.294 | 0.925 |
| rowblock-scan-128x8192-fp32-br4 | fp32/default | 4x8192x1 | inclusive_sum_unordered_tree | strict_exported_fp64_per_element_bound | failed_route | -- | 12.282 | -- |
| rowblock-scan-128x8192-fp32-br8 | fp32/default | 8x8192x1 | inclusive_sum_unordered_tree | strict_exported_fp64_per_element_bound | failed_route | -- | 12.401 | -- |
| rowblock-scan-128x8192-fp16-br1 | fp16/default | 1x8192x1 | inclusive_sum_unordered_tree | strict_exported_fp64_per_element_bound | valid_strict_oracle | 12.351 | 5.578 | 0.452 |
| rowblock-scan-128x8192-fp16-br4 | fp16/default | 4x8192x1 | inclusive_sum_unordered_tree | strict_exported_fp64_per_element_bound | failed_route | -- | 5.578 | -- |
| rowblock-scan-128x8192-fp16-br8 | fp16/default | 8x8192x1 | inclusive_sum_unordered_tree | strict_exported_fp64_per_element_bound | failed_route | -- | 5.475 | -- |
| rowblock-scan-128x8192-bf16-br1 | bf16/default | 1x8192x1 | inclusive_sum_unordered_tree | strict_exported_fp64_per_element_bound | valid_strict_oracle | 12.365 | 5.632 | 0.455 |
| rowblock-scan-128x8192-bf16-br4 | bf16/default | 4x8192x1 | inclusive_sum_unordered_tree | strict_exported_fp64_per_element_bound | failed_route | -- | 6.290 | -- |
| rowblock-scan-128x8192-bf16-br8 | bf16/default | 8x8192x1 | inclusive_sum_unordered_tree | strict_exported_fp64_per_element_bound | failed_route | -- | 6.278 | -- |
| rowblock-reduce_sum-128x8192-fp32-br1 | fp32/default | 1x8192x1 | whole_row_tile_fp32_compute | strict_exported_fp64_per_element_bound | valid_strict_oracle | 3.881 | 4.521 | 1.165 |
| rowblock-reduce_sum-128x8192-fp32-br4 | fp32/default | 4x8192x1 | whole_row_tile_fp32_compute | strict_exported_fp64_per_element_bound | valid_strict_oracle | 4.614 | 4.839 | 1.049 |
| rowblock-reduce_sum-128x8192-fp32-br8 | fp32/default | 8x8192x1 | whole_row_tile_fp32_compute | strict_exported_fp64_per_element_bound | valid_strict_oracle | 4.864 | 4.796 | 0.986 |
| rowblock-reduce_sum-128x8192-fp16-br1 | fp16/default | 1x8192x1 | whole_row_tile_fp32_compute | strict_exported_fp64_per_element_bound | valid_strict_oracle | 2.540 | 2.463 | 0.970 |
| rowblock-reduce_sum-128x8192-fp16-br4 | fp16/default | 4x8192x1 | whole_row_tile_fp32_compute | strict_exported_fp64_per_element_bound | valid_strict_oracle | 3.123 | 2.546 | 0.815 |
| rowblock-reduce_sum-128x8192-fp16-br8 | fp16/default | 8x8192x1 | whole_row_tile_fp32_compute | strict_exported_fp64_per_element_bound | valid_strict_oracle | 3.061 | 2.468 | 0.806 |
| rowblock-reduce_sum-128x8192-bf16-br1 | bf16/default | 1x8192x1 | whole_row_tile_fp32_compute | strict_exported_fp64_per_element_bound | valid_strict_oracle | 2.515 | 2.578 | 1.025 |
| rowblock-reduce_sum-128x8192-bf16-br4 | bf16/default | 4x8192x1 | whole_row_tile_fp32_compute | strict_exported_fp64_per_element_bound | valid_strict_oracle | 3.174 | 2.581 | 0.813 |
| rowblock-reduce_sum-128x8192-bf16-br8 | bf16/default | 8x8192x1 | whole_row_tile_fp32_compute | strict_exported_fp64_per_element_bound | valid_strict_oracle | 3.894 | 2.474 | 0.635 |
| rowblock-reduce_max-128x8192-fp32-br1 | fp32/default | 1x8192x1 | whole_row_tile_fp32_compute | strict_exported_fp64_per_element_bound | valid_strict_oracle | 4.083 | 5.371 | 1.316 |
| rowblock-reduce_max-128x8192-fp32-br4 | fp32/default | 4x8192x1 | whole_row_tile_fp32_compute | strict_exported_fp64_per_element_bound | valid_strict_oracle | 4.755 | 5.358 | 1.127 |
| rowblock-reduce_max-128x8192-fp32-br8 | fp32/default | 8x8192x1 | whole_row_tile_fp32_compute | strict_exported_fp64_per_element_bound | valid_strict_oracle | 5.057 | 4.959 | 0.981 |
| rowblock-reduce_max-128x8192-fp16-br1 | fp16/default | 1x8192x1 | whole_row_tile_fp32_compute | strict_exported_fp64_per_element_bound | valid_strict_oracle | 2.745 | 2.705 | 0.985 |
| rowblock-reduce_max-128x8192-fp16-br4 | fp16/default | 4x8192x1 | whole_row_tile_fp32_compute | strict_exported_fp64_per_element_bound | valid_strict_oracle | 3.258 | 3.034 | 0.931 |
| rowblock-reduce_max-128x8192-fp16-br8 | fp16/default | 8x8192x1 | whole_row_tile_fp32_compute | strict_exported_fp64_per_element_bound | valid_strict_oracle | 3.565 | 2.487 | 0.698 |
| rowblock-reduce_max-128x8192-bf16-br1 | bf16/default | 1x8192x1 | whole_row_tile_fp32_compute | strict_exported_fp64_per_element_bound | valid_strict_oracle | 2.717 | 2.499 | 0.920 |
| rowblock-reduce_max-128x8192-bf16-br4 | bf16/default | 4x8192x1 | whole_row_tile_fp32_compute | strict_exported_fp64_per_element_bound | valid_strict_oracle | 3.215 | 2.798 | 0.870 |
| rowblock-reduce_max-128x8192-bf16-br8 | bf16/default | 8x8192x1 | whole_row_tile_fp32_compute | strict_exported_fp64_per_element_bound | valid_strict_oracle | 3.843 | 2.653 | 0.690 |

## nextrow12

Matrix status: `passed`. Torch ranking contract: `standard`. Torch mode: `max-autotune`. Original matrix SHA-256: `7fabeea8e3520b1293b22dec68a5b48ac3a1edf71cfb87a9d69b5a9770755a0f`.

| Case | Dtype/math | Tile | Realized algorithm | Numerical contract | Compare status | Native us | Torch us | Torch/native |
| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: |
| coverage-rmsnorm-1x4096-bf16-tile1x4096x1 | bf16/default | 1x4096x1 | whole_row_tile_fp32_compute | strict_exported_fp64_per_element_bound | valid_strict_oracle | 1.853 | 1.255 | 0.677 |
| coverage-rmsnorm-32x4096-fp16-tile1x4096x1 | fp16/default | 1x4096x1 | whole_row_tile_fp32_compute | strict_exported_fp64_per_element_bound | valid_strict_oracle | 2.181 | 1.708 | 0.783 |
| coverage-rmsnorm-1024x1024-fp32-tile1x1024x1 | fp32/default | 1x1024x1 | whole_row_tile_fp32_compute | strict_exported_fp64_per_element_bound | valid_strict_oracle | 8.794 | 9.188 | 1.045 |
| coverage-rmsnorm-128x8192-bf16-tile1x8192x1 | bf16/default | 1x8192x1 | whole_row_tile_fp32_compute | strict_exported_fp64_per_element_bound | valid_strict_oracle | 8.431 | 5.771 | 0.684 |
| coverage-layernorm-1x1024-fp32-tile1x1024x1 | fp32/default | 1x1024x1 | whole_row_tile_fp32_compute | strict_exported_fp64_per_element_bound | valid_strict_oracle | 1.334 | 1.088 | 0.816 |
| coverage-layernorm-128x8192-bf16-tile1x8192x1 | bf16/default | 1x8192x1 | whole_row_tile_fp32_compute | strict_exported_fp64_per_element_bound | valid_strict_oracle | 12.320 | 9.096 | 0.738 |
| coverage-softmax-1x8192-fp32-tile1x8192x1 | fp32/default | 1x8192x1 | whole_row_tile_fp32_compute | strict_exported_fp64_per_element_bound | valid_strict_oracle | 3.238 | 2.502 | 0.773 |
| coverage-softmax-32x4096-bf16-tile1x4096x1 | bf16/default | 1x4096x1 | whole_row_tile_fp32_compute | strict_exported_fp64_per_element_bound | valid_strict_oracle | 2.395 | 3.177 | 1.327 |
| coverage-softmax-1024x512-fp16-tile1x512x1 | fp16/default | 1x512x1 | whole_row_tile_fp32_compute | strict_exported_fp64_per_element_bound | valid_strict_oracle | 6.320 | 4.829 | 0.764 |
| coverage-swiglu-1x8192-bf16-tile1x8192x1 | bf16/default | 1x8192x1 | whole_row_tile_fp32_compute | strict_exported_fp64_per_element_bound | valid_strict_oracle | 3.364 | 0.825 | 0.245 |
| coverage-gelu_residual-128x4096-fp16-tile1x4096x1 | fp16/default | 1x4096x1 | whole_row_tile_fp32_compute | strict_exported_fp64_per_element_bound | valid_strict_oracle | 4.957 | 5.234 | 1.056 |
| coverage-rope-1x8192-bf16-tile1x4096x1 | bf16/default | 1x4096x1 | whole_row_tile_fp32_compute | strict_exported_fp64_per_element_bound | valid_strict_oracle | 1.550 | 0.860 | 0.555 |

## nextrest19

Matrix status: `passed`. Torch ranking contract: `standard`. Torch mode: `max-autotune`. Original matrix SHA-256: `6f4e64fca8ba0a998c05b17f3071fe4c365805d3c9f110a5e0845f586d15ba91`.

| Case | Dtype/math | Tile | Realized algorithm | Numerical contract | Compare status | Native us | Torch us | Torch/native |
| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: |
| coverage-reduce_sum-1024x512-fp32-tile4x512x1 | fp32/default | 4x512x1 | whole_row_tile_fp32_compute | strict_exported_fp64_per_element_bound | valid_strict_oracle | 2.911 | 2.753 | 0.946 |
| coverage-reduce_max-1x16384-bf16-tile1x16384x1 | bf16/default | 1x16384x1 | whole_row_tile_fp32_compute | strict_exported_fp64_per_element_bound | valid_strict_oracle | 1.457 | 1.854 | 1.273 |
| coverage-scan-32x2048-fp32-tile1x2048x1 | fp32/default | 1x2048x1 | inclusive_sum_unordered_tree | strict_exported_fp64_per_element_bound | valid_strict_oracle | 1.831 | 1.580 | 0.863 |
| coverage-argmax-1x16384-bf16-tile1x16384x1 | bf16/default | 1x16384x1 | stable_first_index_argmax | strict_exported_fp64_per_element_bound | valid_strict_oracle | 2.219 | 2.499 | 1.126 |
| coverage-sort-4x8192x8192-fp32-tile1x8192x1 | fp32/default | 1x8192x1 | padded_bitonic_full_sort | strict_exported_fp64_per_element_bound | valid_strict_oracle | 331.302 | 50.140 | 0.151 |
| coverage-topk-1x8192x1-fp32-tile1x8192x1-repeated_extrema | fp32/default | 1x8192x1 | stable_repeated_extrema | strict_exported_fp64_per_element_bound | valid_strict_oracle | 1.676 | 37.203 | 22.199 |
| coverage-topk-1x8192x32-bf16-tile1x8192x1-repeated_extrema | bf16/default | 1x8192x1 | stable_repeated_extrema | strict_exported_fp64_per_element_bound | valid_strict_oracle | 42.437 | 25.616 | 0.604 |
| coverage-topk-1x8192x32-bf16-tile1x8192x1-full_sort_prefix | bf16/default | 1x8192x1 | padded_bitonic_full_sort_prefix | strict_exported_fp64_per_element_bound | valid_strict_oracle | 333.629 | 25.610 | 0.077 |
| coverage-topk-32x2048x64-fp16-tile1x2048x1-repeated_extrema | fp16/default | 1x2048x1 | stable_repeated_extrema | strict_exported_fp64_per_element_bound | valid_strict_oracle | 29.149 | 23.113 | 0.793 |
| coverage-topk-32x2048x64-fp16-tile1x2048x1-full_sort_prefix | fp16/default | 1x2048x1 | padded_bitonic_full_sort_prefix | strict_exported_fp64_per_element_bound | valid_strict_oracle | 39.049 | 23.119 | 0.592 |
| coverage-gemm-1x4096x4096-fp16-tile16x64x32 | fp16/default | 16x64x32 | tile_mma_typed_inputs_fp32_accumulator_reassociation | strict_exported_fp64_per_element_bound | valid_strict_oracle | 92.744 | 136.790 | 1.475 |
| coverage-gemm-4x4096x4096-bf16-tile16x64x32 | bf16/default | 16x64x32 | tile_mma_typed_inputs_fp32_accumulator_reassociation | strict_exported_fp64_per_element_bound | valid_strict_oracle | 101.855 | 84.908 | 0.834 |
| coverage-gemm-32x1024x4096-fp32-tile32x32x32 | fp32/default | 32x32x32 | tile_mma_typed_inputs_fp32_accumulator_reassociation | strict_exported_fp64_per_element_bound | valid_strict_oracle | 136.225 | 53.084 | 0.390 |
| coverage-bmm-4x64x256x128-bf16-tile32x64x32 | bf16/default | 32x64x32 | tile_bmm_singleton_batch_typed_inputs_fp32_accumulator_reassociation | strict_exported_fp64_per_element_bound | valid_strict_oracle | 3.647 | 3.048 | 0.836 |
| coverage-attention-1x8x1x1x4096x64x64-fp32-tile1x32x1 | fp32/default | 1x32x1 | causal_online_softmax_gqa | strict_exported_fp64_per_element_bound | valid_strict_oracle | 107.980 | 508.831 | 4.712 |
| coverage-attention_tensorcore-1x8x1x1x4096x64x64-bf16-tile1x32x1 | bf16/default | 1x32x1 | causal_online_softmax_gqa_narrow_pv | {'name': 'attention_single_narrow_probability_v1', 'acceptance_kind': 'predeclared_numerical_envelope', 'stage': 'unnormalized_probability_before_pv_per_kv_block', 'rounding': 'rne', 'fp32_base_envelope': '5e-5*(1+abs(reference))', 'probability_bound_formula': '((1+gamma_n)/(1-gamma_n))*(u_T*max_abs_V_j+eta_T/2*sum_abs_V_j)', 'gamma_formula': 'n*u32/(1-n*u32), n=2*valid_keys+2', 'primary_bound_formula': 'strict_bound+(1+u_T)*probability_rounding_bound', 'valid_keys': 'K-Q+q+1', 'u32': 5.960464477539063e-08, 'u_T': 0.00390625, 'eta_T': 9.183549615799121e-41, 'requirements': 'One probability narrowing, FP32 scores/accumulation, positive normalization; unspecified extra narrowing or FTZ is not covered'} | valid_common_envelope | 105.510 | 138.668 | 1.314 |
| coverage-attention_tensorcore-1x2x1x512x512x64x64-bf16-tile16x32x1 | bf16/default | 16x32x1 | causal_online_softmax_gqa_narrow_pv | {'name': 'attention_single_narrow_probability_v1', 'acceptance_kind': 'predeclared_numerical_envelope', 'stage': 'unnormalized_probability_before_pv_per_kv_block', 'rounding': 'rne', 'fp32_base_envelope': '5e-5*(1+abs(reference))', 'probability_bound_formula': '((1+gamma_n)/(1-gamma_n))*(u_T*max_abs_V_j+eta_T/2*sum_abs_V_j)', 'gamma_formula': 'n*u32/(1-n*u32), n=2*valid_keys+2', 'primary_bound_formula': 'strict_bound+(1+u_T)*probability_rounding_bound', 'valid_keys': 'K-Q+q+1', 'u32': 5.960464477539063e-08, 'u_T': 0.00390625, 'eta_T': 9.183549615799121e-41, 'requirements': 'One probability narrowing, FP32 scores/accumulation, positive normalization; unspecified extra narrowing or FTZ is not covered'} | valid_common_envelope | 62.267 | 20.807 | 0.334 |
| coverage-embedding-16384x1024x32-bf16-tile1x256x1 | bf16/default | 1x256x1 | uniform_int64_row_gather | strict_exported_fp64_per_element_bound | valid_strict_oracle | 0.969 | 0.968 | 1.000 |
| coverage-embedding-32768x128x1024-fp32-tile1x128x1 | fp32/default | 1x128x1 | uniform_int64_row_gather | strict_exported_fp64_per_element_bound | valid_strict_oracle | 1.912 | 1.434 | 0.750 |

Common-envelope SDPA acceptance, where present, is not proof of identical internal arithmetic; secondary strict validation remains in the JSON. A `failed_route`, `unsupported_route`, or `invalid_evidence` record is not a valid performance comparison. Original raw packet/log hashes identify local artifacts even when path labels are normalized. PATH and GPU UUID are omitted.

Checkpoint bytes use UTF-8 with LF newlines. SHA-256: `19532e259631f5d75b826a9438a7c8ff7a18c4b1331aa38c31ccb967476bdc20`.

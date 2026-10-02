# Collective worker-hint sweep

Status: completed_validated. Worker counts below are requested compiler hints, not measured thread counts.
Values are graph-v2 medians in us per complete operation. Only baseline contains Torch; hinted cohorts and the optional native0 drift recheck are native-only.
Changes below 5% are descriptive noise-scale differences; larger changes still need replicated controls before causal claims.

| Case | baseline | hint4 | hint8 | recheck |
|---|---:|---:|---:|---:|
| aligned16-scan-128x8192-fp16 | 12.344 / 5.465 Torch | 12.347 (+0.0%) | 16.206 (+31.3%) | 12.322 (-0.2%) |
| aligned16-scan-128x8192-bf16 | 12.357 / 5.480 Torch | 12.359 (+0.0%) | 16.137 (+30.6%) | 12.323 (-0.3%) |
| aligned16-control-scan-128x8192-fp32 | 13.844 / 10.480 Torch | 13.937 (+0.7%) | 18.408 (+33.0%) | 13.845 (+0.0%) |
| rowblock-reduce_sum-128x65-fp32-br1 | 1.018 / 0.974 Torch | 1.026 (+0.9%) | 1.195 (+17.4%) | 1.025 (+0.7%) |
| rowblock-reduce_sum-128x65-fp32-br4 | 0.942 / 1.061 Torch | 0.945 (+0.3%) | 0.978 (+3.8%) | 0.942 (-0.0%) |
| rowblock-reduce_sum-17x1024-fp16-br1 | 0.924 / 0.984 Torch | 0.927 (+0.3%) | 0.937 (+1.4%) | 0.924 (-0.1%) |
| rowblock-reduce_sum-17x1024-fp16-br4 | 2.199 / 0.908 Torch | 2.207 (+0.4%) | 1.477 (-32.8%) | 2.202 (+0.1%) |
| aligned16-rmsnorm-1x8192-fp16 | 2.890 / 1.414 Torch | 2.905 (+0.5%) | 2.272 (-21.4%) | 2.889 (-0.0%) |
| aligned16-layernorm-1x8192-bf16 | 3.274 / 1.916 Torch | 3.288 (+0.5%) | 2.618 (-20.0%) | 3.275 (+0.0%) |
| coverage-rmsnorm-128x8192-bf16-tile1x8192x1 | 7.732 / 4.882 Torch | 7.823 (+1.2%) | 7.803 (+0.9%) | 7.797 (+0.8%) |
| coverage-layernorm-128x8192-bf16-tile1x8192x1 | 11.216 / 8.091 Torch | 11.235 (+0.2%) | 9.882 (-11.9%) | 11.212 (-0.0%) |
| coverage-softmax-1024x512-fp16-tile1x512x1 | 5.721 / 3.282 Torch | 5.805 (+1.5%) | 8.949 (+56.4%) | 5.777 (+1.0%) |
| rowblock-scan-17x1024-fp16-br1 | 1.014 / 0.926 Torch | 1.017 (+0.4%) | 1.045 (+3.1%) | 1.018 (+0.4%) |
| rowblock-scan-17x1024-fp16-br4 | 2.695 / 1.003 Torch | 2.708 (+0.5%) | 2.970 (+10.2%) | 2.710 (+0.6%) |
| rowblock-reduce_max-17x1024-fp16-br1 | 0.932 / 0.913 Torch | 0.930 (-0.2%) | 0.933 (+0.1%) | 0.928 (-0.4%) |
| rowblock-reduce_max-17x1024-fp16-br4 | 2.165 / 0.914 Torch | 2.172 (+0.3%) | 1.559 (-28.0%) | 2.165 (+0.0%) |

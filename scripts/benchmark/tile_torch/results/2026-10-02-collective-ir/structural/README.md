# Strict structural collective experiment

Status: completed_validated.
All cohorts are native-only. Entries are graph-v2 median microseconds per complete operation, with every raw sample preserved in checkpoint.json.
Source memory/grid evidence is lexical and reported/inferred as labeled, not a driver trace or a general equivalence proof. No robust gain is claimed from one sequential cohort.

| Case | Control | Chunk1024 | Chunk2048 | Independent1 |
|---|---:|---:|---:|---:|
| aligned16-scan-128x8192-fp16 | 12.346 | 13.064 (+5.8%) | 13.010 (+5.4%) | - |
| aligned16-scan-128x8192-bf16 | 12.360 | 13.076 (+5.8%) | 13.141 (+6.3%) | - |
| aligned16-control-scan-128x8192-fp32 | 13.889 | 14.679 (+5.7%) | 14.496 (+4.4%) | - |
| rowblock-reduce_sum-128x65-fp32-br4 | 0.951 | - | - | 1.326 (+39.4%) |
| rowblock-reduce_sum-17x1024-fp16-br4 | 2.208 | - | - | 1.692 (-23.4%) |
| rowblock-scan-17x1024-fp16-br4 | 2.711 | - | - | 2.996 (+10.5%) |
| rowblock-reduce_max-17x1024-fp16-br4 | 2.174 | - | - | 2.614 (+20.3%) |

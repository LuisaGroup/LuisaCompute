# CUDA TIRx subgroup same-cohort result

All six cases and all seven samples are retained. Ratios below one favor the subgroup candidate; drift is native recheck / initial. Fast math is explicit on both routes, with route-specific arithmetic implementations.

| Case | Native initial µs | Subgroup µs | Native recheck µs | Fresh Torch µs | Subgroup / initial | Subgroup / recheck | Subgroup / Torch | Native drift |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| regroup-rmsnorm-128x1024-bf16 | 1.461471 | 3.914547 | 1.466852 | 1.380707 | 2.678498 | 2.668673 | 2.835175 | 1.003682 |
| regroup-rmsnorm-256x1536-fp16 | 2.750970 | 8.244822 | 2.802553 | 2.595424 | 2.997060 | 2.941897 | 3.176677 | 1.018751 |
| regroup-rmsnorm-7x2051-bf16 | 1.172701 | 3.523191 | 1.176151 | 2.099404 | 3.004339 | 2.995527 | 1.678186 | 1.002942 |
| regroup-softmax-1024x512-fp16 | 5.147484 | 6.894814 | 5.181173 | 3.232782 | 1.339453 | 1.330744 | 2.132780 | 1.006545 |
| regroup-softmax-32x512-fp16 | 1.120877 | 1.584007 | 1.125052 | 1.180391 | 1.413185 | 1.407942 | 1.341934 | 1.003724 |
| regroup-softmax-9x769-fp16 | 1.145244 | 1.947852 | 1.149150 | 1.366895 | 1.700819 | 1.695038 | 1.425020 | 1.003411 |

The checkpoint preserves 18 native + 6 fresh Torch validations, 168 primary graph samples, actual graph/resource observations, fixture/source receipts and complete correctness evidence. Saved logical outputs are independently checked against the original full oracle; padded guards and readonly inputs remain the original runtime checks. Driver local bytes are not measured spill traffic. Native output is preallocated; Torch retains its functional allocation contract. No previous cohort, fitting or speed-based admission is used.

## Reproduce this evidence

Run `python reproduce.py` from this directory. It uses only the Python standard library and verifies bundled hashes, all 168 raw primary samples, graph replay normalization, medians/ratios/drift, all 1,800 recorded native graph nodes, source/ABI identity, fixture receipts and loaded-function resources. It does not run GPU work or recheck omitted tensor contents. Original full-output FP64-bound, guard and readonly checks are retained as validation records; inputs, outputs, references and bounds are represented by their measured SHA receipts rather than large payloads.

All six fixed T128/P2/lane-elements8/cache-false candidates regressed against both same-round native controls and fresh Torch. This is a correctness-tested, default-off backend capability experiment; it does not justify automatic deployment or a calibrated CUDA selection model. Registers/shared/local/max-threads are Driver attributes, not occupancy or dynamic memory traffic. In particular the tail BF16 RMS candidate reports 304 local bytes; this packet does not identify a performance cause.

The TIRx route reuses the existing Tile IR pure-DAG reduction mapper; it introduces no DSL/execution-nest primitive. CUDA-source target, complete warp participation, uniform block fences, canonical unordered FP32 reduction identities and the existing noalias contract are required. Explicit fast math is fixed true here: elementwise CUDA fast policy differs from native Tile's fast implementation, while local and warp reduction merges use explicit no-FTZ PTX. This packet does not claim identical arithmetic instructions between implementations.

V1 measured sources and realization markers are preserved as recorded, with implicit unroll=1. A subsequent source revision adds an explicit `:unroll` metadata field; this package does not retrofit that marker into old evidence. The frozen V1 queue's strict marker parser belongs to its recorded snapshot. `source-packet/` contains the exact private adapter/helper/CMake inputs as a reference; rebuilding requires the compatible complete repository and dependencies described by `provenance/source-snapshot.json`. Running this evidence replay does not rebuild them.

`index.json` distinguishes raw/private SHA and bytes from public copies. Source and CSV files are byte-exact. JSON records retain their complete contents with machine-specific path prefixes replaced by `${REPO}`, `${USER_TEMP}`, or `${USER}` and normalized formatting. Raw-result hashes identify the original files; public-file hashes verify the bundled files. No previous smoke or math/row cohort supplies a denominator, and no negative result is removed.

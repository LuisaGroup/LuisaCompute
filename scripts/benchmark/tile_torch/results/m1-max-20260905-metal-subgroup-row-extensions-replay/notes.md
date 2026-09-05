# Balanced LayerNorm and cross-entropy lowering replay

Date: 2026-09-05 Asia/Shanghai (`2026-09-04T23:49:13.115180Z` in the raw
report). Machine: Apple M1 Max, macOS 26.6.2 arm64. PyTorch 2.14.0. Source
revision: `7958d84dd28991ed617b54bf53b5b47039573aa0`.

## Causal question and result

This replay asks whether the new proved SIMD-group execution map, rather than
an incidental framework or build change, causes the improvement. Reference
and candidate use the same executable and dynamic libraries; the only changed
compile policy is `metal_subgroup_reductions=false/true`.

All 64 native variant measurements (8 cases × 4 rounds × 2 policies) pass
their complete FP64 oracles. There are no discarded rows. Every measurement
freshly captures and JIT-compiles the Tile kernel; case, variant and
native/PyTorch order are counterbalanced. Fingerprinted artifacts remain
unchanged from the beginning to the end of the replay.

| Case | Reference µs | Subgroup µs | Paired reference/subgroup median [range] | Candidate-run Torch µs |
|---|---:|---:|---:|---:|
| LayerNorm 1×127 | 131.413 | 4.577 | 28.675× [27.944, 29.063] | 8.239 |
| LayerNorm 17×257 | 337.366 | 5.693 | 58.942× [57.105, 64.220] | 8.633 |
| LayerNorm 128×1024 | 280.352 | 7.517 | 37.333× [36.900, 37.661] | 13.930 |
| LayerNorm 64×4096 | 928.945 | 12.306 | 75.536× [74.338, 82.088] | 24.488 |
| cross-entropy 1×127 | 62.412 | 4.446 | 14.042× [13.737, 14.854] | 108.512 |
| cross-entropy 17×257 | 191.603 | 3.228 | 59.357× [53.681, 61.339] | 107.258 |
| cross-entropy 128×1024 | 74.350 | 4.370 | 17.015× [16.097, 17.463] | 108.205 |
| cross-entropy 64×4096 | 355.493 | 5.774 | 60.879× [59.618, 63.291] | 111.832 |

Displayed times are medians of the four per-round p50 synchronized host-wall
throughput values. Speedups are medians of paired same-round ratios; bracketed
ranges are observed minima and maxima, not confidence intervals. Native
outputs are preallocated in both variants. Therefore the 14.04×--75.54×
native A/B is unaffected by PyTorch's returned-output allocation.

The Torch column is only an external same-round reference. Functional
LayerNorm and cross-entropy allocate their returned outputs inside timing.
Cross-entropy additionally carries the general eager operator's dispatch and
semantic machinery, explaining why its approximately 107--112 µs values must
not be interpreted as a pure MPS kernel baseline.

## Controls

- `reference`: ordinary automatic Metal root mapping, one logical row per
  scalar worker, serial reductions and full logical private Tile storage;
- `candidate`: the same Tile program with proved immutable input views,
  bounded one/two/four/eight-group search and owner-checked resources;
- identical executable and bridge hashes for both variants;
- no parameter search or reuse of the input reports' timing samples;
- fresh capture, JIT, allocation/upload, warmup, timing, download and full
  validation for every variant in every round; and
- MPS fast-math/prefer-Metal/fallback, validation, source-dump and loader
  environment variables removed and recorded before measurement.

Maximum native absolute errors over all rounds are `7.895e-6` (reference) and
`2.913e-7` (candidate) for LayerNorm, and `2.216e-6` (reference) and
`1.109e-6` (candidate) for cross-entropy. This also records the expected
numerical consequence of changing FP32 reduction order rather than hiding it.

## Exact command

```bash
uv run --no-project --python 3.13 --with torch --with numpy python \
  scripts/benchmark/tile_torch/repeat.py \
  --reference /tmp/luisa-row-extensions-reference-20260905-v1/results.json \
  --candidate scripts/benchmark/tile_torch/results/m1-max-20260905-metal-subgroup-row-extensions/results.json \
  --native cmake-build-tirx/bin/benchmark_tile_tirx \
  --output scripts/benchmark/tile_torch/results/m1-max-20260905-metal-subgroup-row-extensions-replay \
  --operations layernorm,cross_entropy --rounds 4 \
  --samples 7 --sample-ms 60 --warmup-ms 80 --threads 8 \
  --capture-sources
```

No build, profiler or other GPU workload ran concurrently.

## Artifact identity

- replay `results.json` SHA-256:
  `e05743e57aef4b254a4e53bc8d2c668fd119a6bb8bf5cab4808dd6c081ddb2c2`
- executable SHA-256 for both variants:
  `ab423b2d4a4e7a8683069ea7d5491162ee8718af755a2372737930a0181428f9`
- TIRx bridge SHA-256 for both variants:
  `4f413da0c70d7b97b465c078b4bb9d42c37c2fbb2a44636a98c3df53cd6ef49c`
- frozen reference configuration report SHA-256:
  `6ae55339f37e13300f258649a168ce118e807cc2fc8859b24ae2e4195386c6ab`
- frozen candidate configuration report SHA-256:
  `8d33bc088ba69a62cf72816ad4ec6a22b91ff6577ed65749946a1037f34d939b`

The raw JSON embeds both frozen plans, all artifact hashes, all 64
implementation orders, every sample/error and 16 unique generated Metal
sources. `artifacts_unchanged=true` confirms no measured binary changed during
the replay.

## Companion source validation

After measurement, the same implementation passed the complete local
`^test_tile_` CTest cohort (**32/32**), the focused guarded-view CPU proof
(**1,572 assertions**) and Python benchmark-contract discovery (**67/67**).
The new ownership diagram is well-formed XML, the handwritten source/docs pass
the whitespace check, and all changed Tile pages pass a Sphinx 9.1 `-W` build
when only the repository's known missing-Doxygen/tutorial warning categories
are suppressed. Generated `.metal` files retain the exact trailing bytes
covered by their SHA-256 filenames. These are post-measurement source/report
checks; they do not alter or contribute samples to the balanced replay.

## Interpretation boundary

This is strong causal evidence that the previous LayerNorm and cross-entropy
gap was an execution/resource realization defect rather than a launch-width
coefficient issue. It does not prove the four-candidate solver globally
optimal, and it does not generalize to other dtypes, layouts, devices,
backward kernels, fused training losses or whole-model performance.

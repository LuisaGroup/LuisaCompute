# Metal MPP cost-model v2 calibration-cohort search

Date: 2026-09-05 Asia/Shanghai (`2026-09-04T19:58:02.528393Z` in the raw report).
Machine: Apple M1 Max, macOS 26.6.2 arm64. Source revision:
`b8c3c54f81f2a4ad947e295f1f75e57605bf8833` plus recorded uncommitted changes.

## Question and answer

The experiment asks whether an MPP-specific analytic score can replace the
old mistake of summing tensor work that separate subgroups execute
concurrently. Within the measured 8-shape × 45-requested-candidate cohort,
cost-model v2 reduces mean top-choice regret from 74.18% to 8.82%, median
regret from 43.05% to 2.59%, and maximum regret from 239.58% to 34.37%.
Exact measured-winner picks increase from 1/8 to 4/8.

This is useful structural evidence, but it is **in-cohort calibration**, not a
held-out generalization result and not proof of a hardware optimum. Search
measurements use only three 10 ms samples and can be noisy. The independently
frozen 14-round replay is in the sibling
[`m1-max-20260905-mpp-cost-v2-replay`](../m1-max-20260905-mpp-cost-v2-replay/notes.md)
directory.

## What changed from v1

Both models enumerate the same legal Tile blocks, threadgroup widths and exact
subgroup factorizations. v2 changes the ranking abstraction:

```text
program_score = weighted_issue_work * state_pressure / participating_subgroups
              + independent_elements * element_weight / participating_subgroups
              + group_setup
concurrent_waves = max(1, programs * participating_subgroups / 512)
kernel_score = program_score * concurrent_waves
```

The 512-subgroup capacity is a replaceable M1-class prior, not queried
occupancy. Other MPP-specific features include memory fragment requests,
asymmetric A/B logical footprints, multiply versus accumulate mode, output
traffic, accumulator initialization, Tile/local aspect terms and live fragment
state. Coefficients are relative work, not nanoseconds.

Legality is separate from score. Exact non-overlapping coverage, thread and
shared-memory limits, fragment/code-size bounds, semantic proofs, and the MPP
descriptor requirement that each local matrix have M or N divisible by 16 are
hard checks. Invalid candidates are retained in `results.json` and cannot win.

## Model choice versus measured finite-set winner

Regret is `measured_time(model_pick) / measured_time(best_valid_candidate) - 1`.
The measurement is the search trial median, so this table diagnoses the model
only inside this finite, noisy cohort.

| Shape | v1 model / measured | v1 regret | v2 model / measured | v2 regret |
|---|---|---:|---|---:|
| 32×32×32 | 32×32×32 @ 64t / @ 256t | 42.05% | 32×32×32 @ 256t / @ 128t | 5.18% |
| 128×128×128 | 32×32×128 @ 64t / @ 256t | 44.05% | 32×32×128 @ 256t / @ 256t | 0.00% |
| 512×512×512 | 64×64×512 @ 128t / 128×32×128 @ 256t | 6.28% | 64×64×512 @ 256t / 32×64×32 @ 128t | 13.81% |
| 1024×1024×1024 | 128×32×1024 @ 128t / @ 128t | 0.00% | 128×32×1024 @ 128t / @ 128t | 0.00% |
| 256×1024×128 | 64×64×128 @ 128t / 32×32×32 @ 64t | 3.45% | 64×64×128 @ 256t / @ 256t | 0.00% |
| 1024×128×256 | 32×64×32 @ 64t / 32×32×128 @ 64t | 84.10% | 32×32×256 @ 128t / 32×32×32 @ 128t | 17.17% |
| 127×193×61 | 32×32×32 @ 64t / @ 256t | 173.93% | 32×32×32 @ 256t / @ 256t | 0.00% |
| 513×257×129 | 32×64×32 @ 64t / 32×32×32 @ 256t | 239.58% | 32×32×32 @ 128t / @ 256t | 34.37% |

The remaining misses are not erased: 512³, 1024×128×256 and
513×257×129 need better cache/layout, edge and launch features or a larger,
more diverse measured shortlist.

## Reproduction

```bash
uv run --no-project --python 3.13 --with numpy --with torch python \
  scripts/benchmark/tile_torch/run.py \
  --native /tmp/luisa-tvm-mpp.VaKmzx/luisa-build/bin/benchmark_tile_tirx \
  --output scripts/benchmark/tile_torch/results/m1-max-20260905-mpp-cost-v2-search \
  --backends metal --operations gemm --execution-scope group \
  --cooperative-matrix --matrix-realization mpp-views --pipeline-window 1 \
  --tune-gemm-blocks '32,32,32;32,64,32;64,64,32;32,32,128;32,128,128;64,64,128;128,32,128;32,32,256;64,64,256;32,128,512;64,64,512;128,32,512;32,128,1024;64,64,1024;128,32,1024' \
  --tune-pipeline-windows 1 --tune-group-threads 64,128,256 \
  --max-tuning-candidates 45 --samples 3 --sample-ms 10 --warmup-ms 50 --threads 8
```

The v1 and v2 executable SHA-256 values are respectively
`ff591dd4fc00bfb4cc96c57c5cf5625c4dbe22248154f23e8fe9abf2762553df`
and `729584994dc8a5b3335ec48a63ae004f001d74f45927990a99ec2170d82076b0`.
The raw reports also fingerprint adjacent Tile libraries, preserve every
candidate result and record the fresh post-selection recapture/JIT timing.

## Interpretation boundary

- The model chose candidates; Torch did not participate in that selection.
- Final rows in `results.md` are fresh post-selection timings, not reused
  search minima.
- Search timings are synchronized host-wall throughput, not GPU-event times.
- The cohort is FP32 GEMM on one M1 Max. It says nothing yet about low precision,
  fused epilogues, reductions, softmax, attention, other devices or CPU.
- A future calibration must train on one set, report held-out regret/top-K
  coverage on another, and keep the measured Staged/JIT winner authoritative.

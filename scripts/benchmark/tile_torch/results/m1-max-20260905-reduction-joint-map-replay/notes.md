# Reduction joint mapping: independent four-round replay

Date: 2026-09-05, Asia/Shanghai. Apple M1 Max, macOS 26.6.2 arm64,
PyTorch 2.14.0, FP32. TIRx Metal only.

All 80 full native outputs pass. The same binary runs the old automatic
policy and the frozen JIT-selected settings in four balanced rounds; there
is no search during replay. Binary/adjacent-library hashes stay unchanged.
Both sum and softmax use preallocated outputs on the native and Torch sides.

## Stable gains, not universal wins

| Operation / shape | Automatic µs | Selected µs | Paired speedup median [min, max] |
|---|---:|---:|---:|
| sum / 1024×4096 | 25.760 | 23.235 | 1.107× [1.100, 1.188] |
| softmax / 1024×4096 | 67.218 | 63.376 | 1.062× [1.053, 1.079] |
| softmax / 1024×257 | 8.386 | 7.953 | 1.051× [1.041, 1.072] |

For the two 4096-wide cases, the selected plan uses four cooperating SIMD
groups and unroll factor four, versus eight SIMD groups and factor one. The
1024×257 softmax uses four packed independent rows instead of eight, without
unrolling. These are mapping/codegen choices for the same source kernel.
The A/B does not separately attribute the wide-row gain to width versus
unrolling.

The other seven cases are not claimed as reliable gains. Their paired ranges
overlap or nearly touch one, and 17×257 softmax has a 0.985× paired median
(a slight regression). The selected 17×257 sum is the unchanged automatic
configuration; its apparent 1.057× median is a useful noise control, not an
optimization. The two 1×4096 cases and 64×4096 softmax are essentially flat.
The [complete table](results.md) includes every case without hiding these
outcomes. Accordingly, unrolling stays opt-in and the default analytic
coefficients are not changed on this evidence alone.

Timing is synchronized device-resident host wall time including dispatch,
not GPU event time. Table times are medians of per-round medians; speedup is
the median of paired ratios. Min/max ranges describe these four observations,
not confidence intervals. New/Torch paired time ratios for the three rows
above are 0.834, 0.494 and 0.238 respectively; this is not MPS/MMA performance.

```bash
uv run --no-project --python 3.13 --with torch --with numpy \
  python scripts/benchmark/tile_torch/repeat.py \
  --reference scripts/benchmark/tile_torch/results/m1-max-20260905-reduction-auto-mapping/results.json \
  --candidate scripts/benchmark/tile_torch/results/m1-max-20260905-reduction-joint-map-search/results.json \
  --native cmake-build-tirx/bin/benchmark_tile_tirx \
  --operations sum,softmax --rounds 4 --samples 7 --sample-ms 40 \
  --warmup-ms 100 --capture-sources --output NEW_EMPTY_DIRECTORY
```

Raw settings, timing order, per-round samples, correctness errors and source
hashes are in [results.json](results.json); generated Metal is in `sources/`.
Independent QA recomputed all per-case medians and paired ratios and verified
one record per `(case, round, variant)`, complete validity and unchanged
run-time fingerprints. Assessment: share with these caveats, not as a claim
that the default cost model or every reduction is now optimal.

# Balanced RMSNorm reference/subgroup replay

Date: 2026-09-05 Asia/Shanghai. Machine: Apple M1 Max, macOS 26.6.2 arm64.
PyTorch 2.14.0. The replay uses one current executable and bridge for both
variants; only the recorded `metal_subgroup_reductions` policy changes.

## Causal question and result

This replay asks whether replacing the old one-thread serial row realization
with the proved SIMD-group realization causes the RMSNorm speedup, while
holding the Tile program, shape, runtime, binary and all other compile options
fixed. The raw report timestamp is `2026-09-04T23:15:21.861523Z`.

All 32 variant measurements (4 shapes × 4 rounds × 2 variants) passed their
complete FP64 oracles. There were no discarded or failed rows. Each row was
freshly captured and JIT-compiled; variant order and case position rotate
across rounds. The candidate is 21.19×–49.87× faster than the reference
lowering by median same-round ratio and is faster than eager Torch in all four
candidate runs.

| Rows×width | Reference µs | Subgroup µs | Paired reference/subgroup median [range] | Candidate-run Torch µs |
|---|---:|---:|---:|---:|
| 1×127 | 103.180 | 3.792 | 27.216× [24.924, 28.020] | 7.164 |
| 17×257 | 268.202 | 5.366 | 49.871× [49.180, 54.574] | 6.141 |
| 128×1024 | 144.082 | 6.805 | 21.192× [20.989, 21.207] | 8.802 |
| 64×4096 | 524.444 | 11.160 | 47.096× [46.344, 50.864] | 12.474 |

Displayed times are medians of the four per-round p50 throughput values.
Speedups are medians of paired same-round ratios; the brackets are the observed
minimum and maximum, not confidence intervals. Timing is synchronized,
device-resident host wall time including dispatch, not a pure GPU event.
Native uses a preallocated output; PyTorch's functional RMSNorm returns a new
output because its public operator has no `out=` overload, so that allocation
is included in the Torch column. The causal reference/candidate speedup compares
the two native variants and is unaffected by this PyTorch API difference.

## Controls

- `reference`: `metal_subgroup_reductions=false`; automatic root binding falls
  back to the old worker mapping and serial row recurrence.
- `candidate`: `metal_subgroup_reductions=true`; the bounded solver chooses the
  reported one/two/four/eight-SIMD-group plan.
- Both variants use the exact same current executable and dynamic Tile bridge.
- The old and new input reports provide frozen configuration only. Their warm
  timings are not reused by this replay.
- Every trial performs capture, JIT, allocation/upload, warmup, timing,
  download and complete output validation. Search minima are not involved.
- Environment variables that could alter MPS fast math, fallback, validation,
  source dumping or loader behavior were removed and recorded.

## Exact command

```bash
uv run --no-project --python 3.13 --with torch --with numpy python \
  scripts/benchmark/tile_torch/repeat.py \
  --reference /tmp/luisa-rmsnorm-before-metal-20260905/results.json \
  --candidate scripts/benchmark/tile_torch/results/m1-max-20260905-metal-subgroup-reductions/results.json \
  --native cmake-build-tirx/bin/benchmark_tile_tirx \
  --output /tmp/luisa-rmsnorm-subgroup-balanced-replay-20260905-final-reported \
  --operations rmsnorm --rounds 4 \
  --samples 9 --sample-ms 60 --warmup-ms 80 --threads 8 \
  --capture-sources
```

No build, profiler or other GPU workload ran concurrently.

## Artifact identity

- replay `results.json` SHA-256:
  `4c1af54b209f7706d09108ff752e2166f80ce00f7ce638d13866a49a8e0e5638`
- executable SHA-256 for both variants:
  `4c0bd9b660085d8d8cd5b3682c8eda4732e2c55853ec77c943986ebd62ac01dd`
- TIRx bridge SHA-256 for both variants:
  `c0374c27b23e6741751f53e08eec5817b4f535b365f8270cfd200a6271670a21`
- frozen reference configuration report SHA-256:
  `b51faffdf7dea5fb93593a6cbc1d737c3221cef78175d6c0f3b2781bc4a3440f`
- frozen candidate configuration report SHA-256:
  `6d075e6f0d5a00c8c53f9026884f936b820f25e801cc369838a5daf8188502d1`

The raw JSON retains all artifact hashes before/after timing, implementation
orders, commands, generated-source hashes, individual samples, correctness
errors and frozen plans.

## Interpretation boundary

This is strong evidence that the measured RMSNorm gap was structural rather
than a small coefficient or launch-width issue. It does not establish that the
current solver is optimal: the replay compares two frozen policies, not every
legal schedule. It also does not extend the conclusion to other hardware,
dtypes, normalization definitions, fusion contexts or whole models.

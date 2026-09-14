# Dependence-aware group synchronization on M1 Max

The bridge now combines compiler-owned group fences across independent memory
effects. At the same 64x64x32 tile, 256 workers, pipeline window 1, and copy
batch 4, four counterbalanced rounds show a **1.019x median speedup over
5a72c0514** for 1024-cubed GEMM, with one round regressing. That shape remains
**1.111x the PyTorch time**. Barrier reduction helps, but does not close the
remaining large-square gap or complete the CPU/Metal performance goal.

## Transformation and safety boundary

The two input copies write distinct fresh shared buffers. They can publish
together before the matrix consumer:

```text
before: copy A -> As; fence; copy B -> Bs; fence; MMA; fence
after:  copy A -> As;        copy B -> Bs; fence; MMA; fence
```

This changes neither operation order nor the execution distribution. The
analysis accumulates effects since the last retained fence; an adjacent-only
comparison would miss `write A; unrelated B; read A`. Any RAW, WAR, or WAW
dependence keeps the cut. All external global buffers share one conservative
alias class, including different parameters with const input types. Only
fresh compiler-owned shared allocations are known distinct.

Unknown calls, storage, explicit synchronization, and control exits remain
boundaries. Compiler fences are identified by emission-local IR identity,
not by matching an intrinsic name. An explicit identical-looking native
barrier is preserved. The final compiler fence of each sequential region
also remains for loop-backedge and enclosing-region consumers. Fences never
move across loops or branches; the full shared-plus-device fence flags are
unchanged. A future storage-reuse pass must invalidate/recheck this plan.

`PlannerOptions::coalesce_group_barriers = false` independently retains the
reference fences; disabling the planner also disables coalescing. No DSL
entity, GEMM-name shortcut, or new external dependency was added.

For aligned 64x64x32 output, subgroup/fragment distribution remains
`(gm, gn, rm, rn) = (2, 4, 4, 2)`, shared storage remains **16 KiB**, and the
live-fragment scalar proxy remains 28 per lane. The number of static compiler
fence sites changes **4 to 3**, including **3 to 2** inside the K loop.
For L K iterations the corresponding dynamic count is `3L + 1` to `2L + 1`.
This is a count derived from the emitted program, not a hardware counter.

The 32-cubed and ragged-output cases retain the guarded shared output path,
32 KiB of shared storage, and change six static sites to five (`3L + 3` to
`2L + 3`). The model reports static before/after counts but does not yet price
barrier latency in its bootstrap relative-work score.

## Controlled implementation comparison

Reference: frozen commit `5a72c051410f6aab739128694d83c43ff8283e69` and adjacent
libraries in `/tmp/luisa-tile-before-barrier-plan-qCk9hq`. Candidate: the
synchronization work based on that commit. The raw report records both
executable hashes and their adjacent Tile/bridge libraries. The respective
bridge SHA-256 hashes are
`cfbdc3f28b0484bd13789eaa7dd0af7e6d36a569a0a7e863611a0a421ca1b949`
and `fb03b04343d3ed64fe6baa72d42ae181c7d5aef787144dc7198cc9c2173ef5c5`.

FP32 inputs and preallocated device-resident output are unchanged. All runs
check the complete output against the FP64 oracle. Four rounds rotate shape
order and counterbalance both implementation and framework order. Every run
freshly captures/JIT-compiles its frozen configuration and records cold/setup
costs separately. Each warm result uses nine 40 ms timing batches after
200 ms warmup. No builds, tests, or profilers ran during timing.

These are synchronized amortized **host-wall** measurements including
dispatch, not GPU-event kernel durations. Values are medians of per-round
medians; paired ranges are observed min-max ranges, not confidence intervals.

| M x N x K | Reference us | Coalesced us | Paired speedup median [range] | Candidate-run PyTorch us |
|---|---:|---:|---:|---:|
| 32 x 32 x 32 | 6.255 | 6.119 | 1.034 [1.004, 1.091] | 26.554 |
| 128 x 128 x 128 | 14.010 | 13.055 | 1.071 [1.063, 1.081] | 27.119 |
| 512 x 512 x 512 | 57.135 | 55.394 | 1.031 [1.026, 1.035] | 48.285 |
| 1024 x 1024 x 1024 | 327.454 | 321.218 | 1.019 [0.986, 1.020] | 289.100 |
| 256 x 1024 x 128 | 19.894 | 19.068 | 1.039 [1.021, 1.103] | 29.476 |
| 1024 x 128 x 256 | 25.524 | 23.507 | 1.089 [1.083, 1.102] | 29.878 |
| 127 x 193 x 61 | 12.323 | 11.919 | 1.035 [1.008, 1.075] | 26.980 |
| 513 x 257 x 129 | 28.111 | 27.531 | 1.026 [0.976, 1.043] | 34.559 |

All 64 measurements validated, with four pairs per shape and zero failures.
No slow rounds were excluded. In particular, the first 1024-cubed candidate
round measured 331.972 us versus its 327.452 us reference; one ragged-output
round also regressed. The 128-cubed and tall-matrix improvements were more
consistent, approximately 7% and 9%. The 512-cubed change was approximately
3%, and it still takes 1.147x the PyTorch time.

Reducing dynamic fences by roughly one third does not produce a matching
time reduction. These measurements isolate synchronization from the earlier
direct-output resource change: tile dimensions, thread count, copy batch,
fragment mapping, shared allocation, and resident output policy were held
fixed. They do not identify the remaining issue/transaction/occupancy costs.
The next cost-model work needs to distinguish equal-work layouts and use
measured ranking error, rather than treating a more powerful solver as a
substitute for missing cost information.

## Reproduction and validation

The [eight-shape pilot](m1-max-20260903-barrier-plan-64x64x32/results.md) fixes
the schedule. The controlled evidence is the [four-round comparison](m1-max-20260903-barrier-plan-repeat/results.md)
and its [raw measurements](m1-max-20260903-barrier-plan-repeat/results.json).

With a fully built candidate and the frozen reference bundle:

```sh
uv run --no-project --python 3.13 --with torch --with numpy python \
  scripts/benchmark/tile_torch/repeat.py \
  --reference scripts/benchmark/tile_torch/results/m1-max-20260903-direct-store-64x64x32/results.json \
  --candidate scripts/benchmark/tile_torch/results/m1-max-20260903-barrier-plan-64x64x32/results.json \
  --native /tmp/luisa-tile-before-barrier-plan-qCk9hq/benchmark_tile_tirx \
  --candidate-native cmake-build-tirx/bin/benchmark_tile_tirx \
  --output /tmp/tile-barrier-plan-fresh-repeat \
  --operations gemm --rounds 4 --samples 9 --sample-ms 40 --warmup-ms 200
```

The output directory must not already exist. Complete the full configured
build before testing or timing, and do not run either concurrently with
performance measurements.

The latest full unit run passed **137/137** in 60.85 seconds, including CPU
and actual Metal. New synchronization regressions cover nonadjacent
dependencies, aliased global parameters, output aliasing across a pipeline
backedge, subgroup-multiple and nonmultiple worker counts, and explicit native
barrier identity. Every numerical case checks the complete output.
The full optional-bridge-off build and its **six** Tile tests also passed,
as did **21** benchmark-driver tests, four per-file syntax checks, and
`git diff --check`.

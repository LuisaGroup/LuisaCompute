# Layout sensitivity missing from the bootstrap model

An experiment reversed only the deterministic subgroup-factor enumeration
order in the exact matrix solver. For 64x64x32 tiles and 256 workers, this
selects a different layout at an **identical model score**. Four counterbalanced
rounds found no overall benefit: 1024-cubed became approximately **3.7% slower**
and 512-cubed approximately **3% slower**. The experiment was reverted; the
production tie order and cost coefficients are unchanged.

A second, independent cooperative-copy layout experiment found approximately
1% large-square gains but small-shape regressions. It was also reverted.
Both studies retain every shape and round, including unfavorable results.

## Isolated change

| Realization | Subgroup grid (gm, gn) | Local fragment rectangle (rm, rn) |
|---|---|---|
| Reference | (2, 4) | (4, 2) |
| Experimental | (4, 2) | (2, 4) |

Both maps cover every output atom exactly once. In all eight pilot rows, the
entire reported plan excluding the matrix-layout tuple is identical: thread
count, shared storage, live-fragment scalar proxy, matrix issues, shared
fragment transfers, direct-output transfers, synchronization-site counts,
copy-batch policy, search counters, and normalized score. Accumulator residency
and direct-output eligibility also remain identical. Kernel source, tile shape,
pipeline window 1, copy batch 4, FP32 policy, and input values are unchanged.

The solver's Pareto frontier keeps the first score/resource-equivalent state.
Sorting the existing subgroup divisors descending rather than ascending
therefore changes only which tied map survives in these cases. This is an
experimental perturbation of ranking, not a new layout primitive or a claimed
target-specific cost formula.

The reference is `1ef0ffa7ddc4480b21446a3c1608d294f01c9d7a`, frozen with its
adjacent libraries in `/tmp/luisa-tile-before-layout-plan-vfVuD5`. The experimental
bundle is `/tmp/luisa-tile-layout-tie-reversed-O61ihU`. Their bridge SHA-256 hashes
are respectively
`fb03b04343d3ed64fe6baa72d42ae181c7d5aef787144dc7198cc9c2173ef5c5`
and `3159491ce56e77011a3ecb28d0fbb13a85b3a89f0d995400b033711f375535a3`.
The experimental source change was just replacing the factor sort with
`std::sort(factors.rbegin(), factors.rend())`, plus a diagnostic comment.

## Four-round comparison

All 64 measurements validate complete outputs against the FP64 oracle. The
eight shapes rotate between rounds; implementation and framework order are
counterbalanced. Each row freshly captures/JIT-compiles the frozen
configuration, uses nine 40 ms timing batches after 200 ms warmup, and keeps
setup/cold costs separate. Inputs and preallocated outputs are device-resident.
No build, test, or profiler ran concurrently with timing.

Times are synchronized amortized **host-wall** measurements including dispatch,
not GPU-event durations. Values are medians of per-round medians. Speedup
ranges are observed paired min-max ranges, not confidence intervals; less
than one means the experimental map is slower.

| M x N x K | Reference us | Experimental us | Paired speedup median [range] | Experimental-run PyTorch us |
|---|---:|---:|---:|---:|
| 32 x 32 x 32 | 6.039 | 6.310 | 0.961 [0.932, 0.970] | 26.797 |
| 128 x 128 x 128 | 13.159 | 13.225 | 0.995 [0.893, 1.032] | 26.865 |
| 512 x 512 x 512 | 55.575 | 57.244 | 0.972 [0.964, 0.973] | 48.404 |
| 1024 x 1024 x 1024 | 321.488 | 333.388 | 0.965 [0.963, 0.967] | 289.348 |
| 256 x 1024 x 128 | 19.358 | 19.863 | 0.977 [0.970, 1.009] | 29.736 |
| 1024 x 128 x 256 | 23.573 | 24.157 | 0.981 [0.967, 0.983] | 29.791 |
| 127 x 193 x 61 | 11.999 | 12.229 | 0.977 [0.958, 1.076] | 27.276 |
| 513 x 257 x 129 | 27.868 | 28.414 | 0.985 [0.962, 0.991] | 34.438 |

No shape or round was removed. A few individual small/ragged rounds favor the
alternative, but none of the eight paired medians does. The large-square
slowdown is consistent across all four rounds.

## Consequence for the model and solver

The exact solver has not missed the model's optimum: these plans tie under
the current objective. The problem is missing discriminatory features. Equal
fragment counts do not capture the different participant/address maps and
issued instruction sequence. This experiment does not determine whether
transactions, bank behavior, instruction scheduling, or another mechanism
causes the measured difference.

Keep exact enumeration/Pareto DP as the mathematical reference. A later
measurement-oriented shortlist should retain layout diversity even across
equal score/resource states; a more elaborate integer solver or annealing
search cannot distinguish plans that the evaluator treats identically. The
production default's win here is not evidence that its tie order generalizes
to other shapes, transposes, architectures, or compiler revisions.

## Independent copy-layout trial

After restoring the production matrix tie order, a separate experiment changed
the cooperative copy's mixed-radix lane/value map. For batch size B, T workers,
batch ordinal c, worker t, and local value i:

```text
reference:    linear = (c * B + i) * T + t
experimental: linear = (c * T + t) * B + i
```

Both are bijections over each complete `T * B` prefix. The experiment kept
the reference guarded remainder, bounded source predicates, destination
ownership/effect checks, and load-before-store batching. It did not insert
unaligned vector loads, change the memory allocation, or assume asynchronous
copies. This is an execution distribution choice, not a new memory resource.

All reported plan features were identical for all eight pilot shapes, including
the restored `(2, 4, 4, 2)` matrix map, barrier counts, work counts, and cost
score. The scalar-copy default B=1 is unchanged. This also exposes a reporting
gap: the bootstrap features do not yet describe the copy lane/value map, so
the code and bridge fingerprint are essential to identify the realization.

The reference is the same `1ef0ffa7d` bundle. The experimental bundle is
`/tmp/luisa-tile-blocked-copy-JiUn4M`, with bridge SHA-256
`6213e0edb8762b8b05da96fa8a6a33257f0b94e96cb1a4a786a8805c18c2c895`.
Its full build and **137/137** unit tests passed in 58.67 seconds, including
bounded/reversed copies with 32, 48, and 256 workers and batch limits 1, 4,
and 16. The changed C++ file passed its syntax check.

An [eight-shape pilot](m1-max-20260903-blocked-copy-64x64x32/results.md) and
[four counterbalanced rounds](m1-max-20260903-blocked-copy-repeat/results.md)
use the same measurement contract as the matrix-layout trial, with no
concurrent build/test/profiler. All 64 repeat measurements validated:

| M x N x K | Interleaved us | Blocked us | Paired speedup median [range] | Blocked-run PyTorch us |
|---|---:|---:|---:|---:|
| 32 x 32 x 32 | 5.679 | 5.988 | 0.971 [0.929, 1.003] | 26.315 |
| 128 x 128 x 128 | 12.905 | 13.148 | 0.978 [0.944, 1.005] | 27.037 |
| 512 x 512 x 512 | 55.363 | 54.843 | 1.011 [1.003, 1.015] | 48.290 |
| 1024 x 1024 x 1024 | 321.976 | 318.454 | 1.010 [1.009, 1.039] | 289.398 |
| 256 x 1024 x 128 | 19.208 | 18.857 | 1.021 [0.979, 1.040] | 29.707 |
| 1024 x 128 x 256 | 23.644 | 23.614 | 1.002 [0.961, 1.009] | 29.524 |
| 127 x 193 x 61 | 11.851 | 11.842 | 1.011 [0.952, 1.016] | 26.815 |
| 513 x 257 x 129 | 27.952 | 27.877 | 1.004 [0.957, 1.010] | 34.686 |

The 512/1024-cubed paired medians improve by approximately 1%, while small
squares regress and other cases have mixed signs. One 1024-cubed reference
round was slower (330.845 us); it is retained. Speedups are medians of paired
ratios, not ratios formed from the two displayed medians. No source-level
vector spelling or hardware issue/transaction attribution follows from these
timings alone. The experiment does not justify replacing the default copy
map, and the production implementation was restored.

Future calibrated/JIT selection can retain both supported layout alternatives;
a universal preference is not established. The two experiments also reinforce
why work features, supported candidate maps, and search must be separate.
The [raw copy-layout comparison](m1-max-20260903-blocked-copy-repeat/results.json)
keeps all hashes, samples, correctness errors, and ordering.

## Reproduction and validation

See the [pilot](m1-max-20260903-layout-tie-64x64x32/results.md),
[controlled comparison](m1-max-20260903-layout-tie-repeat/results.md), and
[raw measurements](m1-max-20260903-layout-tie-repeat/results.json).

After completing the full build, both the solver/layout regressions and real
CPU/Metal coverage passed: **27/27** Tile tests in 56.05 seconds. The changed
C++ file passed its syntax check. The source perturbation was then reverted
after the measurements; both prebuilt bundles remain available for replay:

```sh
uv run --no-project --python 3.13 --with torch --with numpy python \
  scripts/benchmark/tile_torch/repeat.py \
  --reference scripts/benchmark/tile_torch/results/m1-max-20260903-barrier-plan-64x64x32/results.json \
  --candidate scripts/benchmark/tile_torch/results/m1-max-20260903-layout-tie-64x64x32/results.json \
  --native /tmp/luisa-tile-before-layout-plan-vfVuD5/benchmark_tile_tirx \
  --candidate-native /tmp/luisa-tile-layout-tie-reversed-O61ihU/benchmark_tile_tirx \
  --output /tmp/tile-layout-tie-fresh-repeat \
  --operations gemm --rounds 4 --samples 9 --sample-ms 40 --warmup-ms 200
```

Use a new output directory and do not overlap performance runs with builds,
tests, or profiling. These negative results supplement the
[planner design](../../../../docs/source/internals/tile/planner.md); they do not
close the remaining CPU/Metal performance gaps to PyTorch.

To replay the independent copy-layout trial, use the same command and reference,
but set `--candidate` to
`scripts/benchmark/tile_torch/results/m1-max-20260903-blocked-copy-64x64x32/results.json`,
`--candidate-native` to
`/tmp/luisa-tile-blocked-copy-JiUn4M/benchmark_tile_tirx`, and choose another new
output directory. The two experimental binaries must not be mixed: one changes
the matrix tie order, while the other changes only the copy map.

After both reversions, the full configured project was rebuilt and the final
**137/137** unit run passed in 58.26 seconds. Neither experimental C++ change
remains in the working tree. The rebuilt production bridge hash is
`9a37d1986b2d27bf4c59b701abbaa3b04030aba39c79e4a219986b8bc59b4686`;
the measured reference remains the separately fingerprinted frozen bundle.

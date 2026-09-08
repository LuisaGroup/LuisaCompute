# TIRx structural experiments and joint schedule search

The native MPP path does not retire TIRx. The
[five-path comparison](m1-max-20260904-native-lowerings/results.md) keeps native
Tile→MPP, Tile→TIRx, handwritten MPP, direct MPS, and eager Torch separate.
This follow-up investigates the remaining TIRx gap without changing that
comparison into a library substitution.

The later [same-source Runtime controls](m1-max-20260904-runtime-controls/notes.md)
leave the large-GEMM gap essentially unchanged. A subsequent
[fragment-load batching experiment](m1-max-20260904-fragment-batch/notes.md)
checks 144 full outputs but regresses 1024³ by 4.6%/17.8% for batches of two/four.
That emitter change was also reverted; its patch, all sources, and measurements
are retained. Neither result justifies changing the default planner policy.

## Rejected contraction unrolling

The only emitter change in this experiment made the rectangular matrix
reduction loop `kUnrolled` instead of `kSerial` when its contraction extent
was at most 64 (at most eight 8×8 atom steps). It did not change the outer
pipeline, input staging, subgroup distribution, arithmetic order, or buffers.
The source change has been reverted.

[Four counterbalanced rounds](m1-max-20260904-unroll-replay/results.md), with
all eight incumbent schedules frozen, validate 64 complete outputs against
the FP64 oracle. Each run recaptures/JITs, warms for 200 ms, and takes seven
30 ms timing batches. Values below are medians of per-round medians in
microseconds, synchronized host-wall throughput including dispatch—not GPU
event times. No build, test, or profiler ran during timing.

| Shape | Serial incumbent | Unrolled experiment | Observation |
|---|---:|---:|---|
| 512³ | 53.522 | 51.123 | About 4.5% lower time |
| 1024³ | 320.084 | 375.548 | About 17.3% higher time |

The late-prefetch analysis also declined the new loop kind, causing its
structural regression assertions to fail. This is a pass-composition issue,
not evidence that prefetching or unrolling is semantically invalid. The
experiment is not promoted, and the existing CPU/Metal matrix tests pass
again after restoring the serial emitter. Neither those assertions nor the
numerical tolerances were weakened.

## MPP-like group geometry is not sufficient

A separate [six-geometry, 128-thread search](m1-max-20260904-cohort128-search/results.md)
includes `128×32×32` and `32×128×32` group tiles. The former permits four
subgroups to own `32×32` outputs, resembling the successful MPP cohort's
output ownership. However, on 1024³ its exploratory TIRx result was 416 µs;
even the best tested 128-thread geometry remained about 343 µs. These are
selection measurements, not an independent speedup claim.

Matching the output rectangle and participant count does not reproduce MPP's
internal implementation. These experiments do not identify a hardware bank,
register spill, or occupancy limit; source inspection and elapsed time alone
are insufficient to make those claims.

## Independent handwritten SIMD-group probes

The pre-existing, uncommitted `benchmark_tile_manual.mm` was inspected and
run without modification. It is a handwritten **8×8 SIMD-group** experiment,
not the handwritten **MPP** comparison. Its source and executable hashes,
commands, all 80 checked outputs' validation records, errors, and samples are
in [the raw report](m1-max-20260904-manual-probe.json). The original source
remains separately owned work; these notes do not incorporate it into the
production lowering.

Four rounds use rotated/reversed variant order on 512³ and 1024³, five 20 ms
batches and 150 ms warmup. Variant pairs have balanced precedence over each
forward/reverse pair; this is not a complete position-balanced ten-variant
design. All complete outputs pass the FP64 oracle. The helper explicitly
enables Metal fast math. These values compare its variants internally, not
strict-math equivalence or pure GPU times across frameworks.

| Handwritten variant | 512³ µs | 1024³ µs |
|---|---:|---:|
| Shared staging | 55.240 | 328.548 |
| Direct global fragment loads | 54.559 | 439.581 |
| Pad both row strides by one float | 55.482 | 355.507 |
| Pad both row strides by eight floats | 55.135 | 353.194 |
| Stream one A fragment at a time | 55.858 | 328.868 |
| Column-major group order | 55.369 | 329.157 |
| Morton group order | 55.287 | 326.753 |
| Two shared staging slots | 57.331 | 405.250 |

Direct loading, padding, and a second shared slot are therefore not justified
as new defaults by this experiment. The small Morton difference is not proof
of a general scheduling win. No one of these timings is a hardware lower bound.

## Consequence for tuning

The outer Staged/JIT tuner now searches an explicit joint product of block
shape, pipeline window, group width, and copy-batch limit. It still recaptures
ordinary C++ kernels and records each realized planner result. Candidate
budgets reject oversized products; invalid candidates cannot win; the entire
winning configuration is recaptured and remeasured after selection.

The [joint search](m1-max-20260904-joint-search/results.md) includes every
previous incumbent's parameters. It is exploration, not a published speedup;
independent frozen replay must assess its winners. This adds measurement
coverage, not a calibrated cost model, a new DSL primitive, or a claim that
an integer solver can choose an atom family the emitter cannot generate.

### Independent same-binary replay

[Four counterbalanced rounds](m1-max-20260904-joint-replay/results.md) compare
the old and selected schedules using the same compiler/runtime binary, all
eight shapes, seven 30 ms batches and 200 ms warmup. All 64 complete outputs
pass. Each cell below is a median of per-round host-wall medians; the speedup
is the median of paired incumbent/selected ratios.

| Shape | Incumbent µs | Selected µs | Paired speedup |
|---|---:|---:|---:|
| 32³ | 5.249 | 4.748 | 1.116× |
| 128³ | 9.821 | 6.841 | 1.453× |
| 512³ | 53.536 | 53.447 | 0.998× |
| 1024³ | 319.760 | 319.902 | 0.999× |
| 256×1024×128 | 19.326 | 19.086 | 1.024× |
| 1024×128×256 | 21.265 | 19.793 | 1.083× |
| 127×193×61 | 9.174 | 8.683 | 1.054× |
| 513×257×129 | 23.197 | 22.591 | 1.032× |

The strong small-square improvement is real within this replay; neither large
square improves. On 1024³ the selected and incumbent settings are identical,
so its difference reflects measurement variation, not a compiler optimization.
512³ changes only the copy-batch limit and is indistinguishable in this run.
The Torch/MPS large-square gap therefore remains open.

| Shape | Selected block | Group threads | Copy-batch limit |
|---|---|---:|---:|
| 32³ | 32×64×32 | 256 | 8 |
| 128³ | 32×32×128 | 256 | 8 |
| 512³ | 64×64×64 | 256 | 8 |
| 1024³ | 64×64×32 | 256 | 4 |
| 256×1024×128 | 64×64×32 | 256 | 8 |
| 1024×128×256 | 32×64×32 | 256 | 8 |
| 127×193×61 | 32×64×32 | 256 | 8 |
| 513×257×129 | 32×64×32 | 256 | 8 |

Every selected pipeline window is one. The copy number is an upper bound,
not a promise that every copy uses that many values. These are measured
shape-specific configurations, not new global planner defaults.

The subsequent [ten-round five-path replay](m1-max-20260904-joint-lowerings/notes.md)
validates all 400 outputs with native MPP, TIRx, handwritten MPP, MPS and Torch.
Its hashes are stable and its positions/pairwise precedence are balanced.
TIRx remains faster than native MPP on 128³ and 513×257×129; the large-square
Torch/MPS gaps remain. Both lowering paths must stay independently tested.

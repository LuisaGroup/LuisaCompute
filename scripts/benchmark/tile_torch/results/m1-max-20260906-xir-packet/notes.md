# Bounded packet-index proof closes one SIMD codegen disconnect

The final frozen compiler A/B improves the four nontrivial aligned GEMMs in
every one of six paired rounds, in both batched dispatch throughput and
single-call latency. The change is generic Schedule index analysis: no Tile
DSL change, GEMM-name substitution, math relaxation or planner coefficient
change. **It does not close the Torch/Accelerate gap.**

See [protocol](protocol.md), [final raw report](final-replay/results.json),
[independent final audit](final-audit.json), [audit program](audit.py), and
the [separate MPS capture record](mps-capture.md). The earlier
[replay](replay/results.json) and [receipt](audit.json) are retained, not
pooled with final-binary timings. Source review added a conservative W3
rejection between the two replays; W8 generated IR is unchanged.

## Final comparison

Apple M1 Max; FP32 row-major alpha=1/beta=0; fixed Tile 1×1×8, packet W8,
eight requested CPU workers. Torch 2.14.0 at
`08187d9e0fba026dc8217405802ab5381dc88d90` reports Accelerate BLAS. Six balanced
implementation permutations, rotating shape order, five samples, 20 ms windows
and 100 ms warmup. Both compiler arms select the **same execution plan and
relative-work cost**. Inputs and output are preallocated, with full FP64
validation outside timing.

Times are warm synchronized **host-wall dispatch** microseconds, not CPU
kernel-only times. Columns contain medians of per-round p50s; ratios are
medians of same-round ratios, not ratios of the displayed medians. A ratio
above one means the new compiler is slower. Ranges are observations, not
confidence intervals. Every slow sample is retained.

| M×N×K | Old µs | New µs | Torch µs | New/old [min,max] | New slower rounds | New/Torch |
|---|---:|---:|---:|---:|---:|---:|
| 32×32×32 | 51.739 | 38.913 | 0.978 | 0.756 [0.738,0.796] | 0/6 | 39.986 |
| 128×128×128 | 301.521 | 118.771 | 4.936 | 0.398 [0.366,0.459] | 0/6 | 24.215 |
| 512×512×512 | 12528.875 | 4142.333 | 146.611 | 0.326 [0.312,0.401] | 0/6 | 28.267 |
| 1024×1024×1024 | 109013.021 | 39793.646 | 985.279 | 0.367 [0.348,0.403] | 0/6 | 40.493 |
| 128×2048×512 | 13117.948 | 5592.222 | 158.803 | 0.427 [0.359,0.450] | 0/6 | 35.075 |
| 127×193×61 | 246.743 | 247.816 | 6.549 | 1.006 [0.997,1.129] | 5/6 | 38.045 |

Single-call dispatch latency is measured separately:

| M×N×K | Old µs | New µs | Torch µs | New/old [min,max] | New slower rounds |
|---|---:|---:|---:|---:|---:|
| 32×32×32 | 47.855 | 41.730 | 1.042 | 0.864 [0.552,1.195] | 1/6 |
| 128×128×128 | 291.542 | 114.583 | 4.959 | 0.392 [0.356,0.431] | 0/6 |
| 512×512×512 | 12499.500 | 4331.125 | 154.376 | 0.347 [0.297,0.375] | 0/6 |
| 1024×1024×1024 | 104745.125 | 39709.729 | 1007.833 | 0.379 [0.364,0.415] | 0/6 |
| 128×2048×512 | 13171.396 | 5670.916 | 158.188 | 0.429 [0.370,0.483] | 0/6 |
| 127×193×61 | 244.667 | 236.979 | 6.458 | 0.982 [0.818,1.134] | 3/6 |

Median compile time for the five aligned shapes is 24.57–27.32 ms with the
new emitter, versus 37.96–41.05 ms for the old one. The unchanged ragged IR
takes 57.75 versus 57.50 ms. These are separately recorded cold JIT phases,
not included in dispatch speed ratios and not a general compile-time study.

The ragged control has **byte-identical generated LLVM IR** between arms in
every round. Its 0.6% median throughput regression and mixed latency are
retained; they do not prove a changed kernel. The initial replay also had a
large 32-cubed outlier and a ragged latency outlier. Tiny dispatch measurements
are especially sensitive to host/runtime variability; neither control is
dropped or used to invent a thermal attribution.

## Implementation and remaining structural gap

The old Tile index path widens uint32 dispatch x to signed i64, unflattens
with quotient/remainder, and finally casts to an unsigned buffer index.
Schedule analysis discarded consecutive casts and had no packet-aligned
quotient/remainder rule. It consequently emitted 16 gathered operand reads
per static K chunk for these aligned 1×1×8 GEMMs, although the planner's
floating slope estimate preferred broadcast/contiguous memory.

The new proof recognizes `x[lane]=W*q+lane` with nonnegative range bounds,
checks value-preserving casts and aligned offsets, and proves that division
by a positive multiple of W is cohort-equal while remainder is consecutive.
Unknown, cross-row, ragged, negative, narrowing and non-power-of-two cases
remain unproved. These facts enable the **existing** masked memory emitter.
The five aligned shapes now report eight contiguous B reads and eight A
broadcasts. The ragged case retains zero of both. These are IR/emitter counts,
not final CPU instruction counts, cache counters or MPS measurements.

All six shapes still lose to Torch in every final throughput and latency
pair. The nontrivial aligned throughput time ratio is 24.2–40.5×. The current
bridge still expands local Tile elements into per-worker scalar SSA, packs
independent programs across CPU lanes, and lacks a general cache/register-
blocked matrix realization. Better index proofs fix a missing path to existing
vector loads; they do not provide cross-Tile reuse or a BLAS-class matrix
microkernel. The next CPU priority remains this realization family, plus
distribution of wide row/elementwise programs—not fitting this unchanged cost
model to the new times. Multi-operator Runtime tests pass, but this experiment
does **not** claim new LLM-operator performance measurements.

## Verification and provenance

The final audit independently checks all 108 outputs (29,066,094 elements),
all six permutations and actual row order, sample p50s, shapes, strict math
policy, unchanged selected plans, generated memory classifications and
source/output hashes. All outputs match the dyadic FP64 oracle exactly; this
is not a general floating-point accuracy claim. All 38 compiler/runtime/source
artifacts remain unchanged during the final replay. The first replay also
passed 108 complete validations; it belongs to the earlier binary boundary.

The relevant final build and four selected CTests pass: XIR-to-Schedule,
LLVM/JIT codegen, Tile XIR Runtime and Tile XIR LLM. New tests exercise W2/4/8/16
proofs, W3 rejection, ragged/shifted/negative/dynamic divisors, truncation and
bitcasts, plus 24 enabled/disabled JIT executions with sparse lanes, row
crossings, tail packets and output canaries. Runtime GEMMs add six aligned/
ragged M/N/K and multi-row-Tile cases with non-dyadic inputs and guarded views.
Existing RMSNorm, LayerNorm, SwiGLU, RoPE, masked softmax, GELU+residual and
prefill/decode/GQA correctness checks remain in the test run. This is four
selected passing tests, not an all-green whole-worktree claim.

All 95 Python benchmark unit tests pass. Project clangd checks and changed-
region clang-format checks pass for the four modified C++ sources. Initial
test development included a corrected invalid 1×1×1 block fixture; the final
cross-row negative case uses a legal two-dimensional block instead. No
Runtime block-size restriction was weakened.

Five in-memory audit mutations reject a missing round, wrong ordering,
nonfinite sample, false memory classification and changed execution plan.
The two replay source inventories independently confirm identical W8 LLVM
before/after the W3 guard. A fresh Sphinx build/check validates 48 pages,
3,616 local links/assets and 199 compatibility anchors. The strict build
still reports the ten pre-existing missing-Doxygen-XML warnings, not new
handwritten-document errors. Desktop 1440×1050 and mobile 390×844 renders are
inspected: the table fits the 696-pixel desktop content area, and its 338-pixel
mobile container scrolls horizontally to the final column without page overflow.

The reports retain every timing row, command, source/output identity and
per-round plan. Representative round-zero LLVM files are committed; duplicate
round dumps and raw `.f32` outputs remain local under each replay directory.
The full local audit therefore needs those local outputs and frozen binaries;
JSON receipts alone are not a substitute for re-executing validation.

No Metal implementation or default was changed in this checkpoint. The new
8192-cubed MPS capture validates successfully, but Xcode inspection timed out;
its launch/counter attribution remains unresolved. It is recorded separately
and excluded from CPU/GPU performance claims.

# MPP operation-scope screening: no general replacement selected

The fixed five-configuration screen does not justify switching the default
atom to a whole-group collective. At the same 128-thread, 128×32 output-group
geometry, four independent 32×32 operations and one collective 128×32
operation trade places by shape. Other rectangles also have mixed results.
These are single-order observations, not an accepted model or speedup.

The [predeclared protocol](protocol.md) fixes all six shapes and configurations.
All 36 complete outputs pass the unchanged FP64 oracle, atol=rtol=1e-4;
both benchmark executables retain their hashes. Compact row-major FP32,
alpha=1, beta=0, no prepacking/transposes, fast math or relaxed precision.
The source binaries precede the subsequent bounded-K TIRx extension; this
standalone screen neither uses TIRx nor changes a Tile lowering.

## Exact GPU comparison

Values are within-process p50 **command-buffer batch microseconds**, five
samples with a 20 ms target and 100 ms warmup. They are not isolated kernel
instruction timings or compute-pass counter measurements. MPS runs first
for each shape; candidate order rotates across shapes, not repeat rounds.

- A: 4 independent 32×32 operations; 4 subgroups per group, cohort 4×1.
- B: 1 collective 128×32 operation; all 4 subgroups participate.
- C: 1 collective 64×64 operation; all 4 subgroups participate.
- D: 1 collective 32×128 operation; all 4 subgroups participate.
- E: 1 collective 64×32 operation; both of the group's 2 subgroups participate.

| M×N×K | MPS µs | A µs | B µs | C µs | D µs | E µs |
|---|---:|---:|---:|---:|---:|---:|
| 1024³ | 275.954 | 273.242 | 283.757 | 354.057 | 363.837 | 314.706 |
| 4096³ | 22799.583 | 39254.042 | 35525.083 | 46655.750 | 55837.667 | 77917.875 |
| 8192³ | 405878.625 | 568783.083 | 454340.208 | 569715.250 | 482146.875 | 378802.500 |
| 256×11008×4096 | 4244.427 | 6834.056 | 6721.278 | 7322.347 | 10265.771 | 6177.625 |
| 4096×4096×11008 | 61012.833 | 144021.375 | 127577.000 | 162673.875 | 155725.292 | 131438.958 |
| 2049×4097×1025 | 12006.771 | 5803.167 | 4247.313 | 4198.552 | 4595.969 | 2959.549 |

The complete [generated table](search/results.md) retains every arm's host
batch p50. [Raw samples](search/results.json) additionally retain host/GPU
single-call latency, output-error receipts, exact commands and fingerprints.
The companion [independent audit](../m1-max-20260906-mpp-bounded-k/audit.py)
recomputes all four metrics and checks the screen's rotated order.

## Interpretation and limits

B is below A in five shapes, but not at 1024³; it still trails direct MPS
at 4096³, 8192³ and both large rectangles. E's exploratory 8192³ minimum
does not survive as a universal choice: its 4096³ result is worse than A/B.
Do not install these single-order minima into a cost policy. Parameter search
and independent acceptance remain separate.

Large-shape MPS/MPP times differ substantially from other sessions, including
the later bounded-K replay. There was no concurrent build, test or profiler;
this does not establish a thermal, clock, cache or shader-internal cause.
The saved 8192³ MPS capture still lacks recovered launch/counter attribution;
the older 1024³ capture cannot supply it. No cross-session speed ratio is used.

The next realization work should retain operation participation scope as a
backend candidate dimension, alongside K chunking, memory reuse and output
distribution. This experiment supplies a negative control against the claim
that wider MPP participation alone closes the gap. It does not add a frontend
entity or turn a buffer layout into an execution hierarchy.

## Visual contract

The repository's existing Sphinx/Markdown report is the delivery surface.
This exact-lookup table preserves six shapes, five fixed configurations and
the MPS control; a single-scale bar chart would hide the microsecond-to-
hundreds-of-milliseconds range. No rank/minimum chart is used to imply an
accepted winner. Units, single-order status and raw provenance stay adjacent.

# Physical program walks: rejected stripes and an inconclusive rectangle screen

## Technical summary

The hand-MPP probe now supports an **optional bounded 2D program permutation**,
without changing output ownership, memory layouts, MPP scope, K extent or
launch cardinality. The old 2D launch remains the default. This is a benchmark
capability, **not yet a TileIR/TIRx/native backend planner feature**.

A first two-order screen rejects one-column stripes of 4 or 8 program rows:
both are slower than the linear-launch control on all five shapes in both
orders. At 8192³ their GPU batch ratios over linear are 1.551/1.544 and
1.828/1.813. A follow-up with square output-region rectangles has substantial
order reversals and unstable MPS controls. **It is not suitable for fitting
a cost model or declaring a performance win.** All negative and contradictory
observations are retained.

## Scope, definitions and candidates

Apple M1 Max, macOS 26.6.2, September 6, 2026. FP32 compact row-major C=A×B,
no packing/transposes, alpha=1, beta=0, preallocated output, no fast math or
relaxed input precision. Every path validates its entire output against a
NumPy FP64 reference using atol=rtol=1e-4. The inputs are the existing dyadic
benchmark pattern; no new non-dyadic or low-precision claim is made.

Hand MPP uses 32×32 single-SIMD-group operations, four subgroups arranged
4×1: effective group output is 128×32. It processes **whole K**, whereas the
earlier TIRx experiment partitions K. Cross-experiment times must not be used
as a same-code ablation. The MPS control is a separate direct executable;
Torch/native Tile/TIRx were not remeasured in the walk cohorts.

The original `legacy` launch is 2D. `linear` uses one physical launch axis
but maps to the same row-major logical program coordinates. Stripe candidates
visit column-first within program rows. Rectangle candidates visit row-major
inside 2×8, 4×16 or 8×32 program rectangles, corresponding to 256², 512² or
1024² output regions for this fixed group geometry.

Apple's [MPP Programming Guide, §§2.3.3–2.3.4](https://developer.apple.com/download/files/Metal-Performance-Primitives-Programming-Guide.pdf)
motivates testing program locality and K synchronization. Its M5-specific
heuristics are not assumed to apply to this M1 Max. These candidates are
bounded rectangles, not Morton order, and no cache-hit counter was collected.

## The mapping preserves the program domain

For grid rows H, columns W, requested rectangle R×C, first clamp C to W.
For linear physical program i:

```text
stripe_start = floor(i / (R*W)) * R
h = min(R, H - stripe_start)
local = i mod (R*W)
column_start = floor(local / (h*C)) * C
w = min(C, W - column_start)
inside = local mod (h*C)
row = stripe_start + floor(inside / w)
column = column_start + (inside mod w)
```

The last stripe and rectangle use their actual sizes. Each partition has
exactly height×width programs; local row-major enumeration and concatenation
are bijective. There are no padded launches, duplicate outputs or missing
tail programs. This is a permutation of independent `parallel` instances,
not a dependence proof or a guarantee about the GPU's scheduling order.

The C++ probe checks the physical grid and R×W products against uint32 before
allocating tensors, and performs device products unsigned. A separate host
enumeration checks 2,688 rectangular-grid/request combinations. Two complete
GPU outputs additionally exercise a product that exceeds int32 but fits
uint32; four malformed/overflowing requests fail before allocation.

## One-column stripes are rejected

Each cell is **round 0 / round 1** median of five samples, in **GPU
command-buffer batch microseconds**. This includes GPU work and gaps, not
isolated kernel time. Path order is rotated by shape, then exactly reversed
for each shape in the second round; shape order also reverses.

| M×N×K | legacy | linear | stripe4 | stripe8 | mps |
|---|---:|---:|---:|---:|---:|
| 1024×1024×1537 | 503.688 / 507.574 | 482.259 / 498.918 | 521.664 / 521.788 | 532.238 / 532.458 | 433.198 / 437.180 |
| 4096×4096×4096 | 19171.583 / 19596.333 | 17903.667 / 18490.208 | 24421.375 / 22947.125 | 30212.500 / 30346.208 | 17844.083 / 18110.375 |
| 8192×8192×8192 | 375314.250 / 374648.125 | 328866.000 / 330297.875 | 510222.667 / 510074.708 | 601198.875 / 598803.000 | 152791.208 / 155870.333 |
| 4096×4096×11008 | 71396.083 / 71275.375 | 73061.875 / 71882.750 | 103043.208 / 102315.250 | 146566.125 / 146428.125 | 51143.667 / 49948.125 |
| 2049×4097×1025 | 2963.396 / 2954.819 | 2976.819 / 2978.028 | 3011.160 / 3082.325 | 3069.146 / 3100.317 | 2776.375 / 2749.868 |

Both stripes lose all ten pairs to linear, in GPU and E2E batch throughput.
Linear also beats legacy on 1024×1024×1537, 4096³ and 8192³ in both orders,
but loses on the long-K and ragged shapes. Even linear at 8192³ remains
2.152× / 2.119× MPS. Thus neither a universal launch change nor MPS parity
follows from this screen.

TIRx's cooperative group mapper already emits a one-dimensional launch.
The legacy-to-linear hand-probe difference is not a new Tile/TIRx optimization.

## Square output regions do not yield a stable winner

This is a **separate follow-up**, not another round appended to the stripe
cohort. It retains fresh linear/MPS controls; do not mix its absolute values
with the table above.

| M×N×K | linear | rectangle2x8 | rectangle4x16 | rectangle8x32 | mps |
|---|---:|---:|---:|---:|---:|
| 1024×1024×1537 | 483.672 / 824.818 | 513.976 / 885.300 | 481.929 / 876.964 | 489.788 / 904.710 | 440.930 / 834.442 |
| 4096×4096×4096 | 27374.667 / 24159.500 | 19992.125 / 41695.875 | 21329.000 / 47911.417 | 23314.417 / 38369.000 | 23686.208 / 29514.417 |
| 8192×8192×8192 | 362034.625 / 598058.750 | 421905.250 / 472995.708 | 451713.958 / 513553.000 | 459206.000 / 615919.583 | 435384.167 / 385647.833 |
| 4096×4096×11008 | 110839.208 / 89070.042 | 84036.750 / 150857.500 | 98195.042 / 90561.292 | 94314.625 / 122356.250 | 68067.625 / 87038.000 |
| 2049×4097×1025 | 4172.687 / 2946.486 | 3762.575 / 3092.396 | 3138.708 / 2990.604 | 2944.083 / 2948.896 | 4492.062 / 2749.667 |

At 4096³, rectangle2x8/linear GPU batch is 0.730× in round 0 but 1.726× in
round 1; E2E is likewise 0.732× / 1.723×. At 8192³ the same candidate flips
from 1.165× to 0.791× linear. MPS/linear rankings also reverse. At the small
1024×1024×1537 shape, MPS's per-round GPU batch median changes from 440.930
to 834.442 µs. Reporting only the fast half would be misleading.

A read-only process snapshot **after** this cohort observed other interactive
desktop CPU load. It did not measure concurrent per-process GPU use, clocks
or thermal state, so it cannot establish the cause of the variation. No
applications were closed. A quiet-machine, balanced replay is needed before
accepting any rectangle candidate. The user was asked asynchronously whether
a quieter measurement window would be convenient.

## Validation and evidence boundary

The [independent audit](audit.py) checks **244 complete outputs and
2,208,985,892 elements** across the two 50-output screens and three 48-output
development/final-binary correctness cohorts. All saved receipts have
max-absolute-error zero. The first two small cohorts intentionally retain
earlier source/binary versions, not the final rectangle implementation.
The final pre-timing 48-output cohort uses exactly the rectangle-screen
binary. Tests span both single-subgroup cohort orientations and a four-
subgroup collective, with tiny/partial rows, columns and K.

The two subsequent unsigned-boundary outputs in [boundaries.json](boundaries.json)
add 4,258 checked elements; the four negative requests are not numerical
outputs. The final host-format-only build passed those checks. All 95
benchmark Python tests pass, including rejection of a permuted MPP result
presented as the default baseline.

- [Original stripe raw results](results.json), [rectangle raw results](results-rectangles.json)
  and [audit.json](audit.json) retain all four timing metrics: GPU batch/single
  and E2E batch/single. Medians are recomputed from all raw samples; none are
  deleted or silently winsorized.
- Five negative audit probes reject missing rows, wrong round order, wrong
  element count, NaN time and mismatched program walk.
- Executable hashes are unchanged within each cohort. Final formatting changed
  the host executable identity after timing; frozen pre-format binaries are
  retained at `/tmp/luisa-mpp-walk-evidence.9fp9RO` and match the rectangle
  cohort. Archive source hashes remain authoritative per shader variant.
- [Protocol](protocol.md), [driver](experiment.py), source archives and numerical
  receipts preserve the exact experiment. The driver now explicitly requests
  non-default walk metadata from the stricter validator; this does not alter
  the previously compiled/recorded shader work.
- No concurrent builds, hardware tests or GPU profilers were launched by this
  task during the screens. Lightweight read-only source/audit work continued
  on the host; this is not an isolated-laboratory acceptance run.
- Full CMake builds preceded each binary generation's tests. Large timing
  controls are no-counter command-buffer measurements. No new MPS capture,
  per-kernel hardware counters or clock/occupancy attribution is claimed.

Audit retained artifacts without rerunning the benchmark:

```bash
uv run --offline --no-project --python 3.13 python \
  scripts/benchmark/tile_torch/results/m1-max-20260906-mpp-grid-walk/audit.py \
  --current-artifacts --artifact-dir /tmp/luisa-mpp-walk-evidence.9fp9RO --self-test
```

## Implications for the compiler and next questions

Keep physical program traversal distinct from the execution domain and
local memory layout. A future solver candidate should compose a checked
program permutation with the existing subgroup/local-coordinate map and
then the operand address maps. Its features must account for **physical
output aspect and K partition together**; a generic reward for larger row
groups is contradicted by the stripe data. Backend-owned policy should rank
those candidates only after stable held-out tests; the measured/JIT winner
must remain authoritative.

No production lowering, scheduling default or cost coefficient changes in
this checkpoint. Next: obtain a stable comparison window, test K partition
jointly with traversal only if the standalone evidence warrants it, and keep
the larger SIMD register/cache-blocking gap on the work list. The broad
MPS/Torch performance goal remains open.

Report surface: existing repository Markdown/Sphinx. Exact two-order tables
are used for audit lookup rather than statistical error bars or a selected-
minimum chart. Scope, method, uncertainty and next questions surround the
findings; no extra docs hierarchy, site or dashboard is introduced.

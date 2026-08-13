# SIMD CPU backend performance report

Snapshot date: 2026-08-13. This report covers the Release build after merging
`origin/next@62b77df36b6dae05aff558d4db84e415b5e84e75` into
`codex/simd-cpu-backend`, adding coherent direct-CFG lowering, completing the
bindless gradient-sampling vertical slice, and eliminating curve-hit
postprocessing from curve-free acceleration structures and native-width
direct-trace packet copies. The current snapshot additionally promotes
eligible local aggregates before Schedule IR, allowing their varying fields to
remain independent SoA SSA values instead of repeatedly crossing an AoS
storage boundary, and selects a surface-only ray-query runtime for acceleration
structures whose current instances are all triangle meshes. The latest stage
also retains hot ray-query predicates in a proven JIT-side packed status
sidecar and caches the corresponding fixed-vector state handles at W4/W8/W16
while leaving the public query state and plain Embree providers unchanged.
The current final stage treats the cached status entry and its construction-
selected plain provider as an audited internal pair, removing one redundant
JIT callback gather while retaining provider-side fail-closed validation.

## Test host and method

- AMD Ryzen 9 9950X3D, 16 cores / 32 hardware threads;
- LLVM and Clang 22.1.8;
- Embree 4.4.1, reporting native W4/W8/W16 packet support;
- ISPC 1.31.0 for the optional same-algorithm control;
- `CMAKE_BUILD_TYPE=Release`;
- unrelated work was active, so every result uses alternating forward/reverse
  order and a median rather than a best run.

Unless a row states a newer paired sweep, graphics and SDF cells below are
medians of seven independent processes. Image processing repeats its four-
dispatch pipeline 32 times, voxel repeats 16 renders, and Spacex renders four
frames after its upload/update synchronization.
Cutout path tracing uses 64 spp and ordinary path tracing uses 128 spp; both
force one spp per dispatch on both backends to remove a batching asymmetry.
Their current rows use three adjacent fallback/SIMD pairs per width with
reversed order on alternating pairs. The focused triangle-only-provider result
uses twelve W8 pairs, while the other widths use four to six pairs. Image,
voxel, Spacex, and ordinary path tracing compare every measured output with
the repository gallery reference. The refreshed 64-spp cutout processes are
performance-only; a separate 1024-spp run supplies its gallery conformance
gate. SDF uses its internal four-SPP throughput metric; high-SPP SDF image
comparison remains a separate conformance gate. Image/SDF/voxel/Spacex/GEMM
cells retain the earlier seven-process sweep because the relevant kernels have
no eligible aggregate local and unchanged JIT code under this transform.

Speedup is always `fallback time / SIMD time`, or
`SIMD throughput / fallback throughput`, so values above one are wins.

## Current fallback-relative results

| Workload and metric | fallback | W1 | W2 | W4 | W8 | W16 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| non-coro SDF, samples/s | 8.705 | 8.197 (0.942x) | 9.476 (1.089x) | 15.112 (1.736x) | 22.568 (2.593x) | 32.959 (3.786x) |
| image pipeline, ms/iteration | 8.379 | 17.184 (0.488x) | 9.169 (0.914x) | 6.493 (1.290x) | 4.992 (1.678x) | 4.249 (1.972x) |
| voxel render, ms/iteration | 6.904 | 8.127 (0.850x) | 24.122 (0.286x) | 16.128 (0.428x) | 9.386 (0.736x) | 6.479 (1.066x) |
| Spacex, ms/frame | 158.831 | 150.954 (1.052x) | 94.904 (1.674x) | 64.277 (2.471x) | 49.999 (3.177x) | 42.700 (3.720x) |
| ordinary path tracing, fixed 1 spp/dispatch, spp/s | 69.784 | 61.813 (0.863x) | 51.087 (0.730x) | 63.853 (0.932x) | 74.108 (1.108x) | 79.461 (1.133x) |
| cutout path tracing, fixed 1 spp/dispatch, spp/s | 59.366 | 45.860 (0.770x) | 29.919 (0.511x) | 36.802 (0.619x) | 42.791 (0.730x) | 42.283 (0.711x) |
| portable GEMM, GFLOP/s | 64.895 | 23.332 (0.360x) | 25.627 (0.395x) | 115.914 (1.786x) | 190.521 (2.936x) | 316.449 (4.876x) |

The GEMM row is a compute diagnostic rather than a graphics result. It uses
eight explicit SIMD workers and seven independent process medians; every
process performs seven timed samples of 128 complete 256-by-256 dispatches and
validates the output against double-precision accumulation. The fallback
process medians ranged from 41.594 to 88.326 GFLOP/s under shared-host load,
while the SIMD distributions were tight. Its relative speedups must therefore
be treated as host observations, not cross-machine constants.

The current path-tracing rows are paired rather than independent medians because
unrelated host tasks moved the load average during the sweeps. The displayed
fallback cell is the pooled fallback median; each SIMD cell is its three-process
median and the parenthesized speedup is the preferred geometric mean of three
adjacent SIMD/fallback ratios. Every one of the 60 performance processes passed
its required correctness gate: all 30 ordinary processes passed their gallery
comparison, while cutout used the separate 1024-spp comparison described
above. Ordinary W8 and W16 won all three pairs and exceed fallback by 1.1081x
and 1.1329x respectively. W1/W2/W4 remain at
0.8627x/0.7303x/0.9325x. The final-binary cutout row was refreshed with three
adjacent alternating pairs per width after the state-handle cache landed. It
remains below fallback at every width: 0.7695x/0.5105x/0.6193x/0.7298x/0.7111x
from W1 through W16. The displayed throughput cells are process medians; those
ratios are the preferred paired geometric means. Its JIT-side query payload
crossings and sparse cohorts remain the dominant unresolved deficit.

### Pre-schedule aggregate promotion

The ordinary and cutout path kernels each report 11 decomposed aggregate
allocas and 37 inserted leaf allocas. Image processing, voxel, and Spacex
report zero and produce unchanged JIT code. The compiler runs the shared,
target-independent XIR SROA before both `mem2reg` stages; a same-binary
`LUISA_SIMD_DISABLE_AGGREGATE_PROMOTION=1` oracle restores the prior layout.
This is a bounded local AoS-to-field/SoA conversion, not an external resource
layout change or general tile transpose.

Ten alternating W8 ordinary-path pairs at 128 spp measured a 1.3165x promoted/
disabled geometric mean with 10/10 wins; medians were 77.535 and 58.383 spp/s.
Ten W8 cutout pairs at 64 spp measured 1.1469x with 10/10 wins; medians were
43.719 and 38.229 spp/s. Within each workload all 20 outputs were byte-identical
between modes and passed the gallery reference.

The exact W8 ordinary-path JIT object identifies the mechanism:

| Main kernel | promotion disabled | promotion enabled |
| --- | ---: | ---: |
| `.text` bytes | 22,815 | 20,711 |
| static instructions | 3,885 | 3,586 |
| static branches | 237 | 234 |
| stack references | 1,050 | 1,024 |
| stack allocation | 9,728 B | 7,168 B |
| `vgatherqps` / `vscatterqps` | 65 / 55 | 18 / 6 |
| calls / scalar-math calls | 5 / 0 | 5 / 0 |

Three alternating 256-spp `perf stat` pairs measured enabled/disabled mean
ratios of 0.7614 for cycles, 0.9753 for instructions, 0.9993 for branches,
0.9970 for branch misses, 0.8148 for L1 data loads, and 0.5867 for L1
data-load misses. The large cache/load reduction with nearly unchanged branch
count distinguishes this result from dispatcher surgery: the former Ray/Onb
aggregate round trips became independently promotable fields. A separate
exact-PC profile put the shared scheduler dispatcher at only 0.131% of total
process cycles; simple continuation-stealing variants were measured and
rejected because their extra routing was neutral on ordinary path tracing and
regressed SDF.

## Scheduler cost and assembly evidence

The coherent direct-CFG proof accepts a function only when it has no
convergence point and all branch/switch selectors are warp- or cohort-uniform.
It emits ordinary LLVM control flow, keeps cohort values scalar, preserves the
initial inactive-tail mask, and allocates no ready queue or convergence frame.

For the exact DSL GEMM, optimized assembly changes as follows:

| Path | static instructions | static branches | stack-reference instructions |
| --- | ---: | ---: | ---: |
| old scheduled W8 | 753 | 73 | 145 |
| direct W1 | 80 | 5 | 0 |
| direct W2 | 122 | 14 | 0 |
| direct W4 | 73 | 2 | 0 |
| direct W8 | 74 | 2 | 0 |
| direct W16 | 74 | 2 | 0 |
| ISPC AVX2 i32x8 control | 62 | 4 | 0 |

Before direct lowering, an equal-work W8 counter audit retired about 101.2
billion instructions and 9.92 billion branches, versus 6.64 billion and 0.264
billion for ISPC. Both incurred about 3.0 billion L1 load misses. Scheduler
instructions and hot state loads, rather than DRAM or failed LLVM vector
selection, were the dominant gap. The direct W8 body is now close to the ISPC
code shape and contains packed arithmetic, masked contiguous memory, no stack
references, and no scalar-libm lane loop.

The remaining divergent path-tracing deficit has a different signature. One
16-SPP W8/fallback `perf stat` pair measured:

| Counter | fallback | W8 | W8/fallback |
| --- | ---: | ---: | ---: |
| cycles | 36.234 B | 72.389 B | 1.998x |
| instructions | 70.574 B | 106.850 B | 1.514x |
| branches | 6.273 B | 13.193 B | 2.103x |
| branch misses | 55.35 M | 230.88 M | 4.171x |
| L1 data loads | 27.796 B | 55.662 B | 2.002x |
| L1 data-load misses | 161.34 M | 2.183 B | 13.531x |
| last-level cache misses | 39.35 M | 47.39 M | 1.204x |

The ray-tracing main kernel reports `direct_control_flow=false`; its four
setup/accumulation kernels report true. The large L1-miss amplification with
nearly unchanged last-level misses is consistent with scheduler/frame state
cycling through L1/L2, not a DRAM bandwidth wall. Object and runtime audits
confirm W8 calls `rtcIntersect8`/`rtcOccluded8` once per packet and that Embree
advertises native W8 support.

Packet-density instrumentation explains why native traversal alone does not
make this renderer faster than fallback. At W8 and 16 spp, direct closest-hit
issued 8,517,551 packets with a mean 3.992 active lanes (49.90% utilization):
31.88% were singleton and only 25.77% were full. Occlusion issued 7,979,581
packets with a mean 3.772 active lanes (47.15% utilization): 34.35% singleton
and 26.36% full. The counters were an environment-gated diagnostic experiment
and are not present in production. Bounce divergence, not a missing W8 Embree
entry point, is therefore the dominant utilization limit.

### In-place direct-trace packet ABI

The old callback received eight ray vectors, visibility, optional time, and
separate hit scratch. It constructed a second Embree packet, traversed, and
copied instance/primitive/u/v/t back. The JIT now constructs Embree's public
component order directly and the W1/W4/W8/W16 runtime passes that aligned
scratch in place; W2 alone performs its required pad-to-W4 conversion.
Compile-time layout assertions fail the build if the configured Embree headers
do not match. A permanent callback probe checks sparse tails at
W1/W2/W4/W8/W16 and a non-contiguous W8 `0x55` cohort, including inputs,
inactive sanitization, and returned closest/any fields. The runtime accel and
curve gates exercise real Embree.

Isolated old/new backend modules ran ordinary path tracing at 128 spp and one
spp per dispatch, with alternating order and no removed sample. Output files
were byte-identical within every width:

| Width | Pairs | New/old paired geometric mean | Wins | Median ratio |
| --- | ---: | ---: | ---: | ---: |
| W1 | 8 | 1.0450x | 8/8 | 1.0470x |
| W2 | 8 | 1.0435x | 8/8 | 1.0530x |
| W4 | 8 | 1.0293x | 8/8 | 1.0352x |
| W8 | 10 | 1.0221x | 10/10 | 1.0265x |
| W16 | 8 | 1.0263x | 8/8 | 1.0271x |

Three 256-spp W8 `perf stat` repetitions measured candidate/baseline mean
ratios of 0.9710 for cycles, 0.9661 for instructions, 0.9510 for branches,
0.9965 for branch misses, 0.9546 for L1 data loads, and 0.8393 for L1
data-load misses. The JIT main-kernel stack grows from 9,472 to 9,728 bytes
because it now owns the complete public hit packet, but its final assembly
still contains vector arithmetic and no scalar math symbol; runtime object
disassembly shows direct native-width Embree calls without the old packet
initialization/copy path.

The follow-up callback ABI removes its final packed-mask round trip. Luisa
direct trace has no application-visible Embree ray ID, so LLVM sign-extends the
cohort mask into the packet's `ray.id` field and the runtime passes that field
as Embree's `valid` array. Ray queries keep their existing physical lane IDs.
Across five 256-spp `perf stat` pairs, retired instructions changed by stable
candidate/baseline ratios of 0.9907 at W8 and 0.9875 at W16; branches changed
by 0.9903 and 0.9928. The W8 main-kernel stack remains 9,728 bytes with 1,050
stack references, and the final callback object bodies shrink by 51% for
any-hit and 32% for closest-hit. Host load rose above 25 and corrupted several
wall-clock samples, so that checkpoint claimed no additional throughput
uplift. Its then-current fallback-relative row was left unchanged; the
aggregate-promotion sweep at the top of this report now supersedes it.

The then-final eight-process rotating sweep at 128 spp, again forcing one spp per
dispatch for every backend, measured medians of 73.286/62.123/44.564/53.265/
58.537/57.814 spp/s for fallback/W1/W2/W4/W8/W16. Paired geometric-mean
SIMD/fallback throughputs were 0.844x/0.610x/0.726x/0.802x/0.798x. The host
load average rose to about 19 by the end, so the paired measurements are the
preferred ratios. The optimization was real and repeatable, but did not erase
the renderer's divergence-driven fallback deficit at that checkpoint. These
numbers are retained as provenance for the callback stage and are superseded
by the current table.

### Curve-free direct-trace postprocessing

Direct closest-hit traversal historically scanned every active result after
Embree returned, loaded the hit instance's geometry kind, and changed
`bary.y` only for a round curve. A build-time accel summary now records whether
the current instance table contains any curve. Triangle/procedural-only scenes
skip the entire per-lane scan; curve scenes retain the exact old path. The
summary is recomputed after every normal accel build, including instance
replacement and shrink, rather than maintained as a one-way incremental bit.
A permanent W1/W2/W4/W8/W16 regression replaces one instance
`mesh -> curve -> mesh` and checks direct-hit classification after each build.

Isolated old/new backend modules ran the ordinary renderer at 128 spp and one
spp per dispatch. No outlier was removed:

| Width | Pairs | New/old paired geometric mean | Wins | Median ratio |
| --- | ---: | ---: | ---: | ---: |
| W1 | 8 | 1.0063x | 6/8 | 1.0083x |
| W2 | 8 | 0.9974x | 5/8 | 1.0064x |
| W4 | 8 | 1.0102x | 7/8 | 1.0168x |
| W8 | 10 | 1.0163x | 7/10 | 1.0143x |
| W16 | 8 | 1.0196x | 8/8 | 1.0186x |

Three longer W8 `perf stat` pairs measured a 1.0302x throughput geometric
mean. Candidate/baseline ratios were 0.9680--0.9832 for cycles, about 0.9889
for instructions, about 0.9547 for branches, 0.7349--0.7496 for branch
misses, about 0.9974 for L1 data loads, and 0.9146--0.9694 for L1 data-load
misses. A tempting removal of the zero initialization for temporary Embree ray
packets was rejected separately: it changed machine code but ten W8 pairs
measured 0.9952x with only 4/10 wins. That experiment predates the in-place ABI:
native widths no longer materialize a temporary packet, while W2 still clears
its padded W4 storage and the JIT initializes every public packet field.

Fresh 1024-spp ordinary path-tracing comparisons passed at W1/W2/W4/W8/W16
with RGB PSNR 35.43/42.78/40.94/39.22/37.80 dB respectively. The gallery
reference was read-only. These correctness runs and the fixed-dispatch
performance row use the new shared positive-integer
`--max-spp-per-dispatch` example option; default rendering policy is unchanged.

The W8 cutout main kernel contains two sequential ray-query construction sites.
A fail-closed Schedule-IR liveness/interference analysis now colors them into
one per-lane scratch slot; overlapping query objects remain in distinct slots.
This changed the then-current audit assembly as follows. These paired counts
remain a valid code-shape delta, but predate the exact ORC code-model
synchronization described below and are not live-object address maps:

| W8 cutout main kernel | query scratch | stack allocation | instructions | stack references | calls |
| --- | ---: | ---: | ---: | ---: | ---: |
| coloring disabled | 19,456 B | 38,976 B | 6,794 | 1,572 | 5 |
| coloring enabled | 9,728 B | 27,136 B | 6,745 | 1,551 | 5 |

Neither form contains a scalar math call. Across 25 alternating 64-SPP process
pairs, enabled/disabled medians are 34.691/34.506 spp/s (+0.53%). Five-repeat
counter samples measured 6.967 versus 8.586 billion L1 data-load misses
(-18.9%), 280.35 versus 282.82 billion cycles (-0.87%), and 402.06 versus
403.14 billion instructions (-0.27%). The small throughput change despite the
large L1-miss reduction confirms that query allocation was one pressure source,
not the whole divergent-scheduler deficit.

### Exact JIT-object and branch-flow audit

The diagnostic path now captures the exact relocatable object produced by
ORC's compiler alongside an annotated assembly clone. Both explicitly use the
same PIC/small code model. For the current W8 cutout main kernel, the `.text`
section at the profiler checkpoint was 36,856 bytes; assembling the annotated
`.s` produced the same text size and agreed at every basic-block offset.
Padding bytes may differ without changing layout. That report counted 6,456
instructions, 3,902 vector instructions, 506 branches, 1,556 stack references,
and a 27,136-byte stack allocation. The object had no undefined scalar-libm
symbol.

This corrected an earlier profiler attribution error. An independently emitted
large-code-model assembly was 38,532 bytes, so runtime PCs did not identify the
same semantic blocks. In particular, a skid-prone `cycles` profile appeared to
place fast trigonometry's large-range reduction on the hot path even though a
temporary fail-fast probe proved that block was not entered by this workload.
Cycle sampling alone is therefore not used for exact basic-block attribution.

An AMD last-branch-record profile of one W8 64-spp cutout run supplied 320,828
main-kernel user branch edges. After mapping those exact edge endpoints through
the captured object and annotated offsets, the retired branch-flow distribution
was:

| Main-kernel branch category | branch-edge share |
| --- | ---: |
| convergence cascades | 43.07% |
| kernel schedule blocks | 11.10% |
| ray-query state/control | 10.79% |
| dynamic coherence tests | 8.98% |
| mask-stack overflow logic | 8.75% |
| scheduler dispatch/loop | 7.51% |
| unlabeled LLVM blocks | 5.44% |
| fast-math exits | 2.05% |
| convergence target | 1.46% |
| acceleration setup/control | 0.86% |

These are shares of recorded retired branch edges, not shares of cycles or wall
time. They nevertheless quantify substantial state-machine control traffic:
the four hottest individual convergence cascades each contribute roughly
5.5--5.7% of the recorded edges, and `scheduler.dispatch` contributes 4.78%.

Two narrow cascade simplifications were rejected. Bounding a cascade by the
number of static convergence points sharing a target is unsound because loop
iterations can create a dynamic `A -> B -> A` frame chain. Removing the
explicit depth counter is valid for a well-formed state and cut static branches
from 506 to 491, but ten paired path-tracing runs measured a 0.9969x geometric
mean with only two wins, so it was reverted. The next useful scheduler change
must reduce frame/state traffic rather than merely shrink this branch chain.

The first accepted follow-up changes the immutable convergence-static-ID target
map at W4/W8/W16 from an LLVM vector to a private constant array. On x86, a
dynamic vector extract had materialized the complete constant vector on the
stack in every destination cascade. The array form hoists one RIP-relative
base and performs an indexed scalar load. The current W8 main object changes
as follows:

| W8 cutout main kernel | before | constant target array |
| --- | ---: | ---: |
| `.text` bytes | 36,856 | 35,956 |
| static instructions | 6,456 | 6,367 |
| vector instructions | 3,902 | 3,824 |
| static branches | 506 | 506 |
| stack references | 1,556 | 1,487 |
| stack allocation | 27,136 B | 24,192 B |

Ten alternating 128-spp pairs, with one spp per dispatch and reference-image
validation on every process, measured 1.0102x at W4 (8/10 wins), 1.0058x at
W8 (9/10), and 1.0021x at W16 (7/10). The final object still has no undefined
scalar-libm symbol. A separate ten-pair 64-spp W2 array experiment measured
0.9864x with 6/10 wins under shared-host load, so W1/W2 retain their prior
vector lowering rather than inheriting an unproven wide-width heuristic.

Two more register/state-layout experiments were rejected. Packing static ID
and parent token reduced W8 instructions and stack references but measured
0.9969x with 3/10 wins. Storing those fields as scalar arrays cut the accepted
target-array object's instruction count from 6,367 to 6,134 and its stack from
24,192 to 21,440 bytes, yet ten W8 pairs were 0.9999x with 5/10 wins. This is
direct evidence that smaller assembly or fewer L1-resident stack accesses are
not sufficient acceptance criteria; both experimental changes were reverted.

## Runtime-sparse ray-query cohorts

A W8 `perf record` of the 64-spp cutout renderer places 9.14% of sampled
cycles in `_ray_query_proceed`, 5.29% in the argument filter, and 4.22% in
candidate-batch installation. A temporary counter build, removed after the
audit, observed 81.25 million proceed calls and 294.20 million active lanes:
only 3.62 of eight lanes were active on average. Pending scans covered 78.0%
of active lanes, while only 4.6% could publish an already cached candidate.
The candidate distribution also rules out reducing the 32-entry batch: 99.5%
of surface scan-lanes had at most four candidates here, but permanent
40-candidate continuation tests require the existing capacity and overflow
path.

W8 and W16 now select an adaptive host callback at JIT specialization time;
W1/W2/W4 retain the original callback. The adaptive callback uses a dense
fall-through loop when the runtime mask is full and iterates set bits for a
sparse mask. Its packet scan likewise initializes and installs only active
records while still issuing exactly one native `rtcIntersect8/16` or
`rtcOccluded8/16` call. The choice between callbacks is static for a compiled
width, not another varying function-pointer select in each `PROCEED`.

Keeping the providers separate matters for code generation. The original
callback remains 11,881 bytes with the baseline `0x1100` stack frame, versus
11,876 bytes before the change. Two helpers are explicitly force-inlined to
keep that shape stable. A permanent LLVM boundary test supplies distinct
providers and proves W4 selects the original callback while W8 selects the
adaptive one.

The final rejection-kernel assembly reports are 1,368 instructions, 673
vector instructions, and 7,200 bytes of stack at W4; 1,473, 709, and 14,336
bytes at W8; and 2,273, 1,251, and 28,288 bytes at W16. Each specialization
has one callback callsite and zero scalar-math callsites. The runtime-object
disassembly contains `tzcnt` set-bit iteration in the adaptive callback and
native `rtcIntersect8/16` plus `rtcOccluded8/16` callsites; W4 remains on the
original provider and its native packet entrypoints.

Ten alternating W8 cutout process pairs have a paired geometric-mean speedup
of 1.0387x with 9/10 wins; independent medians move from 34.151 to 35.816
spp/s (1.0488x), and every image passes the gallery comparison. This includes
two opposite-side host-load outliers rather than deleting them. Ten W16 pairs
all improve, by a 1.0519x paired geometric mean; independent medians move from
32.785 to 34.525 spp/s (1.0531x). A twelve-pair full-cohort 16-candidate W8
gate is neutral-positive at 1.0024x (8/12 wins). W1/W2/W4 ten-pair gates are
1.0001x, 1.0036x, and 1.0071x respectively. Five final-binary W8 pairs of
the real procedural-callable renderer improve by a 1.0090x geometric mean
with 4/5 wins; all five gallery comparisons pass.

The next `perf annotate` pass localized about forty percent of the surface
filter's own samples to its physical-lane `valid == -1` check, skip branch,
and loop backedge. The accepted W8/W16 filter first compares the fixed Embree
valid array into a small integer bit mask and then visits only set bits with
`countr_zero`/clear-lowest-bit. It remains target-independent C++ and retains
the exact packet width; inactive state pointers are never dereferenced. The
same callback handles sparse valid masks produced from an initially full
cohort, while W1/W2/W4 still use the original dense filter.

This filter is isolated in an append-only translation unit. A shared
standard-layout context base plus a pointer-interconvertible empty derived
type avoids unrelated-struct aliasing. On GCC only, that source disables hot/
cold block partitioning: a measured 65-byte `.text.unlikely` clone otherwise
shifted all established narrow callbacks and produced a repeatable narrow
layout regression. The final narrow filter, batch installer, and proceed
callback have the same addresses, sizes, stack frames, and normalized control
flow as the preceding binary; `.init_array` also remains 104 bytes.

With this final layout, ten alternating W8 64-spp cutout pairs all improve;
the paired geometric-mean speedup is 1.0143x and medians move from 36.196 to
36.591 spp/s. Ten W16 pairs also all improve, by 1.0202x; medians move from
34.212 to 34.688 spp/s. W2 ten-pair and W4 thirty-pair dense rejection gates
are neutral at 1.0046x and 0.9990x. Fifteen W8 procedural-callable pairs are
neutral at 0.9978x, and all reference comparisons pass. An alternative that
let Embree accept opaque closest-query hits while also batching them was
rejected: ten W8 path pairs measured 0.9877x with only 2/10 wins.

### Lazy ray-query batch metadata

The two batch `initialized` fields are the only gates read before the first
scan. W1/W4/W8/W16 query construction now clears only those gates; the first
scan clears count, index, and continuation for both surface and procedural
batches before publishing the gates. W2 remains eager. A same-binary oracle,
`LUISA_SIMD_DISABLE_RAY_QUERY_LAZY_BATCH_INIT=1`, restores all six redundant
construction stores. The exact LLVM fixture locks lazy/unpacked versus
eager/unpacked construction at 31/37 masked-scatter callsites and covers
W1/W2/W4/W8/W16 plus a three-lane W16 tail. The later packed specialization
reduces W4/W8/W16 construction to 26 callsites.

The exact W8 cutout object changes as follows:

| W8 main kernel | eager | lazy |
| --- | ---: | ---: |
| instructions | 6,367 | 6,319 |
| vector instructions | 3,824 | 3,776 |
| stack references | 1,487 | 1,469 |
| stack allocation | 24,192 B | 23,808 B |
| branches / calls | 506 / 5 | 506 / 5 |

Alternating 64-SPP, same-binary cutout processes measured the following paired
geometric means. No outlier was removed:

| Width | Pairs | Lazy/eager | Wins |
| --- | ---: | ---: | ---: |
| W1 | 6 | 1.0144x | 6/6 |
| W2 | 16 | 0.9972x | 12/16 |
| W4 | 6 | 1.0248x | 6/6 |
| W8 | 10 | 1.0294x | 10/10 |
| W16 | 6 | 1.0350x | 6/6 |

W2's padded-W4 path therefore keeps eager initialization. A separate ten-pair
W8 16-candidate rejection chain is directionally positive at 1.0068x with
6/10 wins; it is a non-regression stress gate, not the acceptance result.

A 192-byte-hot/1024-byte-cold split of the existing 1216-byte per-lane query
record was also rejected. Ten W8 rejection-chain pairs measured 1.0025x with
5/10 wins and ten W8 cutout pairs measured 1.0015x with 6/10 wins. The split
object had 6,381 rather than 6,379 instructions and one extra stack reference,
while retaining the same 24,320-byte frame: the extra cold-batch pointer offset
the smaller hot stride. The production ABI remains the single AoS record; a
future SoA experiment must change the packet/state crossing rather than merely
partitioning the same fields.

### Packed ray-query initialization

After lazy batch initialization, five pairs of adjacent fields still carried
identical all-zero or all-one bit patterns. W4/W8/W16 now issue one 64-bit
masked scatter per pair; W1/W2 keep the unpacked 32-bit form. Static ABI
assertions lock every participating offset, and the two potentially unaligned
hot-state pairs truthfully declare four-byte alignment. The committed-hit
pairs are eight-byte aligned. `LUISA_SIMD_DISABLE_RAY_QUERY_PACKED_INIT=1`
restores the unpacked same-binary oracle.

The exact W8 cutout object changes incrementally from lazy/unpacked to
lazy/packed:

| W8 main kernel | unpacked | packed |
| --- | ---: | ---: |
| instructions | 6,319 | 6,281 |
| vector instructions | 3,776 | 3,738 |
| stack references | 1,469 | 1,454 |
| stack allocation | 23,808 B | 23,488 B |
| branches / calls | 506 / 5 | 506 / 5 |

Alternating same-binary 64-SPP cutout processes measured:

| Width | Pairs | Packed/unpacked | Wins | Decision |
| --- | ---: | ---: | ---: | --- |
| W1 | 6 | 0.9994x | 4/6 | keep unpacked |
| W2 | 6 | 0.9877x | 3/6 | keep unpacked |
| W4 | 6 | 1.0185x | 5/6 | enable packed |
| W8 | 10 | 1.0186x | 10/10 | enable packed |
| W16 | 6 | 1.0356x | 6/6 | enable packed |

No outlier was removed. The exact LLVM fixture covers every width, the eager
batch oracle, the unpacked oracle, and a three-lane W16 tail; it counts masked-
scatter callsites rather than intrinsic declarations.

### Triangle-only ray-query runtime

An acceleration structure whose current instance summary contains neither a
curve nor a procedural primitive now selects a separate surface-only query
provider. The summary is recomputed after every build, including motion-child
classification and instance replacement, so one compiled shader can safely
observe `mesh -> procedural -> mesh` rebuilds. The generic provider remains the
only path for mixed, curve, and procedural scenes. The same-binary oracle
`LUISA_SIMD_DISABLE_TRIANGLE_ONLY_RAY_QUERY=1` is sampled when the accel is
created and restores the generic provider without changing JIT code.

The specialized provider preserves the 1216-byte public query-state ABI and
the same W1/W2/W4/W8/W16 Embree mapping. It does not clear, sort, advance, or
test the procedural batch, does not load geometry kind or perform curve
deduplication in the surface filter, and uses a surface-only scan context. It
still rejects every physical Embree candidate into the bounded ordered batch,
retains overflow continuation and cursor ordering, auto-commits opaque
instances, groups by accel/query-any mode, and sanitizes inactive tails before
one packet traversal. W8/W16 retain the sparse set-bit callback; W2 still pads
to one W4 packet. Runtime disassembly contains native `rtcIntersect4/8/16` and
`rtcOccluded4/8/16` callsites plus W1 scalar callsites; it contains no
per-active-lane scalar traversal loop.

Twelve final-binary alternating W8 cutout pairs at 64 spp, with one spp per
dispatch, measured a 1.0069x geometric-mean enabled/disabled speedup and 9/12
wins; independent medians were 43.713 and 43.256 spp/s. Four-pair W1/W2/W4
gates measured 1.0511x/1.0289x/1.0157x, all with 4/4 wins. Six W16 pairs
measured 1.0067x with 4/6 wins. Every output passed the gallery comparison and
every enabled/disabled W8 image had the same SHA-256
`ad97b2a0e41cab86019e7def16f0bd8d63007e640eb4a468d2b71c09a9e74eda`.

Three alternating 128-spp W8 `perf stat` pairs were less noisy than wall time.
Enabled/disabled mean ratios were 0.9893 for cycles, 0.9746 for instructions,
0.9345 for branches, 0.9853 for branch misses, 0.9837 for L1 data loads, and
0.8473 for L1 data-load misses; all three cycle/instruction/branch/load ratios
moved in the expected direction. Ordinary direct-trace path tracing never
calls this provider: five JIT objects, five assembly files, and the output PNG
were byte-identical with the oracle toggled. Six 128-spp W8 pairs were neutral
at 1.0056x with 4/6 wins.

The public 16-candidate rejection-chain benchmark gives the complementary
fallback-relative result. Five independent, alternating processes per width
each took the median of seven samples of 2,097,152 rays and validated the exact
far hit and callback count:

| Backend/width | Median Mray/s | Paired geometric mean vs fallback | Wins |
| --- | ---: | ---: | ---: |
| fallback | 31.6226 | 1.0000x | -- |
| SIMD W1 | 26.6705 | 0.8182x | 0/5 |
| SIMD W2 | 21.2639 | 0.6551x | 0/5 |
| SIMD W4 | 24.8335 | 0.7609x | 0/5 |
| SIMD W8 | 27.9990 | 0.8522x | 0/5 |
| SIMD W16 | 28.0799 | 0.8576x | 0/5 |

The fallback process medians drifted from 36.3795 to 30.0180 Mray/s as other
host work changed, so the paired geometric means are authoritative. The
surface-only provider narrows the gap but does not make dense candidate
rejection faster than fallback; its Embree filter, bounded-batch installation,
and JIT query-state crossings remain measurable costs.

The specialization lives in an append-only translation unit and GCC disables
hot/cold block partitioning for it. In the GCC Release A/B build used for these
measurements, the established generic wide/narrow proceed symbols remain
exactly `0x3433`/`0x2e38` bytes and the generic wide filter remains `0x102a`,
matching the isolated pre-change module built by the same compiler. A rejected
shared-template implementation perturbed generic code layout and measured
0.9907x at W4 and 0.9962x at W8 on the 16-candidate procedural benchmark. The
isolated final layout instead measured 0.9981x across five fresh W8 pairs, so
no procedural throughput claim is made and preserving generic compiler/layout
shape within a matched A/B build is a permanent review constraint.

### Packed ray-query status sidecar

The generated query loop historically reloaded `terminated` and candidate kind
from the 1216-byte lane-private AoS state after every host `PROCEED`. W4/W8/W16
now retain those three predicates in one JIT-owned 64-bit sidecar when
ownership, aliasing, construction-store order, and scratch-color interference
are all proven. W1/W2, disabled scratch coloring, and every unproven query keep
the authoritative gather path. The same-binary oracle is
`LUISA_SIMD_DISABLE_RAY_QUERY_STATUS_CACHE=1`.

The final exact W8 16-candidate rejection object shows the intended tradeoff:

| Main kernel | status disabled | status enabled |
| --- | ---: | ---: |
| status colors | 0 | 1 |
| `.text` bytes | 6,134 | 6,266 |
| static instructions | 1,292 | 1,324 |
| vector instructions | 577 | 572 |
| branches / calls | 128 / 1 | 132 / 1 |
| stack references | 269 | 281 |
| stack allocation | 12,672 B | 12,736 B |
| gather / scatter instructions | 13 / 26 | 8 / 26 |
| scalar-math calls / undefined symbols | 0 / 0 | 0 / 0 |

Thus this is a latency optimization, not an instruction-count optimization:
five gathers disappear while 32 scalar/vector bookkeeping instructions are
added. Three alternating W8 `perf stat` pairs measured enabled/disabled
geometric-mean ratios of 0.9721 for cycles, 1.0329 for instructions, 1.0335
for branches, and 1.0036 for branch misses. The cache events were multiplexed
and are not used as an acceptance claim. Disassembly also caught and rejected
an initial assertion implementation that pulled logging/backtrace construction
into the wrapper; cold no-inline error helpers leave the accepted hot status
entry at 237 bytes with a 40-byte frame. The existing generic and
triangle-only providers are unchanged.

Seven alternating same-binary process pairs per width measured the following
final status-enabled/disabled throughput. No outlier was removed:

| Query benchmark | Width | Enabled/disabled | Wins | Median enabled / disabled |
| --- | ---: | ---: | ---: | ---: |
| 16 rejected triangle candidates | W4 | 1.0253x | 7/7 | 26.726 / 26.199 Mray/s |
| 16 rejected triangle candidates | W8 | 1.0201x | 7/7 | 30.304 / 29.602 Mray/s |
| 16 rejected triangle candidates | W16 | 1.0026x | 6/7 | 29.590 / 29.477 Mray/s |
| 16 rejected procedural candidates | W4 | 1.0437x | 7/7 | 63.896 / 61.401 Mray/s |
| 16 rejected procedural candidates | W8 | 1.0459x | 7/7 | 92.255 / 87.674 Mray/s |
| 16 rejected procedural candidates | W16 | 1.0216x | 7/7 | 106.710 / 104.668 Mray/s |

The real W8 cutout renderer at 64 spp and one spp per dispatch measured 1.0073x
across seven alternating pairs with 6/7 wins; medians were 44.196 and 43.883
spp/s. The 1024-spp procedural-callable renderer measured 1.0199x across five
pairs with 4/5 wins; medians were 3,151.4 and 3,247.6 ms. All 24 renderer
outputs passed their gallery comparisons. Ordinary direct-trace path tracing
allocates zero status colors: all five JIT objects and their assembly are
byte-identical with the oracle toggled, and the output comparison passes.

At the status-sidecar checkpoint, the public triangle rejection sweep remained
below fallback despite that incremental win. Five alternating pairs gave
W1/W2/W4/W8/W16 ratios of
0.7709x/0.6318x/0.7601x/0.8476x/0.8532x. W1 and W2 deliberately generate
byte-identical objects with the status oracle on or off. This result keeps
sparse cohort compaction and a more structural query-state SoA/register
layout, rather than further predicate-cache bookkeeping, as the next major
ray-query targets.

### Proven ray-query state-handle cache

Even with cached status, each query operation previously gathered the state
pointer packet back from the query local. Eligible W4/W8/W16 queries now keep
one fixed-vector state-handle packet per status color. The ordinary masked
local store remains authoritative and publishes the same active lanes to the
cache only afterward. Loads validate every active cached pointer; inactive
lanes are nulled before the host callback. This reuses the status proof for
ownership, construction order, aliasing, and color interference. W1/W2 and
every fail-closed path are unchanged. The independent same-binary oracle is
`LUISA_SIMD_DISABLE_RAY_QUERY_STATE_HANDLE_CACHE=1`.

The exact final W8 16-candidate triangle rejection object changes as follows:

| Main kernel | handle cache disabled | handle cache enabled |
| --- | ---: | ---: |
| handle-cache colors | 0 | 1 |
| `.text` bytes | 6,266 | 6,208 |
| static instructions | 1,324 | 1,316 |
| vector instructions | 572 | 563 |
| branches / calls | 132 / 1 | 133 / 1 |
| stack references | 281 | 284 |
| stack allocation | 12,736 B | 12,800 B |
| gather / scatter instructions | 8 / 26 | 4 / 26 |
| scalar-math calls / undefined symbols | 0 / 0 | 0 / 0 |

The extra 64-byte packet removes four pointer gathers and eight static
instructions. Five `perf stat` repetitions on the final explicitly aligned IR
measured 77.854/79.389 billion cycles and 213.880/214.180 billion instructions
enabled/disabled: ratios of 0.9807 and 0.9986. Branch count was effectively
unchanged at 0.99996; branch misses were 1.0064x but only about 0.01% of retired
branches. This points to gather latency, not scheduler branching, as the source
of the wall-time win.

Seven alternating same-binary pairs per width measured these incremental
enabled/disabled results; no outlier was removed:

| Query benchmark | W4 | W8 | W16 |
| --- | ---: | ---: | ---: |
| 16 rejected triangle candidates | 1.0174x (6/7) | 1.0170x (7/7) | 1.0104x (6/7) |
| 16 rejected procedural candidates | 1.0514x (7/7) | 1.0639x (7/7) | 1.0438x (7/7) |

The real W8 cutout renderer measured 1.0242x across seven 64-spp pairs with
7/7 wins. The 1024-spp procedural-callable renderer measured 1.0188x across
five pairs with 5/5 wins; geometric-mean times were 3,127.9 and 3,186.7 ms.
A separate 1024-spp cutout output passed at 44.17 dB RGB PSNR and procedural
callable passed at 58.37 dB. W1/W2 query objects, all five ordinary-path-tracer
objects at W8, and non-query GEMM objects at W4/W8/W16 are byte-identical with
the oracle toggled.

A more aggressive candidate-payload SoA experiment was rejected. Copying the
AoS payload into a second packet after the host provider removed the candidate
gather but added another active-lane scan at the ABI wrapper: W8 triangle was
only 1.0067x (5/7), procedural was 0.996x (3/7), and cutout was about 0.997x
(1/5). A future payload split must be populated directly by the provider or a
new packet ABI; duplicating the scan is not retained.

The refreshed final-binary 16-candidate triangle-query sweep against fallback
is W1/W2/W4/W8/W16 = 0.7753x/0.6283x/0.7560x/0.8458x/0.8504x across five
adjacent alternating pairs per width. The cache therefore gives a repeatable
incremental improvement but does not close the public query gap.

### Paired ray-query status callback

With a proven status color, construction stores both the plain provider in the
authoritative query state and its paired status entry in the JIT sidecar. The
status entry invokes the plain provider; every production provider validates
that every active state still carries that exact plain callback. JIT therefore
only needs to verify status-entry agreement for the active cohort instead of
gathering and comparing the same plain pointers again. W1/W2, disabled status
caching/coloring, and every unproven query keep both checks. The same-binary
oracle is `LUISA_SIMD_DISABLE_RAY_QUERY_STATUS_CALLBACK_PAIRING=1`.

The exact W8 16-candidate triangle rejection object changes incrementally from
the state-handle-cache checkpoint:

| Main kernel | pairing disabled | pairing enabled |
| --- | ---: | ---: |
| `.text` bytes | 6,208 | 6,144 |
| static instructions | 1,316 | 1,309 |
| vector instructions | 563 | 557 |
| branches / calls | 133 / 1 | 132 / 1 |
| stack references | 284 | 282 |
| stack allocation | 12,800 B | 12,736 B |
| `vpgather` / `vscatter` | 3 / 10 | 2 / 10 |
| scalar-math calls / undefined symbols | 0 / 0 | 0 / 0 |

Five alternating `perf stat` pairs measured enabled/disabled geometric-mean
ratios of 0.9931 for cycles, 0.9989 for retired instructions, and 0.9986 for
branches. Branch misses were 1.0294x, but represented about 0.09% of branches
and did not offset the cycle reduction. Disassembly identifies the removed
instruction as the plain-callback `vpgatherqq`; the status-entry indirect call
remains exactly once.

Seven final-binary alternating pairs per width measured:

| Query benchmark | W4 | W8 | W16 |
| --- | ---: | ---: | ---: |
| 16 rejected triangle candidates | 1.0170x (6/7) | 1.0060x (6/7) | 1.0016x (4/7) |
| 16 rejected procedural candidates | 1.0181x (6/7) | 1.0265x (7/7) | 1.0290x (7/7) |

The real W8 cutout renderer is neutral at 1.0002x with 3/7 wins across seven
64-spp pairs; all fourteen images have the same reference hash. Mixed
procedural callable is positive across fourteen 1024-spp pairs at 1.0106x with
13/14 wins; all twenty-eight images are byte-identical. An attempted direct
provider-specific status wrapper was rejected: in a four-way ablation it was
1.0002x with 4/7 wins alone on the W8 procedural microbenchmark and did not
improve the paired path. The generic status ABI remains unchanged.

The IR fixture fixes W4/W8/W16 pairing-on gathers at zero for its status/handle
probe versus three with the pairing oracle, executes divergent cohorts and
inactive tails, and uses a fatal subprocess to reject mismatched plain
providers. W1/W2 query objects and W4/W8/W16 non-query GEMM assembly are byte-
identical under the oracle.

## Same-algorithm ISPC control and provenance

`benchmark_ispc_gemm.ispc` was independently written to match the DSL loop and
row-major storage: program instances cover consecutive output columns and the
inner K loop remains uniform. It is compiled with `-O2`, precise arithmetic,
FMA disabled, `--cpu=znver5`, and eight workers. Seven-sample medians are:

| ISPC target | Width | GFLOP/s |
| --- | ---: | ---: |
| `avx2-i32x4` | 4 | 93.170 |
| `avx2-i32x8` | 8 | 139.472 |
| `avx512skx-x4` | 4 | 92.101 |
| `avx512skx-x8` | 8 | 142.812 |
| `avx512skx-x16` | 16 | 223.911 |

The compiler is optional and supplied only through
`LUISA_COMPUTE_ISPC_EXECUTABLE`; no machine-local path enters source or tests.
No ISPC implementation, SLEEF implementation, or approximation coefficient is
copied into production. The benchmark tool provenance is official
[ISPC 1.31.0](https://github.com/ispc/ispc/releases/tag/v1.31.0), whose
[license](https://github.com/ispc/ispc/blob/main/LICENSE.txt) is BSD-3-Clause.

## Interpreting widths

W8 is a semantic fixed-vector width, not an AVX-512 contract. On this host LLVM
usually selects YMM arithmetic plus AVX-512VL masks and may use ZMM for eight
64-bit gather addresses. A host without AVX-512 may legalize W8 to AVX2. W16
uses ZMM here, but target legalization is allowed to split it elsewhere.

W1 is also not the fallback backend. It shares SIMD's Schedule ABI, ORC path,
resource callbacks, and runtime; fallback has a separate scalar code generator
that LLVM may horizontally vectorize. In GEMM, fallback vectorizes the inner K
reduction, while SIMD lanes vectorize adjacent output columns. Direct CFG
removes W1 scheduler overhead but cannot make the two compiler pipelines
identical.

## Next measured optimization targets

1. Extend the accepted local aggregate promotion to the remaining ray-query
   payload: publish candidate fields directly into provider-owned packet/SoA
   storage and rematerialize immutable fields across suspension, following the
   liveness/frame principles merged from `next`. A wrapper-side second scan is
   measured and rejected; the accepted state-handle cache covers pointers only.
2. Compact or rebatch sparse ray cohorts before Embree and reduce the remaining
   JIT-side ray-query state crossings. The accepted triangle-only host provider
   removes surface-runtime bookkeeping but does not compact lanes; inlining
   Embree LLVM IR is exploratory and cannot replace this measured scheduler
   work.
3. Move fixed-vector texture tap selection into JIT IR or introduce a measured
   tile/swizzle upload boundary. Preserve row-major public image semantics.
4. Generalize lane-affine recognition into bounded lane/value axis rotation
   only for coherent affine tiles; divergent control and warp operations pin
   lane identity.
5. Add software prefetch only for proven affine lookahead with a stable A/B.
   Immediate masked gathers and L1-sized textures have not justified it.

Device-side instance-opacity mutation is now complete but is intentionally not
reported as a throughput optimization. It adds no work to a trace/query kernel
that does not execute `set_instance_opaque`; when used, it lowers to one scalar
byte store for a uniform operation or one inactive-safe masked byte scatter for
a varying operation, without a host callback. The exact LLVM and W1/W2/W4/W8/
W16 runtime gates validate the capability. The measured ray-query targets above
remain unchanged. A fresh 1024-SPP cutout gallery sweep passed at W1/W2/W4/W8/
W16 with RGB PSNR 42.93/44.65/44.43/44.17/43.89 dB respectively. These are
correctness runs under shared-host load, not a replacement for the alternating
multi-process performance table.

## Bindless gradient-sampling completion

Bindless 2D/3D gradient sampling now derives LOD in JIT IR and passes one level
packet to the existing grouped texture callback. Base extents share the
existing 16-byte texture descriptor by packing three twenty-bit values beside
the sampler code, so ordinary Spacex sampling retains the same slot size and
callback ABI. Varying dependencies use fixed-vector math; the exact W8 JIT
assembly contains the native vector `log2` body and no `log2f` or vector-libm
symbol, and the runtime module adds no `log2f` dependency. A separate ORC
probe with varying coordinates but uniform slot/gradients/minimum LOD contains
one scalar `llvm.log2.f32`, no native vector body, and no extent gather: the
entire uniform LOD chain executes once and splats only at the callback ABI.

The repository `test_bindless_mip simd` now passes instead of failing at its
former capability boundary. A dedicated ORC probe checks the gradient ABI,
inactive-tail sanitization, one callback per varying packet, uniform one-lane
execution, fixed-vector native math, and final assembly symbols. Runtime
coverage spans W1/W2/W4/W8/W16, 2D/3D gradients, stored/explicit samplers,
minimum mip, uniform LOD with varying coordinates, zero/mixed-NaN/infinite
gradients, and a 35-thread W16 tail.

Ordinary non-gradient Spacex was checked with seven alternating before/after
W8 processes at 32 iterations each; every image passed the same gallery
reference. The last three warmed pairs were 50.495/50.488,
51.411/50.801, and 51.526/51.587 ms per frame (before/after), which is neutral
within shared-host noise. No performance gain is claimed for a feature the
workload does not exercise; the gate demonstrates that packing the new
metadata did not regress its hot descriptor layout.

## Validation

The required native-math/fallback-math/runtime-width gate passes 3/3. The
focused `unit_simd` and `integration_simd` gates pass 11/11 and 26/26,
including in-place packet codegen, accel, curve/procedural summary replacement,
local memory, atomics, bindless resources, and three graphics tests. After a
full Release build, the current configured repository CTest suite passes
140/140. The examples runner accepts its default backend matrix and explicit
backend lists, the C++ syntax-check script has 13 passing Python unit tests,
and clangd syntax checks pass for the status wrapper, emitter, and regression
fixture.
This also includes the coroutine-frame tests merged from `next`, the repaired
lazy-dispatch scalar snapshot regression, and the W1/W2/W4/W8/W16 aggregate-
promotion differential test. Separate 1024-SPP gallery gates pass ordinary
and cutout path tracing at all five widths, and non-coro SDF W8 passes at
63.13 dB RGB PSNR.

The state-handle-cache stage reran the required three-test native-math/runtime-
width gate, the accel/curve/procedural/world-ray focus gate (7/7), the complete
SIMD-only Release configuration (129/129, including `integration_simd` 26/26),
and the complete SIMD+fallback Release configuration (140/140). Its dedicated
W4/W8/W16 IR regression fixes the handle-cache gather count at three versus
six for the status-only oracle, while W1/W2/fail-closed paths retain thirteen;
it also executes divergent cohorts and inactive tails. Separate object gates
prove W1/W2 query kernels and W4/W8/W16 non-query GEMM kernels are byte-
identical under the independent handle-cache oracle.

The status-callback-pairing stage reran the required native-math/runtime-width
gate (3/3), the final SIMD-only Release suite (129/129, including
`integration_simd` 26/26 and graphics 3/3), and the final SIMD+fallback Release
suite (140/140). Its dedicated W4/W8/W16 IR oracle fixes the paired status/
handle probe at zero gathers versus three with pairing disabled, and its fatal
subprocess verifies that a mismatched plain provider is still rejected. Final
W8 1024-spp gallery gates pass at 44.17 dB for cutout and 58.37 dB for mixed
procedural callable.

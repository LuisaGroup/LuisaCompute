# SIMD CPU backend performance report

Snapshot date: 2026-08-13. This report covers the Release build after merging
`origin/next@62b77df36b6dae05aff558d4db84e415b5e84e75` into
`codex/simd-cpu-backend`, adding coherent direct-CFG lowering, completing the
bindless gradient-sampling vertical slice, and eliminating curve-hit
postprocessing from curve-free acceleration structures.

## Test host and method

- AMD Ryzen 9 9950X3D, 16 cores / 32 hardware threads;
- LLVM and Clang 22.1.8;
- Embree 4.4.1, reporting native W4/W8/W16 packet support;
- ISPC 1.31.0 for the optional same-algorithm control;
- `CMAKE_BUILD_TYPE=Release`;
- unrelated work was active, so every result uses alternating forward/reverse
  order and a median rather than a best run.

Graphics and SDF cells below are medians of seven independent processes.
Image processing repeats its four-dispatch pipeline 32 times, voxel repeats 16
renders, and Spacex renders four frames after its upload/update synchronization.
Cutout path tracing uses 64 spp and forces one spp per dispatch on both backends
to remove a batching asymmetry. Image, voxel, Spacex, and path tracing compare
every measured output with the repository gallery reference. SDF uses its
internal four-SPP throughput metric; high-SPP SDF image comparison remains a
separate conformance gate. The accepted runtime-sparse studies below supersede
the W8/W16 cutout cells with the latest ten-pair medians; other cells retain
the seven-process sweep.

Speedup is always `fallback time / SIMD time`, or
`SIMD throughput / fallback throughput`, so values above one are wins.

## Current fallback-relative results

| Workload and metric | fallback | W1 | W2 | W4 | W8 | W16 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| non-coro SDF, samples/s | 8.705 | 8.197 (0.942x) | 9.476 (1.089x) | 15.112 (1.736x) | 22.568 (2.593x) | 32.959 (3.786x) |
| image pipeline, ms/iteration | 8.379 | 17.184 (0.488x) | 9.169 (0.914x) | 6.493 (1.290x) | 4.992 (1.678x) | 4.249 (1.972x) |
| voxel render, ms/iteration | 6.904 | 8.127 (0.850x) | 24.122 (0.286x) | 16.128 (0.428x) | 9.386 (0.736x) | 6.479 (1.066x) |
| Spacex, ms/frame | 158.831 | 150.954 (1.052x) | 94.904 (1.674x) | 64.277 (2.471x) | 49.999 (3.177x) | 42.700 (3.720x) |
| ordinary path tracing, fixed 1 spp/dispatch, spp/s | 71.935 | 58.522 (0.814x) | 43.605 (0.606x) | 51.833 (0.721x) | 57.308 (0.797x) | 57.497 (0.799x) |
| cutout path tracing, spp/s | 68.570 | 44.870 (0.654x) | 28.400 (0.414x) | 32.982 (0.481x) | 36.591 (0.534x) | 34.688 (0.506x) |
| portable GEMM, GFLOP/s | 64.895 | 23.332 (0.360x) | 25.627 (0.395x) | 115.914 (1.786x) | 190.521 (2.936x) | 316.449 (4.876x) |

The GEMM row is a compute diagnostic rather than a graphics result. It uses
eight explicit SIMD workers and seven independent process medians; every
process performs seven timed samples of 128 complete 256-by-256 dispatches and
validates the output against double-precision accumulation. The fallback
process medians ranged from 41.594 to 88.326 GFLOP/s under shared-host load,
while the SIMD distributions were tight. Its relative speedups must therefore
be treated as host observations, not cross-machine constants.

The ordinary path-tracing row is the final eight-process sweep at 64 spp. It
uses the shared `--max-spp-per-dispatch 1` option on both backends so the row
measures width, divergent scheduling, resource callbacks, and Embree packets
without a dispatch-batching asymmetry. Its observed ranges were
70.046--73.815, 57.216--59.738, 41.082--44.248, 49.543--52.726,
55.802--58.504, and 55.914--58.484 spp/s from fallback through W16. Paired
geometric-mean speedups were 0.813x/0.601x/0.717x/0.797x/0.795x for
W1/W2/W4/W8/W16. A separate twelve-process real-default sweep, where fallback
uses one spp per dispatch and SIMD uses up to 64, produced medians of
72.205/78.657/55.056/62.260/66.910/65.413 spp/s. That batching policy makes
W1 1.089x fallback but does not make W8 or W16 faster than fallback.

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
measured 0.9952x with only 4/10 wins. The production code keeps the bulk zero.

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

1. Split hot varying scheduler state into register-resident/SoA regions and
   rematerialize immutable launch/ray-query fields across suspension, following
   the liveness/frame principles merged from `next`.
2. Compact or rebatch sparse ray cohorts before Embree and reduce ray-query
   state crossings; inlining Embree LLVM IR is exploratory and cannot replace
   this measured scheduler work.
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
W16 with RGB PSNR 39.10/39.74/39.67/39.58/39.48 dB respectively. These are
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

The required native-math/fallback-math/runtime-width gate passes 3/3. A
focused gate including accel, curve-summary replacement, and example-option
parsing passes 6/6. After a full Release build, the complete configured
repository CTest suite passes 140/140: 26 integration-SIMD, 21 runtime-SIMD,
and three graphics-SIMD tests are included. This also includes the
coroutine-frame tests merged from `next` and the repaired lazy-dispatch scalar
snapshot regression.

# SIMD CPU backend performance report

Snapshot date: 2026-08-14. This report covers the Release build after merging
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
JIT callback gather while retaining provider-side fail-closed validation. A
new W16-only procedural specialization also packs a fully active post-proceed
status with one sequential state-pointer pass; narrower widths and every
non-procedural acceleration structure retain the established callback.
Its provider-native batch installers now place their overwhelmingly common
already-ascending case on one hot branch and all reorderings on an unlikely
edge. The latest control-flow stage also recognizes a bounded nested
select/Phi forwarding ladder at W4/W8, exposing one additional small varying
diamond without applying a whole-function CFG cleanup. A measured W8-only
follow-up permits one further `float3` ladder layer while W4 retains the
original cost boundary.
The newest memory stage independently applies ISPC's bounded-gather lesson to
one much narrower Luisa pattern: eligible W8 direct typed-buffer vectors pack
adjacent 32-bit leaves into legal 64-bit LLVM masked gathers. TargetTransformInfo
gates the rewrite, so W8 itself remains independent of AVX-512 availability.
The newest scheduler stage reuses the incoming active-mask SSA value after a
runtime-coherent varying branch or switch proves that its selected successor
mask is identical. This preserves all-on/partial-tail identity across the hot
edge without changing the genuinely divergent scheduler path.
The current convergence stage bypasses destination-side frame traversal when
the scalar current token is zero and stops a completed cascade as soon as it
restores the root token. Both shortcuts are exact refinements of the formal
target-arrival identity and retain a same-binary oracle.
The latest scheduler stage removes a selected divergent binary child's
redundant ready-record push and immediate LIFO pop. Normal pops and these
children share one PC route and one dispatch switch; a measured state-slot
gate keeps the refinement out of smaller kernels where it is not profitable.

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
Ordinary path tracing uses seven adjacent fallback/SIMD pairs per width with
reversed order on alternating pairs; cutout retains three pairs per width.
The focused triangle-only-provider result uses twelve W8 pairs, while the
other widths use four to six pairs. The refreshed ordinary and voxel processes
keep stable per-backend hashes and use separate gallery conformance runs. The
refreshed 64-spp cutout processes are performance-only; a separate 1024-spp
run supplies its gallery conformance gate. SDF uses its internal four-SPP
throughput metric;
high-SPP SDF image comparison remains a separate conformance gate.
Image/SDF/Spacex/GEMM cells retain the earlier seven-process sweep and are not
performance claims for this checkpoint. Voxel is refreshed with seven
balanced-order fallback/W1/W2/W4/W8/W16 rounds, 64 render iterations per
process, and all variants using 32 workers on logical CPUs 0--31.

Speedup is always `fallback time / SIMD time`, or
`SIMD throughput / fallback throughput`, so values above one are wins.

## Current fallback-relative results

| Workload and metric | fallback | W1 | W2 | W4 | W8 | W16 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| non-coro SDF, samples/s | 8.705 | 8.197 (0.942x) | 9.476 (1.089x) | 15.112 (1.736x) | 22.568 (2.593x) | 32.959 (3.786x) |
| image pipeline, ms/iteration | 8.379 | 17.184 (0.488x) | 9.169 (0.914x) | 6.493 (1.290x) | 4.992 (1.678x) | 4.249 (1.972x) |
| voxel render, ms/iteration | 6.944 | 8.371 (0.825x) | 15.126 (0.458x) | 8.710 (0.793x) | 6.783 (1.021x) | 5.438 (1.266x) |
| Spacex, ms/frame | 158.831 | 150.954 (1.052x) | 94.904 (1.674x) | 64.277 (2.471x) | 49.999 (3.177x) | 42.700 (3.720x) |
| ordinary path tracing, fixed 1 spp/dispatch, spp/s | 72.979 | 64.238 (0.870x) | 52.687 (0.708x) | 65.445 (0.930x) | 79.217 (1.073x) | 79.296 (1.129x) |
| cutout path tracing, fixed 1 spp/dispatch, spp/s | 59.366 | 45.860 (0.770x) | 29.919 (0.511x) | 36.802 (0.619x) | 42.791 (0.730x) | 42.283 (0.711x) |
| portable GEMM, GFLOP/s | 64.895 | 23.332 (0.360x) | 25.627 (0.395x) | 115.914 (1.786x) | 190.521 (2.936x) | 316.449 (4.876x) |

The GEMM row is a compute diagnostic rather than a graphics result. It uses
eight explicit SIMD workers and seven independent process medians; every
process performs seven timed samples of 128 complete 256-by-256 dispatches and
validates the output against double-precision accumulation. The fallback
process medians ranged from 41.594 to 88.326 GFLOP/s under shared-host load,
while the SIMD distributions were tight. Its relative speedups must therefore
be treated as host observations, not cross-machine constants.

The refreshed voxel cells are the medians of the seven balanced rounds.
Parenthesized values are the preferred geometric means of the within-round
fallback/SIMD ratios. Their 95% paired log-space Student-t intervals at
W1/W2/W4/W8/W16 are [0.8118, 0.8389], [0.4491, 0.4661],
[0.7728, 0.8146], [1.0005, 1.0422], and [1.2407, 1.2918]. W16 wins all seven
rounds and W8 wins six; the narrower widths lose all seven. Every fallback
process retains SHA-256
`27455a0e126ecfae23d592a58121751c5884a69d9d7388b20195e8b0a121829a`,
and every SIMD process at all five widths retains
`6172183a6c96704ffa48a6b64d30afcf2a3921431507dc40e3f80f1ae1362e4b`.
Independent gallery comparisons pass at 48.08 dB RGB PSNR for fallback and
82.83 dB for SIMD. The two backends use different floating-point paths and are
not expected to produce identical PNG bytes; same-backend refinement/oracle
outputs below are byte-identical.

The current path-tracing rows are paired rather than independent medians because
unrelated host tasks moved the load average during the sweeps. For ordinary
path tracing the displayed fallback cell is the pooled median of 35 fallback
processes; each SIMD cell is its seven-process median and the parenthesized
speedup is the preferred geometric mean of seven adjacent SIMD/fallback
ratios. W1/W2/W4/W8/W16 measure 0.8699x/0.7078x/0.9304x/1.0734x/1.1294x;
their 95% paired bootstrap intervals are [0.8637, 0.8764], [0.6995, 0.7167],
[0.9221, 0.9417], [1.0592, 1.0824], and [1.1103, 1.1474]. W8 and W16 win all
seven pairs; the other widths lose all seven. Every width/backend retains one
stable output hash across its seven processes, and separate gallery runs
supply correctness conformance. The final-binary cutout row was refreshed with
three adjacent alternating pairs per width after the state-handle cache landed.
It remains below fallback at every width: 0.7695x/0.5105x/0.6193x/0.7298x/0.7111x
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

### Paired direct-buffer vector leaves

ISPC 1.31.0's BSD-3-Clause `GatherCoalesce` pass, independently inspected at
revision `c6adb4f86f56`, bounds its compatible-gather scan, stops at possible
writes, and limits grouping to control register pressure. No source was copied.
The accepted Luisa follow-up deliberately implements only the easiest audited
subset: within one nonvolatile direct typed-buffer read of a top-level vector,
each adjacent pair of non-Boolean 32-bit scalar leaves may share one 64-bit
masked gather. It does not scan across XIR instructions or cross aggregate,
bindless, local, accel, ray-query, byte-address, or volatile boundaries.
The declared buffer element and read type must match, adjacent field offsets
must be exactly four bytes apart, and an odd final leaf stays 32-bit.

Production enables the rewrite only at W8 and only when host LLVM TTI reports
at least one 512-bit fixed-vector register, a legal `<8 x i64>` masked gather,
and no forced scalarization. Therefore semantic W8 still does not promise
AVX-512: AVX2-only hosts retain the four 32-bit gathers. W4 was neutral, and a
temporary W16 gate had a real-render confidence interval crossing one, so both
retain the portable path. `LUISA_SIMD_DISABLE_PAIRED_LEAF_GATHER=1` is the
same-binary oracle.
LLVM legality does not establish relative profitability for this pair, so the
implementation does not cite it as cost proof; it uses only the exact W8
shape, legality/no-scalarization checks, and the measured gate.

The exact ordinary-path W8 main object changes as follows:

| W8 main kernel | 32-bit-leaf oracle | Paired leaves |
| --- | ---: | ---: |
| assembly text bytes | 191,516 | 191,220 |
| static instructions | 3,586 | 3,581 |
| vector instructions | 2,333 | 2,328 |
| branches / calls | 234 / 5 | 234 / 5 |
| stack references / frame | 1,024 / 7,168 B | 1,024 / 7,168 B |
| hardware gathers | 39 | 35 |
| `vgatherqps` / `vpgatherqd` / `vpgatherqq` | 18 / 19 / 2 | 10 / 19 / 6 |

Thus eight 32-bit gathers become four 64-bit gathers without changing the
scheduler, stack frame, branch count, or call boundary. Fourteen alternating
128-SPP same-binary pairs measure 1.00884x candidate/oracle geometric mean,
13/14 wins, and a 95% bootstrap interval of [1.00288, 1.01319]. Candidate and
oracle medians are 79.533/78.530 spp/s, and all 28 PNGs are byte-identical.
Five 256-SPP `perf stat` repetitions reduce aggregate task clock by 1.26% and
cycles by 1.24%, while instructions and branches change only -0.06%/-0.01%.
This isolates a small gather-latency gain, not scheduler-state removal.

The final W8 voxel kernel reports zero paired gathers: enabled/disabled JIT
objects, assembly statistics, and PNG hashes are identical. Permanent tests
cover W4/W8/W16 target selection, candidate/oracle LLVM intrinsic widths,
final x86 gather shape when TTI accepts it, execution equality, an odd `uint3`
leaf, and a 13-element inactive tail.

## Scheduler cost and assembly evidence

The coherent direct-CFG proof accepts a function when all branch/switch
selectors are warp- or cohort-uniform and no convergence remains after the
local predicated-memory refinement described below. It emits ordinary LLVM
control flow, keeps cohort values scalar, preserves the initial inactive-tail
mask, and allocates no ready queue or convergence frame.

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

### Nested select-ladder refinement

The accepted branch-splitting checkpoint handles a narrow pattern left after
the innermost small diamond is if-converted: a newly generated `select` feeds
a single-predecessor PHI-only forwarding block, and that value participates in
the next enclosing diamond. Only Name metadata may move to the unique select;
block, branch, non-Name metadata, multiple-use values, and pre-existing
selects fail closed. The normal four-per-arm, six-total, four-live-out, and
cost-twelve limits are reapplied at every enclosing layer. No whole-function
CFG cleanup is enabled.

The final W4/W8 binary uses
`LUISA_SIMD_DISABLE_PREDICATED_IF_REFINEMENT=1` as its same-binary oracle. Each
default-configuration W4/W8 result below comprises fourteen alternating pairs,
128 voxel renders per process, and 32 workers pinned to logical CPUs 0--31.
W2 and W16 used seven exploratory pairs under the same method in an otherwise
identical temporary measurement build that enabled the candidate at those
widths; the production width gate was restored immediately afterward. Speedup
is oracle time divided by candidate time:

| Width | Production policy | Paired geometric mean | Wins | 95% bootstrap interval | Candidate/oracle median, ms |
| ---: | --- | ---: | ---: | ---: | ---: |
| W2 | rejected | 0.9413x | 0/7 | [0.9340, 0.9468] | 26.399 / 24.916 |
| W4 | enabled | 1.0154x | 13/14 | [1.0116, 1.0183] | 16.087 / 16.351 |
| W8 | enabled | 1.0106x | 14/14 | [1.0074, 1.0142] | 9.394 / 9.490 |
| W16 | rejected as neutral | 1.0004x | 3/7 | [0.9956, 1.0056] | 6.580 / 6.569 |

Every candidate/oracle output used one identical SIMD PNG hash. A separate
W2/W4/W8/W16 inactive-tail regression executes thirteen elements, compares
the accepted widths against the same-binary oracle element by element, and
keeps non-Name metadata plus pre-existing select forwarders unchanged.

The W8 final object explains the modest but stable real-render gain. The
refinement increases accepted diamonds from two to three, reduces Schedule
blocks 37 to 32, convergence points 11 to 10, state slots 45 to 39, and cold
slots 26 to 20. Optimized assembly changes as follows:

| W8 voxel kernel | Oracle | Refinement |
| --- | ---: | ---: |
| assembly text bytes | 134,476 | 124,557 |
| static instructions | 2,959 | 2,724 |
| vector instructions | 1,470 | 1,359 |
| branches | 288 | 260 |
| stack references | 778 | 718 |
| stack allocation | 4,992 B | 4,480 B |
| calls / scalar-math calls | 3 / 2 | 3 / 2 |

W4 similarly falls from 3,013 to 2,756 static instructions, 280 to 251
branches, 736 to 665 stack references, and a 2,504- to 2,264-byte frame. The
gain is therefore removed scheduler state and control, not a math-library or
host-launch change. Ordinary non-query path tracing has no eligible forwarding
block: enabled and disabled W8 main objects are exactly equal at 3,586 static
instructions, 234 branches, 1,024 stack references, and a 7,168-byte frame.
No path-tracing speedup is claimed for this checkpoint.

Worker-count sensitivity was measured separately because the host has sixteen
cores and thirty-two SMT threads. Three 64-render medians with 16 workers on
CPUs 0--15 versus 32 workers on CPUs 0--31 are W1 10.731/8.270,
W2 43.848/24.683, W4 28.209/15.883,
W8 15.617/9.185, and W16 10.527/6.497 ms. The existing SIMD thread pool
therefore benefits substantially from using all logical CPUs; replacing it
with a system `parallel_for` is not supported by this evidence. The accepted
W4/W8 refinement remains positive under both 16-worker and default 32-worker
configurations.

Several broader scheduler experiments were rejected before this narrow rule
was retained. Whole-function `phi_cleanup + simplify_cfg` reduced W8 voxel
blocks 37 to 33 and static instructions 2,959 to 2,953, but seven 128-render
pairs measured 0.9850x. A nonempty-continuation specialization reduced static
instructions to 2,776 and branches to 244 but measured 0.9663x on voxel;
ordinary path tracing was only 1.0075x. Pure-region predication and coherent
forwarding likewise measured 0.9929x/0.9853x and 0.9821x respectively. They
remain out of production: smaller IR is not sufficient when added selects,
mask liveness, or register pressure increase dynamic cost.

This policy was informed by an independent audit of ISPC 1.31.0 source at
revision `c6adb4f86f56` under its BSD-3-Clause license. ISPC's varying-if
emitter uses all-on/mixed paths and permits straight-line predication only
when both arms pass `SafeToRunWithMaskAllOff` and their estimated statement
cost is below six; otherwise it emits mask-aware arm skips. No ISPC source or
threshold is copied. Luisa retains its independently audited safety whitelist,
weighted register-unit cost, and the width-specific real-render gate above.

### W8 deeper select-ladder boundary

The next enclosing Voxel material-selection layer is structurally eligible
after the accepted forwarding refinement, but its two `float3` selects make
the weighted cost fourteen rather than the default limit of twelve. The new
policy raises that limit to sixteen only at W8. Every totality, metadata,
four-per-arm, six-total, and four-live-out rule remains unchanged. The
same-binary oracle
`LUISA_SIMD_DISABLE_DEEP_PREDICATED_IF_REFINEMENT=1` keeps the existing W8
refinement while restoring cost twelve.

Two independent fourteen-pair sweeps used alternating forward/reverse order,
128 Voxel renders per process, 32 workers, and logical CPUs 0--31. The combined
candidate/oracle result is 1.00566x geometric mean, 22/28 wins, and a
bootstrap 95% interval of [1.00299, 1.00848]. Candidate/oracle process medians
are 1,199.413/1,205.234 ms, or 9.370/9.416 ms per render. All 56 PNGs have the
same SHA-256
`6172183a6c96704ffa48a6b64d30afcf2a3921431507dc40e3f80f1ae1362e4b`.

The exact final W8 object changes as follows:

| W8 Voxel kernel | Cost-twelve oracle | Deep ladder |
| --- | ---: | ---: |
| Schedule blocks / convergence points | 32 / 10 | 29 / 9 |
| state slots / cold slots | 39 / 20 | 38 / 19 |
| assembly text bytes | 124,557 | 116,227 |
| static / vector instructions | 2,724 / 1,359 | 2,557 / 1,304 |
| branches / calls | 260 / 3 | 237 / 3 |
| stack references / frame | 718 / 4,480 B | 664 / 4,224 B |

Five alternating 256-render `perf stat` pairs distinguish the mechanism:

| Aggregate counter | Deep ladder vs oracle |
| --- | ---: |
| task clock | -0.569% |
| cycles | -0.497% |
| retired instructions | +0.427% |
| branches | -0.122% |
| branch misses | -6.174% |
| measured render wall time | -0.424% |

Predicating the extra layer executes slightly more dynamic select work, but
removes enough scheduler/control and misprediction cost to win. This is why
the decision is based on paired counters and wall time rather than smaller
static assembly alone.

The identical temporary rule at W4 reduced its object but regressed real
rendering: fourteen pairs measured 0.99502x, 2/14 wins, and a 95% interval of
[0.99112, 0.99962]. W4 therefore remains at cost twelve; W1/W2/W16 also retain
their previous policy. Ordinary and cutout path tracing, image processing,
non-coroutine SDF, Spacex, and game of life report identical W8 optimization
counts under candidate and oracle, so no benefit is claimed for them. A
permanent W2/W4/W8/W16 `float3` ladder regression covers the width gate,
thirteen-element inactive tail, exact candidate/oracle output, Schedule-state
reduction, and final x86 assembly-size direction.

### Predicated masked-memory control

The standalone `masked_stream` control contains one varying diamond whose
taken arm reads two lane-affine typed buffers. Before Schedule-emitter
predication, optimized W8 contained about 744 static instructions, 71 branches,
152 stack-reference instructions, and a 480-byte frame. The accepted lowering
executes the two arms under disjoint masks and removes the complete scheduler
for this CFG. Final W4/W8/W16 objects contain 34/36/36 instructions, one branch,
no calls, no stack references, and no scalar-math symbol. The runtime reports
one predicated-memory diamond and three arm instructions.

Three independent clean seven-process runs were made after an earlier run was
discarded because an unrelated build/test occupied the host. Combining the 21
process medians gives:

| Variant | Mitems/s median | Geomean vs fallback | 95% interval |
| --- | ---: | ---: | ---: |
| fallback | 5,082.447 | 1.000x | [1.000, 1.000] |
| SIMD W1 | 5,861.242 | 1.159x | [1.132, 1.186] |
| SIMD W2 | 5,907.414 | 1.165x | [1.139, 1.191] |
| SIMD W4 | 6,247.237 | 1.220x | [1.188, 1.253] |
| SIMD W8 | 5,852.716 | 1.166x | [1.141, 1.192] |
| SIMD W16 | 5,696.662 | 1.122x | [1.100, 1.145] |

The same-width ISPC/Luisa ratios are 1.034x
([1.013, 1.055]) for AVX2 W4, 1.069x ([1.049, 1.088]) for AVX2 W8,
1.019x ([0.991, 1.047]) for AVX-512 W4, 1.077x
([1.061, 1.093]) for AVX-512 W8, and 1.103x
([1.084, 1.122]) for AVX-512 W16. Thus W4 is statistically tied with the
AVX-512 control, while W8 and W16 retain stable 7.7% and 10.3% gaps. The old
order-of-magnitude scheduler deficit is gone; the remaining difference is in a
small stackless body and width/memory throughput.

This local optimization must not be generalized to graphics. W8 optimization
reports show zero predicated-memory diamonds in all five image-processing
kernels, both game-of-life kernels, the shader-toy main kernel, the 37-block
voxel kernel, and the 31-block ordinary non-coroutine path-tracing kernel. The
image, game-of-life, shader-toy, and voxel gallery comparisons pass; the
one-SPP ordinary path tracer also completes. Image processing was already
single-block/direct, while voxel and path tracing contain larger nested
regions. No real-example speedup is claimed for this stage. Branch
splitting/code motion must expose safe read subregions before this mechanism
can help those kernels.

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

A narrower follow-up fused procedural `inst/prim` publication into the already
mandatory status scan, so it did not add that rejected second lane pass. The
same-binary oracle was
`LUISA_SIMD_DISABLE_RAY_QUERY_PROCEDURAL_CANDIDATE_CACHE=1`; paired runs pinned
sixteen workers to CPUs 0--15 through `LUISA_SIMD_WORKER_COUNT=16`. Fourteen
16-candidate rejection-chain pairs per width measured 1.0411x (14/14, 95%
interval 1.0314--1.0509) at W4, 1.0663x (14/14, 1.0581--1.0747) at W8, and
1.0356x (13/14, 1.0174--1.0541) at W16. Final assembly removed one W8 gather
and two W16 gathers; static instructions changed 1,319 to 1,315 and 1,919 to
1,910, while two SoA vectors increased stack allocation by 128 bytes.

The real 1280x720, 1024-SPP mixed procedural-callable renderer nevertheless
measured 0.9977x at W8 (4/14, 95% interval 0.9923--1.0031) and 0.9985x at W16
(4/14, 0.9892--1.0078). All 56 candidate/oracle images were byte-identical
(SHA-256 `d95cbe53b1cf7c573953986e2f64516494bfa7870536dbd2e37f98b2feb49036`).
The fused cache is therefore also rejected: removing these two gathers is not
the real renderer bottleneck, and a synthetic-only win does not justify the
ABI, analysis, or per-kernel stack cost. Future payload work must let the
traversal provider produce its native packet/SoA representation directly or
remove a larger host/state boundary.

An independent immutable-ray sidecar was also rejected. It cached the seven
construction-stable `origin/t_min/direction` scalars but deliberately left
mutable `t_max` in the authoritative state. W1/W2/W4/W8/W16 world-ray semantic
tests and an inactive-tail ORC fixture passed, and W8 assembly removed fourteen
`vgather` instructions. Fourteen fixed-affinity W16 procedural-callable pairs
were nevertheless only 1.0047x (9/14, 95% interval 0.9937--1.0159); five
`perf stat` pairs were 1.0040x in wall time (3/5). The object grew by nine
static instructions, twenty stack references, and 384 bytes of stack, while
cache references increased about 9.8%. This demonstrates that moving already
immutable fields into an additional JIT sidecar trades gathers for register/
stack pressure rather than removing the state boundary; no immutable-ray code
or oracle remains.

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

### W16 procedural dense status packing

The generic status callback scans an arbitrary active mask by repeatedly
finding and clearing its lowest set bit. Procedural W16 workloads frequently
return all sixteen lanes from one provider call. Acceleration structures built
on a W16 device and proven to contain at least one procedural instance now
install a separate status entry: it calls the same plain provider and retains
its provider-agreement validation, then uses one fixed sequential pass when
the post-call cohort is `0xffff`. Sparse cohorts still use the original set-bit
packer, so inactive or null state pointers are never inspected. Every accel
build recomputes the procedural summary; W1/W2/W4/W8 and triangle/curve-only
accels keep the original callback pointer. The same-binary oracle is
`LUISA_SIMD_DISABLE_PROCEDURAL_WIDE_STATUS_PACK=1`, sampled at accel creation.

This width restriction is measured rather than architectural. An initial
W8/W16 selection was rejected after seven W8 procedural-callable renderer
pairs measured 0.9861x with only 1/7 wins. Restricting the specialization to
W16 leaves that path and the triangle-only path unchanged. Seven final-binary
16-candidate microbenchmark pairs measured:

| Width | Candidate/oracle | Wins | Selection |
| ---: | ---: | ---: | --- |
| W4 | 1.0027x | 3/7 | original callback |
| W8 | 1.0037x | 4/7 | original callback |
| W16 | 1.0122x | 6/7 | enable dense full-mask pack |

Five alternating W16 `perf stat` pairs measured candidate/oracle ratios of
0.9830 for user cycles, 0.9800 for retired instructions, and 1.0014 for
branches. Branch misses were about 0.06% of branches and did not provide a
stable directional signal. Disassembly keeps the original generic wrapper
instruction-for-instruction identical at 237 bytes. The W16 entry is 405 bytes
and the complete backend `.text` grows by 704 bytes.

The real mixed procedural-callable renderer (1280x720, 1024 spp) measured a
1.0067x paired geometric-mean speedup across fourteen alternating W16 pairs,
with 10/14 wins and a log-ratio 95% interval of 1.0007x--1.0127x. Median times
were 2,617.6 ms candidate and 2,633.0 ms oracle. All 28 outputs passed the
gallery comparison and were byte-identical. As a non-procedural control, seven
W16 64-spp cutout pairs measured 1.0001x with 4/7 wins; all fourteen output
hashes matched. This is therefore recorded as a small W16 procedural latency
win, not as a general ray-query or W8 improvement.

### W16 provider-native procedural status publication

The paired W16 entry still crossed the 1216-byte lane records twice: the plain
provider advanced cached candidates or installed an Embree batch, then the
status wrapper reread every active lane. A provider-native entry now publishes
the terminated/surface/procedural bits in those existing passes. Cached lanes
publish immediately after candidate advance; traversal lanes publish while the
newly scanned batch is sorted, installed, and advanced. It adds neither a
sidecar nor a lane pass, and it retains W16 packet traversal. Sparse masks and
multiple accel/terminate-mode groups keep the original grouping semantics. The
exact previous paired-call-plus-pack entry is selected by
`LUISA_SIMD_DISABLE_PROCEDURAL_WIDE_FUSED_STATUS=1`.

Before the change, a fixed-affinity W16 procedural-callable cycle profile put
10.73% in the plain wide provider, 6.17% in candidate-batch installation, and
1.47% in the separate status wrapper. Fourteen alternating strict-oracle
16-candidate microbenchmark pairs measured 1.0710x with 14/14 wins and a 95%
log-ratio interval of 1.0566x--1.0855x.

The real 1280x720, 1024-SPP mixed procedural-callable renderer measured 1.0297x
across fourteen alternating pairs with 11/14 wins and a 95% interval of
1.0110x--1.0486x. All 28 images were byte-identical (SHA-256
`4e3730fa871c450e44b02b77301ca3bb4041cb177c03b1fc786fae8db7f41cac`).
Five additional `perf stat` pairs measured a 1.0469x wall-time ratio; candidate
to oracle ratios were 0.9741 for cycles, 0.9839 for instructions, 0.9846 for
branches, 0.9927 for branch misses, and 0.9359 for cache references. Cache
misses rose to 1.0243x but remained too small to offset the removed state pass.

The original `_ray_query_proceed_wide` retains its exact `0x3556`-byte function
size in an independently compiled HEAD object; only source-line immediates in
diagnostic metadata differ. Cold fused batch-install helpers are outlined, and
the fused entry itself is `0x1fb6` bytes. The permanent W16 procedural gate
runs both implementations and covers the 35-thread inactive tail, query-all/
query-any commit/reject/terminate, 40-candidate continuation, motion, and mixed
surface/procedural rebuilds.

An environment-gated audit of the real W16 procedural renderer counted
101,242,274 proceed calls. Scans occurred on 65.52% of calls; scan groups
averaged 15.00 lanes and 89.62% were full W16 packets, ruling out accel-group
fragmentation in this workload. Surface batches were already ascending in
100% of cases. Procedural batches were ascending in 95.65%, reverse-ordered in
3.52%, and required a general sort in 0.83%; none overflowed into heap form.
The retained branch-layout refinement therefore tests `ascending` first and
moves all three reorderings to the cold edge. Candidate-build invariants make
the transformation semantics-preserving, and it is applied only to the fused
status helpers so the exact previous provider remains an oracle.

Against an independently saved committed backend, fourteen alternating W16
rejection-chain pairs measured 1.0271x with 13/14 wins and a 95% interval of
1.0094x--1.0451x. Fourteen alternating 1280x720, 1024-SPP procedural-callable
pairs measured 1.0338x with 14/14 wins and a 95% interval of
1.0249x--1.0427x. All 28 renderer outputs were byte-identical (SHA-256
`4e3730fa871c450e44b02b77301ca3bb4041cb177c03b1fc786fae8db7f41cac`).
The two full/sparse status installers shrink from 4,635 bytes each to
3,147/3,163 bytes, and the fused proceed entry shrinks from 8,722 to 8,118
bytes. A five-pair hardware-counter attempt was discarded wholesale because
concurrent host load produced paired wall ratios from 0.62x to 2.13x; no
counter claim is made from that run.

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

The compiler is supplied explicitly to a standalone benchmark driver. ISPC
objects and host launchers are created in a temporary directory; no ISPC
compiler path, target, generated object, or comparison executable enters the
project CMake graph or CTest suite, and no machine-local path enters source.
No ISPC implementation, SLEEF implementation, or approximation coefficient is
copied into production. The benchmark tool provenance is official
[ISPC 1.31.0](https://github.com/ispc/ispc/releases/tag/v1.31.0), whose
[license](https://github.com/ispc/ispc/blob/main/LICENSE.txt) is BSD-3-Clause.

The compiler source was independently inspected at revision `c6adb4f86f56`.
The relevant mechanisms are its all-on/all-off/mixed varying-control paths,
the small statement-cost threshold used to choose predication versus
`any(mask)` arm skips, a compatible-gather scan bounded to four operations and
stopped by possible writes, and constant-prefix masked-memory narrowing. No
implementation was copied. Luisa's existing consecutive-buffer-read XIR pass
only joins absolute constant byte offsets and is not a substitute for ISPC's
dynamic typed-buffer gather coalescing.

A current 32-worker sweep including the direct divergent-child route covers
the first four standalone workloads in seven balanced process rounds. Because
the initial path-trace sweep encountered more shared-host noise, that row is
replaced wholesale by a separate fifteen-round run. Every variant is pinned
to logical CPUs 0--31. The table reports the paired geometric mean and 95%
log-space Student-t interval for `ISPC / Luisa SIMD` at the same semantic
width; values above one mean ISPC is faster:

| Workload | W4 AVX2 | W4 AVX-512 | W8 AVX2 | W8 AVX-512 | W16 AVX-512 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Mandelbrot | 0.936x [0.923, 0.950] | 0.922x [0.908, 0.936] | 1.012x [0.988, 1.036] | 0.985x [0.963, 1.007] | 1.091x [1.068, 1.116] |
| masked stream | 0.979x [0.877, 1.092] | 0.945x [0.883, 1.012] | 1.028x [0.957, 1.104] | 1.008x [0.941, 1.081] | 1.073x [1.003, 1.148] |
| AoS to SoA | 1.036x [1.004, 1.069] | 0.996x [0.967, 1.026] | 1.079x [1.020, 1.140] | 1.068x [1.002, 1.138] | 1.062x [1.017, 1.108] |
| GEMM | 0.746x [0.735, 0.757] | 0.749x [0.730, 0.769] | 0.771x [0.764, 0.779] | 0.768x [0.749, 0.788] | 0.789x [0.754, 0.826] |
| analytic path trace | 2.580x [2.519, 2.643] | 2.444x [2.387, 2.503] | 2.387x [2.323, 2.453] | 2.193x [2.104, 2.285] | 2.357x [2.316, 2.399] |

Mandelbrot, masked stream, AoS-to-SoA, and GEMM are bit-identical across all
eight implementations. The asset-free analytic path tracer validates 921,600
floats per implementation with zero tolerance violations; its maximum absolute
and relative errors against Luisa W4 are `1.1921e-7` and `2.7532e-7`.
The Luisa W4/W8/W16 process medians are respectively
713.060/1,182.480/1,770.326 Mitems/s for Mandelbrot,
6,356.242/5,887.078/5,677.879 Mitems/s for masked stream,
2,722.494/2,521.626/2,499.919 Mitems/s for AoS-to-SoA,
263.789/339.163/437.597 GFLOP/s for GEMM, and
734.532/1,071.991/1,219.368 Mitems/s for analytic path tracing.

The balanced intervals matter on this shared host: several small memory-kernel
differences include parity, while GEMM and analytic path tracing remain
unambiguous. Luisa is 27--34% faster than the best matched-width ISPC GEMM
variants. Mandelbrot is at parity through W8 and trails ISPC by 9.1% at W16.
The analytic path tracer remains the outlier: the best same-width ISPC target
is 2.36--2.58x faster, while AVX-512 x8 measures 2.19x. The
compiler executable is passed explicitly to the standalone runner and remains
absent from CMake; this sweep excludes fallback so every ratio is a direct
same-width compiler comparison.

### ISPC all-on ablation and widened update checkpoint

The ISPC source audit was repeated at revision
`c6adb4f86f5678ce6c41951b1e2b59f727455697` under its BSD-3-Clause license.
The relevant implementation is the independently inspected varying-control
emission in `src/stmt.cpp` and function all-on/mixed versioning in
`src/func.cpp`; no source or coefficient was copied. Twenty-one randomized,
single-core process pairs on the matched analytic path tracer gave these
disabled-feature throughput ratios relative to normal ISPC:

| ISPC option | disabled / normal throughput | 95% CI | interpretation |
| --- | ---: | ---: | --- |
| `disable-coalescing` | 1.0028x | [0.9791, 1.0147] | neutral |
| `disable-gather-scatter-optimizations` | 1.0112x | [0.9967, 1.0241] | neutral |
| `disable-all-on-optimizations` | 0.9050x | [0.8833, 0.9226] | stable 9.5% loss |

Normal ISPC's x8 path-trace body has 1,062 static instructions, 67 branches,
43 stack references, 839 vector instructions, and 313 mask references. With
all-on optimization disabled it grows to 1,351 instructions, 78 branches, 263
stack references, 1,095 vector instructions, and 404 mask references. The
gather ablations are neutral because this analytic workload has no compatible
dynamic gather window; the all-on result instead points directly at control
versioning, register residency, and bounded predication.

The first retained Luisa follow-up widens only a fail-closed one-sided state-
update diamond: one arm is empty, the other has five or six pure instructions,
at least two PHIs change, live-outs fit six 32-bit units, and weighted cost is
at most fifty-eight. Floating division has explicit latency weight eight and is
a SIMD-only non-trapping opt-in; integer division, every remainder, memory,
calls, side effects, unsafe casts, and W1 remain rejected. The oracle is
`LUISA_SIMD_DISABLE_WIDENED_PREDICATED_UPDATE=1`.

On the analytic path tracer all enabled widths convert four inner sphere-hit
updates. Candidate/oracle Schedule and final host-assembly changes are:

| Width | blocks | convergence | states | spills | instructions | branches | stack refs | frame bytes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| W2 | 66 -> 58 | 18 -> 14 | 67 -> 63 | 24 -> 20 | 5,328 -> 4,981 | 488 -> 403 | 1,218 -> 1,175 | 6,784 -> 4,032 |
| W4 | 62 -> 54 | 17 -> 13 | 65 -> 61 | 24 -> 20 | 4,109 -> 3,922 | 422 -> 342 | 993 -> 1,014 | 2,712 -> 2,568 |
| W8 | 60 -> 52 | 16 -> 12 | 65 -> 61 | 24 -> 20 | 4,011 -> 3,518 | 406 -> 328 | 1,062 -> 974 | 5,696 -> 5,152 |
| W16 | 66 -> 58 | 18 -> 14 | 67 -> 63 | 24 -> 20 | 4,602 -> 4,420 | 491 -> 416 | 1,187 -> 1,174 | 12,160 -> 11,328 |

Every candidate and oracle assembly has zero calls and zero scalar-math
symbols. Charging floating division at weight eight and setting the exact
fifty-eight-unit boundary reproduces these same final counts.

Fifteen randomized candidate/oracle process pairs were pinned to one core;
each process reports the median of seven timed samples. Speedup is oracle time
divided by candidate time:

| Width | speedup | 95% bootstrap CI | wins |
| --- | ---: | ---: | ---: |
| W2 | 1.2194x | [1.2116, 1.2204] | 15/15 |
| W4 | 1.0128x | [1.0110, 1.0144] | 14/15 |
| W8 | 1.0307x | [1.0277, 1.0351] | 15/15 |
| W16 | 1.0057x | [1.0034, 1.0073] | 14/15 |

Every Luisa result has checksum `a93089e651f98582`. A separate 21-pair W8
single-worker run measured 1.0319x [1.0276, 1.0333], and 28 eight-worker pairs
measured 1.0138x [1.0081, 1.0184]. Ordinary non-query path tracing, Voxel, and
image processing report zero widened candidates; their candidate/oracle JIT
code is identical, so no real-example gain is claimed for this stage. A fresh
W1/W2/W4/W8/W16 gallery sweep passes ordinary path tracing at
35.43/42.78/40.94/39.22/37.80 dB RGB PSNR, while every Voxel and image-
processing width passes at 82.83 and 89.25 dB respectively.

The standardized standalone driver then validated every path-trace output and
ran fourteen alternating 32-worker process rounds. The fallback distribution
was noisy under shared-host load, so its confidence intervals are retained:

| Variant | Mitems/s | vs fallback | 95% CI |
| --- | ---: | ---: | ---: |
| fallback | 576.639 | 1.000x | [1.000, 1.000] |
| SIMD W1 | 737.137 | 1.244x | [1.063, 1.456] |
| SIMD W2 | 291.043 | 0.495x | [0.424, 0.579] |
| SIMD W4 | 485.860 | 0.827x | [0.708, 0.965] |
| SIMD W8 | 835.099 | 1.411x | [1.208, 1.649] |
| SIMD W16 | 1,026.885 | 1.767x | [1.509, 2.069] |
| ISPC AVX-512 x8 | 2,220.705 | 3.746x | [3.195, 4.393] |
| ISPC AVX-512 x16 | 2,619.116 | 4.434x | [3.777, 5.204] |

The paired same-width ISPC/Luisa ratios are 2.655x [2.552, 2.761] at W8 and
2.509x [2.447, 2.573] at W16. Worker count materially changes the ratio for
this small workload: direct fourteen-pair measurements at 1/8/16 workers were
3.575/3.384/3.454x at W8 and 4.253/3.907/3.471x at W16. These are compiler and
parallel-scaling controls, not Embree renderer results.

The widened updates remove real scheduler work, but they do not close the
ISPC gap: W8 still has 3.31x as many static instructions, 4.90x as many
branches, and 22.65x as many stack references as ISPC's normal x8 body. The
next control target was therefore all-on/mixed propagation and register
residency rather than gather prefetching. The bounded follow-up below retains
the profitable identity while rejecting broad cloning that increased code and
register pressure.

### Runtime-coherent successor mask reuse

For a nonempty incoming mask `A`, a varying conditional partitions it into
`T = A & C` and `F = A & !C`. On the existing coherent path exactly one of
`T` and `F` is nonempty, so the selected mask equals `A`. Indexed-switch
case/default masks form the same disjoint partition. Schedule-to-LLVM now
passes `A` itself to that sole successor instead of retaining the derived
mask. The divergent path is unchanged. The same-binary oracle is
`LUISA_SIMD_DISABLE_COHERENT_MASK_REUSE=1`, and the runtime report exposes
`coherent_mask_reuses`.

The analytic path tracer reports 24 eligible static successor edges at W8.
Alternating single-core candidate/oracle processes measured:

| Width | speedup | 95% paired CI | wins |
| ---: | ---: | ---: | ---: |
| W2 | 1.1396x | [1.1335, 1.1457] | 15/15 |
| W4 | 1.2352x | [1.2199, 1.2507] | 15/15 |
| W8 | 1.1629x | [1.1566, 1.1693] | 21/21 |
| W16 | 1.1158x | [1.1133, 1.1184] | 15/15 |

Every result retains checksum `a93089e651f98582`. The exact W8 assembly grows
from 3,518 to 3,697 static instructions and from 328 to 345 static branches,
but stack references fall from 974 to 946 and the frame from 5,152 to 4,960
bytes. This is a useful counterexample to static-size-only selection: removing
the derived-mask dependency changes register allocation and the dynamically
taken coherent path even though LLVM duplicates some surrounding code.

The real Voxel kernel passed the same gate at every SIMD width. Each cell is
seven alternating 64-render candidate/oracle processes on 32 workers:

| Width | speedup | 95% paired CI | wins |
| ---: | ---: | ---: | ---: |
| W2 | 1.1145x | [1.1067, 1.1223] | 7/7 |
| W4 | 1.2355x | [1.2267, 1.2444] | 7/7 |
| W8 | 1.1240x | [1.1154, 1.1327] | 7/7 |
| W16 | 1.0611x | [1.0505, 1.0717] | 7/7 |

Candidate and oracle PNGs at all four widths are byte-identical with SHA-256
`6172183a6c96704ffa48a6b64d30afcf2a3921431507dc40e3f80f1ae1362e4b`.
W8 image processing is neutral at 0.9966x [0.9875, 1.0057], as expected for
its direct CFG. Ordinary Embree path tracing is also neutral at 1.0042x
[0.9934, 1.0150]; this mask identity does not remove its ray-query callbacks or
sparse traversal traffic.

The fallback-relative Voxel table at the top was independently refreshed with
seven balanced fallback/W1/W2/W4/W8/W16 rounds. W16 is now 1.1735x fallback;
W8 improves substantially but remains 0.8744x, so the control-flow gain is not
presented as a complete Voxel layout solution.

The ISPC comparison was refreshed with fifteen balanced single-core rounds.
ISPC remains faster on the same analytic algorithm: AVX2 i32x8 is 3.2946x
[3.2800, 3.3092] over Luisa W8, AVX-512 x8 is 3.0685x
[3.0525, 3.0846], and AVX-512 x16 is 3.8086x [3.7960, 3.8212] over Luisa W16.
These are compiler controls without Embree, not renderer traversal results.

Several broader ISPC-inspired experiments were rejected. Whole-function
all-on/mixed duplication doubled the analytic code body and its no-inline form
measured 0.9796x with one win in fifteen pairs. Cloning only the entry block
improved the analytic W8 case by about two percent but regressed W8 Voxel to
0.9262x and ordinary path tracing to 0.9847x. Unmasked single-cohort prefix
spills were neutral at 1.0016x analytically and neutral on all three real
examples. LLVM O3 measured 0.9988x [0.9960, 1.0016] in 21 pairs; its final
assembly and object were byte-identical to O2. An additional post-O2
`SROA/InstCombine/JumpThreading/SimplifyCFG` sequence was also byte-identical.
The retained change therefore exposes a mask fact unavailable to generic LLVM
rather than merely requesting more optimization passes.

This split result rules out a blanket LLVM-code-quality explanation. The
coherent GEMM body is already faster than the matched ISPC implementation;
the large gap appears specifically in dynamically varying iteration/control
and memory patterns. The analytic benchmark does not call Embree and must not
be presented as the real renderer's traversal comparison.

### Zero-token convergence-cascade guard

The formal scheduler gives token zero one exact meaning: the executing cohort
has no dynamic convergence frame. The destination-side target-arrival cascade
is therefore an identity in that state. Codegen now tests `current.token`
before entering the cascade and returns the incoming mask directly when it is
zero. After releasing a frame, it also stops when the restored parent token is
zero instead of executing one guaranteed no-op iteration. Nonzero parents keep
the full active-frame and target check, including same-target nested cascades.
`LUISA_SIMD_DISABLE_CONVERGENCE_TOKEN_GUARD=1` is the same-binary oracle and
`convergence_token_guards` reports the static guarded destinations.

The analytic path tracer has eight guarded destinations among twelve static
convergence points. Alternating single-core candidate/oracle processes measured
the following throughput speedups; intervals are paired 95% log-space
Student-t intervals:

| Width | speedup | 95% paired CI | wins | candidate/oracle median, Mitems/s |
| ---: | ---: | ---: | ---: | ---: |
| W2 | 1.7830x | [1.7725, 1.7936] | 15/15 | 35.576 / 19.993 |
| W4 | 1.8509x | [1.8453, 1.8566] | 15/15 | 60.247 / 32.584 |
| W8 | 1.3125x | [1.3008, 1.3243] | 21/21 | 93.397 / 71.424 |
| W16 | 1.1369x | [1.1322, 1.1418] | 15/15 | 102.768 / 90.381 |

Every process retains checksum `a93089e651f98582`. The exact W8 candidate is
larger than its oracle: 175,320 versus 166,830 bytes, 3,862 versus 3,697 static
instructions, 364 versus 345 branches, 1,057 versus 946 stack references, and
a 5,152- versus 4,960-byte frame. Neither object calls scalar libm. This stage
is selected by dynamic measurements rather than static size.

Delayed-enable `perf` attribution on the W8 JIT body assigned 36.29% of 248
pre-change samples to convergence cascades and 21.09% of 147 post-change
samples to them. Scheduler/ready handling becomes the next visible component
at 22.45% post-change. Sample shares are diagnostic rather than an independent
speed estimate; the alternating process gate above supplies that estimate.

The real Voxel renderer uses the same paths heavily. Seven alternating
64-render processes on 32 workers produced byte-identical candidate/oracle PNGs
at every width (SHA-256
`6172183a6c96704ffa48a6b64d30afcf2a3921431507dc40e3f80f1ae1362e4b`):

| Width | speedup | 95% paired CI | wins | candidate/oracle median, ms |
| ---: | ---: | ---: | ---: | ---: |
| W2 | 1.4890x | [1.4657, 1.5128] | 7/7 | 15.450 / 22.717 |
| W4 | 1.4242x | [1.3089, 1.5497] | 7/7 | 15.083 / 22.864 |
| W8 | 1.1892x | [1.1635, 1.2156] | 7/7 | 7.106 / 8.449 |
| W16 | 1.0847x | [1.0678, 1.1019] | 7/7 | 5.767 / 6.239 |

Ordinary Embree path tracing at 128 spp and one sample per dispatch is much
less sensitive: W2/W4/W8/W16 measure 1.0437x [1.0207, 1.0672], 1.0101x
[1.0030, 1.0172], 1.0102x [1.0003, 1.0203], and 1.0037x [0.9945, 1.0131].
Candidate and oracle images are byte-identical within each width. This is an
end-to-end renderer result, whereas the ISPC table above remains an
asset-free analytic compiler comparison.

### Shared divergent-child dispatch route

For a genuinely divergent conditional the established LIFO scheduler pushed
the true child, pushed the false child, then immediately popped the false
record. The retained refinement pushes only the true record and enters the
false child through a scalar-PC PHI feeding the normal dispatch switch. It
leaves exactly the same true record and runnable mask, preserves the current
token, and still executes false-edge assignments and destination-side
convergence arrival. `LUISA_SIMD_DISABLE_DIRECT_DIVERGENT_CHILD=1` restores the
push/pop oracle, while `direct_divergent_children` reports accepted static
sites.

The first prototype branched separately from every split to its target. On the
analytic W8 body it grew to 4,066 instructions, 389 branches, 1,098 stack
references, and a 22,431-byte `.text`, versus 3,862/364/1,057/21,289 for the
oracle. It helped selected high-state workloads but duplicated routing enough
to regress SDF. The retained implementation instead shares one PC route and
one switch. Exact final W8 code shape is:

| W8 analytic path trace | shared-route candidate | push/pop oracle |
| --- | ---: | ---: |
| annotated assembly bytes | 162,597 | 175,320 |
| `.text` bytes | 19,048 | 21,289 |
| instructions | 3,468 | 3,862 |
| vector instructions | 1,901 | 2,210 |
| branches | 383 | 364 |
| stack references | 917 | 1,057 |
| stack allocation | 4,992 | 5,152 |
| calls / scalar-math calls | 0 / 0 | 0 / 0 |

Fifteen alternating single-core candidate/oracle processes retain checksum
`a93089e651f98582`. Speedup is candidate throughput divided by oracle
throughput, with paired 95% log-space Student-t intervals:

| Width | speedup | 95% paired CI | wins | candidate/oracle median, Mitems/s |
| ---: | ---: | ---: | ---: | ---: |
| W4 | 1.0049x | [1.0005, 1.0093] | 12/15 | 60.911 / 60.721 |
| W8 | 1.0186x | [1.0013, 1.0362] | 14/15 | 96.095 / 94.063 |
| W16 | 1.1052x | [1.1004, 1.1101] | 15/15 | 113.990 / 103.074 |

The real Voxel kernel has 38 state slots at W8 and also passes the gate. Seven
alternating 64-render candidate/oracle processes on 32 workers produce
byte-identical PNGs at every width with SHA-256
`6172183a6c96704ffa48a6b64d30afcf2a3921431507dc40e3f80f1ae1362e4b`:

| Width | speedup | 95% paired CI | wins | candidate/oracle median, ms |
| ---: | ---: | ---: | ---: | ---: |
| W4 | 1.0569x | [1.0441, 1.0699] | 7/7 | 8.928 / 9.235 |
| W8 | 1.0291x | [1.0138, 1.0446] | 7/7 | 6.989 / 7.192 |
| W16 | 1.0408x | [1.0265, 1.0553] | 7/7 | 5.510 / 5.801 |

Ordinary Embree path tracing has 37 state slots but is statistically neutral:
W4/W8/W16 measure 0.9974x [0.9876, 1.0074], 0.9963x
[0.9899, 1.0026], and 0.9997x [0.9928, 1.0066] over seven 128-SPP pairs.
Candidate and oracle images remain byte-identical within each width. This
confirms that eliminating a scheduler stutter does not remove Embree traversal
or callback traffic.

Broadly enabling the refinement was rejected. The initial direct-target form
made SDF W4/W8/W16 run at 0.9242x/0.9522x/0.9623x of the oracle, and even the
shared route was about three percent slower at W8 before policy selection.
That SDF kernel has only 19 state slots and six force-enabled sites. The final
32-slot threshold emits zero sites and produces exact candidate/oracle SDF
objects: 3,302 instructions, 208 branches, 715 stack references, and a
5,568-byte frame. Image processing already uses direct coherent CFG and also
emits zero sites. W1/W2 are unchanged; the permanent forced W2 regression
requires byte-identical LLVM IR. The threshold admits the 61-slot analytic,
38-slot Voxel, and 37-slot ordinary path-trace kernels while excluding the
measured low-state regression.

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

1. Generalize the predicated-memory result through branch splitting and pure
   code motion: hoist or sink total operations to expose a safe read subregion,
   then use a cost model to choose straight-line masks or `any(mask)` empty-arm
   skips. The first gate is a nonzero hit count and stable gain in voxel or
   ordinary path tracing; the current complete-diamond recognizer hits neither.
2. Extend the completed within-read W8 leaf pairing to compatible nearby
   gathers only when they share base, dynamic offset, scale, and mask. Cap the
   scan window to control register pressure and stop at any possible write.
   Separately narrow known constant prefix-tail masks. Both need inactive-
   address and final-assembly gates; W4/W16 remain disabled until independently
   profitable.
3. Extend the accepted local aggregate promotion to the remaining ray-query
   payload only through a provider-native packet/SoA representation or a
   larger host/state-boundary elimination, following the liveness/frame
   principles merged from `next`. Both a wrapper-side second scan and a fused
   two-field payload cache are measured and rejected on real graphics; the
   accepted provider-native status publication removes the full status scan,
   while the state-handle cache covers pointers only.
4. Compact or rebatch sparse ray cohorts before Embree and reduce the remaining
   JIT-side ray-query state crossings. The accepted triangle-only host provider
   removes surface-runtime bookkeeping but does not compact lanes; inlining
   Embree LLVM IR is exploratory and cannot replace this measured scheduler
   work.
5. Move fixed-vector texture tap selection into JIT IR or introduce a measured
   tile/swizzle upload boundary. Preserve row-major public image semantics.
6. Generalize lane-affine recognition into bounded lane/value axis rotation
   only for coherent affine tiles; divergent control and warp operations pin
   lane identity.
7. Add software prefetch only for proven affine lookahead with a stable A/B.
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

The nested select-ladder stage reran the required native-math/fallback-math/
runtime-width gate (3/3), its Schedule-codegen regression, the combined
SIMD/XIR/runtime/graphics gate (88/88), and the complete configured Release
suite (140/140). The dedicated regression covers W2/W4/W8/W16, a thirteen-
element inactive tail, same-binary W4/W8 oracles, non-Name metadata rejection,
and pre-existing-select provenance. Final fallback and W8 voxel gallery gates
pass at 48.08 and 82.83 dB RGB PSNR respectively.

The paired-leaf-gather stage reran the required native-math/fallback-math/
runtime-width gate (3/3), its Schedule-codegen regression, the combined
SIMD/XIR/runtime/graphics gate (90/90), and the complete configured Release
suite (140/140). Syntax checks pass for every changed C++ translation unit.
The dedicated regression covers W4/W8/W16 target policy, `uint4` execution and
final x86 assembly, an odd `uint3` LLVM-IR tail, a thirteen-element W16 inactive
tail, and exact candidate/oracle output equality.

The W8 deeper-select-ladder stage reran the required native-math/fallback-math/
runtime-width gate (3/3), its Schedule-codegen regression, the currently
enumerated SIMD/XIR/runtime/graphics gate (88/88), and the complete configured
Release suite (140/140). The standalone predicated-if and loop-unswitch
microbenchmarks also retain their existing positive gates. Final W8 gallery
comparisons pass Voxel at 82.83 dB, ordinary path tracing at 39.22 dB, cutout
path tracing at 44.17 dB, and non-coroutine SDF at 63.13 dB RGB PSNR. The
ordinary and cutout runs independently report Embree 4.4.1 W4/W8/W16 native
packet support enabled.

The widened one-sided-update stage reran the required native-math/fallback-
math/runtime-width gate (3/3), the XIR mutation and if-conversion focus gate,
its W1/W2/W4/W8/W16 Schedule-codegen regression, and the complete configured
Release suite (140/140). The standalone ISPC driver validated fallback, all
five SIMD widths, and AVX-512 x8/x16 outputs before fourteen alternating timing
rounds. Fresh all-width ordinary path-tracing, Voxel, and image-processing
gallery comparisons pass; every one of those real kernels reports zero
widened candidates, matching the documented no-gain result.

The runtime-coherent-mask stage reran the required three-test native-math/
runtime-width gate, the focused Schedule-codegen executable, the SIMD-only
suite (129/129), and the complete SIMD+fallback configuration (140/140).
Clangd syntax checks pass for all four changed translation units. Fresh W1/W2/
W4/W8/W16 Voxel and image-processing galleries pass at 82.83/89.25 dB. The
ordinary path tracer passes its 1024-SPP gallery at
35.43/42.78/40.94/39.22/37.80 dB from W1 through W16; W8 cutout passes at
44.17 dB. Both path tracers report Embree 4.4.1 W4/W8/W16 native packet
support. The dedicated candidate/oracle regression covers runtime-coherent
conditional and indexed control, every successor, partial tails, one-lane
cohorts, and a genuinely divergent switch at W2/W4/W8/W16.

The zero-token convergence-cascade stage reran the required native-math/
runtime-width gate plus its focused Schedule-codegen regression (4/4), the
SIMD-only Release suite (129/129), and the complete SIMD+fallback Release suite
(140/140). Formatting, diff-check, and clangd syntax checks pass for every
changed C++ translation unit. Fresh W1/W2/W4/W8/W16 Voxel and image-processing
gallery comparisons pass at 82.83 and 89.25 dB respectively. Ordinary Embree
path tracing passes its 1024-SPP gallery at
35.43/42.78/40.94/39.22/37.80 dB from W1 through W16 and reports native
Embree 4.4.1 W4/W8/W16 packets enabled. W8 cutout path tracing passes at
39.58 dB, and non-coroutine SDF W8 passes at 63.13 dB, both at 1024 spp.

The shared divergent-child-route stage completed fresh full builds of both
Release trees, the required native-math/fallback-math/runtime-width/Schedule-
codegen gate (4/4), the SIMD-only suite (129/129), and the complete SIMD+
fallback suite (140/140). Formatting, diff-check, and clangd syntax checks pass
for every changed C++ translation unit. Fresh W1/W2/W4/W8/W16 Voxel and image-
processing galleries pass at 82.83 and 89.25 dB respectively. Ordinary Embree
path tracing passes its 1024-SPP gallery at
35.43/42.78/40.94/39.22/37.80 dB from W1 through W16 and reports native Embree
4.4.1 W4/W8/W16 packet support enabled; W8 cutout and non-coroutine SDF pass
at 39.58 and 63.13 dB. Fresh fallback checks also pass Voxel at 48.08 dB,
image processing at 100.00 dB, and ordinary path tracing at 62.22 dB. All
gallery outputs were written only in temporary directories; no reference image
was regenerated or modified.

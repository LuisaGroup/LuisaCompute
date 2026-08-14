# SIMD CPU backend performance report

Snapshot date: 2026-08-15. This report covers the Release build after merging
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
The latest layout stage applies the previously planned lane/value axis rotation
at one proven lane-consecutive direct-buffer boundary. W2/W4/W8/W16 transpose
component-major float/int2/3/4 Schedule values to one physical AoS masked
load/store; padded three-component vectors keep their padding masked off. The
analytic path tracer replaces its four output scatters, while image/texture-
based graphics kernels remain unchanged.
The newest texture stage publishes a typed mip-zero `BYTE1` view through the
backend-local bindless descriptor and lowers uniform-slot linear/mirror
sampling into target-independent fixed-vector JIT IR. A proven packet-wide
tail check permits wider W4/W8 gathers without changing public row-major
layout; every other operation retains the grouped callback.
The newest scheduler stage reuses the incoming active-mask SSA value after a
runtime-coherent varying branch or switch proves that its selected successor
mask is identical. This preserves all-on/partial-tail identity across the hot
edge without changing the genuinely divergent scheduler path.
The current control stage additionally versions one bounded coherent region at
W2/W8 when the complete physical packet is active. A measured W8 three-block
minimum excludes a regressing Voxel fragment, while W2 retains its profitable
two-block form; partial tails and mixed cohorts use the unchanged scheduler.
The current convergence stage bypasses destination-side frame traversal when
the scalar current token is zero and stops a completed cascade as soon as it
restores the root token. Both shortcuts are exact refinements of the formal
target-arrival identity and retain a same-binary oracle.
The newest return stage similarly bypasses the bounded per-frame cleanup when
the post-return active-frame bitset is zero. Early returns with live frames
retain the complete release path, while coherent final returns avoid up to W
zero-mask ready-resume regions.
The latest scheduler stage removes a selected divergent binary child's
redundant ready-record push and immediate LIFO pop. Normal pops and these
children share one PC route and one dispatch switch; a measured state-slot
gate keeps the refinement out of smaller kernels where it is not profitable.
The current stage then coalesces move-related Schedule PHI state slots when an
exact per-lane liveness/interference proof shows that their logical lifetimes
do not overlap. This removes redundant masked copies and lets LLVM retain the
smaller physical state set in SSA/registers; a same-binary oracle preserves
the previous slot layout.
The newest W16-only stage stores dynamically indexed convergence-frame static
IDs and parent tokens in scalar LLVM arrays, avoiding whole-vector dynamic
updates and reducing scheduler register/stack pressure. W1/W2/W4/W8 keep the
previous vector layout after independent rejection gates.

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
dispatch pipeline 32 times, the current voxel sweep repeats 64 renders, and
the refreshed Spacex sweep renders eight frames after its upload/update
synchronization.
Cutout path tracing uses 64 spp and ordinary path tracing uses 128 spp; both
force one spp per dispatch on both backends to remove a batching asymmetry.
Ordinary path tracing uses seven adjacent fallback/SIMD pairs per width with
reversed order on alternating pairs; the current cutout sweep also uses seven
pairs per width.
The focused triangle-only-provider result uses twelve W8 pairs, while the
other widths use four to six pairs. The refreshed ordinary and voxel processes
keep stable per-backend hashes and use separate gallery conformance runs. The
refreshed 64-spp cutout processes are performance-only; a separate 1024-spp
run supplies its gallery conformance gate. SDF uses its internal four-SPP
throughput metric;
high-SPP SDF image comparison remains a separate conformance gate.
SDF/GEMM cells retain the earlier seven-process sweep. Image processing,
Voxel, and ordinary path tracing are refreshed after the bounded predicated-
loop stage with seven balanced-order fallback/W1/W2/W4/W8/W16 rounds. Each
image process repeats its four-dispatch pipeline 32 times, each Voxel process
uses 64 render iterations, and each path process uses 128 one-spp dispatches.
Every variant uses 32 workers on logical CPUs 0--31. Spacex retains its prior
seven-round, eight-frame sweep because none of its kernels reaches the new
loop candidate.

Speedup is always `fallback time / SIMD time`, or
`SIMD throughput / fallback throughput`, so values above one are wins.

## Current fallback-relative results

| Workload and metric | fallback | W1 | W2 | W4 | W8 | W16 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| non-coro SDF, samples/s | 8.705 | 8.197 (0.942x) | 9.476 (1.089x) | 15.112 (1.736x) | 22.568 (2.593x) | 32.959 (3.786x) |
| image pipeline, ms/iteration | 10.908 | 18.328 (0.592x) | 9.799 (1.110x) | 6.906 (1.567x) | 5.311 (2.039x) | 4.504 (2.400x) |
| voxel render, ms/iteration | 8.874 | 9.165 (0.951x) | 14.786 (0.588x) | 8.246 (1.050x) | 5.114 (1.692x) | 3.486 (2.482x) |
| Spacex, ms/frame | 162.421 | 125.778 (1.289x) | 64.295 (2.517x) | 34.030 (4.783x) | 18.655 (8.668x) | 11.684 (13.738x) |
| ordinary path tracing, fixed 1 spp/dispatch, spp/s | 66.550 | 59.591 (0.894x) | 50.560 (0.761x) | 65.354 (0.975x) | 75.699 (1.133x) | 78.775 (1.177x) |
| cutout path tracing, fixed 1 spp/dispatch, spp/s | 72.030 | 49.567 (0.692x) | 32.925 (0.465x) | 40.872 (0.575x) | 45.488 (0.642x) | 45.757 (0.642x) |
| portable GEMM, GFLOP/s | 64.895 | 23.332 (0.360x) | 25.627 (0.395x) | 115.914 (1.786x) | 190.521 (2.936x) | 316.449 (4.876x) |

The GEMM row is a compute diagnostic rather than a graphics result. It uses
eight explicit SIMD workers and seven independent process medians; every
process performs seven timed samples of 128 complete 256-by-256 dispatches and
validates the output against double-precision accumulation. The fallback
process medians ranged from 41.594 to 88.326 GFLOP/s under shared-host load,
while the SIMD distributions were tight. Its relative speedups must therefore
be treated as host observations, not cross-machine constants.

The refreshed Voxel cells are the medians of the seven balanced rounds.
Parenthesized values are the preferred geometric means of the within-round
fallback/SIMD ratios. Their 95% paired log-space Student-t intervals at
W1/W2/W4/W8/W16 are [0.9166, 0.9867], [0.5637, 0.6135],
[1.0021, 1.1011], [1.6075, 1.7817], and [2.3514, 2.6191]. W8 and W16 win all
seven rounds; W4 wins six, and W1/W2 lose all seven. Every fallback
process retains SHA-256
`27455a0e126ecfae23d592a58121751c5884a69d9d7388b20195e8b0a121829a`,
and every SIMD process at all five widths retains
`6172183a6c96704ffa48a6b64d30afcf2a3921431507dc40e3f80f1ae1362e4b`.
Independent gallery comparisons pass at 48.08 dB RGB PSNR for fallback and
82.83 dB for SIMD. The two backends use different floating-point paths and are
not expected to produce identical PNG bytes; same-backend refinement/oracle
outputs below are byte-identical.

The refreshed image-pipeline paired 95% intervals are [0.5515, 0.6347],
[1.0349, 1.1903], [1.4516, 1.6908], [1.8937, 2.1956], and
[2.2131, 2.6023] from W1 through W16. W4/W8/W16 win all seven pairs, W2 wins
six, and W1 loses all seven. Every fallback image has SHA-256
`dc3bb32fe870f5c64d4be16e39b92eb1db5670721884907c0fe9387c98daf7b5`;
every SIMD width has
`73d7aa39c1d17b2f2be073f91e5c4615e9233e58fbf673a195b7e43cc43baa31`.
These kernels report zero predicated-loop batches, so this is a refreshed
control workload rather than a gain attributed to the new loop transform.

The current path-tracing rows are paired rather than independent medians because
unrelated host tasks moved the load average during the sweeps. For ordinary
path tracing the displayed fallback cell is the pooled median and each SIMD
cell is its seven-process median. The preferred geometric means of adjacent
SIMD/fallback ratios are 0.8939x/0.7611x/0.9747x/1.1329x/1.1770x from W1
through W16; their paired 95% log-space Student-t intervals are
[0.8750, 0.9133], [0.7469, 0.7756], [0.9634, 0.9861], [1.1064, 1.1601], and
[1.1447, 1.2102]. W8 and W16 win all seven pairs; the other widths lose all
seven. Every width/backend retains one stable output hash across its seven
processes, and separate gallery runs supply correctness conformance.

The final-binary cutout row uses the same seven-pair method. It remains below
fallback at every width: 0.6924x/0.4649x/0.5746x/0.6418x/0.6420x, with paired
95% intervals [0.6821, 0.7029], [0.4532, 0.4769], [0.5631, 0.5863],
[0.6258, 0.6582], and [0.6279, 0.6563]. The displayed throughput cells are
process medians; the ratios are the preferred paired geometric means. Its
JIT-side query payload crossings and sparse cohorts remain the dominant
unresolved deficit.

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

### W8 six-instruction material-ladder boundary

The next Voxel material-selection diamond is not admitted by globally raising
the cost limit. A second pass accepts only one empty arm opposite three scalar
Boolean equality tests and three `float3` selects, with one differing
`float3` PHI, the existing metadata/totality rules, six instructions total,
four live-out register units, and cost nineteen. It runs only at W8 after the
existing deep refinement. Its same-binary oracle is
`LUISA_SIMD_DISABLE_WIDE_PREDICATED_IF_REFINEMENT=1`, and the runtime reports
`predicated_wide_select_ladder_diamonds`.

Twenty-eight alternating pairs used 128 Voxel renders per process, 32 workers,
and logical CPUs 0--31. Oracle time divided by candidate time has a paired
geometric mean of 1.00527x, 17/28 wins, and a log-space 95% Student-t interval
of [1.00073, 1.00983]. Candidate/oracle medians are 773.521/776.306 ms per
process, or 6.043/6.065 ms per render. The first and second fourteen-pair
halves measure 1.00677x and 1.00376x, so the direction does not depend on one
half of the run.

| W8 Voxel kernel | Existing deep policy | Six-instruction layer |
| --- | ---: | ---: |
| Schedule blocks / convergence points | 29 / 9 | 26 / 8 |
| state slots / coalesced slots | 38 / 12 | 37 / 11 |
| assembly text bytes | 113,797 | 105,727 |
| static / vector instructions | 2,443 / 1,212 | 2,273 / 1,160 |
| branches / calls | 277 / 3 | 243 / 3 |
| stack references / frame | 520 / 3,648 B | 507 / 3,520 B |
| scalar-math calls | 2 | 2 |

Five alternating 256-render counter pairs give candidate/oracle changes of
-0.77% cycles, -0.84% task clock, -0.027% retired instructions, -0.28%
branches, -3.96% branch misses, and -1.13% L1 loads. L1 load misses increase
5.57%, so the retained 0.5% wall-time gain is specifically a control/front-
end trade and not an across-the-board cache improvement. The final restricted
implementation's assembly and object are byte-identical to the measured
prototype.

The permanent regression covers W2/W4/W8/W16, a thirteen-element inactive
tail, one W8-only counter hit with six hoisted instructions, exact execution,
and byte-identical non-W8 assembly. A final candidate/oracle gallery pair
passes at 82.834519 dB and shares SHA-256
`6172183a6c96704ffa48a6b64d30afcf2a3921431507dc40e3f80f1ae1362e4b`.
Ordinary and cutout path tracing, image processing, non-coroutine SDF, Spacex,
and game of life report identical candidate/oracle optimization counts; this
stage claims no benefit for those examples.

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
replaced wholesale by a separate fifteen-round run. The W16 scalar-frame
checkpoint independently refreshes Mandelbrot, masked stream, AoS-to-SoA, and
analytic path tracing with thirty alternating rounds; direct-CFG GEMM retains
its unchanged prior measurement. Every variant is pinned to logical CPUs
0--31. The table reports the paired geometric mean and 95% log-space Student-t
interval for `ISPC / Luisa SIMD` at the same semantic width; values above one
mean ISPC is faster:

| Workload | W4 AVX2 | W4 AVX-512 | W8 AVX2 | W8 AVX-512 | W16 AVX-512 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Mandelbrot | 0.928x [0.904, 0.953] | 0.921x [0.904, 0.938] | 1.011x [0.989, 1.033] | 0.985x [0.967, 1.003] | 1.078x [1.068, 1.087] |
| masked stream | 1.000x [0.954, 1.047] | 1.000x [0.958, 1.043] | 1.071x [1.032, 1.111] | 1.066x [1.029, 1.104] | 1.093x [1.076, 1.111] |
| AoS to SoA | 0.991x [0.939, 1.045] | 0.960x [0.892, 1.033] | 1.107x [1.015, 1.206] | 1.075x [1.036, 1.115] | 1.053x [1.033, 1.072] |
| GEMM | 0.754x [0.743, 0.765] | 0.750x [0.745, 0.754] | 0.755x [0.743, 0.767] | 0.762x [0.749, 0.776] | 0.801x [0.779, 0.825] |
| analytic path trace | 2.161x [2.103, 2.222] | 2.024x [1.927, 2.126] | 2.130x [2.047, 2.216] | 2.007x [1.899, 2.120] | 2.199x [2.176, 2.222] |

Mandelbrot, masked stream, AoS-to-SoA, and GEMM are bit-identical across all
eight implementations. The asset-free analytic path tracer validates 921,600
floats per implementation with zero tolerance violations; its maximum absolute
and relative errors against Luisa W4 are `1.1921e-7` and `2.7532e-7`.
The Luisa W4/W8/W16 process medians are respectively
708.907/1,173.274/1,755.470 Mitems/s for Mandelbrot,
6,266.961/5,734.358/5,627.028 Mitems/s for masked stream,
2,752.720/2,578.397/2,562.839 Mitems/s for AoS-to-SoA,
263.854/344.693/431.829 GFLOP/s for GEMM, and
871.429/1,157.886/1,315.437 Mitems/s for analytic path tracing.

The balanced intervals matter on this shared host: several small memory-kernel
differences include parity, while GEMM and analytic path tracing remain
unambiguous. Luisa is 25--33% faster than the matched-width ISPC GEMM
variants. Mandelbrot is at parity through W8 and trails ISPC by 7.8% at W16.
The analytic path tracer remains the outlier: the matched ISPC targets are
2.01--2.20x faster. Their process medians are 1,850.996/1,776.445 Mitems/s at
W4 AVX2/AVX-512, 2,380.414/2,372.723 Mitems/s at W8, and 2,902.211 Mitems/s
at W16. All fifteen W4/W8 rounds and all thirty refreshed W16 rounds favor
ISPC. The
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

### ISPC phase-level attribution

The path-trace gap was also decomposed at the pass boundary instead of treating
ISPC `-O2` as one opaque result. The audit used official ISPC 1.31.0 source at
revision `c6adb4f86f5678ce6c41951b1e2b59f727455697` (BSD-3-Clause), precise
arithmetic with FMA disabled, one worker pinned to CPU 6, and eleven or fifteen
alternating processes per variant. Every ISPC ablation retained checksum
`aff56d522e42c9a`. The table reports disabled-feature throughput divided by
normal ISPC throughput:

| Disabled mechanism | AVX2 W4 | AVX2 W8 | AVX-512 W8 | AVX-512 W16 |
| --- | ---: | ---: | ---: | ---: |
| front-end all-on specialization | 0.8518x | 0.7989x | 0.9151x | 0.9030x |
| all three `ImproveMemoryOpsPass` instances | 0.9460x | 0.8918x | 1.0055x | 1.0009x |
| both mechanisms | 0.7817x | 0.7353x | 0.9228x | 0.9111x |

After the direct-buffer lane/value rotation checkpoint, the AVX2 W8
decomposition was repeated against the current Luisa W8 object in fifteen
rotating single-worker process rounds pinned to CPU 6. Each process again
reports the median of seven timed samples. The fresh medians and paired
geometric-mean ratios are:

| Variant | Mitems/s | variant / normal ISPC | variant / current Luisa W8 |
| --- | ---: | ---: | ---: |
| current Luisa W8 | 162.952 | -- | 1.0000x |
| normal ISPC | 231.288 | 1.0000x | 1.4208x [1.4162, 1.4261] |
| ISPC without front-end all-on specialization | 186.384 | 0.8051x [0.8031, 0.8072] | 1.1439x [1.1405, 1.1475] |
| ISPC without all three `ImproveMemoryOpsPass` instances | 206.497 | 0.8897x [0.8863, 0.8927] | 1.2640x [1.2574, 1.2703] |
| ISPC without both | 170.074 | 0.7301x [0.7240, 0.7352] | 1.0374x [1.0293, 1.0449] |

Every pair favored the faster entry shown by its ratio, and every ISPC
variant retained checksum `aff56d522e42c9a`; Luisa retained
`a93089e651f98582`. Removing both mechanisms closes 89.6% of the current
normal-ISPC/Luisa gap, whether measured as the absolute throughput difference
or as the log throughput ratio. This does not mean the transforms are
independent: normal ISPC is 1.370x faster than its doubly ablated object, not
the product of the two single-feature ratios.

The largest single mechanism is therefore not an LLVM pass. It is ISPC's
front-end varying-control emission in `src/stmt.cpp` together with function
all-on/mixed versioning in `src/func.cpp`. It dynamically distinguishes an
all-on incoming mask and all/none/mixed branch results, then emits a path on
which the mask is a compile-time all-on constant. On AVX2 W8, normal ISPC is
1.242x faster than the same compiler with this mechanism disabled; disabling
it increases retired instructions by 45.0% and cycles by 21.9% while branch
misses remain effectively unchanged. In the final public path-trace function,
the ablation grows LLVM PHIs from 167 to 824 and AVX blend calls from 130 to
274; the assembly stack frame grows from 320 to 4,664 bytes. The gain is
therefore mask/state simplification and register residency, not branch
prediction. Disabling the nominally related
`IntrinsicsOpt` phases 215/216/250/251 produces an object byte-identical to
the baseline, which confirms that the gain is created before those backend
passes.

The largest named ISPC pass is `ImproveMemoryOpsPass`, invoked at phases
211, 256, and 271 around `InstCombine`. It contributes 1.124x on AVX2 W8 and
1.057x on AVX2 W4, but does not help either AVX-512 width in this kernel. The
AVX2 W8 hardware-counter ablation increases both retired instructions and
cycles by about 11.5%/11.2%. Disabling both all-on specialization and all
three memory-pass instances is non-additive but still leaves normal ISPC
1.370x faster than the ablated AVX2 W8 object. In the final public function,
the memory-pass ablation changes 80 to 112 `extractelement` operations, 95 to
zero scalar `getelementptr` operations, and zero to 64 `inttoptr` operations.
For this workload the named pass wins primarily by factoring varying pointer
vectors into a common scalar base plus encodable offsets; it is not a hidden
scalar-libm or target-math advantage.

Ordinary LLVM passes are secondary. Disabling phase-227 `LoopFullUnroll` in a
fresh fifteen-pair run measured 0.98919x [0.98851, 0.98986] with zero wins,
or about a 1.1% contribution. Disabling the late GVN measured 0.99351x
[0.99222, 0.99480], and disabling loop unswitching produced an object
byte-identical to normal ISPC. Luisa's O2/O3 and post-O2 cleanup experiments
above were likewise neutral or byte-identical. The evidence therefore rejects
both "one magic LLVM pass" and "LLVM did not optimize hard enough" as the main
explanation.

The residual is still large. In eleven paired single-core rounds, normal
AVX-512 ISPC is 2.133x [2.121, 2.145] faster than Luisa W8 and 2.574x
[2.551, 2.597] faster than Luisa W16. Disabling ISPC all-on specialization
reduces those ratios only to 1.947x [1.938, 1.957] and 2.319x
[2.301, 2.338]; disabling `ImproveMemoryOpsPass` is neutral at these widths.
This remaining difference is consistent with ISPC's structured mask CFG and
register-resident live state versus Luisa's general independent-PC scheduler
frame, not with one omitted target pass.

Two Luisa follow-ups sharpen the implementation boundary. Replacing both
successor-mask reductions by a first-active-lane seed/broadcast comparison
regressed fifteen W8 pairs to 0.8445x [0.8426, 0.8464] despite preserving the
same result, so the existing reductions are not the bottleneck. Collapsing the
four single-incoming forwarding blocks created by widened sphere updates cut
the analytic W8 Schedule from 52 to 48 blocks, state slots from 61 to 49, and
static assembly from 3,080 to 2,990 instructions, but a separate thirty-pair
run was neutral at 1.00049x [0.99749, 1.00350]. Fewer scheduler objects alone
are therefore insufficient. The actionable target is bounded structured
all-on/mixed region versioning or branch splitting that shortens dynamic state
live ranges without speculating expensive `sqrt`/division work or cloning an
entire function.

### Bounded coherent all-on region versioning

The retained follow-up implements that target as one local branch split. On a
runtime-coherent varying arm it first proves that the incoming physical packet
mask is all one. Only that path clones an acyclic arm-to-convergence-to-next-
split chain under a constant all-one mask. Partial tails and genuinely mixed
packets retain the independent-PC scheduler. The finder accepts no more than
four blocks and twenty-four weighted register units, rejects memory, calls,
effects, loops, foreign/same-target ambiguity, and a terminal predicated-
memory diamond, and versions at most one arm in one region per function. This
is independently authored from ISPC's behavior; no ISPC source was copied.

A first broad W8 experiment cloned five regions (17 blocks, 28 instructions).
It increased static assembly from 3,085 to 3,395 instructions, branches from
379 to 432, stack references from 639 to 668, and the frame from 3,136 to
3,648 bytes. Fifteen alternating single-core pairs measured 0.9871x
[0.9821, 0.9921], with one win. That whole-kernel policy was rejected.
Limiting the transform to the first eligible region leaves the analytic W8
miss path at four blocks/eight instructions. Its assembly is 3,152 versus
3,085 instructions, 390 versus 379 branches, 646 versus 639 stack references,
and a 3,232- versus 3,136-byte frame; both objects have zero calls and zero
scalar-math symbols. Despite the small static growth, the constant-mask clone
shortens the dynamically hot scheduler path.

The initial one-region width ablation selected the production widths:

| Width | candidate/oracle speedup | 95% paired CI | wins | decision |
| ---: | ---: | ---: | ---: | --- |
| W2 | 1.0294x | [1.0269, 1.0320] | 15/15 | enable |
| W4 | 0.9932x | [0.9901, 0.9963] | 1/15 | reject |
| W8 | 1.0311x | [1.0296, 1.0326] | 15/15 | enable |
| W16 | 1.0102x | [0.9929, 1.0278] | 7/15 | reject as inconclusive |

After adding the real-workload profitability boundary, the final analytic
path-trace gate used fifteen alternating candidate/oracle processes pinned to
CPU 6. Each process reported the median of seven samples and retained checksum
`a93089e651f98582`:

| Width | speedup | 95% paired log-space CI | wins |
| ---: | ---: | ---: | ---: |
| W2 | 1.0294x | [1.0258, 1.0330] | 15/15 |
| W8 | 1.0343x | [1.0307, 1.0380] | 15/15 |

The real Voxel gate exposed the fixed-cost boundary. Its W8 candidate covered
only two blocks/four instructions and regressed seven 64-render pairs to
0.9826x [0.9778, 0.9875], with zero wins. Production W8 therefore requires at
least three blocks. The same two-block region is profitable at W2 and remains
enabled. The final fifteen-pair, 32-worker Voxel sweep is:

| Width | accepted regions | speedup | 95% paired CI | wins |
| ---: | ---: | ---: | ---: | ---: |
| W2 | 1 (2 blocks, 4 instructions) | 1.0185x | [1.0130, 1.0240] | 15/15 |
| W8 | 0 | 0.9942x | [0.9854, 1.0030] | 6/15 |

All candidate/oracle Voxel images are byte-identical with SHA-256
`6172183a6c96704ffa48a6b64d30afcf2a3921431507dc40e3f80f1ae1362e4b`.
For W8 the candidate/oracle assembly and object are also byte-identical, so
the interval crossing parity is host noise rather than a generated-code
regression. Ordinary Embree path tracing and image processing likewise report
zero regions and have byte-identical candidate/oracle assembly and objects.
Cutout path tracing, SDF, shader toy, procedural rendering, and blackhole were
also audited and report zero regions at W8.

This recovers only a bounded part of ISPC's broader mechanism. A fresh
fifteen-pair same-algorithm comparison after this change still has ISPC ahead
of Luisa W8 by 1.6802x [1.6763, 1.6842] for AVX2 i32x8 and 1.5654x
[1.5627, 1.5681] for AVX-512 x8. The remaining gap is consistent with ISPC's
whole structured mask CFG and register-resident live state; the local Luisa
clone deliberately does not duplicate arbitrary loops, memory regions, or the
complete function.

### Direct-buffer lane/value axis rotation

The next memory stage targets the concrete address shape exposed by the ISPC
audit without importing target intrinsics or source. A proven lane-consecutive
direct typed-buffer operation whose exact element is a two-to-four-component
32-bit vector now rotates the component-major Schedule value to physical AoS
order with generic fixed-vector shuffles. One `<W * S x T>` masked load/store
replaces `D` leaf gathers/scatters, where `D` is the semantic component count
and `S` includes an optional fourth padding slot for a three-component vector.
The expanded mask repeats each active lane over its semantic components and is
false for padding. Byte-address, volatile, bindless, recursive aggregate,
local, accel, and ray-query accesses fail closed. W1 is unchanged, and
`LUISA_SIMD_DISABLE_LANE_AFFINE_BUFFER=1` restores both this rule and the
earlier scalar lane-affine rule.

Permanent IR/JIT regressions cover W1/W2/W4/W8/W16, uint4 candidate/oracle
intrinsic shape, float2/float3/float4 full and sparse cohorts, a 13-element
tail, preserved float3 padding, inactive sentinels, exact output equality,
LLVM verification, and final assembly without gather/scatter. The analytic
path tracer accepts one transposed `Buffer<float4>` output store. Fifteen
rotating single-core candidate/oracle rounds, each process taking the median
of seven samples, measured:

| Width | candidate/oracle | 95% paired CI | wins | candidate/oracle median, Mitems/s |
| ---: | ---: | ---: | ---: | ---: |
| W2 | 1.0227x | [1.0209, 1.0246] | 15/15 | 48.089 / 46.993 |
| W4 | 1.0696x | [1.0676, 1.0717] | 15/15 | 93.712 / 87.512 |
| W8 | 1.1806x | [1.1778, 1.1834] | 15/15 | 165.420 / 139.982 |
| W16 | 1.2454x | [1.2203, 1.2710] | 15/15 | 253.853 / 205.553 |

An independent fifteen-pair W1 identity gate measured 0.99984x
[0.99946, 1.00022]. Every Luisa process retained checksum
`a93089e651f98582`. A first independent fifteen-pair sweep also found all four
enabled widths positive, including 1.1855x at W8 and 1.2587x at W16, so the
selection does not depend on one noisy batch.

The W8 oracle forms four `<8 x i64>` address vectors and emits four
`vscatterqps`. The candidate legalizes its generic shuffles to
`vpermi2ps`/`vpermi2pd` and emits two masked `vmovups` stores. It has no gather,
scatter, call, or scalar-math symbol. Static code is not a useful cost proxy:

| Width | instructions, candidate/oracle | vector instructions | branches | stack references | frame bytes |
| ---: | ---: | ---: | ---: | ---: | ---: |
| W2 | 4,062 / 4,111 | 1,845 / 1,871 | 419 / 435 | 713 / 712 | 3,584 / 3,520 |
| W4 | 3,043 / 3,032 | 1,369 / 1,369 | 355 / 355 | 474 / 470 | 1,552 / 1,488 |
| W8 | 3,167 / 3,152 | 1,556 / 1,552 | 390 / 390 | 650 / 646 | 3,456 / 3,232 |
| W16 | 3,704 / 3,608 | 1,637 / 1,547 | 468 / 474 | 694 / 660 | 3,456 / 2,792 |

The same rotating batch included official ISPC 1.31.0 controls. ISPC remains
faster, but the direct same-width gap is materially smaller than before this
stage:

| Width/target | ISPC / Luisa SIMD | 95% paired CI | wins |
| --- | ---: | ---: | ---: |
| W4 AVX2 | 1.3285x | [1.3267, 1.3303] | 15/15 |
| W4 AVX-512 | 1.4429x | [1.4407, 1.4451] | 15/15 |
| W8 AVX2 | 1.4146x | [1.4126, 1.4167] | 15/15 |
| W8 AVX-512 | 1.3237x | [1.3216, 1.3258] | 15/15 |
| W16 AVX-512 | 1.3549x | [1.3486, 1.3613] | 15/15 |

The prior W8 gaps were 1.6802x/1.5654x. The remaining difference is therefore
not explained by this final AoS store alone; ISPC's structured mask CFG and
register-resident state remain the principal target.

The same analytic workload also makes the fallback distinction explicit.
Fifteen rotating fallback/W1/W2/W4/W8/W16 rounds produced medians of
580.241/43.302/47.971/93.418/164.995/253.879 Mitems/s. Paired SIMD/fallback
geomeans were 0.0714x, 0.0793x, 0.1544x, 0.2727x, and 0.4187x, with respective
95% intervals [0.0656, 0.0778], [0.0729, 0.0863], [0.1419, 0.1680],
[0.2508, 0.2966], and [0.3849, 0.4553]. A 921,600-float fallback/W16 dump
comparison has zero violations at 1e-6 absolute plus 1e-5 relative tolerance;
maximum absolute/relative errors are `5.9605e-8`/`2.0775e-7`. Fallback's scalar
pipeline can be horizontally vectorized without carrying the independent-PC
frame, so fixing output scatter does not erase that scheduler gap.

Real-example applicability was checked separately. W8 image processing,
Voxel, ordinary Embree path tracing, and non-coroutine SDF each report zero
transposed accesses. For every example, candidate/oracle optimized assembly,
objects, and output PNG are byte-identical. These kernels primarily use
`Image`/texture output, so this stage claims no graphics gain; a distinct
fixed-vector image/tile layout remains required.

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

### Empty-frame return cleanup guard

The return transition first removes the terminating cohort from `live` and
`runnable`. When the scalar `frame.active` bitset is then zero, no expected
mask can shrink, no arrived cohort can be released, and all W calls to the
ready-resume helper carry an empty mask. Codegen now tests the bitset once and
skips that complete unrolled cleanup. Nonzero bitsets enter the original path,
so early return and nested convergence semantics are unchanged.
`LUISA_SIMD_DISABLE_RETURN_FRAME_GUARD=1` is the same-binary oracle and
`return_frame_guards` reports guarded return sites.

The profiler exposed this after the preceding token guard: the W8 analytic
path tracer's hottest remaining samples included the eight repeated
`ready.overflow.resume` regions emitted by its final return. The optimized
machine path is one `testb` plus a branch around them. Fifteen alternating
single-worker candidate/oracle processes pinned to CPU 6 measured the
following validated throughput. Intervals are paired 95% log-space Student-t
intervals, and every process retained checksum `a93089e651f98582`:

| Width | speedup | 95% paired CI | wins | candidate/oracle median, Mitems/s |
| ---: | ---: | ---: | ---: | ---: |
| W2 | 1.1644x | [1.1595, 1.1692] | 15/15 | 45.336 / 38.998 |
| W4 | 1.2437x | [1.2414, 1.2460] | 15/15 | 87.429 / 70.167 |
| W8 | 1.3182x | [1.3138, 1.3226] | 15/15 | 135.401 / 102.472 |
| W16 | 1.5227x | [1.4872, 1.5592] | 15/15 | 201.095 / 133.397 |

W1 uses direct CFG and emits no guard. At W2/W4/W8/W16 the candidate adds only
6/4/5/7 static instructions and 2/2/2/3 branches respectively; stack-frame
sizes remain 3,520/1,488/3,136/2,792 bytes. All objects contain zero calls and
zero scalar-math symbols. Five external W8 `perf stat` pairs, conservatively
including JIT compilation and teardown, measured candidate/oracle geometric
ratios of 0.8766 cycles, 0.8265 instructions, 0.9084 branches, and 0.9980 branch
misses. The unchanged miss count confirms that deleted frame/worklist work,
not improved prediction, supplies the gain.

The same mechanism reaches real non-Embree graphics kernels. With sixteen
workers pinned to CPUs 0--15, ten alternating W8 Voxel processes at 64 renders
measured 1.0277x [1.0160, 1.0394] with 10/10 wins; candidate and oracle PNGs
were byte-identical with SHA-256
`6172183a6c96704ffa48a6b64d30afcf2a3921431507dc40e3f80f1ae1362e4b`.
Ten four-spp SDF pairs measured 1.0174x [1.0037, 1.0312] with 8/10 wins and
byte-identical PNG SHA-256
`beb9deeb53ac4c05ea68bcbec26426987262f0b920eb155bdd330396b1cefc6a`.
The SDF W2 follow-up measured 1.0140x [1.0087, 1.0194] with 7/7 wins; W4 and
W16 intervals included one under concurrent-host load and are not used for a
claim. Image processing's five kernels all take direct CFG, report zero
guards, and are unchanged.

Embree-dominated paths are neutral rather than regressed. Ten alternating W8
ordinary 64-spp, one-spp-per-dispatch pairs measured 1.0014x
[0.9865, 1.0165] with 5/10 wins and byte-identical PNGs. Eight analogous
32-spp cutout pairs measured 1.0008x [0.9882, 1.0136] with 3/8 wins and
byte-identical PNGs. Those intervals deliberately make no traversal-speed
claim. Longer Voxel sweeps at other widths were contaminated by a concurrent
Blender render; no sample was removed and those noisy ratios are excluded.

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

### Liveness-proven PHI state-slot coalescing

Schedule IR keeps logical PHI versions distinct, but the prior LLVM lowering
also gave every version a distinct fixed-vector alloca. The current lowering
builds the exact Schedule CFG, treats each edge assignment as one member of a
parallel PHI copy, and solves per-lane backwards liveness to a fixed point:

```text
live(edge) = uses(edge) union (live_in(target) - defs(edge))
```

Move-related state values may share one physical alloca only when they have
the same nonempty source name, value class, and LLVM storage type, and no pair
in the proposed groups interferes. The name is only a bounded profitability
and provenance hint; the liveness relation is the safety proof. Destinations
also interfere with every other parallel copy's source so source-order
emission cannot clobber an unread value. If a matching source and destination
share a slot, their masked copy is an identity and is omitted.

This is a storage refinement, not a Schedule semantic change. Divergent
cohorts parked at different CFG locations occupy disjoint physical lanes, so
per-lane liveness is the relevant relation for one fixed-vector slot. Logical
`state_slots` remains unchanged, while `coalesced_state_slots` reports removed
physical slots. W1 and coherent direct CFG are byte-identical. Setting
`LUISA_SIMD_DISABLE_STATE_PHI_COALESCING=1` restores the old allocation and
copy path in the same binary.

The analytic path tracer eliminates 31/29/29/31 physical slots at
W2/W4/W8/W16. At W8 the final body falls from 3,468 to 3,080 instructions,
1,901 to 1,545 vector instructions, 917 to 639 stack references, and a
4,992-byte to 3,136-byte frame; branches fall from 383 to 377. Both objects
have zero calls and zero scalar-math calls. After coalescing, the compact
physical state set remains promotable instead of applying the old cold-slot
volatile pinning policy; a separate 21-pair analytic gate and seven-pair Voxel
gate found that reapplying pinning was consistently slower.

Fifteen alternating single-core candidate/oracle processes, each taking the
median of seven timed dispatches, retain checksum `a93089e651f98582`:

| Width | speedup | 95% paired CI | wins | candidate/oracle median, Mitems/s |
| ---: | ---: | ---: | ---: | ---: |
| W2 | 1.1083x | [1.0990, 1.1177] | 15/15 | 39.177 / 35.597 |
| W4 | 1.1618x | [1.1595, 1.1641] | 15/15 | 70.436 / 60.705 |
| W8 | 1.0825x | [1.0787, 1.0863] | 15/15 | 103.631 / 95.529 |
| W16 | 1.1575x | [1.1522, 1.1629] | 15/15 | 131.681 / 113.486 |

The real Voxel kernel eliminates twelve of 38 logical state slots at W8. Seven
alternating 64-render candidate/oracle processes on 32 workers produce
byte-identical PNGs at every width:

| Width | speedup | 95% paired CI | wins | candidate/oracle median, ms |
| ---: | ---: | ---: | ---: | ---: |
| W2 | 1.1202x | [1.1070, 1.1336] | 7/7 | 13.570 / 15.126 |
| W4 | 1.1613x | [1.1521, 1.1706] | 7/7 | 7.542 / 8.724 |
| W8 | 1.1226x | [1.1127, 1.1327] | 7/7 | 6.024 / 6.778 |
| W16 | 1.1633x | [1.1493, 1.1775] | 7/7 | 4.686 / 5.440 |

Ordinary Embree path tracing improves at W4 by 1.0232x
[1.0066, 1.0401], while W8 and W16 are statistically neutral at 0.9972x
[0.9801, 1.0145] and 1.0109x [0.9951, 1.0270]. Cutout path tracing is neutral
at W4/W8 and has a small W16 gain: 1.0066x [0.9985, 1.0148], 1.0076x
[0.9967, 1.0186], and 1.0108x [1.0033, 1.0185]. All candidate/oracle images
are byte-identical within each workload and width. Image processing remains
direct CFG, while SDF and Spacex have no eligible move chain; their final
objects are unchanged. The refreshed fallback and ISPC tables above include
this retained state layout.

### W16 scalar convergence-frame metadata

The scheduler's dynamically indexed `frame.static.id` and
`frame.parent.token` fields previously used `<W x i32>` allocas at every
width. LLVM lowers a W16 dynamic update through whole-vector
extract/insert/broadcast operations and may keep several 512-bit values live
through the dispatcher. The retained W16 policy instead uses `[16 x i32]`
arrays and scalar GEP/load/store. Active-frame and zero-token checks already
provide a valid or sanitized scalar index; frame masks, targets, tokens, and
the formal transition are unchanged. The same-binary vector oracle is
`LUISA_SIMD_DISABLE_SCALAR_FRAME_METADATA=1`, and the runtime report exposes
`scalar_frame_metadata`.

This is deliberately W16-only. Fifteen alternating single-core analytic
candidate/oracle processes rejected the scalar layout at W4 and W8:

| Width | candidate/oracle throughput | 95% paired CI | wins |
| ---: | ---: | ---: | ---: |
| W4 | 0.9609x | [0.9580, 0.9638] | 0/15 |
| W8 | 0.9885x | [0.9815, 0.9956] | 3/15 |

The same W16 experiment was positive. An independent thirty-pair run measured
1.0129x [1.0014, 1.0245] with 27/30 wins; candidate/oracle medians were
132.602/130.439 Mitems/s and every result retained checksum
`a93089e651f98582`. Its exact optimized entry body changes as follows:

| Analytic W16 entry | vector oracle | scalar metadata |
| --- | ---: | ---: |
| static instructions | 3,752 | 3,601 |
| vector instructions | 1,739 | 1,546 |
| static branches | 467 | 471 |
| stack references | 709 | 657 |
| stack allocation | 7,104 B | 2,792 B |
| calls / scalar-math calls | 0 / 0 | 0 / 0 |

The four-extra-branch result is another reason not to select this refinement
from static size alone: reduced dynamic vector updates and register/stack
pressure, rather than branch count, determine its W16 gain.

Real examples were measured with the same binary and alternating oracle order.
Ten W16 Voxel pairs, each repeating 256 renders on 32 workers, measured
1.0284x [1.0139, 1.0432] with 10/10 wins and candidate/oracle medians of
4.976/5.073 ms. Seven ordinary Embree path-tracing pairs at 128 spp were
neutral-positive at 1.0032x [0.9935, 1.0130] with 5/7 wins and medians of
77.371/77.110 spp/s. Seven 64-spp cutout pairs measured 1.0101x
[1.0014, 1.0189] with 5/7 wins and medians of 43.605/43.131 spp/s. The Voxel
main entry shrinks from 3,237 to 3,124 instructions, 1,512 to 1,355 vector
instructions, 651 to 646 stack references, and a 7,872-byte to 4,800-byte
frame; its three calls and two scalar-math calls are unchanged. These gates
justify W16 while the explicit W4/W8 rejection prevents a blanket scalar-state
policy.

A later final-binary fallback refresh ran while the machine's one-minute load
average was 17.5 and is retained as a conservative contention snapshot rather
than replacing the balanced all-width table above. Seven adjacent W16/fallback
pairs measured Voxel at 1.1552x [1.1393, 1.1713] with 7/7 wins, ordinary path
tracing at 1.0623x [1.0522, 1.0725] with 7/7 wins, and cutout at 0.6900x
[0.6683, 0.7124] with 0/7 wins. Their W16 medians were 6.094 ms,
77.615 spp/s, and 45.160 spp/s respectively. The Voxel absolute time is 22%
slower than the 4.976-ms candidate median in the isolated candidate/oracle
gate, confirming that the shared-host epoch is not interchangeable with the
earlier all-width distribution. Both epochs nevertheless agree on the signs:
W16 beats fallback on Voxel and ordinary path tracing, while sparse cutout
query traffic remains slower than fallback.

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
5. Generalize the completed fixed-vector `BYTE1` linear/mirror sampler only
   where a real workload supplies a nonzero hit count. Other address modes,
   formats, mip paths, or a tile/swizzle upload boundary each require their own
   semantics and stable A/B gate; preserve row-major public image semantics.
6. Extend the completed direct-buffer lane/value rotation across bounded
   coherent affine tiles so a profitable layout can remain live across several
   operations. Divergent control, warp operations, and externally visible
   lane-wise effects continue to pin lane identity.
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
packet to the existing grouped texture callback. At this checkpoint base
extents shared the then-16-byte texture descriptor by packing three twenty-bit
values beside the sampler code. The later direct `BYTE1` stage expands that
backend-local descriptor to 24 bytes while retaining the callback ABI. Varying
dependencies use fixed-vector math; the exact W8 JIT
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

## IR-native BYTE1 sampling

The next Spacex profile made the remaining boundary explicit. W8 spent 68.89%
of sampled cycles in the runtime `LINEAR_POINT` texture template; its portable
host compilation contained no `-march=native`, and the 22,754-byte filter
instance lowered mirror-coordinate work and four byte taps to scalar SSE over
all eight lanes. This was not scheduler or pool overhead: the host
`parallel_for` contribution was negligible.

Production now versions the common varying uniform-slot 2D `BYTE1`, mip-zero,
stored `LINEAR_POINT`/`MIRROR` operation directly in JIT IR. The 24-byte
backend-local descriptor publishes a raw pointer only for `BYTE1`; all other
formats and query forms keep the callback. Inactive lanes and NaN/Inf are made
benign before conversion and gather. Public row-major layout is unchanged.
W4/W8 additionally replace byte gathers with alignment-one 32-bit gathers only
after a packet-wide proof that every tap ends at least four bytes before the
allocation boundary. The last three bytes, small textures, W1/W2/W16, and the
disabled oracle retain narrow gathers.

The two transformations were measured separately in fifteen alternating W8
pairs at sixteen frames per process:

| W8 Spacex A/B | enabled ms/frame | oracle ms/frame | paired speedup | wins |
| --- | ---: | ---: | ---: | ---: |
| fixed-vector IR vs complete callback | 21.955 | 51.172 | 2.330x | 15/15 |
| proven wide gather vs narrow IR gather | 18.851 | 21.901 | 1.162x | 15/15 |

Every pair produced byte-identical output and passed the gallery reference at
70.185 dB. The wide-gather width audit used seven pairs and measured
narrow/wide ratios of 0.9698x/0.9815x/1.0275x/1.1585x/1.0074x for
W1/W2/W4/W8/W16. W1 and W2 were stable regressions; W16 won only 5/7 for a
0.74% geometric-mean change. The production cost model therefore enables it
only at W4/W8.

The final production sweep rotated and reversed fallback plus all five SIMD
widths over seven processes, eight frames each:

| Spacex ms/frame | fallback | W1 | W2 | W4 | W8 | W16 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Median | 162.421 | 125.778 | 64.295 | 34.030 | 18.655 | 11.684 |
| Paired speedup | 1.000x | 1.289x | 2.517x | 4.783x | 8.668x | 13.738x |

All widths won 7/7 pairs and retained one deterministic hash per variant.
Three 32-frame `perf stat` repetitions give:

| W8/Spacex counter | direct IR | callback oracle | fallback |
| --- | ---: | ---: | ---: |
| retired instructions | 130.74 B | 545.79 B | 1,655.94 B |
| cycles | 90.65 B | 247.46 B | 788.80 B |
| branches | 8.19 B | 49.47 B | 176.45 B |
| aggregate task-clock | 17.70 s | 48.60 s | 156.39 s |

Thus direct IR removes 4.17x callback instructions and 6.04x callback
branches. In a new cycle profile the former callback no longer received a
sample; `parallel_for` was 0.07%, while scalar uniform math became visible at
the top of the named symbols. Final W8 assembly contains YMM `vroundps`, mask
registers, and native gathers on this Ryzen host. W8 remains a semantic width:
LLVM may select AVX2 or another legal target sequence elsewhere.

The permanent runtime oracle compares stored direct sampling against the
explicit-sampler callback at W1/W2/W4/W8/W16, including mirror-domain inputs,
NaN/Inf, extreme finite coordinates, and a 35-thread inactive tail. The ORC
test requires direct, callback, pre-gather sanitization, and masked-gather IR.
Use
`LUISA_SIMD_DISABLE_IR_BYTE1_TEXTURE_SAMPLING=1` for the full callback oracle
or `LUISA_SIMD_DISABLE_WIDE_BYTE1_GATHERS=1` for narrow direct gathers.

A fresh post-change gallery sweep also passed image processing and Voxel at
every width (89.252 dB and 82.835 dB respectively). The 1024-spp Embree path
tracer passed W1/W2/W4/W8/W16 at
35.427/42.782/40.940/39.219/37.802 dB, and every process reported native
Embree W4/W8/W16 packet support.

### Rejected innermost-loop frame specializations

The W8 Voxel schedule has one parentless natural loop (`l0`, header `bb2`) and
a shared exit `bb5`. Four static convergence points (`c1`, `c2`, `c4`, and
`c7`) target `bb5`; the enclosing `c0` targets `bb8`. This is a useful stress
case for the proposal to reduce push/pop work only on innermost loops. Every
prototype below was compared with a same-binary environment-variable oracle,
used 16 workers pinned to CPUs 0--15, alternated candidate/oracle order, and
required the identical SHA-256 output
`6172183a6c96704ffa48a6b64d30afcf2a3921431507dc40e3f80f1ae1362e4b`.

The experiments establish three distinct costs:

| W8 Voxel prototype | candidate/oracle throughput | decisive counter evidence |
| --- | ---: | --- |
| header-exit parking plus direct body edge | 0.9418x [0.9315, 0.9523], 0/15 wins | code layout and branch behavior regressed |
| parking through the shared scheduler route | 0.9923x [0.9826, 1.0021], 5/15 wins | instructions 0.9958x, branches 0.9873x, but branch misses 1.0443x and cycles 1.0051x |
| fast reuse branch around the header declaration | 0.9971x [0.9883, 1.0059], 8/15 wins | instructions 1.000002x and cycles 1.0001x |
| separate loop mask collector, cascade first | wall clock neutral/noisy | cycles 1.0133x, instructions 1.0073x, branch misses 1.0248x |
| global dynamic-target guard | 0.9883x [0.9826, 0.9941], 2/15 wins | cycles 1.0146x, instructions 1.0855x, branches 1.0496x |
| scalar cached target carried in a separate ready field | 0.9959x [0.9847, 1.0073], 6/15 wins | cycles 0.9933x but instructions 1.0485x; no stable wall-clock win |
| target packed into existing token bits | 0.9966x [0.9872, 1.0061], 5/15 wins | cycles 1.0010x, instructions 1.0340x |
| collector before the generic cascade | 0.9936x [0.9841, 1.0033], 5/15 wins | cycles 1.0106x and branch misses 1.0240x; also invalid as a general shared-target transform because cohorts can carry different inner tokens |
| header branch splitting plus collector | 0.9456x [0.9324, 0.9589], 0/15 wins | cycles 1.0495x, instructions 1.0729x, branches 1.0155x |
| eager frame hoisting to the loop entry | 0.7477x [0.7394, 0.7561], 0/12 wins | cycles 1.3673x, instructions 1.2494x, branches 1.1303x |

The apparently smaller collector object was especially misleading: one
branch-split form removed 37 static instructions, 58 vector instructions, two
branches, and 18 stack references, yet retired 7.29% more dynamic instructions
and ran 4.95% more cycles. The existing convergence frame already aggregates
`c1` in the same cascade loop that processes `c2/c4/c7`; a second collector
duplicates hot aggregation work. Conversely, eager hoisting destroys the
important lazy rule that a frame is created only on the first actually mixed
header evaluation. Many Voxel iterations remain dynamically coherent, so an
eager `c1` adds a live parent layer to inner frames even when no header split
needs it.

No production code or diagnostic knob from these experiments is retained.
The current policy remains lazy-on-mixed declaration, top-frame reuse, and a
single generic arrival cascade. A future loop specialization must preserve all
three properties, distinguish the shared `bb5` token chain, and demonstrate a
stable real-render win rather than relying on static object size.

### Guarded dynamic-trip loop unswitch

The XIR loop-unswitch pass now handles an unknown-trip canonical top-tested
loop without evaluating its invariant varying selector on zero-trip lanes. It
clones the pure initial header condition into an entry guard, substitutes every
header PHI with its preheader input, and sends only entering lanes to the two
specialized loop versions. Exit PHIs and direct live-outs receive a separately
resolved guard incoming value. `LUISA_SIMD_DISABLE_GUARDED_LOOP_UNSWITCH=1`
is the differential oracle, and `guarded_unswitched_loops` reports acceptance.

The permanent W2/W4/W8/W16 regression covers zero, short, and longer per-lane
trip counts plus an inactive tail and exact oracle equality. The current image,
Voxel, Spacex, SDF, and ordinary path-tracing kernels all report zero guarded
unswitched loops, so this checkpoint is a semantic capability expansion rather
than a real-example performance claim. The profitable real DDA result below
comes from the separate non-cloning LLVM batch.

### Bounded predicated innermost-loop batch

The accepted follow-up preserves those three rules by changing a larger unit.
Instead of adding a second exit collector around the existing state machine,
it takes one finite, pure, innermost loop out of the per-iteration PC machine.
Removing annotated backedges leaves an acyclic Schedule region; LLVM executes
that region in topological order under one mask per block, carries a next-mask
PHI, and accumulates one mask PHI per natural-loop exit. A proven bound of `N`
body iterations permits `N + 1` header evaluations, including the final false
test. The loop body is emitted once: there is no recursive ISPC-style all-on/
mixed clone tree.

At the one batch boundary, codegen counts nonempty dynamic destinations. One
header/exit destination continues without a frame. Two or more destinations
recreate the original header convergence once and enqueue the continuation and
exit cohorts under that token. A matching top frame is reused. This is why the
transformation handles the Voxel loop's four exits and shared `bb5` token chain
without any of the rejected collector shortcuts. The permanent regression adds
a post-loop `warp_active_sum`, multiple early-exit destinations, a 13-lane W16
tail, and inactive NaNs before `fptoui`; candidate, disabled oracle, and scalar
reference are exact. Writes and volatile reads independently reject.

The production finder accepts one loop, 6--24 blocks, at most 96 audited
instructions, and a nonzero upper bound no larger than 4096. Only nontrapping
arithmetic/select/compare, casts, and direct nonvolatile typed-buffer reads are
allowed. Every result, state assignment, and read index is varying or a mask.
Nested loops, external non-header entries, undeclared joins, calls, local
pointers, division/remainder/shifts, bindless/texture/accel operations,
collectives, writes, atomics, volatile reads, returns, and barriers fail closed.
Inactive gather indices and float-to-integer operands are sanitized before the
LLVM operation, not after its result.

This policy is target- and parallelism-aware. LLVM TTI must report at least a
512-bit fixed-vector register and a legal non-scalarized masked gather. W16 is
profitable even with one worker; W8 is selected only at 24 or more device
workers. `LUISA_SIMD_DISABLE_PREDICATED_LOOP=1` is the same-binary oracle, and
`LUISA_SIMD_FORCE_PREDICATED_LOOP=1` bypasses profitability only. The Voxel
runtime report is W8 `12 blocks / 40 instructions / 257 header evaluations`
and W16 `13 / 40 / 257`; ordinary path tracing and image processing report
zero accepted loops.

The worker-count crossover used five alternating forced/oracle pairs per cell:

| Voxel width | workers | forced/oracle throughput | 95% paired CI | wins |
| ---: | ---: | ---: | ---: | ---: |
| W8 | 1 | 0.9666x | [0.9544, 0.9788] | 0/5 |
| W8 | 8 | 0.9938x | [0.9733, 1.0148] | 2/5 |
| W8 | 16 | 0.9762x | [0.9527, 1.0003] | 0/5 |
| W8 | 24 | 1.1698x | [1.1409, 1.1995] | 5/5 |
| W8 | 32 | 1.2588x | [1.2268, 1.2916] | 5/5 |
| W16 | 1 | 1.0982x | [1.0912, 1.1052] | 5/5 |
| W16 | 8 | 1.1079x | [1.0891, 1.1270] | 5/5 |
| W16 | 16 | 1.1118x | [1.0887, 1.1355] | 5/5 |
| W16 | 24 | 1.2892x | [1.2638, 1.3150] | 5/5 |
| W16 | 32 | 1.3917x | [1.3805, 1.4029] | 5/5 |

Independent longer 32-worker gates measured W8 at 1.2619x
[1.2506, 1.2734] over seven pairs and W16 at 1.3797x
[1.3599, 1.3998] over ten pairs; every pair won. Forced W2 was inconclusive at
1.0097x [0.9979, 1.0217], and W4 was neutral at 0.9950x
[0.9834, 1.0067], so neither is enabled. Every candidate/oracle PNG is byte-
identical with SHA-256
`6172183a6c96704ffa48a6b64d30afcf2a3921431507dc40e3f80f1ae1362e4b`.

Three-repeat `perf stat` runs over 256 renders quantify the mechanism:

| width/event | generic oracle | predicated batch | reduction |
| --- | ---: | ---: | ---: |
| W8 cycles | 231.623 B | 185.434 B | 19.94% |
| W8 instructions | 503.470 B | 244.689 B | 51.40% |
| W8 branches | 78.443 B | 17.295 B | 77.95% |
| W8 branch misses | 471.575 M | 43.221 M | 90.84% |
| W8 cache misses | 74.928 M | 74.494 M | 0.58% |
| W16 cycles | 175.126 B | 126.507 B | 27.76% |
| W16 instructions | 349.223 B | 161.613 B | 53.72% |
| W16 branches | 53.465 B | 13.889 B | 74.02% |
| W16 branch misses | 220.105 M | 47.772 M | 78.30% |
| W16 cache misses | 74.655 M | 77.771 M | -4.18% |

Cache traffic is effectively unchanged at W8 and slightly worse at W16. The
win is the removed dynamic PC/frame/branch work, not improved locality. This
also explains why static size predicts the wrong sign:

| final Voxel entry | W8 oracle | W8 batch | W16 oracle | W16 batch |
| --- | ---: | ---: | ---: | ---: |
| assembly bytes | 106,093 | 119,101 | 150,759 | 165,183 |
| static instructions | 2,280 | 2,527 | 3,129 | 3,413 |
| vector instructions | 1,162 | 1,246 | 1,357 | 1,514 |
| branches | 244 | 262 | 376 | 393 |
| stack references | 510 | 599 | 646 | 733 |
| stack allocation | 3,520 B | 3,520 B | 4,800 B | 4,864 B |
| calls / scalar-math calls | 3 / 2 | 3 / 2 | 3 / 2 | 3 / 2 |

Both objects have only `sincosf` unresolved; those two calls are pre-existing
uniform scalar camera math. The varying loop contains fixed-vector gathers and
no extract/call/insert scalar-libm loop. The final fallback-relative Voxel
table at the top rises to 1.692x for W8 and 2.482x for W16 in the current
32-worker shared-host epoch.

### Rejected cross-query early gather

The ordinary W8 path tracer has one tempting memory/compute-overlap site. Its
material-buffer index and direct varying read originally follow the shadow
`trace_any` callback by four Schedule instructions. A prototype moved the
two-instruction address/read slice before that read-only acceleration query,
without crossing a write, volatile operation, other resource read, mask
change, or block boundary. Candidate and disabled-oracle outputs were byte-
identical in every run.

The motion reached final machine code: the material `vpgatherqq`/`vgatherqps`
pair moved from after the Embree callback to roughly 53 assembly lines before
its indirect call. A distance sweep from four through 64 Schedule instructions
found that distances 4--12 all collapsed to the same best static object. The
oracle and that object were:

| Ordinary W8 main entry | original order | early gather, distance 4--12 |
| --- | ---: | ---: |
| assembly bytes | 193,787 | 192,598 |
| static instructions | 3,542 | 3,506 |
| vector instructions | 2,297 | 2,272 |
| branches | 273 | 271 |
| stack references | 916 | 906 |
| stack allocation | 6,720 B | 6,720 B |
| calls / scalar-math calls | 5 / 0 | 5 / 0 |

Static size was misleading. Fifteen alternating 128-spp, one-spp-per-dispatch
pairs for a longer 32-instruction motion measured only 0.9988x
[0.9935, 1.0041] candidate/oracle throughput with 8/15 wins. The best static
distance was also neutral in ten alternating 256-spp single-dispatch pairs at
1.0064x [0.9935, 1.0195] with 5/10 wins. Finally, eight alternating 512-spp
hardware-counter pairs measured 0.9979x [0.9903, 1.0056] throughput with only
2/8 wins. It retired 0.1955% fewer instructions in all eight pairs, but used
0.6431% more cycles in all eight; the paired cycle ratio was 1.0064x
[1.0023, 1.0106].

This CPU gather is a value-producing, synchronous instruction rather than an
asynchronous prefetch. Pulling it across the callback extends its live range
and issues gather work earlier, but does not create profitable memory-level
parallelism around Embree on this host. The prototype was therefore removed;
no early-buffer-read pass or diagnostic environment variable is retained.
Future overlap work must use a separately discardable prefetch with a bounded,
sanitized address or batch enough independent rays to amortize the callback,
and must still pass the same final-object and paired-performance gates.

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

The PHI state-slot coalescing stage again completed both full Release builds,
the required four-test native-math/fallback-math/runtime-width/Schedule-codegen
gate, the SIMD-only suite (129/129), and the complete SIMD+fallback suite
(140/140). Formatting and diff checks pass, as do the seven standalone ISPC
driver tests. The standalone compiler control validates all eight variants,
runs the first four workloads for seven balanced rounds, and replaces the
noisier analytic path row with a separate fifteen-round sweep. The permanent
W1/W2/W4/W8/W16 regression compares the disabled
oracle for every active-tail length, exercises a cross-source parallel PHI
swap, and verifies direct-W1 assembly identity.
Fresh all-width Voxel and image-processing galleries pass at 82.83 and
89.25 dB. Ordinary 1024-spp path tracing passes at
35.43/42.78/40.94/39.22/37.80 dB from W1 through W16; W8 cutout and
non-coroutine SDF pass at 39.58 and 63.13 dB. Both path tracers report Embree
4.4.1 W4/W8/W16 native packets enabled. A fresh W8 dump contains five JIT
objects, reports zero scalar-math calls for every kernel, and has no unresolved
scalar f32 libm symbol; the backend dynamically references all
`rtcIntersect4/8/16` and `rtcOccluded4/8/16` packet entries. All gallery and
object artifacts remain in temporary directories.

The W16 scalar-frame-metadata stage completed fresh full builds of both Release
trees, the required native-math/fallback-math/runtime-width/Schedule-codegen
gate (4/4), the SIMD-only suite (129/129), and the complete SIMD+fallback suite
(140/140). The seven standalone-driver tests, clang-format, diff, and clangd
syntax checks also pass. The permanent W1/W2/W4/W8/W16 regression covers all
tail sizes from zero through W, nested same-target convergence, early return,
exact candidate/oracle outputs, byte-identical W1/W2/W4/W8 IR and assembly,
and the required W16 array/vector code-shape distinction. A corrected W16
candidate/oracle gallery rerun selected width with `LUISA_SIMD_WARP_WIDTH=16`:
Voxel passes at 82.834519 dB, ordinary 1024-spp Embree path tracing at
37.801767 dB, and 1024-spp cutout at 43.890250 dB. Each candidate/oracle PNG
pair is byte-identical. Both path tracers report native Embree 4.4.1
W4/W8/W16 packet support enabled. A thirty-round standalone W16/ISPC refresh
supplies the current table above; the ISPC executable and generated objects
remain outside CMake and the repository.

The W8 six-instruction material-ladder stage again completed fresh full builds
of both Release trees, the required native-math/fallback-math/runtime-width/
Schedule-codegen gate (4/4), the SIMD-only suite (129/129), and the complete
SIMD+fallback suite (140/140). The predicated-if and loop-unswitch standalone
performance gates remain positive, the standalone ISPC driver tests pass 7/7,
and formatting, diff, and clangd syntax checks pass for every changed C++
translation unit. Final W8 reference comparisons pass Voxel at 82.834519 dB,
ordinary 1024-spp Embree path tracing at 39.219375 dB, 1024-spp cutout at
44.167099 dB, and non-coroutine SDF at 63.129346 dB. Both path tracers report
native Embree 4.4.1 W4/W8/W16 packet support enabled. No gallery reference was
regenerated or modified.

The empty-frame return-cleanup stage completed a fresh full Release build, the
required native-math/fallback-math/runtime-width/Schedule-codegen gate (4/4),
`unit_simd` (11/11), `integration_simd` (26/26), and the complete configured
suite (140/140). Formatting, diff, and clangd syntax checks pass for every
changed C++ translation unit. Its permanent W2/W4/W8/W16 differential JIT
regression covers both the guarded and unconditional forms, every active-tail
length, early return with live nested frames, final return after release,
inactive sentinels, LLVM verification, and exact output equality. Fresh W1/W2/
W4/W8/W16 Voxel and image-processing galleries pass at 82.834519 and
89.251953 dB. Ordinary 1024-spp Embree path tracing passes at
35.426800/42.781833/40.940546/39.219375/37.801767 dB from W1 through W16;
W8 cutout and non-coroutine SDF pass at 44.167099 and 63.129346 dB. Both path
tracers report native Embree 4.4.1 W4/W8/W16 packet support, and the final
backend dynamically references every `rtcIntersect4/8/16` and
`rtcOccluded4/8/16` entry. Outputs remained in a temporary directory and no
gallery reference was regenerated or modified.

The bounded coherent-all-on-region stage completed a fresh full Release build,
the required native-math/fallback-math/runtime-width gate (3/3), its focused
Schedule-codegen executable, the combined SIMD/XIR/runtime/graphics gate
(88/88), and the complete configured suite (140/140). Clang-format and diff
checks pass for every changed C++ translation unit. The permanent regression
covers W2/W4/W8/W16, coherent and mixed entry conditions, full/partial/one-lane
masks, inactive sentinels, the W8 three-block profitability minimum, and the W2
two-block exception. Fresh W1/W2/W4/W8/W16 Voxel and image-processing gallery
comparisons pass at 82.834519 and 89.251953 dB. Ordinary 1024-spp Embree path
tracing passes at 35.426800/42.781833/40.940546/39.219375/37.801767 dB from
W1 through W16; W8 cutout and non-coroutine SDF pass at 44.167099 and
63.129346 dB. Both path tracers report native Embree 4.4.1 W4/W8/W16 packet
support. Outputs remained in `/tmp`, and no gallery reference was regenerated
or modified.

The direct-buffer lane/value-axis-rotation stage completed another fresh full
Release build, the required native-math/fallback-math/runtime-width gate (3/3),
and the complete configured SIMD+fallback/XIR/runtime/graphics suite (140/140).
Clang-format dry-run and per-translation-unit clangd syntax checks pass for
every changed C++ source, and the syntax-check runner's Python suite passes
13/13. Fresh W1/W2/W4/W8/W16 image-processing and Voxel gallery comparisons
pass at 89.251953 and 82.834519 dB. Ordinary 1024-spp Embree path tracing
passes at 35.426795/42.781582/40.940376/39.219305/37.801771 dB from W1 through
W16; W8 cutout and non-coroutine SDF pass at 39.576002 and 63.129346 dB.
Every path-tracing process reports Embree 4.4.1 native W4/W8/W16 packet support.
The graphics kernels report zero transposed buffer accesses and retain their
prior objects, so these runs are correctness gates rather than a graphics-speed
claim for this buffer-only optimization. Outputs remained in `/tmp`, and no
gallery reference was regenerated or modified.

The guarded-unswitch and bounded-predicated-loop stage completed a fresh full
Release build, the required native-math/fallback-math/runtime-width gate plus
the focused Schedule and XIR regressions (6/6), and the complete configured
SIMD+fallback/XIR/runtime/graphics suite (140/140). Clang-format, diff checks,
the syntax-check runner's Python suite (13/13), and per-translation-unit clangd
checks pass for all fifteen changed C++ sources. The permanent differential
regressions cover W2/W4/W8/W16 guarded zero-trip behavior and a W16 bounded
batch with multiple exits, a 13-lane tail, post-loop collective reconvergence,
inactive NaNs before `fptoui`, and exact disabled-oracle equality.

Fresh W1/W2/W4/W8/W16 image-processing and Voxel gallery comparisons pass at
89.251953 and 82.834519 dB. Ordinary 1024-spp Embree path tracing passes at
35.426795/42.781582/40.940376/39.219305/37.801771 dB from W1 through W16 and
reports native Embree 4.4.1 W4/W8/W16 packet support. Fallback passes the same
three galleries at 100.000000, 48.080453, and 62.223429 dB. Applicability scans
confirm that only Voxel reaches the predicated-loop candidate; image processing,
ordinary path tracing, Spacex, blackhole, SDF, and cutout report zero accepted
batches. All outputs and captured objects remained outside the repository.

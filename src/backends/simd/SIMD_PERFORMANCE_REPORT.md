# SIMD CPU backend performance report

Snapshot date: 2026-08-20. This report covers the Release build after merging
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
The current ray-query control stage if-converts only the exact W8 cutout
instance-filter ladder inside an already active surface handler. Static
aggregate extracts receive an independent bounds/type proof; query traversal,
candidate payload reads, commits, termination, and memory remain under the
ordinary scheduler. A broader whole-query-loop prototype and an extra select-
factoring experiment were rejected by real-renderer measurements.
The newest ray-query stage also consumes canonical inline DSL traversal without
first constructing generic XIR CFG. The `$while` frontend retains its original
condition as non-semantic provenance alongside the historical explicit guard;
AST-to-XIR transactionally proves an exact `$while (query.proceed())` plus
candidate dispatch and emits `RayQueryLoopInst` directly. Serialized,
provenance-free, and diagnostic-oracle ASTs retain fail-closed reconstruction.
SIMD then applies the same selective pipeline lowering: W1/W2 accept every
capture-eligible handler; W4/W8/W16 require either 24 handler instructions or
two eligible query sites in the function. This keeps small single-query
handlers on the scheduler while preserving the real W4 procedural and W8
two-query cutout pipelines. W1 uses the resident provider. Front-end callable
argument/return scratch is forwarded before capture classification, and
non-canonical explicit control remains an ordinary loop.
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
The latest local-control stage recognizes one-sided expensive arithmetic
diamonds inside innermost loops, guards their nonempty arm with `any(mask)`,
and keeps their assignments in one LLVM emission region. W4/W8/W16 may also
fuse a bounded sequence of adjacent diamonds; W8 may absorb one exact nested
assignment tail. This avoids repeated convergence/dispatcher round trips and
Schedule spills without cloning an all-on/mixed loop body.
The current extension keeps a bounded single-entry tail after the last local
diamond in that same emission region. Because the diamond has already restored
its complete incoming cohort, the tail is neither speculated nor cloned. The
ordinary path tracer now avoids one additional merge-to-dispatch boundary and
keeps two more cross-block values in SSA; a width/instruction cost gate rejects
the measured short W4 regression.
Voxel now supplies the complementary two-sided case: both arithmetic arms stay
behind their own nonempty-mask guard and reconverge without a scheduler frame.
This removes 37% of the W8 kernel's dynamic instructions while retaining one
source copy and exact candidate/oracle output.
The initial packet-batch runtime stage moves the serial packet loop for one
block across the JIT boundary. W8 on the recorded wide-register host inlines
one body into one loop; W16 initially unrolled only its sixteen-call shell for
the common 256-thread block. W2/W4 retain a compact dynamic wrapper. The
ordinary packet body is internal, so no object carries two exported
implementations.
The current launch stage makes linear 1D tails explicit, removes redundant
unit-dimension masks/thread decomposition, propagates valid alias facts to the
mutable wrappers, and permits bounded W16 inlining for one small straight-line
shape. A fail-closed block-agnostic proof may concatenate a worker's 1D block
range; multidimensional and block/thread-sensitive kernels keep the generic
loop.
The current state-layout stage then colors compatible non-move scheduler state
roots under the existing exact per-lane interference proof. Production is
limited to W16 schedules with at least 32 logical slots and retains the result
only when at least two additional physical slots disappear. This reduces the
analytic, Voxel, and cutout W16 frames while rolling the neutral one-slot
ordinary-path opportunity back to byte-identical code.

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
dispatch pipeline 32 times, the current voxel sweep repeats 128 renders, and
the refreshed Spacex sweep renders eight frames after its upload/update
synchronization.
Cutout path tracing uses 64 spp and forces one spp per dispatch. Historical
ordinary-path optimization ablations use 64 or 128 spp with that same fixed
batching. The current fallback-relative ordinary row instead uses one fully
synchronized 32-spp dispatch in seven rotated rounds; final FPS is read only
after stream synchronization, so fallback's asynchronous submission is not
mistaken for execution time. It confines both backends to 30 logical CPUs
`0-12,14-28,30-31`; SIMD uses 30 workers and fallback's default pool remains
inside that same affinity. The current cutout sweep uses seven pairs per width.
The focused triangle-only-provider result uses twelve W8 pairs, while the
other widths use four to six pairs. The refreshed ordinary and voxel processes
keep stable per-backend hashes and use separate gallery conformance runs. The
refreshed 64-spp cutout processes are performance-only; a separate 1024-spp
run supplies its gallery conformance gate. SDF uses its internal four-SPP
throughput metric;
high-SPP SDF image comparison remains a separate conformance gate.
SDF/GEMM cells retain the earlier seven-process sweep. Image processing keeps
the bounded-predicated-loop seven-round sweep. Voxel is refreshed after
general W16 state coloring with 128 renders per process, seven alternating
rounds, and the same 30-logical-CPU affinity used by current ordinary path
tracing; SIMD uses 30 workers. Ordinary path tracing is refreshed after
terminal-bridge absorption with seven balanced-order fallback/W1/W2/W4/W8/W16
rounds under the synchronized method above. Spacex retains its prior seven-
round, eight-frame sweep because none of its kernels reaches the new local
candidate.

Speedup is always `fallback time / SIMD time`, or
`SIMD throughput / fallback throughput`, so values above one are wins.

## Current fallback-relative results

| Workload and metric | fallback | W1 | W2 | W4 | W8 | W16 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| non-coro SDF, samples/s | 8.705 | 8.197 (0.942x) | 9.476 (1.089x) | 15.112 (1.736x) | 22.568 (2.593x) | 32.959 (3.786x) |
| image pipeline, ms/iteration | 10.908 | 18.328 (0.592x) | 9.799 (1.110x) | 6.906 (1.567x) | 5.311 (2.039x) | 4.504 (2.400x) |
| voxel render, ms/iteration | 6.938 | 8.924 (0.776x) | 13.363 (0.517x) | 5.913 (1.168x) | 4.977 (1.393x) | 3.375 (2.047x) |
| Spacex, ms/frame | 162.421 | 125.778 (1.289x) | 64.295 (2.517x) | 34.030 (4.783x) | 18.655 (8.668x) | 11.684 (13.738x) |
| ordinary path tracing, synchronized 32-spp dispatch, FPS | 83.868 | 79.252 (0.955x) | 62.834 (0.748x) | 81.722 (0.982x) | 93.845 (1.126x) | 93.961 (1.132x) |
| cutout path tracing, fixed 1 spp/dispatch, spp/s | 70.505 | 48.396 (0.678x) | 33.634 (0.497x) | 40.715 (0.605x) | 45.781 (0.680x) | 44.360 (0.659x) |
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
W1/W2/W4/W8/W16 are [0.7656, 0.7857], [0.5095, 0.5254],
[1.1535, 1.1825], [1.3658, 1.4202], and [2.0270, 2.0678]. W4/W8/W16 win all
seven rounds, and W1/W2 lose all seven. Every fallback
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

The final-binary cutout row uses seven paired rounds after the direct
structured-query stage. W1 was repeated separately after its production gate;
the other widths use one balanced six-configuration rotation. It remains below
fallback at every width: 0.6779x/0.4973x/0.6045x/0.6800x/0.6587x, with paired
95% intervals [0.6636, 0.6924], [0.4761, 0.5194], [0.5892, 0.6203],
[0.6560, 0.7049], and [0.6320, 0.6864]. The displayed fallback cell is the
pooled median of the two fallback sweeps; throughput cells are per-width
process medians, while the ratios are the preferred paired geometric means.
Direct handlers reduce query scheduling work but provider/Embree crossings and
sparse cohorts remain the dominant unresolved deficit.

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

### General W16 state-slot graph coloring

Move provenance is a useful profitability hint but is not required by the
per-lane noninterference proof. The next stage therefore degree-ranks the
physical roots left by move-constrained coalescing and greedily combines
compatible noninterfering groups. Compatibility requires the same value class,
Luisa type, local-lvalue status, and allocated LLVM type. Production restricts
this general coloring to W16 schedules with at least 32 logical state slots and
rolls the complete second stage back unless it removes at least two additional
slots. `LUISA_SIMD_DISABLE_GENERAL_STATE_COLORING=1` retains only the move
stage; `LUISA_SIMD_FORCE_GENERAL_STATE_COLORING=1` bypasses the performance
gates for differential tests.

The current analytic W16 entry removes three additional slots. Its exact
optimized object changes as follows against the same-binary disabled oracle:

| Analytic W16 entry | general coloring disabled | production coloring |
| --- | ---: | ---: |
| logical / all coalesced / generally colored slots | 51 / 31 / 0 | 51 / 34 / 3 |
| object bytes | 22,200 | 21,624 |
| static instructions | 2,655 | 2,571 |
| vector instructions | 1,230 | 1,153 |
| branches | 275 | 276 |
| stack references | 509 | 448 |
| stack allocation | 2,816 B | 2,304 B |
| calls / scalar-math calls | 15 / 0 | 15 / 0 |

The W8 production and disabled objects and assembly are byte-identical. The
permanent W1/W2/W4/W8/W16 regression executes every tail size from zero
through W, compares exact output bits and inactive sentinels, force-exercises
the narrower widths, and includes parallel live ranges with many compatible
colors. A separate W16 graph can remove exactly one slot when forced;
production restores its parent map and emits byte-identical oracle assembly.

Real-example W16 gates used alternating same-binary processes:

| Workload | extra slots | candidate/oracle | 95% paired CI | wins | medians |
| --- | ---: | ---: | ---: | ---: | ---: |
| Voxel, 128 renders, 30 workers | 2 | 1.0207x | [1.0126, 1.0288] | 7/7 | 3.558 / 3.635 ms |
| cutout path, 64 spp | 2 | 1.0092x | [0.9977, 1.0207] | 6/7 | 42.077 / 41.986 frame/s |
| ordinary path, 128 spp, pre-gate | 1 | 0.9984x | [0.9927, 1.0041] | 2/7 | neutral |

The ordinary-path result selected the two-slot retention threshold. Its final
production object now rolls back to the disabled layout; no neutral one-slot
change remains. Candidate/oracle images are byte-identical for Voxel, cutout,
and ordinary path tracing. Voxel production retains 45 logical slots, 21 total
coalesced slots, and two general-coloring slots; cutout reports 50/9/2.

The standalone ISPC driver also fixed a two-variant ordering defect: rotating
and reversing each individual round canceled for two entries and always ran
the same implementation first. It now emits exact `A/B`, `B/A` alternation;
over two complete cycles every implementation occupies every position twice
and every pair precedes each other equally often. Unit tests enforce both
properties. Each analytic path sample contains 256 complete dispatches.

With official ISPC 1.31.0, one worker pinned to physical CPU 6, precise math,
FMA contraction disabled, and thirty alternating W16 process rounds, the final
production binary is slightly but statistically faster:

| Analytic path W16 | median throughput | paired result |
| --- | ---: | ---: |
| Luisa SIMD W16 | 343.651 Mitems/s | 25/30 wins |
| ISPC `avx512skx-x16` | 341.347 Mitems/s | Luisa/ISPC 1.0056x [1.0030, 1.0081] |

The median-throughput ratio is 1.0067x; the paired geometric mean above is the
primary statistic. This is a same-algorithm compiler control, not an ISPC
implementation of the repository's full Embree renderer. It also is not a
blanket Luisa win: a separate fifteen-round W8 run remains behind
`avx512skx-x8`, 196.742 versus 218.066 Mitems/s, or ISPC/Luisa 1.1088x
[1.1075, 1.1101]. The established coherent GEMM result remains in Luisa's
favor.

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

1. Close the remaining analytic W8 gap with a bounded innermost-loop
   continuation refinement. The accepted header proof removes one partition
   but W8 is still 10.29% behind matched AVX-512 x8 ISPC. The next experiment
   should keep the current loop-local mask/token/target in SSA across a proven
   no-suspension trace and enter the generic worklist only at a real mixed
   split or exit. It must retain epoch-separated post-loop convergence, avoid
   whole-loop all-on/mixed cloning, cap static code growth, and beat the
   disabled oracle in final-object counters and repeated analytic runs.
2. Generalize the accepted innermost-loop local regions beyond the completed
   two-sided arithmetic form through bounded branch splitting and pure code
   motion: hoist or sink total single-use operations to expose a safe read
   subregion, and fold compatible
   `select(f(a), f(b), c)` shapes to `f(select(a, b, c))` only after proving
   domain and floating-point equivalence. Voxel now has one profitable two-
   sided hit and path tracing has the earlier one-sided/terminal hit; image
   processing is already branchless direct CFG. The next gate is another real
   nonzero site, exact inactive-tail/oracle equality, a register-pressure-aware
   cost model, and a stable real-example gain beyond those existing sites.
3. Extend the completed within-read W8 leaf pairing to compatible nearby
   gathers only when they share base, dynamic offset, scale, and mask. Cap the
   scan window to control register pressure and stop at any possible write.
   Separately narrow known constant prefix-tail masks. Both need inactive-
   address and final-assembly gates; W4/W16 remain disabled until independently
   profitable.
4. Extend the accepted local aggregate promotion to the remaining ray-query
   payload only through a provider-native packet/SoA representation or a
   larger host/state-boundary elimination, following the liveness/frame
   principles merged from `next`. Both a wrapper-side second scan and a fused
   two-field payload cache are measured and rejected on real graphics; the
   accepted provider-native status publication removes the full status scan,
   while the state-handle cache covers pointers only.
5. Compact or rebatch sparse ray cohorts before Embree and reduce the remaining
   JIT-side ray-query state crossings. The accepted triangle-only host provider
   removes surface-runtime bookkeeping but does not compact lanes; inlining
   Embree LLVM IR is exploratory and cannot replace this measured scheduler
   work.
6. Generalize the completed fixed-vector `BYTE1` linear/mirror sampler only
   where a real workload supplies a nonzero hit count. Other address modes,
   formats, mip paths, or a tile/swizzle upload boundary each require their own
   semantics and stable A/B gate; preserve row-major public image semantics.
7. Extend the completed direct-buffer lane/value rotation across bounded
   coherent affine tiles so a profitable layout can remain live across several
   operations. Divergent control, warp operations, and externally visible
   lane-wise effects continue to pin lane identity.
8. Add software prefetch only for proven affine lookahead with a stable A/B.
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

### Innermost-loop local predication and bounded chaining

This stage targets a smaller unit than the complete predicated loop above. A
one-sided, pure arithmetic diamond in an innermost loop is emitted under its
arm mask, with an `any(mask)` branch around a nonempty 4--24-instruction arm.
Assignment-only diamonds become masked updates directly. A bounded chain keeps
the diamonds and their pure bridges in one LLVM emission region, so values that
were live only across the local scheduler boundary remain in SSA/registers.
There is no loop clone or recursive all-on/mixed version tree.

The same-algorithm analytic path tracer is the stress case. Candidate and
`LUISA_SIMD_DISABLE_LOCAL_PREDICATED_REGIONS=1` were alternated as independent
single-worker processes pinned to CPU 4. Twelve pairs were used at W2/W4/W8;
W16 was rerun for fifteen pairs. Every checksum was
`a93089e651f98582`.

| width | local candidate | generic oracle | candidate/oracle | 95% paired CI | wins |
| ---: | ---: | ---: | ---: | ---: | ---: |
| W2 | 52.918 Mitems/s | 48.209 Mitems/s | 1.0977x | [1.0958, 1.0995] | 12/12 |
| W4 | 100.596 Mitems/s | 93.393 Mitems/s | 1.0771x | [1.0718, 1.0825] | 12/12 |
| W8 | 192.076 Mitems/s | 165.503 Mitems/s | 1.1606x | [1.1553, 1.1658] | 12/12 |
| W16 | 329.738 Mitems/s | 255.340 Mitems/s | 1.2914x | [1.2833, 1.2995] | 15/15 |

Chaining was isolated from the already enabled individual-diamond lowering.
It regressed W2, so production chaining starts at W4. The retained widths were
positive in every pair:

| width | chained / individual diamonds | 95% paired CI | wins | policy |
| ---: | ---: | ---: | ---: | --- |
| W2 | 0.9557x | [0.9497, 0.9617] | 0/10 | disabled |
| W4 | 1.0134x | [1.0112, 1.0156] | 10/10 | enabled |
| W8 | 1.0262x | [1.0249, 1.0275] | 15/15 | enabled |
| W16 | 1.0400x | [1.0332, 1.0470] | 10/10 | enabled |

Absorbing one exact nested assignment tail was then measured independently
over twelve pairs. W8 improved by 1.0092x [1.0070, 1.0114] with 12/12 wins.
W4 measured 0.9936x [0.9906, 0.9967], and W16 measured 0.9968x
[0.9950, 0.9986], so only W8 retains nested-tail chaining. Standalone nested
local predication remains available at W2/W4/W8/W16.

The final analytic objects quantify the removed scheduler state. Counts below
compare production local lowering with the complete local-region oracle; both
objects contain zero calls and zero scalar-math calls.

| width | state slots | instruction spills | static instructions | branches | stack refs | stack frame |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| W2 candidate/oracle | 52 / 63 | 9 / 20 | 3,033 / 4,062 | 289 / 419 | 490 / 713 | 2,752 / 3,584 B |
| W4 candidate/oracle | 50 / 61 | 9 / 20 | 2,098 / 3,043 | 201 / 355 | 364 / 474 | 992 / 1,552 B |
| W8 candidate/oracle | 50 / 61 | 9 / 20 | 1,965 / 3,167 | 192 / 390 | 420 / 650 | 2,304 / 3,456 B |
| W16 candidate/oracle | 52 / 63 | 9 / 20 | 2,647 / 3,704 | 298 / 468 | 487 / 694 | 2,304 / 3,456 B |

The production W4/W8/W16 objects contain one chain. W8 fuses four transitions
including the nested tail; W4/W16 fuse three ordinary transitions. W2 keeps
six independent local diamonds plus one nested region. Convergence guards fall
from 10 to 9 at W2, 9 to 5 at W4, 8 to 3 at W8, and 10 to 6 at W16. This is
direct evidence that the gain comes from fewer PC/frame/branch operations and
better register residency, rather than a different math implementation.

The broadened expensive-arm marker also reaches a real renderer. Ordinary
Embree path tracing has one 19-instruction bounce-loop diamond containing
floating division and `max`, but no square root. Its W8 main entry changes as
follows:

| W8 ordinary path main entry | generic oracle | local candidate |
| --- | ---: | ---: |
| Schedule state slots / spills | 37 / 23 | 34 / 20 |
| static instructions / vector instructions | 3,566 / 2,313 | 3,424 / 2,241 |
| branches / stack references | 276 / 933 | 255 / 894 |
| stack frame | 6,720 B | 6,592 B |
| calls / scalar-math calls | 5 / 0 | 5 / 0 |

Those five calls are unchanged packet/runtime boundaries, including Embree;
the varying math path contains no scalar libm symbol. Embree 4.4.1 reports
native W4/W8/W16 packet traversal enabled. With 64 spp, fifteen workers pinned
to physical CPUs `0-12,14-15`, and alternating candidate/oracle processes, the
real end-to-end result is:

| width | candidate time | oracle time | candidate/oracle throughput | 95% paired CI | wins |
| ---: | ---: | ---: | ---: | ---: | ---: |
| W2 | -- | -- | 1.0065x | [0.9978, 1.0152] | 6/8 |
| W4 | 1,055.874 ms | 1,081.575 ms | 1.0243x | [1.0118, 1.0370] | 9/10 |
| W8 | 927.919 ms | 954.522 ms | 1.0287x | [1.0108, 1.0469] | 10/12 |
| W16 | 877.638 ms | 896.903 ms | 1.0219x | [1.0104, 1.0336] | 10/10 |

Candidate and oracle W8 PNGs are byte-identical, SHA-256
`a7fcc9444150d91ab667964435992d962099c65545fc6ce52891b3847109cbf2`.
Voxel, all five image-processing kernels, non-coroutine SDF, shader toy, game
of life, Mandelbrot, masked stream, AoS-to-SoA, and GEMM report zero local,
nested, or chained regions; this stage makes no throughput claim for them.

For a current fallback-relative renderer checkpoint, both backends were
confined to the same 30 logical CPUs
`0-12,14-28,30-31`, excluding a separately loaded physical core and its SMT
sibling. SIMD used 30 workers; fallback retained its default 32-thread pool
inside that affinity. Five rotated rounds used one synchronized 32-spp dispatch.
The final synchronized FPS is required here: fallback dispatch is asynchronous,
so the pre-synchronize per-dispatch timer is not a valid fallback measurement.

| backend/width | median FPS | mean FPS | throughput / fallback | 95% paired CI |
| --- | ---: | ---: | ---: | ---: |
| fallback | 83.868 | 83.317 | 1.0000x | -- |
| SIMD W1 | 79.252 | 79.542 | 0.9547x | [0.9421, 0.9674] |
| SIMD W2 | 62.834 | 62.352 | 0.7480x | [0.7187, 0.7786] |
| SIMD W4 | 81.722 | 81.846 | 0.9824x | [0.9706, 0.9943] |
| SIMD W8 | 93.845 | 93.815 | 1.1261x | [1.1132, 1.1390] |
| SIMD W16 | 93.961 | 94.348 | 1.1324x | [1.1158, 1.1493] |

W1 is a separate JIT/runtime pipeline, not fallback with width set to one;
fallback's scalar task loop may also be horizontally vectorized by LLVM. W2
does not amortize scheduler and packet-call overhead for this workload. W8 and
W16 combine the local scheduler reductions with native Embree packets and are
currently 12.6% and 13.2% faster than fallback under this synchronized batch
method. W4 is now within about 1.8% of fallback but its interval remains below
parity.

Finally, twelve balanced single-worker rounds compared the same analytic path
algorithm with official ISPC 1.31.0 controls:

| Luisa/ISPC pair | Luisa Mitems/s | ISPC Mitems/s | Luisa/ISPC | 95% paired CI |
| --- | ---: | ---: | ---: | ---: |
| W4 / AVX2 x4 | 100.649 | 123.199 | 0.8185x | [0.8165, 0.8204] |
| W4 / AVX-512 x4 | 100.694 | 134.058 | 0.7523x | [0.7500, 0.7547] |
| W8 / AVX2 x8 | 192.367 | 233.409 | 0.8245x | [0.8231, 0.8260] |
| W8 / AVX-512 x8 | 192.289 | 216.439 | 0.8885x | [0.8872, 0.8898] |
| W16 / AVX-512 x16 | 339.596 | 340.800 | 0.9939x | [0.9891, 0.9986] |

W16 is now about 0.6% behind the matched ISPC x16 control; W8 remains about
11.2% behind AVX-512 x8 and 17.6% behind AVX2 x8. The residual gap is therefore
no longer an order-of-magnitude independent-PC penalty. The next profitable
control work should generalize local regions through bounded branch splitting
and pure code motion, with a register-pressure-aware cost model, instead of
copying ISPC's whole-loop all-on/mixed versioning.

### Local-region terminal bridge

The follow-up extends the accepted local region past its final exclusive
merge. Once both arms have executed, the original cohort is restored; up to
four single-entry innermost-loop blocks and 96 instructions can therefore stay
in the same LLVM emission region without speculation or cloning. The last
ordinary terminator returns to the complete scheduler. This removes one
merge-to-dispatch boundary and lets values defined in the tail remain SSA
until the next real suspension point. The same-binary oracle is
`LUISA_SIMD_DISABLE_LOCAL_PREDICATED_TERMINAL_BRIDGE=1`.

The first broad width ablation demonstrated why this needs an instruction-
and width-aware gate. Ten alternating analytic path-trace pairs measured W2
at 1.0012x [0.9969, 1.0055], W4 at 0.9440x [0.9422, 0.9459], W8 at 1.0073x
[1.0046, 1.0101], and W16 at 1.0404x [1.0259, 1.0552]. Production therefore
leaves W2 disabled and requires at least 32 terminal instructions at W4;
W8/W16 retain bounded short tails. After that gate, W2 and W4 analytic objects
report zero terminal blocks, while eight fresh pairs measure W8 at 1.0030x
[1.0004, 1.0057] and W16 at 1.0419x [1.0237, 1.0605]. Every checksum remains
`a93089e651f98582`.

The real ordinary path tracer supplies the complementary large-tail case. Its
one accepted local hit-update diamond converges into an 81-instruction bounce-
loop block. Candidate and oracle used 64 spp, one spp per dispatch, fifteen
workers pinned to physical CPUs `0-12,14-15`, alternating process order, and
identical output:

| width | terminal candidate/oracle | 95% paired CI | wins |
| ---: | ---: | ---: | ---: |
| W4 | 1.0282x | [1.0169, 1.0397] | 10/10 |
| W8 | 1.0184x | [1.0120, 1.0249] | 9/10 |
| W16 | 1.0096x | [1.0016, 1.0177] | 17/25 |

The W4 result validates the 32-instruction exception rather than a blanket
width policy. W16 required 25 pairs because unrelated work made its shorter
initial interval noisy. W2 remained diagnostic-only after six real pairs were
inconclusive at 1.0081x [0.9806, 1.0363].

Final machine code shows that the transform removes state rather than merely
moving labels:

| ordinary path main entry | W4 candidate/oracle | W8 candidate/oracle | W16 candidate/oracle |
| --- | ---: | ---: | ---: |
| state slots | 32 / 34 | 32 / 34 | 32 / 34 |
| instruction spills | 18 / 20 | 18 / 20 | 18 / 20 |
| static instructions | 3,179 / 3,347 | 3,276 / 3,424 | 3,971 / 4,189 |
| vector instructions | 2,091 / 2,203 | 2,135 / 2,241 | 2,520 / 2,683 |
| branches | 210 / 221 | 251 / 255 | 288 / 299 |
| stack references | 773 / 838 | 834 / 894 | 980 / 1,059 |
| stack frame | 3,192 / 3,256 B | 6,464 / 6,592 B | 10,368 / 10,624 B |
| calls | 5 / 5 | 5 / 5 | 5 / 5 |
| scalar-math calls | 0 / 0 | 0 / 0 | 0 / 0 |

Three-repeat W8 `perf stat` over 128 spp records 170.359 B versus 172.722 B
cycles (-1.37%), 454.422 B versus 459.871 B instructions (-1.18%), and 35.581
B versus 35.994 B branches (-1.15%). Cache misses fall 1.30%; branch misses
are neutral at 1.0005x of the oracle. This directly attributes the end-to-end
gain to reduced scheduler/spill work. Voxel, all image-processing kernels,
cutout path tracing, SDF, and the other audited graphics examples report zero
terminal bridges, so their generated paths are unchanged.

### Two-sided innermost local predication

Voxel's traversal loop contains a varying diamond whose two arms both do real
work: the W8 Schedule form has three instructions on one side and eleven on
the other, followed by a one-instruction merge/split block. The earlier local
recognizer rejected it solely because neither arm was empty. The extension
accepts two single-predecessor arm chains in the same innermost loop when their
combined 4--24 instructions are limited to audited pure arithmetic and
static/bitwise casts. Each arm is still guarded by `any(arm_mask)`, so this is
bounded branch splitting rather than speculative evaluation or whole-loop
all-on/mixed cloning. Off-arm inputs to `fptosi`/`fptoui` are selected to zero
before conversion.

Eight alternating candidate/oracle pairs used 256 renders, fifteen workers,
and physical CPUs `0-12,14-15`. The oracle was
`LUISA_SIMD_DISABLE_TWO_SIDED_LOCAL_PREDICATION=1`:

| width | candidate/oracle | 95% paired CI | median candidate/oracle, ms | wins |
| ---: | ---: | ---: | ---: | ---: |
| W2 | 1.1520x | [1.1409, 1.1632] | 17.705 / 20.344 | 8/8 |
| W4 | 1.3643x | [1.3480, 1.3808] | 8.587 / 11.678 | 8/8 |
| W8 | 1.2730x | [1.2441, 1.3025] | 7.175 / 9.099 | 8/8 |
| W16 | 1.0084x | [0.9944, 1.0224] | 6.114 / 6.161 | 6/8 |

Production enables W2/W4/W8 and leaves W16 on its existing bounded
predicated-loop path. `LUISA_SIMD_FORCE_TWO_SIDED_LOCAL_PREDICATION=1`
retains W16 as a semantic/diagnostic path. That policy matters: recognizing
the local region prevents W16 from selecting its already profitable whole-loop
form, and the measured replacement is neutral.

The W8 final object shrinks materially even though state-slot and instruction-
spill counts remain 37 and 15:

| Voxel W8 main entry | two-sided candidate | disabled oracle |
| --- | ---: | ---: |
| assembly bytes | 102,248 | 106,093 |
| static instructions | 2,138 | 2,280 |
| vector instructions | 1,092 | 1,162 |
| branches | 224 | 244 |
| stack references | 476 | 510 |
| stack allocation | 3,328 B | 3,520 B |
| calls / scalar-math calls | 3 / 2 | 3 / 2 |

Both scalar-math calls are the same uniform `sincosf` camera calculations;
neither object contains a varying scalar-libm lane loop. Five-repeat W8
`perf stat` over 256 renders records 121.383 B versus 160.277 B cycles
(-24.27%), 316.734 B versus 503.465 B instructions (-37.09%), and 45.982 B
versus 78.442 B branches (-41.38%). Branch misses fall 8.68% and cache misses
3.36%. This directly identifies independent-PC routing, rather than LLVM's
fixed-vector arithmetic, as the removed bottleneck.

The refreshed seven-round fallback table above measures W4/W8/W16 at
1.294x/1.566x/2.233x. W2 improves substantially against its own oracle but
still does not amortize the SIMD scheduler relative to fallback. Fresh
W1/W2/W4/W8/W16 gallery runs all pass at 82.834519 dB and produce SHA-256
`6172183a6c96704ffa48a6b64d30afcf2a3921431507dc40e3f80f1ae1362e4b`;
the W8 disabled oracle is byte-identical. Ordinary and cutout path tracing,
SDF, blackhole, and all image-processing kernels report zero two-sided hits,
so their generated paths are unchanged.

The same-algorithm analytic path tracer also reports zero two-sided hits, so
this checkpoint does not claim an ISPC gain for that workload. A fresh
single-worker run pinned to CPU 6 used fifteen alternating process rounds and
official ISPC 1.31.0. ISPC remains 1.223x [1.215, 1.231] faster than Luisa W8
with AVX2 x8, 1.148x [1.141, 1.155] faster with AVX-512 x8, and 1.016x
[1.013, 1.018] faster than Luisa W16 with AVX-512 x16. An independent
thirty-round W16-only repeat measured 1.010x [1.008, 1.012]. W16 is therefore
close but has not statistically crossed parity; GEMM remains the established
same-algorithm workload where Luisa is already faster than matched-width ISPC.

### Block-local packet batching

The host formerly called the JIT packet entry once for every warp in a block.
For the analytic path tracer's 256-thread block that is 128, 64, 32, or 16
host/JIT crossings at W2/W4/W8/W16. The retained wrapper executes those packets
in the same order behind one exported entry. `LUISA_SIMD_DISABLE_PACKET_BATCH_ENTRY=1`
compiles the old single-packet object, so the comparison does not retain an
unused second body or select between two entries after JIT.

Fifteen alternating single-worker rounds were pinned to CPU 6. Each process
used seven samples of 64 dispatches, and every candidate/oracle pair validated
checksum `a93089e651f98582`. One W4 oracle process suffered an unrelated 44%
slowdown; its robust paired median and trimmed geometric mean agree at about
1.023x, so the disturbed raw geometric mean is not used as the claim:

| width | retained lowering | candidate/oracle throughput | 95% paired CI | wins |
| ---: | --- | ---: | ---: | ---: |
| W1 | ordinary entry | disabled by policy | -- | -- |
| W2 | dynamic call loop | 1.0169x | [1.0122, 1.0216] | 15/15 |
| W4 | dynamic call loop | 1.0229x robust | disturbed raw interval | 15/15 |
| W8 | one inlined body/loop | 1.0355x | [1.0309, 1.0402] | 15/15 |
| W16 | sixteen-call shell | 1.0094x | [1.0083, 1.0106] | 15/15 |

The W16 counter result over seven alternating pairs attributes the gain to
boundary/control removal: cycles fell about 0.59%, retired instructions 0.78%,
branches 0.94%, and branch misses 9.1%. Final code shape also proves that the
optimization does not buy speed by cloning a large scheduler body:

| analytic entry | W8 | W16 |
| --- | ---: | ---: |
| externally visible batch body | 10,402 B | 327 B wrapper |
| internal packet body | removed after inlining | 13,346 B |
| static instructions | 1,927 | 2,655 total |
| calls in final entry/object | 0 | 15 calls plus final tail jump |
| packet-body stack allocation | 2,240 B | 2,816 B |
| varying scalar-libm calls | 0 | 0 |

Two plausible broader forms failed their real gates. Inlining the complete W16
body into an outer loop reduced some dynamic instructions but increased live
ranges/cache pressure and regressed throughput about 1.85%. Pinning eight
apparently cold Schedule slots reduced the frame from 2,816 to 2,624 bytes but
regressed throughput roughly 7--8%; LLVM's promotable state remains preferable.
A direct thread-index internal ABI was neutral/slightly negative, and `fastcc`
produced a byte-identical object. Forcing W8 to an AVX2-only target dropped the
same workload from about 197.3 to 146.0 Mitems/s, so W8's remaining gap is not
fixed by suppressing AVX-512 masks and paired 64-bit gathers on this host.

A fresh official ISPC 1.31.0 comparison used 21 rotating process rounds,
single worker, CPU 6, precise math, separately validated output, and no ISPC
path in CMake:

| pair | Luisa median | ISPC median | ISPC/Luisa | 95% paired CI |
| --- | ---: | ---: | ---: | ---: |
| W8 / AVX-512 x8 | 196.834 Mitems/s | 217.222 Mitems/s | 1.1044x | [1.1025, 1.1064] |
| W16 / AVX-512 x16 | 341.459 Mitems/s | 341.561 Mitems/s | 1.0005x | [0.9986, 1.0025] |

W16 is now statistically tied with ISPC, not demonstrably faster. W8 remains
about 9.5% slower. The accepted packet batch closes boundary overhead but does
not erase W8's remaining structured-mask/register-residency gap.

### Canonical early-exit header cohort specialization

The analytic path loop has a canonical counted header plus lane-varying early
exits. Its induction and header comparison correctly remain `varying` after
global uniformity analysis because post-loop state can contain lanes from
different exit epochs. Within one executing continuation, however, the loop
epoch proves that all active lanes have the same induction value. The retained
lowering records that fact only on the direct header comparison, sanitizes
inactive predicate bits, performs one `or.reduce`, and routes the complete
cohort through one edge. It does not scalarize the PHI or remove the post-loop
convergence.

Candidate/oracle measurements used one worker pinned to CPU 6, alternating
process order, seven internal samples of 64 analytic renders, and exact
checksum `a93089e651f98582`. The oracle is the same binary with
`LUISA_SIMD_DISABLE_COHORT_UNIFORM_INDUCTION=1`:

| width | paired rounds | candidate/oracle geomean | 95% paired CI | wins |
| ---: | ---: | ---: | ---: | ---: |
| W4 | 6 | 1.0021x | [0.9997, 1.0045] | 5/6 |
| W8 | 10 | 1.0027x | [1.0014, 1.0041] | 9/10 |
| W16 | 6 | 1.0101x | [1.0080, 1.0122] | 6/6 |

The final optimized objects distinguish the candidate at the intended header
without introducing a call or scalar-math lane loop:

| analytic entry | W8 candidate | W8 oracle | W16 candidate | W16 oracle |
| --- | ---: | ---: | ---: | ---: |
| static instructions | 1,849 | 1,927 | 2,487 | 2,571 |
| vector instructions | 1,109 | 1,140 | 1,120 | 1,153 |
| branches | 165 | 175 | 263 | 276 |
| stack references | 416 | 442 | 418 | 448 |
| stack allocation | 2,112 B | 2,240 B | 2,240 B | 2,304 B |
| calls / varying scalar-math calls | 0 / 0 | 0 / 0 | 15 / 0 | 15 / 0 |

Three-repeat `perf stat` measurements over the same 256-dispatch runner
quantify the removed dynamic scheduler work. Counts below are candidate versus
the disabled oracle; the process includes identical JIT setup, so throughput
claims remain based on the alternating table above:

| width | cycles | instructions | branches | branch misses |
| ---: | ---: | ---: | ---: | ---: |
| W8 | -0.293% | -2.560% | -9.142% | -4.368% |
| W16 | -0.888% | -2.742% | -4.639% | -8.484% |

This is the expected signature of eliminating successor-mask construction and
the generic coherent/divergent decision at one hot loop header: branch work
falls much more than arithmetic, while the fixed-vector math body is unchanged.

A fresh official ISPC 1.31.0 comparison used the same validated analytic
algorithm, AVX-512 x8/x16 targets, pinned single-worker execution, and ten
balanced rotating process rounds. Luisa W16/ISPC x16 reached 1.01608x
[1.01455, 1.01761] with 10/10 wins; the process medians were 347.276 versus
342.154 Mitems/s. W16 has therefore crossed parity by about 1.61% on this
workload. Luisa W8/ISPC x8 remains 0.90668x [0.90516, 0.90821], with process
medians of 197.227 versus 217.341 Mitems/s, so ISPC is still about 10.29%
faster at W8. This is a same-algorithm analytic result, not a claim that the
complete Embree renderer has an equivalent ISPC baseline. The standalone ISPC
executable and generated objects remain outside CMake and the repository; the
raw record is `/tmp/luisa-simd-cohort-final-ispc-w8-w16-10r.json`.

The profitability boundary is deliberately real-workload driven. Removing the
25-Schedule-block minimum made the ordinary Embree path kernel eligible, but
seven candidate/oracle pairs measured only 1.0026x at W4, 0.9991x at W8, and
0.9976x at W16, with just 1/7 W16 wins. Image processing still had no eligible
header, and Voxel's wider paths continued to prefer the existing predicated
loop. The relaxed policy was therefore rejected. A variable seed-lane extract,
a direct seed predicate-bit extract, a second non-header condition, and removal
of convergence metadata also regressed roughly 1.5--5.4%; none is retained.
The last case is additionally outside the semantic contract because earlier
exit epochs may still need the post-loop rendezvous.

### Structured early-exit loop residency

The next W8 stage keeps the complete eligible analytic loop in structured LLVM
control instead of returning to the independent-PC dispatcher between its
control-driving blocks. It retains one shrinking continuation mask, executes
each pure linear early-exit tail under its exit mask, uses the existing local
predication for internal diamonds, and reaches the declared common exit once
under the original cohort. It clones no source block and preserves the
canonical header convergence/token contract. The production oracle is
`LUISA_SIMD_DISABLE_STRUCTURED_EARLY_EXIT_LOOP=1`.

The accepted analytic site contains 47 loop blocks, 172 instructions, and
three absorbed exit-tail blocks. Ten strictly alternating single-worker W8
candidate/oracle processes were pinned to CPU 6; every process performed seven
internal samples over 256 complete dispatches and retained checksum
`a93089e651f98582`:

| W8 analytic path | process median | paired result |
| --- | ---: | ---: |
| structured loop | 235.732 Mitems/s | 10/10 wins |
| disabled general scheduler | 197.187 Mitems/s | structured/oracle 1.19487x [1.19323, 1.19652] |

The exact final JIT objects identify the eliminated scheduler/state-machine
work. Both have zero calls, zero scalar-math calls, and no unresolved symbol:

| W8 analytic entry | structured loop | disabled oracle |
| --- | ---: | ---: |
| assembly bytes | 40,170 | 96,824 |
| static instructions | 639 | 1,849 |
| vector instructions | 577 | 1,109 |
| branches | 30 | 165 |
| stack references | 92 | 416 |
| stack allocation | 960 B | 2,112 B |

Fresh three-repeat `perf stat` runs over the same 256-dispatch process give the
dynamic attribution:

| W8 process | cycles | instructions | branches | branch misses |
| --- | ---: | ---: | ---: | ---: |
| structured loop | 9.716 B | 25.355 B | 1.282 B | 4.589 M |
| disabled oracle | 11.836 B | 38.097 B | 2.413 B | 8.106 M |
| ISPC AVX-512 x8 | 10.406 B | 17.952 B | 1.128 B | 2.657 M |

Relative to the general scheduler, the retained path reduces cycles by about
17.9%, retired instructions by 33.4%, branches by 46.9%, and branch misses by
43.4%. It still retires more instructions and branches than ISPC, but executes
about 6.6% fewer cycles on this host. This is consistent with register/SSA
residency and removal of repeated dispatch/frame traffic, rather than a better
LLVM vector math approximation or an ISA-specific intrinsic.

The official ISPC 1.31.0 comparison was independently rebuilt through the
standalone driver with an explicit executable path, `--cpu=znver5`, precise
math, FMA contraction disabled, one worker on CPU 6, and validated output. W8
uses ten balanced rotating process rounds. Two visibly interrupted W16 samples
in the mixed-width run widened its interval across parity, so the final W16
result uses a separate twenty-round alternating control and retains both slow
samples:

| matched analytic path | rounds | Luisa median | ISPC median | Luisa/ISPC paired geomean | 95% CI |
| --- | ---: | ---: | ---: | ---: | ---: |
| W8 / AVX-512 x8 | 10 | 235.561 | 218.267 Mitems/s | 1.07926x | [1.07785, 1.08067] |
| W16 / AVX-512 x16 | 20 | 347.710 | 342.606 Mitems/s | 1.01209x | [1.00759, 1.01660] |

The W8 structured site wins all ten corresponding process pairs. W16 does not
select this W8-only transformation; its retained earlier optimizations win
18/20 refreshed pairs and remain above parity even with the two interruptions.
The raw official-driver records are
`/tmp/luisa-simd-structured-final-audited-w8-w16-10r.json` and
`/tmp/luisa-simd-structured-final-audited-w16-20r.json`. The compiler and
generated ISPC objects remain outside CMake and the repository; no
`LUISA_COMPUTE_ISPC_EXECUTABLE` variable exists.

This is not an ISPC implementation of the repository's full renderer. Fresh
W1/W2/W4/W8/W16 galleries pass image processing at 89.251953 dB, Voxel at
82.834519 dB, and ordinary 1024-spp Embree path tracing at
35.426795/42.781582/40.940376/39.219305/37.801771 dB. Every one of those real
kernels reports zero structured-loop sites; the path processes report native
Embree 4.4.1 W4/W8/W16 packets. A separate W8 candidate/oracle run produces
byte-identical PNGs for all three workloads. Their hashes are respectively
`73d7aa39c1d17b2f2be073f91e5c4615e9233e58fbf673a195b7e43cc43baa31`,
`6172183a6c96704ffa48a6b64d30afcf2a3921431507dc40e3f80f1ae1362e4b`,
and `6abef9a86fa07dfc11249d491953478e4d6b50ef0861ecb1781b09750902a5d1`.
Those runs are applicability/correctness gates, not renderer speed claims for
this analytic-only hit.

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

### Direct-CFG block-range batching

The block-local packet wrapper still crossed the host/JIT boundary once for
every block. The retained direct-CFG refinement adds one outer JIT loop for the
consecutive flattened block range already claimed by one persistent-pool
worker. It advances the private three-dimensional block ID, resets the packet
origin for every block, and calls the internal packet wrapper with the static
packet count. `LUISA_SIMD_DISABLE_BLOCK_BATCH_ENTRY=1` restores the old
per-block call. Scheduler-backed kernels deliberately keep that oracle path:
enabling the outer loop indiscriminately regressed the analytic W8 path by
about 0.6--0.8% in all seven initial pairs.

At this stage the packet-body ABI also stated the ownership facts that were
already true at runtime: the packed argument record and launch configuration
are read-only and do not overlap each other or the return record. The outer
wrappers were not yet annotated because they mutate launch indices; the later
linear-1D stage below propagates their strictly weaker valid facts. Resource
addresses loaded from the argument record may still alias one another. Nine
alternating W8 pairs isolated the packet-body LLVM attributes with
`LUISA_SIMD_DISABLE_PACKET_ABI_ALIAS_ATTRIBUTES=1`:

| workload | attributes / disabled | 95% paired CI | wins |
| --- | ---: | ---: | ---: |
| AoS-to-SoA | 1.11133x | [1.09644, 1.12643] | 9/9 |
| masked stream | 1.00025x | [0.92126, 1.08602] | 7/9 |
| analytic path | 1.00290x | [1.00139, 1.00442] | 8/9 |

With those attributes fixed in both forms, nine alternating block-range/oracle
pairs on CPU 6 produced:

| width/workload | block range / per block | 95% paired CI | wins |
| --- | ---: | ---: | ---: |
| W8 AoS-to-SoA | 1.02426x | [1.01090, 1.03780] | 8/9 |
| W8 masked stream | 1.00344x | [0.89496, 1.12506] | 4/9 |
| W8 GEMM | 1.00036x | [0.99912, 1.00160] | 6/9 |
| W8 analytic path | 1.00074x | [0.99908, 1.00240] | 5/9 |
| W16 AoS-to-SoA | 1.15230x | [1.13969, 1.16504] | 9/9 |
| W16 masked stream | 1.11146x | [1.06176, 1.16348] | 8/9 |
| W16 GEMM | 1.00627x | [1.00505, 1.00748] | 9/9 |
| W16 analytic path | 1.00160x | [0.98986, 1.01347] | 3/9 |

The path rows are expected null controls because those kernels are
scheduler-backed and do not export a block-range entry. Masked W8 was visibly
disturbed by unrelated machine activity and is not a positive claim. The
accepted evidence is the stable AoS improvement at both widths plus W16 masked
and GEMM; W2/W4 exploratory pairs were neutral on GEMM and positive by roughly
2--3% on AoS.

Final object inspection shows that the gain is not a scalar fallback or libm
call. Candidate/oracle static instruction counts are 161/180 for W8 AoS,
124/123 for W8 masked, 1,247/1,296 for W16 AoS, and 585/624 for W16 masked.
All four candidate objects have zero calls and no undefined symbols. On this
Zen 5 host W8 uses YMM arithmetic with AVX-512VL opmasks (and ZMM operations
where the 64-bit gather shape requires them); W16 is predominantly ZMM. Width
W8 remains a portable eight-lane semantic specialization rather than an
AVX-512 requirement on other targets.

The official ISPC 1.31.0 suite was then rebuilt explicitly with
`--cpu=znver5`, AVX-512 x8/x16 targets, precise arithmetic and FMA contraction
disabled. Fifteen rotating process rounds used one worker pinned to CPU 6.
Mandelbrot, masked stream, AoS-to-SoA, and GEMM were bit-exact across both
implementations; the analytic path output passed its absolute-plus-relative
tolerance:

| workload | Luisa W8 / ISPC x8 | 95% CI | Luisa W16 / ISPC x16 | 95% CI |
| --- | ---: | ---: | ---: | ---: |
| Mandelbrot | **1.40165x** | [1.39871, 1.40460] | **1.54284x** | [1.54012, 1.54558] |
| masked stream | 0.63265x | [0.61440, 0.65144] | 0.69323x | [0.66548, 0.72214] |
| AoS-to-SoA | 0.89964x | [0.88815, 0.91128] | 0.84961x | [0.83840, 0.86096] |
| GEMM | **1.33436x** | [1.33044, 1.33829] | **1.37180x** | [1.37050, 1.37310] |
| analytic path | **1.08682x** | [1.08587, 1.08776] | **1.02091x** | [1.01971, 1.02212] |

Thus Luisa wins three of the five matched compiler controls, including the
analytic path at both widths, but does not claim general superiority over
ISPC: sparse/mixed masking and AoS conversion remain concrete gaps. The raw
record is `/tmp/luisa-simd-ispc-block-batch-final-15r.json`; the ISPC binary
and generated objects remain outside CMake and the repository.

The same standalone algorithms also expose why W1 is not equivalent to the
fallback backend. Fallback owns a different dispatch/code-generation pipeline,
and its scalar block loop can be auto-vectorized vertically by the host
compiler; SIMD W1 still pays packet launch and SIMD-backend ABI costs. Seven
rotating pinned-process rounds completed before a later external build began
contending for the machine:

| workload | W1/fallback | W2/fallback | W4/fallback | W8/fallback | W16/fallback |
| --- | ---: | ---: | ---: | ---: | ---: |
| Mandelbrot | 0.052x | 0.089x | 0.180x | 0.311x | 0.565x |
| masked stream | 0.096x | 0.106x | 0.287x | 0.369x | 0.424x |
| AoS-to-SoA | 0.185x | 0.205x | 0.299x | 0.316x | 0.303x |
| GEMM | 0.047x | 0.052x | 0.234x | 0.388x | 0.654x |

These controls favor fallback's vertical/inner-loop vectorization and motivate
the planned horizontal/vertical layout selection; they are not evidence that
the explicit width should be increased blindly. The path-trace fallback fell
from about 700 to 348 and then 187 Mitems/s when two unrelated HIP compiler
processes started, so that workload's generated interval is rejected and will
not be used. The disturbed raw record is retained at
`/tmp/luisa-simd-block-batch-vs-fallback-all-widths-7r.json` rather than being
used to claim a speedup.

A separate quiet-machine gate measured three repository graphics examples in
the Release build. Processes were rotated in opposite orders on alternating
rounds, restricted to logical CPUs `0-12,14-28,30-31`, and SIMD used thirty
workers. Image processing repeated the complete pipeline 64 times per process,
Voxel repeated 128 renders, and ordinary Embree path tracing rendered 256 spp
at a fixed 32 spp per dispatch. Seven rounds give the following process
medians and paired throughput ratios:

| workload | fallback | W1 | W2 | W4 | W8 | W16 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| image processing, ms/iteration | 8.622 | 18.352 (0.484x) | 9.504 (0.929x) | 6.770 (**1.310x**) | 5.209 (**1.707x**) | 4.468 (**1.992x**) |
| Voxel, ms/iteration | 7.022 | 9.004 (0.793x) | 12.640 (0.565x) | 5.913 (**1.206x**) | 4.930 (**1.444x**) | 3.368 (**2.112x**) |
| ordinary path tracing, spp/s | 89.449 | 82.966 (0.922x) | 65.047 (0.723x) | 84.648 (0.939x) | 96.515 (**1.077x**) | 96.580 (**1.077x**) |

The parenthesized values are paired-geomean SIMD/fallback throughput ratios,
not ratios of the displayed medians. Their 95% bootstrap intervals are
respectively W1 through W16: image processing
`[0.460,0.511]`, `[0.885,0.983]`, `[1.250,1.380]`,
`[1.622,1.810]`, `[1.892,2.112]`; Voxel `[0.782,0.805]`,
`[0.556,0.575]`, `[1.191,1.223]`, `[1.418,1.472]`,
`[2.068,2.156]`; and path tracing `[0.914,0.928]`,
`[0.718,0.727]`, `[0.932,0.945]`, `[1.073,1.081]`,
`[1.071,1.084]`. Thus W8/W16 beat fallback on all three real examples, but
the profitable width is workload-dependent and W1 is not a fallback-equivalent
baseline.

The same image and Voxel processes also alternated the enabled block-range
entry with `LUISA_SIMD_DISABLE_BLOCK_BATCH_ENTRY=1`. Image processing measured
enabled/disabled ratios of 1.0010x, 0.9968x, **1.0127x**, 1.0033x, and 1.0039x
at W1/W2/W4/W8/W16. Only the W4 gain and the small W2 loss excluded one at 95%;
the remaining widths were neutral. Every Voxel interval included one, as
expected because its scheduler-backed kernel cannot export the entry. These
real examples therefore do not reproduce the standalone AoS kernel's larger
block-range gain. Raw records are
`/tmp/luisa-simd-block-image-ab-7r.tsv`,
`/tmp/luisa-simd-block-voxel-ab-7r.tsv`, and
`/tmp/luisa-simd-block-path-vs-fallback-7r.tsv`.

Fresh reference-image sweeps use the actual
`LUISA_SIMD_WARP_WIDTH=1|2|4|8|16` selector. Image processing and Voxel pass at
89.251953 dB and 82.834519 dB at every width. Ordinary 1024-spp Embree path
tracing passes at 35.426795/42.781582/40.940376/39.219305/37.801771 dB, and
non-coroutine 1024-spp SDF passes at 63.129346 dB at every width. Each path
process reports Embree 4.4.1 W4/W8/W16 packet support enabled.

### Linear-1D full-packet and cross-block specialization

The next launch refinement targets the two concrete costs left in straight-line
1D kernels. First, a runtime packet wrapper computes the exact dispatch/block
remainder once, executes a full-width main loop, and calls at most one narrowed
tail. The packet body therefore omits three repeated dispatch compares, and a
1D runtime packet uses its linear lane index directly as `thread_id.x`.
Statically unit block dimensions independently omit their redundant compare.
Second, a proven block-agnostic direct body may concatenate a worker's block
range into that packet loop. The proof rejects block/thread IDs, local storage,
barriers, and every `dispatch_id` use except component zero; a runtime y/z
dispatch guard retains the generic block loop for a multidimensional launch.

The exported packet and block wrappers now carry `noalias readonly` on the
packed argument record and `noalias nonnull` on their mutable launch record.
They deliberately do not mark that record `readonly`. This lets LLVM hoist
descriptor loads out of an inlined packet loop. A bounded W16 policy inlines
only a linear-1D, single-block Schedule body with 8--32 instructions on a host
with at least 512 fixed-vector bits and 32 vector registers. The mixed-mask
control retains its prior call-shell policy.

The individual A/B gates identify the retained effects:

| candidate / disabled oracle | paired result | 95% CI | wins |
| --- | ---: | ---: | ---: |
| W16 AoS wrapper-attribute propagation, 21 pairs | 1.01250x | [1.00631, 1.01873] | 17/21 |
| W8 AoS linear block coalescing, 15 pairs | 1.00764x | [1.00085, 1.01447] | 12/15 |
| W16 AoS linear block coalescing, 15 pairs | 1.01340x | [1.00578, 1.02108] | 12/15 |

The coalescing gate is intentionally small: it buys about 0.8--1.3% on the
memory-bound AoS control and is ineligible for the mixed-mask control and all
measured 2D graphics shaders. Broadly inlining both dynamic wrapper paths grew
the W16 AoS object to 851 instructions without a throughput gain and was
removed. Preventing LLVM's two-packet W16 loop unroll was also rejected after
21 interleaved same-binary pairs measured 0.99837x [0.99103, 1.00577] against
the original, with 9/21 wins. The retained final objects contain 240/274 static
instructions for W8/W16 masked stream and 384/557 for W8/W16 AoS. The two AoS
objects contain two internal dual-path wrapper calls; no object has an
unresolved symbol or varying scalar-libm call.

The final official ISPC 1.31.0 comparison used an explicit standalone ISPC
path, `--cpu=znver5`, AVX-512 x8/x16, precise arithmetic with FMA contraction
disabled, one worker pinned to CPU 15, and rotating process order. The two
linear workloads used separate 21-round width runs; Mandelbrot, GEMM, and the
analytic path used 15 rounds. Exact workloads were bit-identical, and the path
output passed the independent absolute-plus-relative tolerance:

| workload | Luisa W8 / ISPC x8 | 95% CI | Luisa W16 / ISPC x16 | 95% CI |
| --- | ---: | ---: | ---: | ---: |
| Mandelbrot | **1.40329x** | [1.39907, 1.40753] | **1.54555x** | [1.53157, 1.55965] |
| masked stream | **1.27656x** | [1.24271, 1.31134] | **1.19031x** | [1.17620, 1.20458] |
| AoS-to-SoA | 0.99348x | [0.98625, 1.00076] | 0.97479x | [0.96955, 0.98005] |
| GEMM | **1.35374x** | [1.34930, 1.35819] | **1.38332x** | [1.38158, 1.38506] |
| analytic path | **1.10136x** | [1.09715, 1.10560] | **1.02758x** | [1.02454, 1.03062] |

The unweighted five-workload Luisa/ISPC geometric means are **1.21552x at W8**
and **1.20581x at W16**. Luisa therefore wins four controls at each width; W8
AoS is statistically tied and W16 AoS remains 2.52% slower. This is an overall
same-algorithm compiler win, not a claim of universal per-kernel superiority or
an ISPC baseline for the full Embree renderer. Raw records are
`/tmp/luisa-simd-final-attrs-coalesce-ispc-w8-21r.json`,
`/tmp/luisa-simd-final-attrs-coalesce-ispc-w16-21r.json`, and
`/tmp/luisa-simd-final-attrs-coalesce-ispc-rest-15r.json`.

A fresh real-example matrix used seven alternating process rounds on logical
CPUs `0-12,14-28,30-31`; SIMD used 30 workers. Image processing and Voxel ran
eight synchronized iterations, and ordinary Embree path tracing ran 64 spp at
one spp per dispatch:

| workload | fallback | W1 | W2 | W4 | W8 | W16 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| image, median ms | 9.171 | 17.674 (0.531x) | 9.097 (1.028x) | 6.763 (**1.382x**) | 5.268 (**1.776x**) | 4.655 (**1.994x**) |
| Voxel, median ms | 7.324 | 8.508 (0.863x) | 11.724 (0.623x) | 5.598 (**1.308x**) | 4.754 (**1.548x**) | 3.277 (**2.207x**) |
| ordinary path, median spp/s | 70.538 | 60.129 (0.850x) | 52.198 (0.725x) | 67.761 (0.959x) | 78.081 (**1.095x**) | 81.260 (**1.136x**) |

Parenthesized values are paired geometric throughput ratios against fallback.
The W8/W16 95% intervals are image `[1.691,1.865]`/`[1.881,2.114]`, Voxel
`[1.497,1.600]`/`[2.145,2.271]`, and path `[1.069,1.122]`/`[1.098,1.174]`.
Every image-processing kernel and the Voxel kernel report zero linear block
coalescings, confirming that the new 1D proof does not select their 2D launch.
A current-binary 1024-spp reference sweep also passes fallback and every SIMD
width. W1/W2/W4/W8/W16 RGB PSNRs are
35.426795/42.781582/40.940376/39.219305/37.801771 dB; fallback reaches
62.223429 dB. Every SIMD process reports Embree 4.4.1 native W4/W8/W16 packet
support. This final sweep is a correctness gate rather than another paired
throughput claim.

## Validation

The linear-1D specialization stage completed full Release builds of both
configured trees. Each passes the required native-math/fallback-math/runtime-
width/Schedule-codegen gate 4/4, the SIMD/XIR/runtime/graphics gate 35/35, and
the complete configured suite 140/140. The focused Schedule executable covers
W8 and W16 cross-block execution, a nonzero starting block, a 13-lane final
W16 tail (five lanes at W8), exact disabled-oracle launch state,
block/thread-sensitive rejection,
and the minimal two-dimensional `dispatch_id.xy` rejection. The standalone
ISPC-driver and syntax-script Python tests pass 21/21; all seven changed C++
translation units pass clangd 22.1.8. Clang-format 22.1.8 dry-run and
`git diff --check` are clean.

The required native-math/fallback-math/runtime-width/Schedule-codegen gate
passes 4/4, including the new block-range JIT differential. After complete
Release builds, both `build-sdf-bench` and `build-sdf-tbb` pass their configured
repository suites 140/140. A separately repeated SIMD-labelled conformance
gate passes 35/35, covering scheduler, LLVM, runtime, accel, bindless, atomics,
local memory, and graphics. The standalone ISPC-driver suite passes 8/8, the
C++ syntax-check harness passes 13/13, and clangd reports no issue in all five
changed translation units. LLVM 22.1.8 `clang-format --dry-run --Werror` and
`git diff --check` also pass.
This also includes the coroutine-frame tests merged from `next`, the repaired
lazy-dispatch scalar snapshot regression, and the W1/W2/W4/W8/W16 aggregate-
promotion differential test. Separate 1024-SPP gallery gates pass ordinary
and cutout path tracing at all five widths, and non-coro SDF W8 passes at
63.13 dB RGB PSNR.

The block-local packet-batch stage completed fresh full builds of both Release
trees, the required native-math/fallback-math/runtime-width/Schedule-codegen
gate (4/4), and two consecutive complete SIMD-only runs (129/129 each,
including `integration_simd` 26/26 and graphics 3/3), followed by the complete
SIMD+fallback/tutorial configuration (140/140). Clang-format and diff checks
pass, the syntax checker reports no diagnostics for all five changed C++
translation units, and its Python suite passes 13/13. Fresh W1/W2/W4/W8/
W16 image-processing and Voxel galleries pass at 89.251953 and 82.834519 dB.
Ordinary 1024-spp path tracing passes at
35.426795/42.781582/40.940376/39.219305/37.801771 dB and every process reports
Embree 4.4.1 native W4/W8/W16 packet support. All images and raw timing logs
remain under `/tmp`; no reference or benchmark source was modified.

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

The acceleration instance-buffer-only stage completed a fresh full Release
build, the native-math/runtime-width/acceleration focus gate (6/6), and the
complete configured SIMD+fallback/XIR/runtime/graphics suite (140/140). The
W1/W2/W4/W8/W16 permanent regression proves that a host transform/mask/user-id
update and appended primitive become visible through the instance table while
the committed BVH remains unchanged; a following ordinary build with an empty
modification list then commits both deferred geometries. A deferred shrink also
keeps direct traversal and stateful ray queries on the old BVH, including the
removed instance's committed classification, until the following ordinary
build. Mesh-to-curve and curve-to-mesh buffer-only replacements likewise keep
both direct and stateful-query classification tied to the committed primitive
until the deferred build refreshes it. A device-side metadata write is also
routed through buffer-only before its ordinary build, proving that the
intermediate command cannot clear the dirty state. Metadata-only commands avoid
both Embree calls and an instance-summary scan; only count or primitive changes
trigger the O(instance-count) classification refresh.

The repository's real `test_rtx simd` path, which appends an instance through
buffer-only at frame 512 and builds it on the next frame, passes its checked-in
reference at 82.508596 dB. Ordinary W8 1024-spp packet path tracing passes at
39.219305 dB and reports native Embree 4.4.1 W4/W8/W16 packet support. The
motion-blur test has no checked-in reference, so it is not counted as a gallery
pass; instead, fresh SIMD and fallback 512x512/1024-spp outputs compare at
88.279279 dB RGB PSNR, maximum channel error one, with 70 differing pixels.
Their SHA-256 hashes are respectively
`30b450a0dd796f9a1a0b05446114fad578e338461d650a253df5beb0a228f8ad` and
`3cbdcca73e7086a5bcb64a624326211d1bb61ed3412a7ed7f5fd243835dd4208`.
All generated images remained in `/tmp`; this semantic stage makes no new
throughput claim.

The innermost-loop local-predication/chaining stage completed a fresh full
Release build, the required native-math/fallback-math/runtime-width gate plus
the focused Schedule-codegen regression (4/4), and the complete configured
SIMD+fallback/XIR/runtime/graphics suite (140/140). Clang-format 22.1.8 and
diff checks pass for all changed C++ sources. The permanent differential test
covers W2/W4/W8/W16, `W - 1` active lanes, guarded square-root and floating-
division arms, an assignment-only nested tail, every narrower disabled oracle,
and exact bit equality including inactive sentinels. A fresh ordinary W8
1024-spp Embree path-tracing run accepts the checked-in gallery reference at
39.219305 dB and reports native Embree 4.4.1 W4/W8/W16 packet support. Its
optimization report records exactly one local 19-instruction diamond, with no
nested or chained region in that shader. Output remained in `/tmp`; the
checked-in reference was not modified.

The local terminal-bridge extension completed another fresh full Release
build, the required native-math/fallback-math/runtime-width/Schedule-codegen
gate (4/4), and the complete configured SIMD+fallback/XIR/runtime/graphics
suite (140/140). Clang-format 22.1.8, diff checks, the syntax-check runner's
Python suite (13/13), and clangd checks pass for all four changed translation
units and their included headers. Its permanent differential JIT regression
covers forced W2 and production W4/W8/W16, `W - 1` active lanes, inactive NaN
sentinels, a guarded square root, a 41-instruction merge-to-loop-back tail,
LLVM verification, scalar-libm-symbol rejection, and exact disabled-oracle
equality. A fresh ordinary W8 1024-spp path trace accepts the checked-in gallery
reference at 39.219375 dB and reports native Embree 4.4.1 W4/W8/W16 packet
support. The optimization report records one 19-instruction local diamond and
one 81-instruction terminal block, reducing state slots/spills from 34/20 to
32/18. Output remained in `/tmp`; the checked-in reference was not modified.

The two-sided innermost local-predication stage completed a fresh full Release
build, the required native-math/fallback-math/runtime-width/Schedule-codegen
gate (4/4), and the complete configured SIMD+fallback/XIR/runtime/graphics
suite (140/140). Clang-format, diff checks, the syntax-check runner's Python
suite (13/13), per-translation-unit clangd checks, and the standalone ISPC
driver tests (7/7) pass. Its permanent differential JIT regression covers
production W2/W4/W8 and forced W16, `W - 1` active lanes, independently
varying loop iterations, two nonempty arithmetic/cast arms, an inactive NaN
before `fptoui`, operand pre-sanitization, LLVM verification, terminal merge
absorption where enabled, exact candidate/oracle bits, and inactive sentinels.
Fresh W1/W2/W4/W8/W16 Voxel galleries pass at 82.834519 dB with one hit at
W2/W4/W8 and none at W1/W16; the W8 disabled oracle is byte-identical. A fresh
ordinary W8 1024-spp path trace passes at 39.219375 dB, reports native Embree
4.4.1 W4/W8/W16 packet support, and records zero two-sided hits. The current
fifteen-round analytic ISPC comparison and independent thirty-round W16 repeat
are recorded in the two-sided section above. Outputs and raw benchmark JSON
remain in `/tmp`; no checked-in gallery reference was regenerated or modified.

The general W16 state-coloring stage completed both Release builds, the
required native-math/fallback-math/runtime-width/Schedule-codegen gate (4/4),
the SIMD-only suite (129/129), and the complete configured SIMD+fallback/XIR/
runtime/graphics suite (140/140). The standalone ISPC-driver suite passes 8/8,
clang-format 22.1.8 and per-translation-unit clangd syntax checks pass, and the
final diff is whitespace-clean. The permanent regression covers W1/W2/W4/W8/
W16, every tail length from zero through W, forced narrower-width coloring,
complex interference, exact candidate/oracle outputs, narrower production
assembly identity, and the one-slot production rollback.

Fresh W1/W2/W4/W8/W16 Voxel and image-processing gallery comparisons pass at
82.834519 and 89.251953 dB. Ordinary 1024-spp Embree path tracing passes at
35.426795/42.781582/40.940376/39.219305/37.801771 dB from W1 through W16;
W16 cutout and non-coroutine SDF pass at 39.476901 and 63.129346 dB. Both
path tracers report native Embree 4.4.1 W4/W8/W16 packet support. The final
thirty-round W16 official-ISPC comparison uses the corrected strictly
alternating order and is recorded in the general-coloring section above.
Outputs and raw benchmark records remain outside the repository; no checked-in
gallery reference was regenerated or modified.

The canonical early-exit header stage completed fresh full Release builds of
both configured trees, its Schedule/Schedule-projection/LLVM-codegen focus gate
(3/3), the required native-math/fallback-math/runtime-width gate (3/3), two
consecutive SIMD-only runs (129/129 each), and the complete SIMD+fallback/XIR/
runtime/graphics/tutorial suite (140/140). Changed-line clang-format is clean,
all six changed C++ translation units pass the clangd syntax checker, and the
final diff has no whitespace errors. The permanent differential regression
covers W1/W2/W4/W8/W16, an inactive tail, distinct early/normal exits in
different epochs, a post-loop collective, exact disabled-oracle output, and
fail-closed 24-block/nonuniform-bound cases.

Fresh W1/W2/W4/W8/W16 image-processing, Voxel, ordinary 1024-spp path-tracing,
and non-coroutine 1024-spp SDF gallery comparisons all pass. Their RGB PSNRs
are respectively 89.251953, 82.834519,
35.426795/42.781582/40.940376/39.219305/37.801771, and 63.129346 dB. Every
path-tracing process reports Embree 4.4.1 native W4/W8/W16 packet support.
Production W4/W8/W16 applicability scans record zero new header sites in all
image, Voxel, and ordinary path kernels, while the analytic W8/W16 kernel
records exactly one. Fresh dumped analytic objects have no unresolved symbol
and no scalar f32 math call; the W8 object uses YMM arithmetic plus AVX-512
mask/ZMM operations, while W16 is predominantly ZMM. The final ten-round
official-ISPC comparison and three-repeat hardware counters are recorded in
the specialization section above. All images, objects, logs, and raw JSON stay
under `/tmp`; no reference image was regenerated or modified.

The structured early-exit-loop stage completed fresh full Release builds of
both configured trees. The native-math/fallback-math/runtime-width/Schedule-
codegen gate passes 4/4, the scheduler plus runtime/graphics conformance gate
passes 25/25 in both trees, the SIMD-only configuration passes 129/129, and
the larger SIMD+fallback/XIR/runtime/graphics/tutorial configuration passes
140/140. Clang-format 22.1.8, whitespace checks, all seven changed translation
units under clangd, the syntax-check runner suite (13/13), and the standalone
ISPC-driver suite (8/8) pass.

The permanent forced W8 regression and its three fail-closed near misses pass,
including the five-lane inactive tail and pre-sanitized NaN-to-integer cast.
Fresh W1/W2/W4/W8/W16 image-processing, Voxel, and ordinary 1024-spp Embree
path-tracing galleries all accept their references, every path process reports
native Embree 4.4.1 W4/W8/W16 packet support, and a W8 enabled/disabled run is
byte-identical for all three outputs. These real kernels report zero structured
sites. The ten-round W8 and twenty-round W16 matched analytic comparisons,
final-object audit, and three-repeat hardware counters are recorded in the
structured-loop section above; all generated artifacts remain outside the
repository.

## W8 ray-query cutout-filter predication

The next experiment targeted the repository's real cutout renderer rather than
the analytic path control. Predicating the complete lowered ray-query loop was
rejected. In ten alternating 64-spp process pairs, W8 candidate/oracle reached
0.979629x [0.974575, 0.984709], and W16 reached 0.972440x
[0.963077, 0.981893]. W8 hardware counters explained the regression: despite
fewer static/dynamic branches, cycles increased 1.74%, L1D loads 3.51%, and
L1D load misses 6.13%. Extending both handler payload paths kept too much of the
large AoS query state live. The production implementation therefore does not
predicate a query loop, payload read, commit, terminate, callback, or memory
operation.

The retained transform is an exact XIR-side filter specialization. It proves
all aggregate extraction indices constant, nonnegative, in-bounds, and type-
correct, then accepts only the nested `SurfaceHit::inst` cutout ladder described
in the design and execution contract. The two real query sites produce four
W8 conversions. A forced W16 ten-pair control was neutral at 0.999825x
[0.994087, 1.005596], so production remains W8-only. Factoring the two
speculated `fract` paths through an additional select was also rejected:
1.001027x [0.992321, 1.009809], with no stable evidence of a gain.

The final W8 candidate/oracle code-shape comparison is:

| metric | filter predication | disabled oracle |
| --- | ---: | ---: |
| dedicated filter diamonds | 4 | 0 |
| Schedule blocks | 57 | 67 |
| convergence points | 15 | 19 |
| logical state slots | 48 | 50 |
| assembly bytes | 289,473 | 319,880 |
| static instructions | 5,301 | 5,910 |
| vector instructions | 3,153 | 3,341 |
| branches | 476 | 582 |
| calls | 5 | 5 |
| stack references | 1,287 | 1,348 |
| recognized stack allocation | 19,712 B | 20,352 B |
| scalar math calls | 0 | 0 |

Thus the narrow transform removes ten Schedule blocks, four convergence
points, two state slots, 609 static instructions, and 106 branches without
duplicating the query callback boundary. Fresh candidate and oracle objects
have no undefined symbol (`nm -u` is empty), and neither assembly contains a
scalar f32 libm call.

The final throughput gate used the Release build, W8, 30 workers pinned to
logical CPUs `0-12,14-28,30-31`, 64 samples at one sample per dispatch, and a
new JIT process for every entry. Candidate and disabled-oracle order alternated
per pair. Because unrelated machine activity was present, the measurement was
split and reported rather than selecting only the favorable run:

| balanced process set | candidate median | oracle median | candidate/oracle paired geomean | 95% bootstrap CI | wins |
| --- | ---: | ---: | ---: | ---: | ---: |
| first 15 final pairs | 44.757 FPS | 44.567 FPS | 1.002342x | [0.997166, 1.007817] | 8/15 |
| following 20 final pairs | 45.058 FPS | 44.796 FPS | 1.004746x | [1.001735, 1.007572] | 15/20 |
| complete final 35 pairs | 44.930 FPS | 44.778 FPS | **1.003715x** | **[1.000851, 1.006571]** | 23/35 |
| final plus earlier compatible 10 pairs | -- | -- | **1.004665x** | **[1.002075, 1.007287]** | 31/45 |

The primary claim is therefore a small 0.37% gain in the fresh 35-pair batch,
not the earlier batch's larger result. All seventy final rendered PNGs have the
same SHA-256,
`ad97b2a0e41cab86019e7def16f0bd8d63007e640eb4a468d2b71c09a9e74eda`.

Four additional alternating 256-spp `perf stat` pairs give the following
candidate/oracle ratios. Counter runs were multiplexed at approximately 83%,
so the paired ratios, rather than their absolute scaled totals, are the useful
evidence:

| process metric | candidate/oracle | change |
| --- | ---: | ---: |
| FPS | 1.003417x | +0.34% |
| cycles | 0.992568x | -0.74% |
| instructions | 0.993352x | -0.66% |
| branches | 0.989617x | -1.04% |
| branch misses | 0.974477x | -2.55% |
| L1D loads | 0.998808x | -0.12% |
| L1D load misses | 0.930735x | -6.93% |

The permanent differential JIT regression compiles and executes the exact
query/filter shape at W1/W2/W4/W8/W16. W2/W4/W16 use the diagnostic force;
W8 exercises production policy; W1 must remain scalar. Candidate and disabled
oracle consume tall, short, and unrelated instance candidates through the
plain/packed-status query ABI, execute a 13-element inactive tail, and compare
all sixteen output slots exactly. A `SurfaceHit::prim` lookalike must report
zero dedicated conversions. The target-independent XIR test separately keeps
dynamic extraction ineligible and proves that an explicitly enabled nested
structure/vector static extraction is accepted.

Fresh production applicability and reference-image checks cover all five
widths. Only cutout W8 reports dedicated sites (four); cutout W1/W2/W4/W16 and
every image-processing, Voxel, ordinary path-tracing, and non-coroutine SDF
kernel report zero. All reference comparisons pass:

| gallery | W1/W2/W4/W8/W16 RGB PSNR |
| --- | --- |
| image processing | 89.251953 dB at every width |
| Voxel | 82.834519 dB at every width |
| ordinary 1024-spp path tracing | 35.426795 / 42.781582 / 40.940376 / 39.219305 / 37.801771 dB |
| cutout 1024-spp path tracing | 39.100980 / 39.738460 / 39.667107 / 39.576002 / 39.476901 dB |
| non-coroutine 1024-spp SDF | 63.129346 dB at every width |

Every path process reports Embree 4.4.1 native W4/W8/W16 packet support. All
logs, counters, objects, assemblies, and rendered images remain under `/tmp`;
no checked-in reference was regenerated or modified.

Final validation used both Release build trees after a complete (not
target-only) build. `build-sdf-bench` and `build-sdf-tbb` each pass the required
native-math/runtime-width gate 3/3, the scheduler JIT executable including the
new ray-query differential case, and the XIR mutation-safety executable (185
assertions in 28 tests). Each tree then passes its complete CTest inventory
140/140 and a separately repeated SIMD-labelled conformance gate 35/35,
covering scheduler, LLVM, runtime, accel, bindless, atomic, local-memory, and
graphics tests. The six changed translation units pass the project clangd
syntax checker, its Python harness passes 13/13, LLVM 22.1.8
`clang-format --dry-run --Werror` passes every changed C++ source and header,
and `git diff --check` is clean.

## Exact outer-affine SRT motion packets

Quaternion SRT motion under a nonidentity outer affine is now exact rather
than endpoint-matrix interpolation. A private native Embree SRT instance
provides quaternion interpolation and conservative linear bounds. One
top-level user geometry composes `outer * SRT(time)`, inverse-transforms the
ray without normalizing its direction, traverses the child scene once, and
inverse-transpose transforms a committed geometric normal. Identity outer
transforms continue to use a native Embree instance. Nonfinite keyframe/outer
components and zero quaternions are rejected at build time. Nonfinite ray
times, singular composed transforms, and nonrepresentable finite inverses fail
closed as misses; inactive lanes do not evaluate these operations.

The Embree 4 object references exactly one each of
`rtcForwardIntersect{1,4,8,16}` and `rtcForwardOccluded{1,4,8,16}`. It has no
per-lane `rtcIntersect1` fallback. Embree 3.13.5 lacks the forwarding API, so
the compatibility object uses the documented recursive user-geometry route
and references exactly one matching-width `rtcIntersect{1,4,8,16}` and
`rtcOccluded{1,4,8,16}` operation. W2 is padded into W4; only W1 uses the
scalar interface. The Embree 3 check used official tag `v3.13.5`, commit
`698442324ccddd11725fb8875275dc1384f7fb40`, and covered compile plus object-
symbol shape rather than a linked runtime execution.

The permanent acceleration regression executes 35 rays at W1/W2/W4/W8/W16,
including packet tails, and checks closest hit, any hit, stateful query,
callback count, barycentrics, and distance. It also checks host SRT-generation
import without an explicit acceleration modification, buffer-only committed-
payload lifetime, USER-to-INSTANCE route replacement, singular-transform
misses, and restoration to the packet route. The test passes 6,405 assertions.
After complete builds, both Release trees pass the focused native-math,
Schedule-codegen, runtime-width, and acceleration gate 5/5, the SIMD-labelled
conformance gate 35/35, and their complete CTest inventories 140/140.

Fresh correctness smoke runs kept all outputs outside the repository. Ordinary
1024-spp path tracing accepts its checked-in reference at every width; observed
one-process FPS and RGB PSNR were 85.411/35.426871 at W1,
68.137/42.781428 at W2, 88.297/40.940426 at W4, 100.186/39.219318 at W8,
and 99.711/37.801696 at W16. The corresponding fallback observation was
90.441 FPS at 62.136588 dB. A fresh fallback motion-blur image was then used as
the comparison oracle: W1/W2/W4/W8/W16 pass at respectively
88.636435/88.279279/88.279279/88.279279/88.279279 dB. Their one-process render
times were 3213/5589/4332/3703/3474 ms versus 3279 ms for fallback. These are
correctness smokes, not a throughput claim: they are single sequential
observations, and the existing graphics kernels use identity outer transforms,
so they do not select the new nonidentity route.

## Rejected ray-query candidate SoA publication

The next measured boundary was candidate-payload ownership. The retained
status-aware callback already scans active state pointers to publish
`terminated` and candidate-kind bits, while a later JIT candidate read gathers
the `SurfaceHit` leaves from the 1,216-byte-per-lane AoS state. A prototype
therefore let that same status scan optionally publish a field-major packet of
`inst`, `prim`, both barycentrics, and `t`. The status/scratch coloring proof
also owned the packet, inactive lanes remained untouched, and a disabled
same-binary oracle restored the prior callback ABI and gathers. W1/W2/W4/W8/W16
sparse-mask and inactive-tail differential tests were exact. Raw W4+ LLVM IR
removed the seven masked gathers attributable to a complete candidate read.

That IR improvement did not survive the runtime boundary. The W16 fused
provider's optimized object showed five scalar candidate loads followed by five
field-major scalar stores for each surface lane. The first procedural version
unnecessarily paid the same five-field cost; a second version published only
the two defined procedural IDs. Multiple pinned, alternating processes were
run because unrelated load was present. Ratios below are paired geomeans, with
values above one favoring the prototype:

| workload | W8 | W16 | pairs | result |
| --- | ---: | ---: | ---: | --- |
| 16-candidate surface rejection chain, five fields | 0.988640x | 0.999267x | 6 | neutral/regressed |
| 16-candidate procedural rejection chain, five fields | 0.965845x | 0.932894x | 4 | regressed |
| procedural rejection chain, IDs only | 0.988005x | 0.983788x | 4 | still regressed |
| real 64-spp cutout path tracer, five fields | 1.012058x | 1.002874x | 4 | small, non-general gain |

Every cutout image passed the checked-in reference: W8 reported 33.351280 dB
and W16 33.337215 dB in both modes. The real W8 gain was too small to outweigh
the repeatable synthetic and procedural regressions, while W16 was effectively
neutral on the renderer. The prototype, its callback ABI extension, diagnostic
counter, environment oracle, and tests were therefore removed rather than
being presented as a production optimization.

A follow-up tested the suggested lane/value transpose directly. Instead of
five field-major stores, the provider copied one compact lane-major 20-byte
surface record or 8-byte procedural record; the JIT issued one contiguous
masked load and shuffled it into field vectors. It was also rejected after six
alternating pairs:

| compact AoS plus JIT transpose | W8 | W16 |
| --- | ---: | ---: |
| surface rejection chain | 0.992988x | 0.994474x |
| procedural rejection chain | 0.953886x | 0.906576x |

This isolates the boundary more strongly: even a compact copy plus transpose
loses to LLVM's existing direct gathers, especially for the two-field
procedural payload. A future payload experiment must therefore change native
provider ownership rather than copy an already materialized AoS state. The
remaining plausible form is consumer-field demand feeding values that the
provider naturally produces in packet registers, with no intervening per-lane
AoS publication. It should retain the same exact oracle and must win both the
surface/procedural rejection chains and at least one real renderer before being
kept.

After both prototypes were removed, complete Release builds of
`build-sdf-bench` and `build-sdf-tbb` passed. Each tree passed the focused
native-math/Schedule/runtime-width/acceleration gate 5/5, its full CTest
inventory 140/140, and the separately repeated SIMD-labelled conformance gate
35/35. Thus the audit leaves no callback ABI, environment switch, allocation,
or code-generation change in production.

## W16 full-packet coherent direct traversal

The next ordinary-renderer profile used sixteen workers pinned to logical CPUs
0--15. A 256-spp W8 `perf record` attributed 68.76% of sampled cycles to
Embree; the remaining large component was generated JIT code, while the SIMD
runtime was no longer the primary ceiling. Optimized W8/W16 objects also showed
that closest-hit and any-hit calls own separate, non-overlapping packet stack
regions. A function-scoped packet-scratch prototype reduced W8 stack size from
6,464 to 6,080 bytes and W16 from 10,368 to 9,600 bytes without changing the
instruction, vector-instruction, branch, call, or stack-reference counts. It
did not improve throughput: eight alternating 256-spp, one-spp-per-dispatch
pairs measured 1.00093x at W8 and 0.99831x at W16. The scratch reuse, metrics,
test extension, and environment switch were removed.

The productive boundary was instead Embree's traversal hint. Embree 4.4.1
initializes `RTCIntersectArguments` and `RTCOccludedArguments` with
`RTC_RAY_QUERY_FLAG_INCOHERENT`. An initial same-binary experiment selected
`RTC_RAY_QUERY_FLAG_COHERENT` for a full direct packet at every native width.
The ordinary path tracer proved that the policy must be width-specific:

| full-packet coherent / incoherent oracle | paired geomean | 95% CI | wins |
| --- | ---: | ---: | ---: |
| W4, 256 spp, 1 spp/dispatch | 0.90620x | [0.87358, 0.94004] | 0/8 |
| W8, 256 spp, 1 spp/dispatch | 0.98339x | [0.96945, 0.99753] | 2/8 |
| W16, 256 spp, 1 spp/dispatch | **1.02299x** | [1.00578, 1.04049] | 7/8 |

Production therefore changes only a direct W16 closest/any call for which all
sixteen embedded validity entries are active. W1/W2/W4/W8, partial and sparse
W16 packets, and stateful ray queries retain Embree's incoherent default.
`LUISA_SIMD_DISABLE_COHERENT_W16_DIRECT_TRACE=1` restores the complete direct
oracle. Running both modes without the rejected packet-scratch reuse measured
1.02938x [1.02227, 1.03654] across eight W16 pairs with 8/8 wins, proving that
the retained gain is traversal selection rather than stack-layout interaction.

The final binary was then measured on three direct-trace renderers and the
cutout control. Ordinary and HDR use SIMD's default 64 spp per dispatch; Camera
uses its fixed one-spp dispatch loop. Values above one favor coherent W16:

| renderer | workload | paired geomean | 95% CI | wins |
| --- | --- | ---: | ---: | ---: |
| ordinary | 256 spp | **1.04005x** | [1.03394, 1.04619] | 11/11 stable pairs |
| HDR | 256 spp | **1.03234x** | [1.02226, 1.04251] | 8/8 |
| Camera | 64 spp | **1.01714x** | [1.00863, 1.02573] | 8/8 |
| cutout | 64 spp, 1 spp/dispatch | 0.99847x | [0.99183, 1.00515] | 4/10 |

The ordinary run actually contained twelve alternating pairs and all twelve
favored the candidate. One oracle process overlapped the start of an unrelated
HIP renderer and fell to 67.22 from its neighboring 76.33--79.69 spp/s range.
Keeping that externally disturbed pair yields 1.05458x [1.02239, 1.08778]; the
table reports the conservative eleven-pair sensitivity result instead. Cutout
is intentionally retained as neutral evidence, not counted as a gain; every
candidate/oracle cutout PNG was byte-identical.

Five further alternating `perf stat` pairs exposed a stable mechanism. Across
all five candidate/oracle pairs, retired instructions were 0.89777x and retired
branches 0.92085x. In the first four undisturbed pairs, cycles were 0.96351x,
task-clock 0.96228x, and renderer throughput 1.03355x. Branch misses increased
to 1.08270x, so the win is not a generic branch-prediction improvement; Embree's
coherent W16 route executes materially less work despite somewhat less
predictable branches. A fifth candidate process was interrupted by external
load, but its instruction and branch counts remained at the same deterministic
levels.

A follow-up tried to replace the sixteen scalar early-exit validity checks with
a portable bitwise reduction over the documented `-1/0` entries. GCC lowered
the candidate to four unaligned SSE loads and an AND reduction. Two 64-spp
hardware-counter pairs reduced whole-process instructions by about 0.245% and
branches by 2.35%, but that static improvement did not produce a stable real
gain. Fifteen alternating 256-spp pairs measured 0.99833x
[0.98960, 1.00714] with 9/15 wins. The branchless helper, selector field, and
environment oracle were removed; the retained full-packet test remains the
simple early-exit form. The contaminated exploratory record and the clean
rejection gate are `/tmp/luisa-simd-w16-full-check-ab.i36spJ` and
`/tmp/luisa-simd-w16-full-check-clean.aitJsN`.

An eight-pair current-default comparison measured final W16 against fallback
at 256 spp. SIMD used sixteen workers and 64 spp per dispatch; fallback used
its one-spp default. W16 won 8/8 at **1.14574x** [1.12636, 1.16545], with
process medians of 81.397 and 70.878 spp/s. This is the current real Embree
renderer speedup for that machine/load configuration, not an ISPC comparison.
The official same-algorithm ISPC suite remains unchanged: Luisa retains its
1.21552x W8 and 1.20581x W16 five-workload geometric means, but there is still
no matched full-Embree ISPC renderer from which to claim an end-to-end ISPC
win.

A 1024-spp coherent-W16 correctness sweep passed the checked-in ordinary,
HDR, and Camera gallery references at respectively 37.800295, 41.472728, and
41.473939 dB RGB PSNR. The permanent acceleration test now runs once with
the production policy and once with the incoherent oracle; each execution
covers two full W16 packets plus a three-lane tail, in addition to every other
width. Raw process artifacts are
`/tmp/luisa-simd-direct-profile.xP23vW`,
`/tmp/luisa-simd-direct-ab-w8.tPZ8Sa`,
`/tmp/luisa-simd-direct-ab-w16.o6daDM`,
`/tmp/luisa-simd-coherent-w4.ydAgPT`,
`/tmp/luisa-simd-coherent-w8.UkYKII`,
`/tmp/luisa-simd-coherent-w16.hJgrpC`,
`/tmp/luisa-simd-coherent-w16-final.i2ZGIv`,
`/tmp/luisa-simd-coherent-w16-perf-stat-fixed.6v2FOr`,
`/tmp/luisa-simd-coherent-w16-camera.iyvz7W`,
`/tmp/luisa-simd-coherent-w16-hdr.gyHeSU`,
`/tmp/luisa-simd-coherent-w16-cutout.REJFA6`, and
`/tmp/luisa-simd-final-w16-vs-fallback.3867Vc`; correctness outputs are in
`/tmp/luisa-simd-coherent-w16-reference.*`,
`/tmp/luisa-simd-coherent-w16-hdr-reference.*`, and
`/tmp/luisa-simd-coherent-w16-camera-reference.*`.

Final validation completed full Release builds of both `build-sdf-bench` and
`build-sdf-tbb`. Each tree passes the focused native-math/Schedule/runtime-
width/production-accel/oracle-accel gate 6/6, its complete CTest inventory
141/141, and the separately repeated SIMD-labelled gate 36/36. The final
backend object has exactly the expected unresolved packet entries
`rtcIntersect{1,4,8,16}` and `rtcOccluded{1,4,8,16}` and no per-active-lane
scalar fallback. Disassembly confines the coherent-flag construction and
sixteen-entry validity scan to the W16 arms; the W4/W8 arms retain their prior
incoherent call shape.

## Verifier-legal arithmetic completion

The next compiler checkpoint completes Schedule-to-LLVM lowering for the XIR
arithmetic operations that previously reached the explicit unimplemented-op
diagnostic: rotate, `step`, count-leading/trailing-zero, population count, bit
reverse, signed/unsigned integer-exponent power, component reductions,
vector/matrix outer product, and 2x2/3x3/4x4 transpose, determinant, and
inverse. This is a capability and correctness change, not a measured speedup:
the old backend rejected these kernels, and no existing renderer hot path was
changed. Consequently there is no valid candidate/oracle throughput ratio to
report for this checkpoint.

The varying integer-power path is nevertheless machine-audited. Its raw helper
contains one fixed-vector exponentiation-by-squaring loop with vector multiply,
shift, select, and `llvm.vector.reduce.or`, no lane extract/insert loop, and no
target-specific intrinsic. Optimized assembly and object bytes contain no
`powf`. Rotate remains target-independent `llvm.fshl`/`llvm.fshr`. Uniform
inputs continue through one scalar helper invocation rather than one per lane.

Both Release build trees pass 141/141 full CTest and 36/36 separately repeated
SIMD-labelled tests. The focused arithmetic/runtime/native-math/Schedule gate
passes 5/5 in each tree. A fresh real-example sweep passes image processing,
voxel ray tracing, 1024-spp path tracing, and the 1024-spp SDF renderer against
their checked-in references at W1/W2/W4/W8/W16. Image processing reports
89.251953 dB RGB PSNR at every width, voxel reports 82.834519 dB, and SDF
reports 63.129346 dB. Path tracing reports respectively 35.426795, 42.781582,
40.940376, 39.219305, and 37.800295 dB for W1/W2/W4/W8/W16. These single-run
correctness executions are intentionally not treated as stable performance
measurements.

## Direct sampled-texture completion

The next resource checkpoint lowers all eight direct 2D/3D `sample`,
`sample_level`, `sample_grad`, and `sample_grad_level` operations. This is a
capability and correctness stage, not a performance claim: the previous SIMD
compiler rejected these instructions, so there is no valid old/new throughput
ratio. The JIT now computes gradient LOD with target-independent fixed-vector
IR, sanitizes inactive operands before arithmetic and scratch publication, and
crosses one packet callback. Uniform gradients execute one scalar `log2`, and
a uniform sampled result invokes only the first active lane. The runtime groups
divergent sampler codes and interprets every LOD relative to the texture view's
bound base mip. The appended callback consumes existing descriptor padding,
leaving the 64-byte texture argument ABI and prior offsets unchanged.

The implementation also corrects three sampler semantics shared by bindless
and direct paths: `POINT` and `LINEAR_POINT` select the nearest mip, linear
`REPEAT` preserves tap order at the wrap seam, and linear `ZERO` performs the
required border blend rather than collapsing the two taps. A fractional 1.6
POINT-LOD case, a nonzero base-mip direct case, sparse samplers, every runtime
width, and a three-lane W16 tail are permanent independent oracles. The LLVM
shape gate requires one varying callback with the exact tail mask, one
first-active-lane uniform callback, the shared fixed-vector native `log2`
symbol, no lane extract/call/insert path, and no optimized `log2f` or platform
vector-library symbol.

Fresh full Release builds of `build-sdf-bench` and `build-sdf-tbb` pass their
complete 142/142 CTest inventories. The separately repeated SIMD-labelled gate
passes 37/37. Checked-in references remain read-only. At W1/W2/W4/W8/W16,
image processing passes at 89.251953 dB RGB PSNR and Voxel at 82.834519 dB;
1024-spp path tracing passes at 35.426871/42.781428/40.940426/39.219318/
37.800219 dB, and 1024-spp non-coroutine SDF passes at 63.129346 dB at every
width. Every path process reports Embree 4.4.1 native W4/W8/W16 packet support.
These sequential executions are correctness smokes, not new stable throughput
measurements; the official same-algorithm ISPC and fallback-relative tables
above therefore remain the current performance evidence.

## Cooperative block and shared-memory completion

The cooperative-block checkpoint adds shared allocas, shared atomics, and
block barriers without changing the entry selected for an ordinary kernel.
Each fixed-vector packet is one LLVM coroutine, and one exclusive block wrapper
drives the statically known packet handles through barrier phases. The wrapper
rejects mismatched static barrier IDs and completion mixed with live packets;
the inner scheduler independently requires the complete live mask and no
pending cohort before suspension. A 35-thread one-dimensional dispatch and a
`{11, 5}` dispatch over `{8, 4}` blocks permanently cover mixed and non-prefix
edge masks. The existing multi-barrier/shared-atomic programs run at
W1/W2/W4/W8/W16. At this checkpoint barriers in natural loops were still
rejected pending a uniform dynamic-instance proof; the exact loop-epoch
implementation in the next checkpoint supersedes that restriction.

The compiler-focused Release tree passes 143/143 tests. Both maintained
fallback+SIMD Release trees pass 154/154, including the complete tutorial,
graphics, acceleration, ray-query, local-memory, atomic, bindless, and
runtime-width inventories. The focused cooperative gate passes 13/13, and the
required precise/fast native-math plus runtime-width gate passes 3/3. Ten
shared-memory objects across the supported widths have no unresolved symbol;
only the cooperative wrapper is externally visible, while packet, resume, and
destroy functions have local linkage.

Because unrelated host work was active, the ordinary-path ISPC regression
audit used two independent seven-round runs rather than a single process.
Every process was pinned to logical CPU 6 with one worker. Implementation order
rotated and reversed within each run. The control was official ISPC 1.31.0,
`--cpu=znver5`, precise arithmetic with FMA contraction disabled, and
AVX-512 x8/x16 at the matching logical width. All exact workloads were
bit-identical; the analytic path passed its independent tolerance. Combining
all fourteen paired rounds, without removing the visibly disturbed first
Mandelbrot pair, gives:

| workload | Luisa W8 / ISPC x8 | 95% CI | Luisa W16 / ISPC x16 | 95% CI |
| --- | ---: | ---: | ---: | ---: |
| Mandelbrot | **1.40043x** | [1.38054, 1.42060] | **1.54230x** | [1.51886, 1.56610] |
| masked stream | **1.48404x** | [1.43251, 1.53742] | **1.44505x** | [1.40559, 1.48562] |
| AoS-to-SoA | 1.00963x | [1.00008, 1.01927] | 0.98300x | [0.97407, 0.99201] |
| GEMM | **1.34093x** | [1.33743, 1.34444] | **1.38131x** | [1.37978, 1.38284] |
| analytic path | **1.09611x** | [1.09378, 1.09845] | **1.02332x** | [1.02077, 1.02587] |

The unweighted five-workload geometric means are 1.25264x at W8 and 1.25366x
at W16. W8 wins every combined point estimate, although the AoS lower bound is
effectively a tie; W16 wins four of five. This independently confirms the
ordinary-kernel compiler lead after the cooperative changes. It does not
replace the longer official table above, establish a cooperative-kernel
speedup, or supply the still-missing matched full-Embree ISPC renderer. Raw
records are
`/tmp/luisa-simd-ispc-cooperative-confirm-a-7r-20260816.json` and
`/tmp/luisa-simd-ispc-cooperative-confirm-b-7r-20260816.json`.

## Repeated cooperative-barrier completion

The next cooperative checkpoint supports a block barrier inside one or more
natural loops without weakening the fail-closed contract. Only loops that
enclose a static barrier receive state. Each packet coroutine retains one
64-bit epoch per lane for each such loop, increments participating lanes on an
annotated back-edge, rejects overflow, and requires all live lanes to publish
one common epoch before suspension. The outer block wrapper compares the exact
`(static barrier ID, enclosing loop epoch tuple)` across live packets before it
resumes any handle. It neither hashes the tuple nor uses an encounter ordinal,
and it ignores epochs of unrelated loops. Thus a packet that skips a static
site in one iteration cannot rendezvous with a packet reaching that site in a
different iteration.

Permanent regressions include the original minimal XIR failure, a negative
runtime child process where packet zero and the remaining packets reach the
same static site in different iterations, and nested two-level loops with two
barriers at W1/W2/W4/W8/W16 over a 35-thread inactive tail. The AST/JIT shape
test independently requires two tracked loop epochs for one nested static
site. The existing batch-softmax integration now runs at all five widths. Its
final partial block is explicitly padded to a complete 1024-thread reduction:
the kernel already guards the logical element count and initializes padding,
whereas dispatching only 3073 invocations left 1023 shared slots outside the
dispatch extent and accidentally depended on fresh scratch storage.

After complete builds, the compiler-focused tree passes 148/148 tests and both
maintained fallback+SIMD trees pass 159/159. The required native-math and
runtime-width gate passes 3/3. The syntax-check and standalone-ISPC driver
Python suites pass 21/21, and clangd reports no diagnostic for all eleven
changed C++ translation units. Dumped W8 softmax JIT objects contain no
undefined symbol or scalar math call; only the cooperative wrapper is global,
while the packet body, resume, and destroy functions retain local linkage.

Fresh checked-in-reference comparisons pass image processing and Voxel at
W1/W2/W4/W8/W16 with 89.251953 dB and 82.834519 dB RGB PSNR. The ordinary
1024-spp Embree path tracer passes at 35.426795/42.781582/40.940376/39.219305/
37.800295 dB and every process reports Embree 4.4.1 native W4/W8/W16 packet
support. The non-coroutine 1024-spp SDF renderer passes at 63.129346 dB at
every width. These are correctness executions, not performance samples.

The ordinary-kernel ISPC control was rebuilt twice from official ISPC 1.31.0
and measured in two independent seven-round runs. Each process used one worker
pinned to logical CPU 6; orders rotated and reversed, the CPU target was
`znver5`, arithmetic was precise with FMA contraction disabled, exact outputs
were bit-identical, and analytic-path outputs passed the independent tolerance.
Combining all fourteen pairs gives:

| workload | Luisa W8 / ISPC x8 | 95% CI | Luisa W16 / ISPC x16 | 95% CI |
| --- | ---: | ---: | ---: | ---: |
| Mandelbrot | **1.40872x** | [1.40464, 1.41280] | **1.55741x** | [1.55438, 1.56044] |
| masked stream | **1.45065x** | [1.39167, 1.51214] | **1.42450x** | [1.38677, 1.46326] |
| AoS-to-SoA | 0.99406x | [0.98101, 1.00728] | 0.97195x | [0.95997, 0.98408] |
| GEMM | **1.34184x** | [1.33756, 1.34614] | **1.37884x** | [1.37741, 1.38027] |
| analytic path | **1.09687x** | [1.09375, 1.10001] | **1.02321x** | [1.02154, 1.02487] |

The unweighted five-workload geometric means are **1.24489x at W8** and
**1.24922x at W16**. Four workloads win at both widths; W8 AoS-to-SoA is a
statistical tie and W16 AoS-to-SoA remains 2.80% behind. This establishes a
lead over matched-width ISPC for this five-workload compiler suite, including
its asset-free analytic path tracer. It still does not establish a win for the
repository's complete Embree renderer because no matched full-renderer ISPC
implementation exists. Raw records are
`/tmp/luisa-simd-ispc-loop-barrier-a-7r-20260816.json` and
`/tmp/luisa-simd-ispc-loop-barrier-b-7r-20260816.json`.

## Graphics device-loop completion

The offline fire simulation exposed a front-end/JIT scaling defect rather than
a packet-scheduler defect. Its render kernel used a C++ loop over the selected
256 particles, so recording cloned the complete body 256 times into the AST.
The old executable did not produce a frame within 300 seconds, kept one host
thread busy while the worker pool slept, and reached approximately 2,798,044
KiB RSS. Expressing the same iteration space as a DSL `$for` emits one device
loop. The complete 200-frame offline run then finishes in about 5.6 seconds at
approximately 132 MiB RSS. This is a greater-than-53x end-to-end wall-time
lower bound and a greater-than-21x resident-memory reduction; it is primarily
an AST construction and LLVM optimization-size fix, not a claim that the SIMD
scheduler alone became 53x faster.

The output-state oracle and checked-in image comparison pass. Fallback reaches
85.889909 dB RGB PSNR and every SIMD width reaches 85.348272 dB. With the SIMD
pool fixed at 32 workers, independent complete-process repetitions give these
wall-time medians; speedup is fallback time divided by SIMD time:

| backend/width | median seconds | throughput vs fallback |
| --- | ---: | ---: |
| fallback | 11.62 | 1.000x |
| SIMD W1 | 25.13 | 0.462x |
| SIMD W2 | 21.29 | 0.546x |
| SIMD W4 | 10.62 | **1.094x** |
| SIMD W8 | 5.78 | **2.010x** |
| SIMD W16 | 3.24 | **3.586x** |

The first W2 series overlapped unrelated host work and contained two obvious
slow samples. It was discarded in full, then repeated five times; the clean
20.81--21.36-second series supplies the table. Every other entry is the median
of three successful complete runs. These are repeated process medians, not a
paired confidence interval.

W2 is a semantic and correctness width, but not a native two-lane x86
execution width. Its render packet contains 1,495 static instructions versus
1,324 at W4 while processing only half as many lanes; both map principally to
XMM operations. This accounts for the W2 regression without implicating an
incorrect width selection. W16 uses ZMM operations heavily. Both dumped W16
JIT objects have no undefined symbol and no varying `sinf`, `cosf`, `expf`,
`logf`, `powf`, SVML, SLEEF, or other scalar-libm dependency. The retained
artifacts are under `/tmp/luisa-simd-fire-w16-20260820/`.

The fire program is now a permanent offline SIMD graphics test, and
`helloworld` is a permanent backend-startup/ABI test. The latter fixes its
half/ushort aggregate at four-byte alignment; two-byte aggregate alignment
remains deliberately invalid for the cross-backend/DXC structure ABI, with a
core type-system negative regression. Complete Release CTest inventories pass
149/149 in the compiler-focused tree and 161/161 in each of the two maintained
fallback+SIMD/Embree trees. The focused native-math, runtime-width,
`helloworld`, and fire gate passes 5/5.

A broader W8 offline capability sweep also completes base, camera, cutout,
HDR, nested-callable, ray-mask, and spectrum path tracing; photon mapping;
procedural ray query; black-hole, SpaceX, and visual shaders; SDF and both
XIR-to-AST examples; Voxel; image processing; shader toy; game of life; and
the MHA/MLA examples. MHA and MLA pass their independent CPU-reference checks.
The short rendering sweeps are capability coverage, not stable performance
samples. The current official ISPC claim therefore remains scoped to the
matched five-workload compiler suite above: its geomean is 1.24489x at W8 and
1.24922x at W16 in Luisa's favor, while no matched ISPC implementation of the
complete Embree renderer exists.

## Ray-query representation compaction and reconstruction

The two shared XIR lowerings now name their destination explicitly:
`lower_ray_query_to_pipeline` outlines callbacks for fallback/CUDA/HIP/coroutine
consumers, while `lower_ray_query_to_loop` exposes structured loop/if control
for SIMD and native SPIR-V. Deprecated forwarding headers and exported symbols
retain the two former public spellings. Pipeline outlining now accepts multiple
Branch exits from a handler and writes captured outputs at every outlined
return. Loop lowering reuses the update block as the candidate-selection merge,
bypasses exact no-op arms, and removes the obsolete dispatch shell immediately.

The new `reconstruct_ray_query_loop` pass folds the complete canonical
`PROCEED -> IS_TERMINATED -> candidate dispatch -> update` form back into
`RayQueryLoopInst`/`RayQueryDispatchInst`. Besides the shell emitted by
`lower_ray_query_to_loop`, it recognizes the pre-mem2reg `SimpleLoopInst`
generated by the affine DSL `query`/`query_any` API and an exact
`$while (query.proceed())` loop. It preserves the query object,
surface/procedural regions, merge PHIs, and loop/dispatch metadata; nested
structured handler control and multiple handler exits remain legal. Ordinary
loops are ignored. A loop containing `PROCEED` that is only a near-match is an
error, and all function/module candidates are preflighted before the first
mutation. The round-trip, nested-handler, malformed-late-module, multi-exit,
no-op-arm, explicit query-all, reversed query-any-motion dispatch, missing
candidate split, and mixed valid/malformed frontend cases are permanent
regressions. A runtime two-surface query exercises target-dependent commit on
fallback and SIMD with a five-lane dispatch, so the SIMD case also covers an
inactive packet tail.

Fresh full builds of the maintained Release tree and the Clang ASan tree
complete. Release validation passes the focused ray-query/XIR set 7/7, the
SIMD-labelled runtime/graphics gate 47/47, and the complete configured suite
162/162. The explicit runtime query also passes 48 assertions on fallback and
at forced SIMD W1/W2/W4/W8/W16; the five-lane W2/W4/W8/W16 runs exercise
inactive packet tails. Changed-line clang-format is clean and all nine changed
translation units present in the Release compile database pass the clangd
syntax checker (fallback retains one pre-existing unused-include warning).
The directly affected ASan pass and explicit-query runtime cases pass. The
broader ASan `test_simd_accel` still crashes in the pre-existing SRT-motion
forwarding/JIT path (`rtcForwardIntersect8Ex` with default workers, an unrelated
JIT entry with one worker), while the same complete test passes in Release;
therefore this stage makes no whole-suite ASan claim.

For the W8 four-candidate triangle query, a retained pre-change executable and
the new executable report the same five state slots, three instruction spills,
three convergence points, native predicated loop, and 9,728-byte query scratch.
The only structural change is `schedule_blocks=11 -> 9`. Five alternating
old/new process pairs had process-median times of 16.5007 and 16.4047 ms, a
nominal 1.0059x ratio whose direction reversed in two pairs. This is not a
stable throughput result; the accepted result is the two-state reduction.

The shared host was concurrently loaded (observed load average 9.42--12.62), so
the following figures use medians of independent processes, each of which
already reports the median of seven warmed samples. The 16-candidate triangle
query has five processes per entry; throughput is fallback time divided by SIMD
time:

| backend/width | process-median seconds | throughput vs fallback |
| --- | ---: | ---: |
| fallback | 0.056885 | 1.0000x |
| SIMD W1 | 0.073426 | 0.7747x |
| SIMD W2 | 0.082601 | 0.6887x |
| SIMD W4 | 0.070602 | 0.8057x |
| SIMD W8 | 0.064673 | 0.8796x |
| SIMD W16 | 0.067580 | 0.8417x |

The 16-candidate procedural query has three rotated-order processes per entry:

| backend/width | process-median seconds | throughput vs fallback |
| --- | ---: | ---: |
| fallback | 0.075036 | 1.0000x |
| SIMD W1 | 0.148389 | 0.5057x |
| SIMD W2 | 0.131992 | 0.5685x |
| SIMD W4 | 0.092807 | 0.8085x |
| SIMD W8 | 0.073920 | **1.0151x** |
| SIMD W16 | 0.062829 | **1.1943x** |

Thus compaction does not cure the triangle rejection chain, whose Embree and
stateful-callback boundary still favors fallback. The compute-heavier
procedural path amortizes that machinery and reaches parity at W8 and a 1.19x
process-median lead at W16. These are workload-specific measurements, not a
general ray-query speedup claim.

Validation passes 150/150 in the compiler-focused Release tree and 162/162 in
each maintained fallback+SIMD/Embree Release tree. The three focused lowering
tests also pass under Clang ASan with leak detection. The reconstruct suite has
9 tests/117 assertions, pipeline lowering 14/177, destructure 35/246, and the
combined XIR pass suite 392/2,445. A fresh W8 1024-spp ordinary Embree path trace
accepts the checked-in gallery reference at 39.219284 dB RGB PSNR and reports
native Embree 4.4.1 W4/W8/W16 packet support.

## Compile-time Embree triangle-filter indexing

The next profile separated the remaining triangle-query host boundary from the
explicit-PC scheduler. On the isolated W8 16-candidate rejection benchmark,
the pre-change sample attribution was 35.18% Embree traversal, 13.82% in the
wide triangle filter, 7.95% in `ray_query_proceed_triangle_only_impl<true>`,
5.86% in batch installation, and 1.40% in the JIT-visible proceed-status
callback. Individual scheduler/JIT sites were below one percent. Runtime-`N`
`RTCRayN_*`/`RTCHitN_*` addressing and a large inlined full-batch heap path were
therefore a measurable filter cost, while explicit PC dispatch alone was not
the dominant term.

The accepted implementation provides width-specialized filter bodies for
Embree N=1/4/8/16, retains a cold generic runtime-N fail-safe, and outlines the
rare full-32-candidate heap replacement path. The same binary selects the old
generic addressing with
`LUISA_SIMD_DISABLE_SPECIALIZED_TRIANGLE_FILTER=1`. A fresh specialized W8
profile attributes 39.51% to Embree, 8.29% to proceed, 5.58% to the filter,
4.98% to batch installation, and 1.64% to proceed-status. Thus the filter's
own sampled share falls from 13.82% to 5.58%; the larger Embree percentage is
the expected redistribution after shortening the callback.

Five alternating same-binary process pairs give these medians:

| triangle rejection | specialized seconds | generic-oracle seconds | speedup |
| --- | ---: | ---: | ---: |
| SIMD W8 | 0.081142 | 0.085377 | 1.0522x |
| SIMD W16 | 0.081524 | 0.085093 | 1.0438x |

Candidate counts 1, 16, 32, 33, and 64 pass at W1/W2/W4/W8/W16, covering the
under-capacity boundary, first overflow/rescan, padded W2 packet, sparse tail,
and every provider packet width. The permanent 35-candidate device regression
continues to prove exact callback count and the farthest committed instance at
every width. A rejected reverse-batch-cursor experiment changed seven-pair
cycle count by approximately +0.07% and retired essentially the same number of
instructions, so it was removed rather than retained on source-level appeal.

The real 64-spp cutout renderer also uses this filter. Seven alternating pairs
produce:

| renderer | specialized median spp/s | generic-oracle median spp/s | paired geometric mean | wins | 95% CI |
| --- | ---: | ---: | ---: | ---: | ---: |
| SIMD W8 | 47.0006 | 46.2230 | 1.0119x | 6/7 | [1.0003, 1.0237] |
| SIMD W16 | 47.4029 | 46.4882 | 1.0209x | 7/7 | [1.0109, 1.0310] |

Every image hash is identical within a width and oracle pair. A seven-pair W8
ordinary closest-hit path trace, which never enters the ray-query filter, is a
negative control at 1.0022x with a confidence interval crossing one
([0.9990, 1.0053]); its image hashes are also identical.

A separate rotated cutout sweep was interrupted after five clean rounds by an
unrelated full-core build. The contaminated final two rounds are excluded.
The five clean process medians give the current width comparison:

| backend/width | median spp/s | paired throughput vs fallback |
| --- | ---: | ---: |
| fallback | 70.4716 | 1.0000x |
| SIMD W1 | 50.7517 | 0.7203x |
| SIMD W2 | 35.1541 | 0.4959x |
| SIMD W4 | 42.7861 | 0.6075x |
| SIMD W8 | 48.2323 | 0.6853x |
| SIMD W16 | 47.2393 | 0.6689x |

The specialization is therefore a real 1--2% renderer improvement, not a
closure claim: callback-heavy cutout remains slower than fallback. At this
checkpoint, the public DSL already provided the next useful semantic boundary.
`traverse` retains a complete structured handler region that can become
`RayQueryPipelineInst`, whereas explicit `query`/`query_any` exposes each
`proceed()` and at this checkpoint stayed on the loop route. The later
reconstruction stage documented below proves and consumes the canonical inline
form without adding W4/W8/W16 controls to the DSL.

## Capture-free structured ray-query pipeline

The SIMD front door now runs one early local-store-forwarding pass before it
classifies a structured query. In the real cutout kernel this removes two
front-end callable ABI temporaries per handler: one single-store/load
`TriangleHit` argument alloca and one single-store/load Boolean return alloca.
They are private compiler scratch, not `tall_inst`/`short_inst`, resource
captures, or live-outs. Both cutout queries consequently report
`direct_ray_query_pipelines=2`; the W8 main schedule falls from 53 to 31 blocks
and from 31 to 28 instruction spills. A real captured value/resource or
mutable live-out still fails the zero-capture selection and follows the prior
loop implementation. At this stage an explicit DSL `proceed()` loop was a
compile-tested negative control; the later selective reconstruction stage
supersedes that policy for the exact canonical form.

Each outlined handler has the internal width-specialized ABI
`void(i32, i64, ptr, ptr)` for lane count, exact sparse mask, a null-sanitized
array of lane state pointers, and the launch record. The main packet advances
one status-provider loop until every original lane terminates and invokes the
surface/procedural handler only for its classified subset. Width/mask mismatch,
null or divergent callbacks, null active states, overlapping classifications,
and unclassified live lanes fail closed. There is no public DSL, Embree, accel
descriptor, or query-state ABI change.

Same-binary cutout measurements compare the default with
`LUISA_SIMD_DISABLE_DIRECT_RAY_QUERY_PIPELINE=1`. W2 uses the longer 128-spp
confirmation sweep; the other rows use 64 spp. W1 was forced only for this
ablation and is disabled in production after its stable regression.

| Width | direct median spp/s | loop median spp/s | paired geomean | wins | 95% CI |
| ---: | ---: | ---: | ---: | ---: | ---: |
| W1, forced | 47.4825 | 48.4208 | 0.9809x | 0/7 | [0.9755, 0.9862] |
| W2 | 34.1090 | 32.8862 | 1.0341x | 6/7 | [1.0027, 1.0666] |
| W4 | 40.4739 | 39.5092 | 1.0257x | 7/7 | [1.0141, 1.0374] |
| W8 | 45.5710 | 44.6926 | 1.0196x | 10/10 | [1.0151, 1.0242] |
| W16 | 44.3346 | 43.7513 | 1.0152x | 7/7 | [1.0023, 1.0282] |

The final policy is therefore direct W2/W4/W8/W16 and loop W1. Seven fresh
fallback-relative rounds still leave cutout below fallback at every width;
the paired geometric means are 0.6779x/0.4973x/0.6045x/0.6800x/0.6587x from
W1 through W16. W1's loop-gated median is 48.396 spp/s. The balanced
W2/W4/W8/W16 sweep gives medians 33.634/40.715/45.781/44.360 spp/s; its
fallback median is 66.615 spp/s. The W1-specific paired fallback median is
71.830 spp/s. Because unrelated host work changed between sweeps, the paired
ratios, rather than a pooled raw fallback value, are the comparison of record.

At W8, the optimized main object changes from 29,203 to 22,995 text bytes,
6,602 to 5,094 static instructions, 465 to 317 branches, 1,243 to 966 stack
references, and a `0x4cc0` to `0x48c0` stack frame. The direct module also owns
two identical 4,618-byte surface handlers, each with 1,138 instructions, 135
branches, and 166 stack references. Thus total static text grows to 32,231
bytes even while the hot main body shrinks; main callsites rise from five to
seven. Both objects have no undefined symbols.

Three alternating 256-spp `perf stat` pairs confirm a dynamic rather than
source-size win. Direct/loop mean ratios are 0.9744 for cycles, 0.9658 for
retired instructions, 0.9818 for branches, and 0.9743 for aggregate task
clock. Branch misses increase 3.25%, so the gain comes from less total
scheduler work rather than a universally better branch predictor outcome.
A matching direct flat profile attributes 36.36% of cycles to JIT code, 36.28%
to Embree, and 25.38% to the SIMD backend/runtime. The largest named backend
sites are query proceed at 6.95%, the specialized triangle filter at 4.44%,
batch installation at 4.03%, and status publication at 1.95%. The explicit
loop profile remains close at 37.28% JIT, 35.41% Embree, and 24.93% backend;
the dynamic counters provide the clearer scheduler delta.

The permanent DSL/JIT regression covers W1/W2/W4/W8/W16, an inactive tail,
exact candidate versus loop-oracle output, provider call count, front-end
`Callable(TriangleHit)` argument/return scratch, an explicit-proceed negative
control, and a structured-query callable inlined at two call sites. Real
two-spp cutout images are byte-identical between direct and loop at each of all
five widths. This closes the capture-free vertical slice, not general captured
handlers: a later uniform/resource capture ABI must preserve scalar uniform
evaluation and win a resource-using renderer before it is enabled.

## Captured structured ray-query pipeline

The real `examples/rendering/procedural.cpp` query is the first graphics
vertical slice with semantic captures. After the front-end cleanup it retains
four input captures, including the AABB buffer descriptor and mutable
lane-local state used by the procedural handler. The public DSL already
expresses the required whole-query boundary and lexical captures; the missing
piece was the SIMD-private callback ABI, not a width- or Embree-specific DSL
operation.

Schedule IR now keeps the query plus captured arguments as pipeline
dependencies. Outlined handlers receive the caller-proven
`warp_uniform -> cohort_uniform -> varying` parameter classes: uniform values
and resource descriptors remain scalar, varying data keeps fixed-vector/SoA
form, and reference captures use the existing per-lane local handle. The LLVM
handler ABI is privately extended from `void(i32, i64, ptr, ptr)` to
`void(i32, i64, ptr, ptr, capture_0, ...)`. Both callbacks share the exact
typed capture list, and count/type/class mismatches fail closed. No public DSL,
XIR pass-info, runtime descriptor, Embree, or query-state ABI changes.

The profitability sweep used the Ryzen 9 9950X3D's 16 physical cores
(`taskset -c 0-15`, `LUISA_SIMD_WORKER_COUNT=16`) and alternated fresh
processes. W2 used 12 pairs at 64 spp. W8/W16 use the longer 16-pair, 128-spp
confirmation after their first 64-spp intervals crossed one. W4 is the final
12-way rotated direct/loop/fallback run at 64 spp.

| Width | direct median ms | loop median ms | direct/loop speedup | wins | 95% CI |
| ---: | ---: | ---: | ---: | ---: | ---: |
| W2, forced | 481.794 | 477.278 | 0.9759x | 4/12 | [0.9461, 1.0067] |
| W4 | 302.565 | 314.781 | 1.0290x | 9/12 | [1.0026, 1.0561] |
| W8, forced | 449.127 | 447.358 | 0.9900x | 7/16 | [0.9748, 1.0053] |
| W16, forced | 374.910 | 373.298 | 0.9932x | 8/16 | [0.9714, 1.0154] |

Only W4 has repeatable positive evidence, so production enables captured
pipelines only at W4 and only up to four captures. W2/W8/W16 retain the loop;
W1 already retains it. `LUISA_SIMD_DISABLE_CAPTURED_RAY_QUERY_PIPELINE=1`
provides the W4 oracle, while
`LUISA_SIMD_FORCE_CAPTURED_RAY_QUERY_PIPELINE=1` removes the width/count gate
for tests and experiments. Capture-free W2/W4/W8/W16 policy is unchanged.

The same 64-spp renderer gives the current production width comparison against
fallback. Each non-W4 row has ten alternating pairs; W4 uses the twelve rotated
triples above. Raw fallback medians drift between sweeps, so the paired ratio
and interval are the comparison of record.

| backend/width | SIMD median ms | paired fallback median ms | SIMD speedup vs fallback | 95% CI |
| --- | ---: | ---: | ---: | ---: |
| W1 | 534.403 | 269.245 | 0.5065x | [0.4975, 0.5157] |
| W2 | 484.378 | 269.876 | 0.5639x | [0.5467, 0.5817] |
| W4, captured direct | 302.565 | 296.201 | 0.9528x | [0.9214, 0.9852] |
| W8 | 202.186 | 268.751 | 1.3308x | [1.3078, 1.3543] |
| W16 | 173.141 | 275.108 | 1.5775x | [1.5504, 1.6052] |

Thus captured direct lowering closes about three percent at W4 but does not
make W4 beat fallback. The wider packet traversal is what changes the renderer
result: the retained-loop W8 and W16 paths are respectively about 1.33x and
1.58x faster than fallback. W1/W2 expose the opposite limit: packet query
state, callbacks, and scattered candidate memory dominate before useful SIMD
width amortizes them.

At W4, the direct main schedule falls from 25 to 7 blocks, 8 to 2 convergence
points, 22 to 6 state slots, and 15 to 4 instruction spills. Static assembly is
not smaller: direct/loop report 3,147/3,023 instructions, 288/274 branches,
709/749 stack references, 8,256/8,128 stack-allocation bytes, and three calls
each; both report zero scalar-math calls. Five serial 256-spp `perf stat` runs
show why it still wins dynamically:

| counter | direct | loop | direct/loop |
| --- | ---: | ---: | ---: |
| cycles | 89,996,108,496 | 93,488,170,416 | 0.9626 |
| instructions | 354,636,654,267 | 367,210,511,373 | 0.9658 |
| branches | 44,116,384,290 | 44,452,322,492 | 0.9924 |
| branch misses | 126,327,644 | 115,722,102 | 1.0916 |

The direct route retires 3.42% fewer instructions and 3.74% fewer cycles even
though its static body is larger and its branch misses increase. This
quantifies reduced dynamic scheduler turnover rather than an LLVM inlining or
code-size effect. Generic cache counters also show 0.512x cache references and
0.955x misses, but those architecture-generic Zen events are retained only as
directional evidence, not a cache-level traffic model.

An explicit handler-`alwaysinline` experiment was rejected. On the W8 cutout
renderer it removed two calls and both standalone handlers, but total text grew
609 bytes, static instructions grew 64, stack references grew 71, and the main
frame grew 1,216 bytes. Twelve alternating pairs gave 0.9871x with 95% CI
[0.9776, 0.9967] and only 1/12 wins. The production `inlinehint` remains: fewer
call instructions are not useful when they increase ray-query state pressure.

Permanent coverage now includes capture projection, invalid override
diagnostics, uniform parameter preservation, a scalar buffer descriptor, a
varying output reference, exact sparse tail masks, provider-call counts, and
candidate-versus-loop equality at W1/W2/W4/W8/W16. A compile-only DSL case
also carries buffer, image, bindless, accel, and varying-index captures through
one W8 handler. The forced W8 procedural
image is byte-identical to its loop oracle; the default W4 image is likewise
byte-identical, and its fallback reference comparison passes at 48.016 dB
PSNR.

## W1 resident structured ray-query pipeline

The earlier W1 direct experiment lost because it reused the packet status loop
and added a handler call without amortizing useful width. The replacement keeps
the complete ordered advance/scan/commit loop resident in the runtime for one
structured W1 query. The JIT enters it once and supplies one stable
opaque-capture callback target; that thunk validates the candidate kind and
dispatches the surface or procedural handler. LLVM inlines both typed handlers
into the thunk. The runtime still owns candidate collection and sorting.

An attempted Embree-filter streaming implementation was rejected before this
design was accepted. Three overlapping procedural AABBs form the minimal
counterexample: Embree delivered callback order `2, 1, ...`, whereas the
existing SIMD loop oracle publishes deterministic order `0, 1, 2`. Calling the
DSL handler from that filter would therefore change observable callback state.
The resident path instead reuses the same candidate batches, cursors,
continuation scans, opacity, and query-any rules as `PROCEED`.

No public DSL extension was necessary. The high-level
`traverse(...).on_*().trace()` API already creates an AST `RayQueryStmt`, and
AST-to-XIR directly creates `RayQueryLoopInst`. The pipeline pass consumes that
structured node before the later reconstruct pass considers a user-written
inline `proceed()` loop. This stage extends only the private JIT/runtime ABI;
at this checkpoint explicit `proceed()` remained on the loop route. The later
section records the exact shape proof and cost gate that now permit its
canonical immediate-dispatch form to use the same resident implementation.

The profitability run pinned the Ryzen 9 9950X3D's 16 physical cores and used
16 SIMD workers. Each row alternated fresh resident and loop-oracle processes;
the interval is a 95% interval over paired log speedups. The synthetic benchmark
runs 8,388,608 rays per internal sample.

| workload | resident median | loop median | paired geomean | wins | 95% CI |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 procedural candidate | 28.381 ms | 32.813 ms | **1.1538x** | 9/9 | [1.1430, 1.1647] |
| 16 rejected/committed candidates | 151.953 ms | 184.753 ms | **1.2086x** | 9/9 | [1.1930, 1.2244] |
| real procedural renderer, 128 spp | 1089.026 ms | 1096.349 ms | **1.0099x** | 17/24 | [1.0019, 1.0180] |

Five additional 256-spp `perf stat` pairs quantify the final path. Ratios below
are resident divided by loop, so values below one are reductions.

| counter | resident/loop |
| --- | ---: |
| cycles | 0.9925 |
| retired instructions | 0.9429 |
| branches | 0.8842 |
| branch misses | 0.9942 |
| aggregate task clock | 0.9875 |

The resident path therefore removes 5.71% of dynamic instructions and 11.58%
of dynamic branches, while its IPC falls from 2.916 to 2.771 across the
unavoidable runtime-to-JIT callback. The ordinary 24-pair renderer sweep is the
throughput result of record; the instrumented five-pair wall-clock mean was
1.0300x. The final optimized renderer object is 20,056 bytes with 519 static
instructions, 27 branches, 69 stack references, and a 1,464-byte frame. Its
loop oracle is 21,193 bytes, 507 instructions, 28 branches, 83 stack references,
and a 1,480-byte frame. Both have three calls and no scalar math symbols. The
dynamic win is scheduler turnover, not a claim that fewer static instructions
alone predict throughput.

Production now enables capture-free W1 pipelines and captured W1 pipelines up
to four semantic captures. `LUISA_SIMD_DISABLE_DIRECT_RAY_QUERY_PIPELINE=1`
remains the same-binary loop oracle; disabling captured pipelines remains an
independent control. The optimization report exposes
`resident_ray_query_pipelines`, and the permanent W1 regression exercises the
opaque capture ABI, both candidate kinds, commit/terminate state, and the loop
oracle. A real two-spp image is byte-identical between the two paths.

For context, a fresh 128-spp production-policy sweep gives the following
fallback-relative throughput. W1 has nine paired runs; W2/W4/W8 have seven
completed rotated rounds; W16 has nine paired runs. The W1 result shows that
removing scalar scheduler overhead does not substitute for packet traversal.

| backend/width | median SIMD ms | median fallback ms | SIMD/fallback geomean | wins | 95% CI |
| --- | ---: | ---: | ---: | ---: | ---: |
| W1 resident | 1170.301 | 586.417 | 0.5043x | 0/9 | [0.4967, 0.5121] |
| W2 | 1056.247 | 591.011 | 0.5550x | 0/7 | [0.5432, 0.5670] |
| W4 captured direct | 617.162 | 591.011 | 0.9425x | 0/7 | [0.9164, 0.9695] |
| W8 | 453.357 | 591.011 | **1.2764x** | 7/7 | [1.2313, 1.3231] |
| W16 | 360.214 | 575.357 | **1.6033x** | 9/9 | [1.5676, 1.6398] |

## Reconstructed inline ray queries and handler-cost selection

The SIMD production pipeline now applies a second selective
`lower_ray_query_to_pipeline` after `reconstruct_ray_query_loop`. Consequently,
the canonical affine DSL spelling
`$while (query.proceed()) { if (query.is_surface_candidate()) ... }` can use the
same direct or W1-resident implementation as an original
`traverse(...).on_*().trace()` node. Reconstruction remains fail-closed: a
non-canonical use of the `proceed()` result, shared shell payload, nested
`PROCEED`, escaping temporary, unsupported exit, or loop-carried SSA value is
not converted. No public DSL, Embree, query-state, or handler ABI changed.

Blindly outlining every proven loop is not profitable. The representative
single procedural handler contains 9 XIR instructions without a counter
capture and 12 with it. Its original structured/direct form compared with the
explicit scheduled loop as follows on a Ryzen 9 9950X3D, pinned to physical
cores 0--15 with 16 workers and seven alternating fresh-process pairs:

| width/workload | direct versus loop geomean | wins | paired 95% CI |
| --- | ---: | ---: | ---: |
| W1, procedural 16 candidates | 1.2065x | 7/7 | [1.1791, 1.2346] |
| W4, procedural 16 candidates | 0.9489x | 0/7 | [0.9417, 0.9561] |
| W1, triangle 16 candidates | 1.0584x | 7/7 | [1.0422, 1.0749] |
| W4, triangle 16 candidates | 0.9918x | 2/7 | [0.9764, 1.0074] |

Capture-free width sweeps confirmed that the effect is not caused solely by
capture projection. The procedural direct path was 1.1978x/1.0625x at W1/W2,
but 0.9399x/0.9752x/0.9760x at W4/W8/W16. The triangle direct path was a small
1.0306x/1.0150x at W1/W2 and statistically neutral at the wider widths. The
shared lowering therefore preflights all structurally valid loops and applies
two independent selection predicates:

- the existing capture ABI bound: at most four semantic captures at W1/W4,
  capture-free at W2/W8/W16 unless forced;
- W1/W2 have no handler-cost threshold; W4/W8/W16 require at least 24 handler
  XIR instructions or at least two capture-eligible query loops in the same
  function.

The second condition preserves the real 108-instruction procedural handler and
the W8 cutout function's two 8-instruction query sites, but retains a lone
9--12-instruction micro-handler as a loop. The diagnostic
`LUISA_SIMD_DISABLE_RAY_QUERY_PIPELINE_PROFITABILITY=1` bypasses only this cost
gate, providing a same-binary direct oracle. Comparing the final default policy
against that forced direct route gives:

| explicit procedural case | policy median | forced-direct median | policy speedup | wins | paired 95% CI |
| --- | ---: | ---: | ---: | ---: | ---: |
| W4 counted | 108.724 ms | 115.703 ms | 1.0707x | 7/7 | [1.0473, 1.0947] |
| W4 capture-free | 107.774 ms | 113.891 ms | 1.0551x | 7/7 | [1.0443, 1.0660] |
| W8 capture-free | 85.455 ms | 91.668 ms | 1.0358x | 7/7 | [1.0201, 1.0518] |
| W16 capture-free | 73.606 ms | 75.319 ms | 1.0263x | 7/7 | [1.0194, 1.0333] |

The profitable reconstructed W1 counted path was repeated for nine alternating
pairs. Resident execution took a 159.555 ms median versus 191.185 ms for the
global loop oracle: **1.1922x**, 9/9 wins, with paired 95% CI
[1.1816, 1.2029]. Five additional `perf stat` pairs quantify the mechanism;
ratios are resident divided by loop:

| counter | resident/loop |
| --- | ---: |
| cycles | 0.8358 |
| retired instructions | 0.8351 |
| branches | 0.8311 |
| branch misses | 0.7020 |
| aggregate task clock | 0.8343 |

The W1 direct object contains 113 static instructions, 14 branches, 11 stack
references, and a 1,256-byte frame; the loop oracle contains 110 instructions,
17 branches, 19 stack references, and a 1,232-byte frame. Neither object has an
undefined scalar-math symbol. The direct profile attributes 9.64% to the
resident pipeline, while the loop profile attributes 18.77% to
`_ray_query_proceed`; the dynamic reduction comes from avoiding scheduler and
proceed turnover, not merely from static code size.

The final selection preserves both established graphics paths exactly.
Default and profitability-bypassed optimized assembly hashes match for all
three W4 procedural renderer modules and all five W8 cutout modules. Their
two-spp PNGs are byte-identical as well. The optimization report adds
`post_reconstruction_ray_query_pipelines`; at this historical checkpoint W1
explicit counted reported one resident second-pass pipeline, W2 capture-free
reported one packet pipeline, one small W8 explicit query reported zero, two
small W8 queries reported two, the real W4 procedural renderer reported one,
and the real W8 cutout renderer reported two.

## Frontend-preserved inline ray queries

The frontend refinement is now implemented without adding a semantic AST
statement or a backend-specific DSL feature. `LoopStmt` carries an optional
original `$while` condition, while the body retains the same explicit guard.
AST-to-XIR preflights every marked query loop in a function before preserving
any of them. It accepts only the exact same-query `PROCEED` guard, immediate
surface/procedural split, and handler-local structured control; shared shell
payload, outer exits, nested query/proceed, or one malformed marked candidate
leaves the complete function on the legacy route. Ordinary enclosing loops are
not confused with their nested canonical queries. Function duplication keeps
the hint, while callable serialization may deliberately omit it because the
explicit guard remains a complete reconstruction oracle.

`benchmark_ast2xir_inline_ray_query` records a four-query DSL kernel. On the
Ryzen 9 9950X3D host, five independent runs used 500 translations per batch,
11 alternating rounds, and all 32 logical CPUs available. This final-state
repeat immediately followed both complete builds; the post-run load average
was 12.80/14.30/9.73. The table reports the median of the five per-run medians;
the speedup column is the median paired speedup:

| compile slice | direct median | legacy median | direct speedup |
| --- | ---: | ---: | ---: |
| AST-to-XIR translation | 47.080 us | 53.296 us | **1.1320x** |
| translation + reconstruction | 47.486 us | 59.978 us | **1.2638x** |
| translation + two verifications + reconstruction | 125.161 us | 154.760 us | **1.2363x** |

Before normalization, direct preservation produces 29 blocks and 373
instructions versus 45 blocks and 417 instructions for the legacy route: 16
fewer blocks (35.6%) and 44 fewer instructions (10.6%). Both normalize to the
same 29-block/373-instruction XIR, and the permanent regression requires their
complete normalized XIR text, including loop/dispatch comment ownership, to be
identical. A W2 compiler oracle additionally disables frontend preservation,
runs the legacy reconstruction and the full SIMD pipeline, and requires the
captured final assembly to equal the direct route exactly. Consequently this
stage claims compile-time and diagnostic simplification only, not a runtime
speedup. The explicit Embree world-ray test passes all 48 assertions on both
SIMD and fallback after the change.

The analyzer is skipped by the function's existing direct-CallOp bitset when
`RAY_QUERY_PROCEED` is absent. A second four-ordinary-loop kernel (33 blocks,
304 instructions) measured 36.687 us with preservation enabled versus 36.684
us disabled, a paired 0.9987x ratio across the same five runs. This is within
measurement noise and prevents the query proof from becoming a general DSL
compile-time tax.

Live canonical DSL queries are now eligible in the first pipeline-selection
pass and report zero `post_reconstruction_ray_query_pipelines`. That counter is
retained for serialized/provenance-free ASTs, manual XIR, and the diagnostic
`LUISA_SIMD_DISABLE_FRONTEND_RAY_QUERY_PRESERVATION=1` oracle. Packet width,
the 24-instruction threshold, Embree entry points, and resident versus packet
versus scheduler selection remain private to the SIMD compiler.

Final validation passes the complete configured suite 162/162 in both the
default Release build and the TBB/system-parallel-for Release build. The
explicit world-ray query passes 48 assertions on fallback and independently at
W1/W2/W4/W8/W16; the five-lane wide runs include inactive tails. Changed C++
and header files pass clang-format's dry-run check, and the complete diff passes
Git whitespace validation.

## Rejected cloned/prepass narrow-packet status fusion

A fresh W4 procedural rejection-chain profile attributed 20.27% of sampled
cycles to `_ray_query_proceed` and another 7.57% to the generic status wrapper
that packs `terminated` and candidate-kind bits after the provider returns.
This suggested extending the provider-native W16 publication scheme to
W2/W4. Two implementations were evaluated against an environment-selected
same-binary wrapper oracle while pinned to physical CPUs 0--15 with sixteen
workers.

The first implementation fused the complete narrow provider and status
publication. Seven alternating process pairs on the 16-candidate synthetic
procedural chain measured 1.0508x at W2 and 1.0672x at W4, both with 7/7 wins.
Five W4 counter pairs reduced cycles to 0.9363x, instructions to 0.9233x, and
branches to 0.9528x of the oracle; branch misses were 1.0075x. The real
1280x720, 1024-spp mixed procedural renderer instead measured 0.9679x across
five alternating W4 pairs with only 1/5 wins. All ten images were byte-identical
with SHA-256
`d95cbe53b1cf7c573953986e2f64516494bfa7870536dbd2e37f98b2feb49036`.

A second implementation fused only lanes whose initialized candidate metadata
proved that `advance_ray_query_candidate` could publish or terminate without a
new scan; every other lane delegated to the original provider and packer. The
synthetic result fell to 1.0265x at W2 and a neutral 1.0021x at W4. More
importantly, the real renderer remained negative: 0.9859x across three W2 pairs
and 0.9507x across five W4 pairs, with 1 win at each width. The eligibility
pass itself is overhead when most real calls need an initial or continuation
Embree scan.

Both variants were therefore rejected. At that checkpoint the source retained
the original narrow provider and generic status wrapper; after rebuilding, the
SIMD backend shared object was byte-identical to the pre-experiment binary (SHA-256
`3cd40152e3ed0f041715895f95769fb46762cd4d8402cee441780e3f97096087`).
The result narrows future work: a useful register/SoA query-state design must
avoid both the post-provider state pass and a speculative pre-provider
eligibility pass, while keeping the scan-heavy path compact.

## Accepted shared-core W2/W4 status fusion

A third implementation meets that constraint without changing the public
query-state ABI. Plain and status-aware W2/W4 generic entries call one
`LUISA_NEVER_INLINE` provider core with a publication flag. The status entry
packs terminal/surface/procedural bits during the existing candidate advance
pass and immediately after each grouped scan. The plain entry suppresses those
writes. This removes the generic wrapper's second AoS state scan without
cloning the scan-heavy body. Triangle-only acceleration structures keep their
dedicated provider, W8/W16 keep their prior policies, and a measured W1
rejection keeps W1 on its old scalar wrapper. The same-binary control is
`LUISA_SIMD_DISABLE_NARROW_SHARED_STATUS=1`.

The final object has one 10,938-byte shared core plus 7-byte plain and 10-byte
status wrappers. Its `.text` is 3,412,176 bytes, only 448 bytes above the exact
3,411,728-byte pre-experiment object; there is no second roughly 11 KiB
provider. The final backend SHA-256 is
`4ffe67b1475fc24ef6489655e246716f4499d83f3464554bda645579dda0f03d`.

Seven process pairs of the 16-candidate procedural rejection chain compare the
new status entry with its same-binary wrapper oracle. W2 wins 7/7 at 1.0909x
[1.0811, 1.1008]; W4 wins 7/7 at 1.0456x [1.0357, 1.0556]. Because that oracle
also enters the new shared core with publication disabled, these numbers
isolate fused publication but are not substituted for the true old binary.

The real 1280x720, 1024-spp mixed procedural renderer was therefore run from
two isolated runtime directories containing the exact old and candidate
backend objects, pinned to physical CPUs 0--15 with sixteen workers and
alternating order. W4 wins all fifteen primary pairs at 1.0833x [1.0705,
1.0962]; every image has SHA-256
`d95cbe53b1cf7c573953986e2f64516494bfa7870536dbd2e37f98b2feb49036`.
After excluding W1 from the final policy and rebuilding, five additional W4
pairs confirm 5/5 wins at 1.0936x [1.0682, 1.1196]. Three final W4 counter pairs
measure 1.0830x wall-time speedup and the following candidate/old ratios:

| counter | candidate / old |
|---|---:|
| cycles | 0.9150x |
| instructions | 0.9900x |
| branches | 0.9901x |
| branch misses | 0.9597x |

W2 is more exposed to unrelated host work because one render lasts roughly
eight seconds. Fourteen final alternating pairs have a positive 1.0272x point
estimate, 11/14 wins, and a 1.0366x median, but their ordinary t interval
[0.9880, 1.0680] remains inconclusive after large scheduling outliers on both
sides. Three paired `perf stat` runs are internally consistent: wall time wins
3/3 at 1.0255x and cycles fall to 0.9667x, while instructions and branches are
1.0025x/1.0036x because the shared runtime flag costs more than the removed
two-lane pack loop. The stable 1.0909x synthetic result plus the 3.3% real
cycle reduction retain W2, without claiming a converged whole-process
wall-time interval.

Two negative/control gates bound the policy. A forced W1 loop/status chain
regresses 14/14 at 0.9450x [0.9346, 0.9556], so W1 does not select the shared
entry. The W4 triangle-only chain is neutral across fifteen pairs at 1.0003x
[0.9884, 1.0123], proving that the dedicated provider remains selected. A
non-query 128-spp W4 path trace is likewise neutral across ten pairs at 1.0023x
[0.9971, 1.0074], with identical image hashes. The procedural semantic gate
runs production W1/W2/W4/W8, wrapper-oracle W2/W4, and production/oracle W16:
all 1,576 assertions pass, including inactive tails, divergent commit/reject/
terminate, continuation scans, and surface/procedural provider rebuilds.

## Accepted shared-core W8 provider-native status publication

The remaining W8 generic procedural path still invoked its wide provider and
then scanned the complete 1216-byte state record for every active lane to pack
terminal/surface/procedural status. Plain and status-aware W8 entries now
tail-call one `LUISA_NEVER_INLINE` core. Cached candidates publish during its
existing advance pass, while lanes requiring Embree traversal publish during
the existing W8 batch installation pass. Triangle-only and non-procedural W8
scenes retain their prior providers. The same-binary oracle is
`LUISA_SIMD_DISABLE_W8_WIDE_SHARED_STATUS=1`.

The final object contains one 12,368-byte wide shared core plus 7-byte plain
and 10-byte status wrappers. Its `.text` is 3,414,672 bytes, 2,496 bytes above
the preceding 3,412,176-byte object; there is no second roughly 12 KiB
provider. Disassembly contains `rtcIntersect8`/`rtcOccluded8` and no W8
per-lane `rtcIntersect1`/`rtcOccluded1` fallback. The backend SHA-256 at this
checkpoint is
`72baaa661f4ce29d81c2b1fd7a29815836eb9e78e9db7adfe4aa71ba78b78676`.

Seven alternating, physical-CPU-pinned process pairs of the 16-candidate
procedural rejection chain measured 1.0443x [1.0108, 1.0790] with 6/7 wins.
Three paired counter runs produced the following candidate/oracle geometric
mean ratios:

| counter | candidate / oracle |
|---|---:|
| cycles | 0.9461x |
| instructions | 0.9446x |
| branches | 0.9914x |
| branch misses | 0.9970x |
| cache references | 1.0125x |
| cache misses | 0.9915x |

The real 1280x720 procedural renderer wins all seven 128-spp pairs at 1.0774x
[1.0454, 1.1104]. Seven longer 1024-spp pairs also win 7/7 at 1.0574x
[1.0389, 1.0763], with candidate/oracle medians of 3,418.6/3,585.3 ms. Every
candidate and oracle image passes the checked-in gallery reference and the two
W8 paths produce the same image (SHA-256
`801b703092be960c4c3034a505fb9bfec8305d8e3d51a7a634b84704d324a92a` at
128 spp).

Seven independently alternated 1024-spp W8/fallback pairs measured W8 at
1.3285x fallback [1.2974, 1.3603], with medians of 3,428.5 and 4,486.8 ms.
Both backends pass the gallery reference; their different PSNR values are the
pre-existing backend floating-point distinction rather than a candidate/oracle
semantic difference. The permanent procedural gate now runs production and
oracle W8 in addition to all other widths and reports 1,773 assertions,
including inactive tails, commit/reject/terminate, continuation scans, motion,
and mixed surface/procedural rebuilds.

Both maintained Release trees complete after this change: the default runtime
and system/TBB configurations each pass 162/162 CTest cases. The focused
precise/fast native-math, runtime-width, and procedural gates pass separately;
changed C++ is clang-format-clean and the complete diff passes Git whitespace
validation.

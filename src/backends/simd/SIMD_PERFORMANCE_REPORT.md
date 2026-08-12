# SIMD CPU backend performance report

Snapshot date: 2026-08-12. This report covers the Release build after merging
`origin/next@4546cd535ff620f78ae80a1dbe573be8b99ba39d` into
`codex/simd-cpu-backend` and adding coherent direct-CFG lowering.

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
separate conformance gate. The accepted runtime-sparse study below supersedes
the W8/W16 cutout cells with ten-pair medians; other cells retain the
seven-process sweep.

Speedup is always `fallback time / SIMD time`, or
`SIMD throughput / fallback throughput`, so values above one are wins.

## Current fallback-relative results

| Workload and metric | fallback | W1 | W2 | W4 | W8 | W16 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| non-coro SDF, samples/s | 8.705 | 8.197 (0.942x) | 9.476 (1.089x) | 15.112 (1.736x) | 22.568 (2.593x) | 32.959 (3.786x) |
| image pipeline, ms/iteration | 8.379 | 17.184 (0.488x) | 9.169 (0.914x) | 6.493 (1.290x) | 4.992 (1.678x) | 4.249 (1.972x) |
| voxel render, ms/iteration | 6.904 | 8.127 (0.850x) | 24.122 (0.286x) | 16.128 (0.428x) | 9.386 (0.736x) | 6.479 (1.066x) |
| Spacex, ms/frame | 158.831 | 150.954 (1.052x) | 94.904 (1.674x) | 64.277 (2.471x) | 49.999 (3.177x) | 42.700 (3.720x) |
| cutout path tracing, spp/s | 68.570 | 44.870 (0.654x) | 28.400 (0.414x) | 32.982 (0.481x) | 35.816 (0.522x) | 34.525 (0.504x) |
| portable GEMM, GFLOP/s | 64.895 | 23.332 (0.360x) | 25.627 (0.395x) | 115.914 (1.786x) | 190.521 (2.936x) | 316.449 (4.876x) |

The GEMM row is a compute diagnostic rather than a graphics result. It uses
eight explicit SIMD workers and seven independent process medians; every
process performs seven timed samples of 128 complete 256-by-256 dispatches and
validates the output against double-precision accumulation. The fallback
process medians ranged from 41.594 to 88.326 GFLOP/s under shared-host load,
while the SIMD distributions were tight. Its relative speedups must therefore
be treated as host observations, not cross-machine constants.

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

The W8 cutout main kernel contains two sequential ray-query construction sites.
A fail-closed Schedule-IR liveness/interference analysis now colors them into
one per-lane scratch slot; overlapping query objects remain in distinct slots.
This changes the exact optimized assembly as follows:

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

## Validation

The required native-math/fallback-math/runtime-width gate passes 3/3. The
combined SIMD, XIR, runtime, and graphics label gate passes 86/86. After a full
default build, the complete configured repository CTest suite passes 138/138,
including the coroutine-frame tests merged from `next`.

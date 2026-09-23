Software-LBVH performance benchmark

A stress/performance benchmark for the two-level software LBVH in
examples/compute/lbvh.  It builds and traces a catalogue of adversarial scenes
(worst cases of the LBVH algorithm itself, not just "a big model"), reports a
per-stage breakdown of the build and the counters of the traversal, and keeps the
device inside a memory and a per-dispatch time budget so that it cannot break the
hardware or the driver.

It is a measurement tool: only release-mode numbers are meaningful (the debug
build is for correctness - it enables the runtime's own bounds checks and
assertions, which is how the out-of-bounds slice bug below was found).

    xmake build example_software_lbvh_bench
    bin\release\example_software_lbvh_bench.exe <cuda|dx|vk> [options]

--help documents every option.  The important ones:

| option | meaning |
|---|---|
| --scene <name\|all\|worst> | which scene(s) to run (--list prints the catalogue) |
| --triangles/--instances/--rays/--seed | override the scene sizes (the same scene on every backend) |
| --iters/--warmup | timed / untimed iterations; the headline is the minimum |
| --stress-build / --stress-traversal | log2 sweeps with a fitted scaling exponent per stage |
| --budget-gib (5.0) | device-memory budget, checked before anything is allocated |
| --dispatch-budget-ms (1000) | no single submission may exceed this (see below) |
| --force-oversize | override the pre-flight refusal of a size that would exceed it |
| --validate | structural self-check of every tree + the demo's RTX cross-check |
| --repeat-check | the traversal result must be bit-identical across iterations |

Every measurement also prints machine-readable bench_* records (scene, build,
tree, trace, dispatch, memory, skip, oversize) so results from several
backends can be diffed or plotted.

Safety model

Two hard rules, because a benchmark that removes the device is worse than no
benchmark:

 * Memory. The exact byte count of everything one measurement allocates is
   computed before the first allocation and compared against --budget-gib
   (default 5.0 GiB; the machine has 8 GB of VRAM of which < 6 GB are usable).  A
   scene that does not fit is skipped with a warning, never attempted.
 * One dispatch at a time. Windows resets the device (TDR) when a single
   submission runs for a few seconds - --max-seconds cannot help, because a
   dispatch cannot be aborted from the host.  So the benchmark never submits a
   dispatch it predicts will take longer than --dispatch-budget-ms:
     * the traversal is planned from measurements: a small slice is grown while
       its measured time allows, every slice is strided (a contiguous prefix of a
       camera frustum is systematically cheap and would under-estimate the cost),
       and the whole ray set is traced in as few slices as the budget permits;
     * the build is guarded by a pre-flight: a scene is built once at its probe size
       and the requested size is refused (with the predicted time and the
       predicted-safe size) when the extrapolation exceeds the budget; the sweeps stop
       before the first unsafe step;
     * each run reports the worst submission it actually made - for the build that
       is the maximum of the whole recorded chain (one submission) and of the
       largest single stage of the staged rebuild used for the breakdown - so the
       claim is checkable.

The per-dispatch maxima of a default run on the three backends are <= 330 ms
against the 1000 ms budget.

The scene catalogue

Each scene targets a specific worst case of the algorithm and carries the ray
distribution that reaches it (a degenerate tree is only expensive if the rays
actually walk it).  A BLAS has one tree and one radix sort, so the default shape is
one BLAS with N triangles plus a one-instance TLAS.

| scene | stress | worst case it triggers |
|---|---|---|
| uniform | both | baseline: random triangles in a cube, no worst case (every ratio is taken against it) |
| coincident | build | all N Morton codes bit-identical -> a single non-empty sort bin (shared-atomic contention, no scatter parallelism) and the equal-code path of delta() (32 + clz(i ^ j)) |
| grid-duplicates | build | 1024 clustered cells with exactly coincident centroids inside each: partial code duplication and a clustered histogram together |
| exponential | build | "Morton caterpillar": the spacing decays like 2^-i, so every step loses one more common-prefix bit - the most unbalanced radix tree 30-bit codes allow, and the longest determine_range searches and AABB reductions |
| line | build | a long thin chain of overlapping AABBs on the diagonal |
| sliver-soup | traversal | O(N) node visits and triangle tests per ray: long thin slivers fanning through a small ball, so AABB culling is impossible |
| bimodal | both | "teapot in a stadium": 90 % of the primitives in one blob plus 10 % over a huge box -> zero culling near the blob (measured culling ratio exactly 0.5) and a large-scale disparity |
| instance-chain | both | 256 chained meshes -> 256 radix sorts and a chained (caterpillar) TLAS, i.e. the multi-BLAS/multi-instance path |
| --mesh <file.obj> | both | a real asset (a tiny built-in OBJ reader: v/f, fan triangulation, negative indices); nothing is bundled |

Rays are generated in a kernel with a fixed PCG hash, so the ray *set* is
deterministic per (scene, sizes, seed).  It is not bit-identical across backends:
the transcendentals of the frustum setup contract differently, and the three
backends disagree about ~3 of 262144 rays (the hit/miss totals differ by up to 3
counts out of ~94000 hits).  Every check is per-backend, so this only matters when
quoting a hit count as a backend-independent number.

What is measured

 * Build: total ms and Mprim/s, plus the per-stage breakdown (primitive AABBs,
   Morton codes, the four radix-sort passes, the radix-tree construction) through the
   opt-in LbvhBuildTimings hook; the number of nodes.  The headline total is the
   *plain* recorded build (no inter-stage fences); the breakdown comes from a second
   rebuild whose stages are separated by a synchronisation, so for a scene with K
   trees the stage sum carries K x 4 extra fences and is *larger* than the
   headline - by ~2x for `instance-chain` (257 trees).  Use the headline for
   time and the stages for where.
 * Tree shape: max_depth, mean_depth, depth / log2(n) (the unbalance
   measure), the leaf range of every internal node (max_range, mean_range - the
   cost driver of the AABB reduction), and the number of descent failures instead of
   a hang on a malformed tree.
 * Traversal: ms, Mray/s, ns/ray, hit/miss counts, plus the instrumented
   walk's counters (nodes, slab tests, tests that passed, triangle tests, deepest
   stack, culling ratio) - these say why a scene is slow, not just that it is.
   The counters are diagnostics, not a cost model: they are collected by a copy of
   the walk that is driven by the same slice plan, and a scene whose cost is one
   long dependent chain (`line`: 6.05 nodes per ray on average, 68 ms because one
   ray walks 148656 nodes at memory latency) cannot be explained by an average.
 * Ranking: both scenes and the counters are ranked against uniform, per
   primitive for the build and per ray for the traversal.

Findings (RTX 4060, release, --iters 3, min of 3; cuda / dx / vk)

Round 1 - the build was the radix-tree construction

The --stress-build sweep showed the total build scaling at p ~ 1.02-1.05 with a
visible cache cliff between 256 K and 512 K primitives, and the per-stage timing
attributed 87-97 % of the build to the node stage (the Karras radix-tree
construction).  Inside it, a diagnostic run that removed the AABB reduction showed
the two searches cost ~1 % of it: the cost was the reduction, which iterated
sum(leaf depth) ~ n * mean_depth times (~20 M iterations for 1 M primitives), each
doing a random 8 B keys read plus a random 48 B prims[slot] read.  It was also
grossly imbalanced: one thread per internal node means the root reduces the whole
array alone.  Two changes fixed it: the radix tree is built in two passes (leaves
first - the only place that chases a random prims read - then internal nodes, whose
reduction streams the contiguous leaf array), and the internal-node pass uses one
warp per node plus `warp_active_min/max`.

Round 2 (this change) - the sort, and the cost of a node

(a) The radix sort became the dominant stage (42-62 % of the build: 13.45 ms of
21.47 ms for 2^20 primitives on cuda).  It was a *single work-group* 4 x 8-bit LSD
sort with a shared-memory bit-flag scatter machine per 256-element tile, i.e. one
block of 256 threads doing all the work of the device with ~3 barriers per tile.
`lbvh_sort.h` now owns the stage: `Method::single_block` is that implementation
byte for byte, `Method::multi_block` cuts the range into independent tiles of
`block_size * items` elements and gives each one the same algorithm, which needs
four dispatches per pass (per-block histogram -> hierarchical (block,digit)
exclusive scan -> per-digit base -> ranked stable scatter) and one global 8 B key
read plus 8 B read/write per element per pass.  `Method::automatic` picks
single_block below 8192 elements (measured crossover on all three backends: at
4096 the parallel path is 0.97x on cuda / 0.91x on dx / 0.77x on vk; at 8192 it is
already 1.56x / 1.35x / 1.12x) and multi_block above it.  The stage is **16-20x
faster** at scale (cuda 2^20: 11.88 -> 0.78 ms; 2^23: 133.7 -> 7.1 ms; dx 2^20:
12.15 -> 0.79; vk 2^20: 13.27 -> 1.08), which removes it from the picture - and
the small-tree path stays the old single-work-group sort, so the 256-tree scene
keeps the dispatch count it had.

(b) The internal-node reduction was memory-bound and its *access pattern* was the
problem: each lane owned a contiguous slice of the node's leaf range, so the 32
lanes of one warp instruction sat `length / lane_count` records apart (1.5 MiB for
the root of a 1 M-primitive tree) - 32 unrelated cache lines per instruction.  The
lanes now walk the range together (`range.x + lane, + lane_count, ...`), which
makes each instruction read 32 consecutive records.  Node stage on cuda:
7.24 -> 5.33 ms (uniform), 9.71 -> 6.57 (exponential).  Two records per lane per
iteration (the obvious way to add memory-level parallelism) was tried and is
*slower* on all three backends (4.32 -> 5.14 ms), so it is not in the code.

(c) A node was 48 bytes (`float3 lo; float3 hi; uint left; uint right; uint prim;`
- `float3` is 16-byte aligned, so the handles cost a full 16-byte lane), i.e. *two*
32-byte L2 sectors per random node load, 64 bytes of traffic for 32 useful ones.
It is now 32 bytes exactly - one sector, two 16-byte vector loads - with the child
handles bit-cast into the fourth lane of each AABB plane (`lo.w` = left,
`hi.w` = right for an internal node; `lo.w` = invalid, `hi.w` = primitive id for a
leaf).  `lbvh_common.h` owns the packing behind `aabb_lo/hi`, `child_left/right`,
`is_leaf` accessors, so the build passes, both traversals, the benchmark's mirror
and the structural self-check all follow one definition; the node array is also 33 %
smaller (which the memory budget notices).  This is what the traversal had been
waiting for: the walk is dominated by random node loads, and all of the
visit-heavy scenes got 10-25 % faster while the arithmetic is unchanged.

| scene (cuda) | build before -> after | sort | node | trace before -> after |
|---|---|---|---|---|
| uniform (1 M) | 21.5 -> 6.0 ms (3.6x) | 13.45 -> 0.80 | 7.25 -> 4.33 | 3.53 -> 3.06 ms |
| grid-duplicates (1 M) | 17.6 -> 4.9 ms (3.6x) | 8.94 -> 0.30 | 7.11 -> 3.75 | 2.02 -> 1.94 |
| exponential (1 M) | 19.4 -> 6.1 ms (3.2x) | 9.00 -> 0.32 | 9.71 -> 4.91 | 39.9 -> 36.1 |
| line (1 M) | 16.2 -> 4.6 ms (3.6x) | 8.98 -> 0.29 | 6.37 -> 3.40 | 75.9 -> 68.5 |
| coincident (262 K) | 3.40 -> 1.07 ms | 2.20 -> 0.14 | 1.09 -> 0.77 | 234.9 -> 209.9 |
| bimodal (262 K) | 3.44 -> 1.13 ms | 2.29 -> 0.16 | 1.06 -> 0.78 | 198.5 -> 156.6 |
| sliver-soup (16 K) | 0.29 -> 0.17 ms | 0.22 -> 0.09 | 0.12 -> 0.10 | 158.7 -> 123.1 |
| instance-chain (256 trees) | 21.1 -> 19.2 ms | 19.35 -> 18.54 | 11.90 -> 10.77 | 2.40 -> 2.26 |

dx follows (uniform 22.3 -> 7.4 ms, sort 12.3 -> 0.86, node 9.2 -> 5.6; bimodal
199 -> 167; sliver-soup 180 -> 133) and vk follows (uniform 24.3 -> 7.1, node
8.6 -> 5.4; bimodal 212 -> 161; sliver-soup 167 -> 128).  The fitted scaling
exponent of the whole build dropped from 1.02-1.05 to **0.70** (node stage 1.03-1.06
-> 0.72, r2 0.94), the largest size the sweep now completes is its own 4 M cap
(cuda: total 22.4 ms at 4 M, node 15.7), and no single submission of the whole
catalogue comes near the 1000 ms budget (worst traversal slice 326 ms on vk
`coincident`, worst build 63 ms on dx `instance-chain`).

`instance-chain` is the remaining independent story: 257 trees of 4096 primitives
mean ~1300 kernel *submissions* for one build (5 per tree: primitive AABBs, Morton,
4 sort passes as one batch - the single-block path - leaves, internal nodes), and
the measured build of 19 ms (cuda) is dominated by the launch path rather than by
the arithmetic (the same work spread over one tree is ~0.07 ms).  Batching the
trees into one dispatch per stage, and fusing the Morton kernel into the primitive
AABB kernel (removing one submission and one 8 B/element round trip per tree), are
the measured next steps; both change the build interface from one tree per call to
all trees per call.

What the traversal does not do (measured, twice)

The counters show the walk pushing both children and testing them when they are
popped: 141.7 pops and 141.7 slab tests per ray for uniform, of which only 71.3
pass.  Two cheaper-looking walks were implemented and measured:

 * *test a child before pushing it* (so a child the ray cannot enter is never
   pushed).  It halves the pops, and it wins on the scenes that cull (-39 % uniform,
   -61 % line) - but learning that a child is not entered costs a node load, and a
   zero-culling tree pays it for almost every child: coincident +25 %,
   grid-duplicates +8/+13 %.
 * *push only one child and descend into the other directly* (which halves the
   stack traffic to one store and no load per internal node, keeps the visit order
   and the load count identical, and halves the stack depth).  Measured: -10 %
   uniform, -4 % line, -5 % exponential, but +10 % coincident, +15 % bimodal,
   +9 % sliver-soup, +23 % grid-duplicates.

Both were reverted: the benchmark exists to protect the zero-culling worst cases,
and the blind-push walk has no regression on any of them.  What remains for the
traversal is the *number* of visits, which for the worst scenes is the whole tree by
construction, and the triangle test (13946 tests per ray on coincident, 5577 on
sliver-soup, ~15-30 % of those scenes' cost): computing the barycentric acceptance
without the `1/det` division and dividing only on a hit is the cheap candidate left.

Verified behaviour

 * `--validate` self-checks the structure of every tree (reachability,
   parent-vs-children AABBs, the `2n-1` node count) and cross-checks the sampled
   hits against the Luisa RTX reference on the same buffers.  On cuda, dx and vk
   the catalogue reports 0 structural problems and 0 hit/miss, distance and
   instance/primitive mismatches in all 24 scene x backend runs; same-distance ties
   between overlapping primitives (up to 9912 rays on `coincident`, zero on
   `uniform` and `line`) are counted, not hidden, and are not fatal because the
   closest hit is genuinely not unique there.  Rays that are incomparable by
   construction (grazing < ~1.7 degrees to the triangle plane, hits exactly on a
   shared edge, a 1e-12 Moller-Trumbore determinant) are reported separately.
 * `example_software_lbvh` (the demo, one hand-picked scene, exact primitive match
   required) passes unchanged on all three backends, and `example_software_lbvh_test`
   (see below) adds the cases a demo cannot cover.
 * `example_software_lbvh_sort_bench` checks the sort against a host
   `std::stable_sort` on 366 cases (sizes 0..2^21, 8 key distributions, non-zero
   base with garbage around the range, ragged tiles) on all three backends and finds
   0 failures; it is also where the sort's A/B table comes from.

The extra test target

`example_software_lbvh_test <backend>` is the correctness net for changes like this
one.  It builds 78 checks - boundary sizes 1..4097, degenerate geometry (coincident
centroids, zero-area slivers, axis-aligned, 1e-6..1e6 extents, exact axis-aligned
ray directions), 1..17 BLASes with mirrored/scaled/rotated instances, and 32 seeded
random scenes - and compares every traversal against an independent
double-precision brute-force reference *and* against the hardware RTX reference (so
a bug shared by "software LBVH and RTX" cannot hide).  It also verifies slice
invariance (`trace_software`'s strided slices must equal the contiguous trace
element-wise), build determinism (the same scene built twice must produce identical
nodes and hits) and the tree contracts (`validate_tree`, exactly `2n-1` nodes, exact
tiling of the shared buffers).  Its sensitivity was checked with six injected
mutations (dropped hits, perturbed distances, a shifted slice, a wrong validate
range), each of which produced >= 3 failing checks and a non-zero exit code.

It also found three real defects, all fixed here: `validate_tree` used to index the
children of the node it was checking *before* range-checking them (a wrong
`(node_base, count)` crashed the host instead of reporting a problem), the
benchmark's slice planner re-used the ray count of offset 0 for the other sampled
offsets (which walks - and writes - up to one element past the ray/hit buffers
whenever the offset is not a multiple of the stride; the debug build traps it as
`Out of bounds: !(index: 20017 < lc_buffer_size(buffer): 20000)`, release does not),
and the reported worst build submission ignored the plain whole-chain build (it
only tracked the staged rebuild used for the breakdown, under-reporting by up to
~50x on a many-tree scene).  `SoftwareLbvh::trace_software` now also rejects a
strided range that would leave the ray/hit buffers.

Caveats

 * The xmake debug build is *not* an AddressSanitizer build: sanitizers are wired
   into the CMake path only (`LUISA_COMPUTE_ENABLE_SANITIZERS`).  The debug mode
   does enable the runtime's own bounds checks and assertions, which is what caught
   the slice bug above.
 * The RTX cross-check compares only the rays that are comparable; the excluded
   buckets are not negligible on the adversarial scenes (uniform: 2858 of 262144
   rays are near-boundary hits), so "0 mismatches" means "0 mismatches among the
   comparable rays".
 * The per-stage breakdown synchronises between stages.  For a K-tree scene that is
   K x 4 extra fences: use the headline total for time and the stages for shares.
 * The benchmark's own pre-flight is a measured heuristic, not a hardware
   guarantee: `--force-oversize` genuinely risks a driver reset (measured 2697 ms
   `node` submission at 8 M triangles on dx).

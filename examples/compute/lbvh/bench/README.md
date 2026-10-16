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
| --compact (=as-built) | compact the structure after the build (see "Round 4" below); `subtrees` is reserved but not implemented and fails the parse |
| --release-scratch | with `--compact`, also retire the build scratch (opt-in; the storage is traverse-only afterwards) |
| --headroom (2) | size the storage as scene * f, i.e. the loose slack `--compact` reclaims; 1 makes the storage exact and the compaction a no-op |

Every measurement also prints machine-readable bench_ records (scene, build,
tree, trace, dispatch, memory, skip, oversize, and - only when a compaction ran -
compact) so results from several backends can be diffed or plotted.

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

The bindless heap (layout change)

The two-level LBVH used to hand a traversal one shared node buffer and to identify a
BLAS by the absolute node index its region starts at.  Following the fallback RTX
backend (src/backends/common/rtx/fallback_rtx_layout.h), a TLAS now owns a
`BindlessArray` - its *heap* - whose slots hold the node *regions* of the trees
(slot 0 is the null slot, slot 1 the TLAS' own region and slot 2 + i the node region
of BLAS i), and an `LbvhBlas` record names its region by that slot.
`pre_build_accel` registers one `BufferView<LbvhNode>` per tree and records the heap
update; the traversal resolves the top level through slot 1 and each BLAS through the
slot its table record carries, reading node `handle` as `heap[slot][handle -
node_offset]` (a heap entry is a view that starts at the region, so the conversion is
region-local and never negative).  Because a heap entry is a view and not an offset
into one shared descriptor, a TLAS may reference a BLAS that was laid out before it -
the constraint the old ABI could not express.  The instrumented walk mirrors the
change, so its counters still describe the measured walk, and `validate_heap` (used by
the demo, the test and `--validate`) checks on the device that every record resolves
through its slot to the same root node the shared node buffer holds.

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

Round 2 - the sort, and the cost of a node

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

instance-chain is the remaining independent story: 257 trees of 4096 primitives
mean ~1300 kernel submissions for one build (5 per tree: primitive AABBs, Morton,
4 sort passes as one batch - the single-block path - leaves, internal nodes), and
the measured build of 19 ms (cuda) is dominated by the launch path rather than by
the arithmetic (the same work spread over one tree is ~0.07 ms).  Batching the
trees into one dispatch per stage, and fusing the Morton kernel into the primitive
AABB kernel (removing one submission and one 8 B/element round trip per tree), are
the measured next steps; both change the build interface from one tree per call to
all trees per call.

Round 3 (this change) - the searches, the node AABB and the primitive record

The build was revisited with an ablation pass, which is the only way to attribute
it: `LbvhBuildTimings` resolves the four stages (primitive AABBs, Morton, sort,
radix tree) and the radix tree was in turn ablated part by part.  The ablations
are done with the relink of every variant verified (`xmake` decides staleness from
mtimes, and an install that preserves timestamps silently re-runs the *previous*
binary - which is how a first pass at this round produced numbers that did not
reproduce).

For 1 M primitives (cuda) the radix-tree stage used to be 4.5 ms, and ablation
attributed it to the two searches of a node - `determineRange()` and `findSplit()`
- rather than to the AABB reduction, and to the *shape* of the kernel rather than
to the memory traffic:

(a) Both searches are chains of ~3 * log2(range) *dependent* key reads (each step's
address depends on the previous comparison), and the pass ran one *warp* per node,
so all 32 lanes walked the same chain: the same answer was computed 32 times and
the warp had exactly one load in flight.  The searches are now computed by one
*lane* per node (`_plan_kernel`), which puts 32 independent chains in a warp -
32 loads in flight instead of one - and the pass needs no work-group cooperation,
so it is a plain 1D dispatch.  It publishes one `uint4` per node, (first, last,
child_a, child_b), which is also the point where the plan is read back: the
reduction pass gets a node's whole plan in one 16-byte vector load.

(b) The AABB of a node is the union of the leaf AABBs of its range, so the direct
reduction reads every leaf once per ancestor: sum(leaf depth) reads, ~21 M x 32 B
for 1 M primitives.  `_block_kernel` now precomputes the AABB of every block of
`node_reduction_block` (= warp width) aligned leaves - it reads every leaf exactly
once - and a node's range is covered by three disjoint pieces (the leaves up to the
next block boundary, the whole blocks strictly in between, the leaves from the last
boundary).  A node whose range stays inside one block - the overwhelming majority,
the mean range of a 1 M-primitive tree is ~20 leaves - takes the first piece only
and costs exactly what it cost before, and the wide nodes at the top read one block
AABB per 32 leaves instead of every leaf.  Blocks are indexed by the *global node
slot* a leaf starts at, so no per-tree rounding is involved: the whole scene needs
one block per 32 node slots.  min/max are exact and associative, and every element
of the range is still read exactly once, so the resulting tree is **bit-identical**
to the one the direct reduction produced - which is why the determinism, hit and
RTX cross-checks of the test suite pass unchanged.

(c) `LbvhPrim` - the primitive AABB plus the id - was `uint id; float3 lo; float3
hi;`, i.e. 48 bytes, because `float3` is 16-byte aligned: a third more memory, and
the *random* `prims[slot]` read of the leaf pass - the only random access of the
whole build - touched two 32-byte L2 sectors instead of one.  It is now exactly two
`float4` (32 bytes, two vector loads) with the id bit-cast into the spare fourth
lane of the low plane, the same packing `LbvhNode` already used for its handles.
This is where the pass that looked cheapest actually was: writing the record costs
one 16-byte store instead of three scalar stores, and the primitive-AABB pass - the
stage that *fills* it - dropped by ~40 % (see the table).

| scene (cuda, `--iters 5`, min of 5) | build before -> after | prim | node |
|---|---|---|---|
| uniform (1 M, 262 K rays) | 6.54 -> 4.78 ms (1.36x) | 0.84 -> 0.48 | 4.53 -> 2.90 |
| exponential (1 M) | 6.62 -> 3.44 ms (1.92x) | 0.81 -> 0.46 | 5.04 -> 2.65 |
| coincident (262 K) | 1.07 -> 0.69 ms (1.54x) | 0.25 -> 0.14 | 0.75 -> 0.49 |
| bimodal (262 K) | 1.12 -> 0.71 ms (1.57x) | 0.25 -> 0.15 | 0.80 -> 0.50 |
| grid-duplicates (262 K) | 1.08 -> 0.70 ms (1.53x) | 0.26 -> 0.14 | 0.77 -> 0.50 |
| line (1 M) | 4.57 -> 3.26 ms (1.40x) | 0.75 -> 0.46 | 3.42 -> 2.46 |
| sliver-soup (16 K) | 0.17 -> 0.15 ms (1.13x) | 0.06 -> 0.06 | 0.11 -> 0.09 |
| instance-chain (257 trees) | 19.4 -> 20.9 ms | 7.0 -> 8.9 | 10.8 -> 12.9 |

(`prim` is the primitive-AABB pass of the build, `node` the whole radix-tree
construction - leaves, block AABBs, plan, reduction.  Two back-to-back rounds of
the same pair of binaries agree to within a few percent on the `node` and `prim`
columns; the traversal column is deliberately absent, because it is unchanged and
the spread of the *traversal-heavy* scenes between two sessions (up to 20 % on
`coincident`, 200-280 ms for a bit-identical walk) is far larger than anything
these changes did to it.  That spread is also why every conclusion above comes
from an interleaved A/B - the two variants rebuilt and re-run alternately in the
same session - rather than from a before/after pair of files.)

instance-chain is the one scene that does not follow: it is launch-bound and its
four kernels per tree became six (the plan and the block pass), which costs more
than the ~0.6 ms of arithmetic per tree that was saved.  Batching the trees into
one dispatch per stage is still the fix - and now a slightly bigger one.

What is left on the build: the reduction pass is now the largest piece of the
radix tree (1.9 ms of the 2.9 ms node stage for 1 M primitives) and it is limited
by the same sum(leaf depth) leaf traffic the block array could not remove for the
narrow nodes - the small nodes read their own leaves, and their total is what the
block array gives back.  Only a *bottom-up* pass (a node's AABB from its two
children's, two reads per node instead of one per ancestor) removes it, and that
needs an order in which the children of every node are known first; the
Karras layout does not provide one for free (a node's children sit at a lower index
when its range ends at the node and at a higher index when it starts at it).

What the traversal does not do (measured, three times)

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
and the blind-push walk has no regression on any of them.

The third candidate was the triangle test (13946 tests per ray on coincident,
5577 on sliver-soup, ~15-30 % of those scenes' cost): the acceptance test was
decided on the barycentric coordinates after the `1/det` division, and since
multiplying an inequality by `det` only flips it (which a sign trick handles), the
whole test can be decided on the *unnormalized* cross products and the division
(evaluated together with `u` and `v`, which only a hit needs) only happens on a
hit.  It was implemented and measured, and it is **not** in the code: the traversal
moved by less than the run-to-run noise on every scene (uniform -1 %, coincident
+1 %, sliver-soup -3 %) and `bimodal` regressed by a consistent +3-4 % in both
rounds.  The division was apparently not the cost - what the change adds is a
handful of multiplies plus a divergent `$if` on the hit, which is the wrong side of
the trade on a GPU whose reciprocal is already cheap.  The traversal is also
bandwidth-bound rather than arithmetic-bound: for `uniform` it reads 141.7 nodes x
32 B per ray, 1.19 GB in 3.0 ms, i.e. ~390 GB/s - above the card's DRAM bandwidth,
because the tree partially fits the 24 MB L2 - so only *fewer bytes per node* or
*fewer visits* can move it, and both are build-quality/layout questions rather
than traversal-loop questions.

What remains for the traversal is therefore the number of visits, which for the
worst scenes is the whole tree by construction, and the depth a ray reaches: the
deepest stack of the catalogue is 26 entries on `line` (average 1.4-15.5), against
a 64-entry software stack, so the stack is not the lever either.


Verified behaviour

   * `--validate` self-checks the structure of every tree (reachability,
     parent-vs-children AABBs, the `2n-1` node count), checks the bindless heap
     (`validate_heap`: a slot per BLAS, every record's slot resolving to the same
     root node the shared node buffer holds) and cross-checks the sampled
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
  and the tree contracts (`validate_tree`, exactly `2n-1` nodes, exact
  tiling of the shared buffers, `validate_heap`).  Its sensitivity was checked with six injected
mutations (dropped hits, perturbed distances, a shifted slice, a wrong validate
range), each of which produced >= 3 failing checks and a non-zero exit code.

It also found three real defects, all fixed here: validate_tree used to index the
children of the node it was checking before range-checking them (a wrong
(node_base, count) crashed the host instead of reporting a problem), the
benchmark's slice planner re-used the ray count of offset 0 for the other sampled
offsets (which walks - and writes - up to one element past the ray/hit buffers
whenever the offset is not a multiple of the stride; the debug build traps it as
Out of bounds: !(index: 20017 < lc_buffer_size(buffer): 20000), release does not),
and the reported worst build submission ignored the plain whole-chain build (it
only tracked the staged rebuild used for the breakdown, under-reporting by up to
~50x on a many-tree scene).  SoftwareLbvh::trace_software now also rejects a
strided range that would leave the ray/hit buffers.

Round 3 added a fourth, and the *demo* is what found it: the internal-node pass
publishes its plan in a shared buffer, and the two builders rebuild their
`TreeRange` by hand from their resource's accessors, so a field that is not copied
there (the plan offset) is simply left indeterminate.  That failure mode is nasty
precisely because it is *invisible*: the kernel that writes the plan and the kernel
that reads it agree on the same wrong base, so the tree that comes out is correct
whenever the base happens to be in bounds and does not collide with a live plan -
which is why the test suite passed 78/78 while the demo hit
CUDA_ERROR_ILLEGAL_ADDRESS.  `TreeRange` now default-initialises every field and
`build_tree()` checks the range it is handed against the storage' bookkeeping, and
the test checks the *offsets themselves* (each tree's plan base must follow the
previous tree's by that tree's own internal-node count), which is the check that
would have caught it.  That check was verified the same way as the others: with
the plan offsets never set, every boundary scene fails with problems=1.

Round 4 (this change) - storage compaction

The acceleration structure of this example used to be built once, loosely, into a
buffer whose size is the caller's budget for the scene (two nodes per primitive),
and kept that shape for the rest of its life.  This round adds the RTX compaction
flow, which is what `AccelOption::allow_compaction` means on the hardware backends:
after the scene is built, ask the *device* how many nodes the trees actually use,
read that back with a hard synchronise, allocate a node buffer of exactly that size,
copy the built nodes into it with a kernel, re-register every bindless heap view onto
the new buffer (`BindlessArray::update`), and retire the loose buffer plus every
temporary buffer through a *completion callback* of the same command list that
carried the copy.  It is the software translation of
`vkCmdCopyAccelerationStructureKHR` / OptiX `optixAccelCompact` / the DX
`PROPERTY_TYPE_COMPACTED_SIZE` prebuild, and Metal's BLAS/TLAS path is the closest
analogue (write the size, read it back on a callback, hard-sync, copy-and-compact,
release the old handle in a callback).

The flag on `AccelOption` is the caller's intent, never a semantic change (the same
contract the hardware backends state): `SoftwareLbvh::compact()` is the action that
honours it, and it asserts that every BLAS/TLAS was created with the flag.

How the size query is answered without a pass or an atomic

The fused leaf pass of `build_tree()` now writes the tree's node count (`2 * count - 1`)
into its own slot of a per-tree usage buffer.  It is one 4-byte store by lane 0 of a
pass that is already recorded for *every* tree - the `count == 1` case skips the other
three radix-tree passes, but never the leaves - so the query adds no dispatch and no
shared counter to reduce.  DX answers the same question the same way in spirit: its
prebuild carves the 8-byte compacted size out of the build scratch the build already
owns.  `LbvhStorage::compact()` then reads the slots back (its own submission, followed
by `synchronize()` - the host needs the number to size the allocation) and fails closed
if the device count and the host bookkeeping disagree, which is "compaction only on a
full build".  A compacted structure has no spare capacity: reserving *new* trees on
the same storage fails closed (`allocate()` checks the live node buffer), while
re-building an *existing* tree in place stays valid under `as_built` because every
index is unchanged - the test suite checks that, and it is the half of the contract
`subtree_contiguous` (see below) would not keep.

What it costs, what it reclaims (cuda, --iters 3, --validate --repeat-check)

`--headroom f` sizes the storage as scene * f *before* the scene is known - the
storage's own contract - and that headroom is exactly the loose slack the compaction
reclaims.  This is deliberate: an *exact* `estimate()` leaves only `blas_count + 1`
nodes of slack, i.e. 128 B for the demo scene, which is the honest number but not a
useful demonstration.  With the benchmark's default `--headroom 2`:

| scene | nodes loose -> dense | reclaimed | copy ms | loose -> dense ns/ray |
|---|---|---|---|---|
| uniform | 4194308 -> 2097152 | 64.0 MiB | 4.34 | 12.58 -> 11.69 |
| coincident | 1048580 -> 524288 | 16.0 MiB | 2.61 | 3537 -> 3912 |
| grid-duplicates | 4194308 -> 2097152 | 64.0 MiB | 5.11 | 7.53 -> 7.69 |
| exponential | 4194308 -> 2097152 | 64.0 MiB | 4.59 | 150.7 -> 151.0 |
| line | 4194308 -> 2097152 | 64.0 MiB | 4.39 | 291.3 -> 288.1 |
| sliver-soup | 65540 -> 32768 | 1.0 MiB | 2.16 | 2032 -> 2058 |
| bimodal | 1048580 -> 524288 | 16.0 MiB | 2.44 | 20822 -> 20904 |
| instance-chain | 4195328 -> 2097407 | 64.0 MiB | 4.50 | 9.83 -> 9.01 |

`copy ms` is the whole compaction (size readback + synchronise + exact-size allocation
+ the copy submission + the heap update); the allocation is the one-shot part of it
and is what makes the first call of a scene cost more than the copy itself (a 64 MiB
`cuMemAlloc` on this machine, vs ~0.5 ms of pure 32 B/node streaming).

Releasing the build scratch (opt-in, `--release-scratch`)

The node buffer is not the only memory a built scene holds.  The primitive AABBs
(32 B per primitive), the two Morton-key ping-pong buffers (8 B each), the plan
records (16 B per internal node) and the block AABBs (~1 B per node slot) are read
only while a tree is *built*, and together add up to ~66 B per primitive slot - as
large again as the node buffer's 64 B, i.e. roughly twice the node bytes a
`--headroom 2` compaction reclaims.  `--release-scratch` moves them out of the
storage and retires them through the same completion callback as the loose node
buffer (the build has long completed: `compact()` synchronises for its size query, and
nothing the command list records reads the scratch).  For uniform on cuda that is
132.0 MiB of scratch on top of the 64.0 MiB of node bytes.

It is opt-in because it changes the storage contract: after it the storage can only
be traversed and validated - every build stage fails closed on the missing buffers -
so a rebuild needs a new `LbvhStorage`.  That is the same "a compacted structure has
no spare capacity" rule as the node compaction, applied to the build scratch; the
default keeps the current rebuild-valid contract.  The radix sort's own scratch (~1 B
per primitive) is not released.  The demo opts in by default (it compacts once, at the
end, and never rebuilds) and prints both numbers; the test suite releases it once and
then re-validates and re-traces the same scene.

The delivered policy is `as_built`: destination index == source index, so the copy is
a streaming node copy, every handle and every `LbvhBlas::node_offset` stays valid, and
a later rebuild of the same storage stays valid.  That also means the traversal
*cannot* change - the bytes are the same bytes in the same order - and the two ns/ray
columns are session noise, not a result: grid-duplicates appears as 12.4 -> 8.5 in one
session and 7.5 -> 7.7 in the next, with a bit-identical walk (0 hit mismatches in
every scene of both runs), which is the same spread this README warns about for the
traversal everywhere else.  The claim of this round is the bytes, not the ns.

`subtree_contiguous` - designed, rejected here

The second planned policy relabels each tree into DFS preorder (for a Karras tree,
two stable sorts by `(leaf range, subtree size)` give exactly that order), so that a
subtree - the working set of one descended path - is one contiguous range.  That is
the only variant that would add *adjacent access* rather than just shrink the working
set, and it is what the "friendly cache hit" half of the requirement is about.  It is
not implemented, and `--compact=subtrees` fails the parse (and
`LbvhStorage::compact` fails closed on the enum value) instead of silently producing
an unrelabelled structure:

  the relabel needs a second LSD sort that is recorded into the *same* command list
   as the copy - the existing `LbvhRadixSort::sort` submits to a `Stream`, so the
   ordering that makes the copy the last reader of the old buffer would be lost - and
   therefore a `CommandList`-recording overload of the sort, a node-sized scratch
   (keys + the sort's own matrix/scan scratch, ~32 B/node = as large as the node
   buffer), a `remap` pass and a retirement bundle that owns that scratch; and it
   moves every index, so a rebuild after it must either fail closed or re-reserve a
   loose buffer, i.e. it changes the rebuild contract that the delivered policy
   deliberately keeps.
  it is also not measurable *without* that machinery, so this round records it as a
   rejected alternative rather than claiming a win: the index-preserving `as_built`
   policy is the deliverable, and the extension point is the enum value and this
   paragraph.

Verification of the compaction

The demo (`example_software_lbvh --compact`, on by default) prints the nodes/bytes
before and after, re-runs `validate_tree` for every tree and `validate_heap` on the
dense buffer, and traces the same deterministic rays on both structures with an
assert that every hit is bit-identical; `--no-compact` and `--headroom 1` give the
A/B and the no-op case.  The test suite rebuilds every one of its scenes into its own
headroom-sized storage, compacts, and checks the contract (`nodes()`/`node_count()`/
`nodes_after`/`compacted_bytes` agree with the device query, the dense buffer is
exactly `2n-1` per tree), the structure and the heap on the dense buffer, the hit
identity before/after, that the dense node bytes equal the loose ones (`as_built`),
that a second `compact()` reclaims 0 and records no copy, and that several
independent storages compacted back-to-back still validate and trace (the retirement
callbacks must all fire), and that a storage whose build scratch was released is no
longer buildable but still validates and traces.  On cuda, dx and vk: 86 checks in the
full run, 0 failures, 0 compaction problems; the benchmark gate is
`--validate --repeat-check --iters 3 --compact` (and the same with
`--release-scratch`), 8/8 scenes, 0 hit mismatches and `estimated >= allocated` for
every scene (the budget counts the loose *and* the dense node buffer, and the
`--headroom` the storage is sized with).

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

# Software-LBVH performance benchmark

A stress/performance benchmark for the two-level software LBVH in
`examples/compute/lbvh`.  It builds and traces a catalogue of *adversarial* scenes
(worst cases of the LBVH algorithm itself, not just "a big model"), reports a
per-stage breakdown of the build and the counters of the traversal, and keeps the
device inside a memory and a per-dispatch time budget so that it cannot break the
hardware or the driver.

It is a *measurement* tool: only release-mode numbers are meaningful (debug builds
enable AddressSanitizer, and debug runs are for correctness).

```sh
xmake build example_software_lbvh_bench
bin/release/example_software_lbvh_bench.exe --list            # the scene catalogue
bin/release/example_software_lbvh_bench.exe cuda --scene all --iters 3 --validate
bin/release/example_software_lbvh_bench.exe dx   --scene all --iters 3
bin/release/example_software_lbvh_bench.exe cuda --scene uniform --stress-build --stress-traversal
bin/release/example_software_lbvh_bench.exe cuda --mesh path/to/model.obj --instances 8 --validate
# `xmake run example_software_lbvh_bench cuda ...` works too; the backend is argv[1].
```

`--help` documents every option.  The important ones:

| option | meaning |
|---|---|
| `--scene <name\|all\|worst>` | which scene(s) to run (`--list` prints the catalogue) |
| `--triangles/--instances/--rays/--seed` | override the scene sizes (the same scene on every backend) |
| `--iters/--warmup` | timed / untimed iterations; the headline is the **minimum** |
| `--stress-build` / `--stress-traversal` | log2 sweeps with a fitted scaling exponent per stage |
| `--budget-gib` (5.0) | device-memory budget, checked *before* anything is allocated |
| `--dispatch-budget-ms` (1000) | no single submission may exceed this (see below) |
| `--force-oversize` | override the pre-flight refusal of a size that would exceed it |
| `--validate` | structural self-check of every tree + the demo's RTX cross-check |
| `--repeat-check` | the traversal result must be bit-identical across iterations |

Every measurement also prints machine-readable `bench_*` records (`scene`, `build`,
`tree`, `trace`, `dispatch`, `memory`, `skip`, `oversize`) so results from several
backends can be diffed or plotted.

## Safety model

Two hard rules, because a benchmark that removes the device is worse than no
benchmark:

* **Memory.** The exact byte count of everything one measurement allocates is
  computed before the first allocation and compared against `--budget-gib`
  (default 5.0 GiB; the machine has 8 GB of VRAM of which < 6 GB are usable).  A
  scene that does not fit is *skipped* with a warning, never attempted.
* **One dispatch at a time.** Windows resets the device (TDR) when a single
  submission runs for a few seconds — `--max-seconds` cannot help, because a
  dispatch cannot be aborted from the host.  So the benchmark never submits a
  dispatch it predicts will take longer than `--dispatch-budget-ms`:
  * the traversal is planned from **measurements**: a small slice is grown while
    its measured time allows, every slice is **strided** (a contiguous prefix of a
    camera frustum is systematically cheap and would under-estimate the cost), and
    the whole ray set is traced in as few slices as the budget permits;
  * the build is guarded by a pre-flight: a scene is built once at its probe size
    and the requested size is refused (with the predicted time and the
    predicted-safe size) when the extrapolation exceeds the budget; the sweeps stop
    before the first unsafe step;
  * each run reports the worst submission it actually made, so the claim is
    checkable.

The per-dispatch maxima of a default run on the three backends are ≤ 350 ms
against the 1000 ms budget.

## The scene catalogue

Each scene targets a specific worst case of the algorithm and carries the ray
distribution that *reaches* it (a degenerate tree is only expensive if the rays
actually walk it).  A BLAS has one tree and one radix sort, so the default shape is
one BLAS with N triangles plus a one-instance TLAS.

| scene | stress | worst case it triggers |
|---|---|---|
| `uniform` | both | baseline: random triangles in a cube, no worst case (every ratio is taken against it) |
| `coincident` | build | **all** N Morton codes bit-identical → a single non-empty sort bin (shared-atomic contention, no scatter parallelism) and the equal-code path of `delta()` (`32 + clz(i ^ j)`) |
| `grid-duplicates` | build | 1024 clustered cells with exactly coincident centroids inside each: partial code duplication *and* a clustered histogram together |
| `exponential` | build | "Morton caterpillar": the spacing decays like `2^-i`, so every step loses one more common-prefix bit — the most unbalanced radix tree 30-bit codes allow, and the longest `determine_range` searches and AABB reductions |
| `line` | build | a long thin chain of overlapping AABBs on the diagonal |
| `sliver-soup` | traversal | O(N) node visits and triangle tests **per ray**: long thin slivers fanning through a small ball, so AABB culling is impossible |
| `bimodal` | both | "teapot in a stadium": 90 % of the primitives in one blob plus 10 % over a huge box → zero culling near the blob (measured culling ratio exactly 0.5) and a large-scale disparity |
| `instance-chain` | both | 256 chained meshes → 256 radix sorts and a chained (caterpillar) TLAS, i.e. the multi-BLAS/multi-instance path |
| `--mesh <file.obj>` | both | a real asset (a tiny built-in OBJ reader: `v`/`f`, fan triangulation, negative indices); nothing is bundled |

Rays are generated in a kernel with a fixed PCG hash, so all three backends trace
bit-identical rays; `--scene worst` runs only the scenes marked as worst cases.

## What is measured

* **Build**: total ms and `Mprim/s`, plus the per-stage breakdown (primitive AABBs,
  Morton codes, the four radix-sort passes, the radix-tree construction) through the
  opt-in `LbvhBuildTimings` hook; the number of nodes.
* **Tree shape**: `max_depth`, `mean_depth`, `depth / log2(n)` (the unbalance
  measure), the leaf range of every internal node (`max_range`, `mean_range` — the
  cost driver of the AABB reduction), and the number of descent failures instead of
  a hang on a malformed tree.
* **Traversal**: ms, `Mray/s`, ns/ray, hit/miss counts, plus the instrumented
  walk's counters (`nodes`, slab tests, tests that passed, triangle tests, deepest
  stack, culling ratio) — these say *why* a scene is slow, not just that it is.
* **Ranking**: both scenes and the counters are ranked against `uniform`, per
  primitive for the build and per ray for the traversal.

## Findings (RTX 4060, release, `--iters 3`, min of 3; cuda / dx / vk)

### The build was the radix-tree construction, and it was memory- and imbalance-bound

The `--stress-build` sweep showed the total build scaling at `p ≈ 1.02–1.05` with a
visible cache cliff between 256 K and 512 K primitives, and the per-stage timing
attributed **87–97 % of the build to the `node` stage** (the Karras radix-tree
construction) — every other stage was noise.  Inside that stage, a diagnostic run
that removed the AABB reduction showed the two searches cost ~1 % of it: the cost
was the reduction, which iterated `sum(leaf depth) ≈ n · mean_depth` times (~20 M
iterations for 1 M primitives), each doing a random 8 B `keys` read plus a random
**48 B** `prims[slot]` read.  It was also grossly imbalanced: one thread per
internal node means the root reduces the whole array alone.

Two changes fixed it (`examples/compute/lbvh/lbvh_storage.cpp`):

1. the radix tree is built in **two passes** — leaves first (the only place that
   still chases a random `prims` read, one per leaf), then internal nodes, whose
   reduction now streams the **contiguous** leaf-node array and needs no `keys`
   read at all;
2. the internal-node pass uses **one warp per node** (a lane-strided range plus
   `warp_active_min/max`), which removes the imbalance.

| scene | primitives | build cuda before → after | node stage | build dx before → after | node stage |
|---|---|---|---|---|---|
| `uniform` | 1 048 577 | 179.3 → **21.5 ms** (8.3×) | 168.2 → **7.2 ms** (23×) | 347.6 → **22.5 ms** (15.4×) | 334.2 → **9.3 ms** (36×) |
| `grid-duplicates` | 1 048 577 | 176.7 → **16.3 ms** (10.8×) | 166.1 → **6.6 ms** (25×) | 331.8 → **19.0 ms** (17.5×) | 321.2 → **9.3 ms** (35×) |
| `line` | 1 048 577 | 151.6 → **16.1 ms** (9.4×) | 140.3 → **6.3 ms** (22×) | 336.0 → **18.9 ms** (17.8×) | 326.0 → **9.2 ms** (35×) |
| `exponential` | 1 048 577 | 144.8 → **18.6 ms** (7.8×) | 133.5 → **8.9 ms** (15×) | 313.8 → **20.9 ms** (15.0×) | 304.2 → **11.0 ms** (28×) |
| `coincident` | 262 145 | 24.2 → **3.4 ms** (7.1×) | 17.3 → **1.1 ms** (16×) | 43.7 → **4.2 ms** (10.3×) | 41.0 → **1.8 ms** (22×) |
| `bimodal` | 262 145 | 21.7 → **3.5 ms** (6.3×) | 20.1 → **1.0 ms** (20×) | 43.4 → **4.5 ms** (9.6×) | 40.9 → **2.0 ms** (21×) |
| `sliver-soup` | 16 385 | 1.3 → **0.29 ms** (4.6×) | 1.15 → **0.11 ms** (10×) | 3.0 → **0.59 ms** (5.0×) | 2.6 → **0.28 ms** (9.5×) |
| `instance-chain` (256 trees) | 1 048 832 | 84.7 → **20.2 ms** (4.2×) | 76.8 → **11.3 ms** (6.8×) | 201.3 → **61.8 ms** (3.3×) | 177.4 → **36.4 ms** (4.9×) |

vk follows dx (`uniform` 359.7 → 22.5 ms, node 347.2 → 8.2 ms).  The fitted scaling
exponent dropped from `1.02–1.05` to `0.85–0.89` overall and from `1.03–1.06` to
`0.79–0.84` for the `node` stage, the 256 K–512 K cliff is gone, and the largest
size the benchmark's own pre-flight accepts rose from `2^21` to the benchmark's cap
`2^23` primitives (measured: `uniform` at 2^23 in 202 ms, worst submission 64 ms).

### The traversal is latency-bound; the blind push wastes about half of its pops

The counters show the walk existing today pushes both children unconditionally and
tests them when they are popped: for `uniform`, 141.7 nodes are popped, 141.7 slab
tests are executed and only 71.3 pass.  Testing a child before pushing it halves the
pops, and pushing the *farther* child first lets `best.t` tighten earlier.  A full
ordered tested descent was implemented and measured: it wins up to **−61 %**
(`line`), −51 % (`instance-chain`), −39 % (`uniform`), −11 % (`bimodal`) — but it
costs one extra node load per pushed node (the child is read to be tested and read
again when popped), which a zero-culling tree pays in full: `coincident` +25 % and
`grid-duplicates` +8/+13 %.  The traversal therefore keeps the blind-push walk (no
regression anywhere) and the trade-off is documented above; the two remaining ways
to have both are a stack entry that carries the pushed child's child-pointers (so
the pop needs no load — already measured once as slower) or a per-tree choice made
at build time from the fraction of internal nodes whose children have identical
AABBs.

Keeping the stack bound honest was worth one small change: `traversal_stack_size`
went from 92 to 64 with a written justification (30 Morton bits + 32 index bits
bound a path to ≤ 62 nodes, and the observed maximum pending depth is 31).

### Where the time is now

After the fix the radix sort is the dominant build stage (53–63 % of the build on
cuda for the 1 M scenes; 43 % for `instance-chain`), and it is a single-work-group
4×8-bit LSD sort that moves 16 B per element per pass with four `sync_block()`s per
256-element chunk — no inter-block parallelism.  For `instance-chain`, 257
sequential single-work-group submissions also make the per-dispatch overhead itself
dominant (`prim` 19 %, `morton` 18 %, `sort` 30 %).  That is the next thing to
attack: either a multi-block radix sort with per-block histograms and a global scan
per pass, or one work-group per tree with the K trees of a scene batched into one
dispatch per pass.

### Verified behaviour

Every run of the catalogue is also a correctness check: `--validate` self-checks the
structure of every tree (reachability, parent-vs-children AABBs) and cross-checks
the sampled hits against the Luisa RTX reference on the same buffers.  On cuda, dx
and vk the catalogue reports 0 structural problems and 0 hit/miss, distance and
instance/primitive mismatches; the accompanying `example_software_lbvh` demo (which
uses a non-degenerate scene and requires an exact primitive match) still passes
unchanged on all three backends.

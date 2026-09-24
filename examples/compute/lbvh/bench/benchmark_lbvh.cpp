// Performance benchmark of the two-level software LBVH.
//
// The benchmark drives the LBVH with adversarial (worst-case) scenes and reports
// what an optimizer needs to act on:
//
//   * a per-stage build breakdown (primitive AABBs, Morton codes, the four radix
//     passes and the radix-tree construction), obtained from the library's
//     opt-in `LbvhBuildTimings` hook - without it a build is one opaque block of
//     recorded work and the dominant stage is invisible;
//   * the shape of the built trees (leaf depths, internal-node leaf ranges),
//     which is what the internal-node AABB reduction pays for;
//   * the traversal counters of an instrumented copy of the two-level walk
//     (nodes popped, slab tests, tests that passed, triangle tests, deepest
//     stack), which is what says whether a tree actually culls;
//   * a ranking of every scene against the `uniform` baseline; the ratios are
//     normalized per primitive (build) and per ray (traversal), which is exactly
//     the ratio at equal sizes and the only meaningful comparison otherwise;
//   * a device-memory report: the exact byte estimate of every measurement is
//     checked against `--budget-gib` *before* anything is allocated, and a
//     measurement that would not fit is skipped with a warning instead of
//     crashing into an out-of-memory (the machine has 8 GiB of VRAM and less
//     than 6 GiB are safe to use).
//
// Timing is host-observed wall time around `stream << ... << synchronize()`, so
// it includes the submission of the recorded work and the fence: only
// release-mode numbers are meaningful, while a debug/ASan run is for correctness
// (use `--iters 1` and small sizes there).
//
// Usage: example_software_lbvh_bench <backend> [options]   (see --help)

#include "bench_harness.h"
#include "bench_scenes.h"
#include "bench_stats.h"

#include "../software_lbvh.h"

#include <cstdio>
#include <cstring>

#include <algorithm>
#include <cmath>

using namespace luisa;
using namespace luisa::compute;
using namespace luisa::example::lbvh;

namespace {

// ---------------------------------------------------------------------------
// Measurement records and reporting
// ---------------------------------------------------------------------------

// One measurement of one scene (or of one sweep step).  Everything the report
// prints comes from this record, so the human tables and the machine-readable
// `key=value` lines can never disagree.
struct BenchRun {
    luisa::string scene;
    luisa::string step;// empty for the scene's own measurement, "build n=..." / "trace rays=..." otherwise
    bool measured{false};
    bool skipped{false};
    luisa::string skip_reason;
    size_t triangles{0u};
    size_t vertices{0u};
    size_t instances{0u};
    size_t blas_count{0u};
    size_t primitives{0u};
    size_t rays{0u};
    size_t rays_requested{0u};// what was asked for: larger than `rays` only when the
                              // traversal had to be reduced to fit --max-seconds
    uint64_t seed{0u};
    BenchMemoryEstimate memory;
    size_t actual_bytes{0u};
    BenchBuildTiming build;
    BenchTreeStats tree;
    BenchTraceTiming trace;
    BenchRayStats ray_stats;
    size_t tree_problems{0u};// structural self-check (only under --validate)
    size_t walk_ties{0u};    // instrumented vs library walk: same distance, other primitive
    // Safety bookkeeping: the largest single device submission this measurement
    // observed (max over the timed iterations and, for the traversal, over the
    // chunks), and how many chunks the ray range was split into.  A submission
    // cannot be aborted from the host, so these are the numbers that prove the
    // run stayed clear of the driver's reset timeout.
    double worst_build_dispatch_ms{0.0};
    double worst_trace_chunk_ms{0.0};
    size_t trace_chunks{0u};
    double dispatch_budget_ms{0.0};
    // the traversal plan: how many rays one submission covers, the per-ray cost it
    // was based on and what one submission of that size actually took
    size_t trace_chunk_rays{0u};
    double trace_per_ray_ns{0.0};
    double trace_validated_ms{0.0};
    // Storage compaction (`--compact`): the loose structure the build produced,
    // the dense one after the copy, the bytes reclaimed (and, when the optional
    // build-scratch release ran, the scratch bytes reclaimed separately) and the
    // copy time.  The traversal is measured on both structures with the same plan
    // and the hits must agree bit-for-bit.
    bool compacted{false};
    size_t compact_nodes_before{0u};
    size_t compact_nodes_after{0u};
    size_t compact_reclaimed_bytes{0u};
    size_t compact_reclaimed_scratch_bytes{0u};
    double compact_ms{0.0};
    BenchTraceTiming trace_loose;
    size_t compact_hit_mismatches{0u};
};

// Machine-readable records: one line per measurement, `key=value`, no timestamp
// and no logger decoration, so the output of the three backends can be diffed or
// plotted directly.  printf (not LUISA_INFO) is used for exactly that reason:
// the console logger prepends a timestamp to every line.
void print_machine_readable(const BenchRun &run) noexcept {
    auto key = run.scene.c_str();
    auto step = run.step.empty() ? "-" : run.step.c_str();
    if (run.skipped) {
        std::printf("bench_skip scene=%s step=%s estimated_bytes=%llu reason=%s\n",
                    key, step, static_cast<unsigned long long>(run.memory.total_bytes()),
                    run.skip_reason.c_str());
        return;
    }
    std::printf("bench_scene scene=%s step=%s prims=%llu triangles=%llu vertices=%llu instances=%llu blas=%llu rays=%llu rays_requested=%llu seed=%llu tree_problems=%llu walk_ties=%llu\n",
                key, step,
                static_cast<unsigned long long>(run.primitives),
                static_cast<unsigned long long>(run.triangles),
                static_cast<unsigned long long>(run.vertices),
                static_cast<unsigned long long>(run.instances),
                static_cast<unsigned long long>(run.blas_count),
                static_cast<unsigned long long>(run.rays),
                static_cast<unsigned long long>(run.rays_requested),
                static_cast<unsigned long long>(run.seed),
                static_cast<unsigned long long>(run.tree_problems),
                static_cast<unsigned long long>(run.walk_ties));
    if (run.build.total_ms.count() != 0u) {
        auto staged = run.build.prim_ms.min_ms() + run.build.morton_ms.min_ms() +
                      run.build.sort_ms.min_ms() + run.build.node_ms.min_ms();
        auto percent = [staged](double ms) noexcept {
            return staged > 0.0 ? 100.0 * ms / staged : 0.0;
        };
        std::printf("bench_build scene=%s step=%s iters=%llu total_ms_min=%.6f total_ms_median=%.6f total_ms_mean=%.6f mprim_per_s=%.3f stages_ms=%.6f prim_ms=%.6f prim_pct=%.2f morton_ms=%.6f morton_pct=%.2f sort_ms=%.6f sort_pct=%.2f node_ms=%.6f node_pct=%.2f nodes=%llu\n",
                    key, step,
                    static_cast<unsigned long long>(run.build.total_ms.count()),
                    run.build.total_ms.min_ms(), run.build.total_ms.median_ms(),
                    run.build.total_ms.mean_ms(), run.build.mprim_per_s(), staged,
                    run.build.prim_ms.min_ms(), percent(run.build.prim_ms.min_ms()),
                    run.build.morton_ms.min_ms(), percent(run.build.morton_ms.min_ms()),
                    run.build.sort_ms.min_ms(), percent(run.build.sort_ms.min_ms()),
                    run.build.node_ms.min_ms(), percent(run.build.node_ms.min_ms()),
                    static_cast<unsigned long long>(run.build.nodes));
    }
    if (run.tree.measured()) {
        std::printf("bench_tree scene=%s step=%s leaves=%llu internals=%llu max_depth=%llu mean_depth=%.4f depth_per_log2_n=%.4f max_range=%.0f mean_range=%.3f descent_failures=%llu depth_capped=%llu\n",
                    key, step,
                    static_cast<unsigned long long>(run.tree.leaf_count),
                    static_cast<unsigned long long>(run.tree.internal_count),
                    static_cast<unsigned long long>(run.tree.max_depth),
                    run.tree.mean_depth, run.tree.depth_per_log2_n,
                    run.tree.max_range, run.tree.mean_range,
                    static_cast<unsigned long long>(run.tree.descent_failures),
                    static_cast<unsigned long long>(run.tree.depth_capped));
    }
    if (run.trace.total_ms.count() != 0u) {
        std::printf("bench_trace scene=%s step=%s iters=%llu trace_ms_min=%.6f trace_ms_median=%.6f trace_ms_mean=%.6f mray_per_s=%.3f ns_per_ray=%.4f hits=%llu misses=%llu avg_nodes=%.3f avg_aabb_tests=%.3f avg_aabb_hits=%.3f avg_tri_tests=%.3f avg_max_stack=%.3f nodes_max=%llu stack_max=%llu culled_ratio=%.4f\n",
                    key, step,
                    static_cast<unsigned long long>(run.trace.total_ms.count()),
                    run.trace.total_ms.min_ms(), run.trace.total_ms.median_ms(),
                    run.trace.total_ms.mean_ms(), run.trace.mray_per_s(),
                    run.trace.ns_per_ray(),
                    static_cast<unsigned long long>(run.trace.hits),
                    static_cast<unsigned long long>(run.trace.misses),
                    run.ray_stats.avg_nodes, run.ray_stats.avg_aabb_tests,
                    run.ray_stats.avg_aabb_hits, run.ray_stats.avg_tri_tests,
                    run.ray_stats.avg_max_stack,
                    static_cast<unsigned long long>(run.ray_stats.nodes_max),
                    static_cast<unsigned long long>(run.ray_stats.stack_max),
                    run.ray_stats.culled_ratio);
    }
    std::printf("bench_dispatch scene=%s step=%s worst_trace_chunk_ms=%.6f trace_chunks=%llu chunk_rays=%llu per_ray_ns=%.1f validated_ms=%.6f worst_build_dispatch_ms=%.6f budget_ms=%.3f\n",
                key, step, run.worst_trace_chunk_ms,
                static_cast<unsigned long long>(run.trace_chunks),
                static_cast<unsigned long long>(run.trace_chunk_rays),
                run.trace_per_ray_ns, run.trace_validated_ms,
                run.worst_build_dispatch_ms, run.dispatch_budget_ms);
    std::printf("bench_memory scene=%s step=%s estimated_bytes=%llu actual_bytes=%llu\n",
                key, step, static_cast<unsigned long long>(run.memory.total_bytes()),
                static_cast<unsigned long long>(run.actual_bytes));
    if (run.compacted || run.compact_nodes_before != 0u) {
        // only emitted when a compaction ran, so the records of a plain run are
        // byte-for-byte what they were before this option existed
        std::printf("bench_compact scene=%s step=%s compacted=%d iters=%llu nodes_before=%llu nodes_after=%llu reclaimed_bytes=%llu reclaimed_scratch_bytes=%llu compact_ms=%.6f "
                    "loose_trace_ms_min=%.6f dense_trace_ms_min=%.6f loose_ns_per_ray=%.4f dense_ns_per_ray=%.4f hit_mismatches=%llu\n",
                    key, step, run.compacted ? 1 : 0,
                    static_cast<unsigned long long>(run.trace.total_ms.count()),
                    static_cast<unsigned long long>(run.compact_nodes_before),
                    static_cast<unsigned long long>(run.compact_nodes_after),
                    static_cast<unsigned long long>(run.compact_reclaimed_bytes),
                    static_cast<unsigned long long>(run.compact_reclaimed_scratch_bytes),
                    run.compact_ms,
                    run.trace_loose.total_ms.min_ms(), run.trace.total_ms.min_ms(),
                    run.trace_loose.ns_per_ray(), run.trace.ns_per_ray(),
                    static_cast<unsigned long long>(run.compact_hit_mismatches));
    }
    std::fflush(stdout);
}

// The stage shares refer to the *staged* total (the sum of the four stage
// minima), not to the plain build: the staged build synchronises between its
// stages, so it is a little slower than the recorded one by construction.
[[nodiscard]] double staged_total_ms(const BenchBuildTiming &build) noexcept {
    return build.prim_ms.min_ms() + build.morton_ms.min_ms() +
           build.sort_ms.min_ms() + build.node_ms.min_ms();
}

void print_human_summary(const BenchRun &run) noexcept {
    auto staged = staged_total_ms(run.build);
    auto percent = [staged](double ms) noexcept {
        return staged > 0.0 ? 100.0 * ms / staged : 0.0;
    };
    LUISA_INFO("[{}] {} triangles ({} vertices), {} instances, {} BLAS, {} rays, seed {}",
               run.scene, run.triangles, run.vertices, run.instances, run.blas_count,
               run.rays, run.seed);
    if (run.build.total_ms.count() != 0u) {
        LUISA_INFO("  build : min {:.3f} / median {:.3f} / mean {:.3f} ms over {} iteration(s), {:.2f} Mprim/s, {} nodes",
                   run.build.total_ms.min_ms(), run.build.total_ms.median_ms(),
                   run.build.total_ms.mean_ms(), run.build.total_ms.count(),
                   run.build.mprim_per_s(), run.build.nodes);
        LUISA_INFO("  stages: prim {:.3f} ms ({:.1f}%), morton {:.3f} ms ({:.1f}%), sort {:.3f} ms ({:.1f}%), node {:.3f} ms ({:.1f}%)  [staged total {:.3f} ms]",
                   run.build.prim_ms.min_ms(), percent(run.build.prim_ms.min_ms()),
                   run.build.morton_ms.min_ms(), percent(run.build.morton_ms.min_ms()),
                   run.build.sort_ms.min_ms(), percent(run.build.sort_ms.min_ms()),
                   run.build.node_ms.min_ms(), percent(run.build.node_ms.min_ms()), staged);
    }
    if (run.tree.measured()) {
        LUISA_INFO("  tree  : max depth {} ({:.2f} x log2 leaf count), mean depth {:.2f}, leaf range max {:.0f} / mean {:.1f}, {} descent failure(s), {} capped",
                   run.tree.max_depth, run.tree.depth_per_log2_n, run.tree.mean_depth,
                   run.tree.max_range, run.tree.mean_range, run.tree.descent_failures,
                   run.tree.depth_capped);
    }
    if (run.trace.total_ms.count() != 0u) {
        LUISA_INFO("  trace : min {:.3f} / median {:.3f} / mean {:.3f} ms, {:.2f} Mray/s, {:.1f} ns/ray, {} hits / {} misses",
                   run.trace.total_ms.min_ms(), run.trace.total_ms.median_ms(),
                   run.trace.total_ms.mean_ms(), run.trace.mray_per_s(),
                   run.trace.ns_per_ray(), run.trace.hits, run.trace.misses);
        LUISA_INFO("  walk  : {:.2f} nodes, {:.2f} slab tests ({:.1f}% culled), {:.2f} triangle tests, avg stack {:.1f} (deepest ray {}), nodes max {}",
                   run.ray_stats.avg_nodes, run.ray_stats.avg_aabb_tests,
                   100.0 * run.ray_stats.culled_ratio, run.ray_stats.avg_tri_tests,
                   run.ray_stats.avg_max_stack, run.ray_stats.stack_max,
                   run.ray_stats.nodes_max);
    }
    LUISA_INFO("  memory: {:>10} estimated, {:>10} allocated",
               human_bytes(run.memory.total_bytes()), human_bytes(run.actual_bytes));
    if (run.compacted || run.compact_nodes_before != 0u) {
        // The "friendly cache hit" claim, as a number: the dense structure is
        // traced after the loose one with the same rays, so the two ns/ray are an
        // interleaved A/B of the two layouts.
        LUISA_INFO("  compact: {} -> {} node(s), {} reclaimed ({}), {} build scratch reclaimed, copy {:.3f} ms; trace {:.1f} -> {:.1f} ns/ray ({} hit mismatch(es))",
                   run.compact_nodes_before, run.compact_nodes_after,
                   human_bytes(run.compact_reclaimed_bytes),
                   run.compacted ? "dense" : "no-op",
                   human_bytes(run.compact_reclaimed_scratch_bytes),
                   run.compact_ms, run.trace_loose.ns_per_ray(), run.trace.ns_per_ray(),
                   run.compact_hit_mismatches);
    }
    if (run.measured) {
        // The safety claim of the run: one submission is what the driver resets
        // the device for, and these are the largest ones this scene produced.
        LUISA_INFO("  dispatch: worst build submission {:.3f} ms, worst traversal submission {:.3f} ms, {} chunk(s) of {} rays ({:.1f} ns/ray, validated {:.1f} ms), budget {:.1f} ms",
                   run.worst_build_dispatch_ms, run.worst_trace_chunk_ms,
                   run.trace_chunks, run.trace_chunk_rays, run.trace_per_ray_ns,
                   run.trace_validated_ms, run.dispatch_budget_ms);
        // trace_ms is the *sum* over the strided slices: it is the time of the
        // whole traversal, not of one submission (see --help).
    }
}

void print_catalogue() noexcept {
    std::printf("-- scene catalogue --\n");
    std::printf("%-15s  %-9s  %9s  %9s  %8s  %s\n", "scene", "stress", "triangles", "instances",
                "rays", "worst case");
    std::printf("---------------  ---------  ---------  ---------  --------  ------------------------------------------\n");
    for (auto &&info : bench_scene_catalogue()) {
        std::printf("%-15s  %-9s  %9llu  %9llu  %8llu  %s\n", info.name, info.stress,
                    static_cast<unsigned long long>(info.default_triangles),
                    static_cast<unsigned long long>(info.default_instances),
                    static_cast<unsigned long long>(info.default_rays), info.worst_case);
        // the defaults are tuned against the measured cost of one submission,
        // because a multi-second dispatch removes the device (TDR)
        std::printf("%-15s  %-9s  %9s  %9s  %8s  defaults: %s\n", "", "", "", "", "",
                    info.dispatch_note);
    }
    std::printf("---------------  ---------  ---------  ---------  --------  ------------------------------------------\n");
    std::printf("ratios in the ranking are normalized: per primitive for the build,\n"
                "per ray for the traversal, both against the 'uniform' baseline.\n"
                "every run additionally chunks the traversal and pre-flights the build so\n"
                "that no single device submission is predicted to exceed\n"
                "--dispatch-budget-ms (default 1000 ms).\n");
    std::fflush(stdout);
}

// ---------------------------------------------------------------------------
// Device resources of one measurement
// ---------------------------------------------------------------------------

[[nodiscard]] size_t buffer_bytes(const Buffer<float3> &buffer) noexcept {
    return buffer.size() * buffer.stride();
}
[[nodiscard]] size_t buffer_bytes(const Buffer<Triangle> &buffer) noexcept {
    return buffer.size() * buffer.stride();
}
[[nodiscard]] size_t buffer_bytes(const Buffer<LbvhRay> &buffer) noexcept {
    return buffer.size() * buffer.stride();
}
[[nodiscard]] size_t buffer_bytes(const Buffer<LbvhHit> &buffer) noexcept {
    return buffer.size() * buffer.stride();
}
[[nodiscard]] size_t buffer_bytes(const Buffer<LbvhRayStats> &buffer) noexcept {
    return buffer.size() * buffer.stride();
}

// Everything one measurement owns on the device.  It is created after the budget
// check and destroyed before the next measurement, so the peak device memory of
// the whole benchmark is one measurement, not one sweep.
struct SceneResources {
    Buffer<float3> vertices;
    Buffer<Triangle> triangles;
    Buffer<LbvhRay> rays;
    Buffer<LbvhHit> hits;
    Buffer<LbvhHit> reference_hits;// instrumented walk, and the RTX reference
    Buffer<LbvhRayStats> ray_stats;
    luisa::unique_ptr<SoftwareLbvh> lbvh;
    luisa::unique_ptr<BenchStats> bench_stats;
    luisa::vector<Blas> blases;
    Tlas tlas;
    size_t actual_bytes{0u};
    size_t ray_capacity{0u};

    // `ray_count` is clamped to one element: a build-only measurement (a sweep
    // step) allocates no rays but the buffers must stay non-empty.  `headroom`
    // scales the storage capacity (the loose slack `--compact` reclaims) and
    // `allow_compaction` sets the `AccelOption` the compaction requires.
    SceneResources(Device &device, Stream &stream, const BenchScene &scene, size_t ray_count,
                   bool with_stats, double headroom = 1.0,
                   bool allow_compaction = false) noexcept
        : vertices{device.create_buffer<float3>(scene.vertices.size())},
          triangles{device.create_buffer<Triangle>(scene.triangles.size())},
          rays{device.create_buffer<LbvhRay>(std::max<size_t>(ray_count, 1u))},
          hits{device.create_buffer<LbvhHit>(std::max<size_t>(ray_count, 1u))},
          reference_hits{device.create_buffer<LbvhHit>(std::max<size_t>(ray_count, 1u))},
          ray_stats{device.create_buffer<LbvhRayStats>(std::max<size_t>(ray_count, 1u))},
          ray_capacity{std::max<size_t>(ray_count, 1u)} {
        // Upload the geometry once; it is shared by every BLAS (triangle indices
        // are global ids into this buffer, exactly as in the demo).
        stream << vertices.copy_from(luisa::span{scene.vertices})
               << triangles.copy_from(luisa::span{scene.triangles})
               << synchronize();
        // estimate -> create -> pre_build (the backend order, see software_lbvh.h)
        auto capacity_triangles = static_cast<size_t>(
            std::ceil(static_cast<double>(scene.triangles.size()) * headroom));
        auto capacity_instances = static_cast<size_t>(
            std::ceil(static_cast<double>(scene.instances.size()) * headroom));
        auto sizes = SoftwareLbvh::estimate(capacity_triangles, capacity_instances,
                                            scene.meshes.size());
        lbvh = luisa::make_unique<SoftwareLbvh>(device, sizes);
        AccelOption option;
        option.allow_compaction = allow_compaction;
        blases.reserve(scene.meshes.size());
        for (auto &&mesh : scene.meshes) {
            auto blas = lbvh->create_blas(option, mesh.triangle_offset,
                                          mesh.triangle_count, mesh.lo, mesh.hi);
            lbvh->pre_build_blas(blas);
            blases.emplace_back(blas);
        }
        luisa::vector<InstanceDesc> descriptions;
        descriptions.reserve(scene.instances.size());
        for (auto &&instance : scene.instances) {
            descriptions.emplace_back(InstanceDesc{instance.to_world, instance.mesh});
        }
        tlas = lbvh->create_accel(option, static_cast<uint>(descriptions.size()));
        lbvh->pre_build_accel(stream, tlas, luisa::span{blases}, luisa::span{descriptions});
        if (with_stats) {
            bench_stats = luisa::make_unique<BenchStats>(device, sizes.node_capacity,
                                                         sizes.primitive_capacity);
        }
        actual_bytes = buffer_bytes(vertices) + buffer_bytes(triangles) +
                       buffer_bytes(rays) + buffer_bytes(hits) +
                       buffer_bytes(reference_hits) + buffer_bytes(ray_stats) +
                       lbvh_storage_bytes(sizes, allow_compaction) +
                       (with_stats ? bench_stats->scratch_bytes() : 0u);
    }
};

// The ray generator is identical for every scene, so it is compiled once (the
// first compile of a kernel costs seconds, later ones are cache hits).
using BenchRayShader = Shader1D<Buffer<LbvhRay>, uint, uint, uint, uint, float3, float3,
                                float3, float3, float, float, float3, float, float3>;

// ---------------------------------------------------------------------------
// Dispatch budget
//
// A *single* device submission that runs for seconds makes the Windows driver
// reset the device (TDR, ~2 s) - that is exactly what DXGI_ERROR_DEVICE_REMOVED
// and VK_ERROR_DEVICE_LOST are.  A submitted submission cannot be aborted from
// the host, so the only way to be safe at *any* requested size is to never
// submit one that is predicted to be too long:
//
//   * the traversal is one thread per ray with no cross-ray interaction, so a
//     probe of a few hundred rays predicts the cost of any ray count: the rays
//     are split into chunks whose predicted time stays under the budget, and the
//     chunks are separated by a synchronise();
//   * the build is dominated by the `node` stage (92-96 % of it), which grows
//     super-linearly with the primitive count: a scene whose build is predicted
//     to exceed the budget is refused before it is submitted, and the stress
//     sweeps stop at the last size they can predict to be safe.
//
// Every prediction is inflated by `dispatch_safety` so that being wrong is on
// the safe side, and the measured worst submission of every scene is reported
// (`bench_dispatch`) so the claim is checkable.
// ---------------------------------------------------------------------------

constexpr double dispatch_safety = 1.25;

// The largest single submission of one build: the per-stage times of a *scene*
// sum over its trees, which says nothing about the largest submission, so every
// tree is timed separately and the maximum per stage is kept.
struct BenchWorstDispatch {
    double prim_ms{0.0};
    double morton_ms{0.0};
    double sort_ms{0.0};// the 4 LSD passes together
    double node_ms{0.0};

    void add(const LbvhBuildTimings &timings) noexcept {
        prim_ms = std::max(prim_ms, timings.prim_ms);
        morton_ms = std::max(morton_ms, timings.morton_ms);
        sort_ms = std::max(sort_ms, timings.sort_ms);
        node_ms = std::max(node_ms, timings.node_ms);
    }
    // The sort stage is four dispatches of one work-group each, so one of its
    // submissions is a quarter of the stage; `node` is always the largest.
    [[nodiscard]] double value_ms() const noexcept {
        return std::max(std::max(prim_ms, morton_ms), std::max(sort_ms * 0.25, node_ms));
    }
};

// Conservative growth of the largest build stage between two sizes: at least
// 2.5x per doubling (2.5^log2(ratio), the measured L2 cliff), at least
// ratio^1.2, and at least the growth that was just observed.
[[nodiscard]] double predict_growth(double prev_size, double next_size) noexcept {
    auto ratio = next_size > prev_size && prev_size > 0.0 ? next_size / prev_size : 1.0;
    return std::max(std::pow(2.5, std::log2(ratio)), std::pow(ratio, 1.2));
}

[[nodiscard]] double predict_stage_ms(double prev_ms, double prevprev_ms, double prev_size,
                                      double next_size) noexcept {
    auto factor = predict_growth(prev_size, next_size);
    if (prevprev_ms > 0.0 && prev_ms > prevprev_ms) {
        factor = std::max(factor, prev_ms / prevprev_ms);// the growth just measured
    }
    return prev_ms * factor * dispatch_safety;
}

// One full build of every tree of the scene: K BLASes then the TLAS, the order
// the demo and the hardware backends use.  `timings` optionally asks the library
// to separate the stages (which synchronises between them); `worst` keeps the
// largest single submission of the build.
[[nodiscard]] double build_scene(Stream &stream, SceneResources &resources,
                                 LbvhBuildTimings *timings,
                                 BenchWorstDispatch *worst) noexcept {
    Clock clock;
    clock.tic();
    if (timings != nullptr) { *timings = LbvhBuildTimings{}; }
    for (auto &&blas : resources.blases) {
        LbvhBuildTimings tree;
        resources.lbvh->build_blas(stream, blas, resources.vertices, resources.triangles,
                                   AccelBuildRequest::PREFER_UPDATE,
                                   timings != nullptr ? &tree : nullptr);
        if (timings != nullptr) {
            timings->prim_ms += tree.prim_ms;
            timings->morton_ms += tree.morton_ms;
            timings->sort_ms += tree.sort_ms;
            timings->node_ms += tree.node_ms;
            worst->add(tree);
        }
    }
    LbvhBuildTimings tlas;
    resources.lbvh->build_accel(stream, resources.tlas, AccelBuildRequest::PREFER_UPDATE,
                                timings != nullptr ? &tlas : nullptr);
    if (timings != nullptr) {
        timings->prim_ms += tlas.prim_ms;
        timings->morton_ms += tlas.morton_ms;
        timings->sort_ms += tlas.sort_ms;
        timings->node_ms += tlas.node_ms;
        worst->add(tlas);
    }
    stream << synchronize();
    return clock.toc();
}

// One submission of `count` rays starting at `offset` of the ray buffer.
[[nodiscard]] double trace_scene(Stream &stream, SceneResources &resources, size_t count,
                                 size_t offset) noexcept {
    Clock clock;
    clock.tic();
    resources.lbvh->trace_software(stream, resources.vertices, resources.triangles,
                                   resources.rays, resources.hits, resources.tlas,
                                   static_cast<uint>(count), static_cast<uint>(offset));
    stream << synchronize();
    return clock.toc();
}

// One submission of a strided slice of the (instrumented or library) traversal:
// it walks the ray indices `offset, offset + stride, ...` and is followed by a
// synchronise(); returns the host-observed milliseconds.
[[nodiscard]] double time_trace_dispatch(Stream &stream, SceneResources &resources,
                                         bool instrumented, size_t count, size_t offset,
                                         size_t stride) noexcept {
    Clock clock;
    clock.tic();
    if (instrumented) {
        resources.bench_stats->trace_instrumented(
            stream, resources.tlas.heap(), resources.lbvh->blas_table(),
            resources.lbvh->instances(), resources.vertices, resources.triangles,
            resources.rays, resources.reference_hits, resources.ray_stats,
            resources.tlas.node_offset(), static_cast<uint>(count),
            static_cast<uint>(offset), static_cast<uint>(stride));
    } else {
        resources.lbvh->trace_software(stream, resources.vertices, resources.triangles,
                                       resources.rays, resources.hits, resources.tlas,
                                       static_cast<uint>(count), static_cast<uint>(offset),
                                       static_cast<uint>(stride));
    }
    stream << synchronize();
    return clock.toc();
}

// How many rays a strided slice covers.
[[nodiscard]] size_t strided_count(size_t rays, size_t stride, size_t offset) noexcept {
    return offset >= rays ? 0u : (rays - offset + stride - 1u) / stride;
}

// The largest power of two that is not above `value` (at least 1).
[[nodiscard]] size_t round_down_pow2(size_t value) noexcept {
    auto result = size_t{1u};
    while (result <= value / 2u) { result <<= 1u; }
    return result;
}

// The slice policy of `plan_trace`, shared with the reporting of `plan_traversal`.
//
// A slice may use three quarters of the dispatch budget: it is measured warm, while
// the slices of the timed loop re-submit the same work on colder data (the observed
// drift is ~1.4x), so this threshold is the margin that drift needs.
constexpr auto dispatch_accept_fraction = 0.75;
// The smallest slice worth submitting: below this the submission overhead dominates.
constexpr auto min_slice_rays = 128u;
constexpr auto max_growth_steps = 32u;
// A plan that needs more slices than this is dispatch-dominated: its `trace_ms` is
// the sum of that many submissions and measures the submission path as much as the
// walk, so the number is flagged instead of being silently reported.
constexpr auto many_chunk_warning = 16u;
// How many slices of a candidate size are measured before the plan uses it (see
// `measure` in `plan_trace`: one offset is not representative of the whole ray set).
constexpr auto sample_offsets = 4u;
// With a single measurement there is no trend to extrapolate, so the worst case
// (the cost grows linearly with the ray count) is assumed for the first growth step.
constexpr auto initial_growth_exponent = 1.0;

// The plan of one traversal measurement: how many rays are traced, and how they
// are split into strided slices (one submission each).
struct TracePlan {
    size_t rays{0u};        // rays actually traced (see `reduced_from`)
    size_t requested{0u};   // rays the caller asked for
    size_t reduced_from{0u};// nonzero: the caller must *re-generate* `rays` rays (a smaller
                            // frustum) before this plan can be used - the first `rays` rays
                            // of the current buffer are a prefix of the frustum, which is not
                            // a smaller version of the scene
    size_t slices{0u};      // submissions per traversal
    size_t slice_rays{0u};  // rays in the largest slice
    double per_ray_ns{0.0};
    double validated_ms{0.0};// what the largest slice actually took

    [[nodiscard]] bool wants_smaller_ray_set() const noexcept { return reduced_from != 0u; }
    [[nodiscard]] size_t chunks() const noexcept { return slices; }
    // The time the whole traversal is predicted to take: the measured cost of one
    // slice times the number of slices.  This is the sum the timed loop submits,
    // and it is the number `--max-seconds` has to bound.
    [[nodiscard]] double predicted_total_ms() const noexcept {
        return validated_ms * static_cast<double>(slices);
    }
};

// Plans the traversal so that no single submission is known to exceed the
// dispatch budget, and reduces the ray set only as a *last resort*, when even
// the largest slice that fits the budget cannot trace the whole ray set within
// `--max-seconds` (a traversal of minutes is not a measurement, and its slices
// cannot be aborted either).  A reduction is not applied here but reported to the
// caller (`reduced_from`), which re-generates the rays at the smaller size: a
// smaller ray set has to be a smaller *frustum*, not a prefix of the big one (see
// `plan_traversal`).
//
// Four properties of the walk decide the design, all of them measured:
//
//   * the cost of a ray varies by orders of magnitude across the ray set (in a
//     camera frustum the rays that miss the scene cost a few node visits, the
//     ones that enter a dense blob cost hundreds of thousands), so every slice is
//     *strided* - a contiguous prefix can be 1000x cheaper per ray than the whole
//     range, which is exactly how a "safe" prediction submits a slow dispatch;
//   * the per-ray cost *falls* steeply with the size of the submission (a small
//     dispatch cannot hide the memory latency of the walk): measured on
//     `coincident`, 2.7 ms/ray at 128 rays against 3.9 us/ray at 65536 rays -
//     700x.  A per-ray cost measured on a small slice therefore over-estimates a
//     large one by orders of magnitude and must *not* be extrapolated to it: that
//     is what used to cap the plan at 128 rays and report a 73 s traversal for a
//     250 ms one, on one backend only;
//   * the cost of a slice nevertheless grows with its size, so the largest size
//     that fits the budget is found by *measuring* successively larger slices:
//     double the slice while the measured time of the current one leaves room
//     (`dispatch_accept_fraction` of the budget is the margin the colder slices of
//     the timed loop need), and keep the last size that fit;
//   * one slice is not representative of the slices it stands for, so a candidate
//     size is measured at several spread offsets and the *worst* of them decides
//     (see `measure` below: the strided slices of a power-of-two ray count collapse
//     onto a few columns of the frustum, and measuring only the first one is how the
//     plan ends up believing in a slice the timed loop cannot afford).
//
// The doubled size is only *submitted* once even the worst case of the measured
// trend fits the budget: a submission cannot be aborted from the host, so a growth
// step must never be a gamble.  `bound_by_max_seconds` is false for the
// *instrumented* pass: that pass must cover the same rays as the timed walk,
// otherwise the two walk different ray sets and the hit comparison between them is
// meaningless (it would compare freshly written hits against stale ones).
[[nodiscard]] TracePlan plan_trace(Stream &stream, SceneResources &resources,
                                   const BenchOptions &options, bool instrumented,
                                   size_t ray_count,
                                   bool bound_by_max_seconds = true) noexcept {
    auto verbose = std::getenv("LUISA_BENCH_PLAN_VERBOSE") != nullptr;
    TracePlan plan;
    plan.requested = ray_count;
    plan.rays = ray_count;
    auto budget_ms = std::max(options.dispatch_budget_ms, 1.0);
    auto accept_ms = budget_ms * dispatch_accept_fraction;
    // 0) warm the kernel up: the very first dispatch of a kernel includes its
    //    module load (tens of milliseconds), which would otherwise be charged to
    //    the first slice and inflate every prediction
    time_trace_dispatch(stream, resources, instrumented,
                        std::min<size_t>(min_slice_rays, ray_count), 0u, 1u);
    // Measure the slice the timed loop would submit when the ray range is split
    // into `slices` submissions: the stride is the slice count and the offsets
    // enumerate the slices, so the measured slice and the timed ones cover exactly
    // the same ray set (the measured one is the slice of offset 0).
    //
    // One offset is *not* representative of the slices it is drawn from: the timed
    // loop submits `(offset i, stride S)` for i in [0, S), and when the stride is a
    // multiple of the ray generator's grid width every ray of a slice lands in the
    // same column of the frustum - on `coincident` the column of offset 0 misses
    // the blob and costs 6000x less than the average slice, so a plan built on it
    // picks a slice that the real loop cannot afford.  A candidate size is
    // therefore measured at up to `sample_offsets` spread offsets and the *maximum*
    // is what drives the plan: the budget has to hold for the worst slice, not for
    // the cheapest one.  The sampling stops as soon as a slice is over the budget
    // itself, because from there on the size is unusable whatever the other slices
    // cost.
    auto measure = [&](size_t slices) noexcept {
        auto samples = std::min<size_t>(slices, sample_offsets);
        auto worst = 0.0;
        for (auto sample = size_t{0u}; sample < samples; sample++) {
            // with few slices, measure each one; otherwise spread the samples
            auto offset = samples == slices ? sample : slices * sample / samples;
            // every offset needs its *own* ray count: a strided slice covers
            // `(rays - offset + stride - 1) / stride` rays, which is one more
            // than the slice of offset 0 whenever `offset` is not a multiple of
            // the stride.  Re-using the count of offset 0 walks (and writes) past
            // the end of the ray/hit buffers (an out-of-bounds *write* into the
            // hits of the following buffers, which the debug build traps as
            // `index 20017 < buffer size 20000` and release silently accepts).
            auto count = strided_count(plan.rays, slices, offset);
            if (count == 0u) { continue; }
            worst = std::max(worst, time_trace_dispatch(stream, resources, instrumented,
                                                        count, offset, slices));
            if (worst > budget_ms) { break; }
        }
        return worst;
    };
    // The exponent of the measured cost curve, clamped to [0, 1].  What grows with
    // the slice size is the number of rays; the per-ray cost only falls, so a
    // linear growth of the measured step is an upper bound.  The clamp also keeps a
    // single noisy step (a timer hiccup, a cache cliff) from being extrapolated
    // into a projection that would block the growth for the rest of the plan.
    auto exponent_of = [](double ms0, double ms1, double n0, double n1) noexcept {
        if (ms0 <= 0.0 || ms1 <= 0.0 || n0 < 1.0 || n1 <= n0) { return initial_growth_exponent; }
        return std::min(std::max(std::log(ms1 / ms0) / std::log(n1 / n0), 0.0), 1.0);
    };
    // 1) find the largest slice that fits by *measuring* successively larger ones
    auto grow = [&]() noexcept {
        auto slices = std::max<size_t>((plan.rays + min_slice_rays - 1u) / min_slice_rays, 1u);
        auto size = strided_count(plan.rays, slices, 0u);
        auto ms = measure(slices);
        auto exponent = initial_growth_exponent;
        for (auto step = 0u; step < max_growth_steps && slices > 1u; step++) {
            auto next_slices = std::max<size_t>(slices / 2u, 1u);
            auto next_size = strided_count(plan.rays, next_slices, 0u);
            auto projected = size == 0u || next_size <= size ?
                                 ms :
                                 ms * std::pow(static_cast<double>(next_size) / static_cast<double>(size),
                                               exponent);
            if (verbose) {
                std::printf("bench_plan slices=%llu slice_rays=%llu ms=%.4f accept_ms=%.1f "
                            "exponent=%.3f next_slice_rays=%llu projected_ms=%.4f instrumented=%d\n",
                            static_cast<unsigned long long>(slices),
                            static_cast<unsigned long long>(size), ms, accept_ms, exponent,
                            static_cast<unsigned long long>(next_size), projected,
                            instrumented ? 1 : 0);
                std::fflush(stdout);
            }
            // the next size is only submitted when its worst case fits the budget
            if (projected > budget_ms) { break; }
            auto next_ms = measure(next_slices);
            exponent = exponent_of(ms, next_ms, static_cast<double>(size),
                                   static_cast<double>(next_size));
            // the measurement decides: a slice over the accept threshold is not used
            // and the previous size was the largest that fit
            if (next_ms > accept_ms) { break; }
            slices = next_slices;
            size = next_size;
            ms = next_ms;
        }
        plan.slices = std::min(slices, plan.rays);
        plan.slice_rays = strided_count(plan.rays, plan.slices, 0u);
        plan.validated_ms = ms;
        plan.per_ray_ns = plan.slice_rays == 0u ? 0.0 : ms * 1.0e6 / static_cast<double>(plan.slice_rays);
    };
    grow();
    // 2) `--max-seconds` is the last resort: it bounds the *whole* measurement,
    //    which is the sum over the slices, and it is predicted from the measured
    //    cost of one slice - never from an extrapolated per-ray cost, because a
    //    small slice looks orders of magnitude more expensive per ray than the full
    //    walk.  That extrapolation is what used to reduce the ray count on one
    //    backend only, which also made the traversal counters of the very same
    //    scene incomparable between the backends.  The plan only *reports* the
    //    reduction (`reduced_from`): the ray set has to be re-generated at the
    //    smaller size, because the first N rays of the buffer are not a smaller
    //    version of the scene but a prefix of the frustum - its rows can miss the
    //    scene entirely (`uniform` is the one scene where a prefix is harmless),
    //    so a prefix measures neither the scene nor its own backend.
    if (bound_by_max_seconds) {
        auto limit_ms = options.max_seconds * 1.0e3;
        if (plan.predicted_total_ms() > limit_ms) {
            auto per_slice_ms = std::max(plan.validated_ms, 1.0e-3);
            auto affordable_slices = std::max<size_t>(1u, static_cast<size_t>(limit_ms / per_slice_ms));
            auto affordable_rays = affordable_slices * std::max<size_t>(plan.slice_rays, 1u);
            // The reduced size is rounded down to a power of two, so that the same
            // scene and the same ray count reduce to the same ray set on every
            // backend (the backends differ by ~1.2x, and a power of two absorbs it).
            affordable_rays = round_down_pow2(std::min(affordable_rays, plan.rays));
            if (affordable_rays < plan.rays) {
                plan.rays = affordable_rays;
                plan.reduced_from = ray_count;
            }
        }
    }
    return plan;
}

// ---------------------------------------------------------------------------
// Timed measurements
// ---------------------------------------------------------------------------

// Build measurement: the plain build is the headline (no extra synchronisation),
// the staged build attributes the time to the stages.  Both are recorded per
// iteration, so the min/median/mean and the stage shares refer to the same runs.
// `worst_dispatch_ms` keeps the largest single submission over the iterations.
void measure_build(Stream &stream, SceneResources &resources, const BenchOptions &options,
                   BenchBuildTiming &timing, double &worst_dispatch_ms) noexcept {
    for (auto warmup = 0u; warmup < options.warmup; warmup++) {
        build_scene(stream, resources, nullptr, nullptr);
    }
    for (auto iteration = 0u; iteration < options.iterations; iteration++) {
        auto total = build_scene(stream, resources, nullptr, nullptr);
        LbvhBuildTimings stages;
        BenchWorstDispatch worst;
        build_scene(stream, resources, &stages, &worst);
        timing.total_ms.add(total);
        timing.prim_ms.add(stages.prim_ms);
        timing.morton_ms.add(stages.morton_ms);
        timing.sort_ms.add(stages.sort_ms);
        timing.node_ms.add(stages.node_ms);
        // The headline build is a *single* fence-to-fence submission of the whole
        // chain (`total`), while `worst` only knows about the staged rebuild that
        // produced the breakdown.  A multi-*tree* scene submits the whole chain
        // once (129 kernels for 256 trees here), so reporting `worst` alone
        // under-states the largest submission by up to ~50x; the budget check has
        // to see both.
        worst_dispatch_ms = std::max({worst_dispatch_ms, worst.value_ms(), total});
    }
}

// Traversal measurement: `plan` splits the rays so that no submission exceeds the
// dispatch budget (the chunks are separated by a synchronise(), so the reported
// `trace_ms` is the sum over all chunks - it is the time of the whole traversal,
// not of one submission).  Under `--repeat-check` the hits are read back after
// every iteration and compared with the first one, which is the strongest
// determinism check the traversal can be given.
void measure_trace(Stream &stream, SceneResources &resources, const BenchOptions &options,
                   const TracePlan &plan, BenchTraceTiming &timing, bool repeat_check,
                   size_t &repeat_mismatches, luisa::vector<LbvhHit> &host_hits,
                   double &worst_chunk_ms) noexcept {
    auto ray_count = plan.rays;
    // one slice of warm-up: the warm-up must not cost more than the measurement
    if (options.warmup != 0u) {
        time_trace_dispatch(stream, resources, false, plan.slice_rays, 0u, plan.slices);
    }
    luisa::vector<LbvhHit> reference;
    reference.resize(ray_count);
    for (auto iteration = 0u; iteration < options.iterations; iteration++) {
        auto total = 0.0;
        for (auto slice = size_t{0u}; slice < plan.slices; slice++) {
            auto count = strided_count(ray_count, plan.slices, slice);
            if (count == 0u) { continue; }
            auto ms = time_trace_dispatch(stream, resources, false, count, slice, plan.slices);
            total += ms;
            worst_chunk_ms = std::max(worst_chunk_ms, ms);
        }
        timing.total_ms.add(total);
        if (repeat_check) {
            stream << resources.hits.view(0u, ray_count)
                          .copy_to(luisa::span{reference.data(), ray_count})
                   << synchronize();
            // the reference of the first iteration is kept in `host_hits`
            if (iteration == 0u) {
                host_hits.resize(ray_count);
                std::memcpy(host_hits.data(), reference.data(), ray_count * sizeof(LbvhHit));
            } else if (std::memcmp(host_hits.data(), reference.data(),
                                   ray_count * sizeof(LbvhHit)) != 0) {
                repeat_mismatches++;
            }
        }
    }
    host_hits.resize(ray_count);
    stream << resources.hits.view(0u, ray_count).copy_to(luisa::span{host_hits})
           << synchronize();
    timing.rays = ray_count;
    timing.hits = 0u;
    timing.misses = 0u;
    for (auto i = 0u; i < ray_count; i++) {
        if (host_hits[i].inst == invalid_node) {
            timing.misses++;
        } else {
            timing.hits++;
        }
    }
}

// The instrumented walk, run once over the same chunks as the timed traversal
// (never timed: the counters are diagnostics, but its submissions count towards
// the reported worst submission all the same).
void run_instrumented(Stream &stream, SceneResources &resources, const TracePlan &plan,
                      BenchRayStats &stats, luisa::vector<LbvhHit> &host_hits,
                      double &worst_chunk_ms) noexcept {
    auto ray_count = plan.rays;
    for (auto slice = size_t{0u}; slice < plan.slices; slice++) {
        auto count = strided_count(ray_count, plan.slices, slice);
        if (count == 0u) { continue; }
        worst_chunk_ms = std::max(worst_chunk_ms,
                                  time_trace_dispatch(stream, resources, true, count, slice,
                                                      plan.slices));
    }
    luisa::vector<LbvhRayStats> host_stats(ray_count);
    stream << resources.ray_stats.view(0u, ray_count).copy_to(luisa::span{host_stats})
           << resources.reference_hits.view(0u, ray_count).copy_to(luisa::span{host_hits})
           << synchronize();
    stats = summarize_ray_stats(luisa::span{host_stats});
}

// Shape of every tree of the scene, aggregated: the *worst* depth and range of
// the scene (that is what the optimizer has to fix) plus the totals the means
// are taken over.
[[nodiscard]] BenchTreeStats measure_trees(Stream &stream, SceneResources &resources) noexcept {
    BenchTreeStats total;
    size_t depth_sum = 0u;
    double range_sum = 0.0;
    auto add_tree = [&](uint node_base, uint count) noexcept {
        auto tree = resources.bench_stats->measure_tree(stream, resources.lbvh->nodes(),
                                                        node_base, count);
        total.leaf_count += tree.leaf_count;
        total.internal_count += tree.internal_count;
        total.descent_failures += tree.descent_failures;
        total.depth_capped += tree.depth_capped;
        total.max_depth = std::max(total.max_depth, tree.max_depth);
        depth_sum += static_cast<size_t>(tree.mean_depth * static_cast<double>(tree.leaf_count));
        total.max_range = std::max(total.max_range, tree.max_range);
        range_sum += tree.mean_range * static_cast<double>(tree.internal_count);
    };
    for (auto &&blas : resources.blases) {
        add_tree(blas.node_offset(), blas.triangle_count());
    }
    add_tree(resources.tlas.node_offset(), resources.tlas.instance_count());
    if (total.leaf_count != 0u) {
        total.mean_depth = static_cast<double>(depth_sum) / static_cast<double>(total.leaf_count);
        total.depth_per_log2_n = total.leaf_count > 1u && total.max_depth > 0u ? static_cast<double>(total.max_depth) /
                                                                                     std::log2(static_cast<double>(total.leaf_count)) :
                                                                                 0.0;
    }
    if (total.internal_count != 0u) {
        total.mean_range = range_sum / static_cast<double>(total.internal_count);
    }
    return total;
}

// One ray buffer serves the whole measurement: the trace sweep re-dispatches a
// prefix of it, so the sweep never reallocates.
void generate_rays(Stream &stream, SceneResources &resources, const BenchScene &scene,
                   const BenchRayShader &shader, size_t ray_count, uint64_t seed) noexcept {
    auto mode = static_cast<uint>(scene.rays.mode);
    auto grid_w = static_cast<uint>(std::ceil(std::sqrt(static_cast<double>(ray_count))));
    stream << shader(resources.rays, static_cast<uint>(ray_count), grid_w,
                     static_cast<uint>(seed), mode, scene.rays.eye, scene.rays.forward,
                     scene.rays.right, scene.rays.up, scene.rays.half_w, scene.rays.half_h,
                     scene.rays.blob_center, scene.rays.blob_radius, scene.rays.axis)
                  .dispatch(ray_count)
           << synchronize();
}

// Plans the traversal of a scene, applying the `--max-seconds` reduction of
// `plan_trace` by *re-generating* the rays at the smaller size.  The reduced set
// is a smaller frustum with its own ray generator (the grid width of a frustum of
// `n` rays is `ceil(sqrt(n))`), not the prefix of the larger one: a prefix of a
// camera frustum is its top rows, which can miss the scene entirely - on
// `coincident` the first rows never touch the tree, so a prefix would report a
// 0.03 ms traversal of a scene whose full traversal costs 250 ms, with counters to
// match.  Re-generating keeps the reduced measurement a measurement of the scene.
// `note` describes a reduction that was actually performed, so the record says why
// fewer rays were traced than requested.
[[nodiscard]] TracePlan plan_traversal(Stream &stream, SceneResources &resources,
                                       const BenchOptions &options, const BenchScene &scene,
                                       const BenchRayShader &shader, size_t ray_count,
                                       luisa::string &note) noexcept {
    auto plan = plan_trace(stream, resources, options, false, ray_count);
    // the numbers of the *full* plan describe what the reduction avoided
    auto full_slices = plan.slices;
    auto full_slice_rays = plan.slice_rays;
    auto full_slice_ms = plan.validated_ms;
    for (auto attempt = 0u; attempt < 3u && plan.wants_smaller_ray_set(); attempt++) {
        generate_rays(stream, resources, scene, shader, plan.rays, options.seed);
        plan = plan_trace(stream, resources, options, false, plan.rays);
    }
    if (plan.rays < ray_count) {
        note = luisa::format("the traversal of {} rays needs {} strided slice(s) of {} rays "
                             "({:.0f} ms each, {:.1f} s in total), above the {} s limit: only {} "
                             "ray(s) are traced, re-generated as a smaller frustum (the reduced "
                             "size is the same on every backend, so the counters stay comparable)",
                             ray_count, full_slices, full_slice_rays, full_slice_ms,
                             full_slice_ms * static_cast<double>(full_slices) * 1.0e-3,
                             options.max_seconds, plan.rays);
    }
    // the record keeps the *requested* count, so a reduction is visible in it
    plan.requested = ray_count;
    // The warnings describe the plan that will actually be used.  They are emitted
    // here (and not by `plan_trace`) because only the caller knows which of the
    // plans is the final one: the re-plans of a reduced ray set would otherwise
    // repeat them once per attempt.
    auto accept_ms = std::max(options.dispatch_budget_ms, 1.0) * dispatch_accept_fraction;
    if (plan.validated_ms > accept_ms) {
        // even the smallest slice worth submitting is over the accept threshold, so
        // this scene cannot be traced in fewer submissions than the plan needs
        LUISA_WARNING("scene '{}': the smallest slice worth submitting ({} rays) already takes "
                      "{:.0f} ms, above the {:.0f} ms accept threshold derived from the {:.0f} ms "
                      "dispatch budget: this traversal cannot be traced in fewer submissions",
                      scene.name, plan.slice_rays, plan.validated_ms, accept_ms,
                      options.dispatch_budget_ms);
    }
    if (plan.slices > many_chunk_warning) {
        LUISA_WARNING("scene '{}': the traversal needs {} strided slice(s) of {} rays ({:.0f} ms "
                      "each), so trace_ms is the sum over the slices and measures the dispatch path "
                      "as much as the walk - read it with care, or raise --dispatch-budget-ms",
                      scene.name, plan.slices, plan.slice_rays, plan.validated_ms);
    }
    return plan;
}

// ---------------------------------------------------------------------------
// Validation
// ---------------------------------------------------------------------------

[[nodiscard]] auto make_rtx_trace_kernel() noexcept {
    // The demo's RTX reference: the same rays through the Luisa acceleration
    // structure, mapped into the same hit record.
    return Kernel1D{[](AccelVar accel, BufferVar<LbvhRay> rays, BufferVar<LbvhHit> hits,
                       UInt count) noexcept {
        set_block_size(64u);
        UInt index = dispatch_id().x;
        $if (index < count) {
            auto r = rays.read(index);
            auto ray = make_ray(r.origin, r.direction, r.t_min, r.t_max);
            auto hit = accel.intersect(ray, {});
            Var<LbvhHit> result;
            result.inst = invalid_node;
            result.prim = invalid_node;
            result.bary = make_float2(0.0f);
            result.t = r.t_max;
            $if (!hit->miss()) {
                result.inst = hit.inst;
                result.prim = hit.prim;
                result.bary = hit.bary;
                result.t = hit->distance();
            };
            hits.write(index, result);
        };
    }};
}

// Sine of the angle between a ray and the plane of the triangle it hits: 0 means
// the ray is exactly parallel to the plane, 1 means it is perpendicular.  The
// classification is redone on the host from the scene's own geometry, so it is a
// property of the geometry rather than an excuse for a mismatch.
[[nodiscard]] float grazing_sine(const BenchScene &scene, const LbvhRay &ray, uint instance,
                                 uint primitive) noexcept {
    if (instance >= scene.instances.size()) { return 1.0f; }
    auto mesh = scene.instances[instance].mesh;
    if (mesh >= scene.meshes.size()) { return 1.0f; }
    auto range = scene.meshes[mesh];
    if (primitive >= range.triangle_count) { return 1.0f; }
    auto triangle = scene.triangles[range.triangle_offset + primitive];
    if (triangle.i0 >= scene.vertices.size() || triangle.i1 >= scene.vertices.size() ||
        triangle.i2 >= scene.vertices.size()) {
        return 1.0f;
    }
    auto to_world = scene.instances[instance].to_world;
    auto transform = [&to_world](float3 p) noexcept {
        return (to_world * make_float4(p, 1.0f)).xyz();
    };
    auto v0 = transform(scene.vertices[triangle.i0]);
    auto v1 = transform(scene.vertices[triangle.i1]);
    auto v2 = transform(scene.vertices[triangle.i2]);
    auto normal = cross(v1 - v0, v2 - v0);
    auto twice_area = length(normal);
    if (twice_area <= 0.0f) { return 0.0f; }// a degenerate triangle cannot be compared
    return std::abs(dot(normal * (1.0f / twice_area), normalize(ray.direction)));
}

// A hit whose ray is almost parallel to the plane of the triangle it hits.  The
// software walk uses a float Moller-Trumbore and the hardware a (watertight,
// fixed point) intersecter: for a grazing ray the two legitimately disagree, and
// the intersection point itself is ill defined.
[[nodiscard]] bool is_grazing_hit(const BenchScene &scene, const LbvhRay &ray, uint instance,
                                  uint primitive) noexcept {
    // A hit within ~1.7 degrees of the plane of its triangle is not comparable
    // between a float and a watertight (fixed point) intersecter: the intersection
    // distance has a condition number of 1/sin(angle), so the two disagree on the
    // digits of t that a traversal tolerance would look at.  Measured from the
    // geometry, so a hit that is not grazing still has to agree exactly.
    constexpr auto grazing_sin_angle = 3.0e-2f;
    return grazing_sine(scene, ray, instance, primitive) < grazing_sin_angle;
}

// Structural self-check of every tree, then the demo's RTX cross-check (the same
// scene through `Mesh` + `Accel`, the same rays through both).  Returns false
// after logging what went wrong.
[[nodiscard]] bool validate_scene(Device &device, Stream &stream, const BenchScene &scene,
                                  SceneResources &resources, size_t ray_count,
                                  size_t &tree_problems, luisa::string &error) noexcept {
    // ---- structural self-check (validate_tree of every tree that is alive) ----
    tree_problems = 0u;
    for (auto &&blas : resources.blases) {
        tree_problems += resources.lbvh->validate_tree(stream, blas.node_offset(),
                                                       blas.triangle_count());
    }
    tree_problems += resources.lbvh->validate_tree(stream, resources.tlas.node_offset(),
                                                   resources.tlas.instance_count());
    LUISA_INFO("  self-check: {} BLAS + 1 TLAS, {} problem(s)", resources.blases.size(),
               tree_problems);
    if (tree_problems != 0u) {
        error = luisa::format("the software LBVH of scene '{}' is malformed", scene.name);
        return false;
    }
    // The bindless heap of the TLAS: the BLAS records and the TLAS region must
    // resolve through the heap to the very nodes the shared node buffer holds
    // (lbvh_common.h's "The bindless heap of a TLAS").  This is the ABI the
    // traversal runs on, so a wrong slot is reported here instead of silently
    // walking another tree.
    auto heap_problems = resources.lbvh->validate_heap(
        stream, resources.tlas, static_cast<uint>(resources.blases.size()));
    LUISA_INFO("  self-check: bindless heap ({} slot(s)), {} problem(s)",
               resources.tlas.heap_size(), heap_problems);
    if (heap_problems != 0u) {
        error = luisa::format("the bindless heap of the software LBVH of scene '{}' is malformed",
                              scene.name);
        return false;
    }
    // ---- the same scene through the Luisa RTX API ----
    luisa::vector<Mesh> meshes;
    meshes.reserve(scene.meshes.size());
    for (auto &&mesh : scene.meshes) {
        meshes.emplace_back(device.create_mesh(
            resources.vertices,
            resources.triangles.view(mesh.triangle_offset, mesh.triangle_count)));
    }
    Accel accel = device.create_accel();
    for (auto &&instance : scene.instances) {
        accel.emplace_back(meshes[instance.mesh], instance.to_world);
    }
    for (auto &&mesh : meshes) { stream << mesh.build(); }
    stream << accel.build() << synchronize();
    auto rtx_shader = device.compile(make_rtx_trace_kernel());
    luisa::vector<LbvhHit> host_software(ray_count);
    luisa::vector<LbvhHit> host_rtx(ray_count);
    // the download of the software hits must happen *before* the RTX trace
    // overwrites the buffer that holds them
    stream << resources.reference_hits.view(0u, ray_count)
                  .copy_to(luisa::span{host_software})
           << rtx_shader(accel, resources.rays, resources.reference_hits,
                         static_cast<uint>(ray_count))
                  .dispatch(ray_count)
           << synchronize();
    stream << resources.reference_hits.view(0u, ray_count).copy_to(luisa::span{host_rtx})
           << synchronize();
    // The tolerances are the demo's: the hardware traversal uses its own triangle
    // intersection, so distance and barycentrics agree only up to rounding.
    //
    // What the two traversals *must* agree on is what the ray hits: the hit/miss
    // classification, the closest distance (finding the closest hit is the
    // traversal's job, the intersection only refines it) and the instance /
    // primitive unless the distance agrees (several scenes overlap their own
    // primitives on purpose, and then the closest hit is not unique).  A
    // disagreement on a ray that grazes the triangle it hits, or that hits it on
    // its boundary, is reported separately and is not fatal: the intersecter,
    // not the traversal, cannot be compared there.
    constexpr float distance_tolerance = 1.0e-3f;
    constexpr float barycentric_tolerance = 5.0e-3f;
    luisa::vector<LbvhRay> host_rays(ray_count);
    stream << resources.rays.view(0u, ray_count).copy_to(luisa::span{host_rays})
           << synchronize();
    size_t miss_mismatch = 0u;
    size_t hit_id_mismatch = 0u;
    size_t distance_mismatch = 0u;
    size_t tie_mismatch = 0u;
    size_t grazing = 0u;
    size_t boundary = 0u;
    size_t bary_mismatch = 0u;
    // A hit that lands on (or within the barycentric tolerance of) the boundary
    // of its triangle.  Moller-Trumbore and the watertight intersecter of the
    // hardware use different edge rules there, so the two legitimately disagree
    // on such a hit; like `grazing`, this is measured from the hit itself and
    // not assumed.
    auto near_boundary = [](const LbvhHit &hit) noexcept {
        return std::min({hit.bary.x, hit.bary.y,
                         1.0f - hit.bary.x - hit.bary.y}) < barycentric_tolerance;
    };
    float max_distance_error = 0.0f;
    float max_barycentric_error = 0.0f;
    // The maxima of the rays that are *not* comparable (grazing / boundary hits)
    // are kept apart: mixing them into the reported maximum made a scene that
    // agrees perfectly (e.g. `uniform`, 0 distance mismatches) report a 22 %
    // maximum distance error.
    float max_excluded_distance_error = 0.0f;
    float max_excluded_barycentric_error = 0.0f;
    size_t compared = 0u;
    auto verbose = std::getenv("LUISA_BENCH_VALIDATE_VERBOSE") != nullptr;
    auto report_disagreement = [&](size_t index, const LbvhHit &a, const LbvhHit &b) noexcept {
        if (!verbose) { return; }
        auto ray = host_rays[index];
        std::printf("bench_mismatch ray=%llu a_inst=%u a_prim=%u a_t=%.9g a_sin=%.3e b_inst=%u b_prim=%u b_t=%.9g b_sin=%.3e\n",
                    static_cast<unsigned long long>(index), a.inst, a.prim,
                    static_cast<double>(a.t),
                    static_cast<double>(grazing_sine(scene, ray, a.inst, a.prim)),
                    b.inst, b.prim, static_cast<double>(b.t),
                    static_cast<double>(grazing_sine(scene, ray, b.inst, b.prim)));
    };
    for (auto i = 0u; i < ray_count; i++) {
        auto a = host_software[i];
        auto b = host_rtx[i];
        auto a_miss = a.inst == invalid_node;
        auto b_miss = b.inst == invalid_node;
        auto ray = host_rays[i];
        if (a_miss != b_miss) {
            // the ray grazes the triangle that one of the two hit, or hits it
            // on its boundary: there is no second hit to compare it with
            auto &&hit = a_miss ? b : a;
            if (is_grazing_hit(scene, ray, hit.inst, hit.prim)) {
                grazing++;
            } else if (near_boundary(hit)) {
                boundary++;
            } else {
                miss_mismatch++;
                report_disagreement(i, a, b);
            }
            continue;
        }
        if (a_miss) { continue; }
        auto distance_error = std::abs(a.t - b.t) / std::max(1.0f, std::abs(b.t));
        auto barycentric_error = std::max(std::abs(a.bary.x - b.bary.x),
                                          std::abs(a.bary.y - b.bary.y));
        auto distance_ok = distance_error <= distance_tolerance;
        auto barycentric_ok = barycentric_error <= barycentric_tolerance;
        auto same_hit = a.inst == b.inst && a.prim == b.prim;
        auto excluded = is_grazing_hit(scene, ray, a.inst, a.prim) ||
                        is_grazing_hit(scene, ray, b.inst, b.prim) ||
                        near_boundary(a) || near_boundary(b);
        if (excluded) {
            // not comparable: reported, with its own maxima, and never fatal
            if (is_grazing_hit(scene, ray, a.inst, a.prim) ||
                is_grazing_hit(scene, ray, b.inst, b.prim)) {
                grazing++;
            } else {
                boundary++;
            }
            max_excluded_distance_error = std::max(max_excluded_distance_error, distance_error);
            max_excluded_barycentric_error = std::max(max_excluded_barycentric_error,
                                                      barycentric_error);
            continue;
        }
        if (!same_hit) {
            // A different primitive at the same distance: the scenes overlap their
            // own primitives on purpose, so the closest hit is not unique and
            // neither the distance nor the barycentrics are comparable.
            if (distance_ok) {
                tie_mismatch++;
            } else {
                hit_id_mismatch++;
                compared++;
                max_distance_error = std::max(max_distance_error, distance_error);
                report_disagreement(i, a, b);
            }
            continue;
        }
        // same instance and primitive: distance *and* barycentrics are comparable
        compared++;
        max_distance_error = std::max(max_distance_error, distance_error);
        max_barycentric_error = std::max(max_barycentric_error, barycentric_error);
        if (!distance_ok) { distance_mismatch++; }
        // A barycentric mismatch is only counted for a hit the two traversals
        // agree on: the barycentrics of two *different* coincident triangles
        // obviously differ, and that is already reported as a tie.
        if (!barycentric_ok) { bary_mismatch++; }
    }
    LUISA_INFO("  RTX cross-check: hit/miss mismatch {}, closest-hit distance {}, instance/primitive {}, same-distance ties {}, grazing {}, boundary {}, barycentric {}",
               miss_mismatch, distance_mismatch, hit_id_mismatch, tie_mismatch, grazing,
               boundary, bary_mismatch);
    LUISA_INFO("  RTX cross-check ({} comparable hit(s)): max distance error {:.3e}, max barycentric error {:.3e}; excluded rays (grazing/boundary): max distance error {:.3e}, max barycentric error {:.3e}",
               compared, max_distance_error, max_barycentric_error,
               max_excluded_distance_error, max_excluded_barycentric_error);
    if (miss_mismatch != 0u || distance_mismatch != 0u || hit_id_mismatch != 0u) {
        error = luisa::format("the software LBVH traversal of scene '{}' does not match the Luisa RTX reference",
                              scene.name);
        return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// Scene driver
// ---------------------------------------------------------------------------

// Effective sizes of a scene: the CLI overrides win over the scene defaults.
[[nodiscard]] size_t effective_rays(const BenchOptions &options,
                                    const BenchScene &scene) noexcept {
    return options.rays == 0u ? scene.default_rays : options.rays;
}

// Materializes one scene, either from the catalogue or from an OBJ file.
[[nodiscard]] bool make_scene(const BenchOptions &options, const char *name,
                              BenchScene &scene, luisa::string &error) noexcept {
    if (options.mesh_path.empty()) {
        return make_bench_scene(name, options.triangles, options.instances, options.rays,
                                options.seed, scene, error);
    }
    return make_obj_scene(options.mesh_path.c_str(), options.instances, options.rays,
                          options.seed, scene, error);
}

void append_skip_run(luisa::vector<BenchRun> &results, const BenchScene &scene,
                     const char *step, size_t triangles, size_t instances, size_t rays,
                     bool validate, const luisa::string &reason) noexcept {
    BenchRun skip;
    skip.scene = scene.name;
    skip.step = step;
    skip.skipped = true;
    skip.skip_reason = reason;
    skip.triangles = triangles;
    skip.instances = instances;
    skip.rays = rays;
    skip.memory = estimate_bench_memory(triangles, instances, scene.meshes.size(),
                                        scene.vertices.size(), rays, validate);
    results.emplace_back(std::move(skip));
}

// `--stress-build`: the build-only measurement for N = 1<<10, 1<<12, ... up to
// `--max-triangles`, plus the empirical scaling exponent `p` of `log(ms)` vs
// `log(N)` per stage (that is what distinguishes O(N) from O(N log N) from
// worse).  It stops early - with a warning - when the next step would exceed the
// budget or when a step already took longer than `--max-seconds`.
void sweep_build(Device &device, Stream &stream, const BenchOptions &options, const char *name,
                 size_t instances, size_t rays_for_estimate, uint64_t seed,
                 luisa::vector<BenchRun> &results, size_t budget_bytes,
                 bool validate) noexcept {
    luisa::vector<double> sizes;
    luisa::vector<double> totals;
    luisa::vector<double> stages[4];
    auto sweep_options = options;
    // The build cannot be chunked, so a step whose largest submission is
    // predicted to exceed the dispatch budget is skipped instead of submitted:
    // the prediction extrapolates the previous step's largest submission with a
    // conservative growth factor (2.5x per doubling at least).
    auto previous_size = 0.0;
    auto previous_worst_ms = 0.0;
    auto previous_previous_worst_ms = 0.0;
    sweep_options.iterations = std::min<uint32_t>(options.iterations, 3u);
    sweep_options.warmup = 0u;
    std::printf("-- build sweep of '%s' (build only, %u iteration(s) per step) --\n",
                name, sweep_options.iterations);
    for (auto n = size_t{1u} << 10u; n <= options.max_triangles; n <<= 1u) {
        BenchScene scene;
        luisa::string error;
        if (!make_bench_scene(name, n, instances, 0u, seed, scene, error)) {
            LUISA_WARNING("build sweep of '{}' stopped at N={}: {}", name, n, error);
            break;
        }
        auto primitives = scene.triangles.size() + scene.instances.size();
        auto estimate = estimate_bench_memory(scene.triangles.size(), scene.instances.size(),
                                              scene.meshes.size(), scene.vertices.size(),
                                              rays_for_estimate, validate);
        auto step = luisa::format("build n={}", n);
        if (previous_worst_ms > 0.0) {
            auto predicted = predict_stage_ms(previous_worst_ms, previous_previous_worst_ms,
                                              previous_size, static_cast<double>(n));
            if (predicted > options.dispatch_budget_ms) {
                LUISA_WARNING("build sweep of '{}' stopped at N={}: one build submission is "
                              "predicted at {:.0f} ms ({:.1f} ms measured at N={}), above the "
                              "{:.0f} ms dispatch budget - a longer submission removes the device",
                              name, n, predicted, previous_worst_ms,
                              static_cast<size_t>(previous_size), options.dispatch_budget_ms);
                break;
            }
        }
        if (estimate.total_bytes() > budget_bytes) {
            LUISA_WARNING("build sweep of '{}' stopped at N={}: {} needed but only {} left "
                          "of the budget",
                          name, n, human_bytes(estimate.total_bytes()),
                          human_bytes(budget_bytes));
            append_skip_run(results, scene, step.c_str(), scene.triangles.size(),
                            scene.instances.size(), rays_for_estimate, validate,
                            "over budget");
            break;
        }
        SceneResources resources{device, stream, scene, 0u, false};
        BenchRun run;
        run.scene = scene.name;
        run.step = step;
        run.measured = true;
        run.triangles = scene.triangles.size();
        run.vertices = scene.vertices.size();
        run.instances = scene.instances.size();
        run.blas_count = scene.meshes.size();
        run.primitives = primitives;
        run.rays = 0u;
        run.seed = seed;
        run.memory = estimate;
        run.actual_bytes = resources.actual_bytes;
        run.build.primitives = primitives;
        run.build.nodes = resources.lbvh->node_count();
        run.dispatch_budget_ms = options.dispatch_budget_ms;
        measure_build(stream, resources, sweep_options, run.build,
                      run.worst_build_dispatch_ms);
        print_human_summary(run);
        print_machine_readable(run);
        previous_previous_worst_ms = previous_worst_ms;
        previous_worst_ms = run.worst_build_dispatch_ms;
        previous_size = static_cast<double>(n);
        if (previous_worst_ms > options.dispatch_budget_ms) {
            LUISA_WARNING("build sweep of '{}' stopped after N={}: one submission took "
                          "{:.1f} ms, above the {:.0f} ms dispatch budget (the prediction "
                          "was optimistic)",
                          name, n, previous_worst_ms, options.dispatch_budget_ms);
            results.emplace_back(std::move(run));
            break;
        }
        sizes.emplace_back(static_cast<double>(n));
        totals.emplace_back(run.build.total_ms.min_ms());
        stages[0].emplace_back(run.build.prim_ms.min_ms());
        stages[1].emplace_back(run.build.morton_ms.min_ms());
        stages[2].emplace_back(run.build.sort_ms.min_ms());
        stages[3].emplace_back(run.build.node_ms.min_ms());
        auto too_slow = run.build.total_ms.min_ms() > options.max_seconds * 1.0e3;
        results.emplace_back(std::move(run));
        if (too_slow) {
            LUISA_WARNING("build sweep of '{}' stopped after N={}: the step took more than {} s",
                          name, n, options.max_seconds);
            break;
        }
    }
    if (sizes.size() >= 2u) {
        auto span_of = [](const luisa::vector<double> &values) noexcept {
            return luisa::span{values.data(), values.size()};
        };
        auto fit = fit_log_log(span_of(sizes), span_of(totals));
        std::printf("bench_scaling scene=%s stage=total exponent=%.4f r2=%.4f points=%llu\n",
                    name, fit.exponent, fit.r2,
                    static_cast<unsigned long long>(fit.count));
        LUISA_INFO("scaling of '{}': {}", name, fit.to_string("total"));
        for (auto i = 0; i < 4; i++) {
            static constexpr const char *stage_names[] = {"prim", "morton", "sort", "node"};
            auto stage_fit = fit_log_log(span_of(sizes), span_of(stages[i]));
            std::printf("bench_scaling scene=%s stage=%s exponent=%.4f r2=%.4f points=%llu\n",
                        name, stage_names[i], stage_fit.exponent, stage_fit.r2,
                        static_cast<unsigned long long>(stage_fit.count));
            LUISA_INFO("  {}", stage_fit.to_string(stage_names[i]));
        }
    }
}

// `--stress-traversal`: the traversal measurement for
// rays = 1<<12 ... the scene's ray count.  The tree is built once and the ray
// count is swept with the same buffers, so the sweep is cheap; the exponent is
// expected to be ~1.0, anything else is a measurement artefact worth reporting.
void sweep_traversal(Device &device, Stream &stream, const BenchOptions &options,
                     const BenchRayShader &ray_shader, BenchScene &scene, size_t ray_count,
                     size_t budget_bytes, bool validate,
                     luisa::vector<BenchRun> &results) noexcept {
    auto estimate = estimate_bench_memory(scene.triangles.size(), scene.instances.size(),
                                          scene.meshes.size(), scene.vertices.size(),
                                          ray_count, validate, options.compact);
    if (estimate.total_bytes() > budget_bytes) {
        LUISA_WARNING("traversal sweep of '{}' skipped: {} needed", scene.name,
                      human_bytes(estimate.total_bytes()));
        return;
    }
    auto ray_resources = luisa::make_unique<SceneResources>(device, stream, scene, ray_count,
                                                            true);
    generate_rays(stream, *ray_resources, scene, ray_shader, ray_count, options.seed);
    // the tree does not change during the sweep: it is built once, outside the
    // loop, so every step measures the traversal and nothing else
    build_scene(stream, *ray_resources, nullptr, nullptr);
    auto sweep_options = options;
    sweep_options.iterations = std::min<uint32_t>(options.iterations, 3u);
    sweep_options.warmup = 1u;
    std::printf("-- traversal sweep of '%s' (%u iteration(s) per step) --\n",
                scene.name, sweep_options.iterations);
    luisa::vector<double> sizes;
    luisa::vector<double> times;
    luisa::string plan_note;
    for (auto rays = size_t{1u} << 12u; rays <= ray_count; rays <<= 1u) {
        auto step = luisa::format("trace rays={}", rays);
        BenchRun run;
        run.scene = scene.name;
        run.step = step;
        run.measured = true;
        run.triangles = scene.triangles.size();
        run.vertices = scene.vertices.size();
        run.instances = scene.instances.size();
        run.blas_count = scene.meshes.size();
        run.primitives = scene.triangles.size() + scene.instances.size();
        run.rays = rays;
        run.seed = options.seed;
        run.memory = estimate;
        run.actual_bytes = ray_resources->actual_bytes;
        run.build.primitives = run.primitives;
        run.build.nodes = ray_resources->lbvh->node_count();
        // The traversal is chunked by the same plan as the main measurement: a
        // step of the sweep can never submit a too-long dispatch either - and the
        // rays are re-generated for the step, so every step is a frustum of the
        // requested size instead of a prefix of the scene's own ray set (a prefix
        // is systematically cheaper and its exponent would be an artefact).
        generate_rays(stream, *ray_resources, scene, ray_shader, rays, options.seed);
        auto plan = plan_traversal(stream, *ray_resources, sweep_options, scene, ray_shader, rays,
                                   plan_note);
        if (!plan_note.empty()) {
            LUISA_WARNING("traversal sweep of '{}' at {} rays: {}", scene.name, rays, plan_note);
            plan_note.clear();
        }
        run.rays = plan.rays;
        run.rays_requested = plan.requested;
        run.trace_chunks = plan.chunks();
        run.trace_chunk_rays = plan.slice_rays;
        run.trace_per_ray_ns = plan.per_ray_ns;
        run.trace_validated_ms = plan.validated_ms;
        run.dispatch_budget_ms = options.dispatch_budget_ms;
        luisa::vector<LbvhHit> host_hits;
        size_t repeat_mismatches = 0u;
        measure_trace(stream, *ray_resources, sweep_options, plan, run.trace, false,
                      repeat_mismatches, host_hits, run.worst_trace_chunk_ms);
        auto instrumented_plan = plan_trace(stream, *ray_resources, sweep_options, true,
                                            plan.rays, false);
        run_instrumented(stream, *ray_resources, instrumented_plan, run.ray_stats, host_hits,
                         run.worst_trace_chunk_ms);
        print_human_summary(run);
        print_machine_readable(run);
        sizes.emplace_back(static_cast<double>(run.rays));
        times.emplace_back(run.trace.total_ms.min_ms());
        auto too_slow = run.trace.total_ms.min_ms() > options.max_seconds * 1.0e3 ||
                        run.worst_trace_chunk_ms > options.dispatch_budget_ms;
        results.emplace_back(std::move(run));
        if (too_slow) {
            LUISA_WARNING("traversal sweep of '{}' stopped after {} rays: the step took more than {} s",
                          scene.name, rays, options.max_seconds);
            break;
        }
    }
    if (sizes.size() >= 2u) {
        auto fit = fit_log_log(luisa::span{sizes.data(), sizes.size()},
                               luisa::span{times.data(), times.size()});
        std::printf("bench_scaling scene=%s stage=trace exponent=%.4f r2=%.4f points=%llu\n",
                    scene.name, fit.exponent, fit.r2,
                    static_cast<unsigned long long>(fit.count));
        LUISA_INFO("scaling of '{}': {}", scene.name, fit.to_string("trace"));
    }
}

// ---------------------------------------------------------------------------
// Pre-flight of an oversized build
// ---------------------------------------------------------------------------

// The build of one scene is a fixed sequence of submissions that cannot be split
// (the `node` stage alone is 92-96 % of it), so a request that is predicted to
// exceed the dispatch budget has to be refused *before* it is submitted.  The
// prediction comes from a probe: the same scene at a smaller size that is known
// to fit (its own default size, or a bounded prefix of an asset), built once with
// the per-tree timing hook.
struct BuildPreflight {
    bool probed{false};
    double probe_triangles{0.0};
    double probe_worst_dispatch_ms{0.0};
    double predicted_dispatch_ms{0.0};
    double safe_triangles{0.0};
};

[[nodiscard]] BuildPreflight preflight_build(Device &device, Stream &stream,
                                             const BenchOptions &options, const char *name,
                                             const BenchScene &requested, size_t probe_triangles,
                                             size_t requested_triangles) noexcept {
    BuildPreflight result;
    auto budget_bytes = static_cast<size_t>(options.budget_gib * 1024.0 * 1024.0 * 1024.0);
    auto instances = requested.instances.size();
    BenchScene probe;
    luisa::string error;
    auto ok = options.mesh_path.empty() ? make_bench_scene(name, probe_triangles, instances, 0u, options.seed, probe, error) : make_obj_scene(options.mesh_path.c_str(), instances, 0u, options.seed, probe, error, probe_triangles);
    if (!ok) { return result; }
    auto estimate = estimate_bench_memory(probe.triangles.size(), probe.instances.size(),
                                          probe.meshes.size(), probe.vertices.size(), 0u, false);
    if (estimate.total_bytes() > budget_bytes) { return result; }
    {
        SceneResources resources{device, stream, probe, 0u, false};
        LbvhBuildTimings stages;
        BenchWorstDispatch worst;
        build_scene(stream, resources, &stages, &worst);
        result.probed = true;
        result.probe_triangles = static_cast<double>(probe.triangles.size());
        result.probe_worst_dispatch_ms = worst.value_ms();
    }
    auto requested_size = static_cast<double>(requested_triangles);
    result.predicted_dispatch_ms =
        result.probe_worst_dispatch_ms *
        predict_growth(result.probe_triangles, requested_size) * dispatch_safety;
    // invert the model for the largest size that still fits the budget
    constexpr auto growth_exponent = 1.3219280948873623;// log2(2.5): the "2.5 per doubling" floor
    auto affordable = options.dispatch_budget_ms /
                      std::max(result.probe_worst_dispatch_ms * dispatch_safety, 1.0e-9);
    result.safe_triangles = result.probe_triangles *
                            std::pow(std::max(affordable, 1.0), 1.0 / growth_exponent);
    return result;
}

// Outcome of one scene.  That the scene itself cannot be prepared (a missing
// asset, an unusable request) is a recoverable error the driver reports and exits
// on, while a malformed tree or a mismatch with the RTX reference is an
// unconditional failure (see main).  A budget skip is neither: it is a reported
// result.
enum struct RunStatus {
    OK,
    SCENE_ERROR,
    MISMATCH,
};

// The full measurement of one scene.
[[nodiscard]] RunStatus run_scene(Device &device, Stream &stream, const BenchOptions &options,
                                  const BenchRayShader &ray_shader, const char *name,
                                  luisa::vector<BenchRun> &results, bool validate,
                                  bool repeat_check, luisa::string &error) noexcept {
    auto budget_bytes = static_cast<size_t>(options.budget_gib * 1024.0 * 1024.0 * 1024.0);
    BenchScene scene;
    if (!make_scene(options, name, scene, error)) { return RunStatus::SCENE_ERROR; }
    auto ray_count = effective_rays(options, scene);
    auto primitives = scene.triangles.size() + scene.instances.size();
    // the storage is over-sized only for a compaction run, and the estimate must
    // see exactly the storage `SceneResources` will allocate (see --headroom)
    auto storage_headroom = options.compact ? options.headroom : 1.0;
    auto estimate = estimate_bench_memory(scene.triangles.size(), scene.instances.size(),
                                          scene.meshes.size(), scene.vertices.size(),
                                          ray_count, validate, options.compact,
                                          storage_headroom);
    // ---- the budget gate, before anything is allocated ----
    if (estimate.total_bytes() > budget_bytes) {
        LUISA_WARNING("scene '{}' needs {} but the budget is only {} - skipping it "
                      "(raise --budget-gib to run it)",
                      name, human_bytes(estimate.total_bytes()), human_bytes(budget_bytes));
        append_skip_run(results, scene, "", scene.triangles.size(), scene.instances.size(),
                        ray_count, validate, "over budget");
        std::printf("bench_skip scene=%s step=- estimated_bytes=%llu budget_bytes=%llu\n",
                    name, static_cast<unsigned long long>(estimate.total_bytes()),
                    static_cast<unsigned long long>(budget_bytes));
        return RunStatus::OK;
    }
    // ---- the dispatch-budget gate: the build cannot be chunked ----
    // The scene is measured once at its *probe* size (its default size for a
    // catalogue scene, a bounded prefix of the asset for an OBJ) and the largest
    // single submission is extrapolated to the requested size.  A prediction over
    // the budget is refused: a submission of seconds removes the device, and it
    // cannot be aborted from the host once it is submitted.
    auto requested_triangles = scene.triangles.size();
    auto probe_triangles = std::min(requested_triangles,
                                    options.mesh_path.empty() ? std::max<size_t>(scene.default_triangles, 1u) : std::min(requested_triangles, size_t{1u} << 20u));
    if (probe_triangles < requested_triangles && !options.force_oversize) {
        auto preflight = preflight_build(device, stream, options, name, scene, probe_triangles,
                                         requested_triangles);
        if (preflight.probed && preflight.predicted_dispatch_ms > options.dispatch_budget_ms) {
            LUISA_WARNING("scene '{}' at {} triangles would need a {:.0f} ms build submission "
                          "({:.1f} ms measured at {} triangles, extrapolated conservatively); "
                          "the dispatch budget is {:.0f} ms and one submission of seconds "
                          "removes the device - skipping the scene.  The predicted-safe size "
                          "is {} triangles (--force-oversize overrides this and may remove "
                          "the device)",
                          name, requested_triangles, preflight.predicted_dispatch_ms,
                          preflight.probe_worst_dispatch_ms,
                          static_cast<size_t>(preflight.probe_triangles),
                          options.dispatch_budget_ms,
                          static_cast<size_t>(preflight.safe_triangles));
            append_skip_run(results, scene, "", requested_triangles, scene.instances.size(),
                            ray_count, validate, "predicted over the dispatch budget");
            std::printf("bench_oversize scene=%s predicted_dispatch_ms=%.3f budget_ms=%.3f "
                        "safe_triangles=%llu force_oversize=%d\n",
                        name, preflight.predicted_dispatch_ms, options.dispatch_budget_ms,
                        static_cast<unsigned long long>(preflight.safe_triangles),
                        options.force_oversize ? 1 : 0);
            return RunStatus::OK;
        }
        if (preflight.probed) {
            LUISA_INFO("  pre-flight: {:.1f} ms largest build submission at {} triangles -> "
                       "predicted {:.1f} ms at {} (budget {:.0f} ms)",
                       preflight.probe_worst_dispatch_ms,
                       static_cast<size_t>(preflight.probe_triangles),
                       preflight.predicted_dispatch_ms, requested_triangles,
                       options.dispatch_budget_ms);
        }
    } else if (probe_triangles < requested_triangles) {
        LUISA_WARNING("--force-oversize: submitting {} triangles without the pre-flight; a "
                      "single build submission of seconds may remove the device",
                      requested_triangles);
    }
    LUISA_INFO("'{}': {} ({} triangles, {} instances, {} BLAS, {} rays, seed {}), {} estimated",
               name, scene.worst_case, scene.triangles.size(), scene.instances.size(),
               scene.meshes.size(), ray_count, options.seed,
               human_bytes(estimate.total_bytes()));
    BenchRun run;
    run.scene = scene.name;
    run.measured = true;
    run.triangles = scene.triangles.size();
    run.vertices = scene.vertices.size();
    run.instances = scene.instances.size();
    run.blas_count = scene.meshes.size();
    run.primitives = primitives;
    run.rays = ray_count;
    run.seed = options.seed;
    run.memory = estimate;
    bool stress_build = options.stress_build && options.mesh_path.empty();
    bool stress_traversal = options.stress_traversal;
    if (options.stress_build && !options.mesh_path.empty()) {
        LUISA_WARNING("--stress-build is not available for an OBJ scene (its triangle count "
                      "is the asset's); the mesh is only measured as it is");
    }
    run.dispatch_budget_ms = options.dispatch_budget_ms;
    {
        auto resources = luisa::make_unique<SceneResources>(
            device, stream, scene, ray_count, true, storage_headroom, options.compact);
        generate_rays(stream, *resources, scene, ray_shader, ray_count, options.seed);
        run.actual_bytes = resources->actual_bytes;
        run.build.primitives = primitives;
        run.build.blas_count = scene.meshes.size();
        run.build.instance_count = scene.instances.size();
        run.build.nodes = resources->lbvh->node_count();
        measure_build(stream, *resources, options, run.build, run.worst_build_dispatch_ms);
        run.tree = measure_trees(stream, *resources);
        // The traversal is planned by *measuring* successively larger slices so
        // that no submission is known to exceed the dispatch budget; a scene whose
        // whole traversal would exceed --max-seconds is reduced here (a warning).
        luisa::string plan_note;
        auto plan = plan_traversal(stream, *resources, options, scene, ray_shader, ray_count,
                                   plan_note);
        if (!plan_note.empty()) {
            LUISA_WARNING("scene '{}': {}", name, plan_note);
        }
        if (plan.chunks() > 1u) {
            LUISA_INFO("  traversal: {} rays in {} strided slice(s) of {} rays ({:.1f} ns/ray, "
                       "{} ms measured) - the slice size is the largest whose measured cost fits "
                       "the {:.0f} ms dispatch budget, and the slice times are summed into "
                       "trace_ms",
                       plan.rays, plan.chunks(), plan.slice_rays, plan.per_ray_ns,
                       plan.validated_ms, options.dispatch_budget_ms);
        }
        run.rays = plan.rays;
        run.rays_requested = plan.requested;
        run.trace_chunks = plan.chunks();
        run.trace_chunk_rays = plan.slice_rays;
        run.trace_per_ray_ns = plan.per_ray_ns;
        run.trace_validated_ms = plan.validated_ms;
        // ---- storage compaction (`--compact`), after the plan was measured on
        // the loose structure: trace the loose one with the same plan, compact it
        // (device size query -> readback + synchronise -> exact-size dense buffer
        // -> copy -> heap re-registration -> callback retirement), then trace the
        // dense one below.  The two traversals must agree bit-for-bit; the memory
        // the compaction reclaims is `(capacity - used) * 32` bytes.
        luisa::vector<LbvhHit> host_loose;
        if (options.compact) {
            size_t loose_repeat_mismatches = 0u;
            measure_trace(stream, *resources, options, plan, run.trace_loose, false,
                          loose_repeat_mismatches, host_loose, run.worst_trace_chunk_ms);
            auto policy = SoftwareLbvh::CompactionPolicy::as_built;
            Clock compact_clock;
            compact_clock.tic();
            auto result = resources->lbvh->compact(stream, resources->tlas,
                                                   luisa::span{resources->blases}, policy,
                                                   options.release_scratch);
            stream << synchronize();
            run.compact_ms = compact_clock.toc();
            run.compacted = result.compacted();
            run.compact_nodes_before = result.nodes_before;
            run.compact_nodes_after = result.nodes_after;
            run.compact_reclaimed_bytes = result.compacted_bytes;
            run.compact_reclaimed_scratch_bytes = result.reclaimed_scratch_bytes;
            LUISA_INFO("  compaction (as-built): {} -> {} node(s), {} reclaimed, "
                       "{} scratch reclaimed, {:.3f} ms",
                       result.nodes_before, result.nodes_after,
                       human_bytes(result.compacted_bytes),
                       human_bytes(result.reclaimed_scratch_bytes), run.compact_ms);
        }
        luisa::vector<LbvhHit> host_hits;
        size_t repeat_mismatches = 0u;
        measure_trace(stream, *resources, options, plan, run.trace, repeat_check,
                      repeat_mismatches, host_hits, run.worst_trace_chunk_ms);
        if (options.compact) {
            // The hits of the loose and the dense structure must be identical: the
            // same promise `allow_compaction` makes at the runtime level (the
            // option is a hint, never a semantic change).
            auto comparable = std::min(host_loose.size(), host_hits.size());
            for (auto i = 0u; i < comparable; i++) {
                auto &&a = host_loose[i];
                auto &&b = host_hits[i];
                if (a.inst != b.inst || a.prim != b.prim || a.t != b.t ||
                    a.bary.x != b.bary.x || a.bary.y != b.bary.y) {
                    run.compact_hit_mismatches++;
                }
            }
            if (host_loose.size() != host_hits.size()) { run.compact_hit_mismatches++; }
            if (run.compact_hit_mismatches != 0u) {
                error = luisa::format("scene '{}': compaction changed {} of {} hit(s)",
                                      name, run.compact_hit_mismatches, comparable);
                return RunStatus::MISMATCH;
            }
        }
        // A scene whose own measurement already blew through --max-seconds must
        // not be swept: on a debug/ASan run that would look like a hang.
        auto too_slow = run.build.total_ms.min_ms() > options.max_seconds * 1.0e3 ||
                        run.trace.total_ms.min_ms() > options.max_seconds * 1.0e3;
        if (too_slow) {
            LUISA_WARNING("scene '{}' measured {:.1f} ms build / {:.1f} ms trace (> {} s): "
                          "the sweeps of this scene are skipped",
                          name, run.build.total_ms.min_ms(), run.trace.total_ms.min_ms(),
                          options.max_seconds);
            stress_build = false;
            stress_traversal = false;
        }
        // The instrumented walk is a different (larger) kernel, so it gets its own
        // plan: the same ray range can need a different chunk size there.
        auto instrumented_plan = plan_trace(stream, *resources, options, true, run.rays, false);
        luisa::vector<LbvhHit> host_instrumented(run.rays);
        run_instrumented(stream, *resources, instrumented_plan, run.ray_stats,
                         host_instrumented, run.worst_trace_chunk_ms);
        // The instrumented walk is a copy of the library walk with counters, so it
        // must reach the same hits.  "The same" tolerates a different primitive at
        // the same distance: the two kernels are compiled separately, and where a
        // scene creates exact ties (thousands of exactly coincident triangles in
        // `grid-duplicates` / `coincident`) the winner of a tie can differ by one
        // ULP of the intersection distance.  The hit/miss classification and the
        // closest distance may not differ beyond that.
        size_t walk_miss_mismatch = 0u;
        size_t walk_distance_mismatch = 0u;
        constexpr auto walk_distance_tolerance = 1.0e-4f;
        for (auto i = 0u; i < run.rays; i++) {
            auto a = host_hits[i];
            auto b = host_instrumented[i];
            auto a_miss = a.inst == invalid_node;
            auto b_miss = b.inst == invalid_node;
            if (a_miss != b_miss) {
                walk_miss_mismatch++;
                continue;
            }
            if (a_miss) { continue; }
            auto distance_error = std::abs(a.t - b.t) / std::max(1.0f, std::abs(b.t));
            if (distance_error > walk_distance_tolerance) {
                walk_distance_mismatch++;
            } else if (a.inst != b.inst || a.prim != b.prim) {
                run.walk_ties++;
            }
        }
        if (walk_miss_mismatch != 0u || walk_distance_mismatch != 0u) {
            error = luisa::format("the instrumented walk of scene '{}' disagrees with the library traversal "
                                  "({} hit/miss, {} distance)",
                                  name, walk_miss_mismatch, walk_distance_mismatch);
            return RunStatus::MISMATCH;
        }
        if (repeat_check && repeat_mismatches != 0u) {
            error = luisa::format("scene '{}': {} of the {} timed traversals were not bit-identical",
                                  name, repeat_mismatches, options.iterations);
            return RunStatus::MISMATCH;
        }
        if (validate) {
            if (!validate_scene(device, stream, scene, *resources, run.rays,
                                run.tree_problems, error)) {
                return RunStatus::MISMATCH;
            }
        }
    }
    // the safety claim of the scene, checked against the budget it was run with
    if (run.worst_build_dispatch_ms > options.dispatch_budget_ms) {
        LUISA_WARNING("scene '{}': one build submission took {:.1f} ms, above the {:.0f} ms "
                      "dispatch budget (the prediction was optimistic)",
                      name, run.worst_build_dispatch_ms, options.dispatch_budget_ms);
    }
    if (run.worst_trace_chunk_ms > options.dispatch_budget_ms) {
        LUISA_WARNING("scene '{}': one traversal submission took {:.1f} ms, above the {:.0f} ms "
                      "dispatch budget",
                      name, run.worst_trace_chunk_ms, options.dispatch_budget_ms);
    }
    print_human_summary(run);
    print_machine_readable(run);
    auto effective_rays_for_sweep = run.rays;
    results.emplace_back(std::move(run));
    // ---- the sweeps (a scene that was already too slow does not sweep) ----
    if (stress_build) {
        sweep_build(device, stream, options, name, scene.instances.size(),
                    effective_rays_for_sweep, options.seed, results, budget_bytes, validate);
    }
    if (stress_traversal) {
        sweep_traversal(device, stream, options, ray_shader, scene, effective_rays_for_sweep,
                        budget_bytes, validate, results);
    }
    return RunStatus::OK;
}

// ---------------------------------------------------------------------------
// Final report
// ---------------------------------------------------------------------------

// Worst-case ranking: worst build first, then worst traversal first, both
// normalized against the `uniform` baseline (per primitive and per ray, which is
// the ratio at equal sizes and the only meaningful one otherwise).
void print_ranking(luisa::span<const BenchRun> runs) noexcept {
    luisa::vector<const BenchRun *> measured;
    for (auto &&run : runs) {
        if (run.measured && run.step.empty()) { measured.emplace_back(&run); }
    }
    const BenchRun *baseline = nullptr;
    for (auto *run : measured) {
        if (run->scene == "uniform") { baseline = run; }
    }
    auto build_ratio = [baseline](const BenchRun &run) noexcept {
        if (baseline == nullptr || baseline->primitives == 0u || run.primitives == 0u ||
            baseline->build.total_ms.min_ms() <= 0.0) {
            return 0.0;
        }
        return (run.build.total_ms.min_ms() / static_cast<double>(run.primitives)) /
               (baseline->build.total_ms.min_ms() / static_cast<double>(baseline->primitives));
    };
    auto trace_ratio = [baseline](const BenchRun &run) noexcept {
        if (baseline == nullptr || baseline->trace.ns_per_ray() <= 0.0 ||
            run.trace.ns_per_ray() <= 0.0) {
            return 0.0;
        }
        return run.trace.ns_per_ray() / baseline->trace.ns_per_ray();
    };
    auto print_table = [&](const char *title, bool by_build) noexcept {
        auto sorted = measured;
        std::sort(sorted.begin(), sorted.end(), [by_build](const auto *a, const auto *b) noexcept {
            return by_build ? a->build.total_ms.min_ms() > b->build.total_ms.min_ms() : a->trace.total_ms.min_ms() > b->trace.total_ms.min_ms();
        });
        std::printf("\n-- %s (worst first, ratios normalized against 'uniform') --\n", title);
        std::printf("%-15s  %10s  %10s  %9s  %10s  %9s\n", "scene", "primitives",
                    "build_ms", "x/prims", "trace_ms", "x/ray");
        for (auto *run : sorted) {
            auto build_ratio_value = build_ratio(*run);
            auto trace_ratio_value = trace_ratio(*run);
            std::printf("%-15s  %10llu  %10.3f  %9.2f  %10.3f  %9.2f\n", run->scene.c_str(),
                        static_cast<unsigned long long>(run->primitives),
                        run->build.total_ms.min_ms(),
                        build_ratio_value > 0.0 ? build_ratio_value : 1.0,
                        run->trace.total_ms.min_ms(),
                        trace_ratio_value > 0.0 ? trace_ratio_value : 1.0);
        }
        if (baseline == nullptr) {
            std::printf("(no 'uniform' baseline in this run: the ratios default to 1.00)\n");
        }
    };
    print_table("worst-case ranking by build time", true);
    print_table("worst-case ranking by traversal time", false);
}

// Budget report: one line per measurement with the estimate, what was actually
// allocated and the headroom that is left in the budget.
void print_budget_report(luisa::span<const BenchRun> runs, double budget_gib) noexcept {
    auto budget_bytes = static_cast<size_t>(budget_gib * 1024.0 * 1024.0 * 1024.0);
    std::printf("\n-- device memory report (budget %.2f GiB) --\n", budget_gib);
    std::printf("%-15s  %-18s  %12s  %12s  %12s  %s\n", "scene", "step", "estimated",
                "allocated", "headroom", "status");
    for (auto &&run : runs) {
        auto estimate = run.memory.total_bytes();
        auto headroom = estimate < budget_bytes ? budget_bytes - estimate : 0u;
        std::printf("%-15s  %-18s  %12llu  %12llu  %12llu  %s\n", run.scene.c_str(),
                    run.step.empty() ? "-" : run.step.c_str(),
                    static_cast<unsigned long long>(estimate),
                    static_cast<unsigned long long>(run.actual_bytes),
                    static_cast<unsigned long long>(headroom),
                    run.skipped ? "SKIPPED (over budget)" : "ok");
    }
    std::printf("(bytes; 'estimated' is what the budget check used before allocating,\n"
                " 'allocated' is what the measurement actually created - the storage rounds\n"
                " every tree up to its two-node-per-primitive budget, so estimated >= allocated)\n");
}

// The scenes a run selects: the whole catalogue, only the worst cases, or one
// named scene (or the OBJ mesh scene).
[[nodiscard]] bool select_scenes(const BenchOptions &options,
                                 luisa::vector<luisa::string> &scenes,
                                 luisa::string &error) noexcept {
    if (!options.mesh_path.empty()) {
        scenes.emplace_back("mesh");
        if (options.scene != "all" && options.scene != "mesh") {
            error = luisa::format("--mesh runs the file as a single scene, so --scene {} does not apply",
                                  options.scene);
            return false;
        }
        return true;
    }
    if (options.scene == "all") {
        for (auto &&info : bench_scene_catalogue()) { scenes.emplace_back(info.name); }
        return true;
    }
    if (options.scene == "worst") {
        for (auto &&info : bench_scene_catalogue()) {
            // 'uniform' is the baseline and is not a worst case of anything
            if (std::strcmp(info.name, "uniform") != 0) { scenes.emplace_back(info.name); }
        }
        return true;
    }
    auto known = false;
    for (auto &&info : bench_scene_catalogue()) {
        if (options.scene == info.name) { known = true; }
    }
    if (!known) {
        error = luisa::format("unknown scene '{}'", options.scene);
        return false;
    }
    scenes.emplace_back(options.scene);
    return true;
}

}// namespace

int main(int argc, char *argv[]) {
    // The records printed below *are* the result of the benchmark: they must not
    // be lost when the process aborts (a failed --validate) and must not
    // interleave with the console logger, which writes to the same stream.  An
    // unbuffered stdout costs nothing at this output rate and guarantees both.
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    auto executable = argc > 0 && argv != nullptr && argv[0] != nullptr ? argv[0] : "";
    auto parse = parse_bench_options(argc, argv);
    if (!parse.ok()) {
        std::printf("error: %s\n\n", parse.error.c_str());
        print_bench_usage(executable);
        return 1;
    }
    auto &options = parse.options;
    if (options.help) {
        print_bench_usage(executable);
        return 0;
    }
    if (options.list) {
        print_catalogue();
        return 0;
    }
    if (options.backend.empty()) {
        // the same contract as the demo: the backend is the first argument
        print_bench_usage(executable);
        return 1;
    }
    luisa::vector<luisa::string> scenes;
    luisa::string error;
    if (!select_scenes(options, scenes, error)) {
        std::printf("error: %s\n\n", error.c_str());
        print_bench_usage(executable);
        return 1;
    }
    Context context{executable};
    Device device = context.create_device(options.backend);
    Stream stream = device.create_stream();
    std::printf("bench_config backend=%s scenes=%llu iters=%u warmup=%u seed=%llu budget_gib=%.2f max_triangles=%llu max_seconds=%.1f stress_build=%d stress_traversal=%d validate=%d repeat_check=%d\n",
                options.backend.c_str(), static_cast<unsigned long long>(scenes.size()),
                options.iterations, options.warmup,
                static_cast<unsigned long long>(options.seed), options.budget_gib,
                static_cast<unsigned long long>(options.max_triangles), options.max_seconds,
                options.stress_build ? 1 : 0, options.stress_traversal ? 1 : 0,
                options.validate ? 1 : 0, options.repeat_check ? 1 : 0);
    LUISA_INFO("software LBVH benchmark on '{}': {} scene(s), {} iteration(s) per measurement "
               "(a debug/ASan run only proves correctness: the numbers below are host-observed "
               "wall times including submission)",
               options.backend, scenes.size(), options.iterations);
    auto ray_shader = device.compile(make_ray_kernel());
    luisa::vector<BenchRun> results;
    for (auto &&name : scenes) {
        auto status = run_scene(device, stream, options, ray_shader, name.c_str(), results,
                                options.validate, options.repeat_check, error);
        if (status == RunStatus::SCENE_ERROR) {
            // a missing asset or an unusable request: report it and stop, without
            // aborting the process
            std::printf("error: %s\n", error.c_str());
            return 1;
        }
        if (status == RunStatus::MISMATCH) {
            // a malformed tree or a mismatch with the RTX reference: an
            // unconditional failure
            LUISA_ERROR("{}", error);
            return 1;
        }
    }
    print_ranking(luisa::span{results});
    print_budget_report(luisa::span{results}, options.budget_gib);
    std::fflush(stdout);
    return 0;
}

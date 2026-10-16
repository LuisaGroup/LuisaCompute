// Instrumented traversal and tree-shape measurement of the software LBVH.
//
// The library traversal (tlas.cpp / blas.cpp) stays untouched: this header owns
// *its own* copy of the two-level walk with counters, so the counters can never
// cost the measured path anything.  The copy reuses the library's node layout
// and the very same `aabb_test` / `triangle_test` helpers, so it cannot disagree
// with the library walk about what a traversal does - it only counts it.
//
// Two measurements are provided:
//
//   * `trace_instrumented()` - the two-level walk with per-ray counters
//     (nodes popped, slab tests, slab tests that passed, Moller-Trumbore tests,
//     deepest traversal stack).  It quantifies how badly a tree culls, which is
//     the traversal half of the adversarial-scene story.
//   * `measure_tree()` - the shape of one *built* tree: the depth of every leaf
//     and the leaf range of every internal node, i.e. the two quantities that
//     drive the cost of the internal-node AABB reduction and the length of the
//     `determine_range` search.
//
// Neither kernel reduces with global atomics: every result goes to its own
// element and the host reduces.  Same-address atomic reduction over a million
// elements is pathologically slow on some backends (it is what made the DirectX
// device time out at 1<<20 triangles) and it makes a float sum order-dependent,
// while the host reduction is exact and deterministic.

#pragma once

#include "bench_harness.h"

#include "../lbvh_common.h"

#include <cstddef>

// ---------------------------------------------------------------------------
// GPU-side records.
//
// This must be declared at global scope: LUISA_STRUCT opens namespace
// luisa::compute to specialize the DSL types for the struct (the same rule the
// LBVH layout in lbvh_common.h follows).
// ---------------------------------------------------------------------------

// Counters written by the instrumented traversal, one element per ray.
struct LbvhRayStats {
    luisa::uint nodes;     // nodes popped
    luisa::uint aabb_tests;// slab tests executed
    luisa::uint aabb_hits; // slab tests that passed
    luisa::uint tri_tests; // Moller-Trumbore executions
    luisa::uint max_stack; // deepest software traversal stack reached
};
LUISA_STRUCT(LbvhRayStats, nodes, aabb_tests, aabb_hits, tri_tests, max_stack) {};

namespace luisa::example::lbvh {

using namespace luisa;
using namespace luisa::compute;

// Iteration cap of the depth walk.  A 30-bit radix tree cannot be deeper than
// `traversal_stack_size` nodes without the library traversal dropping subtrees,
// so a descent that needs more steps is reported as capped instead of hanging.
inline constexpr uint max_tree_descent_steps = 128u;

// The two high bits of a recorded leaf depth carry the outcome of the walk, so
// one uint per leaf is enough to report success, a capped walk and a failed walk
// without any second buffer or any atomic.
inline constexpr uint leaf_depth_value_mask = 0x3FFFFFFFu;
inline constexpr uint leaf_depth_capped = 0x80000000u;
inline constexpr uint leaf_depth_failed = 0xC0000000u;

// ---------------------------------------------------------------------------
// Host-side aggregates
// ---------------------------------------------------------------------------

// Per-ray counters reduced on the host (the buffer is read back once).
struct BenchRayStats {
    size_t rays{0u};
    double avg_nodes{0.0};
    double avg_aabb_tests{0.0};
    double avg_aabb_hits{0.0};
    double avg_tri_tests{0.0};
    double avg_max_stack{0.0};
    size_t nodes_max{0u};
    size_t stack_max{0u};
    // Fraction of the slab tests that rejected the node: the culling quality of
    // the tree for this ray distribution (1.0 = nothing was ever entered).
    double culled_ratio{0.0};
};

// Reduced shape of one tree (or of every tree of a scene).
struct BenchTreeStats {
    size_t leaf_count{0u};
    size_t internal_count{0u};
    size_t descent_failures{0u};
    size_t depth_capped{0u};
    size_t max_depth{0u};
    double mean_depth{0.0};
    double depth_per_log2_n{0.0};
    double max_range{0.0};
    double mean_range{0.0};
    [[nodiscard]] bool measured() const noexcept { return leaf_count != 0u; }
};

[[nodiscard]] BenchRayStats summarize_ray_stats(luisa::span<const LbvhRayStats> stats) noexcept;

// ---------------------------------------------------------------------------
// Device side
// ---------------------------------------------------------------------------

class BenchStats {

public:
    // `node_capacity` / `primitive_capacity` are the fields of
    // `LbvhStorage::Sizes`: the leaf ranges and the leaf depths are indexed like
    // the shared node / primitive buffers, so one element per slot is enough for
    // every tree of the scene.
    BenchStats(Device &device, size_t node_capacity, size_t primitive_capacity) noexcept;

    // Bytes the statistics own on the device (counted by the memory budget).
    [[nodiscard]] size_t scratch_bytes() const noexcept {
        return _node_range.size() * _node_range.stride() +
               _leaf_depth.size() * _leaf_depth.stride();
    }

    // One instrumented traversal of the strided slice
    // (ray_offset, ray_offset + ray_stride, ...) of the ray buffer; `ray_stats`
    // receives one record per ray of that slice.  `heap` is the TLAS' bindless
    // heap: the instrumented walk resolves every node region through it exactly
    // like the library walk (lbvh_common.h's "The bindless heap of a TLAS").
    void trace_instrumented(Stream &stream, const BindlessArray &heap,
                            const Buffer<LbvhBlas> &blas_table,
                            const Buffer<LbvhInstance> &instances,
                            const Buffer<float3> &vertices,
                            const Buffer<Triangle> &triangles,
                            const Buffer<LbvhRay> &rays, const Buffer<LbvhHit> &hits,
                            const Buffer<LbvhRayStats> &ray_stats,
                            uint tlas_node_offset, uint ray_count, uint ray_offset = 0u,
                            uint ray_stride = 1u) noexcept;

    // Shape of one built tree, read back and reduced on the host.  Not timed: it
    // is a diagnostic run once per scene.
    [[nodiscard]] BenchTreeStats measure_tree(Stream &stream, const Buffer<LbvhNode> &nodes,
                                              uint node_base, uint count) noexcept;

private:
    Buffer<uint2> _node_range;// (first, last) leaf slot of every internal node
    Buffer<uint> _leaf_depth; // encoded depth of every leaf
    luisa::vector<uint2> _host_range;
    luisa::vector<uint> _host_depth;
    Shader1D<BindlessArray, Buffer<LbvhBlas>, Buffer<LbvhInstance>, Buffer<float3>,
             Buffer<Triangle>, Buffer<LbvhRay>, Buffer<LbvhHit>, Buffer<LbvhRayStats>, uint,
             uint, uint, uint>
        _trace_kernel;
    Shader1D<Buffer<LbvhNode>, Buffer<uint2>, uint, uint> _range_kernel;
    Shader1D<Buffer<LbvhNode>, Buffer<uint2>, Buffer<uint>, uint, uint> _depth_kernel;
};

}// namespace luisa::example::lbvh

// Instrumented traversal and tree-shape measurement of the software LBVH: see
// bench_stats.h for what they are for.  The walks below deliberately mirror the
// library ones (blas.cpp / tlas.cpp) statement by statement and reuse the
// library's `LbvhNode` layout plus `aabb_test` / `triangle_test`, so a counter
// can only ever describe the walk the library performs.

#include "bench_stats.h"

#include <algorithm>
#include <cmath>

namespace luisa::example::lbvh {

namespace {

// Instrumented copy of `blas_traversal()`: one ray in the object space of one
// instance, counting instead of only intersecting.  `max_stack` follows the
// deepest the per-ray software stack ever got, which is the quantity that says
// whether the traversal is anywhere near `traversal_stack_size`.
void blas_traversal_instrumented(Var<LbvhHit> &best, UInt instance, const Var<LbvhBlas> &blas,
                                 Float3 origin, Float3 direction, Float t_min,
                                 const BufferVar<LbvhNode> &nodes,
                                 const BufferVar<float3> &vertices,
                                 const BufferVar<Triangle> &triangles,
                                 Var<uint> &nodes_visited, Var<uint> &aabb_tests,
                                 Var<uint> &aabb_hits, Var<uint> &tri_tests,
                                 Var<uint> &max_stack) noexcept {
    auto inv_dir = safe_reciprocal(direction);
    Local<uint> stack{traversal_stack_size};
    // Same walk as the library's `blas_traversal()`, counter for counter.
    stack[0u] = blas.node_offset;
    auto size = def(1u);
    $while (size > 0u) {
        size = size - 1u;
        nodes_visited = nodes_visited + 1u;
        auto node = nodes.read(stack[size]);
        auto node_left = child_left(node);
        aabb_tests = aabb_tests + 1u;
        $if (aabb_test(aabb_lo(node), aabb_hi(node), origin, inv_dir, t_min, best.t)) {
            aabb_hits = aabb_hits + 1u;
            $if (node_left == invalid_node) {
                auto node_prim = child_right(node);
                tri_tests = tri_tests + 1u;
                auto tri = triangles.read(blas.triangle_offset + node_prim);
                auto v0 = vertices.read(tri.i0);
                auto v1 = vertices.read(tri.i1);
                auto v2 = vertices.read(tri.i2);
                auto result = triangle_test(v0, v1, v2, origin, direction, t_min, best.t);
                $if (result.x >= 0.0f) {
                    best.t = result.x;
                    best.bary = make_float2(result.y, result.z);
                    best.prim = node_prim;
                    best.inst = instance;
                };
            }
            $else {
                $if (size + 2u < traversal_stack_size) {
                    stack[size] = node_left;
                    size = size + 1u;
                    stack[size] = child_right(node);
                    size = size + 1u;
                    max_stack = max(max_stack, size);
                };
            };
        };
    };
}

// Instrumented copy of `tlas_traversal()`: the top level drives the (also
// instrumented) bottom level, so both levels land in the same counters.
Var<LbvhHit> tlas_traversal_instrumented(const Var<LbvhRay> &ray, UInt tlas_node_offset,
                                         const BufferVar<LbvhNode> &nodes,
                                         const BufferVar<LbvhBlas> &blas_table,
                                         const BufferVar<LbvhInstance> &instances,
                                         const BufferVar<float3> &vertices,
                                         const BufferVar<Triangle> &triangles,
                                         Var<uint> &nodes_visited, Var<uint> &aabb_tests,
                                         Var<uint> &aabb_hits, Var<uint> &tri_tests,
                                         Var<uint> &max_stack) noexcept {
    auto origin = ray.origin;
    auto direction = ray.direction;
    auto t_min = ray.t_min;
    auto inv_dir = safe_reciprocal(direction);
    Var<LbvhHit> best;
    best.inst = invalid_node;
    best.prim = invalid_node;
    best.bary = make_float2(0.0f);
    best.t = ray.t_max;
    Local<uint> stack{traversal_stack_size};
    stack[0u] = tlas_node_offset;
    auto size = def(1u);
    $while (size > 0u) {
        size = size - 1u;
        nodes_visited = nodes_visited + 1u;
        auto node = nodes.read(stack[size]);
        auto node_left = child_left(node);
        aabb_tests = aabb_tests + 1u;
        $if (aabb_test(aabb_lo(node), aabb_hi(node), origin, inv_dir, t_min, best.t)) {
            aabb_hits = aabb_hits + 1u;
            $if (node_left == invalid_node) {
                auto node_prim = child_right(node);
                auto instance = instances.read(node_prim);
                auto blas = blas_table.read(instance.blas);
                auto o4 = make_float4(origin, 1.0f);
                auto d4 = make_float4(direction, 0.0f);
                auto object_origin = make_float3(dot(o4, instance.to_object_0),
                                                 dot(o4, instance.to_object_1),
                                                 dot(o4, instance.to_object_2));
                auto object_dir = make_float3(dot(d4, instance.to_object_0),
                                              dot(d4, instance.to_object_1),
                                              dot(d4, instance.to_object_2));
                blas_traversal_instrumented(best, node_prim, blas, object_origin, object_dir,
                                            t_min, nodes, vertices, triangles, nodes_visited,
                                            aabb_tests, aabb_hits, tri_tests, max_stack);
            }
            $else {
                $if (size + 2u < traversal_stack_size) {
                    stack[size] = node_left;
                    size = size + 1u;
                    stack[size] = child_right(node);
                    size = size + 1u;
                    max_stack = max(max_stack, size);
                };
            };
        };
    };
    return best;
}

[[nodiscard]] auto make_trace_kernel() noexcept {
    return Kernel1D{[](BufferVar<LbvhNode> nodes, BufferVar<LbvhBlas> blas_table,
                       BufferVar<LbvhInstance> instances, BufferVar<float3> vertices,
                       BufferVar<Triangle> triangles, BufferVar<LbvhRay> rays,
                       BufferVar<LbvhHit> hits, BufferVar<LbvhRayStats> ray_stats,
                       UInt tlas_node_offset, UInt ray_offset, UInt ray_stride,
                       UInt count) noexcept {
        set_block_size(64u);
        UInt i = dispatch_id().x;
        UInt index = ray_offset + i * ray_stride;
        $if (i < count) {
            auto ray = rays.read(index);
            auto nodes_visited = def(0u);
            auto aabb_tests = def(0u);
            auto aabb_hits = def(0u);
            auto tri_tests = def(0u);
            auto max_stack = def(0u);
            auto hit = tlas_traversal_instrumented(ray, tlas_node_offset, nodes, blas_table,
                                                   instances, vertices, triangles,
                                                   nodes_visited, aabb_tests, aabb_hits,
                                                   tri_tests, max_stack);
            hits.write(index, hit);
            Var<LbvhRayStats> stats;
            stats.nodes = nodes_visited;
            stats.aabb_tests = aabb_tests;
            stats.aabb_hits = aabb_hits;
            stats.tri_tests = tri_tests;
            stats.max_stack = max_stack;
            ray_stats.write(index, stats);
        };
    }};
}

// Leaf range of every internal node of one tree.
//
// The range is recovered from the *structure*: the first leaf of the subtree of
// a node is reached by always turning left, the last one by always turning
// right.  This is exact and deterministic, whereas the "containment of the leaf
// centre" walk is ambiguous exactly where the benchmark looks: in the
// adversarial scenes sibling AABBs overlap, in `coincident` / `grid-duplicates`
// they are *identical*, so a centre can be inside both children and any
// containment rule would report an arbitrary depth.
[[nodiscard]] auto make_range_kernel() noexcept {
    return Kernel1D{[](BufferVar<LbvhNode> nodes, BufferVar<uint2> node_range,
                       UInt node_base, UInt count) noexcept {
        set_block_size(sort_block_size);
        UInt i = dispatch_id().x;
        // internal nodes of a tree are [node_base, node_base + count - 2]
        $if (i + 1u < count) {
            auto leaf_base = node_base + count - 1u;
            auto left_index = def(node_base + i);
            auto right_index = def(node_base + i);
            auto first = def(0u);
            auto last = def(0u);
            $for (step, max_tree_descent_steps) {
                auto node = nodes.read(left_index);
                $if (child_left(node) == invalid_node) {
                    first = left_index - leaf_base;
                    $break;
                };
                left_index = child_left(node);
            };
            $for (step, max_tree_descent_steps) {
                auto node = nodes.read(right_index);
                $if (child_left(node) == invalid_node) {
                    last = right_index - leaf_base;
                    $break;
                };
                right_index = child_right(node);
            };
            node_range.write(node_base + i, make_uint2(first, last));
        };
    }};
}

// Depth of every leaf, by descending from the root through the leaf ranges
// computed above.  The iteration cap turns a malformed tree (or a tree deeper
// than the traversal could ever walk) into a recorded outcome instead of a hang,
// and the encoding of `leaf_depth` keeps the outcome in the same element.
[[nodiscard]] auto make_depth_kernel() noexcept {
    return Kernel1D{[](BufferVar<LbvhNode> nodes, BufferVar<uint2> node_range,
                       BufferVar<uint> leaf_depth, UInt node_base, UInt count) noexcept {
        set_block_size(sort_block_size);
        UInt j = dispatch_id().x;
        $if (j < count) {
            auto leaf_base = node_base + count - 1u;
            auto index = def(node_base);
            auto depth = def(0u);
            auto arrived = def(false);
            $for (step, max_tree_descent_steps) {
                auto node = nodes.read(index);
                $if (child_left(node) == invalid_node) {
                    arrived = true;
                    $break;
                };
                // the split of the node's leaf range: the last slot of the left
                // child, which is the slot itself when the left child is a leaf
                auto left = child_left(node);
                auto split = def(0u);
                $if (left >= leaf_base) {
                    split = left - leaf_base;
                }
                $else {
                    split = node_range.read(left).y;
                };
                $if (j <= split) {
                    index = left;
                }
                $else {
                    index = child_right(node);
                };
                depth = depth + 1u;
            };
            // sanity: a finished walk must have ended on *this* leaf
            auto valid = arrived & (index - leaf_base == j);
            $if (valid) {
                leaf_depth.write(j, depth);
            }
            $else {
                $if (arrived) {
                    leaf_depth.write(j, depth | leaf_depth_failed);
                }
                $else {
                    leaf_depth.write(j, depth | leaf_depth_capped);
                };
            };
        };
    }};
}

}// namespace

BenchStats::BenchStats(Device &device, size_t node_capacity,
                       size_t primitive_capacity) noexcept
    : _node_range{device.create_buffer<uint2>(node_capacity)},
      _leaf_depth{device.create_buffer<uint>(primitive_capacity)},
      _trace_kernel{device.compile(make_trace_kernel())},
      _range_kernel{device.compile(make_range_kernel())},
      _depth_kernel{device.compile(make_depth_kernel())} {}

void BenchStats::trace_instrumented(Stream &stream, const Buffer<LbvhNode> &nodes,
                                    const Buffer<LbvhBlas> &blas_table,
                                    const Buffer<LbvhInstance> &instances,
                                    const Buffer<float3> &vertices,
                                    const Buffer<Triangle> &triangles,
                                    const Buffer<LbvhRay> &rays, const Buffer<LbvhHit> &hits,
                                    const Buffer<LbvhRayStats> &ray_stats,
                                    uint tlas_node_offset, uint ray_count, uint ray_offset,
                                    uint ray_stride) noexcept {
    stream << _trace_kernel(nodes, blas_table, instances, vertices, triangles, rays, hits,
                            ray_stats, tlas_node_offset, ray_offset, ray_stride, ray_count)
                  .dispatch(ray_count);
}

BenchTreeStats BenchStats::measure_tree(Stream &stream, const Buffer<LbvhNode> &nodes,
                                        uint node_base, uint count) noexcept {
    BenchTreeStats result;
    if (count == 0u) { return result; }
    stream << _range_kernel(nodes, _node_range, node_base, count).dispatch(count)
           << _depth_kernel(nodes, _node_range, _leaf_depth, node_base, count).dispatch(count);
    // One read-back per tree and the reduction on the host: exact, deterministic
    // and free of the same-address atomics a device-side reduction would need.
    if (count > 1u) {
        _host_range.resize(count - 1u);
        stream << _node_range.view(node_base, count - 1u).copy_to(luisa::span{_host_range});
    } else {
        _host_range.clear();
    }
    _host_depth.resize(count);
    stream << _leaf_depth.view(0u, count).copy_to(luisa::span{_host_depth})
           << synchronize();
    auto range_sum = 0.0;
    for (auto &&range : _host_range) {
        auto size = range.y - range.x + 1u;
        result.max_range = static_cast<double>(std::max(static_cast<uint>(result.max_range), size));
        range_sum += static_cast<double>(size);
    }
    result.internal_count = _host_range.size();
    auto depth_sum = size_t{0u};
    for (auto state : _host_depth) {
        auto value = state & leaf_depth_value_mask;
        if (state == (value | leaf_depth_failed)) {
            result.descent_failures++;
        } else if (state == (value | leaf_depth_capped)) {
            result.depth_capped++;
        } else {
            result.leaf_count++;
            depth_sum += value;
            result.max_depth = std::max(result.max_depth, static_cast<size_t>(value));
        }
    }
    result.mean_depth = result.leaf_count > 0u ? static_cast<double>(depth_sum) / static_cast<double>(result.leaf_count) : 0.0;
    result.depth_per_log2_n = (result.leaf_count > 1u && result.max_depth > 0u) ? static_cast<double>(result.max_depth) /
                                                                                      std::log2(static_cast<double>(result.leaf_count)) :
                                                                                  0.0;
    result.mean_range = result.internal_count > 0u ? range_sum / static_cast<double>(result.internal_count) : 0.0;
    return result;
}

BenchRayStats summarize_ray_stats(luisa::span<const LbvhRayStats> stats) noexcept {
    BenchRayStats result;
    if (stats.empty()) { return result; }
    auto nodes = 0.0;
    auto aabb_tests = 0.0;
    auto aabb_hits = 0.0;
    auto tri_tests = 0.0;
    auto max_stack = 0.0;
    for (auto &&sample : stats) {
        nodes += sample.nodes;
        aabb_tests += sample.aabb_tests;
        aabb_hits += sample.aabb_hits;
        tri_tests += sample.tri_tests;
        max_stack += sample.max_stack;
        result.nodes_max = std::max(result.nodes_max, static_cast<size_t>(sample.nodes));
        result.stack_max = std::max(result.stack_max, static_cast<size_t>(sample.max_stack));
    }
    auto n = static_cast<double>(stats.size());
    result.rays = stats.size();
    result.avg_nodes = nodes / n;
    result.avg_aabb_tests = aabb_tests / n;
    result.avg_aabb_hits = aabb_hits / n;
    result.avg_tri_tests = tri_tests / n;
    result.avg_max_stack = max_stack / n;
    result.culled_ratio = aabb_tests > 0.0 ? 1.0 - aabb_hits / aabb_tests : 0.0;
    return result;
}

}// namespace luisa::example::lbvh

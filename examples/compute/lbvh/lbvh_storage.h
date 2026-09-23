// The storage shared by every LBVH of one scene, together with the tree-agnostic
// build stages: Morton codes, the single-work-group LSD radix sort and the
// Karras radix-tree construction (leaves, internal nodes, node AABBs).
//
// A `BlasBuilder` (blas.h) and a `TlasBuilder` (tlas.h) only supply the
// primitive AABBs (step 1 of the build) and then hand their tree to
// `build_tree()`; both allocate their tree in the same buffers, so a traversal
// can walk a BLAS through the TLAS without any descriptor indexing.

#pragma once

#include "lbvh_common.h"

#include <cstddef>

namespace luisa::example::lbvh {

using namespace luisa;
using namespace luisa::compute;

// Owns the GPU buffers of all trees of one scene.
class LbvhStorage {

public:
    // The contiguous slice of the shared primitive/node buffers that belongs to
    // one tree.
    struct TreeRange {
        uint prim_base;// first primitive slot
        uint node_base;// first node
        uint count;    // primitive (= leaf) count
        [[nodiscard]] uint node_count() const noexcept { return count * 2u - 1u; }
    };

    // Host-side scene size query: the maximum sizes of every shared buffer of a
    // scene with these capacities.  Like the backend build-size query, it runs
    // before any device allocation, so the caller can size the storage from the
    // geometry (triangle/instance/BLAS counts) instead of guessing capacities.
    struct Sizes {
        size_t primitive_capacity{};// BLAS triangles + TLAS instances
        size_t node_capacity{};     // 2 * primitive_capacity (budget)
        size_t blas_capacity{};
        size_t instance_capacity{};
        size_t primitive_bytes{};
        size_t key_bytes{};
        size_t node_bytes{};
        size_t blas_table_bytes{};
        size_t instance_bytes{};
        [[nodiscard]] size_t total_bytes() const noexcept {
            return primitive_bytes + key_bytes + node_bytes +
                   blas_table_bytes + instance_bytes;
        }
    };

    [[nodiscard]] static Sizes estimate(size_t max_triangles, size_t max_instances,
                                        size_t max_blas) noexcept;

    LbvhStorage(Device &device, const Sizes &sizes) noexcept;
    LbvhStorage(Device &device, size_t max_triangles, size_t max_instances,
                size_t max_blas) noexcept
        : LbvhStorage{device, estimate(max_triangles, max_instances, max_blas)} {}

    // Reserve room for `prim_count` primitives and the nodes they need (host
    // side bookkeeping only, no GPU work).
    [[nodiscard]] TreeRange allocate(size_t prim_count) noexcept;

    // Host -> device upload of the per-BLAS table / the TLAS instance records.
    void upload_blas_table(Stream &stream, luisa::span<const LbvhBlas> table) noexcept;
    void upload_instances(Stream &stream, luisa::span<const LbvhInstance> instances) noexcept;

    // Morton codes -> 4 x 8-bit LSD radix sort -> radix tree + node AABBs.
    // The primitive AABBs of `range` must have been filled in already; they are
    // expected to lie inside [lo, hi], which is mapped onto the unit cube.
    void build_tree(Stream &stream, const TreeRange &range, float3 lo, float3 hi) noexcept;

    // Structural self-check of one built tree: every node must be reachable from
    // the root exactly once, and every internal node AABB must be the union of
    // its children's AABBs.  Returns the number of problems found.
    [[nodiscard]] size_t validate_tree(Stream &stream, uint node_base, uint count) noexcept;

    [[nodiscard]] const Buffer<LbvhPrim> &prims() const noexcept { return _prims; }
    [[nodiscard]] const Buffer<LbvhNode> &nodes() const noexcept { return _nodes; }
    [[nodiscard]] const Buffer<LbvhBlas> &blas_table() const noexcept { return _blas_table; }
    [[nodiscard]] const Buffer<LbvhInstance> &instances() const noexcept { return _instances; }
    [[nodiscard]] const Sizes &sizes() const noexcept { return _sizes; }
    [[nodiscard]] size_t primitive_count() const noexcept { return _prim_count; }
    [[nodiscard]] size_t node_count() const noexcept { return _node_count; }

private:
    Sizes _sizes;
    size_t _prim_count{0u};
    size_t _node_count{0u};
    Buffer<LbvhPrim> _prims;
    Buffer<LbvhKey> _keys_a;
    Buffer<LbvhKey> _keys_b;
    Buffer<LbvhNode> _nodes;
    Buffer<LbvhBlas> _blas_table;
    Buffer<LbvhInstance> _instances;
    Shader1D<Buffer<LbvhPrim>, Buffer<LbvhKey>, uint, uint, float3, float3> _morton_kernel;
    Shader1D<Buffer<LbvhKey>, Buffer<LbvhKey>, uint, uint, uint> _sort_kernel;
    Shader1D<Buffer<LbvhKey>, Buffer<LbvhPrim>, Buffer<LbvhNode>, uint, uint, uint> _build_kernel;
};

}// namespace luisa::example::lbvh

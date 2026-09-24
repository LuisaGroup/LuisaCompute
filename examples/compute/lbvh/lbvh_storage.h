// The storage shared by every LBVH of one scene, together with the tree-agnostic
// build stages: Morton codes, the LSD radix sort (lbvh_sort.h) and the Karras
// radix-tree construction (leaves, internal nodes, node AABBs).
//
// A `BlasBuilder` (blas.h) and a `TlasBuilder` (tlas.h) only supply the
// primitive AABBs (step 1 of the build) and then hand their tree to
// `build_tree()`; both allocate their tree in the same buffers, so a traversal
// can walk a BLAS through the TLAS without any descriptor indexing.

#pragma once

#include "lbvh_common.h"
#include "lbvh_sort.h"

#include <cstddef>

namespace luisa::example::lbvh {

using namespace luisa;
using namespace luisa::compute;

// Owns the GPU buffers of all trees of one scene.
class LbvhStorage {

public:
    // The contiguous slice of the shared primitive/node buffers that belongs to
    // one tree.  Every field is initialised: a caller that builds one by hand
    // (the two builders do, from the accessors of their resource) must not be
    // able to leave a field indeterminate.
    struct TreeRange {
        uint prim_base{};// first primitive slot
        uint node_base{};// first node
        uint plan_base{};// first plan slot (see `_plan_kernel`)
        uint count{};    // primitive (= leaf) count
        [[nodiscard]] uint node_count() const noexcept { return count * 2u - 1u; }
        // Internal nodes of the tree, i.e. one plan record each.
        [[nodiscard]] uint internal_count() const noexcept { return count - 1u; }
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
        // Block AABBs of the two-level reduction (see `node_reduction_block`).
        // A block is indexed by the *leaf slot* it starts at, i.e. by the node
        // array (a leaf of a tree lives in the shared node buffer at its global
        // slot), so the whole scene needs one block per `node_reduction_block`
        // node slots - no per-tree rounding, and no bookkeeping to allocate.
        size_t block_capacity{};
        size_t block_bytes{};
        // Per-internal-node records of `_plan_kernel` (one `uint4` per internal
        // node: leaf range + child handles).  A scene has `sum(count) - trees <
        // primitive_capacity` internal nodes, so the primitive capacity bounds it.
        size_t plan_capacity{};
        size_t plan_bytes{};
        size_t blas_table_bytes{};
        size_t instance_bytes{};
        // Scratch of the parallel radix sort (per-block digit histograms and
        // their scan).  Like `scratch_bytes` of a backend build it is not part of
        // the acceleration structure and is only needed while the build runs, but
        // it *is* device memory the caller has to have room for, so the size
        // query reports it.
        size_t sort_scratch_bytes{};
        [[nodiscard]] size_t total_bytes() const noexcept {
            return primitive_bytes + key_bytes + node_bytes + block_bytes +
                   plan_bytes + blas_table_bytes + instance_bytes + sort_scratch_bytes;
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
    // With a non-null `timings` the stages are separated by a synchronisation
    // and their host-observed times are added into `timings` (see
    // `LbvhBuildTimings`); a null pointer keeps the plain recorded build.
    //
    // The radix-tree construction itself is *four* dispatches over the same
    // tree - the leaves, the block AABBs, and the two internal-node passes (the
    // searches of the tree structure, then the AABB reduction) - which is what
    // `node_ms` covers: an internal node's AABB is the union of the leaf AABBs of
    // its range, and once the leaves exist those are contiguous in the node
    // buffer, so the reduction streams them (and one AABB per
    // `node_reduction_block` of them) instead of chasing one random `prims`
    // element per range slot (see `_plan_kernel` / `_build_kernel`).
    void build_tree(Stream &stream, const TreeRange &range, float3 lo, float3 hi,
                    LbvhBuildTimings *timings = nullptr) noexcept;

    // Structural self-check of one built tree: every node must be reachable from
    // the root exactly once, and every internal node AABB must be the union of
    // its children's AABBs.  Returns the number of problems found.
    [[nodiscard]] size_t validate_tree(Stream &stream, uint node_base, uint count) noexcept;

    [[nodiscard]] const Buffer<LbvhPrim> &prims() const noexcept { return _prims; }
    [[nodiscard]] const Buffer<LbvhNode> &nodes() const noexcept { return _nodes; }
    [[nodiscard]] const Buffer<LbvhNode> &blocks() const noexcept { return _blocks; }
    [[nodiscard]] const Buffer<LbvhBlas> &blas_table() const noexcept { return _blas_table; }
    [[nodiscard]] const Buffer<LbvhInstance> &instances() const noexcept { return _instances; }
    [[nodiscard]] const Sizes &sizes() const noexcept { return _sizes; }
    [[nodiscard]] size_t primitive_count() const noexcept { return _prim_count; }
    [[nodiscard]] size_t node_count() const noexcept { return _node_count; }

private:
    Sizes _sizes;
    size_t _prim_count{0u};
    size_t _node_count{0u};
    size_t _plan_count{0u};
    Buffer<LbvhPrim> _prims;
    Buffer<LbvhKey> _keys_a;
    Buffer<LbvhKey> _keys_b;
    Buffer<LbvhNode> _nodes;
    // Block AABBs of `node_reduction_block` consecutive leaves, indexed by the
    // *leaf slot* they start at in the shared node buffer (a block holds the AABB
    // planes of `LbvhNode`; its handle lanes are unused).
    Buffer<LbvhNode> _blocks;
    // Per internal node: (first, last, child_a, child_b), the output of
    // `_plan_kernel` and the input of the reduction pass.
    Buffer<uint4> _plan;
    Buffer<LbvhBlas> _blas_table;
    Buffer<LbvhInstance> _instances;
    // The LSD radix sort of the Morton keys.  It owns its scratch and picks the
    // single-work-group or the parallel implementation from the tree size (see
    // lbvh_sort.h); for the small trees of a multi-BLAS scene that is the old
    // single-work-group sort, byte for byte.
    LbvhRadixSort _sort;
    // Warp (wave/sub-group) width of the device, queried once.  The radix-tree
    // construction dispatches one warp per internal node, so the host side needs
    // the same constant the kernel's `warp_lane_count()` expands to.
    uint _warp_size{32u};
    Shader1D<Buffer<LbvhPrim>, Buffer<LbvhKey>, uint, uint, float3, float3> _morton_kernel;
    Shader1D<Buffer<LbvhKey>, Buffer<LbvhPrim>, Buffer<LbvhNode>, uint, uint, uint> _leaf_kernel;
    Shader1D<Buffer<LbvhKey>, Buffer<uint4>, uint, uint, uint, uint> _plan_kernel;
    Shader1D<Buffer<LbvhNode>, Buffer<LbvhNode>, uint, uint, uint> _block_kernel;
    Shader2D<Buffer<LbvhNode>, Buffer<LbvhNode>, Buffer<uint4>, uint, uint, uint, uint>
        _build_kernel;
};

}// namespace luisa::example::lbvh

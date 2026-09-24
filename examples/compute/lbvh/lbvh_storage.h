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
        // Slot of this tree in `_usage`, the per-tree node counter the build
        // writes and `compact()` reads back.  One `uint` per reserved tree; the
        // builder copies it out of the range exactly like `node_base` (see
        // `Blas::usage_slot` / `Tlas::usage_slot`).
        uint usage_slot{};
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
        // One `uint` node-count slot per tree the storage can reserve: one per
        // BLAS plus the TLAS (`compact()` needs them all).  Like the backend
        // build-size query it is reported before anything is allocated, so the
        // caller can budget for it exactly.
        size_t tree_capacity{};
        size_t usage_bytes{};
        // Scratch of the parallel radix sort (per-block digit histograms and
        // their scan).  Like `scratch_bytes` of a backend build it is not part of
        // the acceleration structure and is only needed while the build runs, but
        // it *is* device memory the caller has to have room for, so the size
        // query reports it.
        size_t sort_scratch_bytes{};
        [[nodiscard]] size_t total_bytes() const noexcept {
            return primitive_bytes + key_bytes + node_bytes + block_bytes +
                   plan_bytes + blas_table_bytes + instance_bytes + usage_bytes +
                   sort_scratch_bytes;
        }
    };

    [[nodiscard]] static Sizes estimate(size_t max_triangles, size_t max_instances,
                                        size_t max_blas) noexcept;

    // How `compact()` lays the kept nodes out in the dense buffer:
    //
    //   * `as_built` (default, delivered): destination index == source index, so
    //     every tree keeps its base, every handle and every `LbvhBlas::node_offset`
    //     stays valid, the copy is a pure streaming node copy, and a later rebuild
    //     of the same storage stays valid (the layout is unchanged).  It reclaims
    //     `(capacity - used) * 32` bytes and shrinks the traversal working set.
    //   * `subtree_contiguous`: reserved but *not implemented* - `compact()`
    //     fails closed on it.  Relabelling a tree into DFS preorder so that every
    //     subtree is one contiguous range is the only variant that adds adjacent
    //     access rather than just shrinking the working set; it needs a second,
    //     node-sized LSD sort recorded into the same command list, a remap pass
    //     and a retirement bundle for the transient scratch, and it changes the
    //     rebuild contract because the indices move.  See bench/README.md,
    //     "Round 4", for the measured/rejected verdict.
    enum class CompactionPolicy { as_built,
                                  subtree_contiguous };

    // Outcome of one `compact()` call; the numbers are what the demo and the
    // benchmark report.
    struct CompactResult {
        CompactionPolicy policy{CompactionPolicy::as_built};
        size_t trees{};           // reserved trees compacted
        size_t nodes_before{};    // reserved (loose) node slots
        size_t nodes_after{};     // kept (dense) node slots
        size_t bytes_before{};
        size_t bytes_after{};
        size_t compacted_bytes{}; // bytes_before - bytes_after
        // Build-scratch bytes retired by the same call when the caller asked for
        // it (`release_build_scratch`, opt-in); 0 otherwise.  Reported separately
        // from `compacted_bytes`: they are transient build memory, not the
        // acceleration structure.
        size_t reclaimed_scratch_bytes{};
        [[nodiscard]] bool compacted() const noexcept { return nodes_after < nodes_before; }
        // The new, exactly-sized node buffer, *until* the heap views have been
        // re-registered onto it (`adopt_nodes()`): the caller registers them and
        // then moves the buffer into the storage, where it is owned from then on.
        // After that hand-off - or when nothing was reclaimed - this field is
        // moved-from and must not be used: read the live buffer through
        // `LbvhStorage::nodes()` (see `SoftwareLbvh::compact`).
        Buffer<LbvhNode> nodes;
    };

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

    // Device-side size query, then the dense copy (requirement steps 2..4).
    //
    // The per-tree node counts are already on the device (the fused leaf pass of
    // `build_tree`), so this first does its *own* submission -
    // `usage.view(0, tree_count).copy_to(host) << synchronize()` - to learn how
    // many nodes the trees actually use, then allocates a node buffer of exactly
    // that size and records the copy into `list` (the caller owns the commit and
    // appends the bindless-heap re-registration plus the retirement callback).
    //
    // The storage is left *untouched*: the loose buffer is taken by the caller
    // with `take_loose_nodes()` (it retires it through a completion callback) and
    // the dense buffer is handed back with `adopt_nodes()` *after* the heap views
    // have been re-registered onto it.  That split is what makes the
    // `BindlessArrayUpdateCommand` (which holds a raw handle until it executes)
    // and the retirement callback safe - the shape Metal's BLAS/TLAS compaction
    // uses (retain the new handle, copy, release the old in a completion
    // callback).
    //
    // Contract (mirrors the hardware backends): only valid on a *full* build
    // (every reserved tree must have been built: the device count must equal
    // `node_count()`), and after compaction the storage has no spare capacity,
    // so a later build of the same storage must fail closed or re-reserve a
    // loose buffer.  `as_built` keeps a later rebuild valid because the indices
    // are unchanged; `subtree_contiguous` does not.  Calling it twice is
    // allowed: the second call reclaims nothing and records no copy.
    //
    // Precedent: DX queries `PROPERTY_TYPE_COMPACTED_SIZE` into a buffer it
    // already owns and copies with
    // `CopyRaytracingAccelerationStructure(..., COPY_MODE_COMPACT)`; CUDA/OptiX
    // call `optixAccelCompact`; Metal writes the size, reads it back on a
    // completion callback, hard-syncs, then `copyAndCompactAccelerationStructure`
    // and releases the old handle in a callback.  This is that flow in DSL.
    [[nodiscard]] CompactResult compact(Stream &stream, CommandList &list,
                                        CompactionPolicy policy = CompactionPolicy::as_built) noexcept;

    // Hand-off of the node buffer around a compaction (see `compact`): the loose
    // buffer is moved out so the caller can retire it *after* the copy (the
    // callback owns it), and the dense buffer is moved in once the heap views
    // point at it.  Both are explicit so the caller can order the heap update
    // between them; neither does any GPU work.
    [[nodiscard]] Buffer<LbvhNode> take_loose_nodes() noexcept;
    void adopt_nodes(Buffer<LbvhNode> &&dense) noexcept;

    // Build scratch the storage can hand over once every tree has been built
    // (opt-in, see `release_build_scratch`).  The caller owns it and retires it
    // through a completion callback, so nothing the GPU still reads is destroyed
    // early.
    struct ReleasedScratch {
        Buffer<LbvhPrim> prims;
        Buffer<LbvhKey> keys_a;
        Buffer<LbvhKey> keys_b;
        Buffer<LbvhNode> blocks;
        Buffer<uint4> plan;
        size_t bytes{0u};
    };

    // Moves the build scratch - the primitive AABBs, the two Morton-key ping-pong
    // buffers, the block AABBs and the node plan - out of the storage.  They are
    // read only while a tree is *built*, so a caller that will only traverse (and
    // validate) can release them and get the bytes back; they are ~66 bytes per
    // primitive slot against the node buffer's 64, i.e. as large again as the
    // acceleration structure itself.
    //
    // Contract: after this the storage can only be traversed/validated - every
    // build stage fails closed on the missing buffers - so a rebuild needs a new
    // `LbvhStorage`.  This is the same "a compacted structure has no spare
    // capacity" rule applied to the scratch, and it is why the call is opt-in:
    // the default keeps the current rebuild-valid contract.  The radix sort's own
    // scratch is not released (~1 byte per primitive).
    [[nodiscard]] ReleasedScratch release_build_scratch() noexcept;

    // Whether the build scratch is still resident (false after
    // `release_build_scratch()`, which every build stage asserts on).
    [[nodiscard]] bool buildable() const noexcept {
        return static_cast<bool>(_prims) && static_cast<bool>(_keys_a) &&
               static_cast<bool>(_keys_b) && static_cast<bool>(_blocks) &&
               static_cast<bool>(_plan);
    }

    // Structural self-check of one built tree: every node must be reachable from
    // the root exactly once, and every internal node AABB must be the union of
    // its children's AABBs.  Returns the number of problems found.
    [[nodiscard]] size_t validate_tree(Stream &stream, uint node_base, uint count) noexcept;

    [[nodiscard]] const Buffer<LbvhPrim> &prims() const noexcept { return _prims; }
    [[nodiscard]] const Buffer<LbvhNode> &nodes() const noexcept { return _nodes; }
    [[nodiscard]] const Buffer<LbvhNode> &blocks() const noexcept { return _blocks; }
    [[nodiscard]] const Buffer<LbvhBlas> &blas_table() const noexcept { return _blas_table; }
    [[nodiscard]] const Buffer<LbvhInstance> &instances() const noexcept { return _instances; }
    // One node-count slot per reserved tree, written by the fused leaf pass (see
    // `_leaf_kernel`) and read back by `compact()`.
    [[nodiscard]] const Buffer<uint> &usage() const noexcept { return _usage; }
    [[nodiscard]] const Sizes &sizes() const noexcept { return _sizes; }
    [[nodiscard]] size_t primitive_count() const noexcept { return _prim_count; }
    [[nodiscard]] size_t node_count() const noexcept { return _node_count; }
    // Node slots the *live* node buffer holds: the reserved capacity until the
    // first `compact()`, exactly the kept node count afterwards (see `compact`).
    [[nodiscard]] size_t node_capacity() const noexcept { return _node_capacity; }
    // Trees reserved so far, i.e. the number of valid slots of `usage()`.
    [[nodiscard]] size_t tree_count() const noexcept { return _tree_count; }

private:
    // The device, needed by `compact()` to allocate the dense node buffer lazily
    // (the build itself only ever receives buffers).
    Device *_device{nullptr};
    Sizes _sizes;
    size_t _prim_count{0u};
    size_t _node_count{0u};
    size_t _plan_count{0u};
    // Number of trees reserved so far, i.e. the high-water mark of `_usage`.
    size_t _tree_count{0u};
    // `_nodes.size()`: the reserved capacity until a `compact()` adopts the dense
    // buffer, exactly the kept node count afterwards.
    size_t _node_capacity{0u};
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
    // One node-count slot per reserved tree, written by the fused leaf pass of
    // `build_tree()` and read back by `compact()` (see `_leaf_kernel`).
    Buffer<uint> _usage;
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
    Shader1D<Buffer<LbvhKey>, Buffer<LbvhPrim>, Buffer<LbvhNode>, Buffer<uint>, uint, uint,
             uint, uint>
        _leaf_kernel;
    // The dense copy of `compact()`: one thread per used node, `dst[i] = src[i]`.
    // With `CompactionPolicy::as_built` the destination index is the source index,
    // so this is a pure streaming copy - the kernel exists to make the bytes
    // *device-side* (the same reason the hardware backends copy on the device and
    // not through the host).
    //
    // lc_optimize note: `set_block_size(256)` plus a `$if (i < count)` tail guard
    // (the node count is `2n - 1` per tree and is not a multiple of the block
    // size).  A variant copying two nodes per thread as a 64-byte `float4` run -
    // the obvious way to add ILP to a pure copy - was considered and *not* kept:
    // the copy moves `used * 32` bytes (64 MiB for the 8 M-primitive catalogue,
    // <= ~0.3 ms of bandwidth) inside a 2.3-7.0 ms compaction that varies by
    // ~2.6 ms run to run (the exact-size allocation and the two-submission
    // readback/copy round trip dominate), so the variant is below the
    // measurement's resolution - and the repo's rule is to keep only a measured
    // win.  No warp intrinsics are involved: this is a streaming copy, not a
    // cross-lane reduction.
    Shader1D<Buffer<LbvhNode>, Buffer<LbvhNode>, uint> _copy_kernel;
    Shader1D<Buffer<LbvhKey>, Buffer<uint4>, uint, uint, uint, uint> _plan_kernel;
    Shader1D<Buffer<LbvhNode>, Buffer<LbvhNode>, uint, uint, uint> _block_kernel;
    Shader2D<Buffer<LbvhNode>, Buffer<LbvhNode>, Buffer<uint4>, uint, uint, uint, uint>
        _build_kernel;
};

}// namespace luisa::example::lbvh

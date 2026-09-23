// The storage shared by every tree of one fallback RTX device, plus the
// tree-agnostic build stages: the region planner, the Morton codes, the LSD
// radix sort (fallback_rtx_sort.h) and the Karras radix-tree construction.
//
// This is the port of examples/compute/lbvh/lbvh_storage.{h,cpp}.  The
// algorithm is that file's algorithm; what changed is the *layout* it writes
// (fallback_rtx_layout.h) and the fact that the sizes are only discovered at
// build time:
//
//   * one *acceleration buffer* (`Buffer<uint4>`) holds every tree of the
//     device, one *region* per tree.  Child handles are the bit pattern of the
//     child's *absolute* uint4 offset, so a traversal needs no descriptor
//     indexing and a TLAS region can descend into a BLAS region
//     (fallback_rtx_layout.h);
//   * a node is two uint4 - (lo.xyz, left) and (hi.xyz, right) - so the build
//     writes the AABB planes and the handles as *integers*, which is why the
//     packing helpers below differ from the example's float-Lane ones;
//   * the build scratch (primitive AABBs, Morton keys, the per-tree volume
//     reduction) is typed and shared, exactly like the example's `_prims` /
//     `_keys_a` / `_keys_b`;
//   * every buffer *grows by appending* (`detail::GrowableBuffer`): a larger
//     buffer is created, the live prefix is copied into the same offsets and
//     the old object is retired rather than destroyed.  An offset baked into a
//     shader descriptor therefore stays valid across a growth, and a command
//     that is still in flight can never observe freed device memory.  The cost
//     is a small, deliberate leak: the retired buffers live until the storage
//     is destroyed.
//
// Region planner (deterministic, append-only):
//
//   a region is laid out as `header | nodes | (index | vertex) or blas table`
//   and takes exactly the number of uint4 the layout header's
//   `blas_region_bytes()` / `tlas_region_bytes()` account for.  Regions are
//   handed out in call order and are never moved, so two builds of the same
//   scene produce the same offsets.
//
//   The *first* region starts at uint4 4, not at 0: the four uint4 at the
//   front are a reserved null region, which makes `base == 0` (the bit
//   pattern of a zeroed blas-table record) unambiguously mean "no tree"
//   (fallback_rtx_layout.h).

#pragma once

#include "fallback_rtx_layout.h"
#include "fallback_rtx_sort.h"

#include <luisa/core/logging.h>
#include <luisa/core/stl/vector.h>
#include <luisa/dsl/local.h>
#include <luisa/dsl/shared.h>
#include <luisa/dsl/struct.h>
#include <luisa/dsl/sugar.h>
#include <luisa/runtime/buffer.h>
#include <luisa/runtime/command_list.h>
#include <luisa/runtime/device.h>

#include <algorithm>
#include <bit>
#include <cstddef>
#include <utility>

// ---------------------------------------------------------------------------
// GPU-side build structures.
//
// Like the sort's key type these have to be at global scope (LUISA_STRUCT opens
// namespace luisa::compute).  A primitive AABB is the build's own input, not
// part of the ABI: the shaders never see it (fallback_rtx_layout.h stores the
// geometry, and the nodes carry the AABBs).
// ---------------------------------------------------------------------------

// Input of one LBVH: the primitive AABB plus the id copied into the leaf node
// (local triangle index for a BLAS, instance index for a TLAS).
struct FallbackRtxPrim {
    luisa::uint id;
    luisa::float3 lo;
    luisa::float3 hi;
};
LUISA_STRUCT(FallbackRtxPrim, id, lo, hi) {};

namespace lc::fallback_rtx {

using namespace luisa;
using namespace luisa::compute;

// ---------------------------------------------------------------------------
// Field access of a node pair in the acceleration buffer.
//
// A node `i` of a region is the two uint4 at `2i` and `2i + 1`.  These helpers
// are the only place that knows how a node packs its AABB and its handles, and
// they exist twice - once for the DSL (`UInt4` expressions, used by the build
// kernels) and once for the host (`uint4`, used by the validator) - because the
// validator walks a downloaded buffer outside the DSL.
// ---------------------------------------------------------------------------

[[nodiscard]] inline UInt4 pack_node_plane(Float3 plane, UInt handle) noexcept {
    return make_uint4(plane.x.bitcast<uint>(), plane.y.bitcast<uint>(),
                      plane.z.bitcast<uint>(), handle);
}

template<typename N>
[[nodiscard]] inline Float3 node_aabb_lo(N &&node) noexcept {
    return make_float3(node.x.template bitcast<float>(),
                       node.y.template bitcast<float>(),
                       node.z.template bitcast<float>());
}

template<typename N>
[[nodiscard]] inline Float3 node_aabb_hi(N &&node) noexcept {
    return make_float3(node.x.template bitcast<float>(),
                       node.y.template bitcast<float>(),
                       node.z.template bitcast<float>());
}

// Left handle of an internal node, `invalid_offset` for a leaf.
template<typename N>
[[nodiscard]] inline UInt node_child_left(N &&node) noexcept { return node.w; }

// Right handle of an internal node, the primitive id of a leaf (only ever read
// when `node_child_left()` says it is a leaf).
template<typename N>
[[nodiscard]] inline UInt node_child_right(N &&node) noexcept { return node.w; }

// ---------------------------------------------------------------------------
// The unit-cube mapping of the Morton codes needs the scene bounds, which the
// host does not know: the geometry of a BLAS and the world AABB of a TLAS are
// discovered by the build itself.  A float min/max has no portable atomic, but
// the *orderable* uint encoding of an IEEE-754 float has one - unsigned min/max
// is available on every backend - so the two stages communicate through a small
// reduction buffer instead of a host round trip:
//
//   key(x) = bits(x) ^ 0x80000000          for x >= 0
//   key(x) = ~bits(x)                      for x <  0
//
// which is monotone in x, so an unsigned min/max over keys is the float min/max.
// ---------------------------------------------------------------------------

[[nodiscard]] inline UInt orderable_key(Float x) noexcept {
    auto bits = x.bitcast<uint>();
    return select(bits ^ 0x80000000u, ~bits, (bits & 0x80000000u) != 0u);
}

[[nodiscard]] inline Float unorderable_key(UInt key) noexcept {
    return select(key ^ 0x80000000u, ~key, (key & 0x80000000u) == 0u).bitcast<float>();
}

namespace detail {

// ---------------------------------------------------------------------------
// A device buffer that only ever grows, by *appending*.
//
// Growth creates a larger buffer, copies the live prefix into the same offsets
// and retires the old object (`_retired`) instead of destroying it: a command
// that was already recorded - in a list the caller commits later, or in a stream
// another submission is still draining - may reference that memory, and the
// fallback has no fence of its own to wait for it.  Retiring costs one buffer
// per growth for the lifetime of the storage, which is the price of keeping the
// already-handed-out *offsets* valid (a shader descriptor baked at offset X
// keeps meaning element X after the storage grew).
// ---------------------------------------------------------------------------

template<typename T>
class GrowableBuffer {

public:
    GrowableBuffer(Device &device, size_t capacity) noexcept
        : _device{&device},
          _buffer{device.create_buffer<T>(std::max<size_t>(capacity, 1u))} {}

    [[nodiscard]] const Buffer<T> &buffer() const noexcept { return _buffer; }
    [[nodiscard]] size_t capacity() const noexcept { return _buffer.size(); }
    [[nodiscard]] size_t size_bytes() const noexcept { return _buffer.size_bytes(); }

    // Make room for `capacity` elements, recording the copy into `commands`.
    // `commands` must be the list that will also hold the first command touching
    // the grown range, so the copy is ordered before every use of it.
    void reserve(size_t capacity, CommandList &commands) noexcept {
        if (capacity <= _buffer.size()) { return; }
        auto grown = _device->create_buffer<T>(std::max(capacity, _buffer.size() * 2u));
        commands << grown.view(0u, _buffer.size()).copy_from(_buffer.view());
        _retired.emplace_back(std::move(_buffer));
        _buffer = std::move(grown);
    }

private:
    Device *_device{nullptr};
    Buffer<T> _buffer;
    luisa::vector<Buffer<T>> _retired;
};

}// namespace detail

// ---------------------------------------------------------------------------
// One tree's place in the shared buffers.
//
// Every field is a uint4 offset into the acceleration buffer unless it says
// otherwise; the offsets are absolute because that is what a node handle stores.
// ---------------------------------------------------------------------------
struct FallbackRtxRegion {
    uint base{};           // u4 offset of the region header
    uint node_base{};      // u4 offset of node 0
    uint node_count{};     // 2 * prim_count - 1
    uint prim_count{};     // leaves: triangles (BLAS) or instances (TLAS)
    uint index_base{};     // u4 offset of the (i0, i1, i2, 0) array (BLAS)
    uint index_count{};    // = prim_count (BLAS)
    uint vertex_base{};    // u4 offset of the (x, y, z, 0) array (BLAS)
    uint vertex_count{};   // vertices of the mesh (BLAS)
    uint blas_table_base{};// u4 offset of the blas-table records (TLAS)
    uint blas_count{};     // records the table holds (TLAS)
    uint prim_offset{};    // slice of the shared primitive scratch
    uint reduce_offset{};  // slice of the shared reduction scratch (uints)
    uint instance_offset{};// slice of the instance buffer (u4, TLAS)
    [[nodiscard]] uint region_u4() const noexcept {
        return header_u4 + node_u4 * node_count +
               index_count + vertex_count + blas_record_u4 * blas_count;
    }
};

// Reduction slot of one region, in uints.
inline constexpr uint reduce_slot_uints = 16u;
// Lanes of the reduction slot (see `orderable_key`).
inline constexpr uint reduce_min_key = 0u;   // 3 lanes: min keys of the AABB
inline constexpr uint reduce_max_key = 3u;   // 3 lanes: max keys of the AABB
inline constexpr uint reduce_volume_lo = 6u; // 3 lanes: scene min (float bits)
inline constexpr uint reduce_volume_inv = 9u;// 3 lanes: reciprocal extent

class FallbackRtxStorage {

public:
    explicit FallbackRtxStorage(Device &device) noexcept;

    // ---- region planner (host side; records the growth copies) -------------

    // Lay a BLAS out in the acceleration buffer: the header, `2n-1` nodes, `n`
    // triangle-index records and the mesh's own `vertex_count` copied vertices.
    // `commands` receives every growth the planning triggers.
    [[nodiscard]] FallbackRtxRegion plan_blas(CommandList &commands,
                                              size_t triangle_count,
                                              size_t vertex_count) noexcept;

    // Lay a TLAS out: the header, `2n-1` nodes and `n` blas-table records, plus
    // the tree's slice of the instance buffer.
    [[nodiscard]] FallbackRtxRegion plan_tlas(CommandList &commands,
                                              size_t instance_count) noexcept;

    // Append one blas-table record to the shared directory and return its *entry
    // index* (the u4 offset of the record is `blas_record_u4 * entry`, which is
    // what a kernel multiplies out).  The record is host-known once a BLAS region
    // has been planned, so it travels as two uniforms instead of an upload - an
    // upload would have to keep the host bytes alive until the command executed,
    // which a plan-time record cannot promise.
    [[nodiscard]] uint append_blas_directory(CommandList &commands,
                                             uint4 record_0, uint4 record_1) noexcept;

    // Write the region header.  The header is host-known at plan time, and this
    // is a kernel (not an upload) for the same lifetime reason as above.
    void write_region_header(CommandList &commands, const FallbackRtxRegion &region,
                             uint flags) noexcept;

    // Start the reduction slot of `region` at the identity of the min/max
    // reduction, so the primitive kernel of the tree can accumulate into it.
    void reset_reduction(CommandList &commands, const FallbackRtxRegion &region) noexcept;

    // ---- build stages ------------------------------------------------------

    // Stages 2..4 of a build, in one list: Morton codes, the 4 x 8-bit LSD
    // radix sort and the Karras radix tree (leaves, then internal nodes).  The
    // caller has already filled the primitive AABBs of `region` and started its
    // reduction slot; this encodes everything else, including the volume
    // reduction the Morton codes are normalized with.
    void build_tree(CommandList &commands, const FallbackRtxRegion &region) noexcept;

    // ---- introspection -----------------------------------------------------

    [[nodiscard]] Device &device() const noexcept { return *_device; }
    [[nodiscard]] const Buffer<uint4> &accel() const noexcept { return _accel.buffer(); }
    [[nodiscard]] const Buffer<uint4> &instances() const noexcept { return _instances.buffer(); }
    [[nodiscard]] const Buffer<FallbackRtxPrim> &prims() const noexcept { return _prims.buffer(); }
    [[nodiscard]] const Buffer<FallbackRtxKey> &keys_a() const noexcept { return _keys_a.buffer(); }
    [[nodiscard]] const Buffer<FallbackRtxKey> &keys_b() const noexcept { return _keys_b.buffer(); }
    [[nodiscard]] const Buffer<uint> &reduce() const noexcept { return _reduce.buffer(); }
    [[nodiscard]] const Buffer<uint4> &blas_directory() const noexcept { return _blas_directory.buffer(); }
    [[nodiscard]] const FallbackRtxSort &sort() const noexcept { return _sort; }
    // The acceleration buffer's live prefix: everything a validator has to read.
    [[nodiscard]] size_t accel_used_u4() const noexcept { return _accel_used; }
    [[nodiscard]] size_t instance_used_u4() const noexcept { return _instance_used; }
    [[nodiscard]] size_t accel_buffer_bytes() const noexcept { return _accel.size_bytes(); }
    [[nodiscard]] size_t instance_buffer_bytes() const noexcept { return _instances.size_bytes(); }
    [[nodiscard]] size_t blas_directory_entries() const noexcept { return _blas_directory_used; }

    // ---- host-side structural check ----------------------------------------

    // Everything a validator needs that does not come from the region header:
    // the downloaded buffers and the offset of the instance slice in the second
    // one.  They are a small struct rather than four parameters because the
    // check recurses into the BLAS regions a TLAS references.
    struct HostView {
        luisa::span<const uint4> accel;
        luisa::span<const uint4> instances;
        uint instance_base_u4{};
        // Number of records the device's blas directory holds, i.e. the upper
        // bound of the directory index an instance record may carry.
        size_t blas_directory_entries{};
    };

    // Walk the tree whose region header is the uint4 at `header_base` of
    // `view.accel` and report the number of structural problems found:
    //
    //   * the header must describe the region it starts (self-consistent
    //     `base` / `node_base` / counts and a `root` that is the node array);
    //   * every node must be reachable from the root exactly once, every handle
    //     must be in range, and every leaf must carry an in-range primitive id;
    //   * every internal node's AABB must be the union of its two children;
    //   * a BLAS must additionally carry in-range triangle indices, so that the
    //     geometry the region carries is internally consistent;
    //   * a TLAS must additionally resolve every instance's blas-table row, find
    //     a self-consistent BLAS region there (validated recursively), and find
    //     every instance's world AABB - the transformed BLAS root AABB - inside
    //     its own root AABB.
    //
    // The decode reads nothing outside the two spans, so a malformed tree is
    // reported instead of crashing the caller.
    [[nodiscard]] static size_t validate_tree(const HostView &view, uint header_base,
                                              bool expect_tlas) noexcept;

private:
    // The four buffers of the layout ABI plus the typed scratch.
    detail::GrowableBuffer<uint4> _accel;
    detail::GrowableBuffer<uint4> _instances;
    detail::GrowableBuffer<FallbackRtxPrim> _prims;
    detail::GrowableBuffer<FallbackRtxKey> _keys_a;
    detail::GrowableBuffer<FallbackRtxKey> _keys_b;
    detail::GrowableBuffer<uint> _reduce;
    detail::GrowableBuffer<uint4> _blas_directory;
    FallbackRtxSort _sort;
    Device *_device{nullptr};

    size_t _accel_used{header_u4};// the first region starts after a null region
    size_t _instance_used{0u};
    size_t _prim_used{0u};
    size_t _reduce_used{0u};
    size_t _blas_directory_used{0u};

    // Warp (wave/sub-group) width of the device, queried once.  The radix-tree
    // construction dispatches one warp per internal node, so the host side needs
    // the same constant the kernel's `warp_lane_count()` expands to.
    uint _warp_size{32u};

    // The identity of both min/max reductions, written by `reset_reduction`.
    Shader1D<Buffer<uint>, uint> _reset_kernel;
    // 30-bit Morton codes of the primitives of one tree.
    Shader1D<Buffer<FallbackRtxPrim>, Buffer<FallbackRtxKey>, Buffer<uint>,
             uint, uint, uint>
        _morton_kernel;
    // Scene bounds -> the unit cube, one thread.
    Shader1D<Buffer<uint>, uint> _volume_kernel;
    // Tree construction, pass 1 of 2: the leaves.
    Shader1D<Buffer<FallbackRtxKey>, Buffer<FallbackRtxPrim>, Buffer<uint4>,
             uint, uint, uint>
        _leaf_kernel;
    // Tree construction, pass 2 of 2: the internal nodes.
    Shader2D<Buffer<FallbackRtxKey>, Buffer<uint4>, uint, uint, uint, uint> _build_kernel;
    // The region header, from the planner's own numbers.
    Shader1D<Buffer<uint4>, uint4, uint4, uint4> _header_kernel;
    // One blas-table record of the shared directory, from two uniforms.
    Shader1D<Buffer<uint4>, uint4, uint4, uint> _directory_kernel;
};

}// namespace lc::fallback_rtx

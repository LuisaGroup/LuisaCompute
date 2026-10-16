// Shared GPU-side layout (and intersection helpers) of the two-level software
// LBVH.
//
// The whole acceleration structure is built *by hand* inside Luisa kernels and
// lives in plain `Buffer`s: no Luisa RTX mesh/accel API is involved.  The build
// follows the algorithm of
//
//   Tero Karras, "Maximizing Parallelism in the Construction of BVHs, Octrees,
//   and k-d Trees", High Performance Graphics 2012,
//
// as implemented by Mirco Werner's VkLBVH reference
// (https://github.com/MircoWerner/VkLBVH):
//
//   1. one AABB per primitive                                    [blas.cpp / tlas.cpp]
//   2. one 30-bit Morton code per primitive (AABB center -> unit cube)
//   3. LSD radix sort of the (code, slot) pairs, 4 passes x 8 bits, executed
//      by a single work-group
//   4. radix-tree construction: leaves, internal nodes and node AABBs
//      (delta / determineRange / findSplit, Karras 2012)         [lbvh_storage.cpp]
//
// Steps 2..4 never look at what a tree indexes, so they are shared by both
// levels (`LbvhStorage`); step 1 does - triangles for a BLAS, instances for a
// TLAS - and is therefore owned by `BlasBuilder` / `TlasBuilder`.
//
// A `Blas` is an LBVH over the triangles of one mesh, a `Tlas` is an LBVH over
// instances where every instance references a `Blas` and a transform.  All
// trees of one scene share a single primitive buffer and a single node buffer,
// each tree owning a contiguous range, and child pointers are stored as
// absolute node indices.  A traversal no longer indexes the shared node buffer
// directly, though: a `Tlas` owns a `BindlessArray` (its *heap*) whose slots
// hold the node region of every tree, and a BLAS is resolved by its *bindless
// slot* exactly like the fallback RTX backend resolves its regions
// (src/backends/common/rtx/fallback_rtx_layout.h).
//
// This header defines the data layout plus the ray/AABB and ray/triangle tests
// shared by the two traversal levels.

#pragma once

#include <luisa/luisa-compute.h>
#include <luisa/dsl/struct.h>
#include <luisa/dsl/sugar.h>
#include <luisa/dsl/rtx/triangle.h>

#include <bit>

// ---------------------------------------------------------------------------
// GPU-side layout.
//
// These structs must be declared at global scope: LUISA_STRUCT opens namespace
// luisa::compute to specialize the DSL types for the struct.
// ---------------------------------------------------------------------------

// Input of one LBVH: the primitive AABB plus the id copied into the leaf node
// (triangle index for a BLAS, instance index for a TLAS).
//
// The record is deliberately two `float4` - exactly 32 bytes, two 16-byte vector
// loads - with the id bit-cast into the spare fourth lane of the low plane, i.e.
// the same trick `LbvhNode` uses for its child handles (see below).  The obvious
// `uint id; float3 lo; float3 hi;` is 48 bytes, because `float3` is 16-byte
// aligned; that costs a third more memory *and* makes the random `prims[slot]`
// read of the radix-tree leaf pass touch two 32-byte L2 sectors instead of one -
// the leaf pass is the only random access of the whole build, so the sector is
// what it pays for.  A struct of scalars also risks being lowered as one scalar
// load per member on some backends, while a pair of vectors is one vector load
// each by construction.
struct LbvhPrim {
    luisa::float4 lo;// xyz = AABB lo, w = id (bit-cast)
    luisa::float4 hi;// xyz = AABB hi, w = unused
};
LUISA_STRUCT(LbvhPrim, lo, hi) {};

// One (Morton code, primitive slot) pair; `slot` indexes the primitive array of
// the tree being built.
struct LbvhKey {
    luisa::uint code;
    luisa::uint slot;
};
LUISA_STRUCT(LbvhKey, code, slot) {};

// LBVH node, 32 bytes.
//
// Nodes are laid out as [internal nodes 0 .. n-2 | leaves n-1 .. 2n-2] and a leaf
// is identified by a `invalid_node` left handle (`prim` is only meaningful there).
// Child pointers are absolute indices into the shared node buffer.
//
// The layout is deliberate: the four floats of each `float4` are one AABB plane
// (xyz) plus one 32-bit handle *bit-cast* into the fourth lane (w), so a node is
// exactly 32 bytes = exactly one 32-byte L2 sector, and two 16-byte aligned
// vector loads read all of it.  The obvious alternative - `float3 lo; float3 hi;
// uint left; uint right; uint prim;` - is 48 bytes, because `float3` is 16-byte
// aligned; a random node load then fetches *two* sectors (64 bytes of traffic for
// 32 useful ones).  Both the internal-node AABB reduction of the build and every
// step of the two traversals are dominated by random node loads, so the sector is
// the natural unit; see bench/README.md for the measured difference.
//
// * internal node: `packed_lo.w` = left child, `packed_hi.w` = right child,
// * leaf: `packed_lo.w` = `invalid_node`, `packed_hi.w` = primitive id.
//
// Handles are *bit-cast*, not value-cast, so the full 32-bit range and the
// `invalid_node` sentinel survive; the field accessors below are the only place
// that knows about the packing.
struct LbvhNode {
    luisa::float4 packed_lo;// xyz = AABB lo, w = left handle (internal) / invalid (leaf)
    luisa::float4 packed_hi;// xyz = AABB hi, w = right handle (internal) / primitive id (leaf)
};
LUISA_STRUCT(LbvhNode, packed_lo, packed_hi) {};

// The point of the layout above: a node is exactly one L2 sector, so a random
// node load never pays for a second one.
static_assert(sizeof(LbvhNode) == 32u && alignof(LbvhNode) == 16u,
              "an LBVH node must stay one 32-byte sector wide");

// One BLAS: where its node array starts, which triangle range it covers, and
// the bindless slot its node region occupies in the owning TLAS' heap (see "The
// bindless heap of a TLAS" below).  A traversal resolves the tree through
// `heap_slot`; `node_offset` is kept because it is the region's base (a node
// handle is absolute in the shared node buffer, see `LbvhNode`), and because the
// host-side build and the structural self-check address the tree through it.
struct LbvhBlas {
    luisa::uint node_offset;
    luisa::uint triangle_offset;
    luisa::uint triangle_count;
    luisa::uint heap_slot;
};
LUISA_STRUCT(LbvhBlas, node_offset, triangle_offset, triangle_count, heap_slot) {};

// One TLAS instance.  `float4x4` is deliberately avoided here: the transforms
// are stored as explicit *rows*, so the buffer layout is unambiguous.
// `to_object_*` are the rows of the world->object matrix (ray transform),
// `to_world_*` the rows of the object->world matrix (AABB corners).
struct LbvhInstance {
    luisa::float4 to_object_0;
    luisa::float4 to_object_1;
    luisa::float4 to_object_2;
    luisa::float4 to_world_0;
    luisa::float4 to_world_1;
    luisa::float4 to_world_2;
    luisa::uint blas;
};
LUISA_STRUCT(LbvhInstance, to_object_0, to_object_1, to_object_2,
             to_world_0, to_world_1, to_world_2, blas) {};

struct LbvhRay {
    luisa::float3 origin;
    luisa::float3 direction;
    float t_min;
    float t_max;
};
LUISA_STRUCT(LbvhRay, origin, direction, t_min, t_max) {};

// The closest hit of a traversal.  `inst == invalid_node` means "miss"; `prim`
// is only ever written together with a hit, so it stays at `invalid_node` while
// `inst` does.  `t` holds the upper distance bound of the traversal (the ray's
// `t_max`) on a miss.
struct LbvhHit {
    luisa::uint inst;
    luisa::uint prim;
    luisa::float2 bary;
    float t;
};
LUISA_STRUCT(LbvhHit, inst, prim, bary, t) {};

namespace luisa::example::lbvh {

using namespace luisa;
using namespace luisa::compute;

// Marks an empty child pointer, i.e. a leaf node (`LbvhNode::left`), and a miss
// in `LbvhHit::inst` / `LbvhHit::prim`.
inline constexpr uint invalid_node = 0xFFFFFFFFu;
// ---------------------------------------------------------------------------
// The bindless heap of a TLAS.
//
// This is the software mirror of the fallback RTX layout
// (src/backends/common/rtx/fallback_rtx_layout.h): a TLAS owns a
// `BindlessArray` whose slots hold the *node regions* of the trees, and a
// `LbvhBlas` record carries the bindless slot of the region it names.  A
// traversal therefore resolves a BLAS through the heap instead of through an
// absolute node offset in one shared node buffer.  A heap entry is a view that
// *starts at the region*, so a TLAS may reference a BLAS laid out before it -
// the constraint a single shared buffer with absolute handles cannot express.
//
// The slot assignment is fixed, so a traversal names the TLAS' own region
// without a second descriptor:
//
//   slot 0 : the null slot - no region.  A `LbvhBlas` record whose slot lane is
//            0 is a BLAS the caller never gave geometry, which a traversal
//            skips exactly like the fallback's null `blas_base`.
//   slot 1 : the TLAS' own node region.
//   slot 2 + i : the node region of BLAS `i`.  The BLAS table of this example
//            is indexed by BLAS (not by instance, as the fallback's table is),
//            so the slot is the BLAS' own index rather than an instance's.
// ---------------------------------------------------------------------------
inline constexpr uint heap_null_slot = 0u;
inline constexpr uint heap_tlas_slot = 1u;
inline constexpr uint heap_first_blas_slot = 2u;
// Work-group size of the radix sort (one single work-group sorts a whole tree).
inline constexpr uint sort_block_size = 256u;
inline constexpr uint sort_radix_bins = 256u;
// Upper bound of one grid dimension of the radix-tree construction.  That kernel
// spends one warp on every internal node, i.e. `primitive_count * warp_size`
// threads, which for any tree of a few hundred thousand primitives is a grid
// larger than the 65535 work-groups per dimension DirectX 12 allows for a single
// Dispatch().  The grid of that kernel is therefore two-dimensional and folded at
// this many work-groups per dimension.
inline constexpr uint max_build_dispatch_groups = 65535u;
// Software traversal stack; a Morton-code radix tree is far shallower than this.
inline constexpr uint traversal_stack_size = 64u;
// Leaves per block of the two-level reduction of the internal-node AABB pass
// (`LbvhStorage::build_tree`).  The AABB of an internal node is the union of the
// leaf AABBs of its range, and a node's range is covered exactly once by
// [range.x, the next block boundary), [the previous block boundary, range.y] and
// the *whole* blocks in between; the two partial ends cost one warp-load each
// and the middle costs one block AABB (see `_build_kernel`) instead of one leaf
// per lane.  The block AABBs are built from the leaves by `_block_kernel`, so the
// whole pass still reads every leaf exactly once plus the blocks of the prefix
// sums - the O(sum of leaf depths) leaf traffic of the direct reduction becomes
// O(count + count / block_size * mean_depth).  The value is the warp width so
// that a partial end is exactly one warp-load on every backend, and the union is
// bit-identical to the direct reduction (min/max are exact and associative).
inline constexpr uint node_reduction_block = 32u;

// ---------------------------------------------------------------------------
// Field access of `LbvhNode` (see its definition above for the packing).
//
// These are templates so that they accept both a `Var<LbvhNode>` and the
// `Expr<LbvhNode>` that `Buffer::read()` returns, and they are the *only* place
// that knows how a node packs its AABB and its handles: the build's two
// radix-tree passes, both traversals and the benchmark's instrumented mirror all
// go through them, so a change of the layout is a change of this block.
// ---------------------------------------------------------------------------

template<typename N>
[[nodiscard]] inline Float3 aabb_lo(N &&node) noexcept {
    return make_float3(node.packed_lo.x, node.packed_lo.y, node.packed_lo.z);
}

template<typename N>
[[nodiscard]] inline Float3 aabb_hi(N &&node) noexcept {
    return make_float3(node.packed_hi.x, node.packed_hi.y, node.packed_hi.z);
}

// Left child handle of an internal node, `invalid_node` for a leaf.
template<typename N>
[[nodiscard]] inline UInt child_left(N &&node) noexcept {
    return node.packed_lo.w.template bitcast<uint>();
}

// Right child handle of an internal node, the primitive id of a leaf (only ever
// read when `child_left()` says it is a leaf).
template<typename N>
[[nodiscard]] inline UInt child_right(N &&node) noexcept {
    return node.packed_hi.w.template bitcast<uint>();
}

template<typename N>
[[nodiscard]] inline Bool is_leaf(N &&node) noexcept {
    return child_left(node) == invalid_node;
}

// ---------------------------------------------------------------------------
// Field access of `LbvhPrim` (see its definition above for the packing): the
// record is two `float4`, with the primitive id bit-cast into the fourth lane of
// the low plane, and these accessors are the only place that knows it.
// ---------------------------------------------------------------------------

template<typename P>
[[nodiscard]] inline Float3 prim_lo(P &&prim) noexcept {
    return make_float3(prim.lo.x, prim.lo.y, prim.lo.z);
}

template<typename P>
[[nodiscard]] inline Float3 prim_hi(P &&prim) noexcept {
    return make_float3(prim.hi.x, prim.hi.y, prim.hi.z);
}

// The primitive id the leaf node carries (triangle index / instance index).
template<typename P>
[[nodiscard]] inline UInt prim_id(P &&prim) noexcept {
    return prim.lo.w.template bitcast<uint>();
}

// Bit-cast a handle into the lane it is stored in (the build side of the
// packing).  `pack_invalid_handle()` is the leaf marker: a *constant* rather than
// a float literal, so no NaN ever reaches a shader source.
[[nodiscard]] inline Float pack_handle(UInt handle) noexcept {
    return handle.bitcast<float>();
}

[[nodiscard]] inline Float pack_invalid_handle() noexcept {
    return def(invalid_node).bitcast<float>();
}

[[nodiscard]] inline Float4 pack_node_plane(Float3 plane, UInt handle) noexcept {
    return make_float4(plane.x, plane.y, plane.z, pack_handle(handle));
}

// Host-side reads of the same lanes, for the structural self-check (which walks a
// read-back node array outside the DSL, so it cannot use the templates above).
[[nodiscard]] inline uint host_child_left(const LbvhNode &node) noexcept {
    return luisa::bit_cast<uint>(node.packed_lo.w);
}

[[nodiscard]] inline uint host_child_right(const LbvhNode &node) noexcept {
    return luisa::bit_cast<uint>(node.packed_hi.w);
}

[[nodiscard]] inline float3 host_aabb_lo(const LbvhNode &node) noexcept {
    return make_float3(node.packed_lo.x, node.packed_lo.y, node.packed_lo.z);
}

[[nodiscard]] inline float3 host_aabb_hi(const LbvhNode &node) noexcept {
    return make_float3(node.packed_hi.x, node.packed_hi.y, node.packed_hi.z);
}

// ---------------------------------------------------------------------------
// Host-side build-size query of one LBVH.
//
// This is the software counterpart of the size query the hardware backends
// issue before they build an acceleration structure:
//
//   * Vulkan : vkGetAccelerationStructureBuildSizesKHR() ->
//              VkAccelerationStructureBuildSizesInfoKHR
//              (accelerationStructureSize, build/updateScratchSize)
//   * DirectX: ID3D12Device::GetRaytracingAccelerationStructurePrebuildInfo()
//              ->
//              D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO
//              (ResultDataMaxSizeInBytes, ScratchDataSizeInBytes)
//
// Like those, it answers "how large must the buffers be at most?" without
// touching the device, so the caller can allocate (or, here, reserve ranges in
// the shared storage) *before* creating and building the tree.
//
// The acceleration structure itself is the node buffer; the primitive AABBs
// and the two ping-pong Morton-key buffers are build scratch: they are written
// by the build and only read back by the traversal through the nodes.
// ---------------------------------------------------------------------------
struct LbvhBuildSizes {
    size_t primitive_count{};             // leaves of the tree
    size_t node_count{};                  // 2 * primitive_count - 1
    size_t acceleration_structure_bytes{};// node buffer (the "accel buffer")
    size_t scratch_bytes{};               // primitive AABBs + 2 x Morton keys
};

// Maximum buffer sizes an LBVH over `primitive_count` primitives needs.
[[nodiscard]] inline LbvhBuildSizes lbvh_build_sizes(size_t primitive_count) noexcept {
    LUISA_ASSERT(primitive_count > 0u, "an LBVH needs at least one primitive.");
    LbvhBuildSizes sizes;
    sizes.primitive_count = primitive_count;
    sizes.node_count = primitive_count * 2u - 1u;
    sizes.acceleration_structure_bytes = sizes.node_count * sizeof(LbvhNode);
    sizes.scratch_bytes = primitive_count * (sizeof(LbvhPrim) + 2u * sizeof(LbvhKey));
    return sizes;
}

// ---------------------------------------------------------------------------
// Opt-in per-stage build timings.
//
// The build is a chain of four dependent kernels (primitive AABBs, Morton
// codes, the four radix-sort passes and the radix-tree construction), and which
// one dominates is not visible from outside: the whole chain is recorded into
// one stream and there is no fence between the stages.  A caller that passes a
// non-null pointer therefore asks the build to *synchronise between the stages*
// and to accumulate the host-observed time of each one, which is what the
// benchmark needs to attribute the build cost to a stage.
//
// The values are wall-clock and include the per-stage submission, so they are
// only meaningful in release builds; `timings == nullptr` takes exactly the
// untimed code path (no extra synchronisation), so the timing hook costs
// nothing to the users that do not ask for it.  The fields *accumulate*, so a
// whole scene (K BLASes + 1 TLAS) can be timed into a single record.
// ---------------------------------------------------------------------------
struct LbvhBuildTimings {
    double prim_ms{0.0};  // primitive AABB kernel (only the BLAS/TLAS builder knows it)
    double morton_ms{0.0};// Morton codes of every primitives
    double sort_ms{0.0};  // the 4 LSD radix-sort passes together
    double node_ms{0.0};  // radix-tree construction (leaves, internals, AABBs)
    [[nodiscard]] double total_ms() const noexcept {
        return prim_ms + morton_ms + sort_ms + node_ms;
    }
};

// A row of a 4x4 matrix as a float4, for the explicit row-major transforms of
// `LbvhInstance`.
[[nodiscard]] inline float4 matrix_row(const float4x4 &m, size_t row) noexcept {
    return make_float4(m[0][row], m[1][row], m[2][row], m[3][row]);
}

// ---------------------------------------------------------------------------
// Ray intersection primitives used by both traversal levels.  These are plain
// C++ helpers that emit DSL expressions into the enclosing function; they don't
// need to be `Callable`s.
// ---------------------------------------------------------------------------

// Slab test against an AABB, bounded by the current best hit.
[[nodiscard]] inline Bool aabb_test(Float3 lo, Float3 hi, Float3 origin, Float3 inv_dir,
                                    Float t_min, Float t_max) noexcept {
    auto t0 = (lo - origin) * inv_dir;
    auto t1 = (hi - origin) * inv_dir;
    auto near_t = min(t0, t1);
    auto far_t = max(t0, t1);
    auto t_near = max(max(near_t.x, near_t.y), max(near_t.z, t_min));
    auto t_far = min(min(far_t.x, far_t.y), min(far_t.z, t_max));
    return t_near <= t_far;
}

// Two-sided Moller-Trumbore; returns (t, u, v) with t < 0 on a miss.  (u, v)
// follow the Luisa barycentric convention used by `triangle_interpolate`, i.e.
// w0 = 1 - u - v, w1 = u, w2 = v.
[[nodiscard]] inline Float3 triangle_test(Float3 v0, Float3 v1, Float3 v2, Float3 origin,
                                          Float3 dir, Float t_min, Float t_max) noexcept {
    auto e1 = v1 - v0;
    auto e2 = v2 - v0;
    auto pv = cross(dir, e2);
    auto det = dot(e1, pv);
    auto inv_det = 1.0f / det;
    auto tv = origin - v0;
    auto u = dot(tv, pv) * inv_det;
    auto qv = cross(tv, e1);
    auto v = dot(dir, qv) * inv_det;
    auto t = dot(e2, qv) * inv_det;
    auto hit = (abs(det) > 1.0e-12f) & (u >= 0.0f) & (v >= 0.0f) &
               (u + v <= 1.0f) & (t >= t_min) & (t <= t_max);
    return select(make_float3(-1.0f, 0.0f, 0.0f), make_float3(t, u, v), hit);
}

// Reciprocal that never produces a NaN from a zero direction.
[[nodiscard]] inline Float3 safe_reciprocal(Float3 d) noexcept {
    return 1.0f / make_float3(
                      select(d.x, 1.0e-20f, abs(d.x) < 1.0e-20f),
                      select(d.y, 1.0e-20f, abs(d.y) < 1.0e-20f),
                      select(d.z, 1.0e-20f, abs(d.z) < 1.0e-20f));
}

}// namespace luisa::example::lbvh

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
// trees of one scene share a single primitive buffer / node buffer, each tree
// owning a contiguous range; child pointers are stored as absolute node
// indices, so a traversal needs no descriptor indexing.
//
// This header defines the data layout plus the ray/AABB and ray/triangle tests
// shared by the two traversal levels.

#pragma once

#include <luisa/luisa-compute.h>
#include <luisa/dsl/struct.h>
#include <luisa/dsl/sugar.h>
#include <luisa/dsl/rtx/triangle.h>

// ---------------------------------------------------------------------------
// GPU-side layout.
//
// These structs must be declared at global scope: LUISA_STRUCT opens namespace
// luisa::compute to specialize the DSL types for the struct.
// ---------------------------------------------------------------------------

// Input of one LBVH: the primitive AABB plus the id copied into the leaf node
// (triangle index for a BLAS, instance index for a TLAS).
struct LbvhPrim {
    luisa::uint id;
    luisa::float3 lo;
    luisa::float3 hi;
};
LUISA_STRUCT(LbvhPrim, id, lo, hi) {};

// One (Morton code, primitive slot) pair; `slot` indexes the primitive array of
// the tree being built.
struct LbvhKey {
    luisa::uint code;
    luisa::uint slot;
};
LUISA_STRUCT(LbvhKey, code, slot) {};

// LBVH node: nodes are laid out as [internal nodes 0 .. n-2 | leaves n-1 .. 2n-2]
// and `left == lbvh::invalid_node` identifies a leaf (`prim` is only meaningful
// there).  Child pointers are absolute indices into the shared node buffer.
struct LbvhNode {
    luisa::float3 lo;
    luisa::float3 hi;
    luisa::uint left;
    luisa::uint right;
    luisa::uint prim;
};
LUISA_STRUCT(LbvhNode, lo, hi, left, right, prim) {};

// One BLAS: where its node array starts, and which triangle range it covers.
struct LbvhBlas {
    luisa::uint node_offset;
    luisa::uint triangle_offset;
    luisa::uint triangle_count;
};
LUISA_STRUCT(LbvhBlas, node_offset, triangle_offset, triangle_count) {};

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
// Work-group size of the radix sort (one single work-group sorts a whole tree).
inline constexpr uint sort_block_size = 256u;
inline constexpr uint sort_radix_bins = 256u;
// Software traversal stack; a Morton-code radix tree is far shallower than this.
inline constexpr uint traversal_stack_size = 92u;

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

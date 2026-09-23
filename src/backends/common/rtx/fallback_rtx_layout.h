// Fallback (software) ray tracing: the GPU-side memory ABI.
//
// This file is the single source of truth of the fallback acceleration-structure
// layout.  Three code generators walk the same bytes:
//
//   * the build kernels of this module (Luisa DSL, fallback_rtx_storage.cpp /
//     fallback_rtx_blas.cpp / fallback_rtx_tlas.cpp),
//   * the HLSL traversal emitted into DX shaders (src/backends/common/hlsl/builtin/
//     fallback_rtx_header.bytes, shared with the Vulkan HLSL->SPIR-V route),
//   * the CUDA traversal in cuda_builtin/cuda_device_resource.h.
//
// The HLSL and CUDA copies cannot include this header, so every rule below is
// restated there; a change of the layout is a change of three files and of the
// `static_assert`s in fallback_rtx_layout_contract.cpp, which pins the byte
// offsets the two text copies hard-code.
//
// ---------------------------------------------------------------------------
// Buffers
// ---------------------------------------------------------------------------
//
// One *acceleration buffer* (`Buffer<uint4>`) holds every tree of the device.
// It is grow-only: a tree region is appended and never moved, so the byte offset
// baked into a shader descriptor stays valid for the lifetime of the tree (the
// buffer object is replaced when the storage grows, but the *offsets* are kept).
//
// One *instance buffer* (`Buffer<uint4>`) holds the TLAS instance records.
//
// Region layout (all offsets are absolute uint4 indices into the acceleration
// buffer; the region starts with its own base, which is how a traversal converts
// an absolute handle back into an index of the descriptor that starts at the
// region):
//
//   -----------------------------------------------------------------------
//   u4   0 : (base,             node_base,        node_count,     prim_count)
//   u4   1 : (blas_table_base,  blas_count,       index_base,     index_count)
//   u4   2 : (vertex_base,      vertex_count,     root,           flags)
//   u4   3 : reserved
//   -----------------------------------------------------------------------
//   node array      : node_count nodes, `node_u4` = 2 u4 each
//   blas table      : blas_count records, `blas_record_u4` = 2 u4 each (TLAS only)
//   index array     : index_count `(i0, i1, i2, 0)` u4 entries     (BLAS only)
//   vertex array    : vertex_count `(x, y, z, 0)` u4 entries       (BLAS only)
//   -----------------------------------------------------------------------
//
// `base` is the absolute uint4 offset of u4 0 of this region: a traversal loads
// it once (`base = accel[0].x`) and reads every other absolute handle `h` as
// `accel[h - base]`, because the descriptor handed to the shader starts at the
// region and not at the buffer.
//
// A node (32 bytes, two u4 - one L2 sector) packs its AABB planes and its child
// handles exactly like the hardware-shaped LBVH of examples/compute/lbvh:
//
//   node[2i + 0] = (lo.x, lo.y, lo.z, left_handle)
//   node[2i + 1] = (hi.x, hi.y, hi.z, right_handle)
//
//   * internal node: left/right hold the *absolute* u4 offset of the child node,
//   * leaf: left holds `invalid_offset`, right holds the triangle index inside the
//     BLAS the leaf belongs to (a TLAS leaf holds the instance index instead).
//
// Handles are the bit pattern of a uint, never a float: the leaf marker has to
// survive the round trip, and a float literal for it would compile to a NaN in a
// shader source.
//
// A blas-table record describes the tree an instance refers to:
//
//   record[2i + 0] = (blas_base, node_base, index_base, vertex_base)
//   record[2i + 1] = (triangle_count, flags, reserved, reserved)
//
// `blas_base` is the base of the referenced BLAS region, so a traversal that
// descends into it can load `base = accel[blas_base - base].x` and continues with
// the BLAS' own frame of reference.
//
// `blas_base == 0` is the *null* reference: the build reserves the four uint4 at
// the front of the acceleration buffer as a dummy region, so no real tree ever
// starts at 0, and a record whose first lane is 0 is an instance the caller never
// gave a mesh.  A traversal must skip such a row instead of descending into
// region 0, and the fallback's validator reports it.
//
// An instance record (128 bytes, eight u4) is the object the shader may read and
// write (RAY_TRACING_INSTANCE_* / RAY_TRACING_SET_INSTANCE_*):
//
//   inst[8i + 0] = to_object_0   (float4 bits)
//   inst[8i + 1] = to_object_1
//   inst[8i + 2] = to_object_2
//   inst[8i + 3] = to_world_0
//   inst[8i + 4] = to_world_1
//   inst[8i + 5] = to_world_2
//   inst[8i + 6] = (blas_index, visibility_mask, user_id, flags)
//   inst[8i + 7] = reserved
//
// `blas_index` is the row of the region's blas table that describes the tree this
// instance refers to, and the build keeps one record per instance, so
// `blas_index == i` and `blas_count == instance_count` always hold: a traversal
// may index the table with the instance index (the `right` lane of a TLAS leaf)
// or with `blas_index`, and reads the same record either way.  The reserved u4 is
// private to the build - it carries the index of the BLAS inside the device's blas
// directory, which is what a rebuild resolves the table rows against - and no
// traversal reads it.
//
// `to_object_*` / `to_world_*` are the explicit *rows* of the world->object and
// object->world matrices, i.e. the same convention the LBVH example uses, so the
// ray transform and the AABB corners are unambiguous.

#pragma once

#include <luisa/core/basic_types.h>
#include <luisa/core/logging.h>
#include <luisa/dsl/struct.h>
#include <luisa/dsl/sugar.h>

#include <bit>
#include <cstddef>

namespace lc::fallback_rtx {

using namespace luisa;
using namespace luisa::compute;

// Left handle of a leaf node, and the "no hit" marker of a traversal.
inline constexpr uint invalid_offset = 0xFFFFFFFFu;
// u4 slots of a region header.
inline constexpr uint header_u4 = 4u;
// u4 slots of one node.
inline constexpr uint node_u4 = 2u;
// u4 slots of one blas-table record.
inline constexpr uint blas_record_u4 = 2u;
// u4 slots of one instance record.
inline constexpr uint instance_u4 = 8u;

// Header slot indices (in uints, i.e. four per u4).
inline constexpr uint h_base = 0u;
inline constexpr uint h_node_base = 1u;
inline constexpr uint h_node_count = 2u;
inline constexpr uint h_prim_count = 3u;
inline constexpr uint h_blas_table_base = 4u;
inline constexpr uint h_blas_count = 5u;
inline constexpr uint h_index_base = 6u;
inline constexpr uint h_index_count = 7u;
inline constexpr uint h_vertex_base = 8u;
inline constexpr uint h_vertex_count = 9u;
inline constexpr uint h_root = 10u;
inline constexpr uint h_flags = 11u;

// Instance-record slot offsets (in u4, relative to the instance).
inline constexpr uint i_to_object = 0u;
inline constexpr uint i_to_world = 3u;
inline constexpr uint i_misc = 6u;
// Misc lanes.
inline constexpr uint im_blas_index = 0u;
inline constexpr uint im_visibility = 1u;
inline constexpr uint im_user_id = 2u;
inline constexpr uint im_flags = 3u;

// Instance flags (mirror of `LC_INSTANCE_FLAG_*`); the build records them so the
// traversal can honour the same two behaviours the hardware does.
inline constexpr uint instance_flag_disable_face_culling = 1u << 0u;
inline constexpr uint instance_flag_flip_facing = 1u << 1u;
inline constexpr uint instance_flag_disable_any_hit = 1u << 2u;
inline constexpr uint instance_flag_enforce_any_hit = 1u << 3u;
// A non-opaque instance still contributes a hit: the fallback has no any-hit
// shading, so opacity only affects `enforce_any_hit` handling.  Recorded so the
// flag survives, and documented as a limitation in the traversal header.
inline constexpr uint instance_flag_opaque = 1u << 4u;

// Region flags.
// A BLAS region (no blas table) and a TLAS region differ only by this bit and by
// the fields the build fills in.
inline constexpr uint region_flag_tlas = 1u << 0u;

// ---------------------------------------------------------------------------
// Host-side region sizes.
//
// The storage planner (fallback_rtx_storage.cpp) uses these to lay a region out;
// they are also what the byte-offset `static_assert`s of the contract TU pin.
// ---------------------------------------------------------------------------

[[nodiscard]] inline constexpr size_t region_header_bytes() noexcept {
    return header_u4 * 16u;
}

// Bytes one LBVH over `prim_count` primitives needs: header, `2n-1` nodes,
// `n` index u4 and `n` vertex u4 (a BLAS only).
[[nodiscard]] inline constexpr size_t blas_region_bytes(size_t prim_count,
                                                        size_t vertex_count) noexcept {
    return region_header_bytes() +
           (prim_count * 2u - 1u) * (node_u4 * 16u) +
           prim_count * 16u + vertex_count * 16u;
}

// Bytes one LBVH over `instance_count` instances needs (a TLAS only).
[[nodiscard]] inline constexpr size_t tlas_region_bytes(size_t instance_count) noexcept {
    return region_header_bytes() +
           (instance_count * 2u - 1u) * (node_u4 * 16u) +
           instance_count * (blas_record_u4 * 16u);
}

static_assert(node_u4 * 16u == 32u, "a fallback node occupies one 32-byte sector");
static_assert(instance_u4 * 16u == 128u, "a fallback instance record is 128 bytes");
static_assert(header_u4 * 16u == 64u, "a fallback region header is 64 bytes");

}// namespace lc::fallback_rtx

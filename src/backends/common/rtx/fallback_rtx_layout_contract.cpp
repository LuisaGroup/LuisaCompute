// The ABI contract of the fallback acceleration structure, pinned as byte
// offsets.
//
// fallback_rtx_layout.h is the single source of truth of this ABI, but two of
// the three implementations that walk the same bytes *cannot* include it:
// the HLSL traversal emitted into DX/Vulkan shaders and the CUDA traversal in
// cuda_builtin/cuda_device_resource.h restate every offset as a literal.  This
// translation unit is where those literals are checked: it asserts the numeric
// values a change of the layout would have to move, so a layout change cannot
// pass silently into two text copies that nobody re-reads.
//
// Nothing here is executable; the file exists to be compiled.

#include "fallback_rtx_layout.h"

#include <cstddef>

namespace lc::fallback_rtx {

namespace {

// ---------------------------------------------------------------------------
// The primitives of the layout: a region is written and read in uint4 units, and
// every offset a shader hard-codes is a byte offset into that stream.
// ---------------------------------------------------------------------------

static_assert(sizeof(uint4) == 16u && alignof(uint4) == 16u,
              "the fallback ABI is a stream of 16-byte uint4");
static_assert(sizeof(float4) == 16u && alignof(float4) == 16u,
              "a node plane and an instance transform row are bit-cast uint4/float4");

static_assert(header_u4 == 4u, "a region header is four uint4");
static_assert(node_u4 == 2u, "a node is two uint4");
static_assert(blas_record_u4 == 2u, "a blas-table record is two uint4");
static_assert(instance_u4 == 8u, "an instance record is eight uint4");

static_assert(region_header_bytes() == 64u, "the region header is 64 bytes");
static_assert(node_u4 * 16u == 32u, "a node is 32 bytes (one L2 sector)");
static_assert(blas_record_u4 * 16u == 32u, "a blas-table record is 32 bytes");
static_assert(instance_u4 * 16u == 128u, "an instance record is 128 bytes");

// ---------------------------------------------------------------------------
// The region header: slot `k` is the uint at byte offset 4 * k from the region's
// first byte.  The traversal copies load these as `header[k]` of a 12-uint
// vector.
// ---------------------------------------------------------------------------

[[nodiscard]] constexpr size_t header_slot_bytes(uint slot) noexcept {
    return static_cast<size_t>(slot) * sizeof(uint);
}

static_assert(header_slot_bytes(h_base) == 0u);
static_assert(header_slot_bytes(h_node_base) == 4u);
static_assert(header_slot_bytes(h_node_count) == 8u);
static_assert(header_slot_bytes(h_prim_count) == 12u);
static_assert(header_slot_bytes(h_blas_table_base) == 16u);
static_assert(header_slot_bytes(h_blas_count) == 20u);
static_assert(header_slot_bytes(h_index_base) == 24u);
static_assert(header_slot_bytes(h_index_count) == 28u);
static_assert(header_slot_bytes(h_vertex_base) == 32u);
static_assert(header_slot_bytes(h_vertex_count) == 36u);
static_assert(header_slot_bytes(h_root) == 40u);
static_assert(header_slot_bytes(h_flags) == 44u);
static_assert(header_slot_bytes(h_flags) + sizeof(uint) <= region_header_bytes(),
              "every header slot is inside the 64-byte header");

// ---------------------------------------------------------------------------
// A node: (lo.xyz, left) then (hi.xyz, right).  The two handles are the *fourth
// lane* of each uint4, i.e. byte 12 and byte 28 of the 32-byte node, and the
// AABB planes are the first three lanes of each.
// ---------------------------------------------------------------------------

struct ContractNode {
    uint4 lo_handle;// xyz = AABB lo (float bits), w = left handle
    uint4 hi_handle;// xyz = AABB hi (float bits), w = right handle (leaf: prim id)
};
static_assert(sizeof(ContractNode) == 32u, "a node is 32 bytes");
static_assert(offsetof(ContractNode, lo_handle) == 0u);
static_assert(offsetof(ContractNode, hi_handle) == 16u);
static_assert(offsetof(ContractNode, lo_handle) + 3u * sizeof(uint) == 12u,
              "the left handle is lane 3 of the first uint4");
static_assert(offsetof(ContractNode, hi_handle) + 3u * sizeof(uint) == 28u,
              "the right handle is lane 3 of the second uint4");

// A leaf is a node whose left handle is this marker; a traversal also uses the
// same value as its "no hit" answer, which is why it must not be a float literal.
static_assert(invalid_offset == 0xFFFFFFFFu, "the leaf/empty marker is all ones");
// The marker must be unreachable as a real handle: a region is four uint4 of
// header plus at least one node, and the acceleration buffer would have to be
// 2^32 uint4 (64 GiB) wide before an offset could collide with it.
static_assert(invalid_offset > static_cast<uint>(region_header_bytes() / 16u));

// ---------------------------------------------------------------------------
// A blas-table record: where the referenced tree lives, and how much of it.
// ---------------------------------------------------------------------------

struct ContractBlasRecord {
    uint4 geometry;// (blas_base, node_base, index_base, vertex_base) - all uint4 offsets
    uint4 misc;    // (triangle_count, flags, heap_slot, reserved)
};
static_assert(sizeof(ContractBlasRecord) == 32u);
static_assert(offsetof(ContractBlasRecord, geometry) == 0u);
static_assert(offsetof(ContractBlasRecord, misc) == 16u);
static_assert(offsetof(ContractBlasRecord, geometry) + 0u * sizeof(uint) == 0u);
static_assert(offsetof(ContractBlasRecord, geometry) + 2u * sizeof(uint) == 8u);
static_assert(offsetof(ContractBlasRecord, misc) + 0u * sizeof(uint) == 16u,
              "a traversal reads the triangle count at byte 16 of the record");
// The record's metadata lanes: `misc.z` is the *bindless slot* of the referenced
// region, which is what a traversal resolves the tree through (the HLSL and CUDA
// copies read it at byte 24) and what a host-side validator never reads.
static_assert(bm_triangle_count == 0u && bm_flags == 1u && bm_heap_slot == 2u,
              "the blas-table metadata lanes are triangle_count, flags, heap_slot");
static_assert(offsetof(ContractBlasRecord, misc) + bm_heap_slot * sizeof(uint) == 24u,
              "the bindless heap slot of a blas-table record is at byte 24");

// ---------------------------------------------------------------------------
// An instance record: three world->object rows, three object->world rows, the
// property uint4 and the builder's private uint4.  The row convention is the
// examples' LBVH one: `to_world_k` is row `k` of the object->world matrix, so a
// point is transformed as `(dot(p, w0), dot(p, w1), dot(p, w2))` with
// `p = (x, y, z, 1)`.
// ---------------------------------------------------------------------------

struct ContractInstance {
    uint4 to_object[3];
    uint4 to_world[3];
    uint4 misc;
    uint4 reserved;
};
static_assert(sizeof(ContractInstance) == 128u, "an instance record is 128 bytes");
static_assert(offsetof(ContractInstance, to_object) == 0u);
static_assert(offsetof(ContractInstance, to_world) == 48u,
              "the object->world rows start at byte 48");
static_assert(offsetof(ContractInstance, misc) == 96u,
              "the instance property lanes are at byte 96");
static_assert(offsetof(ContractInstance, reserved) == 112u,
              "the reserved uint4 is the last 16 bytes");

[[nodiscard]] constexpr size_t instance_lane_bytes(uint u4_slot, uint lane) noexcept {
    return (static_cast<size_t>(u4_slot) * 4u + lane) * sizeof(uint);
}
static_assert(instance_lane_bytes(i_to_object, 0u) == 0u);
static_assert(instance_lane_bytes(i_to_world, 0u) == 48u);
static_assert(instance_lane_bytes(i_misc, im_blas_index) == 96u);
static_assert(instance_lane_bytes(i_misc, im_visibility) == 100u);
static_assert(instance_lane_bytes(i_misc, im_user_id) == 104u);
static_assert(instance_lane_bytes(i_misc, im_flags) == 108u);
static_assert(instance_lane_bytes(i_misc + 1u, im_blas_index) == 112u,
              "the reserved uint4 is the builder's blas-directory lane");

// The instance flags are a mirror of the hardware update kernel's
// `INSTANCE_FLAG_*` (cuda_builtin/cuda_builtin_kernels.cu) and of the DXR/Vulkan
// instance flags the HLSL copy restates: they must keep the same bit values,
// because the build records them and a traversal honours them.
static_assert(instance_flag_disable_face_culling == 1u);
static_assert(instance_flag_flip_facing == 2u);
static_assert(instance_flag_disable_any_hit == 4u);
static_assert(instance_flag_enforce_any_hit == 8u);
static_assert(instance_flag_opaque == 16u);
static_assert(region_flag_tlas == 1u);

// ---------------------------------------------------------------------------
// The region sizes of the layout header and the u4 arithmetic the planner uses
// are the same number; a mismatch would mean a region a traversal reads past its
// end.
// ---------------------------------------------------------------------------

[[nodiscard]] constexpr size_t blas_region_u4(size_t prim_count,
                                              size_t vertex_count) noexcept {
    return header_u4 + node_u4 * (prim_count * 2u - 1u) +
           prim_count /* index records */ + vertex_count;
}

[[nodiscard]] constexpr size_t tlas_region_u4(size_t instance_count) noexcept {
    return header_u4 + node_u4 * (instance_count * 2u - 1u) +
           blas_record_u4 * instance_count;
}

static_assert(blas_region_u4(1u, 1u) * 16u == blas_region_bytes(1u, 1u));
static_assert(blas_region_u4(3u, 5u) * 16u == blas_region_bytes(3u, 5u));
static_assert(blas_region_u4(1024u, 2048u) * 16u == blas_region_bytes(1024u, 2048u));
static_assert(tlas_region_u4(1u) * 16u == tlas_region_bytes(1u));
static_assert(tlas_region_u4(7u) * 16u == tlas_region_bytes(7u));
static_assert(tlas_region_u4(1024u) * 16u == tlas_region_bytes(1024u));

// The explicit sizes of the smallest regions, spelled out: these are the numbers
// a second implementation is most likely to hard-code.
static_assert(blas_region_bytes(1u, 1u) == 64u + 32u + 16u + 16u);
static_assert(tlas_region_bytes(1u) == 64u + 32u + 32u);
static_assert(blas_region_bytes(2u, 3u) == 64u + 3u * 32u + 2u * 16u + 3u * 16u);

}

}// namespace lc::fallback_rtx

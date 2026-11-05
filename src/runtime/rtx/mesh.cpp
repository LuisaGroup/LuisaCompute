#include <luisa/core/logging.h>
#include <luisa/runtime/rtx/mesh.h>
#include <luisa/runtime/rtx/accel.h>

namespace luisa::compute {

namespace mesh_detail {

// Packed layouts of VkClusterAccelerationStructureBuildTriangleClusterInfoNV.
// packed0: [ 8: 0] triangleCount, [17: 9] vertexCount,
//          [23:18] positionTruncateBitCount, [27:24] indexType,
//          [31:28] opacityMicromapIndexType.
// packed1: [23: 0] geometryIndex, [28:24] reserved, [31:29] geometryFlags
//          (VkClusterAccelerationStructureGeometryIndexAndGeometryFlagsNV).
constexpr uint32_t kTriangleCountMask = 0x1ffu;
constexpr uint32_t kVertexCountMask = 0x1ffu;
constexpr uint32_t kPositionTruncateBitCountMask = 0x3fu;
constexpr uint32_t kIndexTypeMask = 0x0fu;
constexpr uint32_t kBaseGeometryIndexMask = 0x00ffffffu;
constexpr uint32_t kGeometryFlagsMask = 0x07u;

constexpr uint32_t kTriangleCountShift = 0u;
constexpr uint32_t kVertexCountShift = 9u;
constexpr uint32_t kPositionTruncateBitCountShift = 18u;
constexpr uint32_t kIndexTypeShift = 24u;
constexpr uint32_t kGeometryFlagsShift = 29u;

}// namespace mesh_detail

luisa::unique_ptr<Command> Mesh::build(Mesh::BuildRequest request) noexcept {
    _check_is_valid();
    if (is_cluster_mode()) {
        return luisa::make_unique<MeshBuildCommand>(
            handle(), request,
            _c_buffer, _c_buffer_offset_bytes, _c_buffer_size_bytes);
    }
    return luisa::make_unique<MeshBuildCommand>(
        handle(), request,
        _v_buffer, _v_buffer_offset_bytes, _v_buffer_size_bytes, _v_stride,
        _t_buffer, _t_buffer_offset_bytes, _t_buffer_size_bytes);
}

Mesh::~Mesh() noexcept {
    if (*this) {
        if (is_cluster_mode()) {
            device()->destroy_mega_geometry(handle());
        } else {
            device()->destroy_mesh(handle());
        }
    }
}

ResourceCreationInfo Mesh::_create_cluster_resource(
    DeviceInterface *device, const AccelOption &option) noexcept {
    auto info = device->create_mega_geometry(option);
#ifdef LUISA_ENABLE_SAFE_MODE
    if (!info.valid()) {
        LUISA_ERROR("Failed to create cluster (mega-geometry) mesh.");
    }
#endif
    return info;
}

BufferView<ClusterBuildInfo> Mesh::cluster_buffer() const noexcept {
    _check_is_valid();
    if (!is_cluster_mode()) { return {}; }
    return {_c_buffer_native_handle, _c_buffer, sizeof(ClusterBuildInfo),
            _c_buffer_offset_bytes, _c_buffer_size_bytes / sizeof(ClusterBuildInfo),
            _c_buffer_total_size_bytes / sizeof(ClusterBuildInfo)};
}

uint32_t Mesh::cluster_count() const noexcept {
    _check_is_valid();
    // triangle meshes have no cluster buffer at all
    if (!is_cluster_mode()) { return 0u; }
    return static_cast<uint32_t>(_c_buffer_size_bytes / sizeof(ClusterBuildInfo));
}

uint64_t Mesh::cluster_address() const noexcept {
    _check_is_valid();
    if (!*this || !is_cluster_mode()) { return 0u; }
    return device()->mega_geometry_device_address(handle());
}

ClusterBuildInfo make_cluster_build_info(
    uint32_t cluster_id, uint32_t triangle_count, uint32_t vertex_count,
    uint64_t index_buffer, uint64_t vertex_buffer,
    ClusterIndexFormat index_format, uint32_t base_geometry_index,
    ClusterGeometryFlags geometry_flags, uint16_t index_buffer_stride,
    uint16_t vertex_buffer_stride, uint32_t position_truncate_bit_count) noexcept {

    using namespace mesh_detail;
    LUISA_ASSERT(cluster_id != 0xffffffffu,
                 "Cluster id 0xffffffff is reserved and must not be used.");
    LUISA_ASSERT(triangle_count <= kTriangleCountMask,
                 "Cluster triangle count {} exceeds the 9-bit field of "
                 "VkClusterAccelerationStructureBuildTriangleClusterInfoNV.",
                 triangle_count);
    LUISA_ASSERT(vertex_count <= kVertexCountMask,
                 "Cluster vertex count {} exceeds the 9-bit field of "
                 "VkClusterAccelerationStructureBuildTriangleClusterInfoNV.",
                 vertex_count);
    LUISA_ASSERT(position_truncate_bit_count <= kPositionTruncateBitCountMask,
                 "Cluster position truncate bit count {} exceeds the 6-bit field of "
                 "VkClusterAccelerationStructureBuildTriangleClusterInfoNV.",
                 position_truncate_bit_count);
    LUISA_ASSERT(index_format == ClusterIndexFormat::UInt8 ||
                     index_format == ClusterIndexFormat::UInt16 ||
                     index_format == ClusterIndexFormat::UInt32,
                 "Invalid cluster index format value {}.",
                 static_cast<uint32_t>(index_format));
    LUISA_ASSERT(base_geometry_index <= kBaseGeometryIndexMask,
                 "Cluster base geometry index {} exceeds the 24-bit field of "
                 "VkClusterAccelerationStructureGeometryIndexAndGeometryFlagsNV.",
                 base_geometry_index);
    LUISA_ASSERT((static_cast<uint32_t>(geometry_flags) & ~kGeometryFlagsMask) == 0u,
                 "Invalid cluster geometry flags value {}.",
                 static_cast<uint32_t>(geometry_flags));

    ClusterBuildInfo info{};
    info.cluster_id = cluster_id;
    info.cluster_flags = 0u;
    info.packed0 = ((triangle_count & kTriangleCountMask) << kTriangleCountShift) |
                   ((vertex_count & kVertexCountMask) << kVertexCountShift) |
                   ((position_truncate_bit_count & kPositionTruncateBitCountMask) << kPositionTruncateBitCountShift) |
                   ((static_cast<uint32_t>(index_format) & kIndexTypeMask) << kIndexTypeShift);
    info.packed1 = (base_geometry_index & kBaseGeometryIndexMask) |
                   ((static_cast<uint32_t>(geometry_flags) & kGeometryFlagsMask) << kGeometryFlagsShift);
    info.index_buffer_stride = index_buffer_stride;
    info.vertex_buffer_stride = vertex_buffer_stride;
    info.geometry_index_and_flags_buffer_stride = 0u;
    info.opacity_micromap_index_buffer_stride = 0u;
    info.index_buffer = index_buffer;
    info.vertex_buffer = vertex_buffer;
    info.geometry_index_and_flags_buffer = 0u;
    info.opacity_micromap_array = 0u;
    info.opacity_micromap_index_buffer = 0u;
    return info;
}

namespace detail {

void check_mesh_vert_align(size_t v_stride, size_t dst) {
    LUISA_ASSERT(v_stride % dst == 0,
                 "Required mesh vertex buffer with stride {} can not be aligned to v_stride {}",
                 dst, v_stride);
}

void check_mesh_vert_buffer_motion_keyframe_count(size_t total_vertex_count, uint motion_keyframe_count) {
    LUISA_ASSERT(motion_keyframe_count <= 1u || total_vertex_count % motion_keyframe_count == 0,
                 "Required vertex count {} can not be aligned to motion keyframe count {}",
                 total_vertex_count, motion_keyframe_count);
}

void check_mesh_triangle_buffer_offset_and_size(size_t offset_bytes, size_t size_bytes) {
    LUISA_ASSERT(offset_bytes % sizeof(uint) == 0,
                 "Required triangle buffer offset {} can not be aligned to triangle index size {}",
                 offset_bytes, sizeof(uint));
    LUISA_ASSERT(size_bytes % sizeof(Triangle) == 0,
                 "Required triangle buffer size {} can not be aligned to triangle size {}",
                 size_bytes, sizeof(Triangle));
}

void check_mesh_vertex_buffer_offset_and_size(size_t offset_bytes, size_t size_bytes, size_t v_stride) {
    LUISA_ASSERT(offset_bytes % v_stride == 0,
                 "Required vertex buffer offset {} can not be aligned to vertex stride {}",
                 offset_bytes, v_stride);
    LUISA_ASSERT(size_bytes % v_stride == 0,
                 "Required vertex buffer size {} can not be aligned to vertex stride {}",
                 size_bytes, v_stride);
}

void check_mesh_cluster_buffer_offset_and_size(size_t offset_bytes, size_t size_bytes) {
    LUISA_ASSERT(offset_bytes % alignof(ClusterBuildInfo) == 0,
                 "Required cluster build info buffer offset {} can not be aligned to cluster build info alignment {}",
                 offset_bytes, alignof(ClusterBuildInfo));
    LUISA_ASSERT(size_bytes % sizeof(ClusterBuildInfo) == 0,
                 "Required cluster build info buffer size {} can not be aligned to cluster build info size {}",
                 size_bytes, sizeof(ClusterBuildInfo));
}

}// namespace detail

}// namespace luisa::compute

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
        LUISA_ASSERT(!has_cutout_mask(),
                     "A cluster (mega-geometry) mesh cannot carry an opacity "
                     "micromap: the cluster build records drive opacity themselves.");
        return luisa::make_unique<MeshBuildCommand>(
            handle(), request,
            _c_buffer, _c_buffer_offset_bytes, _c_buffer_size_bytes);
    }
    auto cmd = luisa::make_unique<MeshBuildCommand>(
        handle(), request,
        _v_buffer, _v_buffer_offset_bytes, _v_buffer_size_bytes, _v_stride,
        _t_buffer, _t_buffer_offset_bytes, _t_buffer_size_bytes);
    if (has_cutout_mask()) {
        // Declare the micromap *and* its payload buffers, so the command-reorder
        // pass can order the bake/upload before this build (see the warning about
        // cluster meshes in mesh.h for the failure mode this avoids).
        cmd->set_micromap(MeshMicromapBinding{
            .micromap = _cutout_mask.micromap,
            .triangle_count = _cutout_mask.triangle_count,
            .index_stride = _cutout_mask.index_stride,
            .data_buffer = _cutout_mask.data_buffer,
            .data_offset = _cutout_mask.data_offset,
            .data_size = _cutout_mask.data_size,
            .index_buffer = _cutout_mask.index_buffer,
            .index_offset = _cutout_mask.index_offset,
            .index_size = _cutout_mask.index_size,
            .assumes_non_opaque_traversal = _cutout_mask.assumes_non_opaque_traversal});
    }
    return cmd;
}

MeshCutoutMask::MeshCutoutMask(const OpacityMicromap &cutout) noexcept {
    // Fail closed: a micromap whose packed bits were never written makes every
    // micro triangle transparent (two-state bit 0), so attaching it would make the
    // whole mesh silently disappear. `ready()` is monotonic, so checking it once
    // here covers every later build of this mesh.
    if (!cutout.ready()) [[unlikely]] {
        LUISA_ERROR_WITH_LOCATION(
            "A mesh cannot attach an opacity micromap that has no bits: call "
            "OpacityMicromap::bake() or OpacityMicromap::upload() first.");
    }
    auto data = cutout.data();
    micromap = cutout.handle();
    data_buffer = data.handle();
    data_offset = data.offset_bytes();
    data_size = data.size_bytes();
    if (cutout.index_format() != OpacityMicromap::IndexFormat::NoIndex) {
        auto index = cutout.index();
        index_buffer = index.handle();
        index_offset = index.offset_bytes();
        index_size = index.size_bytes();
    }
    triangle_count = cutout.triangle_count();
    index_stride = static_cast<uint32_t>(cutout.index_format());
    assumes_non_opaque_traversal = true;
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

void check_mesh_cutout_mask(const MeshCutoutMask &mask,
                            size_t mesh_triangle_count,
                            bool motion_blur) noexcept {
    if (!mask) { return; }// the common, micromap-free case
    LUISA_ASSERT(!motion_blur,
                 "A motion-blur BLAS cannot carry an opacity micromap: "
                 "VkAccelerationStructureGeometryMotionTrianglesDataNV already "
                 "occupies the triangles pNext chain.");
    LUISA_ASSERT(mask.data_buffer != 0u && mask.data_size != 0u,
                 "The opacity micromap of this mesh has no packed bit payload.");
    LUISA_ASSERT(mask.triangle_count != 0u,
                 "The opacity micromap of this mesh has no triangles.");
    LUISA_ASSERT(mask.triangle_count == mesh_triangle_count,
                 "The opacity micromap covers {} triangles, but the mesh has {}. A "
                 "shared micromap needs OpacityMicromap::IndexFormat != None.",
                 mask.triangle_count, mesh_triangle_count);
    LUISA_ASSERT(mask.index_stride == 0u ||
                     (mask.index_buffer != 0u && mask.index_size != 0u),
                 "The opacity micromap declares a per-triangle index stride of {} "
                 "bytes but carries no index buffer.",
                 mask.index_stride);
    LUISA_ASSERT(mask.index_stride == 0u || mask.index_size / mask.index_stride >= mesh_triangle_count,
                 "The opacity micromap index buffer holds {} indices of {} bytes for "
                 "{} mesh triangles.",
                 mask.index_size / (mask.index_stride == 0u ? 1u : mask.index_stride),
                 mask.index_stride, mesh_triangle_count);
}

}// namespace detail

}// namespace luisa::compute

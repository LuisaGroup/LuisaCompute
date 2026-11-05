#pragma once

#include <luisa/core/logging.h>
#include <luisa/runtime/rtx/triangle.h>
#include <luisa/runtime/rhi/resource.h>
#include <luisa/runtime/rhi/command.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/buffer.h>

namespace luisa::compute {

namespace detail {
LUISA_RUNTIME_API void check_mesh_vert_align(size_t v_stride, size_t dst);
LUISA_RUNTIME_API void check_mesh_vert_buffer_motion_keyframe_count(size_t total_vertex_count, uint motion_keyframe_count);
LUISA_RUNTIME_API void check_mesh_triangle_buffer_offset_and_size(size_t offset_bytes, size_t size_bytes);
LUISA_RUNTIME_API void check_mesh_vertex_buffer_offset_and_size(size_t offset_bytes, size_t size_bytes, size_t v_stride);
LUISA_RUNTIME_API void check_mesh_cluster_buffer_offset_and_size(size_t offset_bytes, size_t size_bytes);
}// namespace detail
class Accel;

// Index element size of a cluster's triangle indices. The numeric values are the
// VK_CLUSTER_ACCELERATION_STRUCTURE_INDEX_FORMAT_*_NV enumerants, spelled out so
// that this public header does not have to include <volk.h>.
enum class ClusterIndexFormat : uint32_t {
    UInt8 = 0x1u, // VK_CLUSTER_ACCELERATION_STRUCTURE_INDEX_FORMAT_8BIT_NV
    UInt16 = 0x2u,// VK_CLUSTER_ACCELERATION_STRUCTURE_INDEX_FORMAT_16BIT_NV
    UInt32 = 0x4u,// VK_CLUSTER_ACCELERATION_STRUCTURE_INDEX_FORMAT_32BIT_NV
};

// Per-cluster geometry flags, packed into ClusterBuildInfo::packed1. The numeric
// values are the VK_CLUSTER_ACCELERATION_STRUCTURE_GEOMETRY_*_BIT_NV enumerants.
enum class ClusterGeometryFlags : uint32_t {
    CullDisable = 0x1u,                // ..._GEOMETRY_CULL_DISABLE_BIT_NV
    NoDuplicateAnyHitInvocation = 0x2u,// ..._GEOMETRY_NO_DUPLICATE_ANYHIT_INVOCATION_BIT_NV
    Opaque = 0x4u,                     // ..._GEOMETRY_OPAQUE_BIT_NV
};

[[nodiscard]] constexpr ClusterGeometryFlags operator|(ClusterGeometryFlags lhs, ClusterGeometryFlags rhs) noexcept {
    return static_cast<ClusterGeometryFlags>(static_cast<uint32_t>(lhs) | static_cast<uint32_t>(rhs));
}

[[nodiscard]] constexpr ClusterGeometryFlags operator&(ClusterGeometryFlags lhs, ClusterGeometryFlags rhs) noexcept {
    return static_cast<ClusterGeometryFlags>(static_cast<uint32_t>(lhs) & static_cast<uint32_t>(rhs));
}

// One cluster of triangles for a cluster (mega-geometry) acceleration structure.
// Layout-pinned: byte-identical to
// VkClusterAccelerationStructureBuildTriangleClusterInfoNV so the backend can hand
// an array of these straight to the driver.
struct alignas(8) ClusterBuildInfo {
    uint32_t cluster_id;   // user id, returned by ClusterIDNV; must not be 0xffffffffu
    uint32_t cluster_flags;// VkClusterAccelerationStructureClusterFlagsNV
    // packed0: [ 8: 0] triangle_count
    //          [17: 9] vertex_count
    //          [23:18] position_truncate_bit_count
    //          [27:24] index_type (ClusterIndexFormat value: 1, 2 or 4)
    //          [31:28] opacity_micromap_index_type (0 = none)
    uint32_t packed0;
    // packed1: [23: 0] base_geometry_index
    //          [28:24] reserved (must be 0)
    //          [31:29] geometry_flags (ClusterGeometryFlags value)
    // Mirrors VkClusterAccelerationStructureGeometryIndexAndGeometryFlagsNV, whose
    // bit-fields are geometryIndex:24, reserved:5, geometryFlags:3.
    uint32_t packed1;
    uint16_t index_buffer_stride;                   // 0 = tightly packed
    uint16_t vertex_buffer_stride;                  // 0 = tightly packed
    uint16_t geometry_index_and_flags_buffer_stride;// 0 = unused
    uint16_t opacity_micromap_index_buffer_stride;  // 0 = unused
    uint64_t index_buffer;                          // device address
    uint64_t vertex_buffer;                         // device address
    uint64_t geometry_index_and_flags_buffer;       // 0 = use base_geometry_index/flags
    uint64_t opacity_micromap_array;                // 0 = OMM disabled
    uint64_t opacity_micromap_index_buffer;         // 0 = per-triangle index

    [[nodiscard]] constexpr uint32_t triangle_count() const noexcept { return packed0 & 0x1ffu; }
    [[nodiscard]] constexpr uint32_t vertex_count() const noexcept { return (packed0 >> 9u) & 0x1ffu; }
    [[nodiscard]] constexpr uint32_t position_truncate_bit_count() const noexcept { return (packed0 >> 18u) & 0x3fu; }
    [[nodiscard]] constexpr ClusterIndexFormat index_type() const noexcept {
        return static_cast<ClusterIndexFormat>((packed0 >> 24u) & 0x0fu);
    }
    [[nodiscard]] constexpr uint32_t opacity_micromap_index_type() const noexcept { return (packed0 >> 28u) & 0x0fu; }
    [[nodiscard]] constexpr uint32_t base_geometry_index() const noexcept { return packed1 & 0x00ffffffu; }
    [[nodiscard]] constexpr ClusterGeometryFlags geometry_flags() const noexcept {
        return static_cast<ClusterGeometryFlags>((packed1 >> 29u) & 0x07u);
    }
};

static_assert(sizeof(ClusterBuildInfo) == 64u, "ClusterBuildInfo must match VkClusterAccelerationStructureBuildTriangleClusterInfoNV");
static_assert(alignof(ClusterBuildInfo) == 8u, "ClusterBuildInfo must match VkClusterAccelerationStructureBuildTriangleClusterInfoNV");

// Build a ClusterBuildInfo with tightly packed (or explicitly strided) index and
// vertex buffers, no opacity micromap and geometry flags at bit 29..31.
[[nodiscard]] LUISA_RUNTIME_API ClusterBuildInfo make_cluster_build_info(
    uint32_t cluster_id, uint32_t triangle_count, uint32_t vertex_count,
    uint64_t index_buffer, uint64_t vertex_buffer,
    ClusterIndexFormat index_format = ClusterIndexFormat::UInt16,
    uint32_t base_geometry_index = 0u,
    ClusterGeometryFlags geometry_flags = ClusterGeometryFlags::Opaque,
    uint16_t index_buffer_stride = 0u,
    uint16_t vertex_buffer_stride = 0u,
    uint32_t position_truncate_bit_count = 0u) noexcept;

// A Mesh is a bottom-level acceleration structure (BLAS) for ray-tracing with a set of triangles.
// For custom intersection, see ProceduralPrimitive.
class LUISA_RUNTIME_API Mesh final : public Resource {

public:
    using BuildRequest = AccelBuildRequest;
    // Triangle meshes are built from vertex + index buffers (the historical
    // behaviour); cluster meshes are built from an array of ClusterBuildInfo.
    enum class Mode : uint32_t {
        Triangle = 0u,
        Cluster = 1u,
    };

private:
    uint _triangle_count{};
    uint _motion_keyframe_count{};
    uint64_t _v_buffer{};
    void *_v_buffer_native_handle{};
    size_t _v_buffer_offset_bytes{};
    size_t _v_buffer_size_bytes{};
    size_t _v_buffer_total_size_bytes{};
    size_t _v_stride{};

    uint64_t _t_buffer{};
    void *_t_buffer_native_handle{};
    size_t _t_buffer_offset_bytes{};
    size_t _t_buffer_size_bytes{};
    size_t _t_buffer_total_size_bytes{};

    // cluster (mega-geometry) mode: _v_buffer/_t_buffer* stay zeroed
    Mode _mode{Mode::Triangle};
    uint64_t _c_buffer{};
    void *_c_buffer_native_handle{};
    size_t _c_buffer_offset_bytes{};
    size_t _c_buffer_size_bytes{};
    size_t _c_buffer_total_size_bytes{};

public:
    // WARNING: transform buffers' data without rebuild acceleration structure is undefined behavior
    template<typename VertexType>
    [[nodiscard]] BufferView<VertexType> vertex_buffer() const noexcept {
        LUISA_ASSERT(!is_cluster_mode(),
                     "vertex_buffer() is triangle-only: a cluster mesh carries no vertex buffer.");
        detail::check_mesh_vert_align(_v_stride, sizeof(VertexType));
        return {_v_buffer_native_handle, _v_buffer, sizeof(VertexType),
                _v_buffer_offset_bytes, _v_buffer_size_bytes / sizeof(VertexType),
                _v_buffer_total_size_bytes / sizeof(VertexType)};
    }

    [[nodiscard]] BufferView<Triangle> triangle_buffer() const noexcept {
        LUISA_ASSERT(!is_cluster_mode(),
                     "triangle_buffer() is triangle-only: a cluster mesh carries no triangle buffer.");
        return {_t_buffer_native_handle, _t_buffer, sizeof(Triangle),
                _t_buffer_offset_bytes, _t_buffer_size_bytes / sizeof(Triangle),
                _t_buffer_total_size_bytes / sizeof(Triangle)};
    }

private:
    friend class Device;

    [[nodiscard]] static ResourceCreationInfo _create_cluster_resource(
        DeviceInterface *device, const AccelOption &option) noexcept;

    template<typename VBuffer, typename TBuffer>
        requires is_buffer_or_view_v<VBuffer> &&
                 is_buffer_or_view_v<TBuffer> &&
                 std::same_as<buffer_element_t<TBuffer>, Triangle>
    [[nodiscard]] static ResourceCreationInfo _create_resource(
        DeviceInterface *device, const AccelOption &option,
        const VBuffer &vertex_buffer [[maybe_unused]],
        const TBuffer &triangle_buffer [[maybe_unused]]) noexcept {
        auto info = device->create_mesh(option);
#ifdef LUISA_ENABLE_SAFE_MODE
        if (!info.valid()) {
            LUISA_ERROR("Failed to create mesh.");
        }
#endif
        return info;
    }

private:
    // The spec's `LUISA_ASSERT(buffer_element_t<CBuffer> == ClusterBuildInfo)` is a
    // type-level property, so it is enforced statically by the constraint below
    // (exactly like the triangle constructors constrain on Triangle).
    template<typename CBuffer>
        requires is_buffer_or_view_v<CBuffer> &&
                 std::same_as<buffer_element_t<CBuffer>, ClusterBuildInfo>
    Mesh(DeviceInterface *device,
         const CBuffer &cluster_buffer,
         const AccelOption &option) noexcept
        : Resource{device, Resource::Tag::MESH,
                   _create_cluster_resource(device, option)} {

        BufferView<ClusterBuildInfo> c_view{cluster_buffer};
        detail::check_mesh_cluster_buffer_offset_and_size(c_view.offset_bytes(), c_view.size_bytes());

        _mode = Mode::Cluster;

        _c_buffer = c_view.handle();
        _c_buffer_native_handle = c_view.native_handle();
        _c_buffer_offset_bytes = c_view.offset_bytes();
        _c_buffer_size_bytes = c_view.size_bytes();
        _c_buffer_total_size_bytes = c_view.total_size_bytes();
    }

private:
    template<typename VBuffer, typename TBuffer>
    Mesh(DeviceInterface *device,
         const VBuffer &vertex_buffer, size_t vertex_stride,
         const TBuffer &triangle_buffer,
         const AccelOption &option) noexcept
        : Resource{device, Resource::Tag::MESH,
                   _create_resource(device, option, vertex_buffer, triangle_buffer)} {

        BufferView vb_view{vertex_buffer};
        BufferView tri_view{triangle_buffer};
        detail::check_mesh_vertex_buffer_offset_and_size(vb_view.offset_bytes(), vb_view.size_bytes(), vertex_stride);
        detail::check_mesh_triangle_buffer_offset_and_size(tri_view.offset_bytes(), tri_view.size_bytes());

        _triangle_count = static_cast<uint>(tri_view.size_bytes() / sizeof(Triangle));
        _motion_keyframe_count = option.motion.keyframe_count;
        detail::check_mesh_vert_buffer_motion_keyframe_count(vb_view.size_bytes() / vertex_stride, _motion_keyframe_count);

        _v_buffer = vb_view.handle();
        _v_buffer_native_handle = vb_view.native_handle();
        _v_buffer_offset_bytes = vb_view.offset_bytes();
        _v_buffer_size_bytes = vb_view.size_bytes();
        _v_buffer_total_size_bytes = vb_view.total_size_bytes();
        _v_stride = vertex_stride;

        _t_buffer = tri_view.handle();
        _t_buffer_native_handle = tri_view.native_handle();
        _t_buffer_offset_bytes = tri_view.offset_bytes();
        _t_buffer_size_bytes = tri_view.size_bytes();
        _t_buffer_total_size_bytes = tri_view.total_size_bytes();
    }

    template<typename VBuffer, typename TBuffer>
    Mesh(DeviceInterface *device,
         const VBuffer &vertex_buffer, const TBuffer &triangle_buffer,
         const AccelOption &option) noexcept
        : Mesh{device, vertex_buffer, vertex_buffer.stride(), triangle_buffer, option} {}

public:
    Mesh() noexcept = default;
    ~Mesh() noexcept override;
    Mesh(Mesh &&) noexcept = default;
    Mesh(Mesh const &) noexcept = delete;
    Mesh &operator=(Mesh &&rhs) noexcept {
        _move_from(std::move(rhs));
        return *this;
    }
    Mesh &operator=(Mesh const &) noexcept = delete;
    using Resource::operator bool;
    using Resource::release;
    // build the bottom-level acceleration structure: a triangle BLAS for triangle
    // mode, a cluster BLAS for cluster mode. For cluster mode see the raw-device-
    // address ordering and lifetime warning on Device::create_mesh(ClusterBuildInfo
    // array) below.
    [[nodiscard]] luisa::unique_ptr<Command> build(BuildRequest request = BuildRequest::PREFER_UPDATE) noexcept;

    [[nodiscard]] auto triangle_count() const noexcept {
        _check_is_valid();
        return _triangle_count;
    }

    [[nodiscard]] Mode mode() const noexcept {
        _check_is_valid();
        return _mode;
    }

    [[nodiscard]] bool is_cluster_mode() const noexcept { return _mode == Mode::Cluster; }

    // Array of cluster build records owned by this mesh (empty for triangle mode).
    [[nodiscard]] BufferView<ClusterBuildInfo> cluster_buffer() const noexcept;

    // Number of clusters in cluster_buffer() (0 for triangle mode).
    [[nodiscard]] uint32_t cluster_count() const noexcept;

    // Traceable device address of the cluster BLAS, valid only after a build has
    // been dispatched and completed. Always 0 for triangle mode.
    [[nodiscard]] uint64_t cluster_address() const noexcept;

    [[nodiscard]] auto motion_keyframe_count() const noexcept {
        _check_is_valid();
        return std::max<uint>(1u, _motion_keyframe_count);
    }

    [[nodiscard]] auto vertex_stride() const noexcept {
        _check_is_valid();
        return _v_stride;
    }

    [[nodiscard]] auto vertex_count_per_motion_keyframe() const noexcept {
        _check_is_valid();
        // cluster meshes carry no vertex buffer, so there is no vertex stride to
        // divide by: report an empty vertex set instead of dividing by zero
        if (is_cluster_mode()) { return size_t{0u}; }
        auto n = this->motion_keyframe_count();
        return _v_buffer_size_bytes / _v_stride / n;
    }
};

template<typename VBuffer, typename TBuffer>
    requires is_buffer_or_view_v<VBuffer> &&
             is_buffer_or_view_v<TBuffer> &&
             std::same_as<buffer_element_t<TBuffer>, Triangle>
Mesh Device::create_mesh(VBuffer &&vertices, TBuffer &&triangles, const AccelOption &option) noexcept {
    return this->_create<Mesh>(std::forward<VBuffer>(vertices), std::forward<TBuffer>(triangles), option);
}

template<typename VBuffer, typename TBuffer>
    requires is_buffer_or_view_v<VBuffer> &&
             is_buffer_or_view_v<TBuffer> &&
             std::same_as<buffer_element_t<TBuffer>, Triangle>
Mesh Device::create_mesh(VBuffer &&vertices, size_t vertex_stride, TBuffer &&triangles, const AccelOption &option) noexcept {
    return this->_create<Mesh>(std::forward<VBuffer>(vertices), vertex_stride, std::forward<TBuffer>(triangles), option);
}

    // Build a cluster (mega-geometry) bottom-level acceleration structure from an
    // array of ClusterBuildInfo records; the records carry the vertex/index device
    // addresses, so no vertex/triangle buffer is bound here.
    //
    // WARNING - the vertex and index memory referenced by the records is invisible to
    // the command graph. Unlike a triangle Mesh, which declares its vertex/triangle
    // buffers as build inputs, a cluster Mesh has no buffer handles to declare, so the
    // command-reorder pass CANNOT order the writes that fill that memory before
    // mesh.build(): it may schedule a copy or dispatch that produces the vertices
    // *after* the cluster build, in which case the driver reads stale or uninitialised
    // memory and the resulting BLAS traces the wrong geometry without any error.
    //
    // The caller is therefore responsible for the ordering and the lifetime of every
    // buffer addressed by the records:
    // - submit the uploads/computes that fill the vertex and index data and wait for
    //   them (e.g. `stream << ... << synchronize();`) before recording mesh.build(), or
    //   use a separate, earlier stream/flush for them;
    // - keep those buffers alive until the cluster build has completed on the device -
    //   destroying or reallocating them earlier invalidates raw device addresses the
    //   backend has already handed to the driver.
    //
    // This restriction is a limitation of the current command-reorder pass (see the
    // TODO(cluster) note in src/backends/common/command_reorder_visitor.h) and may be
    // lifted once the pass can follow the addresses stored in the records.
    template<typename CBuffer>
    requires is_buffer_or_view_v<CBuffer> &&
             std::same_as<buffer_element_t<CBuffer>, ClusterBuildInfo>
Mesh Device::create_mesh(CBuffer &&clusters, const AccelOption &option) noexcept {
    return this->_create<Mesh>(std::forward<CBuffer>(clusters), option);
}

}// namespace luisa::compute

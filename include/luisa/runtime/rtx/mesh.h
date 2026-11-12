#pragma once

#include <luisa/core/logging.h>
#include <luisa/runtime/rtx/triangle.h>
#include <luisa/runtime/rhi/resource.h>
#include <luisa/runtime/rhi/command.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/buffer.h>

namespace luisa::compute {

struct MeshCutoutMask;

namespace detail {
LUISA_RUNTIME_API void check_mesh_vert_align(size_t v_stride, size_t dst);
LUISA_RUNTIME_API void check_mesh_vert_buffer_motion_keyframe_count(size_t total_vertex_count, uint motion_keyframe_count);
LUISA_RUNTIME_API void check_mesh_triangle_buffer_offset_and_size(size_t offset_bytes, size_t size_bytes);
LUISA_RUNTIME_API void check_mesh_vertex_buffer_offset_and_size(size_t offset_bytes, size_t size_bytes, size_t v_stride);
LUISA_RUNTIME_API void check_mesh_cluster_buffer_offset_and_size(size_t offset_bytes, size_t size_bytes);
// Validates an attached cut-out mask against the mesh it is attached to: the
// micromap must have bits, the triangle counts must agree, the payload buffers
// must be non-empty, and a motion-blur BLAS may not carry a micromap (the triangles
// pNext chain is already occupied by VkAccelerationStructureGeometryMotionTriangles
// DataNV, and a micromap on a motion geometry is not supported by this build shape).
LUISA_RUNTIME_API void check_mesh_cutout_mask(const MeshCutoutMask &mask,
                                             size_t mesh_triangle_count,
                                             bool motion_blur) noexcept;
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
    // DXR has no per-cluster equivalent; this bit is cleared on DX.
    CullDisable = 0x1u,                // ..._GEOMETRY_CULL_DISABLE_BIT_NV
    NoDuplicateAnyHitInvocation = 0x2u,// ..._GEOMETRY_NO_DUPLICATE_ANYHIT_INVOCATION_BIT_NV
    Opaque = 0x4u,                     // ..._GEOMETRY_OPAQUE_BIT_NV
};

// NOTE: deliberately no free operator| / operator& for ClusterGeometryFlags.
// They would live in namespace luisa::compute and would therefore be found by
// argument-dependent lookup for *every* expression in that namespace, which
// breaks the DSL's own constrained operator templates (e.g. `Var<bool> |
// Var<bool>` in luisa/coro). Combine flags with an explicit uint32_t cast
// instead: static_cast<ClusterGeometryFlags>(a_u32 | b_u32).

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

// What a mesh remembers about its hardware opacity micromap ("cut-out mask").
//
// Holds handles only: an OpacityMicromap is a move-only Resource owned by the
// caller, so a Mesh never copies or owns it (Resource's copy constructor is
// deleted) and must not be able to destroy it. The micromap has to outlive every
// acceleration structure built from this mesh that is still traced, unless the
// driver reports the micromap build sizes as discardable.
//
// The two payload buffer handles are part of the struct because they are the
// command-graph inputs of the build: declaring them is what lets the command
// reorder pass schedule the bake (or the upload) of the packed bits before the
// build instead of after it.
//
// `assumes_non_opaque_traversal` records the shader-side contract: the ray
// traversal that traces this mesh must not pass `OpaqueKHR` (or `NoOpaqueKHR`),
// which overrides both the geometry flag and the micromap and would silently make
// the cut-out a no-op. In practice that means tracing with the loop form -
// `Accel::traverse()` / `Accel::query()` - because `Accel::intersect()` and
// `Accel::intersect_any()` emit `OpaqueKHR` and therefore bypass the micromap.
// `Accel::emplace_back(mesh, ...)` already drops the opaque bit of its instance for
// a mesh that carries a cut-out mask. Leaving this at `true` (the default) is what
// the backend checks before it chains the micromap into the BLAS build inputs;
// setting it to `false` declares that the caller traces with a flag-free traversal,
// and is refused because this library cannot verify a shader's ray flags.
struct LUISA_RUNTIME_API MeshCutoutMask {
    uint64_t micromap{0u};
    uint64_t data_buffer{0u};
    size_t data_offset{0u};
    size_t data_size{0u};
    uint64_t index_buffer{0u};
    size_t index_offset{0u};
    size_t index_size{0u};
    uint32_t triangle_count{0u};
    uint32_t index_stride{0u};
    bool assumes_non_opaque_traversal{true};

    MeshCutoutMask() noexcept = default;
    // Implicit conversion so `create_mesh(vertices, triangles, micromap)` reads like
    // the plain overload. Raises (LUISA_ERROR) when `micromap` was never baked or
    // uploaded: the two-state bit `0` means *transparent*, so attaching a micromap
    // without bits would make the whole mesh silently disappear.
    MeshCutoutMask(const OpacityMicromap &micromap) noexcept;// NOLINT(google-explicit-constructor)
    [[nodiscard]] explicit operator bool() const noexcept { return micromap != 0u; }
};

// A Mesh is a bottom-level acceleration structure (BLAS) for ray-tracing with a set of triangles.
// For custom intersection, see ProceduralPrimitive.
//
// Cut-out masks. A mesh may carry a hardware opacity micromap
// (VK_KHR/EXT_opacity_micromap) so that the traversal itself discards the
// fully-transparent micro triangles of an alpha-cut-out surface, without an
// any-hit shader per candidate hit. See OpacityMicromap for the resource and
// `MeshCutoutMask` below for what a mesh remembers about it.
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

    // The attached cut-out mask, immutable after creation and empty by default.
    // `micromap == 0` means "no micromap", which is what every pre-extension build
    // sees, so all existing behaviour is bit-identical.
    MeshCutoutMask _cutout_mask{};

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
         MeshCutoutMask cutout_mask,
         const AccelOption &option) noexcept
        : Resource{device, Resource::Tag::MESH,
                   _create_resource(device, option, vertex_buffer, triangle_buffer)},
          _cutout_mask{cutout_mask} {

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

        detail::check_mesh_cutout_mask(_cutout_mask, _triangle_count,
                                       option.motion.is_enabled());
    }

    template<typename VBuffer, typename TBuffer>
    Mesh(DeviceInterface *device,
         const VBuffer &vertex_buffer, size_t vertex_stride,
         const TBuffer &triangle_buffer,
         const AccelOption &option) noexcept
        : Mesh{device, vertex_buffer, vertex_stride, triangle_buffer,
               MeshCutoutMask{}, option} {}

    template<typename VBuffer, typename TBuffer>
    Mesh(DeviceInterface *device,
         const VBuffer &vertex_buffer, const TBuffer &triangle_buffer,
         MeshCutoutMask cutout_mask,
         const AccelOption &option) noexcept
        : Mesh{device, vertex_buffer, vertex_buffer.stride(), triangle_buffer,
               cutout_mask, option} {}

    template<typename VBuffer, typename TBuffer>
    Mesh(DeviceInterface *device,
         const VBuffer &vertex_buffer, const TBuffer &triangle_buffer,
         const AccelOption &option) noexcept
        : Mesh{device, vertex_buffer, vertex_buffer.stride(), triangle_buffer,
               MeshCutoutMask{}, option} {}

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

    // Whether this mesh carries a hardware opacity micromap ("cut-out mask").
    [[nodiscard]] bool has_cutout_mask() const noexcept { return _cutout_mask.micromap != 0u; }

    // Handle of the attached OpacityMicromap (0 = none). A handle, not a resource:
    // the Mesh does not own it and must not be able to destroy it. Use it for
    // lifetime bookkeeping, not for baking (bake through the original resource).
    [[nodiscard]] uint64_t cutout_mask() const noexcept { return _cutout_mask.micromap; }

    // Whether the traversal contract of the attached micromap is the default one
    // (no `OpaqueKHR` ray flag). False only for a mesh built with an explicit
    // MeshCutoutMask, which the backend then refuses to attach.
    [[nodiscard]] bool cutout_mask_assumes_non_opaque_traversal() const noexcept {
        return _cutout_mask.micromap == 0u ||
               _cutout_mask.assumes_non_opaque_traversal;
    }

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

// Cut-out mask overloads. `MeshCutoutMask` converts implicitly from an
// `OpacityMicromap`, so `create_mesh(v, t, micromap)` and
// `create_mesh(v, t, MeshCutoutMask{...})` both land here.
//
// UVs are deliberately *not* a mesh input: a micromap needs them only at bake time,
// where they are an explicit buffer argument of `OpacityMicromap::bake`. A mesh
// whose UVs differ per triangle therefore has to be de-indexed by the caller.
template<typename VBuffer, typename TBuffer>
    requires is_buffer_or_view_v<VBuffer> &&
             is_buffer_or_view_v<TBuffer> &&
             std::same_as<buffer_element_t<TBuffer>, Triangle>
Mesh Device::create_mesh(VBuffer &&vertices, TBuffer &&triangles,
                         MeshCutoutMask cutout_mask,
                         const AccelOption &option) noexcept {
    return this->_create<Mesh>(std::forward<VBuffer>(vertices), std::forward<TBuffer>(triangles),
                               cutout_mask, option);
}

template<typename VBuffer, typename TBuffer>
    requires is_buffer_or_view_v<VBuffer> &&
             is_buffer_or_view_v<TBuffer> &&
             std::same_as<buffer_element_t<TBuffer>, Triangle>
Mesh Device::create_mesh(VBuffer &&vertices, size_t vertex_stride, TBuffer &&triangles,
                         MeshCutoutMask cutout_mask,
                         const AccelOption &option) noexcept {
    return this->_create<Mesh>(std::forward<VBuffer>(vertices), vertex_stride, std::forward<TBuffer>(triangles),
                               cutout_mask, option);
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
// The initial DirectX implementation additionally rejects records that reference
// opacity micromaps; use a backend query and the record fields before relying on
// OMM support.
template<typename CBuffer>
    requires is_buffer_or_view_v<CBuffer> &&
             std::same_as<buffer_element_t<CBuffer>, ClusterBuildInfo>
Mesh Device::create_mesh(CBuffer &&clusters, const AccelOption &option) noexcept {
    return this->_create<Mesh>(std::forward<CBuffer>(clusters), option);
}

}// namespace luisa::compute

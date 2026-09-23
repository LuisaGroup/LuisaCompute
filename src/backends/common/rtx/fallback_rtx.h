// Fallback (software) ray tracing: the backend-facing API.
//
// A backend that has no hardware ray tracing - or that the user asked to run in
// fallback mode through its `DeviceConfigExt` - constructs one `FallbackRtxDevice`
// per `DeviceInterface` and routes the acceleration-structure part of
// `DeviceInterface` into it:
//
//   create_mesh()          -> FallbackRtxDevice::create_blas()
//   destroy_mesh(handle)   -> FallbackRtxDevice::destroy_blas(handle)
//   MeshBuildCommand       -> FallbackRtxDevice::build_blas()
//   create_accel()         -> FallbackRtxDevice::create_accel()
//   destroy_accel(handle)  -> FallbackRtxDevice::destroy_accel(handle)
//   AccelBuildCommand      -> FallbackRtxDevice::build_accel()
//   shader argument (accel)-> FallbackRtxDevice::binding()
//
// The device hands out the resource handles itself, so the backend never has to
// keep a map from its own handle space into the fallback one: the handle of a
// fallback `Mesh` *is* the handle of its fallback BLAS.  `owns()` tells the
// backend which handles it has to route here.
//
// The build functions return a `CommandList`.  A backend that is encoding a
// command list splices it in at the position of the corresponding
// `MeshBuildCommand` / `AccelBuildCommand`:
//
//     auto list = _fallback->build_blas(h, geometry);
//     for (auto &&cmd : list.steal_commands()) { cmd->accept(*this); }
//
// so the fallback build kernels end up in the caller's stream, ordered exactly
// where the native acceleration-structure build would have been.
//
// Thread safety: one device per `DeviceInterface`, and the backends already
// serialize resource creation; the builds are recorded from the stream's
// encoding context.  The handles and the storage bookkeeping are what the
// backends touch concurrently, so those are the parts guarded by the lock.

#pragma once

#include <luisa/core/basic_types.h>
#include <luisa/core/stl/memory.h>
#include <luisa/core/stl/optional.h>
#include <luisa/core/stl/unordered_map.h>
#include <luisa/core/stl/vector.h>
#include <luisa/runtime/command_list.h>
#include <luisa/runtime/rhi/command.h>
#include <luisa/runtime/rhi/device_interface.h>
#include <luisa/runtime/rhi/resource.h>

#include <cstddef>

namespace lc::fallback_rtx {

using namespace luisa;
using namespace luisa::compute;

// How a shader argument of type `accel` is bound.
//
// The GPU ABI keeps the two-descriptor shape the native acceleration-structure
// arguments already use: one read-only view of the acceleration buffer and one
// view of the instance buffer.  Both views start at the tree's own region, so a
// traversal indexes them from zero (`fallback_rtx_layout.h`).
struct FallbackAccelBinding {
    // Luisa buffer handle of the shared acceleration buffer (`Buffer<uint4>`).
    uint64_t accel_buffer{};
    // Byte offset of the tree's region inside it; the shader's element 0 is the
    // region's first `uint4`.
    size_t accel_offset_bytes{};
    // Luisa buffer handle of the instance buffer (`Buffer<uint4>`).
    uint64_t instance_buffer{};
    size_t instance_offset_bytes{};
    // `false` before the first build: the backend then binds a null descriptor.
    [[nodiscard]] bool valid() const noexcept { return accel_buffer != 0u; }
};

class FallbackRtxDevice {

public:
    // `device` must outlive this object.
    explicit FallbackRtxDevice(DeviceInterface *device) noexcept;
    ~FallbackRtxDevice() noexcept;

    FallbackRtxDevice(FallbackRtxDevice &&) noexcept = delete;
    FallbackRtxDevice(const FallbackRtxDevice &) noexcept = delete;
    FallbackRtxDevice &operator=(FallbackRtxDevice &&) noexcept = delete;
    FallbackRtxDevice &operator=(const FallbackRtxDevice &) noexcept = delete;

    [[nodiscard]] DeviceInterface *device() const noexcept;

    // ---- resource creation (host side, no GPU work) ----------------------

    // A bottom-level acceleration structure over the triangles of one mesh.  The
    // geometry itself arrives with `build_blas()`.
    [[nodiscard]] uint64_t create_blas(const AccelOption &option) noexcept;
    void destroy_blas(uint64_t blas) noexcept;

    // A top-level acceleration structure over instances.
    [[nodiscard]] uint64_t create_accel(const AccelOption &option) noexcept;
    void destroy_accel(uint64_t accel) noexcept;

    // `true` when `handle` was handed out by this device and has to be routed
    // here instead of into the backend's native acceleration-structure path.
    [[nodiscard]] bool owns(uint64_t handle) const noexcept;
    [[nodiscard]] bool owns_blas(uint64_t handle) const noexcept;
    [[nodiscard]] bool owns_accel(uint64_t handle) const noexcept;

    // ---- shader binding ----------------------------------------------------

    // The buffers an `accel` shader argument has to be bound to, resolved *now*
    // (the storage grows behind the scenes, so a backend must call this every
    // time it writes the argument's descriptors, never once and cache it).
    [[nodiscard]] FallbackAccelBinding binding(uint64_t accel) noexcept;

    // ---- builds ------------------------------------------------------------

    // The geometry of the `MeshBuildCommand` being encoded.
    struct MeshGeometry {
        uint64_t vertex_buffer{};// `float3` vertices, `vertex_stride` apart
        size_t vertex_buffer_offset{};
        size_t vertex_stride{};
        size_t vertex_buffer_size{};// bytes
        uint64_t triangle_buffer{}; // three `uint` indices per primitive
        size_t triangle_buffer_offset{};
        size_t triangle_buffer_size{};// bytes
    };

    // Records the build of `blas` (primitive AABBs -> Morton codes -> LSD radix
    // sort -> Karras radix tree) into a command list the caller splices into its
    // own.  `blas` must have been created by `create_blas()` and not be built
    // twice without an intervening destroy.
    [[nodiscard]] CommandList build_blas(uint64_t blas,
                                         const MeshGeometry &geometry) noexcept;

    // Records the build of `accel`: the modified instance records are uploaded,
    // the BLAS table is refreshed from the live instance table, and - unless
    // `update_instance_buffer_only` - the TLAS radix tree is rebuilt.
    [[nodiscard]] CommandList build_accel(
        uint64_t accel, uint32_t instance_count,
        luisa::span<const AccelBuildCommand::Modification> modifications,
        bool update_instance_buffer_only) noexcept;

    // ---- statistics (tests and diagnostics) -------------------------------

    struct Stats {
        size_t blas_count{};
        size_t accel_count{};
        size_t accel_buffer_bytes{};
        size_t instance_buffer_bytes{};
        size_t blas_builds{};
        size_t accel_builds{};
    };
    [[nodiscard]] Stats stats() const noexcept;

    // ---- structural validation (host-synchronising; debug only) ------------

    // Download a built tree and check its structure on the host:
    //
    //   * every node must be reachable from the root exactly once and every
    //     handle must be in range,
    //   * every internal node's AABB must be the union of its two children,
    //   * a BLAS must carry in-range triangle indices,
    //   * a TLAS must resolve every instance's blas-table row into a
    //     well-formed BLAS region (validated recursively) whose transformed root
    //     AABB lies inside the TLAS' own root AABB.
    //
    // Both calls create a stream through the aliased device and synchronise it,
    // so they are a *debug* helper and must not be used in a render loop.  They
    // return the number of problems found and report the first few through
    // `LUISA_WARNING`; a handle that this device does not own, or a tree that was
    // never built, is a fatal error.
    [[nodiscard]] size_t validate_blas(uint64_t blas) noexcept;
    [[nodiscard]] size_t validate_accel(uint64_t accel) noexcept;

private:
    struct Impl;
    luisa::unique_ptr<Impl> _impl;
};

}// namespace lc::fallback_rtx

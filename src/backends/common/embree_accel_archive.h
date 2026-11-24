#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <utility>

#include <luisa/backends/ext/accel_archive_ext_interface.h>
#include <luisa/core/stl/format.h>
#include <luisa/core/stl/memory.h>
#include <luisa/core/stl/string.h>
#include <luisa/core/stl/vector.h>
#include <luisa/runtime/rtx/mesh.h>

namespace luisa::compute {

enum class EmbreeArchiveStatus : uint32_t {
    OK = 0u,
    INVALID_ARGUMENT = 1u,
    UNSUPPORTED_LAYOUT = 2u,
    CORRUPT = 3u,
    INCOMPATIBLE_CONFIG = 4u,
    INCOMPATIBLE_ISA = 5u,
    BUFFER_TOO_SMALL = 6u,
    INTERNAL_ERROR = 7u,
};

struct EmbreeResolvedTriangleGeometry {
    const std::byte *vertex_buffer{nullptr};
    size_t vertex_offset{0u};
    size_t vertex_stride{0u};
    size_t vertex_count{0u};
    const std::byte *triangle_buffer{nullptr};
    size_t triangle_offset{0u};
    size_t triangle_count{0u};
};

struct EmbreeOutOfTreeABI {
    bool valid{false};
    uint32_t embree_version_number{0u};
    uint32_t build_config_flags{0u};
    uint32_t cpu_isa_mask{0u};
    size_t accel_bounds_ofs{0u};
    size_t accel_type_ofs{0u};
    size_t scene_intersectors_ptr_ofs{0u};
    size_t intersector1_name_ofs{0u};
    size_t bvh_device_ofs{0u};
    size_t bvh_root_ofs{0u};
    size_t scene_vertices_vec_data_ofs{0u};
    bool supports_bvh4{false};
    bool supports_bvh8{false};
    bool supports_compact_vertices{false};
};

struct EmbreeImportedBVHHandle {
    void *slab{nullptr};
    size_t slab_bytes{0u};
    void *bvh_ptr{nullptr};
    size_t bvh_root_ofs{0u};
    uint64_t saved_dummy_root{0u};
    size_t bvh_bounds_ofs{0u};
    std::byte saved_bvh_bounds[64]{};
    bool has_saved_bvh_bounds{false};
    void *rtc_scene{nullptr};
    size_t scene_bounds_ofs{0u};
    std::byte saved_scene_bounds[64]{};
    bool has_saved_scene_bounds{false};
    float **vertices_slot{nullptr};
    float *saved_dummy_verts{nullptr};
    bool has_saved_verts{false};

    EmbreeImportedBVHHandle() noexcept = default;
    EmbreeImportedBVHHandle(const EmbreeImportedBVHHandle &) = delete;
    EmbreeImportedBVHHandle &operator=(const EmbreeImportedBVHHandle &) = delete;
    EmbreeImportedBVHHandle(EmbreeImportedBVHHandle &&other) noexcept;
    EmbreeImportedBVHHandle &operator=(EmbreeImportedBVHHandle &&other) noexcept;
    ~EmbreeImportedBVHHandle() noexcept { restore_and_release(); }

    void restore_and_release() noexcept;
};

struct EmbreeTriangleBufferBinding {
    const std::byte *v_buffer{nullptr};
    size_t v_offset{0u};
    size_t v_stride{0u};
    size_t v_count{0u};
    const std::byte *t_buffer{nullptr};
    size_t t_offset{0u};
    size_t t_count{0u};
};

[[nodiscard]] EmbreeArchiveStatus embree_import_static_triangle_bvh_oot(
    const EmbreeOutOfTreeABI &abi,
    void *rtc_scene,
    const EmbreeResolvedTriangleGeometry &geom,
    luisa::span<const std::byte> bvh_payload,
    EmbreeImportedBVHHandle &out_handle,
    size_t &out_unpacked_bytes) noexcept;

[[nodiscard]] EmbreeArchiveStatus embree_commit_dummy_and_import_bvh_oot(
    const EmbreeOutOfTreeABI &abi,
    void *rtc_scene,
    void *rtc_geometry,
    const EmbreeResolvedTriangleGeometry &geom,
    luisa::span<const std::byte> bvh_payload,
    EmbreeImportedBVHHandle &out_handle,
    size_t &out_unpacked_bytes) noexcept;

template<typename SyncAndRebindOldFn>
[[nodiscard]] inline bool embree_mesh_import_from_archive_locked(
    const EmbreeOutOfTreeABI &abi,
    void *rtc_scene,
    void *rtc_geometry,
    const EmbreeResolvedTriangleGeometry &geom,
    luisa::span<const std::byte> bvh_payload,
    EmbreeImportedBVHHandle &imported_bvh,
    EmbreeTriangleBufferBinding &binding,
    bool &blas_dirty,
    bool &built,
    std::atomic<bool> &is_imported_blas,
    std::atomic<uint64_t> &build_revision,
    SyncAndRebindOldFn &&sync_and_rebind_old,
    size_t &out_unpacked_bytes,
    EmbreeArchiveStatus &out_status) noexcept {
    if (!abi.valid || bvh_payload.empty()) {
        out_status = EmbreeArchiveStatus::INVALID_ARGUMENT;
        return false;
    }
    imported_bvh.restore_and_release();
    const EmbreeTriangleBufferBinding old_binding = binding;
    const bool old_dirty = blas_dirty;
    const bool old_built = built;

    out_status = embree_commit_dummy_and_import_bvh_oot(
        abi,
        rtc_scene,
        rtc_geometry,
        geom,
        bvh_payload,
        imported_bvh,
        out_unpacked_bytes);
    if (out_status != EmbreeArchiveStatus::OK) {
        imported_bvh.restore_and_release();
        binding = old_binding;
        if (old_built && old_binding.v_buffer != nullptr && old_binding.t_buffer != nullptr) {
            sync_and_rebind_old(old_binding);
            blas_dirty = true;
        } else {
            blas_dirty = old_dirty;
        }
        is_imported_blas.store(false, std::memory_order_release);
        return false;
    }

    binding.v_buffer = geom.vertex_buffer;
    binding.v_offset = geom.vertex_offset;
    binding.v_stride = geom.vertex_stride;
    binding.v_count = geom.vertex_count;
    binding.t_buffer = geom.triangle_buffer;
    binding.t_offset = geom.triangle_offset;
    binding.t_count = geom.triangle_count;
    blas_dirty = false;
    built = true;
    is_imported_blas.store(true, std::memory_order_release);
    build_revision.fetch_add(1u, std::memory_order_release);
    return true;
}

struct EmbreeMeshExportState {
    bool is_built{false};
    AccelOption option{};
    uint64_t triangle_count{0u};
    uint64_t vertex_count{0u};
    uint64_t vertex_stride{0u};
    void *committed_rtc_scene{nullptr};
};

struct EmbreeAccelArchiveOps {
    AccelArchiveBackendKind backend_kind{AccelArchiveBackendKind::UNKNOWN};
    void *rtc_device{nullptr};
    luisa::string device_fingerprint;
    void *user_ctx{nullptr};

    void (*synchronize_stream)(void *user_ctx, uint64_t stream_handle) noexcept {nullptr};
    bool (*validate_stream)(void *user_ctx, uint64_t stream_handle) noexcept {nullptr};
    AccelArchiveResult (*get_mesh_export_state)(
        void *user_ctx,
        uint64_t mesh_handle,
        EmbreeMeshExportState &out_state) noexcept {nullptr};
    AccelArchiveResult (*resolve_geometry_binding)(
        void *user_ctx,
        uint64_t mesh_handle,
        const AccelArchiveMeshDesc &expected_desc,
        const AccelArchiveGeometryBinding &binding,
        EmbreeResolvedTriangleGeometry &out_geom) noexcept {nullptr};
    AccelArchiveResult (*import_mesh_bvh)(
        void *user_ctx,
        uint64_t mesh_handle,
        const EmbreeResolvedTriangleGeometry &geom,
        luisa::span<const std::byte> bvh_payload,
        const EmbreeOutOfTreeABI &abi,
        size_t &out_unpacked_bytes,
        EmbreeArchiveStatus &out_embree_status) noexcept {nullptr};
};

class EmbreeAccelArchiveExt final : public AccelArchiveExt {
public:
    explicit EmbreeAccelArchiveExt(EmbreeAccelArchiveOps ops) noexcept;
    ~EmbreeAccelArchiveExt() noexcept;

    [[nodiscard]] AccelArchiveCapabilities capabilities() const noexcept override;

    [[nodiscard]] AccelArchiveResult validate_mesh_archive(
        luisa::span<const std::byte> archive_bytes,
        const AccelArchiveMeshDesc &expected_desc,
        AccelArchiveMetadata *out_meta = nullptr) const noexcept override;

    [[nodiscard]] AccelArchiveResult export_mesh_archive(
        uint64_t stream_handle,
        uint64_t mesh_handle,
        const AccelArchiveMeshDesc &desc,
        luisa::vector<std::byte> &out_archive) noexcept override;

    [[nodiscard]] AccelArchiveResult import_mesh_archive(
        uint64_t stream_handle,
        uint64_t mesh_handle,
        const AccelArchiveMeshDesc &expected_desc,
        const AccelArchiveGeometryBinding &geometry_binding,
        luisa::span<const std::byte> archive_bytes) noexcept override;

    [[nodiscard]] AccelArchiveStagingBuffer acquire_staging_buffer(
        size_t min_bytes) noexcept override;

    void release_staging_buffer(
        AccelArchiveStagingBuffer &&buffer) noexcept override;

    [[nodiscard]] AccelArchiveResult import_mesh_archive_with_options(
        uint64_t stream_handle,
        uint64_t mesh_handle,
        const AccelArchiveMeshDesc &expected_desc,
        const AccelArchiveGeometryBinding &geometry_binding,
        luisa::span<const std::byte> archive_bytes,
        const AccelArchiveImportOptions &options) noexcept override;

private:
    struct CompatBlob {
        uint32_t magic;
        uint32_t version;
        uint32_t embree_version_number;
        uint32_t build_config_flags;
        uint32_t enabled_cpu_features;
        uint32_t payload_magic;
        uint64_t unpacked_as_bytes;
    };
    static_assert(sizeof(CompatBlob) == 32u);

    [[nodiscard]] AccelArchiveResult _check_compat_and_header(
        luisa::span<const std::byte> compat_blob,
        luisa::span<const std::byte> payload,
        uint64_t meta_unpacked_size,
        uint64_t &out_unpacked_bytes) const noexcept;

    struct StagingSlot {
        std::byte *ptr{nullptr};
        size_t capacity{0u};
        bool in_use{false};
        uint64_t lru_tick{0u};
    };

    EmbreeAccelArchiveOps _ops;
    EmbreeOutOfTreeABI _abi{};
    luisa::string _driver_fingerprint;
    bool _supported{false};
    mutable std::mutex _staging_mtx;
    mutable luisa::vector<StagingSlot> _staging_slots;
    mutable uint64_t _staging_tick{0u};
};

[[nodiscard]] inline AccelArchiveResult validate_embree_triangle_binding(
    const AccelOption &mesh_option,
    const AccelArchiveMeshDesc &expected_desc,
    const AccelArchiveGeometryBinding &binding,
    const std::byte *vbuf_data,
    size_t vbuf_total_bytes,
    const std::byte *tbuf_data,
    size_t tbuf_total_bytes,
    EmbreeResolvedTriangleGeometry &out_geom) noexcept {
    if (mesh_option.hint != expected_desc.option.hint ||
        mesh_option.allow_compaction != expected_desc.option.allow_compaction ||
        mesh_option.allow_update != expected_desc.option.allow_update ||
        static_cast<bool>(mesh_option.motion) != static_cast<bool>(expected_desc.option.motion)) {
        return accel_archive::make_archive_error(
            AccelArchiveStatus::OPTIONS_MISMATCH,
            "Target mesh AccelOption does not match archive AccelOption.");
    }
    if (vbuf_data == nullptr || tbuf_data == nullptr) {
        return accel_archive::make_archive_error(
            AccelArchiveStatus::INVALID_ARGUMENT,
            "Geometry binding buffers must be non-null host allocations.");
    }
    if (binding.vertex_stride != expected_desc.vertex_stride || binding.vertex_stride < 12u ||
        (binding.vertex_stride % 4u) != 0u) {
        return accel_archive::make_archive_error(
            AccelArchiveStatus::INVALID_ARGUMENT,
            "Geometry binding vertex_stride does not match archive descriptor.");
    }
    if (expected_desc.vertex_count > SIZE_MAX / expected_desc.vertex_stride ||
        expected_desc.triangle_count > SIZE_MAX / 12u) {
        return accel_archive::make_archive_error(
            AccelArchiveStatus::INVALID_ARGUMENT,
            "Geometry binding byte size overflows.");
    }
    const size_t expected_vb_bytes = static_cast<size_t>(expected_desc.vertex_count * expected_desc.vertex_stride);
    const size_t expected_tb_bytes = static_cast<size_t>(expected_desc.triangle_count * 12u);
    if (binding.vertex_buffer_size != expected_vb_bytes ||
        binding.triangle_buffer_size != expected_tb_bytes) {
        return accel_archive::make_archive_error(
            AccelArchiveStatus::INVALID_ARGUMENT,
            "Geometry binding buffer slice sizes do not match archive descriptor.");
    }
    if (binding.vertex_buffer_offset > vbuf_total_bytes ||
        binding.vertex_buffer_size > vbuf_total_bytes - binding.vertex_buffer_offset ||
        binding.triangle_buffer_offset > tbuf_total_bytes ||
        binding.triangle_buffer_size > tbuf_total_bytes - binding.triangle_buffer_offset) {
        return accel_archive::make_archive_error(
            AccelArchiveStatus::INVALID_ARGUMENT,
            "Geometry binding buffer offset/size exceeds buffer bounds.");
    }
    out_geom.vertex_buffer = vbuf_data;
    out_geom.vertex_offset = binding.vertex_buffer_offset;
    out_geom.vertex_stride = binding.vertex_stride;
    out_geom.vertex_count = static_cast<size_t>(expected_desc.vertex_count);
    out_geom.triangle_buffer = tbuf_data;
    out_geom.triangle_offset = binding.triangle_buffer_offset;
    out_geom.triangle_count = static_cast<size_t>(expected_desc.triangle_count);
    return {};
}

template<typename Traits>
[[nodiscard]] inline luisa::unique_ptr<EmbreeAccelArchiveExt> make_embree_accel_archive_ext(
    AccelArchiveBackendKind backend_kind,
    void *rtc_device,
    luisa::string device_fingerprint) noexcept {
    EmbreeAccelArchiveOps ops{};
    ops.backend_kind = backend_kind;
    ops.rtc_device = rtc_device;
    ops.device_fingerprint = std::move(device_fingerprint);
    ops.user_ctx = nullptr;
    ops.validate_stream = [](void *, uint64_t stream_handle) noexcept -> bool {
        return stream_handle != invalid_resource_handle && stream_handle != 0u;
    };
    ops.synchronize_stream = [](void *, uint64_t stream_handle) noexcept {
        Traits::synchronize_stream(stream_handle);
    };
    ops.get_mesh_export_state = [](void *, uint64_t mesh_handle, EmbreeMeshExportState &out_state) noexcept -> AccelArchiveResult {
        const auto *mesh = Traits::try_cast_mesh(mesh_handle);
        if (mesh == nullptr) {
            return accel_archive::make_archive_error(
                AccelArchiveStatus::INVALID_ARGUMENT,
                luisa::format("Resource handle does not denote a {}.", Traits::mesh_type_name()));
        }
        out_state.option = mesh->option();
        out_state.is_built = mesh->is_built();
        out_state.triangle_count = mesh->last_triangle_count();
        out_state.vertex_count = mesh->last_vertex_count();
        out_state.vertex_stride = mesh->last_vertex_stride();
        out_state.committed_rtc_scene = out_state.is_built ?
                                            Traits::committed_scene(mesh) :
                                            nullptr;
        return {};
    };
    ops.resolve_geometry_binding = [](void *, uint64_t mesh_handle, const AccelArchiveMeshDesc &expected_desc,
                                      const AccelArchiveGeometryBinding &binding,
                                      EmbreeResolvedTriangleGeometry &out_geom) noexcept -> AccelArchiveResult {
        const auto *mesh = Traits::try_cast_mesh(mesh_handle);
        if (mesh == nullptr) {
            return accel_archive::make_archive_error(
                AccelArchiveStatus::INVALID_ARGUMENT,
                luisa::format("Resource handle does not denote a {}.", Traits::mesh_type_name()));
        }
        if (binding.vertex_buffer == invalid_resource_handle || binding.vertex_buffer == 0u ||
            binding.triangle_buffer == invalid_resource_handle || binding.triangle_buffer == 0u) {
            return accel_archive::make_archive_error(
                AccelArchiveStatus::INVALID_ARGUMENT,
                "Embree BLAS import requires valid rebound vertex and triangle buffer handles.");
        }
        const auto [v_ptr, v_bytes] = Traits::buffer_span(binding.vertex_buffer);
        const auto [t_ptr, t_bytes] = Traits::buffer_span(binding.triangle_buffer);
        return validate_embree_triangle_binding(
            mesh->option(),
            expected_desc,
            binding,
            v_ptr,
            v_bytes,
            t_ptr,
            t_bytes,
            out_geom);
    };
    ops.import_mesh_bvh = [](void *, uint64_t mesh_handle, const EmbreeResolvedTriangleGeometry &geom,
                             luisa::span<const std::byte> bvh_payload, const EmbreeOutOfTreeABI &abi,
                             size_t &out_unpacked_bytes, EmbreeArchiveStatus &out_status) noexcept -> AccelArchiveResult {
        auto *mesh = Traits::try_cast_mesh(mesh_handle);
        if (mesh == nullptr) {
            out_status = EmbreeArchiveStatus::INVALID_ARGUMENT;
            return accel_archive::make_archive_error(
                AccelArchiveStatus::INVALID_ARGUMENT,
                luisa::format("Resource handle does not denote a {}.", Traits::mesh_type_name()));
        }
        static_cast<void>(mesh->import_from_archive(
            geom, bvh_payload, abi, out_unpacked_bytes, out_status));
        return {};
    };
    return luisa::make_unique<EmbreeAccelArchiveExt>(std::move(ops));
}

}// namespace luisa::compute

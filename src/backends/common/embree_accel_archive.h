#pragma once

#include <cstddef>
#include <cstdint>

#include <luisa/backends/ext/accel_archive_ext_interface.h>
#include <luisa/core/stl/string.h>
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

using EmbreeArchiveImportFn = EmbreeArchiveStatus (*)(
    void *rtc_scene,
    const void *in_buffer,
    size_t buffer_size,
    size_t *out_unpacked_bytes);

struct EmbreeMeshExportState {
    bool is_built{false};
    AccelOption option{};
    uint64_t triangle_count{0u};
    void *committed_rtc_scene{nullptr};
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
        EmbreeArchiveImportFn import_fn,
        size_t &out_unpacked_bytes,
        EmbreeArchiveStatus &out_embree_status) noexcept {nullptr};
};

class EmbreeAccelArchiveExt final : public AccelArchiveExt {
public:
    explicit EmbreeAccelArchiveExt(EmbreeAccelArchiveOps ops) noexcept;
    ~EmbreeAccelArchiveExt() noexcept = default;

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

    using QueryConfigFn = EmbreeArchiveStatus (*)(void *, void *);
    using ExportBvhFn = EmbreeArchiveStatus (*)(void *, void *, size_t, size_t *, size_t *);

    struct EmbreeConfigInfo {
        uint32_t struct_size;
        uint32_t archive_abi_version;
        uint32_t embree_version_number;
        uint32_t build_config_flags;
        uint32_t enabled_cpu_features;
        uint32_t supports_static_triangle_bvh;
        uint32_t supports_bvh4;
        uint32_t supports_bvh8;
    };

    [[nodiscard]] AccelArchiveResult _check_compat_and_header(
        luisa::span<const std::byte> compat_blob,
        luisa::span<const std::byte> payload,
        uint64_t meta_unpacked_size,
        uint64_t &out_unpacked_bytes) const noexcept;

    EmbreeAccelArchiveOps _ops;
    EmbreeConfigInfo _cfg{};
    luisa::string _driver_fingerprint;
    QueryConfigFn _query_fn{nullptr};
    ExportBvhFn _export_fn{nullptr};
    EmbreeArchiveImportFn _import_fn{nullptr};
    bool _supported{false};
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

}// namespace luisa::compute

#pragma once

#include <luisa/backends/ext/accel_archive_ext_interface.h>

#include <cstdint>
#include <memory>

namespace lc::dx {

class LCDevice;

class DxAccelArchiveExt final : public luisa::compute::AccelArchiveExt {
public:
    explicit DxAccelArchiveExt(LCDevice *device) noexcept;
    ~DxAccelArchiveExt() noexcept;

    [[nodiscard]] luisa::compute::AccelArchiveCapabilities capabilities() const noexcept override;

    [[nodiscard]] luisa::compute::AccelArchiveResult validate_mesh_archive(
        luisa::span<const std::byte> archive_bytes,
        const luisa::compute::AccelArchiveMeshDesc &expected_desc,
        luisa::compute::AccelArchiveMetadata *out_meta = nullptr) const noexcept override;

    [[nodiscard]] luisa::compute::AccelArchiveResult export_mesh_archive(
        uint64_t stream_handle,
        uint64_t mesh_handle,
        const luisa::compute::AccelArchiveMeshDesc &desc,
        luisa::vector<std::byte> &out_archive) noexcept override;

    [[nodiscard]] luisa::compute::AccelArchiveResult import_mesh_archive(
        uint64_t stream_handle,
        uint64_t mesh_handle,
        const luisa::compute::AccelArchiveMeshDesc &expected_desc,
        const luisa::compute::AccelArchiveGeometryBinding &geometry_binding,
        luisa::span<const std::byte> archive_bytes) noexcept override;

    [[nodiscard]] luisa::compute::AccelArchiveStagingBuffer acquire_staging_buffer(
        size_t min_bytes) noexcept override;

    void release_staging_buffer(
        luisa::compute::AccelArchiveStagingBuffer &&buffer) noexcept override;

    [[nodiscard]] luisa::compute::AccelArchiveResult import_mesh_archive_with_options(
        uint64_t stream_handle,
        uint64_t mesh_handle,
        const luisa::compute::AccelArchiveMeshDesc &expected_desc,
        const luisa::compute::AccelArchiveGeometryBinding &geometry_binding,
        luisa::span<const std::byte> archive_bytes,
        const luisa::compute::AccelArchiveImportOptions &options) noexcept override;

    [[nodiscard]] luisa::compute::AccelArchiveResult import_mesh_archive_file(
        uint64_t stream_handle,
        uint64_t mesh_handle,
        const luisa::compute::AccelArchiveMeshDesc &expected_desc,
        const luisa::compute::AccelArchiveGeometryBinding &geometry_binding,
        luisa::string_view file_path,
        const luisa::compute::AccelArchiveImportOptions &options = {}) noexcept override;

private:
    struct Impl;
    struct CompatInfo {
        uint32_t vendor_id;
        uint32_t device_id;
        uint32_t subsys_id;
        uint32_t revision;
        uint32_t magic;
        uint32_t version;
        uint64_t umd_version;
        uint32_t raytracing_tier;
        uint32_t enhanced_barriers;
        uint64_t deserialized_size;
    };

    [[nodiscard]] luisa::compute::AccelArchiveResult _check_compat_and_header(
        luisa::span<const std::byte> compat_blob,
        luisa::span<const std::byte> payload_prefix,
        uint64_t expected_payload_size,
        uint64_t meta_unpacked_size,
        uint64_t &out_deserialized_size) const noexcept;

    LCDevice *_device{nullptr};
    luisa::string _device_fingerprint;
    luisa::string _driver_fingerprint;
    CompatInfo _compat{};
    bool _supported{false};
    bool _gpu_upload_supported{false};
    std::shared_ptr<Impl> _impl;
};

}// namespace lc::dx

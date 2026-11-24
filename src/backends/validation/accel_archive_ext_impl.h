#pragma once

#include <luisa/backends/ext/accel_archive_ext_interface.h>

namespace lc::validation {

using namespace luisa;
using namespace luisa::compute;

class Device;

class AccelArchiveExtImpl final : public AccelArchiveExt {
public:
    Device *_device{nullptr};
    AccelArchiveExt *_native{nullptr};

    AccelArchiveExtImpl(Device *device, AccelArchiveExt *native) noexcept
        : _device{device}, _native{native} {}

    [[nodiscard]] AccelArchiveCapabilities capabilities() const noexcept override {
        return _native->capabilities();
    }

    [[nodiscard]] AccelArchiveResult validate_mesh_archive(
        luisa::span<const std::byte> archive_bytes,
        const AccelArchiveMeshDesc &expected_desc,
        AccelArchiveMetadata *out_meta = nullptr) const noexcept override {
        return _native->validate_mesh_archive(archive_bytes, expected_desc, out_meta);
    }

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
        size_t min_bytes) noexcept override {
        return _native->acquire_staging_buffer(min_bytes);
    }

    void release_staging_buffer(
        AccelArchiveStagingBuffer &&buffer) noexcept override {
        _native->release_staging_buffer(std::move(buffer));
    }

    [[nodiscard]] AccelArchiveResult import_mesh_archive_with_options(
        uint64_t stream_handle,
        uint64_t mesh_handle,
        const AccelArchiveMeshDesc &expected_desc,
        const AccelArchiveGeometryBinding &geometry_binding,
        luisa::span<const std::byte> archive_bytes,
        const AccelArchiveImportOptions &options) noexcept override;

    [[nodiscard]] AccelArchiveResult import_mesh_archive_file(
        uint64_t stream_handle,
        uint64_t mesh_handle,
        const AccelArchiveMeshDesc &expected_desc,
        const AccelArchiveGeometryBinding &geometry_binding,
        luisa::string_view file_path,
        const AccelArchiveImportOptions &options) noexcept override;
};

}// namespace lc::validation

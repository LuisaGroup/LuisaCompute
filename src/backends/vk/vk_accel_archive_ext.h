#pragma once

#include <luisa/backends/ext/accel_archive_ext_interface.h>

namespace lc::vk {

class Device;

class VkAccelArchiveExt final : public luisa::compute::AccelArchiveExt {
    Device *_device;

public:
    explicit VkAccelArchiveExt(Device *device) noexcept;
    ~VkAccelArchiveExt() noexcept = default;

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
};

}// namespace lc::vk

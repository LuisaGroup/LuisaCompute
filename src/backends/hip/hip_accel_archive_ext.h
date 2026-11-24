#pragma once

#include <cstdint>

#include <luisa/backends/ext/accel_archive_ext_interface.h>

namespace luisa::compute::hip {

class HIPDevice;

class HIPAccelArchiveExt final : public AccelArchiveExt {
public:
    explicit HIPAccelArchiveExt(HIPDevice *device) noexcept;
    ~HIPAccelArchiveExt() noexcept = default;

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
    struct CompatInfo {
        uint32_t magic;
        uint32_t struct_size;
        uint32_t pci_device_id;
        int32_t hip_runtime_version;
        int32_t hip_driver_version;
        uint32_t rtip;
        uint32_t gcn_arch_len;
        char gcn_arch[64];
        char hiprt_version[16];
        uint64_t geometry_size;
    };

    [[nodiscard]] AccelArchiveResult _check_compat_and_geom_header(
        luisa::span<const std::byte> compat_blob,
        luisa::span<const std::byte> payload,
        uint64_t meta_unpacked_size,
        uint64_t &out_geom_size) const noexcept;

    HIPDevice *_device{nullptr};
    luisa::string _device_fingerprint;
    luisa::string _driver_fingerprint;
    CompatInfo _compat{};
    bool _supported{false};
};

}// namespace luisa::compute::hip

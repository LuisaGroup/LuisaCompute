#pragma once

#include <cstdint>
#include <memory>

#include <cuda.h>

#include <luisa/backends/ext/accel_archive_ext_interface.h>
#include <luisa/core/stl/string.h>

#include "../optix_api.h"
#include "cuda_accel_archive_codec.h"

namespace luisa::compute::cuda {

class CUDADevice;

class CUDAAccelArchiveExt final : public AccelArchiveExt {
public:
    explicit CUDAAccelArchiveExt(CUDADevice *device) noexcept;
    ~CUDAAccelArchiveExt() noexcept;

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
    struct Impl;

    [[nodiscard]] AccelArchiveResult _validate_compat_and_relocation(
        luisa::span<const std::byte> compat_blob,
        luisa::span<const std::byte> payload,
        uint64_t meta_unpacked_size,
        OptixRelocationInfo &out_reloc_info,
        uint64_t &out_active_bvh_bytes) const noexcept;

    CUDADevice *_device{nullptr};
    luisa::string _device_fingerprint;
    luisa::string _driver_fingerprint;
    uint32_t _optix_abi_version{OPTIX_ABI_VERSION};
    uint32_t _compute_capability{0u};
    uint32_t _driver_version{0u};
    bool _supported{false};
    std::unique_ptr<Impl> _impl;
};

}// namespace luisa::compute::cuda

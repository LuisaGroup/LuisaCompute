#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>

#include <luisa/core/stl/string.h>
#include <luisa/backends/ext/accel_archive_ext_interface.h>

namespace luisa::compute::cuda {

inline constexpr uint32_t kCudaOptixAccelArchiveCompatMagic = 0x4f505458u; // "OPTX"
inline constexpr uint32_t kCudaOptixAccelArchiveCompatVersion = 1u;
inline constexpr uint32_t kCudaOptixAccelBufferAlignment = 128u;

struct CudaOptixAccelArchiveCompatBlob {
    uint32_t magic{kCudaOptixAccelArchiveCompatMagic};
    uint32_t version{kCudaOptixAccelArchiveCompatVersion};
    uint32_t optix_abi_version{0u};
    uint32_t cuda_compute_capability{0u};
    uint32_t cuda_driver_version{0u};
    uint32_t accel_alignment{kCudaOptixAccelBufferAlignment};
    uint64_t active_bvh_bytes{0u};
    uint64_t relocation_info[4]{};
};

static_assert(sizeof(CudaOptixAccelArchiveCompatBlob) == 64u);

[[nodiscard]] inline luisa::vector<std::byte> encode_cuda_optix_compat_blob(
    const CudaOptixAccelArchiveCompatBlob &blob) noexcept {
    luisa::vector<std::byte> out(sizeof(CudaOptixAccelArchiveCompatBlob));
    std::memcpy(out.data(), &blob, sizeof(CudaOptixAccelArchiveCompatBlob));
    return out;
}

[[nodiscard]] inline AccelArchiveResult decode_and_check_cuda_optix_compat_blob(
    luisa::span<const std::byte> compat_bytes,
    luisa::span<const std::byte> payload_bytes,
    uint32_t expected_optix_abi_version,
    uint32_t expected_compute_capability,
    uint32_t expected_driver_version,
    CudaOptixAccelArchiveCompatBlob &out_blob) noexcept {
    if (compat_bytes.size() != sizeof(CudaOptixAccelArchiveCompatBlob)) {
        return accel_archive::make_archive_error(
            AccelArchiveStatus::CORRUPT_HEADER,
            "CUDA/OptiX compatibility blob size mismatch.");
    }
    std::memcpy(&out_blob, compat_bytes.data(), sizeof(CudaOptixAccelArchiveCompatBlob));
    if (out_blob.magic != kCudaOptixAccelArchiveCompatMagic ||
        out_blob.version != kCudaOptixAccelArchiveCompatVersion) {
        return accel_archive::make_archive_error(
            AccelArchiveStatus::UNSUPPORTED_VERSION,
            "CUDA/OptiX compatibility blob magic or version mismatch.");
    }
    if (out_blob.accel_alignment != kCudaOptixAccelBufferAlignment) {
        return accel_archive::make_archive_error(
            AccelArchiveStatus::UNSUPPORTED_LAYOUT,
            "CUDA/OptiX GAS buffer alignment mismatch.");
    }
    if (out_blob.active_bvh_bytes == 0u ||
        out_blob.active_bvh_bytes != payload_bytes.size()) {
        return accel_archive::make_archive_error(
            AccelArchiveStatus::CORRUPT_PAYLOAD,
            "CUDA/OptiX GAS payload byte count does not match compatibility header.");
    }
    if (out_blob.optix_abi_version != expected_optix_abi_version) {
        return accel_archive::make_archive_error(
            AccelArchiveStatus::DRIVER_INCOMPATIBLE,
            "CUDA/OptiX ABI version mismatch.");
    }
    if (out_blob.cuda_compute_capability != expected_compute_capability) {
        return accel_archive::make_archive_error(
            AccelArchiveStatus::DEVICE_INCOMPATIBLE,
            "CUDA device compute capability mismatch.");
    }
    if (expected_driver_version != 0u &&
        out_blob.cuda_driver_version != expected_driver_version) {
        return accel_archive::make_archive_error(
            AccelArchiveStatus::DRIVER_INCOMPATIBLE,
            "CUDA driver version mismatch.");
    }
    return {};
}

}// namespace luisa::compute::cuda

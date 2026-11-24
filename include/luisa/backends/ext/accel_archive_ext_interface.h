#pragma once

#include <cstddef>
#include <cstdint>

#include <luisa/core/dll_export.h>
#include <luisa/core/stl/hash.h>
#include <luisa/core/stl/string.h>
#include <luisa/core/stl/vector.h>
#include <luisa/runtime/rhi/device_interface.h>

namespace luisa::compute {

enum class AccelArchiveStatus : uint32_t {
    OK = 0u,
    UNSUPPORTED_BACKEND,
    UNSUPPORTED_FEATURE,
    UNSUPPORTED_LAYOUT,
    INVALID_ARGUMENT,
    MESH_NOT_BUILT,
    INVALID_MAGIC,
    UNSUPPORTED_VERSION,
    TRUNCATED,
    CORRUPT_HEADER,
    CORRUPT_PAYLOAD,
    KEY_MISMATCH,
    OPTIONS_MISMATCH,
    DEVICE_INCOMPATIBLE,
    DRIVER_INCOMPATIBLE,
    BACKEND_ERROR,
};

[[nodiscard]] LUISA_RUNTIME_API luisa::string_view to_string(AccelArchiveStatus status) noexcept;

enum class AccelArchiveBackendKind : uint32_t {
    UNKNOWN = 0u,
    VULKAN = 1u,
    DX = 2u,
    HIP = 3u,
    CUDA = 4u,
    FALLBACK = 5u,
    SIMD = 6u,
};

[[nodiscard]] LUISA_RUNTIME_API luisa::string_view to_string(AccelArchiveBackendKind kind) noexcept;

enum class AccelArchiveReadiness : uint32_t {
    NONE = 0u,
    READY_IMMEDIATELY = 1u,
    READY_ON_STREAM = 2u,
};

struct AccelArchiveCapabilities {
    bool supported{false};
    bool static_triangle_blas{false};
    bool supports_compacted_export{false};
    bool requires_geometry_rebind{false};
    bool import_is_async_on_stream{false};
    AccelArchiveBackendKind backend_kind{AccelArchiveBackendKind::UNKNOWN};
    uint32_t envelope_version{1u};
    luisa::string device_fingerprint;
    luisa::string driver_fingerprint;
    luisa::string layout_info;
};

struct AccelArchiveMeshDesc {
    uint64_t vertex_count{0u};
    uint64_t triangle_count{0u};
    uint64_t vertex_stride{0u};
    AccelOption option{};
    luisa::Hash128 geometry_content_hash{};
    bool is_cluster{false};
    bool has_cutout_mask{false};
};

// Ephemeral device-local buffer handles used on import when a backend must
// rebind geometry storage (e.g., indexed Embree triangles) or when validation
// tracks buffer lifetimes. Never serialized into archives.
struct AccelArchiveGeometryBinding {
    uint64_t vertex_buffer{invalid_resource_handle};
    size_t vertex_buffer_offset{0u};
    size_t vertex_buffer_size{0u};
    size_t vertex_stride{0u};
    uint64_t triangle_buffer{invalid_resource_handle};
    size_t triangle_buffer_offset{0u};
    size_t triangle_buffer_size{0u};
};

struct AccelArchiveMetadata {
    uint32_t envelope_version{0u};
    AccelArchiveBackendKind backend_kind{AccelArchiveBackendKind::UNKNOWN};
    uint64_t vertex_count{0u};
    uint64_t triangle_count{0u};
    uint64_t vertex_stride{0u};
    AccelOption option{};
    bool is_compacted{false};
    luisa::Hash128 geometry_content_hash{};
    luisa::Hash128 archive_key{};
    uint64_t compat_blob_size{0u};
    uint64_t payload_size{0u};
    uint64_t unpacked_as_size{0u};
};

struct AccelArchiveResult {
    AccelArchiveStatus status{AccelArchiveStatus::OK};
    AccelArchiveReadiness readiness{AccelArchiveReadiness::NONE};
    uint64_t unpacked_as_size{0u};
    uint64_t serialized_payload_size{0u};
    bool is_compacted{false};
    luisa::string message;

    [[nodiscard]] explicit operator bool() const noexcept {
        return status == AccelArchiveStatus::OK;
    }
};

// Optional backend-specific disposable acceleration-structure cache extension.
//
// v1 scope: static triangle BLAS only (no updateable, motion, opacity
// micromap, or cluster BLAS). Archives are non-portable backend caches keyed
// by canonical geometry content and build flags; TLAS must be rebuilt fresh
// from restored BLAS handles and caller-owned instance descriptors.
class AccelArchiveExt : public DeviceExtension {

protected:
    ~AccelArchiveExt() noexcept = default;

public:
    static constexpr luisa::string_view name = "AccelArchiveExt";

    [[nodiscard]] virtual AccelArchiveCapabilities capabilities() const noexcept = 0;

    [[nodiscard]] virtual AccelArchiveResult validate_mesh_archive(
        luisa::span<const std::byte> archive_bytes,
        const AccelArchiveMeshDesc &expected_desc,
        AccelArchiveMetadata *out_meta = nullptr) const noexcept = 0;

    [[nodiscard]] virtual AccelArchiveResult export_mesh_archive(
        uint64_t stream_handle,
        uint64_t mesh_handle,
        const AccelArchiveMeshDesc &desc,
        luisa::vector<std::byte> &out_archive) noexcept = 0;

    [[nodiscard]] virtual AccelArchiveResult import_mesh_archive(
        uint64_t stream_handle,
        uint64_t mesh_handle,
        const AccelArchiveMeshDesc &expected_desc,
        const AccelArchiveGeometryBinding &geometry_binding,
        luisa::span<const std::byte> archive_bytes) noexcept = 0;
};

namespace accel_archive {

inline constexpr uint32_t kEnvelopeVersion = 1u;
inline constexpr uint32_t kStaticTriangleBlasKind = 1u;
inline constexpr char kEnvelopeMagic[8] = {'L', 'C', 'A', 'S', 'A', 'R', 'C', '\0'};

enum EnvelopeOptionFlags : uint32_t {
    kOptionAllowCompaction = 1u << 0u,
    kOptionAllowUpdate = 1u << 1u,
    kOptionIsCompacted = 1u << 2u,
};

struct AccelArchiveEnvelopeHeader {
    char magic[8];
    uint32_t version;
    uint32_t header_size;
    uint32_t backend_kind;
    uint32_t blas_kind;
    uint32_t option_hint;
    uint32_t option_flags;
    uint64_t vertex_count;
    uint64_t triangle_count;
    uint64_t vertex_stride;
    uint8_t geometry_content_hash[16];
    uint8_t archive_key[16];
    uint32_t compat_blob_size;
    uint32_t reserved;
    uint64_t payload_size;
    uint64_t unpacked_as_size;
    uint8_t compat_checksum[16];
    uint8_t payload_checksum[16];
    uint8_t header_checksum[16];
};

static_assert(sizeof(AccelArchiveEnvelopeHeader) == 160u);
static_assert(offsetof(AccelArchiveEnvelopeHeader, header_checksum) == 144u);

struct AccelArchiveEnvelopeView {
    AccelArchiveMetadata meta;
    luisa::span<const std::byte> compat_blob;
    luisa::span<const std::byte> payload;
};

[[nodiscard]] LUISA_RUNTIME_API luisa::Hash128 compute_static_triangle_geometry_hash(
    const void *vertex_data,
    size_t vertex_count,
    size_t vertex_stride,
    const void *triangle_data,
    size_t triangle_count) noexcept;

[[nodiscard]] LUISA_RUNTIME_API luisa::Hash128 compute_archive_key(
    const AccelArchiveMeshDesc &desc) noexcept;

[[nodiscard]] LUISA_RUNTIME_API AccelArchiveResult validate_v1_static_triangle_desc(
    const AccelArchiveMeshDesc &desc) noexcept;

[[nodiscard]] LUISA_RUNTIME_API AccelArchiveResult make_archive_error(
    AccelArchiveStatus status,
    luisa::string message) noexcept;

[[nodiscard]] LUISA_RUNTIME_API AccelArchiveResult encode_envelope(
    AccelArchiveBackendKind backend,
    const AccelArchiveMeshDesc &desc,
    bool is_compacted,
    uint64_t unpacked_as_size,
    luisa::span<const std::byte> compat_blob,
    luisa::span<const std::byte> payload,
    luisa::vector<std::byte> &out_archive) noexcept;

[[nodiscard]] LUISA_RUNTIME_API AccelArchiveResult decode_and_validate_envelope(
    luisa::span<const std::byte> archive_bytes,
    AccelArchiveBackendKind expected_backend,
    const AccelArchiveMeshDesc &expected_desc,
    AccelArchiveEnvelopeView &out_view) noexcept;

}// namespace accel_archive

}// namespace luisa::compute

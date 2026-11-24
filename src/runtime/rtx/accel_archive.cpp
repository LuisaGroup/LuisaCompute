#include <algorithm>
#include <cstring>
#include <limits>

#include <luisa/core/magic_enum.h>
#include <luisa/backends/ext/accel_archive_ext_interface.h>

namespace luisa::compute {

luisa::string_view to_string(AccelArchiveStatus status) noexcept {
    switch (status) {
        case AccelArchiveStatus::OK: return "OK";
        case AccelArchiveStatus::UNSUPPORTED_BACKEND: return "UNSUPPORTED_BACKEND";
        case AccelArchiveStatus::UNSUPPORTED_FEATURE: return "UNSUPPORTED_FEATURE";
        case AccelArchiveStatus::UNSUPPORTED_LAYOUT: return "UNSUPPORTED_LAYOUT";
        case AccelArchiveStatus::INVALID_ARGUMENT: return "INVALID_ARGUMENT";
        case AccelArchiveStatus::MESH_NOT_BUILT: return "MESH_NOT_BUILT";
        case AccelArchiveStatus::INVALID_MAGIC: return "INVALID_MAGIC";
        case AccelArchiveStatus::UNSUPPORTED_VERSION: return "UNSUPPORTED_VERSION";
        case AccelArchiveStatus::TRUNCATED: return "TRUNCATED";
        case AccelArchiveStatus::CORRUPT_HEADER: return "CORRUPT_HEADER";
        case AccelArchiveStatus::CORRUPT_PAYLOAD: return "CORRUPT_PAYLOAD";
        case AccelArchiveStatus::KEY_MISMATCH: return "KEY_MISMATCH";
        case AccelArchiveStatus::OPTIONS_MISMATCH: return "OPTIONS_MISMATCH";
        case AccelArchiveStatus::DEVICE_INCOMPATIBLE: return "DEVICE_INCOMPATIBLE";
        case AccelArchiveStatus::DRIVER_INCOMPATIBLE: return "DRIVER_INCOMPATIBLE";
        case AccelArchiveStatus::BACKEND_ERROR: return "BACKEND_ERROR";
    }
    return "UNKNOWN";
}

luisa::string_view to_string(AccelArchiveBackendKind kind) noexcept {
    switch (kind) {
        case AccelArchiveBackendKind::UNKNOWN: return "unknown";
        case AccelArchiveBackendKind::VULKAN: return "vk";
        case AccelArchiveBackendKind::DX: return "dx";
        case AccelArchiveBackendKind::HIP: return "hip";
        case AccelArchiveBackendKind::CUDA: return "cuda";
        case AccelArchiveBackendKind::FALLBACK: return "fallback";
        case AccelArchiveBackendKind::SIMD: return "simd";
    }
    return "unknown";
}

namespace accel_archive {

namespace {

constexpr uint64_t kGeometryHashSeed = 0x4c43415347454f4dull;
constexpr uint64_t kArchiveKeySeed = 0x4c4341534b455931ull;
constexpr uint64_t kHeaderChecksumSeed = 0x4c43415348445231ull;
constexpr uint64_t kCompatChecksumSeed = 0x4c434153434d5031ull;
constexpr uint64_t kPayloadChecksumSeed = 0x4c43415350415931ull;

[[nodiscard]] luisa::Hash128 hash_span_128(luisa::span<const std::byte> bytes, uint64_t seed) noexcept {
    static const std::byte dummy{0};
    const void *ptr = bytes.empty() ? static_cast<const void *>(&dummy) : static_cast<const void *>(bytes.data());
    return luisa::hash128(ptr, bytes.size(), seed);
}

[[nodiscard]] bool hash_equals(const uint8_t (&lhs)[16], const luisa::Hash128 &rhs) noexcept {
    return std::memcmp(lhs, rhs.data().data(), 16u) == 0;
}

void copy_hash(uint8_t (&dst)[16], const luisa::Hash128 &src) noexcept {
    std::memcpy(dst, src.data().data(), 16u);
}

[[nodiscard]] luisa::Hash128 to_hash128(const uint8_t (&src)[16]) noexcept {
    luisa::Hash128 h;
    std::memcpy(h.data().data(), src, 16u);
    return h;
}

struct CanonicalKeyRecord {
    uint8_t geometry_content_hash[16];
    uint64_t vertex_count;
    uint64_t triangle_count;
    uint64_t vertex_stride;
    uint32_t hint;
    uint8_t allow_compaction;
    uint8_t allow_update;
    uint8_t is_cluster;
    uint8_t has_cutout_mask;
    uint32_t motion_keyframes;
    uint32_t reserved;
};
static_assert(sizeof(CanonicalKeyRecord) == 56u);

}// namespace

luisa::Hash128 compute_static_triangle_geometry_hash(
    const void *vertex_data,
    size_t vertex_count,
    size_t vertex_stride,
    const void *triangle_data,
    size_t triangle_count) noexcept {
    // Hash canonical float3 position bytes per vertex (ignoring any padding
    // bytes beyond 12B in vertex_stride) plus exact triangle index bytes and
    // explicit dimensions so geometry edits always alter the key.
    struct GeometryPrefix {
        uint64_t vertex_count;
        uint64_t triangle_count;
        uint64_t vertex_stride;
    } prefix{
        static_cast<uint64_t>(vertex_count),
        static_cast<uint64_t>(triangle_count),
        static_cast<uint64_t>(vertex_stride),
    };
    const size_t pos_bytes = vertex_count * 12u;
    const size_t tri_bytes = triangle_count * 12u;
    luisa::vector<std::byte> canonical(sizeof(GeometryPrefix) + pos_bytes + tri_bytes);
    std::memcpy(canonical.data(), &prefix, sizeof(GeometryPrefix));
    size_t offset = sizeof(GeometryPrefix);
    if (vertex_data != nullptr && vertex_count > 0u && vertex_stride >= 12u) {
        const auto *src_verts = static_cast<const std::byte *>(vertex_data);
        if (vertex_stride == 12u) {
            std::memcpy(canonical.data() + offset, src_verts, pos_bytes);
        } else {
            for (size_t i = 0u; i < vertex_count; ++i) {
                std::memcpy(canonical.data() + offset + i * 12u, src_verts + i * vertex_stride, 12u);
            }
        }
    }
    offset += pos_bytes;
    if (triangle_data != nullptr && triangle_count > 0u) {
        std::memcpy(canonical.data() + offset, triangle_data, tri_bytes);
    }
    return luisa::hash128(canonical.data(), canonical.size(), kGeometryHashSeed);
}

luisa::Hash128 compute_archive_key(const AccelArchiveMeshDesc &desc) noexcept {
    CanonicalKeyRecord rec{};
    copy_hash(rec.geometry_content_hash, desc.geometry_content_hash);
    rec.vertex_count = desc.vertex_count;
    rec.triangle_count = desc.triangle_count;
    rec.vertex_stride = desc.vertex_stride;
    rec.hint = luisa::to_underlying(desc.option.hint);
    rec.allow_compaction = desc.option.allow_compaction ? 1u : 0u;
    rec.allow_update = desc.option.allow_update ? 1u : 0u;
    rec.is_cluster = desc.is_cluster ? 1u : 0u;
    rec.has_cutout_mask = desc.has_cutout_mask ? 1u : 0u;
    rec.motion_keyframes = desc.option.motion.keyframe_count;
    return luisa::hash128(&rec, sizeof(rec), kArchiveKeySeed);
}

AccelArchiveResult validate_v1_static_triangle_desc(const AccelArchiveMeshDesc &desc) noexcept {
    if (desc.is_cluster) {
        return make_archive_error(
            AccelArchiveStatus::UNSUPPORTED_FEATURE,
            "AccelArchiveExt v1 does not support cluster (mega-geometry) BLAS.");
    }
    if (desc.has_cutout_mask) {
        return make_archive_error(
            AccelArchiveStatus::UNSUPPORTED_FEATURE,
            "AccelArchiveExt v1 does not support opacity-micromap BLAS.");
    }
    if (desc.option.motion.is_enabled()) {
        return make_archive_error(
            AccelArchiveStatus::UNSUPPORTED_FEATURE,
            "AccelArchiveExt v1 does not support motion-blur BLAS.");
    }
    if (desc.option.allow_update) {
        return make_archive_error(
            AccelArchiveStatus::UNSUPPORTED_FEATURE,
            "AccelArchiveExt v1 supports static non-updateable triangle BLAS only.");
    }
    if (desc.vertex_count < 3u || desc.triangle_count == 0u) {
        return make_archive_error(
            AccelArchiveStatus::INVALID_ARGUMENT,
            "Static triangle BLAS must contain at least 3 vertices and 1 triangle.");
    }
    if (desc.vertex_stride < 12u || (desc.vertex_stride % 4u) != 0u) {
        return make_archive_error(
            AccelArchiveStatus::INVALID_ARGUMENT,
            "Static triangle BLAS vertex_stride must be >= 12 and 4-byte aligned.");
    }
    return {};
}

AccelArchiveResult make_archive_error(AccelArchiveStatus status, luisa::string message) noexcept {
    AccelArchiveResult res{};
    res.status = status;
    res.readiness = AccelArchiveReadiness::NONE;
    res.message = std::move(message);
    return res;
}

AccelArchiveResult encode_envelope(
    AccelArchiveBackendKind backend,
    const AccelArchiveMeshDesc &desc,
    bool is_compacted,
    uint64_t unpacked_as_size,
    luisa::span<const std::byte> compat_blob,
    luisa::span<const std::byte> payload,
    luisa::vector<std::byte> &out_archive) noexcept {
    if (backend == AccelArchiveBackendKind::UNKNOWN) {
        return make_archive_error(AccelArchiveStatus::INVALID_ARGUMENT, "Backend kind must not be UNKNOWN.");
    }
    if (auto v = validate_v1_static_triangle_desc(desc); !v) {
        return v;
    }
    if (compat_blob.size() > std::numeric_limits<uint32_t>::max()) {
        return make_archive_error(AccelArchiveStatus::INVALID_ARGUMENT, "Compatibility blob exceeds 32-bit size.");
    }
    if (payload.empty() || unpacked_as_size == 0u) {
        return make_archive_error(AccelArchiveStatus::INVALID_ARGUMENT, "Archive payload and unpacked AS size must be non-zero.");
    }

    AccelArchiveEnvelopeHeader hdr{};
    std::memcpy(hdr.magic, kEnvelopeMagic, sizeof(hdr.magic));
    hdr.version = kEnvelopeVersion;
    hdr.header_size = static_cast<uint32_t>(sizeof(AccelArchiveEnvelopeHeader));
    hdr.backend_kind = luisa::to_underlying(backend);
    hdr.blas_kind = kStaticTriangleBlasKind;
    hdr.option_hint = luisa::to_underlying(desc.option.hint);
    hdr.option_flags = 0u;
    if (desc.option.allow_compaction) {
        hdr.option_flags |= kOptionAllowCompaction;
    }
    if (desc.option.allow_update) {
        hdr.option_flags |= kOptionAllowUpdate;
    }
    if (is_compacted) {
        hdr.option_flags |= kOptionIsCompacted;
    }
    hdr.vertex_count = desc.vertex_count;
    hdr.triangle_count = desc.triangle_count;
    hdr.vertex_stride = desc.vertex_stride;
    copy_hash(hdr.geometry_content_hash, desc.geometry_content_hash);
    copy_hash(hdr.archive_key, compute_archive_key(desc));
    hdr.compat_blob_size = static_cast<uint32_t>(compat_blob.size());
    hdr.reserved = 0u;
    hdr.payload_size = static_cast<uint64_t>(payload.size());
    hdr.unpacked_as_size = unpacked_as_size;
    copy_hash(hdr.compat_checksum, hash_span_128(compat_blob, kCompatChecksumSeed));
    copy_hash(hdr.payload_checksum, hash_span_128(payload, kPayloadChecksumSeed));
    copy_hash(
        hdr.header_checksum,
        luisa::hash128(&hdr, offsetof(AccelArchiveEnvelopeHeader, header_checksum), kHeaderChecksumSeed));

    const size_t total_size = sizeof(AccelArchiveEnvelopeHeader) + compat_blob.size() + payload.size();
    out_archive.resize(total_size);
    std::memcpy(out_archive.data(), &hdr, sizeof(AccelArchiveEnvelopeHeader));
    if (!compat_blob.empty()) {
        std::memcpy(out_archive.data() + sizeof(AccelArchiveEnvelopeHeader), compat_blob.data(), compat_blob.size());
    }
    std::memcpy(
        out_archive.data() + sizeof(AccelArchiveEnvelopeHeader) + compat_blob.size(),
        payload.data(),
        payload.size());

    AccelArchiveResult ok{};
    ok.status = AccelArchiveStatus::OK;
    ok.readiness = AccelArchiveReadiness::NONE;
    ok.unpacked_as_size = unpacked_as_size;
    ok.serialized_payload_size = payload.size();
    ok.is_compacted = is_compacted;
    return ok;
}

AccelArchiveResult decode_and_validate_envelope(
    luisa::span<const std::byte> archive_bytes,
    AccelArchiveBackendKind expected_backend,
    const AccelArchiveMeshDesc &expected_desc,
    AccelArchiveEnvelopeView &out_view) noexcept {
    if (archive_bytes.size() < sizeof(kEnvelopeMagic)) {
        return make_archive_error(AccelArchiveStatus::TRUNCATED, "Archive truncated before envelope magic.");
    }
    if (std::memcmp(archive_bytes.data(), kEnvelopeMagic, sizeof(kEnvelopeMagic)) != 0) {
        return make_archive_error(AccelArchiveStatus::INVALID_MAGIC, "Invalid acceleration-structure archive magic.");
    }
    if (archive_bytes.size() < 12u) {
        return make_archive_error(AccelArchiveStatus::TRUNCATED, "Archive truncated before envelope version.");
    }
    uint32_t version = 0u;
    std::memcpy(&version, archive_bytes.data() + 8u, sizeof(version));
    if (version != kEnvelopeVersion) {
        return make_archive_error(AccelArchiveStatus::UNSUPPORTED_VERSION, "Unsupported acceleration-structure archive envelope version.");
    }
    if (archive_bytes.size() < sizeof(AccelArchiveEnvelopeHeader)) {
        return make_archive_error(AccelArchiveStatus::TRUNCATED, "Archive truncated inside envelope header.");
    }

    AccelArchiveEnvelopeHeader hdr{};
    std::memcpy(&hdr, archive_bytes.data(), sizeof(AccelArchiveEnvelopeHeader));
    if (hdr.header_size != sizeof(AccelArchiveEnvelopeHeader)) {
        return make_archive_error(AccelArchiveStatus::CORRUPT_HEADER, "Unexpected envelope header size.");
    }
    const auto computed_hdr_checksum = luisa::hash128(
        &hdr, offsetof(AccelArchiveEnvelopeHeader, header_checksum), kHeaderChecksumSeed);
    if (!hash_equals(hdr.header_checksum, computed_hdr_checksum)) {
        return make_archive_error(AccelArchiveStatus::CORRUPT_HEADER, "Envelope header checksum mismatch.");
    }
    if (hdr.reserved != 0u || hdr.payload_size == 0u || hdr.unpacked_as_size == 0u) {
        return make_archive_error(AccelArchiveStatus::CORRUPT_HEADER, "Corrupt envelope header metadata.");
    }

    const uint64_t compat_end = uint64_t{hdr.header_size} + uint64_t{hdr.compat_blob_size};
    if (hdr.payload_size > std::numeric_limits<uint64_t>::max() - compat_end) {
        return make_archive_error(AccelArchiveStatus::CORRUPT_HEADER, "Archive payload size overflow.");
    }
    const uint64_t total_expected = compat_end + hdr.payload_size;
    if (archive_bytes.size() < total_expected) {
        return make_archive_error(AccelArchiveStatus::TRUNCATED, "Archive truncated before end of compatibility/payload data.");
    }
    if (archive_bytes.size() > total_expected) {
        return make_archive_error(AccelArchiveStatus::CORRUPT_HEADER, "Archive contains unexpected trailing bytes.");
    }

    const auto compat_span = archive_bytes.subspan(hdr.header_size, hdr.compat_blob_size);
    const auto payload_span = archive_bytes.subspan(static_cast<size_t>(compat_end), static_cast<size_t>(hdr.payload_size));
    if (!hash_equals(hdr.compat_checksum, hash_span_128(compat_span, kCompatChecksumSeed))) {
        return make_archive_error(AccelArchiveStatus::CORRUPT_PAYLOAD, "Compatibility blob checksum mismatch.");
    }
    if (!hash_equals(hdr.payload_checksum, hash_span_128(payload_span, kPayloadChecksumSeed))) {
        return make_archive_error(AccelArchiveStatus::CORRUPT_PAYLOAD, "Native payload checksum mismatch.");
    }

    if (hdr.blas_kind != kStaticTriangleBlasKind || (hdr.option_flags & kOptionAllowUpdate) != 0u) {
        return make_archive_error(AccelArchiveStatus::UNSUPPORTED_FEATURE, "Archive contains unsupported BLAS kind or flags.");
    }
    if (hdr.option_hint != luisa::to_underlying(AccelOption::UsageHint::FAST_TRACE) &&
        hdr.option_hint != luisa::to_underlying(AccelOption::UsageHint::FAST_BUILD)) {
        return make_archive_error(AccelArchiveStatus::CORRUPT_HEADER, "Invalid AccelOption::UsageHint in envelope.");
    }

    if (expected_backend != AccelArchiveBackendKind::UNKNOWN &&
        hdr.backend_kind != luisa::to_underlying(expected_backend)) {
        return make_archive_error(AccelArchiveStatus::UNSUPPORTED_BACKEND, "Archive backend kind does not match target device backend.");
    }

    if (auto v = validate_v1_static_triangle_desc(expected_desc); !v) {
        return v;
    }

    const bool archive_allow_compaction = (hdr.option_flags & kOptionAllowCompaction) != 0u;
    if (hdr.option_hint != luisa::to_underlying(expected_desc.option.hint) ||
        archive_allow_compaction != expected_desc.option.allow_compaction) {
        return make_archive_error(AccelArchiveStatus::OPTIONS_MISMATCH, "Archive build options do not match requested AccelOption.");
    }

    const auto expected_key = compute_archive_key(expected_desc);
    if (hdr.vertex_count != expected_desc.vertex_count ||
        hdr.triangle_count != expected_desc.triangle_count ||
        hdr.vertex_stride != expected_desc.vertex_stride ||
        !hash_equals(hdr.geometry_content_hash, expected_desc.geometry_content_hash) ||
        !hash_equals(hdr.archive_key, expected_key)) {
        return make_archive_error(AccelArchiveStatus::KEY_MISMATCH, "Archive geometry content or dimensions do not match mesh.");
    }

    out_view.meta.envelope_version = hdr.version;
    out_view.meta.backend_kind = static_cast<AccelArchiveBackendKind>(hdr.backend_kind);
    out_view.meta.vertex_count = hdr.vertex_count;
    out_view.meta.triangle_count = hdr.triangle_count;
    out_view.meta.vertex_stride = hdr.vertex_stride;
    out_view.meta.option = {};
    out_view.meta.option.hint = static_cast<AccelOption::UsageHint>(hdr.option_hint);
    out_view.meta.option.allow_compaction = archive_allow_compaction;
    out_view.meta.option.allow_update = false;
    out_view.meta.is_compacted = (hdr.option_flags & kOptionIsCompacted) != 0u;
    out_view.meta.geometry_content_hash = to_hash128(hdr.geometry_content_hash);
    out_view.meta.archive_key = to_hash128(hdr.archive_key);
    out_view.meta.compat_blob_size = hdr.compat_blob_size;
    out_view.meta.payload_size = hdr.payload_size;
    out_view.meta.unpacked_as_size = hdr.unpacked_as_size;
    out_view.compat_blob = compat_span;
    out_view.payload = payload_span;

    AccelArchiveResult ok{};
    ok.status = AccelArchiveStatus::OK;
    ok.readiness = AccelArchiveReadiness::NONE;
    ok.unpacked_as_size = hdr.unpacked_as_size;
    ok.serialized_payload_size = hdr.payload_size;
    ok.is_compacted = out_view.meta.is_compacted;
    return ok;
}

}// namespace accel_archive

}// namespace luisa::compute

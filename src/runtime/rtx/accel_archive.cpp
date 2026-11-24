#include <algorithm>
#include <array>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <thread>
#include <vector>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <malloc.h>
#include <windows.h>
#endif

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

[[nodiscard]] luisa::Hash128 hash_span_128(luisa::span<const std::byte> bytes, uint64_t seed) noexcept {
    static const std::byte dummy{0};
    const void *ptr = bytes.empty() ? static_cast<const void *>(&dummy) : static_cast<const void *>(bytes.data());
    return luisa::hash128(ptr, bytes.size(), seed);
}

inline void fast_checksum_step(
    uint32_t w,
    uint32_t idx,
    uint32_t &lane0,
    uint32_t &lane1,
    uint32_t &lane2,
    uint32_t &lane3) noexcept {
    uint32_t x = w ^ (idx * 0x9E3779B1u + 0x85EBCA6Bu);
    x ^= x >> 16u;
    x *= 0x7FEB352Du;
    x ^= x >> 15u;
    x *= 0x846CA68Bu;
    x ^= x >> 16u;

    uint32_t y = (w + 0x27D4EB2Du) ^ ((idx ^ 0xA5A55A5Au) * 0xC2B2AE35u);
    y ^= y >> 16u;
    y *= 0x85EBCA6Bu;
    y ^= y >> 13u;
    y *= 0xC2B2AE35u;
    y ^= y >> 16u;

    lane0 += x;
    lane1 ^= x;
    lane2 += y;
    lane3 ^= y;
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

void compute_payload_checksum_partial(
    luisa::span<const std::byte> slice,
    uint32_t start_word_index,
    uint32_t (&out_lanes)[4]) noexcept {
    uint32_t l0 = 0u;
    uint32_t l1 = 0u;
    uint32_t l2 = 0u;
    uint32_t l3 = 0u;
    const size_t nbytes = slice.size();
    const std::byte *p = slice.data();
    const size_t full_words = nbytes / 4u;
    size_t i = 0u;

    // 4-way unrolled scalar accumulator loop for ILP / vectorization while keeping exact word-index semantics.
    uint32_t a0 = 0u, a1 = 0u, a2 = 0u, a3 = 0u;
    uint32_t b0 = 0u, b1 = 0u, b2 = 0u, b3 = 0u;
    uint32_t c0 = 0u, c1 = 0u, c2 = 0u, c3 = 0u;
    uint32_t d0 = 0u, d1 = 0u, d2 = 0u, d3 = 0u;
    while (i + 4u <= full_words) {
        uint32_t w0 = 0u, w1 = 0u, w2 = 0u, w3 = 0u;
        std::memcpy(&w0, p + (i + 0u) * 4u, 4u);
        std::memcpy(&w1, p + (i + 1u) * 4u, 4u);
        std::memcpy(&w2, p + (i + 2u) * 4u, 4u);
        std::memcpy(&w3, p + (i + 3u) * 4u, 4u);
        const uint32_t base_idx = start_word_index + static_cast<uint32_t>(i);
        fast_checksum_step(w0, base_idx + 0u, a0, a1, a2, a3);
        fast_checksum_step(w1, base_idx + 1u, b0, b1, b2, b3);
        fast_checksum_step(w2, base_idx + 2u, c0, c1, c2, c3);
        fast_checksum_step(w3, base_idx + 3u, d0, d1, d2, d3);
        i += 4u;
    }
    l0 = a0 + b0 + c0 + d0;
    l1 = a1 ^ b1 ^ c1 ^ d1;
    l2 = a2 + b2 + c2 + d2;
    l3 = a3 ^ b3 ^ c3 ^ d3;

    while (i < full_words) {
        uint32_t w = 0u;
        std::memcpy(&w, p + i * 4u, 4u);
        fast_checksum_step(w, start_word_index + static_cast<uint32_t>(i), l0, l1, l2, l3);
        ++i;
    }
    const size_t rem = nbytes & 3u;
    if (rem != 0u) {
        uint32_t w = 0u;
        std::memcpy(&w, p + full_words * 4u, rem);
        fast_checksum_step(w, start_word_index + static_cast<uint32_t>(full_words), l0, l1, l2, l3);
    }

    out_lanes[0] = l0;
    out_lanes[1] = l1;
    out_lanes[2] = l2;
    out_lanes[3] = l3;
}

void finalize_payload_checksum(
    const uint32_t (&lanes)[4],
    uint64_t total_byte_size,
    uint8_t (&out_checksum)[16]) noexcept {
    const uint32_t sz_lo = static_cast<uint32_t>(total_byte_size & 0xffffffffull);
    const uint32_t sz_hi = static_cast<uint32_t>(total_byte_size >> 32u);
    const uint32_t out_w[4]{
        lanes[0] ^ sz_lo,
        lanes[1] ^ sz_hi,
        lanes[2] + sz_lo * 0x9E3779B1u + 0x243F6A88u,
        lanes[3] ^ ((sz_lo * 0x85EBCA6Bu) ^ 0xB7E15162u),
    };
    std::memcpy(out_checksum, out_w, 16u);
}

namespace {

void run_parallel_copy_and_checksum(
    std::byte *dst,
    luisa::span<const std::byte> payload,
    bool do_copy,
    bool do_checksum,
    uint8_t (&out_checksum)[16]) noexcept {
    const size_t total = payload.size();
    const std::byte *src = payload.data();
    constexpr size_t kParallelThreshold = 256ull * 1024ull;
    if (total < kParallelThreshold) {
        if (do_copy && dst != nullptr && total > 0u) {
            std::memcpy(dst, src, total);
        }
        if (do_checksum) {
            uint32_t lanes[4]{};
            compute_payload_checksum_partial(payload, 0u, lanes);
            finalize_payload_checksum(lanes, total, out_checksum);
        }
        return;
    }

    const unsigned hw = std::max(2u, std::thread::hardware_concurrency());
    const size_t max_workers = std::min<size_t>({static_cast<size_t>(hw), size_t{8u}, std::max<size_t>(1u, total / (128ull * 1024ull))});
    if (max_workers >= 2u) {
        struct SliceTask {
            const std::byte *src{nullptr};
            std::byte *dst{nullptr};
            size_t len{0u};
            uint32_t start_word{0u};
            uint32_t lanes[4]{};
            bool do_copy{false};
            bool do_checksum{false};
#ifdef _WIN32
            PTP_WORK work{nullptr};
#endif
        };

        auto run_task = [](SliceTask *t) noexcept {
            if (t->do_copy && !t->do_checksum) {
                std::memcpy(t->dst, t->src, t->len);
                return;
            }
            if (!t->do_copy && t->do_checksum) {
                compute_payload_checksum_partial(luisa::span<const std::byte>{t->src, t->len}, t->start_word, t->lanes);
                return;
            }
            // Copy + checksum chunked in 128 KiB L1/L2-hot strips.
            constexpr size_t kSubChunk = 128ull * 1024ull;
            uint32_t l0 = 0u, l1 = 0u, l2 = 0u, l3 = 0u;
            size_t pos = 0u;
            while (pos < t->len) {
                const size_t n = std::min(kSubChunk, t->len - pos);
                std::memcpy(t->dst + pos, t->src + pos, n);
                uint32_t sub_lanes[4]{};
                compute_payload_checksum_partial(
                    luisa::span<const std::byte>{t->src + pos, n},
                    t->start_word + static_cast<uint32_t>(pos / 4u),
                    sub_lanes);
                l0 += sub_lanes[0];
                l1 ^= sub_lanes[1];
                l2 += sub_lanes[2];
                l3 ^= sub_lanes[3];
                pos += n;
            }
            t->lanes[0] = l0;
            t->lanes[1] = l1;
            t->lanes[2] = l2;
            t->lanes[3] = l3;
        };

        std::array<SliceTask, 8u> tasks{};
        // Word-aligned boundaries (multiples of 4096 bytes) ensure exact 4-byte word indices across workers.
        const size_t raw_chunk = (total + max_workers - 1u) / max_workers;
        const size_t aligned_chunk = std::max<size_t>(4096u, ((raw_chunk + 4095u) / 4096u) * 4096u);
        size_t active = 0u;
        for (size_t i = 0u; i < max_workers; ++i) {
            const size_t begin = i * aligned_chunk;
            if (begin >= total) {
                break;
            }
            const size_t len = std::min(aligned_chunk, total - begin);
            tasks[active].src = src + begin;
            tasks[active].dst = do_copy ? (dst + begin) : nullptr;
            tasks[active].len = len;
            tasks[active].start_word = static_cast<uint32_t>(begin / 4u);
            tasks[active].do_copy = do_copy;
            tasks[active].do_checksum = do_checksum;
            ++active;
        }

        if (active >= 2u) {
#ifdef _WIN32
            auto work_cb = [](PTP_CALLBACK_INSTANCE, PVOID ctx, PTP_WORK) noexcept {
                auto *t = static_cast<SliceTask *>(ctx);
                if (t->do_copy && !t->do_checksum) {
                    std::memcpy(t->dst, t->src, t->len);
                } else if (!t->do_copy && t->do_checksum) {
                    compute_payload_checksum_partial(luisa::span<const std::byte>{t->src, t->len}, t->start_word, t->lanes);
                } else {
                    constexpr size_t kSubChunk = 128ull * 1024ull;
                    uint32_t l0 = 0u, l1 = 0u, l2 = 0u, l3 = 0u;
                    size_t pos = 0u;
                    while (pos < t->len) {
                        const size_t n = std::min(kSubChunk, t->len - pos);
                        std::memcpy(t->dst + pos, t->src + pos, n);
                        uint32_t sub_lanes[4]{};
                        compute_payload_checksum_partial(
                            luisa::span<const std::byte>{t->src + pos, n},
                            t->start_word + static_cast<uint32_t>(pos / 4u),
                            sub_lanes);
                        l0 += sub_lanes[0];
                        l1 ^= sub_lanes[1];
                        l2 += sub_lanes[2];
                        l3 ^= sub_lanes[3];
                        pos += n;
                    }
                    t->lanes[0] = l0;
                    t->lanes[1] = l1;
                    t->lanes[2] = l2;
                    t->lanes[3] = l3;
                }
            };
            bool all_created = true;
            size_t created = 0u;
            for (size_t i = 0u; i < active; ++i) {
                tasks[i].work = CreateThreadpoolWork(work_cb, &tasks[i], nullptr);
                if (tasks[i].work == nullptr) {
                    all_created = false;
                    break;
                }
                ++created;
                SubmitThreadpoolWork(tasks[i].work);
            }
            for (size_t i = 0u; i < created; ++i) {
                WaitForThreadpoolWorkCallbacks(tasks[i].work, FALSE);
                CloseThreadpoolWork(tasks[i].work);
            }
            if (all_created) {
                if (do_checksum) {
                    uint32_t combined[4]{};
                    for (size_t i = 0u; i < active; ++i) {
                        combined[0] += tasks[i].lanes[0];
                        combined[1] ^= tasks[i].lanes[1];
                        combined[2] += tasks[i].lanes[2];
                        combined[3] ^= tasks[i].lanes[3];
                    }
                    finalize_payload_checksum(combined, total, out_checksum);
                }
                return;
            }
            // Fallback single-threaded if threadpool creation failed partway.
#else
            std::vector<std::thread> threads;
            threads.reserve(active - 1u);
            for (size_t i = 1u; i < active; ++i) {
                threads.emplace_back([&, i]() { run_task(&tasks[i]); });
            }
            run_task(&tasks[0]);
            for (auto &th : threads) {
                th.join();
            }
            if (do_checksum) {
                uint32_t combined[4]{};
                for (size_t i = 0u; i < active; ++i) {
                    combined[0] += tasks[i].lanes[0];
                    combined[1] ^= tasks[i].lanes[1];
                    combined[2] += tasks[i].lanes[2];
                    combined[3] ^= tasks[i].lanes[3];
                }
                finalize_payload_checksum(combined, total, out_checksum);
            }
            return;
#endif
        }
    }

    if (do_copy && dst != nullptr && total > 0u) {
        std::memcpy(dst, src, total);
    }
    if (do_checksum) {
        uint32_t lanes[4]{};
        compute_payload_checksum_partial(payload, 0u, lanes);
        finalize_payload_checksum(lanes, total, out_checksum);
    }
}

}// namespace

void compute_payload_checksum(
    luisa::span<const std::byte> payload,
    uint8_t (&out_checksum)[16]) noexcept {
    run_parallel_copy_and_checksum(nullptr, payload, false, true, out_checksum);
}

AccelArchiveResult encode_envelope(
    AccelArchiveBackendKind backend,
    const AccelArchiveMeshDesc &desc,
    bool is_compacted,
    uint64_t unpacked_as_size,
    luisa::span<const std::byte> compat_blob,
    luisa::span<const std::byte> payload,
    luisa::vector<std::byte> &out_archive) noexcept {
    uint8_t payload_csum[16]{};
    if (!payload.empty()) {
        compute_payload_checksum(payload, payload_csum);
    }
    return encode_envelope_with_payload_checksum(
        backend,
        desc,
        is_compacted,
        unpacked_as_size,
        compat_blob,
        payload,
        payload_csum,
        out_archive);
}

AccelArchiveResult encode_envelope_with_payload_checksum(
    AccelArchiveBackendKind backend,
    const AccelArchiveMeshDesc &desc,
    bool is_compacted,
    uint64_t unpacked_as_size,
    luisa::span<const std::byte> compat_blob,
    luisa::span<const std::byte> payload,
    const uint8_t (&payload_checksum)[16],
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
    std::memcpy(hdr.payload_checksum, payload_checksum, 16u);
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

AccelArchiveResult decode_envelope_prefix(
    luisa::span<const std::byte> file_prefix_bytes,
    uint64_t total_file_size,
    AccelArchiveBackendKind expected_backend,
    const AccelArchiveMeshDesc &expected_desc,
    AccelArchiveMetadata &out_meta,
    uint64_t &out_payload_offset,
    luisa::span<const std::byte> &out_compat_blob,
    uint8_t (&out_expected_payload_checksum)[16]) noexcept {
    if (file_prefix_bytes.size() < sizeof(kEnvelopeMagic) || total_file_size < sizeof(kEnvelopeMagic)) {
        return make_archive_error(AccelArchiveStatus::TRUNCATED, "Archive truncated before envelope magic.");
    }
    if (std::memcmp(file_prefix_bytes.data(), kEnvelopeMagic, sizeof(kEnvelopeMagic)) != 0) {
        return make_archive_error(AccelArchiveStatus::INVALID_MAGIC, "Invalid acceleration-structure archive magic.");
    }
    if (file_prefix_bytes.size() < 12u || total_file_size < 12u) {
        return make_archive_error(AccelArchiveStatus::TRUNCATED, "Archive truncated before envelope version.");
    }
    uint32_t version = 0u;
    std::memcpy(&version, file_prefix_bytes.data() + 8u, sizeof(version));
    if (version != kEnvelopeVersion) {
        return make_archive_error(AccelArchiveStatus::UNSUPPORTED_VERSION, "Unsupported acceleration-structure archive envelope version.");
    }
    if (file_prefix_bytes.size() < sizeof(AccelArchiveEnvelopeHeader) ||
        total_file_size < sizeof(AccelArchiveEnvelopeHeader)) {
        return make_archive_error(AccelArchiveStatus::TRUNCATED, "Archive truncated inside envelope header.");
    }

    AccelArchiveEnvelopeHeader hdr{};
    std::memcpy(&hdr, file_prefix_bytes.data(), sizeof(AccelArchiveEnvelopeHeader));
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
    if (total_file_size < total_expected) {
        return make_archive_error(AccelArchiveStatus::TRUNCATED, "Archive truncated before end of compatibility/payload data.");
    }
    if (total_file_size > total_expected) {
        return make_archive_error(AccelArchiveStatus::CORRUPT_HEADER, "Archive contains unexpected trailing bytes.");
    }
    if (file_prefix_bytes.size() < compat_end) {
        return make_archive_error(AccelArchiveStatus::TRUNCATED, "Archive prefix truncated inside compatibility blob.");
    }

    const auto compat_span = file_prefix_bytes.subspan(hdr.header_size, hdr.compat_blob_size);
    if (!hash_equals(hdr.compat_checksum, hash_span_128(compat_span, kCompatChecksumSeed))) {
        return make_archive_error(AccelArchiveStatus::CORRUPT_PAYLOAD, "Compatibility blob checksum mismatch.");
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

    out_meta.envelope_version = hdr.version;
    out_meta.backend_kind = static_cast<AccelArchiveBackendKind>(hdr.backend_kind);
    out_meta.vertex_count = hdr.vertex_count;
    out_meta.triangle_count = hdr.triangle_count;
    out_meta.vertex_stride = hdr.vertex_stride;
    out_meta.option = {};
    out_meta.option.hint = static_cast<AccelOption::UsageHint>(hdr.option_hint);
    out_meta.option.allow_compaction = archive_allow_compaction;
    out_meta.option.allow_update = false;
    out_meta.is_compacted = (hdr.option_flags & kOptionIsCompacted) != 0u;
    out_meta.geometry_content_hash = to_hash128(hdr.geometry_content_hash);
    out_meta.archive_key = to_hash128(hdr.archive_key);
    out_meta.compat_blob_size = hdr.compat_blob_size;
    out_meta.payload_size = hdr.payload_size;
    out_meta.unpacked_as_size = hdr.unpacked_as_size;
    out_payload_offset = compat_end;
    out_compat_blob = compat_span;
    std::memcpy(out_expected_payload_checksum, hdr.payload_checksum, 16u);

    AccelArchiveResult ok{};
    ok.status = AccelArchiveStatus::OK;
    ok.readiness = AccelArchiveReadiness::NONE;
    ok.unpacked_as_size = hdr.unpacked_as_size;
    ok.serialized_payload_size = hdr.payload_size;
    ok.is_compacted = out_meta.is_compacted;
    return ok;
}

AccelArchiveResult decode_envelope_header_and_compat(
    luisa::span<const std::byte> archive_bytes,
    AccelArchiveBackendKind expected_backend,
    const AccelArchiveMeshDesc &expected_desc,
    AccelArchiveEnvelopeView &out_view,
    uint8_t (&out_expected_payload_checksum)[16]) noexcept {
    uint64_t payload_offset = 0u;
    auto res = decode_envelope_prefix(
        archive_bytes,
        static_cast<uint64_t>(archive_bytes.size()),
        expected_backend,
        expected_desc,
        out_view.meta,
        payload_offset,
        out_view.compat_blob,
        out_expected_payload_checksum);
    if (!res) {
        return res;
    }
    out_view.payload = archive_bytes.subspan(
        static_cast<size_t>(payload_offset),
        static_cast<size_t>(out_view.meta.payload_size));
    return res;
}

AccelArchiveResult verify_payload_checksum(
    luisa::span<const std::byte> payload,
    const uint8_t (&expected_checksum)[16]) noexcept {
    uint8_t actual[16]{};
    compute_payload_checksum(payload, actual);
    if (std::memcmp(actual, expected_checksum, 16u) != 0) {
        return make_archive_error(AccelArchiveStatus::CORRUPT_PAYLOAD, "Native payload checksum mismatch.");
    }
    return {};
}

bool copy_payload_optimized(
    void *dst,
    luisa::span<const std::byte> payload,
    bool verify_checksum,
    const uint8_t (&expected_checksum)[16]) noexcept {
    if (payload.empty()) {
        if (!verify_checksum) {
            return true;
        }
        uint8_t actual[16]{};
        compute_payload_checksum(payload, actual);
        return std::memcmp(actual, expected_checksum, 16u) == 0;
    }
    if (dst == nullptr) {
        return false;
    }
    uint8_t actual[16]{};
    run_parallel_copy_and_checksum(
        static_cast<std::byte *>(dst),
        payload,
        true,
        verify_checksum,
        actual);
    if (verify_checksum && std::memcmp(actual, expected_checksum, 16u) != 0) {
        return false;
    }
    return true;
}

AccelArchiveResult decode_and_validate_envelope(
    luisa::span<const std::byte> archive_bytes,
    AccelArchiveBackendKind expected_backend,
    const AccelArchiveMeshDesc &expected_desc,
    AccelArchiveEnvelopeView &out_view) noexcept {
    uint8_t expected_payload_checksum[16]{};
    auto res = decode_envelope_header_and_compat(
        archive_bytes,
        expected_backend,
        expected_desc,
        out_view,
        expected_payload_checksum);
    if (!res) {
        return res;
    }
    if (auto p = verify_payload_checksum(out_view.payload, expected_payload_checksum); !p) {
        return p;
    }
    return res;
}

}// namespace accel_archive

AccelArchiveStagingBuffer AccelArchiveExt::acquire_staging_buffer(size_t min_bytes) noexcept {
    constexpr size_t kAlign = 65536u;
    const size_t cap = std::max<size_t>(kAlign, ((min_bytes + kAlign - 1u) / kAlign) * kAlign);
#ifdef _WIN32
    auto *ptr = static_cast<std::byte *>(_aligned_malloc(cap, kAlign));
#else
    auto *ptr = static_cast<std::byte *>(std::aligned_alloc(kAlign, cap));
#endif
    if (ptr == nullptr) {
        return {};
    }
    AccelArchiveStagingBuffer buf{};
    buf.handle = reinterpret_cast<uint64_t>(ptr);
    buf.mapped_data = ptr;
    buf.capacity = cap;
    buf.alignment = kAlign;
    buf.is_device_local = false;
    buf.is_pinned_host = false;
    return buf;
}

void AccelArchiveExt::release_staging_buffer(AccelArchiveStagingBuffer &&buffer) noexcept {
    if (buffer.handle != invalid_resource_handle && buffer.handle != 0u) {
#ifdef _WIN32
        _aligned_free(reinterpret_cast<void *>(buffer.handle));
#else
        std::free(reinterpret_cast<void *>(buffer.handle));
#endif
    }
    buffer = {};
}

AccelArchiveResult AccelArchiveExt::import_mesh_archive_with_options(
    uint64_t stream_handle,
    uint64_t mesh_handle,
    const AccelArchiveMeshDesc &expected_desc,
    const AccelArchiveGeometryBinding &geometry_binding,
    luisa::span<const std::byte> archive_bytes,
    const AccelArchiveImportOptions &) noexcept {
    return import_mesh_archive(
        stream_handle, mesh_handle, expected_desc, geometry_binding, archive_bytes);
}

AccelArchiveResult AccelArchiveExt::import_mesh_archive_file(
    uint64_t stream_handle,
    uint64_t mesh_handle,
    const AccelArchiveMeshDesc &expected_desc,
    const AccelArchiveGeometryBinding &geometry_binding,
    luisa::string_view file_path,
    const AccelArchiveImportOptions &options) noexcept {
    if (file_path.empty()) {
        return accel_archive::make_archive_error(
            AccelArchiveStatus::INVALID_ARGUMENT,
            "Archive file path must not be empty.");
    }

#ifdef _WIN32
    const std::string path_str(file_path);
    const int wlen = MultiByteToWideChar(
        CP_UTF8, 0u, path_str.c_str(), static_cast<int>(path_str.size()), nullptr, 0);
    std::vector<wchar_t> wpath(static_cast<size_t>(wlen) + 1u, L'\0');
    if (wlen > 0) {
        MultiByteToWideChar(
            CP_UTF8, 0u, path_str.c_str(), static_cast<int>(path_str.size()), wpath.data(), wlen);
    }

    bool unbuffered = options.bypass_os_file_cache;
    HANDLE h = INVALID_HANDLE_VALUE;
    if (unbuffered) {
        h = CreateFileW(
            wpath.data(),
            GENERIC_READ,
            FILE_SHARE_READ,
            nullptr,
            OPEN_EXISTING,
            FILE_FLAG_NO_BUFFERING | FILE_FLAG_SEQUENTIAL_SCAN,
            nullptr);
        if (h == INVALID_HANDLE_VALUE) {
            unbuffered = false;
        }
    }
    if (h == INVALID_HANDLE_VALUE) {
        h = CreateFileW(
            wpath.data(),
            GENERIC_READ,
            FILE_SHARE_READ,
            nullptr,
            OPEN_EXISTING,
            FILE_FLAG_SEQUENTIAL_SCAN,
            nullptr);
    }
    if (h == INVALID_HANDLE_VALUE) {
        return accel_archive::make_archive_error(
            AccelArchiveStatus::BACKEND_ERROR,
            "Failed to open acceleration-structure archive file for reading.");
    }
    LARGE_INTEGER li{};
    if (!GetFileSizeEx(h, &li) || li.QuadPart < 0) {
        CloseHandle(h);
        return accel_archive::make_archive_error(
            AccelArchiveStatus::BACKEND_ERROR,
            "Failed to query acceleration-structure archive file size.");
    }
    const size_t file_size = static_cast<size_t>(li.QuadPart);
    constexpr size_t kSector = 65536u;
    const size_t padded_size = std::max<size_t>(kSector, ((file_size + kSector - 1u) / kSector) * kSector);
    auto staging = acquire_staging_buffer(padded_size);
    if (staging.mapped_data == nullptr || staging.capacity < padded_size) {
        release_staging_buffer(std::move(staging));
        CloseHandle(h);
        return accel_archive::make_archive_error(
            AccelArchiveStatus::BACKEND_ERROR,
            "Failed to allocate staging buffer for archive file read.");
    }
    if (unbuffered && (reinterpret_cast<uintptr_t>(staging.mapped_data) & 4095ull) != 0ull) {
        CloseHandle(h);
        h = CreateFileW(
            wpath.data(),
            GENERIC_READ,
            FILE_SHARE_READ,
            nullptr,
            OPEN_EXISTING,
            FILE_FLAG_SEQUENTIAL_SCAN,
            nullptr);
        if (h == INVALID_HANDLE_VALUE) {
            release_staging_buffer(std::move(staging));
            return accel_archive::make_archive_error(
                AccelArchiveStatus::BACKEND_ERROR,
                "Failed to reopen archive file for buffered read.");
        }
        unbuffered = false;
    }

    size_t total = 0u;
    bool read_ok = true;
    while (total < file_size) {
        const size_t rem = file_size - total;
        const size_t max_chunk = 32ull * 1024ull * 1024ull;
        const DWORD req = unbuffered ?
                              static_cast<DWORD>(std::min<size_t>(max_chunk, ((rem + kSector - 1u) / kSector) * kSector)) :
                              static_cast<DWORD>(std::min<size_t>(max_chunk, rem));
        DWORD got = 0u;
        if (!ReadFile(h, staging.mapped_data + total, req, &got, nullptr)) {
            if (unbuffered) {
                // Retry sequentially without NO_BUFFERING if FS does not support unbuffered IO.
                CloseHandle(h);
                h = CreateFileW(
                    wpath.data(),
                    GENERIC_READ,
                    FILE_SHARE_READ,
                    nullptr,
                    OPEN_EXISTING,
                    FILE_FLAG_SEQUENTIAL_SCAN,
                    nullptr);
                if (h == INVALID_HANDLE_VALUE) {
                    read_ok = false;
                    break;
                }
                LARGE_INTEGER zero{};
                SetFilePointerEx(h, zero, nullptr, FILE_BEGIN);
                total = 0u;
                unbuffered = false;
                continue;
            }
            read_ok = false;
            break;
        }
        if (got == 0u) {
            break;
        }
        total += got;
    }
    CloseHandle(h);
    if (!read_ok || total != file_size) {
        release_staging_buffer(std::move(staging));
        return accel_archive::make_archive_error(
            AccelArchiveStatus::BACKEND_ERROR,
            "Failed reading acceleration-structure archive file.");
    }
    auto res = import_mesh_archive_with_options(
        stream_handle,
        mesh_handle,
        expected_desc,
        geometry_binding,
        luisa::span<const std::byte>{staging.mapped_data, file_size},
        options);
    release_staging_buffer(std::move(staging));
    return res;
#else
    const std::string path_str(file_path);
    std::ifstream ifs(path_str, std::ios::binary | std::ios::ate);
    if (!ifs.is_open()) {
        return accel_archive::make_archive_error(
            AccelArchiveStatus::BACKEND_ERROR,
            "Failed to open acceleration-structure archive file for reading.");
    }
    const auto sz = ifs.tellg();
    if (sz < 0) {
        return accel_archive::make_archive_error(
            AccelArchiveStatus::BACKEND_ERROR,
            "Failed querying archive file size.");
    }
    const size_t file_size = static_cast<size_t>(sz);
    auto staging = acquire_staging_buffer(std::max<size_t>(file_size, 64u));
    if (staging.mapped_data == nullptr || staging.capacity < file_size) {
        release_staging_buffer(std::move(staging));
        return accel_archive::make_archive_error(
            AccelArchiveStatus::BACKEND_ERROR,
            "Failed allocating staging buffer for archive file read.");
    }
    ifs.seekg(0, std::ios::beg);
    if (file_size > 0u && !ifs.read(reinterpret_cast<char *>(staging.mapped_data), static_cast<std::streamsize>(file_size))) {
        release_staging_buffer(std::move(staging));
        return accel_archive::make_archive_error(
            AccelArchiveStatus::BACKEND_ERROR,
            "Failed reading archive file.");
    }
    auto res = import_mesh_archive_with_options(
        stream_handle,
        mesh_handle,
        expected_desc,
        geometry_binding,
        luisa::span<const std::byte>{staging.mapped_data, file_size},
        options);
    release_staging_buffer(std::move(staging));
    return res;
#endif
}

}// namespace luisa::compute

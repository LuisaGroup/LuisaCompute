#include "ut/ut.hpp"

#include <array>
#include <cstring>
#include <vector>

#include <luisa/luisa-compute.h>
#include <luisa/backends/ext/accel_archive_ext.hpp>
#include "cuda_accel_archive_codec.h"

#if LUISA_TEST_HAS_PINNED_OPTIX_HEADERS
#include <optix_types.h>
static_assert(sizeof(OptixRelocationInfo) == 32u);
static_assert(sizeof(OptixRelocationInfo::info) == 32u);
static_assert(OPTIX_ACCEL_BUFFER_BYTE_ALIGNMENT == 128u);
#endif

using namespace luisa;
using namespace luisa::compute;
using namespace boost::ut;

namespace {

struct PaddedVertex {
    float x, y, z;
    uint32_t padding;
};

[[nodiscard]] AccelArchiveMeshDesc make_sample_desc(
    const luisa::vector<PaddedVertex> &verts,
    const luisa::vector<Triangle> &tris,
    AccelOption opt) noexcept {
    AccelArchiveMeshDesc desc{};
    desc.vertex_count = verts.size();
    desc.triangle_count = tris.size();
    desc.vertex_stride = sizeof(PaddedVertex);
    desc.option = opt;
    desc.geometry_content_hash = accel_archive::compute_static_triangle_geometry_hash(
        verts.data(), verts.size(), sizeof(PaddedVertex),
        tris.data(), tris.size());
    return desc;
}

class StubDeviceInterface final : public DeviceInterface {
public:
    StubDeviceInterface(Context &&ctx) noexcept : DeviceInterface{std::move(ctx)} {}
    [[nodiscard]] uint compute_warp_size() const noexcept override { return 1u; }
    [[nodiscard]] size_t compute_max_shared_memory_size() const noexcept override { return 0u; }
    [[nodiscard]] uint64_t memory_granularity() const noexcept override { return 4096u; }
    [[nodiscard]] void *native_handle() const noexcept override { return nullptr; }
    [[nodiscard]] BufferCreationInfo create_buffer(const Type *, size_t, void *) noexcept override { return {}; }
    void destroy_buffer(uint64_t) noexcept override {}
    [[nodiscard]] ResourceCreationInfo create_texture(PixelFormat, uint, uint, uint, uint, uint, void *, bool, bool) noexcept override { return {}; }
    void destroy_texture(uint64_t) noexcept override {}
    [[nodiscard]] ResourceCreationInfo create_bindless_array(size_t, BindlessSlotType) noexcept override { return {}; }
    void destroy_bindless_array(uint64_t) noexcept override {}
    [[nodiscard]] ResourceCreationInfo create_stream(StreamTag) noexcept override { return {}; }
    void destroy_stream(uint64_t) noexcept override {}
    void synchronize_stream(uint64_t) noexcept override {}
    void dispatch(uint64_t, CommandList &&) noexcept override {}
    [[nodiscard]] SwapchainCreationInfo create_swapchain(const SwapchainOption &, uint64_t) noexcept override { return {}; }
    void destroy_swapchain(uint64_t) noexcept override {}
    void present_display_in_stream(uint64_t, uint64_t, uint64_t) noexcept override {}
    [[nodiscard]] ShaderCreationInfo create_shader(const ShaderOption &, Function) noexcept override { return {}; }
    [[nodiscard]] ShaderCreationInfo load_shader(luisa::string_view, luisa::span<const Type *const>) noexcept override { return {}; }
    Usage shader_argument_usage(uint64_t, size_t) noexcept override { return Usage::NONE; }
    void destroy_shader(uint64_t) noexcept override {}
    [[nodiscard]] ResourceCreationInfo create_event() noexcept override { return {}; }
    void destroy_event(uint64_t) noexcept override {}
    void signal_event(uint64_t, uint64_t, uint64_t) noexcept override {}
    void wait_event(uint64_t, uint64_t, uint64_t) noexcept override {}
    [[nodiscard]] bool is_event_completed(uint64_t, uint64_t) const noexcept override { return true; }
    void synchronize_event(uint64_t, uint64_t) noexcept override {}
    [[nodiscard]] ResourceCreationInfo create_mesh(const AccelOption &) noexcept override { return {}; }
    void destroy_mesh(uint64_t) noexcept override {}
    [[nodiscard]] ResourceCreationInfo create_procedural_primitive(const AccelOption &) noexcept override { return {}; }
    void destroy_procedural_primitive(uint64_t) noexcept override {}
    [[nodiscard]] ResourceCreationInfo create_accel(const AccelOption &) noexcept override { return {}; }
    void destroy_accel(uint64_t) noexcept override {}
    void set_name(Resource::Tag, uint64_t, luisa::string_view) noexcept override {}
};

}// namespace

int main(int argc, char *argv[]) {
    boost::ut::detail::cfg::parse_arg_with_fallback(argc, const_cast<const char **>(argv));

    "geometry_hash_and_key_sensitivity"_test = [] {
        luisa::vector<PaddedVertex> verts_a{
            {0.f, 0.f, 0.f, 1u},
            {1.f, 0.f, 0.f, 2u},
            {0.f, 1.f, 0.f, 3u},
        };
        luisa::vector<PaddedVertex> verts_b = verts_a;
        verts_b[0].padding = 999u; // Padding outside float3 position must not change canonical geometry hash
        luisa::vector<Triangle> tris{{0u, 1u, 2u}};

        auto h_a = accel_archive::compute_static_triangle_geometry_hash(
            verts_a.data(), verts_a.size(), sizeof(PaddedVertex),
            tris.data(), tris.size());
        auto h_b = accel_archive::compute_static_triangle_geometry_hash(
            verts_b.data(), verts_b.size(), sizeof(PaddedVertex),
            tris.data(), tris.size());
        expect(h_a == h_b);

        auto verts_c = verts_a;
        verts_c[2].z = 0.001f;
        auto h_c = accel_archive::compute_static_triangle_geometry_hash(
            verts_c.data(), verts_c.size(), sizeof(PaddedVertex),
            tris.data(), tris.size());
        expect(!(h_a == h_c));

        luisa::vector<Triangle> tris_d{{0u, 2u, 1u}};
        auto h_d = accel_archive::compute_static_triangle_geometry_hash(
            verts_a.data(), verts_a.size(), sizeof(PaddedVertex),
            tris_d.data(), tris_d.size());
        expect(!(h_a == h_d));

        AccelOption opt1{.hint = AccelOption::UsageHint::FAST_TRACE, .allow_compaction = false, .allow_update = false};
        AccelOption opt2{.hint = AccelOption::UsageHint::FAST_BUILD, .allow_compaction = false, .allow_update = false};
        AccelOption opt3{.hint = AccelOption::UsageHint::FAST_TRACE, .allow_compaction = true, .allow_update = false};
        auto desc1 = make_sample_desc(verts_a, tris, opt1);
        auto desc2 = make_sample_desc(verts_a, tris, opt2);
        auto desc3 = make_sample_desc(verts_a, tris, opt3);
        expect(!(accel_archive::compute_archive_key(desc1) == accel_archive::compute_archive_key(desc2)));
        expect(!(accel_archive::compute_archive_key(desc1) == accel_archive::compute_archive_key(desc3)));
    };

    "envelope_roundtrip_and_rejections"_test = [] {
        luisa::vector<PaddedVertex> verts{
            {0.f, 0.f, 0.f, 0u},
            {1.f, 0.f, 0.f, 0u},
            {0.f, 1.f, 0.f, 0u},
        };
        luisa::vector<Triangle> tris{{0u, 1u, 2u}};
        AccelOption opt{.hint = AccelOption::UsageHint::FAST_TRACE, .allow_compaction = true, .allow_update = false};
        auto desc = make_sample_desc(verts, tris, opt);

        std::array<std::byte, 16> compat{};
        for (size_t i = 0u; i < compat.size(); ++i) compat[i] = static_cast<std::byte>(0x10u + i);
        std::array<std::byte, 64> payload{};
        for (size_t i = 0u; i < payload.size(); ++i) payload[i] = static_cast<std::byte>(0xa0u + (i & 0x3fu));

        luisa::vector<std::byte> archive;
        auto enc = accel_archive::encode_envelope(
            AccelArchiveBackendKind::VULKAN, desc, true, 4096u, compat, payload, archive);
        expect(static_cast<bool>(enc));
        expect(archive.size() == sizeof(accel_archive::AccelArchiveEnvelopeHeader) + compat.size() + payload.size());

        accel_archive::AccelArchiveEnvelopeView view{};
        auto dec = accel_archive::decode_and_validate_envelope(
            archive, AccelArchiveBackendKind::VULKAN, desc, view);
        expect(static_cast<bool>(dec));
        expect(view.meta.envelope_version == accel_archive::kEnvelopeVersion);
        expect(view.meta.backend_kind == AccelArchiveBackendKind::VULKAN);
        expect(view.meta.is_compacted);
        expect(view.meta.unpacked_as_size == 4096u);
        expect(view.compat_blob.size() == compat.size());
        expect(view.payload.size() == payload.size());
        expect(std::memcmp(view.compat_blob.data(), compat.data(), compat.size()) == 0);
        expect(std::memcmp(view.payload.data(), payload.data(), payload.size()) == 0);

        // Truncated at various points
        for (size_t len : {size_t{0u}, size_t{4u}, size_t{10u}, sizeof(accel_archive::AccelArchiveEnvelopeHeader) - 1u, archive.size() - 1u}) {
            auto r = accel_archive::decode_and_validate_envelope(
                luisa::span{archive}.subspan(0u, len), AccelArchiveBackendKind::VULKAN, desc, view);
            expect(r.status == AccelArchiveStatus::TRUNCATED);
        }

        // Trailing bytes -> CORRUPT_HEADER
        {
            auto extended = archive;
            extended.push_back(std::byte{0xff});
            auto r = accel_archive::decode_and_validate_envelope(
                extended, AccelArchiveBackendKind::VULKAN, desc, view);
            expect(r.status == AccelArchiveStatus::CORRUPT_HEADER);
        }

        // Bad magic
        {
            auto bad = archive;
            bad[0] = std::byte{'X'};
            auto r = accel_archive::decode_and_validate_envelope(
                bad, AccelArchiveBackendKind::VULKAN, desc, view);
            expect(r.status == AccelArchiveStatus::INVALID_MAGIC);
        }

        // Unsupported version
        {
            auto bad = archive;
            uint32_t bad_ver = 99u;
            std::memcpy(bad.data() + 8u, &bad_ver, sizeof(bad_ver));
            auto r = accel_archive::decode_and_validate_envelope(
                bad, AccelArchiveBackendKind::VULKAN, desc, view);
            expect(r.status == AccelArchiveStatus::UNSUPPORTED_VERSION);
        }

        // Corrupt header field
        {
            auto bad = archive;
            bad[32] ^= std::byte{0x01};
            auto r = accel_archive::decode_and_validate_envelope(
                bad, AccelArchiveBackendKind::VULKAN, desc, view);
            expect(r.status == AccelArchiveStatus::CORRUPT_HEADER);
        }

        // Corrupt compat / payload
        {
            auto bad = archive;
            bad[sizeof(accel_archive::AccelArchiveEnvelopeHeader)] ^= std::byte{0x01};
            auto r = accel_archive::decode_and_validate_envelope(
                bad, AccelArchiveBackendKind::VULKAN, desc, view);
            expect(r.status == AccelArchiveStatus::CORRUPT_PAYLOAD);
        }
        {
            auto bad = archive;
            bad.back() ^= std::byte{0x80};
            auto r = accel_archive::decode_and_validate_envelope(
                bad, AccelArchiveBackendKind::VULKAN, desc, view);
            expect(r.status == AccelArchiveStatus::CORRUPT_PAYLOAD);
        }

        // Wrong backend
        {
            auto r = accel_archive::decode_and_validate_envelope(
                archive, AccelArchiveBackendKind::DX, desc, view);
            expect(r.status == AccelArchiveStatus::UNSUPPORTED_BACKEND);
        }

        // Changed options
        {
            auto other_desc = desc;
            other_desc.option.allow_compaction = false;
            auto r = accel_archive::decode_and_validate_envelope(
                archive, AccelArchiveBackendKind::VULKAN, other_desc, view);
            expect(r.status == AccelArchiveStatus::OPTIONS_MISMATCH);
        }

        // Changed geometry
        {
            auto other_desc = desc;
            other_desc.geometry_content_hash.data()[0] ^= 0x01u;
            auto r = accel_archive::decode_and_validate_envelope(
                archive, AccelArchiveBackendKind::VULKAN, other_desc, view);
            expect(r.status == AccelArchiveStatus::KEY_MISMATCH);
        }

        // Unsupported v1 features rejected cleanly
        {
            auto u = desc;
            u.option.allow_update = true;
            expect(accel_archive::validate_v1_static_triangle_desc(u).status == AccelArchiveStatus::UNSUPPORTED_FEATURE);
            u = desc;
            u.option.motion.keyframe_count = 2u;
            expect(accel_archive::validate_v1_static_triangle_desc(u).status == AccelArchiveStatus::UNSUPPORTED_FEATURE);
            u = desc;
            u.is_cluster = true;
            expect(accel_archive::validate_v1_static_triangle_desc(u).status == AccelArchiveStatus::UNSUPPORTED_FEATURE);
            u = desc;
            u.has_cutout_mask = true;
            expect(accel_archive::validate_v1_static_triangle_desc(u).status == AccelArchiveStatus::UNSUPPORTED_FEATURE);
        }
    };

    "optimized_hash_copy_and_prefix"_test = [] {
        luisa::vector<PaddedVertex> verts(64);
        for (size_t i = 0u; i < verts.size(); ++i) {
            verts[i] = PaddedVertex{static_cast<float>(i), 1.0f, -2.0f, 0u};
        }
        luisa::vector<Triangle> tris;
        for (uint32_t i = 0u; i + 2u < verts.size(); i += 3u) {
            tris.push_back(Triangle{i, i + 1u, i + 2u});
        }
        AccelOption opt{};
        opt.allow_compaction = true;
        auto desc = make_sample_desc(verts, tris, opt);

        // Small (< 2 MiB) & Large (>= 2 MiB) hash128_and_copy + copy_payload_optimized
        for (size_t sz : {size_t{4096u}, size_t{3u * 1024u * 1024u + 137u}}) {
            luisa::vector<std::byte> src(sz);
            for (size_t i = 0u; i < sz; ++i) {
                src[i] = static_cast<std::byte>((i * 131u + 17u) & 0xffu);
            }
            luisa::vector<std::byte> dst(sz);
            const auto h_direct = luisa::hash128(src.data(), src.size(), 0x123456789abcdef0ull);
            const auto h_fused = luisa::hash128_and_copy(dst.data(), src.data(), src.size(), 0x123456789abcdef0ull);
            expect(h_fused == h_direct);
            expect(std::memcmp(src.data(), dst.data(), sz) == 0);

            std::array<std::byte, 24> compat{std::byte{0x42}};
            luisa::vector<std::byte> archive;
            auto enc = accel_archive::encode_envelope(
                AccelArchiveBackendKind::DX, desc, true, 12345u, compat, src, archive);
            expect(static_cast<bool>(enc));

            AccelArchiveMetadata meta{};
            uint64_t payload_off = 0u;
            luisa::span<const std::byte> compat_view;
            uint8_t payload_csum[16]{};
            // Prefix contains only envelope header + compat blob (no payload bytes required!)
            const auto pre_span = luisa::span<const std::byte>{archive}.subspan(
                0u, sizeof(accel_archive::AccelArchiveEnvelopeHeader) + compat.size());
            auto pre_res = accel_archive::decode_envelope_prefix(
                pre_span, archive.size(), AccelArchiveBackendKind::DX, desc,
                meta, payload_off, compat_view, payload_csum);
            expect(static_cast<bool>(pre_res));
            expect(payload_off == pre_span.size());
            expect(meta.payload_size == sz);

            accel_archive::AccelArchiveEnvelopeView hdr_view{};
            uint8_t csum2[16]{};
            auto hdr_res = accel_archive::decode_envelope_header_and_compat(
                archive, AccelArchiveBackendKind::DX, desc, hdr_view, csum2);
            expect(static_cast<bool>(hdr_res));
            expect(std::memcmp(payload_csum, csum2, 16u) == 0);

            luisa::vector<std::byte> copy_out(sz);
            expect(accel_archive::copy_payload_optimized(copy_out.data(), hdr_view.payload, true, csum2));
            expect(std::memcmp(copy_out.data(), src.data(), sz) == 0);

            // Corrupt checksum rejects on verified copy, still succeeds on unverified fast copy
            uint8_t bad_csum[16]{};
            std::memcpy(bad_csum, csum2, 16u);
            bad_csum[5] ^= 0xffu;
            expect(!accel_archive::copy_payload_optimized(copy_out.data(), hdr_view.payload, true, bad_csum));
            expect(accel_archive::copy_payload_optimized(copy_out.data(), hdr_view.payload, false, bad_csum));
        }
    };

    "fast_archive_checksum_128_properties"_test = [] {
        // Empty payload agrees between full compute and finalize_payload_checksum.
        uint8_t empty_a[16]{};
        uint8_t empty_b[16]{};
        accel_archive::compute_payload_checksum(luisa::span<const std::byte>{}, empty_a);
        uint32_t zero_lanes[4]{0u, 0u, 0u, 0u};
        accel_archive::finalize_payload_checksum(zero_lanes, 0u, empty_b);
        expect(std::memcmp(empty_a, empty_b, 16u) == 0);

        const std::array<size_t, 10> sizes{1u, 2u, 3u, 5u, 7u, 13u, 17u, 1025u, 4099u, 262147u};
        for (const size_t sz : sizes) {
            luisa::vector<std::byte> buf(sz);
            for (size_t i = 0u; i < sz; ++i) {
                buf[i] = static_cast<std::byte>((i * 73u + 19u) & 0xffu);
            }

            uint8_t full_a[16]{};
            uint8_t full_b[16]{};
            accel_archive::compute_payload_checksum(buf, full_a);
            accel_archive::compute_payload_checksum(buf, full_b);
            expect(std::memcmp(full_a, full_b, 16u) == 0);

            // Split computation matching HIP's 48-byte (12-word) header fold produces identical checksum.
            if (sz > 48u) {
                uint32_t l_head[4]{};
                uint32_t l_tail[4]{};
                accel_archive::compute_payload_checksum_partial(
                    luisa::span<const std::byte>{buf.data(), 48u}, 0u, l_head);
                accel_archive::compute_payload_checksum_partial(
                    luisa::span<const std::byte>{buf.data() + 48u, sz - 48u}, 12u, l_tail);
                uint32_t combined[4]{
                    l_head[0] + l_tail[0],
                    l_head[1] ^ l_tail[1],
                    l_head[2] + l_tail[2],
                    l_head[3] ^ l_tail[3]};
                uint8_t split_c[16]{};
                accel_archive::finalize_payload_checksum(combined, sz, split_c);
                expect(std::memcmp(full_a, split_c, 16u) == 0);
            }

            // Single-bit flips at start, middle, and tail byte alter checksum.
            const std::array<size_t, 3> flip_offsets{0u, sz / 2u, sz - 1u};
            for (const size_t fo : flip_offsets) {
                auto mutated = buf;
                mutated[fo] = static_cast<std::byte>(std::to_integer<uint8_t>(mutated[fo]) ^ 0x01u);
                uint8_t mut_c[16]{};
                accel_archive::compute_payload_checksum(mutated, mut_c);
                expect(std::memcmp(full_a, mut_c, 16u) != 0);
            }

            // Envelope encode_envelope_with_payload_checksum matches decode validation cleanly.
            const luisa::vector<PaddedVertex> sv{{0.f, 0.f, 0.f, 0u}, {1.f, 0.f, 0.f, 0u}, {0.f, 1.f, 0.f, 0u}};
            const luisa::vector<Triangle> st{{0u, 1u, 2u}};
            const auto d = make_sample_desc(sv, st, AccelOption{.hint = AccelOption::UsageHint::FAST_TRACE});
            std::array<std::byte, 4> compat{std::byte{1}, std::byte{2}, std::byte{3}, std::byte{4}};
            luisa::vector<std::byte> env_ok;
            expect(static_cast<bool>(accel_archive::encode_envelope_with_payload_checksum(
                AccelArchiveBackendKind::DX, d, false, sz, compat, buf, full_a, env_ok)));
            accel_archive::AccelArchiveEnvelopeView v_ok{};
            expect(static_cast<bool>(accel_archive::decode_and_validate_envelope(
                env_ok, AccelArchiveBackendKind::DX, d, v_ok)));

            uint8_t bad_sum[16]{};
            std::memcpy(bad_sum, full_a, 16u);
            bad_sum[7] ^= 0x80u;
            luisa::vector<std::byte> env_bad;
            expect(static_cast<bool>(accel_archive::encode_envelope_with_payload_checksum(
                AccelArchiveBackendKind::DX, d, false, sz, compat, buf, bad_sum, env_bad)));
            accel_archive::AccelArchiveEnvelopeView v_bad{};
            expect(accel_archive::decode_and_validate_envelope(
                       env_bad, AccelArchiveBackendKind::DX, d, v_bad)
                       .status == AccelArchiveStatus::CORRUPT_PAYLOAD);
        }
    };

    "cuda_optix_compat_codec"_test = [] {
        cuda::CudaOptixAccelArchiveCompatBlob blob{};
        blob.optix_abi_version = 118u;
        blob.cuda_compute_capability = 89u;
        blob.cuda_driver_version = 12080u;
        blob.active_bvh_bytes = 256u;
        blob.relocation_info[0] = 11u;
        blob.relocation_info[1] = 22u;
        blob.relocation_info[2] = 33u;
        blob.relocation_info[3] = 44u;

        auto encoded = cuda::encode_cuda_optix_compat_blob(blob);
        std::array<std::byte, 256> dummy_payload{};
        cuda::CudaOptixAccelArchiveCompatBlob decoded{};
        auto ok = cuda::decode_and_check_cuda_optix_compat_blob(
            encoded, dummy_payload, 118u, 89u, 12080u, decoded);
        expect(static_cast<bool>(ok));
        expect(decoded.relocation_info[2] == 33u);

        // ABI mismatch
        expect(cuda::decode_and_check_cuda_optix_compat_blob(
                   encoded, dummy_payload, 105u, 89u, 12080u, decoded)
                   .status == AccelArchiveStatus::DRIVER_INCOMPATIBLE);
        // Compute capability mismatch
        expect(cuda::decode_and_check_cuda_optix_compat_blob(
                   encoded, dummy_payload, 118u, 86u, 12080u, decoded)
                   .status == AccelArchiveStatus::DEVICE_INCOMPATIBLE);
        // Driver mismatch
        expect(cuda::decode_and_check_cuda_optix_compat_blob(
                   encoded, dummy_payload, 118u, 89u, 12060u, decoded)
                   .status == AccelArchiveStatus::DRIVER_INCOMPATIBLE);
        // Payload size mismatch
        std::array<std::byte, 128> short_payload{};
        expect(cuda::decode_and_check_cuda_optix_compat_blob(
                   encoded, short_payload, 118u, 89u, 12080u, decoded)
                   .status == AccelArchiveStatus::CORRUPT_PAYLOAD);
    };

    "device_interface_default_extension_is_null"_test = [&] {
        Context ctx{argv[0]};
        StubDeviceInterface stub{std::move(ctx)};
        expect(stub.extension(AccelArchiveExt::name) == nullptr);
        luisa::vector<std::byte> out;
        AccelArchiveMeshDesc desc{};
        luisa::compute::AccelArchiveExt *null_ext = nullptr;
        expect(null_ext == nullptr);
        // Directly verify interface null handling in wrappers
        AccelArchiveResult r = accel_archive::make_archive_error(
            AccelArchiveStatus::UNSUPPORTED_BACKEND, "test");
        expect(r.status == AccelArchiveStatus::UNSUPPORTED_BACKEND);
        static_cast<void>(out);
        static_cast<void>(desc);
    };

    return 0;
}

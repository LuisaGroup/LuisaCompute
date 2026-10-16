// Test for serialized Metal raster AIR archives.
// This test covers:
// - Mesh-format, fragment-output ABI, argument ABI, and metallib round trips
// - Truncation, trailing data, and header corruption rejection
// - Invalid mesh, usage/stage metadata, and empty-library rejection

#include "ut/ut.hpp"

#include "metal_raster_archive.h"

#include <luisa/core/stl/hash.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>

using namespace luisa;
using namespace luisa::compute;
using namespace luisa::compute::metal;
using namespace boost::ut;
using namespace boost::ut::literals;

namespace {

[[nodiscard]] MetalRasterArchive make_archive() noexcept {
    MetalRasterArchive archive{};
    constexpr std::array stream_zero{
        VertexAttribute{VertexAttributeType::Position, PixelFormat::RGBA32F},
        VertexAttribute{VertexAttributeType::Normal, PixelFormat::RGBA16F}};
    constexpr std::array stream_one{
        VertexAttribute{VertexAttributeType::UV0, PixelFormat::RG32F}};
    archive.mesh_format.emplace_vertex_stream(stream_zero);
    archive.mesh_format.emplace_vertex_stream(stream_one);
    archive.arguments.emplace_back(MetalRasterArchiveArgument{
        .type = "buffer<float>",
        .usage = Usage::READ,
        .stage = MetalRasterArchiveStage::VERTEX});
    archive.arguments.emplace_back(MetalRasterArchiveArgument{
        .type = "float4",
        .usage = Usage::NONE,
        .stage = MetalRasterArchiveStage::FRAGMENT});
    archive.library = {
        std::byte{'M'}, std::byte{'T'}, std::byte{'L'}, std::byte{'B'},
        std::byte{0x01u}, std::byte{0x02u}, std::byte{0x03u}};
    archive.root_argument_size = 32u;
    archive.fragment_output_count = 2u;
    return archive;
}

template<typename T>
void write_pod(span<std::byte> data, size_t offset, T value) noexcept {
    expect(offset <= data.size() && sizeof(T) <= data.size() - offset);
    std::memcpy(data.data() + offset, &value, sizeof(T));
}

void refresh_checksum(luisa::vector<std::byte> &data) noexcept {
    expect(data.size() >= sizeof(uint64_t));
    auto payload_size = data.size() - sizeof(uint64_t);
    auto checksum = luisa::hash64(
        data.data(), payload_size, metal_raster_archive_checksum_seed);
    write_pod<uint64_t>(data, payload_size, checksum);
}

}// namespace

int main(int argc, char *argv[]) {
    boost::ut::detail::cfg::parse_arg_with_fallback(
        argc, const_cast<const char **>(argv));

    "metal_raster_archive_round_trip"_test = [] {
        auto source = make_archive();
        auto data = serialize_metal_raster_archive(source);
        auto archive = deserialize_metal_raster_archive(data);
        expect(archive.has_value());
        if (!archive) { return; }

        expect(eq(archive->mesh_format.vertex_stream_count(), 2u));
        expect(eq(archive->mesh_format.vertex_attribute_count(), 3u));
        auto stream_zero = archive->mesh_format.attributes(0u);
        expect(eq(stream_zero.size(), 2u));
        expect(stream_zero[0u].type == VertexAttributeType::Position);
        expect(stream_zero[0u].format == PixelFormat::RGBA32F);
        expect(stream_zero[1u].type == VertexAttributeType::Normal);
        expect(stream_zero[1u].format == PixelFormat::RGBA16F);
        auto stream_one = archive->mesh_format.attributes(1u);
        expect(eq(stream_one.size(), 1u));
        expect(stream_one[0u].type == VertexAttributeType::UV0);
        expect(stream_one[0u].format == PixelFormat::RG32F);

        expect(eq(archive->arguments.size(), 2u));
        expect(archive->arguments[0u].type == "buffer<float>");
        expect(archive->arguments[0u].usage == Usage::READ);
        expect(archive->arguments[0u].stage ==
               MetalRasterArchiveStage::VERTEX);
        expect(archive->arguments[1u].type == "float4");
        expect(archive->arguments[1u].usage == Usage::NONE);
        expect(archive->arguments[1u].stage ==
               MetalRasterArchiveStage::FRAGMENT);
        expect(eq(archive->root_argument_size, 32u));
        expect(eq(archive->fragment_output_count, 2u));
        expect(static_cast<bool>(archive->library == source.library));
    };

    "metal_raster_archive_depth_only_round_trip"_test = [] {
        auto source = make_archive();
        source.fragment_output_count = 0u;
        auto archive = deserialize_metal_raster_archive(
            serialize_metal_raster_archive(source));
        expect(archive.has_value());
        if (!archive) { return; }
        expect(eq(archive->fragment_output_count, 0u));
        expect(static_cast<bool>(archive->library == source.library));
    };

    "metal_raster_archive_rejects_truncation"_test = [] {
        auto data = serialize_metal_raster_archive(make_archive());
        auto all_prefixes_rejected = true;
        for (auto size = 0u; size < data.size(); size++) {
            if (deserialize_metal_raster_archive(
                    span<const std::byte>{data}.first(size))) {
                all_prefixes_rejected = false;
                break;
            }
        }
        expect(all_prefixes_rejected);
    };

    "metal_raster_archive_rejects_header_corruption"_test = [] {
        auto invalid_magic = serialize_metal_raster_archive(make_archive());
        invalid_magic.front() ^= std::byte{0xffu};
        refresh_checksum(invalid_magic);
        expect(!deserialize_metal_raster_archive(invalid_magic));

        auto invalid_version = serialize_metal_raster_archive(make_archive());
        write_pod<uint32_t>(invalid_version, 8u, 4u);
        refresh_checksum(invalid_version);
        expect(!deserialize_metal_raster_archive(invalid_version));

        auto invalid_root_size = serialize_metal_raster_archive(make_archive());
        write_pod<uint64_t>(invalid_root_size, 24u, 17u);
        refresh_checksum(invalid_root_size);
        expect(!deserialize_metal_raster_archive(invalid_root_size));

        auto too_many_streams = serialize_metal_raster_archive(make_archive());
        write_pod<uint32_t>(too_many_streams, 12u, 5u);
        refresh_checksum(too_many_streams);
        expect(!deserialize_metal_raster_archive(too_many_streams));

        auto too_many_fragment_outputs = serialize_metal_raster_archive(make_archive());
        write_pod<uint32_t>(too_many_fragment_outputs, 20u, 9u);
        refresh_checksum(too_many_fragment_outputs);
        expect(!deserialize_metal_raster_archive(too_many_fragment_outputs));
    };

    "metal_raster_archive_checksum_covers_valid_metadata"_test = [] {
        auto valid_but_wrong_root_size =
            serialize_metal_raster_archive(make_archive());
        write_pod<uint64_t>(valid_but_wrong_root_size, 24u, 16u);
        expect(!deserialize_metal_raster_archive(valid_but_wrong_root_size));

        // The first argument begins at byte 72 and Usage::NONE is structurally
        // valid, but changing READ residency metadata must invalidate the
        // archive rather than silently diverge from the embedded AIR ABI.
        auto valid_but_wrong_usage =
            serialize_metal_raster_archive(make_archive());
        write_pod<uint32_t>(valid_but_wrong_usage, 72u,
                            luisa::to_underlying(Usage::NONE));
        expect(!deserialize_metal_raster_archive(valid_but_wrong_usage));
    };

    "metal_raster_archive_rejects_invalid_mesh_formats"_test = [] {
        {
            auto source = make_archive();
            source.mesh_format.emplace_vertex_stream(
                span<const VertexAttribute>{});
            expect(!validate_metal_raster_mesh_format(source.mesh_format));
            expect(!deserialize_metal_raster_archive(
                serialize_metal_raster_archive(source)));
        }
        {
            MetalRasterArchive source{};
            constexpr std::array attribute{
                VertexAttribute{VertexAttributeType::Position,
                                PixelFormat::RGBA32F}};
            for (auto i = 0u; i < 5u; i++) {
                source.mesh_format.emplace_vertex_stream(attribute);
            }
            source.root_argument_size = 16u;
            source.fragment_output_count = 1u;
            source.library = {std::byte{'M'}};
            expect(!validate_metal_raster_mesh_format(source.mesh_format));
            expect(!deserialize_metal_raster_archive(
                serialize_metal_raster_archive(source)));
        }
        for (auto format : {PixelFormat::R10G10B10A2UInt,
                            PixelFormat::R11G11B10F,
                            PixelFormat::RGBA8SRGB}) {
            MetalRasterArchive source{};
            std::array attribute{
                VertexAttribute{VertexAttributeType::Position, format}};
            source.mesh_format.emplace_vertex_stream(attribute);
            source.root_argument_size = 16u;
            source.fragment_output_count = 1u;
            source.library = {std::byte{'M'}};
            expect(!validate_metal_raster_mesh_format(source.mesh_format));
            expect(!deserialize_metal_raster_archive(
                serialize_metal_raster_archive(source)));
        }
    };

    "metal_raster_archive_rejects_invalid_argument_metadata"_test = [] {
        // Header: magic/version/counts/output count/root size/library size = 40 bytes.
        // Streams: (count + two attributes) + (count + one attribute) = 32 bytes.
        constexpr auto first_argument_offset = 72u;
        auto invalid_usage = serialize_metal_raster_archive(make_archive());
        write_pod<uint32_t>(invalid_usage, first_argument_offset, 0x80u);
        refresh_checksum(invalid_usage);
        expect(!deserialize_metal_raster_archive(invalid_usage));

        auto invalid_stage = serialize_metal_raster_archive(make_archive());
        invalid_stage[first_argument_offset + sizeof(uint32_t)] = std::byte{2u};
        refresh_checksum(invalid_stage);
        expect(!deserialize_metal_raster_archive(invalid_stage));
    };

    "metal_raster_archive_rejects_empty_or_trailing_library"_test = [] {
        auto source = make_archive();
        source.library.clear();
        expect(!deserialize_metal_raster_archive(
            serialize_metal_raster_archive(source)));

        auto trailing = serialize_metal_raster_archive(make_archive());
        // Keep the checksum valid but claim one fewer library byte so the
        // structural reader must reject the remaining payload byte.
        write_pod<uint64_t>(trailing, 32u, 6u);
        refresh_checksum(trailing);
        expect(!deserialize_metal_raster_archive(trailing));
    };
}

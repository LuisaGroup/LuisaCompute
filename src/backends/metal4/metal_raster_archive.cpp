#include <array>
#include <cstring>
#include <limits>

#include <luisa/core/stl/hash.h>

#include "metal_raster_archive.h"

namespace luisa::compute::metal {

namespace {

constexpr uint64_t archive_magic = 0x5249415f43554c4cull;// "LLUC_AIR"
constexpr uint32_t archive_version = 3u;

template<typename T>
void append_pod(luisa::vector<std::byte> &data, const T &value) noexcept {
    static_assert(std::is_trivially_copyable_v<T>);
    auto offset = data.size();
    data.resize(offset + sizeof(T));
    std::memcpy(data.data() + offset, &value, sizeof(T));
}

void append_string(luisa::vector<std::byte> &data, luisa::string_view value) noexcept {
    auto size = static_cast<uint32_t>(value.size());
    append_pod(data, size);
    auto offset = data.size();
    data.resize(offset + size);
    std::memcpy(data.data() + offset, value.data(), size);
}

class ArchiveReader {

private:
    luisa::span<const std::byte> _data;
    size_t _offset{0u};
    bool _valid{true};

public:
    explicit ArchiveReader(luisa::span<const std::byte> data) noexcept
        : _data{data} {}

    template<typename T>
    [[nodiscard]] T read() noexcept {
        static_assert(std::is_trivially_copyable_v<T>);
        T result{};
        if (!_valid || sizeof(T) > _data.size() - std::min(_offset, _data.size())) {
            _valid = false;
            return result;
        }
        std::memcpy(&result, _data.data() + _offset, sizeof(T));
        _offset += sizeof(T);
        return result;
    }

    [[nodiscard]] luisa::string read_string() noexcept {
        auto size = read<uint32_t>();
        if (!_valid || size > _data.size() - std::min(_offset, _data.size())) {
            _valid = false;
            return {};
        }
        luisa::string result{
            reinterpret_cast<const char *>(_data.data() + _offset), size};
        _offset += size;
        return result;
    }

    [[nodiscard]] luisa::span<const std::byte> read_bytes(size_t size) noexcept {
        if (!_valid || size > _data.size() - std::min(_offset, _data.size())) {
            _valid = false;
            return {};
        }
        auto result = _data.subspan(_offset, size);
        _offset += size;
        return result;
    }

    [[nodiscard]] bool finished() const noexcept {
        return _valid && _offset == _data.size();
    }
};

}// namespace

bool validate_metal_raster_mesh_format(
    const MeshFormat &mesh_format, luisa::string *reason) noexcept {
    auto fail = [&](luisa::string message) noexcept {
        if (reason != nullptr) { *reason = std::move(message); }
        return false;
    };
    auto stream_count = mesh_format.vertex_stream_count();
    if (stream_count == 0u) {
        return fail("mesh format has no vertex streams");
    }
    if (stream_count > 4u) {
        return fail("mesh format has more than 4 vertex streams");
    }
    std::array<bool, kVertexAttributeCount> seen{};
    for (auto stream = 0u; stream < stream_count; stream++) {
        auto attributes = mesh_format.attributes(stream);
        if (attributes.empty()) {
            return fail("mesh format contains an empty vertex stream");
        }
        for (auto attribute : attributes) {
            auto semantic = static_cast<size_t>(attribute.type);
            if (semantic >= seen.size()) {
                return fail("mesh format contains an invalid vertex semantic");
            }
            if (seen[semantic]) {
                return fail("mesh format contains a duplicate vertex semantic");
            }
            seen[semantic] = true;
            if (luisa::to_underlying(attribute.format) >= pixel_format_count) {
                return fail("mesh format contains an invalid pixel format");
            }
            if (is_block_compressed(attribute.format)) {
                return fail("block-compressed formats cannot be vertex attributes");
            }
            if (attribute.format == PixelFormat::R10G10B10A2UInt ||
                attribute.format == PixelFormat::R11G11B10F ||
                attribute.format == PixelFormat::RGBA8SRGB) {
                return fail("vertex attribute format is unsupported by the macOS 13 Metal ABI");
            }
        }
    }
    if (reason != nullptr) { reason->clear(); }
    return true;
}

luisa::vector<std::byte>
serialize_metal_raster_archive(const MetalRasterArchive &archive) noexcept {
    luisa::vector<std::byte> data;
    append_pod(data, archive_magic);
    append_pod(data, archive_version);
    append_pod(data, static_cast<uint32_t>(archive.mesh_format.vertex_stream_count()));
    append_pod(data, static_cast<uint32_t>(archive.arguments.size()));
    append_pod(data, archive.fragment_output_count);
    append_pod(data, static_cast<uint64_t>(archive.root_argument_size));
    append_pod(data, static_cast<uint64_t>(archive.library.size()));
    for (auto stream = 0u; stream < archive.mesh_format.vertex_stream_count(); stream++) {
        auto attributes = archive.mesh_format.attributes(stream);
        append_pod(data, static_cast<uint32_t>(attributes.size()));
        for (auto attribute : attributes) {
            append_pod(data, static_cast<uint32_t>(attribute.type));
            append_pod(data, static_cast<uint32_t>(attribute.format));
        }
    }
    for (auto &&argument : archive.arguments) {
        append_pod(data, static_cast<uint32_t>(argument.usage));
        append_pod(data, static_cast<uint8_t>(argument.stage));
        append_string(data, argument.type);
    }
    auto offset = data.size();
    data.resize(offset + archive.library.size());
    std::memcpy(data.data() + offset,
                archive.library.data(), archive.library.size());
    auto checksum = luisa::hash64(
        data.data(), data.size(), metal_raster_archive_checksum_seed);
    append_pod(data, checksum);
    return data;
}

luisa::optional<MetalRasterArchive>
deserialize_metal_raster_archive(luisa::span<const std::byte> data) noexcept {
    if (data.size() < sizeof(uint64_t)) { return luisa::nullopt; }
    auto payload = data.first(data.size() - sizeof(uint64_t));
    uint64_t expected_checksum{};
    std::memcpy(&expected_checksum, data.data() + payload.size(),
                sizeof(expected_checksum));
    auto actual_checksum = luisa::hash64(
        payload.data(), payload.size(), metal_raster_archive_checksum_seed);
    if (actual_checksum != expected_checksum) { return luisa::nullopt; }
    ArchiveReader reader{payload};
    if (reader.read<uint64_t>() != archive_magic ||
        reader.read<uint32_t>() != archive_version) {
        return luisa::nullopt;
    }
    auto stream_count = reader.read<uint32_t>();
    auto argument_count = reader.read<uint32_t>();
    auto fragment_output_count = reader.read<uint32_t>();
    auto root_argument_size = reader.read<uint64_t>();
    auto library_size = reader.read<uint64_t>();
    if (stream_count == 0u || stream_count > 4u ||
        argument_count > 65536u ||
        fragment_output_count == 0u || fragment_output_count > 8u ||
        root_argument_size < 16u || root_argument_size > 65536u ||
        root_argument_size % 16u != 0u ||
        library_size > std::numeric_limits<size_t>::max()) {
        return luisa::nullopt;
    }
    MetalRasterArchive archive{};
    archive.root_argument_size = static_cast<size_t>(root_argument_size);
    archive.fragment_output_count = fragment_output_count;
    for (auto stream = 0u; stream < stream_count; stream++) {
        auto attribute_count = reader.read<uint32_t>();
        if (attribute_count > kVertexAttributeCount) { return luisa::nullopt; }
        luisa::vector<VertexAttribute> attributes;
        attributes.reserve(attribute_count);
        for (auto i = 0u; i < attribute_count; i++) {
            auto semantic = reader.read<uint32_t>();
            auto format = reader.read<uint32_t>();
            if (semantic >= kVertexAttributeCount || format >= pixel_format_count) {
                return luisa::nullopt;
            }
            attributes.emplace_back(VertexAttribute{
                .type = static_cast<VertexAttributeType>(semantic),
                .format = static_cast<PixelFormat>(format)});
        }
        if (attributes.empty()) { return luisa::nullopt; }
        archive.mesh_format.emplace_vertex_stream(attributes);
    }
    if (!validate_metal_raster_mesh_format(archive.mesh_format)) {
        return luisa::nullopt;
    }
    archive.arguments.reserve(argument_count);
    for (auto i = 0u; i < argument_count; i++) {
        auto usage = reader.read<uint32_t>();
        auto stage = reader.read<uint8_t>();
        if ((usage & ~luisa::to_underlying(Usage::READ_WRITE)) != 0u ||
            stage > static_cast<uint8_t>(MetalRasterArchiveStage::FRAGMENT)) {
            return luisa::nullopt;
        }
        auto type = reader.read_string();
        if (type.empty()) { return luisa::nullopt; }
        archive.arguments.emplace_back(MetalRasterArchiveArgument{
            .type = std::move(type),
            .usage = static_cast<Usage>(usage),
            .stage = static_cast<MetalRasterArchiveStage>(stage)});
    }
    auto library = reader.read_bytes(static_cast<size_t>(library_size));
    if (!reader.finished() || library.empty()) { return luisa::nullopt; }
    archive.library.assign(library.begin(), library.end());
    return archive;
}

}// namespace luisa::compute::metal

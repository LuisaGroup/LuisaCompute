#include "remote_protocol.h"

#include <bit>
#include <cstring>

#include <luisa/core/stl/hash.h>

namespace luisa::compute::remote {

namespace {

template<typename T>
void append_little_endian(luisa::vector<std::byte> &bytes, T value) noexcept {
    static_assert(std::is_unsigned_v<T>);
    for (auto i = 0u; i < sizeof(T); i++) {
        bytes.emplace_back(static_cast<std::byte>(value & static_cast<T>(0xffu)));
        value >>= 8u;
    }
}

template<typename T>
[[nodiscard]] T read_little_endian(luisa::span<const std::byte> bytes) noexcept {
    static_assert(std::is_unsigned_v<T>);
    T value{};
    for (auto i = 0u; i < sizeof(T); i++) {
        value |= static_cast<T>(std::to_integer<uint8_t>(bytes[i])) << (i * 8u);
    }
    return value;
}

}// namespace

void Writer::write_u8(uint8_t value) noexcept {
    _bytes.emplace_back(static_cast<std::byte>(value));
}

void Writer::write_u16(uint16_t value) noexcept {
    append_little_endian(_bytes, value);
}

void Writer::write_u32(uint32_t value) noexcept {
    append_little_endian(_bytes, value);
}

void Writer::write_u64(uint64_t value) noexcept {
    append_little_endian(_bytes, value);
}

void Writer::write_i64(int64_t value) noexcept {
    write_u64(std::bit_cast<uint64_t>(value));
}

void Writer::write_bool(bool value) noexcept {
    write_u8(value ? 1u : 0u);
}

void Writer::write_f32(float value) noexcept {
    write_u32(std::bit_cast<uint32_t>(value));
}

void Writer::write_bytes(luisa::span<const std::byte> value) noexcept {
    _bytes.insert(_bytes.end(), value.begin(), value.end());
}

void Writer::write_blob(luisa::span<const std::byte> value) noexcept {
    write_u64(value.size());
    write_bytes(value);
}

void Writer::write_string(luisa::string_view value) noexcept {
    write_u64(value.size());
    write_bytes({reinterpret_cast<const std::byte *>(value.data()), value.size()});
}

bool Reader::_require(size_t size) noexcept {
    if (!ok()) { return false; }
    if (size > remaining()) {
        _fail("Truncated remote protocol payload.");
        return false;
    }
    return true;
}

void Reader::_fail(luisa::string message) noexcept {
    if (_error.empty()) { _error = std::move(message); }
}

uint8_t Reader::read_u8() noexcept {
    if (!_require(1u)) { return 0u; }
    return std::to_integer<uint8_t>(_bytes[_offset++]);
}

uint16_t Reader::read_u16() noexcept {
    if (!_require(sizeof(uint16_t))) { return 0u; }
    auto value = read_little_endian<uint16_t>(_bytes.subspan(_offset, sizeof(uint16_t)));
    _offset += sizeof(uint16_t);
    return value;
}

uint32_t Reader::read_u32() noexcept {
    if (!_require(sizeof(uint32_t))) { return 0u; }
    auto value = read_little_endian<uint32_t>(_bytes.subspan(_offset, sizeof(uint32_t)));
    _offset += sizeof(uint32_t);
    return value;
}

uint64_t Reader::read_u64() noexcept {
    if (!_require(sizeof(uint64_t))) { return 0u; }
    auto value = read_little_endian<uint64_t>(_bytes.subspan(_offset, sizeof(uint64_t)));
    _offset += sizeof(uint64_t);
    return value;
}

int64_t Reader::read_i64() noexcept {
    return std::bit_cast<int64_t>(read_u64());
}

bool Reader::read_bool() noexcept {
    auto value = read_u8();
    if (ok() && value > 1u) { _fail("Invalid remote protocol boolean."); }
    return value != 0u;
}

float Reader::read_f32() noexcept {
    return std::bit_cast<float>(read_u32());
}

luisa::span<const std::byte> Reader::read_bytes(size_t size) noexcept {
    if (!_require(size)) { return {}; }
    auto value = _bytes.subspan(_offset, size);
    _offset += size;
    return value;
}

luisa::span<const std::byte> Reader::read_blob() noexcept {
    auto size = read_u64();
    if (!ok()) { return {}; }
    if (size > _limits.max_frame_payload || size > std::numeric_limits<size_t>::max()) {
        _fail("Remote protocol blob exceeds the configured limit.");
        return {};
    }
    return read_bytes(static_cast<size_t>(size));
}

luisa::string Reader::read_string() noexcept {
    auto size = read_u64();
    if (!ok()) { return {}; }
    if (size > _limits.max_string_size || size > std::numeric_limits<size_t>::max()) {
        _fail("Remote protocol string exceeds the configured limit.");
        return {};
    }
    auto value = read_bytes(static_cast<size_t>(size));
    if (!ok()) { return {}; }
    return {reinterpret_cast<const char *>(value.data()), value.size()};
}

bool Reader::finish() noexcept {
    if (ok() && remaining() != 0u) {
        _fail("Unexpected trailing bytes in remote protocol payload.");
    }
    return ok();
}

luisa::vector<std::byte> encode_frame_header(const FrameHeader &header) noexcept {
    Writer writer{frame_header_size};
    writer.write_u32(protocol_magic);
    writer.write_u16(header.wire_major);
    writer.write_u16(header.wire_minor);
    writer.write_u16(static_cast<uint16_t>(header.kind));
    writer.write_u16(header.flags);
    writer.write_u32(0u);
    writer.write_u64(header.request_id);
    writer.write_u64(header.payload_size);
    writer.write_u64(header.payload_checksum);
    return std::move(writer).take();
}

bool decode_frame_header(luisa::span<const std::byte> bytes,
                         FrameHeader &header,
                         luisa::string &error,
                         ProtocolLimits limits) noexcept {
    if (bytes.size() != frame_header_size) {
        error = "Invalid remote protocol frame-header size.";
        return false;
    }
    Reader reader{bytes, limits};
    if (reader.read_u32() != protocol_magic) {
        error = "Invalid remote protocol magic.";
        return false;
    }
    header.wire_major = reader.read_u16();
    header.wire_minor = reader.read_u16();
    if (header.wire_major != protocol_major ||
        header.wire_minor > protocol_minor) {
        error = "Unsupported remote protocol version.";
        return false;
    }
    header.kind = static_cast<MessageKind>(reader.read_u16());
    header.flags = reader.read_u16();
    if (reader.read_u32() != 0u) {
        error = "Remote protocol reserved frame bits are nonzero.";
        return false;
    }
    header.request_id = reader.read_u64();
    header.payload_size = reader.read_u64();
    header.payload_checksum = reader.read_u64();
    if (!reader.finish()) {
        error = reader.error();
        return false;
    }
    if (header.payload_size > limits.max_frame_payload ||
        header.payload_size > std::numeric_limits<size_t>::max()) {
        error = "Remote protocol frame exceeds the configured payload limit.";
        return false;
    }
    return true;
}

uint64_t payload_checksum(luisa::span<const std::byte> payload) noexcept {
    return luisa::hash64(payload.data(), payload.size(), 0x4c4352505f5631ull);
}

luisa::vector<std::byte> make_response_payload(
    MessageKind request_kind,
    Status status,
    luisa::string_view message,
    luisa::span<const std::byte> body) noexcept {
    Writer writer;
    writer.write_u16(static_cast<uint16_t>(request_kind));
    writer.write_u16(static_cast<uint16_t>(status));
    writer.write_string(message);
    writer.write_bytes(body);
    return std::move(writer).take();
}

bool decode_response_payload(luisa::span<const std::byte> payload,
                             ResponseView &response,
                             luisa::string &error,
                             ProtocolLimits limits) noexcept {
    Reader reader{payload, limits};
    response.request_kind = static_cast<MessageKind>(reader.read_u16());
    response.status = static_cast<Status>(reader.read_u16());
    response.message = reader.read_string();
    if (!reader.ok()) {
        error = reader.error();
        return false;
    }
    response.body = payload.subspan(reader.offset());
    return true;
}

}// namespace luisa::compute::remote

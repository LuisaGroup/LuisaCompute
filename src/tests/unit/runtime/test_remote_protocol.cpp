// Test for the C++ remote backend wire protocol.
// This test covers:
// - canonical little-endian encoding and frame headers
// - bounded payload parsing and malformed input rejection
// - response envelope round-tripping

#include "ut/ut.hpp"

#include <array>
#include <bit>
#include <cstring>

#include "remote_protocol.h"

using namespace luisa;
using namespace luisa::compute::remote;
using namespace boost::ut;
using namespace boost::ut::literals;

namespace {

[[nodiscard]] vector<std::byte> bytes_of(string_view text) {
    vector<std::byte> bytes;
    bytes.resize(text.size());
    std::memcpy(bytes.data(), text.data(), text.size());
    return bytes;
}

void test_primitive_round_trip() {
    Writer writer;
    writer.write_u8(0xabu);
    writer.write_u16(0x1234u);
    writer.write_u32(0x89abcdefu);
    writer.write_u64(0x0123456789abcdefull);
    writer.write_i64(-17);
    writer.write_bool(true);
    writer.write_bool(false);
    writer.write_f32(3.5f);
    writer.write_string("remote");
    auto blob = bytes_of("blob");
    writer.write_blob(blob);

    auto encoded = std::move(writer).take();
    expect(encoded.size() > 40u);
    expect(std::to_integer<uint8_t>(encoded[1]) == 0x34u)
        << "u16 must be little-endian";
    expect(std::to_integer<uint8_t>(encoded[2]) == 0x12u)
        << "u16 must be little-endian";

    Reader reader{encoded};
    expect(reader.read_u8() == 0xabu);
    expect(reader.read_u16() == 0x1234u);
    expect(reader.read_u32() == 0x89abcdefu);
    expect(reader.read_u64() == 0x0123456789abcdefull);
    expect(reader.read_i64() == -17);
    expect(reader.read_bool());
    expect(!reader.read_bool());
    expect(std::bit_cast<uint32_t>(reader.read_f32()) ==
           std::bit_cast<uint32_t>(3.5f));
    expect(reader.read_string() == "remote");
    auto decoded_blob = reader.read_blob();
    expect(decoded_blob.size() == blob.size());
    expect(std::memcmp(decoded_blob.data(), blob.data(), blob.size()) == 0);
    expect(reader.finish());
}

void test_frame_header_round_trip() {
    auto payload = bytes_of("frame payload");
    FrameHeader expected{
        .kind = MessageKind::DISPATCH,
        .flags = 0x12u,
        .request_id = 0x0102030405060708ull,
        .payload_size = payload.size(),
        .payload_checksum = payload_checksum(payload)};
    auto bytes = encode_frame_header(expected);
    expect(bytes.size() == frame_header_size);

    FrameHeader decoded;
    string error;
    expect(decode_frame_header(bytes, decoded, error));
    expect(error.empty());
    expect(decoded.kind == expected.kind);
    expect(decoded.flags == expected.flags);
    expect(decoded.request_id == expected.request_id);
    expect(decoded.payload_size == expected.payload_size);
    expect(decoded.payload_checksum == expected.payload_checksum);
    expect(decoded.wire_major == protocol_major);
    expect(decoded.wire_minor == protocol_minor);
    expect(payload_checksum(payload) != payload_checksum({}));

    expected.wire_minor = 0u;
    bytes = encode_frame_header(expected);
    error.clear();
    expect(decode_frame_header(bytes, decoded, error));
    expect(decoded.wire_minor == 0u);
}

void test_malformed_frame_headers() {
    auto bytes = encode_frame_header(FrameHeader{
        .kind = MessageKind::HELLO,
        .request_id = 1u});
    FrameHeader decoded;
    string error;

    expect(!decode_frame_header(span{bytes}.first(bytes.size() - 1u),
                                decoded, error));
    expect(!error.empty());

    auto bad_magic = bytes;
    bad_magic[0] = std::byte{0u};
    error.clear();
    expect(!decode_frame_header(bad_magic, decoded, error));
    expect(error == "Invalid remote protocol magic.");

    auto bad_version = bytes;
    bad_version[4] = std::byte{protocol_major + 1u};
    error.clear();
    expect(!decode_frame_header(bad_version, decoded, error));
    expect(error == "Unsupported remote protocol version.");

    auto bad_reserved = bytes;
    bad_reserved[12] = std::byte{1u};
    error.clear();
    expect(!decode_frame_header(bad_reserved, decoded, error));
    expect(error == "Remote protocol reserved frame bits are nonzero.");

    auto oversized = bytes;
    for (auto i = 0u; i < 8u; i++) {
        oversized[24u + i] = std::byte{0xffu};
    }
    error.clear();
    expect(!decode_frame_header(oversized, decoded, error,
                                ProtocolLimits{.max_frame_payload = 1024u}));
    expect(error ==
           "Remote protocol frame exceeds the configured payload limit.");
}

void test_reader_rejects_malformed_values() {
    {
        const std::array bytes{std::byte{2u}};
        Reader reader{bytes};
        static_cast<void>(reader.read_bool());
        expect(!reader.ok());
        expect(reader.error() == "Invalid remote protocol boolean.");
    }
    {
        Writer writer;
        writer.write_u64(128u);
        writer.write_bytes(bytes_of("short"));
        auto bytes = std::move(writer).take();
        Reader reader{bytes};
        static_cast<void>(reader.read_blob());
        expect(!reader.ok());
        expect(reader.error() == "Truncated remote protocol payload.");
    }
    {
        Writer writer;
        writer.write_string("too long");
        auto bytes = std::move(writer).take();
        Reader reader{bytes, ProtocolLimits{.max_string_size = 3u}};
        static_cast<void>(reader.read_string());
        expect(!reader.ok());
        expect(reader.error() ==
               "Remote protocol string exceeds the configured limit.");
    }
    {
        Writer writer;
        writer.write_u32(1u);
        writer.write_u32(2u);
        auto bytes = std::move(writer).take();
        Reader reader{bytes};
        expect(reader.read_u32() == 1u);
        expect(!reader.finish());
        expect(reader.error() ==
               "Unexpected trailing bytes in remote protocol payload.");
    }
}

void test_response_round_trip() {
    auto body = bytes_of("result");
    auto payload = make_response_payload(
        MessageKind::CREATE_BUFFER, Status::OK, {}, body);
    ResponseView response;
    string error;
    expect(decode_response_payload(payload, response, error));
    expect(response.request_kind == MessageKind::CREATE_BUFFER);
    expect(response.status == Status::OK);
    expect(response.message.empty());
    expect(response.body.size() == body.size());
    expect(std::memcmp(response.body.data(), body.data(), body.size()) == 0);

    auto failed = make_response_payload(
        MessageKind::CREATE_SHADER,
        Status::UNSUPPORTED,
        "native include is disabled");
    expect(decode_response_payload(failed, response, error));
    expect(response.request_kind == MessageKind::CREATE_SHADER);
    expect(response.status == Status::UNSUPPORTED);
    expect(response.message == "native include is disabled");
    expect(response.body.empty());
}

}// namespace

static auto test_remote_protocol_registration = [] {
    "remote_protocol_primitive_round_trip"_test = test_primitive_round_trip;
    "remote_protocol_frame_header_round_trip"_test =
        test_frame_header_round_trip;
    "remote_protocol_malformed_frame_headers"_test =
        test_malformed_frame_headers;
    "remote_protocol_malformed_values"_test =
        test_reader_rejects_malformed_values;
    "remote_protocol_response_round_trip"_test = test_response_round_trip;
    return 0;
}();

int main(int argc, char *argv[]) {
    boost::ut::detail::cfg::parse_arg_with_fallback(
        argc, const_cast<const char **>(argv));
}

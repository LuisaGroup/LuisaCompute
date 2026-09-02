#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>

#include <luisa/core/stl/memory.h>
#include <luisa/core/stl/string.h>
#include <luisa/core/stl/vector.h>

namespace luisa::compute::remote {

constexpr uint32_t protocol_magic = 0x5052434cu;// "LCRP" in little endian
constexpr uint16_t protocol_major = 1u;
constexpr uint16_t protocol_minor = 1u;
constexpr size_t frame_header_size = 40u;
constexpr uint64_t default_max_frame_payload = 512ull * 1024ull * 1024ull;
constexpr uint64_t default_max_string_size = 1ull * 1024ull * 1024ull;
constexpr uint64_t default_max_array_size = 1ull << 24u;

enum class MessageKind : uint16_t {
    HELLO = 1u,
    RESPONSE = 2u,
    ERROR = 3u,
    CREATE_BUFFER = 10u,
    DESTROY_BUFFER = 11u,
    CREATE_TEXTURE = 12u,
    DESTROY_TEXTURE = 13u,
    CREATE_BINDLESS_ARRAY = 14u,
    DESTROY_BINDLESS_ARRAY = 15u,
    CREATE_STREAM = 16u,
    DESTROY_STREAM = 17u,
    SYNCHRONIZE_STREAM = 18u,
    CREATE_SHADER = 19u,
    LOAD_SHADER = 20u,
    DESTROY_SHADER = 21u,
    CREATE_EVENT = 22u,
    DESTROY_EVENT = 23u,
    SIGNAL_EVENT = 24u,
    WAIT_EVENT = 25u,
    IS_EVENT_COMPLETED = 26u,
    SYNCHRONIZE_EVENT = 27u,
    CREATE_MESH = 28u,
    DESTROY_MESH = 29u,
    CREATE_PROCEDURAL_PRIMITIVE = 30u,
    DESTROY_PROCEDURAL_PRIMITIVE = 31u,
    CREATE_CURVE = 32u,
    DESTROY_CURVE = 33u,
    CREATE_MOTION_INSTANCE = 34u,
    DESTROY_MOTION_INSTANCE = 35u,
    CREATE_ACCEL = 36u,
    DESTROY_ACCEL = 37u,
    DISPATCH = 40u,
    DISPATCH_COMPLETE = 41u,
    QUERY = 42u,
    SET_NAME = 43u,
    STREAM_LOG = 44u,
    GOODBYE = 45u,
    SHADER_ARGUMENT_USAGE = 46u,
    SET_STREAM_LOG_CALLBACK = 47u,
    PREPARE_BLOBS = 48u,
    UPLOAD_BLOBS = 49u,
    BLOB_CACHE_INFO = 50u,
    PROTOCOL_INFO = 51u,
};

enum class Feature : uint64_t {
    BUFFER = 1ull << 0u,
    TEXTURE = 1ull << 1u,
    STREAM = 1ull << 2u,
    AST_SHADER = 1ull << 3u,
    EVENT = 1ull << 4u,
    ASYNC_DISPATCH = 1ull << 5u,
    STREAM_LOG = 1ull << 6u,
    BINDLESS_ARRAY = 1ull << 7u,
    RAY_TRACING = 1ull << 8u,
    BLOB_CACHE = 1ull << 9u,
    LIMIT_NEGOTIATION = 1ull << 10u,
    DEVICE_SELECTION = 1ull << 11u,
};

[[nodiscard]] constexpr uint64_t operator|(Feature lhs, Feature rhs) noexcept {
    return static_cast<uint64_t>(lhs) | static_cast<uint64_t>(rhs);
}

enum class ResourceKind : uint8_t {
    BUFFER = 1u,
    TEXTURE = 2u,
    BINDLESS_ARRAY = 3u,
    STREAM = 4u,
    SHADER = 5u,
    EVENT = 6u,
    MESH = 7u,
    CURVE = 8u,
    PROCEDURAL_PRIMITIVE = 9u,
    MOTION_INSTANCE = 10u,
    ACCEL = 11u,
};

enum class BufferKind : uint8_t {
    BYTE = 0u,
    INDIRECT_DISPATCH = 1u,
};

constexpr uint64_t resource_kind_shift = 56u;
constexpr uint64_t resource_index_mask = (1ull << resource_kind_shift) - 1ull;

[[nodiscard]] constexpr uint64_t make_resource_id(
    ResourceKind kind, uint64_t index) noexcept {
    return (static_cast<uint64_t>(kind) << resource_kind_shift) |
           (index & resource_index_mask);
}

[[nodiscard]] constexpr ResourceKind resource_kind(uint64_t id) noexcept {
    return static_cast<ResourceKind>(id >> resource_kind_shift);
}

enum class Status : uint16_t {
    OK = 0u,
    INVALID_REQUEST = 1u,
    UNSUPPORTED = 2u,
    NOT_FOUND = 3u,
    BACKEND_ERROR = 4u,
    VERSION_MISMATCH = 5u,
    AUTHENTICATION_FAILED = 6u,
    RESOURCE_LIMIT = 7u,
    CONNECTION_CLOSED = 8u,
    TIMEOUT = 9u,
};

struct ProtocolLimits {
    uint64_t max_frame_payload{default_max_frame_payload};
    uint64_t max_string_size{default_max_string_size};
    uint64_t max_array_size{default_max_array_size};
};

struct FrameHeader {
    MessageKind kind{};
    uint16_t flags{};
    uint64_t request_id{};
    uint64_t payload_size{};
    uint64_t payload_checksum{};
    uint16_t wire_major{protocol_major};
    uint16_t wire_minor{protocol_minor};
};

class Writer {

private:
    luisa::vector<std::byte> _bytes;

public:
    Writer() noexcept = default;
    explicit Writer(size_t reserve) noexcept { _bytes.reserve(reserve); }

    void write_u8(uint8_t value) noexcept;
    void write_u16(uint16_t value) noexcept;
    void write_u32(uint32_t value) noexcept;
    void write_u64(uint64_t value) noexcept;
    void write_i64(int64_t value) noexcept;
    void write_bool(bool value) noexcept;
    void write_f32(float value) noexcept;
    void write_bytes(luisa::span<const std::byte> value) noexcept;
    void write_blob(luisa::span<const std::byte> value) noexcept;
    void write_string(luisa::string_view value) noexcept;

    [[nodiscard]] auto bytes() const noexcept { return luisa::span{_bytes}; }
    [[nodiscard]] auto &&take() && noexcept { return std::move(_bytes); }
};

class Reader {

private:
    luisa::span<const std::byte> _bytes;
    ProtocolLimits _limits;
    size_t _offset{};
    luisa::string _error;

private:
    [[nodiscard]] bool _require(size_t size) noexcept;
    void _fail(luisa::string message) noexcept;

public:
    explicit Reader(luisa::span<const std::byte> bytes,
                    ProtocolLimits limits = {}) noexcept
        : _bytes{bytes}, _limits{limits} {}

    [[nodiscard]] uint8_t read_u8() noexcept;
    [[nodiscard]] uint16_t read_u16() noexcept;
    [[nodiscard]] uint32_t read_u32() noexcept;
    [[nodiscard]] uint64_t read_u64() noexcept;
    [[nodiscard]] int64_t read_i64() noexcept;
    [[nodiscard]] bool read_bool() noexcept;
    [[nodiscard]] float read_f32() noexcept;
    [[nodiscard]] luisa::span<const std::byte> read_bytes(size_t size) noexcept;
    [[nodiscard]] luisa::span<const std::byte> read_blob() noexcept;
    [[nodiscard]] luisa::string read_string() noexcept;

    [[nodiscard]] auto offset() const noexcept { return _offset; }
    [[nodiscard]] auto remaining() const noexcept { return _bytes.size() - _offset; }
    [[nodiscard]] auto error() const noexcept { return luisa::string_view{_error}; }
    [[nodiscard]] auto ok() const noexcept { return _error.empty(); }
    [[nodiscard]] bool finish() noexcept;
};

[[nodiscard]] luisa::vector<std::byte>
encode_frame_header(const FrameHeader &header) noexcept;

[[nodiscard]] bool decode_frame_header(
    luisa::span<const std::byte> bytes,
    FrameHeader &header,
    luisa::string &error,
    ProtocolLimits limits = {}) noexcept;

[[nodiscard]] uint64_t payload_checksum(
    luisa::span<const std::byte> payload) noexcept;

[[nodiscard]] luisa::vector<std::byte> make_response_payload(
    MessageKind request_kind,
    Status status,
    luisa::string_view message,
    luisa::span<const std::byte> body = {}) noexcept;

struct ResponseView {
    MessageKind request_kind{};
    Status status{Status::INVALID_REQUEST};
    luisa::string message;
    luisa::span<const std::byte> body;
};

[[nodiscard]] bool decode_response_payload(
    luisa::span<const std::byte> payload,
    ResponseView &response,
    luisa::string &error,
    ProtocolLimits limits = {}) noexcept;

}// namespace luisa::compute::remote

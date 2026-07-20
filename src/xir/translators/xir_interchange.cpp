#include <algorithm>
#include <array>
#include <charconv>
#include <cctype>
#include <cstdint>
#include <cstring>
#include <limits>
#include <type_traits>

#include <luisa/ast/type.h>
#include <luisa/core/stl/format.h>
#include <luisa/core/stl/memory.h>
#include <luisa/core/stl/unordered_map.h>
#include <luisa/xir/builder.h>
#include <luisa/xir/metadata/comment.h>
#include <luisa/xir/metadata/curve_basis.h>
#include <luisa/xir/metadata/location.h>
#include <luisa/xir/metadata/name.h>
#include <luisa/xir/metadata/reg2mem_spill.h>
#include <luisa/xir/metadata/signature_constraint.h>
#include <luisa/xir/translators/xir_interchange.h>
#include <luisa/xir/verifier.h>

#include "../instruction_semantics.h"

namespace luisa::compute::xir {

namespace {

constexpr std::array<std::byte, 8u> bitcode_magic{
    std::byte{'L'}, std::byte{'X'}, std::byte{'I'}, std::byte{'R'},
    std::byte{'B'}, std::byte{'C'}, std::byte{0u}, std::byte{1u}};
constexpr uint32_t interchange_version = 1u;
constexpr uint32_t bitcode_version = 2u;
constexpr size_t bitcode_header_size = 32u;
constexpr size_t max_payload_size = 256u * 1024u * 1024u;
constexpr size_t max_record_count = 1u << 20u;
constexpr size_t max_type_description_size = 16u * 1024u;
constexpr size_t max_string_payload_size = 1u * 1024u * 1024u;

[[nodiscard]] bool canonical_constant_size(
    const Type *type, size_t &size, size_t depth = 0u) noexcept {
    if (type == nullptr || depth > 64u) { return false; }
    switch (type->tag()) {
        case Type::Tag::BOOL:
        case Type::Tag::INT8:
        case Type::Tag::UINT8:
        case Type::Tag::INT16:
        case Type::Tag::UINT16:
        case Type::Tag::INT32:
        case Type::Tag::UINT32:
        case Type::Tag::INT64:
        case Type::Tag::UINT64:
        case Type::Tag::FLOAT16:
        case Type::Tag::FLOAT32:
        case Type::Tag::FLOAT64:
        case Type::Tag::FLOAT8_E4M3:
        case Type::Tag::FLOAT8_E5M2:
            size = type->size();
            return size == 1u || size == 2u || size == 4u || size == 8u;
        case Type::Tag::VECTOR:
        case Type::Tag::ARRAY: {
            size_t element_size = 0u;
            if (!canonical_constant_size(
                    type->element(), element_size, depth + 1u)) {
                return false;
            }
            auto dimension = static_cast<size_t>(type->dimension());
            if (dimension != 0u &&
                element_size > max_payload_size / dimension) {
                return false;
            }
            size = element_size * dimension;
            return true;
        }
        case Type::Tag::MATRIX: {
            size_t element_size = 0u;
            auto dimension = static_cast<size_t>(type->dimension());
            if (!canonical_constant_size(type->element(), element_size, depth + 1u) ||
                element_size > max_payload_size / dimension ||
                element_size * dimension > max_payload_size / dimension) {
                return false;
            }
            size = element_size * dimension * dimension;
            return true;
        }
        case Type::Tag::STRUCTURE: {
            size = 0u;
            for (auto member : type->members()) {
                size_t member_size = 0u;
                if (!canonical_constant_size(member, member_size, depth + 1u) ||
                    member_size > max_payload_size - size) {
                    return false;
                }
                size += member_size;
            }
            return true;
        }
        default: return false;
    }
}

void append_scalar_little_endian(
    luisa::vector<std::byte> &bytes, const std::byte *data, size_t size) noexcept {
    switch (size) {
        case 1u:
            bytes.emplace_back(*data);
            break;
        case 2u: {
            uint16_t value;
            std::memcpy(&value, data, sizeof(value));
            for (auto i = 0u; i < sizeof(value); i++) {
                bytes.emplace_back(static_cast<std::byte>((value >> (i * 8u)) & 0xffu));
            }
            break;
        }
        case 4u: {
            uint32_t value;
            std::memcpy(&value, data, sizeof(value));
            for (auto i = 0u; i < sizeof(value); i++) {
                bytes.emplace_back(static_cast<std::byte>((value >> (i * 8u)) & 0xffu));
            }
            break;
        }
        case 8u: {
            uint64_t value;
            std::memcpy(&value, data, sizeof(value));
            for (auto i = 0u; i < sizeof(value); i++) {
                bytes.emplace_back(static_cast<std::byte>((value >> (i * 8u)) & 0xffu));
            }
            break;
        }
        default: break;
    }
}

[[nodiscard]] bool encode_canonical_constant_impl(
    const Type *type, const std::byte *data,
    luisa::vector<std::byte> &bytes, size_t depth) noexcept {
    if (type == nullptr || data == nullptr || depth > 64u) { return false; }
    if (type->is_scalar()) {
        append_scalar_little_endian(bytes, data, type->size());
        return type->size() == 1u || type->size() == 2u ||
               type->size() == 4u || type->size() == 8u;
    }
    switch (type->tag()) {
        case Type::Tag::VECTOR:
        case Type::Tag::ARRAY: {
            auto element = type->element();
            for (auto i = 0u; i < type->dimension(); i++) {
                if (!encode_canonical_constant_impl(
                        element, data + i * element->size(), bytes, depth + 1u)) {
                    return false;
                }
            }
            return true;
        }
        case Type::Tag::MATRIX: {
            auto column = Type::vector(type->element(), type->dimension());
            for (auto i = 0u; i < type->dimension(); i++) {
                if (!encode_canonical_constant_impl(
                        column, data + i * column->size(), bytes, depth + 1u)) {
                    return false;
                }
            }
            return true;
        }
        case Type::Tag::STRUCTURE: {
            auto offset = size_t{0u};
            for (auto member : type->members()) {
                offset = luisa::align(offset, member->alignment());
                if (!encode_canonical_constant_impl(
                        member, data + offset, bytes, depth + 1u)) {
                    return false;
                }
                offset += member->size();
            }
            return true;
        }
        default: return false;
    }
}

[[nodiscard]] bool encode_canonical_constant(
    const Type *type, const void *data, luisa::vector<std::byte> &bytes) noexcept {
    size_t size = 0u;
    if (!canonical_constant_size(type, size) || data == nullptr) { return false; }
    bytes.clear();
    bytes.reserve(size);
    if (!encode_canonical_constant_impl(
            type, static_cast<const std::byte *>(data), bytes, 0u)) {
        bytes.clear();
        return false;
    }
    return bytes.size() == size;
}

[[nodiscard]] bool read_scalar_little_endian(
    luisa::span<const std::byte> bytes, size_t &offset,
    std::byte *data, size_t size, bool is_bool) noexcept {
    if (offset > bytes.size() || size > bytes.size() - offset) { return false; }
    if (is_bool) {
        auto value = std::to_integer<uint8_t>(bytes[offset]);
        if (size != 1u || value > 1u) { return false; }
        auto boolean = value != 0u;
        std::memcpy(data, &boolean, sizeof(boolean));
        offset++;
        return true;
    }
    switch (size) {
        case 1u:
            *data = bytes[offset];
            break;
        case 2u: {
            auto value = uint16_t{0u};
            for (auto i = 0u; i < sizeof(value); i++) {
                value |= static_cast<uint16_t>(std::to_integer<uint8_t>(bytes[offset + i])) << (i * 8u);
            }
            std::memcpy(data, &value, sizeof(value));
            break;
        }
        case 4u: {
            auto value = uint32_t{0u};
            for (auto i = 0u; i < sizeof(value); i++) {
                value |= static_cast<uint32_t>(std::to_integer<uint8_t>(bytes[offset + i])) << (i * 8u);
            }
            std::memcpy(data, &value, sizeof(value));
            break;
        }
        case 8u: {
            auto value = uint64_t{0u};
            for (auto i = 0u; i < sizeof(value); i++) {
                value |= static_cast<uint64_t>(std::to_integer<uint8_t>(bytes[offset + i])) << (i * 8u);
            }
            std::memcpy(data, &value, sizeof(value));
            break;
        }
        default: return false;
    }
    offset += size;
    return true;
}

[[nodiscard]] bool decode_canonical_constant_impl(
    const Type *type, luisa::span<const std::byte> bytes, size_t &offset,
    std::byte *data, size_t depth) noexcept {
    if (type == nullptr ||
        (data == nullptr && type->size() != 0u) || depth > 64u) {
        return false;
    }
    if (type->is_scalar()) {
        return read_scalar_little_endian(bytes, offset, data, type->size(), type->is_bool());
    }
    switch (type->tag()) {
        case Type::Tag::VECTOR:
        case Type::Tag::ARRAY: {
            auto element = type->element();
            for (auto i = 0u; i < type->dimension(); i++) {
                if (!decode_canonical_constant_impl(
                        element, bytes, offset, data + i * element->size(), depth + 1u)) {
                    return false;
                }
            }
            return true;
        }
        case Type::Tag::MATRIX: {
            auto column = Type::vector(type->element(), type->dimension());
            for (auto i = 0u; i < type->dimension(); i++) {
                if (!decode_canonical_constant_impl(
                        column, bytes, offset, data + i * column->size(), depth + 1u)) {
                    return false;
                }
            }
            return true;
        }
        case Type::Tag::STRUCTURE: {
            auto native_offset = size_t{0u};
            for (auto member : type->members()) {
                native_offset = luisa::align(native_offset, member->alignment());
                if (!decode_canonical_constant_impl(
                        member, bytes, offset, data + native_offset, depth + 1u)) {
                    return false;
                }
                native_offset += member->size();
            }
            return true;
        }
        default: return false;
    }
}

[[nodiscard]] bool decode_canonical_constant(
    const Type *type, luisa::span<const std::byte> bytes,
    luisa::vector<std::byte> &native) noexcept {
    size_t canonical_size = 0u;
    if (!canonical_constant_size(type, canonical_size) || bytes.size() != canonical_size) {
        return false;
    }
    native.assign(type->size(), std::byte{0u});
    auto offset = size_t{0u};
    return decode_canonical_constant_impl(type, bytes, offset, native.data(), 0u) &&
           offset == bytes.size();
}

[[nodiscard]] uint64_t checksum(luisa::span<const std::byte> bytes) noexcept {
    auto hash = uint64_t{14695981039346656037ull};
    for (auto byte : bytes) {
        hash ^= std::to_integer<uint8_t>(byte);
        hash *= 1099511628211ull;
    }
    return hash;
}

void append_uleb(luisa::vector<std::byte> &bytes, uint64_t value) noexcept {
    do {
        auto byte = static_cast<uint8_t>(value & 0x7fu);
        value >>= 7u;
        if (value != 0u) { byte |= 0x80u; }
        bytes.emplace_back(static_cast<std::byte>(byte));
    } while (value != 0u);
}

void append_zigzag(luisa::vector<std::byte> &bytes, int64_t value) noexcept {
    auto encoded = value >= 0 ? static_cast<uint64_t>(value) * 2u :
                                static_cast<uint64_t>(-(value + 1)) * 2u + 1u;
    append_uleb(bytes, encoded);
}

void append_u32(luisa::vector<std::byte> &bytes, uint32_t value) noexcept {
    for (auto i = 0u; i < 4u; i++) {
        bytes.emplace_back(static_cast<std::byte>((value >> (i * 8u)) & 0xffu));
    }
}

void append_u64(luisa::vector<std::byte> &bytes, uint64_t value) noexcept {
    for (auto i = 0u; i < 8u; i++) {
        bytes.emplace_back(static_cast<std::byte>((value >> (i * 8u)) & 0xffu));
    }
}

[[nodiscard]] uint32_t read_u32(luisa::span<const std::byte> bytes, size_t offset) noexcept {
    auto value = uint32_t{0u};
    for (auto i = 0u; i < 4u; i++) {
        value |= static_cast<uint32_t>(std::to_integer<uint8_t>(bytes[offset + i])) << (i * 8u);
    }
    return value;
}

[[nodiscard]] uint64_t read_u64(luisa::span<const std::byte> bytes, size_t offset) noexcept {
    auto value = uint64_t{0u};
    for (auto i = 0u; i < 8u; i++) {
        value |= static_cast<uint64_t>(std::to_integer<uint8_t>(bytes[offset + i])) << (i * 8u);
    }
    return value;
}

[[nodiscard]] XIRInterchangeDiagnostic diagnostic_at(
    luisa::string_view text, size_t offset, luisa::string message) noexcept {
    offset = std::min(offset, text.size());
    auto line = size_t{1u};
    auto column = size_t{1u};
    for (auto i = 0u; i < offset; i++) {
        if (text[i] == '\n') {
            line++;
            column = 1u;
        } else {
            column++;
        }
    }
    return XIRInterchangeDiagnostic{
        .offset = offset,
        .line = line,
        .column = column,
        .message = std::move(message)};
}

void append_quoted(luisa::string &text, luisa::string_view value) noexcept {
    constexpr auto hex = "0123456789abcdef";
    text.push_back('"');
    for (auto c : value) {
        auto u = static_cast<uint8_t>(c);
        switch (c) {
            case '\\': text.append("\\\\"); break;
            case '"': text.append("\\\""); break;
            case '\n': text.append("\\n"); break;
            case '\r': text.append("\\r"); break;
            case '\t': text.append("\\t"); break;
            default:
                if (u < 0x20u || u >= 0x7fu) {
                    text.append("\\x");
                    text.push_back(hex[u >> 4u]);
                    text.push_back(hex[u & 0x0fu]);
                } else {
                    text.push_back(c);
                }
                break;
        }
    }
    text.push_back('"');
}

[[nodiscard]] int hex_value(char c) noexcept {
    if (c >= '0' && c <= '9') { return c - '0'; }
    if (c >= 'a' && c <= 'f') { return c - 'a' + 10; }
    if (c >= 'A' && c <= 'F') { return c - 'A' + 10; }
    return -1;
}

class TextParser {
private:
    luisa::string_view _text;
    size_t _offset{0u};
    luisa::vector<XIRInterchangeDiagnostic> &_diagnostics;

private:
    void _skip_space_and_comments() noexcept {
        for (;;) {
            while (_offset < _text.size()) {
                auto c = _text[_offset];
                if (c != ' ' && c != '\t' && c != '\r' && c != '\n') { break; }
                _offset++;
            }
            if (_offset >= _text.size() || _text[_offset] != ';') { break; }
            while (_offset < _text.size() && _text[_offset] != '\n') { _offset++; }
        }
    }

    [[nodiscard]] bool _fail(luisa::string message) noexcept {
        if (_diagnostics.empty()) {
            _diagnostics.emplace_back(diagnostic_at(_text, _offset, std::move(message)));
        }
        return false;
    }

public:
    TextParser(luisa::string_view text,
               luisa::vector<XIRInterchangeDiagnostic> &diagnostics) noexcept
        : _text{text}, _diagnostics{diagnostics} {}

    [[nodiscard]] size_t offset() const noexcept { return _offset; }
    [[nodiscard]] size_t remaining_size() const noexcept { return _text.size() - _offset; }

    [[nodiscard]] bool keyword(luisa::string_view expected) noexcept {
        _skip_space_and_comments();
        auto begin = _offset;
        while (_offset < _text.size()) {
            auto c = _text[_offset];
            if (!((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
                  (c >= '0' && c <= '9') || c == '_' || c == '.' || c == '-')) {
                break;
            }
            _offset++;
        }
        if (_text.substr(begin, _offset - begin) == expected) { return true; }
        _offset = begin;
        return _fail(luisa::format("Expected '{}'.", expected));
    }

    [[nodiscard]] bool try_keyword(luisa::string_view expected) noexcept {
        _skip_space_and_comments();
        auto begin = _offset;
        while (_offset < _text.size()) {
            auto c = _text[_offset];
            if (!((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
                  (c >= '0' && c <= '9') || c == '_' || c == '.' || c == '-')) {
                break;
            }
            _offset++;
        }
        if (_text.substr(begin, _offset - begin) == expected) { return true; }
        _offset = begin;
        return false;
    }

    [[nodiscard]] bool word(luisa::string &value) noexcept {
        _skip_space_and_comments();
        auto begin = _offset;
        while (_offset < _text.size()) {
            auto c = _text[_offset];
            if (!((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
                  (c >= '0' && c <= '9') || c == '_' || c == '.' || c == '-')) {
                break;
            }
            _offset++;
        }
        if (_offset == begin) { return _fail("Expected a word token."); }
        value.assign(_text.data() + begin, _offset - begin);
        return true;
    }

    [[nodiscard]] bool punctuation(char expected) noexcept {
        _skip_space_and_comments();
        if (_offset < _text.size() && _text[_offset] == expected) {
            _offset++;
            return true;
        }
        return _fail(luisa::format("Expected '{}'.", expected));
    }

    [[nodiscard]] bool string(luisa::string &value) noexcept {
        _skip_space_and_comments();
        if (_offset >= _text.size() || _text[_offset] != '"') {
            return _fail("Expected a quoted string.");
        }
        _offset++;
        value.clear();
        while (_offset < _text.size()) {
            auto c = _text[_offset++];
            if (c == '"') { return true; }
            if (c != '\\') {
                if (static_cast<uint8_t>(c) < 0x20u) {
                    return _fail("Unescaped control character in quoted string.");
                }
                value.push_back(c);
                continue;
            }
            if (_offset >= _text.size()) { return _fail("Truncated string escape."); }
            auto escaped = _text[_offset++];
            switch (escaped) {
                case '\\': value.push_back('\\'); break;
                case '"': value.push_back('"'); break;
                case 'n': value.push_back('\n'); break;
                case 'r': value.push_back('\r'); break;
                case 't': value.push_back('\t'); break;
                case 'x': {
                    if (_offset + 2u > _text.size()) { return _fail("Truncated hexadecimal string escape."); }
                    auto high = hex_value(_text[_offset]);
                    auto low = hex_value(_text[_offset + 1u]);
                    if (high < 0 || low < 0) { return _fail("Invalid hexadecimal string escape."); }
                    value.push_back(static_cast<char>((high << 4) | low));
                    _offset += 2u;
                    break;
                }
                default: return _fail("Unknown quoted-string escape.");
            }
        }
        return _fail("Unterminated quoted string.");
    }

    [[nodiscard]] bool unsigned_integer(uint64_t &value) noexcept {
        _skip_space_and_comments();
        auto begin = _offset;
        value = 0u;
        while (_offset < _text.size() && _text[_offset] >= '0' && _text[_offset] <= '9') {
            auto digit = static_cast<uint64_t>(_text[_offset] - '0');
            if (value > (std::numeric_limits<uint64_t>::max() - digit) / 10u) {
                return _fail("Unsigned integer is out of range.");
            }
            value = value * 10u + digit;
            _offset++;
        }
        if (_offset == begin) { return _fail("Expected an unsigned integer."); }
        return true;
    }

    [[nodiscard]] bool signed_integer(int64_t &value) noexcept {
        _skip_space_and_comments();
        auto negative = false;
        if (_offset < _text.size() && _text[_offset] == '-') {
            negative = true;
            _offset++;
        }
        uint64_t magnitude = 0u;
        auto begin = _offset;
        while (_offset < _text.size() && _text[_offset] >= '0' && _text[_offset] <= '9') {
            auto digit = static_cast<uint64_t>(_text[_offset] - '0');
            constexpr auto max_magnitude = uint64_t{1u} << 63u;
            auto limit = negative ? max_magnitude : max_magnitude - 1u;
            if (magnitude > (limit - digit) / 10u) { return _fail("Signed integer is out of range."); }
            magnitude = magnitude * 10u + digit;
            _offset++;
        }
        if (_offset == begin) { return _fail("Expected a signed integer."); }
        if (negative && magnitude == (uint64_t{1u} << 63u)) {
            value = std::numeric_limits<int64_t>::min();
        } else {
            value = negative ? -static_cast<int64_t>(magnitude) : static_cast<int64_t>(magnitude);
        }
        return true;
    }

    [[nodiscard]] bool count(size_t &value) noexcept {
        uint64_t parsed = 0u;
        if (!unsigned_integer(parsed)) { return false; }
        if (parsed > max_record_count) { return _fail("Record count exceeds the supported limit."); }
        value = static_cast<size_t>(parsed);
        return true;
    }

    [[nodiscard]] bool at_end() noexcept {
        _skip_space_and_comments();
        return _offset == _text.size();
    }

    [[nodiscard]] bool fail(luisa::string message) noexcept { return _fail(std::move(message)); }
};

class ParseBudget {
private:
    size_t _remaining{max_record_count};

public:
    [[nodiscard]] bool consume(TextParser &parser, size_t count) noexcept {
        if (count > _remaining || count > parser.remaining_size()) {
            return parser.fail("Cumulative XIR record count exceeds the supported input budget.");
        }
        _remaining -= count;
        return true;
    }
};

class BinaryReader {
private:
    luisa::span<const std::byte> _bytes;
    size_t _offset{0u};
    size_t _base_offset{0u};
    luisa::vector<XIRInterchangeDiagnostic> &_diagnostics;

private:
    [[nodiscard]] bool _fail(luisa::string message) noexcept {
        if (_diagnostics.empty()) {
            _diagnostics.emplace_back(XIRInterchangeDiagnostic{
                .offset = _base_offset + _offset,
                .message = std::move(message)});
        }
        return false;
    }

public:
    BinaryReader(luisa::span<const std::byte> bytes, size_t base_offset,
                 luisa::vector<XIRInterchangeDiagnostic> &diagnostics) noexcept
        : _bytes{bytes}, _base_offset{base_offset}, _diagnostics{diagnostics} {}

    [[nodiscard]] size_t offset() const noexcept { return _offset; }
    [[nodiscard]] size_t remaining_size() const noexcept { return _bytes.size() - _offset; }
    [[nodiscard]] bool at_end() const noexcept { return _offset == _bytes.size(); }
    [[nodiscard]] bool fail(luisa::string message) noexcept { return _fail(std::move(message)); }

    [[nodiscard]] bool uleb(uint64_t &value) noexcept {
        value = 0u;
        for (auto i = 0u; i < 10u; i++) {
            if (_offset == _bytes.size()) { return _fail("Truncated ULEB128 integer."); }
            auto byte = std::to_integer<uint8_t>(_bytes[_offset++]);
            auto payload = static_cast<uint64_t>(byte & 0x7fu);
            if (i == 9u && payload > 1u) { return _fail("ULEB128 integer is out of range."); }
            value |= payload << (i * 7u);
            if ((byte & 0x80u) == 0u) {
                if (i != 0u && payload == 0u) {
                    return _fail("ULEB128 integer is not minimally encoded.");
                }
                return true;
            }
        }
        return _fail("ULEB128 integer is too long.");
    }

    [[nodiscard]] bool zigzag(int64_t &value) noexcept {
        uint64_t encoded = 0u;
        if (!uleb(encoded)) { return false; }
        if ((encoded & 1u) == 0u) {
            value = static_cast<int64_t>(encoded >> 1u);
        } else {
            value = -1 - static_cast<int64_t>(encoded >> 1u);
        }
        return true;
    }

    [[nodiscard]] bool count(size_t &value) noexcept {
        uint64_t encoded = 0u;
        if (!uleb(encoded)) { return false; }
        if (encoded > max_record_count) { return _fail("Record count exceeds the supported limit."); }
        value = static_cast<size_t>(encoded);
        return true;
    }

    [[nodiscard]] bool bytes(size_t size, luisa::span<const std::byte> &value) noexcept {
        if (size > remaining_size()) { return _fail("Truncated binary record payload."); }
        value = _bytes.subspan(_offset, size);
        _offset += size;
        return true;
    }
};

class BinaryParseBudget {
private:
    size_t _remaining{max_record_count};

public:
    [[nodiscard]] bool consume(BinaryReader &reader, size_t count) noexcept {
        if (count > _remaining || count > reader.remaining_size()) {
            return reader.fail("Cumulative XIR record count exceeds the supported binary budget.");
        }
        _remaining -= count;
        return true;
    }
};

class TypeDescriptionParser {
private:
    enum struct Kind { INVALID,
                       DATA,
                       CUSTOM,
                       RESOURCE,
                       COOPERATIVE_REFERENCE };
    struct Result {
        const Type *type{nullptr};
        Kind kind{Kind::INVALID};
        [[nodiscard]] explicit operator bool() const noexcept { return kind != Kind::INVALID; }
    };

    luisa::string_view _text;
    size_t _offset{0u};

private:
    [[nodiscard]] bool _punctuation(char c) noexcept {
        if (_offset < _text.size() && _text[_offset] == c) {
            _offset++;
            return true;
        }
        return false;
    }

    [[nodiscard]] luisa::string_view _word() noexcept {
        auto begin = _offset;
        if (_offset >= _text.size() ||
            !(std::isalpha(static_cast<unsigned char>(_text[_offset])) || _text[_offset] == '_')) {
            return {};
        }
        _offset++;
        while (_offset < _text.size() &&
               (std::isalnum(static_cast<unsigned char>(_text[_offset])) || _text[_offset] == '_')) {
            _offset++;
        }
        return _text.substr(begin, _offset - begin);
    }

    [[nodiscard]] bool _number(size_t &value) noexcept {
        auto begin = _offset;
        value = 0u;
        while (_offset < _text.size() && _text[_offset] >= '0' && _text[_offset] <= '9') {
            auto digit = static_cast<size_t>(_text[_offset] - '0');
            if (value > (max_record_count - digit) / 10u) { return false; }
            value = value * 10u + digit;
            _offset++;
        }
        return _offset != begin;
    }

    [[nodiscard]] Result _type(size_t depth) noexcept {
        if (depth > 64u) { return {}; }
        auto name = _word();
        if (name.empty()) { return {}; }
        if (name == "bool" || name == "byte" || name == "ubyte" ||
            name == "short" || name == "ushort" || name == "int" ||
            name == "uint" || name == "long" || name == "ulong" ||
            name == "half" || name == "float" || name == "double" ||
            name == "float8e4m3" || name == "float8e5m2") {
            return {Type::from(name), Kind::DATA};
        }
        if (name == "bindless_array" || name == "accel") {
            return {Type::from(name), Kind::RESOURCE};
        }
        if (name == "vector") {
            size_t dimension = 0u;
            if (!_punctuation('<')) { return {}; }
            auto element = _type(depth + 1u);
            if (!element || element.kind != Kind::DATA || element.type == nullptr ||
                !element.type->is_scalar() || !_punctuation(',') || !_number(dimension) ||
                dimension < 2u || dimension > 4u || !_punctuation('>')) {
                return {};
            }
            return {Type::vector(element.type, dimension), Kind::DATA};
        }
        if (name == "matrix") {
            size_t dimension = 0u;
            if (!_punctuation('<') || !_number(dimension) ||
                dimension < 2u || dimension > 4u || !_punctuation('>')) {
                return {};
            }
            return {Type::matrix(dimension), Kind::DATA};
        }
        if (name == "array") {
            size_t dimension = 0u;
            if (!_punctuation('<')) { return {}; }
            auto element = _type(depth + 1u);
            if (!element || element.kind != Kind::DATA || element.type == nullptr ||
                element.type->is_cooperative_vector() || !_punctuation(',') ||
                !_number(dimension) || !_punctuation('>') ||
                (dimension != 0u &&
                 element.type->size() > max_payload_size / dimension)) {
                return {};
            }
            return {Type::array(element.type, dimension), Kind::DATA};
        }
        if (name == "coopvec") {
            size_t dimension = 0u;
            if (!_punctuation('<')) { return {}; }
            auto element = _type(depth + 1u);
            if (!element || element.kind != Kind::DATA || element.type == nullptr ||
                !element.type->is_scalar() || !_punctuation(',') || !_number(dimension) ||
                dimension == 0u || !_punctuation('>') ||
                element.type->size() > max_payload_size / dimension) {
                return {};
            }
            return {Type::cooperative_vector(element.type, dimension), Kind::DATA};
        }
        if (name == "coopvec_ref") {
            size_t dimension = 0u;
            size_t element = 0u;
            if (!_punctuation('<') || !_number(dimension) || dimension == 0u ||
                !_punctuation(',') || !_number(element) || element >= Type::coop_ref_type_size ||
                !_punctuation('>')) {
                return {};
            }
            return {Type::cooperative_vector_ref(static_cast<CoopRefVecType>(element), dimension), Kind::COOPERATIVE_REFERENCE};
        }
        if (name == "coopmat_ref") {
            size_t rows = 0u;
            size_t columns = 0u;
            size_t element = 0u;
            if (!_punctuation('<') || !_number(rows) || rows == 0u ||
                !_punctuation(',') || !_number(columns) || columns == 0u ||
                !_punctuation(',') || !_number(element) || element >= Type::coop_ref_type_size ||
                !_punctuation('>')) {
                return {};
            }
            return {Type::cooperative_matrix_ref(static_cast<CoopRefVecType>(element), rows, columns), Kind::COOPERATIVE_REFERENCE};
        }
        if (name == "struct") {
            size_t alignment = 0u;
            if (!_punctuation('<') || !_number(alignment) ||
                (alignment != 1u && alignment != 4u && alignment != 8u && alignment != 16u) ||
                (_offset >= _text.size() ||
                 (_text[_offset] != ',' && _text[_offset] != '>'))) {
                return {};
            }
            luisa::vector<const Type *> members;
            if (_punctuation(',')) {
                auto member = _type(depth + 1u);
                if (!member || member.kind != Kind::DATA || member.type == nullptr ||
                    member.type->is_cooperative_vector() || member.type->alignment() > alignment) {
                    return {};
                }
                members.emplace_back(member.type);
                while (_punctuation(',')) {
                    member = _type(depth + 1u);
                    if (!member || member.kind != Kind::DATA || member.type == nullptr ||
                        member.type->is_cooperative_vector() || member.type->alignment() > alignment) {
                        return {};
                    }
                    members.emplace_back(member.type);
                }
            }
            if (!_punctuation('>')) { return {}; }
            return {Type::structure(alignment, members), Kind::DATA};
        }
        if (name == "buffer") {
            if (!_punctuation('<')) { return {}; }
            if (_text.substr(_offset, 4u) == "void") {
                _offset += 4u;
                if (!_punctuation('>')) { return {}; }
                return {Type::from("buffer<void>"), Kind::RESOURCE};
            }
            auto element = _type(depth + 1u);
            if (!element || element.kind == Kind::RESOURCE ||
                element.kind == Kind::COOPERATIVE_REFERENCE || element.type == nullptr ||
                element.type->is_cooperative_vector() || !_punctuation('>')) {
                return {};
            }
            return {Type::buffer(element.type), Kind::RESOURCE};
        }
        if (name == "texture") {
            size_t dimension = 0u;
            if (!_punctuation('<') || !_number(dimension) ||
                (dimension != 2u && dimension != 3u) || !_punctuation(',')) {
                return {};
            }
            auto element = _type(depth + 1u);
            if (!element || element.kind != Kind::DATA || element.type == nullptr ||
                !(element.type->is_int32() || element.type->is_uint32() || element.type->is_float32()) ||
                !_punctuation('>')) {
                return {};
            }
            return {Type::texture(element.type, dimension), Kind::RESOURCE};
        }
        return {Type::custom(name), Kind::CUSTOM};
    }

public:
    explicit TypeDescriptionParser(luisa::string_view text) noexcept : _text{text} {}
    [[nodiscard]] const Type *parse() noexcept {
        if (_text.empty() || _text.size() > max_type_description_size) { return nullptr; }
        auto result = _type(0u);
        return result && _offset == _text.size() ? result.type : nullptr;
    }
};

[[nodiscard]] const Type *parse_type(luisa::string_view description) noexcept {
    if (description == "void") { return nullptr; }
    TypeDescriptionParser parser{description};
    return parser.parse();
}

[[nodiscard]] bool round_trippable_type(const Type *type) noexcept {
    return type == nullptr || parse_type(type->description()) == type;
}

[[nodiscard]] int64_t encode_switch_case_value(
    const Type *selector_type, SwitchInst::case_value_type value) noexcept {
    switch (selector_type->tag()) {
        case Type::Tag::INT8:
            return luisa::bit_cast<int8_t>(static_cast<uint8_t>(value));
        case Type::Tag::INT16:
            return luisa::bit_cast<int16_t>(static_cast<uint16_t>(value));
        case Type::Tag::INT32:
            return luisa::bit_cast<int32_t>(static_cast<uint32_t>(value));
        case Type::Tag::INT64:
        case Type::Tag::UINT64: return luisa::bit_cast<int64_t>(value);
        default: return static_cast<int64_t>(value);
    }
}

[[nodiscard]] luisa::optional<SwitchInst::case_value_type>
decode_switch_case_value(const Type *selector_type, int64_t value) noexcept {
    if (selector_type == nullptr) { return luisa::nullopt; }
    auto signed_value = [&]<typename T>() noexcept
        -> luisa::optional<SwitchInst::case_value_type> {
        static_assert(std::is_signed_v<T>);
        if (value < static_cast<int64_t>(std::numeric_limits<T>::min()) ||
            value > static_cast<int64_t>(std::numeric_limits<T>::max())) {
            return luisa::nullopt;
        }
        using U = std::make_unsigned_t<T>;
        return static_cast<SwitchInst::case_value_type>(
            luisa::bit_cast<U>(static_cast<T>(value)));
    };
    auto unsigned_value = [&]<typename T>() noexcept
        -> luisa::optional<SwitchInst::case_value_type> {
        static_assert(std::is_unsigned_v<T>);
        if (value < 0 ||
            static_cast<uint64_t>(value) >
                static_cast<uint64_t>(std::numeric_limits<T>::max())) {
            return luisa::nullopt;
        }
        return static_cast<SwitchInst::case_value_type>(value);
    };
    switch (selector_type->tag()) {
        case Type::Tag::BOOL:
            return value == 0 || value == 1 ?
                       luisa::optional<SwitchInst::case_value_type>{
                           static_cast<SwitchInst::case_value_type>(value)} :
                       luisa::nullopt;
        case Type::Tag::INT8: return signed_value.template operator()<int8_t>();
        case Type::Tag::UINT8: return unsigned_value.template operator()<uint8_t>();
        case Type::Tag::INT16: return signed_value.template operator()<int16_t>();
        case Type::Tag::UINT16: return unsigned_value.template operator()<uint16_t>();
        case Type::Tag::INT32: return signed_value.template operator()<int32_t>();
        case Type::Tag::UINT32: return unsigned_value.template operator()<uint32_t>();
        case Type::Tag::INT64:
        case Type::Tag::UINT64: return luisa::bit_cast<uint64_t>(value);
        default: return luisa::nullopt;
    }
}

[[nodiscard]] luisa::optional<luisa::string_view>
instruction_name(DerivedInstructionTag tag) noexcept {
    using namespace std::string_view_literals;
    switch (tag) {
        case DerivedInstructionTag::IF: return "if"sv;
        case DerivedInstructionTag::SWITCH: return "switch"sv;
        case DerivedInstructionTag::LOOP: return "loop"sv;
        case DerivedInstructionTag::SIMPLE_LOOP: return "simple_loop"sv;
        case DerivedInstructionTag::BRANCH: return "branch"sv;
        case DerivedInstructionTag::CONDITIONAL_BRANCH: return "conditional_branch"sv;
        case DerivedInstructionTag::UNREACHABLE: return "unreachable"sv;
        case DerivedInstructionTag::RASTER_DISCARD: return "raster_discard"sv;
        case DerivedInstructionTag::CORO_SUSPEND: return "coro_suspend"sv;
        case DerivedInstructionTag::CORO_RESUME: return "coro_resume"sv;
        case DerivedInstructionTag::CORO_TERMINATE: return "coro_terminate"sv;
        case DerivedInstructionTag::RETURN: return "return"sv;
        case DerivedInstructionTag::PHI: return "phi"sv;
        case DerivedInstructionTag::ALLOCA: return "alloca"sv;
        case DerivedInstructionTag::LOAD: return "load"sv;
        case DerivedInstructionTag::STORE: return "store"sv;
        case DerivedInstructionTag::GEP: return "gep"sv;
        case DerivedInstructionTag::ATOMIC: return "atomic"sv;
        case DerivedInstructionTag::ARITHMETIC: return "arithmetic"sv;
        case DerivedInstructionTag::THREAD_GROUP: return "thread_group"sv;
        case DerivedInstructionTag::RESOURCE_QUERY: return "resource_query"sv;
        case DerivedInstructionTag::RESOURCE_READ: return "resource_read"sv;
        case DerivedInstructionTag::RESOURCE_WRITE: return "resource_write"sv;
        case DerivedInstructionTag::RAY_QUERY_LOOP: return "ray_query_loop"sv;
        case DerivedInstructionTag::RAY_QUERY_DISPATCH: return "ray_query_dispatch"sv;
        case DerivedInstructionTag::RAY_QUERY_OBJECT_READ: return "ray_query_object_read"sv;
        case DerivedInstructionTag::RAY_QUERY_OBJECT_WRITE: return "ray_query_object_write"sv;
        case DerivedInstructionTag::RAY_QUERY_PIPELINE: return "ray_query_pipeline"sv;
        case DerivedInstructionTag::AUTODIFF_SCOPE: return "autodiff_scope"sv;
        case DerivedInstructionTag::AUTODIFF_INTRINSIC: return "autodiff_intrinsic"sv;
        case DerivedInstructionTag::BREAK: return "break"sv;
        case DerivedInstructionTag::CONTINUE: return "continue"sv;
        case DerivedInstructionTag::CALL: return "call"sv;
        case DerivedInstructionTag::CAST: return "cast"sv;
        case DerivedInstructionTag::PRINT: return "print"sv;
        case DerivedInstructionTag::CLOCK: return "clock"sv;
        case DerivedInstructionTag::DEBUG_BREAK: return "debug_break"sv;
        case DerivedInstructionTag::ASSERT: return "assert"sv;
        case DerivedInstructionTag::ASSUME: return "assume"sv;
        case DerivedInstructionTag::OUTLINE: return "outline"sv;
        default: return luisa::nullopt;
    }
}

[[nodiscard]] luisa::optional<DerivedInstructionTag>
parse_instruction_name(luisa::string_view name) noexcept {
    if (name == "if") { return DerivedInstructionTag::IF; }
    if (name == "switch") { return DerivedInstructionTag::SWITCH; }
    if (name == "loop") { return DerivedInstructionTag::LOOP; }
    if (name == "simple_loop") { return DerivedInstructionTag::SIMPLE_LOOP; }
    if (name == "branch") { return DerivedInstructionTag::BRANCH; }
    if (name == "conditional_branch") { return DerivedInstructionTag::CONDITIONAL_BRANCH; }
    if (name == "unreachable") { return DerivedInstructionTag::UNREACHABLE; }
    if (name == "raster_discard") { return DerivedInstructionTag::RASTER_DISCARD; }
    if (name == "coro_suspend") { return DerivedInstructionTag::CORO_SUSPEND; }
    if (name == "coro_resume") { return DerivedInstructionTag::CORO_RESUME; }
    if (name == "coro_terminate") { return DerivedInstructionTag::CORO_TERMINATE; }
    if (name == "return") { return DerivedInstructionTag::RETURN; }
    if (name == "phi") { return DerivedInstructionTag::PHI; }
    if (name == "alloca") { return DerivedInstructionTag::ALLOCA; }
    if (name == "load") { return DerivedInstructionTag::LOAD; }
    if (name == "store") { return DerivedInstructionTag::STORE; }
    if (name == "gep") { return DerivedInstructionTag::GEP; }
    if (name == "atomic") { return DerivedInstructionTag::ATOMIC; }
    if (name == "arithmetic") { return DerivedInstructionTag::ARITHMETIC; }
    if (name == "thread_group") { return DerivedInstructionTag::THREAD_GROUP; }
    if (name == "resource_query") { return DerivedInstructionTag::RESOURCE_QUERY; }
    if (name == "resource_read") { return DerivedInstructionTag::RESOURCE_READ; }
    if (name == "resource_write") { return DerivedInstructionTag::RESOURCE_WRITE; }
    if (name == "ray_query_loop") { return DerivedInstructionTag::RAY_QUERY_LOOP; }
    if (name == "ray_query_dispatch") { return DerivedInstructionTag::RAY_QUERY_DISPATCH; }
    if (name == "ray_query_object_read") { return DerivedInstructionTag::RAY_QUERY_OBJECT_READ; }
    if (name == "ray_query_object_write") { return DerivedInstructionTag::RAY_QUERY_OBJECT_WRITE; }
    if (name == "ray_query_pipeline") { return DerivedInstructionTag::RAY_QUERY_PIPELINE; }
    if (name == "autodiff_scope") { return DerivedInstructionTag::AUTODIFF_SCOPE; }
    if (name == "autodiff_intrinsic") { return DerivedInstructionTag::AUTODIFF_INTRINSIC; }
    if (name == "break") { return DerivedInstructionTag::BREAK; }
    if (name == "continue") { return DerivedInstructionTag::CONTINUE; }
    if (name == "call") { return DerivedInstructionTag::CALL; }
    if (name == "cast") { return DerivedInstructionTag::CAST; }
    if (name == "print") { return DerivedInstructionTag::PRINT; }
    if (name == "clock") { return DerivedInstructionTag::CLOCK; }
    if (name == "debug_break") { return DerivedInstructionTag::DEBUG_BREAK; }
    if (name == "assert") { return DerivedInstructionTag::ASSERT; }
    if (name == "assume") { return DerivedInstructionTag::ASSUME; }
    if (name == "outline") { return DerivedInstructionTag::OUTLINE; }
    return luisa::nullopt;
}

[[nodiscard]] luisa::optional<int64_t>
parse_integer_token(luisa::string_view token) noexcept {
    int64_t value = 0;
    auto [end, error] = std::from_chars(token.data(), token.data() + token.size(), value);
    if (error != std::errc{} || end != token.data() + token.size()) { return luisa::nullopt; }
    return value;
}

template<typename Op>
struct WireOpToken {
    Op op;
    luisa::string_view name;
};

template<typename Op>
[[nodiscard]] constexpr auto wire_op(Op op, luisa::string_view name) noexcept {
    return WireOpToken<Op>{op, name};
}

using namespace std::string_view_literals;

constexpr std::array alloca_wire_ops{
    wire_op(AllocaOp::LOCAL, "local"sv),
    wire_op(AllocaOp::SHARED, "shared"sv)};

constexpr std::array arithmetic_wire_ops{
    wire_op(ArithmeticOp::UNARY_MINUS, "unary_minus"sv),
    wire_op(ArithmeticOp::UNARY_BIT_NOT, "unary_bit_not"sv),
    wire_op(ArithmeticOp::BINARY_ADD, "binary_add"sv),
    wire_op(ArithmeticOp::BINARY_SUB, "binary_sub"sv),
    wire_op(ArithmeticOp::BINARY_MUL, "binary_mul"sv),
    wire_op(ArithmeticOp::BINARY_DIV, "binary_div"sv),
    wire_op(ArithmeticOp::BINARY_MOD, "binary_mod"sv),
    wire_op(ArithmeticOp::BINARY_BIT_AND, "binary_bit_and"sv),
    wire_op(ArithmeticOp::BINARY_BIT_OR, "binary_bit_or"sv),
    wire_op(ArithmeticOp::BINARY_BIT_XOR, "binary_bit_xor"sv),
    wire_op(ArithmeticOp::BINARY_SHIFT_LEFT, "binary_shift_left"sv),
    wire_op(ArithmeticOp::BINARY_SHIFT_RIGHT, "binary_shift_right"sv),
    wire_op(ArithmeticOp::BINARY_ROTATE_LEFT, "binary_rotate_left"sv),
    wire_op(ArithmeticOp::BINARY_ROTATE_RIGHT, "binary_rotate_right"sv),
    wire_op(ArithmeticOp::BINARY_LESS, "binary_less"sv),
    wire_op(ArithmeticOp::BINARY_GREATER, "binary_greater"sv),
    wire_op(ArithmeticOp::BINARY_LESS_EQUAL, "binary_less_equal"sv),
    wire_op(ArithmeticOp::BINARY_GREATER_EQUAL, "binary_greater_equal"sv),
    wire_op(ArithmeticOp::BINARY_EQUAL, "binary_equal"sv),
    wire_op(ArithmeticOp::BINARY_NOT_EQUAL, "binary_not_equal"sv),
    wire_op(ArithmeticOp::ALL, "all"sv),
    wire_op(ArithmeticOp::ANY, "any"sv),
    wire_op(ArithmeticOp::SELECT, "select"sv),
    wire_op(ArithmeticOp::CLAMP, "clamp"sv),
    wire_op(ArithmeticOp::SATURATE, "saturate"sv),
    wire_op(ArithmeticOp::LERP, "lerp"sv),
    wire_op(ArithmeticOp::SMOOTHSTEP, "smoothstep"sv),
    wire_op(ArithmeticOp::STEP, "step"sv),
    wire_op(ArithmeticOp::ABS, "abs"sv),
    wire_op(ArithmeticOp::MIN, "min"sv),
    wire_op(ArithmeticOp::MAX, "max"sv),
    wire_op(ArithmeticOp::CLZ, "clz"sv),
    wire_op(ArithmeticOp::CTZ, "ctz"sv),
    wire_op(ArithmeticOp::POPCOUNT, "popcount"sv),
    wire_op(ArithmeticOp::REVERSE, "reverse"sv),
    wire_op(ArithmeticOp::ISINF, "isinf"sv),
    wire_op(ArithmeticOp::ISNAN, "isnan"sv),
    wire_op(ArithmeticOp::ACOS, "acos"sv),
    wire_op(ArithmeticOp::ACOSH, "acosh"sv),
    wire_op(ArithmeticOp::ASIN, "asin"sv),
    wire_op(ArithmeticOp::ASINH, "asinh"sv),
    wire_op(ArithmeticOp::ATAN, "atan"sv),
    wire_op(ArithmeticOp::ATAN2, "atan2"sv),
    wire_op(ArithmeticOp::ATANH, "atanh"sv),
    wire_op(ArithmeticOp::COS, "cos"sv),
    wire_op(ArithmeticOp::COSH, "cosh"sv),
    wire_op(ArithmeticOp::SIN, "sin"sv),
    wire_op(ArithmeticOp::SINH, "sinh"sv),
    wire_op(ArithmeticOp::TAN, "tan"sv),
    wire_op(ArithmeticOp::TANH, "tanh"sv),
    wire_op(ArithmeticOp::EXP, "exp"sv),
    wire_op(ArithmeticOp::EXP2, "exp2"sv),
    wire_op(ArithmeticOp::EXP10, "exp10"sv),
    wire_op(ArithmeticOp::LOG, "log"sv),
    wire_op(ArithmeticOp::LOG2, "log2"sv),
    wire_op(ArithmeticOp::LOG10, "log10"sv),
    wire_op(ArithmeticOp::POW, "pow"sv),
    wire_op(ArithmeticOp::POW_INT, "pow_int"sv),
    wire_op(ArithmeticOp::SQRT, "sqrt"sv),
    wire_op(ArithmeticOp::RSQRT, "rsqrt"sv),
    wire_op(ArithmeticOp::CEIL, "ceil"sv),
    wire_op(ArithmeticOp::FLOOR, "floor"sv),
    wire_op(ArithmeticOp::FRACT, "fract"sv),
    wire_op(ArithmeticOp::TRUNC, "trunc"sv),
    wire_op(ArithmeticOp::ROUND, "round"sv),
    wire_op(ArithmeticOp::RINT, "rint"sv),
    wire_op(ArithmeticOp::FMA, "fma"sv),
    wire_op(ArithmeticOp::COPYSIGN, "copysign"sv),
    wire_op(ArithmeticOp::CROSS, "cross"sv),
    wire_op(ArithmeticOp::DOT, "dot"sv),
    wire_op(ArithmeticOp::LENGTH, "length"sv),
    wire_op(ArithmeticOp::LENGTH_SQUARED, "length_squared"sv),
    wire_op(ArithmeticOp::NORMALIZE, "normalize"sv),
    wire_op(ArithmeticOp::FACEFORWARD, "faceforward"sv),
    wire_op(ArithmeticOp::REFLECT, "reflect"sv),
    wire_op(ArithmeticOp::REDUCE_SUM, "reduce_sum"sv),
    wire_op(ArithmeticOp::REDUCE_PRODUCT, "reduce_product"sv),
    wire_op(ArithmeticOp::REDUCE_MIN, "reduce_min"sv),
    wire_op(ArithmeticOp::REDUCE_MAX, "reduce_max"sv),
    wire_op(ArithmeticOp::OUTER_PRODUCT, "outer_product"sv),
    wire_op(ArithmeticOp::MATRIX_COMP_NEG, "matrix_comp_neg"sv),
    wire_op(ArithmeticOp::MATRIX_COMP_ADD, "matrix_comp_add"sv),
    wire_op(ArithmeticOp::MATRIX_COMP_SUB, "matrix_comp_sub"sv),
    wire_op(ArithmeticOp::MATRIX_COMP_MUL, "matrix_comp_mul"sv),
    wire_op(ArithmeticOp::MATRIX_COMP_DIV, "matrix_comp_div"sv),
    wire_op(ArithmeticOp::MATRIX_LINALG_MUL, "matrix_linalg_mul"sv),
    wire_op(ArithmeticOp::MATRIX_DETERMINANT, "matrix_determinant"sv),
    wire_op(ArithmeticOp::MATRIX_TRANSPOSE, "matrix_transpose"sv),
    wire_op(ArithmeticOp::MATRIX_INVERSE, "matrix_inverse"sv),
    wire_op(ArithmeticOp::AGGREGATE, "aggregate"sv),
    wire_op(ArithmeticOp::SHUFFLE, "shuffle"sv),
    wire_op(ArithmeticOp::INSERT, "insert"sv),
    wire_op(ArithmeticOp::EXTRACT, "extract"sv)};

constexpr std::array atomic_wire_ops{
    wire_op(AtomicOp::EXCHANGE, "exchange"sv),
    wire_op(AtomicOp::COMPARE_EXCHANGE, "compare_exchange"sv),
    wire_op(AtomicOp::FETCH_ADD, "fetch_add"sv),
    wire_op(AtomicOp::FETCH_SUB, "fetch_sub"sv),
    wire_op(AtomicOp::FETCH_AND, "fetch_and"sv),
    wire_op(AtomicOp::FETCH_OR, "fetch_or"sv),
    wire_op(AtomicOp::FETCH_XOR, "fetch_xor"sv),
    wire_op(AtomicOp::FETCH_MIN, "fetch_min"sv),
    wire_op(AtomicOp::FETCH_MAX, "fetch_max"sv)};

constexpr std::array autodiff_intrinsic_wire_ops{
    wire_op(AutodiffIntrinsicOp::AUTODIFF_REQUIRES_GRADIENT, "autodiff_requires_gradient"sv),
    wire_op(AutodiffIntrinsicOp::AUTODIFF_GRADIENT, "autodiff_gradient"sv),
    wire_op(AutodiffIntrinsicOp::AUTODIFF_GRADIENT_MARKER, "autodiff_gradient_marker"sv),
    wire_op(AutodiffIntrinsicOp::AUTODIFF_ACCUMULATE_GRADIENT, "autodiff_accumulate_gradient"sv),
    wire_op(AutodiffIntrinsicOp::AUTODIFF_BACKWARD, "autodiff_backward"sv),
    wire_op(AutodiffIntrinsicOp::AUTODIFF_DETACH, "autodiff_detach"sv),
    wire_op(AutodiffIntrinsicOp::AUTODIFF_PROPAGATE_GRADIENT, "autodiff_propagate_gradient"sv),
    wire_op(AutodiffIntrinsicOp::AUTODIFF_OUTPUT_GRADIENT, "autodiff_output_gradient"sv)};

constexpr std::array cast_wire_ops{
    wire_op(CastOp::STATIC_CAST, "static_cast"sv),
    wire_op(CastOp::BITWISE_CAST, "bitwise_cast"sv)};

constexpr std::array ray_query_object_read_wire_ops{
    wire_op(RayQueryObjectReadOp::RAY_QUERY_OBJECT_WORLD_SPACE_RAY, "ray_query_object_world_space_ray"sv),
    wire_op(RayQueryObjectReadOp::RAY_QUERY_OBJECT_PROCEDURAL_CANDIDATE_HIT, "ray_query_object_procedural_candidate_hit"sv),
    wire_op(RayQueryObjectReadOp::RAY_QUERY_OBJECT_TRIANGLE_CANDIDATE_HIT, "ray_query_object_triangle_candidate_hit"sv),
    wire_op(RayQueryObjectReadOp::RAY_QUERY_OBJECT_COMMITTED_HIT, "ray_query_object_committed_hit"sv),
    wire_op(RayQueryObjectReadOp::RAY_QUERY_OBJECT_IS_TRIANGLE_CANDIDATE, "ray_query_object_is_triangle_candidate"sv),
    wire_op(RayQueryObjectReadOp::RAY_QUERY_OBJECT_IS_PROCEDURAL_CANDIDATE, "ray_query_object_is_procedural_candidate"sv),
    wire_op(RayQueryObjectReadOp::RAY_QUERY_OBJECT_IS_TERMINATED, "ray_query_object_is_terminated"sv)};

constexpr std::array ray_query_object_write_wire_ops{
    wire_op(RayQueryObjectWriteOp::RAY_QUERY_OBJECT_COMMIT_TRIANGLE, "ray_query_object_commit_triangle"sv),
    wire_op(RayQueryObjectWriteOp::RAY_QUERY_OBJECT_COMMIT_PROCEDURAL, "ray_query_object_commit_procedural"sv),
    wire_op(RayQueryObjectWriteOp::RAY_QUERY_OBJECT_TERMINATE, "ray_query_object_terminate"sv),
    wire_op(RayQueryObjectWriteOp::RAY_QUERY_OBJECT_PROCEED, "ray_query_object_proceed"sv)};

constexpr std::array resource_query_wire_ops{
    wire_op(ResourceQueryOp::BUFFER_SIZE, "buffer_size"sv),
    wire_op(ResourceQueryOp::BYTE_BUFFER_SIZE, "byte_buffer_size"sv),
    wire_op(ResourceQueryOp::TEXTURE2D_SIZE, "texture2d_size"sv),
    wire_op(ResourceQueryOp::TEXTURE3D_SIZE, "texture3d_size"sv),
    wire_op(ResourceQueryOp::BINDLESS_BUFFER_SIZE, "bindless_buffer_size"sv),
    wire_op(ResourceQueryOp::BINDLESS_BYTE_BUFFER_SIZE, "bindless_byte_buffer_size"sv),
    wire_op(ResourceQueryOp::BINDLESS_TEXTURE2D_SIZE, "bindless_texture2d_size"sv),
    wire_op(ResourceQueryOp::BINDLESS_TEXTURE3D_SIZE, "bindless_texture3d_size"sv),
    wire_op(ResourceQueryOp::BINDLESS_TEXTURE2D_SIZE_LEVEL, "bindless_texture2d_size_level"sv),
    wire_op(ResourceQueryOp::BINDLESS_TEXTURE3D_SIZE_LEVEL, "bindless_texture3d_size_level"sv),
    wire_op(ResourceQueryOp::TEXTURE2D_SAMPLE, "texture2d_sample"sv),
    wire_op(ResourceQueryOp::TEXTURE2D_SAMPLE_LEVEL, "texture2d_sample_level"sv),
    wire_op(ResourceQueryOp::TEXTURE2D_SAMPLE_GRAD, "texture2d_sample_grad"sv),
    wire_op(ResourceQueryOp::TEXTURE2D_SAMPLE_GRAD_LEVEL, "texture2d_sample_grad_level"sv),
    wire_op(ResourceQueryOp::TEXTURE3D_SAMPLE, "texture3d_sample"sv),
    wire_op(ResourceQueryOp::TEXTURE3D_SAMPLE_LEVEL, "texture3d_sample_level"sv),
    wire_op(ResourceQueryOp::TEXTURE3D_SAMPLE_GRAD, "texture3d_sample_grad"sv),
    wire_op(ResourceQueryOp::TEXTURE3D_SAMPLE_GRAD_LEVEL, "texture3d_sample_grad_level"sv),
    wire_op(ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE, "bindless_texture2d_sample"sv),
    wire_op(ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_LEVEL, "bindless_texture2d_sample_level"sv),
    wire_op(ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_GRAD, "bindless_texture2d_sample_grad"sv),
    wire_op(ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_GRAD_LEVEL, "bindless_texture2d_sample_grad_level"sv),
    wire_op(ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE, "bindless_texture3d_sample"sv),
    wire_op(ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE_LEVEL, "bindless_texture3d_sample_level"sv),
    wire_op(ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE_GRAD, "bindless_texture3d_sample_grad"sv),
    wire_op(ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE_GRAD_LEVEL, "bindless_texture3d_sample_grad_level"sv),
    wire_op(ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_SAMPLER, "bindless_texture2d_sample_sampler"sv),
    wire_op(ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_LEVEL_SAMPLER, "bindless_texture2d_sample_level_sampler"sv),
    wire_op(ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_GRAD_SAMPLER, "bindless_texture2d_sample_grad_sampler"sv),
    wire_op(ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_GRAD_LEVEL_SAMPLER, "bindless_texture2d_sample_grad_level_sampler"sv),
    wire_op(ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE_SAMPLER, "bindless_texture3d_sample_sampler"sv),
    wire_op(ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE_LEVEL_SAMPLER, "bindless_texture3d_sample_level_sampler"sv),
    wire_op(ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE_GRAD_SAMPLER, "bindless_texture3d_sample_grad_sampler"sv),
    wire_op(ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE_GRAD_LEVEL_SAMPLER, "bindless_texture3d_sample_grad_level_sampler"sv),
    wire_op(ResourceQueryOp::BUFFER_DEVICE_ADDRESS, "buffer_device_address"sv),
    wire_op(ResourceQueryOp::BINDLESS_BUFFER_DEVICE_ADDRESS, "bindless_buffer_device_address"sv),
    wire_op(ResourceQueryOp::RAY_TRACING_INSTANCE_TRANSFORM, "ray_tracing_instance_transform"sv),
    wire_op(ResourceQueryOp::RAY_TRACING_INSTANCE_USER_ID, "ray_tracing_instance_user_id"sv),
    wire_op(ResourceQueryOp::RAY_TRACING_INSTANCE_VISIBILITY_MASK, "ray_tracing_instance_visibility_mask"sv),
    wire_op(ResourceQueryOp::RAY_TRACING_TRACE_CLOSEST, "ray_tracing_trace_closest"sv),
    wire_op(ResourceQueryOp::RAY_TRACING_TRACE_ANY, "ray_tracing_trace_any"sv),
    wire_op(ResourceQueryOp::RAY_TRACING_QUERY_ALL, "ray_tracing_query_all"sv),
    wire_op(ResourceQueryOp::RAY_TRACING_QUERY_ANY, "ray_tracing_query_any"sv),
    wire_op(ResourceQueryOp::RAY_TRACING_INSTANCE_MOTION_MATRIX, "ray_tracing_instance_motion_matrix"sv),
    wire_op(ResourceQueryOp::RAY_TRACING_INSTANCE_MOTION_SRT, "ray_tracing_instance_motion_srt"sv),
    wire_op(ResourceQueryOp::RAY_TRACING_TRACE_CLOSEST_MOTION_BLUR, "ray_tracing_trace_closest_motion_blur"sv),
    wire_op(ResourceQueryOp::RAY_TRACING_TRACE_ANY_MOTION_BLUR, "ray_tracing_trace_any_motion_blur"sv),
    wire_op(ResourceQueryOp::RAY_TRACING_QUERY_ALL_MOTION_BLUR, "ray_tracing_query_all_motion_blur"sv),
    wire_op(ResourceQueryOp::RAY_TRACING_QUERY_ANY_MOTION_BLUR, "ray_tracing_query_any_motion_blur"sv)};

constexpr std::array resource_read_wire_ops{
    wire_op(ResourceReadOp::BUFFER_READ, "buffer_read"sv),
    wire_op(ResourceReadOp::BUFFER_VOLATILE_READ, "buffer_volatile_read"sv),
    wire_op(ResourceReadOp::BYTE_BUFFER_READ, "byte_buffer_read"sv),
    wire_op(ResourceReadOp::BYTE_BUFFER_VOLATILE_READ, "byte_buffer_volatile_read"sv),
    wire_op(ResourceReadOp::TEXTURE2D_READ, "texture2d_read"sv),
    wire_op(ResourceReadOp::TEXTURE3D_READ, "texture3d_read"sv),
    wire_op(ResourceReadOp::BINDLESS_BUFFER_READ, "bindless_buffer_read"sv),
    wire_op(ResourceReadOp::BINDLESS_BYTE_BUFFER_READ, "bindless_byte_buffer_read"sv),
    wire_op(ResourceReadOp::BINDLESS_TEXTURE2D_READ, "bindless_texture2d_read"sv),
    wire_op(ResourceReadOp::BINDLESS_TEXTURE3D_READ, "bindless_texture3d_read"sv),
    wire_op(ResourceReadOp::BINDLESS_TEXTURE2D_READ_LEVEL, "bindless_texture2d_read_level"sv),
    wire_op(ResourceReadOp::BINDLESS_TEXTURE3D_READ_LEVEL, "bindless_texture3d_read_level"sv),
    wire_op(ResourceReadOp::DEVICE_ADDRESS_READ, "device_address_read"sv)};

constexpr std::array resource_write_wire_ops{
    wire_op(ResourceWriteOp::BUFFER_WRITE, "buffer_write"sv),
    wire_op(ResourceWriteOp::BUFFER_VOLATILE_WRITE, "buffer_volatile_write"sv),
    wire_op(ResourceWriteOp::BYTE_BUFFER_WRITE, "byte_buffer_write"sv),
    wire_op(ResourceWriteOp::BYTE_BUFFER_VOLATILE_WRITE, "byte_buffer_volatile_write"sv),
    wire_op(ResourceWriteOp::TEXTURE2D_WRITE, "texture2d_write"sv),
    wire_op(ResourceWriteOp::TEXTURE3D_WRITE, "texture3d_write"sv),
    wire_op(ResourceWriteOp::BINDLESS_BUFFER_WRITE, "bindless_buffer_write"sv),
    wire_op(ResourceWriteOp::BINDLESS_BYTE_BUFFER_WRITE, "bindless_byte_buffer_write"sv),
    wire_op(ResourceWriteOp::DEVICE_ADDRESS_WRITE, "device_address_write"sv),
    wire_op(ResourceWriteOp::RAY_TRACING_SET_INSTANCE_TRANSFORM, "ray_tracing_set_instance_transform"sv),
    wire_op(ResourceWriteOp::RAY_TRACING_SET_INSTANCE_VISIBILITY_MASK, "ray_tracing_set_instance_visibility_mask"sv),
    wire_op(ResourceWriteOp::RAY_TRACING_SET_INSTANCE_OPACITY, "ray_tracing_set_instance_opacity"sv),
    wire_op(ResourceWriteOp::RAY_TRACING_SET_INSTANCE_USER_ID, "ray_tracing_set_instance_user_id"sv),
    wire_op(ResourceWriteOp::RAY_TRACING_SET_INSTANCE_MOTION_MATRIX, "ray_tracing_set_instance_motion_matrix"sv),
    wire_op(ResourceWriteOp::RAY_TRACING_SET_INSTANCE_MOTION_SRT, "ray_tracing_set_instance_motion_srt"sv),
    wire_op(ResourceWriteOp::INDIRECT_DISPATCH_SET_KERNEL, "indirect_dispatch_set_kernel"sv),
    wire_op(ResourceWriteOp::INDIRECT_DISPATCH_SET_COUNT, "indirect_dispatch_set_count"sv)};

constexpr std::array thread_group_wire_ops{
    wire_op(ThreadGroupOp::SHADER_EXECUTION_REORDER, "shader_execution_reorder"sv),
    wire_op(ThreadGroupOp::RASTER_QUAD_DDX, "raster_quad_ddx"sv),
    wire_op(ThreadGroupOp::RASTER_QUAD_DDY, "raster_quad_ddy"sv),
    wire_op(ThreadGroupOp::WARP_IS_FIRST_ACTIVE_LANE, "warp_is_first_active_lane"sv),
    wire_op(ThreadGroupOp::WARP_FIRST_ACTIVE_LANE, "warp_first_active_lane"sv),
    wire_op(ThreadGroupOp::WARP_ACTIVE_ALL_EQUAL, "warp_active_all_equal"sv),
    wire_op(ThreadGroupOp::WARP_ACTIVE_BIT_AND, "warp_active_bit_and"sv),
    wire_op(ThreadGroupOp::WARP_ACTIVE_BIT_OR, "warp_active_bit_or"sv),
    wire_op(ThreadGroupOp::WARP_ACTIVE_BIT_XOR, "warp_active_bit_xor"sv),
    wire_op(ThreadGroupOp::WARP_ACTIVE_COUNT_BITS, "warp_active_count_bits"sv),
    wire_op(ThreadGroupOp::WARP_ACTIVE_MAX, "warp_active_max"sv),
    wire_op(ThreadGroupOp::WARP_ACTIVE_MIN, "warp_active_min"sv),
    wire_op(ThreadGroupOp::WARP_ACTIVE_PRODUCT, "warp_active_product"sv),
    wire_op(ThreadGroupOp::WARP_ACTIVE_SUM, "warp_active_sum"sv),
    wire_op(ThreadGroupOp::WARP_ACTIVE_ALL, "warp_active_all"sv),
    wire_op(ThreadGroupOp::WARP_ACTIVE_ANY, "warp_active_any"sv),
    wire_op(ThreadGroupOp::WARP_ACTIVE_BIT_MASK, "warp_active_bit_mask"sv),
    wire_op(ThreadGroupOp::WARP_PREFIX_COUNT_BITS, "warp_prefix_count_bits"sv),
    wire_op(ThreadGroupOp::WARP_PREFIX_SUM, "warp_prefix_sum"sv),
    wire_op(ThreadGroupOp::WARP_PREFIX_PRODUCT, "warp_prefix_product"sv),
    wire_op(ThreadGroupOp::WARP_READ_LANE, "warp_read_lane"sv),
    wire_op(ThreadGroupOp::WARP_READ_FIRST_ACTIVE_LANE, "warp_read_first_active_lane"sv),
    wire_op(ThreadGroupOp::SYNCHRONIZE_BLOCK, "synchronize_block"sv)};

template<typename Op, size_t N>
[[nodiscard]] luisa::optional<int64_t>
parse_symbolic_op(luisa::string_view token, const std::array<WireOpToken<Op>, N> &table) noexcept {
    for (auto entry : table) {
        if (entry.name == token) { return static_cast<int64_t>(entry.op); }
    }
    return luisa::nullopt;
}

template<typename Op, size_t N>
[[nodiscard]] luisa::optional<luisa::string_view>
symbolic_op_name(int64_t op, const std::array<WireOpToken<Op>, N> &table) noexcept {
    for (auto entry : table) {
        if (static_cast<int64_t>(entry.op) == op) { return entry.name; }
    }
    return luisa::nullopt;
}

template<typename Op, size_t N>
[[nodiscard]] bool frozen_zero_based_layout(const std::array<WireOpToken<Op>, N> &table) noexcept {
    for (auto i = 0u; i < table.size(); i++) {
        if (static_cast<size_t>(table[i].op) != i) { return false; }
    }
    return true;
}

[[nodiscard]] luisa::optional<int64_t>
parse_instruction_op(DerivedInstructionTag tag, luisa::string_view token) noexcept {
    if (auto numeric = parse_integer_token(token)) {
        switch (tag) {
            case DerivedInstructionTag::ALLOCA:
                if (*numeric == 0) { return static_cast<int64_t>(AllocaOp::LOCAL); }
                if (*numeric == 1) { return static_cast<int64_t>(AllocaOp::SHARED); }
                return luisa::nullopt;
            case DerivedInstructionTag::ARITHMETIC:
                if (frozen_zero_based_layout(arithmetic_wire_ops) &&
                    *numeric >= 0 && static_cast<uint64_t>(*numeric) < arithmetic_wire_ops.size()) {
                    return static_cast<int64_t>(arithmetic_wire_ops[static_cast<size_t>(*numeric)].op);
                }
                return luisa::nullopt;
            case DerivedInstructionTag::CAST:
                if (*numeric == 0) { return static_cast<int64_t>(CastOp::STATIC_CAST); }
                if (*numeric == 1) { return static_cast<int64_t>(CastOp::BITWISE_CAST); }
                return luisa::nullopt;
            default: return *numeric == -1 ? numeric : luisa::nullopt;
        }
    }
    switch (tag) {
        case DerivedInstructionTag::ALLOCA:
            return parse_symbolic_op(token, alloca_wire_ops);
        case DerivedInstructionTag::ARITHMETIC:
            return parse_symbolic_op(token, arithmetic_wire_ops);
        case DerivedInstructionTag::CAST:
            return parse_symbolic_op(token, cast_wire_ops);
        case DerivedInstructionTag::ATOMIC:
            return parse_symbolic_op(token, atomic_wire_ops);
        case DerivedInstructionTag::THREAD_GROUP:
            return parse_symbolic_op(token, thread_group_wire_ops);
        case DerivedInstructionTag::RESOURCE_QUERY:
            return parse_symbolic_op(token, resource_query_wire_ops);
        case DerivedInstructionTag::RESOURCE_READ:
            return parse_symbolic_op(token, resource_read_wire_ops);
        case DerivedInstructionTag::RESOURCE_WRITE:
            return parse_symbolic_op(token, resource_write_wire_ops);
        case DerivedInstructionTag::RAY_QUERY_OBJECT_READ:
            return parse_symbolic_op(token, ray_query_object_read_wire_ops);
        case DerivedInstructionTag::RAY_QUERY_OBJECT_WRITE:
            return parse_symbolic_op(token, ray_query_object_write_wire_ops);
        case DerivedInstructionTag::AUTODIFF_INTRINSIC:
            return parse_symbolic_op(token, autodiff_intrinsic_wire_ops);
        case DerivedInstructionTag::DEBUG_BREAK:
            return token == "null_callback" ? luisa::optional<int64_t>{0} : luisa::nullopt;
        default: return luisa::nullopt;
    }
}

[[nodiscard]] luisa::optional<luisa::string_view>
instruction_op_name(DerivedInstructionTag tag, int64_t op) noexcept {
    using namespace std::string_view_literals;
    switch (tag) {
        case DerivedInstructionTag::ALLOCA:
            return symbolic_op_name(op, alloca_wire_ops);
        case DerivedInstructionTag::ARITHMETIC:
            return symbolic_op_name(op, arithmetic_wire_ops);
        case DerivedInstructionTag::CAST:
            return symbolic_op_name(op, cast_wire_ops);
        case DerivedInstructionTag::ATOMIC:
            return symbolic_op_name(op, atomic_wire_ops);
        case DerivedInstructionTag::THREAD_GROUP:
            return symbolic_op_name(op, thread_group_wire_ops);
        case DerivedInstructionTag::RESOURCE_QUERY:
            return symbolic_op_name(op, resource_query_wire_ops);
        case DerivedInstructionTag::RESOURCE_READ:
            return symbolic_op_name(op, resource_read_wire_ops);
        case DerivedInstructionTag::RESOURCE_WRITE:
            return symbolic_op_name(op, resource_write_wire_ops);
        case DerivedInstructionTag::RAY_QUERY_OBJECT_READ:
            return symbolic_op_name(op, ray_query_object_read_wire_ops);
        case DerivedInstructionTag::RAY_QUERY_OBJECT_WRITE:
            return symbolic_op_name(op, ray_query_object_write_wire_ops);
        case DerivedInstructionTag::AUTODIFF_INTRINSIC:
            return symbolic_op_name(op, autodiff_intrinsic_wire_ops);
        case DerivedInstructionTag::DEBUG_BREAK:
            return op == 0 ? luisa::optional<luisa::string_view>{"null_callback"sv} : luisa::nullopt;
        default: return op == -1 ? luisa::optional<luisa::string_view>{"-1"sv} : luisa::nullopt;
    }
}

[[nodiscard]] luisa::optional<DerivedSpecialRegisterTag>
parse_special_register_name(luisa::string_view name) noexcept {
    if (name == "thread_id") { return DerivedSpecialRegisterTag::THREAD_ID; }
    if (name == "block_id") { return DerivedSpecialRegisterTag::BLOCK_ID; }
    if (name == "warp_lane_id") { return DerivedSpecialRegisterTag::WARP_LANE_ID; }
    if (name == "dispatch_id") { return DerivedSpecialRegisterTag::DISPATCH_ID; }
    if (name == "kernel_id") { return DerivedSpecialRegisterTag::KERNEL_ID; }
    if (name == "object_id") { return DerivedSpecialRegisterTag::RASTER_OBJECT_ID; }
    if (name == "barycentrics") { return DerivedSpecialRegisterTag::RASTER_BARYCENTRICS; }
    if (name == "block_size") { return DerivedSpecialRegisterTag::BLOCK_SIZE; }
    if (name == "warp_size") { return DerivedSpecialRegisterTag::WARP_SIZE; }
    if (name == "dispatch_size") { return DerivedSpecialRegisterTag::DISPATCH_SIZE; }
    return luisa::nullopt;
}

struct MetadataRecord {
    enum struct Kind { NAME,
                       LOCATION,
                       COMMENT,
                       CURVE_BASIS,
                       SIGNATURE_CONSTRAINT,
                       REG2MEM_SPILL } kind;
    luisa::string text;
    int64_t number{0};
};

constexpr uint64_t reg2mem_spill_phi_wire_kind = 0u;
constexpr uint64_t reg2mem_spill_cross_block_wire_kind = 1u;

[[nodiscard]] constexpr luisa::optional<uint64_t>
encode_reg2mem_spill_wire_kind(Reg2MemSpillKind kind) noexcept {
    switch (kind) {
        case Reg2MemSpillKind::PHI:
            return reg2mem_spill_phi_wire_kind;
        case Reg2MemSpillKind::CROSS_BLOCK:
            return reg2mem_spill_cross_block_wire_kind;
    }
    return luisa::nullopt;
}

[[nodiscard]] constexpr luisa::optional<Reg2MemSpillKind>
decode_reg2mem_spill_wire_kind(uint64_t kind) noexcept {
    switch (kind) {
        case reg2mem_spill_phi_wire_kind:
            return Reg2MemSpillKind::PHI;
        case reg2mem_spill_cross_block_wire_kind:
            return Reg2MemSpillKind::CROSS_BLOCK;
        default:
            return luisa::nullopt;
    }
}

[[nodiscard]] bool valid_metadata_name(luisa::string_view name) noexcept {
    if (name.empty()) { return true; }
    auto valid_head = [](char c) noexcept {
        return std::isalpha(static_cast<unsigned char>(c)) || c == '_';
    };
    auto valid_tail = [](char c) noexcept {
        return std::isalnum(static_cast<unsigned char>(c)) || c == '_';
    };
    return valid_head(name.front()) &&
           std::all_of(name.begin() + 1, name.end(), valid_tail);
}

[[nodiscard]] bool parse_metadata_records(
    TextParser &parser, ParseBudget &budget,
    luisa::vector<MetadataRecord> &records) noexcept {
    if (!parser.try_keyword("metadata")) { return true; }
    size_t count = 0u;
    if (!parser.count(count) || !budget.consume(parser, count)) { return false; }
    records.reserve(count);
    constexpr auto valid_curve_basis_mask = (uint64_t{1u} << curve_basis_count) - 1u;
    for (auto i = 0u; i < count; i++) {
        MetadataRecord record{};
        luisa::string kind;
        if (!parser.keyword("md") || !parser.word(kind)) { return false; }
        if (kind == "name") {
            record.kind = MetadataRecord::Kind::NAME;
            if (!parser.string(record.text) || !valid_metadata_name(record.text)) {
                return parser.fail("Invalid XIR name metadata.");
            }
        } else if (kind == "location") {
            record.kind = MetadataRecord::Kind::LOCATION;
            if (!parser.string(record.text) || !parser.signed_integer(record.number)) { return false; }
            if (record.number < std::numeric_limits<int>::min() ||
                record.number > std::numeric_limits<int>::max()) {
                return parser.fail("XIR location line is out of range.");
            }
        } else if (kind == "comment") {
            record.kind = MetadataRecord::Kind::COMMENT;
            if (!parser.string(record.text)) { return false; }
        } else if (kind == "curve_basis") {
            record.kind = MetadataRecord::Kind::CURVE_BASIS;
            uint64_t mask = 0u;
            if (!parser.unsigned_integer(mask)) { return false; }
            if ((mask & ~valid_curve_basis_mask) != 0u) {
                return parser.fail("XIR curve-basis metadata contains unknown bits.");
            }
            record.number = static_cast<int64_t>(mask);
        } else if (kind == "signature_constraint") {
            record.kind = MetadataRecord::Kind::SIGNATURE_CONSTRAINT;
        } else if (kind == "reg2mem_spill") {
            record.kind = MetadataRecord::Kind::REG2MEM_SPILL;
            luisa::string spill_kind;
            if (!parser.word(spill_kind)) { return false; }
            if (spill_kind == "phi") {
                record.number = static_cast<int64_t>(Reg2MemSpillKind::PHI);
            } else if (spill_kind == "cross_block") {
                record.number = static_cast<int64_t>(Reg2MemSpillKind::CROSS_BLOCK);
            } else {
                return parser.fail("Unknown XIR reg2mem-spill metadata kind.");
            }
        } else {
            return parser.fail("Unknown XIR metadata kind.");
        }
        records.emplace_back(std::move(record));
    }
    return true;
}

void apply_metadata_records(
    const luisa::vector<MetadataRecord> &records,
    MetadataListMixin &owner) noexcept {
    for (auto iter = records.rbegin(); iter != records.rend(); ++iter) {
        ManagedPtr<Metadata> metadata;
        switch (iter->kind) {
            case MetadataRecord::Kind::NAME:
                metadata = luisa::make_managed<NameMD>(iter->text);
                break;
            case MetadataRecord::Kind::LOCATION:
                metadata = luisa::make_managed<LocationMD>(luisa::filesystem::path{iter->text}, static_cast<int>(iter->number));
                break;
            case MetadataRecord::Kind::COMMENT:
                metadata = luisa::make_managed<CommentMD>(iter->text);
                break;
            case MetadataRecord::Kind::CURVE_BASIS:
                metadata = luisa::make_managed<CurveBasisMD>(CurveBasisSet::from_u64(static_cast<uint64_t>(iter->number)));
                break;
            case MetadataRecord::Kind::SIGNATURE_CONSTRAINT:
                metadata = luisa::make_managed<SignatureConstraintMD>();
                break;
            case MetadataRecord::Kind::REG2MEM_SPILL:
                metadata = luisa::make_managed<Reg2MemSpillMD>(
                    static_cast<Reg2MemSpillKind>(iter->number));
                break;
        }
        owner.metadata_list().push_front(std::move(metadata));
    }
}

[[nodiscard]] bool append_metadata_records(
    luisa::string &text, const MetadataListMixin &owner,
    luisa::string_view indentation, luisa::string &error) noexcept {
    auto count = owner.metadata_list().count_size();
    if (count > max_record_count) {
        error = "XIR metadata count exceeds the supported limit.";
        return false;
    }
    text.append(indentation);
    luisa::format_to(std::back_inserter(text), "metadata {}\n", count);
    for (auto metadata : owner.metadata_list()) {
        text.append(indentation);
        text.append("md ");
        switch (metadata->derived_metadata_tag()) {
            case DerivedMetadataTag::NAME: {
                auto value = static_cast<const NameMD *>(metadata)->name();
                if (!valid_metadata_name(value)) {
                    error = "XIR name metadata is invalid.";
                    return false;
                }
                text.append("name ");
                append_quoted(text, value);
                break;
            }
            case DerivedMetadataTag::LOCATION: {
                auto value = static_cast<const LocationMD *>(metadata);
                text.append("location ");
                append_quoted(text, value->file().string());
                luisa::format_to(std::back_inserter(text), " {}", value->line());
                break;
            }
            case DerivedMetadataTag::COMMENT: {
                text.append("comment ");
                append_quoted(text, static_cast<const CommentMD *>(metadata)->comment());
                break;
            }
            case DerivedMetadataTag::CURVE_BASIS: {
                auto mask = static_cast<const CurveBasisMD *>(metadata)->curve_basis_set().to_u64();
                text.append("curve_basis ");
                luisa::format_to(std::back_inserter(text), "{}", mask);
                break;
            }
            case DerivedMetadataTag::SIGNATURE_CONSTRAINT:
                text.append("signature_constraint");
                break;
            case DerivedMetadataTag::REG2MEM_SPILL: {
                auto kind = static_cast<const Reg2MemSpillMD *>(metadata)->kind();
                text.append("reg2mem_spill ");
                switch (kind) {
                    case Reg2MemSpillKind::PHI:
                        text.append("phi");
                        break;
                    case Reg2MemSpillKind::CROSS_BLOCK:
                        text.append("cross_block");
                        break;
                    default:
                        error = "XIR reg2mem-spill metadata has an unknown kind.";
                        return false;
                }
                break;
            }
            default:
                error = "XIR contains an unknown metadata kind.";
                return false;
        }
        text.push_back('\n');
    }
    return true;
}

struct GlobalRecord {
    enum struct Kind { CONSTANT,
                       UNDEFINED,
                       SPECIAL } kind;
    uint64_t id{0u};
    luisa::string type;
    luisa::string payload;
    luisa::vector<std::byte> binary_payload;
    bool payload_is_binary{false};
    luisa::vector<MetadataRecord> metadata;
};

struct ArgumentRecord {
    uint64_t id{0u};
    luisa::string kind;
    luisa::string type;
    luisa::vector<MetadataRecord> metadata;
};

struct BasicBlockRecord {
    uint64_t id{0u};
    luisa::vector<MetadataRecord> metadata;
};

struct InstructionRecord {
    uint64_t id{0u};
    uint64_t block_id{0u};
    DerivedInstructionTag tag{};
    luisa::string type;
    int64_t op{-1};
    luisa::vector<int64_t> operands;
    luisa::vector<int64_t> auxiliary;
    luisa::vector<luisa::string> payloads;
    luisa::vector<MetadataRecord> metadata;
};

struct FunctionRecord {
    uint64_t id{0u};
    luisa::string kind;
    luisa::string return_type;
    std::array<uint32_t, 3u> block_size{};
    luisa::vector<MetadataRecord> metadata;
    luisa::vector<ArgumentRecord> arguments;
    luisa::vector<BasicBlockRecord> blocks;
    int64_t body{-1};
    luisa::vector<InstructionRecord> instructions;
};

struct ModuleRecord {
    luisa::vector<MetadataRecord> metadata;
    luisa::vector<GlobalRecord> globals;
    luisa::vector<FunctionRecord> functions;
};

constexpr std::array binary_instruction_tags{
    DerivedInstructionTag::IF,
    DerivedInstructionTag::SWITCH,
    DerivedInstructionTag::LOOP,
    DerivedInstructionTag::SIMPLE_LOOP,
    DerivedInstructionTag::BRANCH,
    DerivedInstructionTag::CONDITIONAL_BRANCH,
    DerivedInstructionTag::UNREACHABLE,
    DerivedInstructionTag::RASTER_DISCARD,
    DerivedInstructionTag::CORO_SUSPEND,
    DerivedInstructionTag::CORO_RESUME,
    DerivedInstructionTag::CORO_TERMINATE,
    DerivedInstructionTag::RETURN,
    DerivedInstructionTag::PHI,
    DerivedInstructionTag::ALLOCA,
    DerivedInstructionTag::LOAD,
    DerivedInstructionTag::STORE,
    DerivedInstructionTag::GEP,
    DerivedInstructionTag::ATOMIC,
    DerivedInstructionTag::ARITHMETIC,
    DerivedInstructionTag::THREAD_GROUP,
    DerivedInstructionTag::RESOURCE_QUERY,
    DerivedInstructionTag::RESOURCE_READ,
    DerivedInstructionTag::RESOURCE_WRITE,
    DerivedInstructionTag::RAY_QUERY_LOOP,
    DerivedInstructionTag::RAY_QUERY_DISPATCH,
    DerivedInstructionTag::RAY_QUERY_OBJECT_READ,
    DerivedInstructionTag::RAY_QUERY_OBJECT_WRITE,
    DerivedInstructionTag::RAY_QUERY_PIPELINE,
    DerivedInstructionTag::AUTODIFF_SCOPE,
    DerivedInstructionTag::AUTODIFF_INTRINSIC,
    DerivedInstructionTag::BREAK,
    DerivedInstructionTag::CONTINUE,
    DerivedInstructionTag::CALL,
    DerivedInstructionTag::CAST,
    DerivedInstructionTag::PRINT,
    DerivedInstructionTag::CLOCK,
    DerivedInstructionTag::DEBUG_BREAK,
    DerivedInstructionTag::ASSERT,
    DerivedInstructionTag::ASSUME,
    DerivedInstructionTag::OUTLINE};

[[nodiscard]] luisa::optional<uint64_t>
binary_instruction_tag_id(DerivedInstructionTag tag) noexcept {
    for (auto i = 0u; i < binary_instruction_tags.size(); i++) {
        if (binary_instruction_tags[i] == tag) { return i; }
    }
    return luisa::nullopt;
}

[[nodiscard]] luisa::optional<DerivedInstructionTag>
binary_instruction_tag(uint64_t id) noexcept {
    if (id >= binary_instruction_tags.size()) { return luisa::nullopt; }
    return binary_instruction_tags[static_cast<size_t>(id)];
}

[[nodiscard]] bool collect_binary_strings(
    const ModuleRecord &record, luisa::vector<luisa::string> &strings,
    luisa::string &error) noexcept {
    auto add = [&](luisa::string_view value) noexcept {
        if (value.size() > max_payload_size) {
            error = "XIR binary string exceeds the supported size limit.";
            return false;
        }
        strings.emplace_back(value);
        return true;
    };
    auto add_metadata = [&](const luisa::vector<MetadataRecord> &metadata) noexcept {
        for (auto &&item : metadata) {
            switch (item.kind) {
                case MetadataRecord::Kind::NAME:
                case MetadataRecord::Kind::LOCATION:
                case MetadataRecord::Kind::COMMENT:
                    if (!add(item.text)) { return false; }
                    break;
                case MetadataRecord::Kind::CURVE_BASIS:
                case MetadataRecord::Kind::SIGNATURE_CONSTRAINT:
                case MetadataRecord::Kind::REG2MEM_SPILL:
                    break;
            }
        }
        return true;
    };
    if (!add_metadata(record.metadata)) { return false; }
    for (auto &&global : record.globals) {
        if (global.kind == GlobalRecord::Kind::SPECIAL) {
            if (!add(global.payload)) { return false; }
        } else if (!add(global.type)) {
            return false;
        }
        if (!add_metadata(global.metadata)) { return false; }
    }
    for (auto &&function : record.functions) {
        if (!add(function.kind) || !add(function.return_type) ||
            !add_metadata(function.metadata)) {
            return false;
        }
        for (auto &&argument : function.arguments) {
            if (!add(argument.kind) || !add(argument.type) ||
                !add_metadata(argument.metadata)) {
                return false;
            }
        }
        for (auto &&block : function.blocks) {
            if (!add_metadata(block.metadata)) { return false; }
        }
        for (auto &&instruction : function.instructions) {
            auto op = instruction_op_name(instruction.tag, instruction.op);
            if (!op) {
                error = "XIR binary encountered an unsupported instruction operation.";
                return false;
            }
            if (!add(instruction.type) || !add(*op) ||
                !add_metadata(instruction.metadata)) {
                return false;
            }
            for (auto &&payload : instruction.payloads) {
                if (!add(payload)) { return false; }
            }
        }
    }
    std::sort(strings.begin(), strings.end());
    strings.erase(std::unique(strings.begin(), strings.end()), strings.end());
    auto total_size = size_t{0u};
    for (auto &&string : strings) {
        if (string.size() > max_payload_size - total_size) {
            error = "XIR binary string table exceeds the supported size limit.";
            return false;
        }
        total_size += string.size();
    }
    return strings.size() <= max_record_count;
}

class BinaryRecordWriter {
private:
    luisa::vector<std::byte> &_bytes;
    const luisa::vector<luisa::string> &_strings;
    luisa::string &_error;
    size_t _records_remaining{max_record_count};

private:
    [[nodiscard]] bool _consume(size_t count) noexcept {
        if (count > _records_remaining) {
            _error = "Cumulative XIR record count exceeds the supported binary output budget.";
            return false;
        }
        _records_remaining -= count;
        return true;
    }

public:
    BinaryRecordWriter(luisa::vector<std::byte> &bytes,
                       const luisa::vector<luisa::string> &strings,
                       luisa::string &error) noexcept
        : _bytes{bytes}, _strings{strings}, _error{error} {}

    void integer(uint64_t value) noexcept { append_uleb(_bytes, value); }
    void signed_integer(int64_t value) noexcept { append_zigzag(_bytes, value); }

    [[nodiscard]] bool count(size_t value) noexcept {
        if (value > max_record_count || !_consume(value)) { return false; }
        integer(value);
        return true;
    }

    [[nodiscard]] bool string(luisa::string_view value) noexcept {
        auto iter = std::lower_bound(
            _strings.begin(), _strings.end(), value,
            [](const luisa::string &lhs, luisa::string_view rhs) noexcept {
                return luisa::string_view{lhs} < rhs;
            });
        if (iter == _strings.end() || luisa::string_view{*iter} != value) {
            _error = "XIR binary string is missing from the canonical string table.";
            return false;
        }
        integer(static_cast<uint64_t>(iter - _strings.begin()));
        return true;
    }

    [[nodiscard]] bool metadata(const luisa::vector<MetadataRecord> &records) noexcept {
        if (!count(records.size())) { return false; }
        for (auto &&record : records) {
            switch (record.kind) {
                case MetadataRecord::Kind::NAME:
                    integer(0u);
                    if (!string(record.text)) { return false; }
                    break;
                case MetadataRecord::Kind::LOCATION:
                    integer(1u);
                    if (!string(record.text)) { return false; }
                    signed_integer(record.number);
                    break;
                case MetadataRecord::Kind::COMMENT:
                    integer(2u);
                    if (!string(record.text)) { return false; }
                    break;
                case MetadataRecord::Kind::CURVE_BASIS:
                    integer(3u);
                    integer(static_cast<uint64_t>(record.number));
                    break;
                case MetadataRecord::Kind::SIGNATURE_CONSTRAINT:
                    integer(4u);
                    break;
                case MetadataRecord::Kind::REG2MEM_SPILL:
                    integer(5u);
                    if (auto kind = encode_reg2mem_spill_wire_kind(
                            static_cast<Reg2MemSpillKind>(record.number))) {
                        integer(*kind);
                    } else {
                        _error = "XIR binary reg2mem-spill metadata has an unknown kind.";
                        return false;
                    }
                    break;
            }
        }
        return true;
    }
};

[[nodiscard]] bool encode_binary_module_record(
    const ModuleRecord &record, luisa::vector<std::byte> &bytes,
    luisa::string &error) noexcept {
    luisa::vector<luisa::string> strings;
    if (!collect_binary_strings(record, strings, error)) {
        if (error.empty()) { error = "XIR binary string table has too many entries."; }
        return false;
    }
    BinaryRecordWriter writer{bytes, strings, error};
    if (!writer.count(strings.size())) { return false; }
    for (auto &&string : strings) {
        writer.integer(string.size());
        auto data = reinterpret_cast<const std::byte *>(string.data());
        bytes.insert(bytes.end(), data, data + string.size());
    }
    if (!writer.metadata(record.metadata) || !writer.count(record.globals.size())) { return false; }
    for (auto &&global : record.globals) {
        switch (global.kind) {
            case GlobalRecord::Kind::CONSTANT: {
                writer.integer(0u);
                writer.integer(global.id);
                if (!writer.string(global.type) || global.payload.size() % 2u != 0u) {
                    if (error.empty()) { error = "XIR constant has malformed hexadecimal data."; }
                    return false;
                }
                writer.integer(global.payload.size() / 2u);
                for (auto i = 0u; i < global.payload.size(); i += 2u) {
                    auto high = hex_value(global.payload[i]);
                    auto low = hex_value(global.payload[i + 1u]);
                    if (high < 0 || low < 0) {
                        error = "XIR constant has malformed hexadecimal data.";
                        return false;
                    }
                    bytes.emplace_back(static_cast<std::byte>((high << 4) | low));
                }
                break;
            }
            case GlobalRecord::Kind::UNDEFINED:
                writer.integer(1u);
                writer.integer(global.id);
                if (!writer.string(global.type)) { return false; }
                break;
            case GlobalRecord::Kind::SPECIAL:
                writer.integer(2u);
                writer.integer(global.id);
                if (!writer.string(global.payload)) { return false; }
                break;
        }
        if (!writer.metadata(global.metadata)) { return false; }
    }
    if (!writer.count(record.functions.size())) { return false; }
    for (auto &&function : record.functions) {
        writer.integer(function.id);
        if (!writer.string(function.kind) || !writer.string(function.return_type)) { return false; }
        for (auto component : function.block_size) { writer.integer(component); }
        if (!writer.metadata(function.metadata) || !writer.count(function.arguments.size())) { return false; }
        for (auto &&argument : function.arguments) {
            writer.integer(argument.id);
            if (!writer.string(argument.kind) || !writer.string(argument.type) ||
                !writer.metadata(argument.metadata)) {
                return false;
            }
        }
        if (!writer.count(function.blocks.size())) { return false; }
        for (auto &&block : function.blocks) {
            writer.integer(block.id);
            if (!writer.metadata(block.metadata)) { return false; }
        }
        writer.signed_integer(function.body);
        if (!writer.count(function.instructions.size())) { return false; }
        for (auto &&instruction : function.instructions) {
            auto tag = binary_instruction_tag_id(instruction.tag);
            if (!tag) {
                error = "XIR binary encountered an unsupported instruction tag.";
                return false;
            }
            writer.integer(instruction.id);
            writer.integer(instruction.block_id);
            writer.integer(*tag);
            if (!writer.string(instruction.type)) { return false; }
            auto op = instruction_op_name(instruction.tag, instruction.op);
            if (!op || !writer.string(*op)) {
                if (error.empty()) { error = "XIR binary encountered an unsupported instruction operation."; }
                return false;
            }
            if (!writer.count(instruction.operands.size())) { return false; }
            for (auto operand : instruction.operands) { writer.signed_integer(operand); }
            if (!writer.count(instruction.auxiliary.size())) { return false; }
            for (auto auxiliary : instruction.auxiliary) { writer.signed_integer(auxiliary); }
            if (!writer.count(instruction.payloads.size())) { return false; }
            for (auto &&payload : instruction.payloads) {
                if (!writer.string(payload)) { return false; }
            }
            if (!writer.metadata(instruction.metadata)) { return false; }
        }
    }
    if (bytes.size() > max_payload_size) {
        error = "XIR binary payload exceeds the supported size limit.";
        return false;
    }
    return true;
}

class BinaryRecordReader {
private:
    BinaryReader &_reader;
    BinaryParseBudget _records;
    luisa::vector<luisa::string> _strings;
    size_t _allocation_remaining{max_payload_size};

private:
    [[nodiscard]] bool _allocate(size_t count, size_t element_size) noexcept {
        if (element_size != 0u && count > _allocation_remaining / element_size) {
            return _reader.fail("XIR binary allocation budget exceeded.");
        }
        _allocation_remaining -= count * element_size;
        return true;
    }

public:
    explicit BinaryRecordReader(BinaryReader &reader) noexcept : _reader{reader} {}

    [[nodiscard]] bool integer(uint64_t &value) noexcept { return _reader.uleb(value); }
    [[nodiscard]] bool signed_integer(int64_t &value) noexcept { return _reader.zigzag(value); }

    template<typename T>
    [[nodiscard]] bool count(size_t &value) noexcept {
        return _reader.count(value) && _records.consume(_reader, value) &&
               _allocate(value, sizeof(T));
    }

    [[nodiscard]] bool strings() noexcept {
        size_t count_value = 0u;
        if (!count<luisa::string>(count_value)) { return false; }
        _strings.reserve(count_value);
        for (auto i = 0u; i < count_value; i++) {
            uint64_t size_u64 = 0u;
            if (!integer(size_u64) || size_u64 > max_payload_size ||
                size_u64 > std::numeric_limits<size_t>::max()) {
                if (size_u64 > max_payload_size) {
                    return _reader.fail("XIR binary string exceeds the supported size limit.");
                }
                return false;
            }
            auto size = static_cast<size_t>(size_u64);
            if (!_allocate(size, 1u)) { return false; }
            luisa::span<const std::byte> data;
            if (!_reader.bytes(size, data)) { return false; }
            luisa::string value;
            value.assign(reinterpret_cast<const char *>(data.data()), data.size());
            if (!_strings.empty() && !(luisa::string_view{_strings.back()} < luisa::string_view{value})) {
                return _reader.fail("XIR binary string table is not strictly sorted and unique.");
            }
            _strings.emplace_back(std::move(value));
        }
        return true;
    }

    [[nodiscard]] bool string(
        luisa::string &value, size_t max_size = max_payload_size) noexcept {
        uint64_t index = 0u;
        if (!integer(index)) { return false; }
        if (index >= _strings.size()) { return _reader.fail("XIR binary string index is out of range."); }
        auto &&source = _strings[static_cast<size_t>(index)];
        if (source.size() > max_size) { return _reader.fail("XIR binary string exceeds its field limit."); }
        if (!_allocate(source.size(), 1u)) { return false; }
        value = source;
        return true;
    }

    [[nodiscard]] bool raw_bytes(luisa::vector<std::byte> &value) noexcept {
        uint64_t size_u64 = 0u;
        if (!integer(size_u64)) { return false; }
        if (size_u64 > max_payload_size || size_u64 > std::numeric_limits<size_t>::max()) {
            return _reader.fail("XIR binary byte payload exceeds the supported size limit.");
        }
        auto size = static_cast<size_t>(size_u64);
        if (!_allocate(size, 1u)) { return false; }
        luisa::span<const std::byte> data;
        if (!_reader.bytes(size, data)) { return false; }
        value.assign(data.begin(), data.end());
        return true;
    }

    [[nodiscard]] bool metadata(luisa::vector<MetadataRecord> &records) noexcept {
        size_t count_value = 0u;
        if (!count<MetadataRecord>(count_value)) { return false; }
        records.reserve(count_value);
        constexpr auto valid_curve_basis_mask = (uint64_t{1u} << curve_basis_count) - 1u;
        for (auto i = 0u; i < count_value; i++) {
            uint64_t kind = 0u;
            MetadataRecord record{};
            if (!integer(kind)) { return false; }
            switch (kind) {
                case 0u:
                    record.kind = MetadataRecord::Kind::NAME;
                    if (!string(record.text) || !valid_metadata_name(record.text)) {
                        return _reader.fail("Invalid XIR binary name metadata.");
                    }
                    break;
                case 1u:
                    record.kind = MetadataRecord::Kind::LOCATION;
                    if (!string(record.text) || !signed_integer(record.number)) { return false; }
                    if (record.number < std::numeric_limits<int>::min() ||
                        record.number > std::numeric_limits<int>::max()) {
                        return _reader.fail("XIR binary location line is out of range.");
                    }
                    break;
                case 2u:
                    record.kind = MetadataRecord::Kind::COMMENT;
                    if (!string(record.text)) { return false; }
                    break;
                case 3u: {
                    record.kind = MetadataRecord::Kind::CURVE_BASIS;
                    uint64_t mask = 0u;
                    if (!integer(mask)) { return false; }
                    if ((mask & ~valid_curve_basis_mask) != 0u) {
                        return _reader.fail("XIR binary curve-basis metadata contains unknown bits.");
                    }
                    record.number = static_cast<int64_t>(mask);
                    break;
                }
                case 4u:
                    record.kind = MetadataRecord::Kind::SIGNATURE_CONSTRAINT;
                    break;
                case 5u: {
                    uint64_t spill_kind = 0u;
                    if (!integer(spill_kind)) { return false; }
                    auto kind = decode_reg2mem_spill_wire_kind(spill_kind);
                    if (!kind) {
                        return _reader.fail(
                            "Unknown XIR binary reg2mem-spill metadata kind.");
                    }
                    record.kind = MetadataRecord::Kind::REG2MEM_SPILL;
                    record.number = static_cast<int64_t>(*kind);
                    break;
                }
                default: return _reader.fail("Unknown XIR binary metadata kind.");
            }
            records.emplace_back(std::move(record));
        }
        return true;
    }
};

[[nodiscard]] bool decode_binary_module_record(
    BinaryReader &reader, ModuleRecord &record) noexcept {
    BinaryRecordReader decoder{reader};
    if (!decoder.strings() || !decoder.metadata(record.metadata)) { return false; }
    size_t global_count = 0u;
    if (!decoder.count<GlobalRecord>(global_count)) { return false; }
    record.globals.reserve(global_count);
    for (auto i = 0u; i < global_count; i++) {
        uint64_t kind = 0u;
        GlobalRecord global{};
        if (!decoder.integer(kind) || !decoder.integer(global.id)) { return false; }
        switch (kind) {
            case 0u:
                global.kind = GlobalRecord::Kind::CONSTANT;
                global.payload_is_binary = true;
                if (!decoder.string(global.type, max_type_description_size) ||
                    !decoder.raw_bytes(global.binary_payload)) {
                    return false;
                }
                break;
            case 1u:
                global.kind = GlobalRecord::Kind::UNDEFINED;
                if (!decoder.string(global.type, max_type_description_size)) { return false; }
                break;
            case 2u:
                global.kind = GlobalRecord::Kind::SPECIAL;
                if (!decoder.string(global.payload, max_string_payload_size)) { return false; }
                break;
            default: return reader.fail("Unknown XIR binary global record kind.");
        }
        if (!decoder.metadata(global.metadata)) { return false; }
        record.globals.emplace_back(std::move(global));
    }
    size_t function_count = 0u;
    if (!decoder.count<FunctionRecord>(function_count)) { return false; }
    record.functions.reserve(function_count);
    for (auto function_index = 0u; function_index < function_count; function_index++) {
        FunctionRecord function{};
        if (!decoder.integer(function.id) ||
            !decoder.string(function.kind, max_string_payload_size) ||
            !decoder.string(function.return_type, max_type_description_size)) {
            return false;
        }
        for (auto &component : function.block_size) {
            uint64_t value = 0u;
            if (!decoder.integer(value)) { return false; }
            if (value > std::numeric_limits<uint32_t>::max()) {
                return reader.fail("XIR binary function block-size component is out of range.");
            }
            component = static_cast<uint32_t>(value);
        }
        if (!decoder.metadata(function.metadata)) { return false; }
        size_t argument_count = 0u;
        if (!decoder.count<ArgumentRecord>(argument_count)) { return false; }
        function.arguments.reserve(argument_count);
        for (auto i = 0u; i < argument_count; i++) {
            ArgumentRecord argument{};
            if (!decoder.integer(argument.id) ||
                !decoder.string(argument.kind, max_string_payload_size) ||
                !decoder.string(argument.type, max_type_description_size) ||
                !decoder.metadata(argument.metadata)) {
                return false;
            }
            function.arguments.emplace_back(std::move(argument));
        }
        size_t block_count = 0u;
        if (!decoder.count<BasicBlockRecord>(block_count)) { return false; }
        function.blocks.reserve(block_count);
        for (auto i = 0u; i < block_count; i++) {
            BasicBlockRecord block{};
            if (!decoder.integer(block.id) || !decoder.metadata(block.metadata)) { return false; }
            function.blocks.emplace_back(std::move(block));
        }
        if (!decoder.signed_integer(function.body)) { return false; }
        if (function.body < -1) {
            return reader.fail("XIR binary body identifiers must be nonnegative or -1.");
        }
        size_t instruction_count = 0u;
        if (!decoder.count<InstructionRecord>(instruction_count)) { return false; }
        function.instructions.reserve(instruction_count);
        for (auto i = 0u; i < instruction_count; i++) {
            InstructionRecord instruction{};
            uint64_t tag_id = 0u;
            if (!decoder.integer(instruction.id) ||
                !decoder.integer(instruction.block_id) ||
                !decoder.integer(tag_id)) {
                return false;
            }
            auto tag = binary_instruction_tag(tag_id);
            if (!tag) { return reader.fail("Unknown XIR binary instruction tag."); }
            instruction.tag = *tag;
            luisa::string op;
            if (!decoder.string(instruction.type, max_type_description_size) ||
                !decoder.string(op, max_string_payload_size)) {
                return false;
            }
            auto parsed_op = parse_instruction_op(instruction.tag, op);
            if (!parsed_op) { return reader.fail("Unknown XIR binary instruction operation."); }
            instruction.op = *parsed_op;
            size_t operand_count = 0u;
            if (!decoder.count<int64_t>(operand_count)) { return false; }
            instruction.operands.reserve(operand_count);
            for (auto j = 0u; j < operand_count; j++) {
                int64_t operand = 0;
                if (!decoder.signed_integer(operand)) { return false; }
                if (operand < -1) {
                    return reader.fail("XIR binary value identifiers must be nonnegative or -1.");
                }
                instruction.operands.emplace_back(operand);
            }
            size_t auxiliary_count = 0u;
            if (!decoder.count<int64_t>(auxiliary_count)) { return false; }
            instruction.auxiliary.reserve(auxiliary_count);
            for (auto j = 0u; j < auxiliary_count; j++) {
                int64_t auxiliary = 0;
                if (!decoder.signed_integer(auxiliary)) { return false; }
                instruction.auxiliary.emplace_back(auxiliary);
            }
            size_t payload_count = 0u;
            if (!decoder.count<luisa::string>(payload_count)) { return false; }
            instruction.payloads.reserve(payload_count);
            for (auto j = 0u; j < payload_count; j++) {
                luisa::string payload;
                if (!decoder.string(payload, max_string_payload_size)) { return false; }
                instruction.payloads.emplace_back(std::move(payload));
            }
            if (!decoder.metadata(instruction.metadata)) { return false; }
            function.instructions.emplace_back(std::move(instruction));
        }
        record.functions.emplace_back(std::move(function));
    }
    if (!reader.at_end()) { return reader.fail("Unexpected trailing bytes in XIR binary payload."); }
    return true;
}

[[nodiscard]] bool parse_module_record(TextParser &parser, ModuleRecord &record) noexcept {
    ParseBudget budget;
    uint64_t version = 0u;
    if (!parser.keyword("xir.text") || !parser.unsigned_integer(version)) { return false; }
    if (version != interchange_version) { return parser.fail("Unsupported XIR text interchange version."); }
    if (!parser.keyword("module") || !parser.punctuation('{') ||
        !parse_metadata_records(parser, budget, record.metadata) ||
        !parser.keyword("globals")) {
        return false;
    }
    size_t global_count = 0u;
    if (!parser.count(global_count) || !budget.consume(parser, global_count)) { return false; }
    record.globals.reserve(global_count);
    for (auto i = 0u; i < global_count; i++) {
        luisa::string kind;
        GlobalRecord global{};
        if (!parser.word(kind) || !parser.unsigned_integer(global.id)) { return false; }
        if (kind == "constant") {
            global.kind = GlobalRecord::Kind::CONSTANT;
            if (!parser.string(global.type) || !parser.string(global.payload)) { return false; }
        } else if (kind == "undefined") {
            global.kind = GlobalRecord::Kind::UNDEFINED;
            if (!parser.string(global.type)) { return false; }
        } else if (kind == "special") {
            global.kind = GlobalRecord::Kind::SPECIAL;
            if (!parser.word(global.payload)) { return false; }
        } else {
            return parser.fail("Unknown XIR global record kind.");
        }
        if (!parse_metadata_records(parser, budget, global.metadata)) { return false; }
        record.globals.emplace_back(std::move(global));
    }
    if (!parser.keyword("functions")) { return false; }
    size_t function_count = 0u;
    if (!parser.count(function_count) || !budget.consume(parser, function_count)) { return false; }
    record.functions.reserve(function_count);
    for (auto function_index = 0u; function_index < function_count; function_index++) {
        FunctionRecord function{};
        uint64_t block_x = 0u;
        uint64_t block_y = 0u;
        uint64_t block_z = 0u;
        if (!parser.keyword("function") || !parser.unsigned_integer(function.id) ||
            !parser.word(function.kind) || !parser.string(function.return_type) ||
            !parser.unsigned_integer(block_x) || !parser.unsigned_integer(block_y) ||
            !parser.unsigned_integer(block_z) || !parser.punctuation('{')) {
            return false;
        }
        if (block_x > std::numeric_limits<uint32_t>::max() ||
            block_y > std::numeric_limits<uint32_t>::max() ||
            block_z > std::numeric_limits<uint32_t>::max()) {
            return parser.fail("Function block size component is out of range.");
        }
        function.block_size = {static_cast<uint32_t>(block_x), static_cast<uint32_t>(block_y), static_cast<uint32_t>(block_z)};
        if (!parse_metadata_records(parser, budget, function.metadata) ||
            !parser.keyword("arguments")) {
            return false;
        }
        size_t argument_count = 0u;
        if (!parser.count(argument_count) || !budget.consume(parser, argument_count)) { return false; }
        function.arguments.reserve(argument_count);
        for (auto i = 0u; i < argument_count; i++) {
            ArgumentRecord argument{};
            if (!parser.keyword("argument") || !parser.unsigned_integer(argument.id) ||
                !parser.word(argument.kind) || !parser.string(argument.type)) {
                return false;
            }
            if (!parse_metadata_records(parser, budget, argument.metadata)) { return false; }
            function.arguments.emplace_back(std::move(argument));
        }
        if (!parser.keyword("blocks")) { return false; }
        size_t block_count = 0u;
        if (!parser.count(block_count) || !budget.consume(parser, block_count)) { return false; }
        function.blocks.reserve(block_count);
        for (auto i = 0u; i < block_count; i++) {
            BasicBlockRecord block{};
            if (!parser.keyword("block") || !parser.unsigned_integer(block.id) ||
                !parse_metadata_records(parser, budget, block.metadata)) {
                return false;
            }
            function.blocks.emplace_back(std::move(block));
        }
        if (!parser.keyword("body") || !parser.signed_integer(function.body)) { return false; }
        if (function.body < -1) {
            return parser.fail("XIR body identifiers must be nonnegative or -1 for no definition.");
        }
        if (!parser.keyword("instructions")) { return false; }
        size_t instruction_count = 0u;
        if (!parser.count(instruction_count) || !budget.consume(parser, instruction_count)) { return false; }
        function.instructions.reserve(instruction_count);
        for (auto i = 0u; i < instruction_count; i++) {
            InstructionRecord instruction{};
            luisa::string tag;
            luisa::string op;
            if (!parser.keyword("instruction") || !parser.unsigned_integer(instruction.id) ||
                !parser.unsigned_integer(instruction.block_id) || !parser.word(tag) ||
                !parser.string(instruction.type) || !parser.word(op)) {
                return false;
            }
            auto parsed_tag = parse_instruction_name(tag);
            if (!parsed_tag) { return parser.fail("Unknown or unsupported XIR instruction kind."); }
            instruction.tag = *parsed_tag;
            auto parsed_op = parse_instruction_op(instruction.tag, op);
            if (!parsed_op) { return parser.fail("Unknown or invalid XIR instruction operation."); }
            instruction.op = *parsed_op;
            size_t operand_count = 0u;
            if (!parser.count(operand_count) || !budget.consume(parser, operand_count)) { return false; }
            instruction.operands.reserve(operand_count);
            for (auto j = 0u; j < operand_count; j++) {
                int64_t operand = -1;
                if (!parser.signed_integer(operand) || operand < -1) {
                    return parser.fail("XIR value identifiers must be nonnegative or -1 for null.");
                }
                instruction.operands.emplace_back(operand);
            }
            size_t auxiliary_count = 0u;
            if (!parser.count(auxiliary_count) || !budget.consume(parser, auxiliary_count)) { return false; }
            instruction.auxiliary.reserve(auxiliary_count);
            for (auto j = 0u; j < auxiliary_count; j++) {
                int64_t auxiliary = 0;
                if (!parser.signed_integer(auxiliary)) { return false; }
                instruction.auxiliary.emplace_back(auxiliary);
            }
            if (parser.try_keyword("payloads")) {
                size_t payload_count = 0u;
                if (!parser.count(payload_count) || !budget.consume(parser, payload_count)) { return false; }
                instruction.payloads.reserve(payload_count);
                for (auto j = 0u; j < payload_count; j++) {
                    luisa::string payload;
                    if (!parser.string(payload)) { return false; }
                    if (payload.size() > max_string_payload_size) {
                        return parser.fail("XIR instruction string payload exceeds the supported limit.");
                    }
                    instruction.payloads.emplace_back(std::move(payload));
                }
            }
            if (!parse_metadata_records(parser, budget, instruction.metadata)) { return false; }
            function.instructions.emplace_back(std::move(instruction));
        }
        if (!parser.punctuation('}')) { return false; }
        record.functions.emplace_back(std::move(function));
    }
    return parser.punctuation('}') && parser.at_end();
}

[[nodiscard]] bool supported_operand_shape(
    DerivedInstructionTag tag, size_t operand_count,
    size_t auxiliary_count, size_t payload_count) noexcept {
    switch (tag) {
        case DerivedInstructionTag::IF: return operand_count == 3u && auxiliary_count == 1u && payload_count == 0u;
        case DerivedInstructionTag::SWITCH: return operand_count >= 2u && auxiliary_count + 1u == operand_count && payload_count == 0u;
        case DerivedInstructionTag::LOOP: return operand_count == 1u && auxiliary_count == 3u && payload_count == 0u;
        case DerivedInstructionTag::SIMPLE_LOOP: return operand_count == 1u && auxiliary_count == 1u && payload_count == 0u;
        case DerivedInstructionTag::BRANCH: return operand_count == 1u && auxiliary_count == 0u && payload_count == 0u;
        case DerivedInstructionTag::CONDITIONAL_BRANCH: return operand_count == 3u && auxiliary_count == 0u && payload_count == 0u;
        case DerivedInstructionTag::UNREACHABLE: return operand_count == 0u && auxiliary_count == 0u && payload_count == 1u;
        case DerivedInstructionTag::RASTER_DISCARD: return operand_count == 0u && auxiliary_count == 0u && payload_count == 0u;
        case DerivedInstructionTag::CORO_SUSPEND: return operand_count == 1u && auxiliary_count == 1u && payload_count == 1u;
        case DerivedInstructionTag::CORO_RESUME: return operand_count == 1u && auxiliary_count == 1u && payload_count == 0u;
        case DerivedInstructionTag::CORO_TERMINATE: return operand_count == 0u && auxiliary_count == 0u && payload_count == 0u;
        case DerivedInstructionTag::RETURN: return operand_count == 1u && auxiliary_count == 0u && payload_count == 0u;
        case DerivedInstructionTag::PHI: return operand_count == auxiliary_count && payload_count == 0u;
        case DerivedInstructionTag::ALLOCA: return operand_count == 0u && auxiliary_count == 0u && payload_count == 0u;
        case DerivedInstructionTag::LOAD: return operand_count == 1u && auxiliary_count == 0u && payload_count == 0u;
        case DerivedInstructionTag::STORE: return operand_count == 2u && auxiliary_count == 0u && payload_count == 0u;
        case DerivedInstructionTag::GEP: return operand_count >= 2u && auxiliary_count == 0u && payload_count == 0u;
        case DerivedInstructionTag::ATOMIC:
        case DerivedInstructionTag::ARITHMETIC:
        case DerivedInstructionTag::THREAD_GROUP:
        case DerivedInstructionTag::RESOURCE_QUERY:
        case DerivedInstructionTag::RESOURCE_READ:
        case DerivedInstructionTag::RESOURCE_WRITE:
        case DerivedInstructionTag::RAY_QUERY_OBJECT_READ:
        case DerivedInstructionTag::RAY_QUERY_OBJECT_WRITE:
        case DerivedInstructionTag::RAY_QUERY_PIPELINE:
        case DerivedInstructionTag::AUTODIFF_INTRINSIC: return auxiliary_count == 0u && payload_count == 0u;
        case DerivedInstructionTag::RAY_QUERY_LOOP: return operand_count == 1u && auxiliary_count == 1u && payload_count == 0u;
        case DerivedInstructionTag::RAY_QUERY_DISPATCH: return operand_count == 4u && auxiliary_count == 0u && payload_count == 0u;
        case DerivedInstructionTag::AUTODIFF_SCOPE: return operand_count == 1u && auxiliary_count == 3u && payload_count == 0u;
        case DerivedInstructionTag::BREAK:
        case DerivedInstructionTag::CONTINUE: return operand_count == 1u && auxiliary_count == 0u && payload_count == 0u;
        case DerivedInstructionTag::CALL: return operand_count >= 1u && auxiliary_count == 0u && payload_count == 0u;
        case DerivedInstructionTag::CAST: return operand_count == 1u && auxiliary_count == 0u && payload_count == 0u;
        case DerivedInstructionTag::PRINT: return auxiliary_count == 0u && payload_count == 1u;
        case DerivedInstructionTag::CLOCK: return operand_count == 0u && auxiliary_count == 0u && payload_count == 0u;
        case DerivedInstructionTag::DEBUG_BREAK: return auxiliary_count == 0u && payload_count == 0u;
        case DerivedInstructionTag::ASSERT:
        case DerivedInstructionTag::ASSUME: return operand_count == 1u && auxiliary_count == 0u && payload_count == 1u;
        case DerivedInstructionTag::OUTLINE: return operand_count == 1u && auxiliary_count == 1u && payload_count == 0u;
        default: return false;
    }
}

[[nodiscard]] bool valid_op(DerivedInstructionTag tag, int64_t op) noexcept {
    switch (tag) {
        case DerivedInstructionTag::ALLOCA: return symbolic_op_name(op, alloca_wire_ops).has_value();
        case DerivedInstructionTag::ARITHMETIC: return symbolic_op_name(op, arithmetic_wire_ops).has_value();
        case DerivedInstructionTag::CAST: return symbolic_op_name(op, cast_wire_ops).has_value();
        case DerivedInstructionTag::ATOMIC: return symbolic_op_name(op, atomic_wire_ops).has_value();
        case DerivedInstructionTag::THREAD_GROUP: return symbolic_op_name(op, thread_group_wire_ops).has_value();
        case DerivedInstructionTag::RESOURCE_QUERY: return symbolic_op_name(op, resource_query_wire_ops).has_value();
        case DerivedInstructionTag::RESOURCE_READ: return symbolic_op_name(op, resource_read_wire_ops).has_value();
        case DerivedInstructionTag::RESOURCE_WRITE: return symbolic_op_name(op, resource_write_wire_ops).has_value();
        case DerivedInstructionTag::RAY_QUERY_OBJECT_READ: return symbolic_op_name(op, ray_query_object_read_wire_ops).has_value();
        case DerivedInstructionTag::RAY_QUERY_OBJECT_WRITE: return symbolic_op_name(op, ray_query_object_write_wire_ops).has_value();
        case DerivedInstructionTag::AUTODIFF_INTRINSIC: return symbolic_op_name(op, autodiff_intrinsic_wire_ops).has_value();
        case DerivedInstructionTag::DEBUG_BREAK: return op == 0;
        default: return op == -1;
    }
}

[[nodiscard]] bool arithmetic_operand_count_valid(ArithmeticOp op, size_t count) noexcept {
    switch (op) {
        case ArithmeticOp::UNARY_MINUS:
        case ArithmeticOp::UNARY_BIT_NOT:
        case ArithmeticOp::ALL:
        case ArithmeticOp::ANY:
        case ArithmeticOp::SATURATE:
        case ArithmeticOp::ABS:
        case ArithmeticOp::CLZ:
        case ArithmeticOp::CTZ:
        case ArithmeticOp::POPCOUNT:
        case ArithmeticOp::REVERSE:
        case ArithmeticOp::ISINF:
        case ArithmeticOp::ISNAN:
        case ArithmeticOp::ACOS:
        case ArithmeticOp::ACOSH:
        case ArithmeticOp::ASIN:
        case ArithmeticOp::ASINH:
        case ArithmeticOp::ATAN:
        case ArithmeticOp::ATANH:
        case ArithmeticOp::COS:
        case ArithmeticOp::COSH:
        case ArithmeticOp::SIN:
        case ArithmeticOp::SINH:
        case ArithmeticOp::TAN:
        case ArithmeticOp::TANH:
        case ArithmeticOp::EXP:
        case ArithmeticOp::EXP2:
        case ArithmeticOp::EXP10:
        case ArithmeticOp::LOG:
        case ArithmeticOp::LOG2:
        case ArithmeticOp::LOG10:
        case ArithmeticOp::SQRT:
        case ArithmeticOp::RSQRT:
        case ArithmeticOp::CEIL:
        case ArithmeticOp::FLOOR:
        case ArithmeticOp::FRACT:
        case ArithmeticOp::TRUNC:
        case ArithmeticOp::ROUND:
        case ArithmeticOp::RINT:
        case ArithmeticOp::LENGTH:
        case ArithmeticOp::LENGTH_SQUARED:
        case ArithmeticOp::NORMALIZE:
        case ArithmeticOp::REDUCE_SUM:
        case ArithmeticOp::REDUCE_PRODUCT:
        case ArithmeticOp::REDUCE_MIN:
        case ArithmeticOp::REDUCE_MAX:
        case ArithmeticOp::MATRIX_COMP_NEG:
        case ArithmeticOp::MATRIX_DETERMINANT:
        case ArithmeticOp::MATRIX_TRANSPOSE:
        case ArithmeticOp::MATRIX_INVERSE: return count == 1u;

        case ArithmeticOp::BINARY_ADD:
        case ArithmeticOp::BINARY_SUB:
        case ArithmeticOp::BINARY_MUL:
        case ArithmeticOp::BINARY_DIV:
        case ArithmeticOp::BINARY_MOD:
        case ArithmeticOp::BINARY_BIT_AND:
        case ArithmeticOp::BINARY_BIT_OR:
        case ArithmeticOp::BINARY_BIT_XOR:
        case ArithmeticOp::BINARY_SHIFT_LEFT:
        case ArithmeticOp::BINARY_SHIFT_RIGHT:
        case ArithmeticOp::BINARY_ROTATE_LEFT:
        case ArithmeticOp::BINARY_ROTATE_RIGHT:
        case ArithmeticOp::BINARY_LESS:
        case ArithmeticOp::BINARY_GREATER:
        case ArithmeticOp::BINARY_LESS_EQUAL:
        case ArithmeticOp::BINARY_GREATER_EQUAL:
        case ArithmeticOp::BINARY_EQUAL:
        case ArithmeticOp::BINARY_NOT_EQUAL:
        case ArithmeticOp::STEP:
        case ArithmeticOp::MIN:
        case ArithmeticOp::MAX:
        case ArithmeticOp::ATAN2:
        case ArithmeticOp::POW:
        case ArithmeticOp::POW_INT:
        case ArithmeticOp::COPYSIGN:
        case ArithmeticOp::CROSS:
        case ArithmeticOp::DOT:
        case ArithmeticOp::REFLECT:
        case ArithmeticOp::OUTER_PRODUCT:
        case ArithmeticOp::MATRIX_COMP_ADD:
        case ArithmeticOp::MATRIX_COMP_SUB:
        case ArithmeticOp::MATRIX_COMP_MUL:
        case ArithmeticOp::MATRIX_COMP_DIV:
        case ArithmeticOp::MATRIX_LINALG_MUL: return count == 2u;

        case ArithmeticOp::SELECT:
        case ArithmeticOp::CLAMP:
        case ArithmeticOp::LERP:
        case ArithmeticOp::SMOOTHSTEP:
        case ArithmeticOp::FMA:
        case ArithmeticOp::FACEFORWARD: return count == 3u;

        case ArithmeticOp::AGGREGATE: return count > 0u;
        case ArithmeticOp::SHUFFLE:
        case ArithmeticOp::EXTRACT: return count >= 2u;
        case ArithmeticOp::INSERT: return count >= 3u;
    }
    return false;
}

[[nodiscard]] bool resource_query_operand_count_valid(ResourceQueryOp op, size_t count) noexcept {
    switch (op) {
        case ResourceQueryOp::BUFFER_SIZE:
        case ResourceQueryOp::BYTE_BUFFER_SIZE:
        case ResourceQueryOp::TEXTURE2D_SIZE:
        case ResourceQueryOp::TEXTURE3D_SIZE:
        case ResourceQueryOp::BUFFER_DEVICE_ADDRESS: return count == 1u;

        case ResourceQueryOp::BINDLESS_BYTE_BUFFER_SIZE:
        case ResourceQueryOp::BINDLESS_TEXTURE2D_SIZE:
        case ResourceQueryOp::BINDLESS_TEXTURE3D_SIZE:
        case ResourceQueryOp::BINDLESS_BUFFER_DEVICE_ADDRESS:
        case ResourceQueryOp::RAY_TRACING_INSTANCE_TRANSFORM:
        case ResourceQueryOp::RAY_TRACING_INSTANCE_USER_ID:
        case ResourceQueryOp::RAY_TRACING_INSTANCE_VISIBILITY_MASK: return count == 2u;

        case ResourceQueryOp::BINDLESS_BUFFER_SIZE:
        case ResourceQueryOp::BINDLESS_TEXTURE2D_SIZE_LEVEL:
        case ResourceQueryOp::BINDLESS_TEXTURE3D_SIZE_LEVEL:
        case ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE:
        case ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE:
        case ResourceQueryOp::RAY_TRACING_TRACE_CLOSEST:
        case ResourceQueryOp::RAY_TRACING_TRACE_ANY:
        case ResourceQueryOp::RAY_TRACING_QUERY_ALL:
        case ResourceQueryOp::RAY_TRACING_QUERY_ANY:
        case ResourceQueryOp::RAY_TRACING_INSTANCE_MOTION_MATRIX:
        case ResourceQueryOp::RAY_TRACING_INSTANCE_MOTION_SRT: return count == 3u;

        case ResourceQueryOp::TEXTURE2D_SAMPLE:
        case ResourceQueryOp::TEXTURE3D_SAMPLE:
        case ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_LEVEL:
        case ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE_LEVEL:
        case ResourceQueryOp::RAY_TRACING_TRACE_CLOSEST_MOTION_BLUR:
        case ResourceQueryOp::RAY_TRACING_TRACE_ANY_MOTION_BLUR:
        case ResourceQueryOp::RAY_TRACING_QUERY_ALL_MOTION_BLUR:
        case ResourceQueryOp::RAY_TRACING_QUERY_ANY_MOTION_BLUR: return count == 4u;

        case ResourceQueryOp::TEXTURE2D_SAMPLE_LEVEL:
        case ResourceQueryOp::TEXTURE3D_SAMPLE_LEVEL:
        case ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_GRAD:
        case ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE_GRAD:
        case ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_SAMPLER:
        case ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE_SAMPLER: return count == 5u;

        case ResourceQueryOp::TEXTURE2D_SAMPLE_GRAD:
        case ResourceQueryOp::TEXTURE3D_SAMPLE_GRAD:
        case ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_GRAD_LEVEL:
        case ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE_GRAD_LEVEL:
        case ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_LEVEL_SAMPLER:
        case ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE_LEVEL_SAMPLER: return count == 6u;

        case ResourceQueryOp::TEXTURE2D_SAMPLE_GRAD_LEVEL:
        case ResourceQueryOp::TEXTURE3D_SAMPLE_GRAD_LEVEL:
        case ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_GRAD_SAMPLER:
        case ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE_GRAD_SAMPLER: return count == 7u;

        case ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_GRAD_LEVEL_SAMPLER:
        case ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE_GRAD_LEVEL_SAMPLER: return count == 8u;
    }
    return false;
}

[[nodiscard]] bool resource_read_operand_count_valid(ResourceReadOp op, size_t count) noexcept {
    switch (op) {
        case ResourceReadOp::BUFFER_READ:
        case ResourceReadOp::BUFFER_VOLATILE_READ:
        case ResourceReadOp::BYTE_BUFFER_READ:
        case ResourceReadOp::BYTE_BUFFER_VOLATILE_READ:
        case ResourceReadOp::TEXTURE2D_READ:
        case ResourceReadOp::TEXTURE3D_READ: return count == 2u;
        case ResourceReadOp::BINDLESS_BUFFER_READ:
        case ResourceReadOp::BINDLESS_BYTE_BUFFER_READ:
        case ResourceReadOp::BINDLESS_TEXTURE2D_READ:
        case ResourceReadOp::BINDLESS_TEXTURE3D_READ: return count == 3u;
        case ResourceReadOp::BINDLESS_TEXTURE2D_READ_LEVEL:
        case ResourceReadOp::BINDLESS_TEXTURE3D_READ_LEVEL: return count == 4u;
        case ResourceReadOp::DEVICE_ADDRESS_READ: return count == 1u;
    }
    return false;
}

[[nodiscard]] bool resource_write_operand_count_valid(ResourceWriteOp op, size_t count) noexcept {
    switch (op) {
        case ResourceWriteOp::BUFFER_WRITE:
        case ResourceWriteOp::BUFFER_VOLATILE_WRITE:
        case ResourceWriteOp::BYTE_BUFFER_WRITE:
        case ResourceWriteOp::BYTE_BUFFER_VOLATILE_WRITE:
        case ResourceWriteOp::TEXTURE2D_WRITE:
        case ResourceWriteOp::TEXTURE3D_WRITE:
        case ResourceWriteOp::RAY_TRACING_SET_INSTANCE_TRANSFORM:
        case ResourceWriteOp::RAY_TRACING_SET_INSTANCE_VISIBILITY_MASK:
        case ResourceWriteOp::RAY_TRACING_SET_INSTANCE_OPACITY:
        case ResourceWriteOp::RAY_TRACING_SET_INSTANCE_USER_ID: return count == 3u;
        case ResourceWriteOp::BINDLESS_BUFFER_WRITE:
        case ResourceWriteOp::BINDLESS_BYTE_BUFFER_WRITE:
        case ResourceWriteOp::RAY_TRACING_SET_INSTANCE_MOTION_MATRIX:
        case ResourceWriteOp::RAY_TRACING_SET_INSTANCE_MOTION_SRT: return count == 4u;
        case ResourceWriteOp::DEVICE_ADDRESS_WRITE:
        case ResourceWriteOp::INDIRECT_DISPATCH_SET_COUNT: return count == 2u;
        case ResourceWriteOp::INDIRECT_DISPATCH_SET_KERNEL: return count == 5u;
    }
    return false;
}

[[nodiscard]] bool thread_group_operand_count_valid(ThreadGroupOp op, size_t count) noexcept {
    switch (op) {
        case ThreadGroupOp::SHADER_EXECUTION_REORDER: return count == 0u || count == 2u;
        case ThreadGroupOp::WARP_IS_FIRST_ACTIVE_LANE:
        case ThreadGroupOp::WARP_FIRST_ACTIVE_LANE:
        case ThreadGroupOp::SYNCHRONIZE_BLOCK: return count == 0u;
        case ThreadGroupOp::WARP_READ_LANE: return count == 2u;
        case ThreadGroupOp::RASTER_QUAD_DDX:
        case ThreadGroupOp::RASTER_QUAD_DDY:
        case ThreadGroupOp::WARP_ACTIVE_ALL_EQUAL:
        case ThreadGroupOp::WARP_ACTIVE_BIT_AND:
        case ThreadGroupOp::WARP_ACTIVE_BIT_OR:
        case ThreadGroupOp::WARP_ACTIVE_BIT_XOR:
        case ThreadGroupOp::WARP_ACTIVE_COUNT_BITS:
        case ThreadGroupOp::WARP_ACTIVE_MAX:
        case ThreadGroupOp::WARP_ACTIVE_MIN:
        case ThreadGroupOp::WARP_ACTIVE_PRODUCT:
        case ThreadGroupOp::WARP_ACTIVE_SUM:
        case ThreadGroupOp::WARP_ACTIVE_ALL:
        case ThreadGroupOp::WARP_ACTIVE_ANY:
        case ThreadGroupOp::WARP_ACTIVE_BIT_MASK:
        case ThreadGroupOp::WARP_PREFIX_COUNT_BITS:
        case ThreadGroupOp::WARP_PREFIX_SUM:
        case ThreadGroupOp::WARP_PREFIX_PRODUCT:
        case ThreadGroupOp::WARP_READ_FIRST_ACTIVE_LANE: return count == 1u;
    }
    return false;
}

[[nodiscard]] bool ray_query_object_write_operand_count_valid(
    RayQueryObjectWriteOp op, size_t count) noexcept {
    return count == (op == RayQueryObjectWriteOp::RAY_QUERY_OBJECT_COMMIT_PROCEDURAL ? 2u : 1u);
}

[[nodiscard]] bool autodiff_intrinsic_operand_count_valid(
    AutodiffIntrinsicOp op, size_t count) noexcept {
    switch (op) {
        case AutodiffIntrinsicOp::AUTODIFF_REQUIRES_GRADIENT:
        case AutodiffIntrinsicOp::AUTODIFF_GRADIENT:
        case AutodiffIntrinsicOp::AUTODIFF_DETACH: return count == 1u;
        case AutodiffIntrinsicOp::AUTODIFF_GRADIENT_MARKER:
        case AutodiffIntrinsicOp::AUTODIFF_ACCUMULATE_GRADIENT:
        case AutodiffIntrinsicOp::AUTODIFF_OUTPUT_GRADIENT: return count == 2u;
        case AutodiffIntrinsicOp::AUTODIFF_BACKWARD: return count == 0u;
        case AutodiffIntrinsicOp::AUTODIFF_PROPAGATE_GRADIENT: return count >= 2u;
    }
    return false;
}

[[nodiscard]] bool instruction_operand_count_valid(
    DerivedInstructionTag tag, int64_t op,
    size_t operand_count, size_t auxiliary_count,
    size_t payload_count) noexcept {
    if (!supported_operand_shape(tag, operand_count, auxiliary_count, payload_count)) { return false; }
    switch (tag) {
        case DerivedInstructionTag::ARITHMETIC:
            return arithmetic_operand_count_valid(static_cast<ArithmeticOp>(op), operand_count);
        case DerivedInstructionTag::ATOMIC:
            return operand_count >= 1u + atomic_op_value_count(static_cast<AtomicOp>(op));
        case DerivedInstructionTag::THREAD_GROUP:
            return thread_group_operand_count_valid(static_cast<ThreadGroupOp>(op), operand_count);
        case DerivedInstructionTag::RESOURCE_QUERY:
            return resource_query_operand_count_valid(static_cast<ResourceQueryOp>(op), operand_count);
        case DerivedInstructionTag::RESOURCE_READ:
            return resource_read_operand_count_valid(static_cast<ResourceReadOp>(op), operand_count);
        case DerivedInstructionTag::RESOURCE_WRITE:
            return resource_write_operand_count_valid(static_cast<ResourceWriteOp>(op), operand_count);
        case DerivedInstructionTag::RAY_QUERY_OBJECT_READ: return operand_count == 1u;
        case DerivedInstructionTag::RAY_QUERY_OBJECT_WRITE:
            return ray_query_object_write_operand_count_valid(static_cast<RayQueryObjectWriteOp>(op), operand_count);
        case DerivedInstructionTag::RAY_QUERY_PIPELINE: return operand_count >= 3u;
        case DerivedInstructionTag::AUTODIFF_INTRINSIC:
            return autodiff_intrinsic_operand_count_valid(static_cast<AutodiffIntrinsicOp>(op), operand_count);
        default: return true;
    }
}

[[nodiscard]] bool integer_scalar_type(const Type *type) noexcept {
    return type != nullptr && (type->is_int() || type->is_uint());
}

[[nodiscard]] bool index32_type(const Type *type) noexcept {
    return type != nullptr && (type->is_int32() || type->is_uint32());
}

[[nodiscard]] bool atomic_value_type(const Type *type) noexcept {
    // This is the scalar atomic surface exposed by the AST/DSL and represented
    // by XIR. Backend-specific feature checks (for example Vulkan's separate
    // buffer/shared Int64Atomics features) belong at the codegen boundary.
    return type != nullptr &&
           (type->is_int32() || type->is_uint32() ||
            type->is_int64() || type->is_uint64() ||
            type->is_float32());
}

[[nodiscard]] bool typed_value_operand_valid(const Value *value) noexcept {
    return value != nullptr && value->type() != nullptr &&
           !value->isa<BasicBlock>() && !value->isa<Function>() &&
           !value->type()->is_resource();
}

[[nodiscard]] bool rvalue_operand_valid(const Value *value) noexcept {
    return typed_value_operand_valid(value) && !value->is_lvalue();
}

[[nodiscard]] bool data_operand_valid(const Value *value) noexcept {
    return rvalue_operand_valid(value) && !value->type()->is_custom();
}

template<typename IndexSpan>
[[nodiscard]] const Type *aggregate_indexed_type(
    const Type *base_type, const IndexSpan &indices) noexcept {
    auto current = base_type;
    for (auto index : indices) {
        if (!data_operand_valid(index) || !integer_scalar_type(index->type()) ||
            current == nullptr) {
            return nullptr;
        }
        switch (current->tag()) {
            case Type::Tag::ARRAY:
            case Type::Tag::VECTOR: {
                uint64_t constant_index = 0u;
                if (index->template isa<Constant>() &&
                    (!try_decode_constant_nonnegative_integer(index, constant_index) ||
                     constant_index >= current->dimension())) {
                    return nullptr;
                }
                current = current->element();
                break;
            }
            case Type::Tag::MATRIX: {
                uint64_t constant_index = 0u;
                if (index->template isa<Constant>() &&
                    (!try_decode_constant_nonnegative_integer(index, constant_index) ||
                     constant_index >= current->dimension())) {
                    return nullptr;
                }
                current = Type::vector(current->element(), current->dimension());
                break;
            }
            case Type::Tag::STRUCTURE: {
                uint64_t member_index = 0u;
                if (!try_decode_constant_nonnegative_integer(index, member_index) ||
                    member_index >= current->members().size()) {
                    return nullptr;
                }
                current = current->members()[member_index];
                break;
            }
            default: return nullptr;
        }
    }
    return current;
}

[[nodiscard]] size_t logical_register_width(const Type *type) noexcept {
    if (type == nullptr || !type->is_scalar_or_vector()) { return 0u; }
    if (type->is_vector()) {
        return type->element()->size() * type->dimension();
    }
    return type->size();
}

[[nodiscard]] bool cast_types_valid(
    CastOp op, const Type *target, const Value *source) noexcept {
    if (target == nullptr || target->is_resource() || target->is_custom() ||
        !data_operand_valid(source)) {
        return false;
    }
    auto source_type = source->type();
    switch (op) {
        case CastOp::STATIC_CAST:
            return source_type->is_scalar_or_vector() &&
                   target->is_scalar_or_vector() &&
                   source_type->dimension() == target->dimension();
        case CastOp::BITWISE_CAST:
            return source_type->is_scalar_or_vector() &&
                   target->is_scalar_or_vector() &&
                   !source_type->is_bool_or_bool_vector() &&
                   !target->is_bool_or_bool_vector() &&
                   logical_register_width(source_type) ==
                       logical_register_width(target);
    }
    return false;
}

[[nodiscard]] bool scalar_or_vector_integer(const Type *type) noexcept {
    return type != nullptr &&
           (type->is_int_or_int_vector() || type->is_uint_or_uint_vector());
}

[[nodiscard]] bool scalar_or_vector_uint32(const Type *type) noexcept {
    return type != nullptr &&
           (type->is_uint32() ||
            (type->is_vector() && type->element()->is_uint32()));
}

[[nodiscard]] bool scalar_or_vector_numeric(const Type *type) noexcept {
    return scalar_or_vector_integer(type) ||
           (type != nullptr && type->is_float_or_float_vector());
}

[[nodiscard]] bool scalar_or_vector_bitwise(const Type *type) noexcept {
    return scalar_or_vector_integer(type) ||
           (type != nullptr && type->is_bool_or_bool_vector());
}

[[nodiscard]] bool same_scalar_or_vector_shape(
    const Type *lhs, const Type *rhs) noexcept {
    return lhs != nullptr && rhs != nullptr && lhs->is_scalar_or_vector() &&
           rhs->is_scalar_or_vector() && lhs->dimension() == rhs->dimension();
}

[[nodiscard]] bool boolean_shape_for(
    const Type *boolean_type, const Type *value_type) noexcept {
    if (boolean_type == nullptr || value_type == nullptr) { return false; }
    if (value_type->is_scalar()) { return boolean_type->is_bool(); }
    return value_type->is_vector() &&
           boolean_type == Type::vector(Type::of<bool>(), value_type->dimension());
}

template<typename OperandSpan>
[[nodiscard]] bool arithmetic_types_valid(
    ArithmeticOp op, const Type *result, const OperandSpan &operands) noexcept {
    if (result == nullptr || result->is_resource() || result->is_custom()) { return false; }
    for (auto operand : operands) {
        if (!data_operand_valid(operand)) { return false; }
    }
    auto all_are = [&](const Type *type) noexcept {
        return std::all_of(operands.begin(), operands.end(),
                           [type](auto operand) noexcept { return operand->type() == type; });
    };
    auto same_or_element = [](const Type *candidate, const Type *type) noexcept {
        return candidate == type ||
               (type != nullptr && type->is_vector() && candidate == type->element());
    };
    switch (op) {
        case ArithmeticOp::UNARY_MINUS:
            return operands[0]->type() == result && scalar_or_vector_numeric(result);
        case ArithmeticOp::UNARY_BIT_NOT:
            return operands[0]->type() == result && scalar_or_vector_bitwise(result);

        case ArithmeticOp::BINARY_ADD:
        case ArithmeticOp::BINARY_SUB:
        case ArithmeticOp::BINARY_MUL:
        case ArithmeticOp::BINARY_DIV:
        case ArithmeticOp::BINARY_MOD:
            return all_are(result) && scalar_or_vector_numeric(result);
        case ArithmeticOp::BINARY_BIT_AND:
        case ArithmeticOp::BINARY_BIT_OR:
        case ArithmeticOp::BINARY_BIT_XOR:
            return all_are(result) && scalar_or_vector_bitwise(result);
        case ArithmeticOp::BINARY_SHIFT_LEFT:
        case ArithmeticOp::BINARY_SHIFT_RIGHT:
        case ArithmeticOp::BINARY_ROTATE_LEFT:
        case ArithmeticOp::BINARY_ROTATE_RIGHT:
            return operands[0]->type() == result && scalar_or_vector_integer(result) &&
                   scalar_or_vector_integer(operands[1]->type()) &&
                   same_scalar_or_vector_shape(result, operands[1]->type());

        case ArithmeticOp::BINARY_LESS:
        case ArithmeticOp::BINARY_GREATER:
        case ArithmeticOp::BINARY_LESS_EQUAL:
        case ArithmeticOp::BINARY_GREATER_EQUAL:
            return operands[0]->type() == operands[1]->type() &&
                   scalar_or_vector_numeric(operands[0]->type()) &&
                   boolean_shape_for(result, operands[0]->type());
        case ArithmeticOp::BINARY_EQUAL:
        case ArithmeticOp::BINARY_NOT_EQUAL:
            return operands[0]->type() == operands[1]->type() &&
                   (scalar_or_vector_numeric(operands[0]->type()) ||
                    operands[0]->type()->is_bool_or_bool_vector()) &&
                   boolean_shape_for(result, operands[0]->type());

        case ArithmeticOp::ALL:
        case ArithmeticOp::ANY:
            return result->is_bool() && operands[0]->type()->is_bool_vector();
        case ArithmeticOp::SELECT: {
            auto condition = operands[2]->type();
            auto condition_valid = condition->is_bool() ||
                                   (result->is_vector() &&
                                    condition == Type::vector(Type::of<bool>(), result->dimension()));
            return operands[0]->type() == result && operands[1]->type() == result &&
                   condition_valid;
        }
        case ArithmeticOp::CLAMP:
            return all_are(result) && scalar_or_vector_numeric(result);
        case ArithmeticOp::SATURATE:
            return operands[0]->type() == result && result->is_float_or_float_vector();
        case ArithmeticOp::LERP:
            return operands[0]->type() == result && operands[1]->type() == result &&
                   result->is_float_or_float_vector() &&
                   same_or_element(operands[2]->type(), result);
        case ArithmeticOp::SMOOTHSTEP:
            return operands[2]->type() == result && result->is_float_or_float_vector() &&
                   same_or_element(operands[0]->type(), result) &&
                   same_or_element(operands[1]->type(), result);
        case ArithmeticOp::STEP:
            return operands[1]->type() == result && result->is_float_or_float_vector() &&
                   same_or_element(operands[0]->type(), result);

        case ArithmeticOp::ABS:
            return operands[0]->type() == result && scalar_or_vector_numeric(result);
        case ArithmeticOp::MIN:
        case ArithmeticOp::MAX:
            return all_are(result) && scalar_or_vector_numeric(result);
        case ArithmeticOp::CLZ:
        case ArithmeticOp::CTZ:
        case ArithmeticOp::POPCOUNT:
        case ArithmeticOp::REVERSE:
            return operands[0]->type() == result &&
                   scalar_or_vector_uint32(result);
        case ArithmeticOp::ISINF:
        case ArithmeticOp::ISNAN:
            return operands[0]->type()->is_float_or_float_vector() &&
                   boolean_shape_for(result, operands[0]->type());

        case ArithmeticOp::ACOS:
        case ArithmeticOp::ACOSH:
        case ArithmeticOp::ASIN:
        case ArithmeticOp::ASINH:
        case ArithmeticOp::ATAN:
        case ArithmeticOp::ATANH:
        case ArithmeticOp::COS:
        case ArithmeticOp::COSH:
        case ArithmeticOp::SIN:
        case ArithmeticOp::SINH:
        case ArithmeticOp::TAN:
        case ArithmeticOp::TANH:
        case ArithmeticOp::EXP:
        case ArithmeticOp::EXP2:
        case ArithmeticOp::EXP10:
        case ArithmeticOp::LOG:
        case ArithmeticOp::LOG2:
        case ArithmeticOp::LOG10:
        case ArithmeticOp::SQRT:
        case ArithmeticOp::RSQRT:
        case ArithmeticOp::CEIL:
        case ArithmeticOp::FLOOR:
        case ArithmeticOp::FRACT:
        case ArithmeticOp::TRUNC:
        case ArithmeticOp::ROUND:
        case ArithmeticOp::RINT:
            return operands[0]->type() == result && result->is_float_or_float_vector();
        case ArithmeticOp::ATAN2:
        case ArithmeticOp::POW:
        case ArithmeticOp::COPYSIGN:
            return all_are(result) && result->is_float_or_float_vector();
        case ArithmeticOp::POW_INT:
            return operands[0]->type() == result && result->is_float_or_float_vector() &&
                   scalar_or_vector_integer(operands[1]->type()) &&
                   (same_scalar_or_vector_shape(result, operands[1]->type()) ||
                    operands[1]->type()->is_scalar());
        case ArithmeticOp::FMA:
            return all_are(result) && result->is_float_or_float_vector();

        case ArithmeticOp::CROSS:
            return all_are(result) && result->is_float_vector() && result->dimension() == 3u;
        case ArithmeticOp::DOT:
            return operands[0]->type() == operands[1]->type() &&
                   operands[0]->type()->is_float_vector() &&
                   result == operands[0]->type()->element();
        case ArithmeticOp::LENGTH:
        case ArithmeticOp::LENGTH_SQUARED:
            return operands[0]->type()->is_float_vector() &&
                   result == operands[0]->type()->element();
        case ArithmeticOp::NORMALIZE:
            return operands[0]->type() == result && result->is_float_vector();
        case ArithmeticOp::FACEFORWARD:
            return all_are(result) && result->is_float_vector();
        case ArithmeticOp::REFLECT:
            return all_are(result) && result->is_float_vector();
        case ArithmeticOp::REDUCE_SUM:
        case ArithmeticOp::REDUCE_PRODUCT:
        case ArithmeticOp::REDUCE_MIN:
        case ArithmeticOp::REDUCE_MAX:
            return scalar_or_vector_numeric(operands[0]->type()) &&
                   operands[0]->type()->is_vector() &&
                   result == operands[0]->type()->element();

        case ArithmeticOp::OUTER_PRODUCT: {
            auto lhs = operands[0]->type();
            auto rhs = operands[1]->type();
            if (lhs->is_float_vector() && rhs == lhs) {
                return result == Type::matrix(lhs->dimension());
            }
            return result->is_matrix() && lhs == result && rhs == result;
        }
        case ArithmeticOp::MATRIX_COMP_NEG:
            return result->is_matrix() && operands[0]->type() == result;
        case ArithmeticOp::MATRIX_COMP_ADD:
        case ArithmeticOp::MATRIX_COMP_SUB:
        case ArithmeticOp::MATRIX_COMP_MUL:
        case ArithmeticOp::MATRIX_COMP_DIV: {
            if (!result->is_matrix()) { return false; }
            auto element = result->element();
            auto lhs = operands[0]->type();
            auto rhs = operands[1]->type();
            return (lhs == result || lhs == element) &&
                   (rhs == result || rhs == element) &&
                   (lhs == result || rhs == result);
        }
        case ArithmeticOp::MATRIX_LINALG_MUL: {
            auto lhs = operands[0]->type();
            auto rhs = operands[1]->type();
            if (lhs->is_matrix() && rhs->is_matrix()) {
                return lhs == rhs && result == lhs;
            }
            if (lhs->is_matrix() && rhs->is_float_vector() &&
                lhs->dimension() == rhs->dimension()) {
                return result == rhs;
            }
            return lhs->is_float_vector() && rhs->is_matrix() &&
                   lhs->dimension() == rhs->dimension() && result == lhs;
        }
        case ArithmeticOp::MATRIX_DETERMINANT:
            return operands[0]->type()->is_matrix() &&
                   result == operands[0]->type()->element();
        case ArithmeticOp::MATRIX_TRANSPOSE:
        case ArithmeticOp::MATRIX_INVERSE:
            return result->is_matrix() && operands[0]->type() == result;

        case ArithmeticOp::AGGREGATE:
            if (result->is_vector() || result->is_array() ||
                result->is_cooperative_vector()) {
                return operands.size() == result->dimension() && all_are(result->element());
            }
            if (result->is_matrix()) {
                return operands.size() == result->dimension() &&
                       all_are(Type::vector(result->element(), result->dimension()));
            }
            if (result->is_structure()) {
                if (operands.size() != result->members().size()) { return false; }
                for (auto i = 0u; i < operands.size(); i++) {
                    if (operands[i]->type() != result->members()[i]) { return false; }
                }
                return true;
            }
            return false;
        case ArithmeticOp::SHUFFLE:
            if (!result->is_vector() || !operands[0]->type()->is_vector() ||
                result->element() != operands[0]->type()->element() ||
                operands.size() != result->dimension() + 1u) {
                return false;
            }
            for (auto index : operands.subspan(1u)) {
                if (!integer_scalar_type(index->type())) { return false; }
                uint64_t constant_index = 0u;
                if (index->template isa<Constant>() &&
                    (!try_decode_constant_nonnegative_integer(index, constant_index) ||
                     constant_index >= operands[0]->type()->dimension())) {
                    return false;
                }
            }
            return true;
        case ArithmeticOp::EXTRACT:
            return aggregate_indexed_type(operands[0]->type(), operands.subspan(1u)) == result;
        case ArithmeticOp::INSERT:
            return operands[0]->type() == result &&
                   aggregate_indexed_type(result, operands.subspan(2u)) == operands[1]->type();
    }
    return false;
}

[[nodiscard]] bool resource_argument_valid(const Value *value) noexcept {
    return value != nullptr && value->isa<ResourceArgument>() &&
           value->type() != nullptr && value->type()->is_resource();
}

[[nodiscard]] const Type *ray_type() noexcept {
    static const auto type = Type::structure(
        16u, {Type::array(Type::of<float>(), 3u), Type::of<float>(),
              Type::array(Type::of<float>(), 3u), Type::of<float>()});
    return type;
}

[[nodiscard]] const Type *surface_hit_type() noexcept {
    static const auto type = Type::structure(
        8u, {Type::of<uint32_t>(), Type::of<uint32_t>(),
             Type::vector(Type::of<float>(), 2u), Type::of<float>()});
    return type;
}

[[nodiscard]] const Type *procedural_hit_type() noexcept {
    static const auto type = Type::structure(
        8u, {Type::of<uint32_t>(), Type::of<uint32_t>()});
    return type;
}

[[nodiscard]] const Type *committed_hit_type() noexcept {
    static const auto type = Type::structure(
        8u, {Type::of<uint32_t>(), Type::of<uint32_t>(),
             Type::vector(Type::of<float>(), 2u), Type::of<uint32_t>(),
             Type::of<float>()});
    return type;
}

[[nodiscard]] const Type *motion_srt_type() noexcept {
    static const auto type = Type::structure(
        16u, {Type::array(Type::of<float>(), 3u),
              Type::array(Type::of<float>(), 4u),
              Type::array(Type::of<float>(), 3u),
              Type::array(Type::of<float>(), 3u),
              Type::array(Type::of<float>(), 3u)});
    return type;
}

[[nodiscard]] const Type *ray_query_all_type() noexcept {
    static const auto type = Type::custom("LC_RayQueryAll");
    return type;
}

[[nodiscard]] const Type *ray_query_any_type() noexcept {
    static const auto type = Type::custom("LC_RayQueryAny");
    return type;
}

[[nodiscard]] const Type *indirect_dispatch_buffer_type() noexcept {
    static const auto type = Type::custom("LC_IndirectDispatchBuffer");
    return type;
}

[[nodiscard]] bool ray_query_type(const Type *type) noexcept {
    return type == ray_query_all_type() || type == ray_query_any_type();
}

[[nodiscard]] bool ray_query_object_valid(const Value *value) noexcept {
    return typed_value_operand_valid(value) && value->is_lvalue() &&
           ray_query_type(value->type());
}

template<typename OperandSpan>
[[nodiscard]] const Type *atomic_addressed_type(
    const Value *base, const OperandSpan &indices) noexcept {
    const Type *current = nullptr;
    if (base != nullptr && base->isa<ResourceArgument>() &&
        base->type() != nullptr && base->type()->is_buffer() &&
        base->type()->element() != nullptr && !indices.empty()) {
        current = base->type();
    } else if (base != nullptr && base->isa<AllocaInst>()) {
        auto alloca = static_cast<const AllocaInst *>(base);
        if (!alloca->is_shared() || alloca->type() == nullptr ||
            !alloca->type()->is_array()) {
            return nullptr;
        }
        current = alloca->type();
    } else {
        return nullptr;
    }
    for (auto index : indices) {
        if (!data_operand_valid(index) ||
            !integer_scalar_type(index->type()) ||
            current == nullptr) {
            return nullptr;
        }
        switch (current->tag()) {
            case Type::Tag::BUFFER:
                current = current->element();
                break;
            case Type::Tag::ARRAY:
            case Type::Tag::VECTOR: {
                uint64_t constant_index = 0u;
                if (index->template isa<Constant>() &&
                    (!try_decode_constant_nonnegative_integer(index, constant_index) ||
                     constant_index >= current->dimension())) {
                    return nullptr;
                }
                current = current->element();
                break;
            }
            case Type::Tag::MATRIX: {
                uint64_t constant_index = 0u;
                if (index->template isa<Constant>() &&
                    (!try_decode_constant_nonnegative_integer(index, constant_index) ||
                     constant_index >= current->dimension())) {
                    return nullptr;
                }
                current = Type::vector(current->element(), current->dimension());
                break;
            }
            case Type::Tag::STRUCTURE: {
                uint64_t member_index = 0u;
                if (!try_decode_constant_nonnegative_integer(index, member_index) ||
                    member_index >= current->members().size()) {
                    return nullptr;
                }
                current = current->members()[member_index];
                break;
            }
            default: return nullptr;
        }
    }
    return current;
}

[[nodiscard]] bool uint_vector_type(const Type *type, uint32_t dimension) noexcept {
    return type == Type::vector(Type::of<uint32_t>(), dimension);
}

[[nodiscard]] bool float_vector_type(const Type *type, uint32_t dimension) noexcept {
    return type == Type::vector(Type::of<float>(), dimension);
}

[[nodiscard]] bool resource_query_base_type_valid(ResourceQueryOp op, const Value *base_value) noexcept {
    if (base_value == nullptr || base_value->type() == nullptr) { return false; }
    auto base = base_value->type();
    switch (op) {
        case ResourceQueryOp::BUFFER_SIZE:
            return resource_argument_valid(base_value) && base->is_buffer() && base->element() != nullptr;
        case ResourceQueryOp::BYTE_BUFFER_SIZE:
            return resource_argument_valid(base_value) && base->is_buffer() && base->element() == nullptr;
        case ResourceQueryOp::BUFFER_DEVICE_ADDRESS:
            return resource_argument_valid(base_value) && base->is_buffer();
        case ResourceQueryOp::TEXTURE2D_SIZE:
        case ResourceQueryOp::TEXTURE2D_SAMPLE:
        case ResourceQueryOp::TEXTURE2D_SAMPLE_LEVEL:
        case ResourceQueryOp::TEXTURE2D_SAMPLE_GRAD:
        case ResourceQueryOp::TEXTURE2D_SAMPLE_GRAD_LEVEL:
            return resource_argument_valid(base_value) && base->is_texture() && base->dimension() == 2u;
        case ResourceQueryOp::TEXTURE3D_SIZE:
        case ResourceQueryOp::TEXTURE3D_SAMPLE:
        case ResourceQueryOp::TEXTURE3D_SAMPLE_LEVEL:
        case ResourceQueryOp::TEXTURE3D_SAMPLE_GRAD:
        case ResourceQueryOp::TEXTURE3D_SAMPLE_GRAD_LEVEL:
            return resource_argument_valid(base_value) && base->is_texture() && base->dimension() == 3u;
        case ResourceQueryOp::BINDLESS_BUFFER_SIZE:
        case ResourceQueryOp::BINDLESS_BYTE_BUFFER_SIZE:
        case ResourceQueryOp::BINDLESS_TEXTURE2D_SIZE:
        case ResourceQueryOp::BINDLESS_TEXTURE3D_SIZE:
        case ResourceQueryOp::BINDLESS_TEXTURE2D_SIZE_LEVEL:
        case ResourceQueryOp::BINDLESS_TEXTURE3D_SIZE_LEVEL:
        case ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE:
        case ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_LEVEL:
        case ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_GRAD:
        case ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_GRAD_LEVEL:
        case ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE:
        case ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE_LEVEL:
        case ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE_GRAD:
        case ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE_GRAD_LEVEL:
        case ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_SAMPLER:
        case ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_LEVEL_SAMPLER:
        case ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_GRAD_SAMPLER:
        case ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_GRAD_LEVEL_SAMPLER:
        case ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE_SAMPLER:
        case ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE_LEVEL_SAMPLER:
        case ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE_GRAD_SAMPLER:
        case ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE_GRAD_LEVEL_SAMPLER:
        case ResourceQueryOp::BINDLESS_BUFFER_DEVICE_ADDRESS:
            return resource_argument_valid(base_value) && base->is_bindless_array();
        case ResourceQueryOp::RAY_TRACING_INSTANCE_TRANSFORM:
        case ResourceQueryOp::RAY_TRACING_INSTANCE_USER_ID:
        case ResourceQueryOp::RAY_TRACING_INSTANCE_VISIBILITY_MASK:
        case ResourceQueryOp::RAY_TRACING_TRACE_CLOSEST:
        case ResourceQueryOp::RAY_TRACING_TRACE_ANY:
        case ResourceQueryOp::RAY_TRACING_QUERY_ALL:
        case ResourceQueryOp::RAY_TRACING_QUERY_ANY:
        case ResourceQueryOp::RAY_TRACING_INSTANCE_MOTION_MATRIX:
        case ResourceQueryOp::RAY_TRACING_INSTANCE_MOTION_SRT:
        case ResourceQueryOp::RAY_TRACING_TRACE_CLOSEST_MOTION_BLUR:
        case ResourceQueryOp::RAY_TRACING_TRACE_ANY_MOTION_BLUR:
        case ResourceQueryOp::RAY_TRACING_QUERY_ALL_MOTION_BLUR:
        case ResourceQueryOp::RAY_TRACING_QUERY_ANY_MOTION_BLUR:
            return resource_argument_valid(base_value) && base->is_accel();
    }
    return false;
}

[[nodiscard]] bool resource_query_result_type_valid(ResourceQueryOp op, const Type *type) noexcept {
    if (type == nullptr) { return false; }
    switch (op) {
        case ResourceQueryOp::BUFFER_SIZE:
        case ResourceQueryOp::BYTE_BUFFER_SIZE: return type->is_uint32() || type->is_uint64();
        case ResourceQueryOp::BINDLESS_BUFFER_SIZE:
        case ResourceQueryOp::BINDLESS_BYTE_BUFFER_SIZE:
            return type->is_uint32() || type->is_uint64();
        case ResourceQueryOp::BUFFER_DEVICE_ADDRESS:
        case ResourceQueryOp::BINDLESS_BUFFER_DEVICE_ADDRESS: return type->is_uint64();
        case ResourceQueryOp::TEXTURE2D_SIZE:
        case ResourceQueryOp::BINDLESS_TEXTURE2D_SIZE:
        case ResourceQueryOp::BINDLESS_TEXTURE2D_SIZE_LEVEL: return uint_vector_type(type, 2u);
        case ResourceQueryOp::TEXTURE3D_SIZE:
        case ResourceQueryOp::BINDLESS_TEXTURE3D_SIZE:
        case ResourceQueryOp::BINDLESS_TEXTURE3D_SIZE_LEVEL: return uint_vector_type(type, 3u);
        case ResourceQueryOp::TEXTURE2D_SAMPLE:
        case ResourceQueryOp::TEXTURE2D_SAMPLE_LEVEL:
        case ResourceQueryOp::TEXTURE2D_SAMPLE_GRAD:
        case ResourceQueryOp::TEXTURE2D_SAMPLE_GRAD_LEVEL:
        case ResourceQueryOp::TEXTURE3D_SAMPLE:
        case ResourceQueryOp::TEXTURE3D_SAMPLE_LEVEL:
        case ResourceQueryOp::TEXTURE3D_SAMPLE_GRAD:
        case ResourceQueryOp::TEXTURE3D_SAMPLE_GRAD_LEVEL:
        case ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE:
        case ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_LEVEL:
        case ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_GRAD:
        case ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_GRAD_LEVEL:
        case ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE:
        case ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE_LEVEL:
        case ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE_GRAD:
        case ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE_GRAD_LEVEL:
        case ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_SAMPLER:
        case ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_LEVEL_SAMPLER:
        case ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_GRAD_SAMPLER:
        case ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_GRAD_LEVEL_SAMPLER:
        case ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE_SAMPLER:
        case ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE_LEVEL_SAMPLER:
        case ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE_GRAD_SAMPLER:
        case ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE_GRAD_LEVEL_SAMPLER: return float_vector_type(type, 4u);
        case ResourceQueryOp::RAY_TRACING_INSTANCE_USER_ID:
        case ResourceQueryOp::RAY_TRACING_INSTANCE_VISIBILITY_MASK: return type->is_uint32();
        case ResourceQueryOp::RAY_TRACING_INSTANCE_TRANSFORM:
        case ResourceQueryOp::RAY_TRACING_INSTANCE_MOTION_MATRIX: return type == Type::matrix(4u);
        case ResourceQueryOp::RAY_TRACING_INSTANCE_MOTION_SRT: return type == motion_srt_type();
        case ResourceQueryOp::RAY_TRACING_TRACE_CLOSEST:
        case ResourceQueryOp::RAY_TRACING_TRACE_CLOSEST_MOTION_BLUR: return type == surface_hit_type();
        case ResourceQueryOp::RAY_TRACING_TRACE_ANY:
        case ResourceQueryOp::RAY_TRACING_TRACE_ANY_MOTION_BLUR: return type->is_bool();
        case ResourceQueryOp::RAY_TRACING_QUERY_ALL:
        case ResourceQueryOp::RAY_TRACING_QUERY_ALL_MOTION_BLUR: return type == ray_query_all_type();
        case ResourceQueryOp::RAY_TRACING_QUERY_ANY:
        case ResourceQueryOp::RAY_TRACING_QUERY_ANY_MOTION_BLUR: return type == ray_query_any_type();
    }
    return false;
}

[[nodiscard]] bool resource_read_base_type_valid(ResourceReadOp op, const Value *base_value) noexcept {
    if (base_value == nullptr || base_value->type() == nullptr) { return false; }
    auto base = base_value->type();
    switch (op) {
        case ResourceReadOp::BUFFER_READ:
        case ResourceReadOp::BUFFER_VOLATILE_READ:
            return resource_argument_valid(base_value) && base->is_buffer() && base->element() != nullptr;
        case ResourceReadOp::BYTE_BUFFER_READ:
        case ResourceReadOp::BYTE_BUFFER_VOLATILE_READ:
            return resource_argument_valid(base_value) && base->is_buffer() && base->element() == nullptr;
        case ResourceReadOp::TEXTURE2D_READ:
            return resource_argument_valid(base_value) && base->is_texture() && base->dimension() == 2u;
        case ResourceReadOp::TEXTURE3D_READ:
            return resource_argument_valid(base_value) && base->is_texture() && base->dimension() == 3u;
        case ResourceReadOp::BINDLESS_BUFFER_READ:
        case ResourceReadOp::BINDLESS_BYTE_BUFFER_READ:
        case ResourceReadOp::BINDLESS_TEXTURE2D_READ:
        case ResourceReadOp::BINDLESS_TEXTURE3D_READ:
        case ResourceReadOp::BINDLESS_TEXTURE2D_READ_LEVEL:
        case ResourceReadOp::BINDLESS_TEXTURE3D_READ_LEVEL:
            return resource_argument_valid(base_value) && base->is_bindless_array();
        case ResourceReadOp::DEVICE_ADDRESS_READ:
            return data_operand_valid(base_value) && base->is_uint64();
    }
    return false;
}

template<typename OperandSpan>
[[nodiscard]] bool resource_query_operand_types_valid(
    ResourceQueryOp op, const OperandSpan &operands) noexcept {
    auto type_is = [&](size_t index, const Type *type) noexcept {
        return operands[index]->type() == type;
    };
    auto uint_at = [&](size_t index) noexcept {
        return operands[index]->type() != nullptr && operands[index]->type()->is_uint32();
    };
    auto index32_at = [&](size_t index) noexcept {
        return index32_type(operands[index]->type());
    };
    auto integer_at = [&](size_t index) noexcept {
        return integer_scalar_type(operands[index]->type());
    };
    auto float_at = [&](size_t index) noexcept {
        return operands[index]->type() != nullptr && operands[index]->type()->is_float32();
    };
    auto float_vector_at = [&](size_t index, uint32_t dimension) noexcept {
        return type_is(index, Type::vector(Type::of<float>(), dimension));
    };
    switch (op) {
        case ResourceQueryOp::BUFFER_SIZE:
        case ResourceQueryOp::BYTE_BUFFER_SIZE:
        case ResourceQueryOp::TEXTURE2D_SIZE:
        case ResourceQueryOp::TEXTURE3D_SIZE:
        case ResourceQueryOp::BUFFER_DEVICE_ADDRESS: return true;
        case ResourceQueryOp::BINDLESS_BUFFER_SIZE: return index32_at(1u) && uint_at(2u);
        case ResourceQueryOp::BINDLESS_BYTE_BUFFER_SIZE:
        case ResourceQueryOp::BINDLESS_TEXTURE2D_SIZE:
        case ResourceQueryOp::BINDLESS_TEXTURE3D_SIZE:
        case ResourceQueryOp::BINDLESS_BUFFER_DEVICE_ADDRESS: return index32_at(1u);
        case ResourceQueryOp::BINDLESS_TEXTURE2D_SIZE_LEVEL:
        case ResourceQueryOp::BINDLESS_TEXTURE3D_SIZE_LEVEL: return index32_at(1u) && integer_at(2u);

        case ResourceQueryOp::TEXTURE2D_SAMPLE:
            return float_vector_at(1u, 2u) && uint_at(2u) && uint_at(3u);
        case ResourceQueryOp::TEXTURE3D_SAMPLE:
            return float_vector_at(1u, 3u) && uint_at(2u) && uint_at(3u);
        case ResourceQueryOp::TEXTURE2D_SAMPLE_LEVEL:
            return float_vector_at(1u, 2u) && float_at(2u) && uint_at(3u) && uint_at(4u);
        case ResourceQueryOp::TEXTURE3D_SAMPLE_LEVEL:
            return float_vector_at(1u, 3u) && float_at(2u) && uint_at(3u) && uint_at(4u);
        case ResourceQueryOp::TEXTURE2D_SAMPLE_GRAD:
            return float_vector_at(1u, 2u) && float_vector_at(2u, 2u) && float_vector_at(3u, 2u) && uint_at(4u) && uint_at(5u);
        case ResourceQueryOp::TEXTURE3D_SAMPLE_GRAD:
            return float_vector_at(1u, 3u) && float_vector_at(2u, 3u) && float_vector_at(3u, 3u) && uint_at(4u) && uint_at(5u);
        case ResourceQueryOp::TEXTURE2D_SAMPLE_GRAD_LEVEL:
            return float_vector_at(1u, 2u) && float_vector_at(2u, 2u) && float_vector_at(3u, 2u) && float_at(4u) && uint_at(5u) && uint_at(6u);
        case ResourceQueryOp::TEXTURE3D_SAMPLE_GRAD_LEVEL:
            return float_vector_at(1u, 3u) && float_vector_at(2u, 3u) && float_vector_at(3u, 3u) && float_at(4u) && uint_at(5u) && uint_at(6u);

        case ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE:
        case ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE:
        case ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_LEVEL:
        case ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE_LEVEL:
        case ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_GRAD:
        case ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE_GRAD:
        case ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_GRAD_LEVEL:
        case ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE_GRAD_LEVEL:
        case ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_SAMPLER:
        case ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE_SAMPLER:
        case ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_LEVEL_SAMPLER:
        case ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE_LEVEL_SAMPLER:
        case ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_GRAD_SAMPLER:
        case ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE_GRAD_SAMPLER:
        case ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_GRAD_LEVEL_SAMPLER:
        case ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE_GRAD_LEVEL_SAMPLER: {
            auto is_2d = op == ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE ||
                         op == ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_LEVEL ||
                         op == ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_GRAD ||
                         op == ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_GRAD_LEVEL ||
                         op == ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_SAMPLER ||
                         op == ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_LEVEL_SAMPLER ||
                         op == ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_GRAD_SAMPLER ||
                         op == ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_GRAD_LEVEL_SAMPLER;
            auto has_level = op == ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_LEVEL ||
                             op == ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE_LEVEL ||
                             op == ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_LEVEL_SAMPLER ||
                             op == ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE_LEVEL_SAMPLER;
            auto has_grad = op == ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_GRAD ||
                            op == ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE_GRAD ||
                            op == ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_GRAD_SAMPLER ||
                            op == ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE_GRAD_SAMPLER;
            auto has_grad_level = op == ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_GRAD_LEVEL ||
                                  op == ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE_GRAD_LEVEL ||
                                  op == ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_GRAD_LEVEL_SAMPLER ||
                                  op == ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE_GRAD_LEVEL_SAMPLER;
            auto sampler = op == ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_SAMPLER ||
                           op == ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE_SAMPLER ||
                           op == ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_LEVEL_SAMPLER ||
                           op == ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE_LEVEL_SAMPLER ||
                           op == ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_GRAD_SAMPLER ||
                           op == ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE_GRAD_SAMPLER ||
                           op == ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_GRAD_LEVEL_SAMPLER ||
                           op == ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE_GRAD_LEVEL_SAMPLER;
            auto dimension = is_2d ? 2u : 3u;
            if (!index32_at(1u) || !float_vector_at(2u, dimension)) { return false; }
            auto next = size_t{3u};
            if (has_level) {
                if (!float_at(next++)) { return false; }
            } else if (has_grad || has_grad_level) {
                if (!float_vector_at(next++, dimension) || !float_vector_at(next++, dimension)) { return false; }
                if (has_grad_level && !float_at(next++)) { return false; }
            }
            return !sampler || (uint_at(next) && uint_at(next + 1u));
        }

        case ResourceQueryOp::RAY_TRACING_INSTANCE_TRANSFORM:
        case ResourceQueryOp::RAY_TRACING_INSTANCE_USER_ID:
        case ResourceQueryOp::RAY_TRACING_INSTANCE_VISIBILITY_MASK: return index32_at(1u);
        case ResourceQueryOp::RAY_TRACING_TRACE_CLOSEST:
        case ResourceQueryOp::RAY_TRACING_TRACE_ANY:
        case ResourceQueryOp::RAY_TRACING_QUERY_ALL:
        case ResourceQueryOp::RAY_TRACING_QUERY_ANY:
            return type_is(1u, ray_type()) && uint_at(2u);
        case ResourceQueryOp::RAY_TRACING_INSTANCE_MOTION_MATRIX:
        case ResourceQueryOp::RAY_TRACING_INSTANCE_MOTION_SRT:
            return index32_at(1u) && index32_at(2u);
        case ResourceQueryOp::RAY_TRACING_TRACE_CLOSEST_MOTION_BLUR:
        case ResourceQueryOp::RAY_TRACING_TRACE_ANY_MOTION_BLUR:
        case ResourceQueryOp::RAY_TRACING_QUERY_ALL_MOTION_BLUR:
        case ResourceQueryOp::RAY_TRACING_QUERY_ANY_MOTION_BLUR:
            return type_is(1u, ray_type()) && float_at(2u) && uint_at(3u);
    }
    return false;
}

template<typename OperandSpan>
[[nodiscard]] bool resource_read_operand_types_valid(
    ResourceReadOp op, const OperandSpan &operands) noexcept {
    auto integer_at = [&](size_t index) noexcept {
        return integer_scalar_type(operands[index]->type());
    };
    auto index32_at = [&](size_t index) noexcept {
        return index32_type(operands[index]->type());
    };
    switch (op) {
        case ResourceReadOp::BUFFER_READ:
        case ResourceReadOp::BUFFER_VOLATILE_READ:
        case ResourceReadOp::BYTE_BUFFER_READ:
        case ResourceReadOp::BYTE_BUFFER_VOLATILE_READ: return integer_at(1u);
        case ResourceReadOp::TEXTURE2D_READ: return uint_vector_type(operands[1]->type(), 2u);
        case ResourceReadOp::TEXTURE3D_READ: return uint_vector_type(operands[1]->type(), 3u);
        case ResourceReadOp::BINDLESS_BUFFER_READ:
        case ResourceReadOp::BINDLESS_BYTE_BUFFER_READ: return index32_at(1u) && integer_at(2u);
        case ResourceReadOp::BINDLESS_TEXTURE2D_READ: return index32_at(1u) && uint_vector_type(operands[2]->type(), 2u);
        case ResourceReadOp::BINDLESS_TEXTURE3D_READ: return index32_at(1u) && uint_vector_type(operands[2]->type(), 3u);
        case ResourceReadOp::BINDLESS_TEXTURE2D_READ_LEVEL:
            return index32_at(1u) && uint_vector_type(operands[2]->type(), 2u) && integer_at(3u);
        case ResourceReadOp::BINDLESS_TEXTURE3D_READ_LEVEL:
            return index32_at(1u) && uint_vector_type(operands[2]->type(), 3u) && integer_at(3u);
        case ResourceReadOp::DEVICE_ADDRESS_READ: return true;
    }
    return false;
}

template<typename OperandSpan>
[[nodiscard]] bool resource_write_operand_types_valid(
    ResourceWriteOp op, const OperandSpan &operands) noexcept {
    auto uint_at = [&](size_t index) noexcept {
        return operands[index]->type() != nullptr && operands[index]->type()->is_uint32();
    };
    auto index32_at = [&](size_t index) noexcept {
        return index32_type(operands[index]->type());
    };
    auto integer_at = [&](size_t index) noexcept {
        return integer_scalar_type(operands[index]->type());
    };
    switch (op) {
        case ResourceWriteOp::BUFFER_WRITE:
        case ResourceWriteOp::BUFFER_VOLATILE_WRITE:
        case ResourceWriteOp::BYTE_BUFFER_WRITE:
        case ResourceWriteOp::BYTE_BUFFER_VOLATILE_WRITE: return integer_at(1u);
        case ResourceWriteOp::TEXTURE2D_WRITE: return uint_vector_type(operands[1]->type(), 2u);
        case ResourceWriteOp::TEXTURE3D_WRITE: return uint_vector_type(operands[1]->type(), 3u);
        case ResourceWriteOp::BINDLESS_BUFFER_WRITE:
        case ResourceWriteOp::BINDLESS_BYTE_BUFFER_WRITE: return index32_at(1u) && integer_at(2u);
        case ResourceWriteOp::DEVICE_ADDRESS_WRITE: return true;
        case ResourceWriteOp::RAY_TRACING_SET_INSTANCE_TRANSFORM:
            return index32_at(1u) && operands[2u]->type() == Type::matrix(4u);
        case ResourceWriteOp::RAY_TRACING_SET_INSTANCE_VISIBILITY_MASK:
        case ResourceWriteOp::RAY_TRACING_SET_INSTANCE_USER_ID:
            return index32_at(1u) && uint_at(2u);
        case ResourceWriteOp::RAY_TRACING_SET_INSTANCE_OPACITY:
            return index32_at(1u) && operands[2]->type() != nullptr && operands[2]->type()->is_bool();
        case ResourceWriteOp::RAY_TRACING_SET_INSTANCE_MOTION_MATRIX:
            return index32_at(1u) && index32_at(2u) && operands[3u]->type() == Type::matrix(4u);
        case ResourceWriteOp::RAY_TRACING_SET_INSTANCE_MOTION_SRT:
            return index32_at(1u) && index32_at(2u) && operands[3u]->type() == motion_srt_type();
        case ResourceWriteOp::INDIRECT_DISPATCH_SET_KERNEL:
            return uint_at(1u) && uint_vector_type(operands[2]->type(), 3u) &&
                   uint_vector_type(operands[3]->type(), 3u) && uint_at(4u);
        case ResourceWriteOp::INDIRECT_DISPATCH_SET_COUNT: return uint_at(1u);
    }
    return false;
}

template<typename OperandSpan>
[[nodiscard]] bool instruction_semantics_valid(
    DerivedInstructionTag tag, int64_t op, const Type *type,
    const OperandSpan &operands) noexcept {
    auto data_operands_valid = [&](size_t begin = 0u) noexcept {
        for (auto i = begin; i < operands.size(); i++) {
            if (!data_operand_valid(operands[i])) { return false; }
        }
        return true;
    };
    auto argument_matches = [](const Argument *formal, const Value *actual) noexcept {
        if (formal == nullptr || actual == nullptr || actual->type() != formal->type()) { return false; }
        if (formal->is_resource()) {
            return actual->isa<ResourceArgument>() && !actual->is_lvalue();
        }
        if (formal->is_reference()) {
            return typed_value_operand_valid(actual) && actual->is_lvalue();
        }
        return rvalue_operand_valid(actual);
    };
    auto value_is = []<typename T>(const Value *value) noexcept {
        return value != nullptr && value->template isa<T>();
    };
    switch (tag) {
        case DerivedInstructionTag::IF:
        case DerivedInstructionTag::CONDITIONAL_BRANCH:
            return type == nullptr && data_operand_valid(operands[0]) &&
                   operands[0]->type()->is_bool() &&
                   value_is.template operator()<BasicBlock>(operands[1]) &&
                   value_is.template operator()<BasicBlock>(operands[2]);
        case DerivedInstructionTag::BRANCH:
            return type == nullptr &&
                   value_is.template operator()<BasicBlock>(operands[0]);
        case DerivedInstructionTag::BREAK:
        case DerivedInstructionTag::CONTINUE:
            return type == nullptr && value_is.template operator()<BasicBlock>(operands[0]);
        case DerivedInstructionTag::SWITCH:
            if (type != nullptr || !data_operand_valid(operands[0]) ||
                !(integer_scalar_type(operands[0]->type()) ||
                  operands[0]->type()->is_bool())) {
                return false;
            }
            for (auto block : operands.subspan(1u)) {
                if (!value_is.template operator()<BasicBlock>(block)) { return false; }
            }
            return true;
        case DerivedInstructionTag::GEP:
            return type != nullptr && typed_value_operand_valid(operands[0]) &&
                   operands[0]->is_lvalue() &&
                   aggregate_indexed_type(operands[0]->type(), operands.subspan(1u)) == type;
        case DerivedInstructionTag::LOAD:
            return type != nullptr && typed_value_operand_valid(operands[0]) &&
                   operands[0]->is_lvalue() && operands[0]->type() == type;
        case DerivedInstructionTag::STORE:
            return type == nullptr && typed_value_operand_valid(operands[0]) &&
                   operands[0]->is_lvalue() && rvalue_operand_valid(operands[1]) &&
                   operands[0]->type() == operands[1]->type();
        case DerivedInstructionTag::RETURN:
            return type == nullptr &&
                   (operands[0] == nullptr || rvalue_operand_valid(operands[0]));
        case DerivedInstructionTag::PHI:
            if (type == nullptr) { return false; }
            for (auto operand : operands) {
                if (!rvalue_operand_valid(operand) || operand->type() != type) {
                    return false;
                }
            }
            return true;
        case DerivedInstructionTag::CORO_SUSPEND:
        case DerivedInstructionTag::CORO_RESUME:
            return type == nullptr &&
                   (operands[0] == nullptr || typed_value_operand_valid(operands[0]));
        case DerivedInstructionTag::ARITHMETIC:
            return arithmetic_types_valid(static_cast<ArithmeticOp>(op), type, operands);
        case DerivedInstructionTag::CAST:
            return cast_types_valid(static_cast<CastOp>(op), type, operands[0]);
        case DerivedInstructionTag::RESOURCE_QUERY: {
            if (!data_operands_valid(1u)) { return false; }
            auto query = static_cast<ResourceQueryOp>(op);
            return resource_query_base_type_valid(query, operands[0]) &&
                   resource_query_result_type_valid(query, type) &&
                   resource_query_operand_types_valid(query, operands);
        }
        case DerivedInstructionTag::RESOURCE_READ: {
            if (!data_operands_valid(1u)) { return false; }
            auto read = static_cast<ResourceReadOp>(op);
            if (type == nullptr || !resource_read_base_type_valid(read, operands[0]) ||
                !resource_read_operand_types_valid(read, operands)) {
                return false;
            }
            switch (read) {
                case ResourceReadOp::BUFFER_READ:
                case ResourceReadOp::BUFFER_VOLATILE_READ: {
                    auto element = operands[0]->type()->element();
                    return element == type;
                }
                case ResourceReadOp::BYTE_BUFFER_READ:
                case ResourceReadOp::BYTE_BUFFER_VOLATILE_READ:
                case ResourceReadOp::BINDLESS_BUFFER_READ:
                case ResourceReadOp::BINDLESS_BYTE_BUFFER_READ:
                case ResourceReadOp::DEVICE_ADDRESS_READ:
                    return !type->is_resource() && !type->is_custom();
                case ResourceReadOp::TEXTURE2D_READ:
                case ResourceReadOp::TEXTURE3D_READ:
                    return type == Type::vector(operands[0]->type()->element(), 4u);
                case ResourceReadOp::BINDLESS_TEXTURE2D_READ:
                case ResourceReadOp::BINDLESS_TEXTURE3D_READ:
                case ResourceReadOp::BINDLESS_TEXTURE2D_READ_LEVEL:
                case ResourceReadOp::BINDLESS_TEXTURE3D_READ_LEVEL: return float_vector_type(type, 4u);
                default: return true;
            }
        }
        case DerivedInstructionTag::RESOURCE_WRITE: {
            if (type != nullptr || !data_operands_valid(1u)) { return false; }
            if (operands[0] == nullptr || operands[0]->type() == nullptr) { return false; }
            auto write = static_cast<ResourceWriteOp>(op);
            auto base = operands[0]->type();
            if (!resource_write_operand_types_valid(write, operands)) { return false; }
            switch (write) {
                case ResourceWriteOp::BUFFER_WRITE:
                case ResourceWriteOp::BUFFER_VOLATILE_WRITE:
                    return resource_argument_valid(operands[0]) && base->is_buffer() &&
                           base->element() != nullptr && base->element() == operands[2]->type();
                case ResourceWriteOp::BYTE_BUFFER_WRITE:
                case ResourceWriteOp::BYTE_BUFFER_VOLATILE_WRITE:
                    return resource_argument_valid(operands[0]) && base->is_buffer() &&
                           base->element() == nullptr;
                case ResourceWriteOp::TEXTURE2D_WRITE:
                    return resource_argument_valid(operands[0]) && base->is_texture() && base->dimension() == 2u &&
                           operands[2]->type() == Type::vector(base->element(), 4u);
                case ResourceWriteOp::TEXTURE3D_WRITE:
                    return resource_argument_valid(operands[0]) && base->is_texture() && base->dimension() == 3u &&
                           operands[2]->type() == Type::vector(base->element(), 4u);
                case ResourceWriteOp::BINDLESS_BUFFER_WRITE:
                case ResourceWriteOp::BINDLESS_BYTE_BUFFER_WRITE:
                    return resource_argument_valid(operands[0]) && base->is_bindless_array();
                case ResourceWriteOp::DEVICE_ADDRESS_WRITE:
                    return data_operand_valid(operands[0]) && base->is_uint64();
                case ResourceWriteOp::RAY_TRACING_SET_INSTANCE_TRANSFORM:
                case ResourceWriteOp::RAY_TRACING_SET_INSTANCE_VISIBILITY_MASK:
                case ResourceWriteOp::RAY_TRACING_SET_INSTANCE_OPACITY:
                case ResourceWriteOp::RAY_TRACING_SET_INSTANCE_USER_ID:
                case ResourceWriteOp::RAY_TRACING_SET_INSTANCE_MOTION_MATRIX:
                case ResourceWriteOp::RAY_TRACING_SET_INSTANCE_MOTION_SRT:
                    return resource_argument_valid(operands[0]) && base->is_accel();
                case ResourceWriteOp::INDIRECT_DISPATCH_SET_KERNEL:
                case ResourceWriteOp::INDIRECT_DISPATCH_SET_COUNT:
                    return (resource_argument_valid(operands[0]) && base->is_buffer()) ||
                           (typed_value_operand_valid(operands[0]) && operands[0]->is_lvalue() &&
                            base == indirect_dispatch_buffer_type());
            }
            return false;
        }
        case DerivedInstructionTag::ATOMIC: {
            auto atomic = static_cast<AtomicOp>(op);
            auto value_count = atomic_op_value_count(atomic);
            auto index_count = operands.size() - 1u - value_count;
            auto base = operands[0];
            if (!atomic_value_type(type)) { return false; }
            auto indices = operands.subspan(1u, index_count);
            if (atomic_addressed_type(base, indices) != type) { return false; }
            for (auto i = operands.size() - value_count; i < operands.size(); i++) {
                if (!data_operand_valid(operands[i]) || operands[i]->type() != type) { return false; }
            }
            if ((atomic == AtomicOp::FETCH_AND || atomic == AtomicOp::FETCH_OR || atomic == AtomicOp::FETCH_XOR) &&
                !type->is_int() && !type->is_uint()) {
                return false;
            }
            return true;
        }
        case DerivedInstructionTag::THREAD_GROUP: {
            if (!data_operands_valid()) { return false; }
            auto thread_group = static_cast<ThreadGroupOp>(op);
            switch (thread_group) {
                case ThreadGroupOp::SHADER_EXECUTION_REORDER:
                    if (type != nullptr) { return false; }
                    for (auto operand : operands) {
                        if (!operand->type()->is_uint32()) { return false; }
                    }
                    return true;
                case ThreadGroupOp::SYNCHRONIZE_BLOCK: return type == nullptr;
                case ThreadGroupOp::RASTER_QUAD_DDX:
                case ThreadGroupOp::RASTER_QUAD_DDY:
                    return type == operands[0]->type() && type != nullptr && type->is_float_or_float_vector();
                case ThreadGroupOp::WARP_IS_FIRST_ACTIVE_LANE: return type != nullptr && type->is_bool();
                case ThreadGroupOp::WARP_FIRST_ACTIVE_LANE: return type != nullptr && type->is_uint32();
                case ThreadGroupOp::WARP_ACTIVE_ALL_EQUAL:
                    if (operands[0]->type()->is_scalar()) { return type != nullptr && type->is_bool(); }
                    return operands[0]->type()->is_vector() && type == Type::vector(Type::of<bool>(), operands[0]->type()->dimension());
                case ThreadGroupOp::WARP_ACTIVE_BIT_AND:
                case ThreadGroupOp::WARP_ACTIVE_BIT_OR:
                case ThreadGroupOp::WARP_ACTIVE_BIT_XOR:
                    return type == operands[0]->type() && type != nullptr && (type->is_int_or_int_vector() || type->is_uint_or_uint_vector());
                case ThreadGroupOp::WARP_ACTIVE_COUNT_BITS:
                case ThreadGroupOp::WARP_PREFIX_COUNT_BITS:
                    return operands[0]->type()->is_bool() && type != nullptr && type->is_uint32();
                case ThreadGroupOp::WARP_ACTIVE_MAX:
                case ThreadGroupOp::WARP_ACTIVE_MIN:
                case ThreadGroupOp::WARP_ACTIVE_PRODUCT:
                case ThreadGroupOp::WARP_ACTIVE_SUM:
                case ThreadGroupOp::WARP_PREFIX_SUM:
                case ThreadGroupOp::WARP_PREFIX_PRODUCT:
                    return type == operands[0]->type() && type != nullptr && type->is_scalar_or_vector() && !type->is_bool_or_bool_vector();
                case ThreadGroupOp::WARP_ACTIVE_ALL:
                case ThreadGroupOp::WARP_ACTIVE_ANY:
                    return operands[0]->type()->is_bool() && type != nullptr && type->is_bool();
                case ThreadGroupOp::WARP_ACTIVE_BIT_MASK:
                    return operands[0]->type()->is_bool() && uint_vector_type(type, 4u);
                case ThreadGroupOp::WARP_READ_LANE:
                    return type == operands[0]->type() && operands[1]->type()->is_uint32();
                case ThreadGroupOp::WARP_READ_FIRST_ACTIVE_LANE:
                    return type != nullptr && type == operands[0]->type();
            }
            return false;
        }
        case DerivedInstructionTag::RAY_QUERY_LOOP:
            return type == nullptr && value_is.template operator()<BasicBlock>(operands[0]);
        case DerivedInstructionTag::RAY_QUERY_DISPATCH:
            return type == nullptr && ray_query_object_valid(operands[0]) &&
                   value_is.template operator()<BasicBlock>(operands[1]) &&
                   value_is.template operator()<BasicBlock>(operands[2]) &&
                   value_is.template operator()<BasicBlock>(operands[3]);
        case DerivedInstructionTag::RAY_QUERY_OBJECT_READ: {
            if (!ray_query_object_valid(operands[0]) || type == nullptr) { return false; }
            switch (static_cast<RayQueryObjectReadOp>(op)) {
                case RayQueryObjectReadOp::RAY_QUERY_OBJECT_WORLD_SPACE_RAY: return type == ray_type();
                case RayQueryObjectReadOp::RAY_QUERY_OBJECT_PROCEDURAL_CANDIDATE_HIT: return type == procedural_hit_type();
                case RayQueryObjectReadOp::RAY_QUERY_OBJECT_TRIANGLE_CANDIDATE_HIT: return type == surface_hit_type();
                case RayQueryObjectReadOp::RAY_QUERY_OBJECT_COMMITTED_HIT: return type == committed_hit_type();
                case RayQueryObjectReadOp::RAY_QUERY_OBJECT_IS_TRIANGLE_CANDIDATE:
                case RayQueryObjectReadOp::RAY_QUERY_OBJECT_IS_PROCEDURAL_CANDIDATE:
                case RayQueryObjectReadOp::RAY_QUERY_OBJECT_IS_TERMINATED: return type->is_bool();
            }
            return false;
        }
        case DerivedInstructionTag::RAY_QUERY_OBJECT_WRITE:
            return type == nullptr && ray_query_object_valid(operands[0]) &&
                   (operands.size() == 1u ||
                    (data_operand_valid(operands[1]) && operands[1]->type()->is_float32()));
        case DerivedInstructionTag::RAY_QUERY_PIPELINE: {
            if (type != nullptr || !ray_query_object_valid(operands[0]) ||
                !value_is.template operator()<CallableFunction>(operands[1]) ||
                !value_is.template operator()<CallableFunction>(operands[2])) {
                return false;
            }
            auto captures = operands.subspan(3u);
            for (auto function_value : {operands[1], operands[2]}) {
                auto function = static_cast<const Function *>(function_value);
                if (function->type() != nullptr ||
                    function->arguments().count_size() != captures.size() + 1u) {
                    return false;
                }
                auto formal = function->arguments().begin();
                if (!(*formal)->is_reference() || (*formal)->type() != operands[0]->type()) { return false; }
                ++formal;
                for (auto capture : captures) {
                    if (!argument_matches(*formal, capture)) { return false; }
                    ++formal;
                }
            }
            return true;
        }
        case DerivedInstructionTag::AUTODIFF_SCOPE:
        case DerivedInstructionTag::OUTLINE:
            return type == nullptr && value_is.template operator()<BasicBlock>(operands[0]);
        case DerivedInstructionTag::AUTODIFF_INTRINSIC: {
            for (auto operand : operands) {
                if (!typed_value_operand_valid(operand)) { return false; }
            }
            switch (static_cast<AutodiffIntrinsicOp>(op)) {
                case AutodiffIntrinsicOp::AUTODIFF_REQUIRES_GRADIENT: return type == nullptr;
                case AutodiffIntrinsicOp::AUTODIFF_GRADIENT:
                case AutodiffIntrinsicOp::AUTODIFF_DETACH:
                    return type != nullptr && type == operands[0]->type();
                case AutodiffIntrinsicOp::AUTODIFF_GRADIENT_MARKER:
                case AutodiffIntrinsicOp::AUTODIFF_ACCUMULATE_GRADIENT:
                    return type == nullptr && operands[0]->type() == operands[1]->type();
                case AutodiffIntrinsicOp::AUTODIFF_BACKWARD: return type == nullptr;
                case AutodiffIntrinsicOp::AUTODIFF_PROPAGATE_GRADIENT:
                    if (type != nullptr) { return false; }
                    for (auto operand : operands.subspan(1u)) {
                        if (operand->type() != operands[0]->type()) { return false; }
                    }
                    return true;
                case AutodiffIntrinsicOp::AUTODIFF_OUTPUT_GRADIENT:
                    return type != nullptr && type == operands[0]->type() &&
                           index32_type(operands[1]->type());
            }
            return false;
        }
        case DerivedInstructionTag::CALL: {
            if (!value_is.template operator()<Function>(operands[0])) { return false; }
            auto callee = static_cast<const Function *>(operands[0]);
            if (type != callee->type() ||
                operands.size() != callee->arguments().count_size() + 1u) {
                return false;
            }
            auto actual = operands.begin() + 1u;
            for (auto formal : callee->arguments()) {
                if (!argument_matches(formal, *actual++)) { return false; }
            }
            return true;
        }
        case DerivedInstructionTag::PRINT:
        case DerivedInstructionTag::DEBUG_BREAK:
            return type == nullptr && data_operands_valid();
        case DerivedInstructionTag::ASSERT:
        case DerivedInstructionTag::ASSUME:
            return type == nullptr && data_operand_valid(operands[0]) &&
                   operands[0]->type()->is_bool();
        default: return true;
    }
}

[[nodiscard]] bool type_shape_valid(DerivedInstructionTag tag, const Type *type) noexcept {
    switch (tag) {
        case DerivedInstructionTag::PHI:
        case DerivedInstructionTag::ALLOCA:
        case DerivedInstructionTag::LOAD:
        case DerivedInstructionTag::GEP:
        case DerivedInstructionTag::ATOMIC:
        case DerivedInstructionTag::ARITHMETIC:
        case DerivedInstructionTag::RESOURCE_QUERY:
        case DerivedInstructionTag::RESOURCE_READ:
        case DerivedInstructionTag::RAY_QUERY_OBJECT_READ:
        case DerivedInstructionTag::CAST: return type != nullptr;
        case DerivedInstructionTag::CLOCK: return type == Type::of<uint64_t>();
        case DerivedInstructionTag::THREAD_GROUP:
        case DerivedInstructionTag::AUTODIFF_INTRINSIC: return true;
        case DerivedInstructionTag::RESOURCE_WRITE:
        case DerivedInstructionTag::RAY_QUERY_OBJECT_WRITE:
        case DerivedInstructionTag::RAY_QUERY_PIPELINE:
        case DerivedInstructionTag::AUTODIFF_SCOPE:
        case DerivedInstructionTag::BREAK:
        case DerivedInstructionTag::CONTINUE:
        case DerivedInstructionTag::PRINT:
        case DerivedInstructionTag::DEBUG_BREAK:
        case DerivedInstructionTag::ASSERT:
        case DerivedInstructionTag::ASSUME:
        case DerivedInstructionTag::OUTLINE: return type == nullptr;
        case DerivedInstructionTag::CALL: return true;
        default: return type == nullptr;
    }
}

[[nodiscard]] bool instruction_auxiliary_values_valid(
    const InstructionRecord &record) noexcept {
    auto all_nonnegative = [&]() noexcept {
        return std::all_of(record.auxiliary.begin(), record.auxiliary.end(),
                           [](int64_t value) noexcept { return value >= 0; });
    };
    switch (record.tag) {
        case DerivedInstructionTag::IF:
            return !record.auxiliary.empty() && record.auxiliary.front() >= -1;
        case DerivedInstructionTag::LOOP:
        case DerivedInstructionTag::SIMPLE_LOOP:
        case DerivedInstructionTag::PHI:
        case DerivedInstructionTag::RAY_QUERY_LOOP:
        case DerivedInstructionTag::OUTLINE: return all_nonnegative();
        case DerivedInstructionTag::SWITCH:
            return !record.auxiliary.empty() && record.auxiliary.front() >= -1;
        case DerivedInstructionTag::CORO_SUSPEND:
        case DerivedInstructionTag::CORO_RESUME:
            return record.auxiliary[0u] >= 0 &&
                   static_cast<uint64_t>(record.auxiliary[0u]) <= std::numeric_limits<uint32_t>::max();
        case DerivedInstructionTag::AUTODIFF_SCOPE: {
            auto merge = record.auxiliary[0u];
            auto forward = record.auxiliary[1u];
            auto n_forward_grads = record.auxiliary[2u];
            if (merge < 0 || (forward != 0 && forward != 1) || n_forward_grads < 0 ||
                static_cast<uint64_t>(n_forward_grads) > std::numeric_limits<size_t>::max()) {
                return false;
            }
            return forward == 0 ? n_forward_grads == 0 : n_forward_grads > 0;
        }
        default: return true;
    }
}

[[nodiscard]] Instruction *allocate_instruction(
    const InstructionRecord &record, BasicBlock *block,
    const Type *type, XIRBuilder &builder) noexcept {
    auto null_operands = luisa::vector<Value *>(record.operands.size(), nullptr);
    ManagedPtr<Instruction> instruction;
    switch (record.tag) {
        case DerivedInstructionTag::IF:
            instruction = luisa::make_managed<IfInst>(block, nullptr);
            break;
        case DerivedInstructionTag::SWITCH: {
            auto value = luisa::make_managed<SwitchInst>(block, nullptr);
            value->set_case_count(record.operands.size() - 2u);
            instruction = std::move(value);
            break;
        }
        case DerivedInstructionTag::LOOP:
            instruction = luisa::make_managed<LoopInst>(block);
            break;
        case DerivedInstructionTag::SIMPLE_LOOP:
            instruction = luisa::make_managed<SimpleLoopInst>(block);
            break;
        case DerivedInstructionTag::BRANCH:
            instruction = luisa::make_managed<BranchInst>(block);
            break;
        case DerivedInstructionTag::CONDITIONAL_BRANCH:
            instruction = luisa::make_managed<ConditionalBranchInst>(block, nullptr);
            break;
        case DerivedInstructionTag::UNREACHABLE:
            instruction = luisa::make_managed<UnreachableInst>(block, record.payloads[0u]);
            break;
        case DerivedInstructionTag::RASTER_DISCARD:
            instruction = luisa::make_managed<RasterDiscardInst>(block);
            break;
        case DerivedInstructionTag::CORO_SUSPEND:
            instruction = luisa::make_managed<CoroSuspendInst>(
                block, static_cast<uint32_t>(record.auxiliary[0u]),
                record.payloads[0u], nullptr);
            break;
        case DerivedInstructionTag::CORO_RESUME:
            instruction = luisa::make_managed<CoroResumeInst>(
                block, static_cast<uint32_t>(record.auxiliary[0u]), nullptr);
            break;
        case DerivedInstructionTag::CORO_TERMINATE:
            instruction = luisa::make_managed<CoroTerminateInst>(block);
            break;
        case DerivedInstructionTag::RETURN:
            instruction = luisa::make_managed<ReturnInst>(block, nullptr);
            break;
        case DerivedInstructionTag::PHI: {
            auto value = luisa::make_managed<PhiInst>(block, type);
            value->set_incoming_count(record.operands.size());
            instruction = std::move(value);
            break;
        }
        case DerivedInstructionTag::ALLOCA:
            instruction = luisa::make_managed<AllocaInst>(block, type, static_cast<AllocaOp>(record.op));
            break;
        case DerivedInstructionTag::LOAD:
            instruction = luisa::make_managed<LoadInst>(block, type, nullptr);
            break;
        case DerivedInstructionTag::STORE:
            instruction = luisa::make_managed<StoreInst>(block, nullptr, nullptr);
            break;
        case DerivedInstructionTag::GEP:
            instruction = luisa::make_managed<GEPInst>(block, type, nullptr, luisa::span<Value *const>{null_operands}.subspan(1u));
            break;
        case DerivedInstructionTag::ATOMIC: {
            auto op = static_cast<AtomicOp>(record.op);
            auto value_count = atomic_op_value_count(op);
            auto index_count = record.operands.size() - 1u - value_count;
            instruction = luisa::make_managed<AtomicInst>(block, type, op, nullptr,
                                                          luisa::span<Value *const>{null_operands}.subspan(1u, index_count));
            break;
        }
        case DerivedInstructionTag::ARITHMETIC:
            instruction = luisa::make_managed<ArithmeticInst>(block, type, static_cast<ArithmeticOp>(record.op), null_operands);
            break;
        case DerivedInstructionTag::THREAD_GROUP:
            instruction = luisa::make_managed<ThreadGroupInst>(block, type, static_cast<ThreadGroupOp>(record.op), null_operands);
            break;
        case DerivedInstructionTag::RESOURCE_QUERY:
            instruction = luisa::make_managed<ResourceQueryInst>(block, type, static_cast<ResourceQueryOp>(record.op), null_operands);
            break;
        case DerivedInstructionTag::RESOURCE_READ:
            instruction = luisa::make_managed<ResourceReadInst>(block, type, static_cast<ResourceReadOp>(record.op), null_operands);
            break;
        case DerivedInstructionTag::RESOURCE_WRITE:
            instruction = luisa::make_managed<ResourceWriteInst>(block, static_cast<ResourceWriteOp>(record.op), null_operands);
            break;
        case DerivedInstructionTag::RAY_QUERY_LOOP:
            instruction = luisa::make_managed<RayQueryLoopInst>(block);
            break;
        case DerivedInstructionTag::RAY_QUERY_DISPATCH:
            instruction = luisa::make_managed<RayQueryDispatchInst>(block, nullptr);
            break;
        case DerivedInstructionTag::RAY_QUERY_OBJECT_READ:
            instruction = luisa::make_managed<RayQueryObjectReadInst>(
                block, type, static_cast<RayQueryObjectReadOp>(record.op), null_operands);
            break;
        case DerivedInstructionTag::RAY_QUERY_OBJECT_WRITE:
            instruction = luisa::make_managed<RayQueryObjectWriteInst>(
                block, static_cast<RayQueryObjectWriteOp>(record.op), null_operands);
            break;
        case DerivedInstructionTag::RAY_QUERY_PIPELINE:
            instruction = luisa::make_managed<RayQueryPipelineInst>(
                block, nullptr, nullptr, nullptr,
                luisa::span<Value *const>{null_operands}.subspan(3u));
            break;
        case DerivedInstructionTag::AUTODIFF_SCOPE:
            instruction = luisa::make_managed<AutodiffScopeInst>(
                block, record.auxiliary[1u] != 0,
                static_cast<size_t>(record.auxiliary[2u]));
            break;
        case DerivedInstructionTag::AUTODIFF_INTRINSIC:
            instruction = luisa::make_managed<AutodiffIntrinsicInst>(
                block, type, static_cast<AutodiffIntrinsicOp>(record.op), null_operands);
            break;
        case DerivedInstructionTag::BREAK:
            instruction = luisa::make_managed<BreakInst>(block);
            break;
        case DerivedInstructionTag::CONTINUE:
            instruction = luisa::make_managed<ContinueInst>(block);
            break;
        case DerivedInstructionTag::CALL:
            instruction = luisa::make_managed<CallInst>(block, type, nullptr, luisa::span<Value *const>{null_operands}.subspan(1u));
            break;
        case DerivedInstructionTag::CAST:
            instruction = luisa::make_managed<CastInst>(block, type, static_cast<CastOp>(record.op), nullptr);
            break;
        case DerivedInstructionTag::PRINT:
            instruction = luisa::make_managed<PrintInst>(block, record.payloads[0u], null_operands);
            break;
        case DerivedInstructionTag::CLOCK:
            instruction = luisa::make_managed<ClockInst>(block);
            break;
        case DerivedInstructionTag::DEBUG_BREAK:
            instruction = luisa::make_managed<DebugBreakInst>(block, nullptr, null_operands);
            break;
        case DerivedInstructionTag::ASSERT:
            instruction = luisa::make_managed<AssertInst>(block, nullptr, record.payloads[0u]);
            break;
        case DerivedInstructionTag::ASSUME:
            instruction = luisa::make_managed<AssumeInst>(block, nullptr, record.payloads[0u]);
            break;
        case DerivedInstructionTag::OUTLINE:
            instruction = luisa::make_managed<OutlineInst>(block);
            break;
        default: return nullptr;
    }
    return builder.append(std::move(instruction));
}

[[nodiscard]] XIRInterchangeParseResult build_module(const ModuleRecord &record) noexcept {
    XIRInterchangeParseResult result;
    auto fail = [&](luisa::string message) noexcept {
        result.diagnostics.emplace_back(XIRInterchangeDiagnostic{.message = std::move(message)});
        return false;
    };
    auto module = luisa::make_unique<Module>();
    apply_metadata_records(record.metadata, *module);
    luisa::unordered_map<uint64_t, Value *> values;
    luisa::unordered_set<Value *> unique_values;
    auto register_value = [&](uint64_t id, Value *value) noexcept {
        if (id > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) || value == nullptr ||
            values.contains(id) || !unique_values.emplace(value).second) {
            return fail("Duplicate, aliased, or out-of-range XIR value identifier.");
        }
        values.emplace(id, value);
        return true;
    };
    auto resolve = [&](int64_t id) noexcept -> Value * {
        if (id < 0) { return nullptr; }
        if (auto iter = values.find(static_cast<uint64_t>(id)); iter != values.end()) { return iter->second; }
        return nullptr;
    };

    for (auto &&global : record.globals) {
        Value *value = nullptr;
        if (global.kind == GlobalRecord::Kind::SPECIAL) {
            auto tag = parse_special_register_name(global.payload);
            if (!tag) {
                fail("Unknown XIR special-register kind.");
                return result;
            }
            value = module->create_special_register(*tag);
        } else {
            auto type = parse_type(global.type);
            if (type == nullptr || type->is_resource()) {
                fail("Invalid type in XIR global record.");
                return result;
            }
            if (global.kind == GlobalRecord::Kind::UNDEFINED) {
                value = module->create_undefined(type);
            } else {
                if (type->is_custom() || type->is_cooperative_vector() ||
                    type->is_cooperative_vector_ref() ||
                    type->is_cooperative_matrix_ref()) {
                    fail("XIR constants cannot use opaque or cooperative types.");
                    return result;
                }
                size_t canonical_size = 0u;
                if (!canonical_constant_size(type, canonical_size) ||
                    canonical_size > max_payload_size) {
                    fail("XIR constant payload size does not match its type.");
                    return result;
                }
                luisa::vector<std::byte> bytes;
                if (global.payload_is_binary) {
                    if (!global.payload.empty() || global.binary_payload.size() != canonical_size) {
                        fail("XIR binary constant payload size does not match its type.");
                        return result;
                    }
                    bytes = global.binary_payload;
                } else {
                    if (canonical_size > std::numeric_limits<size_t>::max() / 2u ||
                        !global.binary_payload.empty() ||
                        global.payload.size() != canonical_size * 2u) {
                        fail("XIR constant payload size does not match its type.");
                        return result;
                    }
                    bytes.resize(canonical_size);
                    for (auto i = 0u; i < bytes.size(); i++) {
                        auto high = hex_value(global.payload[i * 2u]);
                        auto low = hex_value(global.payload[i * 2u + 1u]);
                        if (high < 0 || low < 0) {
                            fail("XIR constant contains invalid hexadecimal data.");
                            return result;
                        }
                        bytes[i] = static_cast<std::byte>((high << 4) | low);
                    }
                }
                luisa::vector<std::byte> native;
                if (!decode_canonical_constant(type, bytes, native)) {
                    fail("XIR constant contains invalid canonical data.");
                    return result;
                }
                value = native.empty() ?
                            static_cast<Value *>(module->create_constant_zero(type)) :
                            static_cast<Value *>(module->create_constant(
                                type, native.data()));
            }
        }
        if (!register_value(global.id, value)) { return result; }
        apply_metadata_records(global.metadata, *value);
    }

    luisa::vector<Function *> functions;
    functions.reserve(record.functions.size());
    for (auto &&function_record : record.functions) {
        auto return_type = parse_type(function_record.return_type);
        if (function_record.return_type != "void" && return_type == nullptr) {
            fail("Invalid XIR function return type.");
            return result;
        }
        Function *function = nullptr;
        if (function_record.kind == "kernel") {
            auto [x, y, z] = function_record.block_size;
            auto block_size = luisa::make_uint3(x, y, z);
            if (return_type != nullptr ||
                !KernelFunction::is_valid_block_size(block_size)) {
                fail("Invalid XIR kernel return type or block size.");
                return result;
            }
            auto kernel = module->create_kernel();
            kernel->set_block_size(block_size);
            function = kernel;
        } else if (function_record.kind == "callable") {
            if (function_record.block_size != std::array<uint32_t, 3u>{}) {
                fail("Callable function has a nonzero block size.");
                return result;
            }
            function = module->create_callable(return_type);
        } else if (function_record.kind == "external") {
            if (function_record.block_size != std::array<uint32_t, 3u>{} ||
                !function_record.blocks.empty() || function_record.body != -1 ||
                !function_record.instructions.empty()) {
                fail("External function contains a definition or block size.");
                return result;
            }
            function = module->create_external_function(return_type);
        } else {
            fail("Unknown XIR function kind.");
            return result;
        }
        if (!register_value(function_record.id, function)) { return result; }
        apply_metadata_records(function_record.metadata, *function);
        functions.emplace_back(function);
    }

    luisa::unordered_map<uint64_t, BasicBlock *> blocks;
    for (auto function_index = 0u; function_index < record.functions.size(); function_index++) {
        auto &&function_record = record.functions[function_index];
        auto function = functions[function_index];
        for (auto &&argument_record : function_record.arguments) {
            auto type = parse_type(argument_record.type);
            if (type == nullptr) {
                fail("Invalid XIR function argument type.");
                return result;
            }
            Argument *argument = nullptr;
            if (argument_record.kind == "value" && !type->is_resource() && !type->is_custom()) {
                argument = function->create_value_argument(type);
            } else if (argument_record.kind == "reference" && !type->is_resource()) {
                argument = function->create_reference_argument(type);
            } else if (argument_record.kind == "resource" && type->is_resource()) {
                argument = function->create_resource_argument(type);
            } else {
                fail("XIR argument kind is incompatible with its type.");
                return result;
            }
            if (!register_value(argument_record.id, argument)) { return result; }
            apply_metadata_records(argument_record.metadata, *argument);
        }
        for (auto &&block_record : function_record.blocks) {
            auto block = function->create_basic_block();
            if (!register_value(block_record.id, block) || blocks.contains(block_record.id)) { return result; }
            apply_metadata_records(block_record.metadata, *block);
            blocks.emplace(block_record.id, block);
        }
        if (auto definition = function->definition()) {
            if (function_record.body < 0) {
                fail("XIR function definition is missing a body block.");
                return result;
            }
            auto body_iter = blocks.find(static_cast<uint64_t>(function_record.body));
            if (body_iter == blocks.end() || body_iter->second->parent_function() != function) {
                fail("XIR function body block is missing or owned by another function.");
                return result;
            }
            definition->set_body_block(body_iter->second);
        }
    }

    luisa::unordered_map<const InstructionRecord *, Instruction *> instructions;
    XIRBuilder builder;
    for (auto function_index = 0u; function_index < record.functions.size(); function_index++) {
        auto &&function_record = record.functions[function_index];
        auto function = functions[function_index];
        for (auto &&instruction_record : function_record.instructions) {
            auto block_iter = blocks.find(instruction_record.block_id);
            if (block_iter == blocks.end() || block_iter->second->parent_function() != function) {
                fail("XIR instruction references a missing or foreign basic block.");
                return result;
            }
            if (!valid_op(instruction_record.tag, instruction_record.op) ||
                !instruction_operand_count_valid(instruction_record.tag, instruction_record.op,
                                                 instruction_record.operands.size(), instruction_record.auxiliary.size(),
                                                 instruction_record.payloads.size()) ||
                !instruction_auxiliary_values_valid(instruction_record)) {
                fail("XIR instruction has an invalid operand, auxiliary, or opcode layout.");
                return result;
            }
            auto type = parse_type(instruction_record.type);
            if ((instruction_record.type != "void" && type == nullptr) ||
                !type_shape_valid(instruction_record.tag, type)) {
                fail("XIR instruction has an invalid result type.");
                return result;
            }
            builder.set_insertion_point(block_iter->second);
            auto instruction = allocate_instruction(instruction_record, block_iter->second, type, builder);
            if (instruction == nullptr || !register_value(instruction_record.id, instruction)) { return result; }
            apply_metadata_records(instruction_record.metadata, *instruction);
            instructions.emplace(&instruction_record, instruction);
        }
    }

    for (auto function_index = 0u; function_index < record.functions.size(); function_index++) {
        auto &&function_record = record.functions[function_index];
        auto function = functions[function_index];
        for (auto &&instruction_record : function_record.instructions) {
            auto instruction = instructions.at(&instruction_record);
            luisa::vector<Value *> resolved_operands;
            resolved_operands.reserve(instruction_record.operands.size());
            for (auto operand_id : instruction_record.operands) {
                auto operand = resolve(operand_id);
                if (operand_id >= 0 && operand == nullptr) {
                    fail("XIR instruction references an unknown value identifier.");
                    return result;
                }
                resolved_operands.emplace_back(operand);
            }
            for (auto operand : resolved_operands) {
                if (operand == nullptr) { continue; }
                if ((operand->isa<BasicBlock>() && static_cast<BasicBlock *>(operand)->parent_function() != function) ||
                    (operand->isa<Argument>() && static_cast<Argument *>(operand)->parent_function() != function) ||
                    (operand->isa<Instruction>() && static_cast<Instruction *>(operand)->parent_function() != function)) {
                    fail("XIR instruction references a local value from another function.");
                    return result;
                }
            }
            if (!instruction_semantics_valid(instruction_record.tag, instruction_record.op,
                                             instruction->type(), luisa::span{resolved_operands})) {
                fail("XIR instruction operands or result type do not match its operation.");
                return result;
            }
            if (instruction_record.tag == DerivedInstructionTag::ARITHMETIC) {
                for (auto operand : resolved_operands) {
                    if (operand == nullptr || operand->type() == nullptr) {
                        fail("XIR arithmetic instruction has a null or typeless operand.");
                        return result;
                    }
                }
            }
            if (instruction_record.tag == DerivedInstructionTag::PHI) {
                auto phi = static_cast<PhiInst *>(instruction);
                for (auto i = 0u; i < resolved_operands.size(); i++) {
                    auto block_id = instruction_record.auxiliary[i];
                    auto block_iter = block_id < 0 ? blocks.end() : blocks.find(static_cast<uint64_t>(block_id));
                    if (block_iter == blocks.end() || block_iter->second->parent_function() != function) {
                        fail("XIR PHI references an unknown or foreign incoming block.");
                        return result;
                    }
                    phi->set_incoming(i, resolved_operands[i], block_iter->second);
                }
            } else {
                for (auto i = 0u; i < resolved_operands.size(); i++) {
                    instruction->set_operand(i, resolved_operands[i]);
                }
            }
            auto resolve_block = [&](int64_t id) noexcept -> BasicBlock * {
                if (id < 0) { return nullptr; }
                auto iter = blocks.find(static_cast<uint64_t>(id));
                return iter == blocks.end() || iter->second->parent_function() != function ? nullptr : iter->second;
            };
            switch (instruction_record.tag) {
                case DerivedInstructionTag::IF: {
                    auto merge_id = instruction_record.auxiliary[0u];
                    auto merge = resolve_block(merge_id);
                    if (merge_id >= 0 && merge == nullptr) {
                        fail("XIR if instruction has an invalid merge block.");
                        return result;
                    }
                    static_cast<IfInst *>(instruction)->set_merge_block(merge);
                    break;
                }
                case DerivedInstructionTag::SWITCH: {
                    auto switch_inst = static_cast<SwitchInst *>(instruction);
                    auto merge_id = instruction_record.auxiliary[0u];
                    auto merge = resolve_block(merge_id);
                    if (merge_id >= 0 && merge == nullptr) {
                        fail("XIR switch instruction has an invalid merge block.");
                        return result;
                    }
                    switch_inst->set_merge_block(merge);
                    for (auto i = 0u; i < switch_inst->case_count(); i++) {
                        auto case_value = decode_switch_case_value(
                            switch_inst->value()->type(),
                            instruction_record.auxiliary[i + 1u]);
                        if (!case_value) {
                            fail("XIR switch case value is outside the selector type range.");
                            return result;
                        }
                        switch_inst->set_case_value(i, *case_value);
                    }
                    break;
                }
                case DerivedInstructionTag::LOOP: {
                    auto loop = static_cast<LoopInst *>(instruction);
                    auto body = resolve_block(instruction_record.auxiliary[0u]);
                    auto update = resolve_block(instruction_record.auxiliary[1u]);
                    auto merge = resolve_block(instruction_record.auxiliary[2u]);
                    if (body == nullptr || update == nullptr || merge == nullptr) {
                        fail("XIR loop instruction has an invalid structured block.");
                        return result;
                    }
                    loop->set_body_block(body);
                    loop->set_update_block(update);
                    loop->set_merge_block(merge);
                    break;
                }
                case DerivedInstructionTag::SIMPLE_LOOP: {
                    auto merge = resolve_block(instruction_record.auxiliary[0u]);
                    if (merge == nullptr) {
                        fail("XIR simple-loop instruction has an invalid merge block.");
                        return result;
                    }
                    static_cast<SimpleLoopInst *>(instruction)->set_merge_block(merge);
                    break;
                }
                case DerivedInstructionTag::RAY_QUERY_LOOP: {
                    auto merge = resolve_block(instruction_record.auxiliary[0u]);
                    if (merge == nullptr) {
                        fail("XIR ray-query loop instruction has an invalid merge block.");
                        return result;
                    }
                    static_cast<RayQueryLoopInst *>(instruction)->set_merge_block(merge);
                    break;
                }
                case DerivedInstructionTag::AUTODIFF_SCOPE: {
                    auto merge = resolve_block(instruction_record.auxiliary[0u]);
                    if (merge == nullptr) {
                        fail("XIR autodiff-scope instruction has an invalid merge block.");
                        return result;
                    }
                    static_cast<AutodiffScopeInst *>(instruction)->set_merge_block(merge);
                    break;
                }
                case DerivedInstructionTag::OUTLINE: {
                    auto merge = resolve_block(instruction_record.auxiliary[0u]);
                    if (merge == nullptr) {
                        fail("XIR outline instruction has an invalid merge block.");
                        return result;
                    }
                    static_cast<OutlineInst *>(instruction)->set_merge_block(merge);
                    break;
                }
                default: break;
            }
            if (instruction_record.tag == DerivedInstructionTag::CALL) {
                auto callee = resolved_operands[0u];
                if (callee == nullptr || !callee->isa<Function>()) {
                    fail("XIR call instruction has an invalid callee.");
                    return result;
                }
            }
        }
    }

    auto verification = xir_verify_module(module.get());
    if (!verification.succeeded()) {
        for (auto &&error : verification.errors) {
            result.diagnostics.emplace_back(XIRInterchangeDiagnostic{
                .message = luisa::format("XIR verification failed after decoding: {}", error.message)});
        }
        return result;
    }
    result.module = std::move(module);
    return result;
}

}// namespace

bool detail::interchange_instruction_semantics_valid(
    DerivedInstructionTag tag, int64_t op, const Type *type,
    luisa::span<const Value *const> operands) noexcept {
    return instruction_semantics_valid(tag, op, type, operands);
}

XIRInterchangeTextWriteResult xir_to_interchange_text(const Module *module) noexcept {
    XIRInterchangeTextWriteResult result;
    auto fail = [&](luisa::string message) noexcept {
        result.text.clear();
        result.diagnostics.emplace_back(XIRInterchangeDiagnostic{.message = std::move(message)});
        return false;
    };
    if (module == nullptr) {
        fail("Cannot serialize a null XIR module.");
        return result;
    }
    auto verification = xir_verify_module(module);
    if (!verification.succeeded()) {
        for (auto &&error : verification.errors) {
            result.diagnostics.emplace_back(XIRInterchangeDiagnostic{
                .message = luisa::format("Cannot serialize malformed XIR: {}", error.message)});
        }
        return result;
    }
    auto writer_records_remaining = max_record_count;
    auto consume_writer_records = [&](size_t count) noexcept {
        if (count > writer_records_remaining) {
            return fail("Cumulative XIR record count exceeds the supported output budget.");
        }
        writer_records_remaining -= count;
        return true;
    };
    auto global_count = size_t{0u};
    for (auto count : {module->constant_list().count_size(),
                       module->undefined_list().count_size(),
                       module->special_register_list().count_size()}) {
        if (count > max_record_count - global_count || !consume_writer_records(count)) {
            if (result.diagnostics.empty()) { fail("XIR global count exceeds the supported limit."); }
            return result;
        }
        global_count += count;
    }
    if (!consume_writer_records(module->function_list().count_size())) { return result; }
    for (auto function : module->function_list()) {
        if (!consume_writer_records(function->arguments().count_size()) ||
            !consume_writer_records(function->basic_blocks().count_size())) {
            return result;
        }
        auto instruction_count = size_t{0u};
        for (auto block : function->basic_blocks()) {
            auto count = block->instructions().count_size();
            if (count > max_record_count - instruction_count) {
                fail("XIR instruction count exceeds the supported limit.");
                return result;
            }
            instruction_count += count;
        }
        if (!consume_writer_records(instruction_count)) { return result; }
    }
    luisa::unordered_map<const Value *, uint64_t> ids;
    auto assign_id = [&](const Value *value) noexcept {
        if (value == nullptr || ids.contains(value)) { return false; }
        ids.emplace(value, static_cast<uint64_t>(ids.size()));
        return true;
    };
    for (auto value : module->constant_list()) { assign_id(value); }
    for (auto value : module->undefined_list()) { assign_id(value); }
    for (auto value : module->special_register_list()) { assign_id(value); }
    for (auto function : module->function_list()) { assign_id(function); }
    for (auto function : module->function_list()) {
        for (auto argument : function->arguments()) { assign_id(argument); }
        for (auto block : function->basic_blocks()) {
            assign_id(block);
            for (auto instruction : block->instructions()) { assign_id(instruction); }
        }
    }
    auto id = [&](const Value *value) noexcept -> luisa::optional<uint64_t> {
        if (auto iter = ids.find(value); iter != ids.end()) { return iter->second; }
        return luisa::nullopt;
    };
    auto append_type = [&](const Type *type) noexcept {
        append_quoted(result.text, type == nullptr ? luisa::string_view{"void"} : type->description());
    };
    auto append_metadata = [&](const MetadataListMixin &owner, luisa::string_view indentation) noexcept {
        if (!consume_writer_records(owner.metadata_list().count_size())) { return false; }
        luisa::string error;
        if (append_metadata_records(result.text, owner, indentation, error)) { return true; }
        fail(std::move(error));
        return false;
    };
    result.text = "; LuisaCompute XIR canonical interchange\n"
                  "xir.text 1\n"
                  "module {\n";
    if (!append_metadata(*module, "  ")) { return result; }
    luisa::format_to(std::back_inserter(result.text), "  globals {}\n", global_count);
    constexpr auto hex = "0123456789abcdef";
    for (auto value : module->constant_list()) {
        if (value->type() == nullptr || value->type()->is_resource() ||
            value->type()->is_custom() || value->type()->is_cooperative_vector() ||
            value->type()->is_cooperative_vector_ref() ||
            value->type()->is_cooperative_matrix_ref() || !round_trippable_type(value->type())) {
            fail("XIR interchange cannot serialize this constant.");
            return result;
        }
        size_t canonical_size = 0u;
        luisa::vector<std::byte> bytes;
        if (!canonical_constant_size(value->type(), canonical_size) ||
            canonical_size > max_payload_size ||
            !encode_canonical_constant(value->type(), value->data(), bytes)) {
            fail("XIR constant payload exceeds the supported size limit.");
            return result;
        }
        luisa::format_to(std::back_inserter(result.text), "  constant {} ", *id(value));
        append_type(value->type());
        result.text.append(" \"");
        for (auto byte_value : bytes) {
            auto byte = std::to_integer<uint8_t>(byte_value);
            result.text.push_back(hex[byte >> 4u]);
            result.text.push_back(hex[byte & 0x0fu]);
        }
        result.text.append("\"\n");
        if (!append_metadata(*value, "    ")) { return result; }
    }
    for (auto value : module->undefined_list()) {
        if (value->type() == nullptr || value->type()->is_resource() ||
            !round_trippable_type(value->type())) {
            fail("XIR interchange cannot serialize this undefined value.");
            return result;
        }
        luisa::format_to(std::back_inserter(result.text), "  undefined {} ", *id(value));
        append_type(value->type());
        result.text.push_back('\n');
        if (!append_metadata(*value, "    ")) { return result; }
    }
    for (auto value : module->special_register_list()) {
        luisa::format_to(std::back_inserter(result.text), "  special {} {}\n", *id(value), to_string(value->derived_special_register_tag()));
        if (!append_metadata(*value, "    ")) { return result; }
    }
    luisa::format_to(std::back_inserter(result.text), "  functions {}\n", module->function_list().count_size());
    for (auto function : module->function_list()) {
        if (function->isa<ExternalFunction>() &&
            (!function->basic_blocks().empty() ||
             function->definition() != nullptr)) {
            fail("XIR external function contains a definition.");
            return result;
        }
        if (!round_trippable_type(function->type())) {
            fail("XIR interchange cannot serialize this function return type.");
            return result;
        }
        auto kind = to_string(function->derived_function_tag());
        auto block_size = function->isa<KernelFunction>() ? static_cast<const KernelFunction *>(function)->block_size() : luisa::make_uint3(0u);
        luisa::format_to(std::back_inserter(result.text), "  function {} {} ", *id(function), kind);
        append_type(function->type());
        luisa::format_to(std::back_inserter(result.text), " {} {} {} {{\n", block_size.x, block_size.y, block_size.z);
        if (!append_metadata(*function, "    ")) { return result; }
        luisa::format_to(std::back_inserter(result.text), "    arguments {}\n", function->arguments().count_size());
        for (auto argument : function->arguments()) {
            if (argument->type() == nullptr ||
                !round_trippable_type(argument->type())) {
                fail("XIR interchange cannot serialize this argument.");
                return result;
            }
            auto argument_kind = argument->is_value() ? "value" : argument->is_reference() ? "reference" :
                                                                                             "resource";
            luisa::format_to(std::back_inserter(result.text), "    argument {} {} ", *id(argument), argument_kind);
            append_type(argument->type());
            result.text.push_back('\n');
            if (!append_metadata(*argument, "      ")) { return result; }
        }
        luisa::format_to(std::back_inserter(result.text), "    blocks {}\n", function->basic_blocks().count_size());
        for (auto block : function->basic_blocks()) {
            luisa::format_to(std::back_inserter(result.text), "    block {}\n", *id(block));
            if (!append_metadata(*block, "      ")) { return result; }
        }
        auto definition = function->definition();
        if (definition != nullptr && definition->body_block() == nullptr) {
            fail("XIR function definition has no body block.");
            return result;
        }
        luisa::format_to(std::back_inserter(result.text), "    body {}\n", definition == nullptr ? -1ll : static_cast<int64_t>(*id(definition->body_block())));
        size_t instruction_count = 0u;
        for (auto block : function->basic_blocks()) { instruction_count += block->instructions().count_size(); }
        luisa::format_to(std::back_inserter(result.text), "    instructions {}\n", instruction_count);
        for (auto block : function->basic_blocks()) {
            for (auto instruction : block->instructions()) {
                auto name = instruction_name(instruction->derived_instruction_tag());
                if (!name ||
                    !round_trippable_type(instruction->type())) {
                    fail("XIR interchange v1 encountered an unsupported instruction.");
                    return result;
                }
                auto op = int64_t{-1};
                switch (instruction->derived_instruction_tag()) {
                    case DerivedInstructionTag::ALLOCA: op = static_cast<int64_t>(static_cast<const AllocaInst *>(instruction)->op()); break;
                    case DerivedInstructionTag::ATOMIC: op = static_cast<int64_t>(static_cast<const AtomicInst *>(instruction)->op()); break;
                    case DerivedInstructionTag::ARITHMETIC: op = static_cast<int64_t>(static_cast<const ArithmeticInst *>(instruction)->op()); break;
                    case DerivedInstructionTag::THREAD_GROUP: op = static_cast<int64_t>(static_cast<const ThreadGroupInst *>(instruction)->op()); break;
                    case DerivedInstructionTag::RESOURCE_QUERY: op = static_cast<int64_t>(static_cast<const ResourceQueryInst *>(instruction)->op()); break;
                    case DerivedInstructionTag::RESOURCE_READ: op = static_cast<int64_t>(static_cast<const ResourceReadInst *>(instruction)->op()); break;
                    case DerivedInstructionTag::RESOURCE_WRITE: op = static_cast<int64_t>(static_cast<const ResourceWriteInst *>(instruction)->op()); break;
                    case DerivedInstructionTag::RAY_QUERY_OBJECT_READ: op = static_cast<int64_t>(static_cast<const RayQueryObjectReadInst *>(instruction)->op()); break;
                    case DerivedInstructionTag::RAY_QUERY_OBJECT_WRITE: op = static_cast<int64_t>(static_cast<const RayQueryObjectWriteInst *>(instruction)->op()); break;
                    case DerivedInstructionTag::AUTODIFF_INTRINSIC: op = static_cast<int64_t>(static_cast<const AutodiffIntrinsicInst *>(instruction)->op()); break;
                    case DerivedInstructionTag::CAST: op = static_cast<int64_t>(static_cast<const CastInst *>(instruction)->op()); break;
                    case DerivedInstructionTag::DEBUG_BREAK: {
                        if (static_cast<const DebugBreakInst *>(instruction)->callback() != nullptr) {
                            fail("XIR interchange cannot serialize a debug-break callback.");
                            return result;
                        }
                        op = 0;
                        break;
                    }
                    default: break;
                }
                luisa::vector<int64_t> auxiliary;
                switch (instruction->derived_instruction_tag()) {
                    case DerivedInstructionTag::IF: {
                        auto merge = static_cast<const IfInst *>(instruction)->merge_block();
                        if (merge != nullptr && !id(merge)) {
                            fail("XIR if instruction has an unowned merge block.");
                            return result;
                        }
                        auxiliary.emplace_back(merge == nullptr ?
                                                   -1ll :
                                                   static_cast<int64_t>(*id(merge)));
                        break;
                    }
                    case DerivedInstructionTag::SWITCH: {
                        auto switch_inst = static_cast<const SwitchInst *>(instruction);
                        auto merge = switch_inst->merge_block();
                        if (merge != nullptr && !id(merge)) {
                            fail("XIR switch instruction has an unowned merge block.");
                            return result;
                        }
                        auxiliary.emplace_back(merge == nullptr ?
                                                   -1ll :
                                                   static_cast<int64_t>(*id(merge)));
                        for (auto case_value : switch_inst->case_values()) {
                            auxiliary.emplace_back(encode_switch_case_value(
                                switch_inst->value()->type(), case_value));
                        }
                        break;
                    }
                    case DerivedInstructionTag::LOOP: {
                        auto loop = static_cast<const LoopInst *>(instruction);
                        if (loop->body_block() == nullptr || loop->update_block() == nullptr || loop->merge_block() == nullptr ||
                            !id(loop->body_block()) || !id(loop->update_block()) || !id(loop->merge_block())) {
                            fail("XIR loop instruction has an unowned structured block.");
                            return result;
                        }
                        auxiliary.emplace_back(static_cast<int64_t>(*id(loop->body_block())));
                        auxiliary.emplace_back(static_cast<int64_t>(*id(loop->update_block())));
                        auxiliary.emplace_back(static_cast<int64_t>(*id(loop->merge_block())));
                        break;
                    }
                    case DerivedInstructionTag::SIMPLE_LOOP: {
                        auto merge = static_cast<const SimpleLoopInst *>(instruction)->merge_block();
                        if (merge == nullptr || !id(merge)) {
                            fail("XIR simple-loop instruction has an unowned merge block.");
                            return result;
                        }
                        auxiliary.emplace_back(static_cast<int64_t>(*id(merge)));
                        break;
                    }
                    case DerivedInstructionTag::PHI: {
                        auto phi = static_cast<const PhiInst *>(instruction);
                        for (auto incoming_block : phi->incoming_blocks()) {
                            if (incoming_block == nullptr || !id(incoming_block)) {
                                fail("XIR PHI instruction has an unowned incoming block.");
                                return result;
                            }
                            auxiliary.emplace_back(static_cast<int64_t>(*id(incoming_block)));
                        }
                        break;
                    }
                    case DerivedInstructionTag::CORO_SUSPEND:
                        auxiliary.emplace_back(static_cast<const CoroSuspendInst *>(instruction)->token());
                        break;
                    case DerivedInstructionTag::CORO_RESUME:
                        auxiliary.emplace_back(static_cast<const CoroResumeInst *>(instruction)->token());
                        break;
                    case DerivedInstructionTag::RAY_QUERY_LOOP: {
                        auto merge = static_cast<const RayQueryLoopInst *>(instruction)->merge_block();
                        if (merge == nullptr || !id(merge)) {
                            fail("XIR ray-query loop instruction has an unowned merge block.");
                            return result;
                        }
                        auxiliary.emplace_back(static_cast<int64_t>(*id(merge)));
                        break;
                    }
                    case DerivedInstructionTag::AUTODIFF_SCOPE: {
                        auto scope = static_cast<const AutodiffScopeInst *>(instruction);
                        auto merge = scope->merge_block();
                        if (merge == nullptr || !id(merge) ||
                            scope->n_forward_grads() > static_cast<size_t>(std::numeric_limits<int64_t>::max())) {
                            fail("XIR autodiff scope has invalid or unowned auxiliary state.");
                            return result;
                        }
                        auxiliary.emplace_back(static_cast<int64_t>(*id(merge)));
                        auxiliary.emplace_back(scope->is_forward() ? 1 : 0);
                        auxiliary.emplace_back(static_cast<int64_t>(scope->n_forward_grads()));
                        break;
                    }
                    case DerivedInstructionTag::OUTLINE: {
                        auto merge = static_cast<const OutlineInst *>(instruction)->merge_block();
                        if (merge == nullptr || !id(merge)) {
                            fail("XIR outline instruction has an unowned merge block.");
                            return result;
                        }
                        auxiliary.emplace_back(static_cast<int64_t>(*id(merge)));
                        break;
                    }
                    default: break;
                }
                luisa::vector<luisa::string_view> payloads;
                switch (instruction->derived_instruction_tag()) {
                    case DerivedInstructionTag::UNREACHABLE:
                        payloads.emplace_back(static_cast<const UnreachableInst *>(instruction)->message());
                        break;
                    case DerivedInstructionTag::CORO_SUSPEND:
                        payloads.emplace_back(static_cast<const CoroSuspendInst *>(instruction)->name());
                        break;
                    case DerivedInstructionTag::PRINT:
                        payloads.emplace_back(static_cast<const PrintInst *>(instruction)->format());
                        break;
                    case DerivedInstructionTag::ASSERT:
                        payloads.emplace_back(static_cast<const AssertInst *>(instruction)->message());
                        break;
                    case DerivedInstructionTag::ASSUME:
                        payloads.emplace_back(static_cast<const AssumeInst *>(instruction)->message());
                        break;
                    default: break;
                }
                for (auto payload : payloads) {
                    if (payload.size() > max_string_payload_size) {
                        fail("XIR instruction string payload exceeds the supported limit.");
                        return result;
                    }
                }
                if (!valid_op(instruction->derived_instruction_tag(), op) ||
                    !instruction_operand_count_valid(instruction->derived_instruction_tag(), op,
                                                     instruction->operand_count(), auxiliary.size(), payloads.size())) {
                    fail("XIR instruction has an unsupported operand or opcode layout.");
                    return result;
                }
                if (!consume_writer_records(instruction->operand_count()) ||
                    !consume_writer_records(auxiliary.size()) ||
                    !consume_writer_records(payloads.size())) {
                    return result;
                }
                auto op_name = instruction_op_name(instruction->derived_instruction_tag(), op);
                if (!op_name) {
                    fail("XIR instruction has an unsupported operation.");
                    return result;
                }
                luisa::vector<const Value *> semantic_operands;
                semantic_operands.reserve(instruction->operand_count());
                for (auto operand_use : instruction->operand_uses()) {
                    semantic_operands.emplace_back(operand_use->value());
                }
                if (!instruction_semantics_valid(instruction->derived_instruction_tag(), op,
                                                 instruction->type(), luisa::span{semantic_operands})) {
                    fail("XIR instruction operands or result type do not match its operation.");
                    return result;
                }
                luisa::format_to(std::back_inserter(result.text), "    instruction {} {} {} ", *id(instruction), *id(block), *name);
                append_type(instruction->type());
                luisa::format_to(std::back_inserter(result.text), " {} {}", *op_name, instruction->operand_count());
                for (auto operand : semantic_operands) {
                    if (operand == nullptr) {
                        result.text.append(" -1");
                    } else if (auto operand_id = id(operand)) {
                        luisa::format_to(std::back_inserter(result.text), " {}", *operand_id);
                    } else {
                        fail("XIR instruction references a value outside the module.");
                        return result;
                    }
                }
                luisa::format_to(std::back_inserter(result.text), " {}", auxiliary.size());
                for (auto value : auxiliary) { luisa::format_to(std::back_inserter(result.text), " {}", value); }
                if (!payloads.empty()) {
                    luisa::format_to(std::back_inserter(result.text), " payloads {}", payloads.size());
                    for (auto payload : payloads) {
                        result.text.push_back(' ');
                        append_quoted(result.text, payload);
                    }
                }
                result.text.push_back('\n');
                if (!append_metadata(*instruction, "      ")) { return result; }
            }
        }
        result.text.append("  }\n");
    }
    result.text.append("}\n");
    if (result.text.size() > max_payload_size) {
        fail("XIR text payload exceeds the supported size limit.");
    }
    return result;
}

XIRInterchangeParseResult xir_from_interchange_text(luisa::string_view text) noexcept {
    XIRInterchangeParseResult result;
    if (text.size() > max_payload_size) {
        result.diagnostics.emplace_back(diagnostic_at(text, 0u, "XIR text payload exceeds the supported size limit."));
        return result;
    }
    TextParser parser{text, result.diagnostics};
    ModuleRecord record;
    if (!parse_module_record(parser, record)) {
        if (result.diagnostics.empty()) {
            result.diagnostics.emplace_back(diagnostic_at(text, parser.offset(), "Malformed XIR interchange text."));
        }
        return result;
    }
    return build_module(record);
}

XIRInterchangeBitcodeWriteResult xir_to_bitcode(const Module *module) noexcept {
    XIRInterchangeBitcodeWriteResult result;
    auto text_result = xir_to_interchange_text(module);
    if (!text_result) {
        result.diagnostics = std::move(text_result.diagnostics);
        return result;
    }
    ModuleRecord record;
    TextParser parser{text_result.text, result.diagnostics};
    if (!parse_module_record(parser, record)) {
        if (result.diagnostics.empty()) {
            result.diagnostics.emplace_back(XIRInterchangeDiagnostic{
                .message = "Internal error while lowering XIR to its binary module record."});
        }
        return result;
    }
    luisa::vector<std::byte> payload;
    luisa::string error;
    if (!encode_binary_module_record(record, payload, error)) {
        result.diagnostics.emplace_back(XIRInterchangeDiagnostic{.message = std::move(error)});
        return result;
    }
    result.bitcode.reserve(bitcode_header_size + payload.size());
    result.bitcode.insert(result.bitcode.end(), bitcode_magic.begin(), bitcode_magic.end());
    append_u32(result.bitcode, bitcode_version);
    append_u32(result.bitcode, 0u);
    append_u64(result.bitcode, payload.size());
    append_u64(result.bitcode, checksum(payload));
    result.bitcode.insert(result.bitcode.end(), payload.begin(), payload.end());
    return result;
}

XIRInterchangeParseResult xir_from_bitcode(luisa::span<const std::byte> bitcode) noexcept {
    XIRInterchangeParseResult result;
    auto fail = [&](luisa::string message, size_t offset = 0u) noexcept {
        result.diagnostics.emplace_back(XIRInterchangeDiagnostic{
            .offset = offset,
            .message = std::move(message)});
    };
    if (bitcode.size() < bitcode_header_size) {
        fail("Truncated XIR bitcode header.", bitcode.size());
        return result;
    }
    for (auto i = 0u; i < bitcode_magic.size(); i++) {
        if (bitcode[i] != bitcode_magic[i]) {
            fail("Invalid XIR bitcode magic.", i);
            return result;
        }
    }
    auto version = read_u32(bitcode, 8u);
    if (version != interchange_version && version != bitcode_version) {
        fail("Unsupported XIR bitcode version.", 8u);
        return result;
    }
    if (read_u32(bitcode, 12u) != 0u) {
        fail("XIR bitcode reserved header bits are nonzero.", 12u);
        return result;
    }
    auto payload_size_u64 = read_u64(bitcode, 16u);
    if (payload_size_u64 > max_payload_size || payload_size_u64 > std::numeric_limits<size_t>::max()) {
        fail("XIR bitcode payload exceeds the supported size limit.", 16u);
        return result;
    }
    auto payload_size = static_cast<size_t>(payload_size_u64);
    if (payload_size > bitcode.size() - bitcode_header_size) {
        fail("Truncated XIR bitcode payload.", bitcode.size());
        return result;
    }
    if (payload_size != bitcode.size() - bitcode_header_size) {
        fail("Unexpected trailing bytes after XIR bitcode payload.", bitcode_header_size + payload_size);
        return result;
    }
    auto payload = bitcode.subspan(bitcode_header_size, payload_size);
    if (checksum(payload) != read_u64(bitcode, 24u)) {
        fail("XIR bitcode payload checksum mismatch.", 24u);
        return result;
    }
    if (version == interchange_version) {
        auto text = luisa::string_view{
            reinterpret_cast<const char *>(payload.data()), payload.size()};
        return xir_from_interchange_text(text);
    }
    ModuleRecord record;
    BinaryReader reader{payload, bitcode_header_size, result.diagnostics};
    if (!decode_binary_module_record(reader, record)) {
        if (result.diagnostics.empty()) {
            fail("Malformed XIR binary module record.", bitcode_header_size + reader.offset());
        }
        return result;
    }
    return build_module(record);
}

}// namespace luisa::compute::xir

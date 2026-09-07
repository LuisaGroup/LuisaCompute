// Metal library container layout adapted from Metal.jl:
// https://github.com/JuliaGPU/Metal.jl/blob/master/src/compiler/library.jl
//
// Copyright (c) 2019-2020 Filippo Vicentini
// Copyright (c) 2021-present Julia Computing and other contributors
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include <algorithm>
#include <array>
#include <limits>
#include <utility>

#include "metal_metallib.h"

namespace luisa::compute::metal {
namespace {

using Digest = std::array<std::byte, 32u>;

[[nodiscard]] constexpr uint32_t rotate_right(uint32_t x, uint32_t n) noexcept {
    return (x >> n) | (x << (32u - n));
}

[[nodiscard]] Digest sha256(luisa::span<const std::byte> data) noexcept {

    static constexpr std::array<uint32_t, 64u> k{
        0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u,
        0x3956c25bu, 0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u,
        0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u,
        0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u, 0xc19bf174u,
        0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu,
        0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau,
        0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u,
        0xc6e00bf3u, 0xd5a79147u, 0x06ca6351u, 0x14292967u,
        0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu, 0x53380d13u,
        0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u,
        0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u,
        0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u,
        0x19a4c116u, 0x1e376c08u, 0x2748774cu, 0x34b0bcb5u,
        0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu, 0x682e6ff3u,
        0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u,
        0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u};

    std::array<uint32_t, 8u> state{
        0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
        0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u};

    auto process_block = [&](const std::byte *block) noexcept {
        std::array<uint32_t, 64u> words{};
        for (auto i = 0u; i < 16u; i++) {
            auto offset = i * 4u;
            words[i] = static_cast<uint32_t>(std::to_integer<uint8_t>(block[offset])) << 24u |
                       static_cast<uint32_t>(std::to_integer<uint8_t>(block[offset + 1u])) << 16u |
                       static_cast<uint32_t>(std::to_integer<uint8_t>(block[offset + 2u])) << 8u |
                       static_cast<uint32_t>(std::to_integer<uint8_t>(block[offset + 3u]));
        }
        for (auto i = 16u; i < 64u; i++) {
            auto x = words[i - 15u];
            auto y = words[i - 2u];
            auto s0 = rotate_right(x, 7u) ^ rotate_right(x, 18u) ^ (x >> 3u);
            auto s1 = rotate_right(y, 17u) ^ rotate_right(y, 19u) ^ (y >> 10u);
            words[i] = words[i - 16u] + s0 + words[i - 7u] + s1;
        }

        auto a = state[0u];
        auto b = state[1u];
        auto c = state[2u];
        auto d = state[3u];
        auto e = state[4u];
        auto f = state[5u];
        auto g = state[6u];
        auto h = state[7u];
        for (auto i = 0u; i < 64u; i++) {
            auto s1 = rotate_right(e, 6u) ^ rotate_right(e, 11u) ^ rotate_right(e, 25u);
            auto choice = (e & f) ^ (~e & g);
            auto temp1 = h + s1 + choice + k[i] + words[i];
            auto s0 = rotate_right(a, 2u) ^ rotate_right(a, 13u) ^ rotate_right(a, 22u);
            auto majority = (a & b) ^ (a & c) ^ (b & c);
            auto temp2 = s0 + majority;
            h = g;
            g = f;
            f = e;
            e = d + temp1;
            d = c;
            c = b;
            b = a;
            a = temp1 + temp2;
        }
        state[0u] += a;
        state[1u] += b;
        state[2u] += c;
        state[3u] += d;
        state[4u] += e;
        state[5u] += f;
        state[6u] += g;
        state[7u] += h;
    };

    auto full_block_count = data.size() / 64u;
    for (auto i = 0u; i < full_block_count; i++) {
        process_block(data.data() + i * 64u);
    }

    std::array<std::byte, 128u> tail{};
    auto remaining = data.size() % 64u;
    if (remaining != 0u) {
        std::copy_n(data.data() + full_block_count * 64u, remaining, tail.data());
    }
    tail[remaining] = std::byte{0x80u};
    auto tail_size = remaining < 56u ? 64u : 128u;
    auto bit_count = static_cast<uint64_t>(data.size()) * 8u;
    for (auto i = 0u; i < 8u; i++) {
        tail[tail_size - 1u - i] = static_cast<std::byte>(bit_count >> (i * 8u));
    }
    process_block(tail.data());
    if (tail_size == 128u) { process_block(tail.data() + 64u); }

    Digest digest{};
    for (auto i = 0u; i < state.size(); i++) {
        digest[i * 4u] = static_cast<std::byte>(state[i] >> 24u);
        digest[i * 4u + 1u] = static_cast<std::byte>(state[i] >> 16u);
        digest[i * 4u + 2u] = static_cast<std::byte>(state[i] >> 8u);
        digest[i * 4u + 3u] = static_cast<std::byte>(state[i]);
    }
    return digest;
}

class Writer {
private:
    luisa::vector<std::byte> _data;

public:
    [[nodiscard]] size_t size() const noexcept { return _data.size(); }

    void reserve(size_t size) noexcept { _data.reserve(size); }

    void write_u8(uint8_t value) noexcept { _data.emplace_back(static_cast<std::byte>(value)); }

    void write_u16(uint16_t value) noexcept {
        write_u8(static_cast<uint8_t>(value));
        write_u8(static_cast<uint8_t>(value >> 8u));
    }

    void write_u32(uint32_t value) noexcept {
        for (auto i = 0u; i < 4u; i++) { write_u8(static_cast<uint8_t>(value >> (i * 8u))); }
    }

    void write_u64(uint64_t value) noexcept {
        for (auto i = 0u; i < 8u; i++) { write_u8(static_cast<uint8_t>(value >> (i * 8u))); }
    }

    void write_ascii(luisa::string_view value) noexcept {
        for (auto c : value) { write_u8(static_cast<uint8_t>(c)); }
    }

    void write_bytes(luisa::span<const std::byte> value) noexcept {
        _data.insert(_data.end(), value.begin(), value.end());
    }

    void patch_u32(size_t offset, uint32_t value) noexcept {
        for (auto i = 0u; i < 4u; i++) { _data[offset + i] = static_cast<std::byte>(value >> (i * 8u)); }
    }

    void patch_u64(size_t offset, uint64_t value) noexcept {
        for (auto i = 0u; i < 8u; i++) { _data[offset + i] = static_cast<std::byte>(value >> (i * 8u)); }
    }

    [[nodiscard]] luisa::vector<std::byte> take() noexcept { return std::move(_data); }
};

void write_tag_header(Writer &writer, luisa::string_view tag, uint16_t size) noexcept {
    writer.write_ascii(tag);
    writer.write_u16(size);
}

void write_string_tag(Writer &writer, luisa::string_view tag, luisa::string_view value) noexcept {
    write_tag_header(writer, tag, static_cast<uint16_t>(value.size() + 1u));
    writer.write_ascii(value);
    writer.write_u8(0u);
}

void write_u8_tag(Writer &writer, luisa::string_view tag, uint8_t value) noexcept {
    write_tag_header(writer, tag, 1u);
    writer.write_u8(value);
}

void write_u64_tag(Writer &writer, luisa::string_view tag, uint64_t value) noexcept {
    write_tag_header(writer, tag, 8u);
    writer.write_u64(value);
}

void write_bytes_tag(Writer &writer, luisa::string_view tag, luisa::span<const std::byte> value) noexcept {
    write_tag_header(writer, tag, static_cast<uint16_t>(value.size()));
    writer.write_bytes(value);
}

void write_function_group(Writer &writer,
                          luisa::string_view name,
                          MetalLibProgramType type,
                          const Digest &module_hash,
                          uint64_t public_metadata_offset,
                          uint64_t private_metadata_offset,
                          uint64_t module_offset,
                          uint64_t module_size,
                          const MetalLibTarget &target) noexcept {
    auto group_start = writer.size();
    writer.write_u32(0u);
    write_string_tag(writer, "NAME", name);
    write_u8_tag(writer, "TYPE", static_cast<uint8_t>(type));
    write_bytes_tag(writer, "HASH", module_hash);
    write_tag_header(writer, "OFFT", 24u);
    writer.write_u64(public_metadata_offset);
    writer.write_u64(private_metadata_offset);
    writer.write_u64(module_offset);
    write_tag_header(writer, "VERS", 8u);
    writer.write_u16(target.air.major);
    writer.write_u16(target.air.minor);
    writer.write_u16(target.metal.major);
    writer.write_u16(target.metal.minor);
    write_u64_tag(writer, "MDSZ", module_size);
    writer.write_ascii("ENDT");
    writer.patch_u32(group_start, static_cast<uint32_t>(writer.size() - group_start));
}

void write_empty_group(Writer &writer) noexcept {
    writer.write_u32(8u);
    writer.write_ascii("ENDT");
}

class Reader {
private:
    luisa::span<const std::byte> _data;
    size_t _position{0u};
    bool _valid{true};

public:
    explicit Reader(luisa::span<const std::byte> data) noexcept : _data{data} {}

    [[nodiscard]] size_t size() const noexcept { return _data.size(); }
    [[nodiscard]] size_t position() const noexcept { return _position; }
    [[nodiscard]] bool valid() const noexcept { return _valid; }

    [[nodiscard]] bool seek(size_t position) noexcept {
        _valid = _valid && position <= _data.size();
        if (_valid) { _position = position; }
        return _valid;
    }

    [[nodiscard]] uint8_t read_u8() noexcept {
        if (_position >= _data.size()) {
            _valid = false;
            return 0u;
        }
        return std::to_integer<uint8_t>(_data[_position++]);
    }

    [[nodiscard]] uint16_t read_u16() noexcept {
        auto x = static_cast<uint16_t>(read_u8());
        return static_cast<uint16_t>(x | static_cast<uint16_t>(read_u8()) << 8u);
    }

    [[nodiscard]] uint32_t read_u32() noexcept {
        auto x = 0u;
        for (auto i = 0u; i < 4u; i++) { x |= static_cast<uint32_t>(read_u8()) << (i * 8u); }
        return x;
    }

    [[nodiscard]] uint64_t read_u64() noexcept {
        auto x = 0ull;
        for (auto i = 0u; i < 8u; i++) { x |= static_cast<uint64_t>(read_u8()) << (i * 8u); }
        return x;
    }

    [[nodiscard]] bool read_ascii(luisa::string_view value) noexcept {
        for (auto c : value) {
            if (read_u8() != static_cast<uint8_t>(c)) { _valid = false; }
        }
        return _valid;
    }

    [[nodiscard]] luisa::span<const std::byte> read_bytes(size_t size) noexcept {
        if (size > _data.size() - std::min(_position, _data.size())) {
            _valid = false;
            return {};
        }
        auto value = _data.subspan(_position, size);
        _position += size;
        return value;
    }

    [[nodiscard]] bool skip(size_t size) noexcept {
        static_cast<void>(read_bytes(size));
        return _valid;
    }
};

struct FunctionRecord {
    luisa::string name;
    MetalLibProgramType type{MetalLibProgramType::KERNEL};
    Digest hash{};
    uint64_t public_metadata_offset{0u};
    uint64_t private_metadata_offset{0u};
    uint64_t module_offset{0u};
    uint64_t module_size{0u};
};

[[nodiscard]] bool is_tag(luisa::span<const std::byte> tag, luisa::string_view expected) noexcept {
    if (tag.size() != expected.size()) { return false; }
    for (auto i = 0u; i < tag.size(); i++) {
        if (std::to_integer<uint8_t>(tag[i]) != static_cast<uint8_t>(expected[i])) { return false; }
    }
    return true;
}

[[nodiscard]] bool read_function_group(Reader &reader, FunctionRecord &record) noexcept {
    auto group_start = reader.position();
    auto group_size = reader.read_u32();
    if (!reader.valid() || group_size < 8u || group_size > reader.size() - group_start) { return false; }
    auto group_end = group_start + group_size;
    auto have_name = false;
    auto have_type = false;
    auto have_hash = false;
    auto have_offsets = false;
    auto have_versions = false;
    auto have_module_size = false;
    auto have_end = false;
    while (reader.valid() && reader.position() + 4u <= group_end) {
        auto tag = reader.read_bytes(4u);
        if (is_tag(tag, "ENDT")) {
            have_end = true;
            break;
        }
        auto size = reader.read_u16();
        auto value = reader.read_bytes(size);
        if (!reader.valid()) { return false; }
        Reader value_reader{value};
        if (is_tag(tag, "NAME")) {
            if (have_name || value.empty() || value.back() != std::byte{}) { return false; }
            record.name.reserve(value.size() - 1u);
            for (auto c : value.first(value.size() - 1u)) { record.name.push_back(static_cast<char>(std::to_integer<uint8_t>(c))); }
            have_name = true;
        } else if (is_tag(tag, "TYPE")) {
            if (have_type || size != 1u) { return false; }
            auto type = value_reader.read_u8();
            if (type > static_cast<uint8_t>(MetalLibProgramType::INTERSECTION)) { return false; }
            record.type = static_cast<MetalLibProgramType>(type);
            have_type = true;
        } else if (is_tag(tag, "HASH")) {
            if (have_hash || size != record.hash.size()) { return false; }
            std::copy(value.begin(), value.end(), record.hash.begin());
            have_hash = true;
        } else if (is_tag(tag, "OFFT")) {
            if (have_offsets || size != 24u) { return false; }
            record.public_metadata_offset = value_reader.read_u64();
            record.private_metadata_offset = value_reader.read_u64();
            record.module_offset = value_reader.read_u64();
            have_offsets = value_reader.valid();
        } else if (is_tag(tag, "VERS")) {
            if (have_versions || size != 8u) { return false; }
            auto air_major = value_reader.read_u16();
            static_cast<void>(value_reader.read_u16());
            auto metal_major = value_reader.read_u16();
            static_cast<void>(value_reader.read_u16());
            if (!value_reader.valid() || air_major == 0u || metal_major == 0u) { return false; }
            have_versions = true;
        } else if (is_tag(tag, "MDSZ")) {
            if (have_module_size || size != 8u) { return false; }
            record.module_size = value_reader.read_u64();
            have_module_size = value_reader.valid();
        }
    }
    return have_end && reader.position() == group_end &&
           have_name && have_type && have_hash && have_offsets && have_versions && have_module_size;
}

[[nodiscard]] bool skip_tag_group(Reader &reader) noexcept {
    auto group_start = reader.position();
    auto group_size = reader.read_u32();
    if (!reader.valid() || group_size < 8u || group_size > reader.size() - group_start) { return false; }
    auto group_end = group_start + group_size;
    while (reader.valid() && reader.position() + 4u <= group_end) {
        auto tag = reader.read_bytes(4u);
        if (is_tag(tag, "ENDT")) { return reader.position() == group_end; }
        auto size = reader.read_u16();
        if (!reader.skip(size)) { return false; }
    }
    return false;
}

[[nodiscard]] bool skip_header_extension(Reader &reader, size_t end) noexcept {
    auto have_end = false;
    while (reader.valid() && reader.position() + 4u <= end) {
        auto tag = reader.read_bytes(4u);
        if (is_tag(tag, "ENDT")) {
            have_end = true;
            break;
        }
        auto size = reader.read_u16();
        if (!reader.skip(size)) { return false; }
    }
    return have_end && reader.position() == end;
}

}// namespace

namespace {

[[nodiscard]] MetalLibTarget metallib_target_for_apple_platform(
    MetalLibPlatform operating_system,
    uint16_t major, uint16_t minor, uint16_t patch) noexcept {
    auto generation = operating_system == MetalLibPlatform::MACOS ?
                          (major >= 16u && major < 26u ?
                               static_cast<uint16_t>(major + 10u) :
                               major) :
                          major;
    auto air_27 = generation >= 27u;
    auto air_26 = generation >= 26u;
    auto air_27_predecessor = operating_system == MetalLibPlatform::MACOS ?
                                  generation >= 15u :
                                  generation >= 18u;
    auto air_26_predecessor = operating_system == MetalLibPlatform::MACOS ?
                                  generation >= 14u :
                                  generation >= 17u;
    auto file_format = air_26             ? MetalLibVersion{1u, 2u, 9u} :
                       air_27_predecessor ? MetalLibVersion{1u, 2u, 8u} :
                                            MetalLibVersion{1u, 2u, 7u};
    auto air = air_27             ? MetalLibVersion{2u, 9u, 0u} :
               air_26             ? MetalLibVersion{2u, 8u, 0u} :
               air_27_predecessor ? MetalLibVersion{2u, 7u, 0u} :
               air_26_predecessor ? MetalLibVersion{2u, 6u, 0u} :
                                    MetalLibVersion{2u, 5u, 0u};
    auto metal = air_27             ? MetalLibVersion{4u, 1u, 0u} :
                 air_26             ? MetalLibVersion{4u, 0u, 0u} :
                 air_27_predecessor ? MetalLibVersion{3u, 2u, 0u} :
                 air_26_predecessor ? MetalLibVersion{3u, 1u, 0u} :
                                      MetalLibVersion{3u, 0u, 0u};
    return MetalLibTarget{
        .operating_system = operating_system,
        .file_format = file_format,
        .platform = MetalLibVersion{generation, minor, patch},
        .air = air,
        .metal = metal};
}

}// namespace

MetalLibTarget metallib_target_for_macos(uint16_t major, uint16_t minor, uint16_t patch) noexcept {
    return metallib_target_for_apple_platform(
        MetalLibPlatform::MACOS, major, minor, patch);
}

MetalLibTarget metallib_target_for_ios(uint16_t major, uint16_t minor, uint16_t patch) noexcept {
    return metallib_target_for_apple_platform(
        MetalLibPlatform::IOS, major, minor, patch);
}

luisa::vector<std::byte> make_metallib(
    const MetalLibTarget &target,
    luisa::span<const MetalLibFunction> functions) noexcept {

    if (functions.empty() || functions.size() > std::numeric_limits<uint32_t>::max() ||
        (target.operating_system != MetalLibPlatform::MACOS &&
         target.operating_system != MetalLibPlatform::IOS) ||
        target.file_format.major != 1u || target.file_format.minor != 2u || target.file_format.patch < 7u ||
        target.platform.major < 13u || target.platform.minor > std::numeric_limits<uint8_t>::max() ||
        target.platform.patch > std::numeric_limits<uint8_t>::max() ||
        target.air.major == 0u || target.metal.major == 0u) { return {}; }
    auto total_module_size = size_t{0u};
    for (auto i = 0u; i < functions.size(); i++) {
        auto &&function = functions[i];
        if (function.name.empty() || function.name.size() >= std::numeric_limits<uint16_t>::max() ||
            function.name.find('\0') != luisa::string_view::npos || function.air_module.empty() ||
            static_cast<uint8_t>(function.type) > static_cast<uint8_t>(MetalLibProgramType::INTERSECTION) ||
            function.air_module.size() > std::numeric_limits<size_t>::max() - total_module_size) { return {}; }
        for (auto j = 0u; j < i; j++) {
            if (functions[j].name == function.name) { return {}; }
        }
        total_module_size += function.air_module.size();
    }

    luisa::vector<std::byte> uuid_input;
    uuid_input.reserve(total_module_size);
    for (auto &&function : functions) {
        uuid_input.insert(uuid_input.end(), function.air_module.begin(), function.air_module.end());
    }
    auto uuid = sha256(uuid_input);
    uuid[6u] = static_cast<std::byte>((std::to_integer<uint8_t>(uuid[6u]) & 0x0fu) | 0x40u);
    uuid[8u] = static_cast<std::byte>((std::to_integer<uint8_t>(uuid[8u]) & 0x3fu) | 0x80u);

    Writer writer;
    if (total_module_size > std::numeric_limits<size_t>::max() - 256u ||
        functions.size() > (std::numeric_limits<size_t>::max() - total_module_size - 256u) / 160u) { return {}; }
    writer.reserve(256u + total_module_size + functions.size() * 160u);
    writer.write_ascii("MTLB");
    auto file_major = target.file_format.major;
    if (target.operating_system == MetalLibPlatform::MACOS) {
        file_major = static_cast<uint16_t>(file_major | 0x8000u);
    }
    writer.write_u16(file_major);
    writer.write_u16(target.file_format.minor);
    writer.write_u16(target.file_format.patch);
    writer.write_u8(0u);
    writer.write_u8(static_cast<uint8_t>(target.operating_system));
    writer.write_u16(target.platform.major);
    writer.write_u8(static_cast<uint8_t>(target.platform.minor));
    writer.write_u8(static_cast<uint8_t>(target.platform.patch));

    auto file_size_offset = writer.size();
    writer.write_u64(0u);
    auto function_list_offset_field = writer.size();
    writer.write_u64(0u);
    auto function_list_size_field = writer.size();
    writer.write_u64(0u);
    auto public_metadata_offset_field = writer.size();
    writer.write_u64(0u);
    writer.write_u64(functions.size() * 8u);
    auto private_metadata_offset_field = writer.size();
    writer.write_u64(0u);
    writer.write_u64(functions.size() * 8u);
    auto module_list_offset_field = writer.size();
    writer.write_u64(0u);
    writer.write_u64(total_module_size);

    auto function_list_offset = writer.size();
    writer.patch_u64(function_list_offset_field, function_list_offset);
    writer.write_u32(static_cast<uint32_t>(functions.size()));
    auto module_offset = uint64_t{0u};
    for (auto i = 0u; i < functions.size(); i++) {
        auto &&function = functions[i];
        auto module_hash = sha256(function.air_module);
        write_function_group(writer, function.name, function.type, module_hash,
                             i * 8u, i * 8u, module_offset, function.air_module.size(), target);
        module_offset += function.air_module.size();
    }
    writer.patch_u64(function_list_size_field, writer.size() - function_list_offset - 4u);

    write_tag_header(writer, "UUID", 16u);
    for (auto i = 0u; i < 8u; i++) { writer.write_u8(std::to_integer<uint8_t>(uuid[7u - i])); }
    for (auto i = 0u; i < 8u; i++) { writer.write_u8(std::to_integer<uint8_t>(uuid[15u - i])); }
    writer.write_ascii("ENDT");

    writer.patch_u64(public_metadata_offset_field, writer.size());
    for (auto i = 0u; i < functions.size(); i++) { write_empty_group(writer); }
    writer.patch_u64(private_metadata_offset_field, writer.size());
    for (auto i = 0u; i < functions.size(); i++) { write_empty_group(writer); }
    writer.patch_u64(module_list_offset_field, writer.size());
    for (auto &&function : functions) { writer.write_bytes(function.air_module); }
    writer.patch_u64(file_size_offset, writer.size());
    return writer.take();
}

bool validate_metallib(
    luisa::span<const std::byte> data,
    luisa::span<const luisa::string_view> expected_entry_points,
    luisa::span<const MetalLibProgramType> expected_program_types) noexcept {

    Reader reader{data};
    if (!reader.read_ascii("MTLB")) { return false; }
    auto file_major = reader.read_u16();
    auto file_minor = reader.read_u16();
    auto file_patch = reader.read_u16();
    auto file_type = reader.read_u8();
    auto platform_type = reader.read_u8();
    auto platform_major = reader.read_u16();
    static_cast<void>(reader.read_u8());
    static_cast<void>(reader.read_u8());
    auto is_macos = platform_type == static_cast<uint8_t>(MetalLibPlatform::MACOS);
    auto is_ios = platform_type == static_cast<uint8_t>(MetalLibPlatform::IOS);
    auto has_macos_file_bit = (file_major & 0x8000u) != 0u;
    if (!reader.valid() || (!is_macos && !is_ios) ||
        has_macos_file_bit != is_macos ||
        (file_major & 0x7fffu) != 1u || file_minor != 2u || file_patch < 7u ||
        file_type != 0u || platform_major < 13u) { return false; }

    auto file_size = reader.read_u64();
    auto function_list_offset = reader.read_u64();
    auto function_list_size = reader.read_u64();
    auto public_metadata_offset = reader.read_u64();
    auto public_metadata_size = reader.read_u64();
    auto private_metadata_offset = reader.read_u64();
    auto private_metadata_size = reader.read_u64();
    auto module_list_offset = reader.read_u64();
    auto module_list_size = reader.read_u64();
    if (!reader.valid() || file_size != data.size() ||
        function_list_offset < reader.position() || function_list_offset > data.size() ||
        public_metadata_offset > data.size() || private_metadata_offset > data.size() ||
        module_list_offset > data.size() ||
        function_list_offset > data.size() - 4u ||
        function_list_size > data.size() - function_list_offset - 4u ||
        public_metadata_size > data.size() - public_metadata_offset ||
        private_metadata_size > data.size() - private_metadata_offset ||
        module_list_size > data.size() - module_list_offset) { return false; }
    if (function_list_offset > public_metadata_offset ||
        function_list_size + 4u > public_metadata_offset - function_list_offset ||
        public_metadata_offset > private_metadata_offset ||
        public_metadata_size > private_metadata_offset - public_metadata_offset ||
        private_metadata_offset > module_list_offset ||
        private_metadata_size > module_list_offset - private_metadata_offset ||
        module_list_size != data.size() - module_list_offset) { return false; }

    if (!reader.seek(function_list_offset)) { return false; }
    auto function_count = reader.read_u32();
    if (!reader.valid() || function_count == 0u ||
        function_count > function_list_size / 8u ||
        (!expected_entry_points.empty() && function_count != expected_entry_points.size()) ||
        (!expected_program_types.empty() && function_count != expected_program_types.size())) { return false; }
    luisa::vector<FunctionRecord> functions(function_count);
    for (auto i = 0u; i < function_count; i++) {
        if (!read_function_group(reader, functions[i])) { return false; }
        if (!expected_entry_points.empty() && functions[i].name != expected_entry_points[i]) { return false; }
        if (!expected_program_types.empty() && functions[i].type != expected_program_types[i]) { return false; }
    }
    if (reader.position() != function_list_offset + function_list_size + 4u ||
        reader.position() > public_metadata_offset ||
        !skip_header_extension(reader, public_metadata_offset)) { return false; }

    if (public_metadata_size < function_count * 8ull ||
        private_metadata_size < function_count * 8ull ||
        !reader.seek(public_metadata_offset)) { return false; }
    auto public_metadata_begin = reader.position();
    for (auto i = 0u; i < function_count; i++) {
        if (functions[i].public_metadata_offset != reader.position() - public_metadata_begin ||
            !skip_tag_group(reader)) { return false; }
    }
    if (reader.position() != public_metadata_offset + public_metadata_size ||
        !reader.seek(private_metadata_offset)) { return false; }
    auto private_metadata_begin = reader.position();
    for (auto i = 0u; i < function_count; i++) {
        if (functions[i].private_metadata_offset != reader.position() - private_metadata_begin ||
            !skip_tag_group(reader)) { return false; }
    }
    if (reader.position() != private_metadata_offset + private_metadata_size ||
        !reader.seek(module_list_offset)) { return false; }

    auto module_list_begin = reader.position();
    for (auto i = 0u; i < function_count; i++) {
        if (functions[i].module_offset != reader.position() - module_list_begin ||
            functions[i].module_size > data.size() - reader.position()) { return false; }
        auto module = reader.read_bytes(functions[i].module_size);
        if (!reader.valid() || sha256(module) != functions[i].hash) { return false; }
    }
    return reader.valid() && reader.position() == data.size();
}

}// namespace luisa::compute::metal

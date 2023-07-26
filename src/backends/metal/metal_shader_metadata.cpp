#include <charconv>

#include <luisa/core/stl/format.h>
#include <luisa/core/logging.h>
#include "metal_shader_metadata.h"

namespace luisa::compute::metal {

luisa::string serialize_metal_shader_metadata(const MetalShaderMetadata &metadata) noexcept {
    luisa::string result;
    result.append(luisa::format("CHECKSUM {:016x} ", metadata.checksum));
    result.append(luisa::format("BLOCK_SIZE {} {} {} ", metadata.block_size.x, metadata.block_size.y, metadata.block_size.z));
    result.append(luisa::format("ARGUMENT_TYPES {} ", metadata.argument_types.size()));
    for (auto &&type : metadata.argument_types) { result.append(type).append(" "); }
    result.append(luisa::format("ARGUMENT_USAGES {}", metadata.argument_usages.size()));
    for (auto usage : metadata.argument_usages) {
        switch (usage) {
            case Usage::NONE: result.append(" NONE"); break;
            case Usage::READ: result.append(" READ"); break;
            case Usage::WRITE: result.append(" WRITE"); break;
            case Usage::READ_WRITE: result.append(" READ_WRITE"); break;
        }
    }
    return result;
}

luisa::optional<MetalShaderMetadata> deserialize_metal_shader_metadata(luisa::string_view metadata) noexcept {

    auto read_token = [&metadata] {
        // skip blanks
        while (!metadata.empty() &&
               isblank(metadata.front())) {
            metadata.remove_prefix(1u);
        }
        // find end
        auto end_pos = 0u;
        while (end_pos < metadata.size() &&
               !isblank(metadata[end_pos])) { end_pos++; }
        auto token = metadata.substr(0u, end_pos);
        // remove token
        metadata.remove_prefix(end_pos);
        return token;
    };

    auto parse_number = [](luisa::string_view token) mutable noexcept {
        auto x = 0ull;
        auto [p, ec] = std::from_chars(token.data(), token.data() + token.size(), x);
        return ec == std::errc{} && p == token.data() + token.size() ?
                   luisa::make_optional(x) :
                   luisa::nullopt;
    };

    auto parse_checksum = [](luisa::string_view token) mutable noexcept {
        auto x = 0ull;
        auto [p, ec] = std::from_chars(token.data(), token.data() + token.size(), x, 16);
        return ec == std::errc{} && p == token.data() + token.size() ?
                   luisa::make_optional(x) :
                   luisa::nullopt;
    };

    luisa::optional<uint64_t> checksum;
    luisa::optional<uint3> block_size;
    luisa::optional<luisa::vector<luisa::string>> argument_types;
    luisa::optional<luisa::vector<Usage>> argument_usages;

    for (;;) {
        auto token = read_token();
        if (token.empty()) { break; }
        if (token == "CHECKSUM") {
            if (checksum.has_value()) {
                LUISA_WARNING_WITH_LOCATION(
                    "Duplicate checksum in shader metadata.");
                return luisa::nullopt;
            }
            auto x = parse_checksum(read_token());
            if (!x.has_value()) {
                LUISA_WARNING("Invalid checksum in shader metadata.");
                return luisa::nullopt;
            }
            checksum = x.value();
        } else if (token == "BLOCK_SIZE") {
            if (block_size.has_value()) {
                LUISA_WARNING_WITH_LOCATION(
                    "Duplicate block size in shader metadata.");
                return luisa::nullopt;
            }
            auto x = parse_number(read_token());
            if (!x.has_value()) {
                LUISA_WARNING_WITH_LOCATION(
                    "Invalid block size in shader metadata.");
                return luisa::nullopt;
            }
            auto y = parse_number(read_token());
            if (!y.has_value()) {
                LUISA_WARNING_WITH_LOCATION(
                    "Invalid block size in shader metadata.");
                return luisa::nullopt;
            }
            auto z = parse_number(read_token());
            if (!z.has_value()) {
                LUISA_WARNING_WITH_LOCATION(
                    "Invalid block size in shader metadata.");
                return luisa::nullopt;
            }
            block_size.emplace(luisa::make_uint3(
                static_cast<uint>(x.value()),
                static_cast<uint>(y.value()),
                static_cast<uint>(z.value())));
        } else if (token == "ARGUMENT_TYPES") {
            if (argument_types.has_value()) {
                LUISA_WARNING_WITH_LOCATION(
                    "Duplicate argument types in shader metadata.");
                return luisa::nullopt;
            }
            auto x = parse_number(read_token());
            if (!x.has_value()) {
                LUISA_WARNING_WITH_LOCATION(
                    "Invalid argument types in shader metadata.");
                return luisa::nullopt;
            }
            auto count = x.value();
            luisa::vector<luisa::string> types;
            types.reserve(count);
            for (auto i = 0ull; i < count; i++) {
                auto type = read_token();
                if (type.empty()) {
                    LUISA_WARNING_WITH_LOCATION(
                        "Invalid argument type in shader metadata.");
                    return luisa::nullopt;
                }
                types.emplace_back(type);
            }
            argument_types.emplace(std::move(types));
        } else if (token == "ARGUMENT_USAGES") {
            if (argument_usages.has_value()) {
                LUISA_WARNING_WITH_LOCATION(
                    "Duplicate argument usages in shader metadata.");
                return luisa::nullopt;
            }
            auto x = parse_number(read_token());
            if (!x.has_value()) {
                LUISA_WARNING_WITH_LOCATION(
                    "Invalid argument usages in shader metadata.");
                return luisa::nullopt;
            }
            auto count = x.value();
            luisa::vector<Usage> usages;
            usages.reserve(count);
            for (auto i = 0ull; i < count; i++) {
                auto usage = read_token();
                if (usage == "NONE") {
                    usages.emplace_back(Usage::NONE);
                } else if (usage == "READ") {
                    usages.emplace_back(Usage::READ);
                } else if (usage == "WRITE") {
                    usages.emplace_back(Usage::WRITE);
                } else if (usage == "READ_WRITE") {
                    usages.emplace_back(Usage::READ_WRITE);
                } else {
                    LUISA_WARNING_WITH_LOCATION(
                        "Invalid argument usage '{}' "
                        "in shader metadata.",
                        usage);
                    return luisa::nullopt;
                }
            }
            argument_usages.emplace(std::move(usages));
        } else {
            LUISA_WARNING_WITH_LOCATION(
                "Invalid token in shader metadata: {}.", token);
            return luisa::nullopt;
        }
    }
    if (!checksum.has_value()) {
        LUISA_WARNING_WITH_LOCATION(
            "Missing checksum in shader metadata.");
        return luisa::nullopt;
    }
    if (!block_size.has_value()) {
        LUISA_WARNING_WITH_LOCATION(
            "Missing block size in shader metadata.");
        return luisa::nullopt;
    }
    if (!argument_types.has_value()) {
        LUISA_WARNING_WITH_LOCATION(
            "Missing argument types in shader metadata.");
        return luisa::nullopt;
    }
    if (!argument_usages.has_value()) {
        LUISA_WARNING_WITH_LOCATION(
            "Missing argument usages in shader metadata.");
        return luisa::nullopt;
    }
    if (argument_types->size() != argument_usages->size()) {
        LUISA_WARNING_WITH_LOCATION(
            "Argument types and usages mismatch in shader metadata.");
        return luisa::nullopt;
    }
    if (argument_types->size() != argument_usages->size()) {
        LUISA_WARNING_WITH_LOCATION(
            "Argument count mismatch in shader metadata.");
        return luisa::nullopt;
    }
    return MetalShaderMetadata{
        .checksum = checksum.value(),
        .block_size = block_size.value(),
        .argument_types = std::move(argument_types.value()),
        .argument_usages = std::move(argument_usages.value())};
}

}// namespace luisa::compute::metal


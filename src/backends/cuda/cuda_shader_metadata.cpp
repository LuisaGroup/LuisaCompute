#include <charconv>

#include <luisa/core/stl/format.h>
#include <luisa/core/logging.h>
#include <luisa/core/magic_enum.h>
#include "cuda_shader_metadata.h"

namespace luisa::compute::cuda {

luisa::string serialize_cuda_shader_metadata(const CUDAShaderMetadata &metadata) noexcept {
    luisa::string result;
    result.append(luisa::format("CHECKSUM {:016x} ", metadata.checksum));
    result.append(luisa::format("KIND {} ", metadata.kind == CUDAShaderMetadata::Kind::UNKNOWN ?
                                                "UNKNOWN" :
                                            metadata.kind == CUDAShaderMetadata::Kind::COMPUTE ?
                                                "COMPUTE" :
                                                "RAY_TRACING"));
    result.append(metadata.enable_debug ? "DEBUG TRUE " : "DEBUG FALSE ");
    result.append(metadata.requires_trace_closest ? "TRACE_CLOSEST TRUE " : "TRACE_CLOSEST FALSE ");
    result.append(metadata.requires_trace_any ? "TRACE_ANY TRUE " : "TRACE_ANY FALSE ");
    result.append(metadata.requires_ray_query ? "RAY_QUERY TRUE " : "RAY_QUERY FALSE ");
    result.append(metadata.requires_printing ? "PRINTING TRUE " : "PRINTING FALSE ");
    result.append(metadata.requires_motion_blur ? "MOTION_BLUR TRUE " : "MOTION_BLUR FALSE ");
    result.append(luisa::format("MAX_REGISTER_COUNT {} ", metadata.max_register_count));
    result.append(luisa::format("BLOCK_SIZE {} {} {} ", metadata.block_size.x, metadata.block_size.y, metadata.block_size.z));
    result.append(luisa::format("ARGUMENT_TYPES {} ", metadata.argument_types.size()));
    for (auto &&type : metadata.argument_types) { result.append(type).append(" "); }
    result.append(luisa::format("ARGUMENT_USAGES {} ", metadata.argument_usages.size()));
    for (auto usage : metadata.argument_usages) {
        switch (usage) {
            case Usage::NONE: result.append("NONE "); break;
            case Usage::READ: result.append("READ "); break;
            case Usage::WRITE: result.append("WRITE "); break;
            case Usage::READ_WRITE: result.append("READ_WRITE "); break;
        }
    }
    result.append(luisa::format("FORMAT_TYPES {} ", metadata.format_types.size()));
    for (auto &&[fmt, type] : metadata.format_types) {
        for (auto c : fmt) { result.append(luisa::format("{:02x}", static_cast<uint>(c))); }
        result.append(" ").append(type).append(" ");
    }
    result.append(luisa::format("CURVE_BASES {} ", metadata.curve_bases.count()));
    for (auto i = 0u; i < curve_basis_count; i++) {
        if (auto basis = static_cast<CurveBasis>(i);
            metadata.curve_bases.test(basis)) {
            result.append(luisa::to_string(basis)).append(" ");
        }
    }
    return result;
}

luisa::optional<CUDAShaderMetadata> deserialize_cuda_shader_metadata(luisa::string_view metadata) noexcept {

    auto read_token = [&metadata] {
        auto is_blank = [](char c) noexcept {
            return isblank(c) || c == '\r' || c == '\n';
        };
        // skip blanks
        while (!metadata.empty() &&
               is_blank(metadata.front())) {
            metadata.remove_prefix(1u);
        }
        // find end
        auto end_pos = 0u;
        while (end_pos < metadata.size() &&
               !is_blank(metadata[end_pos])) { end_pos++; }
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
    luisa::optional<CurveBasisSet> curve_bases;
    luisa::optional<uint3> block_size;
    auto kind = CUDAShaderMetadata::Kind::UNKNOWN;
    luisa::optional<bool> enable_debug;
    luisa::optional<bool> requires_trace_closest;
    luisa::optional<bool> requires_trace_any;
    luisa::optional<bool> requires_ray_query;
    luisa::optional<bool> requires_printing;
    luisa::optional<bool> requires_motion_blur;
    luisa::optional<uint> max_register_count;
    luisa::optional<luisa::vector<luisa::string>> argument_types;
    luisa::optional<luisa::vector<Usage>> argument_usages;
    luisa::optional<luisa::vector<std::pair<luisa::string, luisa::string>>> format_types;

    for (;;) {
        auto token = read_token();
        if (token.empty()) { break; }
        if (token == "KIND") {
            if (kind != CUDAShaderMetadata::Kind::UNKNOWN) {
                LUISA_WARNING_WITH_LOCATION("Duplicate kind in shader metadata.");
                return luisa::nullopt;
            }
            auto x = read_token();
            if (x == "COMPUTE") {
                kind = CUDAShaderMetadata::Kind::COMPUTE;
            } else if (x == "RAY_TRACING") {
                kind = CUDAShaderMetadata::Kind::RAY_TRACING;
            } else {
                LUISA_WARNING_WITH_LOCATION("Invalid kind '{}' in shader metadata.", x);
                return luisa::nullopt;
            }
        } else if (token == "CHECKSUM") {
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
        } else if (token == "CURVE_BASES") {
            if (curve_bases.has_value()) {
                LUISA_WARNING_WITH_LOCATION(
                    "Duplicate curve bases in shader metadata.");
                return luisa::nullopt;
            }
            auto x = parse_number(read_token());
            if (!x.has_value()) {
                LUISA_WARNING_WITH_LOCATION(
                    "Invalid curve bases in shader metadata.");
                return luisa::nullopt;
            }
            auto count = x.value();
            CurveBasisSet bases;
            for (auto i = 0u; i < count; i++) {
                auto name = read_token();
                if (name.empty()) {
                    LUISA_WARNING_WITH_LOCATION(
                        "Empty curve basis name.");
                    return luisa::nullopt;
                }
                using namespace std::string_view_literals;
                if (name == "PIECEWISE_LINEAR"sv) {
                    bases.mark(CurveBasis::PIECEWISE_LINEAR);
                } else if (name == "CUBIC_BSPLINE"sv) {
                    bases.mark(CurveBasis::CUBIC_BSPLINE);
                } else if (name == "CATMULL_ROM"sv) {
                    bases.mark(CurveBasis::CATMULL_ROM);
                } else if (name == "BEZIER"sv) {
                    bases.mark(CurveBasis::BEZIER);
                } else {
                    LUISA_WARNING_WITH_LOCATION(
                        "Invalid curve basis name '{}'.", name);
                    return luisa::nullopt;
                }
            }
            if (bases.count() != count) {
                LUISA_WARNING_WITH_LOCATION(
                    "Invalid curve basis set size.");
                return luisa::nullopt;
            }
            curve_bases.emplace(bases);
        } else if (token == "DEBUG") {
            if (enable_debug.has_value()) {
                LUISA_WARNING_WITH_LOCATION(
                    "Duplicate debug flag in shader metadata.");
                return luisa::nullopt;
            }
            auto x = read_token();
            if (x == "TRUE") {
                enable_debug.emplace(true);
            } else if (x == "FALSE") {
                enable_debug.emplace(false);
            } else {
                LUISA_WARNING_WITH_LOCATION(
                    "Invalid debug flag '{}' in shader metadata.", x);
                return luisa::nullopt;
            }
        } else if (token == "TRACE_CLOSEST") {
            if (requires_trace_closest.has_value()) {
                LUISA_WARNING_WITH_LOCATION(
                    "Duplicate requires_trace_closest flag in shader metadata.");
                return luisa::nullopt;
            }
            auto x = read_token();
            if (x == "TRUE") {
                requires_trace_closest.emplace(true);
            } else if (x == "FALSE") {
                requires_trace_closest.emplace(false);
            } else {
                LUISA_WARNING_WITH_LOCATION(
                    "Invalid requires_trace_closest flag '{}' in shader metadata.", x);
                return luisa::nullopt;
            }
        } else if (token == "TRACE_ANY") {
            if (requires_trace_any.has_value()) {
                LUISA_WARNING_WITH_LOCATION(
                    "Duplicate requires_trace_any flag in shader metadata.");
                return luisa::nullopt;
            }
            auto x = read_token();
            if (x == "TRUE") {
                requires_trace_any.emplace(true);
            } else if (x == "FALSE") {
                requires_trace_any.emplace(false);
            } else {
                LUISA_WARNING_WITH_LOCATION(
                    "Invalid requires_trace_any flag '{}' in shader metadata.", x);
                return luisa::nullopt;
            }
        } else if (token == "RAY_QUERY") {
            if (requires_ray_query.has_value()) {
                LUISA_WARNING_WITH_LOCATION(
                    "Duplicate requires_ray_query flag in shader metadata.");
                return luisa::nullopt;
            }
            auto x = read_token();
            if (x == "TRUE") {
                requires_ray_query.emplace(true);
            } else if (x == "FALSE") {
                requires_ray_query.emplace(false);
            } else {
                LUISA_WARNING_WITH_LOCATION(
                    "Invalid requires_ray_query flag '{}' in shader metadata.", x);
                return luisa::nullopt;
            }
        } else if (token == "PRINTING") {
            if (requires_printing.has_value()) {
                LUISA_WARNING_WITH_LOCATION(
                    "Duplicate requires_printing flag in shader metadata.");
                return luisa::nullopt;
            }
            auto x = read_token();
            if (x == "TRUE") {
                requires_printing.emplace(true);
            } else if (x == "FALSE") {
                requires_printing.emplace(false);
            } else {
                LUISA_WARNING_WITH_LOCATION(
                    "Invalid requires_printing flag '{}' in shader metadata.", x);
                return luisa::nullopt;
            }
        } else if (token == "MOTION_BLUR") {
            if (requires_motion_blur.has_value()) {
                LUISA_WARNING_WITH_LOCATION(
                    "Duplicate requires_motion_blur flag in shader metadata.");
                return luisa::nullopt;
            }
            auto x = read_token();
            if (x == "TRUE") {
                requires_motion_blur.emplace(true);
            } else if (x == "FALSE") {
                requires_motion_blur.emplace(false);
            } else {
                LUISA_WARNING_WITH_LOCATION(
                    "Invalid requires_motion_blur flag '{}' in shader metadata.", x);
                return luisa::nullopt;
            }
        } else if (token == "MAX_REGISTER_COUNT") {
            if (max_register_count.has_value()) {
                LUISA_WARNING_WITH_LOCATION(
                    "Duplicate max register count in shader metadata.");
                return luisa::nullopt;
            }
            auto x = parse_number(read_token());
            if (!x.has_value()) {
                LUISA_WARNING_WITH_LOCATION(
                    "Invalid max register count in shader metadata.");
                return luisa::nullopt;
            }
            max_register_count.emplace(
                std::clamp(static_cast<uint>(x.value()), 0u, 255u));
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
        } else if (token == "FORMAT_TYPES") {
            if (format_types.has_value()) {
                LUISA_WARNING_WITH_LOCATION(
                    "Duplicate format types in shader metadata.");
                return luisa::nullopt;
            }
            auto x = parse_number(read_token());
            if (!x.has_value()) {
                LUISA_WARNING_WITH_LOCATION(
                    "Invalid format types in shader metadata.");
                return luisa::nullopt;
            }
            auto count = x.value();
            luisa::vector<std::pair<luisa::string, luisa::string>> types;
            types.reserve(count);
            for (auto i = 0u; i < count; i++) {
                auto fmt_codes = read_token();
                if (fmt_codes.size() % 2 != 0) {
                    LUISA_WARNING_WITH_LOCATION(
                        "Invalid format string '{}' "
                        "in shader metadata.",
                        fmt_codes);
                    return luisa::nullopt;
                }
                luisa::string fmt;
                fmt.reserve(fmt_codes.size() / 2);
                for (auto j = 0u; j < fmt_codes.size(); j += 2) {
                    auto hi = fmt_codes[j];
                    auto lo = fmt_codes[j + 1];
                    if (!isxdigit(hi) || !isxdigit(lo)) {
                        LUISA_WARNING_WITH_LOCATION(
                            "Invalid format string '{}' "
                            "in shader metadata.",
                            fmt_codes);
                        return luisa::nullopt;
                    }
                    auto hex_to_dec = [](auto x) noexcept {
                        return x >= '0' && x <= '9' ?
                                   x - '0' :
                               x >= 'a' && x <= 'f' ?
                                   x - 'a' + 10 :
                                   x - 'A' + 10;
                    };
                    auto code = (hex_to_dec(hi) << 4u) | hex_to_dec(lo);
                    fmt.push_back(static_cast<char>(code));
                }
                auto type = read_token();
                if (type.empty()) {
                    LUISA_WARNING_WITH_LOCATION(
                        "Invalid format type in shader metadata.");
                    return luisa::nullopt;
                }
                types.emplace_back(fmt, type);
            }
            format_types.emplace(std::move(types));
        } else {
            LUISA_WARNING_WITH_LOCATION(
                "Invalid token in shader metadata: {}.", token);
            return luisa::nullopt;
        }
    }
    if (kind == CUDAShaderMetadata::Kind::UNKNOWN) {
        LUISA_WARNING_WITH_LOCATION(
            "Missing kind in shader metadata.");
        return luisa::nullopt;
    }
    if (!curve_bases.has_value()) {
        LUISA_WARNING_WITH_LOCATION(
            "Missing curve basis set in shader metadata.");
        return luisa::nullopt;
    }
    if (!enable_debug.has_value()) {
        LUISA_WARNING_WITH_LOCATION(
            "Missing debug flag in shader metadata.");
        return luisa::nullopt;
    }
    if (!requires_trace_closest.has_value()) {
        LUISA_WARNING_WITH_LOCATION(
            "Missing requires_trace_closest flag in shader metadata.");
        return luisa::nullopt;
    }
    if (!requires_trace_any.has_value()) {
        LUISA_WARNING_WITH_LOCATION(
            "Missing requires_trace_any flag in shader metadata.");
        return luisa::nullopt;
    }
    if (!requires_ray_query.has_value()) {
        LUISA_WARNING_WITH_LOCATION(
            "Missing requires_ray_query flag in shader metadata.");
        return luisa::nullopt;
    }
    if (!requires_printing.has_value()) {
        LUISA_WARNING_WITH_LOCATION(
            "Missing requires_printing flag in shader metadata.");
        return luisa::nullopt;
    }
    if (!requires_motion_blur.has_value()) {
        LUISA_WARNING_WITH_LOCATION(
            "Missing requires_motion_blur flag in shader metadata.");
        return luisa::nullopt;
    }
    if (!checksum.has_value()) {
        LUISA_WARNING_WITH_LOCATION(
            "Missing checksum in shader metadata.");
        return luisa::nullopt;
    }
    if (!max_register_count.has_value()) {
        LUISA_WARNING_WITH_LOCATION(
            "Missing max register count in shader metadata.");
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
    if (!format_types.has_value()) {
        LUISA_WARNING_WITH_LOCATION(
            "Missing format types in shader metadata.");
        return luisa::nullopt;
    }
    if (format_types->empty() == requires_printing.value()) {
        LUISA_WARNING_WITH_LOCATION(
            "Format types and requires_printing mismatch in shader metadata.");
        return luisa::nullopt;
    }
    return CUDAShaderMetadata{
        .checksum = checksum.value(),
        .curve_bases = curve_bases.value(),
        .kind = kind,
        .enable_debug = enable_debug.value(),
        .requires_trace_closest = requires_trace_closest.value(),
        .requires_trace_any = requires_trace_any.value(),
        .requires_ray_query = requires_ray_query.value(),
        .requires_printing = requires_printing.value(),
        .requires_motion_blur = requires_motion_blur.value(),
        .max_register_count = max_register_count.value(),
        .block_size = block_size.value(),
        .argument_types = std::move(argument_types.value()),
        .argument_usages = std::move(argument_usages.value()),
        .format_types = std::move(format_types.value())};
}

}// namespace luisa::compute::cuda

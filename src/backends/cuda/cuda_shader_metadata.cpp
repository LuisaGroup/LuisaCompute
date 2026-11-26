#include <charconv>
#include <type_traits>

#include <luisa/ast/expression.h>
#include <luisa/ast/statement.h>
#include <luisa/core/stl/format.h>
#include <luisa/core/stl/unordered_map.h>
#include <luisa/core/logging.h>
#include <luisa/core/magic_enum.h>
#include "cuda_shader_metadata.h"

namespace luisa::compute::cuda {

namespace {

[[nodiscard]] luisa::optional<uint32_t> extract_const_uint(const Expression *expr) noexcept {
    while (expr != nullptr && expr->tag() == Expression::Tag::CAST) {
        const auto *cast_expr = static_cast<const CastExpr *>(expr);
        if (cast_expr->op() != CastOp::STATIC) {
            return luisa::nullopt;
        }
        expr = cast_expr->expression();
    }
    if (expr == nullptr || expr->tag() != Expression::Tag::LITERAL) {
        return luisa::nullopt;
    }
    const auto *lit = static_cast<const LiteralExpr *>(expr);
    return luisa::visit(
        [](auto v) noexcept -> luisa::optional<uint32_t> {
            using T = std::decay_t<decltype(v)>;
            if constexpr (std::is_integral_v<T> && !std::is_same_v<T, bool>) {
                return static_cast<uint32_t>(v);
            }
            return luisa::nullopt;
        },
        lit->value());
}

[[nodiscard]] luisa::vector<uint32_t> analyze_function_texture_masks(
    Function f,
    luisa::unordered_map<const detail::FunctionBuilder *, luisa::vector<uint32_t>> &memo,
    luisa::unordered_set<const detail::FunctionBuilder *> &visiting) noexcept {
    const auto *key = f.builder();
    if (auto iter = memo.find(key); iter != memo.end()) {
        return iter->second;
    }
    luisa::vector<uint32_t> masks(f.arguments().size(), 0u);
    auto make_conservative = [&]() noexcept {
        for (size_t i = 0u; i < f.arguments().size(); ++i) {
            if (f.arguments()[i].type()->is_texture()) {
                masks[i] = CUDAShaderMetadata::texture_usage_default_conservative;
            }
        }
    };
    if (!visiting.emplace(key).second) {
        make_conservative();
        return masks;
    }

    luisa::unordered_map<uint32_t, size_t> param_index_by_uid;
    param_index_by_uid.reserve(f.arguments().size());
    bool has_texture_param = false;
    for (size_t i = 0u; i < f.arguments().size(); ++i) {
        param_index_by_uid.emplace(f.arguments()[i].uid(), i);
        if (f.arguments()[i].type()->is_texture()) {
            has_texture_param = true;
        }
    }

    auto resolve_param_index = [&](const Expression *expr) noexcept -> luisa::optional<size_t> {
        if (expr != nullptr && expr->tag() == Expression::Tag::REF) {
            const auto uid = static_cast<const RefExpr *>(expr)->variable().uid();
            if (auto iter = param_index_by_uid.find(uid); iter != param_index_by_uid.end()) {
                return iter->second;
            }
        }
        return luisa::nullopt;
    };

    auto apply_mask_to_expr = [&](const Expression *tex_expr, uint32_t bits) noexcept {
        if (auto idx = resolve_param_index(tex_expr); idx.has_value()) {
            masks[*idx] |= bits;
        } else if (has_texture_param) {
            for (size_t i = 0u; i < f.arguments().size(); ++i) {
                if (f.arguments()[i].type()->is_texture()) {
                    masks[i] |= bits;
                }
            }
        }
    };

    if (f.body() != nullptr) {
        traverse_expressions<true>(
            f.body(),
            [&](const Expression *expr) noexcept {
                if (expr->tag() != Expression::Tag::CALL) {
                    return;
                }
                const auto *call = static_cast<const CallExpr *>(expr);
                const auto args = call->arguments();
                switch (call->op()) {
                    case CallOp::TEXTURE_READ:
                    case CallOp::TEXTURE_WRITE:
                        if (!args.empty()) {
                            apply_mask_to_expr(args.front(), CUDAShaderMetadata::texture_uses_surface_bit);
                        }
                        break;
                    case CallOp::TEXTURE2D_SAMPLE:
                    case CallOp::TEXTURE2D_SAMPLE_LEVEL:
                    case CallOp::TEXTURE2D_SAMPLE_GRAD:
                    case CallOp::TEXTURE2D_SAMPLE_GRAD_LEVEL:
                    case CallOp::TEXTURE3D_SAMPLE:
                    case CallOp::TEXTURE3D_SAMPLE_LEVEL:
                    case CallOp::TEXTURE3D_SAMPLE_GRAD:
                    case CallOp::TEXTURE3D_SAMPLE_GRAD_LEVEL: {
                        if (args.size() >= 3u) {
                            const auto f_const = extract_const_uint(args[args.size() - 2u]);
                            const auto a_const = extract_const_uint(args[args.size() - 1u]);
                            uint32_t bits = CUDAShaderMetadata::texture_sampler_mask_bits;
                            if (f_const.has_value() && a_const.has_value()) {
                                const auto code = ((*f_const & 3u) << 2u) | (*a_const & 3u);
                                bits = 1u << code;
                            }
                            apply_mask_to_expr(args.front(), bits);
                        }
                        break;
                    }
                    case CallOp::CUSTOM: {
                        const auto callee = call->custom();
                        const auto callee_masks = analyze_function_texture_masks(callee, memo, visiting);
                        for (size_t j = 0u; j < args.size() && j < callee_masks.size(); ++j) {
                            if (callee_masks[j] != 0u) {
                                apply_mask_to_expr(args[j], callee_masks[j]);
                            }
                        }
                        break;
                    }
                    default:
                        break;
                }
            },
            [](const Statement *) noexcept {},
            [](const Statement *) noexcept {});
    }

    visiting.erase(key);
    memo.emplace(key, masks);
    return masks;
}

}// namespace

luisa::vector<uint32_t> analyze_cuda_texture_sampler_masks(Function kernel) noexcept {
    bool has_texture = false;
    for (auto arg : kernel.arguments()) {
        if (arg.type()->is_texture()) {
            has_texture = true;
            break;
        }
    }
    if (!has_texture) {
        return {};
    }
    luisa::unordered_map<const detail::FunctionBuilder *, luisa::vector<uint32_t>> memo;
    luisa::unordered_set<const detail::FunctionBuilder *> visiting;
    return analyze_function_texture_masks(kernel, memo, visiting);
}

bool cuda_shader_code_matches_format(luisa::span<const std::byte> code,
                                      CUDAShaderMetadata::CodeFormat format) noexcept {
    if (code.empty()) { return false; }
    auto read_u32 = [&code](size_t offset) noexcept {
        return std::to_integer<uint32_t>(code[offset]) |
               (std::to_integer<uint32_t>(code[offset + 1u]) << 8u) |
               (std::to_integer<uint32_t>(code[offset + 2u]) << 16u) |
               (std::to_integer<uint32_t>(code[offset + 3u]) << 24u);
    };
    auto has_optix_ir_magic = code.size() >= 4u && read_u32(0u) == 0x7f4e43edu;
    if (format == CUDAShaderMetadata::CodeFormat::PTX) { return !has_optix_ir_magic; }
    if (format != CUDAShaderMetadata::CodeFormat::OPTIX_IR ||
        !has_optix_ir_magic || code.size() < 24u) { return false; }
    auto read_u16 = [&code](size_t offset) noexcept {
        return std::to_integer<uint32_t>(code[offset]) |
               (std::to_integer<uint32_t>(code[offset + 1u]) << 8u);
    };
    auto header_size = read_u16(12u);
    auto ir_level = read_u16(14u);
    auto scalar_fields_end = read_u32(16u);
    auto blob_data_end = read_u32(20u);
    return header_size == 24u && ir_level == 2u &&
           scalar_fields_end >= header_size &&
           scalar_fields_end <= blob_data_end && blob_data_end < code.size();
}

luisa::string serialize_cuda_shader_metadata(const CUDAShaderMetadata &metadata) noexcept {
    LUISA_ASSERT(metadata.ray_query_payload_count >= 2u && metadata.ray_query_payload_count <= 32u,
                 "Cannot serialize an unresolved or invalid ray-query payload count.");
    luisa::string result;
    result.append(luisa::format("CHECKSUM {:016x} ", metadata.checksum));
    result.append(luisa::format("KIND {} ", metadata.kind == CUDAShaderMetadata::Kind::UNKNOWN ?
                                                "UNKNOWN" :
                                            metadata.kind == CUDAShaderMetadata::Kind::COMPUTE ?
                                                "COMPUTE" :
                                            metadata.kind == CUDAShaderMetadata::Kind::TILE ?
                                                "TILE" :
                                                "RAY_TRACING"));
    switch (metadata.code_format) {
        case CUDAShaderMetadata::CodeFormat::PTX: result.append("CODE_FORMAT PTX "); break;
        case CUDAShaderMetadata::CodeFormat::OPTIX_IR: result.append("CODE_FORMAT OPTIX_IR "); break;
        default: LUISA_ERROR_WITH_LOCATION("Invalid CUDA shader code format.");
    }
    result.append(metadata.enable_debug ? "DEBUG TRUE " : "DEBUG FALSE ");
    result.append(metadata.requires_trace_closest ? "TRACE_CLOSEST TRUE " : "TRACE_CLOSEST FALSE ");
    result.append(metadata.requires_trace_any ? "TRACE_ANY TRUE " : "TRACE_ANY FALSE ");
    result.append(metadata.requires_ray_query ? "RAY_QUERY TRUE " : "RAY_QUERY FALSE ");
    if (metadata.requires_ray_query) {
        result.append(metadata.requires_procedural_ray_query ? "RAY_QUERY_PROCEDURAL TRUE " : "RAY_QUERY_PROCEDURAL FALSE ");
    }
    result.append(luisa::format("RAY_QUERY_PAYLOAD_COUNT {} ", metadata.ray_query_payload_count));
    if (metadata.ray_query_payload_rw_mask != 0u) {
        result.append(luisa::format("RAY_QUERY_PAYLOAD_RW_MASK {:x} ", metadata.ray_query_payload_rw_mask));
    }
    result.append(metadata.requires_printing ? "PRINTING TRUE " : "PRINTING FALSE ");
    result.append(metadata.requires_grid_dependency_synchronize ? "GRID_DEPENDENCY_SYNC TRUE " : "GRID_DEPENDENCY_SYNC FALSE ");
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
    if (!metadata.texture_sampler_masks.empty()) {
        result.append(luisa::format("TEXTURE_SAMPLER_MASKS {} ", metadata.texture_sampler_masks.size()));
        for (auto mask : metadata.texture_sampler_masks) {
            result.append(luisa::format("{:x} ", mask));
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
    luisa::optional<CUDAShaderMetadata::CodeFormat> code_format;
    luisa::optional<bool> enable_debug;
    luisa::optional<bool> requires_trace_closest;
    luisa::optional<bool> requires_trace_any;
    luisa::optional<bool> requires_ray_query;
    luisa::optional<bool> requires_procedural_ray_query;
    luisa::optional<uint32_t> ray_query_payload_count;
    luisa::optional<uint32_t> ray_query_payload_rw_mask;
    luisa::optional<bool> requires_printing;
    luisa::optional<bool> requires_grid_dependency_synchronize;
    luisa::optional<bool> requires_motion_blur;
    luisa::optional<uint> max_register_count;
    luisa::optional<luisa::vector<luisa::string>> argument_types;
    luisa::optional<luisa::vector<Usage>> argument_usages;
    luisa::optional<luisa::vector<uint32_t>> texture_sampler_masks;
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
            } else if (x == "TILE") {
                kind = CUDAShaderMetadata::Kind::TILE;
            } else {
                LUISA_WARNING_WITH_LOCATION("Invalid kind '{}' in shader metadata.", x);
                return luisa::nullopt;
            }
        } else if (token == "CODE_FORMAT") {
            if (code_format.has_value()) {
                LUISA_WARNING_WITH_LOCATION("Duplicate code format in shader metadata.");
                return luisa::nullopt;
            }
            auto x = read_token();
            if (x == "PTX") {
                code_format.emplace(CUDAShaderMetadata::CodeFormat::PTX);
            } else if (x == "OPTIX_IR") {
                code_format.emplace(CUDAShaderMetadata::CodeFormat::OPTIX_IR);
            } else {
                LUISA_WARNING_WITH_LOCATION("Invalid code format '{}' in shader metadata.", x);
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
        } else if (token == "RAY_QUERY_PROCEDURAL") {
            if (requires_procedural_ray_query.has_value()) {
                LUISA_WARNING_WITH_LOCATION(
                    "Duplicate requires_procedural_ray_query flag in shader metadata.");
                return luisa::nullopt;
            }
            auto x = read_token();
            if (x == "TRUE") {
                requires_procedural_ray_query.emplace(true);
            } else if (x == "FALSE") {
                requires_procedural_ray_query.emplace(false);
            } else {
                LUISA_WARNING_WITH_LOCATION(
                    "Invalid requires_procedural_ray_query flag '{}' in shader metadata.", x);
                return luisa::nullopt;
            }
        } else if (token == "RAY_QUERY_PAYLOAD_COUNT") {
            if (ray_query_payload_count.has_value()) {
                LUISA_WARNING_WITH_LOCATION("Duplicate ray-query payload count in shader metadata.");
                return luisa::nullopt;
            }
            auto x = parse_number(read_token());
            if (!x.has_value() || x.value() < 2u || x.value() > 32u) {
                LUISA_WARNING_WITH_LOCATION("Invalid ray-query payload count in shader metadata; expected 2 through 32 words.");
                return luisa::nullopt;
            }
            ray_query_payload_count.emplace(static_cast<uint32_t>(x.value()));
        } else if (token == "RAY_QUERY_PAYLOAD_RW_MASK") {
            if (ray_query_payload_rw_mask.has_value()) {
                LUISA_WARNING_WITH_LOCATION("Duplicate ray-query payload RW mask in shader metadata.");
                return luisa::nullopt;
            }
            auto x = parse_checksum(read_token());
            if (!x.has_value() || x.value() > 0xffffffffull) {
                LUISA_WARNING_WITH_LOCATION("Invalid ray-query payload RW mask in shader metadata.");
                return luisa::nullopt;
            }
            ray_query_payload_rw_mask.emplace(static_cast<uint32_t>(x.value()));
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
        } else if (token == "GRID_DEPENDENCY_SYNC") {
            if (requires_grid_dependency_synchronize.has_value()) {
                LUISA_WARNING_WITH_LOCATION(
                    "Duplicate requires_grid_dependency_synchronize flag in shader metadata.");
                return luisa::nullopt;
            }
            auto x = read_token();
            if (x == "TRUE") {
                requires_grid_dependency_synchronize.emplace(true);
            } else if (x == "FALSE") {
                requires_grid_dependency_synchronize.emplace(false);
            } else {
                LUISA_WARNING_WITH_LOCATION(
                    "Invalid requires_grid_dependency_synchronize flag '{}' in shader metadata.", x);
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
        } else if (token == "TEXTURE_SAMPLER_MASKS") {
            if (texture_sampler_masks.has_value()) {
                LUISA_WARNING_WITH_LOCATION("Duplicate texture sampler masks in shader metadata.");
                return luisa::nullopt;
            }
            auto x = parse_number(read_token());
            if (!x.has_value()) {
                LUISA_WARNING_WITH_LOCATION("Invalid texture sampler masks count in shader metadata.");
                return luisa::nullopt;
            }
            auto count = x.value();
            luisa::vector<uint32_t> masks;
            masks.reserve(count);
            for (auto i = 0ull; i < count; i++) {
                auto raw = read_token();
                auto parsed = parse_checksum(raw);
                if (!parsed.has_value() || parsed.value() > CUDAShaderMetadata::texture_usage_default_conservative) {
                    LUISA_WARNING_WITH_LOCATION(
                        "Invalid texture sampler mask '{}' in shader metadata.", raw);
                    return luisa::nullopt;
                }
                masks.emplace_back(static_cast<uint32_t>(parsed.value()));
            }
            texture_sampler_masks.emplace(std::move(masks));
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
    if (code_format.value_or(CUDAShaderMetadata::CodeFormat::PTX) == CUDAShaderMetadata::CodeFormat::OPTIX_IR &&
        kind != CUDAShaderMetadata::Kind::RAY_TRACING) {
        LUISA_WARNING_WITH_LOCATION("OptiX IR requires ray-tracing shader metadata.");
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
    if (texture_sampler_masks.has_value() &&
        texture_sampler_masks->size() != argument_types->size()) {
        LUISA_WARNING_WITH_LOCATION(
            "Argument types and texture sampler masks mismatch in shader metadata.");
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
        // Sidecars written before this field contain PTX.
        .code_format = code_format.value_or(CUDAShaderMetadata::CodeFormat::PTX),
        .enable_debug = enable_debug.value(),
        .requires_trace_closest = requires_trace_closest.value(),
        .requires_trace_any = requires_trace_any.value(),
        .requires_ray_query = requires_ray_query.value(),
        // Sidecars written before procedural specialization preserve custom IS capability.
        .requires_procedural_ray_query = requires_ray_query.value() ? requires_procedural_ray_query.value_or(true) : false,
        // Sidecars written before the payload ABI field used the two-word context pointer.
        .ray_query_payload_count = ray_query_payload_count.value_or(2u),
        .ray_query_payload_rw_mask = ray_query_payload_rw_mask.value_or(0u),
        .requires_printing = requires_printing.value(),
        // Sidecars written before this field belong to kernels without a
        // programmatic dependency wait.
        .requires_grid_dependency_synchronize = requires_grid_dependency_synchronize.value_or(false),
        .requires_motion_blur = requires_motion_blur.value(),
        .max_register_count = max_register_count.value(),
        .block_size = block_size.value(),
        .argument_types = std::move(argument_types.value()),
        .argument_usages = std::move(argument_usages.value()),
        .texture_sampler_masks = std::move(texture_sampler_masks).value_or(luisa::vector<uint32_t>{}),
        .format_types = std::move(format_types.value())};
}

}// namespace luisa::compute::cuda

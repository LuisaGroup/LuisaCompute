#pragma once

#include <luisa/core/stl/variant.h>
#include <luisa/core/logging.h>
#include <luisa/ast/type.h>
#include <luisa/ast/type_registry.h>
#include <luisa/runtime/rhi/device_interface.h>

namespace luisa::compute {

/*
 * This class formats Print(fmt, args) statements in shader code, where
 * the args are packed into a struct of the following layout:
 * struct {
 *   uint size;
 *   uint fmt_id;
 *   Args... args;
 * };
 * When printing, the formatter ignores the first two 4B words of each
 * item and outputs the rest of the data according to the fmt string.
 */

namespace detail {

/*
 * The printer payload uses the Luisa type layout: every aggregate is placed at
 * its own alignment and advances by its own size; a matrix stores its columns
 * back to back, each column being a vector of the matrix dimension. The host
 * decoder below and every device-side emitter must agree on this walk, so it
 * lives in exactly one place and is shared by all of them.
 */

/// One aggregate step of the access path from a print argument to a nested value.
struct ShaderPrintPathStep {
    const Type *container;///< structure or array that owns the nested value
    uint32_t index;       ///< member index (structure) or element index (array)
};

/// A matrix-typed sub-value of the printer payload.
struct ShaderPrintMatrixField {
    size_t offset;                          ///< byte offset of the first column within the payload
    size_t column_stride;                   ///< bytes between two consecutive columns
    const Type *type;                       ///< matrix type; element()/dimension() describe a column
    uint32_t argument_index;                ///< index of the owning print argument
    luisa::vector<ShaderPrintPathStep> path;///< aggregate nesting inside that argument
};

/// Walks the printer payload layout of \p type starting at byte \p offset.
/// \p text receives the structural literals, \p leaf every scalar leaf, and
/// \p matrix every matrix-typed sub-value. \p path is the current aggregate nesting.
template<typename TextFn, typename LeafFn, typename MatrixFn>
void traverse_shader_print_layout(Type const *type, size_t offset,
                                  luisa::vector<ShaderPrintPathStep> &path,
                                  TextFn &&text, LeafFn &&leaf, MatrixFn &&matrix) noexcept {
    offset = luisa::align(offset, type->alignment());
    if (type->is_scalar()) {
        leaf(offset, type->tag());
    } else if (type->is_vector() || type->is_array()) {
        auto is_vector = type->is_vector();
        text(is_vector ? "(" : "[");
        auto elem = type->element();
        for (auto i = 0u; i < type->dimension(); i++) {
            if (is_vector) {
                traverse_shader_print_layout(elem, offset, path, text, leaf, matrix);
            } else {
                path.push_back(ShaderPrintPathStep{type, i});
                traverse_shader_print_layout(elem, offset, path, text, leaf, matrix);
                path.pop_back();
            }
            if (i + 1u < type->dimension()) { text(", "); }
            offset += elem->size();
        }
        text(is_vector ? ")" : "]");
    } else if (type->is_matrix()) {
        text("<");
        auto column = Type::vector(type->element(), type->dimension());
        matrix(offset, type, column->size());
        for (auto i = 0u; i < type->dimension(); i++) {
            traverse_shader_print_layout(column, offset, path, text, leaf, matrix);
            if (i + 1u < type->dimension()) { text(", "); }
            offset += column->size();
        }
        text(">");
    } else if (type->is_structure()) {
        text("{");
        for (auto i = 0u; i < type->members().size(); i++) {
            auto member = type->members()[i];
            offset = luisa::align(offset, member->alignment());
            path.push_back(ShaderPrintPathStep{type, i});
            traverse_shader_print_layout(member, offset, path, text, leaf, matrix);
            path.pop_back();
            if (i + 1u < type->members().size()) { text(", "); }
            offset += member->size();
        }
        text("}");
    } else {
        LUISA_ERROR_WITH_LOCATION(
            "Invalid argument type '{}' for printing.",
            type->description());
    }
}

/// Enumerates every matrix-typed sub-value of a print argument pack.
[[nodiscard]] inline luisa::vector<ShaderPrintMatrixField>
shader_print_matrix_fields(luisa::span<Type const *const> arguments) noexcept {
    luisa::vector<ShaderPrintMatrixField> fields;
    luisa::vector<ShaderPrintPathStep> path;
    auto offset = size_t{0u};
    auto noop_text = [](luisa::string_view) noexcept {};
    auto noop_leaf = [](size_t, Type::Tag) noexcept {};
    for (auto i = 0u; i < arguments.size(); i++) {
        auto arg = arguments[i];
        offset = luisa::align(offset, arg->alignment());
        auto on_matrix = [&](size_t matrix_offset, Type const *type, size_t stride) noexcept {
            fields.emplace_back(ShaderPrintMatrixField{
                .offset = matrix_offset,
                .column_stride = stride,
                .type = type,
                .argument_index = i,
                .path = path});
        };
        traverse_shader_print_layout(arg, offset, path, noop_text, noop_leaf, on_matrix);
        offset += arg->size();
    }
    return fields;
}

}// namespace detail

class ShaderPrintFormatter {

public:
    using Primitive = luisa::variant<
        Type::Tag, luisa::string>;

private:
    size_t _size{};
    luisa::vector<size_t> _offsets;
    luisa::vector<Primitive> _primitives;

public:
    ShaderPrintFormatter(luisa::string_view fmt, const Type *arg_pack, bool has_header = true) noexcept {
        auto args = arg_pack->members();
        size_t offset = 0;
        if (has_header) {
            LUISA_ASSERT(arg_pack->members().size() >= 2u &&
                             arg_pack->members()[0] == Type::of<uint>() &&
                             arg_pack->members()[1] == Type::of<uint>(),
                         "Invalid argument pack for shader printer.");
            args = args.subspan(2u);
            offset += 8;
        }
        luisa::string s;
        luisa::string f;
        auto commit_s = [this, &s] {
            if (!s.empty()) {
                _offsets.push_back(0u);
                _primitives.emplace_back(s);
                s.clear();
            }
        };
        while (!fmt.empty()) {
            auto c = fmt.front();
            fmt.remove_prefix(1u);
            if (c == '{') {
                if (!f.empty()) {
                    LUISA_ERROR_WITH_LOCATION("Invalid format string.");
                } else if (fmt.front() == '{') {// escape
                    fmt.remove_prefix(1u);
                    s.push_back('{');
                } else {
                    f.push_back('{');
                    commit_s();
                }
            } else if (c == '}') {
                if (!f.empty()) {// end of format group
                    f.push_back('}');
                    LUISA_ASSERT(f == "{}", "Unsupported format string '{}'.", f);// TODO: support more formats?
                    LUISA_ASSERT(!args.empty(), "Not enough arguments for shader printer.");
                    auto arg = args.front();
                    args = args.subspan(1u);
                    luisa::vector<detail::ShaderPrintPathStep> path;
                    detail::traverse_shader_print_layout(
                        arg, offset, path,
                        [&s, &commit_s](luisa::string_view text) noexcept {
                            s.append(text);
                            commit_s();
                        },
                        [this](size_t leaf_offset, Type::Tag tag) noexcept {
                            _offsets.push_back(leaf_offset);
                            _primitives.emplace_back(tag);
                        },
                        [](size_t, const Type *, size_t) noexcept {});
                    offset = luisa::align(offset, arg->alignment()) + arg->size();
                    f.clear();
                } else {// not in a format group, only escape is allowed
                    if (fmt.front() == '}') {
                        fmt.remove_prefix(1u);
                        s.push_back('}');
                    } else {
                        LUISA_ERROR_WITH_LOCATION("Invalid format string.");
                    }
                }
            } else {
                if (!f.empty()) {
                    f.push_back(c);
                } else {
                    s.push_back(c);
                }
            }
        }
        commit_s();
        LUISA_ASSERT(f.empty(), "Invalid format string.");
        if (!args.empty()) {
            LUISA_WARNING_WITH_LOCATION(
                "Too many arguments for shader printer. Ignored.");
        }
        _size = luisa::align(offset, arg_pack->alignment());
        LUISA_ASSERT(_size <= arg_pack->size(),
                     "Invalid argument pack for shader printer.");
        // optimize the format
        luisa::vector<Primitive> primitives;
        luisa::vector<size_t> offsets;
        primitives.reserve(_primitives.size());
        offsets.reserve(_offsets.size());
        for (auto i = 0u; i < _primitives.size(); i++) {
            luisa::visit(
                [&](auto &&p) noexcept {
                    using T = std::decay_t<decltype(p)>;
                    if constexpr (std::is_same_v<T, Type::Tag>) {
                        primitives.emplace_back(p);
                        offsets.emplace_back(_offsets[i]);
                    } else {
                        static_assert(std::is_same_v<T, luisa::string>);
                        if (primitives.empty() || !luisa::holds_alternative<luisa::string>(primitives.back())) {
                            primitives.emplace_back(p);
                            offsets.emplace_back(_offsets[i]);
                        } else {
                            luisa::get<luisa::string>(primitives.back()).append(p);
                        }
                    }
                },
                _primitives[i]);
        }
        _primitives = std::move(primitives);
        _offsets = std::move(offsets);
    }

    ~ShaderPrintFormatter() noexcept = default;

public:
    [[nodiscard]] auto size() const noexcept { return _size; }

    bool operator()(luisa::string &scratch, luisa::span<const std::byte> item) const noexcept {
        if (item.size() < _size) { return false; }
        for (auto i = 0u; i < _offsets.size(); i++) {
            auto data = item.data() + _offsets[i];
            luisa::visit(
                [&](auto &&p) noexcept {
                    using T = std::decay_t<decltype(p)>;
                    if constexpr (std::is_same_v<T, Type::Tag>) {
                        auto print_primitive = [&]<typename T>(T v) noexcept {
                            using TT = std::decay_t<T>;
                            std::memcpy(&v, data, sizeof(v));
                            if constexpr (std::is_same_v<TT, bool>) {
                                scratch.append(v ? "true" : "false");
                            } else if constexpr (luisa::is_integral_v<TT> && sizeof(TT) <= sizeof(short)) {
                                luisa::format_to(std::back_inserter(scratch), "{}", static_cast<int>(v));
                            } else {
                                if constexpr (std::is_same_v<TT, half>)
                                    luisa::format_to(std::back_inserter(scratch), "{}", static_cast<float>(v));
                                else
                                    luisa::format_to(std::back_inserter(scratch), "{}", v);
                            }
                        };
                        switch (p) {
                            case Type::Tag::BOOL: print_primitive(bool{}); break;
                            case Type::Tag::INT8: print_primitive(int8_t{}); break;
                            case Type::Tag::UINT8: print_primitive(uint8_t{}); break;
                            case Type::Tag::INT16: print_primitive(int16_t{}); break;
                            case Type::Tag::UINT16: print_primitive(uint16_t{}); break;
                            case Type::Tag::INT32: print_primitive(int32_t{}); break;
                            case Type::Tag::UINT32: print_primitive(uint32_t{}); break;
                            case Type::Tag::INT64: print_primitive(int64_t{}); break;
                            case Type::Tag::UINT64: print_primitive(uint64_t{}); break;
                            case Type::Tag::FLOAT16: print_primitive(half{}); break;
                            case Type::Tag::FLOAT32: print_primitive(float{}); break;
                            case Type::Tag::FLOAT64: print_primitive(double{}); break;
                            default: LUISA_ERROR_WITH_LOCATION("Unsupported type for shader printer.");
                        }
                    } else {
                        static_assert(std::is_same_v<T, luisa::string>);
                        scratch.append(p);
                    }
                },
                _primitives[i]);
        }
        return true;
    }
};

inline size_t format_shader_print(luisa::span<const luisa::unique_ptr<ShaderPrintFormatter>> formatters,
                                  luisa::span<const std::byte> contents,
                                  const DeviceInterface::StreamLogCallback &log = {}) noexcept {
    if (contents.empty()) { return 0u; }
    luisa::string scratch;
    scratch.reserve(1_k - 1u);
    auto offset = static_cast<size_t>(0u);
    while (offset < contents.size_bytes()) {
        struct Item {
            uint size;
            uint fmt;
            const std::byte data[];
        };
        static_assert(sizeof(Item) == 8u);
        auto raw = contents.data() + offset;
        auto *item = reinterpret_cast<const Item *>(raw);
        if (item->size == 0u) {
            LUISA_WARNING("Invalid print item size: 0.");
            return false;
        }
        if (auto item_end = offset + item->size;
            item_end > contents.size_bytes()) { break; }
        if (item->fmt < formatters.size()) {
            scratch.clear();
            luisa::span payload{raw, item->size};
            if ((*formatters[item->fmt])(scratch, payload)) {
                if (log) {
                    log(scratch);
                } else {
                    LUISA_INFO("[DEVICE] {}", scratch);
                }
            } else {
                break;
            }
        } else {
            LUISA_WARNING("Unknown print format: {}", item->fmt);
        }
        offset += item->size;
    }
    return offset;
}

}// namespace luisa::compute
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
                    auto encode = [this, &s, &commit_s](auto &&self, size_t offset, const Type *arg) noexcept -> void {
                        offset = luisa::align(offset, arg->alignment());
                        if (arg->is_scalar()) {
                            _offsets.push_back(offset);
                            _primitives.emplace_back(arg->tag());
                        } else if (arg->is_vector()) {
                            s.push_back('(');
                            commit_s();
                            for (auto i = 0u; i < arg->dimension(); i++) {
                                self(self, offset, arg->element());
                                if (i + 1u < arg->dimension()) {
                                    s.append(", ");
                                    commit_s();
                                }
                                offset += arg->element()->size();
                            }
                            s.push_back(')');
                            commit_s();
                        } else if (arg->is_array()) {
                            s.push_back('[');
                            commit_s();
                            for (auto i = 0u; i < arg->dimension(); i++) {
                                self(self, offset, arg->element());
                                if (i + 1u < arg->dimension()) {
                                    s.append(", ");
                                    commit_s();
                                }
                                offset += arg->element()->size();
                            }
                            s.push_back(']');
                            commit_s();
                        } else if (arg->is_matrix()) {
                            s.push_back('<');
                            commit_s();
                            auto column = Type::vector(arg->element(), arg->dimension());
                            for (auto i = 0u; i < arg->dimension(); i++) {
                                self(self, offset, column);
                                if (i + 1u < arg->dimension()) {
                                    s.append(", ");
                                    commit_s();
                                }
                                offset += column->size();
                            }
                            s.push_back('>');
                            commit_s();
                        } else if (arg->is_structure()) {
                            s.push_back('{');
                            commit_s();
                            for (auto i = 0u; i < arg->members().size(); i++) {
                                auto member = arg->members()[i];
                                self(self, offset, member);
                                if (i + 1u < arg->members().size()) {
                                    s.append(", ");
                                    commit_s();
                                }
                                offset += member->size();
                            }
                            s.push_back('}');
                            commit_s();
                        } else {
                            LUISA_ERROR_WITH_LOCATION(
                                "Invalid argument type '{}' for printing.",
                                arg->description());
                        }
                    };
                    offset = luisa::align(offset, arg->alignment());
                    encode(encode, offset, arg);
                    offset += arg->size();
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
    bool operator()(luisa::string &scratch, luisa::span<const std::byte> item) const noexcept {
        if (item.size() < _size) { return false; }
        for (auto i = 0u; i < _offsets.size(); i++) {
            auto data = item.data() + _offsets[i];
            luisa::visit(
                [&](auto &&p) noexcept {
                    using T = std::decay_t<decltype(p)>;
                    if constexpr (std::is_same_v<T, Type::Tag>) {
                        auto print_primitive = [&](auto v) noexcept {
                            using TT = std::decay_t<decltype(v)>;
                            std::memcpy(&v, data, sizeof(v));
                            if constexpr (std::is_same_v<TT, bool>) {
                                scratch.append(v ? "true" : "false");
                            } else if constexpr (luisa::is_integral_v<TT> && sizeof(TT) <= sizeof(short)) {
                                luisa::format_to(std::back_inserter(scratch), "{}", static_cast<int>(v));
                            } else {
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
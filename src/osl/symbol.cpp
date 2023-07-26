#include <luisa/core/logging.h>

#include <luisa/osl/type.h>
#include <luisa/osl/symbol.h>

namespace luisa::compute::osl {

const Symbol *Symbol::root() const noexcept {
    auto p = this;
    while (p->_parent != nullptr) { p = p->_parent; }
    return p;
}

void Symbol::add_child(const Symbol *symbol) noexcept {
    LUISA_ASSERT(symbol != nullptr, "Symbol is null.");
    LUISA_ASSERT(symbol->_parent == this, "Symbol parent mismatch.");
    LUISA_ASSERT(symbol->identifier().starts_with(_identifier),
                 "Symbol identifier mismatch.");
#ifndef NDEBUG
    LUISA_ASSERT(std::none_of(_children.begin(), _children.end(),
                              [symbol](auto &&c) noexcept { return c == symbol; }),
                 "Symbol already added.");
#endif
    _children.emplace_back(symbol);
}

luisa::string Symbol::dump() const noexcept {
    auto tag_string = dump(_tag);
    auto s = _type->tag() == Type::Tag::STRUCT ?
                 luisa::format("{}\tstruct {}", tag_string, _type->identifier()) :
                 luisa::format("{}\t{}", tag_string, _type->identifier());
    if (is_array()) {
        if (is_unbounded()) {
            s.append("[]");
        } else {
            s.append(luisa::format("[{}]", _array_length));
        }
    }
    s.append("\t").append(_identifier);
    if (!_initial_values.empty()) {
        s.append("\t");
        for (auto &&v : _initial_values) {
            s.append(v.dump()).append(" ");
        }
        s.pop_back();
    }
    if (!_hints.empty()) {
        s.append("\t");
        for (auto &&h : _hints) {
            s.append(h.dump()).append(" ");
        }
        s.pop_back();
    }
    if (_type->tag() == Type::Tag::STRUCT) {
        s.append("\t# ").append(static_cast<const StructType *>(_type)->dump());
    }
    if (_parent != nullptr) {
        s.append("\t# ").append("parent: ").append(_parent->identifier());
    }
    if (!_children.empty()) {
        s.append("\t# ").append("children: ");
        for (auto &&c : _children) {
            s.append(c->identifier()).append(" ");
        }
        s.pop_back();
    }
    return s;
}

luisa::string_view Symbol::dump(Symbol::Tag tag) noexcept {
    using namespace std::string_view_literals;
    switch (tag) {
        case Tag::SYM_PARAM: return "param"sv;
        case Tag::SYM_OUTPUT_PARAM: return "oparam"sv;
        case Tag::SYM_LOCAL: return "local"sv;
        case Tag::SYM_TEMP: return "temp"sv;
        case Tag::SYM_GLOBAL: return "global"sv;
        case Tag::SYM_CONST: return "const"sv;
        default: break;
    }
    LUISA_ERROR_WITH_LOCATION("Invalid symbol tag.");
}

Symbol::Symbol(Symbol::Tag tag, const Type *type, int array_length,
               const Symbol *parent, luisa::string identifier,
               luisa::vector<Literal> initial_values,
               luisa::vector<Hint> hints) noexcept
    : _tag{tag}, _array_length{array_length},
      _type{type}, _parent{parent},
      _identifier{std::move(identifier)},
      _initial_values{std::move(initial_values)},
      _hints{std::move(hints)} {}

}// namespace luisa::compute::osl

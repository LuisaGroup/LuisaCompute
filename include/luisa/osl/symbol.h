//
// Created by Mike Smith on 2023/7/24.
//

#pragma once

#include <luisa/core/stl/vector.h>
#include <luisa/osl/literal.h>
#include <luisa/osl/hint.h>

namespace luisa::compute::osl {

class Type;

class LC_OSL_API Symbol final {

public:
    enum struct Tag {
        PARAM,
        OUTPUT_PARAM,
        LOCAL,
        TEMP,
        GLOBAL,
        CONST
    };

private:
    Tag _tag;
    int _array_length;
    const Type *_type;
    const Symbol *_parent;
    luisa::string _identifier;
    luisa::vector<Literal> _initial_values;
    luisa::vector<Hint> _hints;
    luisa::vector<const Symbol *> _children;

public:
    Symbol(Tag tag, const Type *type,
           int array_length,
           const Symbol *parent,
           luisa::string identifier,
           luisa::vector<Literal> initial_values,
           luisa::vector<Hint> hints) noexcept
        : _tag{tag}, _array_length{array_length},
          _type{type}, _parent{parent},
          _identifier{std::move(identifier)},
          _initial_values{std::move(initial_values)},
          _hints{std::move(hints)} {}
    ~Symbol() noexcept = default;
    Symbol(Symbol &&) noexcept = default;
    Symbol(const Symbol &) noexcept = delete;
    Symbol &operator=(Symbol &&) noexcept = default;
    Symbol &operator=(const Symbol &) noexcept = delete;
    [[nodiscard]] auto tag() const noexcept { return _tag; }
    [[nodiscard]] auto type() const noexcept { return _type; }
    [[nodiscard]] auto identifier() const noexcept { return luisa::string_view{_identifier}; }
    [[nodiscard]] auto initial_values() const noexcept { return luisa::span{_initial_values}; }
    [[nodiscard]] auto hints() const noexcept { return luisa::span{_hints}; }
    [[nodiscard]] auto is_array() const noexcept { return _array_length != 0; }
    [[nodiscard]] auto is_unbounded() const noexcept { return _array_length < 0; }
    [[nodiscard]] auto array_length() const noexcept { return _array_length; }
    [[nodiscard]] auto parent() const noexcept { return _parent; }
    [[nodiscard]] auto is_root() const noexcept { return _parent == nullptr; }
    [[nodiscard]] const Symbol *root() const noexcept;
    [[nodiscard]] auto children() const noexcept { return luisa::span{_children}; }
    void add_child(const Symbol *symbol) noexcept;

    // for debugging
    [[nodiscard]] luisa::string dump() const noexcept;
};

}// namespace luisa::compute::osl

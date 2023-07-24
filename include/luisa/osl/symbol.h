//
// Created by Mike Smith on 2023/7/24.
//

#pragma once

#include <luisa/core/stl/vector.h>
#include <luisa/osl/literal.h>

namespace luisa::compute::osl {

class Type;

class Symbol final {

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
    const Type *_type;
    std::string _identifier;
    luisa::vector<Literal> _initial_values;
    luisa::vector<std::string> _hints;

public:
    Symbol(Tag tag, const Type *type,
           std::string identifier,
           luisa::vector<Literal> initial_values,
           luisa::vector<std::string> hints) noexcept
        : _tag{tag}, _type{type},
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
};

}// namespace luisa::compute::osl

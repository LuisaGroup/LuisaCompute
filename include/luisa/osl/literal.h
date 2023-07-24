//
// Created by Mike Smith on 2023/7/24.
//

#pragma once

#include <luisa/core/dll_export.h>
#include <luisa/core/stl/string.h>

namespace luisa::compute::osl {

class LC_OSL_API Literal {

public:
    enum struct Tag : uint32_t {
        INT,
        FLOAT,
        STRING
    };

private:
    Tag _tag;
    uint32_t _string_length;
    union {
        int i;
        float f;
        char *s;
    } _value;

public:
    explicit Literal(int i) noexcept;
    explicit Literal(float f) noexcept;
    explicit Literal(luisa::string_view s) noexcept;
    ~Literal() noexcept;
    Literal(Literal &&) noexcept;
    Literal(const Literal &) noexcept = delete;
    Literal &operator=(Literal &&) noexcept = delete;
    Literal &operator=(const Literal &) noexcept = delete;
    [[nodiscard]] auto tag() const noexcept { return _tag; }
    [[nodiscard]] auto is_int() const noexcept { return _tag == Tag::INT; }
    [[nodiscard]] auto is_float() const noexcept { return _tag == Tag::FLOAT; }
    [[nodiscard]] auto is_string() const noexcept { return _tag == Tag::STRING; }
    [[nodiscard]] int as_int() const noexcept;
    [[nodiscard]] float as_float() const noexcept;
    [[nodiscard]] luisa::string_view as_string() const noexcept;
};

}// namespace luisa::compute::osl

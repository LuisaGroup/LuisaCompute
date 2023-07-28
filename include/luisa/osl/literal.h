#pragma once

#include <luisa/core/dll_export.h>
#include <luisa/core/stl/string.h>

namespace luisa::compute::osl {

class LC_OSL_API Literal {

public:
    enum struct Tag : uint32_t {
        NUMBER,
        STRING
    };

private:
    Tag _tag;
    uint32_t _string_length;
    union {
        double n;
        char *s;
    } _value;

public:
    explicit Literal(double n) noexcept;
    explicit Literal(luisa::string_view s) noexcept;
    ~Literal() noexcept;
    Literal(Literal &&) noexcept;
    Literal(const Literal &) noexcept = delete;
    Literal &operator=(Literal &&) noexcept = delete;
    Literal &operator=(const Literal &) noexcept = delete;
    [[nodiscard]] auto tag() const noexcept { return _tag; }
    [[nodiscard]] auto is_number() const noexcept { return _tag == Tag::NUMBER; }
    [[nodiscard]] auto is_string() const noexcept { return _tag == Tag::STRING; }
    [[nodiscard]] int as_int() const noexcept;
    [[nodiscard]] float as_float() const noexcept;
    [[nodiscard]] luisa::string_view as_string() const noexcept;

    // for debugging
    [[nodiscard]] luisa::string dump() const noexcept;
};

}// namespace luisa::compute::osl

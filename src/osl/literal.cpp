#include <luisa/core/logging.h>
#include <luisa/osl/literal.h>

namespace luisa::compute::osl {

Literal::Literal(double n) noexcept
    : _tag{Tag::NUMBER}, _string_length{}, _value{.n = n} {}

Literal::Literal(luisa::string_view s) noexcept
    : _tag{Tag::STRING},
      _string_length{static_cast<uint32_t>(s.size())},
      _value{.s = luisa::allocate_with_allocator<char>(s.size())} {
    // copy string
    std::memcpy(_value.s, s.data(), s.size());
}

Literal::~Literal() noexcept {
    if (is_string() && _value.s != nullptr) {
        luisa::deallocate_with_allocator(_value.s);
    }
}

Literal::Literal(Literal &&literal) noexcept
    : _tag{literal._tag},
      _string_length{literal._string_length},
      _value{literal._value} {
    if (literal.is_string()) {
        literal._string_length = 0u;
        literal._value.s = nullptr;
    }
}

int Literal::as_int() const noexcept {
    if (!is_number()) {
        LUISA_WARNING_WITH_LOCATION(
            "Literal is not an integer.");
        return 0;
    }
    auto i = static_cast<int>(_value.n);
    if (static_cast<double>(i) != _value.n) {
        LUISA_WARNING_WITH_LOCATION(
            "Literal is a number but cannot "
            "be converted to integer.");
        return 0;
    }
    return i;
}

float Literal::as_float() const noexcept {
    if (!is_number()) {
        LUISA_WARNING_WITH_LOCATION(
            "Literal is not a float.");
        return 0.f;
    }
    return static_cast<float>(_value.n);
}

luisa::string_view Literal::as_string() const noexcept {
    if (!is_string()) {
        LUISA_WARNING_WITH_LOCATION(
            "Literal is not a string.");
        return {};
    }
    return {_value.s, static_cast<size_t>(_string_length)};
}

luisa::string Literal::dump() const noexcept {
    if (is_number()) {
        return luisa::format("{}", _value.n);
    }
    luisa::string s{"\""};
    for (auto c : as_string()) {
        // escape special characters
        if (c == '\t') {
            s.append("\\t");
        } else if (c == '\n') {
            s.append("\\n");
        } else if (c == '\r') {
            s.append("\\r");
        } else if (c == '\f') {
            s.append("\\f");
        } else if (c == '\b') {
            s.append("\\b");
        } else if (c == '\\') {
            s.append("\\\\");
        } else if (c == '\"') {
            s.append("\\\"");
        } else if (c == '\0') {
            s.append("\\0");
        } else {
            s.push_back(c);
        }
    }
    s.append("\"");
    return s;
}

}// namespace luisa::compute::osl

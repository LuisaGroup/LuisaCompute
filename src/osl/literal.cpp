//
// Created by Mike Smith on 2023/7/24.
//

#include <luisa/core/logging.h>
#include <luisa/osl/literal.h>

namespace luisa::compute::osl {

Literal::Literal(int i) noexcept
    : _tag{Tag::INT}, _string_length{}, _value{.i = i} {}

Literal::Literal(float f) noexcept
    : _tag{Tag::FLOAT}, _string_length{}, _value{.f = f} {}

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
    if (!is_int()) {
        LUISA_WARNING_WITH_LOCATION(
            "Literal is not an integer.");
        return 0;
    }
    return _value.i;
}

float Literal::as_float() const noexcept {
    if (!is_float()) {
        LUISA_WARNING_WITH_LOCATION(
            "Literal is not a float.");
        return 0.f;
    }
    return _value.f;
}

luisa::string_view Literal::as_string() const noexcept {
    if (!is_string()) {
        LUISA_WARNING_WITH_LOCATION(
            "Literal is not a string.");
        return {};
    }
    return {_value.s, static_cast<size_t>(_string_length)};
}

}// namespace luisa::compute::osl

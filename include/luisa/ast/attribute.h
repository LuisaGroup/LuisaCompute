#pragma once
#include <luisa/core/stl/string.h>
namespace luisa::compute {
struct Attribute {
    luisa::string key;
    luisa::string value;
    Attribute() noexcept = default;
    [[nodiscard]] operator bool() const noexcept {
        return !key.empty();
    }
    Attribute(
        luisa::string &&key,
        luisa::string &&value) noexcept
        : key(std::move(key)), value(std::move(value)) {}
};
};// namespace luisa::compute
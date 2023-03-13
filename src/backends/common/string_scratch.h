//
// Created by Mike on 3/12/2023.
//

#pragma once

#include <core/basic_types.h>
#include <core/stl/string.h>

namespace luisa::compute {

class StringScratch {

private:
    luisa::string _buffer;

public:
    explicit StringScratch(size_t reserved_size) noexcept;
    StringScratch() noexcept;
    StringScratch &operator<<(std::string_view s) noexcept;
    StringScratch &operator<<(const char *s) noexcept;
    StringScratch &operator<<(const std::string &s) noexcept;
    StringScratch &operator<<(bool x) noexcept;
    StringScratch &operator<<(float x) noexcept;
    StringScratch &operator<<(int x) noexcept;
    StringScratch &operator<<(uint x) noexcept;
    StringScratch &operator<<(size_t x) noexcept;
    [[nodiscard]] luisa::string_view view() const noexcept;
    [[nodiscard]] const char *c_str() const noexcept;
    [[nodiscard]] bool empty() const noexcept;
    [[nodiscard]] size_t size() const noexcept;
    void pop_back() noexcept;
    void clear() noexcept;
    [[nodiscard]] char back() const noexcept;
};
}// namespace luisa::compute

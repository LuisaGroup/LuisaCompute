//
// Created by Mike Smith on 2021/2/3.
//

#pragma once

namespace luisa {

struct Noncopyable {
    Noncopyable() = default;
    Noncopyable(const Noncopyable &) = delete;
    Noncopyable &operator=(const Noncopyable &) = delete;
    Noncopyable(Noncopyable &&) noexcept = default;
    Noncopyable &operator=(Noncopyable &&) noexcept = default;
};

template<typename U>
constexpr auto always_false = false;

template<typename U>
constexpr auto always_true = true;

}// namespace luisa

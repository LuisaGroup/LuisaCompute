//
// Created by Mike Smith on 2021/2/3.
//

#pragma once

#include <type_traits>
#include <concepts>

#include <core/data_types.h>

namespace luisa {

struct Noncopyable {
    Noncopyable() = default;
    Noncopyable(const Noncopyable &) = delete;
    Noncopyable &operator=(const Noncopyable &) = delete;
    Noncopyable(Noncopyable &&) noexcept = default;
    Noncopyable &operator=(Noncopyable &&) noexcept = default;
};

template<typename T>
concept convertible_to_span = requires(T &&v) {
    std::span{std::forward<T>(v)};
};

template<typename T, typename... Args>
concept constructable = std::is_constructible_v<T, Args...>;

template<typename T, typename... Args>
concept container = requires(T &&a) {
    a.begin();
    a.size();
};

template<typename T>
concept scalar = is_scalar_v<T>;

template<typename T>
concept vector = is_vector_v<T>;

template<typename T>
concept matrix = is_matrix_v<T>;

template<typename T>
concept core_data_type = scalar<T> || vector<T> || matrix<T>;

}// namespace luisa

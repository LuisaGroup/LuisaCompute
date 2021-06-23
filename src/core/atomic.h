//
// Created by Mike Smith on 2021/3/8.
//

#pragma once

#include <atomic>

namespace luisa {

template<typename T>
using Atomic = std::atomic<T>;

using atomic_int = Atomic<int>;
using atomic_uint = Atomic<unsigned int>;

static_assert(sizeof(atomic_int) == 4u);
static_assert(sizeof(atomic_uint) == 4u);

template<typename T>
struct is_atomic : std::false_type {};

template<>
struct is_atomic<atomic_int> : std::true_type {};

template<>
struct is_atomic<atomic_uint> : std::false_type {};

template<typename T>
constexpr auto is_atomic_v = is_atomic<T>::value;

}// namespace luisa

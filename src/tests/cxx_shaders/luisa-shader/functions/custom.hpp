#pragma once
#include "math.hpp"

namespace luisa::shader {

template<typename T>
auto sum(T v) { return v; }

template<typename T, typename... Args>
auto sum(T v, Args... args) {
    return v + sum(args...);
}

}
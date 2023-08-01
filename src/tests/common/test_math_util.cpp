/**
 * @file: tests/common/math_util.cpp
 * @author: sailing-innocent
 * @date: 2023-07-28
 * @brief: the implementation for commonly used math utility functions used in tests
*/

#include "test_math_util.h"

using namespace luisa;
using namespace luisa::compute;

namespace luisa::test {

float lcg_host(uint &state) noexcept {
    constexpr auto lcg_a = 1664525u;
    constexpr auto lcg_c = 1013904223u;
    state = lcg_a * state + lcg_c;
    return cast<float>(state & 0x00ffffffu) *
           (1.0f / static_cast<float>(0x01000000u));
}

Callable<float(uint &)> lcg_callable() {
    Callable lcg = [](UInt &state) noexcept {
        constexpr uint lcg_a = 1664525u;
        constexpr uint lcg_c = 1013904223u;
        state = lcg_a * state + lcg_c;
        return cast<float>(state & 0x00ffffffu) *
               (1.0f / static_cast<float>(0x01000000u));
    };
    return lcg;
}

}// namespace luisa::test
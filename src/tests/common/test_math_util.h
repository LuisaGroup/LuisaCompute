#pragma once
/**
 * @file: tests/common/math_util.h
 * @author: sailing-innocent
 * @date: 2023-07-28
 * @brief: commonly used math utility functions used in tests
*/

#include <luisa/dsl/sugar.h>
using namespace luisa;
using namespace luisa::compute;

namespace luisa::test {

// pseudo random number generator
float lcg_host(uint &state) noexcept;
Callable<float(uint &)> lcg_callable();

}// namespace luisa::test
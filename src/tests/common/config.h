#pragma once

#include "doctest.h"

namespace luisa::test {

[[nodiscard]] int argc() noexcept;
[[nodiscard]] const char *const *argv() noexcept;

}// namespace luisa::test


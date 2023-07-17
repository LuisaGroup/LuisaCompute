//
// Created by Mike Smith on 2023/4/5.
//

#pragma once

#define DOCTEST_CONFIG_NO_EXCEPTIONS_BUT_WITH_ALL_ASSERTS
#include "doctest.h"

namespace luisa::test {

[[nodiscard]] int argc() noexcept;
[[nodiscard]] const char *const *argv() noexcept;

}// namespace luisa::test

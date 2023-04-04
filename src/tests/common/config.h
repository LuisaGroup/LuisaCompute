//
// Created by Mike Smith on 2023/4/5.
//

#pragma once

#include <tests/common/doctest.h>

namespace luisa::test {

[[nodiscard]] int argc() noexcept;
[[nodiscard]] const char *const *argv() noexcept;

}// namespace luisa::test

#pragma once

#define DOCTEST_CONFIG_NO_EXCEPTIONS_BUT_WITH_ALL_ASSERTS
#include "doctest.h"

namespace luisa::test {

[[nodiscard]] int argc() noexcept;
[[nodiscard]] const char *const *argv() noexcept;
[[nodiscard]] int supported_backends_count() noexcept;
[[nodiscard]] const char* const *supported_backends() noexcept;

}// namespace luisa::test

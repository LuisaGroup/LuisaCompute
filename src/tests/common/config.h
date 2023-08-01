#pragma once

#define DOCTEST_CONFIG_NO_EXCEPTIONS_BUT_WITH_ALL_ASSERTS
#include "doctest.h"

namespace luisa::test {

[[nodiscard]] int argc() noexcept;
[[nodiscard]] const char *const *argv() noexcept;
[[nodiscard]] int backends_to_test_count() noexcept;
[[nodiscard]] const char *const *backends_to_test() noexcept;

}// namespace luisa::test

// useful macros

// for most test cases. these two headers are necessary
#include <luisa/runtime/context.h>
#include <luisa/runtime/device.h>

#define LUISA_TEST_CASE_WITH_DEVICE(name, condition)                          \
    TEST_CASE(name) {                                                         \
        Context context{luisa::test::argv()[0]};                              \
        for (auto i = 0; i < luisa::test::backends_to_test_count(); i++) {  \
            luisa::string device_name = luisa::test::backends_to_test()[i]; \
            SUBCASE(device_name.c_str()) {                                    \
                Device device = context.create_device(device_name.c_str());   \
                REQUIRE(condition);                                           \
            }                                                                 \
        }                                                                     \
    }
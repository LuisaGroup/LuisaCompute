/**
 * @file test/feat/common/test_device.cpp
 * @author sailing-innocent
 * @date 2023/07/30
 * @brief the device test suite
*/
#include "common/config.h"

#include <luisa/runtime/device.h>
#include <luisa/runtime/context.h>

namespace luisa::test {

class WrappedDevice {
public:
    auto device() noexcept { return m_device; }
private:
    luisa::compute::Device m_device;
};

int test_wrapped_device(luisa::string cwd, luisa::string device_name) {
    return 0;
}
int test_create_device(luisa::string cwd, luisa::string device_name) {
    return 0;
}

}// namespace luisa::test

TEST_SUITE("runtime") {
    TEST_CASE("device::create_device") {
        for (auto i = 0; i < luisa::test::backends_to_test_count(); i++) {
            luisa::string device_name = luisa::test::backends_to_test()[i];
            SUBCASE(device_name.c_str()) {
                REQUIRE(luisa::test::test_create_device(luisa::test::argv()[0], device_name) == 0);
            }
        }
    }
    TEST_CASE("device::wrapped_device") {
        for (auto i = 0; i < luisa::test::backends_to_test_count(); i++) {
            luisa::string device_name = luisa::test::backends_to_test()[i];
            SUBCASE(device_name.c_str()) {
                REQUIRE(luisa::test::test_wrapped_device(luisa::test::argv()[0], device_name) == 0);
            }
        }
    }
}
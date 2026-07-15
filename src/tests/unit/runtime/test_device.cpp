/**
 * @file test/feat/common/test_device.cpp
 * @author sailing-innocent
 * @date 2023/07/30
 * @brief the device test suite
*/
#include "ut/ut.hpp"
#include "test_device.h"

#include <luisa/runtime/device.h>
#include <luisa/runtime/context.h>

using namespace boost::ut;
using namespace boost::ut::literals;

namespace {

int test_wrapped_device(const char *cwd, const char *device_name) {
    return 0;
}
int test_create_device(const char *cwd, const char *device_name) {
    return 0;
}

}// namespace

int main(int argc, char *argv[]) {
    boost::ut::detail::cfg::parse_arg_with_fallback(argc, const_cast<const char **>(argv));

    if (argc > 1) {
        test_create_device(argv[0], argv[1]);
        test_wrapped_device(argv[0], argv[1]);
    }
}

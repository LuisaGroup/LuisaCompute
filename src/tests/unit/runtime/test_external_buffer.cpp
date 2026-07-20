/**
 * @file test/feat/common/test_external_buffer.cpp
 * @author sailing-innocent
 * @date 2023/11/02
 * @brief test import_external_buffer
*/

#include "ut/ut.hpp"
#include "test_device.h"

#include <luisa/runtime/device.h>
#include <luisa/runtime/context.h>
#include <luisa/runtime/stream.h>
#include <luisa/runtime/buffer.h>
#include <luisa/dsl/sugar.h>

using namespace luisa;
using namespace luisa::compute;
using namespace boost::ut;
using namespace boost::ut::literals;

namespace {

int test_external_buffer(Device &device) {
    constexpr uint n = 10u;
    Buffer<int> a = device.create_buffer<int>(n);
    Stream stream = device.create_stream();
    luisa::vector<int> data_init(n, 1);
    luisa::vector<int> data_result(n, 0);

    stream << a.copy_from(luisa::span{data_init});
    stream << synchronize();

    auto b = device.import_external_buffer<int>(a.native_handle(), n);

    stream << b.copy_to(luisa::span{data_result});
    stream << synchronize();

    for (uint idx = 0u; idx < n; idx++) {
        boost::ut::expect(static_cast<bool>(data_result[idx] == 1));
    }

    return 0;
}

}// namespace

int main(int argc, char *argv[]) {
    auto dc = luisa::test::create_device_from_ut(argc, argv);
    if (!dc) {
        return 0;
    }
    boost::ut::detail::cfg::parse_arg_with_fallback(argc, const_cast<const char **>(argv));

    auto &device = dc->device;
    test_external_buffer(device);
}

/**
 * @file test/feat/runtime/test_buffer.cpp
 * @author sailing-innocent
 * @date 2023/11/05
 * @brief test shared memory
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

int test_shared_mem(Device &device) {
    uint block_size = 32u;
    uint n = 1024u;
    Buffer<int> a = device.create_buffer<int>(n);

    Kernel1D test_kernel = [&](BufferVar<int> arr) noexcept {
        set_block_size(block_size);
        auto idx = dispatch_id().x;
        $if (idx > n) { return; };
        Shared<int> *s_data = new Shared<int>(block_size);
        auto thread_idx = thread_id().x;
        (*s_data)[thread_idx] = static_cast<$int>(thread_idx);
        sync_block();
        arr->write(idx, (*s_data)[thread_idx]);
    };
    auto test_shader = device.compile(test_kernel);
    auto stream = device.create_stream();
    stream << test_shader(a).dispatch(n);
    stream << synchronize();
    luisa::vector<int> data(n, 0);
    stream << a.copy_to(luisa::span{data});
    stream << synchronize();

    for (uint i = 0u; i < n; i++) {
        boost::ut::expect(static_cast<bool>(data[i] == i % block_size));
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
    test_shared_mem(device);
}

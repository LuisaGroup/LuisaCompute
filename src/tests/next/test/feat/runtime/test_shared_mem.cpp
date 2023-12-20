/**
 * @file test/feat/runtime/test_buffer.cpp
 * @author sailing-innocent
 * @date 2023/11/05
 * @brief test shared memory
*/

#include "common/config.h"

#include <luisa/runtime/device.h>
#include <luisa/runtime/context.h>
#include <luisa/runtime/stream.h>
#include <luisa/runtime/buffer.h>
#include <luisa/dsl/sugar.h>

using namespace luisa;
using namespace luisa::compute;

namespace luisa::test {

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
    stream << a.copy_to(data.data());
    stream << synchronize();

    for (uint i = 0u; i < n; i++) {
        CHECK_MESSAGE(data[i] == i % block_size, "failed when ", i);
    }
    return 0;
}

}// namespace luisa::test

TEST_SUITE("runtime") {
    LUISA_TEST_CASE_WITH_DEVICE("shared_memory", luisa::test::test_shared_mem(device) == 0);
}

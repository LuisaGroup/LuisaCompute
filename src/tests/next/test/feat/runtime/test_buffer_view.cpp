/**
 * @file test/feat/test_buffer_view.cpp
 * @author sailing-innocent
 * @date 2023/07/29
 * @brief the buffer view test case
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

template<typename T_FloatX>
int test_buffer_view(Device &device, size_t literal_size, size_t align_size = 4) {
    constexpr uint n = 10u;
    auto buffer = device.create_buffer<T_FloatX>(n);
    auto view = buffer.view();
    auto handle_view = BufferView<T_FloatX>{
        buffer.native_handle(),
        buffer.handle(),
        align_size, 0, n, n};
    Stream stream = device.create_stream();
    luisa::vector<float> data_init(n * align_size, 1.f);
    luisa::vector<float> data_result(n * align_size, 0.f);
    stream << buffer.copy_from(data_init.data());
    stream << synchronize();

    // dispatch
    stream << buffer.copy_to(data_result.data());
    stream << synchronize();
    // check init value
    for (auto i = 0; i < n; i++) {
        CHECK_MESSAGE(data_result[i] == 1.f, "failed when ", i);
    }

    Kernel1D selfadd_kernel = [&](BufferVar<T_FloatX> view) noexcept {
        set_block_size(64u);
        UInt index = dispatch_id().x;
        $if (index < n) {
            view->write(index, view->read(index) + 1.0f);
        };
    };

    auto selfadd = device.compile(selfadd_kernel);
    stream << selfadd(view).dispatch(n);
    stream << synchronize();

    // dispatch
    stream << buffer.copy_to(data_result.data());
    stream << synchronize();
    // check byffer value
    for (auto i = 0; i < n; i++) {
        CHECK_MESSAGE(data_result[i] == 2.f, "failed when ", i);
    }

    // handle view
    stream << selfadd(handle_view).dispatch(n);
    stream << synchronize();

    // dispatch
    stream << buffer.copy_to(data_result.data());
    stream << synchronize();
    // check byffer value
    for (auto i = 0; i < n; i++) {
        CHECK_MESSAGE(data_result[i] == 3.f, "failed when ", i);
    }

    return 0;
}
}// namespace luisa::test

TEST_SUITE("runtime") {
    LUISA_TEST_CASE_WITH_DEVICE("buffer_view", luisa::test::test_buffer_view<float4>(device, 4, 4) == 0);
}// TEST_SUITE("runtime")
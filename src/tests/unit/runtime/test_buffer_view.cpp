/**
 * @file test/feat/test_buffer_view.cpp
 * @author sailing-innocent
 * @date 2023/07/29
 * @brief the buffer view test case
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
    stream << buffer.copy_from(luisa::span{data_init});
    stream << synchronize();

    // dispatch
    stream << buffer.copy_to(luisa::span{data_result});
    stream << synchronize();
    // check init value
    for (auto i = 0; i < n; i++) {
        boost::ut::expect(static_cast<bool>(data_result[i] == 1.f));
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
    stream << buffer.copy_to(luisa::span{data_result});
    stream << synchronize();
    // check byffer value
    for (auto i = 0; i < n; i++) {
        boost::ut::expect(static_cast<bool>(data_result[i] == 2.f));
    }

    // handle view
    stream << selfadd(handle_view).dispatch(n);
    stream << synchronize();

    // dispatch
    stream << buffer.copy_to(luisa::span{data_result});
    stream << synchronize();
    // check byffer value
    for (auto i = 0; i < n; i++) {
        boost::ut::expect(static_cast<bool>(data_result[i] == 3.f));
    }

    return 0;
}
int main(int argc, char *argv[]) {
    auto dc = luisa::test::create_device_from_ut(argc, argv);
    if (!dc) {
        return 0;
    }
    boost::ut::detail::cfg::parse_arg_with_fallback(argc, const_cast<const char **>(argv));

    auto &device = dc->device;
    test_buffer_view<float4>(device, 4, 4);
}

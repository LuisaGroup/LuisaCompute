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

namespace {

constexpr auto element_count = 17u;
constexpr auto subview_offset = 3u;
constexpr auto subview_size = 7u;

[[nodiscard]] float4 initial_value(uint i) noexcept {
    auto x = static_cast<float>(i);
    return make_float4(x + 0.25f, x * 2.0f + 0.5f,
                       -x - 0.75f, x * x + 1.0f);
}

constexpr auto increment = make_float4(1.0f, -2.0f, 4.0f, 0.5f);

void check_equal(const float4 &actual, const float4 &expected) noexcept {
    for (auto c = 0u; c < 4u; c++) {
        expect(actual[c] == expected[c]);
    }
}

}// namespace

int main(int argc, char *argv[]) {
    auto dc = luisa::test::create_device_from_ut(argc, argv);
    if (!dc) {
        return 0;
    }
    boost::ut::detail::cfg::parse_arg_with_fallback(argc, const_cast<const char **>(argv));

    auto &device = dc->device;
    auto buffer = device.create_buffer<float4>(element_count);
    auto regular_view = buffer.view();
    auto reconstructed_view = BufferView<float4>{
        buffer.native_handle(), buffer.handle(), buffer.stride(),
        0u, element_count, element_count};
    auto subview = regular_view.subview(subview_offset, subview_size);

    expect(regular_view.stride() == buffer.stride());
    expect(reconstructed_view.stride() == buffer.stride());
    expect(subview.offset() == subview_offset);
    expect(subview.size() == subview_size);

    luisa::vector<float4> expected(element_count);
    for (auto i = 0u; i < element_count; i++) {
        expected[i] = initial_value(i);
    }
    auto result = expected;

    auto stream = device.create_stream();
    stream << buffer.copy_from(luisa::span{expected}) << synchronize();

    Kernel1D add_kernel = [](BufferVar<float4> values) noexcept {
        auto i = dispatch_id().x;
        values->write(i, values->read(i) + increment);
    };
    auto add = device.compile(add_kernel);

    // Exercise the ordinary view and validate every lane of every element.
    stream << add(regular_view).dispatch(element_count)
           << buffer.copy_to(luisa::span{result})
           << synchronize();
    for (auto i = 0u; i < element_count; i++) {
        expected[i] += increment;
        check_equal(result[i], expected[i]);
    }

    // Exercise a view reconstructed from the native handles using the actual
    // backend-reported element stride.
    stream << add(reconstructed_view).dispatch(element_count)
           << buffer.copy_to(luisa::span{result})
           << synchronize();
    for (auto i = 0u; i < element_count; i++) {
        expected[i] += increment;
        check_equal(result[i], expected[i]);
    }

    // A nonzero-offset subview catches implementations that accidentally bind
    // the original buffer base or ignore the view length.
    stream << add(subview).dispatch(subview_size)
           << buffer.copy_to(luisa::span{result})
           << synchronize();
    for (auto i = 0u; i < element_count; i++) {
        if (i >= subview_offset && i < subview_offset + subview_size) {
            expected[i] += increment;
        }
        check_equal(result[i], expected[i]);
    }
}

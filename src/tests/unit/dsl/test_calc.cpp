/**
 * @file test/feat/dsl/test_calc.cpp
 * @author sailing-innocent
 * @date 2023/11/04
 * @brief the calculation in dsl types
*/

#include "ut/ut.hpp"
#include "test_device.h"
#include <luisa/runtime/device.h>
#include <luisa/runtime/stream.h>
#include <luisa/runtime/buffer.h>
#include <luisa/dsl/sugar.h>

using namespace luisa;
using namespace luisa::compute;
using namespace boost::ut;
using namespace boost::ut::literals;

int test_calc(Device &device) {
    Stream stream = device.create_stream();
    Buffer<float2> vert_buf = device.create_buffer<float2>(10);
    Buffer<float2x2> mat_buf = device.create_buffer<float2x2>(10);
    Buffer<float2> out_buf = device.create_buffer<float2>(10);

    Kernel1D mat_vert_prod = [&](BufferVar<float2> vert, BufferVar<float2x2> mat, BufferVar<float2> out) {
        auto idx = dispatch_id().x;
        out.write(idx, mat.read(idx) * vert.read(idx));
    };

    // init
    luisa::vector<float2> vert(10);
    luisa::vector<float2x2> mat(10);
    for (auto i = 0u; i < 10u; i++) {
        vert[i] = make_float2(i, i + 1);
        float2 col_1 = make_float2(i, i + 1);
        float2 col_2 = make_float2(i + 2, i + 3);
        mat[i] = make_float2x2(col_1, col_2);
    }
    stream << vert_buf.copy_from(luisa::span{vert});
    stream << mat_buf.copy_from(luisa::span{mat});
    stream << synchronize();

    auto shader = device.compile(mat_vert_prod);
    stream << shader(vert_buf, mat_buf, out_buf).dispatch(10u);
    stream << synchronize();

    luisa::vector<float2> out(10);
    stream << out_buf.copy_to(luisa::span{out});
    stream << synchronize();

    for (auto i = 0u; i < 10u; i++) {
        boost::ut::expect(static_cast<bool>(out[i][0] == i * i + (i + 1) * (i + 2)));
        boost::ut::expect(static_cast<bool>(out[i][1] == i * (i + 1) + (i + 1) * (i + 3)));
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
    test_calc(device);
}

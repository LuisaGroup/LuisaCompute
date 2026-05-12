/**
 * @file test/feat/dsl/test_matrix.cpp
 * @author sailing-innocent
 * @date 2023/08/26
 * @brief the dsl matrix-relevant operations
*/

#include "ut/ut.hpp"
#include "test_device.h"

#include <cmath>
#include <luisa/runtime/device.h>
#include <luisa/runtime/context.h>
#include <luisa/runtime/stream.h>
#include <luisa/runtime/buffer.h>
#include <luisa/dsl/sugar.h>

using namespace luisa;
using namespace luisa::compute;
using namespace boost::ut;
using namespace boost::ut::literals;

int test_matrix2x2(Device &device) {
    auto m = make_float2x2(1.f, 2.f, 3.f, 4.f);
    // Matrix in LC is col-first order
    // 1 3
    // 2 4
    // M[i][j] means i-th col and j-th row
    boost::ut::expect(static_cast<bool>(std::abs((m[0][0]) - (1.f)) < 0.001f));
    boost::ut::expect(static_cast<bool>(std::abs((m[0][1]) - (2.f)) < 0.001f));
    boost::ut::expect(static_cast<bool>(std::abs((m[1][0]) - (3.f)) < 0.001f));
    boost::ut::expect(static_cast<bool>(std::abs((m[1][1]) - (4.f)) < 0.001f));
    // transpose
    auto mt = transpose(m);
    boost::ut::expect(static_cast<bool>(std::abs((mt[0][0]) - (1.f)) < 0.001f));
    boost::ut::expect(static_cast<bool>(std::abs((mt[0][1]) - (3.f)) < 0.001f));
    boost::ut::expect(static_cast<bool>(std::abs((mt[1][0]) - (2.f)) < 0.001f));
    boost::ut::expect(static_cast<bool>(std::abs((mt[1][1]) - (4.f)) < 0.001f));

    // Matrix-Vector Multiplication
    auto v = make_float2(1.f, 2.f);
    auto mv = m * v;
    // m * v
    // 1 3  x 1 = 7
    // 2 4    2   10
    boost::ut::expect(static_cast<bool>(mv[0] == 7.f));
    boost::ut::expect(static_cast<bool>(mv[1] == 10.f));

    // Matrix-Matrix Multiplication
    auto w = make_float2x2(
        make_float2(5.0f, 6.0f),
        make_float2(7.0f, 8.0f));

    // 1 3  x 5 7 = 23 31
    // 2 4    6 8   34 46
    auto mw = m * w;
    // m^T * w
    boost::ut::expect(static_cast<bool>(std::abs((mw[0][0]) - (23.0f)) < 0.001f));
    boost::ut::expect(static_cast<bool>(std::abs((mw[0][1]) - (34.0f)) < 0.001f));
    boost::ut::expect(static_cast<bool>(std::abs((mw[1][0]) - (31.0f)) < 0.001f));
    boost::ut::expect(static_cast<bool>(std::abs((mw[1][1]) - (46.0f)) < 0.001f));

    // calc inv
    // inv 1 3  = -2   1
    //     2 4  = 1.5 -0.5
    auto inv_m = inverse(m);
    boost::ut::expect(static_cast<bool>(std::abs((inv_m[0][0]) - (-2.0f)) < 0.001f));
    boost::ut::expect(static_cast<bool>(std::abs((inv_m[0][1]) - (+1.0f)) < 0.001f));
    boost::ut::expect(static_cast<bool>(std::abs((inv_m[1][0]) - (+1.5f)) < 0.001f));
    boost::ut::expect(static_cast<bool>(std::abs((inv_m[1][1]) - (-0.5f)) < 0.001f));

    // determinant
    auto det_m = determinant(m);
    boost::ut::expect(static_cast<bool>(det_m == -2.f));

    return 0;
}

int main(int argc, char *argv[]) {
    auto dc = luisa::test::create_device_from_ut(argc, argv);
    if (!dc) {
        return 0;
    }
    boost::ut::detail::cfg::parse_arg_with_fallback(argc, const_cast<const char **>(argv));

    auto &device = dc->device;
    test_matrix2x2(device);
}

/**
 * @file test/feat/dsl/test_matrix.cpp
 * @author sailing-innocent
 * @date 2023/08/26
 * @brief the dsl matrix-relevant operations
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

int test_matrix2x2(Device &device) {
    auto m = make_float2x2(1.f, 2.f, 3.f, 4.f);
    // Matrix in LC is col-first order
    // 1 3
    // 2 4
    // M[i][j] means i-th col and j-th row
    CHECK_MESSAGE(m[0][0] == doctest::Approx(1.f), "failed when (0,0)");
    CHECK_MESSAGE(m[0][1] == doctest::Approx(2.f), "failed when (0,1)");
    CHECK_MESSAGE(m[1][0] == doctest::Approx(3.f), "failed when (1,0)");
    CHECK_MESSAGE(m[1][1] == doctest::Approx(4.f), "failed when (1,1)");
    // transpose
    auto mt = transpose(m);
    CHECK_MESSAGE(mt[0][0] == doctest::Approx(1.f), "failed when (0,0)");
    CHECK_MESSAGE(mt[0][1] == doctest::Approx(3.f), "failed when (0,1)");
    CHECK_MESSAGE(mt[1][0] == doctest::Approx(2.f), "failed when (1,0)");
    CHECK_MESSAGE(mt[1][1] == doctest::Approx(4.f), "failed when (1,1)");

    // Matrix-Vector Multiplication
    auto v = make_float2(1.f, 2.f);
    auto mv = m * v;
    // m * v
    // 1 3  x 1 = 7
    // 2 4    2   10
    CHECK_MESSAGE(mv[0] == 7.f, "failed when (0)");
    CHECK_MESSAGE(mv[1] == 10.f, "failed when (1)");

    // Matrix-Matrix Multiplication
    auto w = make_float2x2(
        make_float2(5.0f, 6.0f),
        make_float2(7.0f, 8.0f));

    // 1 3  x 5 7 = 23 31
    // 2 4    6 8   34 46
    auto mw = m * w;
    // m^T * w
    CHECK(mw[0][0] == doctest::Approx(23.0f));
    CHECK(mw[0][1] == doctest::Approx(34.0f));
    CHECK(mw[1][0] == doctest::Approx(31.0f));
    CHECK(mw[1][1] == doctest::Approx(46.0f));

    // calc inv
    // inv 1 3  = -2   1
    //     2 4  = 1.5 -0.5
    auto inv_m = inverse(m);
    CHECK_MESSAGE(inv_m[0][0] == doctest::Approx(-2.0f), "failed when (0,0)");
    CHECK_MESSAGE(inv_m[0][1] == doctest::Approx(+1.0f), "failed when (0,1)");
    CHECK_MESSAGE(inv_m[1][0] == doctest::Approx(+1.5f), "failed when (1,0)");
    CHECK_MESSAGE(inv_m[1][1] == doctest::Approx(-0.5f), "failed when (1,1)");

    // determinant
    auto det_m = determinant(m);
    CHECK_MESSAGE(det_m == -2.f, "failed when determinant");

    return 0;
}
}// namespace luisa::test

TEST_SUITE("runtime") {
    LUISA_TEST_CASE_WITH_DEVICE("dsl_matrix_float2x2", luisa::test::test_matrix2x2(device) == 0);
}

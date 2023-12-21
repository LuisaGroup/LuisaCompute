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
    // transpose
    auto mt = transpose(m);
    CHECK_MESSAGE(mt[0][0] == 1.f, "failed when (0,0)");
    CHECK_MESSAGE(mt[0][1] == 3.f, "failed when (0,1)");
    CHECK_MESSAGE(mt[1][0] == 2.f, "failed when (1,0)");
    CHECK_MESSAGE(mt[1][1] == 4.f, "failed when (1,1)");
    // col first order
    auto v = make_float2(1.f, 2.f);
    auto mv = m * v;
    // m^T * v
    CHECK_MESSAGE(mv[0] == 7.f, "failed when (0)");
    CHECK_MESSAGE(mv[1] == 10.f, "failed when (1)");
    // calc inv
    auto inv_m = inverse(m);
    CHECK_MESSAGE(inv_m[0][0] == -2.f, "failed when (0,0)");
    CHECK_MESSAGE(inv_m[0][1] == 1.f, "failed when (0,1)");
    CHECK_MESSAGE(inv_m[1][0] == 1.5f, "failed when (1,0)");
    CHECK_MESSAGE(inv_m[1][1] == -0.5f, "failed when (1,1)");

    // determinant
    auto det_m = determinant(m);
    CHECK_MESSAGE(det_m == -2.f, "failed when determinant");

    auto w = make_float2x2(
        make_float2(1.0f, 2.0f),
        make_float2(3.0f, 4.0f));

    CHECK(w[0][0] == 1.0f);
    CHECK(w[0][1] == 2.0f);
    CHECK(w[1][0] == 3.0f);
    CHECK(w[1][1] == 4.0f);

    auto mw = m * w;
    // m^T * w
    CHECK(mw[0][0] == 7.0f);
    CHECK(mw[0][1] == 10.0f);
    CHECK(mw[1][0] == 15.0f);
    CHECK(mw[1][1] == 22.0f);

    return 0;
}
}// namespace luisa::test

TEST_SUITE("runtime") {
    LUISA_TEST_CASE_WITH_DEVICE("dsl_matrix_float2x2", luisa::test::test_matrix2x2(device) == 0);
}

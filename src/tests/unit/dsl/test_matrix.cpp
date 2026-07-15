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

// Struct with float4x4 for testing matrix-in-struct read/write
struct MatrixContainer {
    float3 aa;
    float4x4 m;
    float2 bb;
};
LUISA_STRUCT(MatrixContainer, aa, m, bb) {};

int test_matrix_kernel(Device &device) {
    Stream stream = device.create_stream();
    constexpr auto n = 16u;

    // Create buffers for matrices A, B, and result C
    Buffer<float4x4> buf_a = device.create_buffer<float4x4>(n);
    Buffer<float4x4> buf_b = device.create_buffer<float4x4>(n);
    Buffer<float4x4> buf_c = device.create_buffer<float4x4>(n);

    // Kernel: C[i] = A[i] * B[i]
    Kernel1D mat_mul = [&](BufferVar<float4x4> a, BufferVar<float4x4> b, BufferVar<float4x4> c) {
        auto idx = dispatch_id().x;
        c.write(idx, a.read(idx) * b.read(idx));
    };

    // Initialize host data
    luisa::vector<float4x4> host_a(n);
    luisa::vector<float4x4> host_b(n);
    luisa::vector<float4x4> host_c(n);

    for (auto i = 0u; i < n; i++) {
        // A (col-major): col0=(1,2,3,4), col1=(5,6,7,8), col2=(9,10,11,12), col3=(13,14,15,16)
        host_a[i] = make_float4x4(
            make_float4(1.0f, 2.0f, 3.0f, 4.0f),
            make_float4(5.0f, 6.0f, 7.0f, 8.0f),
            make_float4(9.0f, 10.0f, 11.0f, 12.0f),
            make_float4(13.0f, 14.0f, 15.0f, 16.0f));
        // B (col-major): col0=(17,18,19,20), col1=(21,22,23,24), col2=(25,26,27,28), col3=(29,30,31,32)
        host_b[i] = make_float4x4(
            make_float4(17.0f, 18.0f, 19.0f, 20.0f),
            make_float4(21.0f, 22.0f, 23.0f, 24.0f),
            make_float4(25.0f, 26.0f, 27.0f, 28.0f),
            make_float4(29.0f, 30.0f, 31.0f, 32.0f));
    }

    auto shader = device.compile(mat_mul);
    stream << buf_a.copy_from(luisa::span{host_a})
           << buf_b.copy_from(luisa::span{host_b})
           << shader(buf_a, buf_b, buf_c).dispatch(n)
           << buf_c.copy_to(luisa::span{host_c});
    stream << synchronize();

    // Expected: A*B (col-major, C[col][row])
    //   C[0] = (538, 612, 686, 760)
    //   C[1] = (650, 740, 830, 920)
    //   C[2] = (762, 868, 974, 1080)
    //   C[3] = (874, 996, 1118, 1240)
    for (auto i = 0u; i < n; i++) {
        boost::ut::expect(static_cast<bool>(std::abs(host_c[i][0][0] - 538.0f) < 0.001f));
        boost::ut::expect(static_cast<bool>(std::abs(host_c[i][0][1] - 612.0f) < 0.001f));
        boost::ut::expect(static_cast<bool>(std::abs(host_c[i][0][2] - 686.0f) < 0.001f));
        boost::ut::expect(static_cast<bool>(std::abs(host_c[i][0][3] - 760.0f) < 0.001f));
        boost::ut::expect(static_cast<bool>(std::abs(host_c[i][1][0] - 650.0f) < 0.001f));
        boost::ut::expect(static_cast<bool>(std::abs(host_c[i][1][1] - 740.0f) < 0.001f));
        boost::ut::expect(static_cast<bool>(std::abs(host_c[i][1][2] - 830.0f) < 0.001f));
        boost::ut::expect(static_cast<bool>(std::abs(host_c[i][1][3] - 920.0f) < 0.001f));
        boost::ut::expect(static_cast<bool>(std::abs(host_c[i][2][0] - 762.0f) < 0.001f));
        boost::ut::expect(static_cast<bool>(std::abs(host_c[i][2][1] - 868.0f) < 0.001f));
        boost::ut::expect(static_cast<bool>(std::abs(host_c[i][2][2] - 974.0f) < 0.001f));
        boost::ut::expect(static_cast<bool>(std::abs(host_c[i][2][3] - 1080.0f) < 0.001f));
        boost::ut::expect(static_cast<bool>(std::abs(host_c[i][3][0] - 874.0f) < 0.001f));
        boost::ut::expect(static_cast<bool>(std::abs(host_c[i][3][1] - 996.0f) < 0.001f));
        boost::ut::expect(static_cast<bool>(std::abs(host_c[i][3][2] - 1118.0f) < 0.001f));
        boost::ut::expect(static_cast<bool>(std::abs(host_c[i][3][3] - 1240.0f) < 0.001f));
    }

    return 0;
}

int test_matrix_struct(Device &device) {
    Stream stream = device.create_stream();
    constexpr auto n = 8u;

    // Create buffers of structs containing float4x4
    Buffer<MatrixContainer> buf_in = device.create_buffer<MatrixContainer>(n);
    Buffer<MatrixContainer> buf_out = device.create_buffer<MatrixContainer>(n);

    // Kernel: read struct, transpose matrix, write to output struct
    Kernel1D mat_struct = [&](BufferVar<MatrixContainer> in, BufferVar<MatrixContainer> out) {
        auto idx = dispatch_id().x;
        Var m = in.read(idx).m;
        Var t = transpose(m);
        Var<MatrixContainer> result;
        result.m = t;
        out.write(idx, result);
    };

    // Initialize host input data
    luisa::vector<MatrixContainer> host_in(n);
    luisa::vector<MatrixContainer> host_out(n);

    for (auto i = 0u; i < n; i++) {
        // col-major: col0=(1,2,3,4), col1=(5,6,7,8), col2=(9,10,11,12), col3=(13,14,15,16)
        host_in[i].m = make_float4x4(
            make_float4(1.0f + i, 2.0f + i, 3.0f + i, 4.0f + i),
            make_float4(5.0f + i, 6.0f + i, 7.0f + i, 8.0f + i),
            make_float4(9.0f + i, 10.0f + i, 11.0f + i, 12.0f + i),
            make_float4(13.0f + i, 14.0f + i, 15.0f + i, 16.0f + i));
    }

    auto shader = device.compile(mat_struct);
    stream << buf_in.copy_from(luisa::span{host_in})
           << shader(buf_in, buf_out).dispatch(n)
           << buf_out.copy_to(luisa::span{host_out});
    stream << synchronize();

    // Host-side: transpose each matrix and compare
    for (auto i = 0u; i < n; i++) {
        auto expected = transpose(host_in[i].m);
        // expected: rows become cols
        // col0=(1+i, 5+i, 9+i, 13+i), col1=(2+i, 6+i, 10+i, 14+i),
        // col2=(3+i, 7+i, 11+i, 15+i), col3=(4+i, 8+i, 12+i, 16+i)
        for (auto col = 0; col < 4; col++) {
            for (auto row = 0; row < 4; row++) {
                auto val = host_out[i].m[col][row];
                auto exp = expected[col][row];
                boost::ut::expect(static_cast<bool>(std::abs(val - exp) < 0.001f))
                    << "matrix struct transpose mismatch at i=" << i
                    << " col=" << col << " row=" << row;
            }
        }
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
    test_matrix2x2(device);
    test_matrix_kernel(device);
    test_matrix_struct(device);
}

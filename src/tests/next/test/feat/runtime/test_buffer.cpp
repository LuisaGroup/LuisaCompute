/**
 * @file test/feat/common/test_buffer.cpp
 * @author sailing-innocent
 * @date 2023/07/26
 * @brief the buffer test suite
*/

#include "common/config.h"

#include <luisa/runtime/device.h>
#include <luisa/runtime/context.h>
#include <luisa/runtime/stream.h>
#include <luisa/runtime/buffer.h>
#include <luisa/dsl/syntax.h>
#include <luisa/dsl/sugar.h>

using namespace luisa;
using namespace luisa::compute;

namespace luisa::test {

template<typename T_FloatX>
int test_floatx(Device &device, int literal_size = 1, int align_size = 4) {
    constexpr uint n = 1u;
    Buffer<T_FloatX> a = device.create_buffer<T_FloatX>(n);
    Buffer<T_FloatX> b = device.create_buffer<T_FloatX>(n);
    Buffer<T_FloatX> c = device.create_buffer<T_FloatX>(n);

    Kernel1D add_kernel = [&](BufferVar<T_FloatX> a, BufferVar<T_FloatX> b, BufferVar<T_FloatX> c) noexcept {
        set_block_size(64u);
        UInt index = dispatch_id().x;
        $if (index < n) {
            c->write(index, a->read(index) + b->read(index));
        };
    };
    Shader1D<Buffer<T_FloatX>, Buffer<T_FloatX>, Buffer<T_FloatX>> add = device.compile(add_kernel);

    // init a, b and c

    Stream stream = device.create_stream();
    luisa::vector<float> data_init(n * align_size, 1.f);
    luisa::vector<float> data_result(n * align_size, 0.f);
    stream << a.copy_from(data_init.data());
    stream << b.copy_from(data_init.data());
    stream << c.copy_from(data_result.data());

    stream << add(a, b, c).dispatch(n);
    stream << synchronize();
    stream << c.copy_to(data_result.data());
    stream << synchronize();

    for (uint idx = 0u; idx < n * align_size; idx++) {
        uint i = idx % align_size;
        if (align_size != literal_size && i == align_size - 1) {
            // undefined behavior, depends on backend implementation
        } else {
            CHECK_MESSAGE(data_result[idx] == 2.f, "failed when ", idx);
        }
    }
    return 0;
}

int test_float3x3_order(Device &device) {
    constexpr uint n = 1u;
    Buffer<float3x3> a = device.create_buffer<float3x3>(n);
    Buffer<float3x3> b = device.create_buffer<float3x3>(n);
    Buffer<float3x3> c = device.create_buffer<float3x3>(n);

    Kernel1D add_kernel = [&](BufferVar<float3x3> a, BufferVar<float3x3> b, BufferVar<float3x3> c) noexcept {
        set_block_size(64u);
        UInt index = dispatch_id().x;
        $if (index < n) {
            c->write(index, a->read(index) + b->read(index));
        };
    };
    Shader1D<Buffer<float3x3>, Buffer<float3x3>, Buffer<float3x3>> add = device.compile(add_kernel);

    // init a, b and c

    Stream stream = device.create_stream();
    luisa::vector<float> data_init(n * 12, 1.f);
    // align to col major
    // 1 2 2
    // 1 1 2
    // 1 1 1
    // 0 0 0
    // 3 * vec3 : 1 -> 1 -> 1 -> 0 -> 2 -> 1... -> 1 -> 0
    for (auto i = 0u; i < 3u; i++) {
        for (auto j = 0u; j < 4u; j++) {
            if (j == 3) {
                data_init[i * 4 + j] = 0.f;
            } else {
                if (i > j) {
                    data_init[i * 4 + j] = 2.f;
                } else {
                    data_init[i * 4 + j] = 1.f;
                }
            }
        }
    }
    luisa::vector<float> data_result(n * 12, 0.f);
    stream << a.copy_from(data_init.data());
    stream << b.copy_from(data_init.data());
    stream << c.copy_from(data_result.data());

    stream << add(a, b, c).dispatch(n);
    stream << synchronize();
    stream << c.copy_to(data_result.data());
    stream << synchronize();

    for (uint idx = 0u; idx < n * 12; idx++) {
        uint i = idx / 4;
        uint j = idx % 4;
        if (j == 3) {
            // undefined behaviour depends on backend implementation
        } else {
            if (i > j) {
                CHECK_MESSAGE(data_result[idx] == 4.f, "failed when ", i, ",", j);
            } else {
                CHECK_MESSAGE(data_result[idx] == 2.f, "failed when ", i, ",", j);
            }
        }
    }
    return 0;
}

int test_float3x3(Device &device) {
    constexpr uint n = 1u;
    Buffer<float3x3> a = device.create_buffer<float3x3>(n);
    Buffer<float3x3> b = device.create_buffer<float3x3>(n);
    Buffer<float3x3> c = device.create_buffer<float3x3>(n);

    Kernel1D add_kernel = [&](BufferVar<float3x3> a, BufferVar<float3x3> b, BufferVar<float3x3> c) noexcept {
        set_block_size(64u);
        UInt index = dispatch_id().x;
        $if (index < n) {
            c->write(index, a->read(index) + b->read(index));
        };
    };
    Shader1D<Buffer<float3x3>, Buffer<float3x3>, Buffer<float3x3>> add = device.compile(add_kernel);

    // init a, b and c

    Stream stream = device.create_stream();
    luisa::vector<float> data_init(n * 12, 1.f);
    luisa::vector<float> data_result(n * 12, 0.f);
    stream << a.copy_from(data_init.data());
    stream << b.copy_from(data_init.data());
    stream << c.copy_from(data_result.data());

    stream << add(a, b, c).dispatch(n);
    stream << synchronize();
    stream << c.copy_to(data_result.data());
    stream << synchronize();

    for (uint idx = 0u; idx < n * 12; idx++) {
        uint i = idx / 4;
        uint j = idx % 4;
        if (j == 3) {
            // undefined behaviour depends on backend implementation
        } else {
            CHECK_MESSAGE(data_result[idx] == 2.f, "failed when ", i, ",", j);
        }
    }
    return 0;
}

int test_float4x4(Device &device) {
    constexpr uint n = 1u;
    Buffer<float4x4> a = device.create_buffer<float4x4>(n);
    Buffer<float4x4> b = device.create_buffer<float4x4>(n);
    Buffer<float4x4> c = device.create_buffer<float4x4>(n);

    Kernel1D add_kernel = [&](BufferVar<
                                  float4x4>
                                  a,
                              BufferVar<float4x4> b, BufferVar<float4x4> c) noexcept {
        set_block_size(64u);
        UInt index = dispatch_id().x;
        $if (index < n) {
            c->write(index, a->read(index) + b->read(index));
        };
    };
    Shader1D<Buffer<float4x4>, Buffer<float4x4>, Buffer<float4x4>> add = device.compile(add_kernel);

    // init a, b and c

    Stream stream = device.create_stream();
    luisa::vector<float4x4> data_init(n, make_float4x4(1.f));
    luisa::vector<float> data_result(n * 16, 0.f);
    stream << a.copy_from(data_init.data());
    stream << b.copy_from(data_init.data());
    stream << c.copy_from(data_result.data());

    stream << add(a, b, c).dispatch(n);
    stream << synchronize();
    stream << c.copy_to(data_result.data());
    stream << synchronize();

    for (auto idx = 0u; idx < n * 16; idx++) {
        auto i = idx % 4;
        auto j = idx / 4 % 4;
        if (i == j) {
            CHECK_MESSAGE(data_result[idx] == 2.f, "failed when ", i, ",", j);
        } else {
            CHECK_MESSAGE(data_result[idx] == 0.f, "failed when ", i, ",", j);
        }
    }
    return 0;
}

}// namespace luisa::test

TEST_SUITE("runtime") {
    LUISA_TEST_CASE_WITH_DEVICE("buffer::float3x3", luisa::test::test_float3x3(device) == 0);
    LUISA_TEST_CASE_WITH_DEVICE("buffer::float3x3_order", luisa::test::test_float3x3_order(device) == 0);
    LUISA_TEST_CASE_WITH_DEVICE("buffer::float4x4", luisa::test::test_float4x4(device) == 0);
    LUISA_TEST_CASE_WITH_DEVICE("buffer::float4", luisa::test::test_floatx<float4>(device, 4, 4) == 0);
    LUISA_TEST_CASE_WITH_DEVICE("buffer::float3", luisa::test::test_floatx<float3>(device, 3, 4) == 0);
    LUISA_TEST_CASE_WITH_DEVICE("buffer::float2", luisa::test::test_floatx<float2>(device, 2, 2) == 0);
}
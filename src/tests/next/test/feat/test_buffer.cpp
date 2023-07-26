/**
 * @file test/feat/test_buffer.cpp
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

int test_float3x3(Device &device) {
    constexpr uint n = 1u;
    Buffer<float3x3> a = device.create_buffer<float3x3>(n);
    Buffer<float3x3> b = device.create_buffer<float3x3>(n);
    Buffer<float3x3> c = device.create_buffer<float3x3>(n);

    Kernel1D add_kernel = [&](BufferVar<float3x3> a, BufferVar<float3x3> b, BufferVar<float3x3> c) noexcept {
        set_block_size(64u);
        UInt index = dispatch_id().x;
        $if(index < n){
            c->write(index, a->read(index) + b->read(index));
        };
    };
    Shader1D<Buffer<float3x3>, Buffer<float3x3>, Buffer<float3x3>> add = device.compile(add_kernel);

    // init a, b and c

    Stream stream = device.create_stream();
    luisa::vector<float> data_init(n * 9, 1.f);
    luisa::vector<float> data_result(n * 9, 0.f);
    stream << a.copy_from(data_init.data());
    stream << b.copy_from(data_init.data());
    stream << c.copy_from(data_result.data());
 
    stream << add(a, b, c).dispatch(n);
    stream << synchronize();
    stream << c.copy_to(data_result.data());
    stream << synchronize();

    for (auto idx = 0u; idx < n * 9; idx++) {
        CHECK_MESSAGE(data_result[idx] == data_init[idx] + data_init[idx], "failed when ", idx);
    }
    return 0;
}

int test_float4x4(Device &device) {
    constexpr uint n = 1u;
    Buffer<float4x4> a = device.create_buffer<float4x4>(n);
    Buffer<float4x4> b = device.create_buffer<float4x4>(n);
    Buffer<float4x4> c = device.create_buffer<float4x4>(n);

    Kernel1D add_kernel = [&](BufferVar<
        float4x4> a, BufferVar<float4x4> b, BufferVar<float4x4> c) 
    noexcept {
        set_block_size(64u);
        UInt index = dispatch_id().x;
        $if(index < n){
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


} // namespace luisa::test



TEST_SUITE("feat::buffer") {
    TEST_CASE("buffer::float3x3") {
        Context context{luisa::test::argv()[0]};
       
        for (auto i = 0; i < luisa::test::supported_backends_count(); i++) {
            luisa::string device_name = luisa::test::supported_backends()[i];
            SUBCASE(device_name.c_str()) {
                Device device = context.create_device(device_name.c_str());
                REQUIRE(luisa::test::test_float3x3(device) == 0);
            }
        }
    }

    TEST_CASE("buffer::float4x4") {
        Context context{luisa::test::argv()[0]};
       
        for (auto i = 0; i < luisa::test::supported_backends_count(); i++) {
            luisa::string device_name = luisa::test::supported_backends()[i];
            SUBCASE(device_name.c_str()) {
                Device device = context.create_device(device_name.c_str());
                REQUIRE(luisa::test::test_float4x4(device) == 0);
            }
        }
    }
}
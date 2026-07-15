/**
 * @file test/feat/runtime/test_buffer.cpp
 * @author sailing-innocent
 * @date 2023/07/26
 * @brief the buffer test suite
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
    auto add = device.compile(add_kernel);

    // init a, b and c

    Stream stream = device.create_stream();
    luisa::vector<float> data_init(n * align_size, 1.f);
    luisa::vector<float> data_result(n * align_size, 0.f);
    stream << a.copy_from(luisa::span{data_init});
    stream << b.copy_from(luisa::span{data_init});
    stream << c.copy_from(luisa::span{data_result});

    stream << add(a, b, c).dispatch(n);
    stream << synchronize();
    stream << c.copy_to(luisa::span{data_result});
    stream << synchronize();

    for (uint idx = 0u; idx < n * align_size; idx++) {
        uint i = idx % align_size;
        if (align_size != literal_size && i == align_size - 1) {
            // undefined behavior, depends on backend implementation
        } else {
            boost::ut::expect(static_cast<bool>(data_result[idx] == 2.f));
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
    auto add = device.compile(add_kernel);

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
    stream << a.copy_from(luisa::span{data_init});
    stream << b.copy_from(luisa::span{data_init});
    stream << c.copy_from(luisa::span{data_result});

    stream << add(a, b, c).dispatch(n);
    stream << synchronize();
    stream << c.copy_to(luisa::span{data_result});
    stream << synchronize();

    for (uint idx = 0u; idx < n * 12; idx++) {
        uint i = idx / 4;
        uint j = idx % 4;
        if (j == 3) {
            // undefined behaviour depends on backend implementation
        } else {
            if (i > j) {
                boost::ut::expect(static_cast<bool>(data_result[idx] == 4.f));
            } else {
                boost::ut::expect(static_cast<bool>(data_result[idx] == 2.f));
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
    auto add = device.compile(add_kernel);

    // init a, b and c

    Stream stream = device.create_stream();
    luisa::vector<float> data_init(n * 12, 1.f);
    luisa::vector<float> data_result(n * 12, 0.f);
    stream << a.copy_from(luisa::span{data_init});
    stream << b.copy_from(luisa::span{data_init});
    stream << c.copy_from(luisa::span{data_result});

    stream << add(a, b, c).dispatch(n);
    stream << synchronize();
    stream << c.copy_to(luisa::span{data_result});
    stream << synchronize();

    for (uint idx = 0u; idx < n * 12; idx++) {
        uint i = idx / 4;
        uint j = idx % 4;
        if (j == 3) {
            // undefined behaviour depends on backend implementation
        } else {
            boost::ut::expect(static_cast<bool>(data_result[idx] == 2.f));
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
    auto add = device.compile(add_kernel);

    // init a, b and c

    Stream stream = device.create_stream();
    luisa::vector<float4x4> data_init(n, make_float4x4(1.f));
    luisa::vector<float> data_result(n * 16, 0.f);
    stream << a.copy_from(luisa::span{data_init});
    stream << b.copy_from(luisa::span{data_init});
    stream << c.copy_from(luisa::span{data_result});

    stream << add(a, b, c).dispatch(n);
    stream << synchronize();
    stream << c.copy_to(luisa::span{data_result});
    stream << synchronize();

    for (auto idx = 0u; idx < n * 16; idx++) {
        auto i = idx % 4;
        auto j = idx / 4 % 4;
        if (i == j) {
            boost::ut::expect(static_cast<bool>(data_result[idx] == 2.f));
        } else {
            boost::ut::expect(static_cast<bool>(data_result[idx] == 0.f));
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
    test_float3x3(device);
    test_float3x3_order(device);
    test_float4x4(device);
}

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

template<typename T>
void check_floatx_equal(const T &actual, const T &expected) noexcept {
    if constexpr (is_vector_v<T>) {
        for (auto i = 0u; i < vector_dimension_v<T>; i++) {
            boost::ut::expect(actual[i] == expected[i]);
        }
    } else {
        boost::ut::expect(actual == expected);
    }
}

template<typename T_FloatX, typename LhsFactory, typename RhsFactory>
int test_floatx(Device &device, LhsFactory &&lhs_factory, RhsFactory &&rhs_factory) {
    constexpr uint n = 11u;
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

    Stream stream = device.create_stream();
    luisa::vector<T_FloatX> lhs(n);
    luisa::vector<T_FloatX> rhs(n);
    luisa::vector<T_FloatX> expected(n);
    luisa::vector<T_FloatX> result(n, T_FloatX{});
    for (auto i = 0u; i < n; i++) {
        lhs[i] = lhs_factory(i);
        rhs[i] = rhs_factory(i);
        expected[i] = lhs[i] + rhs[i];
    }
    stream << a.copy_from(luisa::span{lhs});
    stream << b.copy_from(luisa::span{rhs});
    stream << c.copy_from(luisa::span{result});

    stream << add(a, b, c).dispatch(n);
    stream << synchronize();
    stream << c.copy_to(luisa::span{result});
    stream << synchronize();

    for (auto i = 0u; i < n; i++) {
        check_floatx_equal(result[i], expected[i]);
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
    test_floatx<float>(
        device,
        [](auto i) noexcept { return static_cast<float>(i) + 0.25f; },
        [](auto i) noexcept { return static_cast<float>(i * 3u) - 1.5f; });
    test_floatx<float2>(
        device,
        [](auto i) noexcept {
            auto x = static_cast<float>(i);
            return make_float2(x + 0.25f, -x - 0.5f);
        },
        [](auto i) noexcept {
            auto x = static_cast<float>(i);
            return make_float2(x * 2.0f + 1.0f, x * x + 0.75f);
        });
    test_floatx<float3>(
        device,
        [](auto i) noexcept {
            auto x = static_cast<float>(i);
            return make_float3(x + 0.25f, -x - 0.5f, x * x + 1.0f);
        },
        [](auto i) noexcept {
            auto x = static_cast<float>(i);
            return make_float3(x * 2.0f + 1.0f, x + 3.0f, -x * 0.5f);
        });
    test_floatx<float4>(
        device,
        [](auto i) noexcept {
            auto x = static_cast<float>(i);
            return make_float4(x + 0.25f, -x - 0.5f, x * x + 1.0f, x * 4.0f);
        },
        [](auto i) noexcept {
            auto x = static_cast<float>(i);
            return make_float4(x * 2.0f + 1.0f, x + 3.0f, -x * 0.5f, 0.125f - x);
        });
    test_float3x3(device);
    test_float3x3_order(device);
    test_float4x4(device);
}

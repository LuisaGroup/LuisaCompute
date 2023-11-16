/**
 * @file test/feat/common/test_autodiff.cpp
 * @author sailing-innocent
 * @date 2023/08/03
 * @brief the autodiff test suite
*/

#include "common/config.h"

#include <luisa/runtime/context.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/buffer.h>
#include <luisa/runtime/stream.h>
#include <luisa/dsl/sugar.h>
#include <luisa/core/logging.h>

#include <vector>

using namespace luisa;
using namespace luisa::compute;

namespace luisa::test {

int simple_autodiff(Device &device) {
    constexpr uint n = 1024u;
    auto x_buffer = device.create_buffer<float>(n);
    auto y_buffer = device.create_buffer<float2>(n);
    auto dx_buffer = device.create_buffer<float>(n);
    auto dy_buffer = device.create_buffer<float2>(n);
    auto stream = device.create_stream(StreamTag::GRAPHICS);

    // init x and y buffer data

    std::vector<float> x(n);
    std::vector<float2> y(n);
    for (auto i = 0u; i < n; i++) {
        auto v = static_cast<float>(i);
        x[i] = v;
        y[i] = make_float2(v);
    }

    auto dx = std::vector<float>(n);
    auto dy = std::vector<float2>(n);

    stream << x_buffer.copy_from(x.data())
           << synchronize()
           << y_buffer.copy_from(y.data())
           << synchronize();

    // object function: f = x * sin(y)
    static constexpr auto f = [](auto x, auto y) noexcept { return x * sin(y); };

    // autodiff kernel
    Kernel1D kernel = [](BufferFloat x_buffer, BufferFloat2 y_buffer,
                         BufferFloat x_grad_buffer, BufferFloat2 y_grad_buffer) noexcept {
        auto i = dispatch_x();
        auto x = x_buffer.read(i);
        auto y = y_buffer.read(i);

        Callable callable = [](Float x, Float2 y) noexcept {
            auto x_grad = def(0.f);
            auto y_grad = def(make_float2(0.f));

            $autodiff {
                requires_grad(x, y);
                auto z = f(x, y);
                backward(z);
                x_grad = grad(x);
                y_grad = grad(y);
            };
            return make_float3(x_grad, y_grad);
        };

        auto grad = callable(x, y);
        x_grad_buffer.write(i, grad.x);
        y_grad_buffer.write(i, grad.yz());
    };

    auto kernel_shader = device.compile(kernel);
    stream << kernel_shader(x_buffer, y_buffer, dx_buffer, dy_buffer).dispatch(n);
    stream << synchronize();

    // extract dx and dy buffer to cpu
    stream << dx_buffer.copy_to(dx.data())
           << dy_buffer.copy_to(dy.data())
           << synchronize();

    // forward difference
    luisa::vector<float> fd_x(n);
    luisa::vector<float2> fd_y(n);

    auto eps = 1e-4f;

    for (auto i = 0; i < n; i++) {
        auto z = f(x[i], y[i]);
        auto dz_dx = (f(x[i] + eps, y[i]) - z) / eps;
        auto dz_dy0 = (f(x[i], make_float2(y[i].x + eps, y[i].y)) - z) / eps;
        auto dz_dy1 = (f(x[i], make_float2(y[i].x, y[i].y + eps)) - z) / eps;
        fd_x[i] = dz_dx.x + dz_dx.y;
        fd_y[i] = make_float2(dz_dy0.x + dz_dy0.y, dz_dy1.x + dz_dy1.y);
    }

    // theoretical difference
    std::vector<double> td_x(n);
    std::vector<double> td_y_1(n);
    std::vector<double> td_y_2(n);
    for (auto i = 0; i < n; i++) {
        auto x_ = static_cast<double>(x[i]);
        auto y1 = static_cast<double>(y[i].x);
        auto y2 = static_cast<double>(y[i].y);
        td_x[i] = sin(y1) + sin(y2);
        td_y_1[i] = x_ * cos(y1);
        td_y_2[i] = x_ * cos(y2);
    }

    for (auto i = 0u; i < n; i++) {
        auto err_max_x = abs(td_x[i]) * 1e-2f;
        err_max_x = err_max_x > 1e-3f ? err_max_x : 1e-3f;
        // auto err_max_y = (abs(td_y[i]) + 1e-2f) * 1e-2f;
        // CHECK(abs(dx[i] - fd_x[i]) < err_max_x);
        // CHECK(abs(td_x[i] - fd_x[i]) < err_max_x);
        CHECK(abs(dx[i] - td_x[i]) < err_max_x);
        // CHECK(abs(dy[i].x - fd_y[i].x) < err_max_y.x);
        // CHECK(abs(dy[i].y - fd_y[i].y) < err_max_y.y);
        // CHECK(abs(td_y[i].x - fd_y[i].x) < err_max_y.x);
        // CHECK(abs(td_y[i].y - fd_y[i].y) < err_max_y.y);
        // CHECK(abs(dy[i].x - td_y[i].x) < err_max_y.x);
        // CHECK(abs(dy[i].y - td_y[i].y) < err_max_y.y);
    }

    return 0;
}

}// namespace luisa::test

TEST_SUITE("ir") {
    LUISA_TEST_CASE_WITH_DEVICE("autodiff::simple", luisa::test::simple_autodiff(device) == 0);
}
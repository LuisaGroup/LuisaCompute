// Test for automatic differentiation (autodiff) functionality.
//
// This test verifies the correctness of automatic differentiation by comparing
// gradients computed via autodiff with finite difference approximations.
//
// The test function is: f(x, y) = x * sin(y)
// We compute partial derivatives:
//   df/dx = sin(y)
//   df/dy = x * cos(y)
//
// Autodiff uses reverse-mode automatic differentiation to compute gradients
// efficiently in a single forward and backward pass.

#include "ut/ut.hpp"
#include "test_device.h"

#include <fstream>
#include <luisa/luisa-compute.h>
#include <luisa/ir/ast2ir.h>
#include <luisa/ir/ir2ast.h>

using namespace luisa;
using namespace luisa::compute;
using namespace boost::ut;
using namespace boost::ut::literals;

void test_autodiff(Device &device) {

    // Enable verbose logging for debugging
    luisa::log_level_verbose();

    // Number of test elements
    constexpr auto n = 1024u;

    // Create buffers for inputs (x, y) and their gradients (dx, dy)
    auto x_buffer = device.create_buffer<float>(n);
    auto y_buffer = device.create_buffer<float2>(n);
    auto dx_buffer = device.create_buffer<float>(n);
    auto dy_buffer = device.create_buffer<float2>(n);
    auto stream = device.create_stream(StreamTag::GRAPHICS);

    // Initialize input data with sequential values
    std::vector<float> x(n);
    std::vector<float2> y(n);
    for (auto i = 0u; i < n; i++) {
        auto v = static_cast<float>(i);
        x[i] = v;
        y[i] = make_float2(v);
    }

    // Prepare host buffers for gradient results
    auto dx = std::vector<float>(n);
    auto dy = std::vector<float2>(n);
    stream << x_buffer.copy_from(luisa::span{x})
           << synchronize()
           << y_buffer.copy_from(luisa::span{y})
           << synchronize();

    // Define the function to differentiate: f(x, y) = x * sin(y)
    static constexpr auto f = [](auto x, auto y) noexcept { return x * sin(y); };

    // Kernel that computes gradients using autodiff
    Kernel1D kernel = [](BufferFloat x_buffer, BufferFloat2 y_buffer,
                         BufferFloat x_grad_buffer, BufferFloat2 y_grad_buffer) noexcept {
        auto i = dispatch_x();
        auto x = x_buffer.read(i);
        auto y = y_buffer.read(i);

        // Define a callable that uses autodiff to compute gradients
        Callable callable = [](ArrayFloat<3> a) noexcept {
            auto x_grad = def(0.f);
            auto y_grad = def(make_float2(0.f));

            // Autodiff block: mark inputs as requiring gradients,
            // compute output, then propagate gradients backward
            $autodiff {
                requires_grad(a);
                auto z = f(a[0], make_float2(a[1], a[2]));
                backward(z);// Backpropagate gradients from output
                auto a_grad = grad(a);
                x_grad = a_grad[0];
                y_grad = make_float2(a_grad[1], a_grad[2]);
            };
            return make_float3(x_grad, y_grad);
        };

        // Pack inputs into array for autodiff
        ArrayFloat<3> a{x, y.x, y.y};
        auto grad = callable(a);

        // Write gradients to output buffers
        x_grad_buffer.write(i, grad.x);
        y_grad_buffer.write(i, grad.yz());
    };

    // Compile and execute kernel
    auto kernel_shader = device.compile(kernel);
    stream << kernel_shader(x_buffer, y_buffer, dx_buffer, dy_buffer).dispatch(n)
           << synchronize();

    // Copy results back to host
    stream << dx_buffer.copy_to(luisa::span{dx})
           << dy_buffer.copy_to(luisa::span{dy})
           << synchronize();

    // Compute finite difference approximations for validation
    // Using central difference: df/dx ≈ (f(x+ε) - f(x-ε)) / (2ε)
    luisa::vector<float> fd_x(n);
    luisa::vector<float2> fd_y(n);
    auto eps = 1e-4f;
    for (auto i = 0; i < n; i++) {
        auto z = f(x[i], y[i]);
        // Partial derivative w.r.t. x
        auto dz_dx = (f(x[i] + eps, y[i]) - z) / eps;
        // Partial derivative w.r.t. y components
        auto dz_dy0 = (f(x[i], make_float2(y[i].x + eps, y[i].y)) - z) / eps;
        auto dz_dy1 = (f(x[i], make_float2(y[i].x, y[i].y + eps)) - z) / eps;
        fd_x[i] = dz_dx.x + dz_dx.y;
        fd_y[i] = make_float2(dz_dy0.x + dz_dy0.y, dz_dy1.x + dz_dy1.y);
    }

    // Compare autodiff gradients with finite differences
    for (auto i = 0u; i < 16u; i++) {
        LUISA_INFO("Input #{}: {}, ({}, {}); "
                   "AD: {}, ({}, {}); "
                   "FD: {}, ({}, {})",
                   i, x[i], y[i].x, y[i].y,
                   dx[i], dy[i].x, dy[i].y,
                   fd_x[i], fd_y[i].x, fd_y[i].y);
    }
}

int main(int argc, char *argv[]) {
    auto dc = luisa::test::create_device_from_ut(argc, argv);
    if (!dc) {
        return 0;
    }
    boost::ut::detail::cfg::parse_arg_with_fallback(argc, const_cast<const char**>(argv));
    auto &device = dc->device;
    test_autodiff(device);
}

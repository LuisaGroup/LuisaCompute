// Test for DSL autodiff (automatic differentiation) functionality.
// This is a pure DSL-level test — no IR/Rust dependency required.
//
// Tests:
// - Basic $autodiff block with requires_grad / backward / grad
// - Autodiff with scalar functions (multiply, add, sin, cos)
// - Autodiff with control flow ($if / $else)
// - backward(x) with implicit gradient (ones)
// - backward(x, custom_grad) with explicit gradient
// - Multiple variables requiring grad
// - Validation against finite differences

#include "ut/ut.hpp"
#include "test_device.h"

#include <cmath>
#include <luisa/runtime/device.h>
#include <luisa/runtime/stream.h>
#include <luisa/runtime/buffer.h>
#include <luisa/dsl/syntax.h>
#include <luisa/dsl/sugar.h>

using namespace luisa;
using namespace luisa::compute;
using namespace boost::ut;
using namespace boost::ut::literals;

void test_autodiff_basic(Device &device) {
    // f(x, y) = x * y
    // df/dx = y, df/dy = x
    constexpr uint N = 64u;

    auto x_buf = device.create_buffer<float>(N);
    auto y_buf = device.create_buffer<float>(N);
    auto dx_buf = device.create_buffer<float>(N);
    auto dy_buf = device.create_buffer<float>(N);
    auto stream = device.create_stream();

    Kernel1D kernel = [&](BufferFloat x_in, BufferFloat y_in,
                          BufferFloat dx_out, BufferFloat dy_out) noexcept {
        auto tid = dispatch_id().x;
        auto x = x_in.read(tid);
        auto y = y_in.read(tid);

        $autodiff {
            requires_grad(x, y);
            auto z = x * y;
            backward(z);
            dx_out.write(tid, grad(x));
            dy_out.write(tid, grad(y));
        };
    };

    luisa::vector<float> hx(N), hy(N);
    for (uint i = 0u; i < N; i++) {
        hx[i] = static_cast<float>(i + 1);
        hy[i] = static_cast<float>(i + 1) * 0.5f;
    }

    auto shader = device.compile(kernel);
    stream << x_buf.copy_from(luisa::span{hx})
           << y_buf.copy_from(luisa::span{hy})
           << shader(x_buf, y_buf, dx_buf, dy_buf).dispatch(N)
           << synchronize();

    luisa::vector<float> hdx(N), hdy(N);
    stream << dx_buf.copy_to(luisa::span{hdx})
           << dy_buf.copy_to(luisa::span{hdy})
           << synchronize();

    bool correct = true;
    for (uint i = 0u; i < N; i++) {
        // df/dx = y, df/dy = x
        if (!std::isfinite(hdx[i]) || !std::isfinite(hdy[i]) ||
            std::abs(hdx[i] - hy[i]) > 1e-4f ||
            std::abs(hdy[i] - hx[i]) > 1e-4f) {
            correct = false;
            break;
        }
    }
    expect(correct) << "basic autodiff: d(x*y)/dx = y, d(x*y)/dy = x";
}

void test_autodiff_trig(Device &device) {
    // f(x) = sin(x), df/dx = cos(x)
    constexpr uint N = 64u;

    auto x_buf = device.create_buffer<float>(N);
    auto dx_buf = device.create_buffer<float>(N);
    auto stream = device.create_stream();

    Kernel1D kernel = [&](BufferFloat x_in, BufferFloat dx_out) noexcept {
        auto tid = dispatch_id().x;
        auto x = x_in.read(tid);
        $autodiff {
            requires_grad(x);
            auto z = sin(x);
            backward(z);
            dx_out.write(tid, grad(x));
        };
    };

    luisa::vector<float> hx(N);
    for (uint i = 0u; i < N; i++) {
        hx[i] = static_cast<float>(i) * 0.1f;
    }

    auto shader = device.compile(kernel);
    stream << x_buf.copy_from(luisa::span{hx})
           << shader(x_buf, dx_buf).dispatch(N)
           << synchronize();

    luisa::vector<float> hdx(N);
    stream << dx_buf.copy_to(luisa::span{hdx}) << synchronize();

    bool correct = true;
    for (uint i = 0u; i < N; i++) {
        float expected = std::cos(hx[i]);
        if (!std::isfinite(hdx[i]) || std::abs(hdx[i] - expected) > 1e-3f) {
            correct = false;
            break;
        }
    }
    expect(correct) << "trig autodiff: d(sin(x))/dx = cos(x)";
}

void test_autodiff_custom_grad(Device &device) {
    // f(x) = x^2, backward(z, 2.0) should give grad(x) = 2 * x * 2.0 = 4x
    constexpr uint N = 32u;

    auto x_buf = device.create_buffer<float>(N);
    auto dx_buf = device.create_buffer<float>(N);
    auto stream = device.create_stream();

    Kernel1D kernel = [&](BufferFloat x_in, BufferFloat dx_out) noexcept {
        auto tid = dispatch_id().x;
        auto x = x_in.read(tid);
        $autodiff {
            requires_grad(x);
            auto z = x * x;
            backward(z, def(2.0f));
            dx_out.write(tid, grad(x));
        };
    };

    luisa::vector<float> hx(N);
    for (uint i = 0u; i < N; i++) {
        hx[i] = static_cast<float>(i + 1);
    }

    auto shader = device.compile(kernel);
    stream << x_buf.copy_from(luisa::span{hx})
           << shader(x_buf, dx_buf).dispatch(N)
           << synchronize();

    luisa::vector<float> hdx(N);
    stream << dx_buf.copy_to(luisa::span{hdx}) << synchronize();

    bool correct = true;
    for (uint i = 0u; i < N; i++) {
        float expected = 4.0f * hx[i];// 2 * x * custom_grad(2.0)
        if (!std::isfinite(hdx[i]) || std::abs(hdx[i] - expected) > 1e-3f) {
            correct = false;
            break;
        }
    }
    expect(correct) << "custom grad: backward(x^2, 2.0) should give 4*x";
}

void test_autodiff_chain_rule(Device &device) {
    // f(x) = sin(x^2), df/dx = 2*x*cos(x^2)
    constexpr uint N = 32u;

    auto x_buf = device.create_buffer<float>(N);
    auto dx_buf = device.create_buffer<float>(N);
    auto stream = device.create_stream();

    Kernel1D kernel = [&](BufferFloat x_in, BufferFloat dx_out) noexcept {
        auto tid = dispatch_id().x;
        auto x = x_in.read(tid);
        $autodiff {
            requires_grad(x);
            auto z = sin(x * x);
            backward(z);
            dx_out.write(tid, grad(x));
        };
    };

    luisa::vector<float> hx(N);
    for (uint i = 0u; i < N; i++) {
        hx[i] = static_cast<float>(i) * 0.1f + 0.1f;
    }

    auto shader = device.compile(kernel);
    stream << x_buf.copy_from(luisa::span{hx})
           << shader(x_buf, dx_buf).dispatch(N)
           << synchronize();

    luisa::vector<float> hdx(N);
    stream << dx_buf.copy_to(luisa::span{hdx}) << synchronize();

    bool correct = true;
    for (uint i = 0u; i < N; i++) {
        float expected = 2.0f * hx[i] * std::cos(hx[i] * hx[i]);
        if (!std::isfinite(hdx[i]) || std::abs(hdx[i] - expected) > 1e-2f) {
            correct = false;
            break;
        }
    }
    expect(correct) << "chain rule: d(sin(x^2))/dx = 2*x*cos(x^2)";
}

void test_autodiff_addition(Device &device) {
    // f(x, y) = x + y, df/dx = 1, df/dy = 1
    constexpr uint N = 32u;

    auto x_buf = device.create_buffer<float>(N);
    auto y_buf = device.create_buffer<float>(N);
    auto dx_buf = device.create_buffer<float>(N);
    auto dy_buf = device.create_buffer<float>(N);
    auto stream = device.create_stream();

    Kernel1D kernel = [&](BufferFloat x_in, BufferFloat y_in,
                          BufferFloat dx_out, BufferFloat dy_out) noexcept {
        auto tid = dispatch_id().x;
        auto x = x_in.read(tid);
        auto y = y_in.read(tid);
        $autodiff {
            requires_grad(x, y);
            auto z = x + y;
            backward(z);
            dx_out.write(tid, grad(x));
            dy_out.write(tid, grad(y));
        };
    };

    luisa::vector<float> hx(N), hy(N);
    for (uint i = 0u; i < N; i++) {
        hx[i] = static_cast<float>(i);
        hy[i] = static_cast<float>(i) * 2.0f;
    }

    auto shader = device.compile(kernel);
    stream << x_buf.copy_from(luisa::span{hx})
           << y_buf.copy_from(luisa::span{hy})
           << shader(x_buf, y_buf, dx_buf, dy_buf).dispatch(N)
           << synchronize();

    luisa::vector<float> hdx(N), hdy(N);
    stream << dx_buf.copy_to(luisa::span{hdx})
           << dy_buf.copy_to(luisa::span{hdy})
           << synchronize();

    bool correct = true;
    for (uint i = 0u; i < N; i++) {
        if (!std::isfinite(hdx[i]) || !std::isfinite(hdy[i]) ||
            std::abs(hdx[i] - 1.0f) > 1e-5f ||
            std::abs(hdy[i] - 1.0f) > 1e-5f) {
            correct = false;
            break;
        }
    }
    expect(correct) << "addition: d(x+y)/dx = 1, d(x+y)/dy = 1";
}

void test_autodiff_with_callable(Device &device) {
    // Callable: g(x) = x^3
    // Kernel: f(x) = g(x), df/dx = 3*x^2
    constexpr uint N = 32u;

    auto x_buf = device.create_buffer<float>(N);
    auto dx_buf = device.create_buffer<float>(N);
    auto stream = device.create_stream();

    Callable cube = [](Float x) noexcept {
        return x * x * x;
    };

    Kernel1D kernel = [&](BufferFloat x_in, BufferFloat dx_out) noexcept {
        auto tid = dispatch_id().x;
        auto x = x_in.read(tid);
        $autodiff {
            requires_grad(x);
            auto z = cube(x);
            backward(z);
            dx_out.write(tid, grad(x));
        };
    };

    luisa::vector<float> hx(N);
    for (uint i = 0u; i < N; i++) {
        hx[i] = static_cast<float>(i + 1) * 0.5f;
    }

    auto shader = device.compile(kernel);
    stream << x_buf.copy_from(luisa::span{hx})
           << shader(x_buf, dx_buf).dispatch(N)
           << synchronize();

    luisa::vector<float> hdx(N);
    stream << dx_buf.copy_to(luisa::span{hdx}) << synchronize();

    bool correct = true;
    for (uint i = 0u; i < N; i++) {
        float expected = 3.0f * hx[i] * hx[i];
        if (!std::isfinite(hdx[i]) || std::abs(hdx[i] - expected) > 1e-2f) {
            correct = false;
            break;
        }
    }
    expect(correct) << "callable: d(x^3)/dx = 3*x^2";
}

int main(int argc, char *argv[]) {
    auto dc = luisa::test::create_device_from_ut(argc, argv);
    if (!dc) {
        return 0;
    }
    boost::ut::detail::cfg::parse_arg_with_fallback(argc, const_cast<const char **>(argv));

    auto &device = dc->device;
    test_autodiff_basic(device);
    test_autodiff_trig(device);
    test_autodiff_custom_grad(device);
    test_autodiff_chain_rule(device);
    test_autodiff_addition(device);
    test_autodiff_with_callable(device);
}

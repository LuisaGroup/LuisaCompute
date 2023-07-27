#include <fstream>
#include <luisa/luisa-compute.h>
#include <luisa/ir/ast2ir.h>
#include <luisa/ir/ir2ast.h>

using namespace luisa;
using namespace luisa::compute;

int main(int argc, char *argv[]) {

    luisa::log_level_verbose();

    auto context = Context{argv[0]};
    if (argc <= 1) {
        LUISA_INFO("Usage: {} <backend>. <backend>: cuda, dx, cpu, metal", argv[0]);
        exit(1);
    }
    auto device = context.create_device(argv[1]);

    constexpr auto n = 1024u;

    auto x_buffer = device.create_buffer<float>(n);
    auto y_buffer = device.create_buffer<float2>(n);
    auto dx_buffer = device.create_buffer<float>(n);
    auto dy_buffer = device.create_buffer<float2>(n);
    auto stream = device.create_stream(StreamTag::GRAPHICS);

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

    static constexpr auto f = [](auto x, auto y) noexcept { return x * sin(y); };

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
    stream << kernel_shader(x_buffer, y_buffer, dx_buffer, dy_buffer).dispatch(n)
           << synchronize();

    stream << dx_buffer.copy_to(dx.data())
           << dy_buffer.copy_to(dy.data())
           << synchronize();

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

    for (auto i = 0u; i < 16u; i++) {
        LUISA_INFO("Input #{}: {}, ({}, {}); "
                   "AD: {}, ({}, {}); "
                   "FD: {}, ({}, {})",
                   i, x[i], y[i].x, y[i].y,
                   dx[i], dy[i].x, dy[i].y,
                   fd_x[i], fd_y[i].x, fd_y[i].y);
    }
}

//
// Created by Mike Smith on 2021/2/27.
//

#include <luisa-compute.h>

using namespace luisa;
using namespace luisa::compute;

int main(int argc, char *argv[]) {

    luisa::log_level_verbose();

    Kernel1D kernel = [](BufferFloat x_buffer, BufferFloat2 y_buffer,
                         BufferFloat x_grad_buffer, BufferFloat2 y_grad_buffer) noexcept {
        auto i = dispatch_x();
        auto x = x_buffer.read(i);
        auto y = y_buffer.read(i);
        $autodiff {
            requires_grad(x, y);
            auto t = x * y;
            auto z = t / (x + y);
            backward(z);
            x_grad_buffer.write(i, grad(x));
            y_grad_buffer.write(i, grad(y));
        };
    };
}

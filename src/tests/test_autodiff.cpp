//
// Created by Mike Smith on 2021/2/27.
//

#include <fstream>
#include <luisa/luisa-compute.h>
#include <luisa/ir/ast2ir.h>
#include <luisa/ir/ir2ast.h>

using namespace luisa;
using namespace luisa::compute;
#ifndef LUISA_ENABLE_IR
#error "LUISA_ENABLE_IR must be defined."
#endif
int main(int argc, char *argv[]) {
    luisa::log_level_verbose();

    auto context = Context{argv[0]};
    if (argc <= 1) {
        LUISA_INFO("Usage: {} <backend>. <backend>: cuda, dx, ispc, metal", argv[0]);
        exit(1);
    }
    auto device = context.create_device(argv[1]);

    auto x_buffer = device.create_buffer<float>(1024u);
    auto y_buffer = device.create_buffer<float2>(1024u);
    auto dx_buffer = device.create_buffer<float>(1024u);
    auto dy_buffer = device.create_buffer<float2>(1024u);
    auto stream = device.create_stream(StreamTag::GRAPHICS);

    auto dx = std::vector<float>(1024);
    auto dy = std::vector<float2>(1024);
    for (auto i = 0u; i < 1024u; i++) {
        auto f = static_cast<float>(i);
        dx[i] = f;
        dy[i] = make_float2(f);
    }
    stream << x_buffer.copy_from(dx.data())
           << synchronize()
           << y_buffer.copy_from(dy.data())
           << synchronize();

    Kernel1D kernel = [](BufferFloat x_buffer, BufferFloat2 y_buffer,
                         BufferFloat x_grad_buffer, BufferFloat2 y_grad_buffer) noexcept {
        auto i = dispatch_x();
        auto x = x_buffer.read(i);
        auto y = y_buffer.read(i);
        $autodiff {
            requires_grad(x, y);
            auto z = x * sin(y);
            backward(z);
            x_grad_buffer.write(i, grad(x));
            y_grad_buffer.write(i, grad(y));
        };
    };
    auto ir = AST2IR::build_kernel(kernel.function()->function());
    LUISA_INFO("AST2IR done.");

    auto dump = ir::luisa_compute_ir_dump_human_readable(&ir->get()->module);
    LUISA_INFO("IR dump done.");
    std::ofstream out{"autodiff_ir_dump.txt"};
    out << luisa::string_view{reinterpret_cast<const char *>(dump.ptr), dump.len};

    auto reconstructed_ast = IR2AST::build(ir->get());
    LUISA_INFO("IR2AST done.");

    auto reconstructed_ir = AST2IR::build_kernel(reconstructed_ast->function());
    LUISA_INFO("2nd AST2IR done.");

    auto dump2 = ir::luisa_compute_ir_dump_human_readable(&reconstructed_ir->get()->module);
    LUISA_INFO("2nd IR dump done.");
    std::ofstream out2{"autodiff_ir_dump_reconstructed.txt"};
    out2 << luisa::string_view{reinterpret_cast<const char *>(dump2.ptr), dump2.len};

    auto kernel_shader = device.compile<1, Buffer<float>, Buffer<float2>, Buffer<float>, Buffer<float2>>(ir->get());
    stream << kernel_shader(x_buffer, y_buffer, dx_buffer, dy_buffer).dispatch(1024u)
           << synchronize();

    stream << dx_buffer.copy_to(dx.data())
           << dy_buffer.copy_to(dy.data())
           << synchronize();

    for (auto i = 0u; i < 16; i++) {
        printf("%f\n", dx[i]);
    }
    for (auto i = 0u; i < 16; i++) {
        printf("(%f, %f)\n", dy[i].x, dy[i].y);
    }
}

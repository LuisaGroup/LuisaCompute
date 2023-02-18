//
// Created by Mike Smith on 2021/2/27.
//

#include <iostream>

#include <luisa-compute.h>

using namespace luisa;
using namespace luisa::compute;
using luisa::compute::detail::FunctionBuilder;

int main(int argc, char *argv[]) {

    luisa::log_level_verbose();
    Context context{argv[0]};
    if (argc <= 1) {
        LUISA_INFO("Usage: {} <backend>. <backend>: cuda, dx, ispc, metal", argv[0]);
        exit(1);
    }
    auto device = context.create_device(argv[1]);
    auto stream = device.create_stream();

    // auto _builder = FunctionBuilder::define_kernel([] {
    //     // auto a = def(2);
    // });

    // auto shader = device.impl()->create_shader(_builder->function(), {});

    std::cout << Type::of<Buffer<int>>()->description() << "\n";

    auto buf = device.create_buffer<int>(100);

    Kernel1D k1 = [&] {
        buf->write(1, 42);
    };
    auto s = device.compile(k1);
    stream << s().dispatch(1u);
    stream << synchronize();
}

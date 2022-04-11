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
    auto device = context.create_device("ispc");
    auto stream = device.create_stream();


    // auto _builder = FunctionBuilder::define_kernel([] {
    //     // auto a = def(2);
    // });

    // auto shader = device.impl()->create_shader(_builder->function(), {});

    std::cout << Type::of<Buffer<int>>()->description() << "\n";

    auto buf = device.create_buffer<int>(100);

    Kernel1D k1 = [&] {
        buf.write(1,42);
    };
    auto s = device.compile(k1);

    auto command = s().dispatch(1u);
    stream << command;
    stream << synchronize();

}

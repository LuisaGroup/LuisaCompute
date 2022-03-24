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


    auto _builder = FunctionBuilder::define_kernel([] {
        // auto a = def(2);
    });

    auto shader = device.impl()->create_shader(_builder->function(), {});

    std::cout << Type::of<float3>()->description() << "\n";

    LiteralExpr(Type::of<int>(), 3);
    LiteralExpr(Type::of<bool>(), false);
    LiteralExpr(Type::of<float3>(), make_float3(0.4f));

    // Kernel1D k1 = [] {
    //     auto a = def(2);
    // };
    // auto s = device.compile(k1);

    // ===============================================================

    // auto f = make_shared<FunctionBuilder>(Function::Tag::KERNEL);
    // {
    //     f->push(f.get());
    //     f->push_scope(&f->_body);

    //     // auto a = def(2);

    //     f->pop_scope(&f->_body);
    //     f->pop(f.get());
    // }
    // auto _builder = luisa::const_pointer_cast<const FunctionBuilder>(f);
    // auto s = Shader<1>(device, _builder, std::string_view{});


    // auto command = s().dispatch(1024u);
    // stream << command;
    // stream << synchronize();

}

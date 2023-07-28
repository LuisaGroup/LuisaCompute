#include <luisa/luisa-compute.h>

using namespace luisa;
using namespace luisa::compute;
using luisa::compute::detail::FunctionBuilder;

int main(int argc, char *argv[]) {

    luisa::log_level_verbose();
    Context context{argv[0]};
    if (argc <= 1) {
        LUISA_INFO("Usage: {} <backend>. <backend>: cuda, dx, cpu, metal", argv[0]);
        exit(1);
    }
    Device device = context.create_device(argv[1]);
    Stream stream = device.create_stream();

    // auto _builder = FunctionBuilder::define_kernel([] {
    //     // auto a = def(2);
    // });

    // auto shader = device.impl()->create_shader(_builder->function(), {});

    LUISA_INFO("Buffer<int> description: {}", Type::of<Buffer<int>>()->description());

    Buffer<int> buf = device.create_buffer<int>(100);

    auto h = 1.0_h;
    auto f = sin(h);

    LUISA_INFO("h = {}, f = {}, f * f = {}, f + h = {}", h, f, f * f, f + h);

    Kernel1D k1 = [&] {
        buf->write(1, 42);
    };
    Shader1D<> s = device.compile(k1);
    stream << s().dispatch(1u);
    stream << synchronize();
}


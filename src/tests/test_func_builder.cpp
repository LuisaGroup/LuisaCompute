#include <cstdio>
#include <luisa/dsl/sugar.h>
#include <luisa/runtime/context.h>
#include <luisa/runtime/stream.h>
#include <luisa/runtime/image.h>
#include <luisa/runtime/shader.h>
#include <luisa/dsl/syntax.h>

struct Fuck{
    int a;
    uint32_t b;
    std::array<float, 4> c;
};

int main(int argc, char *argv[]) {
    using namespace luisa::compute;
    Context context{argv[0]};
    DeviceConfig config{ .headless = true };
    Device device = context.create_device("dx", &config);
    LUISA_INFO("{}", Type::of<Fuck>()->description());
    auto builder_ptr = luisa::compute::detail::FunctionBuilder::define_kernel([&]() {
        auto cur = luisa::compute::detail::FunctionBuilder::current();
        auto buffer = cur->buffer(Type::of<Buffer<float>>());
        cur->set_block_size(uint3(256, 1, 1));
        cur->mark_variable_usage(buffer->variable().uid(), Usage::WRITE);
        auto idx = cur->literal(Type::of<uint>(), uint(0));
        auto value = cur->literal(Type::of<float>(), float(2.));
        cur->call(CallOp::BUFFER_WRITE, {buffer, idx, value});
    });
    device.impl()->create_shader(
        ShaderOption{
            .name = "test.bin",
            .compile_only = true},
        Function{builder_ptr.get()});
    return 0;
}

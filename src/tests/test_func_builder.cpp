#include <cstdio>
#include <luisa/dsl/sugar.h>
#include <luisa/runtime/context.h>
#include <luisa/runtime/stream.h>
#include <luisa/runtime/image.h>
#include <luisa/runtime/shader.h>
#include <luisa/dsl/syntax.h>
#include <luisa/clangcxx/compiler.h>

struct NVIDIA
{
    int n;
};

struct Fuck{
    int a;
    NVIDIA b;
    std::array<float, 4> c;
};

int main(int argc, char *argv[]) {
    using namespace luisa::compute;
    Context context{argv[0]};
    DeviceConfig config{ .headless = true };
    Device device = context.create_device("dx", &config);
    LUISA_INFO("{}", Type::of<Fuck>()->description());
    {
        auto builder_ptr = luisa::compute::detail::FunctionBuilder::define_kernel([&]() {
            auto cur = luisa::compute::detail::FunctionBuilder::current();
            auto buffer = cur->buffer(Type::of<Buffer<NVIDIA>>());
            cur->set_block_size(uint3(256, 1, 1));
            cur->mark_variable_usage(buffer->variable().uid(), Usage::WRITE);
            auto idx = cur->literal(Type::of<uint>(), uint(0));
            // NVIDIA nv;
            // nv.n = literal;
            auto nv = cur->local(Type::of<NVIDIA>());
            auto nv_member = cur->member(Type::of<int>(), nv, 0);
            cur->assign(nv_member, cur->literal(Type::of<int>(), int(2)));
            cur->call(CallOp::BUFFER_WRITE, {buffer, idx, nv});
        });
        device.impl()->create_shader(
            ShaderOption{
                .name = "test.bin",
                .compile_only = true},
            Function{builder_ptr.get()});
    }
    {
        auto compiler = luisa::clangcxx::Compiler(
            ShaderOption{
                .name = "test.bin",
                .compile_only = true},
            Function{});
        compiler.create_shader(device);
    }
    return 0;
}
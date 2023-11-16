#include <cstdio>
#include <luisa/dsl/sugar.h>
#include <luisa/runtime/context.h>
#include <luisa/runtime/stream.h>
#include <luisa/runtime/image.h>
#include <luisa/runtime/shader.h>
#include <luisa/dsl/syntax.h>
#include <luisa/clangcxx/compiler.h>

int main(int argc, char *argv[]) {
    using namespace luisa::compute;
    Context context{argv[0]};
    DeviceConfig config{ .headless = true };
    Device device = context.create_device("dx", &config);
    {
        auto compiler = luisa::clangcxx::Compiler(
            ShaderOption{
                .compile_only = true,
                .name = "test.bin"
                });
        compiler.create_shader(device);
    }
    return 0;
}
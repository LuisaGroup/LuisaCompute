#include <luisa/core/logging.h>
#include <luisa/dsl/sugar.h>
#include <luisa/runtime/context.h>
#include <luisa/runtime/stream.h>
#include <luisa/runtime/buffer.h>
#include <luisa/runtime/image.h>
#include <luisa/runtime/shader.h>
#include <luisa/runtime/rtx/accel.h>
#include <luisa/clangcxx/compiler.h>
using namespace luisa;
using namespace luisa::compute;
int main(int argc, char *argv[]) {
    Context context{argv[0]};
    // DeviceConfig config{.headless = true};
    Device device = context.create_device("dx", /*&config*/ nullptr);
    Stream stream = device.create_stream();
    {
        auto compiler = luisa::clangcxx::Compiler(
            ShaderOption{
                .compile_only = true,
                .name = "test.bin"});
        compiler.create_shader(context, device);
    }
    auto shader = device.load_shader<1, Buffer<float>>("test.bin");
    auto buffer = device.create_buffer<float>(32);
    vector<float> result;
    result.resize(buffer.size());
    stream << shader(buffer).dispatch(buffer.size())
           << buffer.copy_to(result.data())
           << synchronize();
    string result_str;
    for (auto &i : result) {
        result_str += std::to_string(i) + " ";
    }
    log_info("Result: {}", result_str);
    return 0;
}
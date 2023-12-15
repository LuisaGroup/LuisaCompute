#include <luisa/core/logging.h>
#include <luisa/dsl/sugar.h>
#include <luisa/runtime/context.h>
#include <luisa/runtime/stream.h>
#include <luisa/runtime/buffer.h>
#include <luisa/runtime/image.h>
#include <luisa/runtime/shader.h>
#include <luisa/runtime/rtx/accel.h>
#include <luisa/clangcxx/compiler.h>
#include <luisa/gui/window.h>
#include <luisa/runtime/swapchain.h>
using namespace luisa;
using namespace luisa::compute;
int main(int argc, char *argv[]) {
    Context context{argv[0]};
    // DeviceConfig config{.headless = true};
    static constexpr uint width = 1920;
    static constexpr uint height = 1080;
    Device device = context.create_device("dx", /*&config*/ nullptr);
    Stream stream = device.create_stream(StreamTag::GRAPHICS);
    {
        auto compiler = luisa::clangcxx::Compiler(
            ShaderOption{
                .compile_only = true,
                .name = "test.bin"});
        compiler.create_shader(context, device);
    }
    auto shader = device.load_shader<2, Buffer<float4>>("test.bin");
    auto buffer = device.create_buffer<float4>(width * height);
    Window window{"test func", uint2(width, height)};
    Swapchain swap_chain{device.create_swapchain(
        window.native_handle(),
        stream,
        uint2(width, height),
        false, false, 2)};
    auto ldr_image = device.create_image<float>(swap_chain.backend_storage(), uint2(width, height));

    Callable linear_to_srgb = [&](Var<float3> x) noexcept {
        return saturate(select(1.055f * pow(x, 1.0f / 2.4f) - 0.055f,
                               12.92f * x,
                               x <= 0.00031308f));
    };
    Kernel2D hdr2ldr_kernel = [&](BufferVar<float4> hdr_image, ImageFloat ldr_image, Float scale, Bool is_hdr) noexcept {
        UInt2 coord = dispatch_id().xy();
        Float4 hdr = hdr_image.read(coord.x + coord.y * dispatch_size().x);
        Float3 ldr = hdr.xyz() * scale;
        $if (!is_hdr) {
            ldr = linear_to_srgb(ldr);
        };
        ldr_image.write(coord, make_float4(ldr, 1.0f));
    };
    auto hdr2ldr_shader = device.compile(hdr2ldr_kernel);
    auto blk_size = shader.block_size();
    LUISA_INFO("{}, {}, {}", blk_size.x, blk_size.y, blk_size.z);
    while (!window.should_close()) {
        window.poll_events();
        stream << shader(buffer).dispatch(width, height)
               << hdr2ldr_shader(buffer, ldr_image, 1.0f, false).dispatch(width, height)
               << swap_chain.present(ldr_image);
    }
    return 0;
}
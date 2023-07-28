#include <luisa/luisa-compute.h>
#include <luisa/dsl/sugar.h>
#include <luisa/gui/window.h>

using namespace luisa;
using namespace luisa::compute;

// contributed by @swifly in issue #67
int main(int argc, char *argv[]) {

    Context context{argv[0]};
    if (argc <= 1) {
        LUISA_INFO("Usage: {} <backend>. <backend>: cuda, dx, cpu, metal", argv[0]);
        exit(1);
    }
    Device device = context.create_device(argv[1]);

    constexpr uint2 resolution = make_uint2(1280, 720);

    Stream stream = device.create_stream(StreamTag::GRAPHICS);
    Image<float> device_image1 = device.create_image<float>(PixelStorage::BYTE4, resolution);
    BindlessArray bdls = device.create_bindless_array();
    Buffer<float4> buffer = device.create_buffer<float4>(4);
    std::vector<float4> a{4};
    a[0] = {1, 0, 0, 1};
    a[1] = {0, 1, 0, 1};
    a[2] = {0, 0, 1, 1};
    a[3] = {1, 1, 1, 1};
    stream << buffer.copy_from(a.data()) << synchronize();
    bdls.emplace_on_update(0, buffer);
    stream << bdls.update() << synchronize();

    Kernel2D kernel = [&](Float time) {
        Var coord = dispatch_id().xy();
        UInt i2 = ((coord.x + cast<uint>(time)) / 16 % 4);
        auto vertex_array = bdls->buffer<float4>(0);
        Float4 p = vertex_array.read(i2);
        device_image1->write(coord, make_float4(p));
    };
    Shader2D<float> s = device.compile(kernel);

    Window window{"Display", resolution};

    Swapchain swapchain = device.create_swapchain(
        window.native_handle(), stream, resolution, false);
    Clock clk;
    while (!window.should_close()) {
        stream << s(static_cast<float>(clk.toc() * .05f))
                      .dispatch(1280, 720)
               << swapchain.present(device_image1);
        window.poll_events();
    }
}


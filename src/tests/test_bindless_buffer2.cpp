#include <luisa-compute.h>
#include <dsl/sugar.h>
#include <gui/window.h>

using namespace luisa;
using namespace luisa::compute;

// contributed by @swifly in issue #67
int main(int argc, char* argv[])
{
    Context context{ argv[0] };
    auto device = context.create_device("dx");
    auto stream = device.create_stream();
    auto device_image1 = device.create_image<float>(PixelStorage::BYTE4, 1280, 720, 0u);
    auto bdls = device.create_bindless_array();
    auto buffer = device.create_buffer<float4>(4);
    std::vector<std::array<uint8_t, 4u>> d{ 1280 * 720 };
    std::vector<float4> a{ 4 };
    a[0] = { 1, 0, 0,1 };
    a[1] = { 0, 1, 0,1 };
    a[2] = { 0, 0, 1,1 };
    a[3] = { 1, 1, 1,1 };
    stream << buffer.copy_from(a.data())<<synchronize();
    bdls.emplace(0, buffer);
    stream << bdls.update() << synchronize();

    Kernel2D kernel = [&]()
    {
        Var coord = dispatch_id().xy();

        UInt i2 = ((coord.x / 15) % 4);
        auto vertexArray = bdls.buffer<float4>(0);
        Float4 p = vertexArray.read(i2);
        device_image1.write(coord, make_float4(p));
    };
    auto s = device.compile(kernel);
    stream
        << s().dispatch(1280, 720)
        << device_image1.copy_to(d.data())
        << synchronize();

    Window window{ "Display", uint2{ 1280,720 } };
    window.set_key_callback([&](int key, int action) noexcept
                            {
                                if (action == GLFW_PRESS && key == GLFW_KEY_ESCAPE)
                                {
                                    window.set_should_close();
                                }
                            });
    auto frame_count = 0u;
    window.run([&]
               {
                   window.set_background(d.data(), uint2{ 1280,720 });
               });
}

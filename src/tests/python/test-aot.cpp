#include <stb/stb_image_write.h>
#include <luisa/core/logging.h>
#include <luisa/runtime/context.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/stream.h>
#include <luisa/runtime/image.h>
#include <luisa/runtime/shader.h>
using namespace luisa;
using namespace luisa::compute;

int main(int argc, char *argv[]) {

    log_level_info();
    Context context{argv[0]};
    if (argc <= 1) {
        LUISA_INFO("Usage: {} <backend>. <backend>: cuda, dx, cpu, metal", argv[0]);
        exit(1);
    }
    Device device = context.create_device(argv[1]);
    Stream stream = device.create_stream();
    auto shader = device.load_shader<2, Image<float>>("out_kernel.bytes");
    static constexpr auto width = 512u;
    static constexpr auto height = 512u;
    vector<uint8_t> pixels(width * height * 4u);
    auto ldr_image = device.create_image<float>(PixelStorage::BYTE4, width, height);
    stream
        << shader(ldr_image).dispatch(width, height)
        << ldr_image.copy_to(pixels.data())
        << synchronize();
    stbi_write_png("test_aot.png", width, height, 4, pixels.data(), 0);
    
    return 0;
}

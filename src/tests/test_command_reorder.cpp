//
// Created by ChenXin on 2021/12/9.
//

#include <vector>

#include <runtime/context.h>
#include <runtime/stream.h>
#include <runtime/device.h>
#include <runtime/image.h>

using namespace luisa;
using namespace luisa::compute;

int main(int argc, char *argv[]) {
    Context context{argv[0]};

#if defined(LUISA_BACKEND_CUDA_ENABLED)
    auto device = context.create_device("cuda");
#elif defined(LUISA_BACKEND_METAL_ENABLED)
    auto device = context.create_device("metal");
#else
    auto device = context.create_device("ispc");
#endif
    auto stream = device.create_stream();

    auto width = 1920u, height = 1080u;
    auto device_image = device.create_image<float>(PixelStorage::BYTE4, width, height);
    auto device_image1 = device.create_image<float>(PixelStorage::BYTE4, width, height);
    std::vector<std::array<uint8_t, 4u>> host_image(width * height);
    std::vector<std::array<uint8_t, 4u>> host_image1(width * height);

    stream << device_image.copy_to(host_image.data())
           << device_image1.copy_to(host_image1.data())
           << device_image.copy_from(host_image1.data())
           << synchronize();

    return 0;
}
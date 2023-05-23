//
// Created by Mike on 5/24/2023.
//

#include <luisa-compute.h>

using namespace luisa;
using namespace luisa::compute;

// credit: https://github.com/nvpro-samples/vk_mini_samples/tree/main/samples/texture_3d/shaders (Apache License 2.0)
int main(int argc, char *argv[]) {

    Context context{argv[0]};

    if (argc <= 1) {
        LUISA_INFO("Usage: {} <backend>. <backend>: cuda, dx, ispc, metal", argv[0]);
        exit(1);
    }
    Device device = context.create_device(argv[1]);




}

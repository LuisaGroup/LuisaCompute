#include <fstream>

#include <runtime/context.h>
#include <runtime/device.h>
#include <runtime/stream.h>
#include <runtime/buffer.h>
#include <runtime/image.h>
#include <core/logging.h>
#include <runtime/event.h>
#include <backends/ext/dstorage_ext.hpp>
#include <stb/stb_image_write.h>
#include <core/clock.h>

using namespace luisa;
using namespace luisa::compute;

int main(int argc, char *argv[]) {

    Context context{argv[0]};

    if (argc <= 1) {
        LUISA_INFO("Usage: {} <backend>. <backend>: cuda, dx, ispc, metal", argv[0]);
        exit(1);
    }
    auto device = context.create_device(argv[1]);
    auto dstorage_ext = device.extension<DStorageExt>();

    auto dstorage_stream = dstorage_ext->create_stream();
    auto dstorage_file = dstorage_ext->open_file("test_dstorage_texture_compressed.gdeflate");
    auto image = device.create_image<float>(PixelStorage::BYTE4, make_uint2(512));
    dstorage_stream << dstorage_file.decompress_to(image) << synchronize();

    luisa::vector<uint8_t> pixels(image.size_bytes());
    auto compute_stream = device.create_stream();
    compute_stream << image.copy_to(pixels.data()) << synchronize();

    stbi_write_png("test_dstorage_decompression.png", 512, 512, 4, pixels.data(), 0);
}

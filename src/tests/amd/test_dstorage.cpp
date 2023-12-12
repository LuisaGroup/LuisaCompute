#include <luisa/runtime/context.h>
#include <luisa/runtime/stream.h>
#include <luisa/runtime/sparse_image.h>
#include <luisa/runtime/sparse_command_list.h>
#include <luisa/runtime/shader.h>
#include <luisa/runtime/swapchain.h>
#include <luisa/vstl/common.h>
#include <luisa/gui/window.h>
#include <luisa/backends/ext/dstorage_ext.hpp>

using namespace luisa;
using namespace luisa::compute;

int main(int argc, char *argv[]) {
    static constexpr uint32_t width = 1024;
    static constexpr uint32_t height = 1024;
    Context context{argv[0]};
    Device device = context.create_device("dx");
    Stream stream = device.create_stream(StreamTag::GRAPHICS);

    luisa::vector<uint8_t> pixels(width * height * 4);
    for (size_t x = 0; x < width; ++x)
        for (size_t y = 0; y < height; ++y) {
            size_t pixel_pos = x + y * width;
            float2 uv = make_float2(x, y) / make_float2(width, height);
            pixels[pixel_pos * 4] = static_cast<uint8_t>(uv.x * 255);
            pixels[pixel_pos * 4 + 1] = static_cast<uint8_t>(uv.y * 255);
            pixels[pixel_pos * 4 + 2] = 127;
            pixels[pixel_pos * 4 + 3] = 255;
        }
    auto f = fopen("pixels.bytes", "wb");
    fwrite(pixels.data(), pixels.size_bytes(), 1, f);
    fclose(f);

    auto img = device.create_sparse_image<float>(PixelStorage::BYTE4, width, height, 1, true);
    vector<SparseTextureHeap> heaps;
    SparseCommandList sparse_cmdlist;
    for (auto x : vstd::range(width / 128)) {
        for (auto y : vstd::range(height / 128)) {
            auto &heap = heaps.emplace_back(device.allocate_sparse_texture_heap(pixel_storage_size(img.storage(), uint3(128, 128, 1)), false));
            sparse_cmdlist << img.map_tile(uint2(x, y) * 128u / img.tile_size(), uint2(128) / img.tile_size(), 0, heap);
        }
    }
    stream << sparse_cmdlist.commit() << synchronize();

    auto dstorage_ext = device.extension<DStorageExt>();
    auto file = dstorage_ext->open_file("pixels.bytes");
    auto dstorage_stream = dstorage_ext->create_stream(DStorageStreamOption{DStorageStreamSource::FileSource});
    // this direct-storage command is totally failed at AMD Radeon RX 7900 XTX
    dstorage_stream << file.copy_to(img.view()) << synchronize();
    Window window{"test dstorage", uint2(width, height)};
    Swapchain swap_chain{device.create_swapchain(
        window.native_handle(),
        stream,
        uint2(width, height),
        false, false, 2)};
    while (!window.should_close()) {
        window.poll_events();
        stream << swap_chain.present(img.view());
    }
}
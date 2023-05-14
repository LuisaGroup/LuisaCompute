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
    // Direct Storage only supported for dx currently.
    Device device = context.create_device("dx");
    DStorageExt *dstorage_ext = device.extension<DStorageExt>();
    Stream dstorage_stream = dstorage_ext->create_stream();
    Stream compute_stream = device.create_stream();
    Event event = device.create_event();
    LUISA_INFO("Start test memory and buffer read.");
    // Write a test file
    {
        FILE *file = fopen("test_dstorage_file.txt", "wb");
        if (file) {
            luisa::string_view content = "hello world!";
            fwrite(content.data(), content.size(), 1, file);
            fclose(file);
        }
    }
    {
        DStorageFile file = dstorage_ext->open_file("test_dstorage_file.txt");
        if (!file) {
            LUISA_WARNING("Buffer file not found.");
            exit(1);
        }
        luisa::string file_text;
        file_text.resize(file.size_bytes());
        // create a direct-storage stream
        Buffer<int> buffer = device.create_buffer<int>(file.size_bytes() / sizeof(int));
        luisa::vector<char> buffer_data;
        buffer_data.resize(buffer.size_bytes() + 1);
        // Read buffer from file

        dstorage_stream
            // read to memory
            << file.read_to(file_text.data(), file_text.size())
            // read to memory read to buffer
            << file.read_to(buffer)
            // make event signal
            << event.signal();
        // wait for disk reading and read back to memory.
        compute_stream << event.wait() << buffer.copy_to(buffer_data.data()) << synchronize();
        for (size_t i = file.size_bytes(); i < buffer_data.size(); ++i) {
            buffer_data[i] = 0;
        }
        LUISA_INFO("Memory result: {}", file_text);
        LUISA_INFO("Buffer result: {}", buffer_data.data());
    }
    LUISA_INFO("Start test texture read.");
    static constexpr uint32_t width = 512;
    static constexpr uint32_t height = 512;
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
    {
        Image<float> img = device.create_image<float>(PixelStorage::BYTE4, width, height);
        luisa::vector<uint8_t> out_pixels(width * height * 4u);
        Clock clock{};
        dstorage_stream << DStorageMemory::read_to(img, pixels.data()) << synchronize();
        double time = clock.toc();
        LUISA_INFO("Texture read time: {} ms", time);
        compute_stream << img.copy_to(out_pixels.data()) << synchronize();
        stbi_write_png("test_dstorage_texture.png", width, height, 4, out_pixels.data(), 0);
    }
    LUISA_INFO("Texture result read to test_dstorage_texture.png.");
    LUISA_INFO("Start test texture compress and decompress.");
    luisa::vector<std::byte> compressed_pixels;
    Clock compress_clock{};
    dstorage_ext->gdeflate_compress(luisa::span{reinterpret_cast<std::byte const *>(pixels.data()), pixels.size()}, DStorageExt::CompressQuality::Best, compressed_pixels);
    double compress_time = compress_clock.toc();
    LUISA_INFO("Texture compress time: {} ms, before compress size: {} bytes, after compress size: {} bytes", compress_time, pixels.size_bytes(), compressed_pixels.size_bytes());
    {
        Image<float> img = device.create_image<float>(PixelStorage::BYTE4, width, height);
        luisa::vector<uint8_t> out_pixels(width * height * 4u);
        Clock decompress_clock{};
        dstorage_stream
            << DStorageMemory::decompress_to(img, compressed_pixels.data())
            << synchronize();
        double decompress_time = decompress_clock.toc();
        LUISA_INFO("Texture decompress time: {} ms", decompress_time);
        compute_stream << img.copy_to(out_pixels.data()) << synchronize();
        stbi_write_png("test_dstorage_texture_decompressed.png", width, height, 4, out_pixels.data(), 0);
    }
    LUISA_INFO("Decompressed texture result read to test_dstorage_texture_decompressed.png.");
}
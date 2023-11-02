#include <fstream>

#include <luisa/runtime/context.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/stream.h>
#include <luisa/runtime/buffer.h>
#include <luisa/runtime/image.h>
#include <luisa/core/logging.h>
#include <luisa/runtime/event.h>
#include <luisa/backends/ext/dstorage_ext.hpp>
#include <stb/stb_image_write.h>
#include <luisa/core/clock.h>

using namespace luisa;
using namespace luisa::compute;

int main(int argc, char *argv[]) {

    Context context{argv[0]};

    if (argc <= 1) {
        LUISA_INFO("Usage: {} <backend>. <backend>: cuda, dx, cpu, metal", argv[0]);
        exit(1);
    }
    auto device = context.create_device(argv[1]);
    auto dstorage_ext = device.extension<DStorageExt>();
    static constexpr uint32_t width = 4096;
    static constexpr uint32_t height = 4096;
    Stream dstorage_memory_stream = dstorage_ext->create_stream(DStorageStreamOption{DStorageStreamSource::MemorySource});
    Stream dstorage_file_stream = dstorage_ext->create_stream(DStorageStreamOption{DStorageStreamSource::FileSource,});
    Stream compute_stream = device.create_stream();
    TimelineEvent event = device.create_timeline_event();
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
        dstorage_file_stream
            // read to memory
            << file.copy_to(file_text.data(), file_text.size())
            // read to memory read to buffer
            << file.copy_to(buffer)
            // make event signal
            << event.signal(1);

        // wait for disk reading and read back to memory.
        compute_stream << event.wait(1)
                       << buffer.copy_to(buffer_data.data())
                       << event.signal(2);
        event.synchronize(2);
        for (size_t i = file.size_bytes(); i < buffer_data.size(); ++i) {
            buffer_data[i] = 0;
        }
        LUISA_INFO("Memory result: {}", file_text);
        LUISA_INFO("Buffer result: {}", buffer_data.data());
    }
    LUISA_INFO("Start test texture read.");

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
        DStorageFile pinned_pixels = dstorage_ext->pin_memory(pixels.data(), pixels.size_bytes());
        dstorage_memory_stream << pinned_pixels.copy_to(img) << synchronize();
        double time = clock.toc();
        LUISA_INFO("Texture read time: {} ms", time);
        compute_stream << img.copy_to(out_pixels.data()) << synchronize();
        stbi_write_png("test_dstorage_texture.png", width, height, 4, out_pixels.data(), 0);
    }
    LUISA_INFO("Texture result written to test_dstorage_texture.png.");
    LUISA_INFO("Start test texture compress and decompress.");
    luisa::vector<std::byte> compressed_pixels;
    Clock compress_clock{};
    auto compression = DStorageCompression::GDeflate;
    dstorage_ext->compress(pixels.data(), luisa::span{pixels}.size_bytes(), compression,
                           DStorageCompressionQuality::Best, compressed_pixels);
    double compress_time = compress_clock.toc();
    {
        std::ofstream file{"test_dstorage_texture_compressed.gdeflate", std::ios::binary};
        file.write(reinterpret_cast<const char *>(compressed_pixels.data()),
                   static_cast<ssize_t>(compressed_pixels.size_bytes()));
    }
    LUISA_INFO("Texture compress time: {} ms, before compress size: {} bytes, after compress size: {} bytes", compress_time, pixels.size_bytes(), compressed_pixels.size_bytes());
    {
        Image<float> img = device.create_image<float>(PixelStorage::BYTE4, width, height);
        luisa::vector<std::byte> out_pixels(width * height * 4u);
        Clock decompress_clock{};
        DStorageFile pinned_pixels = dstorage_ext->pin_memory(compressed_pixels.data(), compressed_pixels.size_bytes());
        dstorage_memory_stream << pinned_pixels.copy_to(img, compression)
                               << synchronize();
        double decompress_time = decompress_clock.toc();
        LUISA_INFO("Texture decompress time: {} ms", decompress_time);
        compute_stream << img.copy_to(out_pixels.data()) << synchronize();
        stbi_write_png("test_dstorage_texture_decompressed.png", width, height, 4, out_pixels.data(), 0);
        decompress_clock.tic();
        dstorage_memory_stream << pinned_pixels.copy_to(luisa::span{out_pixels}, compression)
                               << synchronize();
        decompress_time = decompress_clock.toc();
        LUISA_INFO("Memory decompress time: {} ms", decompress_time);
        stbi_write_png("test_dstorage_texture_decompressed_memory.png", width, height, 4, out_pixels.data(), 0);
    }
}

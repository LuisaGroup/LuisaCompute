#include "ut/ut.hpp"
#include "test_device.h"
#include <fstream>

#include <luisa/runtime/context.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/stream.h>
#include <luisa/runtime/buffer.h>
#include <luisa/runtime/image.h>
#include <luisa/core/logging.h>
#include <luisa/runtime/event.h>
#include <luisa/vstl/common.h>
#include <luisa/backends/ext/dstorage_ext.hpp>
#include <luisa/backends/ext/pinned_memory_ext.hpp>
#include "../../reference_image.h"
#include <luisa/core/clock.h>

#include <filesystem>

using namespace luisa;
using namespace luisa::compute;
using namespace boost::ut;
using namespace boost::ut::literals;

void test_dstorage(Device &device) {

    auto opts = luisa::test::ImageTestOptions::parse(
        boost::ut::detail::cfg::largc,
        boost::ut::detail::cfg::largv);
    auto dstorage_ext = device.extension<DStorageExt>();
    auto ref_dir = luisa::test::find_reference_dir(std::filesystem::path{boost::ut::detail::cfg::largv[0]}.parent_path());
    auto comparison_failed = false;
    static constexpr uint32_t width = 4096;
    static constexpr uint32_t height = 4096;
    static constexpr size_t staging_buffer_size = 32ull * 1024ull * 1024ull;
    Stream dstorage_memory_stream = dstorage_ext->create_stream(DStorageStreamOption{DStorageStreamSource::MemorySource, staging_buffer_size});
    Stream dstorage_file_stream = dstorage_ext->create_stream(DStorageStreamOption{DStorageStreamSource::FileSource, staging_buffer_size});
    Stream copy_stream = device.create_stream(StreamTag::COPY);
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
            boost::ut::expect(static_cast<bool>(file)) << "Buffer file not found.";
            return;
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
            << file.copy_to(luisa::span{file_text.data(), file_text.size()})
            // read to memory read to buffer
            << file.copy_to(buffer)
            // make event signal
            << event.signal(1);

        // wait for disk reading and read back to memory.
        copy_stream << event.wait(1)
                    << buffer.copy_to(luisa::span{buffer_data})
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
    auto f = fopen("pixels.bytes", "wb");
    fwrite(pixels.data(), luisa::size_bytes(pixels), 1, f);
    fclose(f);
    {
        auto img = device.create_image<float>(PixelStorage::BYTE4, width, height / 2);
        luisa::vector<uint8_t> out_pixels(width * height * 2);
        DStorageFile pinned_pixels = dstorage_ext->open_file("pixels.bytes");
        auto pinned_ext = device.extension<PinnedMemoryExt>();
        // D3D12_HEAP_TYPE_UPLOAD
        auto buffer = pinned_ext->allocate_pinned_memory<uint>(staging_buffer_size, {true});
        auto evt = device.create_event();
        Clock clock{};
        size_t offset = 0;
        while (offset < out_pixels.size()) {
            auto size = pinned_pixels.size_bytes() - offset;
            size = std::min(size, staging_buffer_size);
            dstorage_file_stream
                // pinned memory
                << pinned_pixels.view(offset).copy_to(buffer.native_handle(), size)
                << evt.signal();
            copy_stream
                << evt.wait()
                // We have to use sub-range copy here
                // this api not opened in front-end, due to dx backend's limitation
                << luisa::make_unique<BufferToTextureCopyCommand>(
                       buffer.handle(),
                       0,
                       img.handle(),
                       img.storage(),
                       0,
                       uint3(width, size / (width * 4), 1),
                       uint3(0, offset / (width * 4), 0))
                << synchronize();
            offset += size;
        }
        double time = clock.toc();
        LUISA_INFO("Texture read time: {} ms", time);
        copy_stream << img.copy_to(luisa::span{out_pixels}) << synchronize();
        stbi_write_png("test_dstorage_texture.png", width, height / 2, 4, out_pixels.data(), 0);
        auto result = luisa::test::save_and_compare(
            out_pixels.data(), static_cast<int>(width), static_cast<int>(height / 2), 4,
            "test_dstorage_texture", opts.output_dir, ref_dir, opts.update_reference);
        LUISA_INFO("Reference comparison: {} ({})", result.passed ? "PASSED" : "FAILED", result.message);
        if (!result.passed) {
            LUISA_ERROR("Reference comparison failed for test_dstorage_texture: {}", result.message);
            comparison_failed = true;
        }
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
                   static_cast<std::streamsize>(luisa::size_bytes(compressed_pixels)));
    }
    LUISA_INFO("Texture compress time: {} ms, before compress size: {} bytes, after compress size: {} bytes", compress_time, luisa::size_bytes(pixels), luisa::size_bytes(compressed_pixels));
    {
        Image<float> img = device.create_image<float>(PixelStorage::BYTE4, width, height / 2);
        luisa::vector<std::byte> out_pixels(width * height * 4u / 2);
        Clock decompress_clock{};
        DStorageFile pinned_pixels = dstorage_ext->pin_memory(compressed_pixels.data(), luisa::size_bytes(compressed_pixels));
        dstorage_memory_stream << pinned_pixels.copy_to(img, compression)
                               << synchronize();
        double decompress_time = decompress_clock.toc();
        LUISA_INFO("Texture decompress time: {} ms", decompress_time);
        copy_stream << img.copy_to(luisa::span{out_pixels}) << synchronize();
        stbi_write_png("test_dstorage_texture_decompressed.png", width, height / 2, 4, out_pixels.data(), 0);
        auto decompressed_result = luisa::test::save_and_compare(
            reinterpret_cast<const uint8_t *>(out_pixels.data()), static_cast<int>(width), static_cast<int>(height / 2), 4,
            "test_dstorage_texture_decompressed", opts.output_dir, ref_dir, opts.update_reference);
        LUISA_INFO("Reference comparison: {} ({})", decompressed_result.passed ? "PASSED" : "FAILED", decompressed_result.message);
        if (!decompressed_result.passed) {
            LUISA_ERROR("Reference comparison failed for test_dstorage_texture_decompressed: {}", decompressed_result.message);
            comparison_failed = true;
        }
        decompress_clock.tic();
        dstorage_memory_stream << pinned_pixels.copy_to(luisa::span{out_pixels}, compression)
                               << synchronize();
        decompress_time = decompress_clock.toc();
        LUISA_INFO("Memory decompress time: {} ms", decompress_time);
        stbi_write_png("test_dstorage_texture_decompressed_memory.png", width, height / 2, 4, out_pixels.data(), 0);
        auto memory_result = luisa::test::save_and_compare(
            reinterpret_cast<const uint8_t *>(out_pixels.data()), static_cast<int>(width), static_cast<int>(height / 2), 4,
            "test_dstorage_texture_decompressed_memory", opts.output_dir, ref_dir, opts.update_reference);
        LUISA_INFO("Reference comparison: {} ({})", memory_result.passed ? "PASSED" : "FAILED", memory_result.message);
        if (!memory_result.passed) {
            LUISA_ERROR("Reference comparison failed for test_dstorage_texture_decompressed_memory: {}", memory_result.message);
            comparison_failed = true;
        }
    }
    if (comparison_failed) {
        if (opts.offline) {
            LUISA_ERROR("One or more reference image comparisons failed in test_dstorage.");
            boost::ut::expect(false) << "One or more reference image comparisons failed in test_dstorage.";
            return;
        }
        boost::ut::expect(false) << "One or more reference image comparisons failed in test_dstorage.";
        return;
    }
}

int main(int argc, char *argv[]) {
    auto dc = luisa::test::create_device_from_ut(argc, argv);
    if (!dc) {
        return 0;
    }
    boost::ut::detail::cfg::parse_arg_with_fallback(argc, const_cast<const char**>(argv));
    auto &device = dc->device;
    test_dstorage(device);
}

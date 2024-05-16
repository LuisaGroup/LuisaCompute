/**
 * @file src/tests/next/test/ext/core/test_dstorage.cpp
 * @author sailing-innocent, on maxwell's previous work
 * @date 2023-12-18
 * @brief the Direct Storage Extension test suite】
 */

#include "common/config.h"
#include <luisa/vstl/common.h>
#include <luisa/runtime/context.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/stream.h>
#include <luisa/runtime/buffer.h>
#include <luisa/runtime/image.h>
// EXTENSION HEADER
#include <luisa/backends/ext/dstorage_ext.hpp>
#include <luisa/backends/ext/pinned_memory_ext.hpp>

// UTILS
#include <luisa/runtime/event.h>
#include <luisa/core/clock.h>
#include <luisa/core/logging.h>
#include <stb/stb_image_write.h>

using namespace luisa;
using namespace luisa::compute;

namespace luisa::test {

int test_dstorage_texture(Device &device, const luisa::string_view file_name = "pixels.bytes") {
    auto dstorage_ext = device.extension<DStorageExt>();
    static constexpr uint32_t width = 4096;
    static constexpr uint32_t height = 4096;
    static constexpr size_t staging_buffer_size = 32ull * 1024ull * 1024ull;
    // Create Streams
    Stream dstorage_memory_stream = dstorage_ext->create_stream(DStorageStreamOption{DStorageStreamSource::MemorySource, staging_buffer_size});
    Stream dstorage_file_stream = dstorage_ext->create_stream(DStorageStreamOption{DStorageStreamSource::FileSource, staging_buffer_size});
    Stream copy_stream = device.create_stream(StreamTag::COPY);

    TimelineEvent event = device.create_timeline_event();
    LUISA_INFO("Start test memory and buffer read.");
    luisa::vector<uint8_t> pixels(width * height * 4);
    // Write pixel
    {
        for (size_t x = 0; x < width; ++x)
            for (size_t y = 0; y < height; ++y) {
                size_t pixel_pos = x + y * width;
                float2 uv = make_float2(x, y) / make_float2(width, height);
                pixels[pixel_pos * 4] = static_cast<uint8_t>(uv.x * 255);
                pixels[pixel_pos * 4 + 1] = static_cast<uint8_t>(uv.y * 255);
                pixels[pixel_pos * 4 + 2] = 127;
                pixels[pixel_pos * 4 + 3] = 255;
            }
        auto f = fopen(file_name.data(), "wb");
        fwrite(pixels.data(), pixels.size_bytes(), 1, f);
        fclose(f);
    }

    {
        // img on gpu
        auto img = device.create_image<float>(PixelStorage::BYTE4, width, height / 2);
        luisa::vector<uint8_t> out_pixels(width * height * 2);
        DStorageFile pinned_pixels = dstorage_ext->open_file(file_name);
        auto pinned_ext = device.extension<PinnedMemoryExt>();
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
        copy_stream << img.copy_to(out_pixels.data()) << synchronize();
        stbi_write_png("test_dstorage_texture.png", width, height / 2, 4, out_pixels.data(), 0);
    }

    return 0;
}

int test_dstorage_str(Device &device, const luisa::string_view file_name = "test_dstorage_file_hello.txt", const luisa::string_view content = "hello world!") {
    // Extension
    auto dstorage_ext = device.extension<DStorageExt>();
    static constexpr uint32_t width = 4096;
    static constexpr uint32_t height = 4096;
    static constexpr size_t staging_buffer_size = 32ull * 1024ull * 1024ull;
    // Create Streams
    Stream dstorage_memory_stream = dstorage_ext->create_stream(DStorageStreamOption{DStorageStreamSource::MemorySource, staging_buffer_size});
    Stream dstorage_file_stream = dstorage_ext->create_stream(DStorageStreamOption{DStorageStreamSource::FileSource, staging_buffer_size});
    Stream copy_stream = device.create_stream(StreamTag::COPY);

    TimelineEvent event = device.create_timeline_event();
    LUISA_INFO("Start test memory and buffer read.");
    // Write a test file
    {
        FILE *file = fopen(file_name.data(), "wb");
        if (file) {
            fwrite(content.data(), content.size(), 1, file);
            fclose(file);
        }
    }
    {
        DStorageFile file = dstorage_ext->open_file(file_name.data());
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

        // Wait for event, then copy
        copy_stream << event.wait(1)
                    << buffer.copy_to(buffer_data.data())
                    << event.signal(2);
        event.synchronize(2);

        for (size_t i = file.size_bytes(); i < buffer_data.size(); ++i) {
            buffer_data[i] = 0;
        }
        CHECK(check_bytes_equal(file_text.data(), content.data(), file_text.size()));
    }
    return 0;
}

}// namespace luisa::test

TEST_SUITE("ext") {
    using namespace luisa::test;
    LUISA_TEST_CASE_WITH_DEVICE("ext_dstorage_str", test_dstorage_str(device) == 0);
    LUISA_TEST_CASE_WITH_DEVICE("ext_dstorage_texture", test_dstorage_texture(device) == 0);
}
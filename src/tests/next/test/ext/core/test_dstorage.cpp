/** [WIP]
 * @file src/tests/next/test/ext/core/test_dstorage.cpp
 * @author sailing-innocent, on maxwell's previous work
 * @date 2023/12/18
 * @brief the Direct Storage Extension test suite】
 */

#include "common/config.h"
#include <luisa/runtime/context.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/stream.h>
// EXTENSION HEADER
#include <luisa/backends/ext/dstorage_ext.hpp>

// UTILS
#include <luisa/runtime/event.h>
#include <luisa/core/clock.h>
#include <luisa/core/logging.h>

using namespace luisa;
using namespace luisa::compute;

namespace luisa::test {

int test_dstorage(Device &device) {
    // Extension
    auto dstorage_ext = device.extension<DStorageExt>();
    // Parameters
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
        FILE *file = fopen("test_dstorage_file.txt", "wb");
        if (file) {
            luisa::string_view content = "hello world!";
            fwrite(content.data(), content.size(), 1, file);
            fclose(file);
        }
    }
    // TODO: test process
    return 0;
}

}// namespace luisa::test

TEST_SUITE("ext") {
    LUISA_TEST_CASE_WITH_DEVICE("ext_dstorage", luisa::test::test_dstorage(device) == 0);
}
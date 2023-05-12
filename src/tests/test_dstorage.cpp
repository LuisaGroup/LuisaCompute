#include <runtime/context.h>
#include <runtime/device.h>
#include <runtime/stream.h>
#include <runtime/buffer.h>
#include <runtime/image.h>
#include <core/logging.h>
#include <runtime/event.h>
#include <backends/ext/dstorage_ext.hpp>
#include <stb/stb_image_write.h>

using namespace luisa;
using namespace luisa::compute;
int main(int argc, char *argv[]) {
    // Write a test file
    {
        FILE *file = fopen("test_dstorage_file.txt", "wb");
        if (file) {
            luisa::string_view content = "hello world!";
            fwrite(content.data(), content.size(), 1, file);
            fclose(file);
        }
    }

    Context context{argv[0]};
    // Direct Storage only supported for dx currently.
    Device device = context.create_device("dx");
    DStorageExt *dstorage_ext = device.extension<DStorageExt>();
    Stream dstorage_stream = dstorage_ext->create_stream();
    Stream compute_stream = device.create_stream();
    Event event = device.create_event();

    DStorageFile file = dstorage_ext->open_file("test_dstorage_file.txt");
    if (!file.valid()) {
        LUISA_WARNING("File not found.");
        exit(1);
    }
    // create a direct-storage stream
    Buffer<int> buffer = device.create_buffer<int>(file.size_bytes() / sizeof(int));
    luisa::vector<char> buffer_data;
    buffer_data.resize(buffer.size_bytes() + 1);
    // Read buffer from file

    dstorage_stream << file.read_to(0, buffer) << event.signal();
    // wait for disk reading and read back to memory.
    compute_stream << event.wait() << buffer.copy_to(buffer_data.data()) << synchronize();
    for (size_t i = file.size_bytes(); i < buffer_data.size(); ++i) {
        buffer_data[i] = 0;
    }
    LUISA_INFO("Direct-Storage result: {}", buffer_data.data());
}
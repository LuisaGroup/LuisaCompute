#include <core/dynamic_module.h>
#include <core/binary_io.h>
#include <runtime/Context.h>
#include <core/logging.h>
#include <iostream>
#include <backends/common/default_binary_io.h>
#include <backends/common/default_binary_io.cpp>
using namespace luisa;
using namespace luisa::compute;
int main(int argc, char *argv[]) {
    Context context{argv[0]};
    auto f = fopen("test_file.txt", "wb");
    if (f) {
        luisa::string_view content = "hello world";
        fwrite(content.data(), content.size(), 1, f);
        fclose(f);
        LUISA_INFO("Pre-write success.");
    } else {
        LUISA_ERROR("Pre-write failed.");
    }
    DynamicModule dylib = DynamicModule::load(context.runtime_directory(), "lc-dstorage");
    // void *create_dstorage_impl(compute::Context const &ctx) noexcept
    auto create_dstorage_impl = dylib.function<void *(Context const &ctx, void* device)>("create_dstorage_impl");
    // void delete_dstorage_impl(void *ptr) noexcept
    auto delete_dstorage_impl = dylib.function<void(void *ptr)>("delete_dstorage_impl");
    // BinaryStream *create_dstorage_stream(void *impl, luisa::string_view const &path) noexcept
    auto create_dstorage_stream = dylib.function<BinaryStream *(void *impl, luisa::string_view path)>("create_dstorage_stream");
    // void delete_dstorage_stream(BinaryStream *stream) noexcept
    void *impl = create_dstorage_impl(context, nullptr);
    BinaryStream *stream = create_dstorage_stream(impl, "test_file.txt");
    LUISA_INFO("File length: {}", stream->length());
    luisa::string str;
    str.resize(stream->length());
    stream->read({reinterpret_cast<std::byte *>(str.data()), str.size()});
    LUISA_INFO("File data: {}", str);
    delete_with_allocator(stream);
    f = fopen("test_file.txt", "wb");
    if (f) {
        luisa::string_view content = "hello world";
        fwrite(content.data(), content.size(), 1, f);
        fclose(f);
        LUISA_INFO("Post-write success.");
    } else {
        LUISA_ERROR("Post-write failed.");
    }
    delete_dstorage_impl(impl);
}
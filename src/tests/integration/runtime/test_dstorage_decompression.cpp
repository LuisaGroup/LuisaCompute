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
#include <luisa/backends/ext/dstorage_ext.hpp>
#include "reference_image.h"
#include <luisa/core/clock.h>

#include <filesystem>

using namespace luisa;
using namespace luisa::compute;
using namespace boost::ut;
using namespace boost::ut::literals;

void test_dstorage_decompression(Device &device) {

    auto argv = boost::ut::detail::cfg::largv;

    auto opts = luisa::test::ImageTestOptions::parse(
        boost::ut::detail::cfg::largc,
        boost::ut::detail::cfg::largv);
    auto dstorage_ext = device.extension<DStorageExt>();

    auto dstorage_stream = dstorage_ext->create_stream();
    auto dstorage_file = dstorage_ext->open_file("test_dstorage_texture_compressed.gdeflate");
    auto image = device.create_image<float>(PixelStorage::BYTE4, make_uint2(4096));
    dstorage_stream << dstorage_file.copy_to(image, DStorageCompression::GDeflate) << synchronize();

    luisa::vector<uint8_t> pixels(image.view().size_bytes());
    auto compute_stream = device.create_stream();
    compute_stream << image.copy_to(luisa::span{pixels}) << synchronize();

    stbi_write_png("test_dstorage_decompression.png", 4096, 4096, 4, pixels.data(), 0);
    if (opts.compare_path) {
        auto result = luisa::test::compare_with_reference_file(
            pixels.data(), 4096, 4096, 4,
            *opts.compare_path);
        LUISA_INFO("Reference comparison [test_dstorage_decompression]: {} ({})", result.passed ? "PASSED" : "FAILED", result.message);
        if (!result.passed) {
            boost::ut::expect(static_cast<bool>(result.passed)) << result.message;
            return;
        }
    }
}

int main(int argc, char *argv[]) {
    auto dc = luisa::test::create_device_from_ut(argc, argv);
    if (!dc) {
        return 0;
    }
    boost::ut::detail::cfg::parse_arg_with_fallback(argc, const_cast<const char**>(argv));
    auto &device = dc->device;
    test_dstorage_decompression(device);
}

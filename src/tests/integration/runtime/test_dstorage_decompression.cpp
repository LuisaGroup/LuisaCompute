// Direct-storage decompression integration test.
// This test covers extension discovery, GDeflate-to-image decompression, and readback.
//
// Two input modes:
//   * `--input <file.gdeflate>` decompresses an externally produced file (the
//     original contract of this test; combine with `--compare` for a reference
//     image check).
//   * Without `--input`, a backend that advertises `DStorageCompression::GDeflate`
//     synthesizes its own input through `DStorageExt::compress()` and verifies
//     the decompressed image against the source, so the decompression path stays
//     covered by a plain `xmake run test_dstorage_decompression <backend>`.

#include "ut/ut.hpp"
#include "test_device.h"

#include <cstring>
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
#include <luisa/core/stl/filesystem.h>

#include <filesystem>

using namespace luisa;
using namespace luisa::compute;
using namespace boost::ut;
using namespace boost::ut::literals;

namespace {

constexpr uint32_t kWidth = 4096u;
constexpr uint32_t kHeight = 4096u;

[[nodiscard]] uint8_t pattern_byte(size_t i) noexcept {
    auto v = static_cast<uint32_t>(i) * 2654435761u + 0x9e3779b9u;
    return static_cast<uint8_t>((v >> 13u) ^ v);
}

/// The pitch-aligned source layout the direct-storage contract requires for a
/// texture destination (`dstorage_ext.hpp`), plus the tightly packed pixel data
/// the destination image is expected to hold.
[[nodiscard]] luisa::vector<uint8_t> make_padded_source(
    PixelStorage storage, uint32_t width, uint32_t height,
    luisa::vector<uint8_t> &tight) noexcept {
    auto pitch = dstorage_texture_row_pitch(storage, width);
    auto tight_row = pixel_storage_size(storage, make_uint3(width, 1u, 1u));
    tight.resize(tight_row * height);
    luisa::vector<uint8_t> padded(pitch * height, 0u);
    for (uint32_t y = 0; y < height; ++y) {
        for (size_t i = 0; i < tight_row; ++i) {
            auto v = pattern_byte(static_cast<size_t>(y) * tight_row + i);
            tight[static_cast<size_t>(y) * tight_row + i] = v;
            padded[static_cast<size_t>(y) * pitch + i] = v;
        }
    }
    return padded;
}

}// namespace

void test_dstorage_decompression(Device &device) {

    auto opts = luisa::test::ImageTestOptions::parse(
        boost::ut::detail::cfg::largc,
        boost::ut::detail::cfg::largv);
    auto dstorage_ext = device.extension<DStorageExt>();
    if (dstorage_ext == nullptr) {
        LUISA_INFO("Skipping direct-storage decompression test: backend '{}' does not provide DStorageExt.", device.backend_name());
        return;
    }

    luisa::filesystem::path compressed_path;
    luisa::vector<uint8_t> expected_pixels;// set when we synthesized the input
    if (opts.input_path) {
        compressed_path = *opts.input_path;
        if (!luisa::filesystem::is_regular_file(compressed_path)) {
            boost::ut::expect(false) << "Missing direct-storage test input: " << compressed_path;
            return;
        }
    } else if (dstorage_ext->supports_compression(DStorageCompression::GDeflate)) {
        LUISA_INFO("test_dstorage_decompression: no --input given; generating a {}x{} "
                   "GDeflate file with the backend's own codec.",
                   kWidth, kHeight);
        auto source = make_padded_source(PixelStorage::BYTE4, kWidth, kHeight,
                                         expected_pixels);
        luisa::vector<std::byte> compressed;
        dstorage_ext->compress(source.data(), source.size(),
                               DStorageCompression::GDeflate,
                               DStorageCompressionQuality::Default, compressed);
        LUISA_INFO("GDeflate: {} byte(s) -> {} byte(s).", source.size(), compressed.size());
        compressed_path =
            std::filesystem::temp_directory_path() / "test_dstorage_decompression_generated.gdeflate";
        std::ofstream out{compressed_path, std::ios::binary | std::ios::trunc};
        out.write(reinterpret_cast<const char *>(compressed.data()),
                  static_cast<std::streamsize>(compressed.size()));
        out.close();
    } else {
        boost::ut::expect(false)
            << "Direct-storage decompression requires --input <file.gdeflate> "
               "(this backend does not provide a GDeflate compressor).";
        return;
    }

    auto dstorage_stream = dstorage_ext->create_stream();
    auto compressed_path_string = compressed_path.string();
    auto dstorage_file = dstorage_ext->open_file(compressed_path_string);
    auto image = device.create_image<float>(PixelStorage::BYTE4, make_uint2(kWidth, kHeight));
    dstorage_stream << dstorage_file.copy_to(image, DStorageCompression::GDeflate) << synchronize();

    luisa::vector<uint8_t> pixels(image.view().size_bytes());
    auto compute_stream = device.create_stream();
    compute_stream << image.copy_to(luisa::span{pixels}) << synchronize();

    stbi_write_png("test_dstorage_decompression.png", kWidth, kHeight, 4, pixels.data(), 0);

    if (!expected_pixels.empty()) {
        // Self-generated input: the strongest available check is a byte-exact
        // round trip against the data we compressed.
        auto mismatches = size_t{0u};
        for (size_t i = 0; i < pixels.size(); ++i) {
            if (pixels[i] != expected_pixels[i]) { ++mismatches; }
        }
        if (mismatches != 0u) {
            LUISA_ERROR("GDeflate round trip: {} of {} byte(s) differ.", mismatches, pixels.size());
        }
        boost::ut::expect(mismatches == 0u)
            << "GDeflate-compressed texture read must reproduce the source exactly";
        std::error_code ec;
        std::filesystem::remove(compressed_path, ec);
    }
    if (opts.compare_path) {
        auto result = luisa::test::compare_with_reference_file(
            pixels.data(), kWidth, kHeight, 4,
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
    boost::ut::detail::cfg::parse_arg_with_fallback(argc, const_cast<const char **>(argv));
    auto &device = dc->device;
    test_dstorage_decompression(device);
}

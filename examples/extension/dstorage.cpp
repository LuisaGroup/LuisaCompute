/**
 * @file examples/extension/dstorage.cpp
 * @brief Direct-storage extension smoke demo.
 *
 * Runs on every backend that provides `DStorageExt`:
 *   * dx     - real DirectStorage (`IDStorageQueue2`)
 *   * vk     - host-staged fallback on an internal COPY stream
 *   * cuda   - file mapping (+ pinned host memory)
 *   * metal  - MTLIO
 *
 * It writes a small pattern file, reads it back into a buffer and into a
 * pitch-aligned 2D image, then reads a pinned host range into a buffer, and
 * verifies everything byte-for-byte.
 *
 * Usage: example_dstorage <backend>   (dx | vk | cuda | metal)
 */

#include <cstring>
#include <filesystem>
#include <fstream>

#include <luisa/runtime/context.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/stream.h>
#include <luisa/runtime/buffer.h>
#include <luisa/runtime/image.h>

// EXTENSION HEADER
#include <luisa/backends/ext/dstorage_ext.hpp>

// UTILS
#include <luisa/core/clock.h>
#include <luisa/core/logging.h>

using namespace luisa;
using namespace luisa::compute;

namespace {

constexpr uint32_t kImageWidth = 100u;// 400 B/row -> padded to a 512 B pitch
constexpr uint32_t kImageHeight = 32u;

[[nodiscard]] uint8_t pattern_byte(size_t i) noexcept {
    auto v = static_cast<uint32_t>(i) * 2654435761u + 0x9e3779b9u;
    return static_cast<uint8_t>((v >> 13u) ^ v);
}

/// A pitch-aligned texture source: rows padded to the 256-byte direct-storage
/// pitch, plus the tightly packed reference the destination image must hold.
struct TextureSource {
    luisa::vector<uint8_t> padded;
    luisa::vector<uint8_t> tight;
};

[[nodiscard]] TextureSource make_texture_source(PixelStorage storage,
                                                uint32_t w, uint32_t h) noexcept {
    auto pitch = dstorage_texture_row_pitch(storage, w);
    auto tight_row = pixel_storage_size(storage, make_uint3(w, 1u, 1u));
    TextureSource source;
    source.tight.resize(tight_row * h);
    source.padded.resize(pitch * h, 0u);
    for (uint32_t y = 0; y < h; ++y) {
        for (size_t i = 0; i < tight_row; ++i) {
            auto v = pattern_byte(static_cast<size_t>(y) * tight_row + i);
            source.tight[static_cast<size_t>(y) * tight_row + i] = v;
            source.padded[static_cast<size_t>(y) * pitch + i] = v;
        }
    }
    return source;
}

[[nodiscard]] bool bytes_equal(luisa::span<const uint8_t> a,
                               luisa::span<const uint8_t> b) noexcept {
    return a.size() == b.size() &&
           std::memcmp(a.data(), b.data(), a.size()) == 0;
}

}// namespace

int test_dstorage(Device &device) {
    auto *dstorage_ext = device.extension<DStorageExt>();
    if (dstorage_ext == nullptr) {
        LUISA_INFO("Backend '{}' does not provide DStorageExt; nothing to do.",
                   device.backend_name());
        return 0;
    }
    LUISA_INFO("Direct storage on '{}': None supported = {}, GDeflate supported = {}.",
               device.backend_name(),
               dstorage_ext->supports_compression(DStorageCompression::None),
               dstorage_ext->supports_compression(DStorageCompression::GDeflate));

    // A single AnySource stream serves both file- and memory-sourced reads.
    DStorageStreamOption option;
    option.source = DStorageStreamSource::AnySource;
    option.staging_buffer_size = 4ull * 1024ull * 1024ull;
    auto dstorage_stream = dstorage_ext->create_stream(option);
    if (!dstorage_stream) {
        LUISA_WARNING("Could not create a DStorage stream on '{}' "
                      "(is the DirectStorage runtime installed?).",
                      device.backend_name());
        return 0;
    }
    auto compute_stream = device.create_stream();

    Clock clock;
    auto failures = 0;

    // ---- file -> buffer ----------------------------------------------------
    {
        auto host = luisa::vector<uint8_t>(1ull << 20);
        for (size_t i = 0; i < host.size(); ++i) { host[i] = pattern_byte(i); }
        auto path = std::filesystem::temp_directory_path() / "luisa_dstorage_example.bin";
        {
            std::ofstream out{path, std::ios::binary | std::ios::trunc};
            out.write(reinterpret_cast<const char *>(host.data()),
                      static_cast<std::streamsize>(host.size()));
        }
        auto file = dstorage_ext->open_file(path.string());
        if (!file) {
            LUISA_ERROR("Failed to open '{}'.", path.string());
            return 1;
        }
        auto buffer = device.create_buffer<uint8_t>(host.size());
        dstorage_stream << file.copy_to(buffer) << synchronize();
        luisa::vector<uint8_t> got(host.size());
        compute_stream << buffer.copy_to(luisa::span{got}) << synchronize();
        auto ok = bytes_equal(got, host);
        LUISA_INFO("[file -> buffer] {} byte(s): {}.", host.size(), ok ? "OK" : "FAILED");
        failures += ok ? 0 : 1;
        std::error_code ec;
        std::filesystem::remove(path, ec);
    }

    // ---- file -> image (pitch-aligned source) ------------------------------
    {
        auto source = make_texture_source(
            PixelStorage::BYTE4, kImageWidth, kImageHeight);
        auto path = std::filesystem::temp_directory_path() / "luisa_dstorage_example_image.bin";
        {
            std::ofstream out{path, std::ios::binary | std::ios::trunc};
            out.write(reinterpret_cast<const char *>(source.padded.data()),
                      static_cast<std::streamsize>(source.padded.size()));
        }
        auto file = dstorage_ext->open_file(path.string());
        auto image = device.create_image<float>(
            PixelStorage::BYTE4, make_uint2(kImageWidth, kImageHeight));
        dstorage_stream << file.copy_to(image) << synchronize();
        luisa::vector<uint8_t> got(image.view().size_bytes());
        compute_stream << image.copy_to(luisa::span{got}) << synchronize();
        auto ok = bytes_equal(got, source.tight);
        LUISA_INFO("[file -> image {}x{} ({} B row pitch)]: {}.",
                   kImageWidth, kImageHeight,
                   dstorage_texture_row_pitch(PixelStorage::BYTE4, kImageWidth),
                   ok ? "OK" : "FAILED");
        failures += ok ? 0 : 1;
        std::error_code ec;
        std::filesystem::remove(path, ec);
    }

    // ---- pinned host memory -> buffer --------------------------------------
    {
        auto host = luisa::vector<uint8_t>(64u * 1024u);
        for (size_t i = 0; i < host.size(); ++i) { host[i] = pattern_byte(i * 3u + 1u); }
        auto pinned = dstorage_ext->pin_memory(host.data(), host.size());
        auto buffer = device.create_buffer<uint8_t>(host.size());
        dstorage_stream << pinned.copy_to(buffer) << synchronize();
        luisa::vector<uint8_t> got(host.size());
        compute_stream << buffer.copy_to(luisa::span{got}) << synchronize();
        auto ok = bytes_equal(got, host);
        LUISA_INFO("[pinned memory -> buffer] {} byte(s): {}.", host.size(), ok ? "OK" : "FAILED");
        failures += ok ? 0 : 1;
    }

    LUISA_INFO("Direct-storage demo finished in {} ms with {} failure(s).",
               clock.toc(), failures);
    return failures == 0 ? 0 : 1;
}

int main(int argc, char *argv[]) {
    if (argc <= 1) {
        LUISA_INFO("Usage: {} <backend>. <backend>: cuda, dx, metal, vk", argv[0]);
        return 1;
    }
    Context context{argv[0]};
    Device device = context.create_device(argv[1]);
    return test_dstorage(device);
}
